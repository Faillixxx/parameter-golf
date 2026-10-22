"""
train_gpt.py - nanoNCM (Next-Concept Modeling) for OpenAI Parameter Golf, in one file.

Stage 1  multiscale residual VQ-VAE over raw UTF-8 bytes (nanoNCM's vq_tokenizer.py)
Stage 2  NCM transformer over the frozen tokenizer's codes (nanoNCM's model.py + train.py)
Eval     two-part code: bits(chunk) = -log2 p(codes) - log2 p(bytes | all levels)

Data: byte260 shards written by fineweb_bytes.py (each doc = BOS 1, then byte b as b + 4).
Every document starts a new seq_len-byte chunk and its last chunk is padded with byte 0;
padding bytes are never counted or scored. Training reads the train shards, monitoring reads
the dev split, and the official val split is read only by the final evaluation.

Settings are environment variables (like the baseline train_gpt.py), parsed like nanoNCM's
configurator.py (a Python literal, else a plain string): VQ_* for the tokenizer, plain names
for NCM and the run. Examples:
    STAGES=tokenizer VQ_MAX_ITERS=3000 python train_gpt.py
    OUT_DIR=/content/drive/MyDrive/nanoNCM/out RESUME=True python train_gpt.py
    torchrun --standalone --nproc_per_node=8 train_gpt.py
dtype=auto uses bf16 autocast on GPUs with native bf16, fp16 with a GradScaler on older GPUs
such as the T4, and fp32 on CPU. Training stages share MAX_WALLCLOCK_SECONDS
(STAGE1_FRACTION of it for the tokenizer); checkpoints in OUT_DIR allow RESUME=True.
"""

import ast
import glob
import hashlib
import inspect
import io
import json
import math
import os
import subprocess
import sys
import time
import uuid
import zlib
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

BOS_ID = 1          # byte260: pad 0, bos 1, eos 2, unk 3, then the 256 bytes
BYTE_OFFSET = 4

# -----------------------------------------------------------------------------
# Settings (configurator.py behaviour, from environment variables)

overrides = set()


def env(name, default):
    """NAME from the environment, parsed like configurator.py: a Python literal, else a string."""
    if name not in os.environ:
        return default
    overrides.add(name)
    value = os.environ[name]
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value      # plain strings like DEVICE=cpu


# -----------------------------------------------------------------------------
# Stage 1 model: multiscale residual VQ-VAE (nanoNCM vq_tokenizer.py)
#
# VAR-style, byte level:
#     256 bytes -> transformer encoder -> features f (256 x code_dim)
#     for each level (16, 64, 256 positions):
#         average-pool the current residual (f - f_hat) down to that length,
#         snap every vector to its nearest codebook entry (shared codebook, 4096 entries),
#         upsample the chosen entries back to 256 and add them to f_hat
#     f_hat -> transformer decoder -> a distribution over the 256 byte values per position
# Coarse levels quantize first; finer levels encode what is still missing.

@dataclass
class VQConfig:
    char_vocab: int = 256              # raw UTF-8 bytes (was: characters in the data)
    seq_len: int = 256                 # bytes per chunk (= finest level length)
    level_lens: tuple = (16, 64, 256)  # must match NCMConfig.level_lens
    codebook_size: int = 4096          # must match NCMConfig.vocab_size
    code_dim: int = 32
    n_embd: int = 192
    n_head: int = 4
    n_layer: int = 3                   # layers in the encoder and in the decoder
    beta: float = 0.25                 # commitment loss weight
    ema_decay: float = 0.99
    prefix_loss_weight: float = 0.5    # full CE + weight * mean(prefix CEs)
    balance_levels: bool = True        # equal EMA contribution per scale
    dead_code_threshold: float = 0.5   # EMA count; restart sooner than 1e-2

    def __post_init__(self):
        self.level_lens = tuple(self.level_lens)
        for name in ("char_vocab", "seq_len", "codebook_size", "code_dim", "n_embd", "n_head", "n_layer"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not self.level_lens or any(
            not isinstance(n, int) or isinstance(n, bool) or n <= 0 for n in self.level_lens
        ):
            raise ValueError("level_lens must contain positive integers")
        if any(a >= b for a, b in zip(self.level_lens, self.level_lens[1:])):
            raise ValueError("level_lens must be strictly increasing")
        if self.level_lens[-1] > self.seq_len:
            raise ValueError("the finest level cannot exceed seq_len")
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        if not 0 <= self.ema_decay < 1:
            raise ValueError("ema_decay must be in [0, 1)")
        if not 0 <= self.dead_code_threshold <= 1:
            raise ValueError("dead_code_threshold must be in [0, 1]")
        for name in ("beta", "prefix_loss_weight"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not isinstance(self.balance_levels, bool):
            raise ValueError("balance_levels must be True or False")


def transformer_stack(cfg):
    layer = nn.TransformerEncoderLayer(
        cfg.n_embd, cfg.n_head, 4 * cfg.n_embd, dropout=0.0,
        activation="gelu", batch_first=True, norm_first=True,
    )
    return nn.TransformerEncoder(layer, cfg.n_layer, enable_nested_tensor=False)


def distributed():
    return dist.is_available() and dist.is_initialized()


class EMACodebook(nn.Module):
    """Codebook updated by exponential moving averages (no gradient), with dead-code restarts.

    Under torchrun every GPU must keep the same codebook: EMA statistics are summed over
    GPUs, and initialization and dead-code restarts are taken from rank 0.
    """

    def __init__(self, num_codes, dim, decay, dead_code_threshold=0.5):
        super().__init__()
        self.decay = decay
        self.dead_code_threshold = dead_code_threshold
        self.register_buffer("embed", torch.randn(num_codes, dim) * 0.1)
        self.register_buffer("embed_avg", self.embed.clone())
        self.register_buffer("cluster_size", torch.ones(num_codes))
        self.register_buffer("initted", torch.tensor(False))

    def lookup(self, x):
        """x (..., D) -> index of the nearest code, shape (...)."""
        flat = x.reshape(-1, x.size(-1))
        dist_ = (flat.pow(2).sum(1, keepdim=True)
                 - 2 * flat @ self.embed.t()
                 + self.embed.pow(2).sum(1)[None, :])
        return dist_.argmin(1).view(x.shape[:-1])

    def vectors(self, idx):
        return F.embedding(idx, self.embed)

    @torch.no_grad()
    def initialize(self, flat, weights):
        """Initialize from real features BEFORE looking up their codes."""
        if not self.initted:
            pick = torch.multinomial(weights, self.embed.size(0), replacement=True)
            self.embed.copy_(flat[pick])
            if distributed():
                dist.broadcast(self.embed, 0)
            self.embed_avg.copy_(self.embed)
            self.cluster_size.fill_(1.0)
            self.initted.fill_(True)

    @torch.no_grad()
    def update(self, flat, idx, weights):
        """Update EMA statistics, with optional scale balancing via weights."""
        K = self.embed.size(0)
        counts = torch.bincount(idx, weights=weights, minlength=K)
        sums = torch.zeros_like(self.embed).index_add_(0, idx, flat * weights[:, None])
        if distributed():
            dist.all_reduce(counts)
            dist.all_reduce(sums)
        self.cluster_size.mul_(self.decay).add_(counts, alpha=1 - self.decay)
        self.embed_avg.mul_(self.decay).add_(sums, alpha=1 - self.decay)
        n = self.cluster_size.sum()
        size = (self.cluster_size + 1e-5) / (n + K * 1e-5) * n     # Laplace smoothing
        self.embed.copy_(self.embed_avg / size[:, None])
        # restart codes nobody uses with random vectors from this batch
        dead = self.cluster_size < self.dead_code_threshold
        if dead.any():
            pick = torch.multinomial(weights, int(dead.sum()), replacement=True)
            self.embed[dead] = flat[pick]
            if distributed():
                dist.broadcast(self.embed, 0)
            self.embed_avg[dead] = self.embed[dead]
            self.cluster_size[dead] = 1.0


class TextVQVAE(nn.Module):
    def __init__(self, cfg: VQConfig):
        super().__init__()
        self.cfg = cfg
        T, C = cfg.seq_len, cfg.n_embd
        # encoder
        self.tok = nn.Embedding(cfg.char_vocab, C)
        self.pos_enc = nn.Parameter(torch.randn(1, T, C) * 0.02)
        self.encoder = transformer_stack(cfg)
        self.to_code = nn.Sequential(nn.LayerNorm(C), nn.Linear(C, cfg.code_dim))
        # quantizer
        self.codebook = EMACodebook(cfg.codebook_size, cfg.code_dim, cfg.ema_decay, cfg.dead_code_threshold)
        # decoder
        self.from_code = nn.Linear(cfg.code_dim, C)
        self.pos_dec = nn.Parameter(torch.randn(1, T, C) * 0.02)
        self.decoder = transformer_stack(cfg)
        self.ln_f = nn.LayerNorm(C)
        self.head = nn.Linear(C, cfg.char_vocab)

    @staticmethod
    def _down(f, n):
        return f if f.size(1) == n else F.adaptive_avg_pool1d(f.transpose(1, 2), n).transpose(1, 2)

    @staticmethod
    def _up(q, n):
        return q if q.size(1) == n else F.interpolate(q.transpose(1, 2), size=n, mode="nearest").transpose(1, 2)

    def encode_features(self, x):
        return self.to_code(self.encoder(self.tok(x) + self.pos_enc))

    def _codebook_batch(self, residuals):
        flat = [r.reshape(-1, r.size(-1)) for r in residuals]
        mean_count = sum(r.size(0) for r in flat) / len(flat)
        weights = torch.cat([
            r.new_full((r.size(0),), mean_count / r.size(0) if self.cfg.balance_levels else 1.0)
            for r in flat
        ])
        return torch.cat(flat), weights

    def quantize(self, f, *, return_prefixes=False):
        """Multiscale residual quantization. Returns codes (list per level), f_hat, residuals that were quantized."""
        # Always fp32 and outside autocast: the nearest-code search decides the codes.
        with torch.autocast(device_type=f.device.type, enabled=False):
            f = f.float()
            if self.training and not self.codebook.initted:
                pooled, weights = self._codebook_batch([self._down(f.detach(), n) for n in self.cfg.level_lens])
                self.codebook.initialize(pooled, weights)
            T = f.size(1)
            f_hat = torch.zeros_like(f)
            codes, residuals, prefixes = [], [], []
            for n in self.cfg.level_lens:
                r = self._down(f - f_hat, n)
                idx = self.codebook.lookup(r)
                f_hat = f_hat + self._up(self.codebook.vectors(idx), T)
                codes.append(idx)
                residuals.append(r)
                prefixes.append(f_hat)
        result = (codes, f_hat, residuals)
        return (*result, prefixes) if return_prefixes else result

    def decode_features(self, f_hat):
        h = self.decoder(self.from_code(f_hat) + self.pos_dec)
        return self.head(self.ln_f(h))

    def forward(self, x):
        f = self.encode_features(x).float()
        codes, f_hat, residuals, prefixes = self.quantize(f.detach(), return_prefixes=True)
        commit = F.mse_loss(f, f_hat.detach())
        # Train the decoder on incomplete prefixes too; otherwise the full-rate
        # final level can solve reconstruction while earlier levels are ignored.
        use_prefixes = self.training and self.cfg.prefix_loss_weight > 0 and len(prefixes) > 1
        features = prefixes if use_prefixes else [f_hat]
        f_st = torch.cat([f + (p - f).detach() for p in features], dim=0)
        all_logits = self.decode_features(f_st)
        losses = F.cross_entropy(
            all_logits.reshape(-1, all_logits.size(-1)).float(), x.repeat(len(features), 1).reshape(-1),
            reduction="none",
        ).view(len(features), -1).mean(dim=1)
        logits, rec = all_logits[-x.size(0):], losses[-1]
        aux = losses[:-1].mean() if use_prefixes else rec.new_zeros(())
        loss = rec + self.cfg.prefix_loss_weight * aux + self.cfg.beta * commit
        # All reconstructions above use the same pre-update codebook snapshot.
        if self.training:
            flat, weights = self._codebook_batch(residuals)
            self.codebook.update(flat, torch.cat([c.reshape(-1) for c in codes]), weights)
        return logits, loss, rec, commit, codes

    @torch.no_grad()
    def encode(self, x):
        """(B, seq_len) byte ids -> list of code tensors, codes[k] of shape (B, level_lens[k]).
        Always fp32 and without gradients (PyTorch's fused fast path): the codes are the
        message that gets scored, so they must not depend on autocast or grad mode."""
        with torch.autocast(device_type=x.device.type, enabled=False):
            return self.quantize(self.encode_features(x))[0]

    @torch.no_grad()
    def decode_logits(self, codes):
        """List of code tensors (a prefix of the hierarchy) -> (B, seq_len, 256) byte logits."""
        if not 1 <= len(codes) <= len(self.cfg.level_lens):
            raise ValueError("decode needs a nonempty prefix of the code hierarchy")
        T = self.cfg.seq_len
        f_hat = sum(self._up(self.codebook.vectors(c), T) for c in codes)
        return self.decode_features(f_hat)

    @torch.no_grad()
    def decode(self, codes):
        """List of code tensors -> (B, seq_len) byte ids (argmax; scoring uses decode_logits)."""
        return self.decode_logits(codes).argmax(-1)


# -----------------------------------------------------------------------------
# Stage 2 model: NCM transformer (nanoNCM model.py, unchanged)
#
# A frozen multiscale residual VQ-VAE supplies discrete codes, coarse to fine.
# codes[k] is a LongTensor of shape (B, level_lens[k]). The model predicts all
# codes in one level in parallel, conditioned only on earlier levels.
#
# Training uses the original block-causal mask. Generation uses a preallocated
# per-layer KV cache by default; use_cache=False provides a reference path.

@dataclass
class NCMConfig:
    vocab_size: int = 4096
    level_lens: tuple = (16, 64, 256)
    n_embd: int = 384
    n_head: int = 6
    n_layer: int = 4
    dropout: float = 0.0
    bias: bool = False
    level_loss_weighting: str = "token"  # "token" or "level"

    def __post_init__(self):
        for name in ("vocab_size", "n_embd", "n_head", "n_layer"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.level_lens = tuple(self.level_lens)
        if not self.level_lens or any(
            not isinstance(n, int) or isinstance(n, bool) or n <= 0
            for n in self.level_lens
        ):
            raise ValueError("level_lens must contain positive integers")
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        if not 0.0 <= self.dropout <= 1.0:
            raise ValueError("dropout must be between 0 and 1")
        if self.level_loss_weighting not in ("token", "level"):
            raise ValueError("level_loss_weighting must be 'token' or 'level'")

    @property
    def num_levels(self):
        return len(self.level_lens)

    @property
    def total_len(self):
        return sum(self.level_lens)


class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def _qkv(self, x):
        B, N, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=-1)
        return tuple(
            t.view(B, N, self.n_head, C // self.n_head).transpose(1, 2)
            for t in (q, k, v)
        )

    def _attend(self, q, k, v, mask=None):
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        B, _, N, _ = q.shape
        y = y.transpose(1, 2).contiguous().view(B, N, self.n_embd)
        return self.resid_dropout(self.c_proj(y))

    def forward(self, x, mask):
        return self._attend(*self._qkv(x), mask=mask)

    def forward_cached(self, x, cache, start, total_len):
        """Inference only: append a complete level to this layer's KV cache."""
        q, k, v = self._qkv(x)
        end = start + x.size(1)
        if cache is None:
            shape = (k.size(0), self.n_head, total_len, k.size(-1))
            cache = (k.new_empty(shape), v.new_empty(shape))
        keys, values = cache
        keys[:, :, start:end].copy_(k)
        values[:, :, start:end].copy_(v)

        # Queries belong only to the current level. Every supplied key belongs
        # to an earlier level or this level, so all are visible. A triangular
        # token mask would incorrectly remove within-level attention.
        return self._attend(q, keys[:, :, :end], values[:, :, :end]), cache


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = SelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, mask):
        x = x + self.attn(self.ln_1(x), mask)
        return x + self.mlp(self.ln_2(x))

    def forward_cached(self, x, cache, start, total_len):
        y, cache = self.attn.forward_cached(self.ln_1(x), cache, start, total_len)
        x = x + y
        return x + self.mlp(self.ln_2(x)), cache


class NCMGPT(nn.Module):
    def __init__(self, config: NCMConfig):
        super().__init__()
        self.config = config
        C = config.n_embd
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, C),
            wpe=nn.Embedding(max(config.level_lens), C),
            wle=nn.Embedding(config.num_levels, C),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(C, bias=config.bias),
        ))
        self.sos = nn.Parameter(torch.zeros(1, config.level_lens[0], C))
        self.lm_head = nn.Linear(C, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        lvl_id = torch.cat([
            torch.full((n,), k, dtype=torch.long)
            for k, n in enumerate(config.level_lens)
        ])
        pos_id = torch.cat([torch.arange(n) for n in config.level_lens])
        mask = (lvl_id[None, :] <= lvl_id[:, None])[None, None]
        self.register_buffer("lvl_id", lvl_id, persistent=False)
        self.register_buffer("pos_id", pos_id, persistent=False)
        self.register_buffer("ncm_mask", mask, persistent=False)

        self.offsets = [0]
        for n in config.level_lens:
            self.offsets.append(self.offsets[-1] + n)

        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
        nn.init.normal_(self.sos, mean=0.0, std=0.02)
        print(f"number of parameters: {self.get_num_params() / 1e6:.2f}M")

    def get_num_params(self, non_embedding=True):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.transformer.wpe.weight.numel()
        return n

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    @staticmethod
    def _resize(e, n):
        if e.size(1) == n:
            return e
        return F.interpolate(e.transpose(1, 2), size=n, mode="nearest").transpose(1, 2)

    def _level_input(self, embeddings, level, batch_size):
        if level == 0:
            return self.sos.expand(batch_size, -1, -1)
        n = self.config.level_lens[level]
        x = self._resize(embeddings[0], n)
        for j in range(1, level):
            # Resize every source directly to n. Repeatedly resizing an
            # accumulated sum is not equivalent for arbitrary level lengths.
            x = x + self._resize(embeddings[j], n)
        return x

    def _level_inputs(self, codes, upto, batch_size):
        embeddings = [self.transformer.wte(codes[j]) for j in range(upto)]
        return torch.cat([
            self._level_input(embeddings, k, batch_size) for k in range(upto + 1)
        ], dim=1)

    def _add_positions(self, x, start, end):
        x = x + self.transformer.wpe(self.pos_id[start:end])
        x = x + self.transformer.wle(self.lvl_id[start:end])
        return self.transformer.drop(x)

    def _trunk(self, x, upto, current_only=False):
        end = self.offsets[upto + 1]
        x = self._add_positions(x, 0, end)
        mask = self.ncm_mask[:, :, :end, :end]
        for block in self.transformer.h:
            x = block(x, mask)
        if current_only:
            x = x[:, self.offsets[upto]:end]
        return self.lm_head(self.transformer.ln_f(x))

    def _trunk_cached(self, x, level, caches):
        start, end = self.offsets[level:level + 2]
        x = self._add_positions(x, start, end)
        for i, block in enumerate(self.transformer.h):
            x, caches[i] = block.forward_cached(
                x, caches[i], start, self.config.total_len
            )
        return self.lm_head(self.transformer.ln_f(x))

    def _check_codes(self, codes):
        if len(codes) != self.config.num_levels:
            raise ValueError(f"expected {self.config.num_levels} levels, got {len(codes)}")
        B = None
        for k, (c, n) in enumerate(zip(codes, self.config.level_lens)):
            if not isinstance(c, torch.Tensor) or c.ndim != 2:
                raise ValueError(f"level {k}: expected a 2D tensor")
            if B is None:
                B = c.size(0)
                if B == 0:
                    raise ValueError("codes must have a nonempty batch")
            if c.shape != (B, n):
                raise ValueError(f"level {k}: expected {(B, n)}, got {tuple(c.shape)}")
            if c.dtype != torch.long:
                raise TypeError(f"level {k}: codes must have dtype torch.long")
            if c.device != self.sos.device:
                raise ValueError(f"level {k}: codes and model must be on the same device")
        # Embedding and cross_entropy check ID bounds without extra per-level
        # min/max reductions and GPU-to-CPU synchronization here.
        return B

    def forward(self, codes):
        """Return (all-level logits, scalar loss, detached per-level losses)."""
        B = self._check_codes(codes)
        L = self.config.num_levels
        x = self._level_inputs(codes, L - 1, B)
        logits = self._trunk(x, L - 1)
        targets = torch.cat(codes, dim=1)
        ce = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="none"
        ).view(B, -1)
        per_level = [
            ce[:, self.offsets[k]:self.offsets[k + 1]].mean() for k in range(L)
        ]
        if self.config.level_loss_weighting == "token":
            loss = ce.mean()
        else:
            loss = torch.stack(per_level).mean()
        return logits, loss, [value.detach() for value in per_level]

    @staticmethod
    def _sample(logits, temperature, top_k, generator):
        if temperature == 0 or top_k == 1:
            return logits.argmax(dim=-1)
        B, N, V = logits.shape
        if top_k is not None and top_k < V:
            values, indices = torch.topk(logits, top_k, dim=-1, sorted=False)
            probs = F.softmax(values.float() / temperature, dim=-1)
            picked = torch.multinomial(
                probs.reshape(-1, top_k), 1, generator=generator
            ).view(B, N, 1)
            return indices.gather(-1, picked).squeeze(-1)
        probs = F.softmax(logits.float() / temperature, dim=-1)
        return torch.multinomial(
            probs.reshape(-1, V), 1, generator=generator
        ).view(B, N)

    def generate(self, batch_size, temperature=1.0, top_k=None, *, use_cache=True, generator=None):
        """
        Generate K levels, sampling all positions in each level in parallel.

        temperature=0 is greedy. top_k uses exactly k candidates, including
        when logits tie at the cutoff; top_k >= vocab_size leaves all entries.
        An optional torch.Generator must match the model's sampling device.
        Evaluation mode is temporary and every module's mode is restored.
        KV caches and code embeddings are local to this call.
        """
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be finite and nonnegative")
        if top_k is not None and (
            not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0
        ):
            raise ValueError("top_k must be a positive integer or None")

        modes = [(module, module.training) for module in self.modules()]
        self.eval()
        try:
            with torch.inference_mode():
                codes, embeddings, inputs = [], [], []
                caches = [None] * self.config.n_layer
                for k in range(self.config.num_levels):
                    x = self._level_input(embeddings, k, batch_size)
                    if use_cache:
                        logits = self._trunk_cached(x, k, caches)
                    else:
                        inputs.append(x)
                        logits = self._trunk(torch.cat(inputs, dim=1), k, current_only=True)
                    nxt = self._sample(logits, temperature, top_k, generator)
                    codes.append(nxt)
                    if k + 1 < self.config.num_levels:
                        embeddings.append(self.transformer.wte(nxt))
            # Preserve the original API's ordinary tensors: generated codes
            # can be fed to a later training pass without inference-tensor errors.
            with torch.inference_mode(False):
                return [code.clone() for code in codes]
        finally:
            for module, training in modes:
                module.training = training

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        params = [p for p in self.parameters() if p.requires_grad]
        groups = [
            {"params": [p for p in params if p.dim() >= 2], "weight_decay": weight_decay},
            {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0},
        ]
        kwargs = {}
        supports_fused = "fused" in inspect.signature(torch.optim.AdamW).parameters
        if torch.device(device_type).type == "cuda" and supports_fused:
            kwargs["fused"] = True
        return torch.optim.AdamW(groups, lr=learning_rate, betas=betas, **kwargs)


# -----------------------------------------------------------------------------
# Data: byte260 shards -> document-aligned chunks

def load_data_shard(file: Path) -> np.ndarray:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    # SHARD HEADER INTS & SHARD_MAGIC
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}: expected {expected_size} bytes")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return tokens_np


class Chunks:
    """
    Documents cut into seq_len-byte chunks, as in evaluation: every document starts a new
    chunk and its last chunk is padded with byte 0 (never scored).
        chunks      (N, seq_len) uint8
        lens        (N,) int16    real bytes per chunk (the rest is padding)
        doc_lens    (D,) int64    bytes per document
        doc_starts  (D,) int64    index of each document's first chunk (chunks are in order)
    """

    def __init__(self, chunks, lens, doc_lens, doc_starts):
        self.chunks, self.lens, self.doc_lens, self.doc_starts = chunks, lens, doc_lens, doc_starts

    def __len__(self):
        return len(self.chunks)

    @property
    def num_bytes(self):
        return int(self.doc_lens.sum())

    @classmethod
    def from_tokens(cls, tokens, seq_len):
        """byte260 tokens that start with BOS -> Chunks (vectorized)."""
        tokens = np.asarray(tokens)
        if tokens.size == 0:
            return cls.concat([], seq_len)
        if tokens[0] != BOS_ID:
            raise ValueError("token stream must start with BOS")
        is_bos = tokens == BOS_ID
        if (tokens[~is_bos] < BYTE_OFFSET).any():
            raise ValueError("unexpected special token inside a document")
        bos = np.flatnonzero(is_bos)
        doc_lens = np.diff(np.append(bos, tokens.size)) - 1
        n_chunks = -(-doc_lens // seq_len)
        doc_starts = np.cumsum(n_chunks) - n_chunks
        total = int(n_chunks.sum())
        out = np.zeros((total * seq_len,), dtype=np.uint8)
        # byte i of doc d sits at token bos[d] + 1 + i and goes to doc_starts[d] * seq_len + i
        shift = doc_starts * seq_len - bos - 1
        idx_dtype = np.int32 if max(total * seq_len, tokens.size) < 2**31 else np.int64
        src = np.flatnonzero(~is_bos).astype(idx_dtype)
        out[src + np.repeat(shift.astype(idx_dtype), doc_lens)] = (tokens[src] - BYTE_OFFSET).astype(np.uint8)
        lens = np.full((total,), seq_len, dtype=np.int16)
        has = n_chunks > 0
        lens[(doc_starts + n_chunks - 1)[has]] = (doc_lens - (n_chunks - 1) * seq_len)[has]
        return cls(out.reshape(total, seq_len), lens, doc_lens.astype(np.int64), doc_starts.astype(np.int64))

    @classmethod
    def concat(cls, parts, seq_len):
        if not parts:
            return cls(np.zeros((0, seq_len), np.uint8), np.zeros((0,), np.int16),
                       np.zeros((0,), np.int64), np.zeros((0,), np.int64))
        offsets = np.cumsum([0] + [len(p) for p in parts[:-1]])
        return cls(np.concatenate([p.chunks for p in parts]), np.concatenate([p.lens for p in parts]),
                   np.concatenate([p.doc_lens for p in parts]),
                   np.concatenate([p.doc_starts + o for p, o in zip(parts, offsets)]))


def load_chunks(pattern, seq_len, max_files=0):
    """All complete documents in the shards matching pattern (in order), as Chunks.
    Documents may straddle shards; the tail after the last shard's final BOS is kept as a
    document of its own (for train shards it may be cut short, which is harmless)."""
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if max_files:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    parts, carry = [], np.zeros((0,), np.uint16)
    for file in files:
        tokens = np.concatenate([carry, load_data_shard(file)])
        bos = np.flatnonzero(tokens == BOS_ID)
        last = int(bos[-1]) if bos.size else 0
        parts.append(Chunks.from_tokens(tokens[:last], seq_len))
        carry = tokens[last:]
    parts.append(Chunks.from_tokens(carry, seq_len))
    return Chunks.concat(parts, seq_len), len(files)


def byte_mask(lens, seq_len):
    """(B,) real-byte counts -> (B, seq_len) bool, True on bytes that are scored."""
    return torch.arange(seq_len, device=lens.device)[None, :] < lens[:, None]


# -----------------------------------------------------------------------------
# Post-training quantization (the baseline's int8 + zlib export, unchanged)

CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights",
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "INT8_KEEP_FLOAT_FP32_NAME_PATTERNS",
        ",".join(CONTROL_TENSOR_NAME_PATTERNS),
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE = torch.float16
INT8_CLIP_PERCENTILE = 99.99984
INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0

def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())

def keep_float_tensor(name: str, t: Tensor, passthrough_orig_dtypes: dict[str, str]) -> Tensor:
    if any(pattern in name for pattern in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t

def quantize_float_tensor(t: Tensor) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        # Matrices get one scale per row, which usually tracks output-channel
        # ranges much better than a single tensor-wide scale.
        clip_abs = (
            torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
            if t32.numel()
            else torch.empty((t32.shape[0],), dtype=torch.float32)
        )
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()

    # Vectors / scalars use a simpler per-tensor scale.
    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale

def quantize_state_dict_int8(state_dict: dict[str, Tensor]):
    # Single supported clean-script export format:
    # - per-row int8 for 2D float tensors
    # - per-tensor int8 for other float tensors
    # - exact passthrough for non-floats
    # - passthrough for small float tensors, stored as fp16 to save bytes
    quantized: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors", "baseline_tensor_bytes", "int8_payload_bytes"),
        0,
    )

    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += tensor_nbytes(t)

        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = t
            stats["int8_payload_bytes"] += tensor_nbytes(t)
            continue

        # Small float tensors are cheap enough to keep directly. We still downcast
        # fp32/bf16 passthrough tensors to fp16 so metadata does not dominate size.
        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += tensor_nbytes(kept)
            continue

        stats["num_float_tensors"] += 1
        q, s = quantize_float_tensor(t)
        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["int8_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)

    obj: dict[str, object] = {
        "__quant_format__": "int8_clean_per_row_v1",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats

def dequantize_state_dict_int8(obj: dict[str, object]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    qmeta = obj.get("qmeta", {})
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        if qmeta.get(name, {}).get("scheme") == "per_row" or s.ndim > 0:
            s = s.to(dtype=torch.float32)
            # Broadcast the saved row scale back across trailing dimensions.
            out[name] = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dtype).contiguous()
        else:
            scale = float(s.item())
            out[name] = (q.float() * scale).to(dtype=dtype).contiguous()
    for name, t in obj["passthrough"].items():
        # Restore small tensors, undoing the temporary fp16 storage cast if needed.
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out


def pack_int8(model):
    """state dict -> (int8 + zlib blob, quantization stats). Tied weights (NCM's wte and
    lm_head) are stored once and re-tied by unpack_int8."""
    state, aliases, seen = {}, {}, {}
    for name, t in model.state_dict().items():
        if t.data_ptr() in seen and t.shape == seen[t.data_ptr()][1].shape:
            aliases[name] = seen[t.data_ptr()][0]
        else:
            seen[t.data_ptr()] = (name, t)
            state[name] = t
    quant_obj, stats = quantize_state_dict_int8(state)
    if aliases:
        quant_obj["aliases"] = aliases
    buf = io.BytesIO()
    torch.save(quant_obj, buf)
    return zlib.compress(buf.getvalue(), level=9), stats


def unpack_int8(model, blob):
    """Load an int8 + zlib blob into model (the artifact round-trip)."""
    quant_state = torch.load(io.BytesIO(zlib.decompress(blob)), map_location="cpu")
    state = dequantize_state_dict_int8(quant_state)
    for name, source in quant_state.get("aliases", {}).items():
        state[name] = state[source]
    model.load_state_dict(state, strict=True)
    return model


# -----------------------------------------------------------------------------
# Tokenizer metrics

@torch.no_grad()
def tokenizer_report(model, data, batch_size, device, ctx, rank=0, world_size=1):
    """
    Over every chunk of `data` (split across ranks), real bytes only:
      acc[k]    byte accuracy of the argmax reconstruction from levels 0..k
      bits[k]   decoder bits per byte, -log2 p(byte | levels 0..k), from the full distribution
    plus distinct codes and code perplexity per level.
    """
    cfg = model.cfg
    L, T = len(cfg.level_lens), cfg.seq_len
    correct = torch.zeros(L, dtype=torch.float64, device=device)
    nll = torch.zeros(L, dtype=torch.float64, device=device)
    hist = torch.zeros(L, cfg.codebook_size, dtype=torch.float64, device=device)
    n_bytes = torch.zeros((), dtype=torch.float64, device=device)
    was_training = model.training
    model.eval()
    lo, hi = len(data) * rank // world_size, len(data) * (rank + 1) // world_size
    for start in range(lo, hi, batch_size):
        end = min(start + batch_size, hi)
        x = torch.from_numpy(data.chunks[start:end]).to(device=device, dtype=torch.long)
        mask = byte_mask(torch.from_numpy(data.lens[start:end]).to(device=device, dtype=torch.long), T)
        with ctx:
            codes = model.encode(x)
        n_bytes += mask.sum()
        for k in range(L):
            hist[k] += torch.bincount(codes[k].reshape(-1), minlength=cfg.codebook_size)
            with ctx:
                logits = model.decode_logits(codes[:k + 1])
            logp = F.log_softmax(logits.float(), dim=-1).gather(-1, x[..., None]).squeeze(-1)
            nll[k] -= logp[mask].double().sum()
            correct[k] += (logits.argmax(-1) == x)[mask].sum()
    if distributed():
        for t in (correct, nll, hist, n_bytes):
            dist.all_reduce(t)
    model.train(was_training)
    probs = [h[h > 0] / h.sum() for h in hist]
    return {
        "acc": (correct / n_bytes).tolist(),
        "bits_per_byte": (nll / n_bytes / math.log(2)).tolist(),
        "used": [int((h > 0).sum()) for h in hist],
        "perplexity": [float((-(p * p.log()).sum()).exp()) for p in probs],
        "bytes": int(n_bytes.item()),
    }


def format_tokenizer_report(name, r):
    acc = ", ".join(f"L0..{k}: {a * 100:.2f}%" for k, a in enumerate(r["acc"]))
    bits = ", ".join(f"L0..{k}: {b:.4f}" for k, b in enumerate(r["bits_per_byte"]))
    ppl = ", ".join(f"{p:.1f}" for p in r["perplexity"])
    return (f"{name} | {r['bytes']:,} bytes | byte accuracy [{acc}] | decoder bits/byte [{bits}] "
            f"| codes seen {r['used']} | code perplexity [{ppl}]")


# -----------------------------------------------------------------------------
# Two-part evaluation
#
# NCM gives p(codes), not p(bytes). The encoder is deterministic, so sending a
# chunk's codes and then its bytes given the codes is a valid code for the bytes:
#     bits(chunk) = -log2 p_ncm(codes) - log2 p_decoder(bytes | all levels)
# Code bits cover every code of a chunk (codes are a function of the padded chunk);
# decoder bits cover real bytes only. Where a document ends is not free either:
# each document pays an Elias-gamma code of (bytes + 1), 2*floor(log2(n + 1)) + 1 bits.
#     val_bpb = (code bits + decoder bits + length bits) / real bytes

def length_bits(doc_lens):
    """Elias-gamma bits for every document's byte count + 1 (exact for counts < 2**53)."""
    _, exponent = np.frexp(np.asarray(doc_lens, dtype=np.float64) + 1)   # = bit_length(n + 1)
    return float((2 * exponent.astype(np.int64) - 1).sum())


@torch.no_grad()
def two_part_bits(tokenizer, ncm, data, batch_size, device, ctx, rank=0, world_size=1):
    """Two-part code length of every chunk in data (split across ranks), in bits."""
    offsets = ncm.offsets
    L = len(offsets) - 1
    code_nats = torch.zeros(L, dtype=torch.float64, device=device)
    decoder_nats = torch.zeros((), dtype=torch.float64, device=device)
    tokenizer.eval()
    ncm.eval()
    T = tokenizer.cfg.seq_len
    lo, hi = len(data) * rank // world_size, len(data) * (rank + 1) // world_size
    for start in range(lo, hi, batch_size):
        end = min(start + batch_size, hi)
        x = torch.from_numpy(data.chunks[start:end]).to(device=device, dtype=torch.long)
        mask = byte_mask(torch.from_numpy(data.lens[start:end]).to(device=device, dtype=torch.long), T)
        codes = tokenizer.encode(x)
        with ctx:
            logits, _, _ = ncm(codes)
        nll = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                              torch.cat(codes, dim=1).reshape(-1), reduction="none").view(x.size(0), -1)
        for k in range(L):
            code_nats[k] += nll[:, offsets[k]:offsets[k + 1]].double().sum()
        with ctx:
            byte_logits = tokenizer.decode_logits(codes)
        logp = F.log_softmax(byte_logits.float(), dim=-1).gather(-1, x[..., None]).squeeze(-1)
        decoder_nats -= logp[mask].double().sum()
    if distributed():
        dist.all_reduce(code_nats)
        dist.all_reduce(decoder_nats)
    code_bits = (code_nats / math.log(2)).tolist()
    decoder_bits = decoder_nats.item() / math.log(2)
    len_bits = length_bits(data.doc_lens)
    total = sum(code_bits) + decoder_bits + len_bits
    return {"code_bits": code_bits, "decoder_bits": decoder_bits, "length_bits": len_bits,
            "total_bits": total, "bytes": data.num_bytes, "val_bpb": total / data.num_bytes}


def format_bits(name, r):
    per_byte = lambda bits: bits / r["bytes"]
    levels = " ".join(f"L{k}:{per_byte(b):.4f}" for k, b in enumerate(r["code_bits"]))
    return (f"{name} val_bpb:{r['val_bpb']:.4f} | code bits/byte [{levels}] "
            f"sum:{per_byte(sum(r['code_bits'])):.4f} | decoder:{per_byte(r['decoder_bits']):.4f} "
            f"| length:{per_byte(r['length_bits']):.4f} | {r['bytes']:,} bytes")


# -----------------------------------------------------------------------------
# Artifact: both int8 + zlib blobs in one file

ARTIFACT_MAGIC = b"NCM1"


def write_artifact(path, vq_cfg, ncm_cfg, tokenizer_blob, ncm_blob):
    header = json.dumps({"vq": asdict(vq_cfg), "ncm": asdict(ncm_cfg),
                         "sizes": [len(tokenizer_blob), len(ncm_blob)]}).encode()
    save_atomic(ARTIFACT_MAGIC + len(header).to_bytes(4, "little") + header + tokenizer_blob + ncm_blob, path)


def read_artifact(path):
    """-> (tokenizer, ncm) rebuilt on CPU from the artifact alone."""
    with open(path, "rb") as f:
        raw = f.read()
    if raw[:4] != ARTIFACT_MAGIC:
        raise ValueError(f"{path} is not a nanoNCM artifact")
    n = int.from_bytes(raw[4:8], "little")
    header = json.loads(raw[8:8 + n])
    a, b = header["sizes"]
    blobs = raw[8 + n:8 + n + a], raw[8 + n + a:8 + n + a + b]
    tokenizer = unpack_int8(TextVQVAE(VQConfig(**header["vq"])), blobs[0])
    ncm = unpack_int8(NCMGPT(NCMConfig(**header["ncm"])), blobs[1])
    return tokenizer, ncm


# -----------------------------------------------------------------------------
# Run

class Hyperparameters:
    def __init__(self):
        cuda = torch.cuda.is_available()
        # run
        self.data_path = env("DATA_PATH", "./data/datasets/fineweb10B_byte260")
        self.train_shards = env("TRAIN_SHARDS", 0)        # 0 = every fineweb_train_*.bin found
        self.out_dir = env("OUT_DIR", "out")              # checkpoints + frozen tokenizer
        self.log_dir = env("LOG_DIR", "logs")
        self.run_id = str(env("RUN_ID", uuid.uuid4()))
        self.seed = env("SEED", 1337)
        self.device = env("DEVICE", "cuda" if cuda else "cpu")
        self.dtype = env("DTYPE", "auto")                 # auto | float32 | bfloat16 | float16
        self.num_threads = env("NUM_THREADS", min(4, torch.get_num_threads()))
        self.stages = env("STAGES", "tokenizer,model,eval,sample")   # any subset, run in this order
        self.resume = env("RESUME", False)                # continue from checkpoints in out_dir
        self.compile = env("COMPILE", False)              # torch.compile the NCM (off: T4-friendly)
        self.max_wallclock_seconds = float(env("MAX_WALLCLOCK_SECONDS", 600.0))   # all training; 0 = no cap
        self.stage1_fraction = env("STAGE1_FRACTION", 0.35)
        self.checkpoint_seconds = env("CHECKPOINT_SECONDS", 600)                 # 0 = only at evals
        # tokenizer training (vq_tokenizer.py's settings)
        self.vq_batch_size = env("VQ_BATCH_SIZE", 32)     # chunks per step, summed over GPUs
        self.vq_max_iters = env("VQ_MAX_ITERS", 3000)
        self.vq_eval_interval = env("VQ_EVAL_INTERVAL", 250)
        self.vq_eval_iters = env("VQ_EVAL_ITERS", 5)
        self.vq_learning_rate = env("VQ_LEARNING_RATE", 1e-3)
        self.vq_warmup_iters = env("VQ_WARMUP_ITERS", 100)
        self.vq_report_batch_size = env("VQ_REPORT_BATCH_SIZE", 64)
        self.vq = VQConfig(**{
            name: env(f"VQ_{name.upper()}", field.default)
            for name, field in VQConfig.__dataclass_fields__.items() if name != "char_vocab"
        })
        known = {f"VQ_{name.upper()}" for name in VQConfig.__dataclass_fields__} | {
            "VQ_BATCH_SIZE", "VQ_MAX_ITERS", "VQ_EVAL_INTERVAL", "VQ_EVAL_ITERS",
            "VQ_LEARNING_RATE", "VQ_WARMUP_ITERS", "VQ_REPORT_BATCH_SIZE"} - {"VQ_CHAR_VOCAB"}
        unknown = sorted(k for k in os.environ if k.startswith("VQ_") and k not in known)
        if unknown:
            raise ValueError(f"unknown setting(s): {unknown}")
        # NCM training (train.py's settings; vocab_size and level_lens come from the tokenizer)
        self.batch_size = env("BATCH_SIZE", 16)           # chunks per step, summed over GPUs
        self.max_iters = env("MAX_ITERS", 1000)
        self.eval_interval = env("EVAL_INTERVAL", 100)
        self.eval_iters = env("EVAL_ITERS", 20)           # batches averaged per eval
        self.log_interval = env("LOG_INTERVAL", 10)
        self.learning_rate = env("LEARNING_RATE", 3e-4)
        self.min_lr = env("MIN_LR", 3e-5)                 # cosine decays to this
        self.warmup_iters = env("WARMUP_ITERS", 100)
        self.weight_decay = env("WEIGHT_DECAY", 1e-2)
        self.grad_clip = env("GRAD_CLIP", 1.0)            # 0 disables clipping
        self.n_embd = env("N_EMBD", 256)
        self.n_head = env("N_HEAD", 4)
        self.n_layer = env("N_LAYER", 4)
        self.dropout = env("DROPOUT", 0.0)
        self.level_loss_weighting = env("LEVEL_LOSS_WEIGHTING", "token")   # or "level"
        self.encode_batch_size = env("ENCODE_BATCH_SIZE", 256)   # chunks per encoder / eval batch
        # sampling (generate.py's settings)
        self.num_samples = env("NUM_SAMPLES", 4)
        self.temperature = env("TEMPERATURE", 0.8)
        self.top_k = env("TOP_K", 50)
        for key in ("batch_size", "eval_interval", "eval_iters", "log_interval", "encode_batch_size",
                    "n_embd", "n_head", "n_layer", "num_samples"):
            value = getattr(self, key)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{key.upper()} must be a positive integer")
        for key in ("max_iters", "warmup_iters"):
            value = getattr(self, key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{key.upper()} must be a nonnegative integer")
        for key in ("vq_batch_size", "vq_eval_interval", "vq_eval_iters", "num_threads", "vq_report_batch_size"):
            value = getattr(self, key)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{key.upper()} must be a positive integer")
        for key in ("vq_max_iters", "vq_warmup_iters"):
            value = getattr(self, key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{key.upper()} must be a nonnegative integer")
        if not 0 < self.stage1_fraction <= 1:
            raise ValueError("STAGE1_FRACTION must be in (0, 1]")


def save_atomic(obj, path):
    """Write to a temp file first so an interrupted session never leaves a broken file."""
    tmp = str(path) + ".tmp"
    if isinstance(obj, bytes):
        with open(tmp, "wb") as f:
            f.write(obj)
    else:
        torch.save(obj, tmp)
    os.replace(tmp, path)


def main() -> None:
    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    stages = {s.strip() for s in str(args.stages).split(",") if s.strip()}
    if not stages <= {"tokenizer", "model", "eval", "sample"}:
        raise ValueError(f"STAGES={args.stages}: choose from tokenizer, model, eval, sample")

    # distributed + device setup
    is_distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device_type = "cuda" if "cuda" in args.device else "cpu"
    device = torch.device("cuda", local_rank) if device_type == "cuda" else torch.device("cpu")
    if device_type == "cuda":
        torch.cuda.set_device(device)
    if is_distributed:
        dist.init_process_group(backend="nccl" if device_type == "cuda" else "gloo")
        dist.barrier()
    master_process = rank == 0
    torch.set_num_threads(args.num_threads)
    # TF32 stays off: the nearest-code search runs in fp32 and decides the codes.

    # Native bf16 needs compute capability >= 8.0. torch.cuda.is_bf16_supported()
    # also reports True for emulated bf16 (e.g. on a T4), which is slow.
    native_bf16 = device_type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8
    dtype = args.dtype
    if dtype == "auto":
        dtype = "bfloat16" if native_bf16 else ("float16" if device_type == "cuda" else "float32")
    assert dtype in ("float32", "bfloat16", "float16"), f"unknown dtype: {dtype}"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    ctx = nullcontext() if dtype == "float32" else torch.autocast(device_type=device_type, dtype=ptdtype)

    logfile = None
    if master_process:
        os.makedirs(args.log_dir, exist_ok=True)
        os.makedirs(args.out_dir, exist_ok=True)
        logfile = os.path.join(args.log_dir, f"{args.run_id}.txt")
        print(logfile)

    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg, flush=True)
        if logfile is not None:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    def barrier():
        if is_distributed:
            dist.barrier()

    def sync_max(value):
        """The same float on every rank (the max), so time-based decisions agree."""
        if not is_distributed:
            return value
        t = torch.tensor(value, dtype=torch.float64, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return float(t.item())

    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"Running Python {sys.version}", console=False)
    log0(f"Running PyTorch {torch.__version__}", console=False)
    if device_type == "cuda":
        log0(subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, check=False).stdout, console=False)
        log0(f"gpu:{torch.cuda.get_device_name(device)} x{world_size}")
    log0("=" * 100, console=False)
    log0(f"settings overridden: {sorted(overrides)}")
    log0(f"device:{device_type} world_size:{world_size} dtype:{dtype}")

    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    # data
    cfg = args.vq
    t_data = time.perf_counter()
    train, n_files = load_chunks(os.path.join(args.data_path, "fineweb_train_*.bin"), cfg.seq_len, args.train_shards)
    dev, _ = load_chunks(os.path.join(args.data_path, "fineweb_dev_*.bin"), cfg.seq_len)
    log0(f"train: {n_files} shard(s), {len(train.doc_lens):,} docs, {train.num_bytes:,} bytes, "
         f"{len(train):,} chunks ({1 - train.num_bytes / (len(train) * cfg.seq_len):.2%} padding)")
    log0(f"dev: {len(dev.doc_lens):,} docs, {dev.num_bytes:,} bytes, {len(dev):,} chunks "
         f"| data loaded in {time.perf_counter() - t_data:.1f}s")

    # -------------------------------------------------------------------------
    # Stage 1: tokenizer

    budget = args.stage1_fraction * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else math.inf
    if args.vq_batch_size % world_size:
        raise ValueError("VQ_BATCH_SIZE must be divisible by the number of GPUs")
    local_batch = args.vq_batch_size // world_size
    frozen_path = os.path.join(args.out_dir, "tokenizer.int8.ptz")
    last_path = os.path.join(args.out_dir, "vq_last.pt")
    model = TextVQVAE(cfg).to(device)
    log0(f"VQ-VAE: {sum(p.numel() for p in model.parameters()):,} parameters, levels {cfg.level_lens}, "
         f"codebook {cfg.codebook_size}, prefix loss weight {cfg.prefix_loss_weight}, "
         f"balanced EMA {cfg.balance_levels}")
    log0(f"stage 1 budget: {f'{budget:.0f}s' if math.isfinite(budget) else 'no cap on'} training time, at most {args.vq_max_iters} iters "
         f"of {args.vq_batch_size} chunks")
    stage1_seconds = None
    trained_tokenizer = False

    if os.path.exists(frozen_path) and (args.resume or "tokenizer" not in stages):
        log0(f"resume: stage 1 already finished, loading {frozen_path}")
    elif "tokenizer" not in stages:
        raise FileNotFoundError(f"{frozen_path} is missing; include 'tokenizer' in STAGES")
    else:
        trained_tokenizer = True
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.vq_learning_rate, weight_decay=0.01)
        scaler = torch.amp.GradScaler(device_type, enabled=(dtype == "float16"))
        train_generator = torch.Generator().manual_seed(args.seed + rank)
        start_iter, train_seconds = 0, 0.0
        if args.resume and os.path.exists(last_path):
            ckpt = torch.load(last_path, map_location=device, weights_only=True)
            if ckpt["config"] != asdict(cfg) or ckpt["world_size"] != world_size:
                raise ValueError("resume: vq_last.pt has a different tokenizer config or GPU count")
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scaler.load_state_dict(ckpt["scaler"])
            train_generator.set_state(ckpt["generators"][rank])
            start_iter, train_seconds = ckpt["iter"], ckpt["train_seconds"]
            log0(f"resume: tokenizer from {last_path} at iter {start_iter} ({train_seconds:.0f}s trained)")
            del ckpt
        ddp_model = DDP(model, device_ids=[local_rank] if device_type == "cuda" else None,
                        broadcast_buffers=False) if is_distributed else model

        def get_batch(src, generator, n):
            ix = torch.randint(0, len(src), (n,), generator=generator).numpy()
            x = torch.from_numpy(src.chunks[ix]).to(dtype=torch.long)
            lens = torch.from_numpy(src.lens[ix]).to(dtype=torch.long)
            return x.to(device, non_blocking=True), lens.to(device, non_blocking=True)

        def get_lr(it, progress):
            if it < args.vq_warmup_iters:
                return args.vq_learning_rate * (it + 1) / args.vq_warmup_iters
            # cosine over iterations, or over the time budget if that runs out first
            ratio = max((it - args.vq_warmup_iters) / max(1, args.vq_max_iters - args.vq_warmup_iters), progress)
            ratio = min(ratio, 1.0)
            return 0.1 * args.vq_learning_rate + 0.45 * args.vq_learning_rate * (1 + math.cos(math.pi * ratio))

        @torch.no_grad()
        def evaluate():
            model.eval()
            # Reset this private generator so every evaluation uses the same dev chunks
            # without consuming training RNG or mutating the codebook.
            generator = torch.Generator().manual_seed(args.seed + 1)
            hist = [torch.zeros(cfg.codebook_size, device=device) for _ in cfg.level_lens]
            rec_total, acc_total, commit_total = 0.0, 0.0, 0.0
            prefix_acc = [0.0] * (len(cfg.level_lens) - 1)
            prefix_rec = [0.0] * (len(cfg.level_lens) - 1)
            for _ in range(args.vq_eval_iters):
                x, lens = get_batch(dev, generator, args.vq_batch_size)
                mask = byte_mask(lens, cfg.seq_len)
                with ctx:
                    logits, _, _, commit, codes = model(x)
                rec_total += F.cross_entropy(logits.float()[mask], x[mask]).item()
                commit_total += commit.item()
                acc_total += (logits.argmax(-1) == x)[mask].float().mean().item()
                prefix = torch.zeros(x.size(0), cfg.seq_len, cfg.code_dim, device=device)
                for k, c in enumerate(codes):
                    hist[k] += torch.bincount(c.reshape(-1), minlength=cfg.codebook_size)
                    if k < len(codes) - 1:
                        prefix = prefix + model._up(model.codebook.vectors(c), cfg.seq_len)
                        with ctx:
                            partial_logits = model.decode_features(prefix).float()
                        prefix_acc[k] += (partial_logits.argmax(-1) == x)[mask].float().mean().item()
                        prefix_rec[k] += F.cross_entropy(partial_logits[mask], x[mask]).item()
            used = [int((h > 0).sum()) for h in hist]
            probs = [h[h > 0] / h.sum() for h in hist]
            perplexity = [float((-(p * p.log()).sum()).exp()) for p in probs]
            model.train()
            return {
                "rec": rec_total / args.vq_eval_iters, "acc": acc_total / args.vq_eval_iters,
                "commit": commit_total / args.vq_eval_iters, "used": used, "perplexity": perplexity,
                "prefix_acc": [a / args.vq_eval_iters for a in prefix_acc],
                "prefix_rec": [r / args.vq_eval_iters for r in prefix_rec],
            }

        def save_checkpoint(it):
            generators = [train_generator.get_state()]
            if is_distributed:
                generators = [None] * world_size
                dist.all_gather_object(generators, train_generator.get_state())
            if master_process:
                save_atomic({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                             "scaler": scaler.state_dict(), "config": asdict(cfg), "iter": it,
                             "train_seconds": train_seconds, "world_size": world_size,
                             "generators": generators}, last_path)
            barrier()

        model.train()
        it = start_iter
        last_save = train_seconds
        if device_type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        while True:
            elapsed = sync_max(train_seconds + time.perf_counter() - t0)
            done = it >= args.vq_max_iters or elapsed >= budget
            if (it % args.vq_eval_interval == 0 or done) and not (args.resume and it == start_iter and it > 0):
                train_seconds = elapsed
                metrics = evaluate()
                pa = ", ".join(f"L0..{k}: {a * 100:.1f}%" for k, a in enumerate(metrics["prefix_acc"]))
                ppl = ", ".join(f"{p:.1f}" for p in metrics["perplexity"])
                log0(f"vq step {it:5d} | dev rec loss {metrics['rec']:.4f} | "
                     f"byte accuracy {metrics['acc'] * 100:5.1f}% ({pa}) "
                     f"| codes seen {metrics['used']} | code perplexity [{ppl}] "
                     f"| train_time {train_seconds:.0f}s")
                save_checkpoint(it)
                last_save = train_seconds
                t0 = time.perf_counter()      # eval and saving are not training time
            elif args.checkpoint_seconds and elapsed - last_save >= args.checkpoint_seconds:
                train_seconds = elapsed
                save_checkpoint(it)
                last_save = train_seconds
                log0(f"vq step {it:5d} | checkpoint saved to {last_path}")
                t0 = time.perf_counter()
            if done:
                if it < args.vq_max_iters:
                    log0(f"stopping_early: wallclock budget train_time:{train_seconds:.0f}s step:{it}/{args.vq_max_iters}")
                break
            lr = get_lr(it, elapsed / budget if math.isfinite(budget) else 0.0)
            for g in optimizer.param_groups:
                g["lr"] = lr
            x, _ = get_batch(train, train_generator, local_batch)
            with ctx:
                _, loss, rec, commit, _ = ddp_model(x)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            it += 1
            if it <= 10 or it % 100 == 0:
                log0(f"vq iter {it:5d} | loss {loss.item():.4f} | rec {rec.item():.4f} | commit {commit.item():.4f} "
                     f"| lr {lr:.2e} | train_time {train_seconds + time.perf_counter() - t0:.0f}s")
        stage1_seconds = train_seconds
        if device_type == "cuda":
            log0(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")
        report = tokenizer_report(model, dev, args.vq_report_batch_size, device, ctx, rank, world_size)
        log0(format_tokenizer_report("tokenizer dev (trained, before int8)", report))

        # Freeze = the artifact's int8 + zlib round trip, done now so that stage 2 is
        # trained on exactly the codes the submitted tokenizer will produce.
        blob, qstats = pack_int8(model)
        if master_process:
            save_atomic(blob, frozen_path)
            log0(f"tokenizer frozen to {frozen_path}: int8+zlib {len(blob):,} bytes "
                 f"(payload {qstats['int8_payload_bytes']:,}, {qstats['param_count']:,} values)")
        barrier()
        del optimizer, ddp_model

    del model
    with open(frozen_path, "rb") as f:
        tokenizer_blob = f.read()
    tokenizer = unpack_int8(TextVQVAE(cfg), tokenizer_blob).to(device).eval().requires_grad_(False)
    tokenizer_sha256 = hashlib.sha256(tokenizer_blob).hexdigest()
    log0(f"frozen tokenizer: {len(tokenizer_blob):,} bytes, sha256 {tokenizer_sha256}")
    if stage1_seconds is None:
        stage1_seconds = 0.0
        if os.path.exists(last_path):
            stage1_seconds = torch.load(last_path, map_location="cpu", weights_only=True)["train_seconds"]
        else:
            log0(f"WARNING: no {last_path}; stage 1 training time unknown, counted as 0")
    if trained_tokenizer:
        report = tokenizer_report(tokenizer, dev, args.vq_report_batch_size, device, ctx, rank, world_size)
        log0(format_tokenizer_report("tokenizer dev (frozen, after int8)", report))
        # one reconstruction, as vq_tokenizer.py printed
        if master_process:
            x = torch.from_numpy(dev.chunks[:1]).to(device=device, dtype=torch.long)
            n = int(dev.lens[0])
            codes = tokenizer.encode(x)
            log0("\n--- original ---\n" + bytes(dev.chunks[0, :n]).decode("utf-8", errors="replace"))
            for k in range(len(codes)):
                recon = tokenizer.decode(codes[:k + 1])[0, :n].tolist()
                log0(f"\n--- reconstructed from levels 0..{k} ---\n" + bytes(recon).decode("utf-8", errors="replace"))

    # -------------------------------------------------------------------------
    # Stage 2: NCM on the frozen tokenizer's codes (train.py)

    config = NCMConfig(
        vocab_size=cfg.codebook_size,
        level_lens=cfg.level_lens,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        dropout=args.dropout,
        level_loss_weighting=args.level_loss_weighting,
    )
    artifact_path = os.path.join(args.out_dir, "final_model.int8.ptz")
    ncm_last_path = os.path.join(args.out_dir, "last.pt")
    ncm_best_path = os.path.join(args.out_dir, "ckpt.pt")
    code_bytes = len(code.encode("utf-8"))

    if "model" in stages and args.resume and os.path.exists(artifact_path):
        log0(f"resume: stage 2 already finished, using {artifact_path}")
    elif "model" in stages:
        if args.batch_size % world_size:
            raise ValueError("BATCH_SIZE must be divisible by the number of GPUs")
        local_batch = args.batch_size // world_size
        budget = args.max_wallclock_seconds - stage1_seconds if args.max_wallclock_seconds > 0 else math.inf
        log0(f"stage 2 budget: {f'{budget:.0f}s' if math.isfinite(budget) else 'no cap on'} training time "
             f"(encoding included), at most {args.max_iters} iters of {args.batch_size} chunks")

        # Encode the training data in memory with the frozen tokenizer (data/prepare_ncm.py):
        # aligned uint16 arrays, one per level; row i of every level is the same chunk.
        def encode_all(data, lo, hi):
            arrays = [np.empty((hi - lo, n), dtype=np.uint16) for n in cfg.level_lens]
            for start in range(lo, hi, args.encode_batch_size):
                end = min(start + args.encode_batch_size, hi)
                x = torch.from_numpy(data.chunks[start:end]).to(device=device, dtype=torch.long)
                for array, c in zip(arrays, tokenizer.encode(x)):
                    array[start - lo:end - lo] = c.cpu().numpy().astype(np.uint16)
            return arrays

        if device_type == "cuda":
            torch.cuda.synchronize()
        t_encode = time.perf_counter()
        data = {"train": encode_all(train, len(train) * rank // world_size, len(train) * (rank + 1) // world_size),
                "dev": encode_all(dev, 0, len(dev))}
        encode_seconds = sync_max(time.perf_counter() - t_encode)
        log0(f"encoded {len(train):,} train chunks (this GPU: {data['train'][0].shape[0]:,}) and "
             f"{len(dev):,} dev chunks in {encode_seconds:.1f}s")

        def get_batch(split):
            """Returns a list of K LongTensors, codes[k] of shape (local batch, level_lens[k])."""
            arrays = data[split]
            ix = np.random.randint(0, arrays[0].shape[0], size=local_batch)
            codes = [torch.from_numpy(a[ix].astype(np.int64)) for a in arrays]   # same rows across levels
            if device_type == "cuda":
                return [c.pin_memory().to(device, non_blocking=True) for c in codes]
            return [c.to(device) for c in codes]

        log0(f"Setting up model on {device} ({dtype}{', GradScaler' if dtype == 'float16' else ''})...")
        ncm = NCMGPT(config).to(device)
        optimizer = ncm.configure_optimizers(args.weight_decay, args.learning_rate, (0.9, 0.95), device_type)
        # fp16 has a narrow exponent range, so its gradients need loss scaling
        scaler = torch.amp.GradScaler(device_type, enabled=(dtype == "float16"))
        start_iter, train_seconds, best_val_loss = 0, 0.0, float("inf")
        if args.resume and os.path.exists(ncm_last_path):
            checkpoint = torch.load(ncm_last_path, map_location=device, weights_only=True)
            if checkpoint["tokenizer_sha256"] != tokenizer_sha256:
                raise ValueError("resume: checkpoint was trained on codes from a different tokenizer")
            if checkpoint["config"] != asdict(config) or checkpoint["world_size"] != world_size:
                raise ValueError("resume: last.pt has a different NCM config or GPU count")
            ncm.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scaler.load_state_dict(checkpoint["scaler"])
            rng = checkpoint["rng"][rank]
            torch.set_rng_state(rng["torch"].cpu())
            np.random.set_state(("MT19937", np.array(rng["numpy"], dtype=np.uint32), *rng["numpy_pos"]))
            start_iter, train_seconds = checkpoint["iter"], checkpoint["train_seconds"]
            best_val_loss = checkpoint["best_val_loss"]
            log0(f"resume: NCM from {ncm_last_path} at iter {start_iter} ({train_seconds:.0f}s trained, "
                 f"best dev {best_val_loss:.4f})")
            checkpoint = None             # free memory
        train_seconds += encode_seconds   # every encoding pass is training compute
        log0(f"NCM: {sum(p.numel() for p in ncm.parameters()):,} parameters, n_layer {config.n_layer}, "
             f"n_head {config.n_head}, n_embd {config.n_embd}, vocab {config.vocab_size}")
        model = torch.compile(ncm) if args.compile else ncm
        if is_distributed:
            model = DDP(model, device_ids=[local_rank] if device_type == "cuda" else None)

        def get_lr(it, progress):
            """Linear warmup, then cosine decay to min_lr (over iterations, or the time budget if shorter)."""
            if it < args.warmup_iters:
                return args.learning_rate * (it + 1) / (args.warmup_iters + 1)
            ratio = max((it - args.warmup_iters) / max(1, args.max_iters - args.warmup_iters), progress)
            return args.min_lr + 0.5 * (1.0 + math.cos(math.pi * min(ratio, 1.0))) * (args.learning_rate - args.min_lr)

        @torch.no_grad()
        def estimate_loss():
            """Average loss (and per-level losses) over eval_iters batches for each split."""
            ncm.eval()
            out = {}
            for split in ("train", "dev"):
                total = torch.zeros(1 + config.num_levels, device=device)
                for _ in range(args.eval_iters):
                    with ctx:
                        _, loss, levels = ncm(get_batch(split))
                    total += torch.stack([loss.float()] + [l.float() for l in levels])
                if is_distributed:
                    dist.all_reduce(total)
                out[split] = (total / (args.eval_iters * world_size)).cpu()
            ncm.train()
            return out

        def save_checkpoint(path, it):
            """Write to a temp file first so an interrupted session never leaves a broken checkpoint."""
            np_state = np.random.get_state()
            rng = {"torch": torch.get_rng_state(), "numpy": np_state[1].tolist(), "numpy_pos": list(np_state[2:])}
            rngs = [rng]
            if is_distributed:
                rngs = [None] * world_size
                dist.all_gather_object(rngs, rng)
            if master_process:
                save_atomic({
                    "model": ncm.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "config": asdict(config),
                    "iter": it,
                    "train_seconds": train_seconds,
                    "best_val_loss": best_val_loss,
                    "tokenizer_sha256": tokenizer_sha256,
                    "world_size": world_size,
                    "rng": rngs,
                }, path)
            barrier()

        log0("Starting training...")
        ncm.train()
        it = start_iter
        last_save = train_seconds
        iters_since_log = 0
        if device_type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        t_log = t0
        while True:
            elapsed = sync_max(train_seconds + time.perf_counter() - t0)
            done = it >= args.max_iters or elapsed >= budget
            # evaluation + checkpoint (skipped on the step we resumed at: it was evaluated before saving)
            if (it % args.eval_interval == 0 or done) and not (args.resume and it == start_iter and it > 0):
                train_seconds = elapsed
                t_eval = time.perf_counter()
                losses = estimate_loss()
                lv = ", ".join(f"L{k}:{v:.3f}" for k, v in enumerate(losses["dev"][1:].tolist()))
                log0(f"step {it:5d} | train {losses['train'][0]:.4f} | dev {losses['dev'][0]:.4f} | "
                     f"dev levels [{lv}] | eval {(time.perf_counter() - t_eval) * 1000:.0f} ms "
                     f"| train_time {train_seconds:.0f}s")
                if losses["dev"][0] < best_val_loss:
                    best_val_loss = losses["dev"][0].item()
                    save_checkpoint(ncm_best_path, it)
                save_checkpoint(ncm_last_path, it)
                last_save = train_seconds
                t0 = t_log = time.perf_counter()   # eval and saving are not training time
                iters_since_log = 0
            elif args.checkpoint_seconds and elapsed - last_save >= args.checkpoint_seconds:
                train_seconds = elapsed
                save_checkpoint(ncm_last_path, it)
                last_save = train_seconds
                log0(f"step {it:5d} | checkpoint saved to {ncm_last_path}")
                t0 = t_log = time.perf_counter()
                iters_since_log = 0
            if done:
                if it < args.max_iters:
                    log0(f"stopping_early: wallclock_cap train_time:{train_seconds:.0f}s step:{it}/{args.max_iters}")
                break

            lr = get_lr(it, elapsed / budget if math.isfinite(budget) else 0.0)
            for group in optimizer.param_groups:
                group["lr"] = lr
            codes = get_batch("train")
            with ctx:
                _, loss, _ = model(codes)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(ncm.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            it += 1
            iters_since_log += 1
            # logging (loss.item() syncs the GPU, so the timing is real)
            if it % args.log_interval == 0:
                lossf = loss.item()
                dt = (time.perf_counter() - t_log) / iters_since_log
                t_log = time.perf_counter()
                iters_since_log = 0
                log0(f"iter {it:5d} | loss {lossf:.4f} | lr {lr:.2e} | {dt * 1000:.1f} ms/iter")

        log0(f"training_time total:{stage1_seconds + train_seconds:.0f}s (tokenizer {stage1_seconds:.0f}s, "
             f"encoding {encode_seconds:.0f}s, NCM {train_seconds - encode_seconds:.0f}s) | best dev {best_val_loss:.4f}")
        if device_type == "cuda":
            log0(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
                 f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB")

        # Serialization: the artifact holds the frozen tokenizer blob unchanged and the
        # NCM's final weights as int8 + zlib, like the baseline's final_model.int8.ptz.
        ncm_blob, qstats = pack_int8(ncm)
        if master_process:
            write_artifact(artifact_path, cfg, config, tokenizer_blob, ncm_blob)
            log0(f"Serialized NCM int8+zlib: {len(ncm_blob):,} bytes (payload {qstats['int8_payload_bytes']:,})")
        barrier()
        del data, model, ncm, optimizer

    if os.path.exists(artifact_path):
        artifact_bytes = os.path.getsize(artifact_path)
        log0(f"Serialized model int8+zlib: {artifact_bytes:,} bytes (tokenizer {len(tokenizer_blob):,} + NCM + header)")
        log0(f"Code size: {code_bytes:,} bytes")
        log0(f"Total submission size int8+zlib: {artifact_bytes + code_bytes:,} bytes "
             f"({'within' if artifact_bytes + code_bytes <= 16_000_000 else 'OVER'} the 16,000,000-byte limit)")

    # -------------------------------------------------------------------------
    # Evaluation after the artifact round trip, and samples

    if stages & {"eval", "sample"}:
        if not os.path.exists(artifact_path):
            raise FileNotFoundError(f"{artifact_path} is missing; include 'model' in STAGES")
        tok_rt, ncm_rt = read_artifact(artifact_path)
        tok_rt, ncm_rt = tok_rt.to(device).eval(), ncm_rt.to(device).eval()

    if "eval" in stages:
        # The val split is read here and nowhere else.
        val, n_val_files = load_chunks(os.path.join(args.data_path, "fineweb_val_*.bin"), cfg.seq_len)
        log0(f"val: {n_val_files} file(s), {len(val.doc_lens):,} docs, {val.num_bytes:,} bytes, {len(val):,} chunks")
        if device_type == "cuda":
            torch.cuda.synchronize()
        t_eval = time.perf_counter()
        result = two_part_bits(tok_rt, ncm_rt, val, args.encode_batch_size, device, ctx, rank, world_size)
        eval_seconds = time.perf_counter() - t_eval
        log0(format_bits("final_int8_zlib_roundtrip", result) + f" | eval_time:{eval_seconds * 1000:.0f}ms")
        log0(f"final_int8_zlib_roundtrip_exact val_bpb:{result['val_bpb']:.8f}")
        if master_process:
            with open(os.path.join(args.out_dir, "eval.json"), "w") as f:
                json.dump({**result, "eval_seconds": eval_seconds, "bytes_code": code_bytes,
                           "bytes_model_int8_zlib": os.path.getsize(artifact_path),
                           "bytes_total": os.path.getsize(artifact_path) + code_bytes,
                           "tokenizer_sha256": tokenizer_sha256, "world_size": world_size,
                           "gpu": torch.cuda.get_device_name(device) if device_type == "cuda" else "cpu"}, f, indent=2)

    if "sample" in stages and master_process:
        # generate.py: sample codes level by level, then decode them to bytes
        generator = torch.Generator(device=device).manual_seed(args.seed)
        codes = ncm_rt.generate(args.num_samples, args.temperature, args.top_k, generator=generator)
        for i, row in enumerate(tok_rt.decode(codes).cpu().tolist()):
            log0(f"\n--- sample {i + 1} (temperature {args.temperature}, top_k {args.top_k}) ---\n"
                 + bytes(row).decode("utf-8", errors="replace"))

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
