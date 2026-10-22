"""Two-part bits/byte: the batched evaluation must match a slow per-chunk reference."""

import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_gpt import (BOS_ID, BYTE_OFFSET, Chunks, NCMConfig, NCMGPT, TextVQVAE, VQConfig,
                       length_bits, pack_int8, read_artifact, two_part_bits, write_artifact)

SEQ_LEN = 16


def tiny_models(seed=3):
    torch.manual_seed(seed)
    vq = VQConfig(seq_len=SEQ_LEN, level_lens=(2, 4, 16), codebook_size=32, code_dim=4,
                  n_embd=16, n_head=2, n_layer=1)
    tokenizer = TextVQVAE(vq)
    tokenizer(torch.randint(256, (8, SEQ_LEN)))       # data-based codebook init + one EMA update
    ncm = NCMGPT(NCMConfig(vocab_size=32, level_lens=vq.level_lens, n_embd=16, n_head=2, n_layer=1))
    return tokenizer.eval(), ncm.eval(), vq


def tiny_data(seed=4):
    rng = np.random.default_rng(seed)
    docs = [rng.integers(0, 256, size=n, dtype=np.uint8).tobytes() for n in (0, 1, 15, 16, 17, 40, 33, 5)]
    tokens = np.concatenate([[BOS_ID] + [b + BYTE_OFFSET for b in d] for d in docs]).astype(np.uint16)
    return Chunks.from_tokens(tokens, SEQ_LEN), docs


def reference_bpb(tokenizer, ncm, docs):
    """One document and one chunk at a time, straight from the definition, in float64."""
    total_bits, total_bytes = 0.0, 0
    for doc in docs:
        total_bits += 2 * (len(doc) + 1).bit_length() - 1          # Elias-gamma(len + 1)
        total_bytes += len(doc)
        for start in range(0, len(doc), SEQ_LEN):
            piece = doc[start:start + SEQ_LEN]
            x = torch.zeros(1, SEQ_LEN, dtype=torch.long)
            x[0, :len(piece)] = torch.tensor(list(piece))
            codes = tokenizer.encode(x)
            with torch.no_grad():
                logits, _, _ = ncm(codes)
                byte_logits = tokenizer.decode_logits(codes)
            code_logp = torch.log_softmax(logits[0].double(), dim=-1)
            for j, target in enumerate(torch.cat(codes, dim=1)[0].tolist()):
                total_bits -= code_logp[j, target].item() / math.log(2)
            byte_logp = torch.log_softmax(byte_logits[0].double(), dim=-1)
            for j, b in enumerate(piece):                         # padding is never scored
                total_bits -= byte_logp[j, b].item() / math.log(2)
    return total_bits / total_bytes


class TwoPartEvalTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.tokenizer, self.ncm, self.vq = tiny_models()
        self.data, self.docs = tiny_data()

    def test_matches_slow_reference(self):
        expected = reference_bpb(self.tokenizer, self.ncm, self.docs)
        for batch_size in (1, 3, 64):
            result = two_part_bits(self.tokenizer, self.ncm, self.data, batch_size, "cpu", torch.no_grad())
            self.assertAlmostEqual(result["val_bpb"], expected, delta=1e-6, msg=f"batch_size={batch_size}")
        self.assertEqual(result["bytes"], sum(len(d) for d in self.docs))

    def test_rank_split_covers_every_chunk_once(self):
        full = two_part_bits(self.tokenizer, self.ncm, self.data, 4, "cpu", torch.no_grad())
        parts = [two_part_bits(self.tokenizer, self.ncm, self.data, 4, "cpu", torch.no_grad(), rank, 3)
                 for rank in range(3)]
        for k in range(3):
            self.assertAlmostEqual(sum(p["code_bits"][k] for p in parts), full["code_bits"][k], delta=1e-6)
        self.assertAlmostEqual(sum(p["decoder_bits"] for p in parts), full["decoder_bits"], delta=1e-6)

    def test_length_bits_exact(self):
        for n in (0, 1, 2, 3, 254, 255, 256, 2**20 - 1, 2**20, 10**9):
            self.assertEqual(length_bits([n]), 2 * (n + 1).bit_length() - 1)

    def test_artifact_roundtrip_and_tied_weights_stored_once(self):
        tokenizer_blob, _ = pack_int8(self.tokenizer)
        ncm_blob, stats = pack_int8(self.ncm)
        self.assertEqual(stats["param_count"], sum(p.numel() for p in self.ncm.parameters()))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "final_model.int8.ptz"
            write_artifact(path, self.vq, self.ncm.config, tokenizer_blob, ncm_blob)
            tok_a, ncm_a = read_artifact(path)
            tok_b, ncm_b = read_artifact(path)
        self.assertIs(ncm_a.transformer.wte.weight, ncm_a.lm_head.weight)
        a = two_part_bits(tok_a.eval(), ncm_a.eval(), self.data, 4, "cpu", torch.no_grad())
        b = two_part_bits(tok_b.eval(), ncm_b.eval(), self.data, 4, "cpu", torch.no_grad())
        self.assertEqual(a["val_bpb"], b["val_bpb"])                  # loading is deterministic
        self.assertLess(abs(a["val_bpb"] - two_part_bits(self.tokenizer, self.ncm, self.data, 4, "cpu",
                                                         torch.no_grad())["val_bpb"]), 0.5)


if __name__ == "__main__":
    unittest.main()
