"""Focused tokenizer checks (ported from nanoNCM). Run: python -m unittest discover -s tests -v"""

from copy import deepcopy
from pathlib import Path
import sys
import unittest

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_gpt import EMACodebook, TextVQVAE, VQConfig, pack_int8, unpack_int8


class TokenizerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(2)
        self.cfg = VQConfig(
            char_vocab=7, seq_len=16, level_lens=(2, 4, 16),
            codebook_size=32, code_dim=4, n_embd=16, n_head=2, n_layer=1,
        )
        self.model = TextVQVAE(self.cfg)
        self.x = torch.randint(7, (2, 16))

    def test_byte_vocab_default(self):
        self.assertEqual(VQConfig().char_vocab, 256)

    def test_initialization_precedes_first_lookup(self):
        original = self.model.codebook.lookup
        calls = []

        def checked_lookup(features):
            self.assertTrue(self.model.codebook.initted.item())
            calls.append(features.shape[1])
            return original(features)

        self.model.codebook.lookup = checked_lookup
        self.model(self.x)
        self.assertEqual(calls, [2, 4, 16])

    def test_equal_ema_mass_per_level(self):
        residuals = [torch.randn(2, n, 4) for n in self.cfg.level_lens]
        _, weights = self.model._codebook_batch(residuals)
        masses = [part.sum() for part in weights.split([2 * n for n in self.cfg.level_lens])]
        torch.testing.assert_close(torch.stack(masses), torch.full((3,), weights.sum() / 3))

    def test_ema_uses_weighted_counts_and_sums(self):
        book = EMACodebook(2, 1, decay=0.0, dead_code_threshold=0.0)
        flat = torch.tensor([[2.0], [4.0], [10.0]])
        weights = torch.tensor([0.5, 0.5, 2.0])
        book.initialize(flat, weights)
        book.update(flat, torch.tensor([0, 0, 1]), weights)
        torch.testing.assert_close(book.cluster_size, torch.tensor([1.0, 2.0]))
        torch.testing.assert_close(book.embed_avg, torch.tensor([[3.0], [20.0]]))
        torch.testing.assert_close(book.embed, torch.tensor([[3.0], [10.0]]), atol=1e-4, rtol=1e-4)

    def test_prefix_loss_and_gradients(self):
        # Start with an initialized codebook so both comparisons share its exact snapshot.
        self.model(self.x)
        plain = deepcopy(self.model)
        plain.cfg.prefix_loss_weight = 0.0
        f = self.model.encode_features(self.x)
        codes, f_hat, _, prefixes = self.model.quantize(f.detach(), return_prefixes=True)
        with torch.no_grad():
            expected = [F.cross_entropy(
                self.model.decode_features(p).reshape(-1, 7), self.x.reshape(-1)
            ) for p in prefixes]
        _, loss, rec, commit, _ = self.model(self.x)
        torch.testing.assert_close(rec, expected[-1])
        torch.testing.assert_close(
            loss, rec + self.cfg.beta * commit + self.cfg.prefix_loss_weight * torch.stack(expected[:-1]).mean()
        )
        loss.backward()
        plain(self.x)[1].backward()
        self.assertTrue(torch.isfinite(self.model.tok.weight.grad).all())
        self.assertGreater((self.model.tok.weight.grad - plain.tok.weight.grad).abs().max().item(), 1e-6)

    def test_eval_roundtrip_and_no_codebook_mutation(self):
        self.model(self.x)
        self.model.eval()
        before = {k: v.clone() for k, v in self.model.codebook.state_dict().items()}
        with torch.no_grad():
            logits, _, _, _, codes = self.model(self.x)
            encoded = self.model.encode(self.x)
            self.assertEqual([tuple(c.shape) for c in codes], [(2, 2), (2, 4), (2, 16)])
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(codes, encoded)))
            torch.testing.assert_close(self.model.decode(codes), logits.argmax(-1))
            torch.testing.assert_close(self.model.decode_logits(codes), logits)
            for k in (1, 2):
                prefix = sum(self.model._up(self.model.codebook.vectors(c), 16) for c in codes[:k])
                torch.testing.assert_close(self.model.decode(codes[:k]), self.model.decode_features(prefix).argmax(-1))
        for key, value in self.model.codebook.state_dict().items():
            torch.testing.assert_close(value, before[key])

    def test_int8_freeze_roundtrip(self):
        # Replaces nanoNCM's legacy-checkpoint test: the artifact is the int8 + zlib blob,
        # and the frozen tokenizer used for stage 2 must be exactly the reloaded artifact.
        self.model(self.x)
        self.model.eval()
        blob, _ = pack_int8(self.model)
        frozen = unpack_int8(TextVQVAE(self.cfg), blob).eval()
        again = unpack_int8(TextVQVAE(self.cfg), blob).eval()
        for a, b in zip(frozen.encode(self.x), again.encode(self.x)):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(frozen.decode_logits(frozen.encode(self.x)),
                                   again.decode_logits(again.encode(self.x)), rtol=0, atol=0)

    def test_codes_ignore_autocast(self):
        self.model(self.x)
        self.model.eval()
        expected = self.model.encode(self.x)
        with torch.no_grad():
            features = self.model.encode_features(self.x)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            codes, f_hat, _ = self.model.quantize(features)
            encoded = self.model.encode(self.x)
        self.assertEqual(f_hat.dtype, torch.float32)
        for a, b, c in zip(codes, expected, encoded):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            torch.testing.assert_close(c, b, rtol=0, atol=0)

    def test_single_level_and_invalid_config(self):
        model = TextVQVAE(VQConfig(
            char_vocab=7, seq_len=16, level_lens=(8,), codebook_size=16,
            code_dim=4, n_embd=16, n_head=2, n_layer=1,
        ))
        logits, loss, _, _, codes = model(self.x)
        self.assertEqual(tuple(logits.shape), (2, 16, 7))
        self.assertEqual(len(codes), 1)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for kwargs in ({"level_lens": (8, 4)}, {"n_embd": 17}, {"prefix_loss_weight": -1}, {"ema_decay": 1}):
            with self.assertRaises(ValueError):
                VQConfig(**kwargs)


if __name__ == "__main__":
    unittest.main()
