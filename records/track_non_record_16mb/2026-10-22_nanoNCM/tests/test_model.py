"""nanoNCM model.py's built-in sanity checks, as unit tests on the ported NCMGPT."""

from dataclasses import replace
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_gpt import NCMConfig, NCMGPT


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        torch.set_num_threads(2)
        self.cfg = NCMConfig(vocab_size=64, level_lens=(4, 9, 23), n_embd=32, n_head=4, n_layer=2, dropout=0.2)
        self.model = NCMGPT(self.cfg).eval()
        self.codes = [torch.randint(self.cfg.vocab_size, (2, n)) for n in self.cfg.level_lens]

    def test_shape_leakage_and_cached_logits(self):
        cfg, model, codes = self.cfg, self.model, self.codes
        with torch.no_grad():
            logits, loss, _ = model(codes)
            self.assertEqual(logits.shape, (2, cfg.total_len, cfg.vocab_size))
            self.assertTrue(torch.isfinite(loss))
            for k in range(cfg.num_levels):
                perturbed = [c.clone() for c in codes]
                perturbed[k] = (perturbed[k] + 1) % cfg.vocab_size
                changed, _, _ = model(perturbed)
                end = model.offsets[k + 1]
                torch.testing.assert_close(logits[:, :end], changed[:, :end])
                if k + 1 < cfg.num_levels:
                    self.assertFalse(torch.allclose(logits[:, end:], changed[:, end:]))

            caches = [None] * cfg.n_layer
            embeddings = [model.transformer.wte(c) for c in codes[:-1]]
            for k in range(cfg.num_levels):
                x = model._level_input(embeddings, k, 2)
                actual = model._trunk_cached(x, k, caches)
                expected = logits[:, model.offsets[k]:model.offsets[k + 1]]
                torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)

    def test_generation_and_mode_restoration(self):
        model = self.model
        model.train()
        model.transformer.h[0].mlp.eval()  # Also verify mixed module modes survive.
        modes = [module.training for module in model.modules()]
        cached = model.generate(2, temperature=0)
        uncached = model.generate(2, temperature=0, use_cache=False)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(cached, uncached)))
        sampled = model.generate(2, top_k=5, generator=torch.Generator().manual_seed(1))
        self.assertEqual([tuple(c.shape) for c in sampled], [(2, n) for n in self.cfg.level_lens])
        self.assertEqual([module.training for module in model.modules()], modes)
        model(sampled)[1].backward()

    def test_one_item_overfit(self):
        # One item avoids conflicting unconditional level-0 targets across a batch.
        fit = NCMGPT(replace(self.cfg, dropout=0.0)).train()
        batch = [c[:1] for c in self.codes]
        opt = fit.configure_optimizers(0.0, 1e-2, (0.9, 0.95), "cpu")
        for _ in range(200):
            opt.zero_grad(set_to_none=True)
            _, loss, _ = fit(batch)
            loss.backward()
            opt.step()
        with torch.no_grad():
            _, loss, _ = fit(batch)
        self.assertLess(loss.item(), 0.05, f"one-item overfit failed: {loss.item():.4f}")


if __name__ == "__main__":
    unittest.main()
