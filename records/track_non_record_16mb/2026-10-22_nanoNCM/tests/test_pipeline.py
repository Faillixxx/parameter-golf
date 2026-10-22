"""Byte data -> chunks -> stage-1 training (single process and torchrun) on tiny synthetic docs."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train_gpt import BOS_ID, BYTE_OFFSET, Chunks, EMACodebook, load_chunks, load_data_shard

TINY = {
    "DEVICE": "cpu", "NUM_THREADS": "2", "MAX_WALLCLOCK_SECONDS": "0",
    "VQ_SEQ_LEN": "16", "VQ_LEVEL_LENS": "(2,4,16)", "VQ_CODEBOOK_SIZE": "32", "VQ_CODE_DIM": "4",
    "VQ_N_EMBD": "16", "VQ_N_HEAD": "2", "VQ_N_LAYER": "1", "VQ_BATCH_SIZE": "4",
    "VQ_EVAL_INTERVAL": "2", "VQ_EVAL_ITERS": "1", "VQ_WARMUP_ITERS": "1", "VQ_REPORT_BATCH_SIZE": "8",
}


def encode(text):
    return [BOS_ID] + [b + BYTE_OFFSET for b in text.encode("utf-8")]


def make_docs(n, seed=0):
    rng = np.random.default_rng(seed)
    words = ["the", "cat", "naïve", "Größe", "€uro", "日本", "data\n", "a", "", "x" * 40]
    return [" ".join(rng.choice(words, size=int(rng.integers(0, 30)))) for _ in range(n)]


def run(script, *args, env=None, timeout=300):
    result = subprocess.run([sys.executable, *script, *args], cwd=env.pop("_cwd", ROOT) if env else ROOT,
                            capture_output=True, text=True, timeout=timeout,
                            env={**{k: v for k, v in os.environ.items() if not k.startswith("VQ_")}, **(env or {})})
    if result.returncode:
        raise AssertionError(result.stdout + result.stderr)
    return result.stdout


class ByteDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.docs = make_docs(400)
        lines = [json.dumps({"text": t}) + "\n" for t in cls.docs]
        (cls.root / "docs.jsonl").write_text("".join(lines), encoding="utf-8")
        cls.offsets = np.cumsum([0] + [len(l.encode()) for l in lines])
        cls.dev_line = 300                     # dev_offset falls inside line 299, so dev starts at 300
        cls.data = cls.root / "bytes"
        cls.prepare(train_shards=2)

    @classmethod
    def prepare(cls, train_shards, out=None):
        return run([str(ROOT / "fineweb_bytes.py")], "--docs", str(cls.root / "docs.jsonl"),
                   "--out", str(out or cls.data), "--train-shards", str(train_shards), "--num-val-docs", "20",
                   "--dev-docs", "30", "--dev-offset", str(int(cls.offsets[cls.dev_line]) - 3),
                   "--shard-size", "1000")

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def stream(self, split, data=None):
        files = sorted((data or self.data).glob(f"fineweb_{split}_*.bin"))
        return np.concatenate([load_data_shard(p) for p in files])

    def test_splits_match_pure_byte_export(self):
        np.testing.assert_array_equal(self.stream("val"), sum((encode(t) for t in self.docs[:20]), []))
        dev = sum((encode(t) for t in self.docs[self.dev_line:self.dev_line + 30]), [])
        np.testing.assert_array_equal(self.stream("dev"), dev)
        train = sum((encode(t) for t in self.docs[20:]), [])
        np.testing.assert_array_equal(self.stream("train"), train[:2000])
        manifest = json.loads((self.data / "bytes_manifest.json").read_text())
        self.assertEqual(manifest["val"]["bytes"], sum(len(t.encode()) for t in self.docs[:20]))

    def test_more_train_shards_resume_the_stream(self):
        data = self.root / "resumed"
        shutil.copytree(self.data, data)
        self.prepare(train_shards=3, out=data)
        train = sum((encode(t) for t in self.docs[20:]), [])
        np.testing.assert_array_equal(self.stream("train", data), train[:3000])
        with self.assertRaises(AssertionError) as caught:  # train must never reach the dev split
            self.prepare(train_shards=200, out=data)
        self.assertIn("reached the dev split", str(caught.exception))

    def test_load_chunks_matches_one_stream(self):
        expected = Chunks.from_tokens(self.stream("val"), 16)
        chunks, n = load_chunks(str(self.data / "fineweb_val_*.bin"), 16)
        self.assertEqual(n, len(list(self.data.glob("fineweb_val_*.bin"))))
        np.testing.assert_array_equal(chunks.chunks, expected.chunks)
        np.testing.assert_array_equal(chunks.lens, expected.lens)
        self.assertEqual(chunks.num_bytes, sum(len(t.encode()) for t in self.docs[:20]))


class ChunkTests(unittest.TestCase):
    def test_document_aligned_padding(self):
        docs = [b"", b"abc", b"x" * 4, b"y" * 5]
        tokens = sum(([BOS_ID] + [b + BYTE_OFFSET for b in d] for d in docs), [])
        c = Chunks.from_tokens(np.array(tokens, dtype=np.uint16), 4)
        np.testing.assert_array_equal(c.doc_lens, [0, 3, 4, 5])
        np.testing.assert_array_equal(c.doc_starts, [0, 0, 1, 2])
        np.testing.assert_array_equal(c.lens, [3, 4, 4, 1])
        self.assertEqual(bytes(c.chunks[0]), b"abc\x00")
        self.assertEqual(bytes(c.chunks[1]), b"xxxx")
        self.assertEqual(bytes(c.chunks[2]), b"yyyy")
        self.assertEqual(bytes(c.chunks[3]), b"y\x00\x00\x00")
        self.assertEqual(c.num_bytes, 12)

    def test_rejects_stream_without_bos(self):
        with self.assertRaises(ValueError):
            Chunks.from_tokens(np.array([5, 6], dtype=np.uint16), 4)


class StageOneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        lines = [json.dumps({"text": t}) + "\n" for t in make_docs(300, seed=1)]
        (cls.root / "docs.jsonl").write_text("".join(lines), encoding="utf-8")
        cls.data = cls.root / "bytes"
        run([str(ROOT / "fineweb_bytes.py")], "--docs", str(cls.root / "docs.jsonl"), "--out", str(cls.data),
            "--train-shards", "2", "--num-val-docs", "20", "--dev-docs", "20",
            "--dev-offset", str(len("".join(lines).encode()) * 3 // 4), "--shard-size", "2000")

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def settings(self, name, **extra):
        return {**TINY, "DATA_PATH": str(self.data), "OUT_DIR": str(self.root / name),
                "LOG_DIR": str(self.root / "logs"), "RUN_ID": name, **extra}

    def test_train_freeze_and_resume(self):
        out = run([str(ROOT / "train_gpt.py")], env=self.settings("run", VQ_MAX_ITERS="3"))
        self.assertIn("tokenizer frozen to", out)
        self.assertIn("tokenizer dev (frozen, after int8)", out)
        frozen = self.root / "run" / "tokenizer.int8.ptz"
        first = frozen.read_bytes()
        # a finished stage 1 is reused, not retrained
        out = run([str(ROOT / "train_gpt.py")], env=self.settings("run", VQ_MAX_ITERS="3", RESUME="True"))
        self.assertIn("stage 1 already finished", out)
        self.assertEqual(frozen.read_bytes(), first)
        # an interrupted stage 1 continues from vq_last.pt
        frozen.unlink()
        out = run([str(ROOT / "train_gpt.py")], env=self.settings("run", VQ_MAX_ITERS="5", RESUME="True"))
        self.assertIn("at iter 3", out)
        self.assertIn("vq iter     5", out)
        log = (self.root / "logs" / "run.txt").read_text()
        self.assertIn("def main()", log)             # the code is logged, like the baseline

    def test_unknown_setting_fails(self):
        with self.assertRaises(AssertionError) as caught:
            run([str(ROOT / "train_gpt.py")], env=self.settings("bad", VQ_BATCHSIZE="4"))
        self.assertIn("unknown setting", str(caught.exception))

    def test_torchrun_two_processes(self):
        out = run(["-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2", str(ROOT / "train_gpt.py")],
                  env=self.settings("ddp", VQ_MAX_ITERS="3"))
        self.assertIn("world_size:2", out)
        self.assertIn("tokenizer dev (frozen, after int8)", out)


class NotebookTests(unittest.TestCase):
    def test_colab_cells_compile(self):
        # google.colab only exists on Colab, so cells are compiled here, not executed
        # (nanoNCM's play.ipynb test executed them); shell lines (!...) become no-ops.
        notebook = json.loads((ROOT / "colab.ipynb").read_text(encoding="utf-8"))
        code = [c for c in notebook["cells"] if c["cell_type"] == "code"]
        self.assertGreaterEqual(len(code), 7)
        for index, cell in enumerate(code):
            source = "".join(cell["source"])
            self.assertNotRegex(source, r"(?i)(token|secret|api_key|password)\s*=")
            lines = ["pass" if line.lstrip().startswith("!") else line for line in source.splitlines()]
            compile("\n".join(lines), f"colab.ipynb:cell{index}", "exec")


def _codebook_worker(rank, world_size, port, out):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    torch.manual_seed(100 + rank)                    # different data and RNG on every rank
    book = EMACodebook(8, 3, decay=0.9, dead_code_threshold=0.95)
    for _ in range(5):
        flat = torch.randn(10, 3) * (rank + 1)
        weights = torch.rand(10) + 0.1
        book.initialize(flat, weights)
        book.update(flat, book.lookup(flat), weights)
    states = [None] * world_size
    dist.all_gather_object(states, {k: v.clone() for k, v in book.state_dict().items()})
    if rank == 0:
        torch.save(states, out)
    dist.destroy_process_group()


class DistributedCodebookTests(unittest.TestCase):
    def test_codebooks_stay_identical_across_ranks(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "states.pt"
            mp.spawn(_codebook_worker, args=(2, 29500 + os.getpid() % 1000, str(out)), nprocs=2)
            a, b = torch.load(out)
        for key in a:
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0, msg=key)


if __name__ == "__main__":
    unittest.main()
