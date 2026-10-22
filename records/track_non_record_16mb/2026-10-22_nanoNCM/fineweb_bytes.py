"""
fineweb_bytes.py - raw UTF-8 FineWeb bytes for nanoNCM, in the repo's byte260 shard format.

The published Hugging Face export has no byte260 shards, so this streams the published docs
cache (datasets/docs_selected.jsonl: the same documents, in the same order, as
fineweb10B_sp1024) with range reads and encodes every document exactly like
data/download_hf_docs_and_tokenize.py's PureByteTokenizer does:
    BOS (1), then each UTF-8 byte b as b + 4; no EOS; shards of 10**8 tokens.

Splits (by position in docs_selected.jsonl):
    val    the first 50,000 docs: the challenge's fixed validation set, used ONLY for final eval
    train  the docs that follow, --train-shards full shards
    dev    --dev-docs docs from a fixed byte offset far beyond any train shard used here;
           for monitoring and tuning, so val is never touched during training

    python fineweb_bytes.py --train-shards 1 --out ./data/datasets/fineweb10B_byte260

Rerunning is cheap: finished splits are skipped and extra train shards resume where the
last one ended (bytes_manifest.json records the offsets), e.g. with a cache on Google Drive.
"""

import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

REPO_ID = os.environ.get("MATCHED_FINEWEB_REPO_ID", "willdepueoai/parameter-golf")
DOCS_PATH = "datasets/docs_selected.jsonl"
NUM_VAL_DOCS = 50_000
SHARD_SIZE = 10**8
BOS_ID = 1
BYTE_OFFSET = 4
DATAFILE_MAGIC = 20240520
DATAFILE_VERSION = 1
DEV_OFFSET = 40_000_000_000   # ~doc 12.8M of 15.4M; 80 train shards end near doc 3M
MANIFEST = "bytes_manifest.json"


def encode(text):
    """PureByteTokenizer.encode with the exporter's leading BOS."""
    data = np.frombuffer(text.encode("utf-8", errors="replace"), dtype=np.uint8)
    toks = np.empty(data.size + 1, dtype="<u2")
    toks[0] = BOS_ID
    toks[1:] = data.astype(np.uint16) + BYTE_OFFSET
    return toks


def write_datafile(path, toks):
    """Same header and layout as data/download_hf_docs_and_tokenize.py; written atomically."""
    header = np.zeros(256, dtype="<i4")
    header[0], header[1], header[2] = DATAFILE_MAGIC, DATAFILE_VERSION, len(toks)
    tmp = Path(str(path) + ".tmp")
    with tmp.open("wb") as f:
        f.write(header.tobytes())
        f.write(np.asarray(toks, dtype="<u2").tobytes())
    tmp.replace(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RangeReader:
    """The HF docs cache as a read-only, seekable file over HTTP range requests
    (only what stream_docs needs: seek, tell, readline)."""

    def __init__(self, url, block_size=16 * 2**20):
        self.url, self.block_size = url, block_size
        self.pos, self.buf_start, self.buf = 0, 0, b""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def seek(self, pos):
        self.pos = pos

    def tell(self):
        return self.pos

    def _fetch(self, start):
        request = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{start + self.block_size - 1}"})
        for attempt in range(6):
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    if response.status != 206:
                        raise RuntimeError(f"server ignored the range request (HTTP {response.status})")
                    data = response.read()
                break
            except urllib.error.HTTPError as e:
                if e.code == 416:          # past the end of the file
                    data = b""
                    break
                if attempt == 5:
                    raise
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                if attempt == 5:
                    raise
            time.sleep(2 ** attempt)
        self.buf_start, self.buf = start, data

    def readline(self):
        parts = []
        while True:
            off = self.pos - self.buf_start
            if not 0 <= off < len(self.buf):
                self._fetch(self.pos)
                off = 0
                if not self.buf:
                    return b"".join(parts)
            newline = self.buf.find(b"\n", off)
            end = len(self.buf) if newline < 0 else newline + 1
            parts.append(self.buf[off:end])
            self.pos = self.buf_start + end
            if newline >= 0:
                return b"".join(parts)


def open_docs(docs):
    """A seekable binary file: a local path (tests), or the HF docs cache read with HTTP ranges."""
    if docs:
        return open(docs, "rb")
    return RangeReader(f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/{DOCS_PATH}")


def stream_docs(f, offset):
    """Yields (line_start, line_end, text) from byte offset `offset`, which starts a line."""
    f.seek(offset)
    while True:
        start = f.tell()
        line = f.readline()
        if not line:
            return
        yield start, f.tell(), json.loads(line)["text"]


def export(f, out, split, offset, *, skip=0, max_docs=None, max_shards=None, first_shard=0,
           shard_size=SHARD_SIZE, stop_offset=None, log=print):
    """
    Writes fineweb_{split}_{i:06d}.bin like the repo's exporter: documents are concatenated
    and cut every shard_size tokens, so one document can straddle two shards. With
    max_shards only full shards are written; each records where the next one resumes
    (line offset + tokens of that line already written). Lines at or after stop_offset are
    never read: train shards must not reach the dev split.
    """
    buf = np.empty(shard_size, dtype="<u2")
    fill, shard, docs, end = 0, first_shard, 0, offset
    files = []

    def flush(resume):
        nonlocal fill, shard
        path = out / f"fineweb_{split}_{shard:06d}.bin"
        sha = write_datafile(path, buf[:fill])
        files.append({"file": path.name, "tokens": int(fill), "sha256": sha, "resume": resume})
        log(f"{split}: wrote {path.name} ({fill:,} tokens)")
        shard += 1
        fill = 0

    for start, end, text in stream_docs(f, offset):
        if stop_offset is not None and start >= stop_offset:
            raise ValueError(f"{split} shards reached the dev split; lower --train-shards")
        if max_docs is not None and docs == max_docs:
            end = start
            break
        toks = encode(text)[skip:]
        pos = 0
        while pos < len(toks):
            take = min(shard_size - fill, len(toks) - pos)
            buf[fill:fill + take] = toks[pos:pos + take]
            fill += take
            pos += take
            if fill == shard_size:
                flush([start, skip + pos])
                if max_shards is not None and shard - first_shard == max_shards:
                    return files, docs + 1, end
        skip = 0
        docs += 1
    if max_shards is not None:
        raise ValueError(f"{split}: the docs ran out before {max_shards} full shards")
    if fill:
        flush([end, 0])
    return files, docs, end


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--train-shards", type=int, default=1)
    parser.add_argument("--dev-docs", type=int, default=10_000)
    parser.add_argument("--out", default="./data/datasets/fineweb10B_byte260")
    parser.add_argument("--docs", default=None, help="local docs_selected.jsonl (tests); default: stream from HF")
    parser.add_argument("--num-val-docs", type=int, default=NUM_VAL_DOCS, help="only for tests")
    parser.add_argument("--dev-offset", type=int, default=DEV_OFFSET, help="only for tests")
    parser.add_argument("--shard-size", type=int, default=SHARD_SIZE, help="only for tests")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / MANIFEST
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    settings = {"source": args.docs or f"{REPO_ID}/{DOCS_PATH}", "num_val_docs": args.num_val_docs,
                "dev_offset": args.dev_offset, "dev_docs": args.dev_docs, "shard_size": args.shard_size}
    if manifest.get("settings", settings) != settings:
        raise ValueError(f"{manifest_path} was made with other settings; use another --out")
    manifest["settings"] = settings

    def save():
        manifest_path.with_suffix(".tmp").write_text(json.dumps(manifest, indent=2) + "\n")
        manifest_path.with_suffix(".tmp").replace(manifest_path)

    def complete(split):
        entry = manifest.get(split)
        return entry is not None and all((out / e["file"]).exists() for e in entry["files"])

    with open_docs(args.docs) as f:
        if not complete("val"):
            files, docs, end = export(f, out, "val", 0, max_docs=args.num_val_docs,
                                       shard_size=args.shard_size)
            if docs != args.num_val_docs:
                raise ValueError(f"val: expected {args.num_val_docs} docs, got {docs}")
            tokens = sum(e["tokens"] for e in files)
            manifest["val"] = {"files": files, "docs": docs, "tokens": tokens,
                               "bytes": tokens - docs, "end_offset": end}
            save()
        if not complete("dev"):
            f.seek(args.dev_offset)
            f.readline()                         # skip to the next complete line
            files, docs, end = export(f, out, "dev", f.tell(), max_docs=args.dev_docs,
                                       shard_size=args.shard_size)
            tokens = sum(e["tokens"] for e in files)
            manifest["dev"] = {"files": files, "docs": docs, "tokens": tokens,
                               "bytes": tokens - docs, "start_offset": args.dev_offset}
            save()
        train = manifest.setdefault("train", {"files": []})
        train["files"] = [e for e in train["files"] if (out / e["file"]).exists()]
        have = len(train["files"])
        if have < args.train_shards:
            offset, skip = train["files"][-1]["resume"] if have else (manifest["val"]["end_offset"], 0)
            files, _, end = export(f, out, "train", offset, skip=skip,
                                   max_shards=args.train_shards - have, first_shard=have,
                                   shard_size=args.shard_size, stop_offset=args.dev_offset)
            train["files"] += files
            save()
    for split in ("val", "dev"):
        e = manifest[split]
        print(f"{split}: {e['docs']:,} docs, {e['bytes']:,} bytes in {len(e['files'])} file(s)")
    print(f"train: {len(manifest['train']['files'])} shard(s) available, {args.train_shards} requested")


if __name__ == "__main__":
    main()
