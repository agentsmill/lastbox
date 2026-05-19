#!/usr/bin/env python3
"""
Embed every passage in the corpus through the local llama-embed server and
save a numpy float32 matrix + parallel id-list. Tiny enough for cosine top-k
in pure numpy at query time (~5 MB for 4k passages × 384 dims).
"""
from __future__ import annotations
import json
import struct
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "gemma4" / "rag" / "corpus" / "corpus.jsonl"
OUT_DIR = ROOT / "gemma4" / "rag" / "index"
OUT_VEC = OUT_DIR / "corpus.f32"
OUT_META = OUT_DIR / "corpus.meta.jsonl"

EMBED_URL = "http://127.0.0.1:11437/v1/embeddings"
BATCH = 8  # bge-small ctx 512; passages average ~500 chars — batch small to stay safe


def embed_batch(texts: list[str]) -> list[list[float]]:
    body = json.dumps({"input": texts}).encode("utf-8")
    req = urllib.request.Request(
        EMBED_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.loads(r.read())
    return [item["embedding"] for item in d["data"]]


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    passages: list[dict] = []
    with CORPUS.open() as f:
        for line in f:
            passages.append(json.loads(line))
    n = len(passages)
    print(f"[index] embedding {n} passages…", file=sys.stderr)

    dim = None
    t0 = time.time()
    with OUT_VEC.open("wb") as vec_f, OUT_META.open("w") as meta_f:
        for batch_start in range(0, n, BATCH):
            batch = passages[batch_start:batch_start + BATCH]
            texts = [p["text"] for p in batch]
            try:
                embs = embed_batch(texts)
            except Exception as e:
                # retry once individually
                print(f"  batch fail at {batch_start}: {e}; retrying solo", file=sys.stderr)
                embs = []
                for t in texts:
                    try:
                        embs.extend(embed_batch([t]))
                    except Exception as ee:
                        print(f"    individual fail: {ee}", file=sys.stderr)
                        embs.append([0.0] * (dim or 384))
            for p, e in zip(batch, embs):
                if dim is None:
                    dim = len(e)
                vec_f.write(struct.pack(f"{dim}f", *e))
                meta_f.write(json.dumps({
                    "id": p["id"],
                    "source": p["source"],
                    "category": p["category"],
                    "kind": p["kind"],
                    "text": p["text"],
                }, ensure_ascii=False) + "\n")
            if batch_start % (BATCH * 20) == 0:
                done = batch_start + len(batch)
                eta = (time.time() - t0) / max(done, 1) * (n - done)
                print(f"  [{done}/{n}] dim={dim} eta {eta:.0f}s", file=sys.stderr)

    print(f"[index] wrote {OUT_VEC} ({OUT_VEC.stat().st_size} bytes), dim={dim}", file=sys.stderr)
    print(f"[index] wrote {OUT_META}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
