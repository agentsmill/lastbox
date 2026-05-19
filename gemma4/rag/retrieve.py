#!/usr/bin/env python3
"""
Tiny stdlib + numpy retriever: load corpus.f32 + corpus.meta.jsonl once,
cosine top-k against a query embedding (which we get from a local
llama-server in --embedding mode).

Used both as a CLI smoke (`python retrieve.py "your query"`) and as a
module by the webapp.
"""
from __future__ import annotations
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import os

# Index dir resolution order:
#   1. RAG_INDEX_DIR env var
#   2. <this file's dir>/index   (matches the lastbox layout ~/lastbox-webapp/rag/index)
#   3. <repo root>/gemma4/rag/index (gx10 dev layout)
_HERE = Path(__file__).resolve().parent
_REPO_INDEX = _HERE.parents[1] / "gemma4" / "rag" / "index"
INDEX_DIR = Path(
    os.environ.get("RAG_INDEX_DIR")
    or (_HERE / "index" if (_HERE / "index" / "corpus.f32").exists() else _REPO_INDEX)
)

_DIM = 384  # bge-small-en-v1.5


def load_index(index_dir: Path = INDEX_DIR) -> tuple[np.ndarray, list[dict]]:
    vec = np.fromfile(index_dir / "corpus.f32", dtype=np.float32).reshape(-1, _DIM)
    # L2-normalise once so cosine == dot product
    norms = np.linalg.norm(vec, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vec = vec / norms
    meta: list[dict] = []
    with (index_dir / "corpus.meta.jsonl").open() as f:
        for line in f:
            meta.append(json.loads(line))
    assert vec.shape[0] == len(meta), "vec/meta length mismatch"
    return vec, meta


def embed_query(text: str, embed_url: str) -> np.ndarray:
    body = json.dumps({"input": [text]}).encode("utf-8")
    req = urllib.request.Request(
        embed_url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        d = json.loads(r.read())
    v = np.array(d["data"][0]["embedding"], dtype=np.float32)
    n = np.linalg.norm(v)
    if n > 0:
        v /= n
    return v


def topk(
    vec: np.ndarray, meta: list[dict], qvec: np.ndarray, k: int = 4,
) -> list[dict]:
    scores = vec @ qvec
    idx = np.argpartition(-scores, k)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return [
        {**meta[i], "score": float(scores[i])}
        for i in idx
    ]


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: retrieve.py <query> [k] [embed_url]", file=sys.stderr)
        return 2
    q = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    embed_url = (
        sys.argv[3] if len(sys.argv) > 3
        else "http://127.0.0.1:11437/v1/embeddings"
    )
    vec, meta = load_index()
    print(f"[retrieve] index: {vec.shape[0]} passages", file=sys.stderr)
    qv = embed_query(q, embed_url)
    hits = topk(vec, meta, qv, k=k)
    print(f"\n=== top {k} for: {q!r} ===\n")
    for h in hits:
        print(f"[{h['score']:.3f}] {h['source']} ({h['category']}/{h['kind']})")
        print(f"  {h['text'][:240]}{'...' if len(h['text']) > 240 else ''}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
