"""Live vision smoke test: send 3 test images to deployed llama-server on lastbox.

Tests Gemma 4 E2B + mmproj-F16 SigLIP encoder via llama.cpp multimodal path.
Measures vision encode + generation latency, prints answer per image.
"""
from __future__ import annotations
import asyncio, base64, json, sys, time
from pathlib import Path
import aiohttp

ENDPOINT = "http://lastbox:11436/v1"
SYSTEM = (
    "You are LastBox - offline survival assistant. Reply ONLY with the final answer "
    "in hybrid format (1 sentence or 1-3 short numbered points). No thinking, no analysis. "
    "lora source = ≤150 bytes UTF-8. Identify safety hazards directly."
)

TESTS = [
    ("rowan_berry.jpg",       "lora", "is this safe to eat?"),
    ("yew_berry_toxic.jpg",   "lora", "is this safe to eat?"),
    ("bleeding_finger.jpg",   "touchscreen", "what happened, first aid?"),
]

async def query_image(session, img_path: Path, source: str, query: str):
    b64 = base64.b64encode(img_path.read_bytes()).decode("ascii")
    payload = {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": [
                {"type": "text", "text": f"[source: {source}] {query}"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ]},
        ],
        "temperature": 0.3,
        "max_tokens": 120,
        "stream": False,
    }
    t0 = time.perf_counter()
    async with session.post(f"{ENDPOINT}/chat/completions", json=payload,
                            timeout=aiohttp.ClientTimeout(total=120)) as r:
        r.raise_for_status()
        data = await r.json()
    dt = time.perf_counter() - t0
    return data["choices"][0]["message"]["content"], dt

async def main():
    base = Path("gemma4/deploy/bundle/test_images")
    async with aiohttp.ClientSession() as s:
        for fname, src, q in TESTS:
            p = base / fname
            print(f"\n=== {fname}  ({(p.stat().st_size/1024):.0f} KB, source={src}) ===")
            print(f"Q: {q}")
            try:
                resp, dt = await query_image(s, p, src, q)
                print(f"A: {resp}")
                print(f"   bytes={len(resp.encode('utf-8'))}  total={dt:.1f}s")
            except Exception as e:
                print(f"ERROR: {e}")

if __name__ == "__main__":
    asyncio.run(main())
