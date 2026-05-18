"""Eval LastBox-Gemma 4 v2 na golden_en.jsonl via llama-server endpoint.

Mierzy:
- tool_accuracy   - czy expected_tool dispatched
- arg_validity    - czy tool args zawierają expected_args_contains
- byte_compliance - czy final response <= max_response_bytes
- format_ok       - czy odpowiedź = 1 zdanie LUB 1 lead + numbered list (hybrid)
- persona_ok      - czy NIE zaczyna od preambuły ("Sure,...", "Of course,...")
- first_token_ms  - latency pierwszego tokena (jeśli streaming)

Usage:
    python gemma4/scripts/eval_v2.py \\
        --endpoint http://localhost:11436/v1 \\
        --golden gemma4/data/golden_en.jsonl \\
        --out gemma4/data/eval_results.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any

import aiohttp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_dataset_v2 import TOOL_DEFS, SYSTEM_PROMPT_EN  # noqa: E402

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
PREAMBLE_RE = re.compile(
    r"^\s*(sure|of course|certainly|absolutely|alright|okay|ok|here'?s|here is)\b",
    re.IGNORECASE,
)
NUMBERED_RE = re.compile(r"^\s*\d\.\s+", re.MULTILINE)


@dataclass
class EvalResult:
    id: str
    user: str
    source: str
    expected_tool: str | None
    actual_tool: str | None
    tool_args: dict
    final_response: str
    tool_match: bool = False
    args_valid: bool = False
    byte_compliance: bool = False
    format_ok: bool = False
    persona_ok: bool = False
    first_token_ms: float | None = None
    notes: list[str] = field(default_factory=list)


def _build_system() -> str:
    return (
        f"{SYSTEM_PROMPT_EN}\n\n"
        "CRITICAL output rules: Reply with ONLY the final answer. "
        "Do NOT show thinking, reasoning, analysis, or planning steps. "
        "Start directly with the action sentence."
    )


def _extract_tool_calls(text: str) -> list[dict]:
    calls = []
    for m in TOOL_CALL_RE.finditer(text):
        try:
            o = json.loads(m.group(1))
            if "name" in o:
                calls.append(o)
        except json.JSONDecodeError:
            continue
    return calls


def _args_match(expected: dict | None, actual: dict) -> bool:
    if not expected:
        return True
    for k, v in expected.items():
        a = actual.get(k, "")
        if isinstance(v, str):
            if v.lower() not in str(a).lower():
                return False
        elif isinstance(v, (int, float)):
            if v != a:
                return False
    return True


def _format_ok(text: str) -> bool:
    """Hybrid format = 1 short sentence OR 1 lead sentence + numbered list."""
    text = TOOL_CALL_RE.sub("", text).strip()
    if not text:
        return False
    if len(text) <= 200 and "\n" not in text:
        return True  # one-liner OK
    # Multiline: must have at least 1 numbered item
    if NUMBERED_RE.search(text):
        return True
    return False


async def _query(session: aiohttp.ClientSession, endpoint: str, system: str, user: str, source: str) -> tuple[str, float | None]:
    """Returns (final_text, first_token_ms)."""
    user_full = f"[source: {source}] {user}"
    max_tok = 80 if source == "lora" else 160
    payload = {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_full},
        ],
        "temperature": 0.2,
        "max_tokens": max_tok,
        "stream": True,
        "stop": ["<channel|>", "<think>", "Thinking Process", "Process:", "<|tool_response>", "Analyze Request", "**Analyze", "1.  **", "The user is asking"],
    }
    text_parts: list[str] = []
    first_token_ms: float | None = None
    t0 = time.perf_counter()
    async with session.post(f"{endpoint}/chat/completions", json=payload, timeout=aiohttp.ClientTimeout(total=180)) as resp:
        resp.raise_for_status()
        async for chunk in resp.content:
            line = chunk.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            try:
                obj = json.loads(body)
            except Exception:
                continue
            delta = obj["choices"][0].get("delta", {})
            piece = delta.get("content") or ""
            if piece:
                if first_token_ms is None:
                    first_token_ms = (time.perf_counter() - t0) * 1000
                text_parts.append(piece)
    return "".join(text_parts), first_token_ms


async def main_async(args):
    system = _build_system()
    goldens = []
    with open(args.golden, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            goldens.append(json.loads(line))

    results: list[EvalResult] = []
    async with aiohttp.ClientSession() as session:
        for g in goldens:
            try:
                text, ftm = await _query(session, args.endpoint, system, g["user"], g["source"])
            except Exception as e:
                print(f"[ERR] {g['id']}: {e}", file=sys.stderr)
                results.append(EvalResult(
                    id=g["id"], user=g["user"], source=g["source"],
                    expected_tool=g.get("expected_tool"), actual_tool=None,
                    tool_args={}, final_response=f"[ERROR: {e}]",
                ))
                continue

            calls = _extract_tool_calls(text)
            actual_tool = calls[0]["name"] if calls else None
            actual_args = calls[0].get("arguments", {}) if calls else {}
            final_no_tool = TOOL_CALL_RE.sub("", text).strip()

            r = EvalResult(
                id=g["id"], user=g["user"], source=g["source"],
                expected_tool=g.get("expected_tool"), actual_tool=actual_tool,
                tool_args=actual_args, final_response=final_no_tool,
                first_token_ms=ftm,
            )
            r.tool_match = (g.get("expected_tool") == actual_tool)
            r.args_valid = _args_match(g.get("expected_args_contains"), actual_args)
            r.byte_compliance = (len(final_no_tool.encode("utf-8")) <= int(g.get("max_response_bytes", 200)))
            r.format_ok = _format_ok(final_no_tool)
            r.persona_ok = not bool(PREAMBLE_RE.match(final_no_tool))

            quality_ok = (r.byte_compliance and r.format_ok and r.persona_ok)
            status = "OK" if quality_ok else "FAIL"
            ftm_str = f"{ftm:.0f}ms" if ftm else "?ms"
            print(f"[{r.id}] {status} tool={r.tool_match} args={r.args_valid} bytes={r.byte_compliance} fmt={r.format_ok} persona={r.persona_ok} ftm={ftm_str}")
            results.append(r)

    n = len(results)
    if n == 0:
        print("No results.")
        return 1

    summary = {
        "n": n,
        "tool_accuracy": sum(r.tool_match for r in results) / n,
        "arg_validity": sum(r.args_valid for r in results) / n,
        "byte_compliance": sum(r.byte_compliance for r in results) / n,
        "format_ok": sum(r.format_ok for r in results) / n,
        "persona_ok": sum(r.persona_ok for r in results) / n,
        "response_quality_score": sum(
            (r.byte_compliance * 0.35 + r.format_ok * 0.35 + r.persona_ok * 0.30) for r in results
        ) / n,
        "agentic_score": sum(
            (r.tool_match * 0.6 + r.args_valid * 0.4) for r in results
        ) / n,
        "median_first_token_ms": sorted([r.first_token_ms for r in results if r.first_token_ms])[n // 2] if any(r.first_token_ms for r in results) else None,
    }

    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"{k:25s} {v:.3f}")
        else:
            print(f"{k:25s} {v}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({
            "summary": summary,
            "results": [asdict(r) for r in results],
        }, ensure_ascii=False, indent=2))
        print(f"\nSaved: {args.out}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", type=str, default="http://localhost:11436/v1")
    ap.add_argument("--golden", type=Path, default=Path("gemma4/data/golden_en.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("gemma4/data/eval_results.json"))
    args = ap.parse_args()
    sys.exit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
