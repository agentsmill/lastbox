"""Eval LastBox v4 — TOOL-ALLOWED variant.

Identical to eval_v2.py except: system prompt is the original SYSTEM_PROMPT_EN
(which describes the tool whitelist), with NO "Reply with ONLY the final answer"
override. This unblocks tool_call emission and gives an honest read on whether
GRPO moved tool_accuracy.

Other metrics (format/byte_cap/persona) are reported separately on the
final answer text only — the <tool_call> block is stripped before scoring.

Usage:
    python gemma4/scripts/eval_v2_tool.py \\
        --endpoint http://lastbox:11436/v1 \\
        --golden gemma4/data/golden_en.jsonl \\
        --out gemma4/data/eval_v4_tool_results.json
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

import aiohttp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_dataset_v2 import SYSTEM_PROMPT_EN  # noqa: E402
from process_v2 import _build_system_prompt_with_tools  # noqa: E402

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
    text = TOOL_CALL_RE.sub("", text).strip()
    if not text:
        return True   # pure tool_call turn is acceptable (the answer comes later)
    if len(text) <= 200 and "\n" not in text:
        return True
    if NUMBERED_RE.search(text):
        return True
    return False


async def _query(session: aiohttp.ClientSession, endpoint: str, system: str,
                 user: str, source: str) -> tuple[str, float | None]:
    user_full = f"[source: {source}] {user}"
    max_tok = 160 if source == "lora" else 256
    payload = {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_full},
        ],
        "temperature": 0.2,
        "max_tokens": max_tok,
        "stream": False,
    }
    t0 = time.perf_counter()
    # Up to 2 retries on disconnect — server is single-slot and occasionally
    # closes a long-lived connection, but recovers immediately.
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            async with session.post(
                f"{endpoint}/chat/completions", json=payload,
                timeout=aiohttp.ClientTimeout(total=300),
            ) as resp:
                resp.raise_for_status()
                obj = await resp.json()
            text = obj["choices"][0]["message"]["content"] or ""
            ttm = (time.perf_counter() - t0) * 1000
            return text, ttm
        except (aiohttp.ServerDisconnectedError, aiohttp.ClientConnectionError,
                aiohttp.ClientPayloadError) as e:
            last_exc = e
            await asyncio.sleep(1.0 + attempt)
    raise last_exc or RuntimeError("query failed")


async def main_async(args):
    # Use the FULL training-time system prompt (with tool defs JSON + format
    # hint). The shorter SYSTEM_PROMPT_EN-only variant suppressed tool emission
    # because the model only learned to emit <tool_call> when the giant tool
    # definition block was present in context — that's how train_v2 was built.
    system = _build_system_prompt_with_tools()
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
                text, ftm = await _query(
                    session, args.endpoint, system, g["user"], g["source"]
                )
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
            r.byte_compliance = len(final_no_tool.encode("utf-8")) <= int(
                g.get("max_response_bytes", 200)
            )
            r.format_ok = _format_ok(final_no_tool)
            r.persona_ok = not bool(PREAMBLE_RE.match(final_no_tool))
            quality_ok = (r.byte_compliance and r.format_ok and r.persona_ok)
            status = "OK" if quality_ok else "FAIL"
            ftm_str = f"{ftm:.0f}ms" if ftm else "?ms"
            print(f"[{r.id}] {status} tool={r.tool_match} (actual={actual_tool}) "
                  f"bytes={r.byte_compliance} fmt={r.format_ok} "
                  f"persona={r.persona_ok} ftm={ftm_str}")
            results.append(r)

    n = len(results)
    if n == 0:
        return 1
    summary = {
        "n": n,
        "tool_accuracy": sum(r.tool_match for r in results) / n,
        "arg_validity": sum(r.args_valid for r in results) / n,
        "tool_emission_rate": sum(1 for r in results if r.actual_tool) / n,
        "byte_compliance": sum(r.byte_compliance for r in results) / n,
        "format_ok": sum(r.format_ok for r in results) / n,
        "persona_ok": sum(r.persona_ok for r in results) / n,
        "response_quality_score": sum(
            (r.byte_compliance * 0.35 + r.format_ok * 0.35 + r.persona_ok * 0.30)
            for r in results
        ) / n,
        "agentic_score": sum(
            (r.tool_match * 0.6 + r.args_valid * 0.4) for r in results
        ) / n,
        "median_first_token_ms": sorted(
            [r.first_token_ms for r in results if r.first_token_ms]
        )[n // 2] if any(r.first_token_ms for r in results) else None,
        "completed": sum(1 for r in results if not r.final_response.startswith("[ERROR")),
    }
    print("\n=== SUMMARY (TOOL-ALLOWED) ===")
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
    ap.add_argument("--out", type=Path,
                    default=Path("gemma4/data/eval_v4_tool_results.json"))
    args = ap.parse_args()
    sys.exit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
