"""
GRPO reward for LastBox v4 — push the v3 SFT checkpoint towards:
  (1) actually emitting <tool_call> when the dataset says it should
  (2) hybrid response format: 1 lead sentence + optional numbered list
  (3) hard byte cap per channel (lora ≤150, touchscreen ≤200)

Composite reward (range 0..1):
    r = 0.5 * tool_match
      + 0.3 * format_ok
      + 0.2 * byte_cap_ok

`tool_match` is the load-bearing term because that's the biggest baseline gap
(2% emission in v3 eval). Format + byte_cap reinforce what SFT already
half-learned.
"""
from __future__ import annotations
import json
import re
from typing import Iterable

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
NUMBERED_LINE_RE = re.compile(r"^\s*\d+[\.\)]\s", re.MULTILINE)


def parse_tool_call(text: str) -> dict | None:
    m = TOOL_CALL_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def utf8_bytes(text: str) -> int:
    return len(text.encode("utf-8"))


def tool_match_score(completion: str, expected_tool: str | None) -> float:
    """1.0 if the right tool is called, 0.0 otherwise.

    'No tool expected' branch: reward NOT calling a tool (this keeps the model
    from spam-emitting tool_calls to game the metric).
    """
    parsed = parse_tool_call(completion)
    if expected_tool is None or expected_tool == "":
        return 1.0 if parsed is None else 0.0
    if parsed is None:
        return 0.0
    return 1.0 if parsed.get("name") == expected_tool else 0.0


def format_ok_score(completion: str) -> float:
    """Hybrid format: at least one sentence; if multi-step, numbered list.

    Returns 1.0 if format is acceptable, 0.5 if borderline (e.g. all numbered
    bullets no lead sentence), 0.0 if the model produced raw tool_call only,
    pre-thinking preamble, or bullet-only output without numbering.
    """
    # Drop the tool_call block for format scoring (its presence is rewarded
    # separately).
    text = TOOL_CALL_RE.sub("", completion).strip()
    if not text:
        # Pure tool_call output is acceptable for the tool-call turn.
        return 1.0
    if text.lower().startswith(("thinking process:", "thinking:", "let me think",
                                "let's think", "sure,", "of course", "okay,")):
        return 0.0
    lines = [l for l in text.splitlines() if l.strip()]
    if len(lines) == 1:
        return 1.0  # single-sentence answer
    has_numbered = bool(NUMBERED_LINE_RE.search(text))
    has_bulleted = any(re.match(r"^\s*[-*•]\s", l) for l in lines)
    if has_numbered:
        return 1.0
    if has_bulleted:
        return 0.5  # better to use numbered for procedures
    return 0.7  # multi-sentence prose — not great, not terrible


def byte_cap_ok_score(completion: str, source: str | None) -> float:
    """Hard cap reward: 1.0 if under cap, 0.0 if over.

    Strip <tool_call> blocks before measuring — they don't go on the wire.
    """
    visible = TOOL_CALL_RE.sub("", completion).strip()
    nb = utf8_bytes(visible)
    if source == "lora":
        return 1.0 if nb <= 150 else 0.0
    # touchscreen or unspecified
    return 1.0 if nb <= 200 else 0.0


def composite_reward(
    completion: str,
    expected_tool: str | None,
    source: str | None,
) -> float:
    """v2 reward shape — break the v1 plateau.

    v1 gave the same 0.5 to "emit correct tool, no answer" and
    "emit no tool, give a direct correct answer". Model converged on the
    easier no-tool path. v2 makes the tool path strictly dominant:

      with expected_tool:
        - correct tool emitted:  +1.0  (plus format/byte bonus if answer present)
        - wrong tool emitted:    +0.0
        - no tool emitted:       -0.5  (active penalty)

      without expected_tool:
        - no tool emitted:       +0.6 base + 0.25*format + 0.15*byte_cap
        - tool emitted:          +0.0  (penalty for spurious tool emission)
    """
    parsed = parse_tool_call(completion)
    fmt = format_ok_score(completion)
    byc = byte_cap_ok_score(completion, source)

    if expected_tool:
        if parsed is None:
            return -0.5
        if parsed.get("name") == expected_tool:
            return 1.0  # max reward — the answer comes from the tool result turn
        return 0.0  # wrong tool

    # No tool expected — reward a clean direct answer
    if parsed is not None:
        return 0.0  # spurious tool emission
    return 0.6 + 0.25 * fmt + 0.15 * byc


# TRL signature: reward_func(prompts, completions, **kwargs) -> list[float]
# kwargs come from the dataset extra columns (we pass expected_tool, source)
def reward_func(
    completions: list[str],
    expected_tool: Iterable[str | None] | None = None,
    source: Iterable[str | None] | None = None,
    **_unused,
) -> list[float]:
    et = list(expected_tool) if expected_tool is not None else [None] * len(completions)
    sr = list(source) if source is not None else [None] * len(completions)
    if len(et) == 1 and len(completions) > 1:
        et = et * len(completions)
    if len(sr) == 1 and len(completions) > 1:
        sr = sr * len(completions)
    return [composite_reward(c, e, s) for c, e, s in zip(completions, et, sr)]


if __name__ == "__main__":
    # Tiny smoke test
    samples = [
        ('<tool_call>{"name":"search_knowledge","arguments":{"query":"x"}}</tool_call>',
         "search_knowledge", "lora"),
        ("1. Apply direct pressure. 2. Elevate the arm.", None, "lora"),
        ("Thinking Process: Let me analyze the situation step by step…",
         None, "touchscreen"),
        ("A" * 200, None, "lora"),  # byte cap fail
    ]
    for c, et, src in samples:
        r = composite_reward(c, et, src)
        print(f"r={r:.3f}  expected={et}  src={src}  text={c[:60]!r}")
