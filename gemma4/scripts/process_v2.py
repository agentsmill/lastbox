"""Validate, normalize, adapt dataset v2 dla SFT.

raw_v2.jsonl -> gemma4/data/train_v2.jsonl + val_v2.jsonl

Walidacja:
- Tool names ∈ whitelist (7 tools)
- Tool args valid JSON
- Final assistant.content ≤ 200B (touchscreen) / ≤ 150B (lora)
- ≥ 1 user msg AND ≥ 1 assistant msg
- Dedup (user_query + source signature)

Adapter (kompatybilne z Gemma 4 chat template, takie samo jak prepare_dataset.py):
- Inject EN system prompt + tools doc jako pierwszy user msg
- Convert structured tool_calls -> inline <tool_call>{json}</tool_call> w assistant.content
- Tool response -> user msg "[tool result]\n..."
- Final assistant -> response w hybrid format
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_dataset_v2 import TOOL_DEFS, TOOL_WHITELIST, SYSTEM_PROMPT_EN


TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"


def _byte_limit(source: str) -> int:
    return 150 if source == "lora" else 200


def _validate(d: dict) -> tuple[bool, str]:
    msgs = d.get("messages")
    if not isinstance(msgs, list) or len(msgs) < 2:
        return False, "no_messages"
    source = d.get("source")
    if source not in ("lora", "touchscreen"):
        return False, f"bad_source:{source}"

    # ostatni assistant w łańcuchu
    last_assistant_text = None
    for m in msgs:
        if m.get("role") == "assistant":
            txt = (m.get("content") or "").strip()
            if txt:
                last_assistant_text = txt
    if not last_assistant_text:
        return False, "no_final_assistant_content"

    # Byte cap
    blimit = _byte_limit(source)
    if len(last_assistant_text.encode("utf-8")) > blimit:
        return False, f"too_long:{len(last_assistant_text.encode('utf-8'))}>{blimit}"

    # Tool whitelist + args valid
    for m in msgs:
        if m.get("role") != "assistant":
            continue
        for tc in (m.get("tool_calls") or []):
            fn = tc.get("function") or tc
            name = fn.get("name")
            if name not in TOOL_WHITELIST:
                return False, f"bad_tool:{name}"
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    return False, "tool_args_not_json"
            if not isinstance(args, dict):
                return False, "tool_args_not_dict"

    # Polish-only check skipped — v2 jest EN
    # Sanity: pierwszy user query niepusty
    first_user = next((m.get("content", "") for m in msgs if m.get("role") == "user"), "")
    if not first_user or len(first_user.strip()) < 4:
        return False, "no_user_query"

    return True, "ok"


def _adapt_to_gemma(d: dict, system_prompt: str) -> dict | None:
    """Konwertuje dialog do formatu używanego przez prepare_dataset.py:
    - pierwszy user: system_prompt + tools doc + actual user query
    - kolejne assistant z tool_calls → inline <tool_call> w content
    - tool messages → user "[tool result]..."
    """
    src_msgs = d["messages"]
    out: list[dict] = []

    # weź oryginalne user query (pierwsze)
    first_user = next((m.get("content", "") for m in src_msgs if m.get("role") == "user"), "")

    # zbuduj pierwszy user msg z system + tools + query
    out.append({"role": "user", "content": f"{system_prompt}\n\n{first_user}"})

    # przepuszczamy resztę zaczynając OD pierwszego non-system non-user-prepended
    user_seen = False
    for m in src_msgs:
        role = m.get("role")
        content = m.get("content", "") or ""
        if role == "user":
            if not user_seen:
                user_seen = True
                continue  # już dodaliśmy go (zmergowanego z prefixem)
            # kolejne user msgs (rzadko) - dorzucamy
            out.append({"role": "user", "content": content})
        elif role == "assistant":
            parts: list[str] = []
            if content:
                parts.append(content)
            for tc in (m.get("tool_calls") or []):
                fn = tc.get("function") or tc
                payload = {"name": fn.get("name"), "arguments": fn.get("arguments", {})}
                parts.append(f"{TOOL_CALL_OPEN}{json.dumps(payload, ensure_ascii=False)}{TOOL_CALL_CLOSE}")
            txt = "\n".join(parts).strip()
            if not txt:
                continue
            if out and out[-1]["role"] == "assistant":
                out[-1]["content"] = out[-1]["content"] + "\n" + txt
            else:
                out.append({"role": "assistant", "content": txt})
        elif role == "tool":
            user_msg = f"[tool result]\n{content}".strip()
            if out and out[-1]["role"] == "user":
                out[-1]["content"] = out[-1]["content"] + "\n\n" + user_msg
            else:
                out.append({"role": "user", "content": user_msg})

    if not any(m["role"] == "assistant" and m["content"].strip() for m in out):
        return None

    return {
        "messages": out,
        "source": d.get("source"),
        "category": d.get("category"),
        "seed_id": d.get("seed_id"),
    }


def _build_system_prompt_with_tools() -> str:
    tools_doc = json.dumps([t["function"] for t in TOOL_DEFS], ensure_ascii=False, indent=2)
    return (
        f"{SYSTEM_PROMPT_EN}\n\n"
        f"## Available tools (call only when needed):\n{tools_doc}\n\n"
        f"Tool call format: {TOOL_CALL_OPEN}"
        '{"name":"<tool>","arguments":{...}}'
        f"{TOOL_CALL_CLOSE}"
    )


def _sig(d: dict) -> str:
    q = next((m.get("content", "") for m in d.get("messages", []) if m.get("role") == "user"), "")
    return hashlib.sha1(f"{d.get('source')}|{q.strip().lower()}".encode("utf-8")).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, default=Path("gemma4/data/raw_v2.jsonl"))
    ap.add_argument("--out-train", type=Path, default=Path("gemma4/data/train_v2.jsonl"))
    ap.add_argument("--out-val", type=Path, default=Path("gemma4/data/val_v2.jsonl"))
    ap.add_argument("--val-ratio", type=float, default=0.10)
    args = ap.parse_args()

    system_prompt = _build_system_prompt_with_tools()

    stats = {"total": 0, "kept": 0, "dropped": 0, "dup": 0}
    reasons: dict[str, int] = {}
    seen_sigs: set[str] = set()
    kept_by_source: dict[str, int] = {"lora": 0, "touchscreen": 0}
    kept: list[dict] = []

    with open(args.inp, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            stats["total"] += 1
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                stats["dropped"] += 1
                reasons["bad_json"] = reasons.get("bad_json", 0) + 1
                continue
            ok, why = _validate(d)
            if not ok:
                stats["dropped"] += 1
                reasons[why] = reasons.get(why, 0) + 1
                continue
            sig = _sig(d)
            if sig in seen_sigs:
                stats["dup"] += 1
                continue
            seen_sigs.add(sig)
            adapted = _adapt_to_gemma(d, system_prompt)
            if not adapted:
                stats["dropped"] += 1
                reasons["adapt_fail"] = reasons.get("adapt_fail", 0) + 1
                continue
            stats["kept"] += 1
            kept_by_source[d.get("source", "?")] = kept_by_source.get(d.get("source", "?"), 0) + 1
            kept.append(adapted)

    # Split train/val deterministically
    import random
    random.seed(42)
    random.shuffle(kept)
    n_val = max(1, int(len(kept) * args.val_ratio))
    val_set = kept[:n_val]
    train_set = kept[n_val:]

    args.out_train.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_train, "w", encoding="utf-8") as f:
        for d in train_set:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    with open(args.out_val, "w", encoding="utf-8") as f:
        for d in val_set:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"[STATS] {stats}")
    print(f"[REASONS] {sorted(reasons.items(), key=lambda x: -x[1])}")
    print(f"[KEPT_BY_SOURCE] {kept_by_source}")
    print(f"[OUT] train={len(train_set)} -> {args.out_train}")
    print(f"[OUT] val={len(val_set)} -> {args.out_val}")


if __name__ == "__main__":
    main()
