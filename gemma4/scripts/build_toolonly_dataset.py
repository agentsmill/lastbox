"""Build a tool-only SFT dataset for v6 warmup.

Filters train_v2.jsonl to keep only the prompts whose FIRST assistant turn is
a <tool_call>. Cuts the dialog off after that turn — no tool result, no final
answer. Output is paired (system+user → tool_call) for raw SFT, forcing the
model to set p(tool_call) ≈ 1.0 on first response.

Output: gemma4/data/train_v2_toolonly.jsonl
"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
IN_PATH = ROOT / "gemma4" / "data" / "train_v2.jsonl"
OUT_PATH = ROOT / "gemma4" / "data" / "train_v2_toolonly.jsonl"
TOOL_RE = re.compile(r"<tool_call>")


def main() -> int:
    out_rows: list[dict] = []
    with IN_PATH.open() as f:
        for line in f:
            d = json.loads(line)
            msgs = d.get("messages", [])
            if len(msgs) < 2:
                continue
            # find first assistant turn
            assistant_idx = None
            for i, m in enumerate(msgs):
                if m["role"] == "assistant":
                    assistant_idx = i
                    break
            if assistant_idx is None:
                continue
            first_assistant = msgs[assistant_idx]
            if not TOOL_RE.search(first_assistant.get("content", "")):
                continue
            # keep [system?, user, assistant_tool_call]
            kept = msgs[: assistant_idx + 1]
            out_rows.append({
                "messages": kept,
                "source": d.get("source"),
                "category": d.get("category"),
                "seed_id": d.get("seed_id"),
            })
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[toolonly] kept {len(out_rows)} pairs → {OUT_PATH}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
