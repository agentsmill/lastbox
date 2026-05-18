"""Adapter datasetu LastBox -> format kompatybilny z Gemma 4 chat template.

Wejście: data/clean/{train,val}.jsonl (Qwen-style messages z tool_calls)
Wyjście: gemma4/data/{train,val}.jsonl (messages w formacie Gemma, tool_calls
        serializowane jako tekst inline <tool_call>{json}</tool_call>)

Gemma 4 nie obsługuje natywnie tool_calls jako oddzielnego pola w chat template
(stan na 2026-05). Najbezpieczniejsza ścieżka to inline serializacja w content
assistant'a — runtime (src/agent/loop.py) parsuje ten format regexem.
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"


def _adapt_message(msg: dict) -> dict | None:
    role = msg.get("role")
    content = msg.get("content", "") or ""
    if role in ("system", "user"):
        return {"role": role, "content": content}
    if role == "assistant":
        parts = []
        if content:
            parts.append(content)
        for tc in msg.get("tool_calls", []) or []:
            fn = tc.get("function", tc)
            name = fn.get("name")
            args = fn.get("arguments", {})
            payload = {"name": name, "arguments": args}
            parts.append(f"{TOOL_CALL_OPEN}{json.dumps(payload, ensure_ascii=False)}{TOOL_CALL_CLOSE}")
        return {"role": "assistant", "content": "\n".join(parts).strip()}
    if role == "tool":
        return {"role": "user", "content": f"[wynik narzędzia]\n{content}"}
    return None


def _adapt_example(ex: dict, system_prompt: str | None) -> dict | None:
    msgs = ex.get("messages") or []
    out: list[dict] = []
    if system_prompt and not any(m.get("role") == "system" for m in msgs):
        out.append({"role": "user", "content": system_prompt})
    for m in msgs:
        am = _adapt_message(m)
        if am is None:
            continue
        if am["role"] == "system":
            am = {"role": "user", "content": am["content"]}
        if out and out[-1]["role"] == am["role"]:
            out[-1]["content"] = f"{out[-1]['content']}\n\n{am['content']}".strip()
        else:
            out.append(am)
    if not out or not any(m["role"] == "assistant" for m in out):
        return None
    return {"messages": out, "source": ex.get("source", "touchscreen"),
            "category": ex.get("category"), "seed_id": ex.get("seed_id")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-train", type=Path, default=Path("data/clean/train.jsonl"))
    ap.add_argument("--in-val", type=Path, default=Path("data/clean/val.jsonl"))
    ap.add_argument("--out-dir", type=Path, default=Path("gemma4/data"))
    ap.add_argument("--inject-system", action="store_true",
                    help="Dodaj system prompt z src/agent/soul.py jako pierwszy user turn")
    args = ap.parse_args()

    sys.path.insert(0, str(Path.cwd()))
    system_prompt = None
    if args.inject_system:
        from src.agent.soul import build_system_prompt, load_soul, get_all_tool_definitions
        soul = load_soul()
        sp = build_system_prompt(soul)
        tools = get_all_tool_definitions()
        tools_doc = json.dumps([t["function"] for t in tools], ensure_ascii=False, indent=2)
        system_prompt = (
            f"{sp}\n\n## Dostępne narzędzia (wywołuj tylko gdy potrzeba):\n{tools_doc}\n\n"
            f"Format wywołania: {TOOL_CALL_OPEN}"
            "{\"name\":\"<tool>\",\"arguments\":{...}}"
            f"{TOOL_CALL_CLOSE}"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split, p in [("train", args.in_train), ("val", args.in_val)]:
        if not p.exists():
            print(f"[WARN] Brak {p}, pomijam {split}")
            continue
        kept = 0
        dropped = 0
        out_path = args.out_dir / f"{split}.jsonl"
        with open(p, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                try:
                    ex = json.loads(line)
                except json.JSONDecodeError:
                    dropped += 1
                    continue
                adapted = _adapt_example(ex, system_prompt)
                if adapted is None:
                    dropped += 1
                    continue
                fout.write(json.dumps(adapted, ensure_ascii=False) + "\n")
                kept += 1
        print(f"[{split}] kept={kept} dropped={dropped} -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
