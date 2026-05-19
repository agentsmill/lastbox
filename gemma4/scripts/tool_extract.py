"""Robust tool_call extraction for LastBox Gemma 4 output.

The v6 fine-tuned model usually emits well-formed
`<tool_call>{"name":"…","arguments":{…}}</tool_call>` blocks, but under
temperature noise / specific tokens it sometimes produces:

  - bareword shorthand:   `search_knowledge{"query":"…"}`
  - missing tags:         `{"name":"…","arguments":{…}}`
  - newlines inside strings (breaks json.loads)
  - stray '<' '>' fragments adjacent to JSON
  - extra '}' inside string values (`"hypother}mia"`)
  - missing trailing `}` (early stop)

This module ships with both `webapp/server.py` (on-device) and the public HF
Space (`space/app.py`). Single source of truth so the same harness behaves
identically on the box and in the cloud demo.

Usage:
    from gemma4.scripts.tool_extract import extract_tool_call, TOOL_NAMES
    call, visible = extract_tool_call(raw_model_output)
"""
from __future__ import annotations

import json
import re

TOOL_NAMES = {
    "search_knowledge",
    "capture_image",
    "analyze_signal",
    "send_lora_message",
    "get_system_status",
    "listen_lora",
    "update_memory",
}

_STRICT_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_LOOSE_RE = re.compile(
    r"(?:<\s*tool_call\s*>\s*)?"
    r"\{[^{}]*?\"name\"\s*:\s*\"(?P<name1>[a-z_]+)\".*?\}"
    r"(?:\s*<\s*/\s*tool_call\s*>)?",
    re.DOTALL,
)
_BARE_RE = re.compile(
    r"(?<![a-z_])(?P<name2>" + "|".join(re.escape(n) for n in TOOL_NAMES) + r")"
    r"\s*\{(?P<body>.*?)\}(?!\s*\})",
    re.DOTALL,
)


def _resolve_name(candidate: str) -> str | None:
    """Map noisy/truncated tool names to canonical ones.

    The v6 model sometimes emits "send_lor>a" or "send_lor" instead of
    "send_lora_message". Match by prefix or by Levenshtein-ish containment.
    """
    if not candidate:
        return None
    cand = candidate.lower().strip()
    if cand in TOOL_NAMES:
        return cand
    # Strip non-[a-z_] chars (drops stray '>', '<', digits, etc.)
    norm = re.sub(r"[^a-z_]", "", cand)
    if norm in TOOL_NAMES:
        return norm
    # Prefix match (e.g. "send_lor" → "send_lora_message")
    for name in TOOL_NAMES:
        if name.startswith(norm) and len(norm) >= 5:
            return name
    # Substring containment (e.g. "lora_message" → "send_lora_message")
    for name in TOOL_NAMES:
        if norm and norm in name and len(norm) >= 5:
            return name
    return None


def _scrub(s: str) -> str:
    """Strip common corruption patterns inside JSON-looking text."""
    # Standalone '<' / '>' that are not part of valid tag (after _STRICT regex matched its own)
    s = re.sub(r'(?<!tool_call)(?<!/tool_call)[<>]', '', s)
    # Normalise newlines / tabs / extra '}' inside JSON string values
    out: list[str] = []
    in_str = False
    escape = False
    for ch in s:
        if escape:
            out.append(ch)
            escape = False
            continue
        if ch == "\\":
            out.append(ch)
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            out.append(ch)
            continue
        if in_str and ch in "\n\r\t":
            out.append(" ")
            continue
        if in_str and ch == "}":
            # broken brace inside a string value — drop it
            continue
        out.append(ch)
    return "".join(out)


def _try_json(text: str) -> dict | None:
    """Parse JSON; on failure run a series of repairs and try again."""
    text = (text or "").strip()
    if not text:
        return None
    candidates: list[str] = [text, _scrub(text)]
    scrubbed = candidates[1]
    # Truncate at first balanced top-level }
    depth = 0
    cut = -1
    for i, ch in enumerate(scrubbed):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                cut = i + 1
                break
    if cut > 0:
        candidates.append(scrubbed[:cut])
    for c in list(candidates):
        candidates += [c + "}", c.rstrip(",") + "}", c + "\"}"]
    for c in candidates:
        try:
            return json.loads(c)
        except (json.JSONDecodeError, ValueError):
            continue
    open_count = scrubbed.count("{")
    close_count = scrubbed.count("}")
    if open_count > close_count:
        try:
            return json.loads(scrubbed + "}" * (open_count - close_count))
        except json.JSONDecodeError:
            pass
    return None


def extract_tool_call(text: str) -> tuple[dict | None, str]:
    """Return (parsed_call, visible_text_with_call_stripped).

    parsed_call schema:
      {"name": <tool>, "arguments": <dict>}              normal
      {"name": <tool>, "arguments_raw": <str>, "_malformed": True}  fallback
    """
    text = text or ""

    # 1. Strict <tool_call>{…}</tool_call>
    m = _STRICT_RE.search(text)
    if m:
        parsed = _try_json(m.group(1))
        if parsed:
            resolved = _resolve_name(parsed.get("name", ""))
            if resolved:
                parsed["name"] = resolved
                return parsed, (text[:m.start()] + text[m.end():]).strip()

    # 2. Loose JSON-looking call (with or without tags)
    m = _LOOSE_RE.search(text)
    if m:
        snippet = m.group(0).replace("<tool_call>", "").replace("</tool_call>", "")
        parsed = _try_json(snippet)
        if parsed and parsed.get("name") in TOOL_NAMES:
            return parsed, (text[:m.start()] + text[m.end():]).strip()

    # 3. Bareword shorthand "toolname{...}"
    m = _BARE_RE.search(text)
    if m:
        name = m.group("name2")
        body = m.group("body")
        # Strip stray inner-string '}' that frequently appears
        cleaned = re.sub(r'"([^"]*?)\}([^"]*?)"', r'"\1\2"', body)
        for envelope in (
            '{"name":"' + name + '","arguments":{' + cleaned + '}}',
            '{"name":"' + name + '","arguments":' + cleaned + '}',
            '{' + cleaned + '}',
        ):
            parsed = _try_json(envelope)
            if parsed and (parsed.get("name") == name or "arguments" in parsed):
                if "name" not in parsed:
                    parsed = {"name": name, "arguments": parsed}
                return parsed, (text[:m.start()] + text[m.end():]).strip()
        # Malformed bareword: model usually emits trailing junk like "mia\"}"
        # after a broken-brace inner string. Extend the strip to the next safe
        # boundary (newline or end of string).
        tail = text[m.end():]
        # Drop characters up to (and including) the next \n, or up to ~30 chars
        # of garbage, whichever comes first.
        tail_clean = re.sub(r"^[^\n]{0,40}(?:\n|$)", "", tail, count=1)
        return (
            {"name": name, "arguments_raw": body.strip(), "_malformed": True},
            (text[:m.start()] + tail_clean).strip(),
        )

    # 4. No usable parse — but the text may still contain a tool_call attempt
    # we want to surface as malformed intent (so the user sees "model tried to
    # call X" instead of an empty reply). Also strip the ruined fragment from
    # the visible text.
    junked = text
    malformed_intent: dict | None = None

    # Try to detect intended tool name from any leftover '<tool_call>{"name":"X"' fragment
    name_match = re.search(
        r"<\s*tool_call\s*>[^{]*?\{[^}]*?\"name\"\s*:\s*\"([a-z_]+)",
        junked, flags=re.DOTALL,
    )
    if name_match:
        intended = _resolve_name(name_match.group(1)) or name_match.group(1)
        malformed_intent = {
            "name": intended,
            "arguments_raw": text[name_match.end():][:120],
            "_malformed": True,
        }

    # Strip well-formed strict block (already handled above, but rerun for safety)
    junked = _STRICT_RE.sub("", junked)

    # Strip `<tool_call>` opening through `</tool_call>` or EOL
    junked = re.sub(
        r"<\s*tool_call\s*>.*?(?:</\s*tool_call\s*>|$)", "", junked, flags=re.DOTALL,
    )

    # Strip JSON-looking blob anchored on a closing `</tool_call>`
    junked = re.sub(
        r"\{[^{}]*?\"name\".*?</\s*tool_call\s*>", "", junked, flags=re.DOTALL,
    )

    # Strip standalone `</tool_call>` orphan closing tag
    junked = re.sub(r"</\s*tool_call\s*>", "", junked)

    # If the remaining text starts with a `{` and ends with `}` and contains
    # `"name"` — it's a leaked tool_call JSON without tags. Drop it.
    stripped = junked.strip()
    if stripped.startswith("{") and stripped.endswith("}") and '"name"' in stripped:
        if malformed_intent is None:
            # Try to grab name from this leftover JSON too
            jm = re.search(r"\"name\"\s*:\s*\"([a-z_]+)\"", stripped)
            if jm:
                malformed_intent = {
                    "name": jm.group(1),
                    "arguments_raw": stripped[:120],
                    "_malformed": True,
                }
        junked = ""

    # If a bareword tool name remains followed by '{', strip up to next '}'
    for n in TOOL_NAMES:
        bare_match = re.search(re.escape(n) + r"\s*\{([^}]*)\}?", junked, flags=re.DOTALL)
        if bare_match and malformed_intent is None:
            malformed_intent = {
                "name": n,
                "arguments_raw": bare_match.group(1)[:120],
                "_malformed": True,
            }
        junked = re.sub(
            re.escape(n) + r"\s*\{[^}]*\}?",
            "", junked, flags=re.DOTALL,
        )

    return malformed_intent, junked.strip()


def clean_visible(text: str) -> str:
    """Light post-processing of visible model output for display."""
    if not text:
        return ""
    text = re.sub(r"^\s*\[source:\s*\w+\]\s*", "", text)
    # Strip any leftover full <tool_call>...</tool_call> block (with JSON inside)
    text = re.sub(
        r"<\s*tool_call\s*>.*?(?:</\s*tool_call\s*>|$)",
        "", text, flags=re.DOTALL,
    )
    # Then strip orphan tags
    text = re.sub(r"<\s*/?\s*tool_call\s*>", "", text)
    # Drop orphan single-line JSON-only blob that survives (model attempted tool_call without tags)
    text = re.sub(r'^\s*\{\s*"name"\s*:.*?\}\s*$', "", text, flags=re.DOTALL)
    # Drop stray single < or > on word boundaries
    text = re.sub(r"\s+>\s*$", "", text)
    text = re.sub(r"^\s*<\s+", "", text)
    # Collapse 3+ newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip trailing standalone braces or commas
    text = re.sub(r"^\s*[}\],]+\s*$", "", text, flags=re.MULTILINE)
    return text.strip()


if __name__ == "__main__":
    samples = [
        '<tool_call>{"name":"search_knowledge","arguments":{"query":"x"}}</tool_call>',
        'search_knowledge{"query":"signs of hypother}mia"}',
        '<tool_call>{"name":"send_lor>a","arguments":{"text":"Stop bleeding."}}</tool_call>',
        '{"name":"search_knowledge","}\n{\n "query":"SOS pattern"\n}</tool_call>',
        'No tool. Just an answer here.',
        '',
        # Combo: text + tool_call mid-sentence (model verbose run)
        'Apply pressure. <tool_call>{"name":"search_knowledge","arguments":{"query":"shock"}}</tool_call> Elevate the limb.',
        # Trailing tool_call junk
        '1. Step one.\nsearch_knowledge{"query":"first aid'
    ]
    for s in samples:
        call, vis = extract_tool_call(s)
        print(f"\nin   : {s!r}")
        print(f"call : {call}")
        print(f"vis  : {vis!r}")
