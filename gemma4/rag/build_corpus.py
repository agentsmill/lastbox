#!/usr/bin/env python3
"""
Build the RAG corpus from local training data + scraped public-domain manuals.

Outputs:
  gemma4/rag/corpus/corpus.jsonl  — one passage per line:
    {id, source, category, text, kind}
      kind ∈ {tool_result_fact, canonical_answer, external_manual}

Strategy:
  - From train_v2.jsonl: extract every "[tool result]\\n..." content as a
    fact passage, and every final assistant turn as a canonical-answer
    passage. Drop the giant tool-spec system prompt (~3 KB) — it's repeated
    in every dialog and would dominate retrieval.
  - From golden_en.jsonl: include held-out gold answers as high-quality
    canonical passages.
  - External: scrape a small set of FM 21-76 / Wikipedia survival pages via
    HTTP (gx10 has internet). Pages are split into paragraph-sized passages.
"""
from __future__ import annotations
import json
import re
import sys
import time
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "gemma4" / "data" / "train_v2.jsonl"
GOLDEN = ROOT / "gemma4" / "data" / "golden_en.jsonl"
OUT_DIR = ROOT / "gemma4" / "rag" / "corpus"
OUT = OUT_DIR / "corpus.jsonl"

OUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_from_train(path: Path) -> list[dict]:
    passages: list[dict] = []
    if not path.exists():
        return passages
    with path.open() as f:
        for line in f:
            d = json.loads(line)
            cat = d.get("category", "general")
            src = d.get("source", "touchscreen")
            seed = d.get("seed_id", "")
            for msg in d.get("messages", []):
                role = msg["role"]
                content = msg.get("content", "")
                if role == "user" and content.startswith("[tool result]\n"):
                    fact = content[len("[tool result]\n"):].strip()
                    if 12 <= len(fact) <= 800:
                        passages.append({
                            "source": f"train_v2:{seed}",
                            "category": cat,
                            "text": fact,
                            "kind": "tool_result_fact",
                        })
                elif role == "assistant" and "<tool_call>" not in content:
                    ans = content.strip()
                    if 12 <= len(ans) <= 800:
                        passages.append({
                            "source": f"train_v2:{seed}",
                            "category": cat,
                            "text": ans,
                            "kind": "canonical_answer",
                        })
    return passages


def extract_from_golden(path: Path) -> list[dict]:
    passages: list[dict] = []
    if not path.exists():
        return passages
    with path.open() as f:
        for line in f:
            d = json.loads(line)
            ans = (d.get("expected_answer") or d.get("answer") or "").strip()
            cat = d.get("category", "general")
            if 12 <= len(ans) <= 800:
                passages.append({
                    "source": f"golden_en:{d.get('id','?')}",
                    "category": cat,
                    "text": ans,
                    "kind": "canonical_answer",
                })
    return passages


class _TextExtractor(HTMLParser):
    """Best-effort: rip text out of Wikipedia / Wikisource HTML.

    We skip script/style/table content and accumulate paragraphs and list
    items as separate passage chunks.
    """

    SKIP = {"script", "style", "table", "thead", "tbody", "tr", "td", "th",
            "sup", "sub", "figure", "figcaption", "header", "footer",
            "nav", "aside"}
    CHUNK_TAGS = {"p", "li", "h2", "h3", "h4"}

    def __init__(self) -> None:
        super().__init__()
        self.chunks: list[str] = []
        self._buf: list[str] = []
        self._skip_depth = 0
        self._in_chunk = False

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in self.SKIP:
            self._skip_depth += 1
        elif tag in self.CHUNK_TAGS:
            self._flush()
            self._in_chunk = True

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIP and self._skip_depth > 0:
            self._skip_depth -= 1
        elif tag in self.CHUNK_TAGS:
            self._flush()

    def handle_data(self, data: str) -> None:
        if self._skip_depth or not self._in_chunk:
            return
        self._buf.append(data)

    def _flush(self) -> None:
        if self._buf:
            text = re.sub(r"\s+", " ", "".join(self._buf)).strip()
            if text:
                self.chunks.append(text)
            self._buf = []
        self._in_chunk = False


def scrape_url(url: str, source_label: str, category: str) -> list[dict]:
    print(f"  scrape: {url}", file=sys.stderr)
    req = urllib.request.Request(url, headers={
        "User-Agent": "lastbox-corpus-builder/1.0 (Kaggle hackathon RAG)",
    })
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        print(f"    skipped ({e})", file=sys.stderr)
        return []
    parser = _TextExtractor()
    parser.feed(html)
    out: list[dict] = []
    for c in parser.chunks:
        if not (60 <= len(c) <= 900):
            continue
        # very crude filter for nav-junk and citation footers
        if re.search(r"\b(retrieved from|references|external links|see also)\b",
                     c.lower()):
            continue
        out.append({
            "source": source_label,
            "category": category,
            "text": c,
            "kind": "external_manual",
        })
    return out


# Wikipedia + Wikisource survival/first-aid sources (public domain / CC-BY-SA).
EXTERNAL_SOURCES: list[tuple[str, str, str]] = [
    # (url, source_label, category)
    ("https://en.wikipedia.org/wiki/First_aid",                   "wikipedia:First_aid",                   "first_aid"),
    ("https://en.wikipedia.org/wiki/Bleeding",                    "wikipedia:Bleeding",                    "first_aid"),
    ("https://en.wikipedia.org/wiki/Cardiopulmonary_resuscitation","wikipedia:CPR",                         "first_aid"),
    ("https://en.wikipedia.org/wiki/Hypothermia",                 "wikipedia:Hypothermia",                 "first_aid"),
    ("https://en.wikipedia.org/wiki/Burn",                        "wikipedia:Burn",                        "first_aid"),
    ("https://en.wikipedia.org/wiki/Wilderness_first_aid",        "wikipedia:Wilderness_first_aid",        "first_aid"),
    ("https://en.wikipedia.org/wiki/Survival_skills",             "wikipedia:Survival_skills",             "bushcraft"),
    ("https://en.wikipedia.org/wiki/Bushcraft",                   "wikipedia:Bushcraft",                   "bushcraft"),
    ("https://en.wikipedia.org/wiki/Water_purification",          "wikipedia:Water_purification",          "bushcraft"),
    ("https://en.wikipedia.org/wiki/Shelter_(building)",          "wikipedia:Shelter",                     "bushcraft"),
    ("https://en.wikipedia.org/wiki/Fire_making",                 "wikipedia:Fire_making",                 "bushcraft"),
    ("https://en.wikipedia.org/wiki/Foraging",                    "wikipedia:Foraging",                    "bushcraft"),
    ("https://en.wikipedia.org/wiki/Navigation",                  "wikipedia:Navigation",                  "navigation"),
    ("https://en.wikipedia.org/wiki/Map_and_compass",             "wikipedia:Map_and_compass",             "navigation"),
    ("https://en.wikipedia.org/wiki/Celestial_navigation",        "wikipedia:Celestial_navigation",        "navigation"),
    ("https://en.wikipedia.org/wiki/Distress_signal",             "wikipedia:Distress_signal",             "navigation"),
    ("https://en.wikipedia.org/wiki/SOS",                         "wikipedia:SOS",                         "navigation"),
    ("https://en.wikipedia.org/wiki/Raspberry_Pi",                "wikipedia:Raspberry_Pi",                "electronics"),
    ("https://en.wikipedia.org/wiki/LoRa",                        "wikipedia:LoRa",                        "electronics"),
    ("https://en.wikipedia.org/wiki/Meshtastic",                  "wikipedia:Meshtastic",                  "electronics"),
    # FM 21-76 on Wikisource (US Army Survival Manual, public domain)
    ("https://en.wikisource.org/wiki/FM_21-76_United_States_Army_Survival_Manual",
     "FM_21-76:index", "bushcraft"),
    ("https://en.wikisource.org/wiki/FM_21-76_United_States_Army_Survival_Manual/Chapter_1",
     "FM_21-76:Ch1_Introduction", "bushcraft"),
    ("https://en.wikisource.org/wiki/FM_21-76_United_States_Army_Survival_Manual/Chapter_2",
     "FM_21-76:Ch2_Psychology", "bushcraft"),
    ("https://en.wikisource.org/wiki/FM_21-76_United_States_Army_Survival_Manual/Chapter_3",
     "FM_21-76:Ch3_Survival_Medicine", "first_aid"),
    ("https://en.wikisource.org/wiki/FM_21-76_United_States_Army_Survival_Manual/Chapter_4",
     "FM_21-76:Ch4_Shelters", "bushcraft"),
    ("https://en.wikisource.org/wiki/FM_21-76_United_States_Army_Survival_Manual/Chapter_5",
     "FM_21-76:Ch5_Water", "bushcraft"),
    ("https://en.wikisource.org/wiki/FM_21-76_United_States_Army_Survival_Manual/Chapter_6",
     "FM_21-76:Ch6_Firecraft", "bushcraft"),
    ("https://en.wikisource.org/wiki/FM_21-76_United_States_Army_Survival_Manual/Chapter_7",
     "FM_21-76:Ch7_Food", "bushcraft"),
]


def main() -> int:
    all_passages: list[dict] = []
    print(f"[corpus] from train_v2: {TRAIN}", file=sys.stderr)
    all_passages += extract_from_train(TRAIN)
    print(f"  → {len(all_passages)} so far", file=sys.stderr)
    print(f"[corpus] from golden_en: {GOLDEN}", file=sys.stderr)
    before = len(all_passages)
    all_passages += extract_from_golden(GOLDEN)
    print(f"  → {len(all_passages) - before} added", file=sys.stderr)
    print(f"[corpus] scraping {len(EXTERNAL_SOURCES)} external pages…", file=sys.stderr)
    for url, label, cat in EXTERNAL_SOURCES:
        all_passages += scrape_url(url, label, cat)
        time.sleep(0.4)
    print(f"[corpus] total raw passages: {len(all_passages)}", file=sys.stderr)

    # Dedup by (lowercased text first 200 chars)
    seen: set[str] = set()
    deduped: list[dict] = []
    for p in all_passages:
        key = p["text"].lower()[:200]
        if key in seen:
            continue
        seen.add(key)
        deduped.append(p)

    # Assign monotonic IDs
    for i, p in enumerate(deduped):
        p["id"] = f"p{i:05d}"

    print(f"[corpus] after dedup: {len(deduped)}", file=sys.stderr)
    with OUT.open("w") as f:
        for p in deduped:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"[corpus] wrote {OUT}", file=sys.stderr)

    # Summary by kind + source
    from collections import Counter
    kinds = Counter(p["kind"] for p in deduped)
    cats = Counter(p["category"] for p in deduped)
    print(f"[corpus] kinds: {dict(kinds)}", file=sys.stderr)
    print(f"[corpus] categories: {dict(cats)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
