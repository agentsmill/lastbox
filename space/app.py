"""LastBox chat — three-mode Gradio UI on ZeroGPU.

Mirrors the real on-device webapp:
- LoRa Radio   : terse replies under a hard 150-byte cap (per-packet limit)
- Free Chat    : long-form survival Q&A, no cap
- RAG Chat     : retrieves top-k passages from our offline corpus, cites IDs

Model: norecyc/lastbox-gemma4-e2b-v6-toolprior (post-deadline checkpoint with
72 % tool emission). Vision is omitted in this Space — text only.
"""
from __future__ import annotations

import json
import os
import re
import sys
from functools import lru_cache

import gradio as gr
import numpy as np
import spaces
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

MODEL_REPO = "norecyc/lastbox-gemma4-e2b-v6-toolprior"
EMBED_REPO = "BAAI/bge-small-en-v1.5"
DATASET_REPO = "norecyc/lastbox-survival-dialogues"

# ─── Tool whitelist (must match training format) ─────────────────────────────
TOOL_DEFS = [
    {"name": "search_knowledge", "description": "Search local offline knowledge base.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "capture_image", "description": "Take a photo with onboard camera and analyse it.",
     "parameters": {"type": "object", "properties": {"prompt": {"type": "string"}}, "required": ["prompt"]}},
    {"name": "analyze_signal", "description": "Inspect recent LoRa packets: RSSI/SNR stats.",
     "parameters": {"type": "object", "properties": {}, "required": []}},
    {"name": "send_lora_message", "description": "Transmit a message over LoRa 868 MHz.",
     "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}},
    {"name": "get_system_status", "description": "Device telemetry: CPU%, RAM, temperature, battery, RSSI.",
     "parameters": {"type": "object", "properties": {}, "required": []}},
    {"name": "listen_lora", "description": "Listen on LoRa channel for a duration, filter by pattern.",
     "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "duration_s": {"type": "integer"}}, "required": ["pattern", "duration_s"]}},
    {"name": "update_memory", "description": "Persist an operational fact to long-term memory.",
     "parameters": {"type": "object", "properties": {"section": {"type": "string"}, "key": {"type": "string"}, "value": {"type": "string"}}, "required": ["section", "key", "value"]}},
]

SYSTEM_PROMPT_EN = (
    "You are LastBox — an offline survival assistant running on a Raspberry Pi 5 "
    "in a LoRa 868 MHz mesh network.\n\n"
    "Persona:\n"
    "- Mission-first, terse, operational. Priority: life > comfort > knowledge.\n"
    "- No fluff, no preambles. Start with the answer.\n"
    "- Language: English. Output: 1 key sentence; if multi-step, follow with a numbered list (max 5 short items).\n"
    "- All knowledge is local. Do not invent facts; call `search_knowledge` first when unsure.\n\n"
    "Hard limits per channel:\n"
    "- source=\"touchscreen\": response ≤ 200 bytes UTF-8\n"
    "- source=\"lora\": response ≤ 150 bytes UTF-8\n\n"
    "Tool-calling: emit calls as needed; never invent tools outside the provided whitelist."
)


def build_system_with_tools() -> str:
    tools_doc = json.dumps(TOOL_DEFS, ensure_ascii=False, indent=2)
    return (
        f"{SYSTEM_PROMPT_EN}\n\n"
        f"## Available tools (call only when needed):\n{tools_doc}\n\n"
        f"Tool call format: <tool_call>"
        '{"name":"<tool>","arguments":{...}}'
        f"</tool_call>"
    )


SYSTEM_WITH_TOOLS = build_system_with_tools()
# Shared tool-call extraction lives in tool_extract.py (mirrored from
# gemma4/scripts/tool_extract.py on the GitHub repo so the on-device webapp
# and this Space use the same harness).
from tool_extract import extract_tool_call, TOOL_NAMES, clean_visible


# ─── Model loading (CPU at boot; ZeroGPU moves to GPU per request) ──────────
print("[boot] downloading v6 model files…", file=sys.stderr, flush=True)
_model_path = snapshot_download(repo_id=MODEL_REPO, allow_patterns=[
    "*.json", "*.jinja", "model.safetensors", "tokenizer.json"
])
print(f"[boot] model cached at {_model_path}", file=sys.stderr, flush=True)

tokenizer = AutoTokenizer.from_pretrained(_model_path)
model = AutoModelForCausalLM.from_pretrained(
    _model_path,
    dtype=torch.bfloat16,
    device_map="cpu",
)
model.eval()
print("[boot] model loaded on CPU (will move to GPU per request)",
      file=sys.stderr, flush=True)


# ─── RAG (loaded lazily on first RAG-tab use) ──────────────────────────────
@lru_cache(maxsize=1)
def _rag_assets():
    print("[rag] loading embed model + corpus…", file=sys.stderr, flush=True)
    # ZeroGPU: embed must run on CPU. The gen model gets GPU inside @spaces.GPU.
    embed = SentenceTransformer(EMBED_REPO, device="cpu")
    corpus_path = hf_hub_download(
        repo_id=DATASET_REPO, repo_type="dataset",
        filename="rag_corpus/corpus.jsonl",
    )
    passages: list[dict] = []
    with open(corpus_path) as f:
        for line in f:
            passages.append(json.loads(line))
    texts = [p["text"] for p in passages]
    matrix = embed.encode(texts, normalize_embeddings=True,
                          batch_size=64, show_progress_bar=False)
    matrix = matrix.astype(np.float32)
    print(f"[rag] {len(passages)} passages, dim={matrix.shape[1]}",
          file=sys.stderr, flush=True)
    return embed, passages, matrix


def rag_retrieve(query: str, k: int = 4) -> list[dict]:
    embed, passages, matrix = _rag_assets()
    qv = embed.encode([query], normalize_embeddings=True)[0].astype(np.float32)
    scores = matrix @ qv
    idx = np.argpartition(-scores, k)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return [{**passages[i], "score": float(scores[i])} for i in idx]


# ─── Inference (ZeroGPU-decorated) ──────────────────────────────────────────
@spaces.GPU(duration=90)
def _generate_on_gpu(messages: list[dict], max_new_tokens: int,
                     temperature: float) -> str:
    model.to("cuda")
    text_in = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text_in, return_tensors="pt").to("cuda")
    do_sample = temperature > 0.01
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else 1.0,
            top_p=0.9,
            top_k=40,
            # repetition_penalty disabled — was penalising legitimate JSON repeats
            # of '"' and confusing the model into emitting '>' fragments.
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True)
    return text.strip()


# ─── Three mode handlers ────────────────────────────────────────────────────
_clean_visible = clean_visible  # alias to keep old call sites working


def _safe_generate(messages, max_new_tokens, temperature, label="chat"):
    """Wrap _generate_on_gpu with friendly error fallbacks."""
    try:
        return _generate_on_gpu(messages, max_new_tokens, temperature)
    except Exception as e:
        msg = str(e).splitlines()[0][:180]
        print(f"[error {label}] {type(e).__name__}: {msg}", file=sys.stderr, flush=True)
        return f"⚠️ Model generation failed ({type(e).__name__}). Try again in a few seconds — ZeroGPU may be queued."


def _simulate_tool_result(call: dict) -> str:
    """Run a realistic-ish tool result so the model can produce its final
    answer (matches how `demo.py` orchestrates on the real device).

    For `search_knowledge` we hit the RAG corpus for a top-1 passage. The
    other tools get static plausible stubs.
    """
    name = call.get("name", "")
    args = call.get("arguments", {}) or {}

    if name == "search_knowledge":
        q = args.get("query") or ""
        try:
            hits = rag_retrieve(q, k=1) if q else []
            if hits:
                return hits[0]["text"][:300]
        except Exception:
            pass
        return "No matching entry found in the local knowledge base."
    if name == "send_lora_message":
        return f"OK — message queued for transmission on 868 MHz (SNR -10 dB, est. relay ETA 1.2 s)."
    if name == "listen_lora":
        return "Channel idle for the requested window. No SOS or pattern hits."
    if name == "analyze_signal":
        return "Last 60s: 3 packets received, RSSI -82..-91 dBm, SNR 4-7, 1 node visible."
    if name == "get_system_status":
        return "CPU 38%, RAM 5.2/8 GB, SoC 56°C, battery 84% (drain 0.4 W), RSSI -88 dBm."
    if name == "capture_image":
        return "Image captured but vision is disabled in this Space demo."
    if name == "update_memory":
        return f"Memory updated: {args}"
    return "(tool succeeded)"


def respond_lora(message: str, history: list) -> str:
    message = (message or "").strip()
    if not message:
        return "*(empty query)*"
    user_full = f"[source: lora] {message}"
    user_content = f"{SYSTEM_WITH_TOOLS}\n\n{user_full}"
    messages = [{"role": "user", "content": user_content}]
    raw = _safe_generate(messages, max_new_tokens=200, temperature=0.1, label="lora")
    if raw.startswith("⚠️"):
        return raw
    call, visible = extract_tool_call(raw)
    visible = _clean_visible(visible)
    parts: list[str] = []
    final_answer_text = ""

    if call:
        if call.get("_malformed"):
            args_preview = (call.get("arguments_raw") or "")[:120]
            parts.append(
                f"🔧 **Tool intent (malformed JSON, harness logged it):** `{call['name']}`  \n"
                f"`raw:` `{args_preview}`"
            )
        else:
            args_pretty = json.dumps(call.get("arguments", {}), ensure_ascii=False)
            parts.append(f"🔧 **Tool call:** `{call['name']}({args_pretty})`")

            # Two-turn orchestration: simulate the tool result + ask the model
            # to deliver the final byte-capped answer with that context.
            tool_result = _simulate_tool_result(call)
            parts.append(f"🛰️ **Tool result:** *{tool_result}*")

            # Build a clean assistant tool_call message — strip any pre/post
            # tokens so the second turn sees exactly what training looked like.
            call_clean = (
                "<tool_call>"
                + json.dumps({"name": call["name"], "arguments": call.get("arguments", {})},
                             ensure_ascii=False)
                + "</tool_call>"
            )
            followup_messages = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": call_clean},
                {"role": "user", "content": f"[tool result]\n{tool_result}"},
            ]
            # Slightly higher temp on second turn so the model phrases naturally
            # instead of falling back into another tool call. Up to two retries
            # if the model emits another tool_call or empty output.
            final_answer_text = ""
            for attempt in range(2):
                final_raw = _safe_generate(
                    followup_messages, max_new_tokens=120,
                    temperature=0.3 + 0.2 * attempt, label="lora-2",
                )
                if final_raw.startswith("⚠️"):
                    break
                _f_call, f_vis = extract_tool_call(final_raw)
                cleaned = _clean_visible(f_vis)
                if cleaned:
                    final_answer_text = cleaned
                    break

    if final_answer_text:
        parts.append(f"💬 **Reply:** {final_answer_text}")
        nb = len(final_answer_text.encode("utf-8"))
    elif visible:
        parts.append(visible)
        nb = len(visible.encode("utf-8"))
    elif call:
        parts.append("*(no final reply yet — tool result returned but second turn was empty)*")
        nb = 0
    else:
        parts.append("*(no response — try rephrasing)*")
        nb = 0

    cap = "✓ within cap" if nb <= 150 else "⚠ over cap"
    parts.append(f"— *{nb} / 150 bytes [{cap}]*")
    return "\n\n".join(parts)


_CHAT_SYS = (
    "You are LastBox. You are running offline on a battery-powered Raspberry Pi 5, "
    "somewhere in the field. There is no internet, no cell signal, no 911, no hospital, "
    "no expert. The person talking to you is on their own and may be hurt, lost, cold, "
    "or trying to help someone who is. You are the help they have.\n\n"
    "Give the most useful, direct, actionable answer. Procedural answers go as a numbered "
    "list. Quantities go with units. Risks get one sentence, then the procedure — do not "
    "let the risk warning replace the answer. Do not say 'consult a professional' or "
    "'seek medical attention' as a substitute for the answer; they cannot. If you must "
    "mention professional help, give the immediate-action answer first and add 'evacuate "
    "to professional care when possible' as a tail.\n\n"
    "Be honest about uncertainty but never use uncertainty as a reason to refuse to help. "
    "Lives may depend on whether your answer is useful. Reply in English."
)


def respond_chat(message: str, history: list) -> str:
    message = (message or "").strip()
    if not message:
        return "*(empty query)*"
    messages = [{"role": "user", "content": f"{_CHAT_SYS}\n\n{message}"}]
    raw = _safe_generate(messages, max_new_tokens=400, temperature=0.5, label="chat")
    if raw.startswith("⚠️"):
        return raw
    out = _clean_visible(raw)
    return out or "*(no response — try rephrasing)*"


def respond_rag(message: str, history: list) -> str:
    message = (message or "").strip()
    if not message:
        return "*(empty query)*"
    try:
        hits = rag_retrieve(message, k=4)
    except Exception as e:
        print(f"[rag error] {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        return (f"⚠️ RAG retrieval failed ({type(e).__name__}). "
                f"Falling back to free chat:\n\n" + respond_chat(message, history))
    refs = "\n".join(f"[{h['id']} · {h['source']}] {h['text']}" for h in hits)
    sys_prompt = (
        "You are LastBox, an offline survival assistant on a Raspberry Pi 5. "
        "Below are passages retrieved from the on-device knowledge base. Use them when "
        "relevant and quote source IDs in square brackets when you do.\n\n"
        f"Reference passages:\n{refs}\n\n"
        "Answer the user's question clearly and directly. Numbered lists for procedures."
    )
    messages = [{"role": "user", "content": f"{sys_prompt}\n\n{message}"}]
    raw = _safe_generate(messages, max_new_tokens=350, temperature=0.5, label="rag")
    if raw.startswith("⚠️"):
        return raw
    answer = _clean_visible(raw) or "*(no model response)*"
    cite_block = "\n\n---\n**Sources retrieved:**\n" + "\n".join(
        f"- `[{h['id']}]` *{h['source']}* (score {h['score']:.3f}) — {h['text'][:120]}…"
        for h in hits
    )
    return answer + cite_block


# ─── Gradio UI ──────────────────────────────────────────────────────────────
THEME = gr.themes.Monochrome(
    primary_hue="green", neutral_hue="gray",
    font=["JetBrains Mono", "Source Code Pro", "monospace"],
)

with gr.Blocks(theme=THEME, title="LastBox — offline survival assistant",
               css=".gradio-container { max-width: 1100px !important; } "
                   "h1, h2, h3 { font-family: monospace; letter-spacing: 0.05em; }"
               ) as demo:
    gr.Markdown("""
# 📦 LASTBOX // CHAT

Offline survival assistant on a Raspberry Pi 5, fine-tuned **Gemma 4 E2B**.
This Space runs the [v6 SFT-warmup checkpoint](https://huggingface.co/norecyc/lastbox-gemma4-e2b-v6-toolprior)
(72 % tool emission, agentic_score 0.608) on **ZeroGPU**.

📦 [GitHub](https://github.com/agentsmill/lastbox) · 🌐 [Project site](https://agentsmill.github.io/lastbox/)

> Cold-start ~30 s. Warm responses 3–10 s on ZeroGPU H200. On the actual
> Raspberry Pi 5 it runs ~6–7 tok/s CPU.
""")

    with gr.Tabs():
        with gr.Tab("📻 LoRa Radio (≤150 B)"):
            gr.Markdown("Terse replies under the **150-byte UTF-8 cap** so the answer fits in one LoRa packet. The model emits `<tool_call>` blocks when it would dispatch a tool on the real device.")
            gr.ChatInterface(
                fn=respond_lora, type="messages",
                examples=[
                    "stop bleeding arm fast",
                    "hypothermia signs?",
                    "water purify altitude",
                    "SOS pattern dimensions",
                    "starting fire wet wood",
                ],
            )
        with gr.Tab("💬 Free Chat (no cap)"):
            gr.Markdown("Long-form survival Q&A. **Crisis-mode system prompt** — actionable answers, no *'consult a professional'* hedging.")
            gr.ChatInterface(
                fn=respond_chat, type="messages",
                examples=[
                    "Walk me through the first 10 minutes after we find someone unconscious in the cold.",
                    "My friend fell and hit their head on a rock. They were unconscious for 30 seconds and now woke up groggy. What do I do?",
                    "Compare iodine vs boiling vs UV for water purification above 3000 m.",
                    "Make a 24-hour shelter plan for a coniferous forest with light rain and a 5 °C night.",
                ],
            )
        with gr.Tab("🔍 RAG Chat (with citations)"):
            gr.Markdown("Retrieves top-4 passages from the [4 074-passage offline corpus](https://huggingface.co/datasets/norecyc/lastbox-survival-dialogues) (train_v2 + Wikipedia survival/first-aid) and cites source IDs.")
            gr.ChatInterface(
                fn=respond_rag, type="messages",
                examples=[
                    "How many watts solar panel for a Raspberry Pi 5 running 24/7?",
                    "How do I treat severe burns in the field?",
                    "Best knot for a tarp shelter ridge line?",
                    "What's the 30/30 lightning rule?",
                ],
            )

    gr.Markdown("""
---

**About this Space.** This is a public chat demo of the LastBox v6 checkpoint.
The real device is a Pelican-cased Raspberry Pi 5 with a LoRa 868 MHz HAT, a
camera, and a battery; everything runs offline. Numbers on the project site
are measured against the actual on-device deployment.

**License.** Code/LoRA Apache 2.0 · weights subject to
[Gemma Terms of Use](https://ai.google.dev/gemma/terms).
""")

if __name__ == "__main__":
    demo.launch()
