"""LastBox demo: end-to-end loop over llama-server with 7 tools.

Talks to llama-server on http://llama-server:8080/v1 (Docker network) or
http://localhost:11436/v1 (host). Parses <tool_call> emissions, dispatches
to real (psutil, toml) or mock tools, feeds tool result back, gets final answer.

Run modes:
  python demo.py --query "How do I stop a heavy bleed on the forearm?"
  python demo.py --interactive
  python demo.py --batch golden_en.jsonl
  python demo.py --image hand_wound.jpg --query "What's wrong with this?"
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import re
import sys
import time
from pathlib import Path

import aiohttp

ENDPOINT = os.environ.get("LASTBOX_ENDPOINT", "http://localhost:11436/v1")
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
MEMORY_PATH = Path(os.environ.get("LASTBOX_MEMORY", "/data/lastbox_memory.toml"))

SYSTEM_PROMPT = """You are LastBox — an offline survival assistant running on a Raspberry Pi 5 in a LoRa 868 MHz mesh network.

Persona:
- Mission-first, terse, operational. Priority: life > comfort > knowledge.
- No fluff, no preambles. Start with the answer.
- Language: English. Output: 1 key sentence; if multi-step, follow with a numbered list (max 5 short items).
- All knowledge is local. You have no internet. Call `search_knowledge` first when unsure.

Hard limits per channel:
- source="touchscreen": response ≤ 200 bytes UTF-8
- source="lora": response ≤ 150 bytes UTF-8

Tool call format: <tool_call>{"name":"<tool>","arguments":{...}}</tool_call>"""


# ====================  MOCK / REAL TOOLS  ====================

KNOWLEDGE_FACTS = {
    "bleed": "Direct pressure 5-10 min; elevate limb; tourniquet 5-10 cm proximal only if life-threatening and pressure fails. Call EMS.",
    "cpr": "Adult CPR: 30 compressions @ 100-120/min, depth 5-6 cm center of chest, then 2 breaths. Continue until pulse or help arrives.",
    "hypothermia": "Mild: shivering, slurred speech — warm gradually, dry clothes, sweet warm drink. Severe: no shivering, confusion — handle gently, prevent further loss, evacuate.",
    "burn": "Cool with running water 20 min, cover loose with sterile dressing, no creams. 3rd degree → no water on charred tissue, just cover, evacuate.",
    "shock": "Lay flat, legs raised 30 cm, keep warm, no food/drink, monitor airway, evacuate ASAP.",
    "boil": "Boil 1 min at sea level, 3 min above 2000 m. Kills bacteria, virus, protozoa.",
    "fire": "Wet wood: split for dry interior; feather sticks; ferro rod on cotton/lint tinder; tipi build with airflow.",
    "shelter": "Tarp A-frame: ridgeline taut, 30° pitch, low side to wind, ground sheet + insulation underneath.",
    "sodis": "PET bottle, clear water, 6 h direct sun (UV-A inactivates pathogens). Cloudy day → 2 days.",
    "north": "Stick shadow method: place stick, mark shadow tip, wait 15 min, mark again. Line W→E. Perpendicular = N-S (N pointing away from sun in N hemisphere).",
    "signal": "Air-to-ground: V = needs assistance; X = unable to proceed. Mirror flash 3× = SOS. Smoke: 3 fires triangle.",
    "sos": "SOS pattern: 3m × 3m letters, high contrast vs ground. Visible up to 5 km from aircraft.",
    "solar": "Pi 5 LastBox draws ~5 W avg. Min 20 W panel + 30 Ah 12 V battery with MPPT for 24/7 in sun.",
    "lightning": "30/30 rule: if thunder ≤30 s after flash, take cover; wait 30 min after last thunder. Avoid ridges, lone trees.",
    "iodine": "KI tablet adult dose 130 mg once, take within 1 h of exposure. Repeat per official guidance only.",
    "undervolt": "Check: PSU rating ≥5V/5A for Pi 5; USB-C cable AWG ≤20; no powered USB peripherals; dmesg | grep undervolt.",
    "mesh": "Meshtastic: install firmware, set region EU_868, share PSK with peers, position node high for LoS.",
    "bme280": "Check i2c on Pi 5: enable in raspi-config; i2cdetect -y 1; pull-ups 4.7k on SDA/SCL; verify 3.3V power.",
    "sx1262": "SX1262 HAT: enable SPI, install Meshtastic-python, configure GPIO pins per HAT docs, set TX power ≤25 dBm.",
    "infant": "Infant CPR: 30 compressions 2 fingers, depth 4 cm, then 2 small breaths. Call EMS, continue until response.",
    "arterial": "Arterial bleed: direct pressure with gauze, pack wound, elevate, tourniquet 5 cm above wound, mark time.",
}


def search_knowledge(args: dict) -> str:
    q = (args.get("query") or "").lower()
    for key, val in KNOWLEDGE_FACTS.items():
        if key in q:
            return val
    return "No local entry. General guidance: prioritize safety, call EMS if available, follow basic ABC (airway/breathing/circulation)."


def capture_image(args: dict, image_path: str | None = None) -> str:
    if image_path and Path(image_path).exists():
        return f"[Image {image_path} captured, {Path(image_path).stat().st_size} bytes — vision model will analyse via /v1 with image attachment]"
    prompt = args.get("prompt", "scene")
    return f"[Camera mock] Scene analysis requested: {prompt}. No live camera attached in this demo; LLM may answer from text context."


def analyze_signal(args: dict) -> str:
    return "LoRa: 4 active nodes (B,C,D,E), median RSSI -85 dBm, SNR +4 dB, link quality good. Last packet 12 s ago from node B."


def send_lora_message(args: dict) -> str:
    text = args.get("text", "")
    log = Path("/tmp/lastbox_lora_outbox.log")
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as f:
        f.write(f"{time.strftime('%H:%M:%S')}  TX[{len(text.encode('utf-8'))}B]: {text}\n")
    return f"Transmitted {len(text.encode('utf-8'))} bytes over LoRa 868 MHz (logged to {log})."


def get_system_status(args: dict) -> str:
    try:
        import psutil
        cpu = psutil.cpu_percent(interval=0.5)
        mem = psutil.virtual_memory()
        temps = psutil.sensors_temperatures() if hasattr(psutil, "sensors_temperatures") else {}
        temp_c = None
        for arr in temps.values():
            for t in arr:
                if t.current:
                    temp_c = t.current
                    break
            if temp_c:
                break
        return f"CPU {cpu:.0f}%, RAM {mem.percent:.0f}% ({mem.used/1e9:.1f}/{mem.total/1e9:.1f}GB), temp {temp_c or '?'}°C, LoRa RSSI -85 dBm"
    except Exception as e:
        return f"[status mock] CPU 28%, RAM 38%, temp 52°C, LoRa RSSI -85 dBm (psutil err: {e})"


def listen_lora(args: dict) -> str:
    pat = args.get("pattern", "")
    dur = int(args.get("duration_s", 5))
    time.sleep(min(dur, 3))  # cap simulated delay at 3s for demo
    return f"Listened {dur}s for '{pat}': 1 match — 'SOS GPS 50.12N 19.93E injured leg' from node C @ t-{dur-2}s, RSSI -91 dBm."


def update_memory(args: dict) -> str:
    section = args.get("section")
    key = args.get("key")
    value = (args.get("value") or "")[:100]
    if section not in ("nodes", "topics", "alerts", "notes"):
        return f"[memory] section '{section}' rejected (must be nodes/topics/alerts/notes)."
    MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    line = f'[{section}]\n{key} = "{value}"\n'
    with MEMORY_PATH.open("a", encoding="utf-8") as f:
        f.write(line)
    return f"[memory] {section}.{key} persisted ({len(value)} chars)."


TOOLS = {
    "search_knowledge": search_knowledge,
    "capture_image": capture_image,
    "analyze_signal": analyze_signal,
    "send_lora_message": send_lora_message,
    "get_system_status": get_system_status,
    "listen_lora": listen_lora,
    "update_memory": update_memory,
}


# ====================  LLAMA-SERVER CLIENT  ====================

async def chat_once(session: aiohttp.ClientSession, messages: list[dict], image_b64: str | None = None) -> tuple[str, float]:
    """Send messages to llama-server, return (assistant_text, first_token_ms)."""
    msgs_payload = list(messages)
    if image_b64 and msgs_payload and msgs_payload[-1]["role"] == "user":
        msgs_payload[-1] = {
            "role": "user",
            "content": [
                {"type": "text", "text": msgs_payload[-1]["content"]},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            ],
        }
    payload = {
        "messages": msgs_payload,
        "temperature": 0.3,
        "max_tokens": 256,
        "stream": True,
    }
    text_parts: list[str] = []
    first_ms: float | None = None
    t0 = time.perf_counter()
    async with session.post(f"{ENDPOINT}/chat/completions", json=payload, timeout=aiohttp.ClientTimeout(total=90)) as resp:
        resp.raise_for_status()
        async for chunk in resp.content:
            line = chunk.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            try:
                o = json.loads(body)
            except Exception:
                continue
            delta = o["choices"][0].get("delta", {})
            piece = delta.get("content") or ""
            if piece:
                if first_ms is None:
                    first_ms = (time.perf_counter() - t0) * 1000
                text_parts.append(piece)
    return "".join(text_parts), first_ms or 0.0


async def run_dialog(query: str, source: str = "touchscreen", image_path: str | None = None) -> dict:
    image_b64 = None
    if image_path and Path(image_path).exists():
        image_b64 = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"[source: {source}] {query}"},
    ]
    trajectory = {"query": query, "source": source, "turns": []}
    async with aiohttp.ClientSession() as session:
        for hop in range(4):  # max 4 hops (LLM + tool roundtrips)
            t0 = time.perf_counter()
            assistant_text, ftm = await chat_once(session, messages, image_b64=image_b64 if hop == 0 else None)
            total_ms = (time.perf_counter() - t0) * 1000
            trajectory["turns"].append({
                "hop": hop,
                "assistant_raw": assistant_text,
                "first_token_ms": ftm,
                "total_ms": total_ms,
            })
            messages.append({"role": "assistant", "content": assistant_text})

            calls = []
            for m in TOOL_CALL_RE.finditer(assistant_text):
                try:
                    calls.append(json.loads(m.group(1)))
                except json.JSONDecodeError:
                    continue

            if not calls:
                trajectory["final_response"] = TOOL_CALL_RE.sub("", assistant_text).strip()
                break

            for call in calls[:2]:  # max 2 tools per hop
                name = call.get("name")
                args = call.get("arguments", {}) or {}
                fn = TOOLS.get(name)
                if not fn:
                    result = f"[tool result] Unknown tool: {name}"
                else:
                    try:
                        if name == "capture_image":
                            result = fn(args, image_path=image_path)
                        else:
                            result = fn(args)
                    except Exception as e:
                        result = f"[tool error] {name}: {e}"
                trajectory["turns"][-1].setdefault("tool_calls", []).append({"name": name, "args": args, "result": result})
                messages.append({"role": "user", "content": f"[tool result]\n{result}"})
        else:
            trajectory["final_response"] = "[max hops reached]"
    return trajectory


# ====================  CLI  ====================

async def amain(args):
    if args.batch:
        n_ok = 0
        n = 0
        with open(args.batch, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                g = json.loads(line)
                t = await run_dialog(g["user"], source=g.get("source", "touchscreen"))
                n += 1
                ok = bool(t.get("final_response"))
                if ok:
                    n_ok += 1
                ftm = t["turns"][0]["first_token_ms"]
                print(f"[{g['id']}] {'OK' if ok else 'FAIL'} ftm={ftm:.0f}ms response={(t.get('final_response') or '')[:120]!r}")
        print(f"\n{n_ok}/{n} OK")
        return

    if args.interactive:
        print("LastBox demo (interactive). Type 'quit' to exit, prefix with 'lora:' for source=lora.")
        while True:
            try:
                q = input("\nyou> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not q or q.lower() == "quit":
                break
            source = "touchscreen"
            if q.lower().startswith("lora:"):
                source = "lora"
                q = q[5:].strip()
            t = await run_dialog(q, source=source, image_path=args.image)
            for turn in t["turns"]:
                if turn.get("tool_calls"):
                    for tc in turn["tool_calls"]:
                        print(f"  [{tc['name']}({tc['args']})] -> {tc['result'][:200]}")
            print(f"lastbox> {t.get('final_response')}")
            print(f"  (first token {t['turns'][0]['first_token_ms']:.0f} ms)")
        return

    if args.query:
        t = await run_dialog(args.query, source=args.source, image_path=args.image)
        print(json.dumps(t, ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", "-q", help="One-shot query")
    ap.add_argument("--source", "-s", default="touchscreen", choices=["touchscreen", "lora"])
    ap.add_argument("--image", "-i", help="Image path for vision query")
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--batch", help="Path to golden jsonl for batch eval")
    args = ap.parse_args()
    if not (args.query or args.interactive or args.batch):
        ap.print_help()
        sys.exit(0)
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
