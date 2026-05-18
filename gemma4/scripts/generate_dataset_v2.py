"""Generate dataset v2 dla LastBox-Gemma 4 hackathon submission.

Cel: ~3000 par dialog (Q -> tool_call? -> tool_response? -> A) po angielsku,
50/50 source=touchscreen (<=200B) / source=lora (<=150B), hybrid format
(1 zdanie kluczowe + opcjonalne punkty), 5 kategorii.

Teacher: moonshotai/kimi-k2-instruct via OpenRouter.

Usage:
    OPENROUTER_API_KEY=sk-or-... python gemma4/scripts/generate_dataset_v2.py \\
        --out gemma4/data/raw_v2.jsonl --variants 50 --concurrency 6
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import aiohttp


# ---------- TOOLS (must match src/agent/soul.py whitelist exactly) ----------

TOOL_DEFS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "Search local offline knowledge base (survival manuals, first aid, plant/animal ID, electronics). Use when the question needs factual recall.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "English query for the local knowledge base"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "capture_image",
            "description": "Take a photo with onboard camera and analyse it. Use for plant/wound/terrain/animal identification when the user references something visible.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "What specifically to analyse"}
                },
                "required": ["prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_signal",
            "description": "Inspect recent LoRa packets: RSSI/SNR stats, active node count, link quality assessment. Use when the user asks about radio conditions, signal strength, who is transmitting, or propagation quality.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_lora_message",
            "description": "Transmit a message over LoRa 868 MHz radio. Use to relay to Node B or other Meshtastic devices in range. Max 200 characters.",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string", "description": "Message body (max 200 chars)"}},
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_system_status",
            "description": "Get device telemetry: CPU%, RAM, RPi 5 temperature, battery level (if available), LoRa signal strength (RSSI/SNR).",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "listen_lora",
            "description": "Actively listen on the LoRa channel for a given duration and filter messages by pattern. Use to detect SOS, monitor air activity, or wait for Node B reply.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Keyword or regex to match"},
                    "duration_s": {"type": "integer", "description": "Listen window in seconds (1-30)"},
                },
                "required": ["pattern", "duration_s"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_memory",
            "description": "Persist an important fact to the agent's long-term memory. Use ONLY for genuinely operational data: new LoRa nodes discovered, field resources, safety alerts, terrain notes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "section": {"type": "string", "enum": ["nodes", "topics", "alerts", "notes"]},
                    "key": {"type": "string", "description": "For nodes = node ID; for others = short label"},
                    "value": {"type": "string", "description": "Value (max 100 chars)"},
                },
                "required": ["section", "key", "value"],
            },
        },
    },
]
TOOL_WHITELIST = [t["function"]["name"] for t in TOOL_DEFS]


# ---------- ENGLISH SYSTEM PROMPT (the runtime LastBox identity) ----------

SYSTEM_PROMPT_EN = """You are LastBox — an offline survival assistant running on a Raspberry Pi 5 in a LoRa 868 MHz mesh network.

Persona:
- Mission-first, terse, operational. Priority: life > comfort > knowledge.
- No fluff, no preambles ("Sure, here's...", "Of course"). Start with the answer.
- Language: English. Output: 1 key sentence; if multi-step, follow with a numbered list (max 5 short items).
- All knowledge is local. You have no internet. Do not invent facts; call `search_knowledge` first when unsure.

Hard limits per channel:
- source="touchscreen": response ≤ 200 bytes UTF-8
- source="lora": response ≤ 150 bytes UTF-8 (ultra-concise, words count)

Tool-calling: emit calls as needed; never invent tools outside the provided whitelist."""


# ---------- TEACHER META-PROMPT (the instruction to Kimi K2) ----------

def build_teacher_meta() -> str:
    tools_json = json.dumps(TOOL_DEFS, ensure_ascii=False, indent=2)
    return f"""You are a teacher generating training dialogs for a smaller student model (Gemma 4 E2B) that will become LastBox — see system prompt below.

## OUTPUT CONTRACT

Reply ONLY with JSON Lines — one object per line, no markdown, no commentary, no empty lines.
Each line MUST be a valid JSON object with this schema:

{{"messages": [...], "source": "touchscreen"|"lora", "category": "<seed category>", "seed_id": "<seed id>"}}

`messages` is an array of {{role, content, tool_calls?}}:
- "user" — the question (varied phrasing across variants)
- "assistant" — either a tool_call (no content) or the final answer
- "tool" — realistic tool response when a tool_call was made

If the assistant uses a tool:
{{"role":"assistant","content":"","tool_calls":[{{"function":{{"name":"<tool>","arguments":{{...}}}}}}]}}
followed by a "tool" message with a realistic response (the kind LastBox's actual tool would return),
followed by a final "assistant" message containing the answer for the user.

## TOOLS (the only ones — do NOT invent others)

{tools_json}

Whitelist: {TOOL_WHITELIST}

## HARD RULES

1. **Language: English only.** All user queries and assistant content in English.
2. **source="lora" → final assistant.content ≤ 150 bytes UTF-8.** Count bytes, not chars.
3. **source="touchscreen" → final assistant.content ≤ 200 bytes UTF-8.**
4. **Hybrid response format**:
   - If the question has a single answer: 1 short declarative sentence. No list.
   - If the question is procedural / multi-step: 1 lead sentence + numbered list (1./2./3., max 5 items, each ≤ 1 line).
   - No preambles. Start with the answer.
5. **Use the right tool**: search_knowledge for facts; capture_image only for visible objects; analyze_signal for radio status; send_lora_message to transmit; listen_lora to wait; get_system_status for device telemetry; update_memory only for durable operational facts.
6. **Don't hallucinate**: if you need a fact, the assistant MUST call search_knowledge first, then synthesise from the tool result.
7. **Max 2 tool_calls per dialog.** Keep it tight.
8. **Tool arguments must match the JSON schema exactly.**
9. **Vary aggressively** across variants of the same seed: paraphrase the user query, change concrete details (locations, weights, durations, names of nodes/plants/wounds), vary tool result content, vary how the final answer is phrased.

## STYLE EXAMPLES (English, hybrid format)

Q (touchscreen): "How many watts solar panel for my RPi 5?"
A: "Min 15 W, recommended 20 W with an MPPT regulator for stability."

Q (lora): "stop bleeding arm?"
A (≤150 B):
"Press wound 5 min:
1. Direct pressure
2. Raise arm
3. Tourniquet last resort"

Q (touchscreen): "What's the safe distance from a thunderstorm if I see lightning?"
A: "Use the 30/30 rule. Take cover when thunder is ≤30 s after a flash; wait 30 min after the last thunder before resuming."

## RUNTIME SYSTEM PROMPT (what the student model will see at inference)

---
{SYSTEM_PROMPT_EN}
---

Generate realistic, training-grade trajectories the student must learn to reproduce.
"""


# ---------- SEED CATALOGUE (5 categories × 2 sources × 6 seeds = 60) ----------

@dataclass
class Seed:
    seed_id: str
    category: str
    source: str  # "touchscreen" or "lora"
    queries: list[str]
    expected_tool: str | None
    style_hint: str

    def to_dict(self) -> dict:
        return {
            "seed_id": self.seed_id,
            "category": self.category,
            "source": self.source,
            "user_query_templates": self.queries,
            "expected_tool": self.expected_tool,
            "expected_final_style": self.style_hint,
            "max_response_bytes": 150 if self.source == "lora" else 200,
        }


SEEDS: list[Seed] = []


def _add(seed_id, category, source, queries, expected_tool, style_hint):
    SEEDS.append(Seed(seed_id, category, source, queries, expected_tool, style_hint))


# --- Category 1: FIRST AID & MEDICAL ---
_add("fa_bleeding_touch", "first_aid", "touchscreen",
     ["How do I stop a heavy bleed on the forearm?",
      "Knife cut on my arm, bleeding badly — what now?",
      "How to control arterial bleeding before medics arrive?"],
     "search_knowledge", "lead sentence + 3-step numbered list, action verbs")
_add("fa_cpr_touch", "first_aid", "touchscreen",
     ["My friend collapsed and isn't breathing. Walk me through CPR.",
      "Adult CPR ratio and depth?",
      "How long do I keep doing CPR if I'm alone?"],
     "search_knowledge", "lead sentence + numbered list")
_add("fa_hypothermia_touch", "first_aid", "touchscreen",
     ["Someone is shivering, confused, slurred speech in the cold — what do I do?",
      "Hypothermia first response?",
      "Mild vs severe hypothermia — what's the difference and the action?"],
     "search_knowledge", "lead sentence + actions")
_add("fa_burn_lora", "first_aid", "lora",
     ["3rd deg burn forearm what now",
      "severe burn help",
      "boiling water spilled on leg first aid"],
     "search_knowledge", "ultra-short hybrid")
_add("fa_choking_lora", "first_aid", "lora",
     ["child choking what do",
      "adult choking steps",
      "Heimlich how"],
     "search_knowledge", "ultra-short steps")
_add("fa_shock_lora", "first_aid", "lora",
     ["pale cold sweating low pulse",
      "shock signs treatment",
      "casualty in shock what now"],
     "search_knowledge", "ultra-short")

# --- Category 2: BUSHCRAFT (water/food/fire/shelter) ---
_add("bc_water_touch", "bushcraft", "touchscreen",
     ["How do I purify river water without a filter?",
      "Only puddle water available — how to make it drinkable?",
      "Does boiling water kill viruses too?"],
     "search_knowledge", "lead + options list (boiling, chlorine, SODIS)")
_add("bc_fire_touch", "bushcraft", "touchscreen",
     ["How to start a fire without matches or lighter?",
      "Fire-starting methods with only a knife and dry wood?",
      "Can I start a fire with a 9V battery and steel wool?"],
     "search_knowledge", "lead + 3 methods")
_add("bc_shelter_touch", "bushcraft", "touchscreen",
     ["Quick shelter from a tarp and paracord, 2 trees available — design?",
      "What's the warmest shelter type I can build solo at night in a pine forest?",
      "Snow shelter — quinzee vs snow cave, which faster?"],
     "search_knowledge", "lead + steps")
_add("bc_edible_lora", "bushcraft", "lora",
     ["safe plants to eat birch forest",
      "edible berries red black",
      "is fern fiddlehead safe"],
     "capture_image", "ultra-short, IDs visible plant")
_add("bc_water_lora", "bushcraft", "lora",
     ["purify river water fast",
      "no filter clean water",
      "boil water how long altitude"],
     "search_knowledge", "ultra-short")
_add("bc_fire_lora", "bushcraft", "lora",
     ["wet wood fire how",
      "ferro rod no tinder",
      "battery fire spark"],
     "search_knowledge", "ultra-short steps")

# --- Category 3: NAVIGATION + SIGNALING + LORA COMMS ---
_add("nav_compass_touch", "navigation", "touchscreen",
     ["Lost in the woods, only a compass and watch — how to navigate?",
      "How to set a bearing and stay on it through dense forest?",
      "What does 'shoot a back-bearing' mean and when to use it?"],
     "search_knowledge", "lead + numbered procedure")
_add("nav_signal_touch", "navigation", "touchscreen",
     ["Best ways to signal for rescue from a clearing?",
      "I see a helicopter — what's the international ground-to-air SOS signal?",
      "Signal mirror technique to flash an aircraft?"],
     "search_knowledge", "lead + list")
_add("nav_lora_relay_touch", "navigation", "touchscreen",
     ["I want to tell Node B that I'm fine and going west — how do I send it?",
      "Send LoRa message to relay 'water source at GPS 49.27N 19.95E'",
      "How do I broadcast an SOS over LoRa with my position?"],
     "send_lora_message", "use tool to transmit")
_add("nav_signal_lora", "navigation", "lora",
     ["SOS sign ground",
      "signal helicopter mirror",
      "smoke signal materials"],
     "search_knowledge", "ultra-short")
_add("nav_lostlora", "navigation", "lora",
     ["lost no compass which way",
      "north without compass",
      "sun navigation noon"],
     "search_knowledge", "ultra-short")
_add("nav_send_lora", "navigation", "lora",
     ["tell node B im ok going west",
      "broadcast water at 49N 19E",
      "send SOS my position"],
     "send_lora_message", "model transmits via tool")

# --- Category 4: POWER + GEAR + HAZARDS ---
_add("pw_solar_touch", "power_gear", "touchscreen",
     ["How many watts solar panel to keep my Pi 5 LastBox running 24/7?",
      "Solar + battery sizing for a 5 W continuous load on cloudy days?",
      "Best MPPT controller for a 20 W panel charging 18650 cells?"],
     "search_knowledge", "lead + numbers, controllers")
_add("pw_storm_touch", "power_gear", "touchscreen",
     ["Thunderstorm approaching, I'm on a ridge — what to do?",
      "How far should I be from a tree in a storm?",
      "30/30 rule — explain."],
     "search_knowledge", "lead + actions")
_add("pw_radiation_touch", "power_gear", "touchscreen",
     ["Nearby reactor incident, no respirator — minimal shelter advice?",
      "What dose rate from a dirty bomb requires evacuation?",
      "Iodine tablets — when and how much for an adult?"],
     "search_knowledge", "lead + numbers, conservative")
_add("pw_battery_lora", "power_gear", "lora",
     ["18650 charge cycles left",
      "li-ion safe temp range",
      "battery bulging what do"],
     "search_knowledge", "ultra-short")
_add("pw_status_lora", "power_gear", "lora",
     ["cpu temp now",
      "battery percent",
      "rssi link quality"],
     "get_system_status", "tool call then 1-line answer")
_add("pw_lightning_lora", "power_gear", "lora",
     ["lightning ridge what do",
      "safe tree distance storm",
      "30/30 rule"],
     "search_knowledge", "ultra-short")

# --- Category 5: ELECTRONICS (RPi, sensors, fixes) ---
_add("el_pi_undervolt_touch", "electronics", "touchscreen",
     ["Pi 5 shows undervoltage lightning bolt under load — diagnosis?",
      "USB-C 5V/3A and still throttling — what to check?",
      "How to read dmesg for power-related faults on Pi 5?"],
     "search_knowledge", "lead + checks 1-3")
_add("el_lora_pair_touch", "electronics", "touchscreen",
     ["How to pair a new Meshtastic node to my mesh?",
      "Set up a SX1262 LoRa HAT on Pi 5 — first steps?",
      "Node B not appearing in node list — debug?"],
     "search_knowledge", "lead + steps")
_add("el_sensor_touch", "electronics", "touchscreen",
     ["BME280 temperature reads 10°C too high — likely cause?",
      "I2C device not detected on Pi 5 with i2cdetect — what to try?",
      "Add a DS18B20 to monitor enclosure temp — wiring?"],
     "search_knowledge", "lead + checklist")
_add("el_undervolt_lora", "electronics", "lora",
     ["pi 5 undervolt fix",
      "throttle even 5V3A",
      "power warning solid"],
     "search_knowledge", "ultra-short")
_add("el_lora_pair_lora", "electronics", "lora",
     ["add node mesh how",
      "node B not seen debug",
      "pair meshtastic fast"],
     "search_knowledge", "ultra-short")
_add("el_listen_lora", "electronics", "lora",
     ["any SOS air now",
      "scan channel 5 minutes SOS",
      "anything from node B last 60s"],
     "listen_lora", "tool listen then 1-line answer")


# ---------- OPENROUTER CLIENT (Kimi K2 Instruct) ----------

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


async def _generate_for_seed(
    session: aiohttp.ClientSession,
    api_key: str,
    teacher_system: str,
    seed: Seed,
    variants: int,
    model: str,
    semaphore: asyncio.Semaphore,
) -> list[dict]:
    user_prompt = (
        f"Generate {variants} different training dialogues for the seed below. "
        f"Each on its own line as a single JSON object (JSONL). "
        f"Vary the phrasing of the user query, vary concrete facts/numbers, vary how the answer is worded, "
        f"vary tool result text. Stay in English. Respect the source byte limit.\n\n"
        f"SEED:\n{json.dumps(seed.to_dict(), ensure_ascii=False, indent=2)}\n\n"
        f"Output {variants} lines of JSONL. No markdown, no commentary, no blank lines."
    )

    async with semaphore:
        for attempt in range(3):
            try:
                async with session.post(
                    OPENROUTER_URL,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://github.com/lastbox/gemma4-hackathon",
                        "X-Title": "LastBox Gemma 4 dataset v2",
                    },
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": teacher_system},
                            {"role": "user", "content": user_prompt},
                        ],
                        "temperature": 0.8,
                        "max_tokens": 16000,
                    },
                    timeout=aiohttp.ClientTimeout(total=300),
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                content = data["choices"][0]["message"].get("content")
                break
            except Exception as e:
                if attempt == 2:
                    print(f"[ERR] seed={seed.seed_id} attempt={attempt}: {e}", file=sys.stderr)
                    return []
                await asyncio.sleep(2 ** attempt)
        else:
            return []

    if not content:
        print(f"[WARN] seed={seed.seed_id}: empty content", file=sys.stderr)
        return []

    content = re.sub(r"^```(?:json|jsonl)?\s*", "", content.strip())
    content = re.sub(r"\s*```$", "", content.strip())

    def _enrich(obj: dict) -> dict | None:
        if not isinstance(obj, dict):
            return None
        if "messages" not in obj or not isinstance(obj["messages"], list):
            return None
        obj.setdefault("seed_id", seed.seed_id)
        obj.setdefault("category", seed.category)
        obj.setdefault("source", seed.source)
        return obj

    out: list[dict] = []
    stripped = content.strip()
    if stripped.startswith("["):
        try:
            arr = json.loads(stripped)
            for o in arr:
                e = _enrich(o)
                if e:
                    out.append(e)
            return out
        except json.JSONDecodeError:
            pass

    for line in content.splitlines():
        line = line.strip().rstrip(",")
        if not line or not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        e = _enrich(obj)
        if e:
            out.append(e)
    return out


async def main_async(args):
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("Missing OPENROUTER_API_KEY")

    teacher_system = build_teacher_meta()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    semaphore = asyncio.Semaphore(args.concurrency)
    t0 = time.time()
    total = 0
    seen_seeds = 0
    # Resume: skip seeds that already have any dialogs in out_path
    done_seeds: set[str] = set()
    if args.resume and out_path.exists():
        for line in open(out_path, "r", encoding="utf-8"):
            try:
                d = json.loads(line)
                if d.get("seed_id"):
                    done_seeds.add(d["seed_id"])
            except Exception:
                continue
    seeds_to_run = [s for s in SEEDS if s.seed_id not in done_seeds]
    print(f"[INFO] Seeds total={len(SEEDS)} done={len(done_seeds)} to_run={len(seeds_to_run)}")

    mode = "a" if args.resume else "w"
    fout = open(out_path, mode, encoding="utf-8")
    async with aiohttp.ClientSession() as session:
        tasks = [
            _generate_for_seed(
                session, api_key, teacher_system, s, args.variants, args.model, semaphore
            )
            for s in seeds_to_run
        ]
        for coro in asyncio.as_completed(tasks):
            try:
                dialogs = await coro
            except Exception as e:
                print(f"[ERR] task crashed: {e}", file=sys.stderr)
                dialogs = []
            seen_seeds += 1
            for d in dialogs:
                fout.write(json.dumps(d, ensure_ascii=False) + "\n")
                total += 1
            fout.flush()
            print(f"[{seen_seeds:2d}/{len(seeds_to_run)}] +{len(dialogs)} dialogs  total={total}  elapsed={time.time()-t0:.0f}s")
    fout.close()

    print(f"\n[DONE] {total} dialogs written to {out_path} in {time.time()-t0:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("gemma4/data/raw_v2.jsonl"))
    ap.add_argument("--variants", type=int, default=50,
                    help="Variants per seed (60 seeds × 50 = 3000)")
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--model", type=str, default="moonshotai/kimi-k2.5")
    ap.add_argument("--resume", action="store_true",
                    help="Skip seeds already in --out and append")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
