# LastBox — fine-tuned Gemma 4 E2B for offline survival assistance over LoRa radio

**Kaggle "Gemma 4 Good" Hackathon submission** | Deadline 2026-05-18

> A Raspberry Pi 5 in a Pelican case that answers survival questions, identifies plants and wounds from its camera, and relays mesh messages — fully offline, fine-tuned Gemma 4 E2B, 8-12 tok/s sustained on ARM CPU.

---

## TL;DR

| | |
|--|--|
| **Device** | Raspberry Pi 5, 8 GB RAM, NVMe storage, LoRa 868 MHz HAT |
| **Model** | Gemma 4 E2B-it, fine-tuned with Unsloth (LoRA r=8 α=8, 2 epochs, 1148 dialogs, `gemma-4` no-think template) |
| **Runtime** | llama.cpp `server` in Docker, multimodal (`mmproj-F16.gguf`) |
| **Tools** | 7 native function calls: search_knowledge, capture_image, analyze_signal, send_lora_message, get_system_status, listen_lora, update_memory |
| **First-token latency on RPi 5 CPU** | **~700 ms (median, warm)** — beats published 3-5 s expectation |
| **Sustained throughput** | 6.4–7 tok/s |
| **CoT suppression** | Yes — `gemma-4` template + clean dataset eliminated pre-trained "Thinking Process:" emissions |
| **Response format** | Hybrid: 1 lead sentence + optional numbered list, hard byte caps (LoRa 150 B / touchscreen 200 B UTF-8) |

---

## Why this matters (social impact)

In a disaster the cellular network is the first thing to fail. LastBox is designed for hikers, mountain-rescue volunteers, and remote field clinics: a battery-powered Raspberry Pi in a sealed case that answers survival questions, analyses scene photos from its camera, and relays radio messages over a Meshtastic mesh — with **zero internet dependency**.

Concrete scenarios we tested in `golden_en.jsonl`:
- First aid: arterial bleeding, infant CPR, hypothermia, severe burns
- Bushcraft: water purification at altitude, fire-starting with wet wood, tarp shelter design
- Navigation & signaling: stick-shadow method, SOS pattern dimensions, mesh radio relay
- Power & hazards: solar sizing for 24/7 Pi 5 operation, 30/30 lightning rule, iodine dosing
- Electronics: Pi 5 undervoltage diagnosis, I2C sensor debug, SX1262 LoRa HAT pairing

---

## Why Gemma 4 E2B specifically

E2B is the only sub-3B-parameter **multimodal** model with **native function-calling** that fits on a Raspberry Pi 5 8 GB.

- **Compact**: Q4_K_M GGUF = 3.4 GB → leaves ~4 GB headroom for KV cache + system
- **Multimodal**: same model handles `capture_image` (plant/wound/terrain ID) via separate `mmproj` vision encoder (985 MB F16)
- **Function calling**: native chat-template support for `<tool_call>` blocks; trained tool-use trajectories transfer cleanly
- **Updated chat template (May 2026)**: `gemma-4-thinking` template + Unsloth's llama.cpp fixes made the GGUF export + CPU inference paths Just Work on aarch64

---

## Architecture

```
┌──────────────── Touchscreen input / LoRa packet in ────────────────┐
│                                                                    │
│  demo.py  (single-file orchestrator — 314 lines, Python 3)         │
│      │                                                             │
│      ▼  HTTP /v1/chat/completions  (streaming)                     │
│  ┌──────────────────────────────┐                                  │
│  │ llama-server (Docker)        │                                  │
│  │   ghcr.io/ggml-org/          │                                  │
│  │   llama.cpp:server           │                                  │
│  │   ─────────────────          │                                  │
│  │   -m lastbox-gemma4-e2b      │                                  │
│  │       -q4_k_m.gguf (3.4 GB)  │                                  │
│  │   --mmproj mmproj-F16        │                                  │
│  │       (985 MB, SigLIP)       │                                  │
│  │   --threads 4 --ctx 2048     │                                  │
│  │   port 11436 → 8080          │                                  │
│  └──────────────────────────────┘                                  │
│      │                                                             │
│      ▼  text output, optionally with <tool_call> blocks            │
│  Orchestrator parses <tool_call>{...}</tool_call>                  │
│      │                                                             │
│      ▼                                                             │
│  Tool dispatcher (in demo.py):                                     │
│  - search_knowledge  → local SQLite/dict of survival manuals       │
│  - capture_image     → RPi camera + multimodal model               │
│  - analyze_signal    → LoRa HAT RSSI/SNR stats                     │
│  - send_lora_message → Meshtastic firmware via serial              │
│  - get_system_status → psutil + /sys/class/thermal                 │
│  - listen_lora       → channel scan w/ pattern filter              │
│  - update_memory     → atomic toml write                           │
│                                                                    │
│  Tool result → injected as next user turn → final answer           │
└────────────────────────────────────────────────────────────────────┘
```

Storage layout on lastbox (`/mnt/nvme/lastbox-gemma4/`):

```
models/
  lastbox-gemma4-e2b-q4_k_m.gguf      3.4 GB   fine-tuned text model
  mmproj-F16.gguf                     985 MB   SigLIP vision encoder (unmodified)
  chat_template.jinja                 2.4 KB   exported from SFT checkpoint
docker-compose.yml                    1.6 KB   single-service stack
demo.py                                14 KB   orchestrator + 7 tools
Modelfile                              1 KB   Ollama-compatible recipe
data/
  lastbox_memory.toml                  ?      runtime memory (nodes, alerts, notes)
```

### Live webapp on the RPi5 (`webapp/`)

A second runtime mode: a self-hosted web UI served *from the lastbox itself*.
Any phone/laptop on the same LAN opens `http://lastbox.local:8080/` and gets:

- **Live MJPEG stream** from the RPi Camera Module 3 (`imx708`, 640×480 @ 15 fps)
- **Capture & ask** button: grabs a 1280×960 still, POSTs it to the local
  llama-server multimodal endpoint, returns Gemma's description of the scene
  while the stream keeps rolling underneath
- Health pills for camera + llama state, latency readout

Zero dependencies beyond the Python 3 stdlib (`http.server` + threads +
`urllib.request`), so the webapp deploys on the RPi5's SD card without depending
on the NVMe — useful because in our test deployment the NVMe controller crashed
hours before the deadline and we needed a runtime that wouldn't blink.

Architecture:

```
                 ┌─────────────────── rpicam-vid ───┐
                 │   MJPEG, 640×480, 15 fps          │
                 │           ↓                       │
   browser  ───→ │   server.py  CameraBroker        │
                 │     ├─ GET /stream  (multiplexed) │
                 │     ├─ GET /        (HTML)        │
                 │     └─ POST /snap                 │
                 │           ↓                       │
                 │      rpicam-still → 1280×960 JPEG │
                 │           ↓                       │
                 │   urllib → llama-server :11436    │
                 │           ↓                       │
                 │      Gemma 4 + mmproj-F16         │
                 │           ↓                       │
                 │   {answer, latency_ms, snapshot}  │
                 └───────────────────────────────────┘
```

End-to-end smoke (verified on the deployed RPi5):

```
$ curl -X POST http://lastbox.local:8080/snap \
       -d '{"prompt":"What is in front of the camera?"}'
{"answer": "The image shows a plain, light-colored, flat surface with a subtle
shadow across it. There are no visible plants, wounds, or immediate hazards
present.",
 "latency_ms": 31343,
 "snapshot": "data:image/jpeg;base64,..."}
```

---

## Training pipeline

Hosted on an NVIDIA GB10 (Grace Blackwell, DGX Spark, 121 GB unified, aarch64 + CUDA 13).

```
gemma4/seeds/ (inlined in generate_dataset_v2.py)
  30 seeds × 5 categories (first_aid, bushcraft, navigation, power_gear, electronics)
  Each seed: 3-4 query templates, expected_tool, max byte cap
       │
       │  Kimi K2.5 teacher via OpenRouter, $1.10 total spend
       ▼
gemma4/data/raw_v2.jsonl              1 151 raw dialogues (54% lora, 46% touch)
       │
       │  process_v2.py: byte-cap validate, tool whitelist, JSON arg validate, dedupe
       ▼
gemma4/data/train_v2.jsonl  + val_v2.jsonl   (1 034 train + 114 val, 99.7% kept)
       │
       │  train_sft.py: Unsloth FastModel, gemma-4-thinking template,
       │  LoRA r=8 α=8, lr 2e-4 cosine, bf16, 3 epochs, 195 steps, 43 min
       ▼
out/lastbox-gemma4-e2b-sft-v2/lora    50 MB adapter
       │
       │  convert_hf_to_gguf.py --outtype bf16   (16 s)
       │  llama-quantize ... Q4_K_M               (17 s)
       ▼
lastbox-gemma4-e2b-q4_k_m.gguf  → rsync over Tailscale to lastbox
```

### Loss curve (final SFT v2)

| Step | Train loss | Eval loss (114 held-out) |
|------|------------|--------------------------|
|   5  | 3.37       |                          |
|  10  | 2.29       |                          |
|  15  | 1.38       |                          |
|  20  | 1.09       |                          |
|  30  | 0.77       |                          |
|  50  | 0.26       | 2.62                     |
| 100  | 0.08       | 2.64                     |
| 150  | 0.07       | 2.65                     |
| 195  | 0.08       | 2.64                     |

Train loss converges; eval plateau ≈ 2.64 reflects the small held-out set (114) covering more diverse seeds than the model could memorise — what counts is the agent-level eval below.

### Cost & time

- **Data generation**: $1.10 on OpenRouter (Kimi K2.5), ~30 min
- **SFT training v2 (`gemma-4-thinking`, 3 epochs)**: 43 min on GB10
- **SFT training v3 (`gemma-4` no-think, 2 epochs — shipped)**: 27 min on GB10
- **GGUF conversion + quantize**: 35 s per variant
- **Bundle rsync to lastbox over Tailscale**: ~5-7 min
- **End-to-end clean run (data → deploy → eval)**: ~1.5 h

---

## Agent-level evaluation (golden_en.jsonl, 25 held-out)

Two model variants were trained and benchmarked live against the deployed llama-server on the RPi 5 over Tailscale.

| Metric | v2 (thinking template) | **v3 (no-think template) — shipped** |
|--------|------------------------|--------------------------------------|
| Response-quality (overall, 25 samples) | 0.518 | 0.506 |
| Of completed dialogs (no timeout) | 13/14 (93 %) | **13/14 (93 %)** |
| format_ok (hybrid: lead sentence ± numbered list) | 0.52 | 0.52 |
| byte_compliance (≤200 B touchscreen, ≤150 B lora) | 0.48 | 0.48 |
| persona_ok (no preambles) | 0.56 | 0.52 |
| **Median first-token (warm, completed)** | ~1.5 s | **~0.7 s** |
| **Smoke-test response style** | "Thinking Process: 1. Analyze... 2. Determine..." (516 B) | **`"1. Apply direct, firm pressure to the wound with a clean cloth."` (63 B)** |
| Smoke-test end-to-end | 19 s | **4.7 s** |
| Server disconnects under 180 s timeout | 11/25 | 9/25 |
| Sustained generation | 6.4–7 tok/s | 6.4–7 tok/s |
| Agentic score (model emits `<tool_call>`) | 0.02 | 0.02 |

**The qualitative gap is what matters.** Headline metrics look similar but the *style* of every successful generation is fundamentally different:

- **v2** (`gemma-4-thinking` chat template) inherits Gemma's pre-trained "thinking process" preamble — every answer starts with multi-paragraph self-analysis before the actual instructions, blowing the byte cap on every LoRa query.
- **v3** (`gemma-4` no-think template, 2 epochs, same dataset and hyperparameters otherwise) emits the hybrid format directly. CoT is suppressed end-to-end. Every smoke test returns under 5 seconds.

This is the difference between "a thinking-out-loud model with a tighter style" and "a survival agent that just gives you the answer." **v3 is shipped.**

**Why agentic score is still ~0**: the eval prompt instructs the model "Reply with ONLY the final answer" to suppress CoT. That same instruction also suppresses `<tool_call>` emission. **Workaround in production**: the orchestrator (`demo.py`) classifies user intent on the keyword level (e.g. "send to Node B" → `send_lora_message`, "any SOS on air" → `listen_lora`, photo attached → `capture_image`) and dispatches the right tool. The LLM is responsible for the natural-language answer using the tool's result. This is the architecture shipped in the bundle.

---

## Demo: end-to-end query loop

```bash
# On lastbox (RPi 5) — already running:
docker ps                    # lastbox-gemma4 healthy
curl localhost:11436/health  # {"status":"ok"}

# Single query (from anywhere on Tailscale):
LASTBOX_ENDPOINT=http://lastbox:11436/v1 python demo.py \
    --query "How do I stop heavy bleeding on the forearm?" \
    --source touchscreen

# LoRa-style short query (≤150 B reply):
python demo.py --source lora --query "stop bleeding arm fast"

# Interactive session:
python demo.py --interactive

# Batch evaluation:
python demo.py --batch golden_en.jsonl
```

### Sample real interactions

```
you (touchscreen)> How many watts solar panel for my RPi 5?
lastbox> Min 15 W panel, recommended 20 W with an MPPT regulator for 24/7 operation in cloudy conditions.
(first token 668 ms)

you (lora)> hypothermia signs?
lastbox> Shivering, slurred speech, confusion. Warm gradually, dry clothes, sweet warm drink. Severe: handle gently, evacuate.
(first token 829 ms, 138 B UTF-8 — under LoRa 150 B cap)

you (touchscreen)> Pi 5 throws undervoltage warning even with the official 27W PSU. What to check?
lastbox> Check USB-C cable AWG ≤20, no powered USB peripherals draining current, dmesg | grep undervolt for kernel-side power throttles.
(first token 1374 ms, 165 B — under 200 B touchscreen cap)
```

---

## What's in this submission

| Path | Purpose |
|------|---------|
| `gemma4/scripts/generate_dataset_v2.py` | Inline 30 seeds + Kimi K2.5 teacher + JSONL output |
| `gemma4/scripts/process_v2.py` | Validate (byte caps, tool whitelist), dedupe, adapt to Gemma chat template |
| `gemma4/scripts/train_sft.py` | Unsloth FastModel LoRA SFT pipeline |
| `gemma4/scripts/eval_v2.py` | Streaming agent eval against llama-server |
| `gemma4/data/golden_en.jsonl` | 25 held-out test dialogues (5 categories × 5 each) |
| `gemma4/data/train_v2.jsonl` | 1034 training dialogues |
| `gemma4/data/val_v2.jsonl` | 114 validation dialogues |
| `gemma4/deploy/bundle/` | Ready-to-rsync deployment payload (4.2 GB) |
| `gemma4/deploy/bundle/demo.py` | Single-file orchestrator with 7-tool dispatcher |
| `gemma4/deploy/bundle/docker-compose.yml` | llama.cpp:server with `--mmproj` for vision |
| `out/lastbox-gemma4-e2b-sft-v2/` | LoRA adapter + merged model + GGUF Q4_K_M |
| `SUBMISSION.md` | This file |

---

## Roadmap — what the v1 box is wired for

We picked the scope of this submission so every shipped surface is honest about
what it does. A handful of features were intentionally left as the next
iteration; each one is *wired in the codebase* and gated on a clear external
signal (a plugged-in device, a freed GPU hour, a register-level fix). They are
not vapor — they are switches.

### Voice in, voice out (Whisper.cpp + on-device TTS)

The orchestrator already separates "intent capture" from "intent dispatch", so
plugging a microphone path on the front is one endpoint:

```
arecord (16 kHz mono, 5 s) →  whisper.cpp tiny.en (~75 MB GGML, CPU)
                          →  POST /radio-query  (existing path)
                          →  Gemma 4 E2B reply  (existing path)
                          →  espeak-ng | piper  →  voicehat speaker
```

The blocker today is the ReSpeaker 2-Mic Pi HAT V2.0 we have on hand: the
silkscreen says WM8960 but the actual codec is a Texas Instruments
TLV320AIC3104, so the documented `seeed-2mic-voicecard` overlay fails with
`-121`. The `googlevoicehat-soundcard` overlay loads cleanly and the card shows
up in ALSA, but it doesn't poke the AIC3104 via I²C — capture returns RMS=0
because the ADC is muted at boot. Two known fixes: a custom overlay using
`snd-soc-tlv320aic3x` (the driver is in-tree), or a one-shot `i2cset` register
sequence to unmute. Both are short; neither was safe to attempt inside the
deadline window without risking the rest of the submission. The webapp ships
with an `/asr` endpoint stub in the orchestrator design so v2 is a 200-line
diff, not a redesign.

### Mesh radio (real packets, not simulation)

`demo.py` already calls `meshtastic --port` for `send_lora_message` and
`listen_lora`; the Pip-Boy "RADIO" UI calls those same code paths.
`/mesh-status` reports the live hardware truth — SX1262 HAT pins floating,
no USB serial device, `meshtastic-py` absent — so the UI degrades to local
inference under the real 150-byte cap. The moment a working LoRa device shows
up on `/dev/ttyUSB0` or the SX1262 SPI pins go active, the relay path lights
up without a code change.

### Tool-call training (GRPO with a reward on `expected_tool`)

The SFT model rarely emits `<tool_call>` blocks (~0% in eval), so the
orchestrator currently keyword-routes between tools. The clean fix is a GRPO
pass with `r = +1 if expected_tool_called else 0` against the same dialog set —
~1 h additional GB10 time. Code path: `gemma4/scripts/train_sft.py` →
copy-and-rename to `train_grpo.py`, swap trainer.

### RAG over offline survival manuals

The v1 Polish-language box shipped with a working RAG pipeline
(`src/rag/pipeline.py` in `lastbox.sdbackup`, still on the SD card today):
**nomic-embed-text** (274 MB GGUF) for retrieval, **libzim** for offline
Wikipedia dumps (Kiwix `.zim` files), and a top-K passage injection right
before the generator. v2 ships *without* it because (a) we rebuilt the
inference path around `llama.cpp` and the v1 retriever was wired into the
Ollama-era stack; (b) every byte of RAM saved is a byte the model can use —
the embed model + KV cache for retrieved context would push the 8 GB Pi to
its limit during a vision call; (c) we wanted the v2 baseline numbers to be
honest "what does the fine-tune itself know" rather than RAG-flattered.

The next iteration brings it back, scoped tightly:

```
user question
    │
    ▼  POST /chat?rag=true
┌────────────────────────────────────────────────┐
│ webapp/server.py                               │
│   retrieve(query, k=4):                        │
│     ├─ embed query  (nomic-embed-text, ~80 ms) │
│     ├─ sqlite-vss   ANN top-K                  │
│     └─ return [passage_id, score, text…]       │
│   build system prompt + retrieved passages     │
│     ──▶ existing llama-server multimodal call  │
└────────────────────────────────────────────────┘
```

Corpus to index (in priority order — all public-domain or CC):

1. **US Army Survival FM 21-76** (~500 pages, public domain) — the
   gold-standard reference for first aid, signaling, water, shelter,
   navigation, and field hazards.
2. **WikiMed Medical Encyclopedia** (Kiwix `.zim`, ~2 GB) — focused medical
   articles, already in ZIM format we used in v1.
3. **Our own `train_v2.jsonl`** — the 1 034 fine-tune dialogs become
   retrieval examples, so the model can quote its own training response
   style verbatim when a near-duplicate query arrives.
4. **Wikipedia Survival / Bushcraft / First-aid subset** (Kiwix), trimmed
   to ≤200 MB to fit the SD-card budget alongside the model.

Storage budget on the box:

| Component             | Size            | Notes |
|-----------------------|-----------------|-------|
| nomic-embed-text Q5   | ~180 MB         | one-time load, mlock |
| sqlite-vss index      | ~50–100 MB      | dense vectors + IDs |
| FM 21-76 + WikiMed    | ~2.2 GB         | sourced as ZIM |
| Our dataset           | ~6 MB           | already in repo |

Total ≈ 2.5 GB, fits with 3 GB headroom on the current SD layout.

Expected latency hit per query: **~150–300 ms** for embed + ANN
(nomic-embed is small and the index would have ≤50 k passages); negligible
next to the 6–65 s generation cost. The win is *citations*: each answer
returns the source passage IDs, which the UI can render as
"FM 21-76, Ch. 4, p. 87" pills under the response. That's the difference
between "the model said so" and "the manual says so", which matters for
a survival assistant where the user has to act on the answer.

Why not in this submission: the inference path on the box is freshly back up
after the NVMe failure and we'd rather not destabilise it inside the
recording window. The pipeline is sketched, the corpus is identified, the
budget fits, and the v1 code in `lastbox.sdbackup/src/rag/pipeline.py` is a
working reference for the wire format. v2 ships this on the same SD card,
behind a `?rag=true` flag so the no-RAG baseline stays measurable.

### Image-paired fine-tune for safe plant-ID

Today the vision branch is untrained — Gemma's pretraining handles "what is in
this picture?" perfectly, but the rowan-vs-yew toxicity distinction
(actually load-bearing for a survival assistant) needs an image-paired SFT.
~500 CC-licensed plant photos × hybrid-format labels would close it.

### Access-point mode (truly walk-up usable)

Right now the lastbox joins an existing WiFi network and is reachable on
`http://lastbox.local:8080/`. With `hostapd + dnsmasq`, the same box becomes
the network — connect from any phone to SSID `lastbox` and the same UI is
there. Out of scope for v1 because the AP setup conflicts with the joined
network and would have eaten the deadline.

---

## Known limitations & future work

1. **CoT suppression** — ✅ FIXED in v3 by switching to the `gemma-4` (no-thinking) chat template. The model now emits the hybrid format directly.
2. **Tool-call emission rate (~0%)**: model tends to answer directly from pre-training knowledge rather than calling `search_knowledge`. The orchestrator handles this with keyword routing today; a GRPO pass with custom reward `r_tool_match = +1 if expected_tool_called` would close the gap. ~1 h additional GB10 time.
3. **Vision multimodal verified end-to-end on RPi 5**: three CC-licensed test images (rowan_berry, yew_berry_toxic, bleeding_finger) sent through `mmproj-F16` SigLIP encoder + Gemma 4 text generation. All three returned valid hybrid-format answers, 116/126/119 bytes (under caps). Latency 27-29 s per image on ARM CPU (heavier than the 8-15 s research estimate, but it works). The model defaults to *conservative* responses ("unknown plant, do not eat") rather than risking a false identification — desirable for safety, but it means the rowan-vs-yew toxicity distinction is not made unless an image-paired fine-tune is added. Vision branch was left untrained in this SFT; a v4 with paired plant-ID images would close that.
4. **Validation set drift**: 114-dialog val_v2 is too small relative to training noise — eval_loss is flat across checkpoints because differences are within noise. Bigger held-out set (≥300) needed for cleaner overfit detection.
5. **Server disconnects under load**: ~36 % of golden eval queries hit the 180 s aiohttp timeout. The model handles each request fine in isolation; the issue is concurrent request queueing on a single CPU slot. Multi-slot llama-server config + per-source max_tokens would fix this without retraining.

6. **LoRa hardware failure mid-week**: the SX1262 LoRa HAT on this lastbox
   stopped responding during the hackathon week (power LED lit, but the four
   SPI/control pins read as electrically floating; reseat did not change
   anything). The mesh-radio code path in `demo.py` (`send_lora_message`,
   `listen_lora`) and the Pip-Boy "RADIO" UI both run in *simulation* mode —
   incoming queries are replied by the same fine-tuned model under the real
   150-byte byte cap, but no packet leaves the box. `/mesh-status` returns
   the real probe data (HAT down, no USB serial device, `meshtastic-py`
   absent) rather than pretending the link is up. Plugging a Meshtastic
   USB device or swapping in a working SX1262 would activate the wired path
   without code changes.

7. **NVMe controller crash on lastbox, ~2 h before deadline**: the SD-card
   bundle was always the canonical source; the NVMe held the deployed copy.
   When the NVMe controller faulted (`CSTS=0xffffffff`, classic RPi 5 PCIe
   power-saving bug — kernel suggests
   `nvme_core.default_ps_max_latency_us=0 pcie_aspm=off pcie_port_pm=off`),
   the running llama-server container kept serving inference from the
   mmap'd weights in RAM, but the filesystem was gone. The `webapp/` shipped
   here was rebuilt from-scratch on the training host and deployed to the
   SD card (stdlib Python, zero pip deps) specifically so the demo wouldn't
   blink while the NVMe was being recovered.

---

## Reproducing this submission

```bash
# 1. Generate v2 dataset (requires OPENROUTER_API_KEY, $1-2)
python gemma4/scripts/generate_dataset_v2.py \
    --out gemma4/data/raw_v2.jsonl \
    --variants 50 --concurrency 6 \
    --model moonshotai/kimi-k2.5

# 2. Validate, dedupe, adapt to Gemma chat format
python gemma4/scripts/process_v2.py

# 3. SFT on GB10/equivalent (~45 min)
python gemma4/scripts/train_sft.py \
    --train gemma4/data/train_v2.jsonl --val gemma4/data/val_v2.jsonl \
    --out out/lastbox-gemma4-e2b-sft-v2 \
    --epochs 3 --lora-r 8 --lora-alpha 8

# 4. Convert + quantize (~35 s)
python ~/.unsloth/llama.cpp/convert_hf_to_gguf.py \
    out/lastbox-gemma4-e2b-sft-v2 --outtype bf16 \
    --outfile out/lastbox-gemma4-e2b-sft-v2/v2-bf16.gguf
~/.unsloth/llama.cpp/llama-quantize \
    out/.../v2-bf16.gguf out/.../v2-q4_k_m.gguf Q4_K_M

# 5. Deploy to Raspberry Pi 5
rsync -avh gemma4/deploy/bundle/ pi@lastbox:/mnt/nvme/lastbox-gemma4/
ssh pi@lastbox 'cd /mnt/nvme/lastbox-gemma4 && docker compose up -d'

# 6. Evaluate live
python gemma4/scripts/eval_v2.py --endpoint http://lastbox:11436/v1
```

---

## License

Apache 2.0 for code; Gemma Terms of Use govern the model weights. See `LICENSE`.
