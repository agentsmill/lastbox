<div align="center">

# 📦 LASTBOX

### Offline survival assistant in a Pelican case.

**Raspberry Pi 5 · LoRa 868 MHz · fine-tuned [Gemma 4 E2B](https://huggingface.co/unsloth/gemma-4-E2B-it) · llama.cpp on ARM CPU**

[![Hugging Face — model](https://img.shields.io/badge/🤗_model-v6_toolprior-yellow?style=for-the-badge)](https://huggingface.co/norecyc/lastbox-gemma4-e2b-v6-toolprior)
[![Hugging Face — dataset](https://img.shields.io/badge/🤗_dataset-survival_dialogues-blue?style=for-the-badge)](https://huggingface.co/datasets/norecyc/lastbox-survival-dialogues)
[![Live site](https://img.shields.io/badge/website-agentsmill.github.io/lastbox-green?style=for-the-badge)](https://agentsmill.github.io/lastbox/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Gemma_4_Good-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/gemma-4-good-hackathon)
[![License](https://img.shields.io/badge/code-Apache_2.0-lightgrey?style=for-the-badge)](LICENSE)
[![License](https://img.shields.io/badge/weights-Gemma_Terms-orange?style=for-the-badge)](https://ai.google.dev/gemma/terms)

</div>

---

> **When the cell tower is the first thing to fail, you still have the box.**
> A battery-powered Pi 5 in a sealed case that answers survival questions,
> identifies plants and wounds from its lens, and relays terse messages over
> a Meshtastic mesh — fully offline. **~700 ms first token, 6–7 tok/s sustained on ARM CPU.**

<table>
<tr>
<td width="33%" align="center">

### 🩹 Survival Q&A
First aid · bushcraft · navigation · power · hazards. One short sentence, optional numbered list. Hard byte cap per channel.

</td>
<td width="33%" align="center">

### 📷 Optical triage
RPi camera → SigLIP → Gemma 4. Plant ID, wound assessment, terrain read. Defaults conservative — *"unknown plant, do not eat"* beats a wrong call.

</td>
<td width="33%" align="center">

### 📻 Mesh radio relay
Every reply destined for LoRa is hard-capped at **150 B UTF-8** so it fits one packet at the legal duty cycle. The box becomes a thinking router.

</td>
</tr>
</table>

---

## ⚡ The 38× result

After two GRPO iterations plateaued, **one 12-minute targeted SFT pass on
tool-only pairs** lifted agentic_score by **38×** and tool emission from
**0% → 72%**. Full retrospective in [`gemma4/SUBMISSION.md`](gemma4/SUBMISSION.md).

| Metric              | v3 (submission) | **v6 (post-deadline)** | Δ |
|---------------------|-----------------|------------------------|------|
| tool_emission_rate  | ~0 %            | **72 %**               | +72 pp |
| tool_accuracy       | 0 %             | **64 %**               | +64 pp |
| **agentic_score**   | 0.016           | **0.608**              | **38×** |
| byte_compliance     | 0.48            | **1.000**              | +0.52 |
| format_ok           | 0.52            | **1.000**              | +0.48 |
| response_quality    | 0.506           | **1.000**              | +0.494 |
| completed / 25      | 14              | **25**                 | perfect |

> 💡 The 0.52 floor on byte/format/persona was never a quality issue — every
> completed dialog already passed. It was the streaming-SSE eval losing 12/25
> dialogs to client-side disconnects. Non-streaming POST + retry → 25/25.

---

## 🚀 Try it in 30 seconds (anywhere with Docker)

```bash
docker run --rm -p 11436:8080 \
  --pull always ghcr.io/ggml-org/llama.cpp:server \
  -hf norecyc/lastbox-gemma4-e2b-v6-toolprior:Q4_K_M \
  --host 0.0.0.0 --port 8080 --jinja --ctx-size 4096

# In another terminal:
curl -sN http://localhost:11436/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"How do I stop heavy bleeding on the forearm?"}],"max_tokens":80}' \
  | jq -r '.choices[0].message.content'
```

Run on the real box: `xdg-open http://lastbox.local:8080/` (live cam stream + Pip-Boy radio + free-form chat — see [website](https://agentsmill.github.io/lastbox/)).

---

## 🏗️ Architecture

```
   📱 phone / 🖥️ laptop / 📻 LoRa packet
                    │
                    ▼   HTTP  (lastbox.local:8080)
   ┌──────────────────────────────────────────────────────┐
   │  webapp/server.py   stdlib Python, zero pip deps     │
   │     ├─ /          live MJPEG + radio chat + RAG chat │
   │     ├─ /snap      rpicam-still → Gemma vision        │
   │     ├─ /radio-query   ≤150 B reply (LoRa cap)        │
   │     └─ /chat?rag=true   embed → top-k → Gemma        │
   └──────────────────────────────────────────────────────┘
                    │
                    ▼   localhost:11436   (Docker, single GPU slot)
   ┌──────────────────────────────────────────────────────┐
   │  llama.cpp server  + Gemma 4 E2B Q4_K_M (3.4 GB)     │
   │                    + mmproj-F16 SigLIP (940 MB)      │
   └──────────────────────────────────────────────────────┘
                    │
                    ▼   localhost:11437   (separate container)
   ┌──────────────────────────────────────────────────────┐
   │  llama.cpp embed-server  +  bge-small-en-v1.5 (36 MB)│
   │                                                       │
   │  4 074-passage RAG index (train_v2 + Wikipedia +     │
   │  survival manuals), numpy cosine top-k, cites IDs    │
   └──────────────────────────────────────────────────────┘
```

---

## 📚 Open-source release

| Asset                          | Where |
|--------------------------------|-------|
| **v3 SFT** (submission state)  | [🤗 norecyc/lastbox-gemma4-e2b-sft-v3](https://huggingface.co/norecyc/lastbox-gemma4-e2b-sft-v3) — 21 GB, Q4_K_M + bf16 GGUF + safetensors + LoRA |
| **v6 SFT-warmup** (best)       | [🤗 norecyc/lastbox-gemma4-e2b-v6-toolprior](https://huggingface.co/norecyc/lastbox-gemma4-e2b-v6-toolprior) — 21 GB, same formats, 72 % tool emission |
| **Dataset**                    | [🤗 norecyc/lastbox-survival-dialogues](https://huggingface.co/datasets/norecyc/lastbox-survival-dialogues) — 1 034 train + 114 val + 25 golden + 4 074 RAG passages |
| **Submission snapshot**        | [`v1-submission` tag](https://github.com/agentsmill/lastbox/tree/v1-submission) — repo state at 2026-05-18 23:59 UTC |
| **Live UI**                    | [agentsmill.github.io/lastbox](https://agentsmill.github.io/lastbox/) — Pip-Boy CRT landing page |

---

## 📂 Repo map

```
gemma4/                The hackathon submission.
├── SUBMISSION.md      Architecture · training · eval · retrospective.
├── scripts/
│   ├── generate_dataset_v2.py   Kimi K2.5 teacher → 1 151 dialogs ($1.10)
│   ├── process_v2.py            byte-cap + tool whitelist + dedupe
│   ├── train_sft.py             Unsloth FastModel LoRA SFT
│   ├── train_grpo.py            TRL GRPOTrainer + reward func
│   ├── eval_v2_tool.py          Agent-level eval (tool-allowed, non-stream)
│   └── build_toolonly_dataset.py  Filter for v6 SFT-warmup
├── data/              train_v2 · val_v2 · golden_en · raw_v2 · toolonly · eval_*
├── grpo/reward.py     Composite reward (v1 + v2 designs)
├── rag/               Embed model · build_corpus · build_index · retrieve
└── deploy/bundle/     Pi 5 rsync target (docker-compose · demo.py · models/)

webapp/                Live UI served from the lastbox itself.
├── server.py          /, /stream, /snap, /radio-query, /chat (stdlib only)
├── static/
│   ├── index.html     Unified Fallout-CRT layout (cam + radio + chat)
│   ├── mesh.html      Pip-Boy radio standalone view
│   └── demo.html      Demo recording layout
└── rag/               retrieve.py + index/ (deployed alongside)

out/lastbox-gemma4-e2b-*/lora/   49 MB LoRA adapters (full weights on 🤗).

docs/index.html        GitHub Pages landing (the same Pip-Boy aesthetic).
```

---

## 🛠️ Reproduce from scratch (~1.5 h on a single CUDA box)

```bash
# 1. Generate teacher dialogs (needs OPENROUTER_API_KEY; ~$1–2)
python gemma4/scripts/generate_dataset_v2.py \
    --out gemma4/data/raw_v2.jsonl \
    --variants 50 --concurrency 6 --model moonshotai/kimi-k2.5

# 2. Validate / dedupe / adapt to Gemma chat format
python gemma4/scripts/process_v2.py

# 3. SFT on any modern CUDA box (~30 min)
python gemma4/scripts/train_sft.py \
    --train gemma4/data/train_v2.jsonl --val gemma4/data/val_v2.jsonl \
    --out out/lastbox-gemma4-e2b-sft-v3 \
    --epochs 2 --chat-template gemma-4

# 4. Convert + quantize (~35 s)
python ~/.unsloth/llama.cpp/convert_hf_to_gguf.py \
    out/lastbox-gemma4-e2b-sft-v3 --outtype bf16 \
    --outfile out/lastbox-gemma4-e2b-sft-v3/v3-bf16.gguf
~/.unsloth/llama.cpp/llama-quantize \
    out/.../v3-bf16.gguf out/.../v3-q4_k_m.gguf Q4_K_M

# 5. For the v6 tool-emission lift: filter train data + 12-min SFT-warmup
python gemma4/scripts/build_toolonly_dataset.py
python gemma4/scripts/train_sft.py \
    --base out/lastbox-gemma4-e2b-sft-v3 \
    --train gemma4/data/train_v2_toolonly.jsonl \
    --out out/lastbox-gemma4-e2b-v6 --epochs 1

# 6. Deploy to the Pi
rsync -avh gemma4/deploy/bundle/ pi@lastbox:/home/pi/lastbox-gemma4/
ssh pi@lastbox 'cd ~/lastbox-gemma4 && docker compose up -d'
```

Full hyperparameters + ablations in [`gemma4/SUBMISSION.md`](gemma4/SUBMISSION.md).

---

## 🔬 What we learned

<details>
<summary><strong>Why pure-GRPO couldn't fix tool emission (and what did)</strong></summary>

Two GRPO iterations (200 steps each, different reward designs) both
plateaued at **0 % tool emission**. The base SFT had `p(tool_call) ≈ 0` at
inference, and GRPO with KL β=0.04 cannot move that to ≈1 inside a small
number of optimization steps — the KL term penalises the required large
policy shift faster than the reward signal grows.

A **12-minute targeted SFT pass** on 1 034 `(prompt, tool_call_only)` pairs
set the prior on first-turn tool emission. Then *no* RL was needed: the
agentic_score jumped 0.016 → 0.608 immediately.

**The pipeline that works:**
1. SFT for new behaviours (set the prior).
2. GRPO for *shaping* an existing behaviour (sharpen what's already there).
3. **Never** try to RL a behaviour from `p ≈ 0` — the KL barrier wins.

</details>

<details>
<summary><strong>Two eval-methodology bugs that hid the real numbers</strong></summary>

The original eval reported `byte_compliance / format_ok / persona_ok` all at
**0.52** — exactly 13/25. We assumed it was a quality plateau. It was not.

1. **Streaming SSE disconnect** — the eval used `stream: true` and `aiohttp`
   was losing 12/25 dialogs to client-side mid-stream disconnects. Server
   logs were clean. Switching to non-streaming POST + 2-retry on
   `ServerDisconnectedError` → 25/25 completed → flags jumped to **1.000**.
2. **Missing tool definitions in system prompt** — training prompts had a
   4 233-char system block with the 7-tool whitelist JSON embedded. The
   eval was passing only the 733-char `SYSTEM_PROMPT_EN` — no tool defs, no
   tool emission. Using `process_v2._build_system_prompt_with_tools()` for
   eval unblocked tool emission completely.

**Lesson kept:** before believing a "quality" plateau, check the dataset
size of the floor. 0.52 = 13/25 was the completion ratio masquerading as
quality.

</details>

<details>
<summary><strong>Hardware reality on the Pi 5</strong></summary>

- **NVMe controller died ~2 h before deadline.** Classic RPi 5 PCIe power-
  saving fault (`CSTS=0xffffffff`). The running llama-server kept serving
  inference from mmap'd weights in RAM, but every new request that touched
  `/mnt/nvme` returned `Input/output error`. We rebuilt the webapp as
  stdlib-only Python + redeployed everything to the SD card mid-rush.
- **ReSpeaker 2-Mic HAT V2.0** had a TLV320AIC3104 codec instead of the
  silkscreened WM8960. The standard `seeed-2mic-voicecard` overlay failed
  with `-121`. Voice in/out shipped as "future work" with a working
  diagnostic path in `webapp/server.py /mesh-status`.
- **SX1262 LoRa HAT** stopped responding mid-week — power LED lit but the
  four SPI/control pins read electrically floating. Mesh radio runs in
  honest simulation mode (Pip-Boy "RADIO" UI calls the same `demo.py` code
  paths; `/mesh-status` reports HW state truthfully).

</details>

---

## 🗺️ Roadmap (wired, not vapor)

Each switch is designed into the codebase, gated on a clear external signal:

- 🎙️ **Voice in/out** — `arecord → whisper.cpp tiny.en → /radio-query → piper/espeak`. Blocked on the ReSpeaker codec mismatch.
- 📡 **Real mesh-radio packets** — `meshtastic --port` wiring in `demo.py`. Activates the moment a working SX1262 / Meshtastic USB device shows up.
- 🌱 **Image-paired SFT for safe plant-ID** — ~500 CC-licensed plant photos × hybrid-format labels to close the rowan-vs-yew distinction.
- 📡 **AP mode** — `hostapd + dnsmasq` so the box *becomes* the network instead of a node on someone else's.
- 🔧 **NVMe power-saving fix** — boot params `nvme_core.default_ps_max_latency_us=0 pcie_aspm=off pcie_port_pm=off` for the next image.

---

## 📜 License & citation

- **Code & LoRA adapters** — [Apache 2.0](LICENSE)
- **Gemma 4 weights** — [Gemma Terms of Use](https://ai.google.dev/gemma/terms)
- **Wikipedia passages in `gemma4/rag/corpus/`** — CC-BY-SA 4.0 (attribution in each row's `source` field)

```bibtex
@misc{lastbox_2026,
  title  = {LastBox: an offline survival assistant on Raspberry Pi 5 with fine-tuned Gemma 4 E2B},
  author = {Mateusz Pawelczuk},
  year   = {2026},
  url    = {https://github.com/agentsmill/lastbox},
  note   = {Kaggle "Gemma 4 Good" Hackathon 2026}
}
```

---

<div align="center">

**Built with ❤️ on a Raspberry Pi 5 and a single NVIDIA GB10. No cloud.**

[Full write-up](gemma4/SUBMISSION.md) · [Live site](https://agentsmill.github.io/lastbox/) · [Model card (v6)](https://huggingface.co/norecyc/lastbox-gemma4-e2b-v6-toolprior) · [Dataset card](https://huggingface.co/datasets/norecyc/lastbox-survival-dialogues)

</div>
