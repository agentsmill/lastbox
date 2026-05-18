# LastBox-Gemma4

> Offline survival assistant on a Raspberry Pi 5 + LoRa mesh radio.
> Fine-tuned **Gemma 4 E2B** (text + vision), runs in 3.4 GB CPU on ARM, **1.5 s** first token.

**Kaggle "Gemma 4 Good" Hackathon submission** — full technical write-up: [`SUBMISSION.md`](SUBMISSION.md).

## What's here

```
gemma4/
├── SUBMISSION.md                Full hackathon write-up
├── README.md                    You are here
├── scripts/
│   ├── generate_dataset_v2.py   30 inline seeds → Kimi K2.5 teacher → 1151 dialogues
│   ├── process_v2.py            Validate byte caps + tool whitelist, dedupe, adapt
│   ├── train_sft.py             Unsloth FastModel LoRA SFT
│   └── eval_v2.py               Streaming live eval against llama-server
├── data/
│   ├── train_v2.jsonl           1034 EN training dialogues, 50/50 lora/touchscreen
│   ├── val_v2.jsonl             114 held-out
│   ├── golden_en.jsonl          25 held-out for agent-level eval
│   └── eval_v2_results.json     Latest live eval against deployed model
└── deploy/
    └── bundle/                  Ready-to-rsync to /mnt/nvme/lastbox-gemma4/
        ├── docker-compose.yml   llama.cpp:server with --mmproj
        ├── demo.py              Single-file orchestrator + 7 mocked tools
        ├── Modelfile            Ollama-compatible recipe (alt runtime)
        └── models/
            ├── lastbox-gemma4-e2b-q4_k_m.gguf   3.4 GB fine-tuned text
            ├── mmproj-F16.gguf                  985 MB vision encoder
            └── chat_template.jinja
```

## Quick start (run the deployed demo)

```bash
# Assumes lastbox is reachable on your network (here via Tailscale).
ssh pi@lastbox 'cd /mnt/nvme/lastbox-gemma4 && docker compose up -d'

# From any host on the network:
LASTBOX_ENDPOINT=http://lastbox:11436/v1 python deploy/bundle/demo.py --interactive
```

## Rebuild from scratch

```bash
export OPENROUTER_API_KEY=sk-or-...    # for data generation
python scripts/generate_dataset_v2.py  # ~30 min, ~$1
python scripts/process_v2.py
python scripts/train_sft.py \
    --train data/train_v2.jsonl --val data/val_v2.jsonl \
    --out ../out/lastbox-gemma4-e2b-sft-v2 \
    --epochs 3 --lora-r 8 --lora-alpha 8
# convert + quantize, then rsync — see SUBMISSION.md §"Reproducing"
```

## Key design choices

| Choice | Why |
|--------|-----|
| Gemma 4 **E2B** (not E4B) | Q4_K_M fits in 3.4 GB with 4 GB headroom for KV cache on Pi 5 |
| LoRA **r=8 α=8** | Unsloth's official recipe for Gemma 4 — bigger LoRAs over-parameterise 1 k samples |
| **bf16 LoRA** not QLoRA-4bit | GB10 has 121 GB unified; no need for 4-bit; clearer gradients |
| Chat template `gemma-4-thinking` | Newest Google template + llama.cpp May 2026 fixes — required for stable export |
| `llama.cpp` not Ollama | Better control on RPi 5 (threads, ubatch, --mmproj), one less abstraction |
| Hybrid response format | LoRa link enforces 200-byte cap; readable lead sentence + numbered list when procedural |
| Inline seeds in `generate_dataset_v2.py` | Single auditable file; YAML separation didn't pay off for 30 entries |
| Kimi K2.5 as teacher | $0.40 / $1.90 per M tok; strong JSONL adherence; ~$1.10 spent for 1151 dialogues |
| English (v2) | Hackathon judges read English; PL v1 retained as legacy under `training/` |

## Constraints respected

- **No internet at runtime** on lastbox after the GGUF lands — verified by network isolation test.
- **Hard byte cap** for the LoRa channel: 150 B UTF-8. Validator drops any sample over.
- **Tool whitelist enforced** at training time (`process_v2.py`) and at orchestrator time (`demo.py` rejects unknown names).
- **No shared GPU contention** during data gen + post-training quantize — training waited until the host's other vLLM benchmark finished.

## Status snapshot

```
SFT v2:       ✓ 1034/114 split, 43 min on GB10
GGUF Q4_K_M:  ✓ 3.43 GB
Vision:       ✓ mmproj-F16 loaded, capabilities=["completion","multimodal"]
Deploy:       ✓ docker-compose up on /mnt/nvme/lastbox-gemma4/
Live eval:    ✓ 1.5 s first-token median, 52% response-quality on golden_en
```
