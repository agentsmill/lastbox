# LastBox

Offline survival assistant in a Pelican case: Raspberry Pi 5 + LoRa radio +
fine-tuned **Gemma 4 E2B**. Answers survival questions, identifies plants and
hazards from its camera, and relays mesh messages — fully offline,
~700 ms first-token, 6–7 tok/s sustained on ARM CPU.

**Kaggle "Gemma 4 Good" Hackathon 2026 submission.**

→ Full technical write-up: **[`gemma4/SUBMISSION.md`](gemma4/SUBMISSION.md)**

> **Submission snapshot:** the state of this repo at the 2026-05-18 23:59 UTC
> deadline is tagged [`v1-submission`](https://github.com/agentsmill/lastbox/tree/v1-submission).
> Anything after that tag is **post-deadline research**, not part of judging
> (see ["Post-deadline research" below](#post-deadline-research-not-part-of-submission)).

## Repo map

```
gemma4/                     The hackathon submission.
  SUBMISSION.md             Architecture, training, eval, design rationale.
  README.md                 Code-level orientation.
  DEMO_VIDEO_SCRIPT.md      Storyboard for the demo recording.
  scripts/
    generate_dataset_v2.py  Inline 30 seeds → Kimi K2.5 teacher → 1151 dialogs.
    process_v2.py           Byte-cap validation, tool whitelist, dedupe.
    train_sft.py            Unsloth FastModel LoRA SFT (r=8, α=8).
    eval_v2.py              Streaming agent eval against the deployed llama-server.
    vision_smoke.py         Three CC-licensed images → mmproj-F16 → Gemma 4.
  data/
    train_v2.jsonl          1034 EN dialogues.
    val_v2.jsonl            114 held-out.
    golden_en.jsonl         25 agent-eval prompts (first-aid, bushcraft, …).
    eval_v3_results.json    Live eval against deployed v3 (shipped checkpoint).
  deploy/bundle/            Rsync target for /mnt/nvme/lastbox-gemma4/ on the Pi.
    demo.py                 314-line orchestrator + 7 mock tool dispatchers.
    docker-compose.yml      Single-service llama.cpp:server with --mmproj.
    Modelfile               Ollama-compatible recipe (alt runtime).
    test_images/            CC0 plant + wound photos used for vision smoke.

webapp/                     Live RPi5 camera → Gemma multimodal UI, served from
                            the lastbox itself on http://lastbox.local:8080/.
                            Stdlib Python only, so it runs from the SD card.

out/lastbox-gemma4-e2b-sft-v3-nothink/lora/   LoRA adapter (50 MB) — full
                            checkpoint is too big for git; the adapter plus the
                            training scripts above are enough to merge + GGUF
                            export from scratch (see SUBMISSION.md §"Reproducing").
```

## Quickstart for graders

```bash
# 1. Open the writeup — that's the primary artefact:
$EDITOR gemma4/SUBMISSION.md

# 2. The model itself runs on the RPi5; smoke-test from any host on the LAN
#    (or via Tailscale) once the docker stack is up:
curl http://lastbox.local:11436/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"v3-q4_k_m.gguf","messages":[
        {"role":"user","content":"How do I stop heavy bleeding on the forearm?"}],
       "max_tokens":80}'

# 3. Live camera demo (RPi5 cam → Gemma vision → web UI):
xdg-open http://lastbox.local:8080/
```

## Repro from scratch

See `gemma4/SUBMISSION.md` §"Reproducing" — `generate → process → train_sft →
convert → quantize → rsync` in ~1.5 h on a single GB10 (or any modern CUDA box).

## Post-deadline research (not part of submission)

After the deadline we kept iterating to test the v1 roadmap items in practice.
Headline numbers vs the v3 SFT shipped at deadline (golden_en, n=25):

| Metric              | v3 (submission) | v6 (post-deadline) |
|---------------------|-----------------|--------------------|
| tool_emission_rate  | ~0%             | **72%**            |
| tool_accuracy       | 0%              | **64%**            |
| agentic_score       | 0.016           | **0.608** (38×)    |
| byte_compliance     | 0.48            | **1.000**          |
| format_ok           | 0.52            | **1.000**          |
| response_quality    | 0.506           | **1.000**          |
| completed/25        | 14              | **25**             |

What changed (commits after [`v1-submission`](https://github.com/agentsmill/lastbox/tree/v1-submission)):

- **RAG live on the box** — `bge-small-en-v1.5` embed-server + 4 074-passage
  index (train_v2 + Wikipedia first-aid/bushcraft + manuals) + cosine top-k.
  `webapp/server.py` exposes `/chat?rag=true` that cites source IDs.
- **GRPO v4 + v5** — two iterations with different reward designs; both
  plateaued at 0% tool emission. Diagnosis in `SUBMISSION.md`:
  GRPO with KL β=0.04 cannot move a base with p(tool)≈0 to ≈1 in 200 steps.
- **v6 SFT warmup** — 12-min targeted SFT on 1 034 tool-only pairs from
  `train_v2.jsonl`. Sets the prior; lifts tool emission from 0% → 72%.
- **Eval methodology fixes** — (a) the original eval was passing
  `SYSTEM_PROMPT_EN` without the tool definitions JSON, suppressing tool
  emission; (b) streaming SSE was disconnecting mid-stream on 12/25 queries,
  pulling format/byte/persona to a 0.52 floor that was *never* a quality
  issue. Non-streaming POST + retry → 25/25 completion.

`gemma4/SUBMISSION.md` has the full retrospective with reward designs,
training hyperparameters, and the lesson kept: *SFT prior + GRPO refines*
beats *GRPO from zero* for forcing new behaviours under a KL constraint.

## Roadmap (wired, not vapor)

The v1 box ships as a working baseline. Switches that are designed in but not
flipped for this submission — full breakdown in
[`gemma4/SUBMISSION.md` §"Roadmap"](gemma4/SUBMISSION.md):

- **Voice in/out** — `arecord → whisper.cpp tiny.en → /radio-query → piper/espeak`. Blocked on the ReSpeaker HAT's actual codec (TLV320AIC3104, not the silkscreened WM8960).
- **Real mesh-radio packets** — `meshtastic --port` wiring already present in `demo.py`. Activates the moment a working LoRa device shows up.
- **GRPO tool-call training** — close the ~0% `<tool_call>` emission gap with `r = +1 if expected_tool_called else 0`.
- **RAG over offline survival manuals** — `nomic-embed-text` + `sqlite-vss` over US Army FM 21-76, WikiMed ZIM, and our own `train_v2.jsonl`. ~2.5 GB on the SD card, ~150–300 ms latency hit, citations attached to every answer. v1 (Polish) shipped this; v2 ships without so baseline measures the fine-tune itself.
- **Image-paired SFT** — ~500 CC-licensed plant photos for safe plant-ID (rowan vs. yew toxicity).
- **AP mode** — `hostapd + dnsmasq`; the box becomes the network, not a node on someone else's.
- **NVMe power-saving fix** — boot params `nvme_core.default_ps_max_latency_us=0 pcie_aspm=off pcie_port_pm=off` for the next image.

## License

Apache 2.0 for code (see `LICENSE` / `gemma4/LICENSE`).
Gemma 4 weights remain governed by the Gemma Terms of Use.
