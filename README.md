# LastBox

Offline survival assistant in a Pelican case: Raspberry Pi 5 + LoRa radio +
fine-tuned **Gemma 4 E2B**. Answers survival questions, identifies plants and
hazards from its camera, and relays mesh messages — fully offline,
~700 ms first-token, 6–7 tok/s sustained on ARM CPU.

**Kaggle "Gemma 4 Good" Hackathon 2026 submission.**

→ Full technical write-up: **[`gemma4/SUBMISSION.md`](gemma4/SUBMISSION.md)**

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

## License

Apache 2.0 for code (see `LICENSE` / `gemma4/LICENSE`).
Gemma 4 weights remain governed by the Gemma Terms of Use.
