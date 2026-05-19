---
title: LastBox Chat
emoji: 📦
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
short_description: Offline survival assistant on RPi 5 (Gemma 4 E2B)
models:
- norecyc/lastbox-gemma4-e2b-v6-toolprior
datasets:
- norecyc/lastbox-survival-dialogues
license: apache-2.0
hf_oauth: false
suggested_hardware: zero-a10g
suggested_storage: small
---

# 📦 LastBox — chat with the offline survival assistant

This Space runs the [v6 SFT-warmup checkpoint](https://huggingface.co/norecyc/lastbox-gemma4-e2b-v6-toolprior)
of LastBox — a fine-tuned **Gemma 4 E2B** designed to live on a battery-powered
Raspberry Pi 5 in a Pelican case, answering survival questions over LoRa 868 MHz
mesh radio with no internet.

**Three modes** (Gradio tabs):

- 📻 **LoRa Radio** — terse replies under the 150-byte UTF-8 cap, with tool-call emission
- 💬 **Free Chat** — long-form survival Q&A, crisis-mode prompt
- 🔍 **RAG Chat** — retrieves top-4 passages from our 4 074-passage offline corpus and cites source IDs

## Headline numbers

| Metric              | v3 (submission) | v6 (this Space) |
|---------------------|-----------------|-----------------|
| tool_emission_rate  | ~0 %            | **72 %**        |
| agentic_score       | 0.016           | **0.608** (38×) |
| byte_compliance     | 0.48            | **1.000**       |
| format_ok           | 0.52            | **1.000**       |

Full retrospective: [GitHub repo](https://github.com/agentsmill/lastbox) · [project site](https://agentsmill.github.io/lastbox/).

## Hardware

ZeroGPU (H200, free, queued). Cold start ~30 s, warm ~5-10 s.

## License

Code & LoRA — Apache 2.0. Gemma 4 weights subject to [Gemma Terms of Use](https://ai.google.dev/gemma/terms).
