# Live vision smoke test on Raspberry Pi 5 (deployed v3 model)

Date: 2026-05-18
Endpoint: `http://lastbox:11436/v1/chat/completions`
Model: `lastbox-gemma4-e2b-q4_k_m.gguf` (v3, no-think template, 3.43 GB)
Vision encoder: `mmproj-F16.gguf` (985 MB, SigLIP from base Gemma 4 E2B)
Host: Raspberry Pi 5, 8 GB RAM, ARM Cortex-A76 (no GPU, no NPU)

## Test setup

3 CC-licensed images from Wikimedia Commons, mixed sources (lora / touchscreen).
Hard byte caps: lora ≤150, touchscreen ≤200.

System prompt:
> You are LastBox - offline survival assistant. Reply ONLY with the final answer
> in hybrid format (1 sentence or 1-3 short numbered points). No thinking, no analysis.
> lora source = ≤150 bytes UTF-8. Identify safety hazards directly.

## Results

| # | Image | Size | Source | Latency | Bytes | Cap | Response |
|---|-------|------|--------|---------|-------|-----|----------|
| 1 | `rowan_berry.jpg` (edible) | 72 KB | lora | **29.0 s** | 116 | 150 ✓ | "1. Unknown plant; do not eat unless positively identified.<br>2. Potential for toxicity or severe stomach upset exists." |
| 2 | `yew_berry_toxic.jpg` (lethal) | 36 KB | lora | **26.8 s** | 126 | 150 ✓ | "1. Unknown plant; avoid consumption.<br>2. Potential for unknown toxins.<br>3. Do not eat unless positively identified by an expert." |
| 3 | `bleeding_finger.jpg` (first aid) | 16 KB | touchscreen | **28.2 s** | 119 | 200 ✓ | "1. Apply direct pressure to stop bleeding.<br>2. Elevate the injured area above the heart.<br>3. Seek immediate medical help." |

## Findings

### What works ✅

- **End-to-end multimodal**: image → base64 → SigLIP encoder → Gemma 4 → text answer. No code changes on the runtime beyond passing `--mmproj` to `llama-server`.
- **Format compliance**: 3/3 responses in clean hybrid format (numbered list, no thinking, no preambles).
- **Byte compliance**: 3/3 under the per-source UTF-8 byte cap.
- **Safety bias**: the model defaults to "do not eat unless positively identified" for ambiguous plant images — this is the correct safety posture even if it sacrifices the rowan-vs-yew discrimination.
- **First-aid scenario** (image #3): the model produced a textbook hybrid-format first-aid response from a single image of a small cut, without any guidance text. This is the strongest end-to-end demonstration of the multimodal capability.

### Limitations ⚠️

- **Latency 27-29 s** per image: ~4-5× slower than the 8-15 s estimate from published RPi 5 benchmarks. SigLIP encoding on Cortex-A76 CPU is bandwidth-bound; an INT8 quant of the mmproj or an NPU offload (Hailo, Coral) would help.
- **No specific plant ID**: the model doesn't distinguish rowan (edible) from yew (toxic). Vision weights were left unchanged during SFT (`finetune_vision_layers=False`). A v4 fine-tune with image-paired survival data would target this — about 200 labeled plant images and another ~1 h of GB10 time would likely suffice.
- **Single-slot serving**: concurrent vision requests would queue behind one slot.

### What this means for the hackathon

The multimodal capability is **demonstrably real** on the device, not a paper claim. The model can ingest a real photograph, produce a survival-appropriate response in the constrained format, and ship it under the LoRa byte cap. The Plant-ID distinction is the next obvious upgrade and is explicitly scoped in the "Known limitations" section.
