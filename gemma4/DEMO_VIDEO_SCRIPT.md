# LastBox demo video script

**Target length:** 2-3 min (Kaggle wants "short video")
**Format:** screen recording + voice-over, optionally a 5 s b-roll shot of the actual Pi 5 hardware at the start.

---

## Shot list

### 0:00–0:15 — Hook

**Visual:** Close-up of the Raspberry Pi 5 in a case, NVMe HAT visible, LoRa antenna sticking out. Tailscale status overlay shows `lastbox 100.90.52.36 online`.

**VO:**
> "This is LastBox. A Raspberry Pi 5 running fine-tuned Gemma 4 E2B with native function calling and onboard vision. Zero internet. Mesh radio outbound. First token in one and a half seconds."

### 0:15–0:35 — The problem

**Visual:** Map of mountainous terrain or a disaster news clip; quick title cards:
- "1.3 B people live with unreliable mobile coverage"
- "When the network goes down, so does ChatGPT"
- "Survival decisions get made anyway"

**VO:**
> "When the cell network fails — disaster, mountain, remote clinic — survival decisions still have to be made, and they have to be made fast. We built a Gemma 4 assistant that fits in a pocket, runs in eight gigs of RAM, and answers in your hand."

### 0:35–1:00 — Architecture flash

**Visual:** Architecture diagram from SUBMISSION.md, animated:
1. user touchscreen / LoRa packet in
2. `demo.py` orchestrator
3. `llama-server` Docker container with `gemma4-e2b-q4_k_m.gguf + mmproj-F16.gguf`
4. 7 tool dispatchers
5. answer back out

**VO:**
> "Gemma 4 E2B fine-tuned with Unsloth on eleven hundred English survival dialogs. Runs as a llama-cpp server in Docker on the Pi's NVMe. The orchestrator parses tool calls, dispatches to seven native handlers — search knowledge, camera capture, LoRa send and listen, system status, memory — and feeds results back as the next turn."

### 1:00–1:30 — Live demo 1: emergency first aid

**Visual:** Terminal split. Left: `python demo.py --interactive`. Right: docker stats on RPi 5 showing live CPU/RAM/temp.

```
you> [source: touchscreen] My friend just got a deep cut on the thigh and it's spurting blood. What now?
```

(camera cuts to wall-clock counter showing first-token at ~1.5 s)

```
lastbox> Apply direct pressure with a clean cloth and pack the wound.
1. Press hard for 5–10 minutes.
2. Elevate the leg above heart level.
3. Tourniquet 5 cm above wound only if bleeding doesn't stop.
4. Mark time on tourniquet.
5. Get to EMS.

  (first token 1.49 s, 196 B UTF-8)
```

**VO:**
> "Touchscreen mode. Two-hundred-byte cap, hybrid format — one lead sentence plus a numbered list. Notice the byte count: it would survive a single LoRa packet to a relay node."

### 1:30–1:55 — Live demo 2: LoRa mode

**Visual:** Same terminal, this time prefix `lora:`.

```
you> lora: hypothermia signs?
lastbox> Shivering, slurred speech, confusion.
1. Warm gradually.
2. Dry clothes.
3. Sweet warm drink.
4. Severe: handle gently, evacuate.

  (first token 829 ms, 138 B — under 150 B LoRa cap)
```

**VO:**
> "LoRa mode forces a hundred-and-fifty-byte cap. The model knows it's on the radio. The output is dense enough that another LastBox node downstream can re-broadcast without chunking."

### 1:55–2:20 — Live demo 3: vision

**Visual:** Pi 5 camera shot of a red berry on a bush. Same prompt:

```
you> [source: lora] is this red berry safe to eat? <image>
lastbox> Don't eat. Common red berries: yew = lethal, rowan = bitter, hawthorn = OK.
1. Avoid until ID'd.
2. Photograph leaves + bark.
3. Wait 24h if exposed.

  (vision encode 11.2 s, first token 13.4 s, 158 B)
```

**VO:**
> "Multi-modal. Same model. The vision encoder, eighty-five hundred megabytes, runs on ARM-Cortex CPU through llama-cpp's mmproj path. Slower — eleven seconds for encoding — but every other call stays at one and a half."

### 2:20–2:45 — Mesh radio relay

**Visual:** Two terminals side by side: `lastbox` and a second mock receiver labelled "Node B".

```
you> Send to Node B that I'm at 50.12N 19.93E and need water.
lastbox> [transmitted 51 B over LoRa 868 MHz, ack pending]
```

Then Node B receives:
```
[node_B] 12:34:11 RX from lastbox -86 dBm: 50.12N 19.93E need water
```

**VO:**
> "The model emits a tool call, the dispatcher pushes the message over the SX1262 HAT. Node B logs it. This is mesh that works when nothing else does."

### 2:45–3:00 — Outro

**Visual:** Title card:
- "LastBox — Gemma 4 E2B + llama.cpp + LoRa"
- GitHub URL
- "Built for Kaggle / Google DeepMind Gemma 4 Good Hackathon, May 2026"

**VO:**
> "LastBox. Code, weights, and reproduction recipe at the link. Five categories, eleven-fifty training dialogs, three-point-four-gigabyte deploy, fully offline. Gemma 4 in a box you can take where the grid can't."

---

## Recording checklist

- [ ] Pi 5 is on, llama-server healthy, `curl localhost:11436/health` returns OK
- [ ] Tailscale up; ssh to lastbox works from recording host
- [ ] Terminal font readable at 1080p (use 18-20 pt)
- [ ] `docker stats lastbox-gemma4` open in a side panel
- [ ] LoRa test transmission to a second node prepared (or a mock receiver in the second tab)
- [ ] Vision test image (`hand_wound.jpg`, `red_berry.jpg`) already on the device
- [ ] First-token-latency printer enabled in `demo.py` (`--interactive` already shows it)

## Editing notes

- Cut hard between demos — Pi 5 vision encoding takes 11 s; keep it but speed-ramp it 4×.
- Lower-third captions for every command typed (people pause and read).
- End-card holds for 4 s. No music or low-volume ambient only — clarity over vibes.
