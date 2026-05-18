# LastBox Gemma 4 — deployment na Raspberry Pi 5 (lastbox via Tailscale)

Izolowany stack: dedykowany namespace `lastbox-gemma4`, port `11436` (omija ollama `:11434`/`:11435`), wolumeny na NVMe.

## Layout NVMe

```
/mnt/nvme/lastbox-gemma4/
├── models/
│   ├── lastbox-gemma4-e2b-q4_k_m.gguf      # ~1.4 GB
│   └── chat_template.jinja                   # Gemma 4 chat template
├── tools/
│   └── server.py                              # FastAPI tool dispatcher (importuje src/agent/)
└── data/
    └── knowledge/                             # baza wiedzy survival
```

## Sync z gx10 → lastbox (Tailscale)

```bash
# Z gx10 (po SFT skończonym):
RPI_USER=<your-pi-user>   # mateusz lub pi, zależnie od konfiguracji systemu
ssh ${RPI_USER}@lastbox 'sudo mkdir -p /mnt/nvme/lastbox-gemma4/{models,tools,data} && sudo chown -R ${RPI_USER}: /mnt/nvme/lastbox-gemma4'

rsync -avP --progress \
  /home/hurloth/lastbox_training/out/lastbox-gemma4-e2b-sft/lastbox-gemma4-e2b-q4_k_m.gguf \
  ${RPI_USER}@lastbox:/mnt/nvme/lastbox-gemma4/models/

rsync -avP \
  /home/hurloth/lastbox_training/gemma4/deploy/docker-compose.yml \
  /home/hurloth/lastbox_training/gemma4/deploy/chat_template.jinja \
  ${RPI_USER}@lastbox:/mnt/nvme/lastbox-gemma4/

rsync -avP \
  /home/hurloth/lastbox_training/src/agent/ \
  ${RPI_USER}@lastbox:/mnt/nvme/lastbox-gemma4/tools/
```

## Start na lastbox

```bash
ssh ${RPI_USER}@lastbox
cd /mnt/nvme/lastbox-gemma4
docker compose up -d
docker compose logs -f llama-server   # poczekaj ~60s na model warm-up
curl http://localhost:11436/health
```

## Test golden eval z gx10 (via Tailscale)

```bash
# Z gx10:
python -m training.eval.run_golden --endpoint http://lastbox:11436/v1 --model lastbox-gemma4-e2b
```

## Oczekiwane wyniki (RPi 5 8GB, Q4_K_M, CPU)

| Metryka                | Wartość                              |
|------------------------|--------------------------------------|
| Pamięć RAM zajęta      | ~2.5 GB (model + KV cache)           |
| First-token latency    | 3-5 s (cold), <2 s (warm)            |
| Sustained generation   | 8-12 tok/s                           |
| Max odpowiedź LoRa     | 200 B (~150 znaków PL UTF-8)         |
| Concurrent requests    | 1 (parallel=1, edge constraint)      |

## Separacja

- Port `11436` (Ollama gx10: 11434, gx10-extra: 11435)
- Network namespace: `lastbox-gemma4` (Docker)
- Volume root: `/mnt/nvme/lastbox-gemma4/` (oddzielne od `/mnt/nvme/*`)
- Container names: `lastbox-gemma4-{llama,tools}`
