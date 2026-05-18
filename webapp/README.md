# LastBox webapp

Live RPi5 camera stream + snap-to-Gemma, served from the lastbox itself.
Stdlib Python only (no pip), so it deploys on the SD card without depending on
the (currently broken) NVMe.

## Run on lastbox

```bash
# from gx10: rsync this dir to the Pi
rsync -avh webapp/ pi@lastbox:/home/pi/lastbox-webapp/

# on lastbox:
cd ~/lastbox-webapp
python3 server.py
```

Open `http://lastbox.local:8080/` on any device on the same LAN.

## Endpoints

| Method | Path        | What it does |
|--------|-------------|--------------|
| GET    | `/`         | UI: live MJPEG + Capture button + answer panel |
| GET    | `/stream`   | `multipart/x-mixed-replace` MJPEG from `rpicam-vid` |
| POST   | `/snap`     | Grab `rpicam-still` JPEG → POST to local llama-server multimodal endpoint → return `{answer, latency_ms, snapshot}` |
| GET    | `/health`   | `{status, camera, llama}` |

## Configuration (env vars)

| Var | Default | Notes |
|-----|---------|-------|
| `LASTBOX_WEBAPP_PORT` | `8080` | HTTP listen port |
| `LLAMA_URL` | `http://127.0.0.1:11436/v1/chat/completions` | llama.cpp server multimodal endpoint |

## Notes

- Camera: `imx708` (RPi Cam Module 3) detected via `rpicam-still --list-cameras`.
- Stream is 640×480 @ 15 fps MJPEG — sized for LAN throughput and low CPU.
- Snapshot is 1280×960 (sharper input to the vision encoder).
- Single `rpicam-vid` producer process; HTTP `/stream` clients are multiplexed.
- Snapshot via separate `rpicam-still` so we don't share-buffer the stream.
