#!/usr/bin/env python3
"""
LastBox webapp — live RPi5 camera stream + snap-to-Gemma.

Stdlib only (http.server + threading + subprocess). Runs on the RPi5 itself,
serves on port 8080. Any device on the same LAN opens http://lastbox.local:8080
to see the live MJPEG stream and ask Gemma about what the camera sees.

Pipeline:
  GET  /             → index.html with <img src="/stream"> + snap button
  GET  /stream       → multipart/x-mixed-replace MJPEG from rpicam-vid
  POST /snap         → grab one rpicam-still JPEG, POST to local llama-server
                       multimodal endpoint, return JSON {answer, latency_ms}
  GET  /health       → {"status":"ok", "camera":..., "llama":...}
"""
from __future__ import annotations

import base64
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PORT = int(os.environ.get("LASTBOX_WEBAPP_PORT", "8080"))
LLAMA_URL = os.environ.get("LLAMA_URL", "http://127.0.0.1:11436/v1/chat/completions")
STATIC_DIR = Path(__file__).parent / "static"

# rpicam-vid stream tuning: 640x480, 15 fps, MJPEG, infinite duration.
# Smaller resolution = lower CPU, faster perceived stream over LAN.
RPICAM_VID_CMD = [
    "rpicam-vid",
    "--width", "640",
    "--height", "480",
    "--framerate", "15",
    "--codec", "mjpeg",
    "--nopreview",
    "--inline",
    "--timeout", "0",
    "--output", "-",
]

RPICAM_STILL_CMD = [
    "rpicam-still",
    "--width", "1280",
    "--height", "960",
    "--encoding", "jpg",
    "--quality", "85",
    "--nopreview",
    "--timeout", "200",
    "--output", "-",
]

SYSTEM_PROMPT = (
    "You are LastBox, an offline survival assistant running on a Raspberry Pi 5. "
    "Describe what you see in the image in 1-2 short sentences. "
    "If you see a plant, terrain, wound, or hazard relevant to survival, mention it. "
    "Be direct and useful. Reply in English."
)


# ─────────────────────────────────────────────────────────────────────
# Camera stream — single rpicam-vid process, multiplexed to many clients
# ─────────────────────────────────────────────────────────────────────

class CameraBroker:
    """Single rpicam-vid producer → many HTTP MJPEG consumers."""

    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None
        self._latest_frame: bytes | None = None
        self._frame_lock = threading.Lock()
        self._frame_cond = threading.Condition(self._frame_lock)
        self._stopped = threading.Event()

    def start(self) -> None:
        try:
            self._proc = subprocess.Popen(
                RPICAM_VID_CMD,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )
        except FileNotFoundError:
            print("[camera] rpicam-vid not found; stream disabled", file=sys.stderr)
            return
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self) -> None:
        assert self._proc and self._proc.stdout
        buf = b""
        SOI = b"\xff\xd8"  # JPEG start
        EOI = b"\xff\xd9"  # JPEG end
        while not self._stopped.is_set():
            chunk = self._proc.stdout.read(4096)
            if not chunk:
                break
            buf += chunk
            while True:
                start = buf.find(SOI)
                if start < 0:
                    buf = buf[-2:]  # keep tail in case marker is split
                    break
                end = buf.find(EOI, start + 2)
                if end < 0:
                    if start > 0:
                        buf = buf[start:]
                    break
                frame = buf[start : end + 2]
                buf = buf[end + 2 :]
                with self._frame_cond:
                    self._latest_frame = frame
                    self._frame_cond.notify_all()

    def get_frame(self, timeout: float = 5.0) -> bytes | None:
        with self._frame_cond:
            self._frame_cond.wait(timeout=timeout)
            return self._latest_frame

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None


CAMERA = CameraBroker()


# ─────────────────────────────────────────────────────────────────────
# Snap → Gemma multimodal call
# ─────────────────────────────────────────────────────────────────────

def grab_still() -> bytes | None:
    """Take a high-res still via rpicam-still (separate process; brief).

    We use rpicam-still rather than reusing the stream frame so the snapshot is
    sharper than the 640x480 stream — gives the model better detail to reason
    about.
    """
    try:
        out = subprocess.check_output(
            RPICAM_STILL_CMD, stderr=subprocess.DEVNULL, timeout=4
        )
        return out
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        # Fallback: use the latest stream frame
        return CAMERA.get_frame(timeout=2.0)


def ask_gemma(jpeg_bytes: bytes, prompt: str | None = None) -> tuple[str, int]:
    b64 = base64.b64encode(jpeg_bytes).decode("ascii")
    payload = {
        "model": "v3-q4_k_m.gguf",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt or "What do you see?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                ],
            },
        ],
        "max_tokens": 160,
        "temperature": 0.6,
        "stream": False,
    }
    t0 = time.time()
    req = urllib.request.Request(
        LLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=90) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    latency_ms = int((time.time() - t0) * 1000)
    msg = data["choices"][0]["message"]["content"].strip()
    return msg, latency_ms


# ─────────────────────────────────────────────────────────────────────
# HTTP handler
# ─────────────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    server_version = "lastbox-webapp/0.1"

    def log_message(self, fmt: str, *args) -> None:
        print(f"[http] {self.address_string()} {fmt % args}", file=sys.stderr)

    def _send_json(self, code: int, obj: dict) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            return self._serve_static("index.html", "text/html; charset=utf-8")
        if self.path == "/stream":
            return self._serve_stream()
        if self.path == "/health":
            llama_ok = False
            try:
                with urllib.request.urlopen(
                    LLAMA_URL.replace("/v1/chat/completions", "/health"), timeout=2
                ) as r:
                    llama_ok = r.status == 200
            except Exception:
                pass
            return self._send_json(
                200,
                {
                    "status": "ok",
                    "camera": CAMERA.is_alive(),
                    "llama": llama_ok,
                },
            )
        self.send_error(404)

    def do_POST(self) -> None:
        if self.path == "/snap":
            return self._handle_snap()
        self.send_error(404)

    def _serve_static(self, name: str, ctype: str) -> None:
        path = STATIC_DIR / name
        if not path.exists():
            self.send_error(404)
            return
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _serve_stream(self) -> None:
        boundary = b"--frame"
        self.send_response(200)
        self.send_header(
            "Content-Type", "multipart/x-mixed-replace; boundary=frame"
        )
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            while True:
                frame = CAMERA.get_frame(timeout=5.0)
                if frame is None:
                    continue
                self.wfile.write(b"\r\n")
                self.wfile.write(boundary)
                self.wfile.write(b"\r\nContent-Type: image/jpeg\r\nContent-Length: ")
                self.wfile.write(str(len(frame)).encode("ascii"))
                self.wfile.write(b"\r\n\r\n")
                self.wfile.write(frame)
        except (BrokenPipeError, ConnectionResetError):
            return

    def _handle_snap(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length else b"{}"
        try:
            req = json.loads(body or b"{}")
        except json.JSONDecodeError:
            req = {}
        prompt = (req.get("prompt") or "").strip() or None

        jpeg = grab_still()
        if not jpeg:
            return self._send_json(503, {"error": "camera unavailable"})
        try:
            answer, latency_ms = ask_gemma(jpeg, prompt=prompt)
        except Exception as e:
            return self._send_json(502, {"error": f"gemma call failed: {e}"})
        b64 = base64.b64encode(jpeg).decode("ascii")
        self._send_json(
            200,
            {
                "answer": answer,
                "latency_ms": latency_ms,
                "snapshot": f"data:image/jpeg;base64,{b64}",
            },
        )


def main() -> int:
    print(f"[lastbox-webapp] starting on 0.0.0.0:{PORT}, llama at {LLAMA_URL}")
    CAMERA.start()
    # Brief warm-up so /stream has frames as soon as a client connects
    time.sleep(1.0)
    httpd = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    host = socket.gethostname()
    print(f"[lastbox-webapp] open http://{host}.local:{PORT}/ on any LAN device")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
