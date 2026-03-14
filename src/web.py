from __future__ import annotations

from typing import Generator, Optional

import cv2
from flask import Flask, Response, render_template_string

from src.config import AppConfig, load_config
from src.tracking import BallTracker
from src.video import open_video_source
from src.visuals import draw_overlay, ensure_bgr


INDEX_HTML = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>Table Tennis Tracking</title>
    <style>
      body { margin: 0; background: #111; color: #eee; font-family: Arial, sans-serif; }
      .wrap { display: grid; place-items: center; height: 100vh; }
      img { max-width: 100vw; max-height: 100vh; }
    </style>
  </head>
  <body>
    <div class="wrap">
      <img src="/video" alt="stream">
    </div>
  </body>
</html>
"""


def run_web_stream(
    video_path: str,
    camera_index: int,
    config: AppConfig,
    host: str,
    port: int,
) -> None:
    app = Flask(__name__)

    cap = open_video_source(video_path, camera_index)
    if not cap.isOpened():
        raise RuntimeError("Unable to open video source")

    fps = cap.get(cv2.CAP_PROP_FPS)
    tracker = BallTracker(config, fps=fps)

    def frame_generator() -> Generator[bytes, None, None]:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frame = ensure_bgr(frame)
            result = tracker.update(frame)
            draw_overlay(frame, result)
            ok, buffer = cv2.imencode(".jpg", frame)
            if not ok:
                continue
            jpg_bytes = buffer.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpg_bytes + b"\r\n"
            )

    @app.get("/")
    def index() -> str:
        return render_template_string(INDEX_HTML)

    @app.get("/video")
    def video() -> Response:
        return Response(
            frame_generator(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    app.run(host=host, port=port, threaded=True)


def run_web_stream_with_config(
    video_path: str,
    camera_index: int,
    host: str,
    port: int,
) -> None:
    config = load_config()
    run_web_stream(video_path, camera_index, config, host, port)
