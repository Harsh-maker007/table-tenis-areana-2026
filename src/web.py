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
      .msg { font-size: 18px; color: #ffcc66; padding: 16px; }
    </style>
  </head>
  <body>
    <div class="wrap">
      <img src="/video" alt="stream">
    </div>
  </body>
</html>
"""


def create_app(
    video_path: str,
    camera_index: int,
    config: AppConfig,
) -> Flask:
    app = Flask(__name__)

    cap = open_video_source(video_path, camera_index)
    capture_ok = cap.isOpened()
    fps = cap.get(cv2.CAP_PROP_FPS) if capture_ok else 0.0
    tracker = BallTracker(config, fps=fps)

    def frame_generator() -> Generator[bytes, None, None]:
        if not capture_ok:
            return
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
        if not capture_ok:
            return render_template_string(
                INDEX_HTML.replace(
                    "<img src=\"/video\" alt=\"stream\">",
                    "<div class=\"msg\">No video source. Set VIDEO_PATH or CAMERA_INDEX.</div>",
                )
            )
        return render_template_string(INDEX_HTML)

    @app.get("/video")
    def video() -> Response:
        if not capture_ok:
            return Response("No video source", status=503)
        return Response(
            frame_generator(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    return app


def run_web_stream(
    video_path: str,
    camera_index: int,
    config: AppConfig,
    host: str,
    port: int,
) -> None:
    app = create_app(video_path, camera_index, config)
    app.run(host=host, port=port, threaded=True)


def run_web_stream_with_config(
    video_path: str,
    camera_index: int,
    host: str,
    port: int,
) -> None:
    config = load_config()
    run_web_stream(video_path, camera_index, config, host, port)
