from __future__ import annotations

from typing import Generator, Optional
from pathlib import Path

import cv2
from flask import Flask, Response, render_template_string, request, redirect, url_for

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
      .wrap { display: grid; place-items: center; min-height: 100vh; gap: 12px; padding: 16px; }
      img { max-width: 100vw; max-height: 75vh; }
      .msg { font-size: 18px; color: #ffcc66; padding: 16px; }
      form { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
      input { padding: 6px 10px; background: #222; color: #eee; border: 1px solid #444; border-radius: 6px; }
      button { padding: 6px 12px; background: #2b6; color: #071; border: none; border-radius: 6px; cursor: pointer; }
      button.secondary { background: #444; color: #eee; }
    </style>
  </head>
  <body>
    <div class="wrap">
      <img src="/video" alt="stream">
      <form method="post" action="/source">
        <input type="hidden" name="mode" value="sample">
        <button type="submit" class="secondary">Use sample frames</button>
      </form>
      <form method="post" action="/source">
        <input type="hidden" name="mode" value="video">
        <input type="text" name="path" placeholder="VIDEO_PATH">
        <button type="submit">Use video path</button>
      </form>
      <form method="post" action="/source">
        <input type="hidden" name="mode" value="camera">
        <input type="text" name="index" placeholder="Camera index (0)">
        <button type="submit">Use camera</button>
      </form>
    </div>
  </body>
</html>
"""


def create_app(
    video_path: str,
    camera_index: int,
    config: AppConfig,
    frames_dir: str | None = None,
) -> Flask:
    app = Flask(__name__)

    state = _StreamState(video_path, camera_index, frames_dir)
    state.start()

    tracker = BallTracker(config, fps=state.fps)

    def frame_generator() -> Generator[bytes, None, None]:
        if not state.capture_ok:
            return
        while True:
            ok, frame = state.next_frame()
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
        if not state.capture_ok:
            return render_template_string(
                INDEX_HTML.replace(
                    "<img src=\"/video\" alt=\"stream\">",
                    "<div class=\"msg\">No video source. Set VIDEO_PATH, CAMERA_INDEX, or SAMPLE_FRAMES_DIR.</div>",
                )
            )
        return render_template_string(INDEX_HTML)

    @app.get("/video")
    def video() -> Response:
        if not state.capture_ok:
            return Response("No video source", status=503)
        return Response(
            frame_generator(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.post("/source")
    def source() -> Response:
        mode = request.form.get("mode", "sample")
        if mode == "video":
            path = request.form.get("path", "")
            state.set_source(video_path=path, use_frames=False)
        elif mode == "camera":
            idx = request.form.get("index", "0")
            try:
                cam_idx = int(idx)
            except ValueError:
                cam_idx = 0
            state.set_source(camera_index=cam_idx, use_frames=False)
        else:
            state.set_source(use_frames=True)
        return redirect(url_for("index"))

    return app


def run_web_stream(
    video_path: str,
    camera_index: int,
    config: AppConfig,
    host: str,
    port: int,
    frames_dir: str | None = None,
) -> None:
    app = create_app(video_path, camera_index, config, frames_dir=frames_dir)
    app.run(host=host, port=port, threaded=True)


def run_web_stream_with_config(
    video_path: str,
    camera_index: int,
    host: str,
    port: int,
) -> None:
    config = load_config()
    run_web_stream(video_path, camera_index, config, host, port)


def _load_frames(frames_dir: str | None) -> list:
    if not frames_dir:
        return []
    path = Path(frames_dir)
    if not path.exists() or not path.is_dir():
        return []
    images = []
    for img_path in sorted(path.glob("*.jpg")):
        img = cv2.imread(str(img_path))
        if img is not None:
            images.append(img)
    return images


class _StreamState:
    def __init__(self, video_path: str, camera_index: int, frames_dir: str | None) -> None:
        self.video_path = video_path
        self.camera_index = camera_index
        self.frames_dir = frames_dir
        self.frames = _load_frames(frames_dir)
        self.use_frames = bool(self.frames)
        self.cap = None
        self.capture_ok = False
        self.fps = 0.0
        self._frame_idx = 0

    def start(self) -> None:
        if self.use_frames:
            self.capture_ok = True
            self.fps = 30.0
            return
        self.cap = open_video_source(self.video_path, self.camera_index)
        self.capture_ok = self.cap.isOpened()
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) if self.capture_ok else 0.0

    def set_source(
        self,
        video_path: str | None = None,
        camera_index: int | None = None,
        use_frames: bool | None = None,
    ) -> None:
        if video_path is not None:
            self.video_path = video_path
        if camera_index is not None:
            self.camera_index = camera_index
        if use_frames is not None:
            self.use_frames = use_frames

        if self.cap is not None:
            self.cap.release()
        self.frames = _load_frames(self.frames_dir) if self.use_frames else []
        self._frame_idx = 0
        self.start()

    def next_frame(self) -> tuple[bool, Optional[any]]:
        if not self.capture_ok:
            return False, None
        if self.use_frames:
            if not self.frames:
                return False, None
            frame = self.frames[self._frame_idx]
            self._frame_idx = (self._frame_idx + 1) % len(self.frames)
            return True, frame
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return False, None
        return True, frame
