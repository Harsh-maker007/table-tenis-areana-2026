import argparse
from pathlib import Path

import cv2
import numpy as np

from src.config import load_config, save_config, AppConfig
from src.calibration import auto_detect_table, manual_select
from src.tracking import BallTracker
from src.video import open_video_source
from src.visuals import draw_overlay, ensure_bgr
from src.web import run_web_stream


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Table tennis ball tracking demo")
    parser.add_argument("--video", type=str, default="", help="Path to a video file")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument("--no-gui", action="store_true", help="Disable imshow windows")
    parser.add_argument("--save", type=str, default="", help="Optional output video path")
    parser.add_argument("--web", action="store_true", help="Serve frames in a browser via Flask")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Web host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Web port (default: 8000)")
    parser.add_argument("--calibrate", action="store_true", help="Calibrate table ROI/quad")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config()

    if args.calibrate:
        _run_calibration(args.video, args.camera, config)
        return

    if args.web:
        run_web_stream(
            video_path=args.video,
            camera_index=args.camera,
            config=config,
            host=args.host,
            port=args.port,
        )
        return

    cap = open_video_source(args.video, args.camera)
    if not cap.isOpened():
        raise RuntimeError("Unable to open video source")

    writer = None
    if args.save:
        output_path = Path(args.save)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    fps = cap.get(cv2.CAP_PROP_FPS)
    tracker = BallTracker(config, fps=fps)

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        frame = ensure_bgr(frame)
        result = tracker.update(frame)
        draw_overlay(frame, result)

        if args.save:
            if writer is None:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(output_path), fourcc, config.output_fps, (w, h))
            writer.write(frame)

        if not args.no_gui:
            cv2.imshow("table-tennis-tracking", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    cap.release()
    if writer is not None:
        writer.release()
    if not args.no_gui:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


def _run_calibration(video_path: str, camera_index: int, config: AppConfig) -> None:
    cap = open_video_source(video_path, camera_index)
    if not cap.isOpened():
        raise RuntimeError("Unable to open video source")

    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError("Unable to read a frame for calibration")

    frame = ensure_bgr(frame)
    quad = auto_detect_table(frame)
    while True:
        canvas = frame.copy()
        if quad:
            pts = np.array(quad, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [pts], True, (0, 255, 255), 2)
        cv2.putText(
            canvas,
            "a=auto  m=manual  s=save  q=quit",
            (12, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (240, 240, 240),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("calibration", canvas)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("a"):
            quad = auto_detect_table(frame)
        elif key == ord("m"):
            quad = manual_select(frame)
        elif key == ord("s") and quad and len(quad) == 4:
            h, w = frame.shape[:2]
            config.table_quad_norm = [(x / w, y / h) for x, y in quad]
            x_vals = [p[0] for p in config.table_quad_norm]
            y_vals = [p[1] for p in config.table_quad_norm]
            config.table_roi_norm = (min(x_vals), min(y_vals), max(x_vals), max(y_vals))
            save_config(config)
            break
        elif key == ord("q"):
            break

    cv2.destroyWindow("calibration")
