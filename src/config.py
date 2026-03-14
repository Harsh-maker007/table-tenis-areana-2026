from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Optional


@dataclass
class AppConfig:
    # HSV bounds for a typical orange table-tennis ball.
    # Adjust these after a few frames with a sample video.
    hsv_lower: tuple[int, int, int] = (5, 120, 120)
    hsv_upper: tuple[int, int, int] = (20, 255, 255)

    # Morphology to clean up the mask
    erode_iters: int = 1
    dilate_iters: int = 2

    # Area thresholds to reject noise
    min_area: int = 20
    max_area: int = 2000

    # Motion gating
    max_jump_px: int = 60
    min_motion_px: int = 2

    # Kalman filter noise terms
    process_noise: float = 1e-2
    measurement_noise: float = 1e-1
    error_cov_post: float = 1.0

    # Output
    output_fps: float = 30.0

    # Trajectory + speed
    trajectory_length: int = 40
    speed_smoothing: float = 0.4  # EMA alpha in [0,1]

    # Bounce detection (pixels/sec)
    bounce_vy_min: float = 120.0
    bounce_cooldown_frames: int = 6

    # Table region of interest in normalized coords (x0, y0, x1, y1).
    # Default is a central crop; update these to the exact table bounds for accuracy.
    table_roi_norm: tuple[float, float, float, float] = (0.08, 0.22, 0.92, 0.88)
    # Optional 4-point polygon (normalized) for accurate in/out under perspective.
    table_quad_norm: Optional[list[tuple[float, float]]] = None


def load_config(path: str = "config.json") -> AppConfig:
    cfg = AppConfig()
    cfg_path = Path(path)
    if not cfg_path.exists():
        return cfg
    try:
        data = json.loads(cfg_path.read_text())
    except json.JSONDecodeError:
        return cfg

    if "table_roi_norm" in data:
        cfg.table_roi_norm = tuple(data["table_roi_norm"])
    if "table_quad_norm" in data and data["table_quad_norm"]:
        cfg.table_quad_norm = [tuple(p) for p in data["table_quad_norm"]]
    return cfg


def save_config(config: AppConfig, path: str = "config.json") -> None:
    cfg_path = Path(path)
    payload = {
        "table_roi_norm": list(config.table_roi_norm),
        "table_quad_norm": (
            [list(p) for p in config.table_quad_norm]
            if config.table_quad_norm
            else None
        ),
    }
    cfg_path.write_text(json.dumps(payload, indent=2))
