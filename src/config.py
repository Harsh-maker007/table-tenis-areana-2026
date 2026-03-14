from dataclasses import dataclass


@dataclass(frozen=True)
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
