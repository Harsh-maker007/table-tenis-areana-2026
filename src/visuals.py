from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class TrackResult:
    center: Optional[tuple[int, int]]
    radius: Optional[int]
    confidence: float
    predicted: Optional[tuple[int, int]]
    mask: np.ndarray
    bbox: Optional[tuple[int, int, int, int]]
    speed: float
    trajectory: list[tuple[int, int]]
    bounced: bool
    in_table: Optional[bool]
    table_roi: tuple[int, int, int, int]
    table_quad: Optional[list[tuple[int, int]]]


def ensure_bgr(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return frame


def draw_overlay(frame: np.ndarray, result: TrackResult) -> None:
    _draw_table_roi(frame, result.table_roi)
    _draw_table_quad(frame, result.table_quad)
    _draw_trajectory(frame, result.trajectory)
    if result.bbox is not None:
        x0, y0, x1, y1 = result.bbox
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 200, 255), 2)
    if result.predicted is not None:
        cv2.circle(frame, result.predicted, 3, (255, 0, 0), -1)
    if result.center is not None and result.radius is not None:
        cv2.circle(frame, result.center, result.radius, (0, 255, 0), 2)
        cv2.circle(frame, result.center, 2, (0, 255, 0), -1)
        _draw_labels(frame, result)


def _draw_labels(frame: np.ndarray, result: TrackResult) -> None:
    x = 12
    y = 22
    color = (235, 235, 235)
    cv2.putText(
        frame,
        f"conf: {result.confidence:.2f}",
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )
    y += 22
    cv2.putText(
        frame,
        f"speed(px/s): {result.speed:.1f}",
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )
    y += 22
    if result.in_table is None:
        in_text = "in/out: --"
        in_color = (180, 180, 180)
    else:
        in_text = "in/out: IN" if result.in_table else "in/out: OUT"
        in_color = (0, 200, 0) if result.in_table else (0, 0, 200)
    cv2.putText(
        frame,
        in_text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        in_color,
        2,
        cv2.LINE_AA,
    )
    if result.bounced:
        cv2.putText(
            frame,
            "BOUNCE",
            (x, y + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )


def _draw_trajectory(frame: np.ndarray, points: list[tuple[int, int]]) -> None:
    if len(points) < 2:
        return
    for i in range(1, len(points)):
        thickness = max(1, int(4 * (i / len(points))))
        cv2.line(frame, points[i - 1], points[i], (50, 180, 255), thickness)


def _draw_table_roi(frame: np.ndarray, roi: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = roi
    cv2.rectangle(frame, (x0, y0), (x1, y1), (80, 80, 80), 1)


def _draw_table_quad(frame: np.ndarray, quad: Optional[list[tuple[int, int]]]) -> None:
    if not quad:
        return
    pts = np.array(quad, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], True, (0, 255, 255), 2)
