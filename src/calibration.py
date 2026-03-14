from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


def auto_detect_table(frame: np.ndarray) -> Optional[list[tuple[int, int]]]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 160)
    edges = cv2.dilate(edges, None, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_quad = None
    best_area = 0.0

    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) != 4:
            continue
        if not cv2.isContourConvex(approx):
            continue
        area = abs(cv2.contourArea(approx))
        if area > best_area:
            best_area = area
            best_quad = approx.reshape(4, 2)

    if best_quad is None:
        return None
    return _order_quad(best_quad)


def manual_select(frame: np.ndarray) -> list[tuple[int, int]]:
    points: list[tuple[int, int]] = []

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))

    window = "calibration"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)

    while True:
        canvas = frame.copy()
        _draw_points(canvas, points)
        cv2.imshow(window, canvas)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("c"):
            points.clear()
        if key == ord("q"):
            break
        if key == ord("s") and len(points) == 4:
            break

    cv2.setMouseCallback(window, lambda *args: None)
    return points


def _order_quad(quad: np.ndarray) -> list[tuple[int, int]]:
    pts = quad.astype(np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return [(int(tl[0]), int(tl[1])), (int(tr[0]), int(tr[1])),
            (int(br[0]), int(br[1])), (int(bl[0]), int(bl[1]))]


def _draw_points(frame: np.ndarray, points: list[tuple[int, int]]) -> None:
    for idx, (x, y) in enumerate(points, start=1):
        cv2.circle(frame, (x, y), 4, (0, 255, 255), -1)
        cv2.putText(
            frame,
            str(idx),
            (x + 6, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
