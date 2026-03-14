from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Optional

import cv2
import numpy as np

from src.config import AppConfig
from src.visuals import TrackResult


@dataclass
class Candidate:
    center: tuple[int, int]
    radius: int
    area: float


class BallTracker:
    def __init__(self, config: AppConfig, fps: Optional[float] = None) -> None:
        self.config = config
        self.kalman = self._init_kalman()
        self.last_center: Optional[tuple[int, int]] = None
        self.last_speed: float = 0.0
        self.fps = fps if fps and fps > 0 else config.output_fps
        self.trajectory: deque[tuple[int, int]] = deque(maxlen=config.trajectory_length)
        self._bounce_cooldown = 0

    def _init_kalman(self) -> cv2.KalmanFilter:
        kalman = cv2.KalmanFilter(4, 2)
        kalman.transitionMatrix = np.array(
            [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 0, 1]],
            dtype=np.float32,
        )
        kalman.measurementMatrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]],
            dtype=np.float32,
        )
        kalman.processNoiseCov = np.eye(4, dtype=np.float32) * self.config.process_noise
        kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * self.config.measurement_noise
        kalman.errorCovPost = np.eye(4, dtype=np.float32) * self.config.error_cov_post
        return kalman

    def update(self, frame: np.ndarray) -> TrackResult:
        mask = self._segment_ball(frame)
        candidates = self._find_candidates(mask)

        predicted = self.kalman.predict()
        predicted_center = (int(predicted[0]), int(predicted[1]))

        best = self._select_candidate(candidates, predicted_center)
        if best is not None:
            measurement = np.array([[np.float32(best.center[0])], [np.float32(best.center[1])]])
            self.kalman.correct(measurement)
            bbox = self._bbox_from_circle(best.center, best.radius, frame.shape)
            speed = self._compute_speed(best.center)
            self.trajectory.append(best.center)
            bounced = self._detect_bounce()
            table_roi = self._table_roi(frame.shape)
            table_quad = self._table_quad(frame.shape)
            in_table = self._in_table(best.center, table_roi, table_quad)
            self.last_center = best.center
            confidence = min(1.0, best.area / max(self.config.min_area, 1))
            return TrackResult(
                center=best.center,
                radius=best.radius,
                confidence=confidence,
                predicted=predicted_center,
                mask=mask,
                bbox=bbox,
                speed=speed,
                trajectory=list(self.trajectory),
                bounced=bounced,
                in_table=in_table,
                table_roi=table_roi,
                table_quad=table_quad,
            )

        self.last_center = None
        self.trajectory.clear()
        self.last_speed = 0.0
        if self._bounce_cooldown > 0:
            self._bounce_cooldown -= 1
        table_roi = self._table_roi(frame.shape)
        table_quad = self._table_quad(frame.shape)
        return TrackResult(
            center=None,
            radius=None,
            confidence=0.0,
            predicted=predicted_center,
            mask=mask,
            bbox=None,
            speed=0.0,
            trajectory=[],
            bounced=False,
            in_table=None,
            table_roi=table_roi,
            table_quad=table_quad,
        )

    def _segment_ball(self, frame: np.ndarray) -> np.ndarray:
        blur = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.config.hsv_lower, self.config.hsv_upper)
        mask = cv2.erode(mask, None, iterations=self.config.erode_iters)
        mask = cv2.dilate(mask, None, iterations=self.config.dilate_iters)
        return mask

    def _find_candidates(self, mask: np.ndarray) -> list[Candidate]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: list[Candidate] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.config.min_area or area > self.config.max_area:
                continue
            (x, y), radius = cv2.minEnclosingCircle(contour)
            if radius <= 0:
                continue
            candidates.append(
                Candidate(center=(int(x), int(y)), radius=int(radius), area=area)
            )
        return candidates

    def _select_candidate(
        self,
        candidates: list[Candidate],
        predicted_center: tuple[int, int],
    ) -> Optional[Candidate]:
        if not candidates:
            return None

        best: Optional[Candidate] = None
        best_score = float("inf")

        for candidate in candidates:
            dist = self._distance(candidate.center, predicted_center)
            if dist > self.config.max_jump_px:
                continue
            score = dist - candidate.area * 0.01
            if score < best_score:
                best_score = score
                best = candidate

        if best is None:
            best = max(candidates, key=lambda c: c.area)
        return best

    @staticmethod
    def _distance(a: tuple[int, int], b: tuple[int, int]) -> float:
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))

    def _compute_speed(self, center: tuple[int, int]) -> float:
        if self.last_center is None:
            self.last_speed = 0.0
            return 0.0
        dist = self._distance(center, self.last_center)
        speed = dist * self.fps
        alpha = self.config.speed_smoothing
        self.last_speed = alpha * speed + (1.0 - alpha) * self.last_speed
        return self.last_speed

    def _bbox_from_circle(
        self,
        center: tuple[int, int],
        radius: int,
        shape: tuple[int, int, int],
    ) -> tuple[int, int, int, int]:
        h, w = shape[:2]
        x0 = max(0, center[0] - radius)
        y0 = max(0, center[1] - radius)
        x1 = min(w - 1, center[0] + radius)
        y1 = min(h - 1, center[1] + radius)
        return x0, y0, x1, y1

    def _detect_bounce(self) -> bool:
        if len(self.trajectory) < 3:
            if self._bounce_cooldown > 0:
                self._bounce_cooldown -= 1
            return False
        p0, p1, p2 = self.trajectory[-3], self.trajectory[-2], self.trajectory[-1]
        vy_prev = (p1[1] - p0[1]) * self.fps
        vy_curr = (p2[1] - p1[1]) * self.fps
        bounced = False
        if self._bounce_cooldown == 0:
            if vy_prev > self.config.bounce_vy_min and vy_curr < -self.config.bounce_vy_min:
                bounced = True
                self._bounce_cooldown = self.config.bounce_cooldown_frames
        else:
            self._bounce_cooldown -= 1
        return bounced

    def _table_roi(self, shape: tuple[int, int, int]) -> tuple[int, int, int, int]:
        h, w = shape[:2]
        x0n, y0n, x1n, y1n = self.config.table_roi_norm
        x0 = int(x0n * w)
        y0 = int(y0n * h)
        x1 = int(x1n * w)
        y1 = int(y1n * h)
        return x0, y0, x1, y1

    def _table_quad(self, shape: tuple[int, int, int]) -> Optional[list[tuple[int, int]]]:
        if not self.config.table_quad_norm:
            return None
        h, w = shape[:2]
        quad = []
        for x_n, y_n in self.config.table_quad_norm:
            quad.append((int(x_n * w), int(y_n * h)))
        return quad

    @staticmethod
    def _in_table(
        center: tuple[int, int],
        roi: tuple[int, int, int, int],
        quad: Optional[list[tuple[int, int]]],
    ) -> bool:
        if quad:
            contour = np.array(quad, dtype=np.int32)
            return cv2.pointPolygonTest(contour, center, False) >= 0
        x0, y0, x1, y1 = roi
        return x0 <= center[0] <= x1 and y0 <= center[1] <= y1
