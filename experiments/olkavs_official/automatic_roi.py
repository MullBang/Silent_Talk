"""Frozen, causal mouth-ROI adapter for the step-03 paired evaluation.

This module accepts pixels only: no transcript, annotation, source-file or model
access. Geometry copies the current backend's 20 outer-lip landmarks and 20%
margin, while BGR output and 96 px suit the official visual model. Detection runs
on a 640 px wide RGB image. Unlike the production helper, this adapter never
back-fills from future frames: initial failure rejects the clip, 1..11 later
misses reuse the previous box on the *current* frame, and 12 reject the clip.
Create a new instance for every utterance; never share tracking across samples.
"""
from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np


LIP_INDICES = (61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
               291, 375, 321, 405, 314, 17, 84, 181, 91, 146)
MARGIN = 0.20
OUTPUT_SIZE = (96, 96)
DETECTION_WIDTH = 640
MAX_CONSECUTIVE_MISS = 12
PROFILE_ID = "mediapipe_0.10.21_lip20_margin20_width640_causal_v1"


def profile() -> dict[str, Any]:
    return {
        "profile_id": PROFILE_ID,
        "mediapipe_version": "0.10.21",
        "landmark_indices": list(LIP_INDICES),
        "margin_each_side": MARGIN,
        "detection_width": DETECTION_WIDTH,
        "detection_color": "RGB",
        "crop_color": "BGR",
        "output_wh": list(OUTPUT_SIZE),
        "interpolation": "INTER_LINEAR",
        "max_num_faces": 4,
        "face_selection": "maximum outer-lip bounding-box area",
        "static_image_mode": False,
        "refine_landmarks": False,
        "min_detection_confidence": 0.5,
        "min_tracking_confidence": 0.5,
        "reset": "new detector for each utterance",
        "initial_miss": "reject entire utterance; no future backfill",
        "later_misses_1_to_11": "last valid bbox applied to current frame",
        "consecutive_misses_12_or_more": "reject entire utterance",
        "annotation_fallback": False,
        "post_failure": "remaining frames are not processed",
    }


def bbox_with_margin(points: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
    """Get clamped xyxy pixel bounds; reject invalid/wholly off-screen boxes."""
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise ValueError("Expected at least two xy points")
    if width < 1 or height < 1 or not np.isfinite(points).all():
        raise ValueError("Invalid image size or nonfinite landmarks")
    low, high = points.min(axis=0), points.max(axis=0)
    extent = high - low
    if np.any(extent <= 0):
        raise ValueError("Degenerate lip bounds")
    # Convert extrema to Python float as in backend.roi_extractor, so rounding
    # follows exactly its floor/ceil margin definition.
    xmin, ymin, xmax, ymax = map(float, (*low, *high))
    bw, bh = xmax - xmin, ymax - ymin
    left, top = int(np.floor(xmin - bw * MARGIN)), int(np.floor(ymin - bh * MARGIN))
    right, bottom = int(np.ceil(xmax + bw * MARGIN)), int(np.ceil(ymax + bh * MARGIN))
    if right <= 0 or bottom <= 0 or left >= width or top >= height:
        raise ValueError("Lip bounds do not intersect the image")
    left, top = max(0, min(left, width - 1)), max(0, min(top, height - 1))
    right, bottom = max(left + 1, min(right, width)), max(top + 1, min(bottom, height))
    return left, top, right, bottom


def largest_lip_points(faces: Any, width: int, height: int) -> np.ndarray | None:
    """Select by normalized lip area and map coordinates to original pixels."""
    best, best_area = None, -1.0
    for face in faces or ():
        try:
            points = np.array([(face.landmark[i].x, face.landmark[i].y)
                               for i in LIP_INDICES], dtype=np.float32)
        except (AttributeError, IndexError, TypeError, ValueError):
            continue
        if not np.isfinite(points).all():
            continue
        span = points.max(axis=0) - points.min(axis=0)
        if np.any(span <= 0):
            continue
        area = float(span[0] * span[1])
        if area > best_area:
            candidate = points * np.array([width, height], dtype=np.float32)
            try:
                bbox_with_margin(candidate, width, height)
            except ValueError:
                continue
            best, best_area = candidate, area
    return best


class AutomaticLipRoi:
    """One-utterance tracker with explicit failure and per-frame telemetry.

    Inject a detector with ``process(rgb)`` and ``close()`` for unit tests only.
    An actual run verifies the frozen MediaPipe version before creating it.
    Once ``failed`` becomes true, the whole utterance must be counted as failed,
    even if earlier calls returned valid pixels. Never infer on the short prefix.
    """

    def __init__(self, detector: Any = None):
        started = time.perf_counter()
        if detector is None:
            import mediapipe as mp
            if mp.__version__ != "0.10.21":
                raise RuntimeError(f"Frozen ROI profile requires MediaPipe 0.10.21, got {mp.__version__}")
            detector = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False, max_num_faces=4, refine_landmarks=False,
                min_detection_confidence=0.5, min_tracking_confidence=0.5,
            )
        self.detector = detector
        self.initialization_seconds = time.perf_counter() - started
        self.failed = False
        self.failure_reason: str | None = None
        self.last_box_xyxy: tuple[int, int, int, int] | None = None
        self.frame_records: list[dict[str, Any]] = []
        self.consecutive_miss = 0
        self.max_consecutive_miss = 0
        self._closed = False
        self._shape: tuple[int, int] | None = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        if not self._closed:
            self.detector.close()
            self._closed = True

    def _fail(self, reason: str):
        self.failed = True
        self.failure_reason = reason

    def process(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        """Return BGR uint8 (96,96,3), or None after utterance failure."""
        if self._closed:
            raise RuntimeError("Tracker is closed")
        if (not isinstance(frame_bgr, np.ndarray) or frame_bgr.dtype != np.uint8
                or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3
                or min(frame_bgr.shape[:2]) < 1):
            raise ValueError("Expected nonempty uint8 BGR frame HxWx3")
        height, width = frame_bgr.shape[:2]
        if self._shape is None:
            self._shape = (height, width)
        elif self._shape != (height, width):
            raise ValueError("Frame size changed inside utterance")
        started = time.perf_counter()
        record: dict[str, Any] = {"frame_index": len(self.frame_records), "bbox_xyxy": None}
        if self.failed:
            record.update(status="skipped_failed_utterance", seconds=0.0)
            self.frame_records.append(record)
            return None
        scaled_height = max(1, round(height * DETECTION_WIDTH / width))
        small = cv2.resize(frame_bgr, (DETECTION_WIDTH, scaled_height), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        try:
            result = self.detector.process(rgb)
            points = largest_lip_points(getattr(result, "multi_face_landmarks", None), width, height)
        except Exception as exc:
            self._fail("detector_exception")
            record.update(status="detector_exception", error_type=type(exc).__name__,
                          error_message=str(exc), seconds=time.perf_counter() - started)
            self.frame_records.append(record)
            return None
        if points is None:
            self.consecutive_miss += 1
            self.max_consecutive_miss = max(self.max_consecutive_miss, self.consecutive_miss)
            if self.last_box_xyxy is None:
                self._fail("initial_detection_miss")
            elif self.consecutive_miss >= MAX_CONSECUTIVE_MISS:
                self._fail("consecutive_detection_misses")
            if self.failed:
                record.update(status=self.failure_reason, seconds=time.perf_counter() - started)
                self.frame_records.append(record)
                return None
            record["status"] = "previous_bbox_current_frame"
        else:
            self.consecutive_miss = 0
            self.last_box_xyxy = bbox_with_margin(points, width, height)
            record["status"] = "detected"
        left, top, right, bottom = self.last_box_xyxy
        crop = frame_bgr[top:bottom, left:right]
        roi = cv2.resize(crop, OUTPUT_SIZE, interpolation=cv2.INTER_LINEAR)
        record.update(bbox_xyxy=list(self.last_box_xyxy), seconds=time.perf_counter() - started)
        self.frame_records.append(record)
        return roi

    def stats(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for row in self.frame_records:
            counts[row["status"]] = counts.get(row["status"], 0) + 1
        detected = counts.get("detected", 0)
        filled = counts.get("previous_bbox_current_frame", 0)
        skipped = counts.get("skipped_failed_utterance", 0)
        attempted = len(self.frame_records) - skipped
        return {
            "profile_id": PROFILE_ID,
            "failed": self.failed,
            "failure_reason": self.failure_reason,
            "frames_received": len(self.frame_records),
            "frames_detection_attempted": attempted,
            "frames_detected": detected,
            "frames_previous_bbox": filled,
            "frames_skipped_after_failure": skipped,
            "detection_rate_among_attempted": detected / attempted if attempted else None,
            "max_consecutive_miss": self.max_consecutive_miss,
            "status_counts": counts,
            "initialization_seconds": self.initialization_seconds,
            "processing_seconds": sum(row["seconds"] for row in self.frame_records),
        }
