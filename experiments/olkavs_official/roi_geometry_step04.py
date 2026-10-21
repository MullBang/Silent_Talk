"""Frozen causal geometry variants for replaying step-03 detector boxes.

Only current BGR pixels and the detector's lip20+20% margin box are inputs.
No annotation, transcript, detector rerun, or future frame is used. Create a
new instance (or call reset) at each utterance boundary. EMA can introduce
lag; these are experimental variants, not a claim of improved recognition.
"""
from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np


ARMS = ("square", "context_square", "stable_context_square")
EMA_ALPHA = 0.4
OUTPUT_SIZE = (96, 96)
PROFILE_ID = "step04_square_context15_causal_ema04_v1"


def profile() -> dict[str, Any]:
    return {
        "profile_id": PROFILE_ID,
        "input_bbox": "step03 automatic lip20 plus 20% per-side margin xyxy",
        "arms": list(ARMS),
        "square_side": "max(input box width, input box height)",
        "context_square_side": "1.5 * max(input box width, input box height)",
        "stable_context_square_side": "1.5 * max(input box width, input box height)",
        "stable_ema_alpha_current": EMA_ALPHA,
        "stable_ema_variables": ["center_x", "center_y", "side"],
        "stable_ema_initialization": "first valid current box without smoothing",
        "integer_geometry": "ceil side; floor(center - integer_side / 2)",
        "boundary_policy": "shift square inside image, never truncate or pad",
        "impossible_size_policy": "reject raw or smoothed ceil side > min(image height, width)",
        "invalid_box_policy": "reject nonfinite, degenerate, or out-of-frame box",
        "output_wh": list(OUTPUT_SIZE),
        "crop_color": "BGR",
        "interpolation": "INTER_LINEAR",
        "reset": "new instance or reset at each utterance boundary",
        "annotation_fallback": False,
        "future_frames": False,
    }


class RoiGeometry:
    """Square cropping and optional causal EMA, with auditable telemetry.

    EMA state uses the floating-point requested center/side, before integer
    rounding or boundary shifting. A boundary shift therefore does not feed
    back into the next frame's smoother. Failed calls leave state unchanged;
    callers must reject a whole clip when preparation fails, not silently skip
    a bad frame.
    """

    def __init__(self, arm: str):
        if arm not in ARMS:
            raise ValueError(f"Unknown geometry arm: {arm!r}")
        self.arm = arm
        self.reset()

    def reset(self) -> None:
        self._previous: np.ndarray | None = None
        self._shape: tuple[int, int] | None = None
        self._frame_index = 0

    def process(self, frame_bgr: np.ndarray, bbox_xyxy: Any) -> tuple[np.ndarray, dict[str, Any]]:
        """Return a square 96 px crop and JSON-serializable frame telemetry."""
        if (not isinstance(frame_bgr, np.ndarray) or frame_bgr.dtype != np.uint8
                or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3
                or min(frame_bgr.shape[:2]) < 1):
            raise ValueError("Expected nonempty uint8 BGR frame HxWx3")
        height, width = frame_bgr.shape[:2]
        if self._shape is not None and self._shape != (height, width):
            raise ValueError("Frame size changed inside utterance")
        try:
            box = np.asarray(bbox_xyxy, dtype=np.float64)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError("Expected finite numeric bbox xyxy") from exc
        if box.shape != (4,) or not np.isfinite(box).all():
            raise ValueError("Expected four finite bbox xyxy coordinates")
        left, top, right, bottom = map(float, box)
        if not (0 <= left < right <= width and 0 <= top < bottom <= height):
            raise ValueError("Expected nondegenerate bbox fully inside the frame")

        factor = 1.0 if self.arm == "square" else 1.5
        raw = np.array([(left + right) / 2, (top + bottom) / 2,
                        factor * max(right - left, bottom - top)], dtype=np.float64)
        if math.ceil(float(raw[2])) > min(height, width):
            raise ValueError("Requested context square exceeds the frame's shorter side")
        smooth = raw.copy()
        smoothed = self.arm == "stable_context_square" and self._previous is not None
        if smoothed:
            smooth = EMA_ALPHA * raw + (1.0 - EMA_ALPHA) * self._previous
        side = math.ceil(float(smooth[2]))
        if side < 1 or side > min(height, width):
            raise ValueError("Smoothed square cannot fit inside the frame")
        requested_left = math.floor(float(smooth[0]) - side / 2)
        requested_top = math.floor(float(smooth[1]) - side / 2)
        final_left = max(0, min(requested_left, width - side))
        final_top = max(0, min(requested_top, height - side))
        requested = [requested_left, requested_top,
                     requested_left + side, requested_top + side]
        final = [final_left, final_top, final_left + side, final_top + side]
        crop = cv2.resize(frame_bgr[final_top:final_top + side, final_left:final_left + side],
                          OUTPUT_SIZE, interpolation=cv2.INTER_LINEAR)
        telemetry = {
            "profile_id": PROFILE_ID,
            "arm": self.arm,
            "frame_index": self._frame_index,
            "raw_bbox_xyxy": box.tolist(),
            "context_factor": factor,
            "pre_smoothing_center_xy": raw[:2].tolist(),
            "pre_smoothing_side": float(raw[2]),
            "post_smoothing_center_xy": smooth[:2].tolist(),
            "post_smoothing_side": float(smooth[2]),
            "smoothing_applied": smoothed,
            "ema_alpha_current": EMA_ALPHA if self.arm == "stable_context_square" else None,
            "requested_square_xyxy": requested,
            "final_square_xyxy": final,
            "integer_side": side,
            "boundary_shifted": requested != final,
            "boundary_shift_xy": [final_left - requested_left, final_top - requested_top],
        }
        self._previous = smooth.copy()
        self._shape = (height, width)
        self._frame_index += 1
        return crop, telemetry
