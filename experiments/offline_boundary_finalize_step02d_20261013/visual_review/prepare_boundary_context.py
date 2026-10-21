"""Inspect all existing 25fps review frames within +/-0.20s of each boundary."""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
PARENT = ROOT / "experiments/offline_boundary_review_step02c_20261013/review_manifest_r1.json"
MEDIA = ROOT / "experiments/offline_alignment_step02_20261012/results/media_manifest.json"
LATEST = ROOT / "experiments/offline_boundary_review_step02c_20261013/visual_review/latest_eight_visual_manifest.json"


def read_json(path: Path) -> dict[str, Any]:
    """Read source metadata without modifying it."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Cannot read {path}") from exc


def sha(path: Path) -> str:
    """Hash a local source or diagnostic artifact."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise RuntimeError(f"Cannot hash {path}") from exc


def decode(spec: dict[str, Any]) -> list[Any]:
    """Hash-check and fully decode the existing ROI preview only."""
    cap = None
    try:
        path = Path(spec["path"])
        if sha(path) != spec["sha256"]:
            raise ValueError("Preview hash differs")
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened() or abs(cap.get(cv2.CAP_PROP_FPS) - 25) > 1e-8:
            raise ValueError("Preview FPS/open failed")
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape != (96, 96, 3):
                raise ValueError("ROI dimensions differ")
            frames.append(frame)
        if len(frames) != spec["frames"]:
            raise ValueError("Preview count differs")
        return frames
    except Exception as exc:
        raise RuntimeError("Read-only ROI decode failed") from exc
    finally:
        if cap is not None:
            cap.release()


def make_sheet(result: dict[str, Any], frames: list[Any]) -> dict[str, Any]:
    """Tile source-sized pixels chronologically, two boundary rows per sheet."""
    try:
        f14 = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 14)
        f12 = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 12)
        canvas = Image.new("RGB", (1320, 500), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 4), result["id"] + " | " + result["text"], font=f14, fill="black")
        draw.text((10, 28), "시작/끝 ±0.20초 범위의 모든 25fps 검수 프레임 / 시간순 / 음절경계 승인 아님", font=f14, fill="black")
        for row_idx, window in enumerate(result["windows"]):
            top = 64 + row_idx * 214
            draw.text((10, top), f'{window["name"]}: {window["boundary_context_time_seconds"]:.6f}s | 앞/뒤 프레임의 모양을 비교', font=f14, fill="black")
            for column, sample in enumerate(window["samples"]):
                x, y = column * 120 + 12, top + 28
                image = Image.fromarray(cv2.cvtColor(frames[sample["review_frame_index"]], cv2.COLOR_BGR2RGB))
                canvas.paste(image, (x, y))
                draw.text((x, y + 98), f'#{sample["review_frame_index"]} {sample["review_frame_time_seconds"]:.3f}s', font=f12, fill="black")
                draw.text((x, y + 115), f'Δ {sample["seconds_from_boundary"]:+.3f}s', font=f12, fill="black")
                draw.text((x, y + 132), f'native {sample["source_trace"]["native_frame_index"]}', font=f12, fill="black")
        path = OUT / (result["id"] + "_boundary_context.png")
        canvas.save(path)
        return {"path": str(path), "sha256": sha(path), "pixel_resize": False,
            "pixel_panels": sum(len(w["samples"]) for w in result["windows"])}
    except Exception as exc:
        raise RuntimeError("Could not create boundary sheet") from exc


def main() -> None:
    """Bind current targets and native trace, preserving all existing artifacts."""
    try:
        parent, media, latest = read_json(PARENT), read_json(MEDIA), read_json(LATEST)
        sources = {str(p): sha(p) for p in (PARENT, MEDIA, LATEST, Path(__file__).resolve())}
        media_rows = {r["id"]: r for r in media["records"]}
        current = sorted([(r, s) for r in parent["records"] for s in r["segments"] if s["recommended"]],
            key=lambda pair: pair[1]["recommendation_rank"])
        if len(current) != 8:
            raise ValueError("Expected current eight phrases")
        result_rows = []
        for (record, segment), prior in zip(current, latest["records"]):
            if record["id"] != prior["id"] or any(segment[k] != prior[k] for k in ("segment_id", "text", "start_sample_16k", "end_sample_16k")):
                raise ValueError("Latest visual and current parent bindings differ")
            src = media_rows[record["id"]]
            video = src["video"]
            spec = video["roi_preview"]
            sources[spec["path"]] = spec["sha256"]
            frames = decode(spec)
            if len(frames) != len(video["trace"]):
                raise ValueError("Video/trace count differs")
            start = segment["start_sample_16k"] / 16000
            end = segment["end_sample_16k"] / 16000
            result = {"id": record["id"], "segment_id": segment["segment_id"], "text": segment["text"],
                "rank": segment["recommendation_rank"], "start_sample_16k": segment["start_sample_16k"],
                "end_sample_16k": segment["end_sample_16k"], "preview_frames": len(frames),
                "source_roi_path": spec["path"], "source_roi_sha256": spec["sha256"],
                "source_raw_member_reference_only": src["source_members"]["video"],
                "available_existing_context": {"first_review_frame_time_seconds": 0,
                    "last_review_frame_time_seconds": (len(frames) - 1) / 25,
                    "video_end_exclusive_seconds": len(frames) / 25,
                    "audio_end_seconds": src["context_audio"]["duration_seconds"],
                    "time_before_target_start_seconds": start,
                    "time_after_target_end_to_last_review_frame_seconds": (len(frames) - 1) / 25 - end,
                    "time_after_target_end_to_audio_end_seconds": src["context_audio"]["duration_seconds"] - end,
                    "context_neighbor_intersections": src["context_neighbor_intersections"],
                    "extra_context_is_not_target_transcript_or_training_padding": True}, "windows": []}
            for name, boundary in (("start", start), ("end", end)):
                first = math.ceil((boundary - .20) * 25 - 1e-9)
                last = math.floor((boundary + .20) * 25 + 1e-9)
                if first < 0 or last >= len(frames):
                    raise ValueError("Requested full 0.2s context is unavailable")
                samples = []
                for index in range(first, last + 1):
                    trace = video["trace"][index]
                    time_s = index / 25
                    target = video["video_clock_start"] + time_s
                    if abs(target - trace["target_video_time"]) > 1e-8 or round(target * video["native_fps"]) != trace["native_frame_index"]:
                        raise ValueError("Frozen native trace mapping differs")
                    if abs(trace["actual_pts_seconds"] - target) > .5 / video["native_fps"] + 1e-8:
                        raise ValueError("Unexpected native time quantization")
                    samples.append({"review_frame_index": index, "review_frame_time_seconds": time_s,
                        "seconds_from_boundary": time_s - boundary, "source_trace": trace,
                        "decoded_roi_bgr_sha256": hashlib.sha256(frames[index].tobytes()).hexdigest()})
                result["windows"].append({"name": name, "boundary_context_time_seconds": boundary,
                    "requested_window_seconds": [boundary - .20, boundary + .20],
                    "all_existing_25fps_indices_in_closed_window": [first, last], "samples": samples})
            result["contact_sheet"] = make_sheet(result, frames)
            result_rows.append(result)
        output = {"schema": "offline-boundary-context-diagnostic-v1", "status": "prepared_for_agent_still_sequence_observation",
            "sources": sources, "records": result_rows, "roi_videos_decoded": len(result_rows),
            "decoded_frames": sum(r["preview_frames"] for r in result_rows),
            "boundary_windows": sum(len(r["windows"]) for r in result_rows),
            "pixel_panels": sum(r["contact_sheet"]["pixel_panels"] for r in result_rows),
            "sampling_rule": "All review-grid frames in each closed [boundary-0.20,boundary+0.20]s window, no inserted/interpolated frames.",
            "limitations": ["Existing resampled and recompressed review video, not fresh raw pixels.",
                "Frozen native trace only; raw-video members not reread.", "Static consecutive frames cannot certify a phonetic boundary.",
                "Visible changes in mouth shape are not automatically speech onset or offset."],
            "human_visual_approvals_created": 0, "training_ready": False, "raw_tar_reads": 0,
            "new_neural_calls": 0, "training_updates": 0}
        (OUT / "boundary_context_manifest.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({k:output[k] for k in ("roi_videos_decoded", "decoded_frames", "boundary_windows", "pixel_panels")}))
    except Exception as exc:
        raise RuntimeError("Boundary context preparation failed") from exc


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
