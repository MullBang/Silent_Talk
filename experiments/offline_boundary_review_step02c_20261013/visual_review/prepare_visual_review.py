"""Tile decoded, existing review frames for a non-approving boundary inspection.

Run from the repository root with .venv-tokenizer/Scripts/python.exe. The
existing review MP4s are read, never rewritten; no raw TAR or model is opened.
"""
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
STEP2 = ROOT / "experiments/offline_alignment_step02_20261012"
PARENT = ROOT / "experiments/offline_boundary_review_step02b_20261013/review_manifest_r1.json"
MEDIA = STEP2 / "results/media_manifest.json"


def sha(path: Path) -> str:
    """Return the SHA-256 of a small local input or output."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise RuntimeError(f"Could not hash {path}") from exc


def read_json(path: Path) -> dict[str, Any]:
    """Read one UTF-8 provenance record."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Could not parse {path}") from exc


def decode(spec: dict[str, Any]) -> list[Any]:
    """Fully decode a hash-verified 25 fps preview, retaining exact decoded pixels."""
    cap = None
    try:
        path = Path(spec["path"])
        if sha(path) != spec["sha256"]:
            raise ValueError(f"Preview hash mismatch: {path}")
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened() or abs(cap.get(cv2.CAP_PROP_FPS) - 25) > 1e-8:
            raise ValueError(f"Unexpected preview stream: {path}")
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape != (spec["height"], spec["width"], 3):
                raise ValueError("Unexpected decoded frame shape")
            frames.append(frame)
        if len(frames) != spec["frames"]:
            raise ValueError(f"Decoded count mismatch: {path}")
        return frames
    except Exception as exc:
        raise RuntimeError("Preview decode failed") from exc
    finally:
        if cap is not None:
            cap.release()


def font(size: int) -> ImageFont.FreeTypeFont:
    """Load the installed Korean UI font for labels only."""
    try:
        return ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", size)
    except OSError as exc:
        raise RuntimeError("Korean label font unavailable") from exc


def tile(record: dict[str, Any], frames: list[Any], samples: list[dict[str, Any]],
         full: bool) -> dict[str, Any]:
    """Paste decoded pixels unchanged and add labels outside image rectangles."""
    try:
        selected = [s for s in samples if s["label"] in ("start", "mid", "end")] if full else samples
        columns, cell_w, cell_h = (3, 640, 418) if full else (5, 206, 164)
        canvas = Image.new("RGB", (columns * cell_w, 70 + math.ceil(len(selected) / columns) * cell_h), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 4), record["id"] + " | " + record["text"], font=font(17), fill="black")
        draw.text((10, 30), "기존 25fps 검수 영상 표본 / 학습 입력·입술 경계 승인 아님", font=font(16), fill="black")
        for i, sample in enumerate(selected):
            x, y = (i % columns) * cell_w, 70 + (i // columns) * cell_h
            frame = frames[sample["review_frame_index"]]
            # BGR->RGB is a lossless channel reorder; no resize/filter/retouch.
            rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            canvas.paste(rgb, (x + (0 if full else 55), y))
            h = frame.shape[0]
            draw.text((x + 4, y + h + 1), f'{sample["label"]} | req {sample["requested_context_time_seconds"]:.4f}s', font=font(13), fill="black")
            draw.text((x + 4, y + h + 20), f'preview #{sample["review_frame_index"]} {sample["review_frame_time_seconds"]:.3f}s', font=font(13), fill="black")
            draw.text((x + 4, y + h + 38), f'native #{sample["source_trace"]["native_frame_index"]}', font=font(13), fill="black")
        path = OUT / (record["id"] + ("_full.png" if full else "_roi.png"))
        canvas.save(path)
        return {"path": str(path), "sha256": sha(path), "sample_count": len(selected), "pixel_resize": False}
    except Exception as exc:
        raise RuntimeError("Could not tile visual diagnostic") from exc


def main() -> None:
    """Verify source bindings and write traceable diagnostic contact sheets."""
    try:
        parent = read_json(PARENT)
        media = read_json(MEDIA)
        if sha(MEDIA) != parent["sources"][str(MEDIA)]:
            raise ValueError("Frozen media manifest changed")
        rows = {r["id"]: r for r in parent["records"]}
        media_rows = {r["id"]: r for r in media["records"]}
        sources = {str(PARENT): sha(PARENT), str(MEDIA): sha(MEDIA), str(Path(__file__).resolve()): sha(Path(__file__).resolve())}
        results = []
        targets = sorted([(r["id"], s["segment_id"], s["recommendation_rank"])
            for r in parent["records"] for s in r["segments"] if s["recommended"]], key=lambda x:x[2])
        if len(targets) != 8:
            raise ValueError("Expected frozen eight recommended phrase candidates")
        for rid, sid, rank in targets:
            row, src = rows[rid], media_rows[rid]
            segment = next(s for s in row["segments"] if s["segment_id"] == sid)
            if rank <= 3 and segment.get("external_audio_evidence", {}).get("decision") != "acceptable":
                raise ValueError("Expected actual, narrow audio-only evidence for first three")
            record_path = STEP2 / "media" / rid / "record.json"
            if read_json(record_path) != src:
                raise ValueError("Record differs from frozen media manifest")
            sources[str(record_path)] = sha(record_path)
            video = src["video"]
            full, roi = decode(video["full_preview"]), decode(video["roi_preview"])
            for name in ("full_preview", "roi_preview"):
                spec = video[name]
                if spec["sha256"] != parent["sources"][spec["path"]]:
                    raise ValueError("Parent preview binding differs")
                sources[spec["path"]] = spec["sha256"]
            if len(full) != len(roi) or len(roi) != len(video["trace"]):
                raise ValueError("Preview/trace count mismatch")
            start, end = segment["start_sample_16k"] / 16000, segment["end_sample_16k"] / 16000
            requests = [(f"start{d:+.2f}" if d else "start", start + d) for d in (-.08, -.04, 0, .04, .08)]
            requests += [("mid", (start + end) / 2)]
            requests += [(f"end{d:+.2f}" if d else "end", end + d) for d in (-.08, -.04, 0, .04, .08)]
            if sid.endswith("end_plus120ms"):
                requests += [("old_end", end - .12)]
            samples = []
            for label, time_s in requests:
                index = round(time_s * 25)
                if not 0 <= index < len(roi):
                    raise ValueError("Requested preview time outside recorded context")
                trace = video["trace"][index]
                target = video["video_clock_start"] + index / 25
                if abs(target - trace["target_video_time"]) > 1e-8:
                    raise ValueError("Trace target timing mismatch")
                if round(target * video["native_fps"]) != trace["native_frame_index"]:
                    raise ValueError("Recorded native frame index inconsistent")
                requested_native_time = src["context_audio"]["actual_source_start_seconds"] + time_s + src["offset_seconds"]
                samples.append({"label": label, "requested_context_time_seconds": time_s,
                    "review_frame_index": index, "review_frame_time_seconds": index / 25,
                    "review_time_quantization_seconds": index / 25 - time_s,
                    "requested_native_video_time_seconds": requested_native_time,
                    "actual_native_minus_requested_seconds": trace["actual_pts_seconds"] - requested_native_time,
                    "source_trace": trace,
                    "decoded_roi_bgr_sha256": hashlib.sha256(roi[index].tobytes()).hexdigest(),
                    "decoded_full_bgr_sha256": hashlib.sha256(full[index].tobytes()).hexdigest()})
            result = {"id": rid, "segment_id": sid, "text": segment["text"],
                "parent_snapshot_audio_evidence_only": segment.get("external_audio_evidence"),
                "recommendation_rank": rank, "start_sample_16k": segment["start_sample_16k"],
                "end_sample_16k": segment["end_sample_16k"], "decoded_each_preview_frames": len(roi),
                "source_video_member_reference_only": src["source_members"]["video"],
                "source_member_reread_this_run": False, "samples": samples}
            result["contact_sheets"] = [tile(result, roi, samples, False), tile(result, full, samples, True)]
            results.append(result)
        output = {"schema": "offline-visual-contact-diagnostic-v1", "status": "prepared_for_agent_visual_observation",
            "sources": sources, "records": results, "record_count": len(results),
            "sample_count": sum(len(r["samples"]) for r in results),
            "decoded_frames_total": 2 * sum(r["decoded_each_preview_frames"] for r in results),
            "selection_rule": "nearest existing review index round(context_seconds * 25); original frozen native trace retained",
            "limitations": ["Existing MP4 previews were resampled and recompressed before this task.",
                "Native trace is a frozen source reference, not a fresh raw-video verification.",
                "Nearest review sampling adds up to 20ms to the prior nearest-native-frame approximation.",
                "Contact sheets cannot establish phoneme boundaries or human approval.",
                "Source raw pixel hashes and decoded recompressed pixel hashes are distinct by design."],
            "human_visual_approvals": 0, "training_ready": False, "model_calls": 0,
            "training_updates": 0, "raw_tar_reads": 0, "raw_downloads": 0,
            "diagnostic_images_are_training_inputs": False}
        (OUT / "visual_contact_manifest.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({k: output[k] for k in ("status", "record_count", "sample_count", "decoded_frames_total")}, ensure_ascii=False))
    except Exception as exc:
        raise RuntimeError("Visual diagnostic preparation failed; no approval may be inferred") from exc


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
