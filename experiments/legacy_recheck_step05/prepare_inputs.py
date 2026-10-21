"""Rebuild only validation20 with the legacy input contract and frozen time offsets.

The original production preparation function is called unchanged. The original
timing must reproduce every stored uint8 pixel before aligned inputs can be used.
No model, test payload, original audio or training code is executed here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "backend"))
from prepare_data import _extract_lip_boxes, _roi_seq_from_boxes  # noqa: E402
from config import FPS, ROI_SIZE, ROI_MARGIN, NORMALIZE_MEAN, NORMALIZE_STD  # noqa: E402
from models.cache import load_clip  # noqa: E402
from models.tokenizer import encode_text, normalize_text  # noqa: E402

SOURCE_PLAN = ROOT / "experiments/olkavs_official/step03_source_plan.json"
SOURCE_PLAN_SHA = "652692aaaf9778bfa51789de0e177797f9c769e7775df0d112676e0cafaab6f7"
ALIGNMENT = ROOT / "experiments/olkavs_official/results/step04_audio_offset_candidates.json"
ALIGNMENT_SHA = "eae3f1d9083b474441c4bc5e210edefb53aaa22cbfaba152e2eb31e0aa93ccb0"
PARENT_MANIFEST = ROOT / "data_cache/datasets/aihub_timed_v4/manifest.json"
PARENT_MANIFEST_SHA = "214ce935150b367a15f24f0896901d34c54c6d4fe73d493a9d397c4c94fe59a6"
ARMS = ("old_timing", "aligned_timing")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def anchor(path: Path, expected: str | None = None) -> dict:
    actual = sha256(path)
    if expected is not None and actual != expected:
        raise ValueError(f"Frozen file changed: {path}")
    return {"path": str(path.resolve()), "sha256": actual}


def save_new_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def offsets(alignment: dict, source_names: set[str]) -> dict[str, float]:
    if alignment.get("status") != "validated":
        raise ValueError("Independent validated alignment required")
    result = {}
    for row in alignment["sources"]:
        name = row["source_name"]
        offset = float(row["label_to_video_offset_seconds"])
        if name in result or name not in source_names or not math.isfinite(offset):
            raise ValueError("Missing, duplicate, foreign or nonfinite offset")
        result[name] = offset
    if set(result) != source_names:
        raise ValueError("Every fixed source requires a frozen offset")
    return result


def target_selection(start: float, end: float, fps: float) -> tuple[list[float], list[int]]:
    # Exact Python rounding and expression order of the original helper.
    if not all(math.isfinite(x) for x in (start, end, fps)) or start < 0 or end <= start or fps <= 0:
        raise ValueError("Invalid source time interval")
    times = [start + k / FPS for k in range(int(round((end - start) * FPS)))]
    return times, [int(round(t * fps)) for t in times]


class TrackedCapture:
    """Transparent wrapper; source pixels/PTS are recorded when helper selects them."""
    def __init__(self, capture):
        self.capture = capture
        self.last_frame = None
        self.frames_read = 0
        self.selected = []
        self.source_digest = hashlib.sha256()

    def read(self):
        ok, frame = self.capture.read()
        if ok:
            self.last_frame = frame
            self.frames_read += 1
        return ok, frame

    def set(self, prop, value):
        return self.capture.set(prop, value)

    def get(self, prop):
        value = self.capture.get(prop)
        if prop == cv2.CAP_PROP_POS_MSEC:
            index = int(round(self.capture.get(cv2.CAP_PROP_POS_FRAMES))) - 1
            if self.last_frame is None:
                raise ValueError("No decoded frame at selection")
            frame_bytes = self.last_frame.tobytes(order="C")
            self.selected.append({"source_frame_index": index, "actual_pts_seconds": float(value) / 1000,
                                  "source_bgr_pixel_sha256": hashlib.sha256(frame_bytes).hexdigest()})
            self.source_digest.update(np.asarray([index], dtype="<i8").tobytes())
            self.source_digest.update(frame_bytes)
        return value


def extract(capture, start: float, end: float, boxes: list, label_fps: float) -> tuple[np.ndarray, dict]:
    tracking = TrackedCapture(capture)
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    times, indices = target_selection(start, end, fps)
    pixels = _roi_seq_from_boxes(tracking, start, end, boxes, label_fps)
    if pixels is None or pixels.dtype != np.uint8 or pixels.shape != (len(indices), 96, 96, 3):
        raise ValueError("Original helper failed or violated shape")
    if [r["source_frame_index"] for r in tracking.selected] != indices:
        raise ValueError("Recorded actual source indices differ from fixed target selection")
    for t, row in zip(times, tracking.selected):
        row["target_video_seconds"] = t
        row["annotation_box_index"] = row["source_frame_index"]
        row["annotation_box_xyxy"] = boxes[row["source_frame_index"]]
    trace = {"native_fps": fps, "target_fps": FPS, "source_frames_decoded": tracking.frames_read,
             "selected_count": len(indices), "selected_source_pixels_sha256": tracking.source_digest.hexdigest(),
             "selection": "round((video_start + k / 25) * actual_fps), k < round(duration * 25)",
             "selected": tracking.selected}
    return pixels, trace


def require_exact_reproduction(pixels: np.ndarray, cached: dict, reference: str, source_text: str) -> None:
    if cached["text"] != reference or cached["source_text"] != source_text:
        raise ValueError("Legacy cached transcript differs from fixed reference")
    if cached["label"].tolist() != encode_text(reference):
        raise ValueError("Legacy cached tokens differ from fixed transcript")
    if pixels.dtype != cached["frames"].dtype or pixels.shape != cached["frames"].shape or not np.array_equal(pixels, cached["frames"]):
        raise ValueError("Old-timing pixels do not exactly reproduce legacy cache; do not relax this gate")


def build_plan(destination: Path) -> dict:
    for path, expected in ((SOURCE_PLAN, SOURCE_PLAN_SHA), (ALIGNMENT, ALIGNMENT_SHA), (PARENT_MANIFEST, PARENT_MANIFEST_SHA)):
        anchor(path, expected)
    source = json.loads(SOURCE_PLAN.read_text(encoding="utf-8"))
    alignment = json.loads(ALIGNMENT.read_text(encoding="utf-8"))
    rows = source["records"]
    if len(rows) != 20 or len({r["id"] for r in rows}) != 20 or any(r["split"] != "val" for r in rows):
        raise ValueError("Exactly original validation20 required")
    source_offsets = offsets(alignment, {Path(r["source_video"]["path"]).name for r in rows})
    for evidence in alignment["evidence_files"]:
        anchor(Path(evidence["path"]), evidence["sha256"])
    code_files = [Path(__file__), ROOT / "scripts/prepare_data.py", ROOT / "backend/preprocessing/roi_extractor.py",
                  ROOT / "backend/models/tokenizer.py", ROOT / "backend/models/cache.py", ROOT / "backend/config.py"]
    plan = {"schema": "legacy-step05-preparation-plan-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
            "frozen_before_extraction_and_inference": True,
            "source_plan": anchor(SOURCE_PLAN), "alignment_sidecar": anchor(ALIGNMENT),
            "parent_manifest": anchor(PARENT_MANIFEST), "code": [anchor(p) for p in code_files],
            "environment": {"python": sys.version, "opencv": cv2.__version__, "numpy": np.__version__},
            "source_ids": [r["id"] for r in rows], "arms": list(ARMS), "records": rows,
            "sources": source["sources"], "source_offsets_seconds": source_offsets,
            "contract": {"helper": "scripts.prepare_data._roi_seq_from_boxes unchanged call", "target_fps": FPS,
              "frames": "round((video_end-video_start)*25)", "selection": "round((video_start+k/25)*actual_fps)",
              "crop": "existing video-indexed annotation bbox yxyx converted xyxy; per-side width/height margin0.20",
              "color": "RGB", "size": list(ROI_SIZE), "margin": ROI_MARGIN, "resize": "cv2.INTER_LINEAR",
              "storage": "lossless uint8 NPZ, no intermediate recompressed MP4", "inference_mean": list(NORMALIZE_MEAN),
              "inference_std": list(NORMALIZE_STD), "inference_scale": "(pixel/255-mean)/std float32",
              "batch1_padding": "none; variable full utterance length"},
            "reproduction_gate": "All20 old-timing re-extractions must be pixel-exact against existing NPZ, label/text unchanged; fail closed",
            "output_directory": str((BASE / "inputs").resolve()), "reads": "Only four specified existing MP4s and JSONs; selected intervals; cloud may cache whole files",
            "audio_read": False, "test_payloads_read": False, "training": False, "original_files_modified": False}
    if (FPS, ROI_SIZE, ROI_MARGIN) != (25, (96, 96), .20):
        raise ValueError("Legacy spatial/temporal contract changed")
    save_new_json(destination, plan)
    return plan


def prepare(plan_path: Path, plan_sha256: str) -> dict:
    anchor(plan_path, plan_sha256)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    for item in [plan["source_plan"], plan["alignment_sidecar"], plan["parent_manifest"], *plan["code"]]:
        anchor(Path(item["path"]), item["sha256"])
    if (plan["environment"]["opencv"], plan["environment"]["numpy"]) != (cv2.__version__, np.__version__):
        raise ValueError("Frozen preparation environment changed")
    output = Path(plan["output_directory"])
    if output.resolve() != BASE / "inputs" or output.exists():
        raise ValueError("Use only new isolated inputs directory; no overwrite")
    output.mkdir()
    (output / "traces").mkdir()
    (output / "aligned_timing").mkdir()
    all_records = []
    started = time.perf_counter()
    control_count = 0
    groups = {}
    for row in plan["records"]:
        groups.setdefault(row["source_video"]["path"], []).append(row)
    try:
        for source_path, rows in groups.items():
            source = Path(source_path)
            before = source.stat()
            expected = rows[0]["source_video"]
            if (before.st_size, before.st_mtime_ns) != (expected["size_bytes"], expected["mtime_ns"]):
                raise ValueError("Source metadata changed")
            label_path = Path(rows[0]["label_path"])
            anchor(label_path, rows[0]["label_sha256"])
            label = json.loads(label_path.read_text(encoding="utf-8-sig"))
            if isinstance(label, list):
                label = label[0]
            boxes, label_fps = _extract_lip_boxes(label, "yxyx")
            if boxes is None:
                raise ValueError("Required annotation boxes missing; no automatic fallback")
            sentences = {s["ID"]: s for s in label["Sentence_info"]}
            capture = cv2.VideoCapture(str(source))
            try:
                if not capture.isOpened():
                    raise ValueError("Source cannot open")
                native_fps = float(capture.get(cv2.CAP_PROP_FPS))
                for row in rows:
                    sentence = sentences[row["sentence_id"]]
                    if (sentence["sentence_text"] != row["source_text"] or normalize_text(sentence["sentence_text"]) != row["text"]
                        or sentence["start_time"] != row["start_time"] or sentence["end_time"] != row["end_time"]):
                        raise ValueError("Original selected transcript or label times changed")
                    if not math.isclose(native_fps, row["video_fps"], rel_tol=1e-12):
                        raise ValueError("Source native FPS changed")
                    cached_path = Path(row["legacy_cache_absolute_path"])
                    anchor(cached_path, row["cache_sha256"])
                    cached = load_clip(cached_path)
                    with np.load(cached_path, allow_pickle=False) as old_npz:
                        metadata = {k: old_npz[k].copy() for k in old_npz.files if k != "frames"}
                    for arm in ARMS:
                        offset = 0.0 if arm == "old_timing" else plan["source_offsets_seconds"][source.name]
                        start, end = row["start_time"] + offset, row["end_time"] + offset
                        clip_started = time.perf_counter()
                        pixels, trace = extract(capture, start, end, boxes, label_fps)
                        if len(pixels) != row["frames"]:
                            raise ValueError("Aligned duration rounding changed frame count")
                        if arm == "old_timing":
                            require_exact_reproduction(pixels, cached, row["text"], row["source_text"])
                            npz_path = cached_path
                            control_count += 1
                        else:
                            npz_path = output / arm / (row["id"] + ".npz")
                            with npz_path.open("xb") as stream:
                                np.savez_compressed(stream, frames=pixels, **metadata,
                                    label_to_video_offset_seconds=offset, original_label_start_time=row["start_time"],
                                    original_label_end_time=row["end_time"], aligned_video_start_time=start, aligned_video_end_time=end,
                                    alignment_sidecar_sha256=ALIGNMENT_SHA)
                            verified = load_clip(npz_path)
                            if not np.array_equal(verified["frames"], pixels) or verified["text"] != row["text"]:
                                raise ValueError("Saved aligned payload failed roundtrip validation")
                        trace.update({"id": row["id"], "arm": arm, "label_to_video_offset_seconds": offset,
                                      "video_start_time": start, "video_end_time": end})
                        trace_path = output / "traces" / f"{row['id']}_{arm}.json"
                        save_new_json(trace_path, trace)
                        all_records.append({"id": row["id"], "speaker_id": row["speaker_id"], "reference": row["text"],
                            "source_text": row["source_text"], "arm": arm, "npz_path": str(npz_path.resolve()),
                            "npz_sha256": sha256(npz_path), "frame_count": len(pixels),
                            "pixel_sha256": hashlib.sha256(pixels.tobytes(order="C")).hexdigest(),
                            "label_tokens_sha256": hashlib.sha256(cached["label"].astype("<i8").tobytes()).hexdigest(),
                            "source_video": expected, "label_path": str(label_path), "original_label_sha256": row["label_sha256"],
                            "label_to_video_offset_seconds": offset, "original_start_time": row["start_time"],
                            "original_end_time": row["end_time"], "video_start_time": start, "video_end_time": end,
                            "native_fps": native_fps, "target_fps": FPS, "selected_trace_path": str(trace_path),
                            "selected_trace_sha256": sha256(trace_path), "selected_source_pixels_sha256": trace["selected_source_pixels_sha256"],
                            "preparation_seconds": time.perf_counter() - clip_started,
                            "old_timing_reproduction_exact": True if arm == "old_timing" else None})
                        print(f"[ok] {row['id']} {arm} T={len(pixels)}", flush=True)
            finally:
                capture.release()
            after = source.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError("Source changed during extraction")
            anchor(label_path, rows[0]["label_sha256"])
        if control_count != 20 or len(all_records) != 40:
            raise ValueError("Full40 input cohort and20 exact controls required")
        result = {"schema": "legacy-step05-input-manifest-v1", "status": "completed",
            "created_utc": datetime.now(timezone.utc).isoformat(), "preparation_plan": anchor(plan_path),
            "source_plan": plan["source_plan"], "alignment_sidecar": plan["alignment_sidecar"],
            "environment": plan["environment"], "contract": plan["contract"],
            "reproduction_gate": {"passed": True, "records": 20, "pixel_exact": 20, "label_exact": 20, "text_exact": 20},
            "record_count": 40, "ids": 20, "arms": list(ARMS), "records": all_records,
            "preparation_seconds": time.perf_counter() - started,
            "test_payloads_read": False, "original_audio_read": False, "training": False,
            "original_files_modified": False, "new_aihub_download": False}
        for row in all_records:
            anchor(Path(row["npz_path"]), row["npz_sha256"])
        anchor(PARENT_MANIFEST, PARENT_MANIFEST_SHA)
        save_new_json(output / "manifest.json", result)
        return result
    except Exception as exc:
        save_new_json(output / "failed_preparation.json", {"status": "failed", "error": str(exc),
            "completed_records": all_records, "passed_controls": control_count, "elapsed_seconds": time.perf_counter() - started})
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-plan", action="store_true")
    parser.add_argument("--plan", type=Path, default=BASE / "preparation_plan.json")
    parser.add_argument("--plan-sha256")
    args = parser.parse_args()
    if args.build_plan:
        build_plan(args.plan)
        print(json.dumps(anchor(args.plan)))
    else:
        if not args.plan_sha256:
            parser.error("--plan-sha256 required for extraction")
        prepare(args.plan, args.plan_sha256)
        print(json.dumps(anchor(BASE / "inputs/manifest.json")))


if __name__ == "__main__":
    main()
