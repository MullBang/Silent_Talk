"""Create stage18 ROI and source-frame review sheets from staged local files.

Only the separate readiness manifest's 120 caches, local labels and staged
videos are opened. No general manifest loader, external source path, model
forward, training or test payload is used. Rendering does not approve quality;
qa_inventory.json remains pending human visual review.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "backend"), str(ROOT / "scripts")]
from config import FPS, ROI_MARGIN
from models.cache import load_clip
from models.tokenizer import ctc_min_input_len
from prepare_data import _extract_lip_boxes, _video_timing_metadata
from preprocessing.roi_extractor import crop_roi_from_box


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path: Path):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            require(key not in value, f"Duplicate JSON key: {key}")
            value[key] = item
        return value
    with path.open(encoding="utf-8") as stream:
        return json.load(stream, object_pairs_hook=unique)


def local_file(base: Path, value: str) -> Path:
    """Reject absolute/relative references outside the explicitly allowed area."""
    require(isinstance(value, str) and bool(value), "Missing local file path")
    candidate = Path(value)
    path = (candidate if candidate.is_absolute() else base / candidate).resolve()
    require(path.is_relative_to(base.resolve()) and path.is_file(), f"Not an existing allowed local file: {path}")
    return path


def font(size):
    for name in ("C:/Windows/Fonts/consola.ttf", "DejaVuSansMono.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def clip_statistics(row: dict, clip: dict) -> dict:
    frames = clip["frames"]
    pixels = frames.astype(np.float32)
    gray = pixels.mean(axis=3)
    spatial_variance = gray.var(axis=(1, 2))
    value = {
        "id": row["id"], "split": row["split"], "recording_id": row["recording_id"],
        "frames": len(frames), "cache_sha256": row["cache_sha256"],
        "minimum_frame_spatial_variance": float(spatial_variance.min()),
        "minimum_frame_spatial_std": float(np.sqrt(spatial_variance).min()),
        "mean_absolute_frame_change": float(np.abs(np.diff(pixels, axis=0)).mean()) if len(frames) > 1 else 0.,
        "near_black_fraction": float((frames <= 3).all(axis=3).mean()),
        "near_white_fraction": float((frames >= 252).all(axis=3).mean()),
        "ctc_slack_frames": int(len(frames) - ctc_min_input_len(clip["label"])),
        "review_flags": [],
    }
    if value["minimum_frame_spatial_std"] < 3:
        value["review_flags"].append("low_spatial_variance")
    if value["mean_absolute_frame_change"] < .1:
        value["review_flags"].append("almost_static")
    if max(value["near_black_fraction"], value["near_white_fraction"]) > .5:
        value["review_flags"].append("mostly_black_or_white")
    return value


def roi_sheet(items: list[dict], first_number: int, total: int) -> Image.Image:
    """Each item has row metadata, three RGB thumbnails and their clip indices."""
    require(0 < len(items) <= 40, "ROI sheet requires1..40 clips")
    columns, cell_w, cell_h = 5, 302, 152
    count_rows = math.ceil(len(items) / columns)
    image = Image.new("RGB", (columns * cell_w + 16, 68 + count_rows * cell_h), "#f4f6f9")
    draw = ImageDraw.Draw(image)
    draw.text((12, 8), f"Stage18 ROI inspection | clips {first_number}-{first_number + len(items)-1}/{total}",
              font=font(22), fill="#182b45")
    draw.text((12, 36), "First / middle / last cache frames | RGB96x96 | manual review pending", font=font(14), fill="#384c65")
    for index, item in enumerate(items):
        row = item["row"]
        x, y = 8 + index % columns * cell_w, 64 + index // columns * cell_h
        color = "#2067a5" if row["split"] == "train" else "#7a479b"
        draw.rectangle((x, y, x + 294, y + 2), fill=color)
        label = f"#{first_number+index:03d} {row['speaker_id']} r{row['recording_id'].rsplit('_',1)[-1]} s{row['sentence_id']:03d} {row['split']}"
        draw.text((x, y + 5), label, font=font(12), fill=color)
        for j, (frame, frame_index) in enumerate(zip(item["thumbnails"], item["indices"])):
            image.paste(Image.fromarray(frame), (x + 98 * j, y + 23))
            draw.text((x + 98 * j, y + 120), f"k={frame_index} T={row['frames']}", font=font(11), fill="#384c65")
    return image


def frame_panel(rgb: np.ndarray, box: list[float], roi: np.ndarray) -> Image.Image:
    """448x252 source view with the exact pre-margin bbox and a96px ROI callout."""
    require(rgb.ndim == 3 and rgb.shape[2] == 3 and rgb.dtype == np.uint8, "Invalid RGB source frame")
    height, width = rgb.shape[:2]
    x1, y1, x2, y2 = box
    require(0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height, "BBox outside decoded image")
    require(roi.shape == (96, 96, 3) and roi.dtype == np.uint8, "Invalid ROI callout")
    panel = Image.new("RGB", (552, 252), "#e8edf3")
    scale = min(448 / width, 252 / height)
    nw, nh = round(width * scale), round(height * scale)
    dx, dy = (448 - nw) // 2, (252 - nh) // 2
    panel.paste(Image.fromarray(rgb).resize((nw, nh), Image.Resampling.LANCZOS), (dx, dy))
    draw = ImageDraw.Draw(panel)
    rect = (dx + x1 * nw / width, dy + y1 * nh / height,
            dx + x2 * nw / width, dy + y2 * nh / height)
    draw.rectangle(rect, outline="#39f57d", width=2)
    panel.paste(Image.fromarray(roi), (454, 72))
    draw.rectangle((453, 71, 550, 168), outline="#2a699c", width=1)
    draw.line((rect[2], (rect[1] + rect[3])/2, 453, 119), fill="#2a699c", width=1)
    draw.text((454, 45), "RGB ROI", font=font(12), fill="#163952")
    draw.text((454, 176), f"+{ROI_MARGIN:.0%} margin", font=font(11), fill="#163952")
    return panel


def source_sheet(items: list[dict], first_number: int, total: int) -> Image.Image:
    """Six recordings per page; three representative frames per recording."""
    require(0 < len(items) <= 6, "Source sheet requires1..6 recordings")
    row_h, tile_w = 308, 560
    image = Image.new("RGB", (3 * tile_w + 20, 70 + row_h * len(items)), "#f4f6f9")
    draw = ImageDraw.Draw(image)
    draw.text((12, 8), f"Stage18 source/bbox review | recordings {first_number}-{first_number + len(items)-1}/{total}",
              font=font(22), fill="#182b45")
    draw.text((12, 36), "Selected-clip timeline first / middle / last | green: label bbox before margin | manual review pending",
              font=font(14), fill="#384c65")
    for index, item in enumerate(items):
        y = 68 + index * row_h
        draw.text((12, y), f"#{first_number+index:02d} {item['recording_id']} | {item['split']} | {item['package']}",
                  font=font(16), fill="#182b45")
        for j, sample in enumerate(item["samples"]):
            x = 12 + j * tile_w
            draw.text((x, y + 24),
                f"s{sample['sentence_id']:03d} k={sample['cache_index']} src={sample['source_index']} PTS={sample['actual_pts_seconds']:.3f}s",
                font=font(13), fill="#384c65")
            image.paste(sample["panel"], (x, y + 44))
    return image


def representative_frames(rows: list[dict]) -> list[dict]:
    """Select three frames only from the union of chosen sentence intervals."""
    times = [(row["start_time"] + k / FPS, row["id"], k, row)
             for row in rows for k in range(row["frames"])]
    times.sort(key=lambda value: (value[0], value[1], value[2]))
    require(bool(times), "No selected frames for source review")
    return [{"row": times[index][3], "cache_index": times[index][2], "target_seconds": times[index][0]}
            for index in (0, len(times)//2, len(times)-1)]


def checked_frame(cap, frame_index: int, target_seconds: float, fps: float) -> tuple[np.ndarray, dict]:
    require(cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index), "Source seek failed")
    success, bgr = cap.read()
    require(success and bgr is not None, "Cannot decode selected source frame")
    position = float(cap.get(cv2.CAP_PROP_POS_FRAMES))
    pts = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000
    require(np.isfinite(position) and abs(position - (frame_index + 1)) <= 1e-6,
            "Decoded source frame index mismatch")
    require(np.isfinite(pts) and abs(pts - frame_index / fps) <= .001,
            "Source PTS differs from CFR index time")
    require(abs(pts - target_seconds) <= .5 / fps + 1e-6,
            "Source frame is not nearest to the selected cache timestamp")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), {
        "source_index": frame_index, "position_after_read": position, "actual_pts_seconds": pts,
        "target_seconds": target_seconds, "absolute_timestamp_error_seconds": abs(pts - target_seconds)}


def inspect(manifest: Path) -> dict:
    manifest = manifest.resolve()
    require(manifest.name == "intake_manifest.json" and manifest.is_relative_to(ROOT / "results/step18-intake"),
            "Only a stage18 readiness intake manifest is allowed")
    base = manifest.parent
    digest = sha256_file(manifest)
    document = read_json(manifest)
    require(document["schema"] == "lipnet-new-source-readiness-v1", "Unsupported intake schema")
    require(document["status"] == "prepared_pending_visual_review", "Intake must await visual review")
    require(document["scope"] == {"training": False, "model_forward": False, "test_payload_access": False,
                                  "default_manifest_changed": False, "readiness_pilot_only": True}, "Incorrect intake scope")
    require(sha256_file(base / "selection.json") == document["selection_sha256"], "Frozen selection changed")
    frozen_selection = read_json(base / "selection.json")
    frozen_rows = {row["id"]: row for row in frozen_selection["records"]}
    rows = document["records"]
    require(len(rows) == len({r["id"] for r in rows}) == 120, "Exactly120 unique new clips required")
    require(set(frozen_rows) == {row["id"] for row in rows}, "Prepared IDs differ from the frozen intake selection")
    for row in rows:
        require(all(row.get(key) == value for key, value in frozen_rows[row["id"]].items()),
                f"Prepared record differs from frozen intake selection: {row['id']}")
    require(Counter(r["split"] for r in rows) == {"train": 96, "val": 24}, "Unexpected pilot split")
    require(all(r["source_partition"] == "Training" and r["angle"] == "A" for r in rows), "Only new Training/A clips allowed")
    groups = defaultdict(list)
    for row in rows:
        groups[row["recording_id"]].append(row)
    require(len(groups) == 30 and all(len(value) == 4 for value in groups.values()), "Expected30 recordings with4 clips each")
    by_id = {r["id"]: r for r in rows}
    receipts = document["source_receipts"]
    require(len(receipts) == 30, "Expected30 local source receipts")
    receipt_map = {}
    for receipt in receipts:
        ids = receipt["selected_ids"]
        require(len(ids) == len(set(ids)) == 4 and all(sample_id in by_id for sample_id in ids), "Unknown/duplicate receipt ID")
        recordings = {by_id[sample_id]["recording_id"] for sample_id in ids}
        require(len(recordings) == 1, "Receipt mixes recordings")
        recording = next(iter(recordings))
        require(recording not in receipt_map and set(ids) == {r["id"] for r in groups[recording]}, "Duplicate/incomplete recording receipt")
        receipt_map[recording] = receipt
    require(set(receipt_map) == set(groups), "Incomplete source receipts")
    out = base / "visual_qa"
    require(not out.exists(), "Refusing to overwrite visual review folder")
    # Preflight path boundaries before any media payload is opened.
    cache_paths = {r["id"]: local_file(base / "clips", str(base / r["cache_path"])) for r in rows}
    label_paths = {recording: local_file(base / "labels", values[0]["label_path"]) for recording, values in groups.items()}
    video_paths = {recording: local_file(base / "selected_videos", receipt["local_path"]) for recording, receipt in receipt_map.items()}
    out.mkdir()
    cv2.setNumThreads(2)
    metrics, sheets, thumbnails = [], [], []
    for number, row in enumerate(rows, 1):
        path = cache_paths[row["id"]]
        require(path.stem == row["id"] and sha256_file(path) == row["cache_sha256"], "Cache identity/hash mismatch")
        clip = load_clip(path)
        require(clip["text"] == row["text"] and clip["source_text"] == row["source_text"], "Cache reference mismatch")
        require(len(clip["frames"]) == row["frames"] == row["expected_frames"], "Cache frame count mismatch")
        metrics.append(clip_statistics(row, clip))
        indices = [0, len(clip["frames"])//2, len(clip["frames"])-1]
        thumbnails.append({"row": row, "indices": indices, "thumbnails": [clip["frames"][index].copy() for index in indices]})
        if len(thumbnails) == 40 or number == len(rows):
            path = out / f"roi_samples_{len(sheets)+1:02d}.png"
            roi_sheet(thumbnails, number-len(thumbnails)+1, len(rows)).save(path)
            sheets.append({"kind": "roi_samples", "path": path.name, "sha256": sha256_file(path),
                           "clip_ids": [item["row"]["id"] for item in thumbnails],
                           "cache_indices": {item["row"]["id"]: item["indices"] for item in thumbnails}})
            thumbnails = []
            print(f"ROI_SHEET {number}/{len(rows)}", flush=True)
    source_audit, source_pages = [], []
    source_items = []
    for number, (recording, selected) in enumerate(sorted(groups.items()), 1):
        first, receipt = selected[0], receipt_map[recording]
        require(len({r["split"] for r in selected}) == len({r["label_path"] for r in selected}) == 1,
                "Recording has multiple roles or label files")
        label_path, video_path = label_paths[recording], video_paths[recording]
        require(video_path.stem == label_path.stem == first["id"].rsplit("_", 1)[0],
                "Staged source/label basename differs from selected recording")
        require(sha256_file(label_path) == first["label_sha256"], "Local label hash changed")
        require(video_path.stat().st_size == receipt["size_bytes"] and sha256_file(video_path) == receipt["sha256"],
                "Staged video hash/size mismatch")
        require(all(r["source_video_sha256"] == receipt["sha256"] for r in selected), "Row/source receipt hash mismatch")
        label = read_json(label_path)
        if isinstance(label, list):
            require(len(label) == 1, "Ambiguous local label list")
            label = label[0]
        boxes, label_fps = _extract_lip_boxes(label, "yxyx")
        require(boxes is not None, "Missing local lip boxes")
        cap = cv2.VideoCapture(str(video_path))
        samples = []
        try:
            require(cap.isOpened(), "Cannot open staged video")
            fps, frame_count = _video_timing_metadata(cap, boxes)
            require(abs(fps - receipt["video_fps"]) <= 1e-10 and frame_count == receipt["frame_count"] == receipt["bbox_count"],
                    "Source receipt timing/bbox count mismatch")
            require(label_fps == receipt["label_fps"], "Source receipt label FPS mismatch")
            require(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == receipt["width"]
                    and int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == receipt["height"], "Source dimensions changed")
            for choice in representative_frames(selected):
                row, k, target = choice["row"], choice["cache_index"], choice["target_seconds"]
                frame_index = int(round(target * fps))
                require(0 <= frame_index < frame_count, "Selected source frame outside video")
                rgb, timing = checked_frame(cap, frame_index, target, fps)
                box = boxes[frame_index]
                crop = crop_roi_from_box(rgb, box)
                require(sha256_file(cache_paths[row["id"]]) == row["cache_sha256"], "Representative cache changed")
                cached = load_clip(cache_paths[row["id"]])["frames"][k]
                require(np.array_equal(crop, cached), "Reconstructed ROI differs from stored selected cache frame")
                samples.append({"id": row["id"], "sentence_id": row["sentence_id"], "cache_index": k,
                    **timing, "bbox_xyxy_before_margin": box, "cached_roi_exact": True,
                    "panel": frame_panel(rgb, box, cached)})
        finally:
            cap.release()
        source_items.append({"recording_id": recording, "split": first["split"], "package": first["package"], "samples": samples})
        source_audit.append({"recording_id": recording, "split": first["split"], "selected_ids": [r["id"] for r in selected],
            "local_video": video_path.relative_to(base).as_posix(), "video_sha256": receipt["sha256"],
            "local_label": label_path.relative_to(base).as_posix(), "label_sha256": first["label_sha256"],
            "video_fps": fps, "label_fps": label_fps, "frame_count": frame_count, "bbox_count": len(boxes),
            "samples": [{key: value for key, value in sample.items() if key != "panel"} for sample in samples]})
        if len(source_items) == 6 or number == len(groups):
            path = out / f"source_bbox_{len(source_pages)+1:02d}.png"
            source_sheet(source_items, number-len(source_items)+1, len(groups)).save(path)
            source_pages.append({"kind": "source_bbox", "path": path.name, "sha256": sha256_file(path),
                                 "recording_ids": [item["recording_id"] for item in source_items]})
            source_items = []
            print(f"SOURCE_SHEET {number}/{len(groups)}", flush=True)
    require(sha256_file(manifest) == digest, "Intake manifest changed during review generation")
    require(sha256_file(base / "selection.json") == document["selection_sha256"], "Selection changed during review generation")
    report = {
        "schema": "lipnet-intake-visual-qa-v1", "status": "manual_review_pending", "manual_review_status": "pending",
        "manifest_sha256": digest, "selection_sha256": document["selection_sha256"],
        "generator_sha256": sha256_file(Path(__file__)), "new_clips_reviewable": len(rows),
        "roi_thumbnail_count": len(rows)*3, "source_recordings_reviewable": len(groups),
        "source_frame_thumbnail_count": len(source_audit)*3, "reconstructed_roi_exact_count": len(source_audit)*3,
        "roi_sheet_count": len(sheets), "source_sheet_count": len(source_pages),
        "sheets": sheets + source_pages, "metrics": metrics, "source_audit": source_audit,
        "automatic_flag_count": sum(bool(value["review_flags"]) for value in metrics),
        "automatic_checks_are_not_recognition_accuracy": True,
        "representation": {"roi_sampling": "Each clip first,middle,last frame", "source_sampling": "Per recording: first,middle,last element of time-sorted frames from all4 selected sentence intervals; no unselected gap frames",
                           "source_display": "448x252 aspect-preserved view plus96x96 cached RGB ROI", "box": "Label yxyx converted to xyxy; displayed rectangle excludes margin", "roi_margin": ROI_MARGIN},
        "scope": {"training": False, "model_forward": False, "test_payload_access": False, "external_source_access": False,
                  "allowed_media_directories": ["clips", "labels", "selected_videos"]},
    }
    with (out / "qa_inventory.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return report


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "results/step18-intake/run01/intake_manifest.json")
    args = parser.parse_args()
    report = inspect(args.run)
    print(json.dumps({"status": report["status"], "roi_sheets": report["roi_sheet_count"],
                      "source_sheets": report["source_sheet_count"], "automatic_flags": report["automatic_flag_count"]}))
