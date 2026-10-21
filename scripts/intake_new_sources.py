"""Stage18: inventory new Training sources and prepare a separate readiness pilot.

Never trains a model or changes the default dataset. Cloud originals are read only.
The pilot has train/val roles and deliberately is not a production train/val/test
manifest. Only selected videos are copied locally; audio and Validation payloads
are not opened.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from models.dataset_manifest import parse_identity, sha256_file, ROI_METHOD, TEMPORAL_METHOD
from models.tokenizer import normalize_text, encode_text, ctc_min_input_len, tokenizer_metadata

PARENT = ROOT / "data_cache/datasets/aihub_timed_v4/manifest.json"
PREVIOUS = ROOT / "results/step17-expansion/run01"
DEFAULT_OUT = ROOT / "results/step18-intake/run01"
PROVENANCE = ("speaker_id", "utterance_id", "recording_id", "angle", "source_partition", "roi_method", "label_sha256")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def package(path: Path, prefix: str) -> str:
    values = {p for p in path.parts if re.fullmatch("(?:" + prefix + r")\d+", p)}
    require(len(values) == 1, f"Unambiguous package required: {path}")
    return next(iter(values))


def snapshot(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path.resolve()), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def inventory(source_root: Path, out: Path) -> None:
    """Enumerate filenames and sizes, without reading any source payload."""
    require(not (out / "inventory.json").exists(), "New inventory required")
    all_parts = {}
    for folder, role in (("1.Training", "Training"), ("2.Validation", "Validation")):
        part = source_root / folder
        require(part.is_dir(), f"Missing official partition: {part}")
        videos, labels, audio = {}, {}, []
        for path in sorted(part.rglob("*")):
            if not path.is_file():
                continue
            ext = path.suffix.lower()
            if ext not in (".mp4", ".json", ".wav"):
                continue
            if ext == ".wav":
                audio.append(snapshot(path))
                continue
            table = videos if ext == ".mp4" else labels
            require(path.stem not in table, f"Duplicate filename: {path}")
            table[path.stem] = {**snapshot(path), **parse_identity(path.stem),
                                "package": package(path, "TS|VS" if ext == ".mp4" else "TL|VL")}
        require(set(videos) == set(labels), f"Video/JSON names do not pair: {role}")
        for stem, video in videos.items():
            require(video["package"][2:] == labels[stem]["package"][2:], f"Package mismatch: {stem}")
        all_parts[role] = {"videos": videos, "labels": labels, "audio": audio,
            "summary": {"videos": len(videos), "labels": len(labels), "audio": len(audio),
                "speakers": sorted({v["speaker_id"] for v in videos.values()}),
                "frontal_videos": sum(v["angle"] == "A" for v in videos.values()),
                "video_bytes": sum(v["size_bytes"] for v in videos.values()),
                "label_bytes": sum(v["size_bytes"] for v in labels.values()),
                "packages": dict(Counter(v["package"] for v in videos.values()))}}
    write_json(out / "inventory.json", {"schema": "lipnet-source-inventory-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(), "source_root": str(source_root),
        "payloads_read": False, "partitions": all_parts})
    print(json.dumps({k: v["summary"] for k, v in all_parts.items()}, ensure_ascii=False), flush=True)


def index(out: Path) -> None:
    """Read only new-speaker frontal Training labels and persist exact label bytes."""
    inv = read_json(out / "inventory.json")
    parent = read_json(PARENT)
    old_speakers = {r["speaker_id"] for r in parent["records"]}
    validation_speakers = set(inv["partitions"]["Validation"]["summary"]["speakers"])
    train = inv["partitions"]["Training"]
    candidates, excluded, sources = [], [], []
    eligible = [(stem, v) for stem, v in train["videos"].items()
                if v["angle"] == "A" and v["speaker_id"] not in old_speakers]
    require(bool(eligible), "No new Training speakers found")
    require(not ({v["speaker_id"] for _, v in eligible} & validation_speakers),
            "New Training speakers overlap official Validation")
    label_dir = out / "labels"
    label_dir.mkdir(exist_ok=False)
    for number, (stem, video) in enumerate(sorted(eligible), 1):
        source_label = train["labels"][stem]
        path = Path(source_label["path"])
        require(snapshot(path) == {k: source_label[k] for k in ("path", "size_bytes", "mtime_ns")}, "Label metadata changed")
        raw = path.read_bytes()
        require(len(raw) == source_label["size_bytes"], "Label size changed")
        data = json.loads(raw)
        if isinstance(data, list):
            require(len(data) == 1, "One recording per label required")
            data = data[0]
        require(data["speaker_info"]["speaker_ID"] == video["speaker_id"], "Speaker label mismatch")
        require(data["Video_env"]["Angle"] == "A" and data["Video_info"]["video_Name"] == stem + ".mp4", "Video label mismatch")
        fps = float(data["Video_info"]["FPS"])
        require(math.isfinite(fps) and fps > 0, "Invalid label FPS")
        boxes = data["Bounding_box_info"]["Lip_bounding_box"]["xtl_ytl_xbr_ybr"]
        require(isinstance(boxes, list) and bool(boxes), "Missing frame boxes")
        local_label = label_dir / (stem + ".json")
        with local_label.open("xb") as stream:
            stream.write(raw)
        digest = hashlib.sha256(raw).hexdigest()
        sources.append({"stem": stem, "video": video, "source_label": source_label,
            "local_label": str(local_label), "label_sha256": digest, "label_fps": fps,
            "bbox_count": len(boxes), "sentence_count": len(data["Sentence_info"])})
        seen = set()
        for sentence in data["Sentence_info"]:
            sid = sentence["ID"]
            require(type(sid) is int and sid >= 0 and sid not in seen, f"Invalid or duplicate sentence ID: {stem}")
            seen.add(sid)
            sample_id = f"{stem}_{sid:03d}"
            try:
                source_text = sentence["sentence_text"]
                text = normalize_text(source_text)
                require(bool(text), "empty_text")
                start, end = float(sentence["start_time"]), float(sentence["end_time"])
                require(math.isfinite(start) and math.isfinite(end) and 0 <= start < end, "invalid_time")
                frames = round((end - start) * 25)
                ctc_min = ctc_min_input_len(encode_text(text))
                require(frames >= ctc_min, "ctc_too_short")
                require(frames <= 200, "over_200_frames")
            except ValueError as exc:
                excluded.append({"id": sample_id, "reason": str(exc)})
                continue
            candidates.append({"id": sample_id, **parse_identity(sample_id), "sentence_id": sid,
                "package": video["package"], "source_partition": "Training", "source_video": video,
                "label_path": str(local_label), "source_label_path": source_label["path"],
                "label_sha256": digest, "source_text": source_text, "text": text,
                "start_time": start, "end_time": end, "expected_frames": frames,
                "ctc_min_frames": ctc_min, "roi_method": ROI_METHOD, "temporal_method": TEMPORAL_METHOD})
        print(f"[labels {number}/{len(eligible)}] {stem} candidates={len(candidates)}", flush=True)
    write_json(out / "source_index.json", {"schema": "lipnet-new-source-index-v1",
        "inventory_sha256": sha256_file(out / "inventory.json"), "parent_sha256": sha256_file(PARENT),
        "excluded_existing_speakers": sorted(old_speakers & set(train["summary"]["speakers"])),
        "official_validation_speakers_reserved": sorted(validation_speakers),
        "sources": sources, "candidates": candidates, "excluded": excluded,
        "excluded_reasons": dict(Counter(r["reason"] for r in excluded)),
        "scope": "Metadata candidates, not yet video-verified or training-ready"})


def frozen_paths(out: Path) -> list[Path]:
    """Required plan anchors; omission cannot disable verification."""
    return [PARENT, PREVIOUS / "plan.json", PREVIOUS / "completed.json", PREVIOUS / "quality_checks.json",
        ROOT / "docs/단계별_개선/17_검증결과.json", out / "inventory.json", out / "source_index.json",
        out / "protected_reference.json", *[ROOT / name for name in (
        "scripts/intake_new_sources.py", "scripts/intake_selection.py", "scripts/prepare_data.py",
        "backend/models/cache.py", "backend/models/tokenizer.py", "backend/models/dataset_manifest.py",
        "backend/preprocessing/roi_extractor.py", "backend/preprocessing/resampler.py", "backend/config.py")]]


def plan(out: Path) -> None:
    """Freeze new-speaker roles and pilot IDs before reading video pixels."""
    from intake_selection import select_intake
    src = read_json(out / "source_index.json")
    old = read_json(PARENT)
    prior = read_json(PREVIOUS / "plan.json")
    require(sha256_file(PARENT) == src["parent_sha256"], "Parent changed")
    require(sha256_file(PARENT) == prior["manifest_sha256"], "Stage17 parent manifest changed")
    held = set(prior["cohorts"]["train60"] + prior["cohorts"]["val20"])
    protected = sorted({r["recording_id"] for r in old["records"] if r["id"] in held or r["split"] == "test"})
    selected = select_intake(src["candidates"], old["records"], protected)
    value = {"schema": "lipnet-new-source-readiness-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "tokenizer": tokenizer_metadata(), **selected,
        "frozen_files": {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in frozen_paths(out)},
        "scope": {"training": False, "model_forward": False, "test_payload_access": False,
                  "default_manifest_changed": False, "readiness_pilot_only": True},
        "video_bytes_to_copy": sum(r["source_video"]["size_bytes"] for r in
            {r["recording_id"]: r for r in selected["records"]}.values())}
    require(value["video_bytes_to_copy"] < 15 * 1024**3, "Pilot source budget exceeds 15GiB")
    write_json(out / "selection.json", value)
    print(json.dumps({k: value[k] for k in ("counts", "speaker_mapping", "video_bytes_to_copy")}, ensure_ascii=False), flush=True)


def verify_frozen(document: dict) -> None:
    for relative, digest in document["frozen_files"].items():
        require((ROOT / relative).resolve().is_relative_to(ROOT), "Frozen path outside project")
        require(sha256_file(ROOT / relative) == digest, f"Frozen artifact changed: {relative}")


def verify_selection(out: Path, document: dict) -> None:
    """Recompute selection and bind every read path to the Training inventory."""
    from intake_selection import select_intake
    require(document.get("schema") == "lipnet-new-source-readiness-v1", "Unsupported readiness schema")
    require(document.get("tokenizer") == tokenizer_metadata(), "Tokenizer contract changed")
    require(set(document["frozen_files"]) == {p.relative_to(ROOT).as_posix() for p in frozen_paths(out)}, "Missing or extra frozen anchors")
    verify_frozen(document)
    src = read_json(out / "source_index.json")
    require(src["inventory_sha256"] == sha256_file(out / "inventory.json") and
            src["parent_sha256"] == sha256_file(PARENT), "Index provenance changed")
    parent = read_json(PARENT)
    prior = read_json(PREVIOUS / "plan.json")
    require(sha256_file(PARENT) == prior["manifest_sha256"], "Stage17 parent manifest changed")
    held = set(prior["cohorts"]["train60"] + prior["cohorts"]["val20"])
    protected = sorted({r["recording_id"] for r in parent["records"] if r["id"] in held or r["split"] == "test"})
    computed = select_intake(src["candidates"], parent["records"], protected)
    for key, value in computed.items():
        require(document.get(key) == value, f"Selection changed: {key}")
    require(document["scope"] == {"training": False, "model_forward": False, "test_payload_access": False,
                "default_manifest_changed": False, "readiness_pilot_only": True}, "Readiness scope changed")
    inv = read_json(out / "inventory.json")["partitions"]["Training"]
    for row in document["records"]:
        stem = row["id"].rsplit("_", 1)[0]
        require(row["source_video"] == inv["videos"][stem], "Source path not bound to Training inventory")
        require(Path(row["label_path"]).resolve() == (out / "labels" / (stem + ".json")).resolve(), "Label path outside local index")
        require(row["source_label_path"] == inv["labels"][stem]["path"], "Source label path mismatch")
    total = sum(r["source_video"]["size_bytes"] for r in {r["recording_id"]: r for r in document["records"]}.values())
    require(document["video_bytes_to_copy"] == total and total < 15 * 1024**3, "Source copy budget changed")


def prepare(out: Path) -> None:
    """Stage selected videos, crop the pilot, and verify complete per-clip provenance."""
    import cv2
    import numpy as np
    import torch
    from models.cache import load_clip
    from prepare_data import prepare_video, _extract_lip_boxes, _video_timing_metadata
    cv2.setNumThreads(2)
    torch.set_num_threads(2)
    document = read_json(out / "selection.json")
    selection_sha = sha256_file(out / "selection.json")
    verify_selection(out, document)
    require(not (out / "intake_manifest.json").exists(), "Pilot already completed")
    clips, staging = out / "clips", out / "selected_videos"
    clips.mkdir(exist_ok=False)
    staging.mkdir(exist_ok=False)
    require(shutil.disk_usage(out).free > document["video_bytes_to_copy"] + 5 * 1024**3, "Insufficient free disk")
    groups = defaultdict(list)
    for row in document["records"]:
        groups[row["recording_id"]].append(row)
    manifest_rows, receipts = [], []
    started = time.perf_counter()
    for number, (_, rows) in enumerate(sorted(groups.items()), 1):
        first = rows[0]
        source = first["source_video"]
        source_path = Path(source["path"])
        require(snapshot(source_path) == {k: source[k] for k in ("path", "size_bytes", "mtime_ns")}, "Video metadata changed")
        label_path = Path(first["label_path"])
        require(sha256_file(label_path) == first["label_sha256"], "Label changed")
        local = staging / source_path.name
        digest, count = hashlib.sha256(), 0
        print(f"[video {number}/{len(groups)}] copy {local.name} {source['size_bytes']/1024**2:.1f}MiB", flush=True)
        with source_path.open("rb") as src, local.open("xb") as dst:
            while block := src.read(1024**2):
                dst.write(block)
                digest.update(block)
                count += len(block)
        require(count == source["size_bytes"] and sha256_file(local) == digest.hexdigest(), "Copied source mismatch")
        require(snapshot(source_path) == {k: source[k] for k in ("path", "size_bytes", "mtime_ns")}, "Video changed during read")
        label = read_json(label_path)
        if isinstance(label, list):
            label = label[0]
        boxes, label_fps = _extract_lip_boxes(label, "yxyx")
        cap = cv2.VideoCapture(str(local))
        try:
            require(cap.isOpened(), "Cannot decode source")
            fps, frame_count = _video_timing_metadata(cap, boxes)
            width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            require(all(0 <= r["start_time"] < r["end_time"] <= frame_count/fps + 1e-6 for r in rows), "Selected interval outside video")
        finally:
            cap.release()
        saved = prepare_video(str(local), str(label_path), str(clips), max_frames=200, roi_mode="bbox",
            sentence_ids=[r["sentence_id"] for r in rows], bbox_coordinate_order="yxyx",
            provenance={r["sentence_id"]: {k: r[k] for k in PROVENANCE} for r in rows})
        require(saved == len(rows), "Incomplete pilot preparation; do not replace failed IDs silently")
        receipt = {"source_path": str(source_path), "local_path": str(local), "sha256": digest.hexdigest(),
            "size_bytes": count, "video_fps": fps, "label_fps": label_fps, "frame_count": frame_count,
            "bbox_count": len(boxes), "width": width, "height": height, "selected_ids": [r["id"] for r in rows]}
        receipts.append(receipt)
        for row in rows:
            cached = clips / (row["id"] + ".npz")
            data = load_clip(cached)
            require(data["text"] == row["text"] and data["source_text"] == row["source_text"], "Cache text mismatch")
            require(len(data["frames"]) == row["expected_frames"], "Cache frame count mismatch")
            require(data["frames"].shape[1:] == (96, 96, 3) and data["frames"].dtype == np.uint8, "Cache pixel contract mismatch")
            require(data["label"].tolist() == encode_text(row["text"]), "Cache encoded label mismatch")
            with np.load(cached, allow_pickle=False) as z:
                for key in PROVENANCE:
                    require(str(z[key]) == str(row[key]), f"Cache provenance mismatch {key}")
                require(str(z["temporal_method"]) == TEMPORAL_METHOD, "Cache timing method mismatch")
                require(float(z["video_fps"]) == fps and float(z["label_fps"]) == label_fps, "Cache FPS mismatch")
            manifest_rows.append({**row, "cache_path": cached.relative_to(out).as_posix(),
                "cache_sha256": sha256_file(cached), "frames": len(data["frames"]), "video_fps": fps,
                "label_fps": label_fps, "source_video_sha256": digest.hexdigest()})
        write_json(out / "receipts" / (local.stem + ".json"), receipt)
    verify_frozen(document)
    require(sha256_file(out / "selection.json") == selection_sha, "Selection changed during preparation")
    write_json(out / "intake_manifest.json", {**document, "records": sorted(manifest_rows, key=lambda r: r["id"]),
        "selection_sha256": selection_sha, "status": "prepared_pending_visual_review",
        "source_receipts": receipts, "runtime_seconds": time.perf_counter() - started,
        "environment": {"python": platform.python_version(), "opencv": cv2.__version__, "numpy": np.__version__, "torch": torch.__version__}})
    print("PILOT_PREPARED", len(manifest_rows), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inventory", "index", "plan", "prepare"))
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out.resolve()
    require(out.is_relative_to(ROOT / "results/step18-intake"), "Stage18 results directory required")
    try:
        if args.command == "inventory":
            require(args.source_root is not None, "Source root required")
            inventory(args.source_root.resolve(), out)
        else:
            {"index": index, "plan": plan, "prepare": prepare}[args.command](out)
    except Exception as exc:
        print(f"STOPPED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
