"""Make paired 96/224 inputs directly from the frozen step-01 source frames.

Only resize size differs between arms; both use one mp4v encode/decode roundtrip.
The regenerated 224 arm must exactly match step 01 after decoding. Source media,
old inputs, reference plan, labels and training cache are never written.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from prepare_inputs import PLAN_SHA256, inspect_mp4, sha256


BASE = Path(__file__).resolve().parent
REFERENCE_MANIFEST_SHA = "fdf8248f9a46ebba35c97607575cb7b30675398655566c804c196862c0fd8cf8"
HELPER_SHA = "a511ccde080510353cf5d25a64de6a31bbcb93fb258ec59f5c0c56b0866b50ff"
SIZES = (96, 224)


def decode(path: Path) -> np.ndarray:
    reader = cv2.VideoCapture(str(path))
    if not reader.isOpened():
        raise ValueError(f"Cannot decode {path}")
    frames = []
    try:
        while True:
            ok, frame = reader.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        reader.release()
    if not frames:
        raise ValueError(f"Empty clip: {path}")
    return np.stack(frames)


def main() -> None:
    plan_path = BASE / "input_plan.json"
    reference_path = BASE / "inputs" / "manifest.json"
    for path, expected in ((plan_path, PLAN_SHA256),
                           (reference_path, REFERENCE_MANIFEST_SHA),
                           (BASE / "prepare_inputs.py", HELPER_SHA)):
        if sha256(path) != expected:
            raise ValueError(f"Frozen reference changed: {path}")
    plan = json.loads(plan_path.read_text(encoding="utf8"))
    reference = json.loads(reference_path.read_text(encoding="utf8"))
    references = {row["id"]: row for row in reference["records"]}
    for name in ("manifest", "inventory", "protected_reference",
                 "upstream_preprocess", "upstream_dataset"):
        anchor = plan[name]
        if sha256(Path(anchor["path"])) != anchor["sha256"]:
            raise ValueError(f"Anchor changed: {name}")
    records = plan["records"]
    if len(records) != 4 or any(row["split"] != "train" for row in records):
        raise ValueError("Expected the same four frozen training clips")
    if set(references) != {row["id"] for row in records}:
        raise ValueError("Step-01 clip set changed")
    for row in references.values():
        if sha256(Path(row["path"])) != row["sha256"]:
            raise ValueError(f"Original 224 clip changed: {row['id']}")
    output = BASE / "inputs_step02"
    if output.exists():
        raise FileExistsError("Refusing to replace previous step-02 artifacts")
    output.mkdir()
    for side in SIZES:
        (output / str(side)).mkdir()
    grouped = defaultdict(list)
    for row in records:
        grouped[row["source_video"]["path"]].append(row)
    sources, results, paired_tiles = {}, [], []
    for source_name, rows in grouped.items():
        source = Path(source_name)
        stat = source.stat()
        expected = rows[0]["source_video"]
        if (stat.st_size, stat.st_mtime_ns) != (expected["size_bytes"], expected["mtime_ns"]):
            raise ValueError(f"Source metadata changed: {source.name}")
        container = inspect_mp4(source)
        label_path = Path(rows[0]["label_path"])
        label_bytes = label_path.read_bytes()
        label_hash = hashlib.sha256(label_bytes).hexdigest()
        if label_hash != rows[0]["label_sha256"]:
            raise ValueError("Label SHA does not match frozen plan")
        labels = json.loads(label_bytes.decode("utf-8-sig"))[0]
        boxes = labels["Bounding_box_info"]["Lip_bounding_box"]["xtl_ytl_xbr_ybr"]
        segments = []
        for row in rows:
            matched = [entry for entry in labels["Sentence_info"]
                       if entry["sentence_text"] == row["source_text"]
                       and abs(float(entry["start_time"]) - row["start_time"]) < 1e-6
                       and abs(float(entry["end_time"]) - row["end_time"]) < 1e-6]
            if len(matched) != 1:
                raise ValueError(f"Sentence alignment changed: {row['id']}")
            start, stop = int(30 * row["start_time"]), int(30 * row["end_time"])
            if not 0 <= start < stop <= len(boxes):
                raise ValueError("Invalid clip bounds")
            writers, paths = {}, {}
            for side in SIZES:
                paths[side] = output / str(side) / f"{row['id']}.mp4"
                writers[side] = cv2.VideoWriter(str(paths[side]), cv2.VideoWriter_fourcc(*"mp4v"),
                                                30.0, (side, side))
                if not writers[side].isOpened():
                    raise RuntimeError(f"Cannot initialize {side} mp4v encoder")
            segments.append({"row": row, "start": start, "stop": stop,
                             "writers": writers, "paths": paths, "written": 0,
                             "crop_hash": hashlib.sha256(), "bbox_min_height": 10**9,
                             "bbox_min_width": 10**9})
        capture = cv2.VideoCapture(str(source))
        try:
            if not capture.isOpened():
                raise RuntimeError(f"Source decoder failed: {source.name}")
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if not math.isfinite(fps) or fps <= 0:
                raise ValueError("Invalid source FPS")
            max_stop = max(item["stop"] for item in segments)
            for index in range(max_stop):
                ok, frame = capture.read()
                if not ok:
                    raise ValueError(f"Source decode failed: {source.name} frame {index}")
                for segment in segments:
                    if not segment["start"] <= index < segment["stop"]:
                        continue
                    top, left, bottom, right = map(int, boxes[index])
                    if not (0 <= top < bottom <= frame.shape[0] and 0 <= left < right <= frame.shape[1]):
                        raise ValueError(f"Invalid bbox at {index}")
                    crop = frame[top:bottom, left:right]
                    segment["crop_hash"].update(np.asarray([index, top, left, bottom, right], dtype="<i8").tobytes())
                    segment["crop_hash"].update(crop.tobytes(order="C"))
                    # Each arm starts from the identical native crop, never another encoded clip.
                    for side in SIZES:
                        resized = cv2.resize(crop, (side, side), interpolation=cv2.INTER_LINEAR)
                        segment["writers"][side].write(resized)
                    segment["written"] += 1
                    segment["bbox_min_height"] = min(segment["bbox_min_height"], bottom - top)
                    segment["bbox_min_width"] = min(segment["bbox_min_width"], right - left)
        finally:
            capture.release()
            for segment in segments:
                for writer in segment["writers"].values():
                    writer.release()
        after = source.stat()
        if (after.st_size, after.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
            raise ValueError("Source metadata changed while reading")
        sources[source.name] = {
            "path": str(source), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "full_source_sha256": None,
            "sha256_omission_reason": "Only necessary prefix was decoded; no full file copy or hash",
            "label_path": str(label_path), "label_sha256": label_hash,
            "native_fps": fps, "reported_frames": count, "width": width, "height": height,
            "decoded_prefix_frames": max_stop, "bbox_count": len(boxes),
            "top_level_container_screen": container, "source_modified": False, "audio_read": False}
        for segment in segments:
            row = segment["row"]
            expected_frames = segment["stop"] - segment["start"]
            if segment["written"] != expected_frames:
                raise ValueError("Frame write count mismatch")
            reference_array = decode(Path(references[row["id"]]["path"]))
            pair = []
            for side in SIZES:
                path = segment["paths"][side]
                array = decode(path)
                if array.shape != (expected_frames, side, side, 3):
                    raise ValueError(f"Wrong decoded shape: {array.shape}")
                file_hash = sha256(path)
                item = {"id": row["id"], "split": "train", "speaker_id": row["speaker_id"],
                        "resolution": side, "path": str(path), "sha256": file_hash,
                        "size_bytes": path.stat().st_size, "source_name": source.name,
                        "source_text": row["source_text"], "text": row["text"],
                        "start_time": row["start_time"], "end_time": row["end_time"],
                        "start_frame": segment["start"], "stop_frame_exclusive": segment["stop"],
                        "frames": expected_frames, "fps": 30.0, "native_fps": fps,
                        "native_time_at_first_selected_frame": segment["start"] / fps,
                        "native_time_at_stop_exclusive_frame": segment["stop"] / fps,
                        "first_frame_time_minus_label_seconds": segment["start"] / fps - row["start_time"],
                        "stop_frame_time_minus_label_seconds": segment["stop"] / fps - row["end_time"],
                        "dtype": str(array.dtype), "shape": list(array.shape), "color_order": "BGR",
                        "min": int(array.min()), "max": int(array.max()),
                        "mean_before_normalization": float(array.mean()),
                        "std_before_normalization_ddof1": float(array.std(ddof=1)),
                        "normalization_applied": False, "all_saved_frames_decoded": True,
                        "native_crop_stream_sha256": segment["crop_hash"].hexdigest(),
                        "native_crop_hash_format": "Per frame: little-endian int64 [index,top,left,bottom,right], then uint8 BGR crop C-order",
                        "min_bbox_height": segment["bbox_min_height"],
                        "min_bbox_width": segment["bbox_min_width"]}
                if item["std_before_normalization_ddof1"] <= 0:
                    raise ValueError("Zero clip variance")
                if side == 224:
                    same_shape = array.shape == reference_array.shape
                    mismatch = int(np.count_nonzero(array != reference_array)) if same_shape else None
                    max_error = int(np.abs(array.astype(np.int16) - reference_array.astype(np.int16)).max()) if same_shape else None
                    item["step01_reproduction"] = {
                        "path": references[row["id"]]["path"],
                        "reference_sha256": references[row["id"]]["sha256"],
                        "encoded_bytes_identical": file_hash == references[row["id"]]["sha256"],
                        "decoded_shape_identical": same_shape, "decoded_mismatched_elements": mismatch,
                        "decoded_max_absolute_difference": max_error}
                    if not same_shape or mismatch != 0:
                        raise ValueError(f"224 reproduction mismatch: {row['id']}")
                results.append(item)
                rgb = cv2.cvtColor(array[len(array) // 2], cv2.COLOR_BGR2RGB)
                tile = Image.new("RGB", (240, 280), "white")
                # Nearest-neighbour display enlargement is solely for QA, never model input.
                tile.paste(Image.fromarray(rgb).resize((224, 224), Image.Resampling.NEAREST), (8, 48))
                draw = ImageDraw.Draw(tile)
                draw.text((8, 5), f"{row['speaker_id']} sentence {row['sentence_id']}: {side}px", fill="black")
                draw.text((8, 23), f"{expected_frames} frames; one mp4v pass", fill="black")
                pair.append(tile)
            paired_tiles.append(pair)
            print(f"PREPARED PAIRED {row['id']} {expected_frames} frames; 224 pixels identical", flush=True)
    sheet = Image.new("RGB", (240 * len(paired_tiles), 560), "white")
    for column, pair in enumerate(paired_tiles):
        for row_index, tile in enumerate(pair):
            sheet.paste(tile, (240 * column, 280 * row_index))
    sheet_path = output / "contact_sheet.png"
    sheet.save(sheet_path)
    contract = dict(plan["contract"])
    contract["profile"] = "paired-published-code-vs-readme-resolution"
    contract["resize"] = "96x96 and 224x224, both from identical native source crop"
    report = {"schema": "olkavs-official-step02-input-manifest-v1",
              "created_utc": datetime.now(timezone.utc).isoformat(), "status": "prepared",
              "purpose": "Controlled resolution diagnostic on the same four step-01 train clips; not generalization evaluation",
              "plan_sha256": PLAN_SHA256, "step01_manifest_sha256": REFERENCE_MANIFEST_SHA,
              "preparation_script_sha256": sha256(Path(__file__)), "helper_script_sha256": HELPER_SHA,
              "opencv_version": cv2.__version__, "numpy_version": np.__version__,
              "audio_used": False, "model_run": False, "training_run": False,
              "val_or_test_payload_read": False, "public_pretraining_overlap_unknown": True,
              "preprocessing_contract": contract, "sources": sources, "records": results,
              "all_224_decoded_pixels_match_step01": True,
              "all_224_encoded_bytes_match_step01": all(r["step01_reproduction"]["encoded_bytes_identical"] for r in results if r["resolution"] == 224),
              "contact_sheet": {"path": str(sheet_path), "sha256": sha256(sheet_path)},
              "visual_qa": "pending separate image inspection",
              "limitation": "Nominal-30fps frame bounds retained; actual-FPS offsets reported. Application decodes only needed prefixes, but Drive provider may cache more bytes."}
    (output / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print("MANIFEST", output / "manifest.json", flush=True)


if __name__ == "__main__":
    main()
