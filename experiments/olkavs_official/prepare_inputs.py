"""Reproduce the published OLKAVS visual preprocessing for four fixed train clips.

This is an isolated inference smoke-test adapter. It never writes to source data,
the existing training cache, upstream code, or any existing output file.
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


BASE = Path(__file__).resolve().parent
PLAN_SHA256 = "1a327d4e2bdfd8e9a0847739df73e87c634c261fd66828d5a11d69b3f04ffd25"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_mp4(path: Path) -> dict:
    """Read top-level box headers only; not a complete video integrity check."""
    size = path.stat().st_size
    boxes, offset = [], 0
    with path.open("rb") as stream:
        while offset < size:
            stream.seek(offset)
            head = stream.read(8)
            if len(head) != 8:
                raise ValueError(f"Incomplete MP4 header at {offset}")
            length = int.from_bytes(head[:4], "big")
            kind = head[4:].decode("ascii", errors="replace")
            header_size = 8
            if length == 1:
                extended = stream.read(8)
                if len(extended) != 8:
                    raise ValueError("Incomplete extended MP4 size")
                length, header_size = int.from_bytes(extended, "big"), 16
            elif length == 0:
                length = size - offset
            if length < header_size or offset + length > size:
                raise ValueError(f"Invalid MP4 box {kind} at {offset}: {length}")
            boxes.append({"type": kind, "offset": offset, "size_bytes": length})
            offset += length
            if len(boxes) > 10000:
                raise ValueError("Too many MP4 top-level boxes")
    kinds = [box["type"] for box in boxes]
    if not kinds or kinds[0] != "ftyp" or "moov" not in kinds or "mdat" not in kinds:
        raise ValueError(f"Required MP4 boxes absent: {kinds}")
    return {"structural_screen_pass": True, "boxes": boxes,
            "full_recording_decode_verified": False}


def main() -> None:
    plan_path = BASE / "input_plan.json"
    if sha256(plan_path) != PLAN_SHA256:
        raise ValueError("Frozen input plan hash changed")
    plan = json.loads(plan_path.read_text(encoding="utf8"))
    for name in ("manifest", "inventory", "protected_reference",
                 "upstream_preprocess", "upstream_dataset"):
        anchor = plan[name]
        if sha256(Path(anchor["path"])) != anchor["sha256"]:
            raise ValueError(f"Anchor changed: {name}")
    records = plan["records"]
    if len(records) != 4 or any(row["split"] != "train" for row in records):
        raise ValueError("Exactly the four frozen train clips are required")
    output = BASE / "inputs"
    output.mkdir(exist_ok=True)
    targets = [output / f"{row['id']}.mp4" for row in records]
    if any(path.exists() for path in targets) or (output / "manifest.json").exists():
        raise FileExistsError("Refusing to overwrite previous input artifacts")
    sources, results, tiles = {}, [], []
    grouped = defaultdict(list)
    for row in records:
        grouped[row["source_video"]["path"]].append(row)

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
            raise ValueError(f"Label differs from existing manifest: {label_path.name}")
        labels = json.loads(label_bytes.decode("utf-8-sig"))[0]
        bboxes = labels["Bounding_box_info"]["Lip_bounding_box"]["xtl_ytl_xbr_ybr"]
        segments = []
        for row in rows:
            matched = [item for item in labels["Sentence_info"]
                       if item["sentence_text"] == row["source_text"]
                       and abs(float(item["start_time"]) - row["start_time"]) < 1e-6
                       and abs(float(item["end_time"]) - row["end_time"]) < 1e-6]
            if len(matched) != 1:
                raise ValueError(f"Sentence alignment mismatch: {row['id']}")
            start, stop = int(30 * row["start_time"]), int(30 * row["end_time"])
            if not 0 <= start < stop <= len(bboxes):
                raise ValueError("Sentence frame bounds exceed label boxes")
            destination = output / f"{row['id']}.mp4"
            writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"mp4v"),
                                     30.0, (224, 224))
            if not writer.isOpened():
                raise RuntimeError("mp4v encoder failed to initialize")
            segments.append({"row": row, "start": start, "stop": stop,
                             "path": destination, "writer": writer, "written": 0,
                             "bbox_min_height": 10**9, "bbox_min_width": 10**9})
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            for segment in segments:
                segment["writer"].release()
            raise RuntimeError(f"Source decoder could not open {source.name}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Invalid source FPS")
        max_stop = max(item["stop"] for item in segments)
        try:
            for index in range(max_stop):
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f"Decode failed before selected range: {source.name} frame {index}")
                for segment in segments:
                    if not segment["start"] <= index < segment["stop"]:
                        continue
                    top, left, bottom, right = map(int, bboxes[index])
                    if not (0 <= top < bottom <= frame.shape[0] and 0 <= left < right <= frame.shape[1]):
                        raise ValueError(f"Invalid yxyx bbox {bboxes[index]} at {index}")
                    crop = cv2.resize(frame[top:bottom, left:right], (224, 224),
                                      interpolation=cv2.INTER_LINEAR)
                    segment["writer"].write(crop)
                    segment["written"] += 1
                    segment["bbox_min_height"] = min(segment["bbox_min_height"], bottom - top)
                    segment["bbox_min_width"] = min(segment["bbox_min_width"], right - left)
        finally:
            cap.release()
            for segment in segments:
                segment["writer"].release()
        after = source.stat()
        if (after.st_size, after.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
            raise ValueError("Source changed during read")
        sources[source.name] = {
            "path": str(source), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "full_source_sha256": None, "sha256_omission_reason": "No full source read/copy required",
            "label_path": str(label_path), "label_sha256": label_hash,
            "native_fps": fps, "reported_frames": count, "width": width, "height": height,
            "bbox_count": len(bboxes), "decoded_prefix_frames": max_stop,
            "top_level_container_screen": container,
            "source_modified": False, "audio_read": False,
        }
        for segment in segments:
            reloaded = []
            reader = cv2.VideoCapture(str(segment["path"]))
            while True:
                ok, frame = reader.read()
                if not ok:
                    break
                reloaded.append(frame)
            reader.release()
            expected_count = segment["stop"] - segment["start"]
            if segment["written"] != expected_count or len(reloaded) != expected_count:
                raise ValueError("Saved clip did not fully decode")
            array = np.stack(reloaded)
            row = segment["row"]
            item = {"id": row["id"], "split": "train", "speaker_id": row["speaker_id"],
                    "path": str(segment["path"]), "sha256": sha256(segment["path"]),
                    "size_bytes": segment["path"].stat().st_size,
                    "source_name": source.name, "source_text": row["source_text"],
                    "text": row["text"], "start_time": row["start_time"], "end_time": row["end_time"],
                    "start_frame": segment["start"], "stop_frame_exclusive": segment["stop"],
                    "frames": len(reloaded), "fps": 30.0, "native_fps": fps,
                    "native_time_at_first_selected_frame": segment["start"] / fps,
                    "native_time_at_stop_exclusive_frame": segment["stop"] / fps,
                    "first_frame_time_minus_label_seconds": segment["start"] / fps - row["start_time"],
                    "stop_frame_time_minus_label_seconds": segment["stop"] / fps - row["end_time"],
                    "dtype": str(array.dtype), "shape": list(array.shape), "color_order": "BGR",
                    "min": int(array.min()), "max": int(array.max()),
                    "mean_before_normalization": float(array.mean()),
                    "std_before_normalization_ddof1": float(array.std(ddof=1)),
                    "normalization_applied": False,
                    "min_bbox_height": segment["bbox_min_height"],
                    "min_bbox_width": segment["bbox_min_width"], "all_saved_frames_decoded": True}
            if item["std_before_normalization_ddof1"] <= 0:
                raise ValueError("Constant clip would cause zero standard deviation")
            results.append(item)
            tile = Image.new("RGB", (224, 274), "white")
            rgb = cv2.cvtColor(array[len(array)//2], cv2.COLOR_BGR2RGB)
            tile.paste(Image.fromarray(rgb), (0, 50))
            draw = ImageDraw.Draw(tile)
            draw.text((6, 5), f"{row['speaker_id']} / sentence {row['sentence_id']}", fill="black")
            draw.text((6, 23), f"{len(array)} frames / 224 x 224", fill="black")
            tiles.append(tile)
            print(f"PREPARED {row['id']} {len(array)} frames", flush=True)

    sheet = Image.new("RGB", (224 * len(tiles), 274), "white")
    for index, tile in enumerate(tiles):
        sheet.paste(tile, (224 * index, 0))
    sheet_path = output / "contact_sheet.png"
    sheet.save(sheet_path)
    report = {"schema": "olkavs-official-input-manifest-v1",
              "created_utc": datetime.now(timezone.utc).isoformat(), "status": "prepared",
              "purpose": plan["purpose"], "plan_sha256": PLAN_SHA256,
              "preparation_script_sha256": sha256(Path(__file__)),
              "audio_used": False, "model_run": False, "training_run": False,
              "val_or_test_payload_read": False, "public_pretraining_overlap_unknown": True,
              "preprocessing_contract": plan["contract"], "sources": sources, "records": results,
              "contact_sheet": {"path": str(sheet_path), "sha256": sha256(sheet_path)},
              "visual_qa": "pending human/model image inspection",
              "limitation": "Code-level published nominal-30fps preprocessing reproduction, including mp4v roundtrip. Native-FPS boundary offsets are reported. Full source videos were not decoded."}
    (output / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print("MANIFEST", output / "manifest.json", flush=True)


if __name__ == "__main__":
    main()
