"""Freeze existing val20 sources without decoding frames or changing source data."""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from prepare_inputs import inspect_mp4

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
MANIFEST = ROOT / "data_cache/datasets/aihub_timed_v4/manifest.json"
INVENTORY = ROOT / "results/step18-intake/run01/inventory.json"
DIAGNOSTIC = BASE / "input_plan.json"
OUTPUT = BASE / "step03_source_plan.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def anchor(path: Path) -> dict:
    return {"path": str(path), "sha256": sha256(path)}


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError("Refusing to replace the frozen source plan")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    diagnostic = json.loads(DIAGNOSTIC.read_text(encoding="utf-8"))
    selected = [row for row in manifest["records"] if row["split"] == "val"]
    if len(selected) != 20 or len({row["id"] for row in selected}) != 20:
        raise ValueError("Expected the unchanged complete val20 split")
    previous_ids = {row["id"] for row in diagnostic["records"]}
    previous_speakers = {row["speaker_id"] for row in diagnostic["records"]}
    if previous_ids.intersection(row["id"] for row in selected):
        raise ValueError("Diagnostic clips overlap evaluation clips")
    if previous_speakers.intersection(row["speaker_id"] for row in selected):
        raise ValueError("Diagnostic speakers overlap evaluation speakers")
    grouped = defaultdict(list)
    for row in selected:
        grouped[row["id"].rsplit("_", 1)[0]].append(row)
    sources = []
    records = []
    for stem, rows in sorted(grouped.items()):
        part = inventory["partitions"][rows[0]["source_partition"]]
        source_meta = part["videos"][stem]
        label_meta = part["labels"][stem]
        source, label_path = Path(source_meta["path"]), Path(label_meta["path"])
        source_stat, label_stat = source.stat(), label_path.stat()
        if (source_stat.st_size, source_stat.st_mtime_ns) != (
            source_meta["size_bytes"], source_meta["mtime_ns"]
        ):
            raise ValueError(f"Source metadata changed: {stem}")
        if source_stat.st_size != rows[0]["source_video"]["size_bytes"]:
            raise ValueError(f"G source differs in byte count from original: {stem}")
        if (label_stat.st_size, label_stat.st_mtime_ns) != (
            label_meta["size_bytes"], label_meta["mtime_ns"]
        ):
            raise ValueError(f"Label metadata changed: {stem}")
        label_bytes = label_path.read_bytes()
        label_hash = hashlib.sha256(label_bytes).hexdigest()
        if any(label_hash != row["label_sha256"] for row in rows):
            raise ValueError(f"Label content differs from original: {stem}")
        labels = json.loads(label_bytes.decode("utf-8-sig"))[0]
        boxes = labels["Bounding_box_info"]["Lip_bounding_box"]["xtl_ytl_xbr_ybr"]
        prefix = max(int(30 * row["end_time"]) for row in rows)
        container = inspect_mp4(source)
        sources.append({
            "stem": stem,
            "source_video": source_meta,
            "label_inventory": label_meta,
            "label_sha256": label_hash,
            "label_bbox_count": len(boxes),
            "selected_records": len(rows),
            "selected_ids": [row["id"] for row in rows],
            "last_required_nominal_frame_exclusive": prefix,
            "nominal_prefix_seconds": prefix / 30,
            "last_label_end_seconds": max(row["end_time"] for row in rows),
            "container_screen": container,
            "full_source_sha256": None,
            "source_full_hash_omission_reason": "Avoid a full cloud source read for metadata planning",
            "video_frames_decoded": 0,
            "source_modified": False,
        })
        for row in rows:
            matches = [sentence for sentence in labels["Sentence_info"]
                       if sentence["sentence_text"] == row["source_text"]
                       and abs(float(sentence["start_time"]) - row["start_time"]) < 1e-6
                       and abs(float(sentence["end_time"]) - row["end_time"]) < 1e-6]
            if len(matches) != 1:
                raise ValueError(f"Sentence content or timing mismatch: {row['id']}")
            start, stop = int(30 * row["start_time"]), int(30 * row["end_time"])
            if not 0 <= start < stop <= len(boxes):
                raise ValueError(f"Nominal bounds exceed bbox records: {row['id']}")
            if any(len(box) != 4 or not (0 <= int(box[0]) < int(box[2])
                    and 0 <= int(box[1]) < int(box[3])) for box in boxes[start:stop]):
                raise ValueError(f"Invalid bbox shape or ordering: {row['id']}")
            records.append({
                **row,
                "source_video_original_manifest": row["source_video"],
                "label_original_manifest": row["label_path"],
                "source_video": source_meta,
                "label_path": str(label_path),
                "label_inventory": label_meta,
                "legacy_cache_absolute_path": str(MANIFEST.parent / row["cache_path"]),
                "legacy_cache_exists": (MANIFEST.parent / row["cache_path"]).is_file(),
                "legacy_cache_used_as_official_model_input": False,
                "source_stem": stem,
                "nominal_start_frame": start,
                "nominal_stop_frame_exclusive": stop,
                "nominal_frames": stop - start,
                "label_sentence_and_bounds_verified": True,
            })
        after = source.stat()
        if (after.st_size, after.st_mtime_ns) != (source_stat.st_size, source_stat.st_mtime_ns):
            raise ValueError(f"Source metadata changed during planning: {stem}")
    records.sort(key=lambda row: row["id"])
    report = {
        "schema": "olkavs-step03-source-plan-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Evaluate all existing val20 clips with frozen step02 official visual model settings",
        "selection_rule": "All and only original split=val records; no output-based filtering or replacement",
        "frozen_before_model_inference": True,
        "manifest": anchor(MANIFEST),
        "inventory": anchor(INVENTORY),
        "diagnostic_plan": anchor(DIAGNOSTIC),
        "planning_script": anchor(Path(__file__)),
        "record_count": len(records),
        "speaker_counts": dict(sorted(Counter(row["speaker_id"] for row in records).items())),
        "source_count": len(sources),
        "source_total_bytes": sum(source["source_video"]["size_bytes"] for source in sources),
        "selected_nominal_frames": sum(row["nominal_frames"] for row in records),
        "selected_label_duration_seconds": sum(row["end_time"] - row["start_time"] for row in records),
        "diagnostic_ids_overlap": [],
        "diagnostic_speakers_overlap": [],
        "test50_preserved": True,
        "test_payloads_read": False,
        "test_targets_used_for_inference_or_scoring": False,
        "test_split_metadata_record_count_only": sum(row["split"] == "test" for row in manifest["records"]),
        "new_training": False,
        "audio_used": False,
        "source_frame_payloads_decoded_at_planning": False,
        "source_read_policy": "Read only these four G files. Labels and MP4 headers checked; later preparation may decode selected intervals or a bounded prefix. Google Drive may cache entire source files.",
        "evaluation_scope": "Existing project validation split; not guaranteed held out from public OLKAVS pretraining, not webcam generalization",
        "sources": sources,
        "records": records,
    }
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"plan": str(OUTPUT), "sha256": sha256(OUTPUT),
                      "records": len(records), "sources": len(sources),
                      "source_total_bytes": report["source_total_bytes"],
                      "nominal_frames": report["selected_nominal_frames"],
                      "speakers": report["speaker_counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
