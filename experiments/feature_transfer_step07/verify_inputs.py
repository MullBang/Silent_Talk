"""Independently audit every saved step07 frame trace without rereading Drive.

This auditor does not import the preparation implementation, its interval
helper, or its tokenizer. Source pixels cannot be independently reconstructed
from hashes; the audit checks saved evidence, correspondence and provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]


def local(path: str | Path) -> Path:
    result = Path(path).resolve()
    if not result.is_relative_to(ROOT):
        raise ValueError(f"Auditor must not read a source/Drive file: {result}")
    return result


def sha(path: str | Path) -> str:
    h = hashlib.sha256()
    with local(path).open("rb") as stream:
        while block := stream.read(2**20):
            h.update(block)
    return h.hexdigest()


def read_json(path: str | Path, expected: str | None = None) -> dict:
    path = local(path)
    if expected is not None and sha(path) != expected:
        raise ValueError(f"Hash differs: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def same(actual, expected, what: str) -> None:
    if actual != expected:
        raise ValueError(f"{what}: {actual!r} != {expected!r}")


def near(actual: float, expected: float, tolerance: float, what: str) -> None:
    if not math.isfinite(actual) or abs(actual - expected) > tolerance:
        raise ValueError(f"{what}: {actual!r} != {expected!r}")


def tokens_from_syllables(raw: str) -> tuple[str, list[int]]:
    """Independent Hangul arithmetic, rather than the production NFD encoder."""
    clean = []
    for character in unicodedata.normalize("NFC", raw):
        if 0xAC00 <= ord(character) <= 0xD7A3:
            clean.append(character)
        elif character.isspace() or unicodedata.category(character).startswith("P"):
            clean.append(" ")
        else:
            raise ValueError("Unsupported original transcript character")
    text = " ".join("".join(clean).split())
    tokens = []
    for character in text:
        if character == " ":
            tokens.append(68)
            continue
        syllable = ord(character) - 0xAC00
        lead, vowel, tail = syllable // 588, (syllable % 588) // 28, syllable % 28
        tokens.extend([lead + 1, vowel + 20])
        if tail:
            tokens.append(tail + 40)
    return text, tokens


def verify_splits(plan: dict) -> dict:
    records = plan["records"]
    same(len(records), 512, "Full planned sentence count")
    same(len({r["id"] for r in records}), 512, "Unique IDs")
    same(dict(Counter(r["split"] for r in records)), {"dev": 71, "train": 441}, "Split counts")
    source_splits, speakers, speaker_sources = defaultdict(set), set(), defaultdict(set)
    for row in records:
        source_splits[row["source_stem"]].add(row["split"])
        speakers.add(row["speaker_id"])
        speaker_sources[row["speaker_id"]].add(row["source_stem"])
    same(len(source_splits), 59, "Source recordings")
    same(len(speakers), 8, "Speakers")
    if any(len(splits) != 1 for splits in source_splits.values()):
        raise ValueError("Train/development recording leakage")
    same(sum(v == {"dev"} for v in source_splits.values()), 8, "Held-out recordings")
    dev_sources = {r["source_stem"] for r in records if r["split"] == "dev"}
    same(dev_sources, {min(sources) for sources in speaker_sources.values()}, "Predetermined per-speaker first recording split")
    if {r["text"] for r in records if r["split"] == "train"} & {r["text"] for r in records if r["split"] == "dev"}:
        raise ValueError("Normalized transcript leakage")
    expected_pilot = []
    for speaker in sorted(speakers):
        eligible = [r for r in records if r["split"] == "train" and r["speaker_id"] == speaker]
        expected_pilot.append(min(eligible, key=lambda r: (-r["frames"], r["id"]))["id"])
    same(plan["preparation"]["pilot_ids"], expected_pilot, "Eight fixed longest training pilots")
    expected_controls = [min(r["id"] for r in records if r["split"] == "train" and r["speaker_id"] == speaker)
                         for speaker in ("C002", "C004", "C282", "C312")]
    same(plan["preparation"]["reproduction_ids"], expected_controls, "Four predetermined controls")
    return {"train_sentences": 441, "development_sentences": 71, "train_recordings": 51, "development_recordings": 8,
            "speakers": 8, "recording_disjoint": True, "normalized_transcript_disjoint": True,
            "development_same_speakers_as_training": True}


def verify_controls(plan: dict, plan_sha: str, inputs: Path) -> dict:
    gate_path = inputs / "reproduction_gate.json"
    gate = read_json(gate_path)
    same(gate["status"], "passed", "Original pixel gate")
    same(gate["plan_sha256"], plan_sha, "Control plan binding")
    same(gate["ids"], plan["preparation"]["reproduction_ids"], "Control IDs")
    same([r["id"] for r in gate["records"]], gate["ids"], "Control result ordering")
    by_id = {r["id"]: r for r in plan["records"]}
    for control in gate["records"]:
        row = by_id[control["id"]]
        same(sha(row["legacy_cache_path"]), row["legacy_cache_sha256"], "Original cache hash")
        with np.load(local(row["legacy_cache_path"]), allow_pickle=False) as old:
            same(hashlib.sha256(old["frames"].tobytes()).hexdigest(), control["student_rgb_sha256"], "Original pixel hash against gate")
            same(old["frames"].shape, (row["frames"], 96, 96, 3), "Original shape")
            same(old["label"].tolist(), row["token_ids"], "Original label tokens")
            same(str(old["text"].item()), row["text"], "Original text")
        same(control["pixel_exact"], True, "Control pixel status")
        same(control["text_and_tokens_exact"], True, "Control label status")
    return {"records": 4, "gate_path": str(gate_path), "gate_sha256": sha(gate_path),
            "old_cache_pixels_tokens_text_against_saved_reproduction_evidence": True,
            "source_video_decode_independently_repeated": False}


def verify_record(row: dict, entry: dict, plan_sha: str, inputs: Path) -> dict:
    path = local(entry["npz_path"])
    same(path, (inputs / "records" / (row["id"] + ".npz")).resolve(), "Isolated NPZ location")
    same(sha(path), entry["npz_sha256"], "NPZ hash")
    metadata_path = inputs / "metadata" / (row["id"] + ".json")
    same(read_json(metadata_path), entry, "Per-record sidecar equals manifest")
    text, label = tokens_from_syllables(row["source_text"])
    same(text, row["text"], "Independent normalized transcript")
    same(label, row["token_ids"], "Arithmetic Hangul encoding")
    if len(label) + sum(a == b for a, b in zip(label, label[1:])) > row["frames"]:
        raise ValueError("Insufficient student CTC length")
    start, end, fps = row["start_time"], row["end_time"], row["video_fps"]
    near(start, row["original_start_time"] + row["offset_seconds"], 1e-12, "Offset start")
    near(end, row["original_end_time"] + row["offset_seconds"], 1e-12, "Offset end")
    expected_count = round((end - start) * 25)
    same(expected_count, row["frames"], "Student count rounding")
    target_times = [start + k / 25 for k in range(expected_count)]
    expected_indices = np.array([round(t * fps) for t in target_times], dtype=np.int64)
    with np.load(path, allow_pickle=False) as payload:
        same(set(payload.files), {"student_rgb", "teacher_bgr", "student_source_indices", "teacher_source_indices",
                                  "student_teacher_rows", "label_tokens", "metadata_json"}, "Payload keys")
        meta = json.loads(str(payload["metadata_json"].item()))
        same(meta["record"], row, "Payload fixed record")
        same(meta["plan_sha256"], plan_sha, "Payload plan binding")
        same(meta["id"], row["id"], "Payload ID")
        frames = meta["source_frame_count"]
        if not 0 <= start < end <= frames / fps + 1e-6:
            raise ValueError("Corrected interval exceeds physical bounds")
        official_first, official_stop = math.ceil(start * fps - 1e-9), math.ceil(end * fps - 1e-9)
        union_first, union_stop = min(official_first, int(expected_indices[0])), max(official_stop, int(expected_indices[-1]) + 1)
        first, stop = max(0, union_first - 2), min(frames, union_stop + 2)
        dense_indices = np.arange(first, stop, dtype=np.int64)
        mapping = expected_indices - first
        for field, expected in (("student_source_indices", expected_indices), ("teacher_source_indices", dense_indices),
                                ("student_teacher_rows", mapping), ("label_tokens", np.array(label, np.int64))):
            values = payload[field]
            same(values.dtype, np.dtype("int64"), f"{field} dtype")
            np.testing.assert_array_equal(values, expected, err_msg=field)
        np.testing.assert_array_equal(dense_indices[mapping], expected_indices, err_msg="Frame identity")
        for field, count in (("student_rgb", expected_count), ("teacher_bgr", len(dense_indices))):
            pixels = payload[field]
            same(pixels.dtype, np.dtype("uint8"), f"{field} dtype")
            same(pixels.shape, (count, 96, 96, 3), f"{field} shape")
            digest = hashlib.sha256(pixels.tobytes()).hexdigest()
            same(digest, meta[field + "_sha256"], f"{field} embedded hash")
            same(digest, entry[field + "_sha256"], f"{field} manifest hash")
        context = {"requested_real_frames_each_side": 2, "official_first": official_first,
                   "official_stop_exclusive": official_stop, "student_union_first": union_first,
                   "student_union_stop_exclusive": union_stop, "actual_first": first, "actual_stop_exclusive": stop,
                   "clipped_left": first != union_first - 2, "clipped_right": stop != union_stop + 2,
                   "student_left_real_context": np.minimum(2, mapping).tolist(),
                   "student_right_real_context": np.minimum(2, len(dense_indices) - 1 - mapping).tolist(),
                   "boundary_policy": "clip to available native frames; teacher front retains its own configured zero padding"}
        same(meta["teacher_context"], context, "Every student context coverage")
        same(entry["teacher_context"], context, "Manifest context")
        student_trace, teacher_trace = meta["student_trace"]["selected"], meta["teacher_trace"]["selected"]
        same(len(student_trace), expected_count, "Student trace length")
        same(len(teacher_trace), len(dense_indices), "Teacher trace length")
        for index, trace in zip(dense_indices, teacher_trace):
            index = int(index)
            same(trace["source_frame_index"], index, "Dense native frame trace")
            same(trace["annotation_box_index"], index, "Native annotation index")
            near(trace["actual_pts_seconds"], index / fps, .001, "Native frame PTS")
            same(trace["final_rectangle_xyxy"], list(map(int, trace["annotation_box_xyxy"])), "Teacher margin-zero integer rectangle")
            rectangle = trace["final_rectangle_xyxy"]
            if not (0 <= rectangle[0] < rectangle[2] and 0 <= rectangle[1] < rectangle[3]):
                raise ValueError("Invalid teacher rectangle")
            if len(trace["source_bgr_pixel_sha256"]) != 64 or any(c not in "0123456789abcdef" for c in trace["source_bgr_pixel_sha256"]):
                raise ValueError("Invalid raw-source pixel SHA")
        for i, trace in enumerate(student_trace):
            index = int(expected_indices[i])
            same(trace["source_frame_index"], index, "Student chosen frame")
            same(trace["annotation_box_index"], index, "Student annotation index")
            near(trace["target_video_seconds"], target_times[i], 1e-12, "25fps target time")
            near(trace["actual_pts_seconds"], index / fps, .001, "Student frame PTS")
            near(trace["actual_pts_seconds"], target_times[i], .5 / fps + 1e-6, "Nearest source frame tolerance")
            linked = teacher_trace[int(mapping[i])]
            same(trace["source_bgr_pixel_sha256"], linked["source_bgr_pixel_sha256"], "Raw source pixel identity across independent decodes")
            same(trace["annotation_box_xyxy"], linked["annotation_box_xyxy"], "Same-source annotation identity")
        expected_source_check = {"video_size_bytes": row["video_size_bytes"], "video_mtime_ns": row["video_mtime_ns"],
                                 "label_sha256_before_and_after": row["label_sha256"], "source_video_unchanged": True}
        same(meta["source_verification"], expected_source_check, "Original metadata/hash gate")
        same(entry["source_verification"], expected_source_check, "Manifest source gate")
        for key, expected in (("id", row["id"]), ("split", row["split"]), ("source_stem", row["source_stem"]),
                              ("student_frames", expected_count), ("teacher_frames", len(dense_indices)), ("video_fps", fps),
                              ("source_frame_count", frames), ("plan_sha256", plan_sha)):
            same(entry[key], expected, "Manifest " + key)
    return {"id": row["id"], "split": row["split"], "student_frames": expected_count,
            "teacher_frames": len(dense_indices), "context_clipped": context["clipped_left"] or context["clipped_right"],
            "metadata_sha256": sha(metadata_path), "npz_sha256": entry["npz_sha256"],
            "all_student_and_teacher_frame_traces_checked": True}


def audit(plan_path: Path, plan_sha: str, manifest_path: Path, manifest_sha: str, output: Path) -> dict:
    started = time.perf_counter()
    plan = read_json(plan_path, plan_sha)
    manifest = read_json(manifest_path, manifest_sha)
    for item in [*plan["bindings"], *plan["preparation"]["code"]]:
        same(sha(item["path"]), item["sha256"], "Frozen dependency hash")
    splits = verify_splits(plan)
    same(manifest["status"], "passed", "Prepared manifest status")
    same(manifest["plan_sha256"], plan_sha, "Manifest plan binding")
    is_pilot = manifest["phase"] == "pilot"
    expected_ids = plan["preparation"]["pilot_ids"] if is_pilot else [r["id"] for r in plan["records"]]
    same(manifest["ids"], expected_ids, "Whole fixed cohort ordering")
    same([r["id"] for r in manifest["records"]], expected_ids, "Manifest exact record coverage")
    same(manifest["record_count"], len(expected_ids), "Manifest count")
    inputs = local(plan["preparation"]["output_directory"])
    controls = verify_controls(plan, plan_sha, inputs)
    by_id = {r["id"]: r for r in plan["records"]}
    mapping_binding = next(item for item in plan["bindings"] if Path(item["path"]).name == "audited_time_mapping_candidates.json")
    mapping = read_json(mapping_binding["path"], mapping_binding["sha256"])
    sources = {r["source_stem"]: r for r in mapping["sources"]}
    same(len(sources), 59, "Audited mapping source count")
    for row in plan["records"]:
        source = sources[row["source_stem"]]
        same(row["offset_seconds"], source["offset_seconds"], "Audited source offset")
        same(row["label_sha256"], source["original_label_sha256"], "Audited original label hash")
        same(row["video_size_bytes"], source["original_video_size_bytes"], "Audited original video size")
        if row["id"] not in source["training_utterance_ids"]:
            raise ValueError("Selected sentence outside audited source cohort")
    verified = []
    for index, entry in enumerate(manifest["records"], 1):
        verified.append(verify_record(by_id[entry["id"]], entry, plan_sha, inputs))
        if index % 32 == 0 or index == len(expected_ids):
            print(f"[verified] {index}/{len(expected_ids)}", flush=True)
    same(sum(r["student_frames"] for r in verified), manifest["student_frames"], "Student frame total")
    same(sum(r["teacher_frames"] for r in verified), manifest["teacher_frames"], "Teacher frame total")
    result = {"schema": "feature-transfer-step07-independent-input-audit-v1", "status": "passed",
              "created_utc": datetime.now(timezone.utc).isoformat(), "code": {"path": str(Path(__file__)), "sha256": sha(Path(__file__))},
              "plan": {"path": str(local(plan_path)), "sha256": plan_sha},
              "manifest": {"path": str(local(manifest_path)), "sha256": manifest_sha},
              "scope": "pilot8" if is_pilot else "all512", "record_count": len(verified), "split_checks": splits,
              "original_reproduction_checks": controls, "records": verified,
              "student_frame_traces_checked": sum(r["student_frames"] for r in verified),
              "teacher_frame_traces_checked": sum(r["teacher_frames"] for r in verified),
              "context_boundary_clipped_records": sum(r["context_clipped"] for r in verified),
              "source_video_audio_labels_reread": False, "test_payloads_read": False, "elapsed_seconds": time.perf_counter() - started,
              "limitations": ["Checks saved full-frame hashes and crop/index traces; original full source pixels were not independently re-decoded or re-cropped.",
                              "Four original pixel re-extraction successes are checked against the saved original cache; this auditor does not independently repeat source extraction.",
                              "Audio alignment does not establish every transcript or lip/audio synchrony."]}
    output = local(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.plan, args.plan_sha256, args.manifest, args.manifest_sha256, args.output)
    print(json.dumps({k: result[k] for k in ("status", "scope", "record_count", "student_frame_traces_checked", "teacher_frame_traces_checked")}), flush=True)


if __name__ == "__main__":
    main()
