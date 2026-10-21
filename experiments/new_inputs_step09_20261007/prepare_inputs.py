"""Cache all 120 visual inputs from bounded D: TAR members, without training.

The frozen producer retains its student/teacher crop contract. Only the 69
provisional teacher candidates persist paired pixels; all other records persist
student arrays only. Original text and canonical tokens remain separately bound.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import time

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
OUTPUT = BASE / "inputs"
PLAN = BASE / "input_execution_plan.json"
DATA_PLAN = BASE / "data_plan.json"
STUDENT_KEYS = {"student_rgb", "student_source_indices", "label_tokens"}
PAIRED_KEYS = STUDENT_KEYS | {"teacher_bgr", "teacher_source_indices", "student_teacher_rows"}
PRIMITIVES = {
    "experiments/feature_transfer_step07/prepare_inputs.py": "c003ccc803d559ddf39e796e0a94629bbca64722354536ccf616fb8212c45cf2",
    "experiments/legacy_recheck_step05/prepare_inputs.py": "5b1d0ca08ee42427c5c8973486462dbb1830bb50d83ff1f191d6693aed9b42cd",
    "scripts/prepare_data.py": "35c727bef7057af63734fb774940bb5cd402d00fbb6a18a90ea622a2013ae3e1",
    "backend/preprocessing/roi_extractor.py": "bab3f9e33d40a9661855aa8bc3337d939034a2108e291e362937b7fecbcbe7c5",
    "backend/models/tokenizer.py": "c73b1c393a739cfd0de03d9bd310e1a6d6339949d2add7121f98794fd7020714",
    "backend/models/cache.py": "2ddbbf55018b822fe392daf2dd93f2d89b1164451e95f91be3f3e89438df4d47",
    "backend/config.py": "6faf3cfdde1d50646d3cceb01350be2c2d9e63911f6ad27eeee9de57ce3383ba",
    "experiments/local_source_step01_20261007/tar_source.py": "05f72840576046683d6b5a3c1121a94b7fe2099fd0d85596c30da511801effb2",
}
UPSTREAM_SHA = {
    "source_plan": "5187f0ccbf84ea099d2f52d0a0e8d94db11d75deed50da1ad093e1a032465929",
    "media_manifest": "fc74e5664fbfc835a9b10173334e397be4abe51dfc7dac6d04c871ba30cdfdf8",
    "label_targets": "e791f567ded1860e5baa5935232eb1d8bc32326a27b34e66f5392472143a18f6",
    "review_data": "7efa45a3154ce78f9c11da717966406eb9ce3aef883fc939496b5a5ecfbbaa42",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pixels_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def local(path):
    path = Path(path).resolve()
    require(path.is_relative_to(ROOT), "Evidence must remain in workspace")
    return path


def bind(path):
    path = local(path)
    return {"path": str(path), "sha256": sha(path)}


def read(path):
    return json.loads(local(path).read_text(encoding="utf-8-sig"))


def verify(ref):
    require(isinstance(ref, dict) and set(ref) == {"path", "sha256"}, "Invalid binding")
    require(bind(ref["path"]) == ref, "Evidence hash/path changed: " + str(ref.get("path")))
    return local(ref["path"])


def save_new(path, value):
    path = local(path)
    require(path.is_relative_to(BASE), "Outputs must stay in Step9")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def environment():
    return {"python": platform.python_version(), "executable": str(Path(sys.executable).resolve()),
            "opencv": cv2.__version__, "numpy": np.__version__}


def code_bindings():
    refs = []
    for relative, expected in PRIMITIVES.items():
        ref = bind(ROOT / relative)
        require(ref["sha256"] == expected, "Frozen preprocessing/transport code changed")
        refs.append(ref)
    return [bind(Path(__file__)), bind(BASE / "test_prepare_inputs.py"), *refs]


def load_helpers():
    # Verify every transitive project helper before importing the frozen producer.
    code_bindings()
    loaded = []
    for name, relative in (("step09_frozen_producer", "experiments/feature_transfer_step07/prepare_inputs.py"),
                           ("step09_frozen_transport", "experiments/local_source_step01_20261007/tar_source.py")):
        spec = importlib.util.spec_from_file_location(name, ROOT / relative)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        loaded.append(module)
    producer, transport = loaded
    require((producer.legacy.FPS, producer.legacy.ROI_SIZE, producer.legacy.ROI_MARGIN) == (25, (96, 96), .2),
            "Spatial/temporal contract differs")
    return producer, transport


def validate_data(data, docs):
    require(data.get("schema") == "lipnet-step09-provisional-data-plan-v1", "Wrong data plan")
    rows = data["records"]
    ids = [row["id"] for row in rows]
    require(len(ids) == len(set(ids)) == 120, "Require exactly 120 unique records")
    require(ids == [r["id"] for r in docs["source_plan"]["records"]] == docs["label_targets"]["ids"] ==
            [r["id"] for r in docs["media_manifest"]["clips"]] == [r["id"] for r in docs["review_data"]["records"]],
            "Frozen cohort/order changed")
    require(data["sources"] == docs["source_plan"]["sources"] and len(data["sources"]) == 30, "Source inventory changed")
    sources = {source["stem"]: source for source in data["sources"]}
    require(len(sources) == 30, "Duplicate recording")
    for source in sources.values():
        for role in ("video", "label"):
            archive = Path(source["members"][role]["archive_path"])
            require(archive.drive.upper() == "D:" and archive.suffix.lower() == ".tar", "Only D: TAR sources allowed")
    for row, original, target, clip, review in zip(rows, docs["source_plan"]["records"],
            docs["label_targets"]["records"], docs["media_manifest"]["clips"], docs["review_data"]["records"]):
        require(Path(row["id"]).name == row["id"] and not any(c in row["id"] for c in "/\\:"), "Unsafe ID")
        require(row["split"] == original["split"] == target["split"] == clip["split"] == review["split"], "Role changed")
        require(row["speaker_id"] == original["speaker_id"] == target["speaker_id"], "Speaker changed")
        require(row["source_stem"] == clip["source_stem"] == target["source_stem"], "Recording identity changed")
        require(row["sentence_id"] == original["sentence_id"] and row["source_text"] == original["source_text"], "Original label changed")
        require(row["original_start_time"] == original["start_time"] and
                row["original_end_time"] == original["end_time"], "Original label times changed")
        timing = clip["timing"]
        require(clip["status"] == "prepared" and clip["timing_status"] == "verified" and clip["local_check"]["verified"],
                "Unverified Step6 timing")
        require(row["start_time"] == timing["corrected_start_time"] and row["end_time"] == timing["corrected_end_time"],
                "Corrected times changed")
        require(math.isclose(row["offset_seconds"], row["start_time"] - row["original_start_time"], abs_tol=1e-10, rel_tol=0)
                and math.isclose(row["offset_seconds"], row["end_time"] - row["original_end_time"], abs_tol=1e-10, rel_tol=0),
                "Offset disagrees with verified times")
        require(row["video_fps"] == timing["actual_video_fps"] == sources[row["source_stem"]]["video_fps"] and
                row["frames"] == timing["student_frames"] == original["expected_frames"] == target["student_frames"],
                "FPS/frame count changed")
        require(row["source_frame_count"] == sources[row["source_stem"]]["video_frame_count"] and
                row["student_native_frame_indices"] == timing["native_frame_indices"], "Planned source frames changed")
        require(row["label_sha256"] == original["label_sha256"] == target["label_sha256"], "Label hash changed")
        require(row["text"] == target["canonical_target"] and row["token_ids"] == target["canonical_tokens"] and
                row["ctc_min_frames"] == target["ctc_min_frames"] <= row["frames"] and target["ctc_feasible"], "Canonical label changed")
        allowed = row["split"] == "train" and review["comparison_group"] == "both_supported"
        require(type(row["teacher_candidate"]) is bool and row["teacher_candidate"] is allowed, "Teacher candidate policy changed")
        require(all(row[k] is False for k in ("training_ready", "full_transcript_human_approved", "human_boundary_approved")),
                "Input preparation does not approve transcript/boundary/training")
    require(Counter(r["split"] for r in rows) == {"train": 96, "val": 24}, "Split counts changed")
    require(sum(r["teacher_candidate"] for r in rows) == 69, "Teacher candidates must be the fixed 69")
    return rows


def load_data(expected_sha):
    require(bind(DATA_PLAN)["sha256"] == expected_sha, "Data plan hash changed")
    data = read(DATA_PLAN)
    require(set(UPSTREAM_SHA) | {"step08_summary"} <= set(data["inputs"]), "Missing upstream references")
    docs = {}
    for name, ref in data["inputs"].items():
        path = verify(ref)
        if name in UPSTREAM_SHA:
            require(ref["sha256"] == UPSTREAM_SHA[name], "Different frozen upstream " + name)
        docs[name] = read(path)
    validate_data(data, docs)
    return data, docs


def correspondence(arrays, meta, row, clip):
    """Compare every native index and all three saved Step6 pixel specimens."""
    pixels = arrays["student_rgb"]
    require(pixels.dtype == np.uint8 and pixels.shape == (row["frames"], 96, 96, 3), "Student shape/dtype differs")
    indices = arrays["student_source_indices"]
    require(indices.dtype == np.int64 and indices.tolist() == clip["timing"]["native_frame_indices"], "Step6 native indices differ")
    trace = meta["student_trace"]["selected"]
    require(len(trace) == row["frames"] and [s["source_frame_index"] for s in trace] == indices.tolist(), "Trace frame identity differs")
    samples = clip["roi_samples"]
    require([s["student_index"] for s in samples] == [0, row["frames"] // 2, row["frames"] - 1], "Step6 specimens changed")
    checks = []
    for sample in samples:
        k = sample["student_index"]
        actual_trace = trace[k]
        require(actual_trace["source_frame_index"] == sample["native_frame_index"] and
                actual_trace["source_bgr_pixel_sha256"] == sample["raw_frame_array"]["sha256"], "Step6 raw frame pixels differ")
        require(abs(actual_trace["actual_pts_seconds"] - sample["decoded_pts_seconds"]) <= 1e-9 and
                abs(actual_trace["target_video_seconds"] - sample["target_time_seconds"]) <= 1e-9,
                "Step6 sample timing differs")
        require(np.array_equal(np.asarray(actual_trace["annotation_box_xyxy"], dtype=np.float32),
                               np.asarray(sample["bbox_xyxy"], dtype=np.float32)), "Step6 annotation box differs")
        bgr = np.ascontiguousarray(pixels[k, :, :, ::-1])
        require(pixels_sha(bgr) == sample["roi_array"]["sha256"] == sample["image"]["pixels"]["sha256"], "Step6 ROI pixels differ")
        image_path = local(sample["image"]["path"])
        raw = image_path.read_bytes()
        require(hashlib.sha256(raw).hexdigest() == sample["image"]["sha256"], "Saved Step6 PNG changed")
        decoded = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        require(decoded is not None and np.array_equal(decoded, bgr), "Student RGB and saved BGR specimen differ")
        checks.append({"position": sample["position"], "student_index": k, "native_frame_index": int(indices[k]),
                       "raw_bgr_sha256": actual_trace["source_bgr_pixel_sha256"], "roi_bgr_sha256": pixels_sha(bgr),
                       "png": {"path": str(image_path), "sha256": sample["image"]["sha256"]}})
    return {"native_indices_exact": True, "all_student_frames": len(indices), "sample_count": 3,
            "student_rgb_to_saved_bgr_exact": True, "samples": checks}


def storage_value(value, row):
    value = copy.deepcopy(value)
    if row["teacher_candidate"]:
        require(row["split"] == "train", "Development cannot persist teacher pixels")
        value["meta"]["schema"] = "lipnet-step09-paired-input-v1"
    else:
        value["arrays"] = {key: value["arrays"][key] for key in STUDENT_KEYS}
        for key in ("teacher_bgr_sha256", "teacher_trace", "teacher_context"):
            value["meta"].pop(key, None)
        value["meta"]["schema"] = "lipnet-step09-student-input-v1"
    value["meta"].update(teacher_pixels_persisted=row["teacher_candidate"], training_ready=False,
                         full_transcript_human_approved=False, human_boundary_approved=False,
                         audio_used_as_model_input=False, teacher_model_run=False)
    return value


def verify_payload(path, row, data_ref, execution_ref, clip, producer):
    with np.load(path, allow_pickle=False) as payload:
        expected = PAIRED_KEYS if row["teacher_candidate"] else STUDENT_KEYS
        require(set(payload.files) == expected | {"metadata_json"}, "NPZ fields differ or disallowed teacher arrays exist")
        meta = json.loads(str(payload["metadata_json"].item()))
        arrays = {key: payload[key] for key in expected}
    require(meta["record"] == row and meta["id"] == row["id"] and meta["plan_sha256"] == data_ref["sha256"] and
            meta["data_plan"] == data_ref and meta["execution_plan"] == execution_ref, "NPZ plan/identity mismatch")
    require(meta["teacher_pixels_persisted"] is row["teacher_candidate"] and
            all(meta[k] is False for k in ("training_ready", "full_transcript_human_approved", "human_boundary_approved",
                                          "audio_used_as_model_input", "teacher_model_run")), "Cache readiness/scope changed")
    require(arrays["label_tokens"].dtype == np.int64 and arrays["label_tokens"].tolist() == row["token_ids"], "Canonical tokens changed")
    require(pixels_sha(arrays["student_rgb"]) == meta["student_rgb_sha256"], "Student pixel hash differs")
    contract = producer.interval_contract(row["start_time"], row["end_time"], row["video_fps"], meta["source_frame_count"])
    require(np.array_equal(arrays["student_source_indices"], contract["student_indices"]), "Student temporal contract differs")
    if row["teacher_candidate"]:
        producer.verify_payload(path, row, data_ref["sha256"])
    else:
        require(not any(k in meta for k in ("teacher_trace", "teacher_context", "teacher_bgr_sha256")), "Teacher metadata persisted for held/dev row")
    checked = correspondence(arrays, meta, row, clip)
    require(meta["step06_correspondence"] == checked, "Saved Step6 correspondence differs")
    return meta


def metadata_record(path, meta):
    row = meta["record"]
    result = {"id": row["id"], "split": row["split"], "speaker_id": row["speaker_id"], "source_stem": row["source_stem"],
              "plan_sha256": meta["plan_sha256"], "npz_path": str(Path(path).resolve()), "npz_sha256": sha(path),
              "student_frames": row["frames"], "teacher_frames": len(meta["teacher_trace"]["selected"]) if row["teacher_candidate"] else 0,
              "teacher_candidate": row["teacher_candidate"], "student_rgb_sha256": meta["student_rgb_sha256"],
              "source_frame_count": meta["source_frame_count"], "video_fps": row["video_fps"],
              "step06_correspondence": meta["step06_correspondence"], "source_verification": meta["source_verification"],
              "preparation_seconds": meta["preparation_seconds"], "training_ready": False}
    if row["teacher_candidate"]:
        result["teacher_bgr_sha256"] = meta["teacher_bgr_sha256"]
    return result


def transport_evidence(source, data_ref, execution_ref):
    return {"transport": "audited-local-tar-v1", "source_stem": source["stem"],
            "data_plan": data_ref, "execution_plan": execution_ref,
            "members": {role: source["members"][role] for role in ("video", "label")},
            "archive_stats_checked_before_and_after": True, "member_hashes_checked_on_materialization": True,
            "temporary_payload_cleaned_before_publish": True, "temporary_paths_persisted": False,
            "audio_used_as_model_input": False, "G_drive_accessed": False}


def saved_record(row, source, data_ref, execution_ref, clip, producer):
    path = OUTPUT / "records" / (row["id"] + ".npz")
    sidecar = OUTPUT / "metadata" / (row["id"] + ".json")
    if not path.exists():
        require(not sidecar.exists(), "Sidecar exists but payload is missing")
        return None
    meta = verify_payload(path, row, data_ref, execution_ref, clip, producer)
    require(meta["source_verification"] == transport_evidence(source, data_ref, execution_ref), "Saved source provenance changed")
    result = metadata_record(path, meta)
    if sidecar.exists():
        require(read(sidecar) == result, "Saved sidecar/NPZ hash differs")
    else:
        # An immutable, self-describing NPZ can recover an interrupted publication.
        save_new(sidecar, result)
    return result


def write_record(row, value, data_ref, execution_ref, clip, producer):
    path = OUTPUT / "records" / (row["id"] + ".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(f".pending-{os.getpid()}.npz")
    with pending.open("xb") as stream:
        np.savez_compressed(stream, **value["arrays"], metadata_json=json.dumps(value["meta"], ensure_ascii=False, allow_nan=False))
    verify_payload(pending, row, data_ref, execution_ref, clip, producer)
    require(not path.exists(), "Existing payload cannot be overwritten")
    pending.rename(path)
    result = metadata_record(path, value["meta"])
    save_new(OUTPUT / "metadata" / (row["id"] + ".json"), result)
    return result


def freeze(expected_data_sha):
    data, _ = load_data(expected_data_sha)
    require((cv2.__version__, np.__version__) == ("5.0.0", "2.3.5"), "Use existing .venv-tokenizer preprocessing runtime")
    code = code_bindings()
    load_helpers()
    plan = {"schema": "lipnet-step09-input-execution-plan-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
            "frozen_before_input_extraction": True, "data_plan": bind(DATA_PLAN), "code": code,
            "environment": environment(), "source_order": sorted(s["stem"] for s in data["sources"]),
            "ids": [r["id"] for r in data["records"]], "teacher_candidate_ids": [r["id"] for r in data["records"] if r["teacher_candidate"]],
            "output_directory": str(OUTPUT), "temporary_directory": str(BASE / "temporary"), "minimum_free_bytes": 10 * 1024**3,
            "rules": {"source_roles": ["video", "label"], "materialize_recordings_at_once": 1,
                      "student": "unchanged RGB96, 20 percent bbox margin, 25fps rounding",
                      "teacher_candidates": "train and frozen Step7 both_supported only, native BGR96 margin0, two context frames",
                      "student_only": "remaining 27 train and all 24 val; teacher arrays/traces discarded before saving",
                      "verification": "all native frame IDs and Step6 first/middle/last raw+ROI pixel equality for all120",
                      "training_ready": False, "student_optimization_steps": 0, "teacher_model_runs": 0}}
    save_new(PLAN, plan)
    print(json.dumps({"execution_plan": bind(PLAN), "records": 120, "teacher_candidates": 69}, ensure_ascii=False))
    return plan


def run(expected_plan_sha):
    require(bind(PLAN)["sha256"] == expected_plan_sha, "Execution plan changed")
    plan, execution_ref = read(PLAN), bind(PLAN)
    require(plan["schema"] == "lipnet-step09-input-execution-plan-v1" and plan["code"] == code_bindings() and
            plan["environment"] == environment(), "Frozen code/environment differs")
    require(plan["output_directory"] == str(OUTPUT) and plan["temporary_directory"] == str(BASE / "temporary") and
            plan["minimum_free_bytes"] == 10 * 1024**3, "Output/temp/reserve policy changed")
    data, docs = load_data(plan["data_plan"]["sha256"])
    require(bind(DATA_PLAN) == plan["data_plan"], "Execution data binding differs")
    producer, transport = load_helpers()
    data_ref = plan["data_plan"]
    clips = {r["id"]: r for r in docs["media_manifest"]["clips"]}
    sources = {s["stem"]: s for s in data["sources"]}
    require(plan["source_order"] == sorted(sources) and plan["ids"] == [r["id"] for r in data["records"]] and
            plan["teacher_candidate_ids"] == [r["id"] for r in data["records"] if r["teacher_candidate"]], "Execution cohort differs")
    allowed_files = {r["id"] + ".npz" for r in data["records"]}
    if (OUTPUT / "records").exists():
        require(all(p.name in allowed_files for p in (OUTPUT / "records").glob("*.npz")), "Unknown or pending payload; inspect before resuming")
    groups, completed = defaultdict(list), {}
    for row in data["records"]:
        groups[row["source_stem"]].append(row)
    started = time.perf_counter()
    progress_path = OUTPUT / "progress.json"
    prior = read(progress_path) if progress_path.exists() else None
    prior_records = {}
    if prior:
        require(prior["execution_plan"] == execution_ref and prior["data_plan"] == data_ref, "Progress plan differs")
        prior_records = {r["id"]: r for r in prior["records"]}
        require(len(prior_records) == len(prior["records"]) and set(prior_records) <= set(plan["ids"]), "Unknown/duplicate progress IDs")
    copied_bytes, sources_this_run = 0, 0
    for stem in plan["source_order"]:
        source, rows = sources[stem], groups[stem]
        pending = []
        for row in rows:
            found = saved_record(row, source, data_ref, execution_ref, clips[row["id"]], producer)
            if row["id"] in prior_records:
                require(found == prior_records[row["id"]], "Previously accepted artifact is missing or changed")
            if found is None:
                pending.append(row)
            else:
                completed[row["id"]] = found
        if pending:
            requested = copy.deepcopy(source)
            requested["members"] = {k: requested["members"][k] for k in ("video", "label")}
            # No result is published until context exit verifies archives and cleans the owned temporary directory.
            with transport.materialize_source(requested, BASE / "temporary", minimum_free_bytes=plan["minimum_free_bytes"]) as paths:
                stat = paths["video"].stat()
                execution_rows = [dict(row, video_path=str(paths["video"]), label_path=str(paths["label"]),
                                       video_size_bytes=stat.st_size, video_mtime_ns=stat.st_mtime_ns) for row in pending]
                values = producer.prepare_source(execution_rows, data_ref["sha256"], reproduction=False)
                require([r["id"] for r, _ in values] == [r["id"] for r in pending], "Producer changed IDs/order")
                prepared = []
                for logical, (_, value) in zip(pending, values):
                    value["meta"].update(record=logical, data_plan=data_ref, execution_plan=execution_ref)
                    require(value["meta"]["source_frame_count"] == source["video_frame_count"], "Source frame count changed")
                    value["meta"]["step06_correspondence"] = correspondence(value["arrays"], value["meta"], logical, clips[logical["id"]])
                    value["meta"]["source_verification"] = transport_evidence(source, data_ref, execution_ref)
                    prepared.append((logical, storage_value(value, logical)))
            for row, value in prepared:
                completed[row["id"]] = write_record(row, value, data_ref, execution_ref, clips[row["id"]], producer)
            copied_bytes += sum(member["size"] for member in requested["members"].values())
            sources_this_run += 1
        progress = {"schema": "lipnet-step09-input-progress-v1", "data_plan": data_ref, "execution_plan": execution_ref,
                    "records": [completed[r["id"]] for r in data["records"] if r["id"] in completed],
                    "complete_recordings": [name for name in plan["source_order"] if all(r["id"] in completed for r in groups[name])],
                    "training_ready": False}
        producer.json_progress(progress_path, progress)
        print(json.dumps({"source": stem, "completed": len(completed), "of": 120}), flush=True)
    require(plan["code"] == code_bindings(), "Code changed during preparation")
    temporary = BASE / "temporary"
    require(not temporary.exists() or not any(temporary.rglob("*")), "Temporary source payload remains")
    records = [completed[row["id"]] for row in data["records"]]
    manifest = {"schema": "lipnet-step09-input-manifest-v1", "status": "passed", "created_utc": datetime.now(timezone.utc).isoformat(),
                "data_plan": data_ref, "execution_plan": execution_ref, "ids": plan["ids"], "record_count": 120, "records": records,
                "student_frames": sum(r["student_frames"] for r in records), "teacher_frames": sum(r["teacher_frames"] for r in records),
                "paired_input_count": sum(r["teacher_candidate"] for r in records), "student_only_count": sum(not r["teacher_candidate"] for r in records),
                "source_recordings": 30, "sources_materialized_this_run": sources_this_run, "copied_payload_bytes_this_run": copied_bytes,
                "input_npz_bytes": sum(Path(r["npz_path"]).stat().st_size for r in records),
                "roi_pixel_comparisons_passed": sum(r["step06_correspondence"]["sample_count"] for r in records),
                "elapsed_seconds_this_run": time.perf_counter() - started, "temporary_payload_files_remaining": 0,
                "training_ready": False, "student_optimization_steps": 0, "teacher_model_runs": 0,
                "audio_used_as_model_input": False, "G_drive_accessed": False}
    destination = OUTPUT / "manifest.json"
    if destination.exists():
        previous = read(destination)
        require(previous["records"] == records and previous["data_plan"] == data_ref and previous["execution_plan"] == execution_ref,
                "Completed manifest changed")
        print(json.dumps({"status": "already_completed_verified", "manifest": bind(destination)}))
        return previous
    save_new(destination, manifest)
    print(json.dumps({"manifest": bind(destination), "records": 120, "paired": 69, "roi_checks": 360}, ensure_ascii=False))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("freeze").add_argument("--data-plan-sha", required=True)
    commands.add_parser("run").add_argument("--expected-plan-sha", required=True)
    args = parser.parse_args()
    freeze(args.data_plan_sha) if args.command == "freeze" else run(args.expected_plan_sha)


if __name__ == "__main__":
    main()
