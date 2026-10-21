"""Extract frozen visual frontend targets for exactly the new train candidates.

Only audited candidate NPZs are opened. No audio, ASR, decoder, student model,
optimizer or learning operation is constructed. Old extraction runners are not
called; unchanged normalization/front-loading/forward primitives are reused.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import torch

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
PILOT = ROOT / "experiments/local_pilot_step02_20261007/teacher_pilot.py"
PRIMITIVES = ROOT / "experiments/feature_transfer_step07/teacher_features.py"
PRIOR_PLAN = ROOT / "experiments/local_dataset_step04_20261007/teacher_plan.json"
PLAN = BASE / "teacher_execution_plan.json"
DATA = BASE / "data_plan.json"
INPUTS = BASE / "inputs/manifest.json"
AUDIT = BASE / "results/input_audit.json"
SELECTION_AUDIT = BASE / "selection_audit.json"
FEATURES = BASE / "features"
RECORDS = BASE / "teacher_results"
PROGRESS = BASE / "results/teacher_progress.json"
RESULT = BASE / "results/teacher_manifest.json"
_spec = importlib.util.spec_from_file_location("step09_readonly_pilot_helpers", PILOT)
pilot = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pilot)
teacher = pilot.teacher


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_bound(ref):
    path = teacher.verify_binding(ref)
    value = read(path)
    teacher.verify_binding(ref)
    return value


def publish(path, writer, replace=False):
    path = Path(path).resolve()
    require(path.is_relative_to(BASE), "Output must stay inside Stage9")
    require(replace or not path.exists(), "Refusing to overwrite teacher evidence")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix="." + path.name + ".", suffix=".tmp", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            require(path.name == PROGRESS.name, "Only the progress index is mutable")
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)


def save_json(path, value, replace=False):
    payload = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    publish(path, lambda stream: stream.write(payload), replace)


def validate_cohort(data, manifest, data_ref):
    """Verify all roles without opening any held/development NPZ payload."""
    require(data["schema"] == "lipnet-step09-provisional-data-plan-v1" and data["training_ready"] is False,
            "Unexpected provisional data plan")
    rows = data["records"]
    ids = [row["id"] for row in rows]
    require(len(ids) == len(set(ids)) == 120 and Counter(row["split"] for row in rows) == {"train": 96, "val": 24},
            "Expected original120 train96/val24")
    selected = [row["id"] for row in rows if row["teacher_candidate"] is True]
    require(len(selected) == 69 and selected == data["teacher_candidate_ids"], "Teacher candidate69 selection differs")
    require(all(row["split"] == "train" for row in rows if row["teacher_candidate"]), "Development teacher input forbidden")
    require({row["source_stem"] for row in rows if row["split"] == "train"}.isdisjoint(
            {row["source_stem"] for row in rows if row["split"] == "val"}), "Recording leakage across roles")
    require(manifest["schema"] == "lipnet-step09-input-manifest-v1" and manifest["status"] == "passed"
            and manifest["data_plan"] == data_ref and manifest["ids"] == ids
            and [row["id"] for row in manifest["records"]] == ids, "Input manifest cohort/plan differs")
    inputs = {}
    for row, item in zip(rows, manifest["records"]):
        require(row["id"] and all(char.isalnum() or char == "_" for char in row["id"]), "Unsafe record ID")
        for key in ("id", "split", "source_stem", "speaker_id", "teacher_candidate"):
            require(item[key] == row[key], "Input role identity differs: " + key)
        require(row["teacher_candidate"] is (row["split"] == "train" and row["comparison_group"] == "both_supported"),
                "Candidate policy must be train AND both_supported")
        require(type(row["teacher_candidate"]) is bool and row["training_ready"] is False and item["training_ready"] is False
                and item["plan_sha256"] == data_ref["sha256"] and item["student_frames"] == row["frames"], "Input frame/plan differs")
        require((item["teacher_frames"] > 0) is row["teacher_candidate"], "Held/development input has teacher frames")
        require(("teacher_bgr_sha256" in item) is row["teacher_candidate"], "Teacher pixel metadata outside allowed cohort")
        require(Path(item["npz_path"]).resolve().is_relative_to(BASE / "inputs"), "NPZ outside prepared Stage9 inputs")
        inputs[row["id"]] = item
    result = {"ids": ids, "selected_ids": selected, "rows": dict(zip(ids, rows)), "inputs": inputs,
            "held_train_ids": [row["id"] for row in rows if row["split"] == "train" and not row["teacher_candidate"]],
            "development_ids": [row["id"] for row in rows if row["split"] == "val"]}
    require(data["ids"] == ids and data["held_train_ids"] == result["held_train_ids"]
            and data["development_ids"] == result["development_ids"], "Declared role ID lists differ")
    return result


def validate_audit(value, refs, cohort):
    require(value["schema"] == "lipnet-step09-independent-input-audit-v1" and value["status"] == "passed" and value["data_plan"] == refs["data_plan"]
            and value["input_manifest"] == refs["input_manifest"]
            and value["execution_plan"] == refs["input_execution_plan"]
            and value["ids"] == cohort["ids"] and [row["id"] for row in value["records"]] == cohort["ids"],
            "Independent input audit does not bind this complete cohort")
    for checked in value["records"]:
        sid = checked["id"]
        row, item = cohort["rows"][sid], cohort["inputs"][sid]
        require(checked["teacher_candidate"] is row["teacher_candidate"]
                and checked["student_frames"] == item["student_frames"] and checked["teacher_frames"] == item["teacher_frames"]
                and checked["npz"] == {"path": str(Path(item["npz_path"]).resolve()), "sha256": item["npz_sha256"]},
                "Independent audit does not verify candidate NPZ separation")
    teacher.verify_binding(value["code"])


def evidence():
    data_ref, manifest_ref = teacher.binding(DATA), teacher.binding(INPUTS)
    data, manifest = read_bound(data_ref), read_bound(manifest_ref)
    cohort = validate_cohort(data, manifest, data_ref)
    refs = {"data_plan": data_ref, "input_manifest": manifest_ref,
            "input_execution_plan": manifest["execution_plan"], "input_audit": teacher.binding(AUDIT),
            "selection_audit": teacher.binding(SELECTION_AUDIT), "prior_teacher_plan": teacher.binding(PRIOR_PLAN)}
    selection = read_bound(refs["selection_audit"])
    require(selection["schema"] == "lipnet-step09-independent-selection-audit-v1" and selection["status"] == "passed"
            and selection["data_plan"] == data_ref and selection["ids"] == cohort["ids"]
            and selection["inputs"] == data["inputs"], "Selection audit does not verify this candidate plan")
    teacher.verify_binding(selection["code"])
    for ref in data["inputs"].values():
        teacher.verify_binding(ref)
    execution = read_bound(refs["input_execution_plan"])
    require(execution["data_plan"] == data_ref, "Input execution belongs to another data plan")
    validate_audit(read_bound(refs["input_audit"]), refs, cohort)
    prior = read_bound(refs["prior_teacher_plan"])
    require(prior["primitives"] == teacher.binding(PRIMITIVES) and prior["pilot_runner"] == teacher.binding(PILOT),
            "Previously frozen teacher helper code changed")
    require(prior["checkpoint"]["sha256"] == teacher.EXPECTED_CHECKPOINT_SHA256, "Wrong teacher checkpoint")
    teacher.verify_binding(prior["checkpoint"])
    require(teacher.upstream_contract() == prior["upstream"], "Teacher upstream differs from previous execution")
    return refs, data, manifest, cohort, prior


def read_candidate(cohort, sid, data_ref, input_execution_ref):
    require(sid in cohort["selected_ids"] and cohort["rows"][sid]["split"] == "train"
            and cohort["rows"][sid]["teacher_candidate"] is True, "Teacher may open only eligible train69")
    item, row = cohort["inputs"][sid], cohort["rows"][sid]
    arrays = teacher.read_prepared_npz(Path(item["npz_path"]), item["npz_sha256"], row, data_ref["sha256"])
    require(arrays["metadata"]["execution_plan"] == input_execution_ref and arrays["metadata"]["training_ready"] is False,
            "Candidate metadata execution/approval differs")
    require(arrays["teacher_bgr"].shape[0] == item["teacher_frames"]
            and arrays["metadata"]["teacher_bgr_sha256"] == item["teacher_bgr_sha256"]
            and arrays["metadata"]["student_rgb_sha256"] == item["student_rgb_sha256"], "Candidate pixels differ from manifest")
    return arrays


def freeze():
    require(not PLAN.exists(), "Teacher execution plan already frozen")
    refs, data, manifest, cohort, prior = evidence()
    # Fully validate only the 69 audited candidate payloads before model execution.
    for sid in cohort["selected_ids"]:
        read_candidate(cohort, sid, refs["data_plan"], refs["input_execution_plan"])
    teacher.configure_runtime()
    runtime = teacher.runtime_info()
    require(runtime == prior["runtime"], "Teacher runtime changed from prior frozen FP32 execution")
    value = {"schema": "lipnet-step09-teacher-execution-plan-v1", "status": "frozen_before_feature_extraction",
             "created_utc": datetime.now(timezone.utc).isoformat(), **refs,
             "selected_ids": cohort["selected_ids"], "held_train_ids": cohort["held_train_ids"], "development_ids": cohort["development_ids"],
             "record_count": 69, "all_ids": cohort["ids"], "checkpoint": prior["checkpoint"], "upstream": prior["upstream"], "runtime": runtime,
             "code": [teacher.binding(Path(__file__)), teacher.binding(BASE / "test_teacher_features.py"), teacher.binding(PILOT), teacher.binding(PRIMITIVES)],
             "input_files": [{"id": sid, "path": str(Path(cohort["inputs"][sid]["npz_path"]).resolve()),
                              "sha256": cohort["inputs"][sid]["npz_sha256"]} for sid in cohort["selected_ids"]],
             "contract": {"module": "encoder.front: Resnet2D_front(3,512)", "checkpoint_load": "weights_only=True; strict encoder.front prefix only",
                          "input": "whole native BGR uint8 window; margin0; one FP32 mean/std(correction1) over all pixels; batch1; no temporal split",
                          "output": "FP32 Tstudent x512, exact original-frame gather with repeated student frames preserved",
                          "state": "eval/fully frozen; all parameters AND buffers hashed before/after every forward",
                          "first_forward_repeat": "first candidate repeated bitwise exactly; no tolerances or alternate settings",
                          "failure_policy": "Abort and preserve partial immutable evidence; no silent skip, replacement, OOM fallback or cohort expansion",
                          "resumption": "Only a checksummed fixed ordered prefix; reject orphan/missing NPZ or records; recheck metadata, features, source indices and diagnostics",
                          "scope": "69 train candidates only; no held27 or val24 teacher targets; provisional inputs, no training approval"},
             "audio_used": False, "decoder_instantiated": False, "optimizer_instantiated": False, "optimization_steps": 0,
             "training_ready": False, "labels_modified": False}
    save_json(PLAN, value)
    return value


def check_feature(feature_ref, record_ref, plan_ref, cohort, sid, initial, arrays, plan):
    require(sid in cohort["selected_ids"], "Forbidden teacher target ID")
    for ref in (feature_ref, record_ref):
        teacher.verify_binding(ref)
    record = read_bound(record_ref)
    row, item = cohort["rows"][sid], cohort["inputs"][sid]
    with np.load(feature_ref["path"], allow_pickle=False) as data:
        require(set(data.files) == {"features", "student_source_indices", "teacher_source_indices", "gather_indices", "metadata_json"},
                "Feature payload fields changed")
        meta = json.loads(str(data["metadata_json"].item()))
        require(record == {**meta, "feature_file": feature_ref, "status": "completed"}, "Record/NPZ metadata mismatch")
        require(meta["schema"] == "lipnet-step09-teacher-record-v1" and meta["id"] == sid and meta["split"] == "train"
                and meta["speaker_id"] == row["speaker_id"] and meta["source_stem"] == row["source_stem"]
                and meta["teacher_plan"] == plan_ref and meta["input_record"] == item
                and meta["data_record_sha256"] == teacher.canonical_sha(row), "Feature identity/plan differs")
        for key in ("data_plan", "input_manifest", "input_execution_plan", "input_audit", "selection_audit", "checkpoint", "runtime"):
            require(meta[key] == plan[key], "Feature provenance differs: " + key)
        require(meta["state_sha256_before"] == meta["state_sha256_after"] == initial, "Feature state changed")
        require(meta["audio_used"] is False and meta["decoder_instantiated"] is False and meta["optimizer_instantiated"] is False
                and meta["transcript_used_for_teacher_forward"] is False and meta["training_ready"] is False
                and meta["optimization_steps"] == meta["trainable_parameters"] == 0, "Unexpected model use or approval")
        features = data["features"]
        require(features.dtype == np.float32 and features.shape == (row["frames"], 512)
                and np.isfinite(features).all() and np.all(np.linalg.norm(features, axis=1) > 1e-8), "Malformed target features")
        require(meta["feature_values_sha256"] == hashlib.sha256(features.tobytes()).hexdigest()
                and meta["feature_diagnostics"] == pilot.feature_diagnostics(features), "Feature hash/diagnostics differ")
        require(meta["native_feature_shape"] == [1, item["teacher_frames"], 512]
                and meta["aligned_feature_shape"] == [row["frames"], 512] and meta["feature_dtype"] == "float32", "Feature shapes differ")
        normalized, normalization = teacher.normalize_teacher(arrays["teacher_bgr"])
        require(meta["normalization"] == normalization, "Input whole-window FP32 normalization differs")
        del normalized
        for key in ("student_source_indices", "teacher_source_indices", "gather_indices"):
            require(data[key].dtype == np.int64 and np.array_equal(data[key], arrays[key]), "Source frame/gather changed: " + key)
        repeat = meta["warm_repeat"]
        if sid == cohort["selected_ids"][0]:
            require(isinstance(repeat, dict) and repeat["bitwise_equal"] is True and repeat["max_absolute_difference"] == 0.
                    and repeat["same_input_sha256"] == item["npz_sha256"], "First candidate repeat evidence absent")
        else:
            require(repeat is None, "Unexpected repeat metadata")
    for ref in (feature_ref, record_ref):
        teacher.verify_binding(ref)
    return record


def load_progress(plan_ref, cohort, plan, initial, path=PROGRESS, features=FEATURES, records=RECORDS):
    path, features, records = Path(path), Path(features), Path(records)
    value = read(path) if path.exists() else {"schema": "lipnet-step09-teacher-progress-v1", "teacher_plan": plan_ref, "records": []}
    require(set(value) == {"schema", "teacher_plan", "records"} and value["schema"] == "lipnet-step09-teacher-progress-v1"
            and value["teacher_plan"] == plan_ref, "Wrong progress schema/plan")
    refs = value["records"]
    require(isinstance(refs, list) and len(refs) <= 69 and [ref["id"] for ref in refs] == cohort["selected_ids"][:len(refs)],
            "Progress must be exact candidate ordered prefix")
    require(not features.exists() or {p.name for p in features.glob("*.npz")} == {ref["id"] + ".npz" for ref in refs},
            "Orphan/forbidden/missing feature file; preserve and investigate")
    require(not records.exists() or {p.name for p in records.glob("*.json")} == {ref["id"] + ".json" for ref in refs},
            "Orphan/forbidden/missing feature record; preserve and investigate")
    completed = []
    for ref in refs:
        sid = ref["id"]
        require(set(ref) == {"id", "feature_file", "record_file"}
                and Path(ref["feature_file"]["path"]).resolve() == (features / (sid + ".npz")).resolve()
                and Path(ref["record_file"]["path"]).resolve() == (records / (sid + ".json")).resolve(), "Unexpected progress paths")
        arrays = read_candidate(cohort, sid, plan["data_plan"], plan["input_execution_plan"])
        completed.append(check_feature(ref["feature_file"], ref["record_file"], plan_ref, cohort, sid, initial, arrays, plan))
    return value, completed


def run(expected_plan_sha256):
    require(not RESULT.exists(), "Completed teacher result already exists")
    plan_ref = teacher.binding(PLAN)
    require(isinstance(expected_plan_sha256, str) and plan_ref["sha256"] == expected_plan_sha256.lower(), "Explicit teacher plan SHA differs")
    plan = read_bound(plan_ref)
    require(plan["schema"] == "lipnet-step09-teacher-execution-plan-v1" and plan["status"] == "frozen_before_feature_extraction", "Wrong execution plan")
    refs = plan["code"] + plan["input_files"] + [plan_ref, plan["checkpoint"]]
    refs.extend(plan["upstream"][key] for key in ("frontend_code", "encoder_code", "inference_config"))
    for ref in refs:
        teacher.verify_binding(ref)
    evidence_refs, _, _, cohort, prior = evidence()
    require(all(plan[key] == value for key, value in evidence_refs.items()) and plan["selected_ids"] == cohort["selected_ids"]
            and plan["all_ids"] == cohort["ids"] and plan["checkpoint"] == prior["checkpoint"] and plan["upstream"] == prior["upstream"],
            "Execution inputs/cohort/model differ from frozen plan")
    teacher.configure_runtime()
    require(teacher.runtime_info() == plan["runtime"], "Frozen runtime differs")
    started = time.perf_counter()
    front, checkpoint_contract, initial = teacher.build_frozen_front(plan)
    load_seconds = time.perf_counter() - started
    try:
        progress, completed = load_progress(plan_ref, cohort, plan, initial)
        reused = len(completed)
        if not PROGRESS.exists():
            save_json(PROGRESS, progress)
        for number, sid in enumerate(cohort["selected_ids"][len(completed):], len(completed) + 1):
            item, row = cohort["inputs"][sid], cohort["rows"][sid]
            clip_started = time.perf_counter()
            arrays = read_candidate(cohort, sid, plan["data_plan"], plan["input_execution_plan"])
            video, normalization = teacher.normalize_teacher(arrays["teacher_bgr"])
            video = video.to(device="cuda:0", dtype=torch.float32)
            before = teacher.state_sha256(front)
            require(before == initial and not any(module.training for module in front.modules())
                    and not any(parameter.requires_grad for parameter in front.parameters()), "Teacher state or eval/freeze changed")
            features, timing, native_shape = pilot.measured_forward(front, video, len(arrays["teacher_bgr"]), arrays["gather_indices"])
            require(teacher.state_sha256(front) == before, "Teacher state changed during forward")
            repeat = None
            if number == 1:
                repeated, repeated_timing, repeated_shape = pilot.measured_forward(front, video, len(arrays["teacher_bgr"]), arrays["gather_indices"])
                require(repeated_shape == native_shape and np.array_equal(repeated, features), "First repeated features differ")
                repeat = {"same_input_sha256": item["npz_sha256"], "bitwise_equal": True,
                          "max_absolute_difference": float(np.max(np.abs(repeated - features))), "timing": repeated_timing}
                del repeated
            after = teacher.state_sha256(front)
            require(after == before, "Teacher parameters/buffers changed")
            teacher.verify_binding({"path": item["npz_path"], "sha256": item["npz_sha256"]})
            meta = {"schema": "lipnet-step09-teacher-record-v1", "id": sid, "split": "train",
                    "speaker_id": row["speaker_id"], "source_stem": row["source_stem"], "created_utc": datetime.now(timezone.utc).isoformat(),
                    "teacher_plan": plan_ref, **{key: plan[key] for key in ("data_plan", "input_manifest", "input_execution_plan", "input_audit", "selection_audit", "checkpoint", "runtime")},
                    "input_record": item, "data_record_sha256": teacher.canonical_sha(row), "checkpoint_contract": checkpoint_contract,
                    "state_sha256_before": before, "state_sha256_after": after, "normalization": normalization,
                    "native_feature_shape": native_shape, "aligned_feature_shape": list(features.shape), "feature_dtype": "float32",
                    "feature_values_sha256": hashlib.sha256(features.tobytes()).hexdigest(), "feature_diagnostics": pilot.feature_diagnostics(features),
                    "feature_timing": timing, "warm_repeat": repeat, "processing_seconds_before_save": time.perf_counter() - clip_started,
                    "transcript_used_for_teacher_forward": False, "audio_used": False, "decoder_instantiated": False,
                    "optimizer_instantiated": False, "optimization_steps": 0, "trainable_parameters": 0, "training_ready": False}
            feature_path, record_path = FEATURES / (sid + ".npz"), RECORDS / (sid + ".json")
            publish(feature_path, lambda stream: np.savez_compressed(stream, features=features,
                    student_source_indices=arrays["student_source_indices"], teacher_source_indices=arrays["teacher_source_indices"],
                    gather_indices=arrays["gather_indices"], metadata_json=np.array(json.dumps(meta, ensure_ascii=False, allow_nan=False))))
            feature_ref = teacher.binding(feature_path)
            result = {**meta, "feature_file": feature_ref, "status": "completed"}
            save_json(record_path, result)
            record_ref = teacher.binding(record_path)
            check_feature(feature_ref, record_ref, plan_ref, cohort, sid, initial, arrays, plan)
            progress["records"].append({"id": sid, "feature_file": feature_ref, "record_file": record_ref})
            save_json(PROGRESS, progress, replace=True)
            completed.append(result)
            print(json.dumps({"completed": number, "total": 69, "id": sid, "shape": list(features.shape), "seconds": timing["seconds"]}), flush=True)
            del video, features, arrays
        require(teacher.state_sha256(front) == initial, "Teacher state changed over extraction")
        for ref in refs + list(evidence_refs.values()):
            teacher.verify_binding(ref)
        checked_progress, checked = load_progress(plan_ref, cohort, plan, initial)
        require(checked_progress == progress and checked == completed, "Final immutable feature verification differs")
        summary = {"schema": "lipnet-step09-teacher-manifest-v1", "status": "completed", "created_utc": datetime.now(timezone.utc).isoformat(),
                   "teacher_plan": plan_ref, **evidence_refs, "ids": cohort["selected_ids"], "records": completed, "record_count": 69,
                   "progress": teacher.binding(PROGRESS), "runtime": plan["runtime"], "checkpoint": plan["checkpoint"],
                   "checkpoint_contract": checkpoint_contract, "state_sha256_before": initial, "state_sha256_after": initial,
                   "reused_records": reused, "teacher_load_seconds": load_seconds, "elapsed_seconds": time.perf_counter() - started,
                   "native_frames": sum(row["native_feature_shape"][1] for row in completed),
                   "student_frames": sum(row["aligned_feature_shape"][0] for row in completed),
                   "feature_gpu_seconds_excluding_repeat": sum(row["feature_timing"]["seconds"] for row in completed),
                   "peak_allocated_bytes": max(timing["peak_allocated_bytes"] for row in completed for timing in
                                               [row["feature_timing"], *([row["warm_repeat"]["timing"]] if row["warm_repeat"] else [])]),
                   "feature_file_bytes": sum(Path(row["feature_file"]["path"]).stat().st_size for row in completed),
                   "held_train_ids": cohort["held_train_ids"], "development_ids": cohort["development_ids"],
                   "held_train_teacher_targets_created": False, "development_teacher_targets_created": False,
                   "audio_used": False, "decoder_instantiated": False, "optimizer_instantiated": False,
                   "optimization_steps": 0, "training_ready": False, "labels_modified": False,
                   "limits": ["Visual feature preparation only; this executes no student training or accuracy evaluation",
                              "Candidates were supported by two runs of the same audio model, not certified by human listening",
                              "Feature temporal variation does not prove semantic recognition or lip/audio synchronization"]}
        save_json(RESULT, summary)
        return summary
    finally:
        del front
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", action="store_true")
    group.add_argument("--run", action="store_true")
    parser.add_argument("--expected-plan-sha256")
    args = parser.parse_args()
    value = freeze() if args.freeze else run(args.expected_plan_sha256)
    print(json.dumps({"status": value["status"], "record_count": value["record_count"]}))


if __name__ == "__main__":
    main()
