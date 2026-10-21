"""Read frozen old/new cached artifacts for the Step10 expansion experiment.

No model, source media, audio, held-training NPZ, or cache mutation is used.
All development records remain available and receive no teacher-feature reads.
The returned split describes the loader role; source_split preserves dev/val.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
STEP9 = ROOT / "experiments/new_inputs_step09_20261007"
HANDOFF = {"path": str(STEP9 / "candidate_handoff.json"),
           "sha256": "8135c7900ad5ad244e98ddb490fad2a05db115b221edba957cb030603a224ae6"}
OLD_LOADER = {"path": str(ROOT / "experiments/local_training_step05_20261007/full_artifacts.py"),
              "sha256": "c11e823ebeb6775f209a26b89283724c3f2611cac9e1351bd7d2a8f74cff8cca"}
STATE_SHA = "08fda31cd2a8ed5c1fe021da7d547229189d47d6f9df99e82cc276524ca19316"
STUDENT_KEYS = {"student_rgb", "student_source_indices", "label_tokens", "metadata_json"}
PAIRED_KEYS = STUDENT_KEYS | {"teacher_bgr", "teacher_source_indices", "student_teacher_rows"}
FEATURE_KEYS = {"features", "student_source_indices", "teacher_source_indices", "gather_indices", "metadata_json"}
_OLD = None


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def pixel_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def old_loader():
    global _OLD
    require(sha(OLD_LOADER["path"]) == OLD_LOADER["sha256"], "Frozen Step5 loader changed")
    if _OLD is None:
        spec = importlib.util.spec_from_file_location("step10_frozen_full_artifacts", OLD_LOADER["path"])
        _OLD = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_OLD)
    return _OLD


def unique(records):
    lookup = {r["id"]: r for r in records}
    require(len(lookup) == len(records), "Duplicate artifact record")
    return lookup


def validate_handoff(handoff, docs):
    require(handoff["schema"] == "lipnet-step09-provisional-candidate-handoff-v1" and
            handoff["status"] == "verified_data_available_pending_new_training_execution_plan", "Wrong Step9 handoff")
    refs = handoff["artifacts"]
    old, new = docs["old_data_plan"]["records"], docs["new_data_plan"]["records"]
    rows = unique(old + new)
    entries = unique(handoff["records"])
    require(len(rows) == len(entries) == 632 and list(rows) == list(entries), "Handoff cohort/order changed")
    inputs = {generation: unique(docs[generation + "_inputs"]["records"]) for generation in ("old", "new")}
    features = {generation: unique(docs[generation + "_features"]["records"]) for generation in ("old", "new")}
    groups = {"control_old441": [], "expanded510": [], "old71": [], "new24": [], "held27": []}
    for generation, logical in (("old", old), ("new", new)):
        require(list(inputs[generation]) == [r["id"] for r in logical], "Input inventory/order changed")
        for row in logical:
            sid = row["id"]
            if generation == "old":
                role = "old_train" if row["split"] == "train" else "old_development"
                require(row["split"] in {"train", "dev"}, "Unexpected old split")
            else:
                require(row["split"] in {"train", "val"}, "Unexpected new split")
                require(type(row["teacher_candidate"]) is bool and
                        not (row["split"] == "val" and row["teacher_candidate"]), "Development cannot be teacher candidate")
                role = "new_provisional_train" if row["teacher_candidate"] else "new_development" if row["split"] == "val" else "new_train_hold"
            item, entry = inputs[generation][sid], entries[sid]
            candidate = role in {"old_train", "new_provisional_train"}
            require(entry["role"] == role and entry["generation"] == generation and entry["original_split"] == row["split"], "Handoff role changed")
            require(entry["speaker_id"] == row["speaker_id"] and entry["source_stem"] == row["source_stem"] and
                    entry["student_frames"] == row["frames"] and entry["logical_record_sha256"] == canonical_sha(row), "Logical record changed")
            require(entry["logical_data_plan"] == refs[generation + "_data_plan"] and
                    entry["input_manifest"] == refs[generation + "_inputs"] and
                    entry["input"] == {"path": item["npz_path"], "sha256": item["npz_sha256"]}, "Input provenance mismatch")
            require(item["split"] == row["split"] and item["student_frames"] == row["frames"], "Input role/length changed")
            require(entry["provisional_training_candidate"] is candidate and entry["new_human_approval_added"] is False,
                    "Provisional scope changed")
            require((sid in features[generation]) is candidate, "Feature/role contamination")
            if candidate:
                require(entry["teacher_feature"] == features[generation][sid]["feature_file"] and
                        entry["teacher_feature_manifest"] == refs[generation + "_features"], "Feature provenance mismatch")
                groups["expanded510"].append(sid)
                if generation == "old":
                    groups["control_old441"].append(sid)
            else:
                require(entry["teacher_feature"] is None and entry["teacher_feature_manifest"] is None and
                        item["teacher_frames"] == 0, "Nontraining role contains teacher artifact")
                groups[{"old_development": "old71", "new_development": "new24", "new_train_hold": "held27"}[role]].append(sid)
    require({key: len(ids) for key, ids in groups.items()} ==
            {"control_old441": 441, "expanded510": 510, "old71": 71, "new24": 24, "held27": 27}, "Group counts changed")
    require(set(features["old"]) == set(groups["control_old441"]) and
            set(features["new"]) == set(groups["expanded510"]) - set(groups["control_old441"]), "Unexpected extra feature record")
    require(handoff["provisional_train_ids"] == groups["expanded510"] and
            handoff["control_old_train_ids"] == groups["control_old441"] and handoff["held_new_train_ids"] == groups["held27"], "Declared cohort differs")
    declared = handoff["development_groups"]
    require(declared["old71_same_training_speakers"] == groups["old71"] and
            declared["new24_all_original_items"] == groups["new24"], "Development cohort changed")
    actual_overlap = []
    for sid in groups["expanded510"]:
        for did in groups["new24"]:
            if rows[sid]["text"].replace(" ", "") == rows[did]["text"].replace(" ", ""):
                actual_overlap.append({"train_id": sid, "val_id": did, "train_text": rows[sid]["text"], "val_text": rows[did]["text"]})
    require(actual_overlap == handoff["text_overlap"] and len(actual_overlap) == 1, "Text overlap evidence changed")
    excluded = {r["val_id"] for r in actual_overlap}
    supplement = [sid for sid in groups["new24"] if sid not in excluded]
    require(supplement == declared["new23_supplement_excluding_training_text_equal_without_spaces"] and len(supplement) == 23,
            "Supplemental development cohort changed")
    train_speakers = {rows[sid]["speaker_id"] for sid in groups["expanded510"]}
    new_dev_speakers = {rows[sid]["speaker_id"] for sid in groups["new24"]}
    require(len(train_speakers) == 20 and len(new_dev_speakers) == 3 and train_speakers.isdisjoint(new_dev_speakers), "New-development speaker leakage")
    require(handoff["training_ready"] is False and handoff["new_human_approvals"] == 0, "Prior review status changed")
    return {"rows": rows, "entries": entries, "inputs": inputs, "features": features, "groups": groups,
            "supplemental_ids": supplement, "excluded_duplicate_ids": [sid for sid in groups["new24"] if sid in excluded],
            "overlap_evidence": actual_overlap}


def _handoff(checked):
    handoff = checked.read(HANDOFF)
    checked.verify(OLD_LOADER)
    checked.verify({"path": str(Path(__file__).resolve()), "sha256": sha(__file__)})
    checked.verify(handoff["generator"])
    refs = handoff["artifacts"]
    require(all(Path(ref["path"]).suffix == ".json" for ref in refs.values()), "Metadata-only handoff has non-JSON artifact")
    docs = {key: checked.read(ref) for key, ref in refs.items()}
    cohort = validate_handoff(handoff, docs)
    for key in ("selection_audit", "input_audit", "feature_audit"):
        require(docs[key]["status"] == "passed" and docs[key]["data_plan"] == refs["new_data_plan"], "Unbound or failed new-data audit")
        checked.verify(docs[key]["code"])
    require(docs["input_audit"]["input_manifest"] == refs["new_inputs"] and
            docs["input_audit"]["execution_plan"] == refs["input_execution_plan"], "Input audit binding differs")
    require(docs["feature_audit"]["teacher_manifest"] == refs["new_features"] and
            docs["feature_audit"]["teacher_plan"] == refs["teacher_execution_plan"], "Feature audit binding differs")
    require(docs["new_inputs"]["status"] == "passed" and docs["new_inputs"]["data_plan"] == refs["new_data_plan"] and
            docs["new_inputs"]["execution_plan"] == refs["input_execution_plan"], "New input manifest differs")
    feature_doc, teacher_plan = docs["new_features"], docs["teacher_execution_plan"]
    require(feature_doc["status"] == "completed" and feature_doc["teacher_plan"] == refs["teacher_execution_plan"], "New features incomplete")
    for key, wanted in (("data_plan", refs["new_data_plan"]), ("input_manifest", refs["new_inputs"]),
                        ("input_execution_plan", refs["input_execution_plan"]), ("input_audit", refs["input_audit"]),
                        ("selection_audit", refs["selection_audit"])):
        require(feature_doc[key] == teacher_plan[key] == wanted, "New teacher binding differs: " + key)
    require(feature_doc["state_sha256_before"] == feature_doc["state_sha256_after"] == STATE_SHA, "Teacher state changed")
    require(feature_doc["held_train_teacher_targets_created"] is False and feature_doc["development_teacher_targets_created"] is False,
            "Forbidden teacher targets created")
    for ref in docs["input_execution_plan"]["code"] + teacher_plan["code"]:
        checked.verify(ref)
    return {"handoff": handoff, "documents": docs, "refs": refs, **cohort}


def verify_handoff():
    """Verify metadata/code and all632 roles, without opening any NPZ payload."""
    checked = old_loader().LocalBindings()
    bundle = _handoff(checked)
    bundle["bindings"] = checked.finish()
    bundle["handoff_ref"] = dict(HANDOFF)
    return bundle


def evaluation_metadata():
    """Expose frozen development IDs and duplicate evidence without pixel I/O."""
    bundle = verify_handoff()
    return {"old71_ids": list(bundle["groups"]["old71"]), "new24_ids": list(bundle["groups"]["new24"]),
            "new23_ids": list(bundle["supplemental_ids"]), "excluded_duplicate_ids": list(bundle["excluded_duplicate_ids"]),
            "overlap_evidence": bundle["overlap_evidence"], "bindings": bundle["bindings"]}


def validate_new_student(row, item, arrays, metadata, bundle, scope):
    candidate = scope == "train"
    require(scope in {"train", "dev"} and row["split"] == ("train" if candidate else "val") and
            row["teacher_candidate"] is candidate, "Wrong or held role requested")
    refs = bundle["refs"]
    require(metadata["record"] == row and metadata["id"] == row["id"] and
            metadata["plan_sha256"] == refs["new_data_plan"]["sha256"] and metadata["data_plan"] == refs["new_data_plan"] and
            metadata["execution_plan"] == refs["input_execution_plan"], "Input embedded logical record/plan differs")
    require(metadata["schema"] == ("lipnet-step09-paired-input-v1" if candidate else "lipnet-step09-student-input-v1") and
            metadata["teacher_pixels_persisted"] is candidate, "Input schema/scope differs")
    require(all(metadata[k] is False for k in ("training_ready", "full_transcript_human_approved", "human_boundary_approved",
                                               "audio_used_as_model_input", "teacher_model_run")), "Input preparation approval/scope changed")
    require(metadata["source_verification"] == item["source_verification"], "Input source provenance differs")
    pixels = arrays["student_rgb"]
    require(pixels.dtype == np.uint8 and pixels.shape == (row["frames"], 96, 96, 3), "Invalid student pixels")
    require(pixel_sha(pixels) == metadata["student_rgb_sha256"] == item["student_rgb_sha256"], "Student pixels changed")
    tokens = row["token_ids"]
    require(tokens and all(type(t) is int and 1 <= t < 69 for t in tokens) and
            len(tokens) + sum(a == b for a, b in zip(tokens, tokens[1:])) <= row["frames"], "Invalid/CTC-infeasible canonical tokens")
    wanted = np.asarray([round((row["start_time"] + k / 25) * row["video_fps"]) for k in range(row["frames"])], np.int64)
    require(wanted.tolist() == row["student_native_frame_indices"] and
            metadata["source_frame_count"] == row["source_frame_count"] and wanted[0] >= 0 and wanted[-1] < row["source_frame_count"],
            "Source-frame contract differs")
    for key, expected in (("student_source_indices", wanted), ("label_tokens", np.asarray(tokens, np.int64))):
        require(arrays[key].dtype == np.int64 and np.array_equal(arrays[key], expected), "Student frame/token arrays changed")
    trace = metadata["student_trace"]["selected"]
    require(len(trace) == row["frames"] and [r["source_frame_index"] for r in trace] == wanted.tolist(), "Student trace frame mapping differs")
    require(metadata["step06_correspondence"] == item["step06_correspondence"] and
            metadata["step06_correspondence"]["sample_count"] == 3, "Prior sample correspondence differs")
    if not candidate:
        require(item["teacher_frames"] == 0 and not any(k in metadata for k in ("teacher_trace", "teacher_context", "teacher_bgr_sha256")),
                "Development has teacher metadata")


def validate_new_target(row, item, feature, arrays, payload, metadata, bundle):
    require(row["split"] == "train" and row["teacher_candidate"] is True, "Held/development teacher target forbidden")
    refs, plan = bundle["refs"], bundle["documents"]["teacher_execution_plan"]
    require({**metadata, "feature_file": feature["feature_file"], "status": "completed"} == feature,
            "Feature embedded metadata differs from manifest")
    for key, expected in (("id", row["id"]), ("split", "train"), ("speaker_id", row["speaker_id"]),
                          ("source_stem", row["source_stem"]), ("input_record", item),
                          ("data_record_sha256", canonical_sha(row)), ("teacher_plan", refs["teacher_execution_plan"]),
                          ("state_sha256_before", STATE_SHA), ("state_sha256_after", STATE_SHA),
                          ("aligned_feature_shape", [row["frames"], 512]), ("feature_dtype", "float32"),
                          ("optimization_steps", 0), ("trainable_parameters", 0)):
        require(metadata[key] == expected, "Feature contract differs: " + key)
    for key in ("data_plan", "input_manifest", "input_execution_plan", "input_audit", "selection_audit", "checkpoint", "runtime"):
        require(metadata[key] == plan[key], "Feature evidence changed: " + key)
    require(all(metadata[k] is False for k in ("audio_used", "decoder_instantiated", "optimizer_instantiated",
                                               "transcript_used_for_teacher_forward", "training_ready")), "Feature scope differs")
    features = payload["features"]
    require(features.dtype == np.float32 and features.shape == (row["frames"], 512) and np.isfinite(features).all(), "Invalid feature shape/dtype/values")
    require(np.all(np.linalg.norm(features.astype(np.float64), axis=1) > 1e-8) and not np.all(features == features[0]), "Collapsed teacher features")
    require(pixel_sha(features) == metadata["feature_values_sha256"], "Teacher features changed")
    for key, source_key in (("student_source_indices", "student_source_indices"), ("teacher_source_indices", "teacher_source_indices"),
                            ("gather_indices", "student_teacher_rows")):
        require(payload[key].dtype == np.int64 and arrays[source_key].dtype == np.int64 and
                np.array_equal(payload[key], arrays[source_key]), "Feature/input original-frame mapping changed")
    first = max(0, min(math.ceil(row["start_time"] * row["video_fps"] - 1e-9), int(arrays["student_source_indices"][0])) - 2)
    stop = min(row["source_frame_count"], max(math.ceil(row["end_time"] * row["video_fps"] - 1e-9),
                                            int(arrays["student_source_indices"][-1]) + 1) + 2)
    dense, gather = payload["teacher_source_indices"], payload["gather_indices"]
    require(np.array_equal(dense, np.arange(first, stop, dtype=np.int64)) and gather.shape == (row["frames"],) and
            np.all(gather >= 0) and np.all(gather < len(dense)) and
            np.array_equal(dense[gather], arrays["student_source_indices"]), "Teacher native context/gather differs")
    require(metadata["native_feature_shape"] == [1, len(dense), 512] and item["teacher_frames"] == len(dense), "Native teacher shape changed")


def _load_new(bundle, ids, scope, checked):
    allowed = [sid for sid in bundle["groups"]["expanded510"] if bundle["entries"][sid]["generation"] == "new"] if scope == "train" else bundle["groups"]["new24"]
    require(scope in {"train", "dev"} and ids == allowed, "Only complete fixed new69/new24 scopes may be opened")
    input_audits = unique(bundle["documents"]["input_audit"]["records"])
    feature_audits = unique(bundle["documents"]["feature_audit"]["records"])
    clips = {}
    for sid in ids:
        row, item = bundle["rows"][sid], bundle["inputs"]["new"][sid]
        require(row["teacher_candidate"] is (scope == "train") and row["split"] == ("train" if scope == "train" else "val"), "Forbidden payload role")
        input_ref = {"path": item["npz_path"], "sha256": item["npz_sha256"]}
        require(input_audits[sid]["npz"] == input_ref, "Input audit references another NPZ")
        path = checked.verify(input_ref)
        require(path == STEP9 / "inputs/records" / (sid + ".npz"), "New input path changed")
        require(checked.read(input_audits[sid]["metadata"]) == item, "Input sidecar differs from audited manifest")
        with np.load(path, allow_pickle=False) as saved:
            require(set(saved.files) == (PAIRED_KEYS if scope == "train" else STUDENT_KEYS), "Disallowed input arrays")
            keys = STUDENT_KEYS - {"metadata_json"}
            if scope == "train":
                keys |= {"teacher_source_indices", "student_teacher_rows"}
            arrays = {key: saved[key] for key in keys}
            metadata = json.loads(str(saved["metadata_json"].item()))
        validate_new_student(row, item, arrays, metadata, bundle, scope)
        clip = {"id": sid, "split": scope, "source_split": row["split"], "speaker_id": row["speaker_id"],
                "source_stem": row["source_stem"], "text": row["text"], "pixels": arrays["student_rgb"], "tokens": list(row["token_ids"])}
        if scope == "train":
            feature = bundle["features"]["new"][sid]
            require(feature_audits[sid]["feature_file"] == feature["feature_file"], "Feature audit references another payload")
            path = checked.verify(feature["feature_file"])
            require(path == STEP9 / "features" / (sid + ".npz"), "New feature path changed")
            with np.load(path, allow_pickle=False) as saved:
                require(set(saved.files) == FEATURE_KEYS, "Unexpected feature fields")
                payload = {key: saved[key] for key in FEATURE_KEYS - {"metadata_json"}}
                feature_metadata = json.loads(str(saved["metadata_json"].item()))
            validate_new_target(row, item, feature, arrays, payload, feature_metadata, bundle)
            clip["teacher_features"] = payload["features"]
        clips[sid] = clip
    return clips


def merge_bindings(*groups):
    result = {}
    for group in groups:
        for ref in group:
            require(ref["path"] not in result or result[ref["path"]] == ref, "Conflicting old/new artifact bindings")
            result[ref["path"]] = ref
    return list(result.values())


def _result(clips, ids, scope, bundle, bindings):
    require(list(clips) == ids, "Loaded record order differs")
    training = scope in {"control_old441", "expanded510"}
    counts = {"clips": len(clips), "student_frames": sum(len(c["pixels"]) for c in clips.values()),
              "train": len(clips) if training else 0, "dev": 0 if training else len(clips), "test": 0,
              "speakers": len({c["speaker_id"] for c in clips.values()}), "feature_width": 512 if training else 0,
              "pixel_bytes": sum(c["pixels"].nbytes for c in clips.values()),
              "feature_bytes": sum(c["teacher_features"].nbytes for c in clips.values()) if training else 0}
    require(counts["student_frames"] == sum(bundle["rows"][sid]["frames"] for sid in ids), "Loaded frame total differs")
    require(all(("teacher_features" in clip) is training for clip in clips.values()), "Teacher features crossed role boundary")
    return {"clips": clips, "ids": list(ids), "counts": counts, "scope": scope, "handoff": dict(HANDOFF),
            "bindings": bindings, "artifact_refs": bundle["refs"], "held_training_ids_opened": [],
            "source_media_opened": False, "new_human_approvals_added": 0}


def load_training(cohort="expanded510"):
    """Return old441 control or old441+new69, with train-only frozen targets."""
    require(cohort in {"control_old441", "expanded510"}, "Unknown training cohort; held/development requests are forbidden")
    loader = old_loader()
    checked = loader.LocalBindings()
    bundle = _handoff(checked)
    old = loader.load_full("train")
    require(old["ids"] == bundle["groups"]["control_old441"] and old["handoff"] == bundle["refs"]["old_handoff"], "Old cohort provenance changed")
    clips = {sid: dict(old["clips"][sid], source_split="train") for sid in old["ids"]}
    if cohort == "expanded510":
        ids = [sid for sid in bundle["groups"]["expanded510"] if bundle["entries"][sid]["generation"] == "new"]
        clips.update(_load_new(bundle, ids, "train", checked))
    return _result(clips, bundle["groups"][cohort], cohort, bundle, merge_bindings(checked.finish(), old["bindings"]))


def load_development(group="new24"):
    """Return every frozen old71 or new24 item; never open a teacher-feature NPZ."""
    require(group in {"old71", "new24"}, "Only complete old71/new24 development groups are allowed")
    loader = old_loader()
    checked = loader.LocalBindings()
    bundle = _handoff(checked)
    if group == "old71":
        old = loader.load_full("dev")
        require(old["ids"] == bundle["groups"]["old71"] and old["handoff"] == bundle["refs"]["old_handoff"], "Old development provenance changed")
        clips = {sid: dict(old["clips"][sid], source_split="dev") for sid in old["ids"]}
        bindings = merge_bindings(checked.finish(), old["bindings"])
    else:
        clips = _load_new(bundle, bundle["groups"]["new24"], "dev", checked)
        bindings = checked.finish()
    result = _result(clips, bundle["groups"][group], group, bundle, bindings)
    result.update(supplemental_ids=bundle["supplemental_ids"] if group == "new24" else [],
                  excluded_duplicate_ids=bundle["excluded_duplicate_ids"] if group == "new24" else [],
                  overlap_evidence=bundle["overlap_evidence"] if group == "new24" else [],
                  duplicate_rule="canonical target equality after removing spaces; full24 remains primary",
                  teacher_feature_npz_opened=False)
    return result
