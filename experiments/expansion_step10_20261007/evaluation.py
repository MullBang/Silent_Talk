"""Predeclared Step10 development evaluation with invalid-jamo preservation.

Both variants are inferred anew on old71 and new24; new23 is only a fixed
subset aggregation. No teacher/audio/held inputs, optimization, or checkpoint
selection occurs. Existing Step5 evidence is read-only control replay evidence.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import unicodedata

import numpy as np
import torch

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
STEP5 = ROOT / "experiments/local_training_step05_20261007"


def load_module(name, path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


decoder = load_module("step10_frozen_prediction_decoder", STEP5 / "evaluate_prediction_decode.py")
original = decoder.original
runner, core, require = original.runner, original.core, original.require
bind, verify, read = original.bind, original.verify, original.read
CONDITIONS = original.CONDITIONS
VARIANTS = ("control_old441", "expanded510")
SEEDS = (42, 43)
CONTROL_AUDIT = {"path": str(STEP5 / "results/decode_correction/independent_audit.json"),
                 "sha256": "080f5b27f5fb9f3304a6d34a7101058141d3abbf3cd7b20d91a273644555923a"}


def evaluation_code():
    paths = [Path(__file__), BASE / "test_evaluation.py", BASE / "mixed_artifacts.py",
             STEP5 / "evaluate.py", STEP5 / "evaluate_prediction_decode.py",
             ROOT / "experiments/olkavs_official/diagnostic_utils.py",
             ROOT / "experiments/olkavs_official/evaluation_metrics.py"]
    paths.extend(Path(ref["path"]) for ref in runner._code_bindings())
    return [bind(path) for path in dict.fromkeys(paths)]


def validate_metadata(value):
    old, new, subset = value["old71_ids"], value["new24_ids"], value["new23_ids"]
    excluded = value["excluded_duplicate_ids"]
    require(len(old) == len(set(old)) == 71 and len(new) == len(set(new)) == 24
            and len(subset) == len(set(subset)) == 23 and len(excluded) == 1, "Incomplete development groups")
    require(set(old).isdisjoint(new) and subset == [sid for sid in new if sid not in excluded]
            and set(excluded) <= set(new), "Changed development overlap subset")
    require(len(value["overlap_evidence"]) == 1, "Expected one frozen whitespace-free transcript overlap")
    overlap = value["overlap_evidence"][0]
    text_key = lambda text: "".join(unicodedata.normalize("NFC", text).split())
    require(overlap["val_id"] == excluded[0] and overlap["train_id"] not in set(old + new)
            and text_key(overlap["train_text"]) == text_key(overlap["val_text"]), "Wrong overlap evidence")
    return value


def evaluation_contract():
    """Metadata-only policy for the training comparison plan, before updates."""
    loader = load_module("step10_evaluation_metadata_loader", BASE / "mixed_artifacts.py")
    metadata = validate_metadata(loader.evaluation_metadata())
    verify(CONTROL_AUDIT)
    audit = read(CONTROL_AUDIT["path"])
    require(audit["schema"] == "lipnet-step05-independent-decode-correction-audit-v1"
            and audit["status"] == "passed", "Prior corrected-control audit is not passed")
    verify(audit["summary"])
    summary = read(audit["summary"]["path"])
    audited = {(row["seed"], row["arm"]): row["evaluation"] for row in audit["models"]}
    controls = {}
    for seed in SEEDS:
        path = STEP5 / f"results/decode_correction/evaluation_seed{seed}_B_feature.json"
        ref = bind(path)
        require(audited[(seed, "B_feature")] == ref and ref in summary["evaluations"], "Prior control differs from independent audit")
        value = read(path)
        require(value["schema"] == "lipnet-step05-posthoc-prediction-decode-evaluation-v1"
                and value["status"] == "completed" and value["seed"] == seed and value["arm"] == "B_feature"
                and value["completed_updates"] == 3200 and value["ids"] == metadata["old71_ids"], "Wrong prior control evaluation")
        controls[str(seed)] = ref
    return {"schema": "lipnet-step10-frozen-evaluation-contract-v1", "declared_before_training": True,
            "conditions": list(CONDITIONS), "development": metadata, "control_old71_replay": controls,
            "control_replay_audit": CONTROL_AUDIT, "control_replay_summary": audit["summary"],
            "checkpoint": "Only the frozen seed's B_feature final3200 student; no best selection or teacher forward",
            "decoder": "Frame argmax; collapse adjacent repeats before blank0 removal; existing compose_jamos(strict=False); retain every invalid jamo as U+FFFD",
            "raw_token_paths": "Save all frame argmax IDs, collapsed IDs and strict label-composer diagnostic; no language model/beam/token erasure",
            "primary_metric": "Pooled NFC CER including internal spaces; NFC strip only; failed inference is empty prediction in full denominator",
            "secondary_metrics": "WER, exact sentences, whitespace-collapsed CER and exact; per-speaker values; retained U+FFFD counts",
            "temporal_conditions": "Frozen midpoint static frame and ID-derived step07-order-v1 shuffle; same permutation across variants and seeds",
            "new23": "Reaggregate new24 records after excluding the one predeclared whitespace-free transcript overlap for both variants; no re-inference or role changes",
            "control_replay": "Re-infer old71 and require exact raw IDs, collapsed IDs, prediction, status, metrics, temporal and per-speaker aggregates against Step5 corrected control evidence; preserve mismatching new output before failing",
            "failure_policy": "Artifact/identity/hash failures abort; individual input/inference failures retain every fixed ID with empty prediction; no cohort replacement",
            "optimization_steps": 0, "teacher_targets_read": False, "audio_read": False,
            "limits": ["Development cohorts were selected and reviewed; neither is an untouched final test",
                       "Old71 contains speakers also represented in old training recordings",
                       "new23 removes the declared text overlap but does not establish general unseen-speaker or webcam performance",
                       "Static/shuffled comparisons are diagnostic; text performance alone does not prove reliance on lip motion"]}


def validate_final_target(target, variant, seed, payload):
    require(variant in VARIANTS and seed in SEEDS, "Unknown requested comparison")
    plan, result = target["plan"], target["result"]
    original.validate_final_payload(plan, result, payload)
    require(plan["seed"] == seed and payload["identity"]["arm"] == result["arm"] == "B_feature", "Wrong model seed/method")
    count = 441 if variant == "control_old441" else 510
    require(len(plan["training_data"]["clips"]) == count and
            set(sid for batch in plan["schedule"] for sid in batch) == set(plan["training_data"]["clips"]), "Wrong training cohort")
    require(result["plan_file"] == target["plan_file"], "Run result plan-file binding differs")
    require(payload["initial_state_hashes"] == result["initial_state_hashes"] == target["initial_state_hashes_expected"],
            "Initial state differs from the matched fresh seed")
    if variant == "expanded510":
        require(plan["variant"] == result["variant"] == variant and result["experiment_plan"] == target["experiment_plan"],
                "Expanded experiment identity differs")


def summarize_condition(records):
    speakers = sorted({row["speaker_id"] for row in records})
    return {"records": records, "aggregate": core.aggregate(records),
            "per_speaker": {speaker: core.aggregate([row for row in records if row["speaker_id"] == speaker]) for speaker in speakers},
            "sentences_with_replacement_characters": sum(row["replacement_characters"] > 0 for row in records),
            "replacement_characters": sum(row["replacement_characters"] for row in records)}


def evaluate_clips(model, loaded, group, device):
    """All planned rows survive runtime failures; metadata is checked before inference."""
    ids, clips = loaded["ids"], loaded["clips"]
    rows = loaded["rows"] if "rows" in loaded else [clips[sid] for sid in ids]
    require(ids and len(ids) == len(set(ids)) and [row["id"] for row in rows] == ids, "Changed development order")
    require(all(row["split"] == "dev" for row in rows) and set(clips) <= set(ids), "Nondevelopment input")
    require(not model.training and not any(parameter.requires_grad for parameter in model.parameters()), "Student must be frozen eval")
    before = runner.nested_hash(model.state_dict())
    conditions = {}
    with torch.inference_mode():
        for condition in CONDITIONS:
            records = []
            for row in rows:
                started = time.perf_counter()
                prediction, error, temporal, frames = "", None, None, None
                tokens = {"raw_argmax_ids": [], "collapsed_ids": [], "replacement_characters": 0,
                          "strict_label_decoder_diagnostic": None}
                try:
                    clip = clips[row["id"]]
                    require(clip["id"] == row["id"] and clip["split"] == "dev" and clip["text"] == row["text"]
                            and clip["speaker_id"] == row["speaker_id"] and clip["source_stem"] == row["source_stem"], "Development identity changed")
                    require("teacher_features" not in clip and "audio" not in clip, "Forbidden development model inputs")
                    pixels = clip["pixels"]
                    require(isinstance(pixels, np.ndarray) and pixels.dtype == np.uint8 and pixels.ndim == 4
                            and len(pixels) > 0 and pixels.shape[1:] == (96, 96, 3), "Malformed student pixels")
                    frames = len(pixels)
                    temporal = original.condition_metadata(pixels, condition, row["id"])
                    posterior = model(core.prepare_student(core.perturb_pixels(pixels, condition, row["id"]), device))
                    require(posterior.shape == (frames, 1, 69) and torch.isfinite(posterior).all().item(), "Malformed/nonfinite posterior")
                    decoded = decoder.decode_prediction(posterior[:, 0].float().cpu().numpy())
                    prediction = decoded["prediction"]
                    tokens = {key: decoded[key] for key in ("raw_argmax_ids", "collapsed_ids", "replacement_characters")}
                    tokens["strict_label_decoder_diagnostic"] = decoded["strict_replay"]
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    prediction = ""
                records.append({"id": row["id"], "speaker_id": row["speaker_id"], "source_stem": row["source_stem"],
                                "split": "dev", "source_split": row.get("source_split", "dev"), "group": group,
                                "condition": condition, "reference": row["text"], "prediction": prediction,
                                "status": "ok" if error is None else "failed", "error": error, "frames": frames,
                                "temporal": temporal, "seconds": time.perf_counter() - started, **tokens,
                                "metrics": core.counts(row["text"], prediction)})
            conditions[condition] = summarize_condition(records)
    after = runner.nested_hash(model.state_dict())
    require(before == after, "Evaluation changed student state")
    return {"ids": ids, "record_count": len(ids), "conditions": conditions,
            "model_state_before": before, "model_state_after": after, "model_state_unchanged": True,
            "aggregation_only": False}


def subset_cohort(parent, selected_ids, excluded_ids):
    require(selected_ids == [sid for sid in parent["ids"] if sid not in excluded_ids]
            and len(set(selected_ids)) == len(selected_ids) and set(excluded_ids) <= set(parent["ids"]), "Subset order/exclusions changed")
    conditions = {condition: summarize_condition([row for row in value["records"] if row["id"] in selected_ids])
                  for condition, value in parent["conditions"].items()}
    return {"ids": selected_ids, "record_count": len(selected_ids), "conditions": conditions,
            "parent_group": "new24", "excluded_ids": excluded_ids, "aggregation_only": True,
            "model_state_before": parent["model_state_before"], "model_state_after": parent["model_state_after"],
            "model_state_unchanged": parent["model_state_unchanged"]}


def control_replay(current, prior):
    """Return evidence instead of discarding a mismatching fresh evaluation."""
    mismatches = []
    if current["ids"] != prior["ids"]:
        mismatches.append({"field": "ids"})
    keys = ("id", "speaker_id", "source_stem", "reference", "prediction", "status", "error", "frames", "temporal",
            "raw_argmax_ids", "collapsed_ids", "replacement_characters", "metrics")
    for condition in CONDITIONS:
        old, new = prior["conditions"][condition], current["conditions"][condition]
        if len(old["records"]) != len(new["records"]):
            mismatches.append({"condition": condition, "field": "record_count"})
        for one, two in zip(new["records"], old["records"]):
            different = [key for key in keys if one[key] != two[key]]
            if one["strict_label_decoder_diagnostic"] != two["strict_replay"]:
                different.append("strict_label_decoder_diagnostic")
            if different:
                mismatches.append({"condition": condition, "id": one["id"], "fields": different})
        for key in ("aggregate", "per_speaker", "sentences_with_replacement_characters", "replacement_characters"):
            if new[key] != old[key]:
                mismatches.append({"condition": condition, "field": key})
    return {"status": "passed" if not mismatches else "failed", "compared_records": sum(len(value["records"]) for value in current["conditions"].values()),
            "mismatches": mismatches, "prior_predictions_used_as_outputs": False}


def validate_loaded(loaded, group, contract):
    expected = contract["development"][group + "_ids"]
    require(loaded["ids"] == expected and set(loaded["clips"]) == set(expected), "Loaded development cohort differs")
    for sid in expected:
        clip = loaded["clips"][sid]
        require(clip["id"] == sid and clip["split"] == "dev" and "teacher_features" not in clip and "audio" not in clip,
                "Wrong development identity or forbidden modality")
        require(clip.get("source_split", "dev") == ("dev" if group == "old71" else "val"), "Original split role changed")
    if group == "new24":
        require(loaded["supplemental_ids"] == contract["development"]["new23_ids"]
                and loaded["excluded_duplicate_ids"] == contract["development"]["excluded_duplicate_ids"], "Supplemental exclusion differs")


def save_result(output, value):
    """Keep a failed replay reviewable, and never overwrite previous evidence."""
    require(not Path(output).exists(), "Evaluation output already exists")
    runner._save_json(Path(output), value)
    require(value["status"] == "completed", "Control replay mismatch: new results preserved; investigate without changing frozen code")


def run_evaluation(seed, variant, output):
    output = Path(output).resolve()
    require(output.is_relative_to(BASE / "results") and not output.exists(), "Output must be a new file inside Stage10 results")
    training = load_module("step10_training_target_access", BASE / "training.py")
    target = training.evaluation_target(variant, seed)
    for key in ("experiment_plan", "plan_file", "run_result"):
        verify(target[key])
    policy = read(target["experiment_plan"]["path"])
    contract = policy["evaluation_contract"]
    require(contract == evaluation_contract(), "Frozen evaluation policy or evidence changed")
    frozen_code = {ref["path"]: ref for ref in policy["execution_code"]}
    for ref in evaluation_code():
        require(frozen_code.get(ref["path"]) == ref, "Evaluator dependency not frozen before training")
    for ref in policy["execution_code"]:
        verify(ref)
    require(target["plan"] == read(target["plan_file"]["path"]) and target["result"] == read(target["run_result"]["path"]),
            "Target access returned different documents")
    plan, result = target["plan"], target["result"]
    checkpoint = result["checkpoint"]
    checkpoint_ref, sidecar_ref = bind(checkpoint["path"]), bind(checkpoint["sidecar"])
    require(checkpoint_ref["sha256"] == checkpoint["sha256"] and sidecar_ref["sha256"] == checkpoint["sidecar_sha256"], "Final checkpoint bytes changed")
    payload = runner.load_checkpoint(Path(checkpoint["path"]))
    validate_final_target(target, variant, seed, payload)
    loader = load_module("step10_evaluation_data_access", BASE / "mixed_artifacts.py")
    datasets = {group: loader.load_development(group) for group in ("old71", "new24")}
    for group, loaded in datasets.items():
        validate_loaded(loaded, group, contract)
    core.setup_runtime()
    device = torch.device("cuda:0")
    runtime = runner._runtime(device)
    require(runtime == plan["runtime"] == payload["identity"]["runtime"], "Evaluation runtime differs from checkpoint")
    model = core.LipNetBaseline()
    model.load_state_dict(payload["state"]["model"], strict=True)
    require(runner.nested_hash(model.state_dict()) == payload["state_hashes"]["model"], "Loaded student state differs")
    model.to(device).eval().requires_grad_(False)
    started = time.perf_counter()
    cohorts = {group: evaluate_clips(model, loaded, group, device) for group, loaded in datasets.items()}
    cohorts["new23"] = subset_cohort(cohorts["new24"], contract["development"]["new23_ids"], contract["development"]["excluded_duplicate_ids"])
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    replay = {"status": "not_applicable_to_new_model"}
    if variant == "control_old441":
        control_ref = contract["control_old71_replay"][str(seed)]
        verify(control_ref)
        previous = read(control_ref["path"])
        require(previous["checkpoint"] == checkpoint, "Control replay checkpoint changed")
        replay = {**control_replay(cohorts["old71"], previous), "reference": control_ref}
    all_bindings = policy["execution_code"] + contract["development"]["bindings"]
    for loaded in datasets.values():
        all_bindings.extend(loaded["bindings"])
    all_bindings += [target[key] for key in ("experiment_plan", "plan_file", "run_result")] + [checkpoint_ref, sidecar_ref]
    for ref in all_bindings:
        verify(ref)
    value = {"schema": "lipnet-step10-development-evaluation-v1", "status": "failed_control_replay" if replay["status"] == "failed" else "completed",
             "created_utc": datetime.now(timezone.utc).isoformat(), "seed": seed, "variant": variant, "arm": "B_feature",
             "completed_updates": 3200, "experiment_plan": target["experiment_plan"], "training_plan": target["plan_file"],
             "run_result": target["run_result"], "checkpoint": checkpoint, "initial_state_hashes": payload["initial_state_hashes"],
             "runtime": runtime, "evaluation_code": evaluation_code(), "data_bindings": {group: loaded["bindings"] for group, loaded in datasets.items()},
             "cohorts": cohorts, "inferred_distinct_records": 95, "inferred_condition_records": 285,
             "old_control_replay": replay, "elapsed_seconds": elapsed, "new_optimizer_updates": 0,
             "audio_read": False, "original_media_read": False, "teacher_targets_read": False, "teacher_executed": False,
             "optimizer_executed": False, "held_training_records_read": False, "service_weights_updated": False,
             "decoding": contract["decoder"], "failure_policy": contract["failure_policy"], "limits": contract["limits"]}
    save_result(output, value)
    print(json.dumps({"output": str(output), "status": value["status"], "variant": variant, "seed": seed,
                      "normal": {group: cohort["conditions"]["normal"]["aggregate"] for group, cohort in cohorts.items()}}, ensure_ascii=False), flush=True)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", required=True, type=int, choices=SEEDS)
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run_evaluation(args.seed, args.variant, args.output)


if __name__ == "__main__":
    main()
