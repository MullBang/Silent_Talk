"""Post-hoc prediction-decoder correction, preserving all frozen evaluations.

The original greedy decoder calls strict target-label validation. This separate
diagnostic retains malformed predicted jamos as U+FFFD with the already existing
prediction composer. It adds no training, checkpoint selection, language model,
token deletion, or changes to the original executable comparison contract.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import torch

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
OUTPUT = BASE / "results/decode_correction"
PLAN_PATH = OUTPUT / "posthoc_plan.json"
PAIRS = ((42, "A_ctc"), (42, "B_feature"), (43, "A_ctc"), (43, "B_feature"))
INVALID_JAMO_PREFIX = "ValueError: 잘못된 자모 순서/토큰"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


original = load_module("step05_original_evaluation_for_decode_diagnostic", BASE / "evaluate.py")
runner, core, require = original.runner, original.core, original.require
tokenizer = load_module("step05_prediction_composer", ROOT / "backend/models/tokenizer.py")
diagnostics = load_module("step05_bound_diagnostic_utils", ROOT / "experiments/olkavs_official/diagnostic_utils.py")


class ReplayMismatch(ValueError):
    """An output differs beyond the declared decoder correction; stop the audit."""


def evidence(condition: bool, message: str) -> None:
    if not condition:
        raise ReplayMismatch(message)


def decode_prediction(posterior: np.ndarray) -> dict:
    """Preserve raw/collapsed CTC IDs and use the existing tolerant composer."""
    require(isinstance(posterior, np.ndarray) and posterior.ndim == 2
            and posterior.shape[1] == 69 and len(posterior) > 0
            and np.isfinite(posterior).all(), "Invalid finite 69-class frame posterior")
    raw_ids = posterior.argmax(-1).tolist()
    collapsed = diagnostics.collapse_ctc(raw_ids, blank_id=0)
    prediction = tokenizer.compose_jamos([tokenizer.IDX_TO_GRAPHEME[index] for index in collapsed], strict=False)
    strict_prediction, strict_error = None, None
    try:
        strict_prediction = core.greedy(posterior)
    except ValueError as exc:
        strict_error = f"ValueError: {exc}"
    return {"raw_argmax_ids": raw_ids, "collapsed_ids": collapsed, "prediction": prediction,
            "replacement_characters": prediction.count("\ufffd"),
            "strict_replay": {"status": "ok" if strict_error is None else "failed",
                              "prediction": strict_prediction, "error": strict_error}}


def compare_original(decoded: dict, previous: dict) -> dict:
    """Correct only the documented invalid-jamo failure, never unrelated faults."""
    strict = decoded["strict_replay"]
    if previous["status"] == "ok":
        evidence(strict["status"] == "ok" and strict["prediction"] == previous["prediction"],
                 "Originally successful greedy output did not reproduce")
        evidence(decoded["prediction"] == previous["prediction"], "Tolerant composer changed a valid prediction")
        return {"status": "ok", "error": None, "prediction": decoded["prediction"],
                "correction": "unchanged_valid_prediction", "original_success_prediction_match": True,
                "strict_replay_matches_original": True}
    if str(previous.get("error", "")).startswith(INVALID_JAMO_PREFIX):
        evidence(strict["status"] == "failed" and strict["error"] == previous["error"],
                 "Original invalid-jamo failure did not reproduce exactly")
        evidence(decoded["replacement_characters"] > 0, "Invalid predicted jamo was not retained as U+FFFD")
        return {"status": "ok", "error": None, "prediction": decoded["prediction"],
                "correction": "invalid_predicted_jamo_retained_as_replacement_character",
                "original_success_prediction_match": None, "strict_replay_matches_original": True}
    return {"status": "failed", "error": "Original non-decoder failure retained: " + str(previous.get("error")),
            "prediction": "", "correction": "not_eligible_original_non_decoder_failure",
            "original_success_prediction_match": None, "strict_replay_matches_original": False}


def check_original_document(value: dict, seed: int, arm: str) -> None:
    require(value.get("schema") == "lipnet-step05-final-development-evaluation-v1"
            and value.get("status") == "completed" and value.get("seed") == seed and value.get("arm") == arm
            and value.get("completed_updates") == 3200 and value.get("record_count") == 71,
            "Incorrect original evaluation")
    require(len(value["ids"]) == len(set(value["ids"])) == 71, "Incomplete original development IDs")
    require(set(value["conditions"]) == set(original.CONDITIONS), "Incomplete original conditions")
    for condition, data in value["conditions"].items():
        require([row["id"] for row in data["records"]] == value["ids"], "Original condition cohort/order mismatch")
        require(all(row["condition"] == condition and row["split"] == "dev" for row in data["records"]),
                "Original evaluation split mismatch")
        require(core.aggregate(data["records"]) == data["aggregate"], "Original aggregate mismatch")
        for row in data["records"]:
            require(row["status"] in ("ok", "failed") and (row["status"] == "ok" or row["prediction"] == ""),
                    "Original failure policy mismatch")
            require(row["metrics"] == core.counts(row["reference"], row["prediction"]), "Original metrics mismatch")


def code_bindings() -> list[dict]:
    return [original.bind(path) for path in (Path(__file__), BASE / "test_evaluate_prediction_decode.py",
            BASE / "evaluate.py", BASE / "full_artifacts.py", ROOT / "backend/models/tokenizer.py",
            ROOT / "experiments/olkavs_official/diagnostic_utils.py")]


def freeze_plan() -> dict:
    require(not PLAN_PATH.exists(), "Post-hoc settings already frozen")
    entries = []
    for seed, arm in PAIRS:
        path = BASE / f"results/evaluation_seed{seed}_{arm}.json"
        previous = original.read(path)
        check_original_document(previous, seed, arm)
        specs = [previous["plan"], previous["run_result"], previous["full_data_plan"]]
        for spec in specs:
            original.verify(spec)
        result = original.read(previous["run_result"]["path"])
        require(result["checkpoint"] == previous["checkpoint"], "Original evaluation checkpoint mismatch")
        checkpoint = previous["checkpoint"]
        ckpt_ref = original.bind(checkpoint["path"])
        sidecar_ref = original.bind(checkpoint["sidecar"])
        require(ckpt_ref["sha256"] == checkpoint["sha256"] and sidecar_ref["sha256"] == checkpoint["sidecar_sha256"],
                "Original checkpoint or sidecar changed")
        entries.append({"seed": seed, "arm": arm, "original_evaluation": original.bind(path),
                        "plan": previous["plan"], "run_result": previous["run_result"],
                        "checkpoint": ckpt_ref, "checkpoint_sidecar": sidecar_ref,
                        "output": str((OUTPUT / f"evaluation_seed{seed}_{arm}.json").resolve())})
    value = {"schema": "lipnet-step05-posthoc-prediction-decode-plan-v1", "status": "frozen_before_inference",
             "created_utc": datetime.now(timezone.utc).isoformat(), "analysis_type": "posthoc_decoder_correction",
             "original_protocol_and_results_preserved": True, "models": entries, "code_bindings": code_bindings(),
             "changes": ["Replace strict target-label composition with existing compose_jamos(strict=False) for predictions",
                         "Keep each invalid jamo as U+FFFD and count it as an ordinary error character"],
             "invariants": ["Same four fixed update3200 students, same71 rows and normal/static/shuffled pixels",
                            "Same blank0/repeat collapse, no token erasure, beam search or language model",
                            "All originally successful outputs must reproduce exactly",
                            "No training or model changes; no rewriting frozen evaluation artifacts"],
             "failure_policy": "Retain all71 rows; only reproduced invalid-jamo errors eligible for decoder correction; other faults remain failed/empty",
             "primary_results": "The pre-registered original strict-decoder results remain the official frozen comparison",
             "new_optimizer_updates": 0}
    runner._save_json(PLAN_PATH, value)
    return value


def evaluate_rows(model, clips: dict, rows: list[dict], previous: dict, device: torch.device) -> dict:
    require(not model.training and len(rows) == 71 and [row["id"] for row in rows] == previous["ids"],
            "Changed model mode or development cohort")
    before = runner.nested_hash(model.state_dict())
    output = {}
    with torch.inference_mode():
        for condition in original.CONDITIONS:
            records = []
            for row, old in zip(rows, previous["conditions"][condition]["records"]):
                require(row["id"] == old["id"] and row["text"] == old["reference"], "Frozen reference mismatch")
                began = time.perf_counter()
                decoded = {"raw_argmax_ids": [], "collapsed_ids": [], "replacement_characters": 0, "strict_replay": None}
                result = {"status": "failed", "error": "Input/inference not completed", "prediction": "",
                          "correction": "runtime_failure", "original_success_prediction_match": None,
                          "strict_replay_matches_original": False}
                temporal, frames = None, None
                try:
                    clip = clips[row["id"]]
                    require(clip["id"] == row["id"] and clip["split"] == "dev" and clip["text"] == row["text"]
                            and "teacher_features" not in clip, "Invalid development clip")
                    pixels = clip["pixels"]
                    frames = len(pixels)
                    temporal = original.condition_metadata(pixels, condition, row["id"])
                    require(temporal == old["temporal"], "Temporal condition differs from original")
                    posterior = model(core.prepare_student(core.perturb_pixels(pixels, condition, row["id"]), device))
                    require(posterior.shape == (frames, 1, 69) and torch.isfinite(posterior).all().item(), "Invalid posterior")
                    decoded = decode_prediction(posterior[:, 0].float().cpu().numpy())
                    result = compare_original(decoded, old)
                except ReplayMismatch:
                    raise
                except Exception as exc:
                    if old["status"] == "ok" or str(old.get("error", "")).startswith(INVALID_JAMO_PREFIX):
                        raise ReplayMismatch(f"Cannot reproduce original inference for {row['id']} ({condition}): {exc}") from exc
                    result["error"] = f"{type(exc).__name__}: {exc}"
                records.append({"id": row["id"], "speaker_id": row["speaker_id"], "source_stem": row["source_stem"],
                    "split": "dev", "condition": condition, "reference": row["text"], "frames": frames,
                    "temporal": temporal, "seconds": time.perf_counter() - began, **decoded, **result,
                    "original": {key: old[key] for key in ("status", "prediction", "error", "metrics")},
                    "metrics": core.counts(row["text"], result["prediction"])})
            speakers = sorted({row["speaker_id"] for row in rows})
            aggregate = core.aggregate(records)
            output[condition] = {"aggregate": aggregate, "records": records,
                "per_speaker": {speaker: core.aggregate([row for row in records if row["speaker_id"] == speaker]) for speaker in speakers},
                "original_aggregate": previous["conditions"][condition]["aggregate"],
                "posthoc_minus_original_cer_pp": aggregate["cer_percent"] - previous["conditions"][condition]["aggregate"]["cer_percent"],
                "corrected_invalid_jamo_failures": sum(row["correction"] == "invalid_predicted_jamo_retained_as_replacement_character" for row in records),
                "unchanged_original_successes": sum(row["original_success_prediction_match"] is True for row in records),
                "sentences_with_replacement_characters": sum(row["replacement_characters"] > 0 for row in records),
                "replacement_characters": sum(row["replacement_characters"] for row in records)}
    after = runner.nested_hash(model.state_dict())
    evidence(before == after, "Post-hoc inference changed model state")
    return {"conditions": output, "model_state_before": before, "model_state_after": after, "model_state_unchanged": True}


def evaluate_entry(entry: dict, posthoc: dict) -> dict:
    for name in ("original_evaluation", "plan", "run_result", "checkpoint", "checkpoint_sidecar"):
        original.verify(entry[name])
    require(not Path(entry["output"]).exists(), "Refusing to overwrite post-hoc evaluation")
    previous = original.read(entry["original_evaluation"]["path"])
    check_original_document(previous, entry["seed"], entry["arm"])
    plan, result = original.read(entry["plan"]["path"]), original.read(entry["run_result"]["path"])
    require(plan["plan_sha256"] == runner.canonical_plan_hash(plan) and result["plan_file"] == entry["plan"], "Training plan mismatch")
    require(previous["plan"] == entry["plan"] and previous["run_result"] == entry["run_result"]
            and previous["checkpoint"] == result["checkpoint"], "Original result provenance mismatch")
    for ref in plan["execution_code"] + [plan["full_data_plan"]]:
        original.verify(ref)
    require(plan["full_data_plan"] == previous["full_data_plan"], "Changed handoff")
    payload = runner.load_checkpoint(Path(entry["checkpoint"]["path"]))
    original.validate_final_payload(plan, result, payload)
    loader = load_module("step05_posthoc_full_artifacts", BASE / "full_artifacts.py")
    loaded = loader.load_full(scope="dev")
    handoff = original.read(plan["full_data_plan"]["path"])
    original.verify(handoff["parent_plan"])
    rows = [row for row in original.read(handoff["parent_plan"]["path"])["records"] if row["split"] == "dev"]
    require(loaded["ids"] == previous["ids"] == handoff["development_ids"], "Changed development IDs")
    device = torch.device("cuda:0")
    core.setup_runtime()
    runtime = runner._runtime(device)
    require(runtime == plan["runtime"] == payload["identity"]["runtime"] == previous["runtime"], "Runtime changed")
    model = core.LipNetBaseline()
    model.load_state_dict(payload["state"]["model"], strict=True)
    require(runner.nested_hash(model.state_dict()) == payload["state_hashes"]["model"], "Wrong final model state")
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    started = time.perf_counter()
    evaluated = evaluate_rows(model, loaded["clips"], rows, previous, device)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    for name in ("original_evaluation", "plan", "run_result", "checkpoint", "checkpoint_sidecar"):
        original.verify(entry[name])
    for ref in posthoc["code_bindings"] + loaded["bindings"] + plan["execution_code"]:
        original.verify(ref)
    value = {"schema": "lipnet-step05-posthoc-prediction-decode-evaluation-v1", "status": "completed",
        "analysis_type": "posthoc_decoder_correction", "created_utc": datetime.now(timezone.utc).isoformat(),
        "posthoc_plan": original.bind(PLAN_PATH), "original_evaluation": entry["original_evaluation"],
        "seed": entry["seed"], "arm": entry["arm"], "completed_updates": 3200, "record_count": 71,
        "ids": loaded["ids"], "plan": entry["plan"], "run_result": entry["run_result"], "checkpoint": result["checkpoint"],
        "full_data_plan": plan["full_data_plan"], "data_bindings": loaded["bindings"], "runtime": runtime,
        "elapsed_seconds": elapsed, "new_optimizer_updates": 0, "audio_read": False,
        "teacher_targets_read": False, "teacher_executed": False, "service_weights_updated": False,
        "decoding": "Frame argmax, adjacent-repeat collapse before blank0 removal, compose_jamos(strict=False), invalid jamos retained as U+FFFD",
        "official_frozen_results_replaced": False,
        "limits": ["Post-hoc decoder correction discovered after examining original evaluation failures",
                   "Original evaluation did not store argmax IDs; exact prior token-path identity cannot be established retroactively",
                   "Same-speaker held-out recordings, not an untouched final test or webcam validation"], **evaluated}
    runner._save_json(Path(entry["output"]), value)
    print(json.dumps({"seed": entry["seed"], "arm": entry["arm"], "output": entry["output"],
                      "normal": value["conditions"]["normal"]["aggregate"]}, ensure_ascii=False), flush=True)
    return value


def run_all() -> dict:
    posthoc = original.read(PLAN_PATH)
    require(posthoc["schema"] == "lipnet-step05-posthoc-prediction-decode-plan-v1"
            and posthoc["status"] == "frozen_before_inference" and posthoc["code_bindings"] == code_bindings(),
            "Post-hoc plan/code mismatch")
    require([(entry["seed"], entry["arm"]) for entry in posthoc["models"]] == list(PAIRS), "Changed four-model scope")
    plan_ref = original.bind(PLAN_PATH)
    require(all(not Path(entry["output"]).exists() for entry in posthoc["models"]), "Post-hoc outputs already exist")
    values = []
    for entry in posthoc["models"]:
        original.verify(plan_ref)
        values.append(evaluate_entry(entry, posthoc))
    original.verify(plan_ref)
    summary = {"schema": "lipnet-step05-posthoc-prediction-decode-summary-v1", "status": "completed",
               "analysis_type": "posthoc_decoder_correction", "posthoc_plan": plan_ref,
               "evaluations": [original.bind(entry["output"]) for entry in posthoc["models"]],
               "official_frozen_results_replaced": False, "new_optimizer_updates": 0,
               "models": [{"seed": value["seed"], "arm": value["arm"],
                           "conditions": {condition: {key: data[key] for key in (
                               "aggregate", "original_aggregate", "posthoc_minus_original_cer_pp", "corrected_invalid_jamo_failures",
                               "unchanged_original_successes", "sentences_with_replacement_characters", "replacement_characters")}
                               for condition, data in value["conditions"].items()}} for value in values]}
    runner._save_json(OUTPUT / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", action="store_true")
    group.add_argument("--run", action="store_true")
    args = parser.parse_args()
    value = freeze_plan() if args.freeze else run_all()
    print(json.dumps({"status": value["status"], "analysis_type": value["analysis_type"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
