"""Final-update development evaluation for the frozen full-data A/B experiment.

All 71 development sentences remain in every condition. This module loads only
the student's final state and performs inference, without optimizer creation,
teacher execution, audio, original media, checkpoint selection, or service writes.
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
CONDITIONS = ("normal", "static", "shuffled")


def module_at(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = module_at("step05_eval_frozen_runner", ROOT / "experiments/local_training_step03_20261007/runner.py")
core = runner.core
require = runner.require


def bind(path: Path | str) -> dict:
    path = Path(path).resolve()
    return {"path": str(path), "sha256": runner._sha_file(path)}


def verify(spec: dict) -> None:
    require(bind(spec["path"]) == spec, f"Changed evaluation evidence: {spec['path']}")


def read(path: Path | str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_final_payload(plan: dict, result: dict, payload: dict) -> None:
    """Reject any other budget, seed, arm, data, schedule, code, or final state."""
    require(plan.get("updates") == 3200 and plan.get("batch_size") == 2
            and plan.get("seed") in (42, 43), "Wrong full-training recipe")
    plan_sha = runner.canonical_plan_hash(plan)
    require(plan.get("plan_sha256") == plan_sha, "Plan digest mismatch")
    require(result.get("schema") == "local-paired-training-segment-result-v1", "Wrong result schema")
    require(payload.get("schema") == runner.SCHEMA, "Wrong checkpoint schema")
    require(result.get("completed_updates") == payload.get("completed_updates") == 3200,
            "Evaluation requires exactly final update 3200")
    identity = payload["identity"]
    require(identity["arm"] in core.ARMS and result["arm"] == identity["arm"], "Arm mismatch")
    require(identity["seed"] == plan["seed"] and identity["total_updates"] == 3200
            and identity["batch_size"] == 2, "Checkpoint recipe mismatch")
    require(identity["plan_sha256"] == result["plan_sha256"] == plan_sha, "Checkpoint plan mismatch")
    require(identity["schedule_sha256"] == runner.nested_hash(plan["schedule"]), "Checkpoint schedule mismatch")
    require(len(plan["schedule"]) == 3200 and all(len(batch) == 2 for batch in plan["schedule"]),
            "Malformed full schedule")
    require(identity["data_sha256"] == plan["training_data"]["data_sha256"]
            and identity["clip_bindings"] == plan["training_data"]["clips"], "Checkpoint data mismatch")
    require(identity["code_bindings"] == runner._code_bindings(), "Checkpoint code mismatch")
    require(result["identity_sha256"] == runner.nested_hash(identity), "Result identity mismatch")
    require(result["runtime"] == identity["runtime"] == plan["runtime"], "Result/plan runtime mismatch")
    require(identity["optimization"] == {"optimizer": "Adam", "lr": .001, "betas": [.9, .999],
            "eps": 1e-8, "weight_decay": 0., "gradient_clip": 1., "accumulation_clips": 2,
            "lambda": 0. if identity["arm"] == "A_ctc" else 1., "ctc_device": "cpu", "precision": "float32"},
            "Optimization recipe mismatch")
    history = payload["history"]
    require(len(history) == 3200 and all(row["update"] == i + 1 and row["ids"] == plan["schedule"][i]
            for i, row in enumerate(history)), "Incomplete or altered update history")
    hashes = runner.state_hashes(payload["state"])
    require(hashes == payload["state_hashes"] == result["final_state_hashes"]
            == result["checkpoint"]["state_hashes"] == history[-1]["state_hashes"], "Final state mismatch")
    require(result["checkpoint"]["completed_updates"] == 3200, "Result checkpoint is not final")


def condition_metadata(pixels: np.ndarray, condition: str, sample_id: str) -> dict:
    """Publish exact temporal manipulations using the frozen core's permutation."""
    length = len(pixels)
    require(length > 0, "Empty frame sequence")
    positions = np.arange(length, dtype=np.int64)[:, None]
    order = core.perturb_pixels(positions, condition, sample_id).reshape(-1).tolist()
    return {"frames": length, "source_positions": order,
            "source_positions_sha256": runner.nested_hash(order),
            "static_position": length // 2 if condition == "static" else None,
            "shuffle_policy": "sha256(step07-order-v1: + sample_id), first8 little endian, numpy.default_rng.permutation"
            if condition == "shuffled" else None}


def evaluate_clips(model, clips: dict, rows: list[dict], device: torch.device) -> dict:
    """Evaluate every fixed row, preserving failed inference as empty prediction."""
    ids = [row["id"] for row in rows]
    require(len(ids) == len(set(ids)) and ids, "Empty or duplicate development cohort")
    require(all(row["split"] == "dev" for row in rows), "Non-development evaluation row")
    require(set(clips).issubset(ids), "Unexpected evaluation clip")
    require(not model.training, "Evaluation model must be in eval mode")
    before = runner.nested_hash(model.state_dict())
    by_condition = {}
    with torch.inference_mode():
        for condition in CONDITIONS:
            records = []
            for row in rows:
                prediction, error, temporal, frames = "", None, None, None
                began = time.perf_counter()
                try:
                    clip = clips[row["id"]]
                    require(clip["id"] == row["id"] and clip["split"] == "dev"
                            and clip["text"] == row["text"] and clip["speaker_id"] == row["speaker_id"],
                            "Development identity/text mismatch")
                    require("teacher_features" not in clip, "Development teacher target is forbidden")
                    pixels = clip["pixels"]
                    require(isinstance(pixels, np.ndarray) and pixels.dtype == np.uint8 and pixels.ndim == 4
                            and pixels.shape[1:] == (96, 96, 3) and len(pixels) > 0, "Invalid development pixels")
                    frames = len(pixels)
                    temporal = condition_metadata(pixels, condition, row["id"])
                    posterior = model(core.prepare_student(core.perturb_pixels(pixels, condition, row["id"]), device))
                    require(posterior.shape == (frames, 1, 69) and torch.isfinite(posterior).all().item(),
                            "Nonfinite or malformed CTC posterior")
                    prediction = core.greedy(posterior[:, 0].float().cpu().numpy())
                except Exception as exc:
                    # Evidence checks occur before this function. Runtime input/inference
                    # failure does not remove a denominator or admit another checkpoint.
                    error = f"{type(exc).__name__}: {exc}"
                    prediction = ""
                records.append({"id": row["id"], "speaker_id": row["speaker_id"], "source_stem": row["source_stem"],
                    "split": "dev", "condition": condition, "reference": row["text"], "prediction": prediction,
                    "status": "ok" if error is None else "failed", "error": error, "frames": frames,
                    "temporal": temporal, "seconds": time.perf_counter() - began,
                    "metrics": core.counts(row["text"], prediction)})
            speakers = sorted({row["speaker_id"] for row in rows})
            by_condition[condition] = {"aggregate": core.aggregate(records), "records": records,
                "per_speaker": {speaker: core.aggregate([row for row in records if row["speaker_id"] == speaker])
                                for speaker in speakers}}
    after = runner.nested_hash(model.state_dict())
    require(before == after, "Evaluation changed student model state")
    return {"conditions": by_condition, "model_state_before": before, "model_state_after": after,
            "model_state_unchanged": True}


def run_evaluation(plan_path: Path, run_result_path: Path, output: Path, device: str = "cuda:0") -> dict:
    require(not output.exists(), f"Refusing to overwrite evaluation: {output}")
    plan_binding, result_binding = bind(plan_path), bind(run_result_path)
    plan, result = read(plan_path), read(run_result_path)
    require(plan.get("plan_sha256") == runner.canonical_plan_hash(plan), "Plan digest mismatch")
    require(result.get("plan_file") == plan_binding, "Run result plan-file binding mismatch")
    for binding in plan["execution_code"]:
        verify(binding)
    expected_code = {str(Path(binding["path"]).resolve()): binding for binding in plan["execution_code"]}
    for path in (Path(__file__), BASE / "test_evaluate.py"):
        require(expected_code.get(str(path.resolve())) == bind(path), "Evaluator was not frozen before training")
    verify(plan["full_data_plan"])
    handoff = read(plan["full_data_plan"]["path"])
    for binding in handoff["verified_code"]:
        verify(binding)
    for binding in handoff["artifacts"].values():
        verify(binding)
    parent = read(handoff["parent_plan"]["path"])
    verify(handoff["parent_plan"])
    dev_rows = [row for row in parent["records"] if row["split"] == "dev"]
    require(len(dev_rows) == 71 and [row["id"] for row in dev_rows] == handoff["development_ids"],
            "Full development cohort mismatch")
    require(set(handoff["train_ids"]) == set(plan["training_data"]["clips"]), "Training cohort mismatch")
    checkpoint_info = result["checkpoint"]
    checkpoint_path = Path(checkpoint_info["path"])
    require(runner._sha_file(checkpoint_path) == checkpoint_info["sha256"], "Run result checkpoint SHA mismatch")
    require(runner._sha_file(Path(checkpoint_info["sidecar"])) == checkpoint_info["sidecar_sha256"],
            "Run result checkpoint sidecar SHA mismatch")
    payload = runner.load_checkpoint(checkpoint_path)
    validate_final_payload(plan, result, payload)
    loader = module_at("step05_eval_full_artifacts", BASE / "full_artifacts.py")
    loaded = loader.load_full(scope="dev")
    require(loaded["ids"] == handoff["development_ids"] and len(loaded["clips"]) == 71, "Loader development cohort mismatch")
    selected_device = torch.device(device)
    require(selected_device.type in ("cpu", "cuda")
            and (selected_device.type != "cuda" or torch.cuda.is_available()), "Unavailable evaluation device")
    core.setup_runtime()
    runtime = runner._runtime(selected_device)
    require(runtime == payload["identity"]["runtime"], "Evaluation runtime differs from training")
    model = core.LipNetBaseline()
    model.load_state_dict(payload["state"]["model"], strict=True)
    require(runner.nested_hash(model.state_dict()) == payload["state_hashes"]["model"], "Loaded model differs from final state")
    model.to(selected_device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    began = time.perf_counter()
    evaluated = evaluate_clips(model, loaded["clips"], dev_rows, selected_device)
    if selected_device.type == "cuda":
        torch.cuda.synchronize(selected_device)
    elapsed = time.perf_counter() - began
    for binding in loaded["bindings"] + [plan_binding, result_binding, plan["full_data_plan"]] + plan["execution_code"]:
        verify(binding)
    require(runner._sha_file(checkpoint_path) == checkpoint_info["sha256"], "Checkpoint changed during evaluation")
    result = {"schema": "lipnet-step05-final-development-evaluation-v1", "status": "completed",
        "created_utc": datetime.now(timezone.utc).isoformat(), "plan": plan_binding, "run_result": result_binding,
        "plan_sha256": plan["plan_sha256"], "checkpoint": checkpoint_info,
        "seed": plan["seed"], "arm": payload["identity"]["arm"], "completed_updates": 3200,
        "full_data_plan": plan["full_data_plan"], "data_bindings": loaded["bindings"],
        "evaluation_code": [bind(Path(__file__)), bind(BASE / "test_evaluate.py")],
        "runtime": runtime, "elapsed_seconds": elapsed, "ids": loaded["ids"], "record_count": 71,
        "audio_read": False, "original_media_read": False, "teacher_targets_read": False,
        "teacher_executed": False, "optimizer_executed": False, "checkpoint_selection": "fixed final update3200",
        "decode": "frozen greedy CTC; no beam search or language model", "primary_metric": "pooled NFC CER retaining internal spaces",
        "failure_policy": "Strict artifact-integrity failures abort evaluation; individual input/inference failures retain row and empty prediction",
        "limits": ["Same eight speakers as training, disjoint recordings; development cohort is not an untouched final test",
                   "Two seeds and this small existing cohort do not establish unseen-speaker or webcam generalization"],
        **evaluated}
    runner._save_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0", help="Must reproduce the frozen training runtime/device")
    args = parser.parse_args()
    result = run_evaluation(args.plan, args.run_result, args.output, args.device)
    print(json.dumps({"output": str(args.output.resolve()), "seed": result["seed"], "arm": result["arm"],
                      "conditions": {key: value["aggregate"] for key, value in result["conditions"].items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
