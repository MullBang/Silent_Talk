"""Freeze and execute a bounded paired GPU training/restart preflight.

This command exposes only the eight accepted training pilots. The generic
engine can consume future verified cohorts, but this plan authorizes four
updates per arm and a replay of the fourth update, not the full experiment.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import artifacts
import runner

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
PLAN_PATH = BASE / "preflight_plan.json"


def binding(path: Path) -> dict:
    """Bind a local file to the frozen execution contract."""
    try:
        return runner.core.bind(path)
    except OSError as exc:
        raise RuntimeError(f"Cannot bind {path}") from exc


def canonical_plan_sha(plan: dict) -> str:
    """Use the same nonrecursive canonical plan identity as the engine."""
    import hashlib
    try:
        core = {key: value for key, value in plan.items() if key != "plan_sha256"}
        payload = json.dumps(core, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":"), allow_nan=False).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()
    except (TypeError, ValueError) as exc:
        raise ValueError("Plan cannot be canonicalized") from exc


def code_bindings() -> list[dict]:
    """Bind execution, checks, and the original model/training implementation."""
    try:
        paths = [BASE / name for name in ("train.py", "runner.py", "artifacts.py", "compare_replay.py",
                                          "test_runner.py", "test_artifacts.py", "test_compare_replay.py")]
        paths.extend(ROOT / name for name in (
            "experiments/feature_transfer_step07/train_compare.py",
            "experiments/feature_transfer_step07/test_training_contract.py",
            "backend/models/baseline.py", "backend/models/tokenizer.py", "backend/models/__init__.py",
            "experiments/olkavs_official/evaluation_metrics.py"))
        return [binding(path) for path in paths]
    except OSError as exc:
        raise RuntimeError("Training implementation cannot be frozen") from exc


def freeze() -> dict:
    """Declare cohorts, schedule, and replay tolerances before observing updates."""
    try:
        runner.core.setup_runtime()
        loaded = artifacts.load_pilot()
        ids = loaded["ids"]
        schedule = runner.core.fixed_schedule(ids, seed=42, updates=4, batch_size=2)
        if sorted(key for batch in schedule for key in batch) != sorted(ids):
            raise ValueError("The pilot schedule must cover each fixed training clip once")
        plan = {
            "schema": "lipnet-step03-paired-training-preflight-v1", "seed": 42,
            "updates": 4, "batch_size": 2, "checkpoint_interval": 1,
            "ids": ids, "schedule": schedule, "arms": list(runner.core.ARMS),
            "scope": "eight fixed training pilots; four optimizer updates per arm, plus separate update4 replay",
            "data_bindings": loaded["bindings"], "parent_plan": loaded["parent_plan"],
            "input_manifest": loaded["input_manifest"], "feature_manifest": loaded["feature_manifest"],
            "teacher_plan": loaded["teacher_plan"], "source_plan": loaded["source_plan"],
            "code_bindings": code_bindings(), "runtime": runner.core.runtime_environment(),
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "optimizer": {"name": "Adam", "lr": .001, "betas": [.9, .999], "eps": 1e-8,
                          "weight_decay": 0., "gradient_clip_norm": 1.0},
            "loss": {"A_ctc": "mean of two per-clip FP32 CPU CTC losses with GPU autograd",
                     "B_feature": "same CTC plus mean normalized squared visual-feature distance",
                     "feature_lambda": 1.0, "feature_projection": "live pre-dropout student256 -> trainable linear512",
                     "teacher": "saved FP32 targets only, detached; no teacher model in training process"},
            "initialization": "fresh same seed student and adapter in both arms; no old student checkpoint",
            "accumulation": "one variable-length clip at a time; divide each loss by two, step once",
            "replay": {"checkpoint_after_update": 3, "replay_update": 4,
                       "restore_model_adapter_optimizer_rng_exact": True,
                       "tolerances": {"model": {"atol": 1e-6, "rtol": 1e-5},
                                      "adapter": {"atol": 1e-6, "rtol": 1e-5},
                                      "exp_avg": {"atol": 1e-7, "rtol": 1e-4},
                                      "exp_avg_sq": {"atol": 1e-10, "rtol": 1e-4},
                                      "loss": {"atol": 1e-5, "rtol": 1e-5},
                                      "gradient_norm": {"atol": 1e-4, "rtol": 1e-4}},
                       "other_state": "exact, including optimizer step and all RNG",
                       "reason": "CUDA spatial pooling backward may be nondeterministic; do not change architecture or relax tolerances after results"},
            "dev_test_audio_used": False, "service_weights_updated": False,
        }
        plan["plan_sha256"] = canonical_plan_sha(plan)
        runner.core.save_json(PLAN_PATH, plan)
        return plan
    except Exception as exc:
        raise RuntimeError("Cannot freeze the Step3 preflight plan") from exc


def verify_plan() -> tuple[dict, dict]:
    """Verify code/data/runtime before every new process or resumed segment."""
    try:
        plan = runner.core.read(PLAN_PATH)
        if plan["plan_sha256"] != canonical_plan_sha(plan):
            raise ValueError("Training plan canonical hash differs")
        runner.core.setup_runtime()
        if plan["runtime"] != runner.core.runtime_environment():
            raise ValueError("Training runtime differs from frozen plan")
        if code_bindings() != plan["code_bindings"]:
            raise ValueError("Training implementation changed")
        if os.environ["CUBLAS_WORKSPACE_CONFIG"] != plan["cublas_workspace_config"]:
            raise ValueError("CUDA workspace configuration changed")
        loaded = artifacts.load_pilot()
        if loaded["ids"] != plan["ids"] or loaded["bindings"] != plan["data_bindings"]:
            raise ValueError("Training artifacts changed")
        return plan, loaded
    except Exception as exc:
        raise RuntimeError("Frozen Step3 plan verification failed") from exc


def execute(arm: str, run_name: str) -> dict:
    """Run one arm in its own process; resumed execution loads update3 from disk."""
    try:
        plan, loaded = verify_plan()
        output = BASE / "runs" / arm / run_name
        if (output / "run_result.json").exists():
            raise FileExistsError("Keep the completed run result immutable")
        resume = None
        if run_name == "resumed":
            original = runner.core.read(BASE / "runs" / arm / "continuous/run_result.json")
            choices = []
            for item in original["checkpoints"]:
                path = Path(item["path"])
                if runner.load_checkpoint(path)["completed_updates"] == plan["replay"]["checkpoint_after_update"]:
                    choices.append(path)
            if len(choices) != 1:
                raise ValueError("A unique completed update3 checkpoint is required")
            resume = choices[0]
        result = runner.run_segment(plan, loaded["clips"], arm, output,
                                    stop_update=plan["updates"], resume_checkpoint=resume, device="cuda:0")
        result["plan_file"] = binding(PLAN_PATH)
        result["run_name"] = run_name
        runner.core.save_json(output / "run_result.json", result)
        return result
    except Exception as exc:
        raise RuntimeError(f"Step3 {arm}/{run_name} failed") from exc


def main() -> None:
    """Expose only bounded preflight actions, never automatically expand training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--arm", choices=("A_ctc", "B_feature"))
    parser.add_argument("--run-name", choices=("continuous", "resumed"))
    args = parser.parse_args()
    if args.freeze:
        plan = freeze()
        print(json.dumps({"frozen_plan": binding(PLAN_PATH), "canonical_plan_sha256": plan["plan_sha256"]}), flush=True)
    else:
        if not args.arm or not args.run_name:
            parser.error("--arm and --run-name are required for execution")
        result = execute(args.arm, args.run_name)
        print(json.dumps({key: result.get(key) for key in ("arm", "status", "checkpoint", "restore_verification")}), flush=True)


if __name__ == "__main__":
    main()
