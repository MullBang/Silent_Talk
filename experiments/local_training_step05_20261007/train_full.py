"""Freeze and run the authorized, fixed-budget full-data paired experiment.

Uses the unchanged Step3 optimizer/checkpoint engine. Each arm runs in a fresh
process, uses all441 training clips, and never loads development inputs. No
original media or service model is accessed by this entrypoint.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
STEP3 = ROOT / "experiments/local_training_step03_20261007"
STEP4 = ROOT / "experiments/local_dataset_step04_20261007"
HANDOFF_SHA = "3003209c04109a01f9737593b3645ffe7aef28e16530d84e53ba0caca77fca4b"
RUNNER_SHA = "71ef1e815ce21a0e52b2aff37257f570c4b07fa81531de9f38e1681c695e8693"
SCHEMA = "lipnet-step05-full-comparison-plan-v1"
SEEDS = (42, 43)
ARMS = ("A_ctc", "B_feature")


def load_runner():
    """Import the byte-identical verified Step3 implementation."""
    import hashlib
    path = STEP3 / "runner.py"
    if hashlib.sha256(path.read_bytes()).hexdigest() != RUNNER_SHA:
        raise ValueError("The frozen Step3 engine changed")
    spec = importlib.util.spec_from_file_location("step05_frozen_step03_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load_runner()
core = runner.core


def bound_json(ref: dict) -> dict:
    """Read only a hash-matching workspace JSON artifact."""
    path = Path(ref["path"]).resolve()
    if not path.is_relative_to(ROOT) or core.bind(path) != ref:
        raise ValueError("Changed or nonlocal execution evidence")
    return core.read(path)


def execution_code() -> list[dict]:
    """Freeze loader, training, evaluation and the original model implementation."""
    paths = [BASE / name for name in ("train_full.py", "test_train_full.py", "full_artifacts.py",
                                      "test_full_artifacts.py", "evaluate.py", "test_evaluate.py")]
    paths.extend([STEP3 / "runner.py", ROOT / "experiments/feature_transfer_step07/train_compare.py",
                  ROOT / "backend/models/baseline.py", ROOT / "backend/models/__init__.py",
                  ROOT / "backend/models/tokenizer.py", ROOT / "experiments/olkavs_official/evaluation_metrics.py",
                  ROOT / "experiments/olkavs_official/diagnostic_utils.py"])
    return [core.bind(path) for path in paths]


def handoff() -> tuple[dict, dict, dict]:
    """Require the accepted full-data handoff and original comparison recipe."""
    ref = {"path": str(STEP4 / "full_data_plan.json"), "sha256": HANDOFF_SHA}
    data = bound_json(ref)
    recipe = bound_json(data["artifacts"]["recipe"])
    if (data["status"] != "data_and_schedules_frozen_pending_step05_execution_binding"
            or recipe["seeds"] != list(SEEDS) or recipe["arms"] != list(ARMS)
            or recipe["updates_per_seed_per_arm"] != 3200
            or recipe["accumulated_clips_per_update"] != 2
            or recipe["checkpoint_interval_updates"] != 100):
        raise ValueError("Full-data comparison budget changed")
    return ref, data, recipe


def schedule_for(data: dict, seed: int) -> list[list[str]]:
    """Require exact frozen schedule coverage and exclusion of development IDs."""
    if seed not in SEEDS:
        raise ValueError("Only the two frozen comparison seeds are allowed")
    selected = [s for s in data["training_schedules"] if s["seed"] == seed]
    if len(selected) != 1:
        raise ValueError("Missing or duplicate seeded schedule")
    item = selected[0]
    expected = core.fixed_schedule(data["train_ids"], seed, 3200, 2)
    if (item["schedule"] != expected or item["arms"] != list(ARMS)
            or item["updates"] != 3200 or item["clips_per_update"] != 2
            or set(sid for batch in expected for sid in batch) != set(data["train_ids"])
            or set(data["train_ids"]) & set(data["development_ids"])):
        raise ValueError("Frozen schedule or train/dev separation differs")
    return item["schedule"]


def plan_path(seed: int) -> Path:
    """Resolve a bounded seed plan path."""
    if seed not in SEEDS:
        raise ValueError("Unexpected seed")
    return BASE / "plans" / f"seed{seed}.json"


def check_plan_contract(plan: dict, seed: int, data: dict, recipe: dict) -> None:
    """Reject a changed budget, algorithm, cohort, schedule or canonical identity."""
    if (plan.get("schema") != SCHEMA or plan.get("seed") != seed
            or plan.get("updates") != 3200 or plan.get("batch_size") != 2
            or plan.get("checkpoint_interval") != 100 or plan.get("arms") != list(ARMS)
            or plan.get("ids") != data["train_ids"] or plan.get("development_ids") != data["development_ids"]
            or plan.get("schedule") != schedule_for(data, seed)
            or plan.get("optimizer") != recipe["optimizer"] or plan.get("loss") != recipe["loss"]
            or plan.get("primary_checkpoint") != recipe["primary_checkpoint"]
            or plan.get("evaluation") != recipe["evaluation"]
            or plan.get("plan_sha256") != runner.canonical_plan_hash(plan)):
        raise ValueError("Full execution contract was changed")


def freeze() -> dict:
    """Bind the actual full train payload and both complete execution plans."""
    import full_artifacts
    core.setup_runtime()
    if not runner.torch.cuda.is_available():
        raise ValueError("Validated GPU runtime is unavailable")
    ref, data, recipe = handoff()
    previous = bound_json(recipe["preflight_plan"])
    if core.runtime_environment() != previous["runtime"]:
        raise ValueError("Training runtime differs from the accepted Step3 preflight")
    code = execution_code()
    loaded = full_artifacts.load_full(scope="train")
    if loaded["ids"] != data["train_ids"] or loaded["handoff"] != ref:
        raise ValueError("Full loader returned another cohort or handoff")
    plans = []
    for seed in SEEDS:
        plan = {"schema": SCHEMA, "seed": seed, "updates": 3200, "batch_size": 2,
                "checkpoint_interval": 100, "ids": data["train_ids"],
                "development_ids": data["development_ids"], "arms": list(ARMS),
                "schedule": schedule_for(data, seed), "full_data_plan": ref,
                "comparison_recipe": data["artifacts"]["recipe"],
                "step03_preflight_plan": recipe["preflight_plan"],
                "step03_replay_audit": recipe["replay_audit"],
                "parent_plan": loaded["parent_plan"], "source_plan": loaded["source_plan"],
                "input_manifest": loaded["input_manifest"], "feature_manifest": loaded["feature_manifest"],
                "teacher_plan": loaded["teacher_plan"], "data_bindings": loaded["bindings"],
                "execution_code": code, "runtime": runner._runtime(runner.torch.device("cuda:0")),
                "optimizer": recipe["optimizer"], "loss": recipe["loss"],
                "initialization": recipe["model_initialization"], "precision": recipe["precision"],
                "primary_checkpoint": recipe["primary_checkpoint"], "evaluation": recipe["evaluation"],
                "scope": "train441; fresh A/B perseed; exactly3200updates each; final dev71 normal/static/shuffled",
                "service_weights_updated": False, "audio_used": False,
                "recovery": "same plan/arm/state and budget in a new output directory from a complete hashed checkpoint"}
        validated = runner.validate_plan(plan, loaded["clips"], "B_feature")
        plan["training_data"] = {k: validated[k] for k in ("data_sha256", "clips")}
        plan["plan_sha256"] = runner.canonical_plan_hash(plan)
        check_plan_contract(plan, seed, data, recipe)
        path = plan_path(seed)
        if path.exists():
            if core.read(path) != plan:
                raise ValueError("Existing full execution plan differs; preserve it")
        else:
            runner._save_json(path, plan)
        plans.append(core.bind(path))
    return {"status": "frozen", "plans": plans, "train_clips": 441,
            "student_optimization_steps": 0, "teacher_model_loaded": False}


def verify_plan(seed: int) -> tuple[dict, dict]:
    """Verify current code/runtime and all input bytes before every process."""
    import full_artifacts
    core.setup_runtime()
    ref, data, recipe = handoff()
    plan = core.read(plan_path(seed))
    check_plan_contract(plan, seed, data, recipe)
    if (plan["full_data_plan"] != ref or plan["execution_code"] != execution_code()
            or plan["runtime"] != runner._runtime(runner.torch.device("cuda:0"))):
        raise ValueError("Execution code, handoff or GPU runtime changed")
    loaded = full_artifacts.load_full(scope="train")
    if loaded["ids"] != plan["ids"] or loaded["bindings"] != plan["data_bindings"]:
        raise ValueError("Training payload bindings changed")
    actual = runner.validate_plan(plan, loaded["clips"], "B_feature")
    if plan["training_data"] != {k: actual[k] for k in ("data_sha256", "clips")}:
        raise ValueError("Decoded training pixels, tokens or feature arrays changed")
    return plan, loaded


def execute(seed: int, arm: str, resume: Path | None = None, recovery_name: str | None = None) -> dict:
    """Execute exactly one fixed arm or continue its complete saved state."""
    if arm not in ARMS:
        raise ValueError("Unexpected arm")
    if shutil.disk_usage(BASE).free < 10 * 1024**3:
        raise ValueError("At least10GiB free is required for checkpoints and reserve")
    started = time.perf_counter()
    plan, loaded = verify_plan(seed)
    if resume is None:
        if recovery_name is not None:
            raise ValueError("Recovery name without checkpoint")
        output = BASE / "runs" / f"seed{seed}" / arm / "continuous"
    else:
        resume = resume.resolve()
        if not resume.is_relative_to(BASE / "runs") or resume.suffix != ".pt":
            raise ValueError("Recovery requires this experiment's own checkpoint")
        if not recovery_name or not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", recovery_name):
            raise ValueError("Recovery requires a simple isolated name")
        output = BASE / "runs" / f"seed{seed}" / arm / "recoveries" / recovery_name
    result = runner.run_segment(plan, loaded["clips"], arm, output,
                                stop_update=3200, resume_checkpoint=resume, device="cuda:0")
    result.update(status="completed", plan_file=core.bind(plan_path(seed)), seed=seed,
                  scope=plan["scope"], elapsed_seconds=time.perf_counter() - started,
                  full_data_plan=plan["full_data_plan"], dev_inputs_loaded=False,
                  audio_used=False, teacher_model_loaded=False, service_weights_updated=False)
    runner._save_json(output / "run_result.json", result)
    print(json.dumps({"status": "completed", "seed": seed, "arm": arm,
                      "run_result": core.bind(output / "run_result.json"),
                      "elapsed_seconds": result["elapsed_seconds"]}), flush=True)
    return result


def write_progress(value: dict) -> None:
    """Publish a replaceable progress view, separate from immutable evidence."""
    path = BASE / "results/training_progress.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for attempt in range(50):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            # Windows readers may briefly hold a handle without delete sharing.
            # The progress display must not interrupt the checkpointed worker.
            time.sleep(.01)
    print("Progress view is temporarily locked; immutable run evidence continues.", flush=True)


def completed_run_result(seed: int, arm: str) -> dict | None:
    """Recognize one verified final run, including an explicit recovery result."""
    import evaluate
    home = BASE / "runs" / f"seed{seed}" / arm
    candidates = [p for p in [home / "continuous/run_result.json",
                              *sorted((home / "recoveries").glob("*/run_result.json"))] if p.is_file()]
    if not candidates:
        return None
    if len(candidates) != 1:
        raise ValueError("Ambiguous multiple final runs; do not choose a favorable trajectory")
    path = candidates[0]
    result, plan = core.read(path), core.read(plan_path(seed))
    ref, data, recipe = handoff()
    check_plan_contract(plan, seed, data, recipe)
    if (result.get("status") != "completed" or result.get("seed") != seed or result.get("arm") != arm
            or result.get("plan_file") != core.bind(plan_path(seed))
            or plan["full_data_plan"] != ref or plan["execution_code"] != execution_code()):
        raise ValueError("Existing run does not match the frozen full comparison")
    info = result["checkpoint"]
    checkpoint_path, sidecar = Path(info["path"]).resolve(), Path(info["sidecar"]).resolve()
    if (not checkpoint_path.is_relative_to(home) or sidecar != checkpoint_path.with_suffix(".pt.json")
            or core.sha(checkpoint_path) != info["sha256"] or core.sha(sidecar) != info["sidecar_sha256"]):
        raise ValueError("Existing result checkpoint/sidecar binding differs")
    checkpoint = runner.load_checkpoint(checkpoint_path)
    evaluate.validate_final_payload(plan, result, checkpoint)
    return core.bind(path)


def run_all() -> dict:
    """Run the four previously frozen comparisons sequentially in fresh processes."""
    for seed in SEEDS:
        if not plan_path(seed).is_file():
            raise ValueError("Freeze both full-data plans before starting any run")
    logs = BASE / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    completed = []
    for seed in SEEDS:
        for arm in ARMS:
            output = BASE / "runs" / f"seed{seed}" / arm / "continuous"
            result_path = output / "run_result.json"
            accepted = completed_run_result(seed, arm)
            if accepted is not None:
                completed.append(accepted)
                continue
            if output.exists() and any(output.iterdir()):
                raise ValueError("Incomplete prior arm detected; resume a verified checkpoint explicitly")
            progress = {"status": "running", "seed": seed, "arm": arm, "update": 0,
                        "updates_per_arm": 3200, "completed_arms": completed,
                        "started_utc": datetime.now(timezone.utc).isoformat()}
            write_progress(progress)
            command = [sys.executable, "-u", str(Path(__file__)), "--seed", str(seed), "--arm", arm]
            log = logs / f"seed{seed}_{arm}.log"
            with log.open("x", encoding="utf-8", newline="\n") as stream:
                with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                      text=True, encoding="utf-8", errors="replace",
                                      creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)) as child:
                    progress["worker_pid"] = child.pid
                    write_progress(progress)
                    for line in child.stdout:
                        stream.write(line)
                        stream.flush()
                        try:
                            row = json.loads(line)
                        except (ValueError, TypeError):
                            continue
                        if row.get("arm") == arm and type(row.get("update")) is int:
                            progress.update(update=row["update"], loss=row["loss"],
                                            last_update_seconds=row["seconds"],
                                            updated_utc=datetime.now(timezone.utc).isoformat())
                            if row["update"] % 10 == 0:
                                write_progress(progress)
                            if row["update"] % 100 == 0:
                                print(json.dumps({"seed": seed, **row}), flush=True)
                    exit_code = child.wait()
            if exit_code != 0 or not result_path.is_file():
                progress.update(status="failed", exit_code=exit_code, log=str(log))
                write_progress(progress)
                raise RuntimeError(f"Stopped at seed{seed}/{arm}; inspect {log}")
            completed.append(core.bind(result_path))
    result = {"status": "completed", "run_results": completed, "optimizer_updates": 12800,
              "evaluation_executed": False, "service_weights_updated": False}
    path = BASE / "results/training_complete.json"
    runner._save_json(path, result)
    write_progress(result)
    return result


def main() -> None:
    """Expose plan freeze, four-run execution, or bounded same-plan recovery."""
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--freeze", action="store_true")
    actions.add_argument("--all", action="store_true")
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--recovery-name")
    args = parser.parse_args()
    if args.freeze:
        result = freeze()
    elif args.all:
        result = run_all()
    else:
        if args.seed is None or args.arm is None:
            parser.error("--seed and --arm are required")
        result = execute(args.seed, args.arm, args.resume, args.recovery_name)
    print(json.dumps({key: value for key, value in result.items() if key in ("status", "plans", "optimizer_updates", "elapsed_seconds")}), flush=True)


if __name__ == "__main__":
    main()
