"""Stage13: reproduce stage12 B, then extend the same train64 trajectory to6400.

All original stage12 computation sources remain immutable. A strict numeric
prefix gate runs before attempt1601. Checkpoints and full training states have
distinct schemas; neither is a generalization-selected or deployed model.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "backend"), str(ROOT / "scripts")]
import diagnose_learning_transition as prior
from models.dataset_manifest import load_manifest, sha256_file
from transition_diagnostics import make_training_schedule, parameter_groups, select_transition_cohorts
from learning_budget_state import (save_training_state, compare_state_dicts,
    compare_telemetry, compare_schedule_prefix)

PRIOR_RUN = ROOT / "results/step12-transition/run01"
STEPS = [*prior.STEPS, 2400, 3200, 4800, 6400]
CONTROL_STEPS = [*prior.CONTROL_STEPS, 3200, 6400]
BEAM_STEPS = [*prior.BEAM_STEPS, 3200, 6400]
STATE_STEPS = [1600, 3200, 6400]
SOURCE_PATHS = [*prior.SOURCE_PATHS, "scripts/diagnose_learning_budget.py", "scripts/learning_budget_state.py"]
require, write_json, read_json = prior.require, prior.write_json, prior.read_json


def source_hashes() -> dict:
    """Record actual training, evaluation and state-persistence implementation."""
    return {name: sha256_file(ROOT / name) for name in SOURCE_PATHS}


def prior_snapshot(step: int) -> Path:
    """Resolve only the immutable stage12 common start or B snapshots."""
    return PRIOR_RUN / ("common/step_0000" if step == 0 else f"B_expand64/step_{step:04d}")


def prior_checkpoint(step: int) -> Path:
    """The source checkpoint for step0 predates both experiments."""
    return prior.INITIAL if step == 0 else PRIOR_RUN / f"B_expand64/step_{step:04d}.pt"


def create_plan(out: Path) -> dict:
    """Verify immutable stage12 inputs and freeze the budget before training."""
    require(not out.exists(), "Output exists; refusing to overwrite or resume implicitly")
    previous = read_json(PRIOR_RUN / "plan.json")
    completed = read_json(PRIOR_RUN / "completed.json")
    require(completed["status"] == "completed" and
            sha256_file(PRIOR_RUN / "plan.json") == completed["plan_sha256"], "Invalid prior run")
    environment = {"torch": str(torch.__version__), "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(), "numpy": np.__version__, "python": sys.version}
    require(environment == completed["environment"], "Stage12 runtime environment changed")
    require(prior.source_hashes() == previous["source_hashes"], "Stage12 sources changed")
    document, metadata = load_manifest(prior.MANIFEST, verify_files=False)
    require(metadata["manifest_sha256"] == prior.MANIFEST_SHA, "Manifest changed")
    require(select_transition_cohorts(document["records"]) == previous["cohorts"], "Train64 selection changed")
    require(sha256_file(prior.INITIAL) == prior.INITIAL_SHA, "Initial checkpoint changed")
    branch = read_json(PRIOR_RUN / "B_expand64/completed.json")
    require(sha256_file(PRIOR_RUN / "B_expand64/completed.json") == completed["branch_completed_sha256"]["B_expand64"],
            "Prior B completion changed")
    require(sha256_file(PRIOR_RUN / "B_expand64/telemetry.jsonl") == branch["telemetry_sha256"], "Prior telemetry changed")
    files = ["plan.json", "completed.json", "B_expand64/completed.json", "B_expand64/telemetry.jsonl"]
    for step in prior.STEPS:
        folder = prior_snapshot(step)
        summary = read_json(folder / "summary.json")
        expected_summary = (completed["common_summary_sha256"] if step == 0 else
            next(x["summary_sha256"] for x in branch["snapshots"] if x["step"] == step))
        require(sha256_file(folder / "summary.json") == expected_summary, "Prior snapshot summary changed")
        for filename, field in (("rows.jsonl", "rows_sha256"), ("posteriors.npz", "posteriors_sha256")):
            require(sha256_file(folder / filename) == summary[field], "Prior evaluation changed")
        require(sha256_file(prior_checkpoint(step)) == summary["checkpoint_sha256"], "Prior checkpoint changed")
        files.extend((folder / filename).relative_to(PRIOR_RUN).as_posix()
                     for filename in ("summary.json", "rows.jsonl", "posteriors.npz"))
        if step:
            files.append(prior_checkpoint(step).relative_to(PRIOR_RUN).as_posix())
    schedule = make_training_schedule(previous["cohorts"]["expanded64"], attempt_updates=6400)
    require(schedule["batches"][:1600] == previous["schedules"]["B_expand64"]["batches"], "Schedule prefix mismatch")
    settings = {**previous["settings"], "attempt_updates": 6400,
        "normal_steps": STEPS, "control_steps": CONTROL_STEPS, "beam_steps": BEAM_STEPS}
    plan = {"schema": "lipnet-learning-budget-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "prior_run": str(PRIOR_RUN), "prior_files": {name: sha256_file(PRIOR_RUN / name) for name in files},
        "manifest": str(prior.MANIFEST), "manifest_sha256": prior.MANIFEST_SHA,
        "initial_checkpoint": str(prior.INITIAL), "initial_checkpoint_sha256": prior.INITIAL_SHA,
        "source_hashes": source_hashes(), "cohorts": previous["cohorts"], "records": previous["records"],
        "settings": settings, "schedule": schedule,
        "environment": environment,
        "scope": {"train_only": True, "validation_cache_access": False, "test_cache_access": False,
            "generalization_evaluation": False, "deployment": False},
        "prefix_gate_policy": {"comparison": "exact", "before_attempt": 1601,
            "model": "every state_dict tensor at original evaluation steps",
            "outputs": "every raw posterior and prediction row at original evaluation steps",
            "telemetry": "all first1600 numeric/ID/counter values, excluding wall time seconds only",
            "on_mismatch": "write failure and stop before any further training; do not relax silently"},
        "resume_state_steps": STATE_STEPS, "nonfinite_policy": previous["nonfinite_policy"],
        "interpretation": ["Same64, same initialization and optimizer, same ordered batches; only budget extends.",
            "1600/3200/6400 are points on one trajectory, not independent seed replicates.",
            "No performance early stopping or best-checkpoint selection; report all predeclared points.",
            "Train memorization and temporal-control sensitivity do not prove webcam/generalization performance."]}
    out.mkdir(parents=True)
    write_json(out / "plan.json", plan)
    return plan


@torch.inference_mode()
def evaluate_snapshot(model: torch.nn.Module, plan: dict, cache: dict, records: dict, step: int,
                      folder: Path, checkpoint_sha: str) -> dict:
    """Use stage12 evaluation unchanged for its prefix and the same math afterwards."""
    if step in prior.STEPS:
        return prior.evaluate_snapshot(model, plan, cache, records, step, folder, checkpoint_sha)
    folder.mkdir()
    started = time.perf_counter()
    conditions = ["normal", "static_middle", "shuffle"] if step in CONTROL_STEPS else ["normal"]
    beam_performed = step in BEAM_STEPS
    arrays, rows = {}, []
    with prior.preserve_rng_and_mode(model):
        for index, sample_id in enumerate(plan["cohorts"]["expanded64"]):
            source, item = records[sample_id], cache[sample_id]
            normal, normal_text = None, None
            for condition in conditions:
                frames, transform = prior.transform_frames(item["frames"], condition, sample_id=sample_id, seed=42)
                lp, _ = prior.forward(model, frames, capture=False)
                greedy = prior.ctc_greedy_decode(lp)
                beam = prior.ctc_beam_search_decode(lp, beam_width=10) if beam_performed else None
                diagnostics = prior.analyze_ctc(lp, item["label"], beam_text=beam if beam_performed else "")
                if not beam_performed:
                    diagnostics = {k: v for k, v in diagnostics.items() if not k.startswith("beam")}
                target = torch.tensor(item["label"], dtype=torch.long)
                loss = torch.nn.functional.ctc_loss(torch.from_numpy(lp[:, None, :]), target,
                    torch.tensor([len(lp)]), torch.tensor([len(target)]), blank=0, reduction="mean", zero_infinity=False)
                require(bool(torch.isfinite(loss)), "Nonfinite evaluation loss")
                name = "shuffle_42" if condition == "shuffle" else condition
                key = f"sample_{index:03d}_{name}"
                arrays[key] = lp
                row = {"id": sample_id, "step": step, "split": "train", "condition": name,
                    "cohort": "short16" if sample_id in plan["cohorts"]["short16"] else "additional48",
                    "speaker_id": source["speaker_id"], "reference": source["text"], "frames": len(lp),
                    "greedy": prior.prediction_row(source, greedy),
                    "beam": prior.prediction_row(source, beam) if beam_performed else None,
                    "ctc": diagnostics, "ctc_loss_per_target_token": loss.item(),
                    "posterior_key": key, "transform": transform, "comparison_to_normal": None}
                if condition == "normal":
                    normal, normal_text = lp, greedy
                else:
                    row["comparison_to_normal"] = prior.compare_posteriors(normal, lp,
                        normal_text=normal_text, control_text=greedy)
                rows.append(row)
    with (folder / "rows.jsonl").open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    np.savez_compressed(folder / "posteriors.npz", **arrays)
    summary = {"step": step, "beam_performed": beam_performed,
        "conditions": ["shuffle_42" if c == "shuffle" else c for c in conditions],
        "rows": len(rows), "checkpoint_sha256": checkpoint_sha,
        "rows_sha256": sha256_file(folder / "rows.jsonl"),
        "posteriors_sha256": sha256_file(folder / "posteriors.npz"),
        "seconds": time.perf_counter() - started, "cohorts": {}}
    for cohort in ("short16", "additional48"):
        normal_rows = [r for r in rows if r["cohort"] == cohort and r["condition"] == "normal"]
        summary["cohorts"][cohort] = {"count": len(normal_rows),
            "greedy_exact": sum(r["greedy"]["exact"] for r in normal_rows),
            "greedy_mean_cer": float(np.mean([r["greedy"]["cer"] for r in normal_rows]))}
    write_json(folder / "summary.json", summary)
    print(json.dumps({"evaluation_step": step, **summary["cohorts"], "seconds": summary["seconds"]}), flush=True)
    return summary


def compare_prefix_snapshot(model: torch.nn.Module, step: int, folder: Path) -> dict:
    """Compare tensors and arrays, never serialization bytes or merely rounded CER."""
    with prior.preserve_rng_and_mode(model):
        previous = torch.load(prior_checkpoint(step), map_location="cpu", weights_only=True)
        state = compare_state_dicts(previous["state_dict"], model.state_dict())
        old_folder = prior_snapshot(step)
        old_rows = [json.loads(x) for x in (old_folder / "rows.jsonl").read_text(encoding="utf-8").splitlines()]
        new_rows = [json.loads(x) for x in (folder / "rows.jsonl").read_text(encoding="utf-8").splitlines()]
        rows_equal = old_rows == new_rows
        arrays_equal, count, max_difference = True, 0, 0.
        with np.load(old_folder / "posteriors.npz", allow_pickle=False) as old, np.load(folder / "posteriors.npz", allow_pickle=False) as new:
            arrays_equal = old.files == new.files
            if arrays_equal:
                for key in old.files:
                    a, b = old[key], new[key]
                    same = a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b)
                    arrays_equal = arrays_equal and same
                    if a.shape == b.shape:
                        max_difference = max(max_difference, float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64)))))
                    count += 1
        return {"step": step, "model_state": state, "prediction_rows_exact": rows_equal,
            "posterior_arrays_exact": arrays_equal, "posterior_arrays_compared": count,
            "posterior_max_absolute_difference": max_difference}


def snapshot_exact(result: dict) -> bool:
    """A numeric prefix passes only if every comparison category is exact."""
    return result["model_state"]["exact"] and result["prediction_rows_exact"] and result["posterior_arrays_exact"]


def finish_prefix_gate(plan: dict, out: Path, checks: list[dict], telemetry: list[dict]) -> dict:
    """Refuse the extension if the original1600-step trajectory was not reproduced."""
    require([r["step"] for r in checks] == prior.STEPS, "Prefix snapshot inventory incomplete")
    require(len(telemetry) == 1600, "Prefix telemetry must contain all1600 attempts")
    old_plan = read_json(PRIOR_RUN / "plan.json")
    old_rows = [json.loads(x) for x in (PRIOR_RUN / "B_expand64/telemetry.jsonl").read_text(encoding="utf-8").splitlines()]
    schedule = compare_schedule_prefix(old_plan["schedules"]["B_expand64"]["batches"], plan["schedule"]["batches"], attempts=1600)
    telemetry_result = compare_telemetry(old_rows, telemetry, excluded_keys=("seconds",))
    passed = schedule["exact"] and telemetry_result["exact"] and all(snapshot_exact(r) for r in checks)
    result = {"status": "passed" if passed else "failed", "checked_before_attempt": 1601,
        "plan_sha256": sha256_file(out / "plan.json"), "steps": [r["step"] for r in checks],
        "snapshots": checks, "schedule": schedule, "telemetry": telemetry_result,
        "inference_only_source": True, "prior_optimizer_state_available": False,
        "skipped_attempts": [r["attempt"] for r in telemetry if r["skipped"]]}
    write_json(out / "prefix_gate.json", result)
    require(passed, "Stage12 prefix did not reproduce exactly; attempt1601 is not authorized by this plan")
    print("PREFIX GATE PASSED: all model tensors, outputs, 1600 update records and ordered batches are exact", flush=True)
    return result


def run(out: Path) -> None:
    """Execute a fixed continuous run after its immutable-data and prefix gates."""
    require(torch.cuda.is_available(), "CUDA required")
    torch.set_num_threads(2)
    plan = create_plan(out)
    started = time.perf_counter()
    records = {r["id"]: r for r in plan["records"]}
    cache = {s: prior.load_train_cache(records[s]) for s in plan["cohorts"]["expanded64"]}
    prior.seed_training(42)
    model = prior.load_initial(plan["cohorts"])
    evaluate_snapshot(model, plan, cache, records, 0, out / "step_0000", prior.INITIAL_SHA)
    checks = [compare_prefix_snapshot(model, 0, out / "step_0000")]
    if not snapshot_exact(checks[-1]):
        write_json(out / "prefix_snapshot_failure.json", checks[-1])
        raise ValueError("Shared initial evaluation did not reproduce")
    del model
    torch.cuda.empty_cache()
    # Match stage12 B exactly: reset seed, instantiate/load, new Adam/scaler, train.
    prior.seed_training(42)
    model = prior.load_initial(plan["cohorts"]).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=.0003)
    scaler = torch.amp.GradScaler("cuda", enabled=True, init_scale=1024)
    groups = parameter_groups(model)
    torch.cuda.reset_peak_memory_stats()
    snapshots = [{"step": 0, "folder": "step_0000", "checkpoint": str(prior.INITIAL),
        "checkpoint_sha256": prior.INITIAL_SHA, "summary_sha256": sha256_file(out / "step_0000/summary.json")}]
    resume_states, prefix_rows = [], []
    updates, skips = 0, 0
    exposures = Counter({s: 0 for s in plan["cohorts"]["expanded64"]})
    successful = exposures.copy()
    provenance = {"mode": "smoke_only", "purpose": "train_subset_budget_diagnostic",
        "manifest_sha256": prior.MANIFEST_SHA, "sample_ids": plan["cohorts"]["expanded64"],
        "generalization_evaluation": False, "deployment": False}
    plan_sha = sha256_file(out / "plan.json")
    with (out / "telemetry.jsonl").open("x", encoding="utf-8") as stream:
        for entry in plan["schedule"]["batches"]:
            result = prior.optimizer_attempt(model, optimizer, scaler,
                prior.collate([cache[s] for s in entry["ids"]]), groups, torch.device("cuda"))
            skips += int(result["skipped"])
            updates += int(not result["skipped"])
            exposures.update(entry["ids"])
            if not result["skipped"]:
                successful.update(entry["ids"])
            result.update(entry)
            result.update(optimizer_steps=updates, skipped_optimizer_steps=skips,
                frames=[records[s]["frames"] for s in entry["ids"]],
                target_lengths=[len(cache[s]["label"]) for s in entry["ids"]])
            stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            step = entry["attempt"]
            if step <= 1600:
                prefix_rows.append(result)
            if step in STEPS:
                checkpoint = out / f"step_{step:04d}.pt"
                model.save_checkpoint(str(checkpoint), training_data=provenance, training_run={
                    "budget_diagnostic": True, "attempt_updates": step, "optimizer_steps": updates,
                    "skipped_optimizer_steps": skips, "plan_sha256": plan_sha,
                    "initial_checkpoint_sha256": prior.INITIAL_SHA, "inference_only": True,
                    "selection": "fixed_attempt_schedule", "settings": plan["settings"]})
                sha = sha256_file(checkpoint)
                folder = out / f"step_{step:04d}"
                evaluate_snapshot(model, plan, cache, records, step, folder, sha)
                snapshots.append({"step": step, "folder": folder.name, "checkpoint": checkpoint.name,
                    "checkpoint_sha256": sha, "summary_sha256": sha256_file(folder / "summary.json")})
                if step <= 1600:
                    comparison = compare_prefix_snapshot(model, step, folder)
                    checks.append(comparison)
                    if not snapshot_exact(comparison):
                        write_json(out / "prefix_snapshot_failure.json", comparison)
                        raise ValueError(f"Prefix snapshot mismatch at {step}; stopped without relaxing the gate")
            if step == 1600:
                finish_prefix_gate(plan, out, checks, prefix_rows)
                prefix_rows.clear()
            if step in STATE_STEPS:
                state_file = out / f"training_state_{step:04d}.pt"
                save_training_state(state_file, model=model, optimizer=optimizer, scaler=scaler,
                    attempt=step, schedule={k: plan["schedule"][k] for k in
                        ("sample_ids", "attempt_updates", "batch_size", "seed", "batches_sha256")},
                    provenance={**provenance, "plan_sha256": plan_sha},
                    source_hashes=plan["source_hashes"], settings=plan["settings"],
                    counters={"optimizer_steps": updates, "skipped_optimizer_steps": skips,
                        "attempted_exposures": dict(exposures), "successful_update_exposures": dict(successful)})
                resume_states.append({"step": step, "file": state_file.name, "sha256": sha256_file(state_file)})
            if step % 400 == 0:
                print(json.dumps({"attempt": step, "optimizer_steps": updates, "skips": skips,
                    "loss": result["loss"], "elapsed_seconds": time.perf_counter()-started}), flush=True)
    require(dict(exposures) == plan["schedule"]["exposures"], "Exposure count mismatch")
    require(source_hashes() == plan["source_hashes"], "Experiment source changed")
    for name, digest in plan["prior_files"].items():
        require(sha256_file(PRIOR_RUN / name) == digest, "Prior artifact changed")
    require(sha256_file(prior.INITIAL) == prior.INITIAL_SHA and sha256_file(prior.MANIFEST) == prior.MANIFEST_SHA,
            "Initial checkpoint or manifest changed")
    write_json(out / "completed.json", {"status": "completed", "plan_sha256": plan_sha,
        "prefix_gate_sha256": sha256_file(out / "prefix_gate.json"), "snapshots": snapshots,
        "resume_states": resume_states, "attempt_updates": 6400,
        "optimizer_steps": updates, "skipped_optimizer_steps": skips,
        "attempted_exposures": dict(exposures), "successful_update_exposures": dict(successful),
        "telemetry_sha256": sha256_file(out / "telemetry.jsonl"),
        "train_cache_ids_opened": list(cache), **plan["scope"],
        "seconds": time.perf_counter()-started,
        "peak_allocated_mib": torch.cuda.max_memory_allocated()/2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved()/2**20,
        "environment": {"torch": str(torch.__version__), "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(), "numpy": np.__version__, "python": sys.version}})


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results/step13-budget/run01")
    args = parser.parse_args()
    try:
        run(args.out.resolve())
    except Exception as exc:
        print(f"STAGE13 STOPPED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
