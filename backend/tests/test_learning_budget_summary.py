"""Saved stage13 artifacts are verified without model inference or GPU work."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "backend"), str(ROOT / "scripts"), str(Path(__file__).parent)]
import summarize_learning_budget as summary
import summarize_learning_transition as prior_summary
import test_learning_transition_summary as fixtures


def extended_snapshot(tmp_path, step):
    records = {s: fixtures.source(s) for s in ("old", "new")}
    plan = {"cohorts": {"short16": ["old"], "additional48": ["new"], "expanded64": ["old", "new"]}}
    folder = tmp_path / f"step_{step:04d}"
    old_step = 1600 if step == 3200 else 800
    rows, stored = fixtures.make_snapshot(folder, plan, records, old_step)
    for row in rows:
        row["step"] = step
    stored["step"] = step
    fixtures.write_rows(folder / "rows.jsonl", rows)
    stored["rows_sha256"] = summary.sha256_file(folder / "rows.jsonl")
    fixtures.write_json(folder / "summary.json", stored)
    return folder, plan, records, rows, stored


@pytest.mark.parametrize("step,beam,count", [(2400, False, 2), (3200, True, 6)])
def test_extended_snapshots_recompute_metrics_without_mutating_prior_protocol(tmp_path, step, beam, count, monkeypatch):
    import temporal_diagnostics
    monkeypatch.setattr(temporal_diagnostics, "ctc_beam_search_decode", lambda *a, **k: pytest.fail("No beam rerun"))
    prior = (copy.deepcopy(prior_summary.STEPS), copy.deepcopy(prior_summary.CONTROL_STEPS), copy.deepcopy(prior_summary.BEAM_STEPS))
    folder, plan, records, _, _ = extended_snapshot(tmp_path, step)
    result = summary.validate_snapshot(folder, plan, records, step, "a"*64)
    assert result["beam_performed"] is beam and result["rows"] == count
    assert (prior_summary.STEPS, prior_summary.CONTROL_STEPS, prior_summary.BEAM_STEPS) == prior


@pytest.mark.parametrize("mutation", ["ctc", "loss", "beam", "scope", "control"])
def test_extended_snapshot_rejects_rehashed_semantic_mutations(tmp_path, mutation):
    folder, plan, records, rows, stored = extended_snapshot(tmp_path, 3200)
    if mutation == "ctc":
        rows[0]["ctc"]["blank_posterior_mean"] += .1
    elif mutation == "loss":
        rows[0]["ctc_loss_per_target_token"] += .1
    elif mutation == "beam":
        rows[0]["beam"] = None
    elif mutation == "scope":
        rows[0]["split"] = "test"
    else:
        rows[1]["comparison_to_normal"]["mean_total_variation"] += .1
    fixtures.write_rows(folder / "rows.jsonl", rows)
    stored["rows_sha256"] = summary.sha256_file(folder / "rows.jsonl")
    fixtures.write_json(folder / "summary.json", stored)
    with pytest.raises(ValueError):
        summary.validate_snapshot(folder, plan, records, 3200, "a"*64)


def test_extended_fixed_windows_are_disjoint_attempt_ranges():
    template = fixtures.telemetry_fixture(1)[0][0]
    endpoints = [1, 100, 101, 400, 401, 800, 801, 1600, 1601, 2400, 2401, 3200, 3201, 4800, 4801, 6400]
    rows = [{**copy.deepcopy(template), "attempt": attempt} for attempt in endpoints]
    result = summary.fixed_window_summary(rows)
    assert list(result) == [f"{a:04d}-{b:04d}" for a, b in summary.WINDOWS]
    assert all(value["attempts_observed"] == 2 for value in result.values())
    assert sum(value["attempts_observed"] for value in result.values()) == 16


def prefix_fixture(tmp_path):
    prior, run = tmp_path / "prior", tmp_path / "run"
    prior.mkdir()
    run.mkdir()
    batches = [{"attempt": i, "ids": ["a", "b"]} for i in range(1, 1601)]
    fixtures.write_json(prior / "plan.json", {"schedules": {"B_expand64": {"batches": batches}}})
    (prior / "B_expand64").mkdir()
    telemetry = [{"attempt": i, "loss": .01, "seconds": float(i), "skipped": False} for i in range(1, 1601)]
    fixtures.write_rows(prior / "B_expand64/telemetry.jsonl", telemetry)
    new_telemetry = [{**row, "seconds": 0.} for row in telemetry]
    initial = tmp_path / "initial.pt"
    torch.save({"state_dict": {"weight": torch.ones(3)}}, initial)
    old_specs, new_specs = [], []
    for step in (0, 1600):
        old_rel = "common/step_0000" if step == 0 else "B_expand64/step_1600"
        for root, relative in ((prior, old_rel), (run, f"step_{step:04d}")):
            folder = root / relative
            folder.mkdir(parents=True)
            fixtures.write_rows(folder / "rows.jsonl", [{"id": "a", "step": step}])
            np.savez_compressed(folder / "posteriors.npz", a=np.ones((4, 69), np.float32))
        if step:
            torch.save({"state_dict": {"weight": torch.ones(3)}}, prior / "B_expand64/step_1600.pt")
            torch.save({"state_dict": {"weight": torch.ones(3)}}, run / "step_1600.pt")
            old_specs.append({"step": step, "folder": old_rel, "checkpoint": "B_expand64/step_1600.pt"})
        new_specs.append({"step": step, "folder": f"step_{step:04d}",
                          "checkpoint": str(initial) if step == 0 else "step_1600.pt"})
    plan = {"prior_run": str(prior), "initial_checkpoint": str(initial), "schedule": {"batches": batches}}
    fixtures.write_json(run / "plan.json", plan)
    state = {"weight": torch.ones(3)}
    gate = {"status": "passed", "checked_before_attempt": 1601,
        "plan_sha256": summary.sha256_file(run / "plan.json"), "steps": [0, 1600],
        "snapshots": [{"step": step, "model_state": summary.compare_state_dicts(state, state),
            "prediction_rows_exact": True, "posterior_arrays_exact": True, "posterior_arrays_compared": 1,
            "posterior_max_absolute_difference": 0.} for step in (0, 1600)],
        "schedule": summary.compare_schedule_prefix(batches, batches, attempts=1600),
        "telemetry": summary.compare_telemetry(telemetry, new_telemetry, excluded_keys=("seconds",)),
        "inference_only_source": True, "prior_optimizer_state_available": False, "skipped_attempts": []}
    fixtures.write_json(run / "prefix_gate.json", gate)
    completed = {"snapshots": new_specs, "prefix_gate_sha256": summary.sha256_file(run / "prefix_gate.json")}
    return run, plan, completed, new_telemetry, {"snapshots": old_specs}


def test_prefix_reproduction_checks_raw_artifacts_allowing_runtime_difference(tmp_path):
    args = prefix_fixture(tmp_path)
    result = summary.validate_prefix(*args)
    assert result["passed"] and result["telemetry_attempts_exact_excluding_seconds"] == 1600
    assert [row["step"] for row in result["snapshots"]] == [0, 1600]
    assert all(row["model_tensors_exact"] and row["posteriors_exact"] for row in result["snapshots"])


@pytest.mark.parametrize("mutation", ["schedule", "telemetry", "rows", "posteriors", "weight", "dtype", "gate"])
def test_reproduction_cannot_be_claimed_with_altered_artifacts(tmp_path, mutation):
    run, plan, completed, telemetry, branch = prefix_fixture(tmp_path)
    if mutation == "schedule":
        plan["schedule"]["batches"][99]["ids"].reverse()
    elif mutation == "telemetry":
        telemetry[99]["loss"] += 1e-12
    elif mutation == "rows":
        fixtures.write_rows(run / "step_1600/rows.jsonl", [{"id": "changed", "step": 1600}])
    elif mutation == "posteriors":
        array = np.ones((4, 69), np.float32)
        array[0, 0] = 2
        np.savez_compressed(run / "step_1600/posteriors.npz", a=array)
    elif mutation == "weight":
        torch.save({"state_dict": {"weight": torch.ones(3) + .1}}, run / "step_1600.pt")
    elif mutation == "dtype":
        torch.save({"state_dict": {"weight": torch.ones(3, dtype=torch.float64)}}, run / "step_1600.pt")
    else:
        fixtures.write_json(run / "prefix_gate.json", {"status": "failed"})
        completed["prefix_gate_sha256"] = summary.sha256_file(run / "prefix_gate.json")
    with pytest.raises(ValueError):
        summary.validate_prefix(run, plan, completed, telemetry, branch)


def test_length_bins_use_fixed_boundaries_and_only_additional_normal_clips():
    rows = []
    for index, frames in enumerate((1, 50, 51, 100, 101, 150, 151, 200)):
        row, _ = fixtures.make_row(fixtures.source(f"s{index}"), "additional48")
        row["frames"] = frames
        rows.append(row)
    old, _ = fixtures.make_row(fixtures.source("old"), "short16")
    static = copy.deepcopy(rows[0])
    static["condition"] = "static_middle"
    result = summary.summarize_length_bins([*rows, old, static])
    assert list(result) == ["001-050", "051-100", "101-150", "151-200"]
    assert all(value["summary"]["count"] == 2 for value in result.values())
    assert sum(value["summary"]["count"] for value in result.values()) == 8
    rows[0]["frames"] = 201
    with pytest.raises(ValueError, match="outside"):
        summary.summarize_length_bins(rows)


def test_sentence_trajectory_contains_all_ids_and_rejects_missing_or_changed_reference():
    rows = [fixtures.make_row(fixtures.source(s), "short16" if s == "old" else "additional48")[0]
            for s in ("old", "new")]
    by_step = {0: rows, 1600: copy.deepcopy(rows), 3200: copy.deepcopy(rows), 6400: copy.deepcopy(rows)}
    result = summary.sentence_trajectories(by_step, ["old", "new"])
    assert [row["id"] for row in result] == ["old", "new"]
    assert all(list(row["snapshots"]) == ["0000", "1600", "3200", "6400"] for row in result)
    by_step[6400][1]["reference"] = "나"
    with pytest.raises(ValueError, match="trajectory"):
        summary.sentence_trajectories(by_step, ["old", "new"])
    by_step[6400].pop()
    with pytest.raises(ValueError, match="scope"):
        summary.sentence_trajectories(by_step, ["old", "new"])


def test_incomplete_or_existing_summary_is_not_written(tmp_path, monkeypatch):
    fixtures.write_json(tmp_path / "plan.json", {})
    fixtures.write_json(tmp_path / "completed.json", {"status": "running"})
    output = tmp_path / "output.json"
    monkeypatch.setattr(sys, "argv", ["summarize", "--run", str(tmp_path), "--out", str(output)])
    with pytest.raises(ValueError, match="not completed"):
        summary.main()
    assert not output.exists()
    output.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        summary.main()
    assert output.read_bytes() == b"keep"


def resume_fixture(tmp_path):
    sample_ids = ["old", "new"]
    schedule = {"sample_ids": sample_ids, "attempt_updates": 6400, "batch_size": 2,
                "seed": 42, "batches_sha256": "b"*64}
    plan = {"resume_state_steps": [1600, 3200, 6400], "schedule": schedule,
            "cohorts": {"expanded64": sample_ids}, "manifest_sha256": "m"*64,
            "source_hashes": {}, "settings": {"learning_rate": .0003}}
    environment = {"torch": "test", "numpy": "test", "cuda": "test"}
    completed = {"resume_states": [], "snapshots": [], "plan_sha256": "p"*64, "environment": environment}
    telemetry = [{"attempt": i, "ids": sample_ids, "skipped": False, "optimizer_steps": i,
                  "skipped_optimizer_steps": 0, "scale_after": 1024. * 2**(i//2000)} for i in range(1, 6401)]
    weights = {"weight": torch.ones(2)}
    for step in (1600, 3200, 6400):
        checkpoint = tmp_path / f"step_{step:04d}.pt"
        torch.save({"state_dict": weights}, checkpoint)
        value = {"schema": "lipnet-learning-budget-resume-v1", "inference_only": False, "deployment": False,
            "boundary": "after_scaler_update_before_next_zero_grad", "attempt": step,
            "model_class": "models.baseline.LipNetBaseline", "optimizer_class": "torch.optim.adam.Adam",
            "scaler_class": "torch.amp.grad_scaler.GradScaler", "scaler_enabled": True, "model_training": True,
            "source_hashes": {}, "settings": plan["settings"], "schedule": schedule,
            "provenance": {"mode": "smoke_only", "purpose": "train_subset_budget_diagnostic",
                "manifest_sha256": "m"*64, "sample_ids": sample_ids, "generalization_evaluation": False,
                "deployment": False, "plan_sha256": "p"*64},
            "counters": {"optimizer_steps": step, "skipped_optimizer_steps": 0,
                "attempted_exposures": {s: step for s in sample_ids},
                "successful_update_exposures": {s: step for s in sample_ids}},
            "model_state_dict": weights,
            "parameter_signature": [{"name": "weight", "shape": [2], "dtype": "torch.float32", "requires_grad": True}],
            "optimizer_parameter_names": [["weight"]],
            "optimizer_state_dict": {
                "param_groups": [{"params": [0], "lr": .0003, "eps": 1e-8, "weight_decay": 0.,
                    "amsgrad": False, "maximize": False, "capturable": False, "differentiable": False, "betas": (.9, .999)}],
                "state": {0: {"step": torch.tensor(float(step)), "exp_avg": torch.ones(2), "exp_avg_sq": torch.ones(2)}}},
            "scaler_state_dict": {"scale": 1024.*2**(step//2000), "growth_factor": 2., "backoff_factor": .5,
                "growth_interval": 2000, "_growth_tracker": step%2000},
            "rng_state": {"python": (3, (), None), "torch_cpu": torch.zeros(8, dtype=torch.uint8),
                "torch_cuda": [torch.zeros(8, dtype=torch.uint8)], "numpy": {"bit_generator": "MT19937",
                    "keys": torch.zeros(624, dtype=torch.int64), "position": 0, "has_gauss": 0, "cached_gaussian": 0.}},
            "module_training": {"": True}, "backend_state": {"torch_threads": 2, "cudnn_benchmark": False,
                "cudnn_deterministic": True, "cudnn_allow_tf32": False, "matmul_allow_tf32": False},
            "environment": environment}
        path = tmp_path / f"training_state_{step:04d}.pt"
        torch.save(value, path)
        completed["snapshots"].append({"step": step, "checkpoint": checkpoint.name})
        completed["resume_states"].append({"step": step, "file": path.name, "sha256": summary.sha256_file(path)})
    return tmp_path, plan, completed, telemetry


def test_resume_states_are_cross_checked_without_restoring_rng_or_optimizer(tmp_path, monkeypatch):
    import learning_budget_state
    monkeypatch.setattr(learning_budget_state, "restore_rng_state", lambda *a: pytest.fail("No RNG restore"))
    result = summary.validate_resume_states(*resume_fixture(tmp_path))
    assert [row["step"] for row in result] == [1600, 3200, 6400]
    assert all(row["checkpoint_tensors_exact"] and row["adam_parameter_states"] == 1 for row in result)


@pytest.mark.parametrize("mutation", ["purpose", "weight", "moment_nan", "counter", "scale", "rng_dtype", "exposure", "mode"])
def test_resume_metadata_and_moments_cannot_be_hidden_by_rehashing(tmp_path, mutation):
    args = resume_fixture(tmp_path)
    run, plan, completed, telemetry = args
    path = run / "training_state_1600.pt"
    value = torch.load(path, map_location="cpu", weights_only=True)
    if mutation == "purpose":
        value["provenance"]["purpose"] = "train_subset_overfit"
    elif mutation == "weight":
        value["model_state_dict"]["weight"][0] = 2.
    elif mutation == "moment_nan":
        value["optimizer_state_dict"]["state"][0]["exp_avg"][0] = float("nan")
    elif mutation == "counter":
        value["optimizer_state_dict"]["state"][0]["step"] += 1.
    elif mutation == "scale":
        value["scaler_state_dict"]["scale"] *= 2
    elif mutation == "rng_dtype":
        value["rng_state"]["torch_cuda"][0] = torch.zeros(8, dtype=torch.float32)
    elif mutation == "exposure":
        value["counters"]["successful_update_exposures"]["new"] -= 1
    else:
        value["module_training"][""] = False
    torch.save(value, path)
    completed["resume_states"][0]["sha256"] = summary.sha256_file(path)
    with pytest.raises(ValueError):
        summary.validate_resume_states(run, plan, completed, telemetry)
