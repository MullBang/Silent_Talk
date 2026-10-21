"""Stage14 saved-artifact verification rejects substantive paired-run mistakes."""
from __future__ import annotations

import copy
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "backend"), str(ROOT / "scripts"), str(Path(__file__).parent)]
import summarize_learning_rate as summary
import summarize_learning_budget as budget
import test_learning_transition_summary as fixtures
import test_learning_budget_summary as budget_fixtures
from transition_diagnostics import GROUP_PREFIXES, make_training_schedule


def continuation_fixture():
    rows, _, records, settings = fixtures.telemetry_fixture(4)
    schedule = make_training_schedule(["a", "b"], attempt_updates=12)
    initial = {"attempt": 8, "counters": {"optimizer_steps": 7, "skipped_optimizer_steps": 1,
        "attempted_exposures": {"a": 8, "b": 8}, "successful_update_exposures": {"a": 7, "b": 7}},
        "model_state_dict": {prefix + "weight": torch.ones(2) for prefix in GROUP_PREFIXES.values()},
        "scaler_state_dict": {"scale": 1024., "growth_factor": 2., "backoff_factor": .5,
                              "growth_interval": 2000, "_growth_tracker": 1999}}
    for row, batch, scale_before, scale_after in zip(rows, schedule["batches"][8:],
                                                   [1024., 2048., 2048., 1024.], [2048., 2048., 1024., 1024.]):
        row.update(copy.deepcopy(batch))
        row["optimizer_steps"] += 7
        row["skipped_optimizer_steps"] += 1
        row["scale_before"], row["scale_after"] = scale_before, scale_after
    return rows, schedule, records, settings, initial


def test_resumed_telemetry_preserves_absolute_attempts_and_counters():
    args = continuation_fixture()
    before = copy.deepcopy(args)
    result = summary.validate_extension_telemetry(*args)
    assert result["extension"]["attempt_updates"] == 4
    assert result["extension"]["optimizer_steps"] == 3
    assert result["cumulative"] == {"optimizer_steps": 10, "skipped_optimizer_steps": 2,
        "attempted_exposures": {"a": 12, "b": 12}, "successful_update_exposures": {"a": 10, "b": 10}}
    assert result["final_scaler_state"]["scale"] == 1024.
    assert result["final_scaler_state"]["_growth_tracker"] == 1
    assert summary.compare_nested(before, args)["exact"]


@pytest.mark.parametrize("mutation", ["reset_counter", "fresh_scaler", "tracker", "source_weights", "learning_rate", "schedule", "skip"])
def test_continuation_rejects_resumed_state_and_telemetry_mismatch(mutation):
    rows, schedule, records, settings, initial = continuation_fixture()
    if mutation == "reset_counter":
        rows[0]["optimizer_steps"] -= 7
    elif mutation == "fresh_scaler":
        rows[0]["scale_before"] = 512.
    elif mutation == "tracker":
        initial["scaler_state_dict"]["_growth_tracker"] = 0
    elif mutation == "source_weights":
        initial["model_state_dict"]["fc.weight"] += .1
    elif mutation == "learning_rate":
        rows[0]["learning_rate"] = .0001
    elif mutation == "schedule":
        rows[0]["ids"].reverse()
    else:
        rows[2]["skipped"] = False
    with pytest.raises(ValueError):
        summary.validate_extension_telemetry(rows, schedule, records, settings, initial)


def initial_pair_fixture():
    source = {key: {"value": torch.ones(2)} for key in (
        "schema", "inference_only", "deployment", "boundary", "attempt", "model_class", "optimizer_class",
        "scaler_class", "scaler_enabled", "parameter_signature", "optimizer_parameter_names",
        "model_state_dict", "scaler_state_dict", "rng_state", "model_training", "module_training",
        "backend_state", "counters", "environment")}
    source["optimizer_state_dict"] = {"state": {0: {"step": torch.tensor(6395.), "exp_avg": torch.ones(2)}},
                                      "param_groups": [{"lr": .0003, "params": [0], "betas": (.9, .999)}]}
    candidate = copy.deepcopy(source)
    candidate["optimizer_state_dict"]["param_groups"][0]["lr"] = .0001
    return source, candidate


def test_branch_initial_state_allows_only_declared_adam_learning_rate():
    source, candidate = initial_pair_fixture()
    assert summary.compare_initial_states(source, candidate, .0001)["adam_moments_and_step_counters_exact"]
    assert source["optimizer_state_dict"]["param_groups"][0]["lr"] == .0003


@pytest.mark.parametrize("mutation", ["weights", "adam_moment", "adam_step", "rng", "scaler", "backend", "lr", "beta"])
def test_branch_initial_state_rejects_other_differences(mutation):
    source, candidate = initial_pair_fixture()
    if mutation in ("weights", "rng", "scaler", "backend"):
        field = {"weights": "model_state_dict", "rng": "rng_state", "scaler": "scaler_state_dict", "backend": "backend_state"}[mutation]
        candidate[field]["value"][0] += 1e-7
    elif mutation == "adam_moment":
        candidate["optimizer_state_dict"]["state"][0]["exp_avg"][0] += 1e-7
    elif mutation == "adam_step":
        candidate["optimizer_state_dict"]["state"][0]["step"] -= 1
    elif mutation == "lr":
        candidate["optimizer_state_dict"]["param_groups"][0]["lr"] = .0003
    else:
        candidate["optimizer_state_dict"]["param_groups"][0]["betas"] = (.8, .999)
    with pytest.raises(ValueError):
        summary.compare_initial_states(source, candidate, .0001)


def snapshot_fixture(tmp_path, step):
    records = {s: fixtures.source(s) for s in ("old", "new")}
    plan = {"cohorts": {"short16": ["old"], "additional48": ["new"], "expanded64": ["old", "new"]}}
    folder = tmp_path / f"step_{step:04d}"
    old_step = 1600 if step in summary.BEAM_STEPS else 400 if step in summary.CONTROL_STEPS else 800
    rows, stored = fixtures.make_snapshot(folder, plan, records, old_step)
    for row in rows:
        row["step"] = step
    stored["step"] = step
    fixtures.write_rows(folder / "rows.jsonl", rows)
    stored["rows_sha256"] = summary.sha256_file(folder / "rows.jsonl")
    fixtures.write_json(folder / "summary.json", stored)
    return folder, plan, records, rows, stored


@pytest.mark.parametrize("step,beam,count", [(6401, False, 2), (7200, False, 6), (8000, True, 6)])
def test_snapshot_recomputes_metrics_with_declared_controls_and_beam(tmp_path, monkeypatch, step, beam, count):
    import temporal_diagnostics
    monkeypatch.setattr(temporal_diagnostics, "ctc_beam_search_decode", lambda *a, **k: pytest.fail("No beam rerun"))
    prior = (copy.deepcopy(budget.STEPS), copy.deepcopy(budget.CONTROL_STEPS), copy.deepcopy(budget.BEAM_STEPS))
    folder, plan, records, _, _ = snapshot_fixture(tmp_path, step)
    result = summary.validate_snapshot(folder, plan, records, step, "a" * 64)
    assert result["beam_performed"] is beam and result["rows"] == count
    assert (budget.STEPS, budget.CONTROL_STEPS, budget.BEAM_STEPS) == prior


@pytest.mark.parametrize("mutation", ["ctc", "loss", "beam", "scope", "control"])
def test_snapshot_rejects_rehashed_semantic_tampering(tmp_path, mutation):
    folder, plan, records, rows, stored = snapshot_fixture(tmp_path, 8000)
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
        summary.validate_snapshot(folder, plan, records, 8000, "a" * 64)


def full_state_fixture(tmp_path):
    budget_fixtures.resume_fixture(tmp_path)
    source = torch.load(tmp_path / "training_state_6400.pt", map_location="cpu", weights_only=True)
    candidate = copy.deepcopy(source)
    candidate["attempt"] = 9600
    candidate["optimizer_state_dict"]["param_groups"][0]["lr"] = .0001
    candidate["counters"] = {"optimizer_steps": 9598, "skipped_optimizer_steps": 2,
        "attempted_exposures": {"old": 9600, "new": 9600},
        "successful_update_exposures": {"old": 9598, "new": 9598}}
    candidate["optimizer_state_dict"]["state"][0]["step"] = torch.tensor(9598.)
    return source, candidate


def test_full_state_checks_adam_counters_and_checkpoint_without_model(tmp_path):
    source, candidate = full_state_fixture(tmp_path)
    result = summary.validate_full_state(candidate, source, candidate["model_state_dict"], step=9600,
        learning_rate=.0001, counters=candidate["counters"], scaler=candidate["scaler_state_dict"])
    assert result["optimizer_steps"] == 9598 and result["checkpoint_tensors_exact"]


@pytest.mark.parametrize("mutation", ["moment_nan", "negative_variance", "counter", "dtype", "rng", "mode", "adam_lr"])
def test_full_state_rejects_invalid_optimizer_or_rng_artifacts(tmp_path, mutation):
    source, candidate = full_state_fixture(tmp_path)
    checkpoint = copy.deepcopy(candidate["model_state_dict"])
    if mutation == "moment_nan":
        candidate["optimizer_state_dict"]["state"][0]["exp_avg"][0] = float("nan")
    elif mutation == "negative_variance":
        candidate["optimizer_state_dict"]["state"][0]["exp_avg_sq"][0] = -1.
    elif mutation == "counter":
        candidate["optimizer_state_dict"]["state"][0]["step"] += 1
    elif mutation == "dtype":
        candidate["model_state_dict"]["weight"] = candidate["model_state_dict"]["weight"].double()
    elif mutation == "rng":
        candidate["rng_state"]["torch_cuda"][0] = torch.zeros(8, dtype=torch.float32)
    elif mutation == "mode":
        candidate["module_training"][""] = False
    else:
        candidate["optimizer_state_dict"]["param_groups"][0]["lr"] = .0003
    with pytest.raises(ValueError):
        summary.validate_full_state(candidate, source, checkpoint, step=9600,
            learning_rate=.0001, counters=candidate["counters"], scaler=candidate["scaler_state_dict"])


def test_evaluation_baseline_requires_raw_array_equality(tmp_path):
    import shutil
    folder, _, _, _, _ = snapshot_fixture(tmp_path, 6400)
    candidate = tmp_path / "candidate"
    shutil.copytree(folder, candidate)
    assert summary.compare_saved_evaluation(folder, candidate)["posterior_arrays_compared"] == 6
    with np.load(candidate / "posteriors.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    key = next(iter(arrays))
    arrays[key][0, 0] += .001
    np.savez_compressed(candidate / "posteriors.npz", **arrays)
    with pytest.raises(ValueError):
        summary.compare_saved_evaluation(folder, candidate)


def gates_fixture(tmp_path):
    import shutil
    prior, run = tmp_path / "prior", tmp_path / "run"
    prior.mkdir()
    run.mkdir()
    original, _, _, _, _ = snapshot_fixture(prior, 6400)
    source, _ = initial_pair_fixture()
    initial = {branch: copy.deepcopy(source) for branch in summary.BRANCHES}
    for branch, rate in summary.BRANCHES.items():
        initial[branch]["optimizer_state_dict"]["param_groups"][0]["lr"] = rate
    plan = {"prior_run": str(prior), "prior_state_sha256": "s" * 64}
    completed = {"plan_sha256": "p" * 64}
    branch_completed, state_specs, snapshot_specs, gate_digests = {}, {}, {}, {}
    source_runtime = {key: source[key] for key in summary.RUNTIME_FIELDS}
    for branch, rate in summary.BRANCHES.items():
        folder = run / branch
        folder.mkdir()
        shutil.copytree(original, folder / "step_6400")
        state_specs[branch] = {"step": 6400, "file": "training_state_6400.pt", "sha256": branch[0] * 64}
        snapshot_specs[branch] = {"step": 6400, "folder": "step_6400", "checkpoint": "step_6400.pt"}
        runtime = {key: initial[branch][key] for key in summary.RUNTIME_FIELDS}
        report = summary.compare_nested(runtime, runtime)
        gate = {"status": "passed", "source_runtime_exact": summary.compare_nested(source_runtime, source_runtime),
            "lr_only_runtime_exact": report, "learning_rate": rate, "before_lrs": [.0003],
            "evaluation_preserved_runtime": report,
            "initial_outputs": summary.compare_saved_evaluation(original, folder / "step_6400"),
            "source_training_state_sha256": plan["prior_state_sha256"],
            "initial_training_state_sha256": state_specs[branch]["sha256"]}
        fixtures.write_json(folder / "restore_gate.json", gate)
        gate_digests[branch] = summary.sha256_file(folder / "restore_gate.json")
        branch_completed[branch] = {"restore_gate_sha256": gate_digests[branch],
            "resume_states": [state_specs[branch]], "snapshots": [snapshot_specs[branch]]}
    expected = copy.deepcopy({key: initial["A_keep"][key] for key in summary.RUNTIME_FIELDS})
    expected["optimizer_state_dict"]["param_groups"][0]["lr"] = .0001
    preflight = {"status": "passed", "plan_sha256": completed["plan_sha256"],
        "checked_before_either_branch_training": True,
        "cross_branch_runtime_lr_only_exact": summary.compare_nested(expected,
            {key: initial["B_reduce"][key] for key in summary.RUNTIME_FIELDS}),
        "restore_gate_sha256": gate_digests, "initial_states": state_specs, "initial_snapshots": snapshot_specs}
    fixtures.write_json(run / "preflight_gate.json", preflight)
    completed["preflight_gate_sha256"] = summary.sha256_file(run / "preflight_gate.json")
    for branch, rate in summary.BRANCHES.items():
        runtime = {key: initial[branch][key] for key in summary.RUNTIME_FIELDS}
        report = summary.compare_nested(runtime, runtime)
        gate = {"status": "passed", "source_runtime_exact": report, "lr_only_runtime_exact": report,
            "learning_rate": rate, "before_lrs": [rate],
            "initial_training_state_sha256": state_specs[branch]["sha256"],
            "preflight_gate_sha256": completed["preflight_gate_sha256"], "next_attempt": 6401}
        fixtures.write_json(run / branch / "start_gate.json", gate)
        branch_completed[branch]["start_gate_sha256"] = summary.sha256_file(run / branch / "start_gate.json")
    return run, plan, completed, source, initial, branch_completed


def test_preflight_and_start_evidence_crosschecks_saved_states_and_outputs(tmp_path):
    result = summary.validate_gates(*gates_fixture(tmp_path))
    assert result["passed"] and result["cross_branch_runtime_exact_except_learning_rate"]
    assert set(result["branches"]) == {"A_keep", "B_reduce"}


@pytest.mark.parametrize("mutation", ["baseline", "initial_moment", "restore_flag", "start_hash", "preflight_flag"])
def test_gates_reject_rehashed_or_cross_branch_tampering(tmp_path, mutation):
    run, plan, completed, source, states, branches = gates_fixture(tmp_path)
    if mutation == "baseline":
        rows = summary.read_rows(run / "B_reduce/step_6400/rows.jsonl")
        rows[0]["reference"] = "changed"
        fixtures.write_rows(run / "B_reduce/step_6400/rows.jsonl", rows)
    elif mutation == "initial_moment":
        states["B_reduce"]["optimizer_state_dict"]["state"][0]["exp_avg"][0] += .01
    elif mutation == "restore_flag":
        path = run / "A_keep/restore_gate.json"
        value = summary.read_json(path)
        value["source_runtime_exact"]["exact"] = False
        fixtures.write_json(path, value)
        branches["A_keep"]["restore_gate_sha256"] = summary.sha256_file(path)
    elif mutation == "start_hash":
        path = run / "B_reduce/start_gate.json"
        value = summary.read_json(path)
        value["initial_training_state_sha256"] = "x" * 64
        fixtures.write_json(path, value)
        branches["B_reduce"]["start_gate_sha256"] = summary.sha256_file(path)
    else:
        path = run / "preflight_gate.json"
        value = summary.read_json(path)
        value["checked_before_either_branch_training"] = False
        fixtures.write_json(path, value)
        completed["preflight_gate_sha256"] = summary.sha256_file(path)
        # Also rebind start files so semantic verification, not an old hash, fails.
        for branch in summary.BRANCHES:
            start = run / branch / "start_gate.json"
            gate = summary.read_json(start)
            gate["preflight_gate_sha256"] = completed["preflight_gate_sha256"]
            fixtures.write_json(start, gate)
            branches[branch]["start_gate_sha256"] = summary.sha256_file(start)
    with pytest.raises(ValueError):
        summary.validate_gates(run, plan, completed, source, states, branches)
