"""Stage16 safety and experiment contracts using synthetic fixtures only."""
from __future__ import annotations

from collections import Counter
import copy
import json
from pathlib import Path
import random
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "backend")]
import diagnose_training_diversity as runner
from learning_budget_state import capture_rng_state, compare_nested, save_training_state
from models.tokenizer import encode_text
from transition_diagnostics import make_training_schedule


def exact(expected, actual):
    result = compare_nested(expected, actual)
    assert result["exact"], result


def runtime(model, optimizer, scaler):
    return copy.deepcopy(runner.lr.runtime_state(model, optimizer, scaler))


def synthetic_contract():
    return {"attempt": 1, "schedule": {"sample_ids": ["a", "b"], "attempt_updates": 2,
                "batches_sha256": "a" * 64},
            "source_hashes": {"synthetic.py": "b" * 64},
            "provenance": {"mode": "smoke_only", "purpose": "synthetic_test"},
            "settings": {"learning_rate": .0001, "batch_size": 2},
            "counters": {"optimizer_steps": 1, "skipped_optimizer_steps": 0}}


def trained_objects():
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Dropout(.1), torch.nn.Linear(4, 2)).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=.0001)
    scaler = torch.amp.GradScaler("cpu", init_scale=128)
    scaler.scale(model(torch.arange(9, dtype=torch.float32).reshape(3, 3)).square().mean()).backward()
    scaler.step(optimizer)
    scaler.update()
    return model, optimizer, scaler


def fixture_plan():
    cohorts = {"original64": ["old_a", "old_b"], "additional192": ["added"],
               "train60": ["held_train"], "val20": ["held_val"]}
    records = {}
    cache = {}
    for index, sample_id in enumerate(i for ids in cohorts.values() for i in ids):
        records[sample_id] = {"id": sample_id, "split": "val" if sample_id == "held_val" else "train",
            "recording_id": f"recording_{index}", "speaker_id": "new" if sample_id == "held_val" else "known",
            "text": "간" if sample_id not in ("held_train", "held_val") else ("도" if sample_id == "held_train" else "말"),
            "frames": 8}
        cache[sample_id] = {"frames": np.arange(24, dtype=np.uint8).reshape(8, 1, 1, 3),
                            "label": np.asarray(encode_text(records[sample_id]["text"]), dtype=np.int64)}
    plan = {"cohorts": cohorts, "branches": {
        "A_keep64": {"train_ids": cohorts["original64"]},
        "B_expand256": {"train_ids": cohorts["original64"] + cohorts["additional192"]}},
        "selection": {"excluded_recording_ids": [records["held_train"]["recording_id"]]},
        "manifest_sha256": "c"*64, "prior_state_sha256": "d"*64,
        "settings": runner.settings(128), "source_hashes": {"synthetic.py": "e"*64},
        "initial_counters": {"optimizer_steps": 9599, "skipped_optimizer_steps": 1}}
    return plan, records, cache


def test_identical_full_runtime_restored_without_reinitializing_adam_or_scaler(tmp_path):
    model, optimizer, scaler = trained_objects()
    before = runtime(model, optimizer, scaler)
    path = tmp_path / "state.pt"
    save_training_state(path, model=model, optimizer=optimizer, scaler=scaler, **synthetic_contract())
    restored = copy.deepcopy(model)
    fresh_optimizer = torch.optim.Adam(restored.parameters(), lr=.7)
    fresh_scaler = torch.amp.GradScaler("cpu", init_scale=7)
    random.random(); np.random.rand(); torch.rand(4)
    audit = runner.restore_exact(path, restored, fresh_optimizer, fresh_scaler, synthetic_contract())
    assert audit["exact"]
    exact(before, runtime(restored, fresh_optimizer, fresh_scaler))
    assert fresh_optimizer.param_groups[0]["lr"] == .0001
    assert all(int(state["step"]) == 1 and torch.count_nonzero(state["exp_avg"]) > 0
               for state in fresh_optimizer.state.values())


def test_restore_rejects_wrong_contract_before_mutating_current_runtime(tmp_path):
    model, optimizer, scaler = trained_objects()
    path = tmp_path / "state.pt"
    save_training_state(path, model=model, optimizer=optimizer, scaler=scaler, **synthetic_contract())
    with torch.no_grad():
        next(model.parameters()).add_(2.)
    before = runtime(model, optimizer, scaler)
    wrong = synthetic_contract(); wrong["settings"]["learning_rate"] = .0003
    with pytest.raises(ValueError, match="contract mismatch: settings"):
        runner.restore_exact(path, model, optimizer, scaler, wrong)
    exact(before, runtime(model, optimizer, scaler))


@pytest.mark.parametrize("field", runner.RUNTIME_FIELDS)
def test_restore_gate_rejects_incomplete_restoration(tmp_path, monkeypatch, field):
    model, optimizer, scaler = trained_objects()
    path = tmp_path / "state.pt"
    save_training_state(path, model=model, optimizer=optimizer, scaler=scaler, **synthetic_contract())
    real_load = runner.load_training_state
    thread_count = torch.get_num_threads()
    saved_rng = capture_rng_state()
    def broken_load(*args, **kwargs):
        value = real_load(*args, **kwargs)
        if field == "model_state_dict":
            with torch.no_grad():
                next(model.parameters()).add_(1)
        elif field == "optimizer_state_dict":
            optimizer.param_groups[0]["eps"] = .01
        elif field == "scaler_state_dict":
            state = scaler.state_dict(); state["scale"] *= 2
            scaler.load_state_dict(state)
        elif field == "rng_state":
            torch.rand(1)
        elif field == "model_training":
            model.training = False
        elif field == "module_training":
            model[1].training = False
        elif field == "backend_state":
            torch.set_num_threads(1 if thread_count != 1 else 2)
        return value
    monkeypatch.setattr(runner, "load_training_state", broken_load)
    try:
        with pytest.raises(ValueError, match="Full runtime restore mismatch"):
            runner.restore_exact(path, model, optimizer, scaler, synthetic_contract())
    finally:
        from learning_budget_state import restore_rng_state
        torch.set_num_threads(thread_count)
        restore_rng_state(saved_rng)


def test_sampler_is_fresh_local_and_fixed_budget_means_100_vs_25_repetitions():
    ids64 = [f"sample_{i:03d}" for i in range(64)]
    ids256 = [f"sample_{i:03d}" for i in range(256)]
    before = capture_rng_state()
    for ids, repeats in ((ids64, 100), (ids256, 25)):
        schedule = make_training_schedule(ids, attempt_updates=3200, batch_size=2, seed=42)
        assert schedule["batches"][0]["attempt"] == 1
        assert schedule["batches"][0]["epoch"] == 1
        assert schedule["batches"][-1]["attempt"] == 3200
        assert Counter(i for row in schedule["batches"] for i in row["ids"]) == Counter({i: repeats for i in ids})
        assert sum(schedule["exposures"].values()) == 6400
        assert make_training_schedule(ids, attempt_updates=3200, batch_size=2, seed=42) == schedule
    exact(before, capture_rng_state())


@pytest.mark.parametrize("bad_ids", [["old_a"], ["old_a", "old_a"], ["old_a", "held_train"],
                                      ["old_a", "held_val"], ["old_a", "added"], ["old_a", "unknown"]])
def test_batch_guard_rejects_wrong_size_duplicates_unselected_or_heldout(bad_ids):
    plan, records, _ = fixture_plan()
    with pytest.raises(ValueError):
        runner.validate_training_ids(bad_ids, plan, "A_keep64", records)


@pytest.mark.parametrize("leak", ["split", "recording", "transcript"])
def test_training_guard_rejects_metadata_leakage_even_for_allowed_id(leak):
    plan, records, _ = fixture_plan()
    if leak == "split":
        records["old_b"]["split"] = "val"
    elif leak == "recording":
        records["old_b"]["recording_id"] = records["held_train"]["recording_id"]
    else:
        records["old_b"]["text"] = "  " + records["held_val"]["text"] + "  "
    with pytest.raises(ValueError, match="Training/evaluation leakage"):
        runner.validate_training_ids(["old_a", "old_b"], plan, "A_keep64", records)


def test_training_guard_allows_only_planned_added_data_in_expanded_branch():
    plan, records, _ = fixture_plan()
    runner.validate_training_ids(["old_a", "added"], plan, "B_expand256", records)
    runner.validate_training_ids(["old_a", "old_b"], plan, "A_keep64", records)


@pytest.mark.parametrize("source", [{"id": "old_a", "split": "test"},
                                   {"id": "unknown", "split": "train"}, {"id": "unknown", "split": "val"}])
def test_cache_guard_rejects_before_hash_or_file_open(monkeypatch, source):
    plan, _, _ = fixture_plan()
    def forbidden(*args):
        pytest.fail("cache verification/open must not occur")
    monkeypatch.setattr(runner, "verify_cache", forbidden)
    with pytest.raises(ValueError):
        runner.load_selected_cache(source, plan)


@pytest.mark.parametrize("sample_id", ["old_a", "added", "held_train", "held_val"])
def test_selected_cache_access_accepts_only_plan_members(monkeypatch, sample_id):
    plan, records, _ = fixture_plan()
    seen = []
    def verify(source):
        seen.append(source["id"])
        return {"synthetic": True}
    monkeypatch.setattr(runner, "verify_cache", verify)
    assert runner.load_selected_cache(records[sample_id], plan) == {"synthetic": True}
    assert seen == [sample_id]


@pytest.mark.parametrize("branch", runner.BRANCHES)
def test_step_zero_provenance_retains_actual64_lineage_separately_from_planned_data(branch):
    plan, _, _ = fixture_plan()
    zero = runner.provenance(plan, branch, 0)
    final = runner.provenance(plan, branch, 3200)
    assert zero["sample_ids"] == plan["cohorts"]["original64"]
    assert zero["planned_sample_ids"] == plan["branches"][branch]["train_ids"]
    assert zero["cumulative_attempt"] == 9600
    assert final["sample_ids"] == plan["branches"][branch]["train_ids"]
    assert final["cumulative_attempt"] == 12800
    assert zero["development_only"] and zero["deployment"] is False and zero["mode"] == "smoke_only"


@pytest.mark.parametrize("step", runner.STEPS)
def test_eval_uses_declared_grid_preserves_rng_and_mode_and_never_beams_controls(tmp_path, monkeypatch, step):
    plan, records, cache = fixture_plan()
    model = torch.nn.Linear(2, 2).train()
    logits = torch.full((8, 69), -5.)
    logits[0:2, 1] = 2.; logits[2:4, 20] = 2.; logits[4:6, 44] = 2.; logits[6:, 0] = 2.
    lp = logits.log_softmax(-1).numpy()
    seen, beams = [], []
    before = capture_rng_state()
    weights = copy.deepcopy(model.state_dict())
    def forward(current, frames, capture):
        assert not current.training and torch.is_inference_mode_enabled() and capture is False
        seen.append(frames.copy())
        random.random(); np.random.rand(); torch.rand(1)
        return lp.copy(), {}
    def beam(probs, beam_width):
        assert beam_width == 10
        beams.append(True)
        return runner.prior.ctc_greedy_decode(probs)
    monkeypatch.setattr(runner.prior, "forward", forward)
    monkeypatch.setattr(runner.prior, "ctc_beam_search_decode", beam)
    folder = tmp_path / f"step_{step:04d}"
    summary = runner.evaluate_snapshot(model, plan, cache, records, "A_keep64", step, folder, "f"*64)
    rows = [json.loads(line) for line in (folder / "rows.jsonl").read_text(encoding="utf-8").splitlines()]
    controls = step in (0, 3200)
    assert len(seen) == len(rows) == (9 if controls else 5)
    assert len(beams) == (5 if controls else 0)
    assert summary["step"] == step and summary["cumulative_attempt"] == 9600+step
    assert {(r["id"], r["condition"]) for r in rows} == {
        (sample_id, condition) for cohort, ids in plan["cohorts"].items() for sample_id in ids
        for condition in (["normal", "static_middle", "shuffle_42"]
            if controls and cohort in runner.HELDOUT else ["normal"])}
    for row in rows:
        assert (row["beam"] is not None) == (controls and row["condition"] == "normal")
        assert (row["comparison_to_normal"] is None) == (row["condition"] == "normal")
        assert row["frames"] == 8
    assert model.training
    exact(before, capture_rng_state())
    exact(weights, model.state_dict())


def test_saved_local_zero_keeps_existing_adam_step_and_separate_source_attempt(tmp_path):
    plan, _, _ = fixture_plan()
    model, optimizer, scaler = trained_objects()
    ids = plan["branches"]["A_keep64"]["train_ids"]
    plan["branches"]["A_keep64"]["schedule"] = make_training_schedule(ids, attempt_updates=3200, batch_size=2, seed=42)
    before = runtime(model, optimizer, scaler)
    spec = runner.save_state(tmp_path, model, optimizer, scaler, plan, "A_keep64", 0,
                             runner.zero_counters(ids), "f"*64)
    saved = torch.load(tmp_path / spec["file"], map_location="cpu", weights_only=True)
    assert saved["attempt"] == 0 and saved["provenance"]["source_attempt"] == 9600
    assert saved["counters"]["optimizer_steps"] == 0
    assert all(int(s["step"]) == 1 for s in saved["optimizer_state_dict"]["state"].values())
    exact(before, runtime(model, optimizer, scaler))
    exact({k: saved[k] for k in runner.RUNTIME_FIELDS}, before)


@pytest.mark.parametrize("branch,step", [("unknown", 0), ("A_keep64", 1),
                                        ("B_expand256", 9600), ("A_keep64", -1)])
def test_unplanned_evaluation_rejected_before_output_or_forward(tmp_path, monkeypatch, branch, step):
    plan, records, cache = fixture_plan()
    def forbidden(*args, **kwargs):
        pytest.fail("Unplanned evaluation cannot run a model")
    monkeypatch.setattr(runner.prior, "forward", forbidden)
    folder = tmp_path / "must_not_exist"
    with pytest.raises(ValueError):
        runner.evaluate_snapshot(torch.nn.Linear(2, 2), plan, cache, records, branch, step, folder, "f"*64)
    assert not folder.exists()


def test_failed_evaluation_restores_rng_and_mode(tmp_path, monkeypatch):
    plan, records, cache = fixture_plan()
    model = torch.nn.Linear(2, 2).train()
    before = capture_rng_state()
    def broken(*args, **kwargs):
        random.random(); np.random.rand(); torch.rand(1)
        raise ValueError("synthetic forward failure")
    monkeypatch.setattr(runner.prior, "forward", broken)
    with pytest.raises(ValueError, match="synthetic forward failure"):
        runner.evaluate_snapshot(model, plan, cache, records, "A_keep64", 0, tmp_path / "failed", "f"*64)
    exact(before, capture_rng_state())
    assert model.training


def write_outputs(folder, rows, arrays):
    folder.mkdir(parents=True)
    (folder / "rows.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    np.savez_compressed(folder / "posteriors.npz", **arrays)


def reproduction_fixture(tmp_path, monkeypatch):
    plan = {"cohorts": {"original64": [f"old_{i}" for i in range(64)],
            "train60": [f"held_{i}" for i in range(60)], "val20": [f"val_{i}" for i in range(20)]}}
    source = tmp_path / "source14"
    previous = tmp_path / "source15"
    monkeypatch.setattr(runner, "SOURCE", source)
    monkeypatch.setattr(runner, "PREVIOUS", previous)
    all_rows, all_arrays = [], {}
    for ids, conditions, folder in (
        (plan["cohorts"]["original64"], ["normal"], source / "B_reduce/step_9600"),
        (plan["cohorts"]["train60"] + plan["cohorts"]["val20"], ["normal", "static_middle", "shuffle_42"],
         previous / "B_reduce9600")):
        rows, arrays = [], {}
        for sample_id in ids:
            for condition in conditions:
                key = sample_id + "__" + condition
                row = {"id": sample_id, "condition": condition, "posterior_key": key,
                    "reference": "간", "frames": 8, "greedy": {"prediction": "간"},
                    "beam": {"prediction": "간"} if condition == "normal" else None,
                    "ctc": {"synthetic": True}, "transform": {"condition": condition},
                    "comparison_to_normal": None if condition == "normal" else {"same": True},
                    "ctc_loss_per_target_token": .4}
                rows.append(row)
                arrays[key] = np.full((8, 69), -4., dtype=np.float32)
        write_outputs(folder, rows, arrays)
        all_rows.extend(copy.deepcopy(rows)); all_arrays.update(copy.deepcopy(arrays))
    current = tmp_path / "current"
    return plan, current, all_rows, all_arrays


def test_initial_reproduction_checks_all_304_posteriors_and_comparable_fields(tmp_path, monkeypatch):
    plan, current, rows, arrays = reproduction_fixture(tmp_path, monkeypatch)
    # Different reduction implementations may incur a tiny floating-point loss difference.
    rows[-1]["ctc_loss_per_target_token"] += 1e-7
    write_outputs(current, rows, arrays)
    result = runner.compare_initial_outputs(current, plan)
    assert result["prediction_rows_compared"] == result["posterior_arrays_compared"] == 304
    assert result["posterior_arrays_exact"] and result["posterior_max_absolute_difference"] == 0


@pytest.mark.parametrize("corruption", ["prediction", "transform", "ctc_loss", "array_value", "array_dtype", "array_shape"])
def test_initial_reproduction_rejects_prediction_and_posterior_mismatch(tmp_path, monkeypatch, corruption):
    plan, current, rows, arrays = reproduction_fixture(tmp_path, monkeypatch)
    key = rows[-1]["posterior_key"]
    if corruption == "prediction":
        rows[-1]["greedy"]["prediction"] = "다른"
    elif corruption == "transform":
        rows[-1]["transform"]["condition"] = "normal"
    elif corruption == "ctc_loss":
        rows[-1]["ctc_loss_per_target_token"] += .01
    elif corruption == "array_value":
        arrays[key][0, 0] += .001
    elif corruption == "array_dtype":
        arrays[key] = arrays[key].astype(np.float64)
    else:
        arrays[key] = arrays[key][:-1]
    write_outputs(current, rows, arrays)
    with pytest.raises(ValueError, match="Initial"):
        runner.compare_initial_outputs(current, plan)


@pytest.mark.parametrize("mutation", [None, "prediction", "posterior", "order"])
def test_cross_branch_start_requires_496_identical_outputs_except_branch(tmp_path, mutation):
    rows = [{"branch": "A_keep64", "id": str(i), "posterior_key": str(i), "prediction": "간"}
            for i in range(496)]
    arrays = {str(i): np.full((2, 69), -.1, dtype=np.float32) for i in range(496)}
    second = [{**row, "branch": "B_expand256"} for row in rows]
    changed = copy.deepcopy(arrays)
    if mutation == "prediction":
        second[-1]["prediction"] = "다른"
    elif mutation == "posterior":
        changed["495"][0, 0] = -.11
    elif mutation == "order":
        second.reverse()
    first_dir, second_dir = tmp_path / "a", tmp_path / "b"
    write_outputs(first_dir, rows, arrays)
    write_outputs(second_dir, second, changed)
    if mutation is None:
        result = runner.compare_branch_outputs(first_dir, second_dir)
        assert result["prediction_rows_exact"] and result["posterior_arrays_compared"] == 496
    else:
        with pytest.raises(ValueError, match="Initial cross-branch"):
            runner.compare_branch_outputs(first_dir, second_dir)
