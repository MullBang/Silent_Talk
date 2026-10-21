"""Stage14 continuation/fork contracts using synthetic data only.

No cached training clips or diagnostic checkpoints are opened by these tests.
The compact CUDA test checks the actual AMP/CTC update implementation.
"""
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
from learning_budget_state import (capture_rng_state, compare_nested, compare_schedule_prefix,
                                  save_training_state)
from transition_diagnostics import make_training_schedule


def exact(expected, actual):
    result = compare_nested(expected, actual)
    assert result["exact"], result


def live_state(model, optimizer, scaler):
    return {"model": copy.deepcopy(model.state_dict()),
            "optimizer": copy.deepcopy(optimizer.state_dict()),
            "scaler": copy.deepcopy(scaler.state_dict()),
            "rng": capture_rng_state(),
            "modes": {name: module.training for name, module in model.named_modules()}}


def synthetic_contract(attempt=1):
    return {"attempt": attempt,
            "schedule": {"sample_ids": ["synthetic_a", "synthetic_b"],
                         "batches_sha256": "a" * 64, "attempt_updates": 2},
            "provenance": {"mode": "smoke_only", "purpose": "synthetic_test"},
            "source_hashes": {"synthetic.py": "b" * 64},
            "settings": {"learning_rate": .0003, "batch_size": 2},
            "counters": {"optimizer_steps": attempt, "skipped_optimizer_steps": 0}}


def test_full_schedule_continues_epoch201_and_each_branch_gets_100_new_exposures():
    ids = [f"sample_{index:02d}" for index in range(64)]
    old = make_training_schedule(ids, attempt_updates=6400)
    full = make_training_schedule(ids, attempt_updates=9600)
    assert compare_schedule_prefix(old, full, attempts=6400)["exact"]
    continued = full["batches"][6400:]
    assert len(continued) == 3200
    assert (continued[0]["attempt"], continued[0]["epoch"], continued[0]["batch_in_epoch"]) == (6401, 201, 1)
    assert (continued[-1]["attempt"], continued[-1]["epoch"], continued[-1]["batch_in_epoch"]) == (9600, 300, 32)
    assert Counter(key for row in continued for key in row["ids"]) == Counter({key: 100 for key in ids})
    assert [row["ids"] for row in continued] != [row["ids"] for row in full["batches"][:3200]]
    # Plan construction owns a local generator, so branch dropout RNG is unaffected.
    rng = capture_rng_state()
    assert make_training_schedule(ids, attempt_updates=9600) == full
    exact(capture_rng_state(), rng)


def _restore_and_set_rate(path, model, optimizer, scaler, learning_rate):
    """The actual runner must load full state first, then change only group LR."""
    import diagnose_learning_rate as runner
    gate = runner.restore_branch(path, model=model, optimizer=optimizer, scaler=scaler,
                                 expected_contract=synthetic_contract(), learning_rate=learning_rate)
    assert gate["status"] == "passed"
    assert gate["source_runtime_exact"]["exact"] and gate["lr_only_runtime_exact"]["exact"]
    assert gate["learning_rate"] == learning_rate and gate["before_lrs"] == [.0003]
    return gate


@pytest.mark.parametrize("learning_rate", [.0003, .0001])
def test_cpu_lr_fork_preserves_adam_scaler_rng_and_predictions(tmp_path, learning_rate):
    random.seed(813); np.random.seed(814); torch.manual_seed(815)
    model = torch.nn.Sequential(torch.nn.Linear(4, 6), torch.nn.Dropout(.2), torch.nn.Linear(6, 2)).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=.0003)
    scaler = torch.amp.GradScaler("cpu", init_scale=128)
    x = torch.randn(3, 4)
    scaler.scale(model(x).square().mean()).backward()
    scaler.step(optimizer); scaler.update()
    original = live_state(model, optimizer, scaler)
    path = tmp_path / "synthetic_state.pt"
    save_training_state(path, model=model, optimizer=optimizer, scaler=scaler, **synthetic_contract())

    model.eval()
    with torch.no_grad():
        predicted = model(x).clone()
    restored = copy.deepcopy(model).train()
    resumed_optimizer = torch.optim.Adam(restored.parameters(), lr=.99)
    resumed_scaler = torch.amp.GradScaler("cpu", init_scale=7)
    random.random(); np.random.rand(); torch.rand(10)
    _restore_and_set_rate(path, restored, resumed_optimizer, resumed_scaler, learning_rate)
    expected = copy.deepcopy(original)
    for group in expected["optimizer"]["param_groups"]:
        group["lr"] = learning_rate
    exact(expected, live_state(restored, resumed_optimizer, resumed_scaler))
    restored.eval()
    with torch.no_grad():
        assert torch.equal(predicted, restored(x))
    assert all(int(state["step"]) == 1 and torch.count_nonzero(state["exp_avg"]) > 0
               for state in resumed_optimizer.state.values())


def test_restore_changes_every_optimizer_group_lr_but_preserves_other_group_settings(tmp_path):
    import diagnose_learning_rate as runner
    model = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Linear(3, 1)).train()
    groups = [{"params": model[0].parameters(), "betas": (.8, .99)},
              {"params": model[1].parameters(), "betas": (.9, .999)}]
    optimizer = torch.optim.Adam(groups, lr=.0003)
    scaler = torch.amp.GradScaler("cpu", init_scale=128)
    scaler.scale(model(torch.ones(1, 2)).square().mean()).backward()
    scaler.step(optimizer); scaler.update()
    path = tmp_path / "two_groups.pt"
    save_training_state(path, model=model, optimizer=optimizer, scaler=scaler, **synthetic_contract())
    original = live_state(model, optimizer, scaler)
    for group in optimizer.param_groups:
        group["lr"] = .777
        group["betas"] = (.1, .2)
    gate = runner.restore_branch(path, model=model, optimizer=optimizer, scaler=scaler,
        expected_contract=synthetic_contract(), learning_rate=.0001)
    expected = copy.deepcopy(original)
    for group in expected["optimizer"]["param_groups"]:
        group["lr"] = .0001
    exact(expected, live_state(model, optimizer, scaler))
    assert gate["before_lrs"] == [.0003, .0003]
    assert gate["source_runtime_exact"]["exact"] and gate["lr_only_runtime_exact"]["exact"]


def test_wrong_source_contract_stops_before_state_or_rng_mutation(tmp_path):
    import diagnose_learning_rate as runner
    model = torch.nn.Linear(2, 2).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=.0003)
    scaler = torch.amp.GradScaler("cpu")
    path = tmp_path / "source.pt"
    save_training_state(path, model=model, optimizer=optimizer, scaler=scaler, **synthetic_contract())
    with torch.no_grad():
        model.weight.add_(1.)
    before = live_state(model, optimizer, scaler)
    wrong = synthetic_contract(); wrong["settings"]["learning_rate"] = .99
    with pytest.raises(ValueError, match="contract mismatch: settings"):
        runner.restore_branch(path, model=model, optimizer=optimizer, scaler=scaler,
            expected_contract=wrong, learning_rate=.0001)
    exact(before, live_state(model, optimizer, scaler))


@pytest.mark.parametrize("corrupted_field", ["model_state_dict", "optimizer_state_dict", "scaler_state_dict",
    "rng_state", "model_training", "module_training", "backend_state"])
def test_restore_gate_rejects_each_runtime_category_if_restore_is_incomplete(tmp_path, monkeypatch, corrupted_field):
    """A broken/partial restore cannot pass because only predictions happen to match."""
    import diagnose_learning_rate as runner
    model = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Dropout(.2), torch.nn.Linear(3, 1)).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=.0003)
    scaler = torch.amp.GradScaler("cpu", init_scale=128)
    path = tmp_path / "source.pt"
    save_training_state(path, model=model, optimizer=optimizer, scaler=scaler, **synthetic_contract())
    real_load = runner.load_training_state
    original_threads = torch.get_num_threads()
    original_rng = capture_rng_state()

    def incomplete_load(*args, **kwargs):
        loaded = real_load(*args, **kwargs)
        if corrupted_field == "model_state_dict":
            with torch.no_grad():
                next(model.parameters()).add_(1.)
        elif corrupted_field == "optimizer_state_dict":
            optimizer.param_groups[0]["betas"] = (.1, .2)
        elif corrupted_field == "scaler_state_dict":
            changed = scaler.state_dict(); changed["scale"] *= 2
            scaler.load_state_dict(changed)
        elif corrupted_field == "rng_state":
            torch.rand(1)
        elif corrupted_field == "model_training":
            model.training = False
        elif corrupted_field == "module_training":
            model[1].training = False
        elif corrupted_field == "backend_state":
            torch.set_num_threads(1 if original_threads != 1 else 2)
        return loaded

    monkeypatch.setattr(runner, "load_training_state", incomplete_load)
    try:
        with pytest.raises(ValueError, match="Source runtime did not restore exactly"):
            runner.restore_branch(path, model=model, optimizer=optimizer, scaler=scaler,
                expected_contract=synthetic_contract(), learning_rate=.0001)
        assert optimizer.param_groups[0]["lr"] == .0003
    finally:
        from learning_budget_state import restore_rng_state
        torch.set_num_threads(original_threads)
        restore_rng_state(original_rng)


@pytest.mark.parametrize("learning_rate", [0., -.0001, .001, float("nan"), float("inf")])
def test_unplanned_learning_rate_rejected_before_read_or_mutation(tmp_path, learning_rate):
    import diagnose_learning_rate as runner
    model = torch.nn.Linear(2, 2).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=.0003)
    scaler = torch.amp.GradScaler("cpu")
    before = live_state(model, optimizer, scaler)
    with pytest.raises(ValueError, match="Unplanned learning rate"):
        runner.restore_branch(tmp_path / "must_not_be_read.pt", model=model, optimizer=optimizer, scaler=scaler,
            expected_contract=synthetic_contract(), learning_rate=learning_rate)
    exact(before, live_state(model, optimizer, scaler))


@pytest.mark.parametrize("step,control,beam", [
    (6400, True, True), (6401, False, False), (6500, False, False), (6800, False, False),
    (7200, True, False), (8000, True, True), (8800, False, False), (9600, True, True)])
def test_evaluation_preserves_rng_mode_and_uses_only_declared_controls_and_beam(tmp_path, monkeypatch, step, control, beam):
    import diagnose_learning_rate as runner
    ids = ["fake_short", "fake_long"]
    records = {key: {"id": key, "speaker_id": "fake", "text": "간", "frames": 12} for key in ids}
    cache = {key: {"frames": np.arange(36, dtype=np.uint8).reshape(12, 1, 1, 3),
                   "label": np.asarray([1, 20, 44], dtype=np.int64)} for key in ids}
    plan = {"cohorts": {"short16": ids[:1], "expanded64": ids}}
    logits = np.full((12, 69), -12.)
    logits[np.arange(12), [0, 1, 0, 20, 0, 44, 0, 0, 0, 0, 0, 0]] = 4.
    posterior = (logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))).astype(np.float32)
    model = torch.nn.Linear(1, 1).train()
    calls = []

    def forward(actual, frames, capture):
        assert actual is model and not actual.training and not torch.is_grad_enabled()
        random.random(); np.random.rand(); torch.rand(1)
        return posterior.copy(), {}

    def beam_decode(values, beam_width):
        calls.append(beam_width)
        return "간"

    monkeypatch.setattr(runner.prior, "forward", forward)
    monkeypatch.setattr(runner.prior, "ctc_beam_search_decode", beam_decode)
    before_rng = capture_rng_state()
    before_model = copy.deepcopy(model.state_dict())
    folder = tmp_path / str(step)
    summary = runner.evaluate_snapshot(model, plan, cache, records, step, folder, "fake")
    rows = [json.loads(line) for line in (folder / "rows.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == (6 if control else 2)
    assert len(calls) == (len(rows) if beam else 0)
    assert set(calls) <= {10}
    assert summary["beam_performed"] is beam
    assert all((row["beam"] is not None) is beam for row in rows)
    assert all(("beam_text" in row["ctc"]) is beam for row in rows)
    assert model.training
    exact(before_rng, capture_rng_state())
    exact(before_model, model.state_dict())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="compact AMP branch equivalence requires CUDA")
def test_cuda_compact_fork_matches_uninterrupted_and_only_lr_changes_first_update(tmp_path):
    import diagnose_learning_transition as prior
    from models.baseline import LipNetBaseline
    from transition_diagnostics import parameter_groups

    torch.set_num_threads(2)
    prior.seed_training(819)
    model = LipNetBaseline().cuda().train()
    optimizer = torch.optim.Adam(model.parameters(), lr=.0003)
    scaler = torch.amp.GradScaler("cuda", init_scale=1024, growth_interval=2)
    generator = np.random.default_rng(820)
    samples = [{"frames": generator.integers(0, 256, size=(length, 96, 96, 3), dtype=np.uint8),
                "label": np.asarray(labels, dtype=np.int64)}
               for length, labels in ((10, [1, 20, 44]), (12, [3, 24, 48]))]
    batch = prior.collate(samples)
    first = prior.optimizer_attempt(model, optimizer, scaler, batch, parameter_groups(model), torch.device("cuda"))
    assert not first["skipped"]
    saved = live_state(model, optimizer, scaler)
    path = tmp_path / "synthetic_compact_state.pt"
    save_training_state(path, model=model, optimizer=optimizer, scaler=scaler, **synthetic_contract())
    with prior.preserve_rng_and_mode(model), torch.inference_mode():
        predicted, _ = prior.forward(model, samples[0]["frames"], capture=False)
    exact(saved, live_state(model, optimizer, scaler))
    uninterrupted_row = prior.optimizer_attempt(model, optimizer, scaler, batch, parameter_groups(model), torch.device("cuda"))
    uninterrupted = live_state(model, optimizer, scaler)

    branch_states, branch_rows = {}, {}
    for name, learning_rate in (("A", .0003), ("B", .0001)):
        restored = LipNetBaseline().cuda().eval()
        restored_optimizer = torch.optim.Adam(restored.parameters(), lr=.99)
        restored_scaler = torch.amp.GradScaler("cuda", init_scale=7)
        _restore_and_set_rate(path, restored, restored_optimizer, restored_scaler, learning_rate)
        expected = copy.deepcopy(saved)
        for group in expected["optimizer"]["param_groups"]:
            group["lr"] = learning_rate
        exact(expected, live_state(restored, restored_optimizer, restored_scaler))
        with prior.preserve_rng_and_mode(restored), torch.inference_mode():
            actual, _ = prior.forward(restored, samples[0]["frames"], capture=False)
        assert np.array_equal(predicted, actual)
        exact(expected, live_state(restored, restored_optimizer, restored_scaler))
        branch_rows[name] = prior.optimizer_attempt(restored, restored_optimizer, restored_scaler, batch,
                                                    parameter_groups(restored), torch.device("cuda"))
        branch_states[name] = live_state(restored, restored_optimizer, restored_scaler)
        assert not branch_rows[name]["skipped"]
        del restored, restored_optimizer, restored_scaler

    expected_row = {key: value for key, value in uninterrupted_row.items() if key != "seconds"}
    actual_row = {key: value for key, value in branch_rows["A"].items() if key != "seconds"}
    exact(expected_row, actual_row)
    exact(uninterrupted, branch_states["A"])
    for key in ("loss", "preclip", "postclip", "clip_coefficient", "clip_applied",
                "scale_before", "scale_after", "skipped"):
        exact(branch_rows["A"][key], branch_rows["B"][key])
    for key in ("scaler", "rng", "modes"):
        exact(branch_states["A"][key], branch_states["B"][key])
    exact(branch_states["A"]["optimizer"]["state"], branch_states["B"]["optimizer"]["state"])
    assert not compare_nested(branch_states["A"]["model"], branch_states["B"]["model"])["exact"]
    ratio = branch_rows["B"]["parameter_update"]["total"]["absolute_l2"] / branch_rows["A"]["parameter_update"]["total"]["absolute_l2"]
    assert ratio == pytest.approx(1 / 3, rel=2e-5)
    assert branch_states["A"]["scaler"]["scale"] == branch_states["B"]["scaler"]["scale"] == 2048
