"""Exact stage13 resume/prefix checks on synthetic arrays only, never project data."""
from __future__ import annotations

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
from learning_budget_state import (  # noqa: E402
    SCHEMA, capture_rng_state, compare_nested, compare_schedule_prefix,
    compare_state_dicts, compare_telemetry, load_training_state,
    restore_rng_state, save_training_state,
)


def require_exact(left, right):
    result = compare_nested(left, right)
    assert result["exact"], result


def contract(attempt=1):
    return {"attempt": attempt,
            "schedule": {"sample_ids": ["fake_a", "fake_b"], "batches_sha256": "a"*64,
                         "attempt_updates": 2},
            "provenance": {"mode": "smoke_only", "train_only": True, "sample_ids": ["fake_a", "fake_b"]},
            "source_hashes": {"synthetic.py": "b"*64},
            "settings": {"lr": .0003, "batch_size": 2, "seed": 712},
            "counters": {"optimizer_steps": attempt, "skipped_optimizer_steps": 0}}


class TinyDropout(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(torch.nn.Linear(4, 6), torch.nn.Dropout(.4),
                                          torch.nn.Tanh(), torch.nn.Linear(6, 2))

    def forward(self, value):
        return self.layers(value)


def objects():
    model = TinyDropout().train()
    return (model, torch.optim.Adam(model.parameters(), lr=.0003),
            torch.amp.GradScaler("cpu", init_scale=128, growth_interval=2))


def tiny_step(model, optimizer, scaler, overflow=False):
    optimizer.zero_grad(set_to_none=True)
    # All three RNG families affect the update, so restoring only torch is insufficient.
    x = torch.rand(3, 4) + random.random() + float(np.random.random())
    loss = model(x).square().mean()
    hook = next(model.parameters()).register_hook(lambda g: torch.full_like(g, float("inf"))) if overflow else None
    try:
        scaler.scale(loss).backward()
    finally:
        if hook is not None:
            hook.remove()
    scaler.unscale_(optimizer)
    if not overflow:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    scaler.step(optimizer)
    scaler.update()
    return loss.item()


def object_state(model, optimizer, scaler):
    return {"model": copy.deepcopy(model.state_dict()), "optimizer": copy.deepcopy(optimizer.state_dict()),
            "scaler": copy.deepcopy(scaler.state_dict()), "rng": capture_rng_state(),
            "modes": {name: module.training for name, module in model.named_modules()}}


@pytest.mark.parametrize("overflow_first", [False, True])
def test_cpu_resume_matches_uninterrupted_parameters_adam_scaler_and_all_rng(tmp_path, overflow_first):
    random.seed(712); np.random.seed(713); torch.manual_seed(714)
    model, optimizer, scaler = objects()
    tiny_step(model, optimizer, scaler, overflow=overflow_first)
    before_save = object_state(model, optimizer, scaler)
    path = tmp_path / "training_state.pt"
    metadata = save_training_state(path, model=model, optimizer=optimizer, scaler=scaler, **contract())
    assert metadata["schema"] == SCHEMA and metadata["attempt"] == 1
    require_exact(object_state(model, optimizer, scaler), before_save)
    # Restricted loading must work without any allowlisted NumPy/Python classes.
    stored = torch.load(path, weights_only=True, map_location="cpu")
    assert stored["rng_state"]["numpy"]["keys"].dtype == torch.int64
    assert stored["inference_only"] is False and stored["deployment"] is False
    expected_loss = tiny_step(model, optimizer, scaler)
    expected = object_state(model, optimizer, scaler)

    resumed, resumed_optimizer, resumed_scaler = objects()
    random.seed(123); np.random.seed(456); torch.manual_seed(789)
    loaded = load_training_state(path, model=resumed, optimizer=resumed_optimizer,
                                 scaler=resumed_scaler, expected_contract=contract())
    assert loaded["attempt"] == 1
    require_exact(object_state(resumed, resumed_optimizer, resumed_scaler), before_save)
    assert tiny_step(resumed, resumed_optimizer, resumed_scaler) == expected_loss
    require_exact(object_state(resumed, resumed_optimizer, resumed_scaler), expected)
    assert compare_state_dicts(model.state_dict(), resumed.state_dict())["exact"]
    if not overflow_first:
        assert resumed_scaler.get_scale() == 256  # growth tracker must also resume.


def test_state_save_refuses_overwrite_and_preserves_mixed_module_modes(tmp_path):
    model, optimizer, scaler = objects()
    model.layers[1].eval()
    before = object_state(model, optimizer, scaler)
    path = tmp_path / "state.pt"
    save_training_state(path, model=model, optimizer=optimizer, scaler=scaler, **contract(attempt=0))
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        save_training_state(path, model=model, optimizer=optimizer, scaler=scaler, **contract(attempt=0))
    assert path.read_bytes() == original
    require_exact(object_state(model, optimizer, scaler), before)
    resumed, resumed_optimizer, resumed_scaler = objects()
    load_training_state(path, model=resumed, optimizer=resumed_optimizer, scaler=resumed_scaler)
    require_exact(object_state(resumed, resumed_optimizer, resumed_scaler), before)
    assert resumed.training and not resumed.layers[1].training


def test_contract_mismatch_fails_before_model_optimizer_or_rng_mutation(tmp_path):
    model, optimizer, scaler = objects()
    tiny_step(model, optimizer, scaler)
    path = tmp_path / "state.pt"
    save_training_state(path, model=model, optimizer=optimizer, scaler=scaler, **contract())
    resumed, resumed_optimizer, resumed_scaler = objects()
    before = object_state(resumed, resumed_optimizer, resumed_scaler)
    wrong = contract(); wrong["settings"]["lr"] = .001
    with pytest.raises(ValueError, match="contract mismatch: settings"):
        load_training_state(path, model=resumed, optimizer=resumed_optimizer,
                            scaler=resumed_scaler, expected_contract=wrong)
    require_exact(object_state(resumed, resumed_optimizer, resumed_scaler), before)


def test_swapped_optimizer_order_is_rejected_even_when_shapes_match(tmp_path):
    model = torch.nn.Linear(2, 2, bias=False)
    model.second = torch.nn.Linear(2, 2, bias=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=.0003)
    scaler = torch.amp.GradScaler("cpu")
    path = tmp_path / "state.pt"
    save_training_state(path, model=model, optimizer=optimizer, scaler=scaler, **contract(attempt=0))
    swapped = torch.optim.Adam(list(model.parameters())[::-1], lr=.0003)
    with pytest.raises(ValueError, match="optimizer_parameter_names"):
        load_training_state(path, model=model, optimizer=swapped, scaler=scaler)


def test_capture_restore_rng_preserves_cached_numpy_gaussian_and_cpu_cuda_states():
    random.seed(42); np.random.seed(43); torch.manual_seed(44)
    np.random.normal()  # Leaves the second Box-Muller value cached in RandomState.
    before = capture_rng_state()
    assert before["numpy"]["has_gauss"] == 1
    expected = (random.random(), np.random.normal(), torch.rand(8))
    expected_cuda = [torch.rand(4, device=f"cuda:{i}") for i in range(torch.cuda.device_count())]
    restore_rng_state(before)
    assert random.random() == expected[0] and np.random.normal() == expected[1]
    assert torch.equal(torch.rand(8), expected[2])
    assert all(torch.equal(torch.rand(4, device=f"cuda:{i}"), value) for i, value in enumerate(expected_cuda))


def test_exact_comparisons_do_not_hide_small_changes_nonfinite_or_dtypes():
    baseline = {"weight": torch.tensor([1., 2.], dtype=torch.float32), "counter": torch.tensor(3)}
    assert compare_state_dicts(baseline, copy.deepcopy(baseline))["exact"]
    changed = copy.deepcopy(baseline)
    changed["weight"][0] = torch.nextafter(changed["weight"][0], torch.tensor(float("inf")))
    report = compare_state_dicts(baseline, changed)
    assert not report["exact"] and report["matching_tensor_count"] == 1
    assert report["max_abs_numeric_difference"] == 2**-23
    wrong_dtype = {key: value.double() for key, value in baseline.items()}
    assert not compare_state_dicts(baseline, wrong_dtype)["exact"]
    for value in (float("nan"), float("inf"), -float("inf")):
        result = compare_state_dicts({"x": torch.tensor(value)}, {"x": torch.tensor(value)})
        assert not result["exact"] and result["nonfinite_difference_count"] == 1
        json.dumps(result, allow_nan=False)
        result = compare_nested(value, value)
        assert not result["exact"] and result["nonfinite_difference_count"] == 1
        json.dumps(result, allow_nan=False)


def telemetry():
    return [{"attempt": i, "loss": 1./i, "scale_before": 1024., "skipped": False,
             "preclip": {"total": {"l2_norm": 3.}}, "postclip": {"total": {"l2_norm": 1.}},
             "parameter_update": {"total": {"absolute_l2": .004}}, "seconds": float(i)}
            for i in range(1, 1601)]


def test_all_1600_telemetry_attempts_are_exact_checked_except_seconds():
    expected = telemetry()
    actual = copy.deepcopy(expected)
    for row in actual:
        row["seconds"] *= 123
    assert compare_telemetry(expected, actual)["exact"]
    actual[-1]["parameter_update"]["total"]["absolute_l2"] += 1e-10
    report = compare_telemetry(expected, actual)
    assert not report["exact"] and report["difference_count"] == 1
    assert report["differences"][0]["path"] == "root[1599].parameter_update.total.absolute_l2"
    assert report["max_abs_numeric_difference"] > 0
    assert not compare_telemetry(expected, actual[:-1])["exact"]
    assert not compare_nested({"skipped": False}, {"skipped": 0})["exact"]


def test_telemetry_details_are_bounded_but_counts_and_maximum_cover_all_rows():
    expected = telemetry()
    actual = copy.deepcopy(expected)
    for i, row in enumerate(actual):
        row["loss"] += i+1
    report = compare_telemetry(expected, actual)
    assert report["difference_count"] == 1600 and len(report["differences"]) == 100
    assert report["max_abs_numeric_difference"] == 1600 and report["details_truncated"]
    json.dumps(report, allow_nan=False)


def test_schedule_prefix_checks_order_and_completeness_without_comparing_future_batches():
    old = [{"attempt": i, "epoch": 1, "ids": ["a", "b"], "indices": [0, 1]} for i in range(1, 4)]
    extended = copy.deepcopy(old) + [{"attempt": 4, "epoch": 2, "ids": ["b", "a"], "indices": [1, 0]}]
    assert compare_schedule_prefix({"batches": old}, {"batches": extended}, 3)["exact"]
    assert not compare_schedule_prefix(old, old, 4)["exact"]
    extended[1]["ids"].reverse()
    assert not compare_schedule_prefix(old, extended, 3)["exact"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA compact resume equivalence requires GPU")
def test_cuda_compact_amp_resume_matches_uninterrupted_two_attempts_exactly(tmp_path):
    import diagnose_learning_transition as runner
    from models.baseline import LipNetBaseline
    from transition_diagnostics import parameter_groups

    torch.set_num_threads(2)
    runner.seed_training(931)
    model = LipNetBaseline().to("cuda").train()
    optimizer = torch.optim.Adam(model.parameters(), lr=.0003)
    scaler = torch.amp.GradScaler("cuda", init_scale=1024, growth_interval=2)
    synthetic_rng = np.random.default_rng(932)
    batch = runner.collate([{"frames": synthetic_rng.integers(0, 256, size=(length, 96, 96, 3), dtype=np.uint8),
                             "label": np.asarray(labels, dtype=np.int64)}
                            for length, labels in ((10, [1, 20, 44]), (12, [3, 24, 48]))])
    groups = parameter_groups(model)
    first = runner.optimizer_attempt(model, optimizer, scaler, batch, groups, torch.device("cuda"))
    assert not first["skipped"]
    before_save = object_state(model, optimizer, scaler)
    path = tmp_path / "cuda_training_state.pt"
    save_training_state(path, model=model, optimizer=optimizer, scaler=scaler, **contract())
    require_exact(object_state(model, optimizer, scaler), before_save)
    expected_second = runner.optimizer_attempt(model, optimizer, scaler, batch, groups, torch.device("cuda"))
    expected = object_state(model, optimizer, scaler)

    resumed = LipNetBaseline().to("cuda").eval()
    resumed_optimizer = torch.optim.Adam(resumed.parameters(), lr=.0003)
    resumed_scaler = torch.amp.GradScaler("cuda", init_scale=7)  # saved scaler config is authoritative.
    load_training_state(path, model=resumed, optimizer=resumed_optimizer,
                        scaler=resumed_scaler, expected_contract=contract())
    require_exact(object_state(resumed, resumed_optimizer, resumed_scaler), before_save)
    actual_second = runner.optimizer_attempt(resumed, resumed_optimizer, resumed_scaler, batch,
                                             parameter_groups(resumed), torch.device("cuda"))
    assert compare_telemetry([expected_second], [actual_second])["exact"]
    require_exact(object_state(resumed, resumed_optimizer, resumed_scaler), expected)
    # Adam moment tensors follow parameters to CUDA; ordinary scalar step stays on CPU.
    for state in resumed_optimizer.state.values():
        assert state["exp_avg"].device.type == state["exp_avg_sq"].device.type == "cuda"
        assert state["step"].device.type == "cpu"
    assert resumed_scaler.get_scale() == 2048
