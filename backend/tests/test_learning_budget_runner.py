"""Stage13 budget/evaluation contracts, with no real clips or training run."""
from __future__ import annotations

import json
from pathlib import Path
import random
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "backend")]
import diagnose_learning_budget as runner
import diagnose_learning_transition as prior
from transition_diagnostics import make_training_schedule


def fixture():
    """Two tiny fake clips with valid tokens, normalized posteriors and scopes."""
    ids = ["old", "additional"]
    source = {key: {"id": key, "speaker_id": "fake", "text": "간", "frames": 12} for key in ids}
    cache = {key: {"frames": np.arange(36, dtype=np.uint8).reshape(12, 1, 1, 3),
                   "label": np.asarray([1, 20, 44], dtype=np.int64)} for key in ids}
    logits = np.full((12, 69), -12.)
    logits[np.arange(12), [0, 1, 0, 20, 0, 44, 0, 0, 0, 0, 0, 0]] = 4.
    logp = (logits-np.log(np.exp(logits).sum(axis=1, keepdims=True))).astype(np.float32)
    return {"cohorts": {"short16": ids[:1], "expanded64": ids}}, cache, source, logp


def test_fixed_budget_has_identical_1600_prefix_and_200_exposures():
    """Epoch continuation uses one local RNG stream, not reseeding at the boundary."""
    ids = [f"sample_{i}" for i in range(64)]
    short = make_training_schedule(ids)
    long = make_training_schedule(ids, attempt_updates=6400)
    assert long["batches"][:1600] == short["batches"]
    assert set(long["exposures"].values()) == {200}
    assert long["batches"][1600]["attempt"] == 1601
    assert long["batches"][1600]["epoch"] == 51


@pytest.mark.parametrize("step,controls,beam", [(2400, False, False), (3200, True, True),
                                               (4800, False, False), (6400, True, True)])
def test_future_evaluation_schedule_and_rng_mode_preservation(monkeypatch, tmp_path, step, controls, beam):
    """Only declared future steps decode beam; every evaluation restores training RNG."""
    plan, cache, source, lp = fixture()
    model = torch.nn.Linear(1, 1).train()
    calls = []
    def forward(m, frames, capture):
        assert m is model and not m.training and not torch.is_grad_enabled()
        random.random(); np.random.rand(); torch.rand(1)
        return lp.copy(), {}
    def decode(values, beam_width):
        calls.append(beam_width)
        return "간"
    monkeypatch.setattr(prior, "forward", forward)
    monkeypatch.setattr(prior, "ctc_beam_search_decode", decode)
    before = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    folder = tmp_path / str(step)
    summary = runner.evaluate_snapshot(model, plan, cache, source, step, folder, "fake")
    rows = [json.loads(line) for line in (folder / "rows.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == (6 if controls else 2)
    assert len(calls) == (len(rows) if beam else 0)
    assert summary["beam_performed"] is beam
    assert all((r["beam"] is not None) is beam for r in rows)
    assert all(("beam_text" in r["ctc"]) is beam for r in rows)
    assert model.training and random.getstate() == before[0]
    now = np.random.get_state()
    assert now[0] == before[1][0] and now[2:] == before[1][2:]
    np.testing.assert_array_equal(now[1], before[1][1])
    assert torch.equal(torch.get_rng_state(), before[2])


def test_future_evaluation_math_exactly_matches_original_end_point(monkeypatch, tmp_path):
    """Same inputs yield identical rows/posteriors at old1600 and future6400."""
    plan, cache, source, lp = fixture()
    model = torch.nn.Linear(1, 1).train()
    monkeypatch.setattr(prior, "forward", lambda *a, **k: (lp.copy(), {}))
    monkeypatch.setattr(prior, "ctc_beam_search_decode", lambda *a, **k: "간")
    prior.evaluate_snapshot(model, plan, cache, source, 1600, tmp_path / "old", "fake")
    runner.evaluate_snapshot(model, plan, cache, source, 6400, tmp_path / "new", "fake")
    old = [json.loads(x) for x in (tmp_path / "old/rows.jsonl").read_text(encoding="utf-8").splitlines()]
    new = [json.loads(x) for x in (tmp_path / "new/rows.jsonl").read_text(encoding="utf-8").splitlines()]
    for a, b in zip(old, new):
        b["step"] = 1600
        assert a == b
    with np.load(tmp_path / "old/posteriors.npz") as a, np.load(tmp_path / "new/posteriors.npz") as b:
        assert a.files == b.files
        for key in a.files:
            np.testing.assert_array_equal(a[key], b[key])


def test_prefix_gate_rejects_missing_checks_before_io(tmp_path):
    """An empty all() must never allow an unverified extension."""
    with pytest.raises(ValueError, match="inventory incomplete"):
        runner.finish_prefix_gate({}, tmp_path, [], [])
    with pytest.raises(ValueError, match="all1600"):
        runner.finish_prefix_gate({}, tmp_path, [{"step": s} for s in prior.STEPS], [])


@pytest.mark.parametrize("kind", ["model", "rows", "posterior"])
def test_every_gate_component_must_be_exact(kind):
    """Matching text cannot hide a changed model or posterior."""
    result = {"model_state": {"exact": True}, "prediction_rows_exact": True, "posterior_arrays_exact": True}
    if kind == "model": result["model_state"]["exact"] = False
    elif kind == "rows": result["prediction_rows_exact"] = False
    else: result["posterior_arrays_exact"] = False
    assert runner.snapshot_exact(result) is False
