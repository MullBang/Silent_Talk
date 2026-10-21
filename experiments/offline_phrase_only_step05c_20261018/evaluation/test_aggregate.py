"""Strict gate and cohort selection tests without models or visual arrays."""
import importlib.util
import json
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location("stage5c_aggregate_tests", Path(__file__).with_name("aggregate.py"))
agg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(agg)
ev = agg.ev


def make_summaries(tmp_path, fault=None):
    old = json.loads(ev.helper.PLAN_PATH.read_text(encoding="utf-8"))
    items = [r for r in old["items"] if r["split"] == "train" and r["kind"] == "phrase"]
    paths = []
    for seed in [42, 43]:
        for update in ev.CHECKPOINTS:
            identity = {"seed": seed, "completed_update": update}
            rows = []
            for item in items:
                for condition in ev.CONDITIONS[str(update)]:
                    correct = not (fault == "penultimate_wrong" and seed == 43 and update == 1260)
                    prediction = item["text"] if correct else ""
                    rows.append({"clip_id": item["clip_id"], "identity": identity, "reference_text": item["text"],
                                 "raw_text": prediction, "condition": condition, "split": "train", "kind": "phrase",
                                 "metrics": ev.contract.score_prediction(item["text"], prediction)})
            if fault == "wrong_clip" and seed == 43 and update == 1680:
                rows[-1]["clip_id"] = "unplanned_phrase"
            summary = {"identity": identity, "count": len(rows), "rows": rows, "metrics": ev.summarize_rows(rows),
                       "failures": 0, "status": "completed", "state_isolation_passed": True,
                       "forward_calls": len(rows), "attempted_forward_frames": sum(r["T"] for r in items) * len(ev.CONDITIONS[str(update)])}
            path = tmp_path / f"{seed}_{update}.json"
            ev.write_json(path, summary)
            paths.append(path)
    return paths


def test_all_eight_cohorts_and_144_rows_are_retained(tmp_path):
    result = agg.aggregate_summaries(make_summaries(tmp_path))
    assert result["evaluation_count"] == 144 and result["frames"] == 5772
    assert result["fit_gate"]["passed"] and len(result["diagnostics"]) == 12
    assert not result["dev_evaluated"] and not result["generalization_measured"]


def test_final_success_cannot_hide_penultimate_failure(tmp_path):
    result = agg.aggregate_summaries(make_summaries(tmp_path, "penultimate_wrong"))
    assert not result["fit_gate"]["passed"]
    assert [r["passed"] for r in result["fit_gate"]["per_seed"]] == [True, False]


def test_duplicate_checkpoint_or_unplanned_clip_fails(tmp_path):
    paths = make_summaries(tmp_path)
    with pytest.raises(ValueError, match="Duplicate or omitted"):
        agg.aggregate_summaries(paths[:-1] + paths[:1])


def test_unplanned_clip_fails(tmp_path):
    with pytest.raises(ValueError, match="Unexpected clip"):
        agg.aggregate_summaries(make_summaries(tmp_path, "wrong_clip"))
