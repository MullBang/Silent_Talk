"""CPU-only scope, accounting and resume guards; no neural inference."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location("stage5c_eval_tests", Path(__file__).with_name("evaluator.py"))
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)


def plan_fixture():
    old = json.loads(ev.helper.PLAN_PATH.read_text(encoding="utf-8"))
    return {"items": [r for r in old["items"] if r["split"] == "train" and r["kind"] == "phrase"],
            "checkpoints": ev.CHECKPOINTS, "conditions_by_update": copy.deepcopy(ev.CONDITIONS)}


@pytest.mark.parametrize("update,count", [(420, 12), (840, 12), (1260, 12), (1680, 36)])
def test_exact_checkpoint_scope(update, count):
    plan = plan_fixture()
    conditions = ev.validate_plan(plan, {"seed": 42, "completed_update": update}, verify_sources=False)
    assert len(ev.expected_pairs(plan, conditions)) == count
    assert sum(len(ev.expected_pairs(plan, c)) for c in ev.CONDITIONS.values()) * 2 == 144


@pytest.mark.parametrize("fault", ["dev", "sentence", "duplicate", "permutation", "early_control", "wrong_checkpoint", "bool_seed"])
def test_scope_widening_and_relabelled_frames_rejected(fault):
    plan = copy.deepcopy(plan_fixture())
    identity = {"seed": 42, "completed_update": 420}
    if fault == "dev":
        plan["items"][0]["split"] = "dev"
    elif fault == "sentence":
        plan["items"][0]["kind"] = "full_sentence"
    elif fault == "duplicate":
        plan["items"][-1] = plan["items"][0]
    elif fault == "permutation":
        plan["items"][0]["shuffled_frame_indices"].reverse()
    elif fault == "early_control":
        plan["conditions_by_update"]["420"].append("static")
    elif fault == "wrong_checkpoint":
        identity["completed_update"] = 419
    else:
        identity["seed"] = True
    with pytest.raises(ValueError):
        ev.validate_plan(plan, identity, verify_sources=False)


def test_failed_row_not_dropped_from_metric_denominator():
    rows = [{"clip_id": "a", "split": "train", "kind": "phrase", "condition": "normal", "metrics": ev.contract.score_prediction("가 나", "", failed=True)},
            {"clip_id": "b", "split": "train", "kind": "phrase", "condition": "normal", "metrics": ev.contract.score_prediction("가", "가")}]
    score = ev.summarize_rows(rows)["cohorts"]["train/phrase/normal"]
    assert score["CER"] == .75 and score["count"] == 2 and score["failure_count"] == 1
    with pytest.raises(ValueError, match="Duplicate"):
        ev.summarize_rows([rows[0], rows[0]])
    rows[0]["split"] = "dev"
    with pytest.raises(ValueError, match="Non-phrase"):
        ev.summarize_rows(rows)


def test_resume_rejects_incomplete_attempt_and_changed_identity(tmp_path, monkeypatch):
    plan = plan_fixture()
    identity = {"seed": 42, "completed_update": 420}
    with pytest.raises(ValueError, match="Incomplete existing"):
        ev.reuse_completed(tmp_path, plan, identity, ["normal"])
    path = tmp_path / "plan.json"
    ev.write_json(path, plan)
    monkeypatch.setattr(ev, "PLAN_PATH", path)
    ev.write_json(tmp_path / "evaluation_summary.json", {"identity": {**identity, "seed": 43}, "evaluation_plan_sha256": ev.file_sha256(path)})
    with pytest.raises(ValueError, match="identity/plan"):
        ev.reuse_completed(tmp_path, plan, identity, ["normal"])


def test_complete_resume_rechecks_saved_files_without_forward(tmp_path, monkeypatch):
    plan = plan_fixture()
    identity = {"seed": 42, "completed_update": 420}
    plan_path = tmp_path / "plan.json"
    ev.write_json(plan_path, plan)
    monkeypatch.setattr(ev, "PLAN_PATH", plan_path)
    rows = []
    for item in plan["items"]:
        posterior = tmp_path / (item["clip_id"] + ".npz")
        posterior.write_bytes(b"test-file-hash-only")
        rows.append({"clip_id": item["clip_id"], "split": "train", "kind": "phrase", "condition": "normal", "identity": identity,
                     "metrics": ev.contract.score_prediction(item["text"], item["text"]), "posterior": ev.ref(posterior)})
    log = tmp_path / "rows.jsonl"
    log.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    isolation = ev.write_json(tmp_path / "state_isolation.json", {"fatal": None, "model_unchanged": True,
        "gradients_unchanged": True, "rng_restored_exact": True, "module_training_flags_restored": True})
    summary = {"identity": identity, "evaluation_plan_sha256": ev.file_sha256(plan_path), "status": "completed", "failures": 0,
               "rows": rows, "metrics": ev.summarize_rows(rows), "state_isolation": isolation, "rows_jsonl": ev.ref(log),
               "count": 12, "forward_calls": 12, "attempted_forward_frames": sum(r["T"] for r in plan["items"]),
               "seconds": 1., "peak_allocated_cuda_bytes": 0, "peak_reserved_cuda_bytes": 0,
               "cuda_reserved_budget_passed": True, "state_isolation_passed": True}
    ev.write_json(tmp_path / "evaluation_summary.json", summary)
    result = ev.reuse_completed(tmp_path, plan, identity, ["normal"])
    assert result["reused"] and result["new_forward_calls"] == 0 and result["forward_calls"] == 12
    Path(rows[0]["posterior"]["path"]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="artifact changed"):
        ev.reuse_completed(tmp_path, plan, identity, ["normal"])
