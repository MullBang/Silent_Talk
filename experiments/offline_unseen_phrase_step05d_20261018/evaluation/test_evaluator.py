"""CPU-only scope/checkpoint, failure-accounting and recovery guards."""
import copy
import importlib.util
import json
from pathlib import Path
import unicodedata

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location("stage5d_eval_tests", Path(__file__).with_name("evaluator.py"))
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)


def fixture(seed=42):
    old = json.loads(ev.helper.PLAN_PATH.read_text(encoding="utf-8"))
    handoff = json.loads(ev.HANDOFF_PATH.read_text(encoding="utf-8"))
    return {"items": [r for r in old["items"] if r["split"] == "dev" and r["kind"] == "phrase"],
            "checkpoints": [1680], "conditions_by_update": copy.deepcopy(ev.CONDITIONS),
            "resource_caps": {"cuda_reserved_bytes": ev.CUDA_RESERVED_BUDGET_BYTES, "seconds_per_model": ev.SECONDS_PER_MODEL}}, {
            "seed": seed, "mode": "evaluation_only", "completed_update": 1680, "checkpoint": handoff["checkpoints"][[42, 43].index(seed)]}


@pytest.mark.parametrize("seed", [42, 43])
def test_three_dev_phrases_exact_frozen_checkpoint(seed):
    plan, identity = fixture(seed)
    conditions = ev.validate_plan(plan, identity, verify_sources=False)
    assert len(ev.expected_pairs(plan, conditions)) == 9
    assert sum(r["T"] for r in plan["items"]) * len(conditions) * 2 == 576


@pytest.mark.parametrize("fault", ["train", "sentence", "duplicate", "permutation", "missing_control", "wrong_update", "wrong_seed_checkpoint", "checkpoint_hash", "resource_cap", "mode"])
def test_scope_or_checkpoint_changes_rejected(fault):
    plan, identity = copy.deepcopy(fixture())
    if fault == "train": plan["items"][0]["split"] = "train"
    elif fault == "sentence": plan["items"][0]["kind"] = "full_sentence"
    elif fault == "duplicate": plan["items"][-1] = plan["items"][0]
    elif fault == "permutation": plan["items"][0]["shuffled_frame_indices"].reverse()
    elif fault == "missing_control": plan["conditions_by_update"]["1680"] = ["normal"]
    elif fault == "wrong_update": identity["completed_update"] = 1260
    elif fault == "wrong_seed_checkpoint": identity["seed"] = 43
    elif fault == "resource_cap": plan["resource_caps"]["cuda_reserved_bytes"] += 1
    elif fault == "mode": identity["mode"] = "main"
    else: identity["checkpoint"]["sha256"] = "0" * 64
    with pytest.raises(ValueError): ev.validate_plan(plan, identity, verify_sources=False)


def test_load_validation_uses_evaluation_only_purpose():
    plan, _ = fixture()
    item = plan["items"][0]
    reverse = {v: k for k, v in ev.contract.IDX_TO_GRAPHEME.items()}
    loaded = {"clip_id": item["clip_id"], "parent_id": item["parent_id"], "split": "dev", "kind": "phrase",
              "length": item["T"], "purpose": "evaluation", "labels": np.asarray([reverse[c] for c in unicodedata.normalize("NFD", item["text"])], dtype=np.int64),
              "audit": {"source_training_ready": False, "source_provisional": True, "approval_changed": False}}
    ev.validate_loaded(loaded, item)
    loaded["purpose"] = "validation"
    with pytest.raises(ValueError, match="use differs"): ev.validate_loaded(loaded, item)


def test_failed_row_keeps_denominator():
    rows = [{"clip_id": "a", "split": "dev", "kind": "phrase", "condition": "normal", "metrics": ev.contract.score_prediction("가 나", "", failed=True)},
            {"clip_id": "b", "split": "dev", "kind": "phrase", "condition": "normal", "metrics": ev.contract.score_prediction("가", "가")}]
    score = ev.summarize_rows(rows)["cohorts"]["dev/phrase/normal"]
    assert score["CER"] == .75 and score["failure_count"] == 1 and score["count"] == 2
    with pytest.raises(ValueError, match="Duplicate"): ev.summarize_rows([rows[0], rows[0]])


def test_resume_rejects_incomplete_and_changed_identity(tmp_path, monkeypatch):
    plan, identity = fixture()
    with pytest.raises(ValueError, match="Incomplete existing"): ev.reuse_completed(tmp_path, plan, identity, ev.CONDITIONS["1680"])
    path = tmp_path / "plan.json"
    ev.write_json(path, plan)
    monkeypatch.setattr(ev, "PLAN_PATH", path)
    ev.write_json(tmp_path / "evaluation_summary.json", {"identity": {**identity, "seed": 43}, "evaluation_plan_sha256": ev.file_sha256(path)})
    with pytest.raises(ValueError, match="identity/plan"): ev.reuse_completed(tmp_path, plan, identity, ev.CONDITIONS["1680"])


def test_completed_reuse_verifies_saved_files_and_does_not_infer(tmp_path, monkeypatch):
    plan, identity = fixture()
    plan_path = tmp_path / "plan.json"
    ev.write_json(plan_path, plan)
    monkeypatch.setattr(ev, "PLAN_PATH", plan_path)
    rows = []
    for item in plan["items"]:
        for condition in ev.CONDITIONS["1680"]:
            posterior = tmp_path / f"{item['clip_id']}_{condition}.npz"
            posterior.write_bytes(b"test-file-hash-only")
            rows.append({"clip_id": item["clip_id"], "split": "dev", "kind": "phrase", "condition": condition, "identity": identity,
                         "metrics": ev.contract.score_prediction(item["text"], item["text"]), "posterior": ev.ref(posterior)})
    log = tmp_path / "rows.jsonl"
    log.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    isolation = ev.write_json(tmp_path / "state_isolation.json", {"fatal": None, "model_unchanged": True, "gradients_unchanged": True,
        "rng_restored_exact": True, "module_training_flags_restored": True})
    summary = {"identity": identity, "evaluation_plan_sha256": ev.file_sha256(plan_path), "status": "completed", "failures": 0,
               "rows": rows, "metrics": ev.summarize_rows(rows), "state_isolation": isolation, "rows_jsonl": ev.ref(log),
               "count": 9, "forward_calls": 9, "attempted_forward_frames": 288, "seconds": 1., "peak_allocated_cuda_bytes": 0,
               "peak_reserved_cuda_bytes": 0, "cuda_reserved_budget_passed": True, "state_isolation_passed": True}
    ev.write_json(tmp_path / "evaluation_summary.json", summary)
    result = ev.reuse_completed(tmp_path, plan, identity, ev.CONDITIONS["1680"])
    assert result["reused"] and result["new_forward_calls"] == 0 and result["count"] == 9
    Path(rows[0]["posterior"]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact changed"): ev.reuse_completed(tmp_path, plan, identity, ev.CONDITIONS["1680"])


def aggregate_fixture(tmp_path, fault=None):
    spec = importlib.util.spec_from_file_location("stage5d_aggregate_test_module", Path(__file__).with_name("aggregate.py"))
    agg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agg)
    paths = []
    for seed in [42, 43]:
        plan, identity = fixture(seed)
        rows = []
        for item in plan["items"]:
            for condition in ev.CONDITIONS["1680"]:
                rows.append({"clip_id": item["clip_id"], "identity": identity, "reference_text": item["text"], "raw_text": "",
                             "condition": condition, "split": "dev", "kind": "phrase", "metrics": ev.contract.score_prediction(item["text"], "")})
        if fault == "unplanned_clip" and seed == 43:
            rows[-1]["clip_id"] = "unplanned"
        path = tmp_path / f"seed{seed}.json"
        ev.write_json(path, {"identity": identity, "count": 9, "rows": rows, "metrics": ev.summarize_rows(rows), "status": "completed",
                            "failures": 0, "state_isolation_passed": True, "forward_calls": 9, "attempted_forward_frames": 288})
        paths.append(path)
    return agg, paths


def test_aggregate_preserves_both_models_and_all_conditions(tmp_path):
    agg, paths = aggregate_fixture(tmp_path)
    result = agg.aggregate_summaries(paths)
    assert result["evaluation_count"] == 18 and result["frames"] == 576
    assert len(result["diagnostics"]) == 6 and result["primary"]["independent_utterances"] == 3
    assert result["primary"]["mean_CER"] == 1 and result["optimizer_updates"] == 0
    assert result["exploratory_reused_development_probe"] and not result["generalization_established"]


def test_aggregate_rejects_duplicated_checkpoint(tmp_path):
    agg, paths = aggregate_fixture(tmp_path)
    with pytest.raises(ValueError, match="duplicated"): agg.aggregate_summaries([paths[0], paths[0]])


def test_aggregate_rejects_unplanned_clip(tmp_path):
    agg, paths = aggregate_fixture(tmp_path, "unplanned_clip")
    with pytest.raises(ValueError, match="Unexpected clip"): agg.aggregate_summaries(paths)
