"""Test fail-closed handoff gates and future schedule cohort isolation."""
from copy import deepcopy
import json

import pytest

import freeze_handoff as handoff


@pytest.fixture
def ready():
    train = [f"train{i}" for i in range(441)]
    dev = [f"dev{i}" for i in range(71)]
    refs = {key: {"path": key, "sha256": key} for key in ("recipe", "inputs", "input_audit", "features", "teacher_plan")}
    refs["recipe"]["sha256"] = handoff.RECIPE_SHA
    docs = {
        "recipe": {"train_ids": train, "development_ids": dev, "parent_plan": {"path": "parent", "sha256": "parent"},
                   "source_plan": {"path": "source", "sha256": "source"}},
        "inputs": {"status": "passed", "record_count": 512,
                   "records": [{"id": sid, "split": "train", "teacher_frames": 12} for sid in train]
                   + [{"id": sid, "split": "dev", "teacher_frames": 0} for sid in dev]},
        "features": {"status": "completed", "record_count": 441, "ids": train,
                     "records": [{"id": sid} for sid in train], "input_manifest": refs["inputs"],
                     "input_audit": refs["input_audit"], "teacher_plan": refs["teacher_plan"]},
        "input_audit": {"schema": "local-tar-step04-independent-input-audit-v1", "scope": "all512",
                        "status": "passed", "record_count": 512, "manifest": refs["inputs"],
                        "plan": {"path": "parent", "sha256": "parent"}, "source_plan": {"path": "source", "sha256": "source"},
                        "reused_pilot_records_unchanged": 8, "student_only_dev_records": 71},
        "visual_review": {"status": "passed_sampled_review", "input_manifest": refs["inputs"]},
        "feature_audit": {"schema": "local-tar-step04-independent-feature-audit-v1", "status": "passed",
                          "record_count": 441, "manifest": refs["features"], "input_manifest": refs["inputs"],
                          "teacher_plan": refs["teacher_plan"], "input_audit": refs["input_audit"],
                          "reused_pilot_records_unchanged": 8, "new_records": 433, "development_target_count": 0}}
    return docs, refs


def test_accepted_gates(ready):
    handoff.validate_ready(*ready)


@pytest.mark.parametrize("key,field", [("input_audit", "manifest"), ("features", "teacher_plan"),
                                     ("feature_audit", "manifest"), ("visual_review", "input_manifest")])
def test_passed_status_for_other_artifacts_is_rejected(ready, key, field):
    docs, refs = deepcopy(ready)
    docs[key][field] = {"path": "other", "sha256": "other"}
    with pytest.raises(ValueError):
        handoff.validate_ready(docs, refs)


def test_development_cannot_acquire_teacher_targets(ready):
    docs, refs = deepcopy(ready)
    docs["inputs"]["records"][-1]["teacher_frames"] = 1
    with pytest.raises(ValueError):
        handoff.validate_ready(docs, refs)


def test_changed_train_order_is_rejected(ready):
    docs, refs = deepcopy(ready)
    docs["features"]["records"][0], docs["features"]["records"][1] = docs["features"]["records"][1], docs["features"]["records"][0]
    with pytest.raises(ValueError):
        handoff.validate_ready(docs, refs)


@pytest.mark.parametrize("key,value", [("scope", "pilot8"), ("student_only_dev_records", 0),
                                      ("plan", {"path": "other", "sha256": "other"})])
def test_incomplete_or_different_input_audit_is_rejected(ready, key, value):
    docs, refs = deepcopy(ready)
    docs["input_audit"][key] = value
    with pytest.raises(ValueError):
        handoff.validate_ready(docs, refs)


def test_real_full_schedules_have_exact_cohort_and_epoch_coverage():
    recipe = json.loads((handoff.STEP3 / "comparison_protocol.json").read_text(encoding="utf-8"))
    schedules = handoff.schedules(recipe)
    assert [item["seed"] for item in schedules] == [42, 43]
    assert schedules[0]["schedule"] != schedules[1]["schedule"]
    for item in schedules:
        assert item["arms"] == ["A_ctc", "B_feature"]
        stream = [sid for batch in item["schedule"] for sid in batch]
        assert len(stream) == 6400 and not set(stream) & set(recipe["development_ids"])
        for start in range(0, 441 * 14, 441):
            assert set(stream[start:start + 441]) == set(recipe["train_ids"])
        assert len(set(stream[441 * 14:])) == 226
