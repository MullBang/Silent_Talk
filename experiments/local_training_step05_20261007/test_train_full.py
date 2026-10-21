"""Check fixed-budget execution cannot drift from the accepted comparison."""
from copy import deepcopy
import json

import pytest

import train_full as train


@pytest.fixture
def contract():
    ref, data, recipe = train.handoff()
    plan = {"schema": train.SCHEMA, "seed": 42, "updates": 3200, "batch_size": 2,
            "checkpoint_interval": 100, "ids": data["train_ids"], "development_ids": data["development_ids"],
            "arms": list(train.ARMS), "schedule": train.schedule_for(data, 42),
            "optimizer": recipe["optimizer"], "loss": recipe["loss"],
            "primary_checkpoint": recipe["primary_checkpoint"], "evaluation": recipe["evaluation"]}
    plan["plan_sha256"] = train.runner.canonical_plan_hash(plan)
    return plan, data, recipe


def test_accepted_frozen_execution_contract(contract):
    plan, data, recipe = contract
    train.check_plan_contract(plan, 42, data, recipe)


@pytest.mark.parametrize("field,value", [("updates", 3201), ("batch_size", 3), ("checkpoint_interval", 1),
                                       ("primary_checkpoint", "best-dev"), ("seed", 43)])
def test_rehashed_plan_cannot_change_budget_or_selection(contract, field, value):
    plan, data, recipe = deepcopy(contract)
    plan[field] = value
    plan["plan_sha256"] = train.runner.canonical_plan_hash(plan)
    with pytest.raises(ValueError):
        train.check_plan_contract(plan, 42, data, recipe)


def test_development_id_cannot_enter_rehashed_schedule(contract):
    plan, data, recipe = deepcopy(contract)
    plan["schedule"][0][0] = data["development_ids"][0]
    plan["plan_sha256"] = train.runner.canonical_plan_hash(plan)
    with pytest.raises(ValueError):
        train.check_plan_contract(plan, 42, data, recipe)


def test_schedule_forbids_different_order_even_with_same_cohort(contract):
    _, data, _ = deepcopy(contract)
    item = next(s for s in data["training_schedules"] if s["seed"] == 42)
    item["schedule"][0].reverse()
    with pytest.raises(ValueError):
        train.schedule_for(data, 42)


def test_missing_seed_schedule_is_rejected(contract):
    _, data, _ = deepcopy(contract)
    data["training_schedules"] = [s for s in data["training_schedules"] if s["seed"] != 43]
    with pytest.raises(ValueError):
        train.schedule_for(data, 43)


def test_unsafe_recovery_name_fails_before_training(monkeypatch, contract):
    plan, _, _ = contract
    monkeypatch.setattr(train, "verify_plan", lambda seed: (plan, {"clips": {}}))
    with pytest.raises(ValueError, match="simple isolated name"):
        train.execute(42, "A_ctc", train.BASE / "runs/seed42/A_ctc/continuous/update_0100.pt", "../../escape")


def test_foreign_checkpoint_cannot_be_resumed(monkeypatch, contract):
    plan, _, _ = contract
    monkeypatch.setattr(train, "verify_plan", lambda seed: (plan, {"clips": {}}))
    with pytest.raises(ValueError, match="own checkpoint"):
        train.execute(42, "A_ctc", train.STEP3 / "runs/A_ctc/continuous/update_0004.pt", "recovered")


def test_ambiguous_completed_trajectories_cannot_be_selected(monkeypatch, tmp_path):
    monkeypatch.setattr(train, "BASE", tmp_path)
    home = tmp_path / "runs/seed42/A_ctc"
    for path in [home / "continuous/run_result.json", home / "recoveries/duplicate/run_result.json"]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match="Ambiguous"):
        train.completed_run_result(42, "A_ctc")
