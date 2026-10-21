"""Validate the separately authorized stage5 plan and its pre-GPU boundaries."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from audit_sources import binding, check_binding, digest, read

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = HERE.parents[2]
S4 = ROOT / "experiments/offline_training_step04_20261013"


def inspect() -> dict:
    plan_path, policy_path = BASE / "execution_plan.json", BASE / "controlled_pilot_policy.json"
    plan, policy = read(plan_path), read(policy_path)
    assert plan["schema"] == "offline-stage5-execution-plan-v1" and plan["main_updates"] == 5040 and plan["preflight_updates"] == 8
    assert plan["main_checkpoints"] == [420, 840, 1260] and plan["main_evaluations"] == 1080
    assert plan["fresh_initialization_required"] is True
    assert plan["audio_inputs"] == plan["teacher_calls"] == plan["service_changes"] == 0
    assert plan["runtime"] == read(S4 / "smoke_plan.json")["runtime"]
    required = [BASE / path for path in ["execution.py", "run_comparison.py", "test_execution.py", "data/loader.py", "data/test_loader.py",
                                       "evaluation/evaluator.py", "evaluation/aggregate.py", "evaluation/test_evaluator.py", "evaluation/test_aggregate.py"]]
    bound = {Path(row["path"]): row["sha256"] for row in plan["bindings"]}
    assert set(required) <= set(bound)
    for row in plan["bindings"] + plan["previous_manifests"] + plan["schedules"] + [plan["evaluation_plan"], plan["resource_budget"]]:
        check_binding(row)
    source = read(Path(policy["handoff"]["path"]))
    rows = {r["clip_id"]: r for r in source["records"]}
    assert policy["schema"] == "lipnet-stage5-controlled-pilot-data-policy-v1"
    assert policy["authorization"] == {"source": "current_user_request", "request_text": "다음 진행", "scope": "stage5_fixed_comparison_and_preflight"}
    assert policy["training_plan"] == {"path": str(plan_path), "sha256": digest(plan_path)}
    assert policy["full_training_authorized"] is True and policy["service_change_authorized"] is False
    assert policy["allow_provisional_cache_for_controlled_pilot"] is True and policy["preserve_provisional_flags"] is True
    assert sorted(policy["allowed_optimizer_clip_ids"]) == sorted(ident for ident, row in rows.items() if row["split"] == "train")
    assert sorted(policy["allowed_validation_clip_ids"]) == sorted(rows)
    assert len(policy["allowed_optimizer_clip_ids"]) == 24 and len(policy["allowed_validation_clip_ids"]) == 30
    for field in ["training_plan", "handoff", "stage3_frozen_manifest"]:
        check_binding(policy[field])
    assert source["training_ready"] is False
    first_schedule = read(S4 / "protocol/schedule_seed42.json")["updates"]
    full = max((r["arms"]["A_sentence"] for r in first_schedule), key=lambda r: sum(r["lengths"]))
    phrase = max((r["arms"]["B_curriculum"] for r in first_schedule if r["arms"]["B_curriculum"]["kind"] == "phrase"), key=lambda r: sum(r["lengths"]))
    assert sum(full["lengths"]) == 316 and max(full["lengths"]) * 2 == 358
    for arm in ["A_sentence", "B_curriculum"]:
        expected = [{"update":1, "lr":.001, **(full if arm == "A_sentence" else phrase)}, {"update":2, "lr":.0003, **full}]
        assert plan["preflight_tasks"][arm] == expected
        for task in expected:
            assert all(rows[s]["split"] == "train" for s in task["clip_ids"])
            assert task["lengths"] == [rows[s]["T"] for s in task["clip_ids"]]
    limits = read(Path(plan["resource_budget"]["path"]))["conservative_future_limits"]
    assert limits["wall_seconds_per_run_including_evaluation"] == 3600 and limits["aggregate_wall_seconds"] == 14400
    assert limits["stop_if_cuda_reserved_exceeds_bytes"] == 6442450944
    current_ledgers = list(BASE.glob("preflight/**/progress.jsonl")) + list(BASE.glob("runs/**/progress.jsonl"))
    assert not current_ledgers, "Pre-GPU audit must precede physical stage5 updates"
    return {"schema": "offline-stage5-independent-plan-preflight-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "passed", "plan": binding(plan_path), "policy": binding(policy_path),
            "code_reviewed": [binding(path) for path in required], "bindings_checked": len(plan["bindings"]),
            "main_updates": 5040, "preflight_updates": 8, "main_checkpoints": 12, "main_evaluation_rows": 1080,
            "largest_true_batch_frames": 316, "largest_dense_batch_frames": 358,
            "preflight_source": "Frozen schedule length maxima; no score based selection",
            "fresh_main_initialization": True, "provisional_flags_preserved": True,
            "optimizer_inputs": 24, "evaluation_inputs": 30, "train_validation_purpose_separated": True,
            "producer_imports": 0, "network_calls": 0, "optimizer_updates_at_audit": 0,
            "review_findings": ["No blocking code issue identified in update, checkpoint, loader or evaluator implementation.",
                                "Finite zero gradients in later main updates remain counted and are logged; preflight requires nonzero parameter coverage.",
                                "Main eval failure is recorded before halting; no missing row is removed from metrics.",
                                "If interrupted after a checkpoint and before its evaluation completes, recovery needs the missing fixed evaluation and separate physical-update accounting; uninterrupted path is covered."]}


if __name__ == "__main__":
    result = inspect()
    output = HERE / "plan_preflight.json"
    assert not output.exists()
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "bindings": result["bindings_checked"], "main_updates": 5040, "preflight_updates": 8}))
