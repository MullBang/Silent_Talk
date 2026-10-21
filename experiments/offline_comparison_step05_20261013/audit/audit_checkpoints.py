"""Audit complete saved tensors, physical updates and fixed schedules on CPU."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from audit_common import fingerprint, state_hashes, unpack
from audit_sources import binding, check_binding, digest, read

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = HERE.parents[2]
S4 = ROOT / "experiments/offline_training_step04_20261013"


def check_state(state: dict, done: int, lr: float) -> dict:
    assert set(state) == {"model", "optimizer", "rng"}
    model, optimizer, rng = state["model"], state["optimizer"], state["rng"]
    assert sum(x.numel() for x in model.values()) == 3287781
    assert all(x.dtype == torch.float32 and torch.isfinite(x).all() for x in model.values())
    groups = optimizer["param_groups"]
    assert len(groups) == 1
    group = groups[0]
    assert group["lr"] == lr and tuple(group["betas"]) == (.9, .999)
    assert group["eps"] == 1e-8 and group["weight_decay"] == 0 and group["amsgrad"] is False
    assert len(group["params"]) == len(set(group["params"])) == len(model) == len(optimizer["state"])
    assert set(group["params"]) == set(optimizer["state"])
    for ident, parameter in zip(group["params"], model.values()):
        saved = optimizer["state"][ident]
        assert set(saved) == {"step", "exp_avg", "exp_avg_sq"}
        assert saved["step"].numel() == 1 and float(saved["step"]) == done
        assert saved["exp_avg"].shape == saved["exp_avg_sq"].shape == parameter.shape
        assert saved["exp_avg"].dtype == saved["exp_avg_sq"].dtype == parameter.dtype
        assert torch.isfinite(saved["exp_avg"]).all() and torch.isfinite(saved["exp_avg_sq"]).all()
        assert (saved["exp_avg_sq"] >= 0).all()
    assert set(rng) == {"python", "numpy", "torch", "cuda"}
    assert len(rng["python"]) == 3 and rng["python"][0] == 3
    assert len(rng["numpy"]) == 5 and rng["numpy"][0] == "MT19937" and rng["numpy"][1].dtype == np.uint32
    assert rng["torch"].dtype == torch.uint8 and rng["torch"].ndim == 1
    assert len(rng["cuda"]) == 1 and rng["cuda"][0].dtype == torch.uint8
    return {"parameters": 3287781, "parameter_tensors": len(model), "Adam_states": len(optimizer["state"]), "Adam_steps": done}


def check_history(history: list[dict], tasks: list[dict], lookup: dict) -> None:
    for number, row in enumerate(history, 1):
        task = tasks[number - 1]
        assert row["update"] == number and row["kind"] == task["kind"]
        assert row["clip_ids"] == task["clip_ids"] and row["learning_rate"] == task["lr"]
        expected = [lookup[clip] for clip in row["clip_ids"]]
        assert all(r["split"] == "train" for r in expected)
        lengths = [r["T"] for r in expected]
        assert row["lengths"] == lengths and row["true_frames"] == sum(lengths)
        assert row["dense_frames"] == 2 * max(lengths) and row["padding_frames"] == 2 * max(lengths) - sum(lengths)
        assert row["target_tokens"] == sum(r["label_length"] for r in expected)
        assert row["optimizer_updated"] is True
        assert 0 <= row["nonzero_gradient_tensors"] <= row["parameter_tensors"]
        assert 0 <= row["gradient_norm_after_clip"] <= 1.0001
        assert all(math.isfinite(row[key]) for key in ["loss", "gradient_norm_before_clip", "gradient_norm_after_clip", "seconds"])
        assert row["seconds"] >= 0 and row["cuda_peak_reserved_bytes"] <= 6442450944
        assert row["cuda_peak_allocated_bytes"] <= row["cuda_peak_reserved_bytes"]


def inspect(mode: str) -> dict:
    torch.set_num_threads(4)
    plan_path = BASE / "execution_plan.json"
    if not plan_path.exists():
        choices = [p for p in BASE.glob("*.json") if read(p).get("schema") == "offline-stage5-execution-plan-v1"]
        assert len(choices) == 1, choices
        plan_path = choices[0]
    plan = read(plan_path)
    for item in plan["bindings"]:
        check_binding(item)
    for item in plan["previous_manifests"]:
        check_binding(item)
    policies = [p for p in BASE.glob("*.json") if read(p).get("schema") == "lipnet-stage5-controlled-pilot-data-policy-v1"]
    assert len(policies) == 1
    policy_path = policies[0]
    policy = read(policy_path)
    assert policy["training_plan"] == {"path": str(plan_path), "sha256": digest(plan_path)}
    lookup = {r["clip_id"]: r for r in read(Path(policy["handoff"]["path"]))["records"]}
    old_initial = read(S4 / "results/smoke_summary.json")["paired_initial_state_hashes"]
    results = []
    for path in sorted(BASE.glob("**/segment_result.json")):
        value = read(path)
        if value["identity"]["mode"] == mode:
            results.append((path, value))
    assert len(results) == (6 if mode == "preflight" else 4), len(results)
    updates = true_frames = dense_frames = checkpoint_count = 0
    maximum_memory = 0
    all_progress = []
    contracts = []
    checkpoints = []
    finals = {}
    initial_by_seed = {}
    main_identities = set()
    for path, value in results:
        identity = value["identity"]
        seed, arm = identity["seed"], identity["arm"]
        assert identity["plan_sha256"] == digest(plan_path) and identity["policy_sha256"] == digest(policy_path)
        assert identity["runtime"] == plan["runtime"]
        if mode == "preflight":
            tasks = plan["preflight_tasks"][arm]
            assert seed == 42
        else:
            schedule = read(S4 / f"protocol/schedule_seed{seed}.json")["updates"]
            tasks = [{"update":r["update"], "lr":r["lr"], **r["arms"][arm]} for r in schedule]
            assert (seed, arm) not in main_identities
            main_identities.add((seed, arm))
        assert identity["tasks_sha256"] == hashlib.sha256(json.dumps(tasks, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
        initial = value["initial_state_hashes"]
        assert initial == old_initial[str(seed)]
        if seed in initial_by_seed:
            assert initial_by_seed[seed] == initial
        initial_by_seed[seed] = initial
        contract = read(path.parent / "segment_contract.json")
        assert contract["identity"] == identity and contract["initial_state_hashes"] == initial
        contracts.append(binding(path.parent / "segment_contract.json"))
        check_binding(value["progress"])
        progress = [json.loads(line) for line in Path(value["progress"]["path"]).read_text(encoding="utf-8").splitlines()]
        start, stop = contract["start_update"], contract["stop_update"]
        assert len(progress) == value["executed_updates"] == stop - start
        assert [r["update"] for r in progress] == list(range(start + 1, stop + 1))
        if start:
            assert contract["resume"] and contract["restore"]["exact"] is True
            check_binding(contract["resume"])
            resumed = unpack(torch.load(contract["resume"]["path"], map_location="cpu", weights_only=True))
            assert resumed["state_hashes"] == contract["restore"]["state_hashes"]
            assert contract["restore"]["next_update"] == start + 1
        else:
            assert contract["resume"] is None and contract["restore"] is None
        for reference in value["checkpoints"]:
            check_binding(reference)
            payload = unpack(torch.load(reference["path"], map_location="cpu", weights_only=True))
            assert payload["schema"] == "offline-stage5-complete-checkpoint-v1" and payload["identity"] == identity
            done = payload["completed_updates"]
            assert payload["initial_state_hashes"] == initial
            assert len(payload["history"]) == done
            assert payload["history"][start:done] == progress[:done - start]
            check_history(payload["history"], tasks, lookup)
            assert payload["next_task"] == (tasks[done] if done < len(tasks) else None)
            assert payload["state_hashes"] == state_hashes(payload["state"])
            assert payload["state_hashes"]["model"] != initial["model"]
            checked = check_state(payload["state"], done, tasks[done - 1]["lr"])
            checkpoints.append({"checkpoint": reference, "mode": mode, "seed": seed, "arm": arm, "update": done,
                                "state_hashes": payload["state_hashes"], **checked})
            checkpoint_count += 1
            if done == stop:
                finals[str(path.parent)] = payload["state_hashes"]
        assert value["status"] == "completed" and value["completed_updates"] == stop
        assert value["audio_inputs"] == value["teacher_calls"] == value["service_changes"] == 0
        if mode == "main":
            assert stop == 1260 and start == 0 and len(value["checkpoints"]) == len(value["evaluations"]) == 3
            for evaluation, expected_update in zip(value["evaluations"], [420, 840, 1260]):
                assert evaluation["count"] == evaluation["forward_calls"] == 90 and evaluation["failures"] == 0
                assert evaluation["frames"] == 6069 and evaluation["state_isolation_passed"] is True
                assert evaluation["identity"]["completed_update"] == expected_update
                assert all(evaluation["identity"][key] == val for key, val in identity.items())
                check_binding(evaluation["summary"])
        else:
            assert value["evaluations"] == []
            assert all(r["nonzero_gradient_tensors"] == r["parameter_tensors"] for r in progress)
        all_progress.extend(progress)
        updates += len(progress)
        true_frames += sum(r["true_frames"] for r in progress)
        dense_frames += sum(r["dense_frames"] for r in progress)
        maximum_memory = max(maximum_memory, max(r["cuda_peak_reserved_bytes"] for r in progress))
    assert updates == (8 if mode == "preflight" else 5040)
    assert checkpoint_count == (8 if mode == "preflight" else 12)
    if mode == "main":
        assert (true_frames, dense_frames) == (911540, 1085994)
        assert main_identities == {(s, a) for s in [42, 43] for a in ["A_sentence", "B_curriculum"]}
    else:
        assert max(r["true_frames"] for r in all_progress) == 316
        assert max(r["dense_frames"] for r in all_progress) == 358
        for arm in ["A_sentence", "B_curriculum"]:
            terminal = [row["state_hashes"] for row in checkpoints if row["arm"] == arm and row["update"] == 2]
            assert len(terminal) == 2 and terminal[0] == terminal[1]
    return {"schema": "offline-stage5-independent-complete-state-audit-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "passed", "mode": mode, "producer_imports": 0, "neural_forward_calls": 0, "optimizer_updates_by_audit": 0,
            "execution_plan": binding(plan_path), "policy": binding(policy_path), "segment_count": len(results),
            "physical_training_updates": updates, "checkpoints": checkpoint_count, "true_frames": true_frames, "dense_frames": dense_frames,
            "max_true_batch_frames": max(r["true_frames"] for r in all_progress), "max_dense_batch_frames": max(r["dense_frames"] for r in all_progress),
            "max_cuda_reserved_bytes": maximum_memory, "initial_states_equal_to_frozen_fresh_initialization": True,
            "replay_all_bitwise_exact": True if mode == "preflight" else None,
            "zero_gradient_tensor_update_count": sum(r["nonzero_gradient_tensors"] < r["parameter_tensors"] for r in all_progress),
            "contracts": contracts, "checkpoint_details": checkpoints}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["preflight", "main"])
    args = parser.parse_args()
    result = inspect(args.mode)
    output = HERE / f"{args.mode}_checkpoint_audit.json"
    assert not output.exists()
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ["status", "mode", "physical_training_updates", "checkpoints", "true_frames", "dense_frames"]}))
