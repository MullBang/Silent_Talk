"""Check independent-process checkpoint replay against predeclared tolerances."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

BASE = Path(__file__).resolve().parent


def compare_state(reference, actual, tolerances: dict) -> dict:
    """Compare all model/optimizer/RNG leaves; only named FP arrays get tolerance."""
    records, failures = [], []

    def visit(want, got, path: str) -> None:
        """Walk nested saved states without ignoring optimizer or random states."""
        if type(want) is not type(got):
            failures.append({"path": path, "reason": "type"})
            return
        if isinstance(want, dict):
            if want.keys() != got.keys():
                failures.append({"path": path, "reason": "keys"})
                return
            for key in want:
                visit(want[key], got[key], f"{path}.{key}" if path else str(key))
        elif isinstance(want, (list, tuple)):
            if len(want) != len(got):
                failures.append({"path": path, "reason": "length"})
                return
            for index, (left, right) in enumerate(zip(want, got)):
                visit(left, right, f"{path}.{index}")
        elif isinstance(want, (torch.Tensor, np.ndarray)):
            if isinstance(want, torch.Tensor):
                left, right = want.detach().cpu().numpy(), got.detach().cpu().numpy()
            else:
                left, right = want, got
            if left.shape != right.shape or left.dtype != right.dtype:
                failures.append({"path": path, "reason": "shape/dtype"})
                return
            exact = np.array_equal(left, right)
            category = ("model" if path.startswith("model.") else
                        "adapter" if path.startswith("adapter.") else
                        "exp_avg_sq" if path.endswith(".exp_avg_sq") else
                        "exp_avg" if path.endswith(".exp_avg") else "exact")
            policy = tolerances.get(category, {"atol": 0., "rtol": 0.})
            if np.issubdtype(left.dtype, np.floating):
                finite = bool(np.isfinite(left).all() and np.isfinite(right).all())
                difference = np.abs(left.astype(np.float64) - right.astype(np.float64))
                bound = policy["atol"] + policy["rtol"] * np.abs(left.astype(np.float64))
                passed = finite and bool(np.all(difference <= bound))
                max_abs = float(difference.max(initial=0)) if finite else None
                max_rel = float((difference / np.maximum(np.abs(left.astype(np.float64)), 1e-12)).max(initial=0)) if finite else None
            else:
                passed, max_abs, max_rel = exact, 0. if exact else None, 0. if exact else None
            item = {"path": path, "category": category, "shape": list(left.shape),
                    "dtype": str(left.dtype), "bitwise_equal": bool(exact),
                    "passed": bool(passed), "max_absolute_difference": max_abs,
                    "max_relative_difference_denominator_floor_1e12": max_rel,
                    "atol": policy["atol"], "rtol": policy["rtol"]}
            records.append(item)
            if not passed:
                failures.append({"path": path, "reason": "values", "max_absolute_difference": max_abs})
        elif want != got:
            failures.append({"path": path, "reason": "scalar value"})

    try:
        visit(reference, actual, "")
        return {"passed": not failures, "bitwise_equal": not failures and all(r["bitwise_equal"] for r in records),
                "arrays_checked": len(records), "arrays_with_bitwise_difference": sum(not r["bitwise_equal"] for r in records),
                "maximum_absolute_difference": max((r["max_absolute_difference"] or 0. for r in records), default=0.),
                "records": records, "failures": failures}
    except Exception as exc:
        raise RuntimeError("Cannot compare checkpoint state") from exc


def compare_metrics(reference: dict, actual: dict, tolerances: dict) -> dict:
    """Compare losses and pre-clipping gradient norm; timing need not match."""
    try:
        results = {}
        for key in ("ctc_loss", "feature_loss", "loss", "gradient_norm_before_clipping"):
            policy = tolerances["gradient_norm" if key == "gradient_norm_before_clipping" else "loss"]
            left, right = float(reference[key]), float(actual[key])
            difference = abs(left - right)
            results[key] = {"reference": left, "replayed": right, "absolute_difference": difference,
                            **policy, "passed": bool(np.isfinite(left) and np.isfinite(right)
                                                     and difference <= policy["atol"] + policy["rtol"] * abs(left))}
        for key in ("ids", "lambda", "optimizer_updated"):
            if reference[key] != actual[key]:
                raise ValueError(f"Replayed update changed {key}")
        return {"passed": all(item["passed"] for item in results.values()), "records": results}
    except Exception as exc:
        raise RuntimeError("Cannot compare update metrics") from exc


def audit() -> dict:
    """Read all four run artifacts and publish one immutable replay decision."""
    import runner
    import train
    try:
        plan, _ = train.verify_plan()
        bindings, arm_results, originals = [], {}, {}
        for arm in runner.core.ARMS:
            original_path = BASE / "runs" / arm / "continuous/run_result.json"
            replay_path = BASE / "runs" / arm / "resumed/run_result.json"
            original, replay = runner.core.read(original_path), runner.core.read(replay_path)
            bindings.extend([runner.core.bind(original_path), runner.core.bind(replay_path)])
            originals[arm] = original
            if original["arm"] != arm or replay["arm"] != arm:
                raise ValueError("Incorrect arm result")
            if not replay["restore_verification"]["exact"]:
                raise ValueError("Checkpoint restoration was not exact")
            if len(original["updates"]) != 4 or len(replay["updates"]) != 1:
                raise ValueError("Expected four new updates and one replay")
            left = runner.load_checkpoint(Path(original["checkpoint"]["path"]))
            right = runner.load_checkpoint(Path(replay["checkpoint"]["path"]))
            if left["completed_updates"] != right["completed_updates"] or left["completed_updates"] != 4:
                raise ValueError("Resume ended at a different schedule cursor")
            values = compare_state(left["state"], right["state"], plan["replay"]["tolerances"])
            metrics = compare_metrics(original["updates"][-1], replay["updates"][-1], plan["replay"]["tolerances"])
            initial, final = original["initial_state_hashes"], original["final_state_hashes"]
            if initial["model"] == final["model"]:
                raise ValueError("Student weights did not change")
            adapter_changed = initial["adapter"] != final["adapter"]
            if adapter_changed != (arm == "B_feature"):
                raise ValueError("The auxiliary adapter updated in the wrong arm")
            if arm == "A_ctc" and any(u["feature_loss"] != 0 for u in original["updates"]):
                raise ValueError("A unexpectedly used feature loss")
            if arm == "B_feature" and any(u["feature_loss"] <= 0 for u in original["updates"]):
                raise ValueError("B feature loss did not contribute")
            arm_results[arm] = {"passed": values["passed"] and metrics["passed"],
                                "checkpoint_restore_exact": True, "completed_updates": 4,
                                "student_weights_changed": True, "adapter_weights_changed": adapter_changed,
                                "state_comparison": values, "metric_comparison": metrics,
                                "continuous_checkpoint": original["checkpoint"], "resumed_checkpoint": replay["checkpoint"]}
        for key in ("model", "adapter", "rng"):
            if originals["A_ctc"]["initial_state_hashes"][key] != originals["B_feature"]["initial_state_hashes"][key]:
                raise ValueError(f"Paired fresh initialization differs for {key}")
        if [u["ids"] for u in originals["A_ctc"]["updates"]] != plan["schedule"] or [u["ids"] for u in originals["B_feature"]["updates"]] != plan["schedule"]:
            raise ValueError("Arms did not follow the same frozen schedule")
        result = {"schema": "step03-independent-process-replay-audit-v1",
                  "status": "passed" if all(a["passed"] for a in arm_results.values()) else "failed",
                  "plan_file": runner.core.bind(train.PLAN_PATH), "plan_sha256": plan["plan_sha256"],
                  "code": runner.core.bind(Path(__file__)), "run_bindings": bindings,
                  "paired_initial_model_adapter_rng_equal": True, "paired_schedule_equal": True,
                  "arms": arm_results, "strict_bitwise_next_update_required": False,
                  "limitations": ["Replay covers the next update after update3 in a new process on the recorded device/runtime.",
                                  "Four updates per arm do not demonstrate recognition accuracy or feature-transfer benefit."]}
        runner.core.save_json(BASE / "results/replay_audit.json", result)
        return result
    except Exception as exc:
        raise RuntimeError("Step3 replay audit did not publish a passing result") from exc


if __name__ == "__main__":
    result = audit()
    print(json.dumps({"status": result["status"], "arms": {key: {
        "passed": value["passed"], "bitwise_next_update": value["state_comparison"]["bitwise_equal"],
        "max_abs": value["state_comparison"]["maximum_absolute_difference"]} for key, value in result["arms"].items()}}), flush=True)
