"""Bounded visual-only stage4 backpropagation and complete-state recovery engine.

This entry point accepts only the six-update preparation plan. It cannot launch
the future comparison experiment. Previously frozen model/helpers stay intact.
"""
from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from torch.nn import functional as F

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BASE / "data"))
from backend.models.baseline import LipNetBaseline
from loader import ControlledPilotDataset, collate

RUNNER_PATH = ROOT / "experiments/local_training_step03_20261007/runner.py"
POOL_PATH = ROOT / "experiments/diversity_training_step16r1_20261008/deterministic_pool.py"


def require(condition: bool, message: str) -> None:
    """Reject a violated execution contract before further mutation."""
    if not condition:
        raise ValueError(message)


def sha(path: Path) -> str:
    """Hash an explicit local input or output with bounded memory."""
    try:
        with Path(path).open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()
    except OSError as exc:
        raise RuntimeError(f"Cannot hash {path}") from exc


def read(path: Path) -> Any:
    """Read JSON evidence without changing it."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Cannot read {path}") from exc


def save(path: Path, value: Any) -> None:
    """Create new stage4 JSON only; never overwrite a completed artifact."""
    try:
        require(path.resolve().is_relative_to(BASE), "Output outside stage4")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Cannot create {path}") from exc


def module(path: Path, name: str) -> Any:
    """Import a local frozen helper without invoking its CLI."""
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "Missing helper module")
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


require(sha(RUNNER_PATH) == "71ef1e815ce21a0e52b2aff37257f570c4b07fa81531de9f38e1681c695e8693", "Frozen runner changed")
runner = module(RUNNER_PATH, "offline4_safe_state_primitives")
pool = module(POOL_PATH, "offline4_frozen_deterministic_pool")
core = runner.core


def setup() -> dict[str, Any]:
    """Select the previously verified deterministic FP32 GPU execution contract."""
    pool.setup_runtime(core)
    require(torch.cuda.is_available(), "CUDA is required for the declared GPU smoke")
    return {"python": sys.version, "torch": str(torch.__version__), "numpy": np.__version__,
        "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "device": torch.cuda.get_device_name(0), "precision": "GPU FP32 and CPU FP32 CTC autograd",
        "strict_deterministic": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cublas": os.environ.get("CUBLAS_WORKSPACE_CONFIG"), "threads": torch.get_num_threads(),
        "tf32": False, "amp": False, "pooling": pool.pooling_contract()}


def objects(seed: int) -> tuple[torch.nn.Module, torch.optim.Optimizer]:
    """Create only a fresh compact student and Adam; no historical weights/teacher."""
    core.seed_all(seed)
    model = pool.configure_model(LipNetBaseline()).to("cuda:0").train()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, betas=(.9, .999), eps=1e-8, weight_decay=0.)
    return model, optimizer


def state(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    """Capture independent CPU copies plus all Python/NumPy/Torch/CUDA RNG states."""
    return {"model": core.cpu_clone(model.state_dict()), "optimizer": core.cpu_clone(optimizer.state_dict()),
            "rng": core.capture_rng()}


def hashes(value: dict[str, Any]) -> dict[str, str]:
    """Hash every state component, including optimizer counters and RNG bytes."""
    return {**{k: runner.nested_hash(value[k]) for k in ("model", "optimizer", "rng")},
            "complete": runner.nested_hash(value)}


def restore(value: dict[str, Any], model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
    """Restore RNG last, after creating/loading model and optimizer objects."""
    model.load_state_dict(value["model"], strict=True)
    optimizer.load_state_dict(value["optimizer"])
    core.restore_rng(value["rng"])


def verify_files(plan: dict[str, Any]) -> None:
    """Enforce the frozen code, data and explicitly limited smoke scope."""
    require(plan["schema"] == "offline-stage4-six-update-smoke-plan-v1" and plan["updates"] == 6, "Only stage4 six-update smoke allowed")
    require(plan["seeds"] == [42, 43] and plan["arms"] == ["A_sentence", "B_curriculum"], "Smoke trajectories changed")
    require(plan["optimizer"] == {"name": "Adam", "betas": [.9, .999], "eps": 1e-8, "weight_decay": 0., "clip_norm": 1.}, "Optimizer contract changed")
    for item in plan["bindings"]:
        require(sha(Path(item["path"])) == item["sha256"], f"Bound source changed: {item['path']}")


def dataset(plan_path: Path, policy_path: Path) -> ControlledPilotDataset:
    """Open the explicitly authorized controlled-pilot training split only."""
    policy = read(policy_path)
    require(policy["training_plan"] == {"path": str(plan_path.resolve()), "sha256": sha(plan_path)}, "Policy is for another smoke plan")
    handoff = policy["handoff"]
    return ControlledPilotDataset(Path(handoff["path"]), handoff["sha256"], policy_path, sha(policy_path), split="train")


def make_batch(data: ControlledPilotDataset, ids: list[str]) -> dict[str, Any]:
    """Load exact scheduled training examples, rejecting any developer/test mixing."""
    items = [data.load_clip(ident, purpose="optimizer") for ident in ids]
    result = collate(items, purpose="optimizer")
    require(result["clip_ids"] == ids and all(s == "train" for s in result["splits"]), "Optimizer batch identity changed")
    return result


def ctc(log_probs: torch.Tensor, batch: dict[str, Any]) -> torch.Tensor:
    """Compute true-length, target-length-normalized CPU FP32 CTC with gradients."""
    lengths = torch.from_numpy(batch["lengths"])
    targets = torch.from_numpy(batch["labels"])
    target_lengths = torch.from_numpy(batch["label_lengths"])
    require(log_probs.shape == (int(lengths.max()), len(lengths), 69), "Model failed to preserve padded T")
    require(bool(torch.isfinite(log_probs).all()), "Nonfinite log probabilities")
    return F.ctc_loss(log_probs.float().cpu(), targets, lengths, target_lengths,
                      blank=0, reduction="mean", zero_infinity=False)


def update(model: torch.nn.Module, optimizer: torch.optim.Optimizer, batch: dict[str, Any], lr: float) -> dict[str, Any]:
    """Apply one dense batch2 update; account for true/padded frames and finite gradients."""
    try:
        require(len(batch["lengths"]) == 2, "Exactly two logical clips required")
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:
            group["lr"] = lr
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        began = time.perf_counter()
        inputs = torch.from_numpy(batch["inputs"]).to("cuda:0")
        lengths = torch.from_numpy(batch["lengths"])
        output = model(inputs, lengths=lengths)
        loss = ctc(output, batch)
        require(bool(torch.isfinite(loss)), "Nonfinite CTC loss")
        loss.backward()
        parameters = list(model.parameters())
        require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in parameters), "Missing/nonfinite parameter gradient")
        nonzero = sum(bool(torch.count_nonzero(p.grad)) for p in parameters)
        require(nonzero == len(parameters), "A parameter tensor received no gradient")
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        after_norm = math.sqrt(sum(float(torch.sum(p.grad.double().square()).cpu()) for p in parameters))
        require(after_norm <= 1.0001, "Gradient clip bound exceeded")
        optimizer.step()
        require(all(bool(torch.isfinite(p).all()) for p in parameters), "Nonfinite optimized parameter")
        torch.cuda.synchronize()
        return {"clip_ids": batch["clip_ids"], "lengths": batch["lengths"].tolist(),
            "true_frames": int(batch["lengths"].sum()), "dense_frames": int(len(lengths)*lengths.max()),
            "padding_frames": int(len(lengths)*lengths.max()-lengths.sum()), "loss": float(loss.detach()),
            "gradient_norm_before_clip": float(norm), "gradient_norm_after_clip": after_norm,
            "nonzero_gradient_tensors": nonzero, "parameter_tensors": len(parameters),
            "optimizer_updated": True, "learning_rate": lr, "seconds": time.perf_counter()-began,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved()}
    except Exception as exc:
        raise RuntimeError("Stage4 dense CTC update failed; do not hide/skips its loss") from exc


def write_checkpoint(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    """Write a new complete checkpoint using restricted-loader-compatible values."""
    require(path.resolve().is_relative_to(BASE) and not path.exists(), "Unsafe/existing checkpoint")
    with path.open("xb") as stream:
        torch.save(runner._pack_safe(payload), stream)
    return {"path": str(path.resolve()), "sha256": sha(path), "bytes": path.stat().st_size}


def load_checkpoint(reference: dict[str, Any]) -> dict[str, Any]:
    """Reject byte changes before a weights-only load of a trusted stage4 checkpoint."""
    path = Path(reference["path"]).resolve()
    require(path.is_relative_to(BASE) and sha(path) == reference["sha256"], "Checkpoint path/hash mismatch")
    return runner._unpack_safe(torch.load(path, map_location="cpu", weights_only=True))


def validate_payload(payload: dict[str, Any], identity: dict[str, Any], schedule: list[dict[str, Any]]) -> None:
    """Check trajectory identity, complete state and cursor/phase/LR before restoration."""
    require(payload["schema"] == "offline-stage4-complete-checkpoint-v1" and payload["identity"] == identity, "Wrong checkpoint identity")
    done = payload["completed_updates"]
    require(type(done) is int and 0 < done <= 6 and len(payload["history"]) == done, "Invalid checkpoint update/history")
    require(payload["state_hashes"] == hashes(payload["state"]), "Checkpoint state digest mismatch")
    for index, row in enumerate(payload["history"]):
        require(row["update"] == index+1 and row["clip_ids"] == schedule[index]["clip_ids"] and row["learning_rate"] == schedule[index]["lr"] and row["kind"] == schedule[index]["kind"] and row["optimizer_updated"] is True, "Checkpoint schedule/history differs")
    require(payload["next_schedule_row"] == (schedule[done] if done < 6 else None), "Resume cursor/LR/phase differs")
    require(all(group["lr"] == schedule[done-1]["lr"] for group in payload["state"]["optimizer"]["param_groups"]), "Optimizer LR differs")
    require(payload["state"]["optimizer"]["state"] and all(float(item["step"]) == done for item in payload["state"]["optimizer"]["state"].values()), "Adam update counters differ")
    require(payload["history"][-1]["state_hashes"] == payload["state_hashes"], "History final state differs")


def run_segment(plan_path: Path, policy_path: Path, seed: int, arm: str, output: Path,
                stop: int, resume: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run only a declared smoke trajectory; resume across the phase and LR boundaries."""
    began = time.perf_counter()
    plan = read(plan_path)
    verify_files(plan)
    runtime = setup()
    require(runtime == plan["runtime"] and seed in plan["seeds"] and arm in plan["arms"], "Runtime/trajectory changed")
    require(stop in (2, 6), "Smoke stop must be 2 or 6")
    require(output.resolve().is_relative_to(BASE) and not output.exists(), "Output must be a new stage4 directory")
    output.mkdir(parents=True)
    identity = {"plan_sha256": sha(plan_path), "policy_sha256": sha(policy_path), "seed": seed, "arm": arm, "runtime": runtime}
    schedule = plan["schedules"][arm]
    data = dataset(plan_path, policy_path)
    model, optimizer = objects(seed)
    initial = hashes(state(model, optimizer))
    saved = load_checkpoint(resume) if resume else None
    history, restore_record = [], None
    if saved:
        validate_payload(saved, identity, schedule)
        require(saved["completed_updates"] == 2 and stop == 6 and saved["initial_state_hashes"] == initial, "Only the fixed update2 checkpoint may resume")
        restore(saved["state"], model, optimizer)
        restored = hashes(state(model, optimizer))
        require(restored == saved["state_hashes"], "Restoration was not bitwise exact")
        restore_record = {"exact": True, "restored_hashes": restored, "next_update": 3, "next_schedule_row": schedule[2]}
        history = list(saved["history"])
    checkpoints, executed = [], []
    save(output / "segment_contract.json", {"identity": identity, "initial_state_hashes": initial,
        "start_update": len(history), "stop_update": stop, "resume": resume, "restore": restore_record})
    try:
        for index in range(len(history), stop):
            require(time.perf_counter()-began < plan["time_limit_seconds_per_segment"], "Smoke segment exceeded time budget")
            task = schedule[index]
            batch = make_batch(data, task["clip_ids"])
            before = runner.nested_hash(model.state_dict())
            metrics = update(model, optimizer, batch, task["lr"])
            current = state(model, optimizer)
            measured = hashes(current)
            require(before != measured["model"], "Optimizer did not change model values")
            row = {"update": index+1, "kind": task["kind"], **metrics, "state_hashes": measured}
            history.append(row)
            executed.append(row)
            with (output / "progress.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")
            if index+1 in (2, 6):
                payload = {"schema": "offline-stage4-complete-checkpoint-v1", "identity": identity,
                    "completed_updates": index+1, "history": history, "state": current, "state_hashes": measured,
                    "initial_state_hashes": initial, "next_schedule_row": schedule[index+1] if index+1 < 6 else None}
                reference = write_checkpoint(output / f"update_{index+1:04d}.pt", payload)
                validate_payload(load_checkpoint(reference), identity, schedule)
                checkpoints.append(reference)
            print(json.dumps({"seed": seed, "arm": arm, "segment": output.name, "update": index+1, "kind": task["kind"], "loss": row["loss"]}), flush=True)
        result = {"status": "completed", "identity": identity, "initial_state_hashes": initial,
            "executed_updates": executed, "completed_updates": stop, "restore": restore_record,
            "checkpoints": checkpoints, "final_checkpoint": checkpoints[-1],
            "elapsed_seconds": time.perf_counter()-began, "dev_neural_inputs": 0, "teacher_calls": 0, "audio_inputs": 0}
        save(output / "segment_result.json", result)
        return result
    except Exception as exc:
        save(output / "failure.json", {"error": repr(exc), "completed_updates": len(history), "executed_this_segment": len(executed), "last_saved": checkpoints[-1] if checkpoints else None})
        raise
    finally:
        del model, optimizer, data
        gc.collect()
        torch.cuda.empty_cache()
