"""Bound, resumable fixed-schedule execution of the frozen A/B primitives.

This module does not select data, download anything, evaluate a development
set, or promote service weights. The caller must supply verified train clips
and an immutable plan. Checkpoints contain the complete runtime state and are
loaded with PyTorch's restricted weights-only loader.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import struct
import tempfile
import time
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
CORE_PATH = ROOT / "experiments/feature_transfer_step07/train_compare.py"
_SPEC = importlib.util.spec_from_file_location("step03_frozen_training_core", CORE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Cannot load frozen training primitives: {CORE_PATH}")
core = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(core)

SCHEMA = "local-paired-training-checkpoint-v1"
NUMPY_MARKER = "__step03_numpy_array_v1__"


def require(condition: bool, message: str) -> None:
    """Reject invalid evidence before an update is applied."""
    if not condition:
        raise ValueError(message)


def canonical_plan_hash(plan: dict) -> str:
    """Hash the JSON plan excluding its optional self-declared digest."""
    content = {key: value for key, value in plan.items() if key != "plan_sha256"}
    return hashlib.sha256(json.dumps(content, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def nested_hash(value: Any) -> str:
    """Hash tensor/RNG values exactly, including types, shapes and dictionary keys."""
    digest = hashlib.sha256()

    def feed(blob: bytes) -> None:
        digest.update(struct.pack("<Q", len(blob)))
        digest.update(blob)

    def visit(item: Any) -> None:
        if torch.is_tensor(item):
            array = item.detach().cpu().contiguous()
            feed(b"torch")
            feed(str(array.dtype).encode())
            feed(str(tuple(array.shape)).encode())
            feed(array.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, np.ndarray):
            require(not item.dtype.hasobject, "Object arrays cannot be checkpointed")
            feed(b"numpy")
            feed(item.dtype.str.encode())
            feed(str(item.shape).encode())
            feed(np.ascontiguousarray(item).tobytes())
        elif isinstance(item, dict):
            feed(b"dict")
            for key in sorted(item, key=lambda key: (type(key).__name__, repr(key))):
                visit(key)
                visit(item[key])
            feed(b"end_dict")
        elif isinstance(item, (list, tuple)):
            feed(type(item).__name__.encode())
            for child in item:
                visit(child)
            feed(b"end_sequence")
        elif item is None:
            feed(b"none")
        elif isinstance(item, bool):
            feed(b"bool")
            feed(str(item).encode())
        elif isinstance(item, int):
            feed(b"int")
            feed(str(item).encode())
        elif isinstance(item, float):
            require(math.isfinite(item), "Nonfinite scalar in bound state")
            feed(b"float")
            feed(struct.pack("<d", item))
        elif isinstance(item, str):
            feed(b"str")
            feed(item.encode("utf-8"))
        else:
            raise TypeError(f"Unsupported state type: {type(item).__name__}")

    visit(value)
    return digest.hexdigest()


def state_hashes(state: dict) -> dict:
    """Produce component hashes to distinguish restore from next-update replay."""
    return {**{key: nested_hash(state[key]) for key in ("model", "adapter", "optimizer", "rng")},
            "complete": nested_hash(state)}


def _pack_safe(value: Any) -> Any:
    """Represent NumPy RNG arrays using only restricted-loader-supported objects."""
    if isinstance(value, np.ndarray):
        require(not value.dtype.hasobject, "Object arrays cannot be checkpointed")
        array = np.ascontiguousarray(value)
        return {NUMPY_MARKER: True, "dtype": array.dtype.str, "shape": list(array.shape),
                "bytes": torch.from_numpy(np.frombuffer(array.tobytes(), dtype=np.uint8).copy())}
    if isinstance(value, dict):
        require(NUMPY_MARKER not in value, "Reserved checkpoint key")
        return {key: _pack_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_pack_safe(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_pack_safe(item) for item in value)
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise TypeError(f"Unsupported checkpoint value: {type(value).__name__}")


def _unpack_safe(value: Any) -> Any:
    """Restore NumPy arrays without importing pickle globals."""
    if isinstance(value, dict) and value.get(NUMPY_MARKER) is True:
        require(set(value) == {NUMPY_MARKER, "dtype", "shape", "bytes"}, "Malformed NumPy state")
        dtype = np.dtype(value["dtype"])
        require(not dtype.hasobject, "Object arrays cannot be restored")
        shape = value["shape"]
        blob = value["bytes"]
        require(isinstance(shape, list) and all(type(x) is int and x >= 0 for x in shape), "Malformed array shape")
        require(torch.is_tensor(blob) and blob.dtype == torch.uint8 and blob.ndim == 1, "Malformed array bytes")
        require(blob.numel() == math.prod(shape) * dtype.itemsize, "Array byte count mismatch")
        return np.frombuffer(blob.cpu().numpy().tobytes(), dtype=dtype).reshape(shape).copy()
    if isinstance(value, dict):
        return {key: _unpack_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_unpack_safe(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_unpack_safe(item) for item in value)
    return value


def _sha_file(path: Path) -> str:
    """Read the artifact's current SHA-256."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _atomic_bytes(path: Path, content: bytes) -> None:
    """Publish a complete immutable file; never overwrite an existing artifact."""
    require(not path.exists(), f"Refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix="." + path.name + ".", suffix=".tmp",
                                         dir=path.parent, delete=False) as stream:
            temp = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        # Hard-link publication is atomic and fails if another writer won.
        os.link(temp, path)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def _save_json(path: Path, value: dict) -> None:
    """Save readable finite JSON as a new immutable artifact."""
    _atomic_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))


def save_checkpoint(output: Path, payload: dict) -> dict:
    """Atomically publish a complete weights-only-compatible checkpoint and sidecar."""
    update = payload["completed_updates"]
    path = output / f"update_{update:04d}.pt"
    sidecar = path.with_suffix(".pt.json")
    require(not path.exists() and not sidecar.exists(), f"Checkpoint already exists: {path}")
    temp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix="." + path.name + ".", suffix=".tmp",
                                         dir=output, delete=False) as stream:
            temp = Path(stream.name)
            torch.save(_pack_safe(payload), stream)
            stream.flush()
            os.fsync(stream.fileno())
        checkpoint_sha = _sha_file(temp)
        info = {"schema": SCHEMA, "path": str(path.resolve()), "sha256": checkpoint_sha,
                "bytes": temp.stat().st_size, "completed_updates": update,
                "identity_sha256": nested_hash(payload["identity"]),
                "state_hashes": payload["state_hashes"]}
        os.link(temp, path)
        _save_json(sidecar, info)
        return {**info, "sidecar": str(sidecar.resolve()), "sidecar_sha256": _sha_file(sidecar)}
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def load_checkpoint(path: Path) -> dict:
    """Verify bytes and state evidence, then load with the restricted Torch loader.

    Checksums detect corruption and provenance mismatch; they are not signatures.
    This reader is for this experiment's locally generated checkpoints only.
    """
    path = Path(path).resolve()
    sidecar = path.with_suffix(".pt.json")
    info = json.loads(sidecar.read_text(encoding="utf-8"))
    require(info.get("schema") == SCHEMA, "Unknown checkpoint sidecar schema")
    require(path.stat().st_size == info["bytes"] and _sha_file(path) == info["sha256"], "Checkpoint SHA-256 mismatch")
    require(Path(info["path"]).resolve() == path, "Checkpoint location changed")
    with path.open("rb") as stream:
        payload = _unpack_safe(torch.load(stream, map_location="cpu", weights_only=True))
    require(payload.get("schema") == SCHEMA, "Unknown checkpoint schema")
    require(payload["completed_updates"] == info["completed_updates"], "Checkpoint position mismatch")
    require(nested_hash(payload["identity"]) == info["identity_sha256"], "Checkpoint identity mismatch")
    actual_hashes = state_hashes(payload["state"])
    require(actual_hashes == payload["state_hashes"] == info["state_hashes"], "Checkpoint state digest mismatch")
    require(_sha_file(path) == info["sha256"], "Checkpoint changed while loading")
    return payload


def _clip_binding(clip_id: str, clip: dict) -> dict:
    """Bind exact training pixels, token sequence and optional teacher target."""
    require(clip.get("id") == clip_id, f"Clip ID mismatch: {clip_id}")
    require(clip.get("split", "train") == "train", f"Non-training clip: {clip_id}")
    pixels, tokens = clip["pixels"], clip["tokens"]
    require(isinstance(pixels, np.ndarray) and pixels.dtype == np.uint8 and pixels.ndim == 4
            and pixels.shape[1:] == (96, 96, 3) and len(pixels) > 0, f"Invalid pixels: {clip_id}")
    require(isinstance(tokens, list) and tokens and all(type(t) is int and 1 <= t < 69 for t in tokens),
            f"Invalid CTC tokens: {clip_id}")
    require(len(pixels) >= core.ctc_min_input_len(tokens), f"CTC infeasible clip: {clip_id}")
    feature = clip.get("teacher_features")
    if feature is not None:
        require(isinstance(feature, np.ndarray) and feature.dtype == np.float32
                and feature.shape == (len(pixels), 512) and np.isfinite(feature).all()
                and (np.linalg.norm(feature, axis=-1) > 1e-8).all(), f"Invalid teacher target: {clip_id}")
    return {"id": clip_id, "frames": len(pixels), "pixels_sha256": nested_hash(pixels),
            "tokens_sha256": nested_hash(tokens), "teacher_sha256": nested_hash(feature)}


def validate_plan(plan: dict, clips: dict[str, dict], arm: str) -> dict:
    """Require the whole fixed budget and bind all in-memory data before training."""
    require(arm in core.ARMS, "Unknown arm")
    require(type(plan.get("seed")) is int and 0 <= plan["seed"] < 2**32, "Invalid seed")
    require(type(plan.get("updates")) is int and plan["updates"] > 0, "Invalid update budget")
    require(plan.get("batch_size") == 2, "This comparison requires two clips per update")
    require(type(plan.get("checkpoint_interval", 1)) is int and plan.get("checkpoint_interval", 1) > 0,
            "Invalid checkpoint interval")
    schedule = plan.get("schedule")
    require(isinstance(schedule, list) and len(schedule) == plan["updates"], "Schedule length mismatch")
    require(all(isinstance(batch, list) and len(batch) == plan["batch_size"]
                and all(isinstance(item, str) and item in clips for item in batch) for batch in schedule),
            "Unknown or malformed scheduled clip")
    scheduled_ids = set(item for batch in schedule for item in batch)
    require(set(clips) == scheduled_ids, "Input clips must equal the scheduled training scope")
    declared = plan.get("plan_sha256")
    plan_sha = canonical_plan_hash(plan)
    require(declared is None or declared == plan_sha, "Plan SHA-256 mismatch")
    bindings = {clip_id: _clip_binding(clip_id, clips[clip_id]) for clip_id in sorted(clips)}
    if arm == "B_feature":
        require(all(clips[key].get("teacher_features") is not None for key in clips), "Missing teacher target")
    return {"plan_sha256": plan_sha, "data_sha256": nested_hash(bindings), "clips": bindings}


def _code_bindings() -> list[dict]:
    """Bind experiment code and model/tokenizer definitions used by this process."""
    paths = [Path(__file__), CORE_PATH, ROOT / "backend/models/baseline.py",
             ROOT / "backend/models/__init__.py", ROOT / "backend/models/tokenizer.py"]
    return [{"path": str(path.resolve()), "sha256": _sha_file(path)} for path in paths]


def _runtime(device: torch.device) -> dict:
    """Capture execution settings that must match when resuming a checkpoint."""
    result = dict(core.runtime_environment())
    # torch.__version__ can be a TorchVersion subclass; keep JSON-safe primitives.
    result["torch"] = str(result["torch"])
    result.update({"requested_device": str(device), "threads": torch.get_num_threads(),
                   "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                   "cudnn_benchmark": torch.backends.cudnn.benchmark,
                   "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                   "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                   "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()})
    if device.type == "cuda":
        result["selected_gpu"] = torch.cuda.get_device_name(device)
        result["selected_capability"] = list(torch.cuda.get_device_capability(device))
    return result


def _gradient_summary(parameters: Any) -> dict:
    """Inspect retained post-clipping gradients without affecting optimizer state."""
    count, nonzero, norm_squared = 0, 0, 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach()
        require(torch.isfinite(grad).all().item(), "Nonfinite post-update gradient")
        count += 1
        nonzero += int(torch.count_nonzero(grad).item() > 0)
        norm_squared += float(grad.double().square().sum().cpu())
    return {"parameter_tensors_with_gradient": count, "parameter_tensors_with_nonzero_gradient": nonzero,
            "norm_after_clipping": math.sqrt(norm_squared)}


def run_segment(plan: dict, clips: dict[str, dict], arm: str, output: Path,
                stop_update: int | None = None, resume_checkpoint: Path | None = None,
                device: str = "cuda:0") -> dict:
    """Execute a fixed schedule prefix or resume it without changing the budget.

    The output directory must be absent or empty. Checkpoints are written at the
    fixed interval and at the segment end. A separate empty directory is used for replay, so the
    original trajectory is retained. Runtime restoration is checked byte-exactly;
    the caller separately assesses any numerical CUDA next-update differences.
    """
    data = validate_plan(plan, clips, arm)
    stop = plan["updates"] if stop_update is None else stop_update
    require(type(stop) is int and 1 <= stop <= plan["updates"], "Invalid stop update")
    selected_device = torch.device(device)
    require(selected_device.type in ("cpu", "cuda"), "Unsupported training device")
    require(selected_device.type != "cuda" or torch.cuda.is_available(), "CUDA is unavailable")
    core.setup_runtime()
    identity = {"plan_sha256": data["plan_sha256"], "seed": plan["seed"], "arm": arm,
                "total_updates": plan["updates"], "batch_size": plan["batch_size"],
                "schedule_sha256": nested_hash(plan["schedule"]), "data_sha256": data["data_sha256"],
                "clip_bindings": data["clips"], "code_bindings": _code_bindings(),
                "runtime": _runtime(selected_device),
                "optimization": {"optimizer": "Adam", "lr": .001, "betas": [.9, .999],
                                 "eps": 1e-8, "weight_decay": 0., "gradient_clip": 1.,
                                 "accumulation_clips": 2, "lambda": 0. if arm == "A_ctc" else 1.,
                                 "ctc_device": "cpu", "precision": "float32"}}
    saved = load_checkpoint(resume_checkpoint) if resume_checkpoint is not None else None
    if saved is not None:
        require(saved["identity"] == identity, "Resume identity mismatch (plan, arm, data, code, or runtime)")
        start = saved["completed_updates"]
        require(type(start) is int and 0 <= start < stop, "Resume must advance the saved schedule position")
        require(len(saved["history"]) == start, "Checkpoint history length mismatch")
        require(all(row["update"] == index + 1 and row["ids"] == plan["schedule"][index]
                    for index, row in enumerate(saved["history"])), "Checkpoint history/schedule mismatch")
    else:
        start = 0
    output = Path(output).resolve()
    require(not output.exists() or (output.is_dir() and not any(output.iterdir())),
            f"Output must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    model, adapter, optimizer = core.make_objects(plan["seed"], arm, selected_device)
    initial_hashes = state_hashes(core.runtime_state(model, adapter, optimizer))
    history: list[dict] = []
    restore_verification = None
    if saved is not None:
        require(initial_hashes == saved["initial_state_hashes"], "Fresh seeded initialization differs")
        core.restore_state(saved["state"], model, adapter, optimizer)
        restored_hashes = state_hashes(core.runtime_state(model, adapter, optimizer))
        require(restored_hashes == saved["state_hashes"], "Restore did not reproduce full state exactly")
        restore_verification = {"exact": True, "expected_hashes": saved["state_hashes"],
                                "actual_hashes": restored_hashes}
        history = list(saved["history"])
    _save_json(output / "segment_contract.json", {"schema": SCHEMA, "identity": identity,
               "start_update": start, "stop_update": stop, "initial_state_hashes": initial_hashes,
               "resume_checkpoint": str(Path(resume_checkpoint).resolve()) if resume_checkpoint else None,
               "resume_checkpoint_sha256": _sha_file(Path(resume_checkpoint)) if resume_checkpoint else None})
    checkpoints, current_updates = [], []
    for index in range(start, stop):
        batch = [clips[clip_id] for clip_id in plan["schedule"][index]]
        before_model = core.tensor_hash(model.state_dict())
        before_adapter = core.tensor_hash(adapter.state_dict())
        hook_count = len(model.projection[2]._forward_hooks)
        if selected_device.type == "cuda":
            torch.cuda.synchronize(selected_device)
            torch.cuda.reset_peak_memory_stats(selected_device)
        began = time.perf_counter()
        try:
            metrics = core.optimizer_update(model, adapter, optimizer, batch, arm, selected_device)
        finally:
            require(len(model.projection[2]._forward_hooks) == hook_count, "Feature hook leaked after update")
        if selected_device.type == "cuda":
            torch.cuda.synchronize(selected_device)
        seconds = time.perf_counter() - began
        # Snapshot before FP64 gradient telemetry creates its own CUDA temporaries.
        peak_allocated = torch.cuda.max_memory_allocated(selected_device) if selected_device.type == "cuda" else 0
        peak_reserved = torch.cuda.max_memory_reserved(selected_device) if selected_device.type == "cuda" else 0
        student_grad = _gradient_summary(model.parameters())
        adapter_grad = _gradient_summary(adapter.parameters())
        after_model = core.tensor_hash(model.state_dict())
        after_adapter = core.tensor_hash(adapter.state_dict())
        require(before_model != after_model and student_grad["parameter_tensors_with_nonzero_gradient"] > 0,
                "Student did not receive a nonzero gradient and update")
        require((before_adapter != after_adapter and adapter_grad["parameter_tensors_with_nonzero_gradient"] > 0)
                if arm == "B_feature" else (before_adapter == after_adapter and adapter_grad["parameter_tensors_with_gradient"] == 0),
                "Adapter update disagrees with arm")
        combined_norm = math.hypot(student_grad["norm_after_clipping"], adapter_grad["norm_after_clipping"])
        require(combined_norm <= 1.0001, "Post-clipping gradient norm exceeds contract")
        row = {**metrics, "update": index + 1, "seconds_optimizer_update": seconds,
               "frames": [len(clip["pixels"]) for clip in batch], "student_gradients": student_grad,
               "adapter_gradients": adapter_grad, "gradient_norm_after_clipping": combined_norm,
               "student_changed": before_model != after_model, "adapter_changed": before_adapter != after_adapter,
               "feature_hook_count_before": hook_count, "feature_hook_count_after": len(model.projection[2]._forward_hooks),
               "model_sha256": after_model, "adapter_sha256": after_adapter,
               "cuda_peak_allocated_bytes": peak_allocated,
               "cuda_peak_reserved_bytes": peak_reserved}
        state = core.runtime_state(model, adapter, optimizer)
        hashes = state_hashes(state)
        row["state_hashes"] = hashes
        history.append(row)
        current_updates.append(row)
        payload = {"schema": SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(),
                   "identity": identity, "completed_updates": index + 1,
                   "initial_state_hashes": initial_hashes, "state_hashes": hashes,
                   "state": state, "history": history}
        if (index + 1) % plan.get("checkpoint_interval", 1) == 0 or index + 1 == stop:
            checkpoints.append(save_checkpoint(output, payload))
        print(json.dumps({"arm": arm, "update": index + 1, "loss": row["loss"],
                          "seconds": seconds, "cuda_peak_allocated_bytes": row["cuda_peak_allocated_bytes"]}), flush=True)
    result = {"schema": "local-paired-training-segment-result-v1", "arm": arm,
              "plan_sha256": data["plan_sha256"], "identity_sha256": nested_hash(identity),
              "start_update": start, "completed_updates": stop, "initial_state_hashes": initial_hashes,
              "restored_from": str(Path(resume_checkpoint).resolve()) if resume_checkpoint else None,
              "restore_verification": restore_verification, "updates": current_updates,
              "final_state_hashes": history[-1]["state_hashes"], "checkpoint": checkpoints[-1],
              "checkpoints": checkpoints, "runtime": identity["runtime"]}
    _save_json(output / "segment_result.json", result)
    return result
