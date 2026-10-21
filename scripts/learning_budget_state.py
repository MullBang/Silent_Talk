"""Exact training-state resume and stage12 prefix checks for stage13.

These files are deliberately separate from inference checkpoints. Save only at
an optimizer-attempt boundary, after GradScaler.update(), before the next
zero_grad(). Gradients need not be saved because the next attempt clears them.
No project data, checkpoints, or CUDA training are opened on module import.
"""
from __future__ import annotations

import math
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch

SCHEMA = "lipnet-learning-budget-resume-v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _plain(value: Any) -> Any:
    """Clone only weights-only-loadable values; never retain live tensor aliases."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if type(value) in (str, int, float, bool, type(None)):
        if type(value) is float:
            _require(math.isfinite(value), "Nonfinite scalar in resume metadata")
        return value
    if isinstance(value, dict):
        _require(all(type(k) in (str, int, float, bool, type(None)) for k in value),
                 "Resume dictionary keys must be plain values")
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        items = [_plain(v) for v in value]
        return tuple(items) if isinstance(value, tuple) else items
    raise TypeError(f"Unsupported resume value: {type(value).__name__}")


def capture_rng_state() -> dict:
    """Serialize NumPy's uint32 MT state as a tensor, with no pickle globals."""
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {"bit_generator": numpy_state[0],
                  "keys": torch.from_numpy(numpy_state[1].astype(np.int64)),
                  "position": int(numpy_state[2]), "has_gauss": int(numpy_state[3]),
                  "cached_gaussian": float(numpy_state[4])},
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [state.clone().cpu() for state in torch.cuda.get_rng_state_all()]
                      if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict) -> None:
    """Restore all recorded generators; reject a different CUDA device count."""
    cuda_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    _require(len(state["torch_cuda"]) == cuda_count, "CUDA RNG device-count mismatch")
    numpy_state = state["numpy"]
    keys = numpy_state["keys"].cpu().numpy()
    _require(keys.dtype == np.int64 and bool(((keys >= 0) & (keys <= 2**32-1)).all()),
             "Invalid NumPy RNG keys")
    random.setstate(state["python"])
    np.random.set_state((numpy_state["bit_generator"], keys.astype(np.uint32),
                         numpy_state["position"], numpy_state["has_gauss"],
                         numpy_state["cached_gaussian"]))
    torch.set_rng_state(state["torch_cpu"].cpu())
    if cuda_count:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda"]])


def _backend_state() -> dict:
    return {"torch_threads": torch.get_num_threads(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled()}


def _restore_backend_state(state: dict) -> None:
    if torch.get_num_threads() != state["torch_threads"]:
        torch.set_num_threads(state["torch_threads"])
    torch.backends.cudnn.benchmark = state["cudnn_benchmark"]
    torch.backends.cudnn.deterministic = state["cudnn_deterministic"]
    torch.backends.cudnn.allow_tf32 = state["cudnn_allow_tf32"]
    torch.backends.cuda.matmul.allow_tf32 = state["matmul_allow_tf32"]
    torch.use_deterministic_algorithms(state["deterministic_algorithms"],
                                     warn_only=state["deterministic_warn_only"])


def _class_name(value: Any) -> str:
    return f"{type(value).__module__}.{type(value).__qualname__}"


def _parameter_signature(model: torch.nn.Module) -> list[dict]:
    return [{"name": name, "shape": list(parameter.shape), "dtype": str(parameter.dtype),
             "requires_grad": parameter.requires_grad}
            for name, parameter in model.named_parameters()]


def _optimizer_parameter_names(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> list[list[str]]:
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    result, seen = [], set()
    for group in optimizer.param_groups:
        ids = [id(parameter) for parameter in group["params"]]
        _require(all(key in names for key in ids), "Optimizer contains a non-model parameter")
        _require(not seen.intersection(ids) and len(ids) == len(set(ids)),
                 "Optimizer contains duplicate parameters")
        seen.update(ids)
        result.append([names[key] for key in ids])
    _require({id(p) for p in model.parameters() if p.requires_grad}.issubset(seen),
             "Optimizer is missing trainable parameters")
    return result


def save_training_state(path: str | Path, *, model: torch.nn.Module,
                        optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler,
                        attempt: int, schedule: dict, provenance: dict,
                        source_hashes: dict, settings: dict, counters: dict | None = None) -> dict:
    """Write a new, complete attempt-boundary state without changing any RNG.

    ``schedule`` may contain all batches or a hash-bound summary with
    ``attempt_updates``. ``attempt`` counts completed attempts, so the next
    batch is the experiment plan's ``batches[attempt]``. Hash/contract checks
    are supplied by the experiment runner when restoring.
    """
    _require(type(attempt) is int and attempt >= 0, "Nonnegative integer attempt required")
    _validate_schedule(schedule, attempt)
    value = _plain({"schema": SCHEMA, "inference_only": False, "deployment": False,
        "boundary": "after_scaler_update_before_next_zero_grad", "attempt": attempt,
        "model_class": _class_name(model), "optimizer_class": _class_name(optimizer),
        "scaler_class": _class_name(scaler), "scaler_enabled": scaler.is_enabled(),
        "parameter_signature": _parameter_signature(model),
        "optimizer_parameter_names": _optimizer_parameter_names(model, optimizer),
        "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(), "rng_state": capture_rng_state(),
        "model_training": model.training,
        "module_training": {name: module.training for name, module in model.named_modules()},
        "backend_state": _backend_state(), "schedule": schedule, "provenance": provenance,
        "source_hashes": source_hashes, "settings": settings, "counters": counters or {},
        "environment": {"torch": str(torch.__version__), "numpy": str(np.__version__),
                        "cuda": torch.version.cuda}})
    path = Path(path)
    with path.open("xb") as stream:
        torch.save(value, stream)
    return {"schema": SCHEMA, "path": str(path), "attempt": attempt,
            "inference_only": False, "deployment": False}


def load_training_state(path: str | Path, *, model: torch.nn.Module,
                        optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler,
                        expected_contract: dict | None = None) -> dict:
    """Validate, then restore weights, Adam, scaler, modes, backend flags and RNG.

    ``expected_contract`` contains complete values for selected top-level keys,
    for example schedule/provenance/settings/source_hashes/attempt. Every
    provided value must compare exactly; no tolerance or partial subtree match
    is applied. RNG is restored last, after newly constructed objects load.
    """
    value = torch.load(path, map_location="cpu", weights_only=True)
    _require(isinstance(value, dict) and value.get("schema") == SCHEMA, "Resume schema mismatch")
    _require(value.get("boundary") == "after_scaler_update_before_next_zero_grad",
             "Unsupported resume boundary")
    for name, expected in (expected_contract or {}).items():
        _require(name in value and compare_nested(expected, value[name])["exact"],
                 f"Resume contract mismatch: {name}")
    for name, actual in (("model_class", _class_name(model)),
                         ("optimizer_class", _class_name(optimizer)),
                         ("scaler_class", _class_name(scaler)),
                         ("scaler_enabled", scaler.is_enabled()),
                         ("parameter_signature", _parameter_signature(model)),
                         ("optimizer_parameter_names", _optimizer_parameter_names(model, optimizer))):
        _require(compare_nested(value[name], actual)["exact"], f"Resume topology mismatch: {name}")
    _require(set(value["module_training"]) == {name for name, _ in model.named_modules()},
             "Resume module topology mismatch")
    _require(value["model_training"] == value["module_training"][""], "Inconsistent model training mode")
    _validate_schedule(value["schedule"], value["attempt"])
    _require(len(value["rng_state"]["torch_cuda"]) ==
             (torch.cuda.device_count() if torch.cuda.is_available() else 0), "CUDA RNG device-count mismatch")
    current_state = model.state_dict()
    saved_state = value["model_state_dict"]
    _require(current_state.keys() == saved_state.keys(), "Resume model state keys mismatch")
    _require(all(current_state[key].shape == saved_state[key].shape and
                 current_state[key].dtype == saved_state[key].dtype for key in current_state),
             "Resume model state shape/dtype mismatch")
    model.load_state_dict(saved_state, strict=True)
    optimizer.load_state_dict(value["optimizer_state_dict"])
    scaler.load_state_dict(value["scaler_state_dict"])
    # Setting each flag preserves deliberately mixed train/eval submodule modes.
    for name, module in model.named_modules():
        module.training = value["module_training"][name]
    _restore_backend_state(value["backend_state"])
    restore_rng_state(value["rng_state"])
    return value


def _validate_schedule(schedule: dict, attempt: int) -> None:
    _require(type(attempt) is int and attempt >= 0, "Nonnegative integer attempt required")
    _require(isinstance(schedule, dict), "Schedule dictionary required")
    if "batches" in schedule:
        _require(isinstance(schedule["batches"], list), "Schedule batches must be a list")
        _require(attempt <= len(schedule["batches"]), "Attempt exceeds schedule length")
        _require(all(entry.get("attempt") == index for index, entry in
                     enumerate(schedule["batches"], start=1)), "Schedule attempts must be consecutive")
        if "attempt_updates" in schedule:
            _require(schedule["attempt_updates"] == len(schedule["batches"]), "Schedule length mismatch")
    else:
        _require(type(schedule.get("attempt_updates")) is int and
                 attempt <= schedule["attempt_updates"], "Attempt exceeds declared schedule length")
        _require(isinstance(schedule.get("sample_ids"), list) and bool(schedule["sample_ids"])
                 and len(set(schedule["sample_ids"])) == len(schedule["sample_ids"]),
                 "Summary schedule needs unique sample IDs")
        digest = schedule.get("batches_sha256")
        _require(isinstance(digest, str) and len(digest) == 64 and
                 all(character in "0123456789abcdef" for character in digest),
                 "Summary schedule needs SHA256 batch digest")


def compare_nested(expected: Any, actual: Any, *, excluded_keys: tuple[str, ...] = (),
                   detail_limit: int = 100) -> dict:
    """Compare all leaves with exact values; report every mismatch's count.

    Only the first ``detail_limit`` differing paths are stored, while maximum
    numerical differences and the total difference count cover the entire tree.
    Tensor equality is torch.equal plus identical shape/dtype. No tolerance is
    used. Nonfinite differences remain explicit and are never serialized as NaN.
    """
    _require(type(detail_limit) is int and detail_limit >= 0, "Nonnegative detail_limit required")
    report = {"exact": True, "compared_leaves": 0, "difference_count": 0,
              "max_abs_numeric_difference": 0.0, "nonfinite_difference_count": 0,
              "differences": [], "excluded_keys": list(excluded_keys)}

    def mismatch(path: str, kind: str, **details: Any) -> None:
        report["exact"] = False
        report["difference_count"] += 1
        if len(report["differences"]) < detail_limit:
            report["differences"].append({"path": path, "kind": kind, **details})

    def walk(left: Any, right: Any, path: str) -> None:
        if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
            report["compared_leaves"] += 1
            if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
                mismatch(path, "tensor_type")
                return
            left, right = left.detach().cpu(), right.detach().cpu()
            if left.shape != right.shape or left.dtype != right.dtype:
                mismatch(path, "tensor_shape_or_dtype", expected_shape=list(left.shape),
                         actual_shape=list(right.shape), expected_dtype=str(left.dtype), actual_dtype=str(right.dtype))
                return
            nonfinite = 0
            if left.is_floating_point() or left.is_complex():
                finite = torch.isfinite(left) & torch.isfinite(right)
                nonfinite = int((~finite).sum().item())
            else:
                finite = torch.ones_like(left, dtype=torch.bool)
            if nonfinite == 0 and torch.equal(left, right):
                return
            difference = (left.to(torch.complex128) - right.to(torch.complex128)).abs() if left.is_complex() else (
                left.to(torch.float64) - right.to(torch.float64)).abs()
            maximum = float(difference[finite].max().item()) if bool(finite.any()) else 0.0
            report["max_abs_numeric_difference"] = max(report["max_abs_numeric_difference"], maximum)
            report["nonfinite_difference_count"] += nonfinite
            mismatch(path, "tensor_values", unequal_elements=int((left != right).sum().item()),
                     finite_max_abs_difference=maximum, nonfinite_unequal_elements=nonfinite)
            return
        if isinstance(left, dict) and isinstance(right, dict):
            left_keys, right_keys = set(left) - set(excluded_keys), set(right) - set(excluded_keys)
            if left_keys != right_keys:
                mismatch(path, "dictionary_keys", missing=[str(k) for k in sorted(left_keys-right_keys,key=str)],
                         extra=[str(k) for k in sorted(right_keys-left_keys,key=str)])
            for key in sorted(left_keys & right_keys, key=lambda k:(type(k).__name__, str(k))):
                walk(left[key], right[key], f"{path}.{key}")
            return
        if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
            if type(left) is not type(right):
                mismatch(path, "sequence_type", expected_type=type(left).__name__, actual_type=type(right).__name__)
            if len(left) != len(right):
                mismatch(path, "sequence_length", expected=len(left), actual=len(right))
            for index, (a, b) in enumerate(zip(left, right)):
                walk(a, b, f"{path}[{index}]")
            return
        report["compared_leaves"] += 1
        if type(left) is not type(right):
            mismatch(path, "value_type", expected_type=type(left).__name__, actual_type=type(right).__name__)
        elif (type(left) is float and (not math.isfinite(left) or not math.isfinite(right))) or left != right:
            details = {"expected": left, "actual": right}
            if type(left) in (int, float):
                if math.isfinite(left) and math.isfinite(right):
                    maximum = float(abs(left-right))
                    report["max_abs_numeric_difference"] = max(report["max_abs_numeric_difference"], maximum)
                    details["abs_difference"] = maximum
                else:
                    report["nonfinite_difference_count"] += 1
                    details = {"expected": str(left), "actual": str(right)}
            mismatch(path, "value", **details)

    walk(expected, actual, "root")
    report["details_truncated"] = report["difference_count"] > len(report["differences"])
    return report


def compare_state_dicts(expected: dict, actual: dict) -> dict:
    """Check every saved state tensor, rather than a lossy aggregate/hash proxy."""
    _require(isinstance(expected, dict) and isinstance(actual, dict), "State dictionaries required")
    _require(all(isinstance(v, torch.Tensor) for v in [*expected.values(), *actual.values()]),
             "All state entries must be tensors")
    result = compare_nested(expected, actual)
    result.update(expected_tensor_count=len(expected), actual_tensor_count=len(actual),
                  matching_tensor_count=sum(key in actual and expected[key].dtype == actual[key].dtype
                    and expected[key].shape == actual[key].shape and bool(torch.isfinite(expected[key]).all())
                    and bool(torch.isfinite(actual[key]).all()) and torch.equal(expected[key].cpu(), actual[key].cpu())
                    for key in expected))
    result["tensors"] = {key: compare_nested(expected[key], actual[key], detail_limit=1)
                         for key in expected if key in actual}
    return result


def compare_telemetry(expected_rows: list[dict], actual_rows: list[dict], *,
                      excluded_keys: tuple[str, ...] = ("seconds",)) -> dict:
    """Compare every telemetry field and attempt, excluding wall-clock seconds only."""
    result = compare_nested(expected_rows, actual_rows, excluded_keys=excluded_keys)
    result.update(expected_rows=len(expected_rows), actual_rows=len(actual_rows))
    return result


def compare_schedule_prefix(expected_batches: list[dict] | dict,
                            actual_batches: list[dict] | dict, attempts: int) -> dict:
    """Verify a declared batch prefix, including attempt/epoch/order/IDs/indices."""
    _require(type(attempts) is int and attempts >= 0, "Nonnegative prefix length required")
    left = expected_batches["batches"] if isinstance(expected_batches, dict) else expected_batches
    right = actual_batches["batches"] if isinstance(actual_batches, dict) else actual_batches
    result = compare_nested(left[:attempts], right[:attempts])
    result.update(attempts=attempts, expected_available=len(left), actual_available=len(right),
                  prefix_complete=len(left) >= attempts and len(right) >= attempts)
    result["exact"] = result["exact"] and result["prefix_complete"]
    return result
