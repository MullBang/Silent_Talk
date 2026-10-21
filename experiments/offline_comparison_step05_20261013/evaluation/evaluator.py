"""Fixed stage5 visual-only checkpoint diagnostics, with complete state isolation.

The caller owns training and checkpoint selection. This module evaluates exactly
the frozen 30 clips in three conditions and never drops a failed inference from
the metric denominator. Importing this module does not create or run a model.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import random
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
STAGE5 = ROOT / "experiments/offline_comparison_step05_20261013"
STAGE4 = ROOT / "experiments/offline_training_step04_20261013"
PLAN_PATH = STAGE4 / "protocol/evaluation_plan.json"
PLAN_SHA256 = "7ddf7a6d301f7cc909b8c8070d73c4ed7eae74dea60a877926d31cd70d277839"
CONTRACT_PATH = STAGE4 / "evaluation_contract.py"
CONTRACT_SHA256 = "654d83a6958622ebcec310bed5e97a9e5001a32bc4fbc1592a16dcefb7619dc3"
CUDA_RESERVED_BUDGET_BYTES = 6 * 1024 ** 3


class ArtifactWriteError(RuntimeError):
    """Artifact persistence failed, so the caller must stop rather than hide it."""


def require(condition: bool, message: str) -> None:
    """Reject a changed evaluation contract before inference."""
    if not condition:
        raise ValueError(message)


def file_sha256(path: Path) -> str:
    """Hash an artifact without loading any model or original media."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


require(file_sha256(CONTRACT_PATH) == CONTRACT_SHA256, "Frozen metric contract changed")
_spec = importlib.util.spec_from_file_location("offline_stage4_frozen_metric_contract", CONTRACT_PATH)
require(_spec is not None and _spec.loader is not None, "Metric contract import unavailable")
contract = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(contract)


def _json(path: Path, value: Any) -> dict:
    """Save a new UTF-8 JSON artifact, refusing to overwrite prior evidence."""
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return {"path": str(path.resolve()), "sha256": file_sha256(path), "bytes": path.stat().st_size}


def digest_state(value: Any) -> str:
    """Hash nested RNG/model values with explicit types, shapes and raw bytes."""
    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            array = item.detach().cpu().contiguous().numpy()
            digest.update(b"torch:")
            visit(array)
        elif isinstance(item, np.ndarray):
            require(not item.dtype.hasobject, "Object arrays cannot be state evidence")
            digest.update(b"numpy:" + item.dtype.str.encode() + repr(item.shape).encode() + b":")
            digest.update(np.ascontiguousarray(item).tobytes())
        elif isinstance(item, Mapping):
            digest.update(b"dict{")
            for key in sorted(item, key=lambda k: (type(k).__name__, repr(k))):
                visit(key)
                visit(item[key])
            digest.update(b"}")
        elif isinstance(item, (tuple, list)):
            digest.update(type(item).__name__.encode() + b"[")
            for child in item:
                visit(child)
            digest.update(b"]")
        elif item is None or isinstance(item, (str, int, float, bool)):
            encoded = repr(item).encode("utf-8")
            digest.update(type(item).__name__.encode() + b":" + str(len(encoded)).encode() + b":" + encoded)
        else:
            raise TypeError(f"Unsupported state evidence: {type(item).__name__}")

    visit(value)
    return digest.hexdigest()


def capture_rng() -> dict:
    """Copy Python, NumPy, torch CPU and all initialized CUDA generator states."""
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state().clone(),
            "torch_cuda": [s.clone() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []}


def restore_rng(states: dict) -> None:
    """Restore the caller's generators after diagnostics, including failures."""
    random.setstate(states["python"])
    np.random.set_state(states["numpy"])
    torch.set_rng_state(states["torch_cpu"])
    if states["torch_cuda"]:
        torch.cuda.set_rng_state_all(states["torch_cuda"])


def condition_input(pixels: np.ndarray, item: dict, condition: str) -> tuple[np.ndarray, np.ndarray]:
    """Materialize CTHW input using only the frozen condition's frame indices."""
    length = item["T"]
    require(type(length) is int and length > 0, "Invalid true frame count")
    require(pixels.dtype == np.float32 and pixels.shape == (3, length, 96, 96)
            and np.isfinite(pixels).all(), "Expected normalized float32 CTHW visual input")
    if condition == "normal":
        indices = np.arange(length, dtype=np.int64)
    elif condition == "static":
        require(item["static_frame_index"] == length // 2, "Static index must be T//2")
        indices = np.full(length, length // 2, dtype=np.int64)
    elif condition == "shuffled":
        order = item["shuffled_frame_indices"]
        require(len(order) == length and all(type(i) is int for i in order)
                and sorted(order) == list(range(length)), "Shuffled indices are not a full permutation")
        indices = np.asarray(order, dtype=np.int64)
    else:
        raise ValueError(f"Unknown condition: {condition}")
    return np.ascontiguousarray(pixels[:, indices], dtype=np.float32), indices


def summarize_rows(rows: Sequence[dict]) -> dict:
    """Aggregate every fixed split/kind/condition cohort, including failed rows."""
    require(len(rows) > 0, "Evaluation rows are empty")
    seen = [(r["clip_id"], r["condition"]) for r in rows]
    require(len(seen) == len(set(seen)), "Duplicate evaluation row identity")
    cohorts = {}
    for key in sorted({(r["split"], r["kind"], r["condition"]) for r in rows}):
        selected = [r["metrics"] for r in rows if (r["split"], r["kind"], r["condition"]) == key]
        cohorts["/".join(key)] = contract.corpus_scores(selected)
    return {"cohorts": cohorts, "all_rows": contract.corpus_scores([r["metrics"] for r in rows]),
            "pooled_all_rows_is_diagnostic_only": True,
            "primary_cohort_key": "dev/full_sentence/normal",
            "primary_is_only_final_update1260": True}


def validate_plan(evaluation_plan: dict, identity: dict) -> None:
    """Require the exact prior evaluation specification and a fixed checkpoint."""
    require(file_sha256(PLAN_PATH) == PLAN_SHA256, "Frozen evaluation plan changed")
    require(evaluation_plan == json.loads(PLAN_PATH.read_text(encoding="utf-8")), "Evaluation plan differs from frozen stage4")
    require(identity.get("seed") in (42, 43), "Unexpected comparison initialization")
    require(identity.get("arm") in ("A_sentence", "B_curriculum"), "Unexpected comparison arm")
    require(identity.get("completed_update") in (420, 840, 1260), "Checkpoint is outside the frozen evaluation schedule")
    require(len(evaluation_plan["items"]) == 30 and len({r["clip_id"] for r in evaluation_plan["items"]}) == 30,
            "Evaluation needs exactly 30 distinct clips")
    require(evaluation_plan["conditions"] == ["normal", "static", "shuffled"], "Condition order changed")


def _model_state(model: torch.nn.Module) -> dict:
    return {"model": digest_state(model.state_dict()),
            "gradients": digest_state({name: p.grad for name, p in model.named_parameters()}),
            "module_training_flags": {name: bool(m.training) for name, m in model.named_modules()}}


def _validate_loaded(loaded: dict, item: dict) -> None:
    require(loaded["clip_id"] == item["clip_id"] and loaded["split"] == item["split"]
            and loaded["kind"] == item["kind"] and loaded["parent_id"] == item["parent_id"], "Loaded clip identity differs")
    require(loaded["length"] == item["T"] and loaded["purpose"] == "validation", "Loaded true length/use differs")
    require(loaded["audit"]["source_training_ready"] is False and loaded["audit"]["source_provisional"] is True
            and loaded["audit"]["approval_changed"] is False, "Source provisional state was promoted")
    labels = loaded["labels"]
    require(labels.dtype == np.int64 and labels.ndim == 1 and labels.size > 0, "Invalid reference CTC labels")
    decoded = contract.compose_jamos((contract.IDX_TO_GRAPHEME[int(i)] for i in labels), strict=True)
    require(contract.metric_text(decoded) == contract.metric_text(item["text"]), "Reference label/text mismatch")


def evaluate_checkpoint(model: torch.nn.Module, datasets: dict, evaluation_plan: dict,
                        output_dir: Path, identity: dict) -> dict:
    """Evaluate a trained CUDA model without advancing model, gradients or RNG.

    Each dataset must authorize ``load_clip(clip_id, purpose='validation')`` for
    its split, including train diagnostics. Files are created only in a new
    stage5 output directory. Per-clip inference failures produce explicit empty
    predictions; state corruption or artifact-write failure aborts the caller.
    """
    validate_plan(evaluation_plan, identity)
    output_dir = Path(output_dir).resolve()
    require(output_dir.is_relative_to(STAGE5.resolve()) and output_dir != STAGE5.resolve(), "Output must remain inside stage5")
    require(not output_dir.exists(), "Refusing to overwrite an evaluation attempt")
    require(set(datasets) == {"train", "dev"}, "Both fixed dataset splits are required")
    device = next(model.parameters()).device
    require(device.type == "cuda", "Production evaluation requires the caller's CUDA model")
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "posteriors").mkdir()
    _json(output_dir / "attempt.json", {"schema": "offline-stage5-evaluation-attempt-v1", "identity": identity,
          "evaluation_plan": {"path": str(PLAN_PATH), "sha256": PLAN_SHA256},
          "metric_contract": {"path": str(CONTRACT_PATH), "sha256": CONTRACT_SHA256}})
    initial_rng = capture_rng()
    before = _model_state(model)
    before["rng"] = digest_state(initial_rng)
    torch.cuda.reset_peak_memory_stats(device)
    rows: list[dict] = []
    forward_calls = successful_frames = attempted_frames = 0
    started = time.perf_counter()
    fatal: dict | None = None
    model.eval()
    try:
        with torch.no_grad():
            for item in evaluation_plan["items"]:
                for condition in evaluation_plan["conditions"]:
                    visual = output = None
                    row = {"clip_id": item["clip_id"], "parent_id": item["parent_id"], "split": item["split"],
                           "kind": item["kind"], "condition": condition, "T": item["T"], "identity": identity,
                           "reference_text": item["text"], "forward_attempted": False}
                    try:
                        loaded = datasets[item["split"]].load_clip(item["clip_id"], purpose="validation")
                        _validate_loaded(loaded, item)
                        inputs, indices = condition_input(loaded["input"], item, condition)
                        visual = torch.from_numpy(inputs).unsqueeze(0).to(device=device, dtype=torch.float32)
                        lengths = torch.tensor([item["T"]], dtype=torch.int64, device="cpu")
                        row["forward_attempted"] = True
                        forward_calls += 1
                        attempted_frames += item["T"]
                        output = model(visual, lengths)
                        require(torch.cuda.max_memory_reserved(device) <= CUDA_RESERVED_BUDGET_BYTES,
                                "Evaluation CUDA reserved-memory budget exceeded (6 GiB)")
                        require(output.dtype == torch.float32 and tuple(output.shape) == (item["T"], 1, 69), "Unexpected model log-posterior shape/dtype")
                        values = np.ascontiguousarray(output.detach().cpu().numpy()[:, 0, :], dtype=np.float32)
                        require(np.isfinite(values).all(), "Nonfinite log probabilities")
                        require(np.allclose(np.exp(values.astype(np.float64)).sum(axis=1), 1.0, atol=1e-5, rtol=1e-5), "Output is not normalized log probabilities")
                        raw_ids = values.argmax(axis=1).astype(np.int64)
                        decoded = contract.decode_prediction(raw_ids)
                        row.update(decoded)
                        row["metrics"] = contract.score_prediction(item["text"], decoded["raw_text"])
                        row["status"] = "success"
                        row["error"] = None
                        row["input_audit"] = loaded["audit"]
                        row["input_array_sha256"] = digest_state(inputs)
                        row["frame_indices"] = indices.tolist()
                        row["reference_label_ids"] = loaded["labels"].tolist()
                        row["log_probs_array_sha256"] = digest_state(values)
                        posterior = output_dir / "posteriors" / f"{item['clip_id']}__{condition}.npz"
                        try:
                            with posterior.open("xb") as stream:
                                np.savez_compressed(stream, log_probs=values, raw_argmax_ids=raw_ids,
                                    collapsed_ids=np.asarray(decoded["collapsed_ids"], dtype=np.int64),
                                    frame_indices=indices, length=np.asarray(item["T"], dtype=np.int64),
                                    metadata_json=np.asarray(json.dumps({"schema": "offline-stage5-posterior-v1", "identity": identity,
                                        "clip_id": item["clip_id"], "condition": condition, "layout": "T,69", "audio_input": False},
                                        ensure_ascii=False, sort_keys=True)))
                            row["posterior"] = {"path": str(posterior), "sha256": file_sha256(posterior), "bytes": posterior.stat().st_size}
                        except OSError as exc:
                            raise ArtifactWriteError(f"Cannot persist posterior: {posterior}") from exc
                        successful_frames += item["T"]
                        del lengths, values, loaded, inputs
                    except ArtifactWriteError:
                        raise
                    except Exception as exc:
                        row.update({"status": "failed", "raw_argmax_ids": [], "collapsed_ids": [], "raw_text": "",
                                    "metric_text": "", "posterior": None,
                                    "error": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()},
                                    "metrics": contract.score_prediction(item["text"], "", failed=True)})
                    finally:
                        output = visual = None
                        if row.get("status") == "failed":
                            torch.cuda.empty_cache()
                    rows.append(row)
                    with (output_dir / "rows.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
                        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    except BaseException as exc:
        fatal = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        raise
    finally:
        rng_after_before_restore = digest_state(capture_rng())
        for name, module in model.named_modules():
            module.training = before["module_training_flags"][name]
        restore_rng(initial_rng)
        after = _model_state(model)
        after["rng"] = digest_state(capture_rng())
        isolation = {"before": before, "after": after, "rng_before_restore": rng_after_before_restore,
                     "model_unchanged": before["model"] == after["model"],
                     "gradients_unchanged": before["gradients"] == after["gradients"],
                     "rng_restored_exact": before["rng"] == after["rng"],
                     "module_training_flags_restored": before["module_training_flags"] == after["module_training_flags"],
                     "fatal": fatal, "rows_saved": len(rows)}
        isolation_ref = _json(output_dir / "state_isolation.json", isolation)
    require(all(isolation[key] for key in ("model_unchanged", "gradients_unchanged", "rng_restored_exact", "module_training_flags_restored")),
            "Evaluation changed caller state; training must stop")
    require(len(rows) == 90, "Incomplete fixed evaluation cohort")
    expected = {(i["clip_id"], c) for i in evaluation_plan["items"] for c in evaluation_plan["conditions"]}
    require({(r["clip_id"], r["condition"]) for r in rows} == expected, "Evaluation omitted a fixed row")
    scores = summarize_rows(rows)
    allocated_peak = torch.cuda.max_memory_allocated(device)
    reserved_peak = torch.cuda.max_memory_reserved(device)
    summary = {"schema": "offline-stage5-checkpoint-evaluation-v1", "identity": identity,
        "status": "completed" if scores["all_rows"]["failure_count"] == 0 else "completed_with_explicit_failures",
        "rows": rows, "metrics": scores, "count": len(rows), "failures": scores["all_rows"]["failure_count"],
        "forward_calls": forward_calls, "planned_frames": sum(i["T"] for i in evaluation_plan["items"]) * 3,
        "attempted_forward_frames": attempted_frames, "successful_posterior_frames": successful_frames,
        "posterior_count": sum(r["posterior"] is not None for r in rows),
        "peak_allocated_cuda_bytes": allocated_peak, "peak_reserved_cuda_bytes": reserved_peak,
        "cuda_reserved_budget_bytes": CUDA_RESERVED_BUDGET_BYTES,
        "cuda_reserved_budget_passed": reserved_peak <= CUDA_RESERVED_BUDGET_BYTES,
        "seconds": time.perf_counter() - started, "state_isolation": isolation_ref,
        "state_isolation_passed": True, "optimizer_updates": 0, "audio_input": False, "teacher_model_used": False,
        "evaluation_plan_sha256": PLAN_SHA256, "evaluation_contract_sha256": CONTRACT_SHA256,
        "primary_result": scores["cohorts"]["dev/full_sentence/normal"] if identity["completed_update"] == 1260 else None}
    summary_ref = _json(output_dir / "evaluation_summary.json", summary)
    return {"summary": summary_ref, "count": len(rows), "failures": summary["failures"],
            "forward_calls": forward_calls, "frames": attempted_frames, "seconds": summary["seconds"],
            "peak_allocated_cuda_bytes": allocated_peak, "peak_reserved_cuda_bytes": reserved_peak,
            "cuda_reserved_budget_passed": summary["cuda_reserved_budget_passed"],
            "metrics": scores, "primary_result": summary["primary_result"], "state_isolation": isolation_ref,
            "state_isolation_passed": True, "identity": identity}
