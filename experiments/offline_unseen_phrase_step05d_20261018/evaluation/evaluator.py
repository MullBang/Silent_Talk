"""Unseen-dev-phrase-only frozen-checkpoint diagnostics; no model is created by this module.

The stage5 numerical helpers are read-only. The old thirty-clip evaluator is
never invoked. A completed evaluation may be reused only after its identity,
source bindings, rows, posterior files and isolation evidence are verified.
"""
from __future__ import annotations

import importlib.util
import json
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
STAGE = ROOT / "experiments/offline_unseen_phrase_step05d_20261018"
OLD = ROOT / "experiments/offline_comparison_step05_20261013/evaluation/evaluator.py"
OLD_SHA = "1f75bcc10865d07c6d67c8806a9e082b5e23c4c3ca3c6c276694839aeaa29277"
PLAN_PATH = STAGE / "evaluation_plan.json"
EXECUTION_PATH = STAGE / "execution_plan.json"
POLICY_PATH = STAGE / "controlled_pilot_policy.json"
CHECKPOINTS = [1680]
CONDITIONS = {"1680": ["normal", "static", "shuffled"]}
HANDOFF_PATH = ROOT / "experiments/offline_phrase_only_step05c_20261018/results/next_step_handoff.json"
HANDOFF_SHA = "4d291591311d6e4bb7eba0efebf0f4163f9db20da7a39821b7ce8fdd8fd3cb6f"
CUDA_RESERVED_BUDGET_BYTES = 2 * 1024 ** 3
SECONDS_PER_MODEL = 120


def require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def file_sha256(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


require(file_sha256(OLD) == OLD_SHA, "Frozen stage5 numerical helpers changed")
_spec = importlib.util.spec_from_file_location("stage5d_readonly_stage5_evaluation_helpers", OLD)
helper = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(helper)
contract = helper.contract
digest_state = helper.digest_state
condition_input = helper.condition_input
capture_rng = helper.capture_rng
restore_rng = helper.restore_rng


def ref(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": file_sha256(path), "bytes": path.stat().st_size}


def write_json(path: Path, value: Any) -> dict:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return ref(path)


def validate_plan(plan: dict, identity: dict, *, verify_sources: bool = True) -> list[str]:
    """Reject widened data scope, changed temporal controls and unknown stages."""
    require(type(identity.get("seed")) is int and identity["seed"] in [42, 43], "Unexpected seed")
    require(type(identity.get("completed_update")) is int and identity["completed_update"] in CHECKPOINTS,
            "Unexpected checkpoint update")
    require(identity.get("mode") == "evaluation_only", "Evaluation-only identity required")
    require(file_sha256(HANDOFF_PATH) == HANDOFF_SHA, "Frozen stage5c handoff changed")
    handoff = json.loads(HANDOFF_PATH.read_text(encoding="utf-8"))
    expected_checkpoint = handoff["checkpoints"][[42, 43].index(identity["seed"])]
    binding = identity.get("checkpoint", {})
    require(Path(binding.get("path", "")).resolve() == Path(expected_checkpoint["path"]).resolve()
            and binding.get("sha256") == expected_checkpoint["sha256"]
            and binding.get("bytes") == expected_checkpoint["bytes"], "Only the seed-matched frozen final1680 checkpoint is allowed")
    old_plan = json.loads(helper.PLAN_PATH.read_text(encoding="utf-8"))
    require(file_sha256(helper.PLAN_PATH) == helper.PLAN_SHA256, "Frozen source evaluation plan changed")
    expected_items = [r for r in old_plan["items"] if r["split"] == "dev" and r["kind"] == "phrase"]
    require(len(expected_items) == 3 and plan.get("items") == expected_items,
            "Evaluation must contain exactly the fixed three dev phrases and frame controls")
    require(plan.get("checkpoints") == CHECKPOINTS and plan.get("conditions_by_update") == CONDITIONS,
            "Checkpoint conditions differ from the fixed frozen-model dev-phrase probe")
    require(plan.get("resource_caps") == {"cuda_reserved_bytes": CUDA_RESERVED_BUDGET_BYTES, "seconds_per_model": SECONDS_PER_MODEL},
            "Fixed evaluation resource caps changed")
    if verify_sources:
        require(plan == json.loads(PLAN_PATH.read_text(encoding="utf-8")), "Loaded evaluation plan changed")
        for path, key in [(EXECUTION_PATH, "plan_sha256"), (POLICY_PATH, "policy_sha256")]:
            require(identity.get(key) == file_sha256(path), f"Changed {key} binding")
        binding = identity.get("checkpoint", {})
        checkpoint = Path(binding.get("path", "")).resolve()
        require(checkpoint.suffix == ".pt", "Checkpoint must be a saved state")
        require(binding.get("sha256") == file_sha256(checkpoint), "Checkpoint digest differs")
        require(isinstance(identity.get("runtime"), dict) and bool(identity["runtime"]), "Runtime binding is missing")
    return CONDITIONS[str(identity["completed_update"])]


def summarize_rows(rows: list[dict]) -> dict:
    require(bool(rows), "Evaluation rows are empty")
    keys = [(r["clip_id"], r["condition"]) for r in rows]
    require(len(keys) == len(set(keys)), "Duplicate evaluation row")
    require(all(r["split"] == "dev" and r["kind"] == "phrase" for r in rows), "Non-phrase/dev evaluation row")
    return {
        "cohorts": {f"dev/phrase/{c}": contract.corpus_scores([r["metrics"] for r in rows if r["condition"] == c])
                    for c in sorted({r["condition"] for r in rows})},
        "all_rows": contract.corpus_scores([r["metrics"] for r in rows]),
        "pooled_all_rows_is_diagnostic_only": True,
        "primary_cohort_key": "dev/phrase/normal",
        "generalization_measured": False,
        "exploratory_reused_development_probe": True,
    }


def validate_loaded(loaded: dict, item: dict) -> None:
    require(loaded["clip_id"] == item["clip_id"] and loaded["split"] == "dev"
            and loaded["kind"] == "phrase" and loaded["parent_id"] == item["parent_id"], "Loaded dev phrase differs")
    require(loaded["length"] == item["T"] and loaded["purpose"] == "evaluation", "Loaded true length/use differs")
    require(loaded["audit"]["source_training_ready"] is False and loaded["audit"]["source_provisional"] is True
            and loaded["audit"]["approval_changed"] is False, "Source provisional state was promoted")
    labels = loaded["labels"]
    require(labels.dtype == np.int64 and labels.ndim == 1 and labels.size > 0, "Invalid reference CTC labels")
    decoded = contract.compose_jamos((contract.IDX_TO_GRAPHEME[int(i)] for i in labels), strict=True)
    require(contract.metric_text(decoded) == contract.metric_text(item["text"]), "Reference label/text mismatch")


def expected_pairs(plan: dict, conditions: list[str]) -> set[tuple[str, str]]:
    return {(r["clip_id"], c) for r in plan["items"] for c in conditions}


def return_value(summary: dict, path: Path, *, reused: bool) -> dict:
    return {
        "status": summary["status"], "summary": ref(path), "count": summary["count"],
        "failures": summary["failures"], "forward_calls": summary["forward_calls"],
        "new_forward_calls": 0 if reused else summary["forward_calls"],
        "frames": summary["attempted_forward_frames"], "seconds": summary["seconds"],
        "peak_allocated_cuda_bytes": summary["peak_allocated_cuda_bytes"],
        "peak_reserved_cuda_bytes": summary["peak_reserved_cuda_bytes"],
        "cuda_reserved_budget_passed": summary["cuda_reserved_budget_passed"],
        "metrics": summary["metrics"], "primary_result": summary["metrics"]["cohorts"]["dev/phrase/normal"],
        "state_isolation": summary["state_isolation"], "state_isolation_passed": summary["state_isolation_passed"],
        "identity": summary["identity"], "reused": reused,
    }


def reuse_completed(output_dir: Path, plan: dict, identity: dict, conditions: list[str]) -> dict:
    path = output_dir / "evaluation_summary.json"
    require(path.is_file(), "Incomplete existing evaluation: preserve it and stop for explicit recovery")
    summary = json.loads(path.read_text(encoding="utf-8"))
    require(summary["identity"] == identity and summary["evaluation_plan_sha256"] == file_sha256(PLAN_PATH),
            "Existing evaluation identity/plan differs")
    require(summary["status"] == "completed" and summary["failures"] == 0, "Existing evaluation has failures")
    rows = summary["rows"]
    require(len(rows) == len(expected_pairs(plan, conditions)) and
            {(r["clip_id"], r["condition"]) for r in rows} == expected_pairs(plan, conditions), "Existing cohort differs")
    require(summary["metrics"] == summarize_rows(rows), "Existing evaluation scores differ from rows")
    require(all(r["identity"] == identity for r in rows), "Saved row identity differs")
    for binding in [summary["state_isolation"], summary["rows_jsonl"], *[r["posterior"] for r in rows]]:
        artifact = Path(binding["path"]).resolve()
        require(artifact.is_relative_to(output_dir) and ref(artifact) == binding, "Existing evaluation artifact changed")
    jsonl = [json.loads(line) for line in Path(summary["rows_jsonl"]["path"]).read_text(encoding="utf-8").splitlines()]
    require(jsonl == rows, "Saved summary rows differ from incremental log")
    isolation = json.loads(Path(summary["state_isolation"]["path"]).read_text(encoding="utf-8"))
    require(isolation["fatal"] is None and all(isolation[k] for k in
            ["model_unchanged", "gradients_unchanged", "rng_restored_exact", "module_training_flags_restored"]),
            "Existing state isolation failed")
    return return_value(summary, path, reused=True)


def evaluate(model: torch.nn.Module, dataset: Any, identity: dict, output_dir: Path) -> dict:
    """Run fixed dev-phrase diagnostics; caller owns model/state/checkpoint choice."""
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    conditions = validate_plan(plan, identity)
    output_dir = Path(output_dir).resolve()
    require(output_dir.is_relative_to(STAGE.resolve()) and output_dir != STAGE.resolve(), "Output must stay inside stage5d")
    if output_dir.exists():
        return reuse_completed(output_dir, plan, identity, conditions)
    device = next(model.parameters()).device
    require(device.type == "cuda", "Production evaluation requires caller CUDA model")
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "posteriors").mkdir()
    write_json(output_dir / "attempt.json", {"schema": "offline-stage5d-evaluation-attempt-v1", "identity": identity,
               "evaluation_plan": ref(PLAN_PATH), "conditions": conditions, "input_split": "dev", "input_kind": "phrase"})
    initial_rng = capture_rng()
    before = helper._model_state(model)
    before["rng"] = digest_state(initial_rng)
    torch.cuda.reset_peak_memory_stats(device)
    rows = []
    forward_calls = attempted_frames = successful_frames = 0
    started = time.perf_counter()
    fatal = None
    model.eval()
    try:
        with torch.no_grad():
            for item in plan["items"]:
                for condition in conditions:
                    visual = output = None
                    row = {"clip_id": item["clip_id"], "parent_id": item["parent_id"], "split": "dev", "kind": "phrase",
                           "condition": condition, "T": item["T"], "identity": identity,
                           "reference_text": item["text"], "forward_attempted": False}
                    try:
                        require(time.perf_counter() - started <= SECONDS_PER_MODEL, "Evaluation wall-time budget exceeded")
                        loaded = dataset.get(item["clip_id"], purpose="evaluation")
                        validate_loaded(loaded, item)
                        inputs, indices = condition_input(loaded["input"], item, condition)
                        visual = torch.from_numpy(inputs).unsqueeze(0).to(device=device, dtype=torch.float32)
                        lengths = torch.tensor([item["T"]], dtype=torch.int64, device="cpu")
                        row["forward_attempted"] = True
                        forward_calls += 1
                        attempted_frames += item["T"]
                        output = model(visual, lengths)
                        require(time.perf_counter() - started <= SECONDS_PER_MODEL, "Evaluation wall-time budget exceeded")
                        require(torch.cuda.max_memory_reserved(device) <= CUDA_RESERVED_BUDGET_BYTES, "Evaluation CUDA budget exceeded")
                        require(output.dtype == torch.float32 and tuple(output.shape) == (item["T"], 1, 69), "Unexpected model output")
                        values = np.ascontiguousarray(output.detach().cpu().numpy()[:, 0, :], dtype=np.float32)
                        require(np.isfinite(values).all() and np.allclose(np.exp(values.astype(np.float64)).sum(axis=1), 1., atol=1e-5, rtol=1e-5),
                                "Output is not finite normalized log probabilities")
                        raw_ids = values.argmax(axis=1).astype(np.int64)
                        decoded = contract.decode_prediction(raw_ids)
                        row.update(decoded)
                        row.update({"status": "success", "error": None, "metrics": contract.score_prediction(item["text"], decoded["raw_text"]),
                                    "input_audit": loaded["audit"], "input_array_sha256": digest_state(inputs),
                                    "frame_indices": indices.tolist(), "reference_label_ids": loaded["labels"].tolist(),
                                    "log_probs_array_sha256": digest_state(values)})
                        posterior = output_dir / "posteriors" / f"{item['clip_id']}__{condition}.npz"
                        try:
                            with posterior.open("xb") as stream:
                                np.savez_compressed(stream, log_probs=values, raw_argmax_ids=raw_ids,
                                    collapsed_ids=np.asarray(decoded["collapsed_ids"], dtype=np.int64), frame_indices=indices,
                                    length=np.asarray(item["T"], dtype=np.int64), metadata_json=np.asarray(json.dumps({
                                        "schema": "offline-stage5d-posterior-v1", "identity": identity, "clip_id": item["clip_id"],
                                        "condition": condition, "layout": "T,69", "audio_input": False}, ensure_ascii=False, sort_keys=True)))
                            row["posterior"] = ref(posterior)
                        except OSError as exc:
                            raise helper.ArtifactWriteError(f"Could not persist posterior: {posterior}") from exc
                        successful_frames += item["T"]
                        del lengths, values, loaded, inputs
                    except helper.ArtifactWriteError:
                        raise
                    except Exception as exc:
                        row.update({"status": "failed", "raw_argmax_ids": [], "collapsed_ids": [], "raw_text": "", "metric_text": "",
                                    "posterior": None, "error": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()},
                                    "metrics": contract.score_prediction(item["text"], "", failed=True)})
                    finally:
                        output = visual = None
                        if row.get("status") == "failed":
                            torch.cuda.empty_cache()
                    rows.append(row)
                    with (output_dir / "rows.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
                        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    require(row["status"] == "success", "Evaluation failed; fixed row is preserved and execution must stop")
    except BaseException as exc:
        fatal = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        raise
    finally:
        rng_unrestored = digest_state(capture_rng())
        for name, module in model.named_modules():
            module.training = before["module_training_flags"][name]
        restore_rng(initial_rng)
        after = helper._model_state(model)
        after["rng"] = digest_state(capture_rng())
        isolation = {"before": before, "after": after, "rng_before_restore": rng_unrestored,
                     "model_unchanged": before["model"] == after["model"], "gradients_unchanged": before["gradients"] == after["gradients"],
                     "rng_restored_exact": before["rng"] == after["rng"],
                     "module_training_flags_restored": before["module_training_flags"] == after["module_training_flags"],
                     "fatal": fatal, "rows_saved": len(rows)}
        isolation_ref = write_json(output_dir / "state_isolation.json", isolation)
    require(all(isolation[key] for key in ["model_unchanged", "gradients_unchanged", "rng_restored_exact", "module_training_flags_restored"]),
            "Evaluation changed caller state")
    require(len(rows) == len(expected_pairs(plan, conditions)) and
            {(r["clip_id"], r["condition"]) for r in rows} == expected_pairs(plan, conditions), "Incomplete evaluation")
    scores = summarize_rows(rows)
    require(time.perf_counter() - started <= SECONDS_PER_MODEL, "Evaluation final wall-time budget exceeded")
    summary = {"schema": "offline-stage5d-checkpoint-evaluation-v1", "identity": identity, "status": "completed", "rows": rows,
               "metrics": scores, "count": len(rows), "failures": scores["all_rows"]["failure_count"], "forward_calls": forward_calls,
               "planned_frames": sum(i["T"] for i in plan["items"]) * len(conditions), "attempted_forward_frames": attempted_frames,
               "successful_posterior_frames": successful_frames, "posterior_count": len(rows),
               "peak_allocated_cuda_bytes": torch.cuda.max_memory_allocated(device), "peak_reserved_cuda_bytes": torch.cuda.max_memory_reserved(device),
               "cuda_reserved_budget_bytes": CUDA_RESERVED_BUDGET_BYTES,
               "wall_seconds_budget": SECONDS_PER_MODEL,
               "cuda_reserved_budget_passed": torch.cuda.max_memory_reserved(device) <= CUDA_RESERVED_BUDGET_BYTES,
               "seconds": time.perf_counter() - started, "state_isolation": isolation_ref, "state_isolation_passed": True,
               "rows_jsonl": ref(output_dir / "rows.jsonl"), "evaluation_plan_sha256": file_sha256(PLAN_PATH),
               "evaluation_contract_sha256": helper.CONTRACT_SHA256, "optimizer_updates": 0, "audio_input": False,
               "teacher_model_used": False, "dev_evaluated": True, "generalization_measured": False, "exploratory_reused_development_probe": True}
    path = output_dir / "evaluation_summary.json"
    write_json(path, summary)
    return return_value(summary, path, reused=False)
