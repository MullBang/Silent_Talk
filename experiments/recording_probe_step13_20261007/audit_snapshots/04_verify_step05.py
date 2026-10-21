"""Read-only CPU verification of four final runs and all development metrics.

This auditor independently implements checkpoint digests, restricted NumPy-state
unpacking, schedule construction and edit distance. It imports no training,
evaluation, model or data-preparation implementation and reads no media drives.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import struct
import time
import unicodedata

import numpy as np
import torch

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
STEP4 = ROOT / "experiments/local_dataset_step04_20261007"
PARENT = ROOT / "experiments/feature_transfer_step07/data_plan.json"
PARENT_SHA = "c33dd5c2b77ed7298c497e58218d70a7552b476b947659f3accd33748af13b30"
HANDOFF_SHA = "3003209c04109a01f9737593b3645ffe7aef28e16530d84e53ba0caca77fca4b"
SEEDS = (42, 43)
ARMS = ("A_ctc", "B_feature")
CONDITIONS = ("normal", "static", "shuffled")
CHECKPOINT_SCHEMA = "local-paired-training-checkpoint-v1"
NUMPY_MARKER = "__step03_numpy_array_v1__"


def require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def same(actual, expected, message: str) -> None:
    require(actual == expected, message)


def local(path: str | Path) -> Path:
    result = Path(path).resolve()
    require(result.is_relative_to(ROOT), f"Only workspace artifacts allowed: {path}")
    return result


def sha(path: str | Path) -> str:
    with local(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def binding(path: str | Path) -> dict:
    return {"path": str(local(path)), "sha256": sha(path)}


def verify_binding(item: dict, expected: Path | None = None) -> Path:
    path = local(item["path"])
    if expected is not None:
        same(path, expected.resolve(), "Unexpected artifact path")
    same(sha(path), item["sha256"], f"Changed artifact: {path}")
    return path


def read_json(path: str | Path, expected_sha: str | None = None) -> dict:
    data = local(path).read_bytes()
    if expected_sha is not None:
        same(hashlib.sha256(data).hexdigest(), expected_sha, "Changed JSON artifact")
    def invalid(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")
    return json.loads(data, parse_constant=invalid)


def canonical_hash(plan: dict) -> str:
    value = {key: item for key, item in plan.items() if key != "plan_sha256"}
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def nested_hash(value) -> str:
    """Independent implementation of the frozen typed, length-prefixed format."""
    digest = hashlib.sha256()
    def add(value: bytes):
        digest.update(struct.pack("<Q", len(value)))
        digest.update(value)
    def visit(item):
        if isinstance(item, torch.Tensor):
            cpu = item.detach().cpu().contiguous()
            add(b"torch")
            add(str(cpu.dtype).encode())
            add(str(tuple(cpu.shape)).encode())
            add(cpu.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, np.ndarray):
            require(not item.dtype.hasobject, "Object arrays forbidden")
            add(b"numpy")
            add(item.dtype.str.encode())
            add(str(item.shape).encode())
            add(np.ascontiguousarray(item).tobytes())
        elif isinstance(item, dict):
            add(b"dict")
            for key in sorted(item, key=lambda x: (type(x).__name__, repr(x))):
                visit(key)
                visit(item[key])
            add(b"end_dict")
        elif isinstance(item, (list, tuple)):
            add(type(item).__name__.encode())
            for child in item:
                visit(child)
            add(b"end_sequence")
        elif item is None:
            add(b"none")
        elif isinstance(item, bool):
            add(b"bool")
            add(str(item).encode())
        elif isinstance(item, int):
            add(b"int")
            add(str(item).encode())
        elif isinstance(item, float):
            require(math.isfinite(item), "Nonfinite bound scalar")
            add(b"float")
            add(struct.pack("<d", item))
        elif isinstance(item, str):
            add(b"str")
            add(item.encode("utf-8"))
        else:
            raise ValueError(f"Unsupported hash object: {type(item)}")
    visit(value)
    return digest.hexdigest()


def unpack(value):
    if isinstance(value, dict) and value.get(NUMPY_MARKER) is True:
        same(set(value), {NUMPY_MARKER, "dtype", "shape", "bytes"}, "Malformed packed NumPy state")
        dtype, shape, data = np.dtype(value["dtype"]), value["shape"], value["bytes"]
        require(not dtype.hasobject and isinstance(shape, list)
                and all(type(n) is int and n >= 0 for n in shape), "Invalid packed dtype/shape")
        require(isinstance(data, torch.Tensor) and data.dtype == torch.uint8 and data.ndim == 1
                and data.numel() == math.prod(shape) * dtype.itemsize, "Invalid packed array bytes")
        return np.frombuffer(data.cpu().numpy().tobytes(), dtype=dtype).reshape(shape).copy()
    if isinstance(value, dict):
        return {key: unpack(item) for key, item in value.items()}
    if isinstance(value, list):
        return [unpack(item) for item in value]
    if isinstance(value, tuple):
        return tuple(unpack(item) for item in value)
    return value


def state_hashes(state: dict) -> dict:
    same(set(state), {"model", "adapter", "optimizer", "rng"}, "Incomplete full checkpoint state")
    return {**{key: nested_hash(state[key]) for key in state}, "complete": nested_hash(state)}


def tensor_hash(state: dict) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str((tensor.shape, str(tensor.dtype))).encode())
        digest.update(tensor.tobytes())
    return digest.hexdigest()


def finite_tree(value) -> None:
    if isinstance(value, torch.Tensor):
        require(torch.isfinite(value).all().item(), "Nonfinite checkpoint tensor")
    elif isinstance(value, np.ndarray):
        require(np.isfinite(value).all(), "Nonfinite checkpoint array")
    elif isinstance(value, dict):
        for item in value.values():
            finite_tree(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            finite_tree(item)
    elif isinstance(value, float):
        require(math.isfinite(value), "Nonfinite checkpoint scalar")


def load_checkpoint(info: dict) -> dict:
    path = verify_binding(info)
    same(path.stat().st_size, info["bytes"], "Checkpoint byte size")
    sidecar = path.with_suffix(".pt.json")
    same(local(info["sidecar"]), sidecar, "Checkpoint sidecar location")
    side = read_json(sidecar, info["sidecar_sha256"])
    same(side, {key: value for key, value in info.items() if key not in ("sidecar", "sidecar_sha256")}, "Sidecar/result exact agreement")
    with path.open("rb") as stream:
        payload = unpack(torch.load(stream, map_location="cpu", weights_only=True))
    same(payload.get("schema"), CHECKPOINT_SCHEMA, "Checkpoint schema")
    same(side.get("schema"), CHECKPOINT_SCHEMA, "Checkpoint sidecar schema")
    same(payload["completed_updates"], side["completed_updates"], "Checkpoint update position")
    same(nested_hash(payload["identity"]), side["identity_sha256"], "Checkpoint identity hash")
    finite_tree(payload["state"])
    actual = state_hashes(payload["state"])
    same(actual, payload["state_hashes"], "Checkpoint state hashes")
    same(actual, side["state_hashes"], "Sidecar state hashes")
    same(sha(path), info["sha256"], "Checkpoint changed during CPU load")
    return payload


def independent_schedule(ids: list[str], seed: int, updates: int) -> list[list[str]]:
    generator = random.Random(seed + 17000)
    stream = []
    while len(stream) < updates * 2:
        epoch = sorted(ids)
        generator.shuffle(epoch)
        stream.extend(epoch)
    return [stream[index:index + 2] for index in range(0, updates * 2, 2)]


def validate_history(history: list[dict], plan: dict, arm: str, rows: dict) -> dict:
    same(len(history), plan["updates"], "Missing or duplicated update history")
    for index, record in enumerate(history):
        same(record["update"], index + 1, "Noncontiguous optimizer updates")
        same(record["ids"], plan["schedule"][index], "Training order changed")
        same(record["frames"], [rows[sid]["frames"] for sid in record["ids"]], "Training frame count changed")
        require(record["optimizer_updated"] is True and record["student_changed"] is True, "Skipped or unchanged student update")
        for key in ("loss", "ctc_loss", "feature_loss", "gradient_norm_before_clipping", "gradient_norm_after_clipping", "seconds_optimizer_update"):
            require(isinstance(record[key], (int, float)) and math.isfinite(record[key]) and record[key] >= 0, f"Invalid training metric: {key}")
        require(abs(record["loss"] - record["ctc_loss"] - record["feature_loss"]) <= 1e-9, "Training loss arithmetic changed")
        same(record["lambda"], 0. if arm == "A_ctc" else 1., "Feature loss coefficient changed")
        if arm == "A_ctc":
            same(record["feature_loss"], 0., "CTC-only arm consumed feature loss")
            require(record["adapter_changed"] is False and record["adapter_gradients"]["parameter_tensors_with_gradient"] == 0,
                    "CTC-only arm updated adapter")
        else:
            require(record["adapter_changed"] is True
                    and record["adapter_gradients"]["parameter_tensors_with_nonzero_gradient"] > 0,
                    "Feature arm did not update adapter")
            require(record["feature_loss"] <= 4.00001, "Normalized feature distance exceeds mathematical bound")
        require(record["student_gradients"]["parameter_tensors_with_nonzero_gradient"] > 0, "No student gradient")
        norm = math.hypot(record["student_gradients"]["norm_after_clipping"], record["adapter_gradients"]["norm_after_clipping"])
        for group in ("student_gradients", "adapter_gradients"):
            require(math.isfinite(record[group]["norm_after_clipping"]) and record[group]["norm_after_clipping"] >= 0,
                    "Invalid gradient norm component")
        require(math.isfinite(norm) and norm <= 1.0001 and abs(norm - record["gradient_norm_after_clipping"]) <= 1e-12,
                "Clipping evidence differs")
        same(record["feature_hook_count_before"], 0, "Unexpected feature hooks")
        same(record["feature_hook_count_after"], 0, "Leaked feature hooks")
        require(0 < record["cuda_peak_allocated_bytes"] <= record["cuda_peak_reserved_bytes"], "GPU memory evidence differs")
        same(set(record["state_hashes"]), {"model", "adapter", "optimizer", "rng", "complete"}, "Missing per-update state component")
        for digest in [record["model_sha256"], record["adapter_sha256"], *record["state_hashes"].values()]:
            require(isinstance(digest, str) and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest), "Invalid state digest")
    return {"updates": len(history), "clip_presentations": 2 * len(history),
            "loss_first": history[0]["loss"], "loss_last": history[-1]["loss"],
            "ctc_loss_first": history[0]["ctc_loss"], "ctc_loss_last": history[-1]["ctc_loss"],
            "feature_loss_first": history[0]["feature_loss"], "feature_loss_last": history[-1]["feature_loss"],
            "seconds_optimizer_updates": sum(r["seconds_optimizer_update"] for r in history),
            "peak_allocated_bytes": max(r["cuda_peak_allocated_bytes"] for r in history),
            "peak_reserved_bytes": max(r["cuda_peak_reserved_bytes"] for r in history),
            "per_clip_presentations": dict(sorted(Counter(s for r in history for s in r["ids"]).items())),
            "all_updates_finite_and_applied": True}


def validate_optimizer(state: dict, arm: str, updates: int) -> None:
    optimizer = state["optimizer"]
    same(set(optimizer), {"state", "param_groups"}, "Unexpected optimizer state structure")
    same(len(optimizer["param_groups"]), 1, "Unexpected parameter grouping")
    group = optimizer["param_groups"][0]
    for key, expected in (("lr", .001), ("betas", (.9, .999)), ("eps", 1e-8), ("weight_decay", 0.),
                          ("amsgrad", False), ("maximize", False), ("capturable", False), ("differentiable", False),
                          ("foreach", None), ("fused", None), ("decoupled_weight_decay", False)):
        same(group[key], expected, f"Adam contract changed: {key}")
    parameters = list(state["model"].values()) + (list(state["adapter"].values()) if arm == "B_feature" else [])
    same(group["params"], list(range(len(parameters))), "Adam parameter IDs or arm differs")
    same(set(optimizer["state"]), set(group["params"]), "Missing Adam parameter states")
    for pid, parameter in enumerate(parameters):
        saved = optimizer["state"][pid]
        same(set(saved), {"step", "exp_avg", "exp_avg_sq"}, "Unexpected Adam buffers")
        require(isinstance(saved["step"], torch.Tensor) and saved["step"].numel() == 1
                and saved["step"].item() == updates, "Adam did not apply the full update budget")
        for key in ("exp_avg", "exp_avg_sq"):
            same(saved[key].shape, parameter.shape, "Adam parameter/buffer shape mismatch")
            same(saved[key].dtype, torch.float32, "Adam buffer precision changed")
            require(torch.isfinite(saved[key]).all().item(), "Nonfinite Adam buffer")
        require((saved["exp_avg_sq"] >= 0).all().item(), "Negative squared-gradient state")


def edit_distance(first, second) -> int:
    previous = list(range(len(second) + 1))
    for row, left in enumerate(first, 1):
        current = [row]
        for column, right in enumerate(second, 1):
            current.append(min(current[-1] + 1, previous[column] + 1,
                               previous[column - 1] + int(left != right)))
        previous = current
    return previous[-1]


def metric_counts(reference: str, prediction: str) -> dict:
    reference = unicodedata.normalize("NFC", reference).strip()
    prediction = unicodedata.normalize("NFC", prediction).strip()
    norm_ref, norm_pred = " ".join(reference.split()), " ".join(prediction.split())
    return {"character_edits": edit_distance(reference, prediction), "reference_characters": len(reference),
            "word_edits": edit_distance(reference.split(), prediction.split()), "reference_words": len(reference.split()),
            "exact": reference == prediction, "normalized_character_edits": edit_distance(norm_ref, norm_pred),
            "normalized_reference_characters": len(norm_ref), "normalized_exact": norm_ref == norm_pred}


def aggregate_metrics(records: list[dict]) -> dict:
    require(records and len({r["id"] for r in records}) == len(records), "Empty or duplicated evaluation cohort")
    counts = [metric_counts(r["reference"], r["prediction"]) for r in records]
    sums = {key: sum(item[key] for item in counts) for key in (
        "character_edits", "reference_characters", "word_edits", "reference_words",
        "normalized_character_edits", "normalized_reference_characters")}
    require(sums["reference_characters"] > 0 and sums["reference_words"] > 0, "Empty metric denominator")
    return {**sums, "clips": len(records), "failed_inputs": sum(r["status"] != "ok" for r in records),
            "exact_sentences": sum(r["exact"] for r in counts),
            "normalized_exact_sentences": sum(r["normalized_exact"] for r in counts),
            "cer_percent": 100 * sums["character_edits"] / sums["reference_characters"],
            "wer_percent": 100 * sums["word_edits"] / sums["reference_words"],
            "normalized_cer_percent": 100 * sums["normalized_character_edits"] / sums["normalized_reference_characters"]}


def temporal_order(frames: int, condition: str, sid: str) -> list[int]:
    if condition == "normal":
        return list(range(frames))
    if condition == "static":
        return [frames // 2] * frames
    if condition == "shuffled":
        seed = int.from_bytes(hashlib.sha256(("step07-order-v1:" + sid).encode()).digest()[:8], "little")
        return np.random.default_rng(seed).permutation(frames).tolist()
    raise ValueError("Unknown evaluation condition")


def validate_condition(result: dict, condition: str, dev_rows: list[dict]) -> dict:
    records = result["records"]
    same([r["id"] for r in records], [r["id"] for r in dev_rows], "Evaluation excludes, duplicates or reorders fixed cohort")
    for row, record in zip(dev_rows, records, strict=True):
        for key, expected in (("speaker_id", row["speaker_id"]), ("source_stem", row["source_stem"]),
                              ("split", "dev"), ("condition", condition), ("reference", row["text"])):
            same(record[key], expected, f"Evaluation identity/reference changed: {key}")
        require(isinstance(record["prediction"], str), "Prediction must remain text")
        require(record["status"] in ("ok", "failed"), "Unknown evaluation status")
        if record["status"] == "failed":
            require(record["prediction"] == "" and isinstance(record["error"], str) and record["error"],
                    "Failed evaluation did not retain empty-prediction denominator")
        else:
            same(record["error"], None, "Successful row unexpectedly contains failure")
            same(record["frames"], row["frames"], "Evaluation used different frame count")
            require(record["temporal"] is not None, "Missing temporal manipulation evidence")
        if record["temporal"] is not None:
            temporal = record["temporal"]
            positions = temporal_order(row["frames"], condition, row["id"])
            same(temporal["frames"], row["frames"], "Temporal evidence frame count")
            same(temporal["source_positions"], positions, "Evaluation temporal condition changed")
            same(temporal["source_positions_sha256"], nested_hash(positions), "Temporal evidence hash")
            same(temporal["static_position"], row["frames"] // 2 if condition == "static" else None, "Static frame selection")
        require(math.isfinite(record["seconds"]) and record["seconds"] >= 0, "Invalid inference time")
        same(record["metrics"], metric_counts(row["text"], record["prediction"]), "Per-record independent metrics differ")
    aggregate = aggregate_metrics(records)
    same(result["aggregate"], aggregate, "Pooled independent metrics differ")
    per_speaker = {speaker: aggregate_metrics([r for r in records if r["speaker_id"] == speaker])
                   for speaker in sorted({r["speaker_id"] for r in dev_rows})}
    same(result["per_speaker"], per_speaker, "Per-speaker independent metrics differ")
    return {"aggregate": aggregate, "per_speaker": per_speaker}


def compare_conditions(evaluations: dict) -> dict:
    result = {}
    for condition in CONDITIONS:
        per_seed = {}
        for seed in SEEDS:
            a = evaluations[(seed, "A_ctc")][condition]["aggregate"]
            b = evaluations[(seed, "B_feature")][condition]["aggregate"]
            per_seed[str(seed)] = {"A_cer_percent": a["cer_percent"], "B_cer_percent": b["cer_percent"],
                                   "B_minus_A_cer_percentage_points": b["cer_percent"] - a["cer_percent"],
                                   "A_exact_sentences": a["exact_sentences"], "B_exact_sentences": b["exact_sentences"],
                                   "B_minus_A_exact_sentences": b["exact_sentences"] - a["exact_sentences"]}
        result[condition] = {"per_seed": per_seed,
                            "mean_B_minus_A_cer_percentage_points": sum(r["B_minus_A_cer_percentage_points"] for r in per_seed.values()) / len(SEEDS),
                            "mean_A_cer_percent": sum(r["A_cer_percent"] for r in per_seed.values()) / len(SEEDS),
                            "mean_B_cer_percent": sum(r["B_cer_percent"] for r in per_seed.values()) / len(SEEDS)}
    return result


def validate_npz_binding_scope(bindings: list[dict], expected: list[dict]) -> None:
    """Check the selected payload references without rereading or decoding them."""
    actual = {}
    for ref in bindings:
        path = local(ref["path"])
        if path.suffix == ".npz":
            require(str(path) not in actual, "Duplicate payload binding")
            actual[str(path)] = ref["sha256"]
    wanted = {str(local(ref["path"])): ref["sha256"] for ref in expected}
    same(len(wanted), len(expected), "Duplicate expected payload binding")
    same(actual, wanted, "Payload bindings include another split, omit inputs or add teacher targets")


def validate_plan_file(seed: int, parent: dict, handoff: dict) -> tuple[dict, dict]:
    path = BASE / "plans" / f"seed{seed}.json"
    plan = read_json(path)
    for key, expected in (("schema", "lipnet-step05-full-comparison-plan-v1"), ("seed", seed),
                          ("updates", 3200), ("batch_size", 2), ("checkpoint_interval", 100), ("arms", list(ARMS)),
                          ("ids", handoff["train_ids"]), ("development_ids", handoff["development_ids"]),
                          ("full_data_plan", binding(STEP4 / "full_data_plan.json")),
                          ("plan_sha256", canonical_hash(plan)), ("audio_used", False), ("service_weights_updated", False)):
        same(plan[key], expected, "Frozen execution plan differs: " + key)
    expected_schedule = independent_schedule(handoff["train_ids"], seed, 3200)
    same(plan["schedule"], expected_schedule, "Independently reconstructed seeded schedule differs")
    schedule = next(item for item in handoff["training_schedules"] if item["seed"] == seed)
    same(plan["schedule"], schedule["schedule"], "Step4 scheduled order differs")
    same(set(plan["training_data"]["clips"]), set(handoff["train_ids"]), "Frozen training data cohort differs")
    same(nested_hash(plan["training_data"]["clips"]), plan["training_data"]["data_sha256"], "Frozen clip data digest differs")
    rows = {r["id"]: r for r in parent["records"]}
    for sid, item in plan["training_data"]["clips"].items():
        same(item["id"], sid, "Clip binding ID changed")
        same(item["frames"], rows[sid]["frames"], "Clip binding frame count changed")
        same(item["tokens_sha256"], nested_hash(rows[sid]["token_ids"]), "Clip binding transcript tokens changed")
    prepared = read_json(plan["input_manifest"]["path"], plan["input_manifest"]["sha256"])
    features = read_json(plan["feature_manifest"]["path"], plan["feature_manifest"]["sha256"])
    wanted_payloads = [{"path": r["npz_path"], "sha256": r["npz_sha256"]} for r in prepared["records"] if r["split"] == "train"]
    wanted_payloads += [r["feature_file"] for r in features["records"]]
    validate_npz_binding_scope(plan["data_bindings"], wanted_payloads)
    # Evidence references are checked without reopening every input NPZ; Step4
    # independently checked them, and the frozen loader checks them before each run.
    for key in ("comparison_recipe", "step03_preflight_plan", "step03_replay_audit", "parent_plan", "source_plan",
                "input_manifest", "feature_manifest", "teacher_plan"):
        verify_binding(plan[key])
    recipe = read_json(plan["comparison_recipe"]["path"], plan["comparison_recipe"]["sha256"])
    for key in ("optimizer", "loss", "primary_checkpoint", "evaluation", "precision"):
        same(plan[key], recipe[key], "Frozen Step3 recipe changed: " + key)
    expected_code_paths = [BASE / name for name in ("train_full.py", "test_train_full.py", "full_artifacts.py",
                                                   "test_full_artifacts.py", "evaluate.py", "test_evaluate.py")]
    expected_code_paths += [ROOT / name for name in (
        "experiments/local_training_step03_20261007/runner.py", "experiments/feature_transfer_step07/train_compare.py",
        "backend/models/baseline.py", "backend/models/__init__.py", "backend/models/tokenizer.py",
        "experiments/olkavs_official/evaluation_metrics.py", "experiments/olkavs_official/diagnostic_utils.py")]
    same(plan["execution_code"], [binding(path) for path in expected_code_paths], "Execution implementation changed after freeze")
    require(plan["runtime"]["requested_device"] == "cuda:0" and plan["runtime"]["cublas_workspace_config"] == ":4096:8",
            "Training runtime/selected device differs")
    return plan, binding(path)


def checkpoint_info(path: Path) -> dict:
    sidecar = path.with_suffix(".pt.json")
    info = read_json(sidecar)
    same(local(info["path"]), path.resolve(), "Checkpoint sidecar path changed")
    return {**info, "sidecar": str(sidecar), "sidecar_sha256": sha(sidecar)}


def audit_checkpoint_chain(directory: Path, final: dict, plan: dict) -> list[dict]:
    """Verify checkpoint bytes and boundary state hashes, including resumed prefixes."""
    end, seen, boundaries = 3200, set(), {}
    identity, initial, history = final["identity"], final["initial_state_hashes"], final["history"]
    while True:
        directory = local(directory)
        require(directory.is_relative_to(BASE / "runs") and directory not in seen and len(seen) < 64,
                "Invalid or cyclic checkpoint recovery chain")
        seen.add(directory)
        contract = read_json(directory / "segment_contract.json")
        same(contract["identity"], identity, "Segment identity changed within recovery chain")
        same(contract["initial_state_hashes"], initial, "Fresh initialization changed across recovery")
        start = contract["start_update"]
        require(type(start) is int and 0 <= start < end <= contract["stop_update"] <= 3200, "Invalid segment bounds")
        wanted = list(range(((start // 100) + 1) * 100, end + 1, 100))
        if end not in wanted:
            wanted.append(end)
        for update in wanted:
            info = checkpoint_info(directory / f"update_{update:04d}.pt")
            verify_binding(info)
            same(local(info["path"]).stat().st_size, info["bytes"], "Boundary checkpoint byte count")
            same(info["completed_updates"], update, "Boundary checkpoint update")
            same(info["identity_sha256"], nested_hash(identity), "Boundary checkpoint identity")
            same(info["state_hashes"], history[update - 1]["state_hashes"], "Boundary checkpoint/history states differ")
            require(update not in boundaries, "Duplicate checkpoint boundary across selected chain")
            boundaries[update] = info
        if start == 0:
            same(contract["resume_checkpoint"], None, "Initial run unexpectedly resumed")
            same(contract["resume_checkpoint_sha256"], None, "Initial run checkpoint reference changed")
            break
        previous_path = local(contract["resume_checkpoint"])
        same(sha(previous_path), contract["resume_checkpoint_sha256"], "Resume checkpoint binding changed")
        prior = load_checkpoint(checkpoint_info(previous_path))
        same(prior["identity"], identity, "Resume checkpoint identity differs")
        same(prior["completed_updates"], start, "Resume checkpoint position differs")
        same(prior["history"], history[:start], "Resumed trajectory does not preserve prior history")
        same(prior["initial_state_hashes"], initial, "Resume changed original initialization")
        directory, end = previous_path.parent, start
    require(set(range(100, 3201, 100)).issubset(boundaries), "Missing planned checkpoint boundaries")
    return [boundaries[key] for key in sorted(boundaries)]


def audit_run(result_ref: dict, seed: int, arm: str, plan: dict, plan_ref: dict, rows: dict) -> tuple[dict, dict]:
    path = verify_binding(result_ref)
    require(path.name == "run_result.json" and path.is_relative_to(BASE / "runs" / f"seed{seed}" / arm), "Run result path outside seeded arm")
    result = read_json(path, result_ref["sha256"])
    for key, expected in (("status", "completed"), ("seed", seed), ("arm", arm), ("completed_updates", 3200),
                          ("plan_file", plan_ref), ("plan_sha256", plan["plan_sha256"]),
                          ("full_data_plan", plan["full_data_plan"]), ("dev_inputs_loaded", False), ("audio_used", False),
                          ("teacher_model_loaded", False), ("service_weights_updated", False)):
        same(result[key], expected, "Run completion contract: " + key)
    segment = read_json(path.parent / "segment_result.json")
    for key, value in segment.items():
        same(result[key], value, "Augmented run result changed engine evidence: " + key)
    payload = load_checkpoint(result["checkpoint"])
    identity = payload["identity"]
    expected_core_paths = [ROOT / name for name in (
        "experiments/local_training_step03_20261007/runner.py", "experiments/feature_transfer_step07/train_compare.py",
        "backend/models/baseline.py", "backend/models/__init__.py", "backend/models/tokenizer.py")]
    for key, expected in (("seed", seed), ("arm", arm), ("total_updates", 3200), ("batch_size", 2),
                          ("plan_sha256", plan["plan_sha256"]), ("schedule_sha256", nested_hash(plan["schedule"])),
                          ("data_sha256", plan["training_data"]["data_sha256"]), ("clip_bindings", plan["training_data"]["clips"]),
                          ("code_bindings", [binding(p) for p in expected_core_paths]), ("runtime", plan["runtime"])):
        same(identity[key], expected, "Checkpoint identity differs: " + key)
    same(identity["optimization"], {"optimizer": "Adam", "lr": .001, "betas": [.9, .999], "eps": 1e-8,
         "weight_decay": 0., "gradient_clip": 1., "accumulation_clips": 2, "lambda": 0. if arm == "A_ctc" else 1.,
         "ctc_device": "cpu", "precision": "float32"}, "Actual optimizer recipe differs")
    same(result["identity_sha256"], nested_hash(identity), "Run identity digest differs")
    same(result["runtime"], identity["runtime"], "Run runtime differs")
    same(payload["completed_updates"], 3200, "Final checkpoint is not final update3200")
    history = payload["history"]
    stats = validate_history(history, plan, arm, rows)
    same(result["updates"], history[result["start_update"]:], "Final segment is not the exact history suffix")
    same(result["initial_state_hashes"], payload["initial_state_hashes"], "Initial state evidence changed")
    same(result["final_state_hashes"], payload["state_hashes"], "Final result state differs")
    same(history[-1]["state_hashes"], payload["state_hashes"], "Final history state differs")
    same(history[-1]["model_sha256"], tensor_hash(payload["state"]["model"]), "Final model bytes differ from history")
    same(history[-1]["adapter_sha256"], tensor_hash(payload["state"]["adapter"]), "Final adapter bytes differ from history")
    model, adapter = payload["state"]["model"], payload["state"]["adapter"]
    same(len(model), 31, "Student state tensor count differs")
    same(sum(t.numel() for t in model.values()), 3287781, "Student parameter count differs")
    same(len(adapter), 2, "Feature adapter tensor count differs")
    same(sum(t.numel() for t in adapter.values()), 131584, "Adapter parameter count differs")
    require(all(t.dtype == torch.float32 for t in [*model.values(), *adapter.values()]), "Student/adapter precision differs")
    validate_optimizer(payload["state"], arm, 3200)
    require(payload["initial_state_hashes"]["model"] != payload["state_hashes"]["model"], "Student never changed from initialization")
    same(payload["initial_state_hashes"]["adapter"] == payload["state_hashes"]["adapter"], arm == "A_ctc", "Final adapter behavior disagrees with arm")
    contract = read_json(path.parent / "segment_contract.json")
    same(contract["start_update"], result["start_update"], "Segment start differs")
    same(contract["stop_update"], 3200, "Segment ends before budget")
    same(contract["resume_checkpoint"], result["restored_from"], "Resume source differs")
    if result["start_update"]:
        restore = result["restore_verification"]
        require(restore["exact"] is True, "Resume was not exact")
        same(restore["expected_hashes"], history[result["start_update"] - 1]["state_hashes"], "Resume expected state differs")
        same(restore["actual_hashes"], restore["expected_hashes"], "Resume actual state differs")
    else:
        same(result["restore_verification"], None, "Fresh run has unexpected restore state")
    boundaries = audit_checkpoint_chain(path.parent, payload, plan)
    expected_segment = [item for item in boundaries if item["completed_updates"] > result["start_update"]]
    same(result["checkpoints"], expected_segment, "Segment checkpoint inventory differs")
    same(result["checkpoint"], expected_segment[-1], "Selected checkpoint is not final boundary")
    require(math.isfinite(result["elapsed_seconds"]) and result["elapsed_seconds"] > 0, "Invalid run elapsed time")
    summary = {"seed": seed, "arm": arm, "run_result": result_ref, "plan": plan_ref,
               "checkpoint": result["checkpoint"], "identity_sha256": nested_hash(identity),
               "initial_state_hashes": payload["initial_state_hashes"], "final_state_hashes": payload["state_hashes"],
               "data_sha256": identity["data_sha256"], "schedule_sha256": identity["schedule_sha256"],
               "checkpoint_count": len(boundaries), "checkpoint_bytes": sum(i["bytes"] for i in boundaries),
               "elapsed_seconds_reported_segment": result["elapsed_seconds"], "recovered": result["start_update"] > 0,
               "student_parameters": 3287781, "adapter_parameters": 131584, **stats}
    return summary, result


def validate_paired_runs(runs: dict) -> dict:
    pairs = {}
    for seed in SEEDS:
        a, b = runs[(seed, "A_ctc")], runs[(seed, "B_feature")]
        for component in ("model", "adapter", "rng"):
            same(a["initial_state_hashes"][component], b["initial_state_hashes"][component], "Paired fresh initialization differs: " + component)
        require(a["initial_state_hashes"]["optimizer"] != b["initial_state_hashes"]["optimizer"], "Different optimizer parameter sets unexpectedly share identity")
        for key in ("data_sha256", "schedule_sha256", "per_clip_presentations", "updates", "clip_presentations"):
            same(a[key], b[key], "A/B data, order or update budget differs: " + key)
        pairs[str(seed)] = {"same_initial_model_adapter_rng": True, "same_data_and_schedule": True,
                            "updates_per_arm": 3200, "data_sha256": a["data_sha256"], "schedule_sha256": a["schedule_sha256"],
                            "final_rng_hashes_equal": a["final_state_hashes"]["rng"] == b["final_state_hashes"]["rng"],
                            "final_rng_interpretation": "Descriptive paired-dropout-consumption check; not a requirement that trained weights match."}
    require(runs[(42, "A_ctc")]["initial_state_hashes"]["model"] != runs[(43, "A_ctc")]["initial_state_hashes"]["model"], "Different seeds produced the same initial model")
    return pairs


def audit_evaluation(seed: int, arm: str, plan: dict, plan_ref: dict, run: dict, run_ref: dict,
                     dev_rows: list[dict]) -> tuple[dict, dict]:
    path = BASE / "results" / f"evaluation_seed{seed}_{arm}.json"
    evaluation = read_json(path)
    for key, expected in (("schema", "lipnet-step05-final-development-evaluation-v1"), ("status", "completed"),
                          ("seed", seed), ("arm", arm), ("completed_updates", 3200), ("record_count", 71),
                          ("ids", [r["id"] for r in dev_rows]), ("plan", plan_ref), ("run_result", run_ref),
                          ("checkpoint", run["checkpoint"]), ("plan_sha256", plan["plan_sha256"]),
                          ("full_data_plan", plan["full_data_plan"]), ("runtime", plan["runtime"]),
                          ("model_state_before", run["final_state_hashes"]["model"]),
                          ("model_state_after", run["final_state_hashes"]["model"]), ("model_state_unchanged", True),
                          ("audio_read", False), ("original_media_read", False), ("teacher_targets_read", False),
                          ("teacher_executed", False), ("optimizer_executed", False),
                          ("checkpoint_selection", "fixed final update3200")):
        same(evaluation[key], expected, "Final evaluation provenance differs: " + key)
    same(evaluation["evaluation_code"], [binding(BASE / "evaluate.py"), binding(BASE / "test_evaluate.py")], "Evaluation code binding differs")
    prepared = read_json(plan["input_manifest"]["path"], plan["input_manifest"]["sha256"])
    validate_npz_binding_scope(evaluation["data_bindings"], [
        {"path": r["npz_path"], "sha256": r["npz_sha256"]} for r in prepared["records"] if r["split"] == "dev"])
    same(set(evaluation["conditions"]), set(CONDITIONS), "Missing or extra evaluation conditions")
    metrics = {condition: validate_condition(evaluation["conditions"][condition], condition, dev_rows) for condition in CONDITIONS}
    return metrics, binding(path)


def audit(output: Path) -> dict:
    started = time.perf_counter()
    parent = read_json(PARENT, PARENT_SHA)
    handoff = read_json(STEP4 / "full_data_plan.json", HANDOFF_SHA)
    rows = {r["id"]: r for r in parent["records"]}
    dev = [r for r in parent["records"] if r["split"] == "dev"]
    same([r["id"] for r in dev], handoff["development_ids"], "Fixed dev cohort changed")
    completion_path = BASE / "results/training_complete.json"
    completion = read_json(completion_path)
    same(completion["status"], "completed", "Training incomplete")
    same(completion["optimizer_updates"], 12800, "Full optimizer budget incomplete")
    same(len(completion["run_results"]), 4, "Missing or duplicate final runs")
    refs = {}
    for ref in completion["run_results"]:
        run = read_json(verify_binding(ref), ref["sha256"])
        key = (run["seed"], run["arm"])
        require(key in {(seed, arm) for seed in SEEDS for arm in ARMS} and key not in refs, "Unexpected or duplicated final run")
        refs[key] = ref
    runs, evaluations, evaluation_refs = {}, {}, {}
    for seed in SEEDS:
        plan, plan_ref = validate_plan_file(seed, parent, handoff)
        for arm in ARMS:
            key = (seed, arm)
            runs[key], result = audit_run(refs[key], seed, arm, plan, plan_ref, rows)
            evaluations[key], evaluation_refs[key] = audit_evaluation(seed, arm, plan, plan_ref, result, refs[key], dev)
            print(f"[step05 independent audit] seed{seed}/{arm}: 3200 updates and 71x3 evaluation rows passed", flush=True)
    pairs = validate_paired_runs(runs)
    result = {"schema": "lipnet-step05-independent-final-audit-v1", "status": "passed",
              "created_utc": datetime.now(timezone.utc).isoformat(), "code": binding(Path(__file__)),
              "tests": binding(BASE / "test_verify_step05.py"), "training_complete": binding(completion_path),
              "full_data_plan": binding(STEP4 / "full_data_plan.json"), "parent_plan": binding(PARENT),
              "run_count": 4, "optimizer_updates": 12800, "clip_presentations": 25600,
              "development_sentences_per_condition": 71, "conditions_per_model": 3, "evaluation_rows": 852,
              "runs": [runs[(seed, arm)] for seed in SEEDS for arm in ARMS], "paired_checks": pairs,
              "evaluation_metrics": [{"seed": seed, "arm": arm, "artifact": evaluation_refs[(seed, arm)],
                                      "conditions": evaluations[(seed, arm)]} for seed in SEEDS for arm in ARMS],
              "B_minus_A_comparison": compare_conditions(evaluations),
              "checkpoint_count": sum(r["checkpoint_count"] for r in runs.values()),
              "checkpoint_bytes": sum(r["checkpoint_bytes"] for r in runs.values()),
              "elapsed_seconds": time.perf_counter() - started, "model_or_producer_code_imported": False,
              "inference_repeated": False, "original_media_read": False, "student_optimization_steps": 0,
              "limits": ["Saved checkpoint, history and metric evidence are independently verified; inference and training are not rerun.",
                         "Per-update intermediate tensors are represented by recorded digests; full state is independently loaded at final and recovery checkpoints.",
                         "This existing same-speaker development split is not an untouched final test and does not establish webcam or unseen-speaker performance."]}
    output = local(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def main() -> None:
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=BASE / "results/independent_audit.json")
    args = parser.parse_args()
    result = audit(args.output)
    print(json.dumps({"status": result["status"], "optimizer_updates": result["optimizer_updates"],
                      "evaluation_rows": result["evaluation_rows"], "output": str(args.output.resolve())}), flush=True)


if __name__ == "__main__":
    main()
