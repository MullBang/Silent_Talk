"""Extract fixed OLKAVS visual-front targets; never instantiate an ASR decoder.

The frontend processes each complete native-rate temporal window in one call.
Rows are gathered by exact original-frame IDs onto the student's 25-fps inputs.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import torch

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
OFFICIAL = ROOT / "experiments/olkavs_official"
UPSTREAM = OFFICIAL / "upstream"
EXPECTED_COMMIT = "af2feda3e3ac91ff3fef2454fcee662366ce9f4b"
EXPECTED_CHECKPOINT_SHA256 = "f41d742380af11b72f9cafe178cd59d0a9016c02f5a976145f5f7595d44326a5"
STATE_PREFIX = "encoder.front."


def sha256(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def binding(path: str | Path) -> dict:
    return {"path": str(Path(path).resolve()), "sha256": sha256(path)}


def verify_binding(item: dict) -> Path:
    path = Path(item["path"]).resolve()
    if sha256(path) != item["sha256"]:
        raise ValueError(f"Frozen file changed: {path}")
    return path


def canonical_sha(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def save_new_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def save_new_npz(path: Path, **arrays) -> dict:
    """Exclusive creation prevents silent replacement of cached targets."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    return binding(path)


def exact_gather_indices(teacher_indices: np.ndarray, student_indices: np.ndarray) -> np.ndarray:
    """Preserve repeated student frames; refuse interpolation or edge clamping."""
    for name, values in (("teacher", teacher_indices), ("student", student_indices)):
        if values.dtype != np.dtype("int64") or values.ndim != 1 or values.size == 0:
            raise ValueError(f"{name} source indices must be a nonempty int64 vector")
        if np.any(values < 0):
            raise ValueError(f"{name} source indices cannot be negative")
    if not np.all(np.diff(teacher_indices) == 1):
        raise ValueError("Teacher source indices must cover every frame in increasing order")
    if np.any(np.diff(student_indices) < 0):
        raise ValueError("Student source indices cannot go backwards")
    indices = student_indices - teacher_indices[0]
    if np.any(indices < 0) or np.any(indices >= teacher_indices.size):
        raise ValueError("A student source frame is outside the complete teacher window")
    if not np.array_equal(teacher_indices[indices], student_indices):
        raise ValueError("Teacher/student original-frame correspondence is not exact")
    return indices.astype(np.int64, copy=False)


def normalize_teacher(pixels: np.ndarray) -> tuple[torch.Tensor, dict]:
    """Published BGR whole-utterance mean/std normalization (sample std, ddof=1)."""
    if (pixels.dtype != np.uint8 or pixels.ndim != 4 or pixels.shape[0] < 1
            or pixels.shape[1:] != (96, 96, 3)):
        raise ValueError("Teacher requires nonempty T x 96 x 96 x 3 uint8 BGR pixels")
    # float() creates a separate allocation; the input bytes stay immutable.
    video = torch.from_numpy(pixels).float()
    mean = video.mean()
    video -= mean
    std = video.std(correction=1)
    if not torch.isfinite(std) or float(std) <= 0:
        raise ValueError("Teacher input has invalid or zero whole-utterance variance")
    video /= std
    return video.permute(3, 0, 1, 2).unsqueeze(0).contiguous(), {
        "channel_order": "BGR", "scope": "whole native temporal window, all channels and pixels",
        "std_correction": 1, "raw_mean": float(mean), "raw_std": float(std),
        "normalized_mean": float(video.mean()), "normalized_std": float(video.std(correction=1)),
        "dtype": "float32"}


def load_front_state(front: torch.nn.Module, state: Mapping) -> dict:
    """Strictly isolate the named visual frontend without loading other modules."""
    if (not isinstance(state, Mapping) or not state
            or any(not isinstance(k, str) or not isinstance(v, torch.Tensor) for k, v in state.items())):
        raise ValueError("Checkpoint must be a nonempty direct tensor state dictionary")
    selected = {key[len(STATE_PREFIX):]: value for key, value in state.items()
                if key.startswith(STATE_PREFIX)}
    expected = front.state_dict()
    missing = sorted(set(expected) - set(selected))
    unexpected = sorted(set(selected) - set(expected))
    mismatched = {key: {"actual": list(selected[key].shape), "expected": list(expected[key].shape)}
                  for key in set(expected) & set(selected) if selected[key].shape != expected[key].shape}
    wrong_dtype = {key: {"actual": str(selected[key].dtype), "expected": str(expected[key].dtype)}
                   for key in set(expected) & set(selected) if selected[key].dtype != expected[key].dtype}
    if missing or unexpected or mismatched or wrong_dtype:
        raise ValueError(json.dumps({"missing": missing, "unexpected": unexpected,
                                     "shape_mismatch": mismatched, "dtype_mismatch": wrong_dtype}))
    if any(value.is_floating_point() and not torch.isfinite(value).all() for value in selected.values()):
        raise ValueError("Visual frontend checkpoint contains nonfinite tensors")
    front.load_state_dict(selected, strict=True)
    front.eval().requires_grad_(False)
    return {"state_prefix": STATE_PREFIX, "strict": True, "state_entries": len(selected),
            "total_checkpoint_state_entries": len(state), "other_state_entries_loaded": 0,
            "parameters": sum(p.numel() for p in front.parameters()),
            "missing": [], "unexpected": [], "shape_mismatch": {}, "dtype_mismatch": {}}


def read_prepared_npz(path: Path, expected_sha: str, record: dict | None = None,
                      data_plan_sha: str | None = None) -> dict:
    if sha256(path) != expected_sha:
        raise ValueError(f"Prepared input changed: {path}")
    with np.load(path, allow_pickle=False) as data:
        required = {"teacher_bgr", "student_rgb", "teacher_source_indices", "student_source_indices"}
        if not required.issubset(data.files):
            raise ValueError(f"Prepared input lacks arrays: {sorted(required - set(data.files))}")
        arrays = {key: data[key] for key in required}
        if record is not None:
            expected_keys = required | {"student_teacher_rows", "label_tokens", "metadata_json"}
            if set(data.files) != expected_keys:
                raise ValueError("Prepared payload has unexpected or absent metadata arrays")
            metadata = json.loads(str(data["metadata_json"].item()))
            if (metadata["record"] != record or metadata["id"] != record["id"]
                    or metadata["plan_sha256"] != data_plan_sha):
                raise ValueError("Payload metadata differs from the frozen data-plan record")
            if (data["label_tokens"].dtype != np.int64
                    or data["label_tokens"].tolist() != record["token_ids"]):
                raise ValueError("Prepared original transcript tokens changed")
            arrays["student_teacher_rows"] = data["student_teacher_rows"]
            arrays["metadata"] = metadata
    teacher, student = arrays["teacher_bgr"], arrays["student_rgb"]
    for name, pixels in (("teacher", teacher), ("student", student)):
        if pixels.dtype != np.uint8 or pixels.ndim != 4 or pixels.shape[1:] != (96, 96, 3):
            raise ValueError(f"{name} video must be T x 96 x 96 x 3 uint8")
    gather = exact_gather_indices(arrays["teacher_source_indices"], arrays["student_source_indices"])
    if teacher.shape[0] != arrays["teacher_source_indices"].size or student.shape[0] != gather.size:
        raise ValueError("Pixel frame counts differ from original-frame index arrays")
    # The teacher uses margin=0; the student uses the legacy 20% margin.
    # Their pixel arrays must NOT be treated as identical geometry.
    if record is not None:
        if (arrays["student_teacher_rows"].dtype != np.int64
                or not np.array_equal(gather, arrays["student_teacher_rows"])):
            raise ValueError("Saved teacher rows differ from exact source-ID mapping")
        meta = arrays["metadata"]
        for key in ("teacher_bgr", "student_rgb"):
            if hashlib.sha256(arrays[key].tobytes()).hexdigest() != meta[key + "_sha256"]:
                raise ValueError(f"Prepared {key} pixel evidence differs")
        fps, start, end = record["video_fps"], record["start_time"], record["end_time"]
        wanted_student = np.array([int(round((start + k / 25) * fps))
                                   for k in range(int(round((end - start) * 25)))], dtype=np.int64)
        first = max(0, min(math.ceil(start * fps - 1e-9), int(wanted_student[0])) - 2)
        stop = min(meta["source_frame_count"], max(math.ceil(end * fps - 1e-9), int(wanted_student[-1]) + 1) + 2)
        wanted_teacher = np.arange(first, stop, dtype=np.int64)
        if (not np.array_equal(arrays["student_source_indices"], wanted_student)
                or not np.array_equal(arrays["teacher_source_indices"], wanted_teacher)
                or len(wanted_student) != record["frames"]):
            raise ValueError("Prepared source-frame arrays differ from frozen timing and real context")
        students, teachers = meta["student_trace"]["selected"], meta["teacher_trace"]["selected"]
        if len(students) != len(gather) or len(teachers) != len(wanted_teacher):
            raise ValueError("Preparation source traces have inconsistent lengths")
        for i, row in enumerate(students):
            mate = teachers[int(gather[i])]
            if (row["source_frame_index"] != int(wanted_student[i])
                    or mate["source_frame_index"] != row["source_frame_index"]
                    or mate["source_bgr_pixel_sha256"] != row["source_bgr_pixel_sha256"]):
                raise ValueError("Teacher/student inputs do not trace to identical original frame pixels")
    if sha256(path) != expected_sha:
        raise ValueError("Prepared input changed while being read")
    arrays["gather_indices"] = gather
    return arrays


def validate_front_output(output: torch.Tensor, native_frames: int, gather: np.ndarray) -> np.ndarray:
    if output.dtype != torch.float32 or tuple(output.shape) != (1, native_frames, 512):
        raise ValueError(f"Expected FP32 frontend output 1 x {native_frames} x 512; got {output.dtype} {tuple(output.shape)}")
    if not torch.isfinite(output).all():
        raise ValueError("Teacher frontend output contains nonfinite values")
    index = torch.from_numpy(gather).to(output.device)
    selected = output[0].index_select(0, index)
    if torch.any(torch.linalg.vector_norm(selected, dim=-1) <= 1e-8):
        raise ValueError("Teacher target contains a zero or near-zero feature row")
    return selected.cpu().numpy().copy()


def runtime_info() -> dict:
    return {"python": platform.python_version(), "executable": str(Path(sys.executable).resolve()),
            "numpy": np.__version__, "torch": torch.__version__, "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "gpu": torch.cuda.get_device_name(0),
            "cuda_device_count": torch.cuda.device_count(), "dtype": "float32",
            "batch_size": 1, "autocast": False, "num_threads": torch.get_num_threads(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32}


def configure_runtime() -> None:
    torch.set_num_threads(4)
    torch.manual_seed(2207)
    np.random.seed(2207)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("The frozen extraction contract requires CUDA")


def state_sha256(front: torch.nn.Module) -> str:
    """Hash all parameters AND state buffers, including BatchNorm counters."""
    digest = hashlib.sha256()
    for name, tensor in sorted(front.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        header = json.dumps({"name": name, "shape": list(value.shape), "dtype": str(value.dtype)},
                            sort_keys=True, separators=(",", ":")).encode("utf-8")
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def upstream_contract() -> dict:
    commit = subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(UPSTREAM), "status", "--porcelain"], text=True).strip()
    if commit != EXPECTED_COMMIT or dirty:
        raise ValueError("Pinned upstream repository must remain clean and unchanged")
    return {"commit": commit, "clean": True,
            "frontend_code": binding(UPSTREAM / "avsr/models/resnet/resnet.py"),
            "encoder_code": binding(UPSTREAM / "avsr/models/encoder.py"),
            "inference_config": binding(UPSTREAM / "configs/v_inference.yaml")}


def build_frozen_front(plan: dict) -> tuple[torch.nn.Module, dict, str]:
    """Load one pinned module directly, avoiding encoder/decoder construction."""
    code = verify_binding(plan["upstream"]["frontend_code"])
    checkpoint = verify_binding(plan["checkpoint"])
    if plan["checkpoint"]["sha256"] != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("Expected the previously verified public checkpoint")
    spec = importlib.util.spec_from_file_location("step07_pinned_visual_front", code)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    front = module.Resnet2D_front(3, 512)
    conv = front.conv1[0]
    if (not isinstance(conv, torch.nn.Conv3d) or conv.kernel_size != (5, 7, 7)
            or conv.stride != (1, 2, 2) or conv.padding != (2, 0, 0)):
        raise ValueError("Unexpected temporal frontend architecture")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    contract = load_front_state(front, state)
    del state
    initial = state_sha256(front)
    front.to(device="cuda:0", dtype=torch.float32)
    if front.training or any(p.requires_grad for p in front.parameters()):
        raise ValueError("Teacher must be frozen in evaluation mode")
    if any(module.training for module in front.modules()):
        raise ValueError("Every teacher submodule must be in evaluation mode")
    if state_sha256(front) != initial:
        raise ValueError("Moving the frontend to CUDA changed its state")
    return front, contract, initial


def validate_data_plan(data_plan: dict) -> dict[str, dict]:
    if data_plan.get("schema") != "feature-transfer-step07-data-plan-v1":
        raise ValueError("Unexpected source data plan")
    rows = data_plan["records"]
    lookup = {row["id"]: row for row in rows}
    if (len(rows) != len(lookup) or len(rows) != 512
            or sum(row["split"] == "train" for row in rows) != 441
            or sum(row["split"] == "dev" for row in rows) != 71):
        raise ValueError("Require the fixed 441-train / 71-dev recording split")
    for sid in lookup:
        if Path(sid).name != sid or any(ch in sid for ch in ("/", "\\", ":")):
            raise ValueError("Unsafe sample ID")
    training = [row for row in rows if row["split"] == "train"]
    dev = [row for row in rows if row["split"] == "dev"]
    if {r["source_stem"] for r in training} & {r["source_stem"] for r in dev}:
        raise ValueError("Recording leakage in new training split")
    if {r["text"] for r in training} & {r["text"] for r in dev}:
        raise ValueError("Exact transcript overlap in new split")
    speakers = sorted({r["speaker_id"] for r in rows})
    pilot = [min((r for r in training if r["speaker_id"] == speaker),
                 key=lambda r: (-r["frames"], r["id"]))["id"] for speaker in speakers]
    if len(speakers) != 8 or pilot != data_plan["preparation"]["pilot_ids"]:
        raise ValueError("Pilot must be the predetermined longest train clip of each speaker")
    for evidence in [*data_plan["bindings"], *data_plan["preparation"]["code"], data_plan["planner"]]:
        verify_binding(evidence)
    return lookup


def validate_manifest(manifest: dict, data_plan: dict, data_sha: str, phase: str) -> dict[str, dict]:
    if (manifest.get("schema") != "feature-transfer-step07-input-manifest-v1"
            or manifest.get("status") != "passed" or manifest["plan_sha256"] != data_sha):
        raise ValueError("Prepared input manifest is not linked to the frozen data plan")
    expected = (data_plan["preparation"]["pilot_ids"] if phase == "pilot"
                else [row["id"] for row in data_plan["records"]])
    rows = manifest["records"]
    if (len(rows) != len(expected) or manifest["ids"] != expected
            or [r["id"] for r in rows] != expected or len(set(expected)) != len(expected)
            or manifest["record_count"] != len(expected)):
        raise ValueError("Prepared manifest does not contain the complete fixed phase cohort")
    lookup = {r["id"]: r for r in rows}
    for record in data_plan["records"]:
        if record["id"] not in lookup:
            continue
        item = lookup[record["id"]]
        if (item["split"] != record["split"] or item["source_stem"] != record["source_stem"]
                or item["student_frames"] != record["frames"] or item["video_fps"] != record["video_fps"]
                or item["plan_sha256"] != data_sha):
            raise ValueError("Prepared record metadata differs from fixed training/dev selection")
    return lookup


def freeze_plan(data_path: Path, pilot_path: Path, destination: Path) -> dict:
    data_binding, pilot_binding = binding(data_path), binding(pilot_path)
    data_plan = json.loads(data_path.read_text(encoding="utf-8"))
    validate_data_plan(data_plan)
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    validate_manifest(pilot, data_plan, data_binding["sha256"], "pilot")
    configure_runtime()
    checkpoint = binding(OFFICIAL / "weights/multiview_lip.pt")
    if checkpoint["sha256"] != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("Official model checkpoint changed")
    plan = {"schema": "feature-transfer-step07-teacher-plan-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
            "frozen_before_feature_extraction": True, "data_plan": data_binding, "pilot_manifest": pilot_binding,
            "checkpoint": checkpoint, "upstream": upstream_contract(), "runner": binding(Path(__file__)),
            "contract_tests": binding(BASE / "teacher_tests.py"), "runtime": runtime_info(),
            "train_ids": [r["id"] for r in data_plan["records"] if r["split"] == "train"],
            "pilot_ids": data_plan["preparation"]["pilot_ids"],
            "contract": {"layer": "model.encoder.front", "module": "Resnet2D_front(3,512)",
                         "state_prefix": STATE_PREFIX, "strict": True, "batch_size": 1,
                         "dtype": "float32", "frozen": True, "eval": True,
                         "input": "native-rate BGR96 annotation bbox margin0, with real temporal context from data plan",
                         "normalization": "one scalar mean and sample std(correction=1) over the entire prepared native window",
                         "time": "whole native temporal window in one forward call; no independent-frame inference",
                         "temporal_kernel": 5, "temporal_stride": 1,
                         "mapping": "gather exact student original-frame IDs from native rows, no interpolation or clamping",
                         "output": "student-aligned float32 Tstudent x512",
                         "minimum_target_l2_norm_exclusive": 1e-8,
                         "teacher_student_crops_identical": False,
                         "state_integrity": "all parameters and buffers hashed before and after each extraction",
                         "scope": "train441 only; dev71 receives no teacher targets; no audio or ASR text output"},
            "output_directory": str((BASE / "features").resolve()),
            "result_directory": str((BASE / "teacher_results").resolve()),
            "full_manifest_policy": "May expand the fixed pilot into exactly all512 data-plan records; preserve every pilot record. Only train441 NPZ payloads may be read by this runner."}
    save_new_json(destination, plan)
    print(json.dumps({"frozen_teacher_plan": binding(destination), "train": 441, "pilot": 8}, ensure_ascii=False), flush=True)
    return plan


def verify_saved_feature(path: Path, result_path: Path, plan_sha: str, input_record: dict,
                         row: dict, expected_state_hash: str | None = None) -> dict:
    with np.load(path, allow_pickle=False) as payload:
        required = {"features", "student_source_indices", "teacher_source_indices", "gather_indices", "metadata_json"}
        if set(payload.files) != required:
            raise ValueError("Saved feature payload fields differ")
        meta = json.loads(str(payload["metadata_json"].item()))
        if (meta["teacher_plan_sha256"] != plan_sha or meta["id"] != row["id"]
                or meta["input_record"] != input_record or meta["data_record_sha256"] != canonical_sha(row)
                or meta["state_sha256_before"] != meta["state_sha256_after"]):
            raise ValueError("Saved feature provenance differs from the fixed extraction")
        if expected_state_hash is not None and meta["state_sha256_before"] != expected_state_hash:
            raise ValueError("Saved features were made by a different frontend state")
        features = payload["features"]
        if (features.dtype != np.float32 or features.shape != (row["frames"], 512)
                or not np.isfinite(features).all() or np.any(np.linalg.norm(features, axis=1) <= 1e-8)):
            raise ValueError("Saved features have incorrect shape, dtype, or finite values")
        gathered = exact_gather_indices(payload["teacher_source_indices"], payload["student_source_indices"])
        if not np.array_equal(gathered, payload["gather_indices"]):
            raise ValueError("Saved feature mapping changed")
        if payload["student_source_indices"].size != row["frames"]:
            raise ValueError("Saved feature rows disagree with student frames")
        if hashlib.sha256(features.tobytes()).hexdigest() != meta["feature_values_sha256"]:
            raise ValueError("Saved feature value hash differs")
    result = {**meta, "feature_file": binding(path), "status": "completed"}
    if result_path.exists():
        if json.loads(result_path.read_text(encoding="utf-8")) != result:
            raise ValueError("Saved feature result and payload metadata differ")
    else:
        # Recover interruption between exclusive NPZ publication and JSON publication.
        save_new_json(result_path, result)
    return result


def run(plan_path: Path, manifest_path: Path, phase: str) -> dict:
    plan_sha = sha256(plan_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("schema") != "feature-transfer-step07-teacher-plan-v1":
        raise ValueError("Unexpected teacher extraction plan")
    if verify_binding(plan["runner"]) != Path(__file__).resolve():
        raise ValueError("This is not the frozen runner")
    verify_binding(plan["contract_tests"])
    data_path, pilot_path = verify_binding(plan["data_plan"]), verify_binding(plan["pilot_manifest"])
    data_plan = json.loads(data_path.read_text(encoding="utf-8"))
    rows = validate_data_plan(data_plan)
    manifest_binding = binding(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    inputs = validate_manifest(manifest, data_plan, plan["data_plan"]["sha256"], phase)
    pilot_manifest = json.loads(pilot_path.read_text(encoding="utf-8"))
    pilot_inputs = validate_manifest(pilot_manifest, data_plan, plan["data_plan"]["sha256"], "pilot")
    if phase == "pilot" and manifest_binding != plan["pilot_manifest"]:
        raise ValueError("Pilot manifest must match its frozen path and SHA")
    if phase == "all-train" and any(inputs[sid] != item for sid, item in pilot_inputs.items()):
        raise ValueError("Expanded manifest changed a frozen pilot input")
    train_ids = [r["id"] for r in data_plan["records"] if r["split"] == "train"]
    if plan["train_ids"] != train_ids or plan["pilot_ids"] != data_plan["preparation"]["pilot_ids"]:
        raise ValueError("Teacher cohort differs from data plan")
    selected = plan["pilot_ids"] if phase == "pilot" else train_ids
    if any(rows[sid]["split"] != "train" for sid in selected):
        raise ValueError("Teacher target extraction is strictly train-only")
    output, results = Path(plan["output_directory"]), Path(plan["result_directory"])
    if output.resolve() != (BASE / "features").resolve() or results.resolve() != (BASE / "teacher_results").resolve():
        raise ValueError("Feature results must remain in isolated stage07 directories")
    if phase == "all-train":
        pilot_result_path = results / "pilot_manifest.json"
        if not pilot_result_path.exists():
            raise ValueError("Complete and review the fixed eight-feature pilot before extracting all train targets")
        pilot_result = json.loads(pilot_result_path.read_text(encoding="utf-8"))
        if (pilot_result.get("status") != "completed" or pilot_result["ids"] != plan["pilot_ids"]
                or pilot_result["teacher_plan"] != binding(plan_path)):
            raise ValueError("Feature pilot does not belong to this frozen teacher plan")
    if upstream_contract() != plan["upstream"]:
        raise ValueError("Upstream code changed")
    configure_runtime()
    runtime = runtime_info()
    if runtime != plan["runtime"]:
        raise ValueError("Feature extraction runtime differs from frozen plan")
    started = time.perf_counter()
    front, checkpoint_contract, initial_state = build_frozen_front(plan)
    completed, reused = [], 0
    try:
        for number, sid in enumerate(selected, 1):
            input_record, row = inputs[sid], rows[sid]
            input_path = Path(input_record["npz_path"])
            if sha256(input_path) != input_record["npz_sha256"]:
                raise ValueError("Prepared NPZ changed before feature extraction/resume")
            feature_path, result_path = output / f"{sid}.npz", results / "records" / f"{sid}.json"
            if feature_path.exists():
                result = verify_saved_feature(feature_path, result_path, plan_sha, input_record, row, initial_state)
                completed.append(result)
                reused += 1
                print(f"[teacher reuse] {number}/{len(selected)} {sid}", flush=True)
                continue
            if result_path.exists():
                raise ValueError("Feature result exists without its immutable payload")
            item_started = time.perf_counter()
            arrays = read_prepared_npz(input_path, input_record["npz_sha256"], row, plan["data_plan"]["sha256"])
            video, normalization = normalize_teacher(arrays["teacher_bgr"])
            video = video.to("cuda:0")
            before = state_sha256(front)
            if (before != initial_state or any(module.training for module in front.modules())
                    or any(p.requires_grad for p in front.parameters())):
                raise ValueError("Teacher state or frozen evaluation mode changed")
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            feature_started = time.perf_counter()
            with torch.inference_mode():
                native_output = front(video)
            torch.cuda.synchronize()
            timing = {"seconds": time.perf_counter() - feature_started,
                      "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                      "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                      "scope": "CUDA-resident whole native video to visual-front output; excludes normalization, transfer, source preparation and file writes"}
            features = validate_front_output(native_output, len(arrays["teacher_bgr"]), arrays["gather_indices"])
            after = state_sha256(front)
            if after != before:
                raise ValueError("Teacher parameters or BatchNorm buffers changed during extraction")
            if sha256(input_path) != input_record["npz_sha256"]:
                raise ValueError("Prepared NPZ changed during feature extraction")
            meta = {"schema": "feature-transfer-step07-teacher-record-v1", "id": sid,
                    "split": "train", "speaker_id": row["speaker_id"], "source_stem": row["source_stem"],
                    "created_utc": datetime.now(timezone.utc).isoformat(), "teacher_plan_sha256": plan_sha,
                    "data_plan": plan["data_plan"], "input_manifest": manifest_binding,
                    "input_record": input_record, "data_record_sha256": canonical_sha(row),
                    "checkpoint": plan["checkpoint"], "checkpoint_contract": checkpoint_contract,
                    "runtime": runtime, "state_sha256_before": before, "state_sha256_after": after,
                    "normalization": normalization, "native_feature_shape": list(native_output.shape),
                    "aligned_feature_shape": list(features.shape), "feature_dtype": "float32",
                    "feature_values_sha256": hashlib.sha256(features.tobytes()).hexdigest(),
                    "feature_timing": timing, "processing_seconds_before_save": time.perf_counter() - item_started,
                    "transcript_used_for_teacher_forward": False, "audio_used": False,
                    "decoder_instantiated": False, "trainable_parameters": 0}
            save_new_npz(feature_path, features=features,
                         student_source_indices=arrays["student_source_indices"],
                         teacher_source_indices=arrays["teacher_source_indices"], gather_indices=arrays["gather_indices"],
                         metadata_json=np.array(json.dumps(meta, ensure_ascii=False, allow_nan=False)))
            result = verify_saved_feature(feature_path, result_path, plan_sha, input_record, row, initial_state)
            completed.append(result)
            print(f"[teacher] {number}/{len(selected)} {sid} {len(arrays['teacher_bgr'])}->{len(features)}frames "
                  f"{timing['seconds']:.3f}s peak={timing['peak_allocated_bytes'] / 1024**2:.1f}MiB", flush=True)
            del video, native_output, features, arrays
        if state_sha256(front) != initial_state:
            raise ValueError("Teacher state changed across the extraction run")
    finally:
        del front
        torch.cuda.empty_cache()
    summary = {"schema": "feature-transfer-step07-teacher-manifest-v1", "status": "completed", "phase": phase,
               "created_utc": datetime.now(timezone.utc).isoformat(), "teacher_plan": binding(plan_path),
               "input_manifest": manifest_binding, "runtime": runtime, "ids": selected,
               "records": completed, "record_count": len(completed), "reused_records": reused,
               "state_sha256_before": initial_state, "state_sha256_after": initial_state,
               "checkpoint_contract": checkpoint_contract, "elapsed_seconds": time.perf_counter() - started,
               "native_frames": sum(r["native_feature_shape"][1] for r in completed),
               "student_frames": sum(r["aligned_feature_shape"][0] for r in completed),
               "feature_gpu_seconds": sum(r["feature_timing"]["seconds"] for r in completed),
               "peak_allocated_bytes": max(r["feature_timing"]["peak_allocated_bytes"] for r in completed),
               "peak_reserved_bytes": max(r["feature_timing"]["peak_reserved_bytes"] for r in completed),
               "feature_file_bytes": sum(Path(r["feature_file"]["path"]).stat().st_size for r in completed),
               "audio_used": False, "dev_teacher_targets_created": False, "decoder_instantiated": False}
    destination = results / ("pilot_manifest.json" if phase == "pilot" else "manifest.json")
    if destination.exists():
        prior = json.loads(destination.read_text(encoding="utf-8"))
        if (prior["teacher_plan"] != summary["teacher_plan"] or prior["input_manifest"] != summary["input_manifest"]
                or prior["records"] != summary["records"]):
            raise ValueError("Existing completed teacher manifest differs")
        summary = prior
    else:
        save_new_json(destination, summary)
    print(json.dumps({"teacher_manifest": binding(destination), "record_count": len(completed),
                      "new_records": len(completed) - reused, "status": "completed"}, ensure_ascii=False), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--data-plan", type=Path, default=BASE / "data_plan.json")
    parser.add_argument("--pilot-manifest", type=Path, default=BASE / "inputs/pilot_manifest.json")
    parser.add_argument("--plan", type=Path, default=BASE / "teacher_plan.json")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--phase", choices=["pilot", "all-train"])
    args = parser.parse_args()
    if args.freeze:
        if args.phase or args.manifest:
            parser.error("--freeze cannot be combined with execution phase/manifest")
        freeze_plan(args.data_plan.resolve(), args.pilot_manifest.resolve(), args.plan.resolve())
    else:
        if not args.phase or not args.manifest:
            parser.error("Extraction requires --phase and --manifest")
        run(args.plan.resolve(), args.manifest.resolve(), args.phase)


if __name__ == "__main__":
    main()
