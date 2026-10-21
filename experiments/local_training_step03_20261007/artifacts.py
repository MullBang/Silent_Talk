"""Read the completed fixed-eight pilot without media I/O or model loading.

The immutable step02 completion hash is the trust root.  Only prepared local
NPZ files are consumed; G:/D: references in provenance remain descriptive.
This intentionally cannot masquerade as the not-yet-prepared 512-clip dataset.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
PILOT = ROOT / "experiments/local_pilot_step02_20261007"
SUMMARY = PILOT / "results/step02_summary.json"
SUMMARY_SHA256 = "c26a4b9b4a90ad553cacddecfd68fe22aad6835af507b9ce921ccfb6d5ec856e"
PARENT_SHA256 = "c33dd5c2b77ed7298c497e58218d70a7552b476b947659f3accd33748af13b30"
INPUT_SHA256 = "d807b5a67e109019b1cba82181dbba8c6204e736ecc78db09b5c73fbcf381fac"
FEATURE_SHA256 = "f36bc7bebbc30ab6dc3650a298488b9dc09abe52bd0ae36ceea182fa4eac2ab8"


def sha256(path: Path | str) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical_sha(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class LocalBindings:
    """Hash only workspace files; cache repeated references within one load."""

    def __init__(self, root: Path = ROOT):
        self.root = root.resolve()
        self.items: dict[str, dict] = {}
        self.stats: dict[str, tuple[int, int]] = {}

    def path(self, name: str | Path) -> Path:
        path = Path(name).resolve()
        if not path.is_relative_to(self.root) or not path.is_file():
            raise ValueError(f"Only existing local workspace files are allowed: {name}")
        return path

    def verify(self, item: dict) -> Path:
        path = self.path(item["path"])
        key = str(path)
        wanted = item["sha256"]
        stat = path.stat()
        identity = (stat.st_size, stat.st_mtime_ns)
        if key in self.items:
            if self.items[key]["sha256"] != wanted or self.stats[key] != identity:
                raise ValueError(f"Conflicting or changed file binding: {path}")
            return path
        if len(wanted) != 64 or sha256(path) != wanted:
            raise ValueError(f"Frozen file hash changed: {path}")
        end = path.stat()
        if identity != (end.st_size, end.st_mtime_ns):
            raise ValueError(f"File changed during verification: {path}")
        self.items[key] = {"path": key, "sha256": wanted}
        self.stats[key] = identity
        return path

    def read(self, item: dict) -> dict:
        path = self.verify(item)
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != item["sha256"]:
            raise ValueError(f"JSON changed while reading: {path}")
        return json.loads(raw)

    def finish(self) -> list[dict]:
        for key, previous in self.stats.items():
            stat = Path(key).stat()
            if (stat.st_size, stat.st_mtime_ns) != previous:
                raise ValueError(f"File changed during artifact load: {key}")
        return list(self.items.values())


def validate_cohort(parent: dict, source: dict, inputs: dict, features: dict,
                    teacher_plan: dict) -> tuple[list[str], dict, dict, dict]:
    """Reject replacement, reordered, repeated, expanded, or dev/test pilots."""
    ids = parent["preparation"]["pilot_ids"]
    rows = {row["id"]: row for row in parent["records"]}
    if len(parent["records"]) != 512 or len(rows) != 512:
        raise ValueError("Require the unchanged 512-record logical parent")
    if (len(ids) != 8 or len(set(ids)) != 8
            or any(rows[sid]["split"] != "train" for sid in ids)
            or len({rows[sid]["speaker_id"] for sid in ids}) != 8):
        raise ValueError("Require exactly eight fixed training speakers")
    if source["records"] != parent["records"] or source["parent_plan"]["sha256"] != PARENT_SHA256:
        raise ValueError("D source logical records changed")
    if (inputs.get("status") != "passed" or inputs.get("phase") != "pilot"
            or inputs.get("record_count") != 8 or features.get("status") != "completed"
            or features.get("phase") != "pilot" or teacher_plan["pilot_ids"] != ids):
        raise ValueError("Input/feature pilot is incomplete or has another scope")
    for manifest in (inputs, features):
        if manifest["ids"] != ids or [row["id"] for row in manifest["records"]] != ids:
            raise ValueError("Manifest cohort differs from the fixed ordered eight IDs")
        if any(row["split"] != "train" for row in manifest["records"]):
            raise ValueError("Pilot artifacts must be strictly train-only")
    return ids, rows, {r["id"]: r for r in inputs["records"]}, {r["id"]: r for r in features["records"]}


def validate_clip(row: dict, arrays: dict, payload: dict, feature_record: dict) -> None:
    """Cross-check the saved teacher target against this exact student input."""
    if row["split"] != "train" or feature_record["split"] != "train":
        raise ValueError("No development or test targets are allowed")
    for key in ("id", "source_stem", "speaker_id"):
        if feature_record[key] != row[key]:
            raise ValueError(f"Feature identity mismatch: {key}")
    if (arrays["metadata"]["record"] != row
            or arrays["metadata"]["id"] != row["id"]
            or arrays["metadata"]["plan_sha256"] != PARENT_SHA256):
        raise ValueError("Student transcript/timing/source metadata changed")
    pixels = arrays["student_rgb"]
    target = payload["features"]
    if pixels.dtype != np.uint8 or pixels.shape != (row["frames"], 96, 96, 3):
        raise ValueError("Student must be unchanged T x 96 x 96 x 3 uint8")
    if (target.dtype != np.float32 or target.shape != (row["frames"], 512)
            or not np.isfinite(target).all()
            or np.any(np.linalg.norm(target.astype(np.float64), axis=1) <= 1e-8)):
        raise ValueError("Invalid finite nonzero T x 512 teacher target")
    for key in ("student_source_indices", "teacher_source_indices", "gather_indices"):
        if (payload[key].dtype != np.int64 or arrays[key].dtype != np.int64
                or not np.array_equal(payload[key], arrays[key])):
            raise ValueError(f"Student and teacher target source mapping differs: {key}")
    if hashlib.sha256(target.tobytes()).hexdigest() != feature_record["feature_values_sha256"]:
        raise ValueError("Feature values differ from frozen evidence")
    tokens = row["token_ids"]
    if (not tokens or any(type(t) is not int or not 1 <= t < 69 for t in tokens)
            or len(tokens) + sum(a == b for a, b in zip(tokens, tokens[1:])) > row["frames"]):
        raise ValueError("Invalid or CTC-infeasible unmodified transcript tokens")
    if (feature_record["data_record_sha256"] != canonical_sha(row)
            or feature_record["state_sha256_before"] != feature_record["state_sha256_after"]
            or feature_record["audio_used"] is not False
            or feature_record["decoder_instantiated"] is not False
            or feature_record["transcript_used_for_teacher_forward"] is not False
            or feature_record["optimization_steps"] != 0
            or feature_record["trainable_parameters"] != 0):
        raise ValueError("Frozen visual teacher provenance changed")


def load_pilot() -> dict:
    """Load only the completed step02 fixed-eight cohort, with frozen evidence."""
    checked = LocalBindings()
    completion_binding = {"path": str(SUMMARY), "sha256": SUMMARY_SHA256}
    summary = checked.read(completion_binding)
    if (summary["schema"] != "lipnet-roadmap-20261007-step02-completion-v1"
            or summary["status"] != "completed" or summary["record_count"] != 8
            or summary["student_optimization_steps"] != 0 or summary["teacher_optimization_steps"] != 0):
        raise ValueError("Step02 has not completed the required inference-only pilot")
    bindings = summary["artifacts"]
    documents = {key: checked.read(item) for key, item in bindings.items()}
    input_plan, inputs = documents["input_plan"], documents["inputs"]
    teacher_plan, features = documents["teacher_plan"], documents["teacher_features"]
    parent_binding, source_binding = teacher_plan["data_plan"], teacher_plan["source_plan"]
    if (parent_binding["sha256"] != PARENT_SHA256
            or bindings["inputs"]["sha256"] != INPUT_SHA256
            or bindings["teacher_features"]["sha256"] != FEATURE_SHA256
            or teacher_plan["pilot_manifest"] != bindings["inputs"]
            or teacher_plan["input_audit"] != bindings["input_audit"]
            or features["teacher_plan"] != bindings["teacher_plan"]
            or features["input_manifest"] != bindings["inputs"]
            or features["input_audit"] != bindings["input_audit"]
            or inputs["source_plan"] != source_binding
            or inputs["plan_sha256"] != PARENT_SHA256
            or input_plan["parent_plan"] != parent_binding
            or input_plan["source_plan"] != source_binding):
        raise ValueError("Step02 source/plan/manifest bindings disagree")
    parent, source = checked.read(parent_binding), checked.read(source_binding)
    ids, rows, input_rows, feature_rows = validate_cohort(parent, source, inputs, features, teacher_plan)
    code_bindings = [*parent["bindings"], *parent["preparation"]["code"], parent["planner"],
                     input_plan["preparation_runner"], *input_plan["transport_code"],
                     input_plan["reproduction_gate"], teacher_plan["runner"], teacher_plan["primitives"],
                     *teacher_plan["contract_tests"], teacher_plan["checkpoint"],
                     *[value for value in teacher_plan["upstream"].values() if isinstance(value, dict)]]
    for item in code_bindings:
        checked.verify(item)
    # Import only the hash-pinned NumPy validation helpers.  This module imports
    # torch but neither constructs a model nor touches a CUDA context on import.
    primitive_path = checked.verify(teacher_plan["primitives"])
    spec = importlib.util.spec_from_file_location("step03_frozen_teacher_checks", primitive_path)
    teacher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(teacher)
    teacher.validate_data_plan(parent)
    audit, review, visual = documents["input_audit"], documents["feature_review"], documents["visual_review"]
    if (audit["status"] != "passed" or audit["scope"] != "pilot8"
            or audit["plan"] != parent_binding or audit["manifest"] != bindings["inputs"]
            or audit["source_plan"] != source_binding
            or [r["id"] for r in audit["records"]] != ids
            or review["status"] != "passed" or review["ids"] != ids
            or review["teacher_manifest"] != bindings["teacher_features"]
            or review["teacher_plan"] != bindings["teacher_plan"]
            or visual["status"] != "passed_sampled_frames" or visual["ids"] != ids
            or visual["manifest"] != bindings["inputs"]):
        raise ValueError("Independent audits do not prove this pilot")
    for item in [audit["code"], audit["numerical_ancestor"], visual["contact_sheet"], *review["verified_bindings"]]:
        checked.verify(item)
    review_rows = {r["id"]: r for r in review["records"]}
    clips = {}
    state_hash = None
    for sid in ids:
        row, input_row, feature_row = rows[sid], input_rows[sid], feature_rows[sid]
        input_binding = {"path": input_row["npz_path"], "sha256": input_row["npz_sha256"]}
        input_path = checked.verify(input_binding)
        feature_path = checked.verify(feature_row["feature_file"])
        # verify_saved_feature has a recovery-write branch.  Requiring an
        # existing local sidecar before calling it keeps this reader read-only.
        sidecar = checked.path(PILOT / "teacher_results/records" / f"{sid}.json")
        sidecar_binding = {"path": str(sidecar), "sha256": sha256(sidecar)}
        if checked.read(sidecar_binding) != feature_row:
            raise ValueError("Feature sidecar differs from the frozen feature manifest")
        arrays = teacher.read_prepared_npz(input_path, input_binding["sha256"], row, PARENT_SHA256)
        verified = teacher.verify_saved_feature(feature_path, sidecar, bindings["teacher_plan"]["sha256"],
                                                input_row, row, state_hash)
        if verified != feature_row:
            raise ValueError("Feature payload differs from the frozen manifest")
        state_hash = feature_row["state_sha256_before"]
        with np.load(feature_path, allow_pickle=False) as saved:
            payload = {key: saved[key] for key in ("features", "student_source_indices", "teacher_source_indices", "gather_indices")}
        validate_clip(row, arrays, payload, feature_row)
        prior = review_rows[sid]
        if (prior["feature_file"] != feature_row["feature_file"] or prior["input_file"] != input_binding
                or prior["feature_values_sha256"] != feature_row["feature_values_sha256"]):
            raise ValueError("Independent reviewed feature does not match the loaded target")
        # Confirm exact D transport provenance without opening any archive.
        provenance = arrays["metadata"]["source_verification"]
        source_row = next(s for s in source["sources"] if s["stem"] == row["source_stem"])
        if (provenance["transport"] != "audited-local-tar-v1"
                or provenance["source_plan_path"] != source_binding["path"]
                or provenance["source_plan_sha256"] != source_binding["sha256"]
                or provenance["source_stem"] != row["source_stem"]
                or provenance["members"] != {k: v for k, v in source_row["members"].items() if k != "audio"}
                or provenance["audio_used_as_model_input"] is not False):
            raise ValueError("Prepared D source provenance changed")
        if sha256(feature_path) != feature_row["feature_file"]["sha256"]:
            raise ValueError("Feature payload changed while loading")
        clips[sid] = {"id": sid, "pixels": arrays["student_rgb"], "tokens": list(row["token_ids"]),
                      "teacher_features": payload["features"], "text": row["text"],
                      "source_stem": row["source_stem"], "speaker_id": row["speaker_id"], "split": "train"}
    counts = {"clips": len(clips), "student_frames": sum(len(v["pixels"]) for v in clips.values()),
              "teacher_native_frames": summary["teacher_frames"], "feature_width": 512,
              "train": 8, "dev": 0, "test": 0, "speakers": 8}
    if counts["student_frames"] != summary["student_frames"] or counts["student_frames"] != 1543:
        raise ValueError("Prepared pilot frame count changed")
    return {"clips": clips, "ids": list(ids), "bindings": checked.finish(),
            "parent_plan": parent_binding, "input_manifest": bindings["inputs"],
            "feature_manifest": bindings["teacher_features"], "teacher_plan": bindings["teacher_plan"],
            "source_plan": source_binding, "counts": counts, "step02_summary": completion_binding,
            "scope": "fixed8 train-only smoke test; full512 not yet available"}
