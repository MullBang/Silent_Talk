"""Freeze verified full-data references and future paired training schedules.

This publishes a data handoff only. It never constructs a model, reads original
media, or performs an optimizer update. Step5 must bind its full-data loader and
entrypoint to this artifact before running the already frozen comparison recipe.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
STEP3 = ROOT / "experiments/local_training_step03_20261007"
CORE = ROOT / "experiments/feature_transfer_step07/train_compare.py"
CORE_SHA = "eb65e48cf6d4d35825feee1c3d0bd885952357c50c011049944358081790c62a"
RECIPE_SHA = "d2612b2e96e6439505aca8b34af18517bd2bb32daa226c1c7adcb6a405127bb6"


def binding(path: Path) -> dict:
    """Bind a workspace artifact without opening any external source."""
    try:
        path = path.resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError("Handoff evidence must be inside the project")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        return {"path": str(path), "sha256": digest}
    except OSError as exc:
        raise RuntimeError(f"Cannot bind {path}") from exc


def read(ref: dict) -> dict:
    """Verify a JSON reference before decoding it."""
    try:
        path = Path(ref["path"])
        if binding(path) != ref:
            raise ValueError("Handoff evidence changed")
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RuntimeError("Cannot read bound handoff evidence") from exc


def schedules(recipe: dict) -> list[dict]:
    """Use the unchanged training primitive and store every future batch."""
    if binding(CORE)["sha256"] != CORE_SHA:
        raise ValueError("Frozen schedule implementation changed")
    spec = importlib.util.spec_from_file_location("step04_frozen_schedule_core", CORE)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    ids = recipe["train_ids"]
    if len(ids) != 441 or len(set(ids)) != 441 or recipe["seeds"] != [42, 43]:
        raise ValueError("Expected the fixed441 train cohort and two seeds")
    if recipe["updates_per_seed_per_arm"] != 3200 or recipe["accumulated_clips_per_update"] != 2:
        raise ValueError("Training budget differs from the frozen recipe")
    result = []
    for seed in recipe["seeds"]:
        schedule = core.fixed_schedule(ids, seed, updates=3200, batch_size=2)
        counts = Counter(sid for batch in schedule for sid in batch)
        if len(schedule) != 3200 or any(len(batch) != 2 for batch in schedule):
            raise ValueError("Incomplete full schedule")
        if set(counts) != set(ids) or set(counts.values()) != {14, 15}:
            raise ValueError("Unexpected repeated-epoch schedule coverage")
        result.append({"seed": seed, "arms": recipe["arms"], "updates": 3200,
                       "clips_per_update": 2, "schedule": schedule,
                       "per_clip_uses": dict(sorted(counts.items()))})
    return result


def validate_ready(docs: dict, refs: dict) -> None:
    """Reject incomplete cohorts or audits that refer to another artifact."""
    recipe, inputs, features = docs["recipe"], docs["inputs"], docs["features"]
    if refs["recipe"]["sha256"] != RECIPE_SHA:
        raise ValueError("Frozen comparison recipe changed")
    train_ids, dev_ids = recipe["train_ids"], recipe["development_ids"]
    if (len(train_ids) != 441 or len(dev_ids) != 71 or set(train_ids) & set(dev_ids)
            or inputs["status"] != "passed" or inputs["record_count"] != 512
            or features["status"] != "completed" or features["record_count"] != 441
            or features["ids"] != train_ids):
        raise ValueError("Full input/feature coverage is incomplete")
    if ([r["id"] for r in inputs["records"] if r["split"] == "train"] != train_ids
            or [r["id"] for r in inputs["records"] if r["split"] == "dev"] != dev_ids
            or [r["id"] for r in features["records"]] != train_ids
            or any(r["teacher_frames"] != 0 for r in inputs["records"] if r["split"] == "dev")):
        raise ValueError("Fixed order or student-only development scope differs")
    input_audit = docs["input_audit"]
    if (input_audit.get("schema") != "local-tar-step04-independent-input-audit-v1"
            or input_audit.get("scope") != "all512"
            or input_audit["status"] != "passed" or input_audit["manifest"] != refs["inputs"]
            or input_audit["record_count"] != 512
            or input_audit.get("plan") != recipe["parent_plan"]
            or input_audit.get("source_plan") != recipe["source_plan"]
            or input_audit.get("reused_pilot_records_unchanged") != 8
            or input_audit.get("student_only_dev_records") != 71
            or features["input_manifest"] != refs["inputs"]
            or features["input_audit"] != refs["input_audit"]
            or features["teacher_plan"] != refs["teacher_plan"]):
        raise ValueError("Input audit and teacher provenance do not bind these artifacts")
    review = docs["visual_review"]
    if review["status"] != "passed_sampled_review" or review["input_manifest"] != refs["inputs"]:
        raise ValueError("Completed sampled visual review is required")
    feature_audit = docs["feature_audit"]
    if (feature_audit.get("schema") != "local-tar-step04-independent-feature-audit-v1"
            or feature_audit.get("status") != "passed" or feature_audit.get("record_count") != 441
            or feature_audit.get("manifest") != refs["features"]
            or feature_audit.get("input_manifest") != refs["inputs"]
            or feature_audit.get("teacher_plan") != refs["teacher_plan"]
            or feature_audit.get("input_audit") != refs["input_audit"]
            or feature_audit.get("reused_pilot_records_unchanged") != 8
            or feature_audit.get("new_records") != 433
            or feature_audit.get("development_target_count") != 0):
        raise ValueError("Independent feature audit does not bind all441 targets")


def freeze() -> dict:
    """Publish an immutable handoff after all independent reviews pass."""
    paths = {"recipe": STEP3 / "comparison_protocol.json", "inputs": BASE / "inputs/manifest.json",
             "input_audit": BASE / "results/input_audit.json", "teacher_plan": BASE / "teacher_plan.json",
             "features": BASE / "teacher_results/train_manifest.json",
             "feature_audit": BASE / "results/feature_audit.json",
             "visual_review": BASE / "results/visual_review.json"}
    refs = {key: binding(path) for key, path in paths.items()}
    docs = {key: read(ref) for key, ref in refs.items()}
    validate_ready(docs, refs)
    recipe = docs["recipe"]
    code_refs = [docs["input_audit"]["code"], docs["feature_audit"]["code"],
                 docs["feature_audit"]["independent_input_verifier"], *recipe["evaluation_code"]]
    for ref in code_refs:
        if binding(Path(ref["path"])) != ref:
            raise ValueError("Audited code or frozen evaluation implementation changed")
    parent, source = read(recipe["parent_plan"]), read(recipe["source_plan"])
    if source["records"] != parent["records"] or docs["inputs"]["ids"] != [r["id"] for r in parent["records"]]:
        raise ValueError("Parent records changed")
    result = {"schema": "lipnet-step04-full-data-handoff-v1",
              "status": "data_and_schedules_frozen_pending_step05_execution_binding",
              "created_utc": datetime.now(timezone.utc).isoformat(), "artifacts": refs,
              "parent_plan": recipe["parent_plan"], "source_plan": recipe["source_plan"],
              "generator": binding(Path(__file__)), "schedule_code": binding(CORE),
              "contract_tests": binding(BASE / "test_freeze_handoff.py"), "verified_code": code_refs,
              "train_ids": recipe["train_ids"], "development_ids": recipe["development_ids"],
              "training_schedules": schedules(recipe),
              "comparison_contract": {k: recipe[k] for k in ("arms", "model_initialization", "optimizer", "loss",
                                       "precision", "checkpoint_interval_updates", "primary_checkpoint", "evaluation")},
              "step05_requirements": ["Bind a new full-data loader and entrypoint before the first update; Step3 CLI still permits only eight pilot inputs.",
                                      "Verify every referenced NPZ and preserve original pilot feature provenance plus new full-feature provenance.",
                                      "Use the exact stored schedule for both A/B paths of each seed; initialize fresh rather than resuming Step3 smoke checkpoints.",
                                      "Keep development inputs out of training batches and use them only for final normal/static/shuffled evaluation."],
              "student_optimization_steps_this_step": 0, "new_data_download_required": False,
              "webcam_accuracy_established": False, "service_weights_updated": False}
    destination = BASE / "full_data_plan.json"
    with destination.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return {"full_data_plan": binding(destination), "train_records": 441, "development_records": 71,
            "schedules": 2, "updates_per_schedule": 3200, "training_executed": False}


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(freeze(), ensure_ascii=False), flush=True)
