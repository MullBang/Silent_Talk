"""Summarize completed, independently checked Step4 artifacts without training."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from freeze_handoff import BASE, ROOT, binding, read, validate_ready


def complete() -> dict:
    """Check frozen evidence and publish the bounded completion record."""
    plan_ref = binding(BASE / "full_data_plan.json")
    plan = read(plan_ref)
    refs = plan["artifacts"]
    docs = {key: read(ref) for key, ref in refs.items()}
    validate_ready(docs, refs)
    for ref in [plan["generator"], plan["schedule_code"], plan["contract_tests"], *plan["verified_code"]]:
        if binding(Path(ref["path"])) != ref:
            raise ValueError("Frozen handoff implementation changed")
    prior = {
        "step01_reproduction_gate": ("local_source_step01_20261007/results/reproduction_gate.json", "173ea79a90c22f59ec5294d423f10c1462192415ba7e9fa5ccce35261343d739"),
        "step02_completion": ("local_pilot_step02_20261007/results/step02_summary.json", "c26a4b9b4a90ad553cacddecfd68fe22aad6835af507b9ce921ccfb6d5ec856e"),
        "step03_completion": ("local_training_step03_20261007/results/step03_summary.json", "aa62fc9f6ad16f6d2b05843096baf2cd3b3f4332108f2883417f39fa76b224fd"),
    }
    preserved = {}
    for key, (relative, expected) in prior.items():
        ref = binding(ROOT / "experiments" / relative)
        if ref["sha256"] != expected:
            raise ValueError("Prior completion evidence changed")
        preserved[key] = ref
    tmp = ROOT / "experiments/local_source_step01_20261007/tmp"
    if tmp.exists() and any(path.is_file() for path in tmp.rglob("*")):
        raise ValueError("Temporary source payloads remain")
    inputs, features = docs["inputs"], docs["features"]
    entries = inputs["records"]
    storage = {"all512_input_npz_bytes": sum(Path(r["npz_path"]).stat().st_size for r in entries),
               "new504_input_npz_bytes": inputs["new_input_npz_bytes"],
               "train441_input_npz_bytes": sum(Path(r["npz_path"]).stat().st_size for r in entries if r["split"] == "train"),
               "dev71_input_npz_bytes": sum(Path(r["npz_path"]).stat().st_size for r in entries if r["split"] == "dev"),
               "all441_feature_npz_bytes": features["all441_feature_file_bytes"],
               "new433_feature_npz_bytes": features["new433_feature_file_bytes"],
               "c_free_bytes_after": shutil.disk_usage(ROOT).free}
    if storage["all512_input_npz_bytes"] != inputs["input_npz_bytes"]:
        raise ValueError("Prepared input storage changed")
    storage["all_referenced_input_and_feature_npz_bytes"] = storage["all512_input_npz_bytes"] + storage["all441_feature_npz_bytes"]
    storage["new_input_and_feature_npz_bytes"] = storage["new504_input_npz_bytes"] + storage["new433_feature_npz_bytes"]
    source = read(plan["source_plan"])
    temporary_max = max(sum(s["members"][role]["size"] for role in ("video", "label")) for s in source["sources"])
    materialized_bytes = sum(s["members"][role]["size"] for s in source["sources"] for role in ("video", "label"))
    groups = [("preparation", "test_prepare_full.py", 11, ".venv-tokenizer"),
              ("independent_inputs", "test_verify_full.py", 35, ".venv-tokenizer"),
              ("teacher_contract", "test_teacher_full.py", 10, ".venv-gpu"),
              ("independent_features", "test_verify_features.py", 8, ".venv-tokenizer"),
              ("handoff", "test_freeze_handoff.py", 11, ".venv-tokenizer")]
    summary = {"schema": "lipnet-local-step04-completion-v1", "status": "completed",
               "created_utc": datetime.now(timezone.utc).isoformat(), "full_data_plan": plan_ref,
               "artifacts": refs, "parent_plan": plan["parent_plan"], "source_plan": plan["source_plan"],
               "preserved_prior_evidence": preserved,
               "inputs": {"total": 512, "train": 441, "dev": 71, "reused_pilot": 8, "new": 504,
                          "student_frames": inputs["student_frames"], "train_native_teacher_frames": inputs["teacher_frames"],
                          "source_recordings": 59, "elapsed_seconds": inputs["elapsed_seconds"]},
               "teacher": {"total": 441, "reused_pilot": 8, "new": 433, "development_targets": 0,
                           "aligned_student_frames": features["student_frames"], "feature_width": 512,
                           "elapsed_seconds": features["elapsed_seconds_this_run"],
                           "new433_gpu_forward_seconds": features["new433_gpu_forward_seconds"],
                           "new433_peak_allocated_bytes": features["new433_peak_allocated_bytes"],
                           "new433_peak_reserved_bytes": features["new433_peak_reserved_bytes"],
                           "state_sha256": features["state_sha256_after"]},
               "visual_sample": {"sources": 59, "sentences": 59, "images": 177,
                                 "review": refs["visual_review"], "scope": "sampled student ROI images only"},
               "tests": {"reported_passed_total": sum(g[2] for g in groups),
                         "groups": [{"name": name, "file": binding(BASE / filename), "passed": count, "environment": runtime}
                                    for name, filename, count, runtime in groups]},
               "storage": storage, "maximum_temporary_video_label_payload_bytes": temporary_max,
               "unique_selected_video_label_payload_bytes": materialized_bytes,
               "temporary_source_payload_files_remaining": 0,
               "student_optimization_steps": 0, "teacher_optimization_steps": 0,
               "new_downloads": 0, "G_drive_accessed": False, "audio_used": False,
               "service_weights_updated": False, "full_student_training_started": False,
               "next": "Step5 binds a full-data loader/entrypoint to full_data_plan.json and executes the fixed paired comparison.",
               "limitations": ["Preparation checks do not establish transcript accuracy or model learning benefit.",
                               "Development71 shares the training speaker pool and is not an untouched final test.",
                               "Only sampled crop appearance was visually reviewed; no whole-video manual transcript review.",
                               "GPU extraction memory is not a measurement of full student training memory."],
               "summary_code": binding(Path(__file__))}
    with (BASE / "results/step04_summary.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return summary


if __name__ == "__main__":
    value = complete()
    print(json.dumps({"status": value["status"], "summary": binding(BASE / "results/step04_summary.json"),
                      "storage": value["storage"], "inputs": value["inputs"], "teacher": value["teacher"]}, ensure_ascii=False), flush=True)
