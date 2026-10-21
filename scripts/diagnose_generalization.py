"""Stage15: fixed diagnostic checkpoints on preselected unseen clips; no training.

This explicitly scoped diagnostic does not change the deployment/evaluation guards
for smoke_only checkpoints. Test caches are rejected before any cache operation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "backend"), str(ROOT / "scripts")]
from models.dataset_manifest import load_manifest, sha256_file
from models.decoder import ctc_beam_search_decode, ctc_greedy_decode
from models.registry import load_model_checkpoint
from diagnose_temporal_use import MANIFEST, forward, length_bin, prediction_row, verify_cache, write_json
from diagnose_learning_transition import seed_training
from generalization_selection import select_generalization
from learning_budget_state import capture_rng_state, compare_nested, compare_state_dicts
from summarize_learning_transition import read_json, require
from temporal_diagnostics import analyze_ctc, compare_posteriors, transform_frames

MANIFEST_SHA = "214ce935150b367a15f24f0896901d34c54c6d4fe73d493a9d397c4c94fe59a6"
PRIOR13 = ROOT / "results/step13-budget/run01"
PRIOR14 = ROOT / "results/step14-learning-rate/run01"
MODEL_SPECS = {
    "source6400": (PRIOR13, 6400, "af2a579832b585556967d6f885170610d8e7beb1a8a56967f5350ba059d8e519"),
    "A_keep9600": (PRIOR14 / "A_keep", 9600, "7d5a8bcf1dea50729ba02734deacacb3f38289b4307c8cb71b10fc62915c1c5f"),
    "B_reduce9600": (PRIOR14 / "B_reduce", 9600, "2b5c43f1f0dc1a79300b04f0d0d6a5059de0143b241b067633fa7439a07864ed"),
}
CONDITIONS = ["normal", "static_middle", "shuffle_42"]
SETTINGS = {
    "conditions": CONDITIONS, "shuffle_seed": 42, "selection_seed": 42,
    "beam_width": 10, "beam_conditions": ["normal"], "primary_decoder": "greedy",
    "precision": "fp32_unpadded_B1", "batch_size": 1, "torch_threads": 2,
    "cudnn_deterministic": True, "cudnn_benchmark": False, "tf32": False,
    "model_selection": False, "training": False, "test_cache_access": False,
    "expected_forwards": 720, "expected_beam_rows": 240,
}


def state_digest(state: dict) -> str:
    """Hash tensor names, types, shapes and exact bytes, independent of serialization."""
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        require(bool(torch.isfinite(tensor).all()), "Nonfinite model state")
        header = json.dumps([name, str(tensor.dtype), list(tensor.shape)], separators=(",", ":")).encode()
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def validate_checkpoint_metadata(metadata: dict, key: str, training_ids: list[str], plan_sha: str) -> None:
    """Allow only the three fixed compact inference snapshots and exact 64-ID lineage."""
    require(key in MODEL_SPECS, "Unapproved checkpoint key")
    data, run = metadata["training_data"], metadata["training_run"]
    require(metadata["architecture_version"] == "lipnet-compact-v1" and metadata["num_classes"] == 69,
            "Compact 69-token architecture required")
    require(data["mode"] == "smoke_only" and data["manifest_sha256"] == MANIFEST_SHA
            and data["sample_ids"] == training_ids and len(training_ids) == len(set(training_ids)) == 64,
            "Training data lineage mismatch")
    require(data.get("deployment") is False and run.get("inference_only") is True
            and run.get("selection") == "fixed_attempt_schedule" and run["plan_sha256"] == plan_sha,
            "Fixed diagnostic checkpoint required")
    require(run["attempt_updates"] == MODEL_SPECS[key][1], "Checkpoint attempt mismatch")
    purpose = "train_subset_budget_diagnostic" if key == "source6400" else "train_subset_lr_diagnostic"
    require(data["purpose"] == purpose, "Checkpoint purpose mismatch")
    if key != "source6400":
        branch = "A_keep" if key == "A_keep9600" else "B_reduce"
        require(data["branch"] == run["branch"] == branch, "Checkpoint branch mismatch")
    expected_lr = .0001 if key == "B_reduce9600" else .0003
    require(run["settings"]["learning_rate"] == expected_lr, "Checkpoint LR mismatch")


def cache_for_evaluation(source: dict, cohorts: dict) -> dict:
    """Reject test or unselected IDs before verify_cache can hash/open a cache."""
    require(source["split"] in ("train", "val"), "Test cache access forbidden")
    cohort = "train60" if source["split"] == "train" else "val20"
    require(source["id"] in cohorts[cohort], "Unselected cache access forbidden")
    return verify_cache(source)


def evaluate_clip(model, source: dict, item: dict, *, key: str, cohort: str,
                  forward_fn=forward) -> tuple[list[dict], dict]:
    """One selected clip, three identical-T conditions, beam on normal only."""
    require(not model.training, "Evaluation mode required")
    rows, arrays = [], {}
    normal_probs, normal_text = None, None
    for condition in CONDITIONS:
        transform_name = "shuffle" if condition == "shuffle_42" else condition
        frames, transform = transform_frames(item["frames"], transform_name, sample_id=source["id"], seed=42)
        probs, _ = forward_fn(model, frames, capture=False)
        require(probs.dtype == np.float32 and probs.shape == (source["frames"], 69)
                and np.isfinite(probs).all(), "Invalid model posterior")
        greedy_text = ctc_greedy_decode(probs)
        beam_text = ctc_beam_search_decode(probs, beam_width=10) if condition == "normal" else None
        diagnostics = analyze_ctc(probs, item["label"], beam_text=beam_text if beam_text is not None else "")
        if beam_text is None:
            diagnostics = {name: value for name, value in diagnostics.items() if not name.startswith("beam")}
        labels = torch.tensor(item["label"], dtype=torch.long)
        loss = torch.nn.functional.ctc_loss(torch.from_numpy(probs[:, None, :]), labels,
            torch.tensor([len(probs)]), torch.tensor([len(labels)]), blank=0, reduction="none", zero_infinity=False)
        require(len(labels) > 0 and bool(torch.isfinite(loss).all()), "Invalid CTC loss")
        posterior_key = source["id"] + "__" + condition
        row = {"model": key, "id": source["id"], "split": source["split"], "cohort": cohort,
            "speaker_id": source["speaker_id"], "recording_id": source["recording_id"],
            "length_bin": length_bin(source["frames"]), "reference": source["text"], "frames": source["frames"],
            "condition": condition, "posterior_key": posterior_key, "greedy": prediction_row(source, greedy_text),
            "beam": prediction_row(source, beam_text) if beam_text is not None else None,
            "ctc": diagnostics, "ctc_loss_per_target_token": float(loss.item()/len(labels)),
            "transform": transform, "comparison_to_normal": None}
        if condition == "normal":
            normal_probs, normal_text = probs, greedy_text
        else:
            row["comparison_to_normal"] = compare_posteriors(normal_probs, probs,
                normal_text=normal_text, control_text=greedy_text)
        arrays[posterior_key] = probs
        rows.append(row)
    return rows, arrays


def build_plan() -> tuple[dict, dict]:
    """Read metadata/checkpoints only; selection is frozen before opening any clip."""
    require(sha256_file(MANIFEST) == MANIFEST_SHA, "Manifest changed")
    document, _ = load_manifest(MANIFEST, verify_files=False)
    prior_plan = read_json(PRIOR14 / "plan.json")
    training_ids = prior_plan["cohorts"]["expanded64"]
    selection = select_generalization(document["records"], training_ids, count=60, seed=42)
    require(all(value is (False if name == "test_selected" else True)
                for name, value in selection["disjointness"].items()), "Evaluation overlap detected")
    records = {r["id"]: r for r in document["records"]}
    require(len(selection["cohorts"]["train60"]) == 60 and len(selection["cohorts"]["val20"]) == 20,
            "Stage15 requires exactly train60 and val20")
    require(len({records[i]["speaker_id"] for i in training_ids}) == 8
            and len({records[i]["speaker_id"] for i in selection["cohorts"]["train60"]}) == 8
            and len({records[i]["speaker_id"] for i in selection["cohorts"]["val20"]}) == 2,
            "Stage15 speaker scope mismatch")
    require(set(selection["cohorts"]["val20"]) == {r["id"] for r in document["records"] if r["split"] == "val"},
            "All existing val20 required")
    prior_paths = [PRIOR13 / "plan.json", PRIOR13 / "completed.json", PRIOR14 / "plan.json", PRIOR14 / "completed.json",
                   PRIOR14 / "A_keep/completed.json", PRIOR14 / "B_reduce/completed.json",
                   ROOT / "docs/단계별_개선/14_검증결과.json"]
    completed14 = read_json(PRIOR14 / "completed.json")
    require(completed14["status"] == "completed" and completed14["plan_sha256"] == sha256_file(PRIOR14 / "plan.json"),
            "Stage14 completion/plan mismatch")
    for branch in ("A_keep", "B_reduce"):
        require(sha256_file(PRIOR14 / branch / "completed.json") == completed14["branch_completed_sha256"][branch],
                "Stage14 branch completion changed")
    require(read_json(ROOT / "docs/단계별_개선/14_검증결과.json")["status"] == "completed_and_verified",
            "Verified stage14 required")
    models = {}
    for key, (folder, step, expected_sha) in MODEL_SPECS.items():
        checkpoint = folder / f"step_{step:04d}.pt"
        require(sha256_file(checkpoint) == expected_sha, "Fixed checkpoint hash mismatch")
        completion = read_json(folder / "completed.json")
        prior_plan_path = (PRIOR13 if key == "source6400" else PRIOR14) / "plan.json"
        require(completion["status"] == "completed" and completion["plan_sha256"] == sha256_file(prior_plan_path),
                "Prior completion/plan mismatch")
        snapshot = [s for s in completion["snapshots"] if s["step"] == step]
        require(len(snapshot) == 1 and snapshot[0]["checkpoint_sha256"] == expected_sha, "Prior snapshot mismatch")
        metadata = torch.load(checkpoint, map_location="cpu", weights_only=True)
        validate_checkpoint_metadata(metadata, key, training_ids, sha256_file(prior_plan_path))
        models[key] = {"checkpoint": str(checkpoint), "checkpoint_sha256": expected_sha,
                       "training_data": metadata["training_data"], "training_run": metadata["training_run"],
                       "state_sha256": state_digest(metadata["state_dict"])}
        prior_paths.extend([checkpoint, folder / f"step_{step:04d}/summary.json"])
        require(sha256_file(prior_paths[-1]) == snapshot[0]["summary_sha256"], "Prior snapshot summary changed")
    source_hashes = dict(prior_plan["source_hashes"])
    for path, digest in source_hashes.items():
        require(sha256_file(ROOT / path) == digest, f"Frozen prior source changed: {path}")
    for path in ["scripts/diagnose_generalization.py", "scripts/generalization_selection.py",
                 "scripts/summarize_learning_transition.py", "backend/models/__init__.py"]:
        source_hashes[path] = sha256_file(ROOT / path)
    selected = selection["cohorts"]["train60"] + selection["cohorts"]["val20"]
    plan = {"schema": "lipnet-generalization-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
            "manifest": str(MANIFEST), "manifest_sha256": MANIFEST_SHA, "models": models,
            "training_ids": training_ids, "selection": selection, "settings": SETTINGS,
            "source_hashes": source_hashes,
            "prior_files": {str(p.relative_to(ROOT)).replace("\\", "/"): sha256_file(p) for p in prior_paths},
            "records": [records[i] for i in selected],
            "interpretation": {"val20": "Previously viewed development evaluation, not untouched final test",
                "train60": "Unseen by this 64-clip checkpoint lineage; familiar speakers and new recordings",
                "cohort_difference": "Speaker, content, recording and length differ; not isolated speaker effect",
                "controls": "Same T; limited corruption sensitivity evidence, not a recognition proof",
                "no_population_accuracy": True, "no_webcam_evaluation": True,
                "training_performed": False, "test_cache_access": False, "deployment": False}}
    return plan, records


def run_model(key: str, plan: dict, records: dict, cached: dict, out: Path, plan_sha: str) -> dict:
    spec = plan["models"][key]
    require(sha256_file(Path(spec["checkpoint"])) == spec["checkpoint_sha256"], "Checkpoint changed before evaluation")
    model = load_model_checkpoint(spec["checkpoint"]).to("cuda").eval()
    before = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    before_digest = state_digest(before)
    require(before_digest == spec["state_sha256"], "Loaded model state mismatch")
    before_rng = capture_rng_state()
    folder = out / key
    folder.mkdir()
    arrays, row_count, beam_count = {}, 0, 0
    started = time.perf_counter()
    selected = plan["selection"]["cohorts"]["train60"] + plan["selection"]["cohorts"]["val20"]
    with (folder / "rows.jsonl").open("x", encoding="utf-8") as stream:
        for index, sample_id in enumerate(selected):
            source = records[sample_id]
            cohort = "train60" if source["split"] == "train" else "val20"
            rows, probabilities = evaluate_clip(model, source, cached[sample_id], key=key, cohort=cohort)
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            arrays.update(probabilities)
            row_count += len(rows)
            beam_count += sum(row["beam"] is not None for row in rows)
            stream.flush()
            if (index + 1) % 10 == 0:
                print(json.dumps({"model": key, "clips": index+1, "forwards": row_count,
                                  "seconds": round(time.perf_counter()-started, 1)}), flush=True)
    np.savez_compressed(folder / "posteriors.npz", **arrays)
    state_audit = compare_state_dicts(before, model.state_dict())
    rng_audit = compare_nested(before_rng, capture_rng_state())
    require(state_audit["exact"] and rng_audit["exact"], "Inference mutated model state or RNG")
    require(all(not module.training for module in model.modules()), "Evaluation mode changed")
    require(sha256_file(Path(spec["checkpoint"])) == spec["checkpoint_sha256"], "Checkpoint changed after evaluation")
    report = {"status": "completed", "model": key, "plan_sha256": plan_sha,
        "checkpoint_sha256": spec["checkpoint_sha256"], "clips": len(selected),
        "forwards": row_count, "beam_rows": beam_count,
        "rows_sha256": sha256_file(folder / "rows.jsonl"), "posteriors_sha256": sha256_file(folder / "posteriors.npz"),
        "weights_unchanged": True, "rng_unchanged": True, "evaluation_mode_unchanged": True,
        "state_before_sha256": before_digest, "state_after_sha256": state_digest(model.state_dict()),
        "state_audit": state_audit, "rng_audit": rng_audit,
        "training_performed": False, "test_cache_access": False,
        "seconds": time.perf_counter()-started}
    write_json(folder / "report.json", report)
    del model, arrays, before
    torch.cuda.empty_cache()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    torch.set_num_threads(2)
    require(torch.cuda.is_available(), "CUDA required")
    seed_training(42)
    out = Path(args.out).resolve()
    require(out.is_relative_to(ROOT / "results") and not out.exists(), "New results subfolder required")
    started = time.perf_counter()
    plan, records = build_plan()
    out.mkdir(parents=True)
    write_json(out / "plan.json", plan)
    plan_sha = sha256_file(out / "plan.json")
    print(json.dumps({"plan_frozen": plan_sha, "cohorts": {k: len(v) for k, v in plan["selection"]["cohorts"].items()},
                      "quotas": plan["selection"]["quotas"]}), flush=True)
    cached, opened = {}, []
    for sample_id in plan["selection"]["cohorts"]["train60"] + plan["selection"]["cohorts"]["val20"]:
        cached[sample_id] = cache_for_evaluation(records[sample_id], plan["selection"]["cohorts"])
        opened.append(sample_id)
    write_json(out / "cache_audit.json", {"plan_sha256": plan_sha, "cache_ids_opened": opened,
        "cache_sha256": {i: records[i]["cache_sha256"] for i in opened}, "test_cache_access": False})
    reports = {key: run_model(key, plan, records, cached, out, plan_sha) for key in MODEL_SPECS}
    require(sha256_file(out / "plan.json") == plan_sha, "Frozen plan changed")
    for path, digest in {**plan["source_hashes"], **plan["prior_files"]}.items():
        require(sha256_file(ROOT / path) == digest, f"Frozen file changed: {path}")
    forwards, beams = sum(r["forwards"] for r in reports.values()), sum(r["beam_rows"] for r in reports.values())
    require(forwards == 720 and beams == 240, "Evaluation scope mismatch")
    completed = {"status": "completed", "plan_sha256": plan_sha,
        "models": {key: sha256_file(out / key / "report.json") for key in reports},
        "cache_audit_sha256": sha256_file(out / "cache_audit.json"), "cache_ids_opened": opened,
        "forwards": forwards, "beam_rows": beams, "training_performed": False,
        "test_cache_access": False, "source_hashes_unchanged": True, "seconds": time.perf_counter()-started,
        "environment": {"torch": torch.__version__, "cuda": torch.version.cuda,
                        "gpu": torch.cuda.get_device_name(0), "numpy": np.__version__, "python": sys.version}}
    write_json(out / "completed.json", completed)
    print(json.dumps({"status": "completed", "forwards": forwards, "beam_rows": beams,
                      "seconds": completed["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
