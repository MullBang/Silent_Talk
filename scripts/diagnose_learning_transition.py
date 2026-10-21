"""Stage12: fixed train-only branches with AMP/gradient/update telemetry.

This is an inference-weight restart experiment, not optimizer-state resume.
No validation/test cache is opened, no best checkpoint is selected or deployed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "backend"), str(ROOT / "scripts")]
from models.cache import load_clip
from models.baseline import LipNetBaseline
from models.dataset_manifest import load_manifest, sha256_file
from models.decoder import ctc_beam_search_decode, ctc_greedy_decode
from models.registry import load_model_checkpoint
from train import collate, ctc_training_loss
from diagnose_temporal_use import forward, prediction_row
from temporal_diagnostics import analyze_ctc, compare_posteriors, transform_frames
from transition_diagnostics import (select_transition_cohorts, make_training_schedule,
    parameter_groups, gradient_metrics, snapshot_parameters, parameter_update_metrics)

MANIFEST = ROOT / "data_cache/datasets/aihub_timed_v4/manifest.json"
INITIAL = ROOT / "results/step9-diagnostics/compact-short16-v4/last.pt"
INITIAL_SHA = "a426118a54b4214d7a53fc31ea02cf538cc69ccab7e025600cd62d2cf2226613"
MANIFEST_SHA = "214ce935150b367a15f24f0896901d34c54c6d4fe73d493a9d397c4c94fe59a6"
STEPS = [0, 1, 5, 10, 25, 50, 100, 200, 400, 800, 1600]
CONTROL_STEPS = [0, 100, 400, 1600]
BEAM_STEPS = [0, 1600]
BRANCHES = {"A_keep16": "short16", "B_expand64": "expanded64"}
SOURCE_PATHS = ["scripts/diagnose_learning_transition.py", "scripts/transition_diagnostics.py",
    "scripts/train.py", "scripts/diagnose_temporal_use.py", "scripts/temporal_diagnostics.py",
    "scripts/test_model.py", "backend/models/baseline.py", "backend/models/registry.py",
    "backend/models/cache.py", "backend/models/dataset_manifest.py", "backend/models/tokenizer.py",
    "backend/models/decoder.py", "backend/config.py"]


def require(condition: bool, message: str) -> None:
    """Reject a broken experiment contract even under optimized Python."""
    if not condition:
        raise ValueError(message)


def write_json(path: Path, value: dict) -> None:
    """Write a new UTF-8 artifact, refusing replacement and nonfinite JSON."""
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def read_json(path: Path) -> dict:
    """Read a UTF-8 result document."""
    return json.loads(path.read_text(encoding="utf-8"))


def source_hashes() -> dict:
    """Hash all source files directly involved in inference and training."""
    return {path: sha256_file(ROOT / path) for path in SOURCE_PATHS}


def seed_training(seed: int) -> None:
    """Reset each branch's RNG and common CUDA precision settings."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


@contextmanager
def preserve_rng_and_mode(model: torch.nn.Module):
    """Evaluation cannot advance dropout RNG or leave the model in eval mode."""
    python, numpy = random.getstate(), np.random.get_state()
    cpu = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    training = model.training
    try:
        model.eval()
        yield
    finally:
        random.setstate(python)
        np.random.set_state(numpy)
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)
        model.train(training)


def load_initial(cohorts: dict) -> LipNetBaseline:
    """Load only the verified compact short16 inference checkpoint."""
    require(sha256_file(INITIAL) == INITIAL_SHA, "Initial checkpoint changed")
    model = load_model_checkpoint(str(INITIAL))
    data, run = model.training_data, model.training_run
    require(type(model) is LipNetBaseline, "Expected compact model")
    require(data.get("mode") == "smoke_only" and data.get("purpose") == "train_subset_overfit"
            and data.get("manifest_sha256") == MANIFEST_SHA
            and data.get("sample_ids") == cohorts["short16"], "Initial training provenance mismatch")
    require(run.get("overfit_only") is True and run.get("step") == 1600,
            "Expected original short16 step1600 inference weights")
    return model.to("cuda")


def load_train_cache(row: dict) -> dict:
    """Verify and open one selected train cache; reject other splits first."""
    require(row["split"] == "train", "Only train cache access is allowed")
    path = MANIFEST.parent / row["cache_path"]
    require(sha256_file(path) == row["cache_sha256"], "Cache hash mismatch")
    item = load_clip(path)
    require(item["text"] == row["text"] and item["source_text"] == row["source_text"]
            and len(item["frames"]) == row["frames"], "Cache content mismatch")
    with np.load(path, allow_pickle=False) as data:
        for name in ("speaker_id", "utterance_id", "recording_id", "angle", "source_partition",
                     "roi_method", "label_sha256", "temporal_method", "video_fps", "label_fps"):
            require(str(data[name]) == str(row[name]), f"Cache metadata mismatch: {name}")
    return item


def create_plan(out: Path) -> dict:
    """Freeze the cohorts, schedule and source hashes before any new training."""
    require(not out.exists(), "Output exists; refusing to overwrite or resume")
    document, metadata = load_manifest(MANIFEST, verify_files=False)
    require(metadata["manifest_sha256"] == MANIFEST_SHA, "Manifest changed")
    cohorts = select_transition_cohorts(document["records"], seed=42)
    require(sha256_file(INITIAL) == INITIAL_SHA, "Initial checkpoint changed")
    schedules = {key: make_training_schedule(cohorts[cohort]) for key, cohort in BRANCHES.items()}
    plan = {"schema": "lipnet-learning-transition-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(MANIFEST), "manifest_sha256": MANIFEST_SHA,
        "initial_checkpoint": str(INITIAL), "initial_checkpoint_sha256": INITIAL_SHA,
        "initial_source": "compact short16 step1600, inference weights only; original lr0.001",
        "source_hashes": source_hashes(), "cohorts": cohorts, "schedules": schedules,
        "records": [r for r in document["records"] if r["id"] in cohorts["expanded64"]],
        "settings": {"attempt_updates": 1600, "batch_size": 2, "learning_rate": .0003,
            "adam_betas": [.9, .999], "adam_eps": 1e-8, "weight_decay": 0.,
            "seed": 42, "grad_clip": 1., "grad_scaler_init_scale": 1024.,
            "precision": "amp_fp16_ctc_fp32", "ctc_reduction": "mean",
            "ctc_zero_infinity": False, "evaluation_precision": "fp32_unpadded_B1",
            "normal_steps": STEPS, "control_steps": CONTROL_STEPS, "beam_steps": BEAM_STEPS,
            "beam_width": 10, "shuffle_seed": 42, "fresh_optimizer_per_branch": True,
            "sampler": "epoch-full-index-shuffle-v1; differs from prior fixed length-pair sampler",
            "early_stopping": False, "validation_selection": False,
            "torch_threads": 2, "cudnn_deterministic": True, "cudnn_benchmark": False, "tf32": False},
        "scope": {"train_only": True, "validation_cache_access": False, "test_cache_access": False,
            "generalization_evaluation": False, "deployment": False,
            "A_additional48": "not trained in A; diagnostic subset of existing train split",
            "B_additional48": "trained in B; not a held-out evaluation set"},
        "metric_interpretation": ["Fixed attempted updates, unequal per-ID exposures: A200/B50.",
            "Gradient norms after AMP unscale, before and after clipping; actual parameter deltas.",
            "Static/shuffle controls retain sequence length and appearance; sensitivity is not correctness.",
            "First observed old16 exact<16 is descriptive, not an early stopping trigger.",
            "Expansion changes length, text, speaker mix and exposure; this does not isolate dataset size."],
        "nonfinite_policy": "Nonfinite loss aborts. Nonfinite gradient: no clip; scaler skips update and lowers scale. No performance-based abort."}
    out.mkdir(parents=True)
    write_json(out / "plan.json", plan)
    return plan


@torch.inference_mode()
def evaluate_snapshot(model: torch.nn.Module, plan: dict, cache: dict, records: dict, step: int,
                      folder: Path, checkpoint_sha: str) -> dict:
    """Save actual posteriors and outputs; no fabricated beam values in intermediate rows."""
    folder.mkdir()
    started = time.perf_counter()
    conditions = ["normal", "static_middle", "shuffle"] if step in CONTROL_STEPS else ["normal"]
    beam_performed = step in BEAM_STEPS
    arrays, rows = {}, []
    with preserve_rng_and_mode(model):
        for index, sample_id in enumerate(plan["cohorts"]["expanded64"]):
            source, item = records[sample_id], cache[sample_id]
            normal, normal_text = None, None
            for condition in conditions:
                frames, transform = transform_frames(item["frames"], condition, sample_id=sample_id, seed=42)
                lp, _ = forward(model, frames, capture=False)
                greedy = ctc_greedy_decode(lp)
                beam = ctc_beam_search_decode(lp, beam_width=10) if beam_performed else None
                # Supplying an unused placeholder avoids analyze_ctc's automatic beam decode.
                # Strip every beam field when no beam search actually ran.
                diagnostics = analyze_ctc(lp, item["label"], beam_text=beam if beam_performed else "")
                if not beam_performed:
                    diagnostics = {k: v for k, v in diagnostics.items() if not k.startswith("beam")}
                target = torch.tensor(item["label"], dtype=torch.long)
                loss = torch.nn.functional.ctc_loss(torch.from_numpy(lp[:, None, :]), target,
                    torch.tensor([len(lp)]), torch.tensor([len(target)]), blank=0, reduction="mean", zero_infinity=False)
                require(bool(torch.isfinite(loss)), "Nonfinite evaluation loss")
                name = "shuffle_42" if condition == "shuffle" else condition
                key = f"sample_{index:03d}_{name}"
                arrays[key] = lp
                row = {"id": sample_id, "step": step, "split": "train", "condition": name,
                    "cohort": "short16" if sample_id in plan["cohorts"]["short16"] else "additional48",
                    "speaker_id": source["speaker_id"], "reference": source["text"], "frames": len(lp),
                    "greedy": prediction_row(source, greedy),
                    "beam": prediction_row(source, beam) if beam_performed else None,
                    "ctc": diagnostics, "ctc_loss_per_target_token": loss.item(),
                    "posterior_key": key, "transform": transform, "comparison_to_normal": None}
                if condition == "normal":
                    normal, normal_text = lp, greedy
                else:
                    row["comparison_to_normal"] = compare_posteriors(normal, lp,
                        normal_text=normal_text, control_text=greedy)
                rows.append(row)
    with (folder / "rows.jsonl").open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    np.savez_compressed(folder / "posteriors.npz", **arrays)
    summary = {"step": step, "beam_performed": beam_performed,
        "conditions": ["shuffle_42" if c == "shuffle" else c for c in conditions],
        "rows": len(rows), "checkpoint_sha256": checkpoint_sha,
        "rows_sha256": sha256_file(folder / "rows.jsonl"),
        "posteriors_sha256": sha256_file(folder / "posteriors.npz"),
        "seconds": time.perf_counter() - started, "cohorts": {}}
    for cohort in ("short16", "additional48"):
        normal_rows = [r for r in rows if r["cohort"] == cohort and r["condition"] == "normal"]
        summary["cohorts"][cohort] = {"count": len(normal_rows),
            "greedy_exact": sum(r["greedy"]["exact"] for r in normal_rows),
            "greedy_mean_cer": float(np.mean([r["greedy"]["cer"] for r in normal_rows]))}
    write_json(folder / "summary.json", summary)
    print(json.dumps({"evaluation_step": step, **summary["cohorts"], "seconds": summary["seconds"]}), flush=True)
    return summary


def optimizer_attempt(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                      scaler: torch.amp.GradScaler, batch: tuple, groups: dict,
                      device: torch.device) -> dict:
    """A normal training update with read-only measurements surrounding clipping/step."""
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    loss = ctc_training_loss(model, batch, device, use_amp=True)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    preclip = gradient_metrics(groups)
    finite = preclip["total"]["all_present_gradients_finite"]
    if finite:
        returned = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        coefficient = min(1., 1. / (float(returned) + 1e-6))
    else:
        # GradScaler already recorded inf during unscale and will skip this update.
        coefficient = None
    postclip = gradient_metrics(groups)
    before = snapshot_parameters(groups)
    old_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    new_scale = scaler.get_scale()
    skipped = new_scale < old_scale
    delta = parameter_update_metrics(groups, before)
    require(delta["total"]["all_parameters_finite"], "Nonfinite parameters")
    require(not skipped or delta["total"]["absolute_l2"] == 0., "AMP skip changed parameters")
    require(finite != skipped, "Gradient finiteness and AMP skip disagree")
    require(not finite or preclip["total"]["missing_grad_tensors"] == 0, "Missing model gradients")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return {"loss": float(loss.item()), "preclip": preclip, "postclip": postclip,
        "clip_coefficient": coefficient, "clip_applied": finite,
        "scale_before": old_scale, "scale_after": new_scale, "skipped": skipped,
        "parameter_update": delta, "learning_rate": optimizer.param_groups[0]["lr"],
        "seconds": time.perf_counter() - started}


def run_branch(key: str, out: Path, plan: dict, cache: dict, records: dict) -> dict:
    """Execute one predeclared branch with a fresh optimizer and fixed attempts."""
    seed_training(42)
    model = load_initial(plan["cohorts"]).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=.0003)
    scaler = torch.amp.GradScaler("cuda", enabled=True, init_scale=1024)
    groups = parameter_groups(model)
    torch.cuda.reset_peak_memory_stats()
    folder = out / key
    folder.mkdir()
    schedule = plan["schedules"][key]
    provenance = {"mode": "smoke_only", "purpose": "train_subset_transition_diagnostic",
        "manifest_sha256": MANIFEST_SHA, "sample_ids": schedule["sample_ids"],
        "generalization_evaluation": False, "deployment": False}
    updates, skips = 0, 0
    exposures, successful = Counter(), Counter({s: 0 for s in schedule["sample_ids"]})
    snapshots = []
    started = time.perf_counter()
    with (folder / "telemetry.jsonl").open("x", encoding="utf-8") as stream:
        for entry in schedule["batches"]:
            result = optimizer_attempt(model, optimizer, scaler,
                collate([cache[s] for s in entry["ids"]]), groups, torch.device("cuda"))
            skips += int(result["skipped"])
            updates += int(not result["skipped"])
            exposures.update(entry["ids"])
            if not result["skipped"]:
                successful.update(entry["ids"])
            result.update(entry)
            result.update(optimizer_steps=updates, skipped_optimizer_steps=skips,
                frames=[records[s]["frames"] for s in entry["ids"]],
                target_lengths=[len(cache[s]["label"]) for s in entry["ids"]])
            stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            step = entry["attempt"]
            if step in STEPS:
                checkpoint = folder / f"step_{step:04d}.pt"
                model.save_checkpoint(str(checkpoint), training_data=provenance, training_run={
                    "transition_diagnostic": True, "branch": key, "attempt_updates": step,
                    "optimizer_steps": updates, "skipped_optimizer_steps": skips,
                    "plan_sha256": sha256_file(out / "plan.json"),
                    "initial_checkpoint_sha256": INITIAL_SHA, "inference_only": True,
                    "selection": "fixed_attempt_schedule", "settings": plan["settings"]})
                sha = sha256_file(checkpoint)
                snapshot = folder / f"step_{step:04d}"
                evaluate_snapshot(model, plan, cache, records, step, snapshot, sha)
                snapshots.append({"step": step, "folder": str(snapshot.relative_to(out)),
                    "checkpoint": str(checkpoint.relative_to(out)), "checkpoint_sha256": sha,
                    "summary_sha256": sha256_file(snapshot / "summary.json")})
            if step % 100 == 0:
                print(json.dumps({"branch": key, "attempt": step, "loss": result["loss"],
                    "updates": updates, "skips": skips, "elapsed_seconds": time.perf_counter()-started}), flush=True)
    require(dict(exposures) == schedule["exposures"], "Exposure schedule mismatch")
    report = {"branch": key, "status": "completed", "attempt_updates": len(schedule["batches"]),
        "optimizer_steps": updates, "skipped_optimizer_steps": skips,
        "attempted_exposures": dict(exposures), "successful_update_exposures": dict(successful),
        "snapshots": snapshots, "step0_folder": "common/step_0000",
        "telemetry_sha256": sha256_file(folder / "telemetry.jsonl"),
        "last_checkpoint_sha256": snapshots[-1]["checkpoint_sha256"],
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "seconds": time.perf_counter()-started}
    write_json(folder / "completed.json", report)
    del optimizer, model, groups
    torch.cuda.empty_cache()
    return report


def run(out: Path) -> None:
    """Validate selected inputs, evaluate the shared start, then run both branches."""
    require(torch.cuda.is_available(), "CUDA is required for the fixed protocol")
    torch.set_num_threads(2)
    plan = create_plan(out)
    started = time.perf_counter()
    records = {r["id"]: r for r in plan["records"]}
    cache = {s: load_train_cache(records[s]) for s in plan["cohorts"]["expanded64"]}
    seed_training(42)
    model = load_initial(plan["cohorts"])
    (out / "common").mkdir()
    common = evaluate_snapshot(model, plan, cache, records, 0, out / "common/step_0000", INITIAL_SHA)
    require(common["cohorts"]["short16"]["greedy_exact"] == 16, "Initial short16 reproduction failed")
    del model
    torch.cuda.empty_cache()
    branches = {key: run_branch(key, out, plan, cache, records) for key in BRANCHES}
    require(source_hashes() == plan["source_hashes"], "Running source changed during experiment")
    require(sha256_file(MANIFEST) == MANIFEST_SHA and sha256_file(INITIAL) == INITIAL_SHA,
            "Source artifacts changed during experiment")
    write_json(out / "completed.json", {"status": "completed", "plan_sha256": sha256_file(out / "plan.json"),
        "common_summary_sha256": sha256_file(out / "common/step_0000/summary.json"),
        "branch_completed_sha256": {key: sha256_file(out / key / "completed.json") for key in BRANCHES},
        "branches": {key: {k: v for k, v in value.items() if k in (
            "attempt_updates", "optimizer_steps", "skipped_optimizer_steps", "seconds")} for key, value in branches.items()},
        "environment": {"torch": str(torch.__version__), "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(), "numpy": np.__version__, "python": sys.version},
        "seconds": time.perf_counter()-started, "train_cache_ids_opened": list(cache),
        "validation_cache_access": False, "test_cache_access": False, "deployment": False})


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results/step12-transition/run01")
    args = parser.parse_args()
    run(args.out.resolve())
