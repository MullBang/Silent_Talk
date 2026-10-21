"""Independently audit the fixed stage17 256-versus512 continuation experiment.

Saved FP32 posteriors are decoded and scored on CPU, including every declared
normal beam. No model forward, new training or test-cache access occurs here.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "backend"), str(ROOT / "scripts")]
from models import GRAPHEME_VOCAB
from models.dataset_manifest import load_manifest, sha256_file
from models.decoder import ctc_beam_search_decode
from models.tokenizer import encode_text, validate_tokenizer_metadata
from diagnose_temporal_use import prediction_row, length_bin
from temporal_diagnostics import analyze_ctc, compare_posteriors, transform_frames
from transition_diagnostics import make_training_schedule, GROUP_PREFIXES
from learning_budget_state import compare_nested, compare_state_dicts
from summarize_learning_transition import (
    require, finite_tree, read_json, read_rows, compare, describe, checked_hash,
    inside, summarize_group, validate_telemetry,
)
from summarize_learning_rate import parameter_norms, scaler_after, validate_full_state as validate_prior_full_state

BASE_STEP = 12800
STEPS = [0, 1600, 3200, 6400]
ENDPOINTS = [0, 6400]
BRANCHES = {"A_keep256": "existing256", "B_expand512": "expanded512"}
COHORTS = ["original64", "additional192", "additional256", "train60", "val20"]
HELDOUT = ["train60", "val20"]
CONDITIONS = ["normal", "static_middle", "shuffle_42"]
MANIFEST_SHA = "214ce935150b367a15f24f0896901d34c54c6d4fe73d493a9d397c4c94fe59a6"


def exact(actual, expected, where: str) -> None:
    require(compare_nested(expected, actual)["exact"], f"Exact state mismatch: {where}")


def ordered_ids(plan: dict) -> list[str]:
    return [sample_id for cohort in COHORTS for sample_id in plan["cohorts"][cohort]]


def snapshot_pairs(plan: dict, step: int) -> list[list[str]]:
    require(step in STEPS, "Undeclared snapshot step")
    controls = set(s for c in HELDOUT for s in plan["cohorts"][c]) if step in ENDPOINTS else set()
    return [[sample_id, condition] for sample_id in ordered_ids(plan)
            for condition in (CONDITIONS if sample_id in controls else ["normal"])]


def validate_snapshot(folder: Path, plan: dict, records: dict, branch: str,
                      step: int, checkpoint_sha: str) -> tuple[dict, list[dict]]:
    """Validate the complete predeclared grid, then recompute all saved outputs."""
    require(branch in BRANCHES, "Unknown expansion branch")
    stored = read_json(folder / "summary.json")
    expected_pairs = snapshot_pairs(plan, step)
    controls = HELDOUT if step in ENDPOINTS else []
    conditions = CONDITIONS if controls else ["normal"]
    for field, value in {"step": step, "cumulative_attempt": BASE_STEP + step, "branch": branch,
        "rows": len(expected_pairs), "checkpoint_sha256": checkpoint_sha, "conditions": conditions,
        "control_cohorts": controls, "beam_performed": step in ENDPOINTS}.items():
        compare(stored[field], value, f"Snapshot summary.{field}")
    require(stored["seconds"] >= 0, "Negative snapshot duration")
    checked_hash(folder / "rows.jsonl", stored["rows_sha256"])
    checked_hash(folder / "posteriors.npz", stored["posteriors_sha256"])
    rows = read_rows(folder / "rows.jsonl")
    compare([[r["id"], r["condition"]] for r in rows], expected_pairs, "Exact snapshot grid and order")
    membership = {s: c for c in COHORTS for s in plan["cohorts"][c]}
    normal = {}
    beam_count = 0
    with np.load(folder / "posteriors.npz", allow_pickle=False) as posteriors:
        require(len({r["posterior_key"] for r in rows}) == len(rows)
                and set(posteriors.files) == {r["posterior_key"] for r in rows}, "Posterior inventory mismatch")
        for row in rows:
            sample_id, condition = row["id"], row["condition"]
            source = records[sample_id]
            require(source["split"] in ("train", "val"), "Test posterior is forbidden")
            for field, value in {"step": step, "cumulative_attempt": BASE_STEP + step, "branch": branch,
                "split": source["split"], "cohort": membership[sample_id], "speaker_id": source["speaker_id"],
                "recording_id": source["recording_id"], "reference": source["text"], "frames": source["frames"],
                "length_bin": length_bin(source["frames"]), "posterior_key": sample_id + "__" + condition}.items():
                compare(row[field], value, f"Snapshot row.{field}")
            logp = posteriors[row["posterior_key"]]
            require(logp.dtype == np.float32 and logp.shape == (source["frames"], 69)
                    and bool(np.isfinite(logp).all()), "Invalid finite FP32 unpadded posterior")
            labels = np.asarray(encode_text(source["text"]), dtype=np.int64)
            has_beam = step in ENDPOINTS and condition == "normal"
            require((row["beam"] is not None) == has_beam, "Beam schedule mismatch")
            beam = ctc_beam_search_decode(logp, beam_width=10) if has_beam else ""
            beam_count += int(has_beam)
            diagnostic = analyze_ctc(logp, labels, beam_text=beam)
            if not has_beam:
                diagnostic = {name: value for name, value in diagnostic.items() if not name.startswith("beam")}
            compare(row["ctc"], diagnostic, "Recomputed CTC diagnostics")
            compare(row["greedy"], prediction_row(source, diagnostic["greedy_text"]), "Recomputed greedy score")
            if has_beam:
                compare(row["beam"], prediction_row(source, beam), "Independently decoded beam score")
            loss = torch.nn.functional.ctc_loss(torch.from_numpy(logp[:, None, :]), torch.from_numpy(labels),
                torch.tensor([len(logp)]), torch.tensor([len(labels)]), blank=0, reduction="mean", zero_infinity=False)
            compare(row["ctc_loss_per_target_token"], float(loss), "Recomputed CTC loss", atol=2e-5, rtol=2e-5)
            if condition == "normal":
                require(row["comparison_to_normal"] is None, "Normal row has control comparison")
                normal[sample_id] = (logp, diagnostic["greedy_text"])
            else:
                normal_logp, normal_text = normal[sample_id]
                compare(row["comparison_to_normal"], compare_posteriors(normal_logp, logp,
                    normal_text=normal_text, control_text=diagnostic["greedy_text"]), "Recomputed normal/control change")
            _, transform = transform_frames(np.zeros((source["frames"], 1, 1, 3), np.uint8),
                "shuffle" if condition == "shuffle_42" else condition, sample_id=sample_id, seed=42)
            transform["input_shape"] = [source["frames"], 96, 96, 3]
            changed = row["transform"]["pixel_changed_frame_count"]
            require(type(changed) is int and 0 <= changed <= source["frames"], "Invalid changed-pixel count")
            if condition == "normal":
                require(changed == 0, "Normal transform changed pixels")
            transform["pixel_changed_frame_count"] = changed
            compare(row["transform"], transform, "Reconstructed transform indices")
    cohorts, by_speaker, by_length = {}, {}, {}
    for cohort in COHORTS:
        values = [r for r in rows if r["cohort"] == cohort]
        cohorts[cohort] = {condition: summarize_group([r for r in values if r["condition"] == condition])
            for condition in conditions if any(r["condition"] == condition for r in values)}
        normal_rows = [r for r in values if r["condition"] == "normal"]
        for field, destination in (("speaker_id", by_speaker), ("length_bin", by_length)):
            destination[cohort] = {group: {"sample_ids": [r["id"] for r in normal_rows if r[field] == group],
                "summary": summarize_group([r for r in normal_rows if r[field] == group])}
                for group in sorted({r[field] for r in normal_rows})}
    return {**stored, "beam_rows": beam_count, "cohorts": cohorts,
            "normal_by_speaker": by_speaker, "normal_by_length": by_length}, rows


def validate_extension_telemetry(rows: list[dict], schedule: dict, records: dict,
                                 settings: dict, source: dict, initial_counters: dict) -> dict:
    """Verify local counters while binding cumulative counters to the source."""
    # Stage16's saved counters are LOCAL to stage16. The separately verified
    # merged history is the cumulative reference for the actual Adam counter.
    initial = initial_counters
    result = validate_telemetry(rows, schedule, records,
        {**settings, "grad_scaler_init_scale": source["scaler_state_dict"]["scale"]})
    for row in rows:
        compare(row["cumulative_attempt"], BASE_STEP + row["attempt"], "Cumulative attempt")
        compare(row["cumulative_optimizer_steps"], initial["optimizer_steps"] + row["optimizer_steps"],
            "Cumulative optimizer updates")
        compare(row["cumulative_skipped_optimizer_steps"], initial["skipped_optimizer_steps"] + row["skipped_optimizer_steps"],
            "Cumulative skipped updates")
        compare(row["raw_frame_presentations"], sum(records[s]["frames"] for s in row["ids"]), "Unpadded frames per attempt")
        compare(row["padded_frame_presentations"], max(records[s]["frames"] for s in row["ids"]) * len(row["ids"]),
            "Padded frames per attempt")
    first_norms = parameter_norms(source["model_state_dict"])
    for group, norm in first_norms.items():
        entry = rows[0]["parameter_update"]["total"] if group == "total" else rows[0]["parameter_update"]["groups"][group]
        compare(entry["before_l2"], norm, "First local update connects to source weights", rtol=2e-6)
    actual_frames = sum(sum(records[s]["frames"] for s in r["ids"]) for r in rows)
    padded_frames = sum(max(records[s]["frames"] for s in r["ids"]) * len(r["ids"]) for r in rows)
    successful_frames = sum(sum(records[s]["frames"] for s in r["ids"]) for r in rows if not r["skipped"])
    cumulative = {"optimizer_steps": initial["optimizer_steps"] + result["optimizer_steps"],
        "skipped_optimizer_steps": initial["skipped_optimizer_steps"] + result["skipped_optimizer_steps"],
        "attempted_exposures": {s: initial["attempted_exposures"].get(s, 0) + result["attempted_exposures"][s]
            for s in schedule["sample_ids"]},
        "successful_update_exposures": {s: initial["successful_update_exposures"].get(s, 0) + result["successful_update_exposures"][s]
            for s in schedule["sample_ids"]}}
    return {"extension": result, "cumulative": cumulative,
        "final_scaler_state": scaler_after(rows, source["scaler_state_dict"]),
        "compute": {"attempted_clip_presentations": sum(len(r["ids"]) for r in rows),
            "attempted_unpadded_frames": actual_frames, "attempted_padded_frames": padded_frames,
            "successful_update_unpadded_frames": successful_frames,
            "padding_fraction": (padded_frames - actual_frames) / padded_frames,
            "training_seconds": sum(r["seconds"] for r in rows),
            "note": "Frame presentations are measured workload proxies, not FLOP measurements; skipped attempts still execute forward/backward."}}


def merge_counters(history: dict, local: dict) -> dict:
    """Merge source history and one completed local stage without mutating either."""
    fields = {"optimizer_steps", "skipped_optimizer_steps", "attempted_exposures", "successful_update_exposures"}
    require(set(history) == set(local) == fields, "Counter inventory mismatch")
    result = {}
    for name in ("optimizer_steps", "skipped_optimizer_steps"):
        require(all(type(value[name]) is int and value[name] >= 0 for value in (history, local)), "Invalid update counter")
        result[name] = history[name] + local[name]
    for value in (history, local):
        require(set(value["attempted_exposures"]) == set(value["successful_update_exposures"]), "Exposure key mismatch")
        for sample_id, attempted in value["attempted_exposures"].items():
            successful = value["successful_update_exposures"][sample_id]
            require(isinstance(sample_id, str) and sample_id and type(attempted) is int and type(successful) is int
                    and 0 <= successful <= attempted, "Invalid sample exposure counter")
    for name in ("attempted_exposures", "successful_update_exposures"):
        require(set(history[name]).issubset(local[name]), "Earlier training IDs missing from source-local counters")
        ids = sorted(local[name])
        result[name] = {sample_id: history[name].get(sample_id, 0) + local[name].get(sample_id, 0) for sample_id in ids}
    return result


def paired_normal(before_rows: list[dict], after_rows: list[dict]) -> dict:
    """Matched-clip differences, with after-minus-before sign, for each cohort."""
    result = {}
    for cohort in COHORTS:
        a = {r["id"]: r for r in before_rows if r["cohort"] == cohort and r["condition"] == "normal"}
        b = {r["id"]: r for r in after_rows if r["cohort"] == cohort and r["condition"] == "normal"}
        require(a.keys() == b.keys() and bool(a), "Unmatched paired evaluation clips")
        metrics = {}
        for decoder in ("greedy", "beam"):
            require(all(a[s][decoder] is not None and b[s][decoder] is not None for s in a), "Missing paired endpoint decoder")
            deltas = [b[s][decoder]["cer"] - a[s][decoder]["cer"] for s in a]
            metrics[decoder] = {"cer_delta": describe(deltas),
                "improved_count": sum(d < -1e-12 for d in deltas),
                "unchanged_count": sum(abs(d) <= 1e-12 for d in deltas),
                "worsened_count": sum(d > 1e-12 for d in deltas),
                "text_agreement_count": sum(a[s][decoder]["prediction"] == b[s][decoder]["prediction"] for s in a),
                "before_exact": sum(a[s][decoder]["exact"] for s in a),
                "after_exact": sum(b[s][decoder]["exact"] for s in b)}
        result[cohort] = {"count": len(a), **metrics}
    return result


def validate_plan(plan: dict) -> tuple[dict, dict]:
    """Rebuild metadata selection and schedules without using any output score."""
    from expansion_selection import select_expansion
    require(plan["schema"] == "lipnet-training-expansion-v1", "Unsupported stage17 schema")
    checked_hash(Path(plan["manifest"]), MANIFEST_SHA)
    compare(plan["manifest_sha256"], MANIFEST_SHA, "Timed manifest")
    required_sources = {"scripts/diagnose_training_expansion.py", "scripts/expansion_selection.py",
        "scripts/learning_budget_state.py", "scripts/transition_diagnostics.py", "scripts/temporal_diagnostics.py",
        "backend/models/decoder.py", "backend/models/tokenizer.py"}
    require(required_sources.issubset(plan["source_hashes"]), "Missing frozen source dependencies")
    for field in ("source_hashes", "prior_files"):
        require(bool(plan[field]), "Missing frozen artifacts")
        for relative, digest in plan[field].items():
            checked_hash(inside(ROOT, relative), digest)
    p16_path = ROOT / "results/step16-diversity/run01/plan.json"
    c16_path = ROOT / "results/step16-diversity/run01/B_expand256/completed.json"
    s16_path = ROOT / "docs/단계별_개선/16_검증결과.json"
    p16, c16, s16 = read_json(p16_path), read_json(c16_path), read_json(s16_path)
    for path in (p16_path, c16_path, s16_path):
        require(path.relative_to(ROOT).as_posix() in plan["prior_files"], "Unbound previous contract")
    checked_hash(p16_path, c16["plan_sha256"])
    compare(s16["provenance"]["plan_sha256"], c16["plan_sha256"], "Previous verified source plan")
    require(s16["status"] == "completed_and_verified" and c16["status"] == "completed", "Previous stage is incomplete")
    source_path = Path(plan["prior_state"]).resolve()
    require(source_path == (c16_path.parent / "training_state_3200.pt").resolve(), "Unexpected source state")
    source_specs = [s for s in c16["resume_states"] if s["step"] == 3200]
    require(len(source_specs) == 1, "Source full-state inventory")
    compare(plan["prior_state_sha256"], source_specs[0]["sha256"], "Source full-state binding")
    checked_hash(source_path, plan["prior_state_sha256"])
    source = torch.load(source_path, map_location="cpu", weights_only=True)
    compare(source["attempt"], 3200, "Source stage16 LOCAL attempt")
    exact(plan["source_contract"], {key: source[key] for key in plan["source_contract"]}, "Source full-state contract")
    require(set(plan["source_contract"]) == {"attempt", "source_hashes", "settings", "schedule", "provenance", "counters"},
            "Incomplete source restore contract")
    local = {key: c16[key] for key in ("optimizer_steps", "skipped_optimizer_steps", "attempted_exposures", "successful_update_exposures")}
    exact(source["counters"], local, "Source stage16 local counters")
    cumulative = merge_counters(p16["initial_counters"], local)
    exact(plan["initial_counters"], cumulative, "Merged source cumulative counters")
    compare(plan["source_local_attempt"], 3200, "Source local attempt binding")
    compare(plan["source_history"], {"prior_attempt": 9600, "local_attempt": 3200,
        "prior_counters": p16["initial_counters"], "local_counters": local}, "Explicit source history decomposition")
    compare(cumulative, s16["branches"]["B_expand256"]["cumulative"], "Previous verified cumulative history")
    exact(plan["initial_scaler"], source["scaler_state_dict"], "Source scaler")
    compare(cumulative["optimizer_steps"], 12797, "Source cumulative successful updates")
    compare(cumulative["skipped_optimizer_steps"], 3, "Source cumulative skipped updates")
    compare(local["optimizer_steps"], 3198, "Source local successful updates")
    compare(local["skipped_optimizer_steps"], 2, "Source local skipped updates")
    checkpoint_path = Path(plan["source_checkpoint"]).resolve()
    require(checkpoint_path == (c16_path.parent / "step_3200.pt").resolve(), "Unexpected source inference checkpoint")
    checkpoint_specs = [s for s in c16["snapshots"] if s["step"] == 3200]
    require(len(checkpoint_specs) == 1, "Source inference snapshot inventory")
    compare(plan["source_checkpoint_sha256"], checkpoint_specs[0]["checkpoint_sha256"], "Fixed source checkpoint")
    compare(plan["source_checkpoint_sha256"], s16["branches"]["B_expand256"]["snapshots"]["3200"]["checkpoint_sha256"],
        "Previous verified source weights")
    checked_hash(checkpoint_path, plan["source_checkpoint_sha256"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    validate_tokenizer_metadata(checkpoint, str(checkpoint_path))
    require(compare_state_dicts(source["model_state_dict"], checkpoint["state_dict"])["exact"], "Source full/inference weights disagree")
    original, previous = p16["cohorts"]["original64"], p16["cohorts"]["additional192"]
    compare(checkpoint["training_data"]["sample_ids"], original + previous, "Source trained sample IDs")
    document, _ = load_manifest(plan["manifest"], verify_files=False)
    selection = select_expansion(document["records"], original, previous,
        {cohort: p16["cohorts"][cohort] for cohort in HELDOUT}, total=512, seed=42)
    compare(plan["selection"], selection, "Rebuilt metadata-only expansion")
    cohorts = {"original64": selection["original64"], "additional192": selection["additional192"],
        "additional256": selection["additional256"], **selection["heldout"]}
    compare(plan["cohorts"], cohorts, "Five disjoint evaluation cohorts")
    require([len(cohorts[c]) for c in COHORTS] == [64, 192, 256, 60, 20], "Fixed cohort counts")
    ids = ordered_ids(plan)
    require(len(ids) == len(set(ids)) == 592, "Overlapping evaluation IDs")
    all_records = {r["id"]: r for r in document["records"]}
    records = {sample_id: all_records[sample_id] for sample_id in ids}
    compare(plan["records"], list(records.values()), "Selected record inventory")
    fixed_settings = {
        "attempt_updates": 6400, "source_attempt": 12800, "batch_size": 2, "learning_rate": .0001,
        "adam_betas": [.9, .999], "adam_eps": 1e-8, "weight_decay": 0., "seed": 42, "grad_clip": 1.,
        "grad_scaler_init_scale": source["scaler_state_dict"]["scale"], "precision": "amp_fp16_ctc_fp32",
        "ctc_reduction": "mean", "ctc_zero_infinity": False, "evaluation_precision": "fp32_unpadded_B1",
        "normal_steps": STEPS, "control_steps": ENDPOINTS, "control_cohorts": HELDOUT,
        "beam_steps": ENDPOINTS, "beam_conditions": ["normal"], "beam_width": 10, "shuffle_seed": 42,
        "fresh_optimizer_per_branch": False, "sampler": "fresh local epoch-full-index-shuffle-v1",
        "early_stopping": False, "validation_selection": False, "torch_threads": 2,
        "cudnn_deterministic": True, "cudnn_benchmark": False, "tf32": False}
    compare(plan["settings"], fixed_settings, "Fixed training/evaluation settings")
    compare(plan["steps"], STEPS, "Snapshot steps")
    compare(plan["resume_state_steps"], STEPS, "Resume snapshot steps")
    compare(plan["source_attempt"], BASE_STEP, "Source attempt")
    compare(plan["additional_attempt_updates"], 6400, "Additional attempt budget")
    compare(plan["scope"], {"training_performed": True, "validation_cache_access": True, "test_cache_access": False,
        "development_evaluation": True, "deployment": False, "best_checkpoint_selection": False}, "Experiment scope")
    compare(plan["evaluation"], {"cohort_order": COHORTS, "normal_steps": STEPS, "control_steps": ENDPOINTS,
        "control_cohorts": HELDOUT, "beam_steps": ENDPOINTS, "beam_conditions": ["normal"],
        "expected_forwards": 5376, "expected_beam_rows": 2368, "primary_decoder": "greedy"}, "Evaluation schedule")
    require(set(plan["branches"]) == set(BRANCHES), "Branch inventory")
    for branch, cohort in BRANCHES.items():
        spec = plan["branches"][branch]
        compare(spec["train_ids"], selection[cohort], "Branch training IDs")
        compare(spec["settings"], fixed_settings, "Both branches share training settings")
        compare(spec["schedule"], make_training_schedule(selection[cohort], attempt_updates=6400, batch_size=2, seed=42),
            "Deterministic local sampler schedule")
    for sample_id, row in records.items():
        expected_split = "val" if sample_id in cohorts["val20"] else "train"
        compare(row["split"], expected_split, "Selected split before cache access")
        path = inside(Path(plan["manifest"]).parent, row["cache_path"])
        checked_hash(path, row["cache_sha256"])
        with np.load(path, allow_pickle=False) as cache:
            validate_tokenizer_metadata(cache, str(path))
            for field in ("text", "source_text", "speaker_id", "utterance_id", "recording_id", "angle",
                          "source_partition", "roi_method", "label_sha256", "temporal_method", "video_fps", "label_fps"):
                require(str(cache[field]) == str(row[field]), f"Cache metadata mismatch: {sample_id}/{field}")
            compare(cache["label"].tolist(), encode_text(row["text"]), "Cached token labels")
    return records, source


def expected_provenance(plan: dict, branch: str, step: int) -> dict:
    return {"mode": "smoke_only", "purpose": "train_expansion_diagnostic", "branch": branch,
        "manifest_sha256": plan["manifest_sha256"],
        "sample_ids": plan["selection"]["existing256"] if step == 0 else plan["branches"][branch]["train_ids"],
        "planned_sample_ids": plan["branches"][branch]["train_ids"],
        "source_state_sha256": plan["prior_state_sha256"], "source_attempt": BASE_STEP,
        "cumulative_attempt": BASE_STEP + step, "generalization_evaluation": True,
        "development_only": True, "deployment": False}


def validate_checkpoint(path: Path, plan: dict, branch: str, step: int, counters: dict, plan_sha: str) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    validate_tokenizer_metadata(checkpoint, str(path))
    require(checkpoint["architecture_version"] == "lipnet-compact-v1" and checkpoint["num_classes"] == 69
            and checkpoint["lstm_hidden"] == 256 and checkpoint["vocab"] == GRAPHEME_VOCAB,
            "Unexpected checkpoint architecture")
    state = checkpoint["state_dict"]
    require(all(isinstance(v, torch.Tensor) and v.is_floating_point() and bool(torch.isfinite(v).all()) for v in state.values())
            and sum(v.numel() for v in state.values()) == 3287781, "Checkpoint parameter inventory/finiteness")
    compare(checkpoint["training_data"], expected_provenance(plan, branch, step), "Checkpoint diagnostic provenance")
    compare(checkpoint["training_run"], {"expansion_diagnostic": True, "branch": branch, "attempt_updates": step,
        "cumulative_attempt": BASE_STEP + step, "optimizer_steps": counters["optimizer_steps"],
        "skipped_optimizer_steps": counters["skipped_optimizer_steps"],
        "cumulative_optimizer_steps": plan["initial_counters"]["optimizer_steps"] + counters["optimizer_steps"],
        "cumulative_skipped_optimizer_steps": plan["initial_counters"]["skipped_optimizer_steps"] + counters["skipped_optimizer_steps"],
        "plan_sha256": plan_sha, "source_training_state_sha256": plan["prior_state_sha256"],
        "inference_only": True, "selection": "fixed_attempt_schedule", "settings": plan["settings"]}, "Checkpoint training record")
    return state


def schedule_metadata(schedule: dict) -> dict:
    return {key: schedule[key] for key in ("sample_ids", "attempt_updates", "batch_size", "seed", "batches_sha256")}


def local_counters(rows: list[dict], sample_ids: list[str]) -> dict:
    attempted = Counter({s: 0 for s in sample_ids})
    successful = Counter({s: 0 for s in sample_ids})
    for row in rows:
        attempted.update(row["ids"])
        if not row["skipped"]:
            successful.update(row["ids"])
    return {"optimizer_steps": sum(not row["skipped"] for row in rows),
        "skipped_optimizer_steps": sum(row["skipped"] for row in rows),
        "attempted_exposures": dict(attempted), "successful_update_exposures": dict(successful)}


def validate_resume_state(value: dict, source: dict, checkpoint_state: dict, plan: dict,
                          branch: str, step: int, telemetry: list[dict], plan_sha: str) -> dict:
    """State attempt/counters are local; the actual Adam counter stays cumulative."""
    spec = plan["branches"][branch]
    local = local_counters(telemetry[:step], spec["train_ids"])
    exact(value["counters"], local, "Local full-state counters")
    exact(value["schedule"], schedule_metadata(spec["schedule"]), "Local full-state schedule")
    exact(value["provenance"], {**expected_provenance(plan, branch, step), "plan_sha256": plan_sha}, "Full-state provenance")
    exact(value["source_hashes"], plan["source_hashes"], "Full-state frozen sources")
    exact(value["settings"], plan["settings"], "Full-state settings")
    cumulative = {**local, "optimizer_steps": plan["initial_counters"]["optimizer_steps"] + local["optimizer_steps"],
        "skipped_optimizer_steps": plan["initial_counters"]["skipped_optimizer_steps"] + local["skipped_optimizer_steps"]}
    # The reusable structural validator expects Adam's count in counters. Pass a
    # separate view; the saved local counters were checked exactly just above.
    viewed = {**value, "counters": cumulative}
    scaler = scaler_after(telemetry[:step], source["scaler_state_dict"])
    result = validate_prior_full_state(viewed, source, checkpoint_state, step=step,
        learning_rate=.0001, counters=cumulative, scaler=scaler)
    if step == 0:
        for field in ("model_state_dict", "optimizer_state_dict", "scaler_state_dict", "rng_state",
                      "model_training", "module_training", "backend_state"):
            exact(value[field], source[field], f"Initial full-state runtime.{field}")
    return {**result, "local_attempt": step, "cumulative_attempt": BASE_STEP + step,
        "local_optimizer_steps": local["optimizer_steps"], "cumulative_optimizer_steps": cumulative["optimizer_steps"],
        "initial_source_runtime_exact": step == 0}


def validate_exact_audit(audit: dict, where: str) -> None:
    for field, expected in {"exact": True, "difference_count": 0, "max_abs_numeric_difference": 0.,
        "nonfinite_difference_count": 0, "differences": [], "excluded_keys": [], "details_truncated": False}.items():
        compare(audit[field], expected, f"{where}.{field}")
    require(type(audit["compared_leaves"]) is int and audit["compared_leaves"] > 0, "Empty runtime audit")


def validate_checkpoint_telemetry(state: dict, row: dict) -> None:
    """Bind each saved weight snapshot to the recorded post-update weight norms."""
    for group, norm in parameter_norms(state).items():
        entry = row["parameter_update"]["total"] if group == "total" else row["parameter_update"]["groups"][group]
        compare(entry["after_l2"], norm, "Saved checkpoint/post-update weight norm", rtol=2e-6)


def verify_initial_posteriors(run: Path, plan: dict) -> dict:
    """Compare both step0 grids with the complete stage16 B endpoint grid."""
    prior = ROOT / "results/step16-diversity/run01/B_expand256/step_3200"
    expected = {(sample_id, "normal") for sample_id in plan["selection"]["existing256"]}
    expected |= {(sample_id, condition) for cohort in HELDOUT for sample_id in plan["cohorts"][cohort]
        for condition in CONDITIONS}
    old_rows = read_rows(prior / "rows.jsonl")
    require({(r["id"], r["condition"]) for r in old_rows} == expected and len(old_rows) == 496,
            "Source initial reproduction inventory")
    result = {}
    with np.load(prior / "posteriors.npz", allow_pickle=False) as original:
        for branch in BRANCHES:
            path = run / branch / "step_0000/posteriors.npz"
            checked = 0
            with np.load(path, allow_pickle=False) as candidate:
                for row in old_rows:
                    key = row["id"] + "__" + row["condition"]
                    old_key = row["posterior_key"]
                    require(candidate[key].dtype == original[old_key].dtype and np.array_equal(candidate[key], original[old_key]),
                            "Initial posterior differs from stage16 source")
                    checked += 1
            result[branch] = {"source_posteriors_exact": checked, "heldout": 240, "existing256_normal": 256}
    with np.load(run / "A_keep256/step_0000/posteriors.npz", allow_pickle=False) as a, \
            np.load(run / "B_expand512/step_0000/posteriors.npz", allow_pickle=False) as b:
        require(set(a.files) == set(b.files), "Initial branch posterior inventory")
        require(all(a[key].dtype == b[key].dtype and np.array_equal(a[key], b[key]) for key in a.files),
                "Initial branch posteriors differ")
        result["all_initial_branch_posteriors_exact"] = len(a.files)
    return result


def validate_preflight(run: Path, plan: dict, completed: dict, plan_sha: str) -> dict:
    checked_hash(run / "preflight_gate.json", completed["preflight_gate_sha256"])
    gate = read_json(run / "preflight_gate.json")
    for key, expected in {"status": "passed", "plan_sha256": plan_sha, "checked_before_either_branch_training": True}.items():
        compare(gate[key], expected, f"Preflight.{key}")
    validate_exact_audit(gate["cross_branch_runtime_exact"], "Initial cross-branch runtime")
    compare(gate["cross_branch_initial_outputs"], {"status": "passed", "prediction_rows_exact": True,
        "posterior_arrays_exact": True, "posterior_arrays_compared": 752}, "Initial cross-branch output gate")
    require(set(gate["restore_gate_sha256"]) == set(gate["initial_states"]) == set(gate["initial_snapshots"]) == set(BRANCHES),
            "Preflight branch inventory")
    for branch in BRANCHES:
        folder = run / branch
        checked_hash(folder / "restore_gate.json", gate["restore_gate_sha256"][branch])
        restore = read_json(folder / "restore_gate.json")
        for key, expected in {"status": "passed", "source_training_state_sha256": plan["prior_state_sha256"],
            "initial_training_state_sha256": gate["initial_states"][branch]["sha256"],
            "stage_local_counters_reset_only": True, "optimizer_state_reset": False}.items():
            compare(restore[key], expected, f"Source restore gate.{key}")
        validate_exact_audit(restore["source_runtime_exact"], "Source restore runtime")
        compare(restore["initial_outputs"], {"status": "passed", "prediction_rows_compared": 496,
            "posterior_arrays_compared": 496, "posterior_arrays_exact": True,
            "posterior_max_absolute_difference": 0.}, "Reproduced source outputs gate")
    return {"stored_gate_sha256": completed["preflight_gate_sha256"],
        "source_and_initial_runtime_audits_passed": True,
        "raw_posterior_recomparison": verify_initial_posteriors(run, plan)}


def build_summary(run: Path) -> dict:
    run = run.resolve()
    plan, completed = read_json(run / "plan.json"), read_json(run / "completed.json")
    plan_sha = checked_hash(run / "plan.json", completed["plan_sha256"])
    require(completed["status"] == "completed", "Experiment is incomplete")
    records, source = validate_plan(plan)
    for field, expected in {**plan["scope"], "forwards": 5376, "beam_rows": 2368,
        "cache_ids_opened": ordered_ids(plan), "environment": plan["environment"]}.items():
        compare(completed[field], expected, f"Completed experiment.{field}")
    checked_hash(run / "cache_audit.json", completed["cache_audit_sha256"])
    compare(read_json(run / "cache_audit.json"), {"plan_sha256": plan_sha, "cache_ids_opened": ordered_ids(plan),
        "cache_sha256": {s: records[s]["cache_sha256"] for s in ordered_ids(plan)}, "test_cache_access": False}, "Selected cache audit")
    preflight = validate_preflight(run, plan, completed, plan_sha)
    raw_gate = read_json(run / "preflight_gate.json")
    require(set(completed["branch_completed_sha256"]) == set(BRANCHES), "Completed branch inventory")
    branches, raw_rows = {}, {}
    for branch in BRANCHES:
        folder = run / branch
        checked_hash(folder / "completed.json", completed["branch_completed_sha256"][branch])
        report = read_json(folder / "completed.json")
        for field, expected in {"status": "completed", "branch": branch, "plan_sha256": plan_sha,
            "attempt_updates": 6400, "cumulative_attempt": 19200, **plan["scope"]}.items():
            compare(report[field], expected, f"Completed branch.{field}")
        checked_hash(folder / "telemetry.jsonl", report["telemetry_sha256"])
        telemetry = read_rows(folder / "telemetry.jsonl")
        validated = validate_extension_telemetry(telemetry, plan["branches"][branch]["schedule"], records,
            plan["settings"], source, plan["initial_counters"])
        for key, expected in validated["extension"].items():
            compare(report[key], expected, f"Completed branch local counters.{key}")
        for key in ("optimizer_steps", "skipped_optimizer_steps"):
            compare(report["cumulative_" + key], validated["cumulative"][key], f"Completed branch cumulative.{key}")
        compare(report["compute_totals"], {"raw_frame_presentations": validated["compute"]["attempted_unpadded_frames"],
            "padded_frame_presentations": validated["compute"]["attempted_padded_frames"]}, "Measured branch workload")
        for name in ("restore_gate", "start_gate"):
            checked_hash(folder / (name + ".json"), report[name + "_sha256"])
        compare(report["restore_gate_sha256"], raw_gate["restore_gate_sha256"][branch], "Same source restore gate")
        start = read_json(folder / "start_gate.json")
        for field, expected in {"status": "passed", "initial_training_state_sha256": raw_gate["initial_states"][branch]["sha256"],
            "preflight_gate_sha256": completed["preflight_gate_sha256"], "next_local_attempt": 1,
            "next_cumulative_attempt": 12801}.items():
            compare(start[field], expected, f"Prepared state restore.{field}")
        validate_exact_audit(start["prepared_runtime_exact"], "Prepared state actual restore")
        compare([s["step"] for s in report["snapshots"]], STEPS, "Fixed snapshot schedule")
        compare([s["step"] for s in report["resume_states"]], STEPS, "Fixed full-state schedule")
        compare(report["snapshots"][0], raw_gate["initial_snapshots"][branch], "Preflight snapshot unchanged")
        compare(report["resume_states"][0], raw_gate["initial_states"][branch], "Preflight full-state unchanged")
        snapshots, raw_rows[branch], states = {}, {}, []
        for spec, state_spec in zip(report["snapshots"], report["resume_states"]):
            step = spec["step"]
            compare(spec["folder"], f"step_{step:04d}", "Snapshot folder")
            compare(spec["checkpoint"], f"step_{step:04d}.pt", "Checkpoint path")
            compare(state_spec["file"], f"training_state_{step:04d}.pt", "Full-state path")
            snapshot_folder, checkpoint_path = inside(folder, spec["folder"]), inside(folder, spec["checkpoint"])
            checked_hash(snapshot_folder / "summary.json", spec["summary_sha256"])
            checked_hash(snapshot_folder / "runtime_audit.json", spec["runtime_audit_sha256"])
            validate_exact_audit(read_json(snapshot_folder / "runtime_audit.json"), "Evaluation preserves full runtime")
            checked_hash(checkpoint_path, spec["checkpoint_sha256"])
            counters = local_counters(telemetry[:step], plan["branches"][branch]["train_ids"])
            checkpoint_state = validate_checkpoint(checkpoint_path, plan, branch, step, counters, plan_sha)
            if step > 0:
                validate_checkpoint_telemetry(checkpoint_state, telemetry[step - 1])
            snapshots[f"{step:04d}"], raw_rows[branch][step] = validate_snapshot(snapshot_folder, plan, records,
                branch, step, spec["checkpoint_sha256"])
            state_path = inside(folder, state_spec["file"])
            checked_hash(state_path, state_spec["sha256"])
            value = torch.load(state_path, map_location="cpu", weights_only=True)
            states.append({**validate_resume_state(value, source, checkpoint_state, plan, branch, step, telemetry, plan_sha),
                "sha256": state_spec["sha256"]})
            del value, checkpoint_state
        for field in ("seconds", "peak_allocated_mib", "peak_reserved_mib"):
            require(report[field] >= 0, "Negative runtime or memory")
        branches[branch] = {**validated, "snapshots": snapshots, "resume_state_validation": states,
            "train_ids": plan["branches"][branch]["train_ids"], "telemetry_sha256": report["telemetry_sha256"],
            "runtime": {key: report[key] for key in ("seconds", "peak_allocated_mib", "peak_reserved_mib")},
            "skipped_attempts": [{key: r[key] for key in ("attempt", "cumulative_attempt", "ids", "loss", "scale_before", "scale_after")}
                for r in telemetry if r["skipped"]],
            "exposure_summary": {kind: {cohort: describe([validated["extension"][kind].get(s, 0)
                for s in plan["cohorts"][cohort]]) for cohort in COHORTS}
                for kind in ("attempted_exposures", "successful_update_exposures")}}
    rows_verified = sum(s["rows"] for b in branches.values() for s in b["snapshots"].values())
    beams_verified = sum(s["beam_rows"] for b in branches.values() for s in b["snapshots"].values())
    compare(rows_verified, 5376, "Verified posterior count")
    compare(beams_verified, 2368, "Independently redecoded beam count")
    paired = {branch + "_final_minus_initial": paired_normal(raw_rows[branch][0], raw_rows[branch][6400]) for branch in BRANCHES}
    paired["B_expand512_final_minus_A_keep256_final"] = paired_normal(raw_rows["A_keep256"][6400], raw_rows["B_expand512"][6400])
    predictions = {}
    choices = {"initial": raw_rows["A_keep256"][0], **{b: raw_rows[b][6400] for b in BRANCHES}}
    lookups = {key: {r["id"]: r for r in values if r["condition"] == "normal"} for key, values in choices.items()}
    for cohort in COHORTS:
        predictions[cohort] = [{"id": s, "reference": records[s]["text"], "speaker_id": records[s]["speaker_id"],
            "recording_id": records[s]["recording_id"], "frames": records[s]["frames"],
            "models": {key: {field: lookup[s][field] for field in ("greedy", "beam", "ctc_loss_per_target_token")}
                for key, lookup in lookups.items()}} for s in plan["cohorts"][cohort]]
    require(completed["seconds"] >= 0, "Negative total runtime")
    result = {"schema": "lipnet-training-expansion-summary-v1", "status": "completed_and_verified",
        "created_utc": datetime.now(timezone.utc).isoformat(), "run": str(run),
        "provenance": {"plan_sha256": plan_sha, "completed_sha256": sha256_file(run / "completed.json"),
            "manifest_sha256": plan["manifest_sha256"], "source_state_sha256": plan["prior_state_sha256"],
            "source_hashes": plan["source_hashes"], "prior_files": plan["prior_files"],
            "summarizer_sha256": sha256_file(Path(__file__)), "selected_cache_metadata_and_labels_verified": len(records),
            "test_cache_access": False, "model_forward_calls": 0, "training_performed_by_audit": False,
            "saved_posterior_rows_verified": rows_verified, "saved_normal_beams_independently_redecoded": beams_verified,
            "full_states_verified": 8,
            "limits": ["No model forward or optimizer replay occurs in the audit.",
                "Initial full runtime is compared against source tensors; subsequent weights/counters/Adam steps/scaler are cross-checked, moments and RNG schema/hash-checked.",
                "Runtime exact comparisons are stored runner audits; pixel change counts are range-checked while indices and cache hashes are reconstructed."]},
        "protocol": {"settings": plan["settings"], "evaluation": plan["evaluation"], "scope": plan["scope"],
            "selection": plan["selection"], "cohorts": plan["cohorts"], "source_counters": plan["initial_counters"],
            "source_history": plan["source_history"], "source_local_attempt": plan["source_local_attempt"],
            "schedules": {branch: {key: value for key, value in plan["branches"][branch]["schedule"].items() if key != "batches"}
                for branch in BRANCHES},
            "metric_definitions": {"mean_sentence_jamo_cer": "Arithmetic mean of per-sentence NFD jamo edit distance/reference NFD length; may exceed1.",
                "corpus_jamo_cer": "Pooled edit distance divided by pooled reference length; differs from sentence mean.",
                "exact_count": "Full reference and prediction are equal.",
                "paired_delta": "After CER minus before CER on the same clip; negative is improvement.",
                "attempts": "Stage17 local0..6400 plus fixed cumulative source12800; source stage16 state has local3200 and local counters, merged independently with its preceding history."}},
        "preflight": preflight, "branches": branches, "paired_normal": paired, "normal_predictions": predictions,
        "runtime": {"seconds": completed["seconds"], "environment": completed["environment"]},
        "interpretation": [
            "Both branches start from identical source weights, Adam moments/step, scaler and global RNG; only the sample set and its new local sampler differ.",
            "Equal6400 attempted batches and12800 sample presentations imply50 new repetitions per existing256 in A versus25 per expanded512 in B; actual frame compute differs.",
            "Merged source counters record325 earlier attempted presentations per original64 and25 per additional192. The initial short16 also had earlier initialization training beyond those counters; additional256 is untrained in this lineage until branch B.",
            "additional256 is training data in B and an untrained diagnostic cohort in A. Its B fit is not generalization. Both branches train original64 and additional192.",
            "The same train60 recording IDs and normalized transcripts are excluded from both training branches; val20 has unseen speakers but is previously used development data.",
            "The expanded512 still contains the same eight training speakers and59 training recordings as existing256; the new data increases utterance coverage within those recordings.",
            "More samples also changes sentence/length/recording distributions and rehearsal frequency. This paired run does not isolate a pure causal data-volume effect.",
            "Static/shuffle sensitivity is not proof of correct lip-motion recognition, and one shuffle/training lineage does not establish repeatability.",
            "All snapshot steps were fixed before evaluation. There is no validation-based checkpoint selection, test50 access or webcam deployment.",
            "Development-set CER is not webcam accuracy; do not label1-CER as program accuracy."]}
    finite_tree(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", "--run-dir", dest="run", type=Path, default=ROOT / "results/step17-expansion/run01")
    parser.add_argument("--out", "--output", dest="out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite summary: {args.out}")
    torch.set_num_threads(2)
    result = build_summary(args.run)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.out.resolve()), "status": result["status"]}, ensure_ascii=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
