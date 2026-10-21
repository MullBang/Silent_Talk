"""Verify completed stage12 artifacts and summarize the fixed train-only experiment.

No model is instantiated or run. Stored beam strings are scored, not decoded again.
Greedy, CTC loss, token statistics and controls are recomputed from saved posteriors.
Gradient/update telemetry is checked for internal consistency, not recreated from
training: raw gradients, optimizer state and parameter snapshots were not saved.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "backend"), str(ROOT / "scripts")]
from models import GRAPHEME_VOCAB
from models.dataset_manifest import load_manifest, sha256_file
from models.tokenizer import encode_text, validate_tokenizer_metadata
from diagnose_temporal_use import prediction_row
from diagnose_learning_transition import (
    STEPS, CONTROL_STEPS, BEAM_STEPS, BRANCHES, SOURCE_PATHS, INITIAL_SHA, MANIFEST_SHA,
)
from temporal_diagnostics import analyze_ctc, compare_posteriors, transform_frames
from transition_diagnostics import (
    GROUP_PREFIXES, METRIC_DEFINITIONS, make_training_schedule, select_transition_cohorts,
)

WINDOWS = ((1, 100), (101, 400), (401, 800), (801, 1600))
CTC_FIELDS = ("blank_posterior_mean", "argmax_blank_fraction", "mean_frame_entropy_nats",
              "greedy_collapsed_token_count", "greedy_collapsed_unique_token_count",
              "greedy_decoded_nfd_length", "reference_token_length", "reference_ctc_min_frames",
              "ctc_slack_frames")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def finite_tree(value, location: str = "document") -> None:
    """Null is valid for undefined telemetry, but JSON NaN/Infinity never is."""
    if isinstance(value, dict):
        for key, item in value.items():
            finite_tree(item, f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            finite_tree(item, f"{location}[{index}]")
    elif isinstance(value, float):
        require(math.isfinite(value), f"Nonfinite value: {location}")


def _unique_object(pairs: list[tuple]) -> dict:
    value = {}
    for key, item in pairs:
        require(key not in value, f"Duplicate JSON key: {key}")
        value[key] = item
    return value


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    require(isinstance(value, dict), f"JSON object required: {path}")
    finite_tree(value, str(path))
    return value


def read_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        require(bool(line.strip()), f"Blank JSONL row: {path}")
        row = json.loads(line, object_pairs_hook=_unique_object)
        require(isinstance(row, dict), f"JSONL object required: {path}")
        finite_tree(row, str(path))
        rows.append(row)
    return rows


def compare(actual, expected, where: str, *, atol: float = 1e-8, rtol: float = 1e-7) -> None:
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), f"Fields mismatch: {where}")
        for key, item in expected.items():
            compare(actual[key], item, f"{where}.{key}", atol=atol, rtol=rtol)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), f"Length mismatch: {where}")
        for index, (a, b) in enumerate(zip(actual, expected)):
            compare(a, b, f"{where}[{index}]", atol=atol, rtol=rtol)
    elif isinstance(expected, float):
        require(isinstance(actual, (int, float)) and not isinstance(actual, bool)
                and math.isfinite(actual) and math.isfinite(expected)
                and math.isclose(actual, expected, abs_tol=atol, rel_tol=rtol), f"Number mismatch: {where}")
    else:
        require(type(actual) is type(expected) and actual == expected, f"Value mismatch: {where}")


def describe(values: list[float | None]) -> dict:
    """Null norms are counted separately, never silently converted to zero."""
    observed = [value for value in values if value is not None]
    result = {"count": len(values), "finite_count": len(observed), "null_count": len(values)-len(observed)}
    if observed:
        require(all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                    for v in observed), "Nonfinite/nonnumeric summary input")
        array = np.asarray(observed, dtype=np.float64)
        result.update(mean=float(array.mean()), median=float(np.median(array)),
                      minimum=float(array.min()), maximum=float(array.max()))
    else:
        result.update(mean=None, median=None, minimum=None, maximum=None)
    return result


def checked_hash(path: Path, expected: str) -> str:
    require(isinstance(expected, str) and len(expected) == 64, f"Invalid SHA256: {path}")
    actual = sha256_file(path)
    require(actual == expected, f"SHA256 mismatch: {path}")
    return actual


def inside(base: Path, relative: str) -> Path:
    require(isinstance(relative, str) and not Path(relative).is_absolute(), "Relative artifact path required")
    path = (base / relative).resolve()
    require(path.is_relative_to(base.resolve()), "Artifact escapes its directory")
    return path


def validate_plan(plan: dict) -> dict[str, dict]:
    """Validate the frozen protocol and train-only cache hashes without model inference."""
    require(plan["schema"] == "lipnet-learning-transition-v1", "Unsupported plan schema")
    require(plan["manifest_sha256"] == MANIFEST_SHA and plan["initial_checkpoint_sha256"] == INITIAL_SHA,
            "Unexpected stage12 source artifacts")
    checked_hash(Path(plan["manifest"]), plan["manifest_sha256"])
    checked_hash(Path(plan["initial_checkpoint"]), plan["initial_checkpoint_sha256"])
    require(set(plan["source_hashes"]) == set(SOURCE_PATHS), "Frozen source inventory mismatch")
    for relative, digest in plan["source_hashes"].items():
        checked_hash(inside(ROOT, relative), digest)
    document, _ = load_manifest(plan["manifest"], verify_files=False)
    compare(plan["cohorts"], select_transition_cohorts(document["records"], seed=42), "cohorts")
    expected_records = [row for row in document["records"] if row["id"] in plan["cohorts"]["expanded64"]]
    compare(plan["records"], expected_records, "plan.records")
    records = {row["id"]: row for row in expected_records}
    require(len(records) == 64 and all(row["split"] == "train" for row in records.values()), "Train64 scope mismatch")
    expected_settings = {
        "attempt_updates": 1600, "batch_size": 2, "learning_rate": .0003,
        "adam_betas": [.9, .999], "adam_eps": 1e-8, "weight_decay": 0., "seed": 42,
        "grad_clip": 1., "grad_scaler_init_scale": 1024., "precision": "amp_fp16_ctc_fp32",
        "ctc_reduction": "mean", "ctc_zero_infinity": False, "evaluation_precision": "fp32_unpadded_B1",
        "normal_steps": STEPS, "control_steps": CONTROL_STEPS, "beam_steps": BEAM_STEPS,
        "beam_width": 10, "shuffle_seed": 42, "fresh_optimizer_per_branch": True,
        "early_stopping": False, "validation_selection": False, "torch_threads": 2,
        "cudnn_deterministic": True, "cudnn_benchmark": False, "tf32": False,
    }
    for name, value in expected_settings.items():
        compare(plan["settings"][name], value, f"settings.{name}")
    for name, value in {"train_only": True, "validation_cache_access": False, "test_cache_access": False,
                        "generalization_evaluation": False, "deployment": False}.items():
        compare(plan["scope"][name], value, f"scope.{name}")
    require(set(plan["schedules"]) == set(BRANCHES), "Unexpected branch schedule")
    for branch, cohort in BRANCHES.items():
        compare(plan["schedules"][branch], make_training_schedule(plan["cohorts"][cohort]), f"schedule.{branch}")
    # Only train64 cache metadata/labels are read. Hashing checks the entire file;
    # no validation/test cache or video pixel array is opened for evaluation.
    for row in records.values():
        path = inside(Path(plan["manifest"]).parent, row["cache_path"])
        checked_hash(path, row["cache_sha256"])
        with np.load(path, allow_pickle=False) as cache:
            validate_tokenizer_metadata(cache, str(path))
            for name in ("text", "source_text", "speaker_id", "utterance_id", "recording_id", "angle",
                         "source_partition", "roi_method", "label_sha256", "temporal_method", "video_fps", "label_fps"):
                require(str(cache[name]) == str(row[name]), f"Cache metadata mismatch: {row['id']}/{name}")
            require(cache["label"].tolist() == encode_text(row["text"]), "Cache target tokens mismatch")
    validate_checkpoint(Path(plan["initial_checkpoint"]), plan, initial=True)
    return records


def validate_checkpoint(path: Path, plan: dict, *, initial: bool = False,
                        branch: str | None = None, step: int | None = None,
                        updates: int | None = None, skips: int | None = None,
                        plan_sha: str | None = None) -> None:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    validate_tokenizer_metadata(checkpoint, str(path))
    require(checkpoint["architecture_version"] == "lipnet-compact-v1"
            and checkpoint["num_classes"] == 69 and checkpoint["lstm_hidden"] == 256
            and checkpoint["vocab"] == GRAPHEME_VOCAB, "Checkpoint architecture/tokenizer mismatch")
    require(not any(key in checkpoint for key in ("optimizer", "optimizer_state_dict", "scaler", "rng_state")),
            "Expected inference-only checkpoint")
    require(isinstance(checkpoint["state_dict"], dict) and bool(checkpoint["state_dict"]), "Missing model tensors")
    require(all(isinstance(value, torch.Tensor) and bool(torch.isfinite(value).all())
                for value in checkpoint["state_dict"].values()), "Nonfinite checkpoint tensors")
    data, run = checkpoint["training_data"], checkpoint["training_run"]
    require(data["mode"] == "smoke_only" and data["manifest_sha256"] == plan["manifest_sha256"]
            and data["generalization_evaluation"] is False, "Checkpoint source scope mismatch")
    if initial:
        require(data["purpose"] == "train_subset_overfit" and data["sample_ids"] == plan["cohorts"]["short16"]
                and run["overfit_only"] is True and run["step"] == 1600, "Initial short16 source mismatch")
    else:
        compare(data, {"mode": "smoke_only", "purpose": "train_subset_transition_diagnostic",
                       "manifest_sha256": plan["manifest_sha256"],
                       "sample_ids": plan["schedules"][branch]["sample_ids"],
                       "generalization_evaluation": False, "deployment": False}, "checkpoint.training_data")
        compare(run, {"transition_diagnostic": True, "branch": branch, "attempt_updates": step,
                      "optimizer_steps": updates, "skipped_optimizer_steps": skips,
                      "plan_sha256": plan_sha, "initial_checkpoint_sha256": plan["initial_checkpoint_sha256"],
                      "inference_only": True, "selection": "fixed_attempt_schedule", "settings": plan["settings"]},
                "checkpoint.training_run")


def aggregate_predictions(rows: list[dict], decoder: str) -> dict:
    values = [row[decoder] for row in rows]
    require(values and all(isinstance(value, dict) for value in values), f"Missing {decoder} predictions")
    chars = sum(value["reference_jamo_count"] for value in values)
    words = sum(value["reference_word_count"] for value in values)
    return {"count": len(values), "mean_sentence_jamo_cer": float(np.mean([v["cer"] for v in values])),
            "corpus_jamo_cer": sum(v["jamo_edit_distance"] for v in values) / chars,
            "mean_sentence_wer": float(np.mean([v["wer"] for v in values])),
            "corpus_wer": sum(v["word_edit_distance"] for v in values) / words,
            "exact_count": sum(v["exact"] for v in values), "empty_count": sum(v["prediction"] == "" for v in values),
            "reference_jamo_count": chars, "reference_word_count": words,
            "jamo_edit_distance": sum(v["jamo_edit_distance"] for v in values),
            "word_edit_distance": sum(v["word_edit_distance"] for v in values)}


def cohort_metadata(plan: dict, records: dict) -> dict:
    result = {}
    for cohort in ("short16", "additional48", "expanded64"):
        rows = [records[s] for s in plan["cohorts"][cohort]]
        counts = Counter(token for row in rows for token in encode_text(row["text"]))
        top = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:4]
        total = sum(counts.values())
        result[cohort] = {
            "count": len(rows), "frames": describe([r["frames"] for r in rows]),
            "unique_texts": len({r["text"] for r in rows}),
            "speaker_count": len({r["speaker_id"] for r in rows}),
            "reference_token_length": describe([len(encode_text(r["text"])) for r in rows]),
            "reference_token_count": total, "reference_unique_token_count": len(counts),
            "reference_counts_by_token_id": {str(k): v for k, v in sorted(counts.items())},
            "reference_pooled_top4_fraction": sum(v for _, v in top)/total if total else 0.,
            "reference_pooled_top4": [{"token_id": k, "token": GRAPHEME_VOCAB[k], "count": v,
                                       "fraction": v/total} for k, v in top],
        }
    return result


def summarize_group(rows: list[dict]) -> dict:
    require(bool(rows), "Empty cohort/condition")
    require(len({row["id"] for row in rows}) == len(rows), "Duplicate sample in group")
    require(len({row["cohort"] for row in rows}) == len({row["condition"] for row in rows}) == 1,
            "Cohorts/conditions must not be pooled")
    beam_present = [row["beam"] is not None for row in rows]
    require(all(beam_present) or not any(beam_present), "Mixed beam availability")
    tokens, reference = Counter(), Counter()
    for row in rows:
        tokens.update(row["ctc"]["greedy_collapsed_token_ids"])
        reference.update(encode_text(row["reference"]))
    total = sum(tokens.values())
    top = sorted(tokens.items(), key=lambda item: (-item[1], item[0]))[:4]
    result = {"count": len(rows), "greedy": aggregate_predictions(rows, "greedy"),
              "ctc_loss_per_target_token": describe([r["ctc_loss_per_target_token"] for r in rows]),
              "ctc": {name: describe([r["ctc"][name] for r in rows]) for name in CTC_FIELDS},
              "repeat3_fraction": describe([r["ctc"]["collapsed_ngram_repetition"]["3"]["repeat_fraction"] for r in rows]),
              "frames": describe([r["frames"] for r in rows]),
              "token_frequency": {"collapsed_token_count": total, "collapsed_unique_token_count": len(tokens),
                  "collapsed_counts_by_token_id": {str(k): v for k, v in sorted(tokens.items())},
                  "reference_counts_by_token_id": {str(k): v for k, v in sorted(reference.items())},
                  "pooled_top4_fraction": sum(v for _, v in top)/total if total else 0.,
                  "pooled_top4": [{"token_id": k, "token": GRAPHEME_VOCAB[k], "count": v,
                                   "fraction": v/total} for k, v in top]}}
    if all(beam_present):
        result["beam"] = aggregate_predictions(rows, "beam")
        result["beam_decoded_nfd_length"] = describe([r["ctc"]["beam_decoded_nfd_length"] for r in rows])
    comparisons = [r["comparison_to_normal"] for r in rows if r["comparison_to_normal"] is not None]
    if comparisons:
        require(len(comparisons) == len(rows), "Incomplete normal/control comparison")
        result["comparison_to_normal"] = {
            name: describe([r[name] for r in comparisons]) for name in (
                "mean_total_variation", "max_total_variation", "argmax_path_agreement_fraction",
                "blank_posterior_mean_difference")}
        result["comparison_to_normal"].update(
            greedy_text_agreement_count=sum(r["text_equal"] for r in comparisons),
            greedy_text_agreement_fraction=sum(r["text_equal"] for r in comparisons)/len(rows))
    return result


def validate_snapshot(folder: Path, plan: dict, records: dict, step: int, checkpoint_sha: str) -> dict:
    stored = read_json(folder / "summary.json")
    conditions = ["normal", "static_middle", "shuffle_42"] if step in CONTROL_STEPS else ["normal"]
    beam_performed = step in BEAM_STEPS
    for name, value in {"step": step, "beam_performed": beam_performed, "conditions": conditions,
                        "rows": len(records)*len(conditions), "checkpoint_sha256": checkpoint_sha}.items():
        compare(stored[name], value, f"snapshot.{name}")
    require(stored["seconds"] >= 0, "Negative snapshot duration")
    checked_hash(folder / "rows.jsonl", stored["rows_sha256"])
    checked_hash(folder / "posteriors.npz", stored["posteriors_sha256"])
    rows = read_rows(folder / "rows.jsonl")
    require(len(rows) == stored["rows"], "Snapshot row count mismatch")
    expected_pairs = [(sample_id, condition) for sample_id in plan["cohorts"]["expanded64"] for condition in conditions]
    compare([[r["id"], r["condition"]] for r in rows], [list(pair) for pair in expected_pairs], "snapshot row scope/order")
    normal = {}
    with np.load(folder / "posteriors.npz", allow_pickle=False) as posteriors:
        require(len(set(r["posterior_key"] for r in rows)) == len(rows)
                and set(posteriors.files) == {r["posterior_key"] for r in rows}, "Posterior inventory mismatch")
        for row in rows:
            source, condition = records[row["id"]], row["condition"]
            cohort = "short16" if row["id"] in plan["cohorts"]["short16"] else "additional48"
            for name, value in {"step": step, "split": "train", "cohort": cohort, "speaker_id": source["speaker_id"],
                                "reference": source["text"], "frames": source["frames"]}.items():
                compare(row[name], value, f"row.{name}")
            logp = posteriors[row["posterior_key"]]
            require(logp.dtype == np.float32 and logp.shape == (row["frames"], 69) and np.isfinite(logp).all(),
                    "Nonfinite or invalid posterior")
            require((row["beam"] is not None) == beam_performed, "Beam availability mismatch")
            labels = np.asarray(encode_text(row["reference"]), dtype=np.int64)
            diagnostics = analyze_ctc(logp, labels, beam_text=row["beam"]["prediction"] if beam_performed else "")
            if not beam_performed:
                diagnostics = {k: v for k, v in diagnostics.items() if not k.startswith("beam")}
            compare(row["ctc"], diagnostics, "posterior CTC statistics")
            compare(row["greedy"], prediction_row(source, diagnostics["greedy_text"]), "greedy prediction")
            if beam_performed:
                compare(row["beam"], prediction_row(source, row["beam"]["prediction"]), "beam scoring")
            loss = torch.nn.functional.ctc_loss(torch.from_numpy(logp[:, None, :]), torch.from_numpy(labels),
                torch.tensor([len(logp)]), torch.tensor([len(labels)]), blank=0, reduction="mean", zero_infinity=False)
            compare(row["ctc_loss_per_target_token"], float(loss), "posterior CTC loss", atol=2e-5, rtol=2e-5)
            if condition == "normal":
                require(row["comparison_to_normal"] is None, "Normal has control comparison")
                normal[row["id"]] = (logp, diagnostics["greedy_text"])
            else:
                normal_logp, normal_text = normal[row["id"]]
                compare(row["comparison_to_normal"], compare_posteriors(normal_logp, logp,
                        normal_text=normal_text, control_text=diagnostics["greedy_text"]), "normal/control posterior")
            # Source index metadata can be reconstructed without loading pixels.
            _, transform = transform_frames(np.zeros((row["frames"], 1, 1, 3), np.uint8),
                "shuffle" if condition == "shuffle_42" else condition, sample_id=row["id"], seed=42)
            transform["input_shape"] = [row["frames"], 96, 96, 3]
            changed = row["transform"]["pixel_changed_frame_count"]
            require(isinstance(changed, int) and not isinstance(changed, bool) and 0 <= changed <= row["frames"],
                    "Invalid changed-frame count")
            if condition == "normal":
                require(changed == 0, "Normal transform changed pixels")
            transform["pixel_changed_frame_count"] = changed
            compare(row["transform"], transform, "transform metadata")
    cohorts = {cohort: {condition: summarize_group([r for r in rows if r["cohort"] == cohort and r["condition"] == condition])
                        for condition in conditions} for cohort in ("short16", "additional48")}
    expected_quick = {cohort: {"count": values["normal"]["count"],
                              "greedy_exact": values["normal"]["greedy"]["exact_count"],
                              "greedy_mean_cer": values["normal"]["greedy"]["mean_sentence_jamo_cer"]}
                      for cohort, values in cohorts.items()}
    compare(stored["cohorts"], expected_quick, "snapshot quick summary")
    return {"step": step, "beam_performed": beam_performed, "conditions": conditions, "rows": len(rows),
            "checkpoint_sha256": checkpoint_sha, "seconds": stored["seconds"], "cohorts": cohorts}


def _nonnegative_int(value, where: str) -> None:
    require(isinstance(value, int) and not isinstance(value, bool) and value >= 0, f"Invalid count: {where}")


def _gradient_metrics(value: dict, where: str) -> None:
    require(set(value) == {"groups", "total"} and set(value["groups"]) == set(GROUP_PREFIXES), "Gradient groups mismatch")
    counts = ("parameter_tensors", "parameter_elements", "present_grad_tensors", "present_grad_elements",
              "missing_grad_tensors", "zero_grad_tensors", "zero_grad_elements", "nonfinite_grad_tensors", "nonfinite_grad_elements")
    for name, stats in [*value["groups"].items(), ("total", value["total"])]:
        for field in counts:
            _nonnegative_int(stats[field], f"{where}.{name}.{field}")
        require(stats["present_grad_tensors"] + stats["missing_grad_tensors"] == stats["parameter_tensors"]
                and stats["present_grad_elements"] <= stats["parameter_elements"], "Gradient presence counts mismatch")
        require(stats["zero_grad_tensors"] + stats["nonfinite_grad_tensors"] <= stats["present_grad_tensors"]
                and stats["zero_grad_elements"] + stats["nonfinite_grad_elements"] <= stats["present_grad_elements"],
                "Gradient zero/nonfinite counts mismatch")
        is_finite = stats["nonfinite_grad_elements"] == 0
        compare(stats["all_present_gradients_finite"], is_finite, "Gradient finiteness flag")
        require(math.isfinite(stats["finite_l2_norm"]) and stats["finite_l2_norm"] >= 0, "Invalid finite gradient norm")
        if is_finite:
            compare(stats["l2_norm"], float(stats["finite_l2_norm"]), "Gradient finite norm")
        else:
            require(stats["l2_norm"] is None, "Nonfinite gradient needs null norm")
    for field in counts:
        compare(value["total"][field], sum(v[field] for v in value["groups"].values()), "Gradient group count sum")
    compare(value["total"]["finite_l2_norm"], math.sqrt(sum(v["finite_l2_norm"]**2 for v in value["groups"].values())),
            "Gradient group norm sum", rtol=2e-6)


def _update_metrics(value: dict) -> None:
    require(set(value) == {"groups", "total"} and set(value["groups"]) == set(GROUP_PREFIXES), "Update groups mismatch")
    for stats in [*value["groups"].values(), value["total"]]:
        for field in ("parameter_tensors", "parameter_elements", "before_nonfinite_elements", "after_nonfinite_elements", "changed_finite_elements"):
            _nonnegative_int(stats[field], f"update.{field}")
        require(stats["before_nonfinite_elements"] == stats["after_nonfinite_elements"] == 0
                and stats["all_parameters_finite"] is True, "Nonfinite model update")
        require(stats["changed_finite_elements"] <= stats["parameter_elements"], "Changed-element count exceeds parameters")
        for field in ("before_l2", "after_l2", "absolute_l2", "relative_l2"):
            require(isinstance(stats[field], (int, float)) and math.isfinite(stats[field]) and stats[field] >= 0,
                    "Invalid parameter-update norm")
        compare(stats["relative_l2"], stats["absolute_l2"]/max(stats["before_l2"], 1e-12), "Update relative norm")
        require((stats["absolute_l2"] == 0) == (stats["changed_finite_elements"] == 0), "Update norm/changed count mismatch")
    for field in ("parameter_tensors", "parameter_elements", "before_nonfinite_elements", "after_nonfinite_elements", "changed_finite_elements"):
        compare(value["total"][field], sum(v[field] for v in value["groups"].values()), "Update group count sum")
    for field in ("before_l2", "after_l2", "absolute_l2"):
        compare(value["total"][field], math.sqrt(sum(v[field]**2 for v in value["groups"].values())), "Update group norm sum", rtol=2e-6)


def validate_telemetry(rows: list[dict], schedule: dict, records: dict, settings: dict) -> dict:
    require(len(rows) == schedule["attempt_updates"] == len(schedule["batches"]), "Telemetry attempt count mismatch")
    updates, skips = 0, 0
    exposures = Counter({s: 0 for s in schedule["sample_ids"]})
    successful = Counter({s: 0 for s in schedule["sample_ids"]})
    previous_scale = settings["grad_scaler_init_scale"]
    previous_after_norms = None
    sizes = None
    for row, batch in zip(rows, schedule["batches"]):
        finite_tree(row, "telemetry")
        for name, value in batch.items():
            compare(row[name], value, f"schedule/telemetry.{name}")
        compare(row["frames"], [records[s]["frames"] for s in batch["ids"]], "telemetry.frames")
        compare(row["target_lengths"], [len(encode_text(records[s]["text"])) for s in batch["ids"]], "telemetry.targets")
        require(isinstance(row["skipped"], bool) and isinstance(row["clip_applied"], bool), "Telemetry boolean required")
        require(isinstance(row["loss"], (int, float)) and math.isfinite(row["loss"]) and row["loss"] >= 0
                and row["seconds"] >= 0, "Invalid loss/duration")
        compare(row["learning_rate"], settings["learning_rate"], "learning rate")
        compare(row["scale_before"], float(previous_scale), "AMP scale continuity")
        require(row["scale_before"] > 0 and row["scale_after"] > 0, "Invalid AMP scale")
        compare(row["skipped"], row["scale_after"] < row["scale_before"], "AMP skip flag")
        _gradient_metrics(row["preclip"], "preclip")
        _gradient_metrics(row["postclip"], "postclip")
        _update_metrics(row["parameter_update"])
        finite = row["preclip"]["total"]["all_present_gradients_finite"]
        require(finite != row["skipped"] and row["clip_applied"] == finite, "Gradient/AMP/clip flags disagree")
        current_sizes = {name: (v["parameter_tensors"], v["parameter_elements"])
                         for name, v in row["preclip"]["groups"].items()}
        require(sizes is None or current_sizes == sizes, "Parameter inventory changed during training")
        sizes = current_sizes
        for group in [*GROUP_PREFIXES, "total"]:
            before = row["preclip"]["total"] if group == "total" else row["preclip"]["groups"][group]
            after = row["postclip"]["total"] if group == "total" else row["postclip"]["groups"][group]
            delta = row["parameter_update"]["total"] if group == "total" else row["parameter_update"]["groups"][group]
            if previous_after_norms is not None:
                compare(delta["before_l2"], previous_after_norms[group], "Parameter norm continuity")
            for field in ("parameter_tensors", "parameter_elements", "present_grad_tensors", "present_grad_elements",
                          "missing_grad_tensors", "nonfinite_grad_tensors", "nonfinite_grad_elements"):
                compare(after[field], before[field], f"Clip count preservation: {field}")
            for field in ("parameter_tensors", "parameter_elements"):
                compare(delta[field], before[field], "Update parameter inventory")
            if finite:
                require(before["missing_grad_tensors"] == 0, "Missing finite model gradient")
                compare(after["l2_norm"], before["l2_norm"]*row["clip_coefficient"], "Clipped norm", atol=2e-7, rtol=3e-5)
            else:
                compare(after, before, "Nonfinite gradient must remain unclipped")
        if finite:
            expected = min(1., settings["grad_clip"]/(row["preclip"]["total"]["l2_norm"]+1e-6))
            compare(row["clip_coefficient"], expected, "Clipping coefficient", atol=2e-7, rtol=3e-5)
        else:
            require(row["clip_coefficient"] is None and row["parameter_update"]["total"]["absolute_l2"] == 0,
                    "Skipped AMP update changed weights or used clipping")
        updates += int(not row["skipped"])
        skips += int(row["skipped"])
        compare(row["optimizer_steps"], updates, "Cumulative optimizer updates")
        compare(row["skipped_optimizer_steps"], skips, "Cumulative skipped updates")
        exposures.update(batch["ids"])
        if not row["skipped"]:
            successful.update(batch["ids"])
        previous_scale = row["scale_after"]
        previous_after_norms = {group: row["parameter_update"]["total"]["after_l2"] if group == "total"
                                else row["parameter_update"]["groups"][group]["after_l2"]
                                for group in [*GROUP_PREFIXES, "total"]}
    compare(dict(exposures), schedule["exposures"], "Attempted exposure schedule")
    return {"attempt_updates": len(rows), "optimizer_steps": updates, "skipped_optimizer_steps": skips,
            "attempted_exposures": dict(exposures), "successful_update_exposures": dict(successful)}


def fixed_window_summary(rows: list[dict]) -> dict:
    result = {}
    for start, end in WINDOWS:
        values = [row for row in rows if start <= row["attempt"] <= end]
        if not values:
            continue
        result[f"{start:04d}-{end:04d}"] = {
            "first_attempt": start, "last_attempt": end, "attempts_observed": len(values),
            "optimizer_steps": sum(not r["skipped"] for r in values), "skipped_count": sum(r["skipped"] for r in values),
            "clip_invoked_count": sum(r["clip_applied"] for r in values),
            "effective_clipping_count": sum(r["clip_coefficient"] is not None and r["clip_coefficient"] < 1 for r in values),
            "zero_parameter_update_count": sum(r["parameter_update"]["total"]["absolute_l2"] == 0 for r in values),
            "loss": describe([r["loss"] for r in values]), "seconds_per_attempt": describe([r["seconds"] for r in values]),
            "clip_coefficient": describe([r["clip_coefficient"] for r in values]),
            "scale_before": describe([r["scale_before"] for r in values]),
            "scale_after": describe([r["scale_after"] for r in values]),
            "groups": {}}
        for group in [*GROUP_PREFIXES, "total"]:
            def metrics(row, key):
                return row[key]["total"] if group == "total" else row[key]["groups"][group]
            result[f"{start:04d}-{end:04d}"]["groups"][group] = {
                key: {field: describe([metrics(r, key)[field] for r in values]) for field in fields}
                for key, fields in {
                    "preclip": ("l2_norm", "finite_l2_norm", "nonfinite_grad_elements", "missing_grad_tensors", "zero_grad_tensors"),
                    "postclip": ("l2_norm", "finite_l2_norm", "nonfinite_grad_elements", "missing_grad_tensors", "zero_grad_tensors"),
                    "parameter_update": ("absolute_l2", "relative_l2", "changed_finite_elements"),
                }.items()}
    return result


def first_observed_drop(snapshots: dict, expected_exact: int = 16) -> dict:
    previous = 0
    for key in sorted(snapshots, key=int):
        snapshot = snapshots[key]
        exact = snapshot["cohorts"]["short16"]["normal"]["greedy"]["exact_count"]
        if exact < expected_exact:
            return {"step": snapshot["step"], "greedy_exact_count": exact,
                    "previous_observed_step": previous,
                    "note": "First sampled snapshot below16; the exact onset between snapshots is unknown. Not early stopping."}
        previous = snapshot["step"]
    return {"step": None, "note": "No sampled normal snapshot below16; changes between snapshots are unobserved."}


def build_summary(runpath: str | Path) -> dict:
    run = Path(runpath).resolve()
    plan, completed = read_json(run / "plan.json"), read_json(run / "completed.json")
    require(completed["status"] == "completed", "Experiment is not completed")
    plan_sha = checked_hash(run / "plan.json", completed["plan_sha256"])
    for name in ("validation_cache_access", "test_cache_access", "deployment"):
        require(completed[name] is False, "Forbidden completed scope")
    records = validate_plan(plan)
    compare(completed["train_cache_ids_opened"], plan["cohorts"]["expanded64"], "Opened train cache IDs")
    require(set(completed["branch_completed_sha256"]) == set(completed["branches"]) == set(BRANCHES), "Completed branch inventory")
    checked_hash(run / "common/step_0000/summary.json", completed["common_summary_sha256"])
    common = validate_snapshot(run / "common/step_0000", plan, records, 0, plan["initial_checkpoint_sha256"])
    require(common["cohorts"]["short16"]["normal"]["greedy"]["exact_count"] == 16, "Initial short16 reproduction failed")
    branches = {}
    for branch in BRANCHES:
        folder = run / branch
        checked_hash(folder / "completed.json", completed["branch_completed_sha256"][branch])
        report = read_json(folder / "completed.json")
        require(report["status"] == "completed" and report["branch"] == branch
                and report["step0_folder"] == "common/step_0000", "Branch completion/source mismatch")
        checked_hash(folder / "telemetry.jsonl", report["telemetry_sha256"])
        telemetry = read_rows(folder / "telemetry.jsonl")
        validated = validate_telemetry(telemetry, plan["schedules"][branch], records, plan["settings"])
        for name, value in validated.items():
            compare(report[name], value, f"completed.{branch}.{name}")
        compare(completed["branches"][branch], {name: report[name] for name in
                ("attempt_updates", "optimizer_steps", "skipped_optimizer_steps", "seconds")}, "Top-level branch completion")
        compare([item["step"] for item in report["snapshots"]], STEPS[1:], "Fixed snapshot schedule")
        snapshots = {"0000": common}
        for spec in report["snapshots"]:
            step = spec["step"]
            compare(Path(spec["folder"]).as_posix(), f"{branch}/step_{step:04d}", "Snapshot path")
            compare(Path(spec["checkpoint"]).as_posix(), f"{branch}/step_{step:04d}.pt", "Checkpoint path")
            snapshot_folder = inside(run, spec["folder"])
            checkpoint = inside(run, spec["checkpoint"])
            checked_hash(snapshot_folder / "summary.json", spec["summary_sha256"])
            checked_hash(checkpoint, spec["checkpoint_sha256"])
            event = telemetry[step-1]
            validate_checkpoint(checkpoint, plan, branch=branch, step=step,
                updates=event["optimizer_steps"], skips=event["skipped_optimizer_steps"], plan_sha=plan_sha)
            snapshots[f"{step:04d}"] = validate_snapshot(snapshot_folder, plan, records, step, spec["checkpoint_sha256"])
        require(report["last_checkpoint_sha256"] == report["snapshots"][-1]["checkpoint_sha256"], "Last checkpoint mismatch")
        for key in ("seconds", "peak_allocated_mib", "peak_reserved_mib"):
            require(report[key] >= 0, "Negative runtime/memory")
        branches[branch] = {**validated, "trained_cohort": BRANCHES[branch], "common_step0_reference": "common_step0",
            "first_observed_short16_greedy_exact_below16": first_observed_drop(snapshots),
            "snapshots": {k: v for k, v in snapshots.items() if k != "0000"},
            "fixed_windows": fixed_window_summary(telemetry),
            "exposure_summary": {kind: {cohort: describe([validated[kind].get(s, 0) for s in plan["cohorts"][cohort]])
                for cohort in ("short16", "additional48")} for kind in ("attempted_exposures", "successful_update_exposures")},
            "runtime": {k: report[k] for k in ("seconds", "peak_allocated_mib", "peak_reserved_mib")},
            "last_checkpoint_sha256": report["last_checkpoint_sha256"], "telemetry_sha256": report["telemetry_sha256"]}
    final_key = f"{STEPS[-1]:04d}"
    final = {}
    for cohort in ("short16", "additional48"):
        final[cohort] = {}
        for condition in common["conditions"]:
            entries = {"initial": common["cohorts"][cohort][condition], **{
                branch: branches[branch]["snapshots"][final_key]["cohorts"][cohort][condition] for branch in BRANCHES}}
            final[cohort][condition] = {key: {field: row[field] for field in ("count", "greedy", "beam", "ctc_loss_per_target_token")}
                                        for key, row in entries.items()}
            final[cohort][condition]["B_minus_A"] = {decoder: {
                metric: entries["B_expand64"][decoder][metric]-entries["A_keep16"][decoder][metric]
                for metric in ("mean_sentence_jamo_cer", "corpus_jamo_cer", "mean_sentence_wer", "corpus_wer", "exact_count")}
                for decoder in ("greedy", "beam")}
    result = {"schema": "lipnet-learning-transition-summary-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed_and_verified", "run": str(run),
        "provenance": {"plan_sha256": plan_sha, "completed_sha256": sha256_file(run / "completed.json"),
            "manifest_sha256": plan["manifest_sha256"], "initial_checkpoint_sha256": plan["initial_checkpoint_sha256"],
            "summarizer_sha256": sha256_file(Path(__file__)), "source_hashes": plan["source_hashes"],
            "train_cache_count_verified": len(records), "validation_cache_access": False, "test_cache_access": False,
            "model_forward_calls": 0, "beam_decodes": 0,
            "saved_posterior_rows_verified": common["rows"] + sum(
                s["rows"] for b in branches.values() for s in b["snapshots"].values()),
            "saved_beam_rows_rescored": common["rows"] + sum(
                s["rows"] for b in branches.values() for s in b["snapshots"].values() if s["beam_performed"]),
            "raw_posterior_recalculation": ["greedy text and CER/WER", "CTC loss", "blank/entropy/token statistics", "normal/control comparison"],
            "limits": ["Saved beam strings are rescored but beam search is not rerun.",
                "Raw gradients and optimizer states were not saved; telemetry is hash-checked and internally cross-checked.",
                "Cache hashes and transform indices are checked; pixel_changed_frame_count is range-checked, not recomputed."]},
        "protocol": {"settings": plan["settings"], "scope": plan["scope"], "cohorts": plan["cohorts"],
            "cohort_metadata": cohort_metadata(plan, records),
            "schedules": {branch: {key: value for key, value in schedule.items() if key != "batches"}
                          for branch, schedule in plan["schedules"].items()}, "metric_definitions": METRIC_DEFINITIONS},
        "common_step0": common, "branches": branches, "final_comparison": final,
        "interpretation": [
            "All64 clips belong to the existing train split. A's additional48 is untrained in A but is not an external validation set; B trains on it.",
            "Both branches restart from the same inference weights with fresh Adam/GradScaler; optimizer state is not resumed.",
            "Equal1600 attempted batches are unequal per-ID exposures: A200 and B50; successful exposures exclude AMP-skipped batches.",
            "The branches also differ in clip lengths, text and speakers; this experiment does not isolate the causal effect of dataset size.",
            "Cohorts16/48 and conditions are separate; common step0 is stored once, never treated as extra independent observations.",
            "CER/WER values are ratios, not percentages. Sentence means and pooled edit-count rates have different denominators.",
            "Token top4 is pooled over raw collapsed greedy token IDs; beam NFD length is decoded text length, not a hidden beam token path.",
            "clip_invoked_count follows runner clip_applied; effective_clipping_count requires coefficient<1. Null nonfinite norms are not zeros.",
            "Static/shuffle response measures sensitivity, not correct recognition; no generalization or deployment claim is supported."]}
    finite_tree(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "results/step12-transition/run01")
    parser.add_argument("--out", type=Path, required=True)
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
