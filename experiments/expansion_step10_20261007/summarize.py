"""Recount the four fixed Step10 evaluations without loading pixels or models.

The separate old71/new24 cohorts remain primary development results. new23
is the predeclared overlap sensitivity analysis, never a replacement cohort.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import unicodedata

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
SEEDS = (42, 43)
VARIANTS = ("control_old441", "expanded510")
CONDITIONS = ("normal", "static", "shuffled")
GROUPS = ("old71", "new24", "new23")
PAIRS = tuple((seed, variant) for seed in SEEDS for variant in VARIANTS)
INTEGER_FIELDS = ("character_edits", "reference_characters", "word_edits", "reference_words",
                  "normalized_character_edits", "normalized_reference_characters")
RATE_FIELDS = ("cer_percent", "wer_percent", "normalized_cer_percent", "exact_percent")


def require(value, message):
    if not value:
        raise ValueError(message)


def local(path):
    path = Path(path).resolve()
    require(path.is_relative_to(ROOT), "Only workspace artifacts may be read")
    return path


def bind(path):
    path = local(path)
    with path.open("rb") as stream:
        return {"path": str(path), "sha256": hashlib.file_digest(stream, "sha256").hexdigest()}


def verify(ref):
    require(bind(ref["path"]) == {"path": ref["path"], "sha256": ref["sha256"]}, "Changed evidence: " + ref["path"])


def read(ref):
    verify(ref)
    return json.loads(local(ref["path"]).read_text(encoding="utf-8"))


def distance(left, right):
    previous = list(range(len(right) + 1))
    for i, one in enumerate(left, 1):
        current = [i]
        for j, two in enumerate(right, 1):
            current.append(min(previous[j] + 1, current[-1] + 1, previous[j - 1] + (one != two)))
        previous = current
    return previous[-1]


def counts(reference, prediction):
    reference, prediction = (unicodedata.normalize("NFC", text).strip() for text in (reference, prediction))
    normal_ref, normal_pred = " ".join(reference.split()), " ".join(prediction.split())
    return {"character_edits": distance(reference, prediction), "reference_characters": len(reference),
            "word_edits": distance(reference.split(), prediction.split()), "reference_words": len(reference.split()),
            "exact": reference == prediction, "normalized_character_edits": distance(normal_ref, normal_pred),
            "normalized_reference_characters": len(normal_ref), "normalized_exact": normal_ref == normal_pred}


def aggregate(records):
    require(records and len({row["id"] for row in records}) == len(records), "Empty/duplicate cohort records")
    result = dict.fromkeys(INTEGER_FIELDS, 0)
    for row in records:
        require(row["status"] in ("ok", "failed") and isinstance(row["reference"], str)
                and isinstance(row["prediction"], str), "Malformed evaluation record")
        require(row["status"] != "failed" or row["prediction"] == "", "Failure must retain empty prediction")
        measured = counts(row["reference"], row["prediction"])
        require(measured == row["metrics"], "Saved metric differs from independent recount: " + row["id"])
        require(measured["reference_characters"] > 0 and measured["reference_words"] > 0, "Empty reference")
        for key in INTEGER_FIELDS:
            result[key] += measured[key]
    result.update(clips=len(records), failed_inputs=sum(row["status"] != "ok" for row in records),
                  exact_sentences=sum(row["metrics"]["exact"] for row in records),
                  normalized_exact_sentences=sum(row["metrics"]["normalized_exact"] for row in records))
    result.update(cer_percent=100 * result["character_edits"] / result["reference_characters"],
                  wer_percent=100 * result["word_edits"] / result["reference_words"],
                  normalized_cer_percent=100 * result["normalized_character_edits"] / result["normalized_reference_characters"])
    return result


def with_rates(value):
    return {**value, "exact_percent": 100 * value["exact_sentences"] / value["clips"]}


def collapse_ids(values):
    require(isinstance(values, list) and values and all(type(x) is int and 0 <= x < 69 for x in values), "Invalid argmax IDs")
    return [token for i, token in enumerate(values) if token != 0 and (i == 0 or token != values[i - 1])]


def validate_metadata(policy):
    require(policy["schema"] == "lipnet-step10-expansion-comparison-plan-v1"
            and policy["status"] == "frozen_before_expanded_training_and_new_evaluation", "Wrong frozen comparison")
    require(policy["seeds"] == list(SEEDS) and policy["variants"] == list(VARIANTS)
            and policy["method"] == "B_feature", "Changed variants/seeds")
    require(policy["updates_per_model"] == 3200 and policy["clips_per_update"] == 2
            and policy["new_optimizer_updates_planned"] == policy["reused_historical_optimizer_updates"] == 6400, "Budget changed")
    require(policy["new_full_transcript_approvals"] == 0 and policy["training_ready_flags_promoted"] is False
            and policy["service_weights_updated"] is False and policy["audio_used"] is False, "Approval/service policy changed")
    contract = policy["evaluation_contract"]
    require(contract["declared_before_training"] is True and contract["conditions"] == list(CONDITIONS), "Evaluation was not predeclared")
    meta = contract["development"]
    for group, count in zip(GROUPS, (71, 24, 23), strict=True):
        ids = meta[group + "_ids"]
        require(len(ids) == len(set(ids)) == count, "Wrong fixed " + group)
    old, new, excluded = meta["old71_ids"], meta["new24_ids"], meta["excluded_duplicate_ids"]
    require(set(old).isdisjoint(new) and len(excluded) == 1 and set(excluded) <= set(new)
            and meta["new23_ids"] == [sid for sid in new if sid not in excluded], "Wrong supplemental cohort")
    control, expanded, held = policy["control_train_ids"], policy["provisional_train_ids"], policy["held_train_ids"]
    require(len(control) == len(set(control)) == 441 and len(expanded) == len(set(expanded)) == 510
            and len(held) == len(set(held)) == 27 and expanded[:441] == control, "Wrong training roles")
    require(set(expanded).isdisjoint(old + new + held) and set(held).isdisjoint(old + new), "Training/development/hold contamination")
    evidence = meta["overlap_evidence"]
    strip = lambda text: "".join(unicodedata.normalize("NFC", text).split())
    require(len(evidence) == 1 and evidence[0]["val_id"] == excluded[0] and evidence[0]["train_id"] in expanded
            and strip(evidence[0]["train_text"]) == strip(evidence[0]["val_text"]), "Overlap evidence changed")
    return meta


def validate_evaluation(value, policy, tokenizer=None):
    meta = validate_metadata(policy)
    require(value["schema"] == "lipnet-step10-development-evaluation-v1" and value["status"] == "completed", "Incomplete evaluation")
    require((value["seed"], value["variant"]) in PAIRS and value["arm"] == "B_feature"
            and value["completed_updates"] == 3200, "Wrong evaluation identity/checkpoint")
    require(value["inferred_distinct_records"] == 95 and value["inferred_condition_records"] == 285
            and set(value["cohorts"]) == set(GROUPS), "Wrong inference coverage")
    require(value["new_optimizer_updates"] == 0 and all(value[key] is False for key in
            ("audio_read", "original_media_read", "teacher_targets_read", "teacher_executed", "optimizer_executed",
             "held_training_records_read", "service_weights_updated")), "Evaluation role changed")
    expected_replay = "passed" if value["variant"] == "control_old441" else "not_applicable_to_new_model"
    require(value["old_control_replay"]["status"] == expected_replay, "Control replay did not pass")
    if expected_replay == "passed":
        require(value["old_control_replay"]["mismatches"] == [] and value["old_control_replay"]["compared_records"] == 213,
                "Incomplete control replay")
    summary = {}
    for group in GROUPS:
        cohort = value["cohorts"][group]
        ids = meta[group + "_ids"]
        require(cohort["ids"] == ids and cohort["record_count"] == len(ids) and set(cohort["conditions"]) == set(CONDITIONS), "Changed cohort/conditions")
        require(cohort["model_state_unchanged"] is True and cohort["model_state_before"] == cohort["model_state_after"], "Evaluation changed model")
        require(cohort["aggregation_only"] is (group == "new23"), "Wrong supplemental inference role")
        if group == "new23":
            require(cohort["parent_group"] == "new24" and cohort["excluded_ids"] == meta["excluded_duplicate_ids"], "Changed subset evidence")
        summary[group] = {}
        normal_identity = None
        for condition in CONDITIONS:
            part = cohort["conditions"][condition]
            rows = part["records"]
            require([row["id"] for row in rows] == ids, "Missing/reordered/duplicate fixed ID")
            measured = aggregate(rows)
            require(part["aggregate"] == measured, "Saved aggregate differs")
            identities = [(r["id"], r["speaker_id"], r["source_stem"], r["reference"]) for r in rows]
            normal_identity = identities if normal_identity is None else normal_identity
            require(identities == normal_identity, "References/speakers changed by temporal condition")
            speakers = {speaker: aggregate([r for r in rows if r["speaker_id"] == speaker])
                        for speaker in sorted({r["speaker_id"] for r in rows})}
            require(part["per_speaker"] == speakers, "Saved per-speaker aggregate differs")
            require(part["replacement_characters"] == sum(r["prediction"].count("\ufffd") for r in rows)
                    and part["sentences_with_replacement_characters"] == sum("\ufffd" in r["prediction"] for r in rows), "Replacement characters lost")
            for row in rows:
                require(row["split"] == "dev" and row["group"] == ("old71" if group == "old71" else "new24")
                        and row["condition"] == condition and row["source_split"] == ("dev" if group == "old71" else "val"), "Wrong row role")
                require(row["replacement_characters"] == row["prediction"].count("\ufffd"), "Row replacement count differs")
                if row["status"] == "ok":
                    require(row["error"] is None and len(row["raw_argmax_ids"]) == row["frames"]
                            and collapse_ids(row["raw_argmax_ids"]) == row["collapsed_ids"], "Changed CTC token path")
                    if tokenizer is not None:
                        decoded = tokenizer.compose_jamos([tokenizer.IDX_TO_GRAPHEME[x] for x in row["collapsed_ids"]], strict=False)
                        require(decoded == row["prediction"], "Saved prediction differs from retained CTC tokens")
                else:
                    require(isinstance(row["error"], str) and row["error"], "Failed row missing error")
            if group == "new23":
                selected = [r for r in value["cohorts"]["new24"]["conditions"][condition]["records"] if r["id"] in ids]
                require(rows == selected, "new23 must reuse exact new24 records")
            summary[group][condition] = {"aggregate": with_rates(measured),
                                         "per_speaker": {k: with_rates(v) for k, v in speakers.items()},
                                         "replacement_characters": part["replacement_characters"],
                                         "sentences_with_replacement_characters": part["sentences_with_replacement_characters"]}
    return summary


def differences(one, two):
    """Rates are percentage-point differences: first argument minus second."""
    result = {field.replace("_percent", "_percentage_points"): one[field] - two[field] for field in RATE_FIELDS}
    result.update(exact_sentences=one["exact_sentences"] - two["exact_sentences"],
                  character_edits=one["character_edits"] - two["character_edits"])
    return result


def mean_values(values):
    require(len(values) == 2 and set(values[0]) == set(values[1]), "Exactly two matched seeds required")
    return {key: statistics.mean(value[key] for value in values) for key in values[0]}


def prediction_changes(control, expanded):
    require([r["id"] for r in control] == [r["id"] for r in expanded], "Prediction pair IDs differ")
    rows = []
    for one, two in zip(control, expanded, strict=True):
        require(all(one[key] == two[key] for key in ("id", "speaker_id", "source_stem", "reference", "condition")), "Prediction pair reference differs")
        delta = two["metrics"]["character_edits"] - one["metrics"]["character_edits"]
        rows.append({"id": one["id"], "speaker_id": one["speaker_id"], "reference": one["reference"],
                     "control_prediction": one["prediction"], "expanded_prediction": two["prediction"],
                     "control_status": one["status"], "expanded_status": two["status"],
                     "control_metrics": one["metrics"], "expanded_metrics": two["metrics"],
                     "prediction_changed": one["prediction"] != two["prediction"],
                     "character_edits_change": delta, "outcome": "improved" if delta < 0 else "worsened" if delta > 0 else "tied"})
    totals = {name: sum(row["outcome"] == name for row in rows) for name in ("improved", "worsened", "tied")}
    totals.update(clips=len(rows), changed_predictions=sum(row["prediction_changed"] for row in rows),
                  unchanged_predictions=sum(not row["prediction_changed"] for row in rows),
                  net_character_edits_change=sum(row["character_edits_change"] for row in rows))
    return {"totals": totals, "records": rows}


def training_summary(result, plan, policy, variant):
    require(variant in VARIANTS, "Unknown training variant")
    expected = policy["control_train_ids"] if variant == "control_old441" else policy["provisional_train_ids"]
    require(result["status"] == "completed" and result["arm"] == "B_feature" and result["seed"] == plan["seed"] in SEEDS
            and result["completed_updates"] == plan["updates"] == 3200, "Incomplete training result")
    require(result["start_update"] == 0 and result["restored_from"] is None,
            "Partial resumed history: a bound full-history reconciliation is required before summarizing")
    require(result["plan_sha256"] == plan["plan_sha256"] and result["runtime"] == plan["runtime"] == policy["runtime"], "Wrong training plan/runtime")
    require(set(plan["training_data"]["clips"]) == set(expected) and plan["ids"] == expected, "Wrong training IDs")
    require(len(result["updates"]) == len(plan["schedule"]) == 3200 and plan["batch_size"] == 2, "Missing actual optimizer history")
    require(all(result[key] is False for key in ("dev_inputs_loaded", "audio_used", "teacher_model_loaded", "service_weights_updated")), "Training role changed")
    if variant == "expanded510":
        require(result["held_inputs_loaded"] is False and result["variant"] == variant, "Held training role changed")
    exposure = Counter()
    frame_exposure = Counter()
    for index, (update, batch) in enumerate(zip(result["updates"], plan["schedule"], strict=True), 1):
        require(update["update"] == index and update["ids"] == batch and len(batch) == 2
                and update["frames"] == [plan["training_data"]["clips"][sid]["frames"] for sid in batch], "Actual schedule/frames differ")
        require(update["optimizer_updated"] is True and update["student_changed"] is True and update["adapter_changed"] is True, "Missing optimizer change")
        for key in ("loss", "ctc_loss", "feature_loss", "seconds_optimizer_update"):
            require(isinstance(update[key], (float, int)) and math.isfinite(update[key]) and update[key] >= 0, "Nonfinite/negative training measure")
        require(update["lambda"] == 1.0 and math.isclose(update["loss"], update["ctc_loss"] + update["feature_loss"], rel_tol=1e-10, abs_tol=1e-10), "Loss components disagree")
        require(all(type(update[key]) is int and update[key] >= 0 for key in ("cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes")), "Invalid GPU telemetry")
        for sid, frames in zip(batch, update["frames"], strict=True):
            exposure[sid] += 1
            frame_exposure[sid] += frames
    require(set(exposure) == set(expected), "Not all planned clips were presented")
    require(result["updates"][-1]["state_hashes"] == result["final_state_hashes"] == result["checkpoint"]["state_hashes"]
            and result["checkpoint"]["completed_updates"] == 3200, "Final state/history differs")
    old = set(policy["control_train_ids"])
    rows = [{"id": sid, "cohort": "old441" if sid in old else "new69", "clip_frames": plan["training_data"]["clips"][sid]["frames"],
             "presentations": exposure[sid], "processed_frames": frame_exposure[sid]} for sid in expected]
    roles = {}
    for role in ("old441", "new69"):
        selected = [r for r in rows if r["cohort"] == role]
        roles[role] = {"unique_clips": len(selected), "presentations": sum(r["presentations"] for r in selected),
                       "processed_frames": sum(r["processed_frames"] for r in selected),
                       "presentations_min": min((r["presentations"] for r in selected), default=0),
                       "presentations_max": max((r["presentations"] for r in selected), default=0),
                       "mean_presentations_per_clip": statistics.mean(r["presentations"] for r in selected) if selected else 0}
    require(math.isfinite(result["elapsed_seconds"]) and result["elapsed_seconds"] > 0, "Invalid wall time")
    windows = {name: {field: statistics.mean(row[field] for row in records) for field in ("loss", "ctc_loss", "feature_loss")}
               for name, records in (("first100", result["updates"][:100]), ("last100", result["updates"][-100:]))}
    return {"seed": result["seed"], "variant": variant, "optimizer_updates": 3200,
            "new_optimizer_updates": 3200 if variant == "expanded510" else 0,
            "reused_historical_optimizer_updates": 3200 if variant == "control_old441" else 0,
            "clip_presentations": sum(exposure.values()), "processed_frames": sum(frame_exposure.values()),
            "wall_seconds": result["elapsed_seconds"],
            "optimizer_update_seconds_sum": sum(row["seconds_optimizer_update"] for row in result["updates"]),
            "cuda_peak_allocated_bytes": max(row["cuda_peak_allocated_bytes"] for row in result["updates"]),
            "cuda_peak_reserved_bytes": max(row["cuda_peak_reserved_bytes"] for row in result["updates"]),
            "loss_window_means": windows, "loss_window_updates": {"first100": [1, 100], "last100": [3101, 3200]},
            "exposure_by_role": roles, "per_clip_exposures": rows,
            "accounting_note": "Wall time is the saved training invocation including preparation/checkpoints; optimizer seconds exclude telemetry. Control time is historical and was not spent again in Step10."}


def compare(evaluations, policy, tokenizer=None):
    require(set(evaluations) == set(PAIRS), "Exactly two variants times two seeds are required")
    scores, per_model, by_seed, basis = {}, [], [], []
    baseline_identity = {}
    for seed, variant in PAIRS:
        value = evaluations[(seed, variant)]
        require((value["seed"], value["variant"]) == (seed, variant), "Evaluation mapping identity differs")
        scores[(seed, variant)] = validate_evaluation(value, policy, tokenizer)
        temporal = {}
        for group in GROUPS:
            temporal[group] = {condition: differences(scores[(seed, variant)][group][condition]["aggregate"],
                                                       scores[(seed, variant)][group]["normal"]["aggregate"])
                               for condition in ("static", "shuffled")}
            for condition in CONDITIONS:
                rows = value["cohorts"][group]["conditions"][condition]["records"]
                identity = [(r["id"], r["speaker_id"], r["source_stem"], r["reference"]) for r in rows]
                key = group, condition
                baseline_identity.setdefault(key, identity)
                require(baseline_identity[key] == identity, "Evaluation models use different references/speakers")
                for row in rows:
                    basis.append({"seed": seed, "variant": variant, "group": group, "condition": condition,
                                  "id": row["id"], "speaker_id": row["speaker_id"], "status": row["status"], **row["metrics"]})
        per_model.append({"seed": seed, "variant": variant, "groups": scores[(seed, variant)], "temporal_minus_normal": temporal})
    for seed in SEEDS:
        groups = {}
        for group in GROUPS:
            conditions = {}
            for condition in CONDITIONS:
                old, new = (scores[(seed, variant)][group][condition] for variant in VARIANTS)
                require(set(old["per_speaker"]) == set(new["per_speaker"]), "Speaker cohorts differ")
                conditions[condition] = {"expanded_minus_control": differences(new["aggregate"], old["aggregate"]),
                    "per_speaker": {speaker: {"control": old["per_speaker"][speaker], "expanded": new["per_speaker"][speaker],
                        "expanded_minus_control": differences(new["per_speaker"][speaker], old["per_speaker"][speaker])}
                        for speaker in old["per_speaker"]},
                    "prediction_changes": prediction_changes(*(evaluations[(seed, variant)]["cohorts"][group]["conditions"][condition]["records"] for variant in VARIANTS))}
            groups[group] = conditions
        by_seed.append({"seed": seed, "groups": groups})
    means = {group: {condition: {variant: mean_values([{field: scores[(seed, variant)][group][condition]["aggregate"][field]
                           for field in (*RATE_FIELDS, "exact_sentences")} for seed in SEEDS]) for variant in VARIANTS}
                     for condition in CONDITIONS} for group in GROUPS}
    for group in GROUPS:
        for condition in CONDITIONS:
            means[group][condition]["expanded_minus_control"] = mean_values([entry["groups"][group][condition]["expanded_minus_control"] for entry in by_seed])
    temporal_means = {variant: {group: {condition: mean_values([entry["temporal_minus_normal"][group][condition]
                                 for entry in per_model if entry["variant"] == variant])
                                 for condition in ("static", "shuffled")} for group in GROUPS} for variant in VARIANTS}
    return {"per_model": per_model, "paired_seed_comparisons": by_seed, "two_seed_arithmetic_means": means,
            "two_seed_temporal_minus_normal_arithmetic_means": temporal_means,
            "integer_accounting_rows": basis,
            "accounting_rows_note": "old71/new24 rows are inference evidence; new23 rows repeat their fixed new24 subset solely for separate aggregation. Never sum the three groups as independent data."}


def summarize(evaluation_paths, output=BASE / "results/comparison.json"):
    output = local(output)
    require(output.is_relative_to(BASE / "results") and not output.exists(), "Output must be a new Step10 results file")
    freeze_ref = bind(BASE / "plans/freeze_manifest.json")
    frozen = read(freeze_ref)
    require(frozen["status"] == "frozen", "Missing frozen execution plan")
    policy_ref = frozen["experiment_plan"]
    policy = read(policy_ref)
    meta = validate_metadata(policy)
    frozen_code = {ref["path"]: ref for ref in policy["execution_code"]}
    evidence = [freeze_ref, policy_ref, policy["handoff"], policy["mixed_loader_validation"], *policy["execution_code"], *meta["bindings"]]
    for ref in evidence:
        verify(ref)
    handoff = read(policy["handoff"])
    require(handoff["training_ready"] is False and handoff["new_human_approvals"] == 0, "Candidate approval changed")
    require(policy["provisional_train_ids"] == handoff["provisional_train_ids"]
            and policy["control_train_ids"] == handoff["control_old_train_ids"]
            and policy["held_train_ids"] == handoff["held_new_train_ids"]
            and policy["development_groups"] == handoff["development_groups"], "Frozen handoff roles differ")
    tokenizer_path = ROOT / "backend/models/tokenizer.py"
    require(str(tokenizer_path) in frozen_code, "Tokenizer must be bound by pretraining policy")
    spec = importlib.util.spec_from_file_location("step10_summary_bound_tokenizer", tokenizer_path)
    tokenizer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tokenizer)
    evaluations, training_runs, refs = {}, [], []
    require(len(evaluation_paths) == 4, "Exactly four evaluation paths required")
    for path in evaluation_paths:
        ref = bind(path)
        require(local(path).is_relative_to(BASE / "results"), "Expected Step10 evaluation output")
        value = read(ref)
        key = value["seed"], value["variant"]
        require(key in PAIRS and key not in evaluations, "Duplicate/unknown evaluation model")
        require(value["experiment_plan"] == policy_ref, "Evaluation belongs to another experiment")
        require(value["decoding"] == policy["evaluation_contract"]["decoder"]
                and value["failure_policy"] == policy["evaluation_contract"]["failure_policy"], "Evaluation protocol differs")
        require(value["evaluation_code"] and value["data_bindings"]["old71"] and value["data_bindings"]["new24"], "Missing evaluation evidence bindings")
        for code in value["evaluation_code"]:
            require(frozen_code.get(code["path"]) == code, "Evaluation code was not frozen before training")
        plan, run = read(value["training_plan"]), read(value["run_result"])
        require(plan["seed"] == run["seed"] == key[0] and value["runtime"] == run["runtime"], "Evaluation seed/runtime differs from trained model")
        require(value["training_plan"] == run["plan_file"] and value["checkpoint"] == run["checkpoint"], "Evaluation/run bindings differ")
        require(value["initial_state_hashes"] == run["initial_state_hashes"] == policy["controls"][str(key[0])]["initial_state_hashes"], "Initial state differs")
        require(all(value["cohorts"][group]["model_state_before"] == run["final_state_hashes"]["model"] for group in GROUPS), "Evaluated model differs from final training state")
        if key[1] == "control_old441":
            entry = policy["controls"][str(key[0])]
            require(value["training_plan"] == entry["plan_file"] and value["run_result"] == entry["run_result"], "Historical control changed")
            require(value["old_control_replay"]["reference"] == policy["evaluation_contract"]["control_old71_replay"][str(key[0])], "Control replay reference changed")
        else:
            require(value["training_plan"] in frozen["plans"] and run["experiment_plan"] == plan["experiment_plan"] == policy_ref, "Expanded training binding differs")
        checkpoint = run["checkpoint"]
        evidence.extend([ref, value["training_plan"], value["run_result"], {"path": checkpoint["path"], "sha256": checkpoint["sha256"]},
                         {"path": checkpoint["sidecar"], "sha256": checkpoint["sidecar_sha256"]}])
        for group in ("old71", "new24"):
            evidence.extend(value["data_bindings"][group])
        entry = training_summary(run, plan, policy, key[1])
        training_runs.append({**entry, "run_result": value["run_result"], "plan": value["training_plan"]})
        evaluations[key], refs = value, refs + [ref]
    table = compare(evaluations, policy, tokenizer)
    require(all(value["evaluation_code"] == evaluations[PAIRS[0]]["evaluation_code"]
                and value["data_bindings"] == evaluations[PAIRS[0]]["data_bindings"] for value in evaluations.values()),
            "Evaluation code/data bindings differ between models")
    unique_evidence = list({(ref["path"], ref["sha256"]): ref for ref in evidence}.values())
    for ref in unique_evidence:
        verify(ref)
    result = {"schema": "lipnet-step10-expansion-comparison-v1", "status": "completed",
              "created_utc": datetime.now(timezone.utc).isoformat(), "experiment_plan": policy_ref,
              "summary_code": [bind(Path(__file__)), bind(BASE / "test_summarize.py")],
              "evaluations": refs, "verified_evidence": unique_evidence, **table,
              "training_runs": sorted(training_runs, key=lambda entry: (entry["seed"], VARIANTS.index(entry["variant"]))),
              "optimizer_accounting": {"new_updates_executed_in_step10": sum(r["new_optimizer_updates"] for r in training_runs),
                                       "historical_control_updates_reused": sum(r["reused_historical_optimizer_updates"] for r in training_runs),
                                       "updates_executed_by_summary": 0},
              "cohort_policy": {"primary_development_groups_reported_separately": ["old71", "new24"],
                                 "supplemental_group": "new23", "supplemental_ids": meta["new23_ids"],
                                 "excluded_duplicate_ids": meta["excluded_duplicate_ids"], "overlap_evidence": meta["overlap_evidence"],
                                 "pooled95_primary_score_computed": False, "best_seed_selected": False,
                                 "seed_mean": "Equal arithmetic mean of the two fixed seeds; no best-seed or checkpoint selection"},
              "new_full_transcript_approvals": 0, "training_ready": False, "training_ready_flags_promoted": False,
              "service_weights_updated": False, "audio_read": False, "pixels_read": False, "teacher_features_read": False,
              "limitations": [*policy["limitations"],
                              "Development results only; no untouched final test or live webcam recognition validation.",
                              "new23 is an overlap sensitivity supplement; it does not replace full new24.",
                              "Equal updates/presentations do not imply equal frames, epochs, FLOPs or elapsed time.",
                              "Static/shuffled deltas are diagnostics and cannot alone establish lip-motion reliance."]}
    require(result["optimizer_accounting"]["new_updates_executed_in_step10"] == result["optimizer_accounting"]["historical_control_updates_reused"] == 6400,
            "Incorrect new/reused optimizer accounting")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluations", nargs=4, required=True, type=Path)
    parser.add_argument("--output", type=Path, default=BASE / "results/comparison.json")
    args = parser.parse_args()
    result = summarize(args.evaluations, args.output)
    print(json.dumps({"output": bind(args.output), "status": result["status"],
                      "means": result["two_seed_arithmetic_means"]}, ensure_ascii=False))
