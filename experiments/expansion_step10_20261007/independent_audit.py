"""Independent CPU audit of the fixed-budget data expansion experiment.

This imports only the hash-pinned previous independent numerical auditor.
No model, training runner, evaluation producer or mixed-loader module is imported.
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
import time
import unicodedata

import numpy as np

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
OLD = ROOT / "experiments/local_training_step05_20261007"
NUMERICAL_PATH = OLD / "verify_step05.py"
NUMERICAL_SHA = "a4462634845cf1a888abe5f0fae7cbe8b554a6edba271006b6d14c68833e5d76"
if hashlib.sha256(NUMERICAL_PATH.read_bytes()).hexdigest() != NUMERICAL_SHA:
    raise ValueError("Previous independent numerical auditor changed")
spec = importlib.util.spec_from_file_location("step10_independent_numerical_ancestor", NUMERICAL_PATH)
num = importlib.util.module_from_spec(spec)
spec.loader.exec_module(num)
same, require = num.same, num.require
read, bind, verify, sha = num.read_json, num.binding, num.verify_binding, num.sha
SEEDS = (42, 43)
VARIANTS = ("control_old441", "expanded510")
CONDITIONS = ("normal", "static", "shuffled")
HANDOFF = ROOT / "experiments/new_inputs_step09_20261007/candidate_handoff.json"
HANDOFF_SHA = "8135c7900ad5ad244e98ddb490fad2a05db115b221edba957cb030603a224ae6"


def collapse(raw, frames):
    require(type(frames) is int and frames > 0 and isinstance(raw, list) and len(raw) == frames,
            "Missing frame-level argmax IDs")
    require(all(type(t) is int and 0 <= t < 69 for t in raw), "Invalid argmax token")
    return [t for i, t in enumerate(raw) if t != 0 and (i == 0 or t != raw[i - 1])]


def compose(ids):
    """Independent Unicode syllable arithmetic; every malformed jamo becomes U+FFFD."""
    require(isinstance(ids, list) and all(type(t) is int and 1 <= t <= 68 for t in ids), "Invalid collapsed token")
    out, index, first_error = [], 0, None
    while index < len(ids):
        t = ids[index]
        if t == 68:
            out.append(" "); index += 1
        elif 1 <= t <= 19 and index + 1 < len(ids) and 20 <= ids[index + 1] <= 40:
            tail = ids[index + 2] - 40 if index + 2 < len(ids) and 41 <= ids[index + 2] <= 67 else 0
            out.append(chr(0xAC00 + ((t - 1) * 21 + ids[index + 1] - 20) * 28 + tail))
            index += 3 if tail else 2
        else:
            if first_error is None:
                grapheme = chr(0x1100 + t - 1) if t <= 19 else chr(0x1161 + t - 20) if t <= 40 else chr(0x11A8 + t - 41)
                first_error = f"ValueError: 잘못된 자모 순서/토큰 (위치 {index}): {grapheme!r}"
            out.append("\ufffd"); index += 1
    text = "".join(out)
    return {"prediction": text, "replacement_characters": text.count("\ufffd"),
            "strict": {"status": "ok" if first_error is None else "failed", "prediction": text if first_error is None else None, "error": first_error}}


def cohort_source():
    handoff = read(HANDOFF, HANDOFF_SHA)
    rows = {}
    for generation in ("old", "new"):
        ref = handoff["artifacts"][generation + "_data_plan"]
        verify(ref)
        for row in read(ref["path"])["records"]:
            require(row["id"] not in rows, "Repeated logical ID")
            rows[row["id"]] = row
    same(len(rows), 632, "Logical632")
    old_dev = handoff["development_groups"]["old71_same_training_speakers"]
    new_dev = handoff["development_groups"]["new24_all_original_items"]
    train = handoff["provisional_train_ids"]
    strip = lambda text: "".join(unicodedata.normalize("NFC", text).split())
    clean = [sid for sid in new_dev if strip(rows[sid]["text"]) not in {strip(rows[t]["text"]) for t in train}]
    groups = {"old71": old_dev, "new24": new_dev, "new23": clean}
    same([len(x) for x in groups.values()], [71, 24, 23], "Development denominators")
    same(set(new_dev) - set(clean), {"lip_J_1_F_05_C059_A_002_036"}, "Only known transcript-overlap exclusion")
    same(set(train) & set(old_dev + new_dev), set(), "Train/dev clip separation")
    same({rows[x]["speaker_id"] for x in train} & {rows[x]["speaker_id"] for x in new_dev}, set(), "New development speaker separation")
    return handoff, rows, groups


def independent_training_bindings(handoff, rows):
    """Read original saved uint8/tokens/FP32 features directly, without a loader."""
    entries = {r["id"]: r for r in handoff["records"]}
    clips = {}
    for sid in handoff["provisional_train_ids"]:
        item, row = entries[sid], rows[sid]
        verify(item["input"]); verify(item["teacher_feature"])
        require(item["role"] in ("old_train", "new_provisional_train"), "Held/development train payload")
        with np.load(item["input"]["path"], allow_pickle=False) as saved:
            pixels, tokens = saved["student_rgb"], saved["label_tokens"]
            meta = json.loads(str(saved["metadata_json"].item()))
        same(meta["record"], row, "Raw NPZ logical record")
        same(pixels.shape, (row["frames"], 96, 96, 3), "Original uint8 shape")
        same(pixels.dtype, np.dtype("uint8"), "Original uint8 precision")
        same(tokens.dtype, np.dtype("int64"), "Original labels precision")
        same(tokens.tolist(), row["token_ids"], "Original label tokens")
        with np.load(item["teacher_feature"]["path"], allow_pickle=False) as saved:
            features = saved["features"]
        same(features.shape, (row["frames"], 512), "Saved teacher feature shape")
        same(features.dtype, np.dtype("float32"), "Saved teacher precision")
        require(np.isfinite(features).all() and (np.linalg.norm(features, axis=1) > 1e-8).all(), "Invalid teacher target")
        clips[sid] = {"id": sid, "frames": len(pixels), "pixels_sha256": num.nested_hash(pixels),
                      "tokens_sha256": num.nested_hash(tokens.tolist()), "teacher_sha256": num.nested_hash(features)}
    require(len(clips) == 510, "Train510 payload count")
    variants = {}
    for variant, ids in (("control_old441", handoff["control_old_train_ids"]), ("expanded510", handoff["provisional_train_ids"])):
        values = {sid: clips[sid] for sid in sorted(ids)}
        variants[variant] = {"data_sha256": num.nested_hash(values), "clips": values}
    return variants


def validate_prediction(record, row, condition):
    for key, expected in (("id", row["id"]), ("speaker_id", row["speaker_id"]), ("source_stem", row["source_stem"]),
                          ("reference", row["text"]), ("condition", condition), ("split", "dev")):
        same(record[key], expected, "Evaluation identity " + key)
    same(record["source_split"], row["split"], "Original split retained")
    require(record["status"] in ("ok", "failed"), "Unknown evaluation status")
    if record["status"] == "ok":
        same(record["frames"], row["frames"], "Prediction frame count")
        ids = collapse(record["raw_argmax_ids"], row["frames"])
        same(record["collapsed_ids"], ids, "CTC blank/repeat handling")
        reconstructed = compose(ids)
        same(record["prediction"], reconstructed["prediction"], "Independent Hangul composition")
        same(record["replacement_characters"], reconstructed["replacement_characters"], "Malformed predicted token retention")
        same(record["strict_label_decoder_diagnostic"], reconstructed["strict"], "Independent strict diagnostic")
        same(record["error"], None, "Successful inference error")
    else:
        same(record["prediction"], "", "Failed input keeps empty prediction")
        require(isinstance(record["error"], str) and record["error"], "Failed input needs error")
        same(record["raw_argmax_ids"], [], "Failed input raw tokens")
        same(record["collapsed_ids"], [], "Failed input collapsed tokens")
        same(record["replacement_characters"], 0, "Failed input malformed tokens")
        same(record["strict_label_decoder_diagnostic"], None, "Failed input strict diagnostic")
    temporal = record["temporal"]
    if temporal is not None:
        positions = num.temporal_order(row["frames"], condition, row["id"])
        same(temporal["source_positions"], positions, "Independent temporal order")
        same(temporal["source_positions_sha256"], num.nested_hash(positions), "Order digest")
        same(temporal["frames"], row["frames"], "Temporal length")
        same(temporal["static_position"], row["frames"] // 2 if condition == "static" else None, "Middle static frame")
    else:
        require(record["status"] == "failed", "Successful result missing temporal evidence")
    same(record["metrics"], num.metric_counts(row["text"], record["prediction"]), "Independent per-clip metrics")


def validate_cohort_conditions(doc, ids, rows):
    same(doc["ids"], ids, "Full fixed cohort order")
    same(doc["record_count"], len(ids), "Full denominator")
    same(set(doc["conditions"]), set(CONDITIONS), "Temporal conditions")
    results = {}
    for condition in CONDITIONS:
        item = doc["conditions"][condition]
        same([r["id"] for r in item["records"]], ids, "Evaluation rows exact")
        for record in item["records"]:
            validate_prediction(record, rows[record["id"]], condition)
        aggregate = num.aggregate_metrics(item["records"])
        same(item["aggregate"], aggregate, "Independent pooled aggregate")
        per_speaker = {speaker: num.aggregate_metrics([r for r in item["records"] if r["speaker_id"] == speaker])
                       for speaker in sorted({rows[sid]["speaker_id"] for sid in ids})}
        same(item["per_speaker"], per_speaker, "Independent per-speaker metrics")
        same(item["sentences_with_replacement_characters"], sum(r["prediction"].count("\ufffd") > 0 for r in item["records"]), "Malformed sentence count")
        same(item["replacement_characters"], sum(r["prediction"].count("\ufffd") for r in item["records"]), "Malformed character count")
        results[condition] = aggregate
    return results


def validate_supplement(evaluation, groups):
    for condition in CONDITIONS:
        full = evaluation["cohorts"]["new24"]["conditions"][condition]["records"]
        expected = [r for r in full if r["id"] in set(groups["new23"])]
        same(evaluation["cohorts"]["new23"]["conditions"][condition]["records"], expected,
             "New23 must be a reuse of the fixed23 full-cohort predictions")


def compare(results):
    """Per-seed and two-seed mean differences, always expanded minus old."""
    output = {}
    for cohort in ("old71", "new24", "new23"):
        output[cohort] = {}
        for condition in CONDITIONS:
            per_seed = {}
            for seed in SEEDS:
                a = results[("control_old441", seed)][cohort][condition]
                b = results[("expanded510", seed)][cohort][condition]
                per_seed[str(seed)] = {"old_cer_percent": a["cer_percent"], "expanded_cer_percent": b["cer_percent"],
                                       "expanded_minus_old_cer_percentage_points": b["cer_percent"] - a["cer_percent"],
                                       "old_exact_sentences": a["exact_sentences"], "expanded_exact_sentences": b["exact_sentences"],
                                       "expanded_minus_old_exact_sentences": b["exact_sentences"] - a["exact_sentences"]}
            output[cohort][condition] = {"per_seed": per_seed,
                "mean_old_cer_percent": sum(x["old_cer_percent"] for x in per_seed.values()) / 2,
                "mean_expanded_cer_percent": sum(x["expanded_cer_percent"] for x in per_seed.values()) / 2,
                "mean_expanded_minus_old_cer_percentage_points": sum(x["expanded_minus_old_cer_percentage_points"] for x in per_seed.values()) / 2}
    return output


def validate_plan_contract(policy, plan, handoff, old_plan, old_run, actual_data):
    seed = plan["seed"]
    require(seed in SEEDS, "Unknown seed")
    for key, value in (("variant", "expanded510"), ("arm", "B_feature"), ("updates", 3200),
                       ("batch_size", 2), ("checkpoint_interval", 100),
                       ("ids", handoff["provisional_train_ids"]), ("training_data", actual_data["expanded510"]),
                       ("runtime", old_plan["runtime"]), ("optimizer", old_plan["optimizer"]),
                       ("loss", old_plan["loss"]), ("initial_state_hashes_expected", old_run["initial_state_hashes"]),
                       ("development_groups", handoff["development_groups"]), ("service_weights_updated", False)):
        same(plan[key], value, "Expanded plan " + key)
    same(plan["plan_sha256"], num.canonical_hash(plan), "Plan canonical digest")
    same(plan["schedule"], num.independent_schedule(plan["ids"], seed, 3200), "Independent expanded schedule")
    same(old_plan["training_data"], actual_data["control_old441"], "Historical441 exact input tensors")
    same(old_plan["schedule"], num.independent_schedule(handoff["control_old_train_ids"], seed, 3200), "Historical schedule")
    same(policy["runtime"], old_plan["runtime"], "Matched runtime")
    same(policy["optimizer"], old_plan["optimizer"], "Matched optimizer")
    same(policy["loss"], old_plan["loss"], "Matched loss")
    same(policy["controls"][str(seed)]["initial_state_hashes"], old_run["initial_state_hashes"], "Matched complete initialization")
    gate = policy["reuse_gates"][str(seed)]
    same(gate, {"old441_data_sha256": actual_data["control_old441"]["data_sha256"],
                "old441_tensor_bindings_exact": True, "fresh_initial_state_hashes": old_run["initial_state_hashes"],
                "fresh_initial_state_exact": True, "student_updates_executed": 0}, "Pre-update reuse evidence")
    forbidden = set(handoff["held_new_train_ids"])
    for ids in handoff["development_groups"].values():
        if isinstance(ids, list) and all(isinstance(x, str) for x in ids): forbidden.update(ids)
    same(set(plan["ids"]) & forbidden, set(), "Forbidden training roles")


def preflight():
    handoff, rows, groups = cohort_source()
    policy_path = BASE / "comparison_plan.json"
    policy = read(policy_path)
    manifest = read(BASE / "plans/freeze_manifest.json")
    same(manifest["status"], "frozen", "Plan freeze")
    same(manifest["experiment_plan"], bind(policy_path), "Frozen policy binding")
    same(manifest["new_optimizer_updates_planned"], 6400, "New update budget")
    same(manifest["optimizer_updates_executed"], 0, "Before-training freeze")
    for key, value in (("schema", "lipnet-step10-expansion-comparison-plan-v1"),
                       ("status", "frozen_before_expanded_training_and_new_evaluation"),
                       ("handoff", bind(HANDOFF)), ("seeds", list(SEEDS)), ("variants", list(VARIANTS)),
                       ("method", "B_feature"), ("updates_per_model", 3200), ("clips_per_update", 2),
                       ("new_optimizer_updates_planned", 6400), ("reused_historical_optimizer_updates", 6400),
                       ("provisional_train_ids", handoff["provisional_train_ids"]),
                       ("control_train_ids", handoff["control_old_train_ids"]),
                       ("held_train_ids", handoff["held_new_train_ids"]), ("development_groups", handoff["development_groups"]),
                       ("pending_listening", handoff["pending_listening"]), ("new_full_transcript_approvals", 0),
                       ("service_weights_updated", False), ("audio_used", False), ("training_ready_flags_promoted", False)):
        same(policy[key], value, "Policy " + key)
    for ref in policy["execution_code"]: verify(ref)
    verify(policy["mixed_loader_validation"])
    same(read(policy["mixed_loader_validation"]["path"])["status"], "passed", "Loader validation")
    actual_data = independent_training_bindings(handoff, rows)
    historical_path = OLD / "results/independent_audit.json"
    historical = read(historical_path, "821c722f43320ab672cd3ae773305b970a5d841908bde88b5729562e1771f982")
    same(historical["status"], "passed", "Historical evidence accepted")
    plans, old_plans, old_runs = {}, {}, {}
    for ref in manifest["plans"]:
        verify(ref); plan = read(ref["path"]); seed = plan["seed"]
        require(seed not in plans, "Duplicate frozen seed")
        same(ref, bind(BASE / f"plans/seed{seed}.json"), "Fixed seed plan path")
        same(plan["experiment_plan"], bind(policy_path), "Plan policy")
        same(plan["handoff"], bind(HANDOFF), "Plan handoff")
        control = policy["controls"][str(seed)]
        for key in ("historical_audit", "plan_file", "run_result", "segment_contract", "checkpoint", "checkpoint_sidecar"):
            verify(control[key])
        same(control["historical_audit"], bind(historical_path), "Pinned historical audit")
        matching = [r for r in historical["runs"] if r["seed"] == seed and r["arm"] == "B_feature"]
        same(len(matching), 1, "Unique historical audited control")
        accepted = matching[0]
        same(control["run_result"], accepted["run_result"], "Audited control result")
        same(control["plan_file"], accepted["plan"], "Audited control plan")
        same(control["checkpoint"], {k: accepted["checkpoint"][k] for k in ("path", "sha256")}, "Audited control checkpoint")
        old_plan, old_run = read(control["plan_file"]["path"]), read(control["run_result"]["path"])
        validate_plan_contract(policy, plan, handoff, old_plan, old_run, actual_data)
        entries = {r["id"]: r for r in handoff["records"]}
        selected_payloads = [entries[sid][field] for sid in handoff["provisional_train_ids"] for field in ("input", "teacher_feature")]
        num.validate_npz_binding_scope(plan["data_bindings"], selected_payloads)
        for reference in plan["data_bindings"]:
            if Path(reference["path"]).suffix != ".npz": verify(reference)
        plans[seed], old_plans[seed], old_runs[seed] = plan, old_plan, old_run
    same(set(plans), set(SEEDS), "Exactly paired seed plans")
    ev = policy["evaluation_contract"]
    same(ev["conditions"], list(CONDITIONS), "Fixed temporal conditions")
    for name, ids in groups.items(): same(ev["development"][name + "_ids"], ids, "Frozen development IDs")
    same(ev["declared_before_training"], True, "Evaluation declared first")
    corrected_path = OLD / "results/decode_correction/independent_audit.json"
    corrected = read(corrected_path, "080f5b27f5fb9f3304a6d34a7101058141d3abbf3cd7b20d91a273644555923a")
    same(ev["control_replay_audit"], bind(corrected_path), "Corrected control audit anchor")
    for seed in SEEDS:
        prior = next(x["evaluation"] for x in corrected["models"] if x["seed"] == seed and x["arm"] == "B_feature")
        same(ev["control_old71_replay"][str(seed)], prior, "Anchored corrected control output"); verify(prior)
    report = {"schema": "lipnet-step10-independent-preflight-v1", "status": "passed",
              "experiment_plan": bind(policy_path), "freeze_manifest": bind(BASE / "plans/freeze_manifest.json"),
              "handoff": bind(HANDOFF), "directly_read_training_inputs": 510, "directly_read_teacher_targets": 510,
              "data_sha256": {k: v["data_sha256"] for k, v in actual_data.items()},
              "control_tensor_identity_exact": True, "matched_initial_states_exact": True,
              "train_clip_counts": {"control_old441": 441, "expanded510": 510},
              "development_counts": {k: len(v) for k, v in groups.items()},
              "new_optimizer_updates_planned": 6400, "reused_historical_optimizer_updates": 6400,
              "policy_notes": ["Equal optimizer updates and utterance presentations, not equal frame counts or compute",
                               "No independent human transcript approval inferred from ASR agreement"],
              "auditor_code": [bind(Path(__file__)), bind(BASE / "test_independent_audit.py"), bind(NUMERICAL_PATH)]}
    return report, (handoff, rows, groups, policy, plans, old_plans, old_runs)


def checkpoint_chain(directory, final, allowed_root):
    """Rehash every completed boundary file; deserialize only final/resume states."""
    end, seen, boundaries = 3200, set(), {}
    identity, initial, history = final["identity"], final["initial_state_hashes"], final["history"]
    while True:
        directory = num.local(directory)
        require(directory.is_relative_to(allowed_root) and directory not in seen and len(seen) < 64,
                "Invalid checkpoint recovery chain")
        seen.add(directory)
        contract = read(directory / "segment_contract.json")
        same(contract["identity"], identity, "Recovery identity")
        same(contract["initial_state_hashes"], initial, "Recovery fresh state")
        start = contract["start_update"]
        require(type(start) is int and 0 <= start < end <= contract["stop_update"] <= 3200, "Segment bounds")
        wanted = list(range((start // 100 + 1) * 100, end + 1, 100))
        if end not in wanted: wanted.append(end)
        for update in wanted:
            info = num.checkpoint_info(directory / f"update_{update:04d}.pt")
            verify(info)
            same(num.local(info["path"]).stat().st_size, info["bytes"], "Boundary bytes")
            same(info["completed_updates"], update, "Boundary update")
            same(info["identity_sha256"], num.nested_hash(identity), "Boundary identity")
            same(info["state_hashes"], history[update - 1]["state_hashes"], "Boundary state/history")
            require(update not in boundaries, "Duplicate selected checkpoint boundary")
            boundaries[update] = info
        if start == 0:
            same(contract["resume_checkpoint"], None, "Fresh segment resume source")
            same(contract["resume_checkpoint_sha256"], None, "Fresh segment resume hash")
            break
        previous = num.local(contract["resume_checkpoint"])
        same(sha(previous), contract["resume_checkpoint_sha256"], "Recovery source hash")
        prior = num.load_checkpoint(num.checkpoint_info(previous))
        same(prior["identity"], identity, "Restored identity")
        same(prior["completed_updates"], start, "Restored update position")
        same(prior["history"], history[:start], "Restored history prefix")
        same(prior["initial_state_hashes"], initial, "Restored fresh state")
        directory, end = previous.parent, start
    require(set(range(100, 3201, 100)) <= set(boundaries), "Missing planned checkpoint boundary")
    return [boundaries[n] for n in sorted(boundaries)]


def audit_expanded_run(result_ref, seed, plan, rows):
    path = verify(result_ref)
    home = BASE / "runs" / f"seed{seed}"
    require(path.name == "run_result.json" and path.is_relative_to(home), "Expanded run location")
    result = read(path)
    for key, value in (("status", "completed"), ("seed", seed), ("variant", "expanded510"), ("arm", "B_feature"),
                       ("completed_updates", 3200), ("plan_file", bind(BASE / f"plans/seed{seed}.json")),
                       ("experiment_plan", bind(BASE / "comparison_plan.json")), ("plan_sha256", plan["plan_sha256"]),
                       ("dev_inputs_loaded", False), ("held_inputs_loaded", False), ("audio_used", False),
                       ("teacher_model_loaded", False), ("service_weights_updated", False)):
        same(result[key], value, "Completed expanded run " + key)
    for key, value in read(path.parent / "segment_result.json").items():
        same(result[key], value, "Engine evidence retained: " + key)
    payload = num.load_checkpoint(result["checkpoint"])
    identity = payload["identity"]
    paths = [ROOT / p for p in ("experiments/local_training_step03_20261007/runner.py",
             "experiments/feature_transfer_step07/train_compare.py", "backend/models/baseline.py",
             "backend/models/__init__.py", "backend/models/tokenizer.py")]
    for key, value in (("seed", seed), ("arm", "B_feature"), ("total_updates", 3200), ("batch_size", 2),
                       ("plan_sha256", plan["plan_sha256"]), ("schedule_sha256", num.nested_hash(plan["schedule"])),
                       ("data_sha256", plan["training_data"]["data_sha256"]), ("clip_bindings", plan["training_data"]["clips"]),
                       ("code_bindings", [bind(p) for p in paths]), ("runtime", plan["runtime"])):
        same(identity[key], value, "Checkpoint actual identity " + key)
    same(identity["optimization"], {"optimizer": "Adam", "lr": .001, "betas": [.9, .999], "eps": 1e-8,
         "weight_decay": 0., "gradient_clip": 1., "accumulation_clips": 2, "lambda": 1.,
         "ctc_device": "cpu", "precision": "float32"}, "Actual optimization recipe")
    same(result["identity_sha256"], num.nested_hash(identity), "Actual identity digest")
    same(result["runtime"], identity["runtime"], "Actual runtime")
    same(payload["completed_updates"], 3200, "Final update")
    history = payload["history"]
    stats = num.validate_history(history, plan, "B_feature", rows)
    same(result["updates"], history[result["start_update"]:], "Selected segment history suffix")
    same(payload["initial_state_hashes"], plan["initial_state_hashes_expected"], "Paired fresh state inside checkpoint")
    same(result["initial_state_hashes"], payload["initial_state_hashes"], "Result initial state")
    same(result["final_state_hashes"], payload["state_hashes"], "Result final state")
    same(history[-1]["state_hashes"], payload["state_hashes"], "History final state")
    state = payload["state"]
    for component, count, parameters in (("model", 31, 3287781), ("adapter", 2, 131584)):
        same(history[-1][component + "_sha256"], num.tensor_hash(state[component]), "Final tensor bytes")
        same(len(state[component]), count, "Tensor count")
        same(sum(t.numel() for t in state[component].values()), parameters, "Parameter count")
        require(all(t.dtype == num.torch.float32 for t in state[component].values()), "FP32 state")
        require(payload["initial_state_hashes"][component] != payload["state_hashes"][component], "Unchanged trained component")
    num.validate_optimizer(state, "B_feature", 3200)
    contract = read(path.parent / "segment_contract.json")
    same(contract["start_update"], result["start_update"], "Segment start")
    same(contract["stop_update"], 3200, "Segment final budget")
    same(contract["resume_checkpoint"], result["restored_from"], "Resume provenance")
    if result["start_update"]:
        restore = result["restore_verification"]
        same(restore["exact"], True, "Exact recovery")
        same(restore["expected_hashes"], history[result["start_update"] - 1]["state_hashes"], "Recovery expected state")
        same(restore["actual_hashes"], restore["expected_hashes"], "Recovery actual state")
    else: same(result["restore_verification"], None, "Unexpected recovery")
    boundaries = checkpoint_chain(path.parent, payload, home)
    selected = [b for b in boundaries if b["completed_updates"] > result["start_update"]]
    same(result["checkpoints"], selected, "Checkpoint inventory")
    same(result["checkpoint"], selected[-1], "Final fixed checkpoint selection")
    require(math.isfinite(result["elapsed_seconds"]) and result["elapsed_seconds"] > 0, "Elapsed segment time")
    return {"seed": seed, "variant": "expanded510", "run_result": result_ref, "plan": result["plan_file"],
            "checkpoint": result["checkpoint"], "initial_state_hashes": payload["initial_state_hashes"],
            "final_state_hashes": payload["state_hashes"], "checkpoint_count": len(boundaries),
            "checkpoint_bytes": sum(b["bytes"] for b in boundaries),
            "elapsed_seconds_reported_segment": result["elapsed_seconds"], "recovered": result["start_update"] > 0,
            "processed_frames": sum(sum(r["frames"]) for r in history), **stats}, result


def validate_control_replay(evaluation, prior):
    count = 0
    keys = ("id", "speaker_id", "source_stem", "reference", "prediction", "status", "error", "frames", "temporal",
            "raw_argmax_ids", "collapsed_ids", "replacement_characters", "metrics")
    for condition in CONDITIONS:
        current, previous = evaluation["cohorts"]["old71"]["conditions"][condition], prior["conditions"][condition]
        same(len(current["records"]), 71, "Control replay full denominator")
        same(len(previous["records"]), 71, "Prior corrected full denominator")
        for now, old in zip(current["records"], previous["records"]):
            for key in keys: same(now[key], old[key], "Fresh control replay " + key)
            same(now["strict_label_decoder_diagnostic"], old["strict_replay"], "Strict diagnostic replay")
            count += 1
        for key in ("aggregate", "per_speaker", "sentences_with_replacement_characters", "replacement_characters"):
            same(current[key], previous[key], "Fresh control aggregate replay")
    return count


def audit_evaluation(path, variant, seed, run, run_ref, plan_ref, policy, handoff, rows, groups):
    evaluation = read(path)
    for key, value in (("schema", "lipnet-step10-development-evaluation-v1"), ("status", "completed"),
                       ("seed", seed), ("variant", variant), ("arm", "B_feature"), ("completed_updates", 3200),
                       ("experiment_plan", bind(BASE / "comparison_plan.json")), ("training_plan", plan_ref),
                       ("run_result", run_ref), ("checkpoint", run["checkpoint"]),
                       ("initial_state_hashes", run["initial_state_hashes"]), ("runtime", policy["runtime"]),
                       ("inferred_distinct_records", 95), ("inferred_condition_records", 285), ("new_optimizer_updates", 0)):
        same(evaluation[key], value, "Evaluation provenance " + key)
    for key in ("audio_read", "original_media_read", "teacher_targets_read", "teacher_executed", "optimizer_executed",
                "held_training_records_read", "service_weights_updated"):
        same(evaluation[key], False, "Evaluation false side effect " + key)
    require(math.isfinite(evaluation["elapsed_seconds"]) and evaluation["elapsed_seconds"] > 0, "Evaluation elapsed time")
    for ref in evaluation["evaluation_code"]:
        require(ref in policy["execution_code"], "Evaluation code not frozen"); verify(ref)
    required_code = {str((BASE / "evaluation.py").resolve()), str((BASE / "mixed_artifacts.py").resolve())}
    require(required_code <= {r["path"] for r in evaluation["evaluation_code"]}, "Missing critical evaluation code bindings")
    entries = {r["id"]: r for r in handoff["records"]}
    for group in ("old71", "new24"):
        expected_npz = [entries[sid]["input"] for sid in groups[group]]
        num.validate_npz_binding_scope(evaluation["data_bindings"][group], expected_npz)
        for ref in evaluation["data_bindings"][group]: verify(ref)
    same(set(evaluation["cohorts"]), set(groups), "Fixed evaluation cohorts")
    metrics = {}
    for group, ids in groups.items():
        doc = evaluation["cohorts"][group]
        for key in ("model_state_before", "model_state_after"):
            same(doc[key], run["final_state_hashes"]["model"], "Student unchanged during inference")
        same(doc["model_state_unchanged"], True, "No model mutation")
        same(doc["aggregation_only"], group == "new23", "Supplement aggregation flag")
        expected_group = "new24" if group == "new23" else group
        for condition in CONDITIONS:
            for row in doc["conditions"][condition]["records"]:
                same(row["group"], expected_group, "Original evaluation cohort")
                require(math.isfinite(row["seconds"]) and row["seconds"] >= 0, "Per-item inference elapsed time")
        metrics[group] = validate_cohort_conditions(doc, ids, rows)
    validate_supplement(evaluation, groups)
    same(evaluation["cohorts"]["new23"]["parent_group"], "new24", "Supplement parent")
    same(evaluation["cohorts"]["new23"]["excluded_ids"], [sid for sid in groups["new24"] if sid not in groups["new23"]], "Fixed overlap exclusion")
    replay_count = 0
    if variant == "control_old441":
        ref = policy["evaluation_contract"]["control_old71_replay"][str(seed)]
        verify(ref); replay_count = validate_control_replay(evaluation, read(ref["path"]))
        replay = evaluation["old_control_replay"]
        for key, value in (("status", "passed"), ("compared_records", 213), ("mismatches", []),
                           ("prior_predictions_used_as_outputs", False), ("reference", ref)):
            same(replay[key], value, "Saved replay gate " + key)
    else: same(evaluation["old_control_replay"]["status"], "not_applicable_to_new_model", "New model replay role")
    return metrics, {"seed": seed, "variant": variant, "evaluation": bind(path),
                     "unique_inference_rows": 285, "control_replayed_rows": replay_count,
                     "failed_rows": sum(metrics[g][c]["failed_inputs"] for g in ("old71", "new24") for c in CONDITIONS)}


def close_tree(actual, expected, message):
    """Derived floating means permit rounding only; integers/IDs remain exact."""
    if isinstance(expected, dict):
        same(set(actual), set(expected), message + " keys")
        for key in expected: close_tree(actual[key], expected[key], message + "/" + str(key))
    elif isinstance(expected, list):
        same(len(actual), len(expected), message + " length")
        for i, (a, e) in enumerate(zip(actual, expected)): close_tree(a, e, message + "/" + str(i))
    elif isinstance(expected, float):
        require(type(actual) in (float, int) and math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-10), message)
    else: same(actual, expected, message)


def with_rates(metric):
    return {**metric, "exact_percent": 100 * metric["exact_sentences"] / metric["clips"]}


def delta(first, second):
    return {**{field.replace("_percent", "_percentage_points"): first[field] - second[field]
               for field in ("cer_percent", "wer_percent", "normalized_cer_percent", "exact_percent")},
            "exact_sentences": first["exact_sentences"] - second["exact_sentences"],
            "character_edits": first["character_edits"] - second["character_edits"]}


def changed_predictions(old, new):
    same([r["id"] for r in old], [r["id"] for r in new], "Paired prediction IDs")
    records = []
    for before, after in zip(old, new):
        changes = after["metrics"]["character_edits"] - before["metrics"]["character_edits"]
        records.append({"id": before["id"], "speaker_id": before["speaker_id"], "reference": before["reference"],
                        "control_prediction": before["prediction"], "expanded_prediction": after["prediction"],
                        "control_status": before["status"], "expanded_status": after["status"],
                        "control_metrics": before["metrics"], "expanded_metrics": after["metrics"],
                        "prediction_changed": before["prediction"] != after["prediction"],
                        "character_edits_change": changes,
                        "outcome": "improved" if changes < 0 else "worsened" if changes > 0 else "tied"})
    totals = {name: sum(r["outcome"] == name for r in records) for name in ("improved", "worsened", "tied")}
    totals.update(clips=len(records), changed_predictions=sum(r["prediction_changed"] for r in records),
                  unchanged_predictions=sum(not r["prediction_changed"] for r in records),
                  net_character_edits_change=sum(r["character_edits_change"] for r in records))
    return {"totals": totals, "records": records}


def preserved_preflight():
    path = BASE / "results/independent_preflight.json"
    original = read(path)
    same(original["status"], "passed", "Historical preflight status")
    snapshots = []
    for ref, filename in zip(original["auditor_code"][:2], ("independent_preflight_v1.py.txt", "test_independent_preflight_v1.py.txt")):
        snapshot = bind(BASE / "audit_snapshots" / filename)
        same(snapshot["sha256"], ref["sha256"], "Historical preflight code preserved byte-exact")
        snapshots.append({"original_code_reference": ref, "preserved_snapshot": snapshot, "byte_exact": True})
    verify(original["auditor_code"][2])
    return {"result": bind(path), "code_snapshots": snapshots,
            "note": "Original preflight paths were subsequently developed into the final auditor; these immutable snapshots preserve the exact code bytes recorded at preflight."}


def audit_comparison(evaluation_paths, run_summaries, run_results, policy):
    path = BASE / "results/comparison.json"
    result = read(path)
    for key, expected in (("schema", "lipnet-step10-expansion-comparison-v1"), ("status", "completed"),
                          ("experiment_plan", bind(BASE / "comparison_plan.json")), ("new_full_transcript_approvals", 0)):
        same(result[key], expected, "Summary identity " + key)
    for key in ("training_ready", "training_ready_flags_promoted", "service_weights_updated", "audio_read", "pixels_read", "teacher_features_read"):
        same(result[key], False, "Summary false flag " + key)
    same(result["summary_code"], [bind(BASE / "summarize.py"), bind(BASE / "test_summarize.py")], "Summary code bytes")
    refs = [bind(p) for p in evaluation_paths.values()]
    same(sorted(result["evaluations"], key=lambda x: x["path"]), sorted(refs, key=lambda x: x["path"]), "Summary four evaluation bindings")
    for ref in result["verified_evidence"]: verify(ref)
    metrics, docs, models, basis = {}, {}, [], []
    for seed in SEEDS:
        for variant in VARIANTS:
            key = variant, seed
            doc = docs[key] = read(evaluation_paths[key])
            table, temporal = {}, {}
            for group in ("old71", "new24", "new23"):
                table[group] = {}
                for condition in CONDITIONS:
                    rows = doc["cohorts"][group]["conditions"][condition]["records"]
                    table[group][condition] = {
                        "aggregate": with_rates(num.aggregate_metrics(rows)),
                        "per_speaker": {speaker: with_rates(num.aggregate_metrics([r for r in rows if r["speaker_id"] == speaker])) for speaker in sorted({r["speaker_id"] for r in rows})},
                        "replacement_characters": sum(r["prediction"].count("\ufffd") for r in rows),
                        "sentences_with_replacement_characters": sum("\ufffd" in r["prediction"] for r in rows)}
                    for row in rows:
                        basis.append({"seed": seed, "variant": variant, "group": group, "condition": condition,
                                      "id": row["id"], "speaker_id": row["speaker_id"], "status": row["status"],
                                      **num.metric_counts(row["reference"], row["prediction"])})
                temporal[group] = {c: delta(table[group][c]["aggregate"], table[group]["normal"]["aggregate"]) for c in ("static", "shuffled")}
            metrics[key] = table
            models.append({"seed": seed, "variant": variant, "groups": table, "temporal_minus_normal": temporal})
    close_tree(result["per_model"], models, "All summary model metrics")
    same(result["integer_accounting_rows"], basis, "Integer accounting including repeated supplemental rows")
    same(len(basis), 1416, "1140 distinct inference +276 supplemental rows")
    paired, means = [], {g: {c: {} for c in CONDITIONS} for g in ("old71", "new24", "new23")}
    for seed in SEEDS:
        groups = {}
        for group in means:
            groups[group] = {}
            for condition in CONDITIONS:
                old, new = (metrics[v, seed][group][condition] for v in VARIANTS)
                groups[group][condition] = {"expanded_minus_control": delta(new["aggregate"], old["aggregate"]),
                    "per_speaker": {s: {"control": old["per_speaker"][s], "expanded": new["per_speaker"][s],
                        "expanded_minus_control": delta(new["per_speaker"][s], old["per_speaker"][s])} for s in old["per_speaker"]},
                    "prediction_changes": changed_predictions(*(docs[v, seed]["cohorts"][group]["conditions"][condition]["records"] for v in VARIANTS))}
        paired.append({"seed": seed, "groups": groups})
    close_tree(result["paired_seed_comparisons"], paired, "Seed-paired metrics and prediction changes")
    for group in means:
        for condition in CONDITIONS:
            for variant in VARIANTS:
                means[group][condition][variant] = {field: sum(metrics[variant, seed][group][condition]["aggregate"][field] for seed in SEEDS) / 2
                    for field in ("cer_percent", "wer_percent", "normalized_cer_percent", "exact_percent", "exact_sentences")}
            deltas = [x["groups"][group][condition]["expanded_minus_control"] for x in paired]
            means[group][condition]["expanded_minus_control"] = {key: sum(x[key] for x in deltas) / 2 for key in deltas[0]}
    close_tree(result["two_seed_arithmetic_means"], means, "Equal two-seed mean comparison")
    temporal_means = {}
    for variant in VARIANTS:
        temporal_means[variant] = {}
        for group in means:
            temporal_means[variant][group] = {}
            for condition in ("static", "shuffled"):
                values = [delta(metrics[variant, seed][group][condition]["aggregate"],
                                metrics[variant, seed][group]["normal"]["aggregate"]) for seed in SEEDS]
                temporal_means[variant][group][condition] = {key: sum(item[key] for item in values) / 2 for key in values[0]}
    close_tree(result["two_seed_temporal_minus_normal_arithmetic_means"], temporal_means, "Two-seed temporal delta means")
    same(len(result["training_runs"]), 4, "Four training provenance summaries")
    seen = set()
    for row in result["training_runs"]:
        key = row["variant"], row["seed"]
        require(key not in seen and key in run_summaries, "Duplicate/unplanned training summary")
        seen.add(key); stats, run = run_summaries[key], run_results[key]
        expected_ids = policy["control_train_ids"] if key[0] == "control_old441" else policy["provisional_train_ids"]
        for field, expected in (("run_result", stats["run_result"]), ("plan", stats["plan"]), ("optimizer_updates", 3200),
                                ("new_optimizer_updates", 3200 if key[0] == "expanded510" else 0),
                                ("reused_historical_optimizer_updates", 3200 if key[0] == "control_old441" else 0),
                                ("clip_presentations", 6400), ("processed_frames", stats["processed_frames"]),
                                ("wall_seconds", stats["elapsed_seconds_reported_segment"]),
                                ("optimizer_update_seconds_sum", stats["seconds_optimizer_updates"]),
                                ("cuda_peak_allocated_bytes", stats["peak_allocated_bytes"]),
                                ("cuda_peak_reserved_bytes", stats["peak_reserved_bytes"])):
            close_tree(row[field], expected, "Training accounting " + field)
        clips = read(stats["plan"]["path"])["training_data"]["clips"]
        exposures = [{"id": sid, "cohort": "old441" if sid in set(policy["control_train_ids"]) else "new69",
                      "clip_frames": clips[sid]["frames"], "presentations": stats["per_clip_presentations"][sid],
                      "processed_frames": clips[sid]["frames"] * stats["per_clip_presentations"][sid]} for sid in expected_ids]
        same(row["per_clip_exposures"], exposures, "All per-clip exposure accounting")
        roles = {}
        for name in ("old441", "new69"):
            selected = [r for r in exposures if r["cohort"] == name]
            presentations = [r["presentations"] for r in selected]
            roles[name] = {"unique_clips": len(selected), "presentations": sum(presentations),
                "processed_frames": sum(r["processed_frames"] for r in selected),
                "presentations_min": min(presentations, default=0), "presentations_max": max(presentations, default=0),
                "mean_presentations_per_clip": sum(presentations) / len(selected) if selected else 0}
        close_tree(row["exposure_by_role"], roles, "New/old exposure accounting")
        windows = {name: {field: math.fsum(r[field] for r in records) / len(records) for field in ("loss", "ctc_loss", "feature_loss")}
                   for name, records in (("first100", run["updates"][:100]), ("last100", run["updates"][-100:]))}
        close_tree(row["loss_window_means"], windows, "Loss window accounting")
        same(row["loss_window_updates"], {"first100": [1, 100], "last100": [3101, 3200]}, "Loss windows")
    same(result["optimizer_accounting"], {"new_updates_executed_in_step10": 6400, "historical_control_updates_reused": 6400,
                                         "updates_executed_by_summary": 0}, "New vs reused optimization accounting")
    cp = result["cohort_policy"]
    for key, expected in (("primary_development_groups_reported_separately", ["old71", "new24"]),
                          ("supplemental_group", "new23"), ("pooled95_primary_score_computed", False), ("best_seed_selected", False)):
        same(cp[key], expected, "Summary cohort policy")
    meta = policy["evaluation_contract"]["development"]
    for field, source in (("supplemental_ids", "new23_ids"), ("excluded_duplicate_ids", "excluded_duplicate_ids"), ("overlap_evidence", "overlap_evidence")):
        same(cp[field], meta[source], "Summary overlap evidence")
    return {"comparison": bind(path), "all_model_and_speaker_metrics_recomputed": True,
            "integer_accounting_rows": len(basis), "unique_inference_rows": 1140,
            "paired_prediction_changes_recomputed": True, "training_exposure_and_compute_accounting_recomputed": True}


def audit():
    started = time.perf_counter()
    pre, context = preflight()
    handoff, rows, groups, policy, plans, old_plans, old_runs = context
    completed_path = BASE / "results/training_complete.json"
    completed = read(completed_path)
    for key, value in (("status", "completed"), ("experiment_plan", bind(BASE / "comparison_plan.json")),
                       ("new_optimizer_updates", 6400), ("reused_historical_optimizer_updates", 6400),
                       ("service_weights_updated", False)):
        same(completed[key], value, "Completed training manifest " + key)
    same(len(completed["run_results"]), 2, "Only two fresh completed runs")
    expanded_refs = {}
    for ref in completed["run_results"]:
        verify(ref); seed = read(ref["path"])["seed"]
        require(seed not in expanded_refs, "Duplicated completed seed")
        expanded_refs[seed] = ref
    same(set(expanded_refs), set(SEEDS), "Completed both fixed seeds")
    run_summaries, run_results, run_refs = {}, {}, {}
    for seed in SEEDS:
        control = policy["controls"][str(seed)]
        stats, result = num.audit_run(control["run_result"], seed, "B_feature", old_plans[seed], control["plan_file"], rows)
        stats.update(variant="control_old441", processed_frames=sum(sum(r["frames"]) for r in result["updates"]))
        run_summaries[("control_old441", seed)], run_results[("control_old441", seed)], run_refs[("control_old441", seed)] = stats, result, control["run_result"]
        stats, result = audit_expanded_run(expanded_refs[seed], seed, plans[seed], rows)
        same(result["initial_state_hashes"], old_runs[seed]["initial_state_hashes"], "Matched seed initialization")
        run_summaries[("expanded510", seed)], run_results[("expanded510", seed)], run_refs[("expanded510", seed)] = stats, result, expanded_refs[seed]
    evaluations = {}
    for path in sorted((BASE / "results").glob("evaluation_*.json")):
        value = read(path)
        if value.get("schema") != "lipnet-step10-development-evaluation-v1": continue
        key = (value["variant"], value["seed"])
        require(key not in evaluations and key[0] in VARIANTS and key[1] in SEEDS, "Duplicate/unplanned evaluation")
        evaluations[key] = path
    same(set(evaluations), {(v, s) for v in VARIANTS for s in SEEDS}, "Exactly four final evaluations")
    numerical, evidence = {}, []
    for key, path in evaluations.items():
        variant, seed = key
        plan_ref = policy["controls"][str(seed)]["plan_file"] if variant == "control_old441" else bind(BASE / f"plans/seed{seed}.json")
        metrics, item = audit_evaluation(path, variant, seed, run_results[key], run_refs[key], plan_ref, policy, handoff, rows, groups)
        numerical[key] = metrics; evidence.append(item)
    same(sum(x["unique_inference_rows"] for x in evidence), 1140, "Total fresh inference records")
    same(sum(x["control_replayed_rows"] for x in evidence), 426, "Exact old control replay records")
    temporal = []
    for (variant, seed), cohorts in numerical.items():
        temporal.append({"variant": variant, "seed": seed, "cohorts": {name: {
            c + "_minus_normal_cer_percentage_points": metrics[c]["cer_percent"] - metrics["normal"]["cer_percent"]
            for c in ("static", "shuffled")} for name, metrics in cohorts.items()}})
    summary_validation = audit_comparison(evaluations, run_summaries, run_results, policy)
    return {"schema": "lipnet-step10-independent-result-audit-v1", "status": "passed",
            "created_utc": datetime.now(timezone.utc).isoformat(), "experiment_plan": bind(BASE / "comparison_plan.json"),
            "freeze_manifest": bind(BASE / "plans/freeze_manifest.json"), "training_complete": bind(completed_path),
            "comparison": bind(BASE / "results/comparison.json"),
            "handoff": bind(HANDOFF), "preflight_recomputed": pre, "runs": list(run_summaries.values()),
            "preserved_preflight": preserved_preflight(), "comparison_validation": summary_validation,
            "evaluations": evidence, "comparisons": compare(numerical), "temporal_diagnostics": temporal,
            "unique_inference_rows": 1140, "old_control_replay_rows": 426,
            "new_optimizer_updates": 6400, "reused_historical_optimizer_updates": 6400,
            "all_fixed_inputs_and_raw_tokens_audited": True, "new_full_transcript_approvals": 0,
            "training_ready_flags_promoted": False, "service_weights_updated": False,
            "pending_listening": handoff["pending_listening"],
            "limits": ["No fresh optimization replay by the auditor; final CPU tensor/optimizer states and all saved boundary bytes verified",
                       "Intermediate boundary states are cross-checked through sidecars and full file digests; only final/resume tensors deserialized",
                       "Input audit reuses frozen source crop audit; no new original-media decode or human listening",
                       "Matched budget means updates and utterance presentations; processed frames, time, and epochs differ",
                       "ASR-selected provisional69 and these development cohorts do not establish label truth or webcam generalization"],
            "auditor_code": [bind(Path(__file__)), bind(BASE / "test_independent_audit.py"), bind(NUMERICAL_PATH)],
            "elapsed_seconds": time.perf_counter() - started}


def save(path, value):
    path = Path(path).resolve()
    require(path.is_relative_to(BASE / "results"), "Audit output outside new results")
    path.parent.mkdir(exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    report = preflight()[0] if args.preflight else audit()
    output = BASE / "results" / ("independent_preflight.json" if args.preflight else "independent_audit.json")
    save(output, report)
    print(json.dumps({"status": report["status"], "output": str(output), "sha256": sha(output)}))
