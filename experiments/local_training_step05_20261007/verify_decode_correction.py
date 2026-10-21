"""Read-only independent audit of the separate post-hoc decoder correction.

No model, tokenizer, training runner, or evaluation producer is imported. Saved
frame IDs are collapsed independently; Hangul syllables use Unicode arithmetic.
The completed original audit supplies immutable checkpoint-state evidence.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import verify_step05 as old_audit

BASE, ROOT = old_audit.BASE, old_audit.ROOT
OUTPUT = BASE / "results/decode_correction"
PLAN = OUTPUT / "posthoc_plan.json"
ORIGINAL_AUDIT = BASE / "results/independent_audit.json"
ORIGINAL_AUDIT_SHA = "821c722f43320ab672cd3ae773305b970a5d841908bde88b5729562e1771f982"
PAIRS = tuple((seed, arm) for seed in (42, 43) for arm in ("A_ctc", "B_feature"))
CONDITIONS = old_audit.CONDITIONS
require, same = old_audit.require, old_audit.same
bind, verify, read = old_audit.binding, old_audit.verify_binding, old_audit.read_json
INVALID_PREFIX = "ValueError: 잘못된 자모 순서/토큰"
SUMMARY_KEYS = ("aggregate", "original_aggregate", "posthoc_minus_original_cer_pp",
                "corrected_invalid_jamo_failures", "unchanged_original_successes",
                "sentences_with_replacement_characters", "replacement_characters")


def collapse(raw: list[int], frames: int) -> list[int]:
    require(type(frames) is int and frames > 0 and isinstance(raw, list) and len(raw) == frames,
            "Missing or wrong-length frame argmax IDs")
    require(all(type(token) is int and 0 <= token <= 68 for token in raw), "Invalid frame argmax ID")
    # Compare with the actual previous frame, including blank, before dropping it.
    return [token for index, token in enumerate(raw)
            if token and (index == 0 or raw[index - 1] != token)]


def grapheme(token: int) -> str:
    if 1 <= token <= 19:
        return chr(0x1100 + token - 1)
    if 20 <= token <= 40:
        return chr(0x1161 + token - 20)
    if 41 <= token <= 67:
        return chr(0x11A8 + token - 41)
    same(token, 68, "Invalid nonblank token")
    return " "


def reconstruct(tokens: list[int]) -> dict:
    """Independent 19x21x28 Hangul composition and first strict-error evidence."""
    require(isinstance(tokens, list) and all(type(t) is int and 1 <= t <= 68 for t in tokens),
            "Invalid collapsed token ID")
    text, first_error, index = [], None, 0
    while index < len(tokens):
        token = tokens[index]
        if token == 68:
            text.append(" ")
            index += 1
        elif 1 <= token <= 19 and index + 1 < len(tokens) and 20 <= tokens[index + 1] <= 40:
            lead, vowel = token - 1, tokens[index + 1] - 20
            tail = tokens[index + 2] - 40 if index + 2 < len(tokens) and 41 <= tokens[index + 2] <= 67 else 0
            text.append(chr(0xAC00 + (lead * 21 + vowel) * 28 + tail))
            index += 3 if tail else 2
        else:
            if first_error is None:
                first_error = f"{INVALID_PREFIX} (위치 {index}): {grapheme(token)!r}"
            text.append("\ufffd")
            index += 1
    prediction = "".join(text)
    return {"prediction": prediction, "replacement_characters": prediction.count("\ufffd"),
            "strict_replay": {"status": "ok" if first_error is None else "failed",
                              "prediction": prediction if first_error is None else None,
                              "error": first_error}}


def validate_row(record: dict, previous: dict, source: dict, condition: str) -> dict:
    for key in ("id", "speaker_id", "source_stem", "split", "condition", "reference", "frames", "temporal"):
        same(record[key], previous[key], "Changed original row identity/input condition: " + key)
    same(record["frames"], source["frames"], "Changed source frame count")
    same(record["original"], {key: previous[key] for key in ("status", "prediction", "error", "metrics")},
         "Changed original row snapshot")
    # This audited cohort has no original failures outside strict jamo composition.
    # A new input/inference failure must not masquerade as a decoder-only change.
    eligible = previous["status"] == "ok" or str(previous["error"]).startswith(INVALID_PREFIX)
    require(eligible, "Original non-decoder failure needs a separate audit scope")
    tokens = collapse(record["raw_argmax_ids"], source["frames"])
    same(record["collapsed_ids"], tokens, "Incorrect adjacent-repeat/blank CTC collapse")
    decoded = reconstruct(tokens)
    for key in ("prediction", "replacement_characters", "strict_replay"):
        same(record[key], decoded[key], "Independent predicted-jamo reconstruction differs: " + key)
    same(record["status"], "ok", "New runtime failure in a decoder-only replay")
    same(record["error"], None, "Unexpected corrected-row error")
    same(record["strict_replay_matches_original"], True, "Original strict replay not established")
    replay = decoded["strict_replay"]
    same(replay["status"], previous["status"], "Original strict status did not reproduce")
    same(replay["error"], previous["error"], "Original strict error did not reproduce exactly")
    if previous["status"] == "ok":
        same(record["prediction"], previous["prediction"], "Originally valid text changed")
        same(record["original_success_prediction_match"], True, "Original valid-output match missing")
        same(record["correction"], "unchanged_valid_prediction", "Unexpected successful-output correction")
    else:
        same(previous["prediction"], "", "Original failed row did not retain empty prediction")
        require(decoded["replacement_characters"] > 0, "Malformed jamo disappeared")
        same(record["original_success_prediction_match"], None, "Failed output presented as original success")
        same(record["correction"], "invalid_predicted_jamo_retained_as_replacement_character", "Wrong correction category")
    same(record["metrics"], old_audit.metric_counts(source["text"], record["prediction"]),
         "Independent corrected metrics differ")
    return decoded


def validate_condition(value: dict, previous: dict, sources: list[dict], condition: str) -> dict:
    checked = old_audit.validate_condition(value, condition, sources)
    old_checked = old_audit.validate_condition(previous, condition, sources)
    for row, old, source in zip(value["records"], previous["records"], sources, strict=True):
        validate_row(row, old, source, condition)
    same(value["original_aggregate"], old_checked["aggregate"], "Original aggregate was changed")
    same(value["posthoc_minus_original_cer_pp"],
         checked["aggregate"]["cer_percent"] - old_checked["aggregate"]["cer_percent"],
         "Post-hoc minus original CER sign/value differs")
    expected = {
        "corrected_invalid_jamo_failures": sum(r["status"] == "failed" for r in previous["records"]),
        "unchanged_original_successes": sum(r["status"] == "ok" for r in previous["records"]),
        "sentences_with_replacement_characters": sum(r["prediction"].count("\ufffd") > 0 for r in value["records"]),
        "replacement_characters": sum(r["prediction"].count("\ufffd") for r in value["records"])}
    for key, result in expected.items():
        same(value[key], result, "Incorrect correction diagnostics: " + key)
    return {**checked, **expected, "posthoc_minus_original_cer_pp": value["posthoc_minus_original_cer_pp"]}


def validate_provenance(value: dict, previous: dict, entry: dict, plan_ref: dict, run: dict, freeze_time: str) -> None:
    for key, expected in (("schema", "lipnet-step05-posthoc-prediction-decode-evaluation-v1"),
            ("status", "completed"), ("analysis_type", "posthoc_decoder_correction"),
            ("posthoc_plan", plan_ref), ("original_evaluation", entry["original_evaluation"]),
            ("seed", entry["seed"]), ("arm", entry["arm"]), ("completed_updates", 3200), ("record_count", 71),
            ("new_optimizer_updates", 0), ("audio_read", False), ("teacher_targets_read", False),
            ("teacher_executed", False), ("service_weights_updated", False), ("official_frozen_results_replaced", False),
            ("model_state_before", run["final_state_hashes"]["model"]),
            ("model_state_after", run["final_state_hashes"]["model"]), ("model_state_unchanged", True)):
        same(value[key], expected, "Incorrect post-hoc provenance: " + key)
    for key in ("ids", "plan", "run_result", "checkpoint", "full_data_plan", "data_bindings", "runtime"):
        same(value[key], previous[key], "Changed original evaluation binding/scope: " + key)
    same(len(value["ids"]), 71, "Wrong development denominator")
    same(len(set(value["ids"])), 71, "Duplicated development IDs")
    same(set(value["conditions"]), set(CONDITIONS), "Changed condition scope")
    require(datetime.fromisoformat(value["created_utc"]) >= datetime.fromisoformat(freeze_time),
            "Evaluation predates frozen correction plan")
    require(isinstance(value["limits"], list) and any("argmax" in line for line in value["limits"]),
            "Original token-path limitation omitted")


def validate_posthoc_plan(plan: dict, audited: dict) -> None:
    for key, expected in (("schema", "lipnet-step05-posthoc-prediction-decode-plan-v1"),
            ("status", "frozen_before_inference"), ("analysis_type", "posthoc_decoder_correction"),
            ("original_protocol_and_results_preserved", True), ("new_optimizer_updates", 0)):
        same(plan[key], expected, "Wrong post-hoc plan contract: " + key)
    same([(e["seed"], e["arm"]) for e in plan["models"]], list(PAIRS), "Wrong four-model scope")
    paths = [BASE / "evaluate_prediction_decode.py", BASE / "test_evaluate_prediction_decode.py",
             BASE / "evaluate.py", BASE / "full_artifacts.py", ROOT / "backend/models/tokenizer.py",
             ROOT / "experiments/olkavs_official/diagnostic_utils.py"]
    same(plan["code_bindings"], [bind(path) for path in paths], "Changed post-hoc execution code or tests")
    runs = {(r["seed"], r["arm"]): r for r in audited["runs"]}
    evals = {(r["seed"], r["arm"]): r["artifact"] for r in audited["evaluation_metrics"]}
    for entry in plan["models"]:
        key = (entry["seed"], entry["arm"])
        run, checkpoint = runs[key], runs[key]["checkpoint"]
        same(entry["original_evaluation"], evals[key], "Original evaluation differs from completed independent audit")
        for name in ("plan", "run_result"):
            same(entry[name], run[name], "Changed audited run binding: " + name)
        same(entry["checkpoint"], {"path": checkpoint["path"], "sha256": checkpoint["sha256"]}, "Changed final checkpoint")
        same(entry["checkpoint_sidecar"], {"path": checkpoint["sidecar"], "sha256": checkpoint["sidecar_sha256"]}, "Changed checkpoint sidecar")
        same(old_audit.local(entry["output"]), OUTPUT / f"evaluation_seed{key[0]}_{key[1]}.json", "Unexpected correction output path")
        for name in ("original_evaluation", "plan", "run_result", "checkpoint", "checkpoint_sidecar"):
            verify(entry[name])
        original_plan = read(entry["plan"]["path"])
        for ref in original_plan["execution_code"] + [original_plan["full_data_plan"]]:
            verify(ref)


def audit(output: Path) -> dict:
    started = time.perf_counter()
    audited = read(ORIGINAL_AUDIT, ORIGINAL_AUDIT_SHA)
    same(audited["status"], "passed", "Original final audit did not pass")
    verify(audited["code"], BASE / "verify_step05.py")
    verify(audited["tests"], BASE / "test_verify_step05.py")
    parent = read(old_audit.PARENT, old_audit.PARENT_SHA)
    sources = [r for r in parent["records"] if r["split"] == "dev"]
    same(len(sources), 71, "Changed independent development source scope")
    plan_ref, plan = bind(PLAN), read(PLAN)
    validate_posthoc_plan(plan, audited)
    summary_path = OUTPUT / "summary.json"
    summary = read(summary_path)
    for key, expected in (("schema", "lipnet-step05-posthoc-prediction-decode-summary-v1"), ("status", "completed"),
            ("analysis_type", "posthoc_decoder_correction"), ("posthoc_plan", plan_ref),
            ("official_frozen_results_replaced", False), ("new_optimizer_updates", 0)):
        same(summary[key], expected, "Incorrect post-hoc summary: " + key)
    expected_refs = [bind(entry["output"]) for entry in plan["models"]]
    same(summary["evaluations"], expected_refs, "Missing or changed correction evaluations")
    runs = {(r["seed"], r["arm"]): r for r in audited["runs"]}
    metrics, raw_frames, rows_checked, models_summary = {}, 0, 0, []
    per_model = []
    for entry, output_ref in zip(plan["models"], expected_refs, strict=True):
        key = (entry["seed"], entry["arm"])
        previous = read(entry["original_evaluation"]["path"])
        value = read(output_ref["path"])
        validate_provenance(value, previous, entry, plan_ref, runs[key], plan["created_utc"])
        metrics[key] = {c: validate_condition(value["conditions"][c], previous["conditions"][c], sources, c)
                        for c in CONDITIONS}
        raw_frames += sum(len(r["raw_argmax_ids"]) for data in value["conditions"].values() for r in data["records"])
        rows_checked += sum(len(data["records"]) for data in value["conditions"].values())
        models_summary.append({"seed": key[0], "arm": key[1], "conditions": {
            c: {name: value["conditions"][c][name] for name in SUMMARY_KEYS} for c in CONDITIONS}})
        per_model.append({"seed": key[0], "arm": key[1], "evaluation": output_ref, "conditions": metrics[key]})
        print(f"[independent decode audit] seed{key[0]}/{key[1]}: 71x3 raw-token/text/metric rows passed", flush=True)
    same(summary["models"], models_summary, "Summary model diagnostics differ from independently checked outputs")
    same(rows_checked, 852, "Wrong corrected evaluation denominator")
    same(raw_frames, sum(r["frames"] for r in sources) * 12, "Wrong total frame argmax evidence")
    verify(plan_ref)
    for ref in expected_refs:
        verify(ref)
    result = {"schema": "lipnet-step05-independent-decode-correction-audit-v1", "status": "passed",
        "created_utc": datetime.now(timezone.utc).isoformat(), "code": bind(Path(__file__)),
        "tests": bind(BASE / "test_verify_decode_correction.py"), "independent_metric_helper": audited["code"],
        "original_audit": bind(ORIGINAL_AUDIT), "posthoc_plan": plan_ref, "summary": bind(summary_path),
        "analysis_type": "posthoc_decoder_correction", "model_count": 4, "evaluation_rows": rows_checked,
        "raw_frame_argmax_ids_checked": raw_frames, "all_strict_replays_match_original": True,
        "all_valid_original_predictions_unchanged": True, "all_invalid_jamos_retained": True,
        "models": per_model, "B_minus_A_comparison": old_audit.compare_conditions(metrics),
        "original_B_minus_A_comparison": audited["B_minus_A_comparison"],
        "original_frozen_artifacts_preserved": True, "inference_repeated": False,
        "model_or_tokenizer_or_producer_code_imported": False, "input_npz_or_media_read": False,
        "new_optimizer_updates": 0, "elapsed_seconds": time.perf_counter() - started,
        "limits": ["Saved argmax evidence, text reconstruction and metrics are independently checked; inference is not repeated.",
                   "The original strict evaluations did not save argmax IDs; their exact prior token paths cannot be established retrospectively.",
                   "Final checkpoint bytes match the original passed audit; model-state immutability during replay is checked against recorded before/after digests.",
                   "This is a separate post-hoc implementation correction, not a replacement for the frozen original comparison or an untouched final test."]}
    output = old_audit.local(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def main() -> None:
    old_audit.torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT / "independent_audit.json")
    args = parser.parse_args()
    result = audit(args.output)
    print(json.dumps({"status": result["status"], "evaluation_rows": result["evaluation_rows"],
                      "output": str(args.output.resolve())}), flush=True)


if __name__ == "__main__":
    main()
