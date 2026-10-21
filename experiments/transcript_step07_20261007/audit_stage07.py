"""Independent read-only Stage7 label/context evidence audit; never runs models.

Uses the frozen independent Step6 verifier only for file/text/ByteLevel helpers.
Does not import the context recognizer, target producer, or production tokenizer.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
from itertools import product
import json
import math
from pathlib import Path
import re
import sys
import time
import unicodedata
import wave

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
STEP6 = BASE.parent / "new_data_step06_20261007"
HELPER = STEP6 / "verify_asr.py"
HELPER_SHA = "ea1b21daa0eb835d976228358efbb208375553e2deb1f524ba99fe6e207a70db"
if hashlib.sha256(HELPER.read_bytes()).hexdigest() != HELPER_SHA:
    raise ValueError("Frozen independent helper changed")
spec = importlib.util.spec_from_file_location("stage7_frozen_independent_helpers", HELPER)
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
require, same, close, read, bind, verify = old.require, old.same, old.close, old.read, old.bind, old.verify
TOKENIZER_SHA = "c73b1c393a739cfd0de03d9bd310e1a6d6339949d2add7121f98794fd7020714"
CONTEXT_FIELDS = set("id split speaker_id source_stem reference start_time end_time schema context_plan context_audio original_audio original_interval_in_context baseline status full_context_text raw_segments transcription_info error context_segment_boundary_flags decoder_confidence_flags inference_seconds human_listening_confirmed automatic_approval label_modified training_ready words selected_word_indices selected_text all_word_text boundary_cues word_timing_flags full_context_comparison selected_comparison".split())
CUE_NAMES = ("crosses_label_start", "crosses_label_end", "near_label_start", "near_label_end")


def raw_sha(text):
    return hashlib.sha256(text.encode("utf8")).hexdigest()


def parse_independent(text):
    """Independent character scanner for exact flat alternatives, with spans."""
    require(type(text) is str, "Raw text type")
    def structural(char):
        return char in "/\\⁄∕／＼" or unicodedata.category(char) in ("Ps", "Pe")
    segments, alternatives, i, last = [], [], 0, 0
    while i < len(text):
        if not structural(text[i]):
            i += 1
            continue
        start = i
        if text[i] != "(":
            raise ValueError("Unsupported structural character")
        close_a = text.find(")", i+1)
        require(close_a >= 0 and text[close_a:close_a+3] == ")/(", "Malformed alternative")
        close_b = text.find(")", close_a+3)
        require(close_b >= 0, "Unclosed alternative")
        ranges = [[start+1, close_a], [close_a+3, close_b]]
        options = [text[a:b] for a, b in ranges]
        require(all(option.strip() and not any(structural(char) for char in option) for option in options), "Nested/empty alternative")
        if last < start:
            segments.append({"kind": "text", "span": [last, start], "raw": text[last:start]})
        item = {"alternative_id": "alt_" + str(len(alternatives)), "span": [start, close_b+1],
                "raw": text[start:close_b+1], "options": options, "option_spans": ranges}
        alternatives.append(item)
        segments.append({"kind": "alternative", **item})
        i = last = close_b + 1
    if last < len(text):
        segments.append({"kind": "text", "span": [last, len(text)], "raw": text[last:]})
    require("".join(row["raw"] for row in segments) == text, "Raw reconstruction")
    return {"status": "alternatives" if alternatives else "plain", "raw_text": text,
            "segments": segments, "alternatives": alternatives, "error": None}


def target(text, frames):
    """Arithmetic Hangul decomposition, independent of production Unicode NFD."""
    normalized = old.normalize(text)
    require(normalized and all(char == " " or 0xAC00 <= ord(char) <= 0xD7A3 for char in normalized), "Unsupported target")
    tokens = []
    for char in normalized:
        if char == " ":
            tokens.append(68)
        else:
            syllable = ord(char) - 0xAC00
            tokens.extend([1 + syllable // 588, 20 + syllable % 588 // 28])
            tail = syllable % 28
            if tail:
                tokens.append(40 + tail)
    minimum = len(tokens) + sum(tokens[i] == tokens[i-1] for i in range(1, len(tokens)))
    return {"normalized_text": normalized, "tokens": tokens,
            "tokens_sha256": hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode("ascii")).hexdigest(),
            "ctc_min_frames": minimum, "student_frames": frames, "ctc_feasible": minimum <= frames}


def tokenizer_metadata():
    vocab = ["<blank>"] + [chr(n) for a, b in ((0x1100, 0x1113), (0x1161, 0x1176), (0x11A8, 0x11C3)) for n in range(a, b)] + [" "]
    return {"tokenizer_version": "ko-jamo-v2", "normalization_version": "ko-text-v1",
            "vocab_sha256": hashlib.sha256(json.dumps(vocab, ensure_ascii=False, separators=(",", ":")).encode("utf8")).hexdigest(), "num_classes": 69}


def check_target_record(row, source, overrides):
    for key in ("id", "split", "speaker_id", "source_text", "start_time", "end_time", "label_sha256"):
        same(row[key], source[key], "Target immutable " + key)
    same(row["source_stem"], source["id"].rsplit("_", 1)[0], "Target source")
    same(row["raw_text_sha256"], raw_sha(source["source_text"]), "Raw label digest")
    same(row["original_normalized_text"], source["text"], "Previous normalized label")
    same(row["student_frames"], source["expected_frames"], "Student frames")
    same(row["tokenizer"], tokenizer_metadata(), "Tokenizer metadata")
    require(row["policy_version"] == "ko-alternative-target-v1", "Target policy")
    require(all(row[key] is False for key in ("training_ready", "full_transcript_human_approved", "human_boundary_approved", "asr_used_to_choose_target")), "Unwarranted target approval")
    parsed = parse_independent(source["source_text"])
    same(row["parser"], parsed, "Independent parser structure/spans")
    require(len(parsed["alternatives"]) <= 4, "Candidate limit")
    previous = target(source["source_text"], source["expected_frames"])
    same(row["previous_target"], previous, "Historical target")
    same(previous["normalized_text"], source["text"], "Historical normalized selection")
    same(previous["ctc_min_frames"], source["ctc_min_frames"], "Historical CTC")
    candidates = []
    for index, choice in enumerate(product((0, 1), repeat=len(parsed["alternatives"]))):
        choices = [{"alternative_id": "alt_" + str(i), "option": option} for i, option in enumerate(choice)]
        selected = dict((item["alternative_id"], item["option"]) for item in choices)
        rendered = "".join(part["raw"] if part["kind"] == "text" else part["options"][selected[part["alternative_id"]]] for part in parsed["segments"])
        candidates.append({"candidate_id": "candidate_" + str(index), "choices": choices,
                           "raw_candidate_text": rendered, **target(rendered, source["expected_frames"])})
    same(row["candidates"], candidates, "All candidate text/tokens/CTC")
    if parsed["status"] == "plain":
        require(source["id"] not in overrides, "Ordinary label override")
        selected = candidates[0]
        require(row["ordinary_tokens_identical"] is True and row["selection_evidence"] is None, "Ordinary target preservation")
        status = "unchanged_ordinary_target"
    elif source["id"] not in overrides:
        require(row["target_status"] == "needs_alternative_resolution" and row["canonical_target"] is None
                and row["canonical_tokens"] is None and row["selection_evidence"] is None, "Unresolved alternative selected")
        return
    else:
        override = overrides[source["id"]]
        require(override["raw_text_sha256"] == raw_sha(source["source_text"]), "Choice raw hash")
        path = verify(override["evidence"])
        evidence = read(path)
        require(evidence["schema"] == "lipnet-step07-alternative-resolution-evidence-v1"
                and evidence["id"] == source["id"] and evidence["raw_text_sha256"] == override["raw_text_sha256"], "Choice evidence identity")
        same(evidence["choices"], override["choices"], "Individual choices")
        require(evidence["evidence_type"] == "user_listening" and evidence["scope"] == "alternative_choice_only"
                and evidence["full_transcript_human_approved"] is False and evidence["training_ready"] is False
                and evidence["original_labels_modified"] is False, "User choice scope")
        require(evidence["selection_statement"] == "‘애쓰는’이 한 번 들림" and evidence["selected_alternative_text"] == "애쓰는", "Specific user answer")
        same(row["selection_evidence"], {"binding": override["evidence"], "evidence": evidence}, "Preserved user evidence")
        selected = next(item for item in candidates if item["choices"] == override["choices"])
        require(row["ordinary_tokens_identical"] is None, "Alternative ordinary claim")
        status = "resolved_alternative_choice"
    same(row["selected_candidate_id"], selected["candidate_id"], "Selected candidate")
    same(row["ctc_min_frames"], selected["ctc_min_frames"], "Selected CTC")
    same(row["ctc_feasible"], selected["ctc_feasible"], "Selected feasibility")
    if selected["ctc_feasible"]:
        same(row["canonical_target"], selected["normalized_text"], "Canonical target")
        same(row["canonical_tokens"], selected["tokens"], "Canonical token sequence")
        require(row["target_status"] == status and row["hold_reason"] is None, "Target state")
    else:
        require(row["target_status"] == "hold_ctc_infeasible" and row["canonical_target"] is None
                and row["canonical_tokens"] is None, "Infeasible target not held")


def interval(clip):
    context, audio = clip["context_audio"], clip["audio"]
    require(context["source_sha256"] == audio["source_sha256"] and context["clock"] == audio["clock"] == "original_label_wav", "Source clock")
    start = clip["start_time"] - context["start_seconds"]
    end = clip["end_time"] - context["start_seconds"]
    close(start, .5, "Context prefix", 1e-8)
    close(context["end_seconds"] - clip["end_time"], .5, "Context suffix", 1e-8)
    duration = context["frames"] / 16000
    close(duration, context["end_seconds"] - context["start_seconds"], "Context PCM length", 2/16000)
    close(end - start, audio["frames"] / 16000, "Original PCM length", 2/16000)
    require(0 < start < end < duration, "Interval bounds")
    return {"start_seconds": start, "end_seconds": end, "context_duration_seconds": duration,
            "original_duration_seconds": clip["end_time"] - clip["start_time"]}


def interpret(segments, period):
    start, end, duration = (period[key] for key in ("start_seconds", "end_seconds", "context_duration_seconds"))
    words, flags = [], set()
    cues = {name: [] for name in CUE_NAMES}
    previous_end = -math.inf
    for si, segment in enumerate(segments):
        require(type(segment["words"]) is list, "Missing raw words")
        for wi, word in enumerate(segment["words"]):
            require(set(word) == {"word", "start", "end", "probability"} and type(word["word"]) is str, "Raw word schema")
            left, right, probability = (old.finite(word[key], key) for key in ("start", "end", "probability"))
            require(right >= left and 0 <= probability <= 1, "Invalid raw word")
            if left < -.02 or right > duration + .02:
                flags.add("word_timestamp_outside_context")
            if left < previous_end - .02:
                flags.add("word_timestamps_overlap_or_out_of_order")
            if left == right:
                flags.add("zero_duration_word")
            previous_end = right
            midpoint = (left + right) / 2
            row = {"index": len(words), "segment_index": si, "segment_word_index": wi,
                   **word, "midpoint": midpoint, "selected_by_midpoint": start <= midpoint < end,
                   "crosses_label_start": left < start < right, "crosses_label_end": left < end < right,
                   "near_label_start": min(abs(left-start), abs(right-start)) <= .12,
                   "near_label_end": min(abs(left-end), abs(right-end)) <= .12}
            for name in CUE_NAMES:
                if row[name]:
                    cues[name].append(row["index"])
            words.append(row)
    selected = [word["index"] for word in words if word["selected_by_midpoint"]]
    if not words:
        flags.add("no_word_timestamps")
    if not selected:
        flags.add("no_midpoint_selected_words")
    return {"words": words, "selected_word_indices": selected,
            "selected_text": "".join(words[i]["word"] for i in selected),
            "all_word_text": "".join(word["word"] for word in words),
            "boundary_cues": cues, "word_timing_flags": sorted(flags)}


def check_context(row, clip, baseline, plan_ref, decoder):
    require(set(row) == CONTEXT_FIELDS and row["schema"] == "step07-context-asr-record-v1", "Context record schema")
    same(row["context_plan"], plan_ref, "Context plan")
    for key in ("id", "split", "speaker_id", "source_stem", "reference", "start_time", "end_time", "context_audio"):
        same(row[key], clip[key], "Context immutable " + key)
    same(row["original_audio"], clip["audio"], "Original audio")
    same(row["original_interval_in_context"], interval(clip), "Relative interval")
    same(row["baseline"], {key: baseline[key] for key in ("status", "asr_text", "metrics", "boundary_flags", "confidence_flags")}, "Baseline row")
    require(all(row[key] is False for key in ("human_listening_confirmed", "automatic_approval", "label_modified", "training_ready")), "Context approval claim")
    require(old.finite(row["inference_seconds"], "runtime") >= 0 and row["status"] in ("ok", "asr_failed"), "Runtime/status")
    if row["status"] == "ok":
        require(row["error"] is None, "Successful error")
        info = copy.deepcopy(row["transcription_info"])
        require(info["transcription_options"]["word_timestamps"] is True, "Word timestamp option")
        # All other recognizer option metadata must match the frozen Step6 contract.
        info["transcription_options"]["word_timestamps"] = False
        old.check_info(info, clip["context_audio"]["frames"] / 16000, decoder)
        for index, segment in enumerate(row["raw_segments"], 1):
            require(set(segment) == old.SEGMENT_FIELDS and type(segment["id"]) is int
                    and segment["id"] == index and type(segment["seek"]) is int and segment["seek"] >= 0, "Raw segment identity")
            require(segment["temperature"] == 0.0 and type(segment["words"]) is list, "Raw segment options")
            for key in ("start", "end", "avg_logprob", "compression_ratio", "no_speech_prob"):
                old.finite(segment[key], key)
            require(0 <= segment["no_speech_prob"] <= 1 and segment["compression_ratio"] >= 0, "Raw decoder statistic")
            same(decoder.decode(segment["tokens"]), segment["text"], "Raw token text")
        same(row["full_context_text"], "".join(s["text"] for s in row["raw_segments"]), "Full context concatenation")
        boundary, confidence = old.segment_flags(row["raw_segments"], clip["context_audio"]["frames"] / 16000)
        if not old.normalize(row["full_context_text"]):
            confidence = sorted(set(confidence) | {"empty_normalized_transcription"})
    else:
        require(row["full_context_text"] == "" and row["raw_segments"] == [] and row["transcription_info"] is None
                and type(row["error"]) is str and row["error"], "Context failure evidence")
        boundary, confidence = [], []
    same(row["context_segment_boundary_flags"], boundary, "Context boundary flags")
    same(row["decoder_confidence_flags"], confidence, "Context confidence flags")
    for key, value in interpret(row["raw_segments"], interval(clip)).items():
        same(row[key], value, "Independent word derivation " + key)
    same(row["full_context_comparison"], old.comparison(clip["reference"], row["full_context_text"]), "Full context metrics")
    same(row["selected_comparison"], old.comparison(clip["reference"], row["selected_text"]), "Selected context metrics")


def context_summary(rows):
    comparisons = {}
    for key in ("full_context_comparison", "selected_comparison"):
        values = [row[key]["metrics"] for row in rows]
        e, r, ne, nr = (sum(value[name] for value in values) for name in
                       ("character_edits", "reference_characters", "without_spaces_character_edits", "without_spaces_reference_characters"))
        comparisons[key] = {"exact": sum(value["exact"] for value in values),
            "without_spaces_exact": sum(value["without_spaces_exact"] for value in values),
            "character_edits": e, "reference_characters": r, "pooled_cer_percent": 100. * e / r,
            "without_spaces_character_edits": ne, "without_spaces_reference_characters": nr,
            "without_spaces_pooled_cer_percent": 100. * ne / nr}
    return {"records": len(rows), "status_counts": dict(Counter(row["status"] for row in rows)),
        "comparison_metrics": comparisons,
        "cue_sentence_counts": {key: sum(bool(row["boundary_cues"][key]) for row in rows) for key in CUE_NAMES},
        "word_timing_flag_counts": dict(Counter(flag for row in rows for flag in row["word_timing_flags"])),
        "context_segment_boundary_flag_counts": dict(Counter(flag for row in rows for flag in row["context_segment_boundary_flags"])),
        "decoder_confidence_flag_counts": dict(Counter(flag for row in rows for flag in row["decoder_confidence_flags"])),
        "selected_text_equal_to_baseline_normalized": sum(old.normalize(row["selected_text"]) == old.normalize(row["baseline"]["asr_text"]) for row in rows),
        "human_listening_confirmed_count": 0, "automatic_approval_count": 0, "labels_modified": False, "training_executed": False}


def check_targets(document, plan, source, media, overrides):
    require(document["schema"] == "lipnet-transcript-step07-targets-v1" and document["policy_version"] == "ko-alternative-target-v1", "Target artifact schema")
    ids = [row["id"] for row in source["records"]]
    require(len(ids) == len(set(ids)) == 120, "Target120 denominator")
    same(document["ids"], ids, "Target ID order")
    same([row["id"] for row in document["records"]], ids, "Target record order")
    same(dict(Counter(row["split"] for row in source["records"])), {"train": 96, "val": 24}, "Original split")
    train = {row["speaker_id"] for row in source["records"] if row["split"] == "train"}
    development = {row["speaker_id"] for row in source["records"] if row["split"] == "val"}
    require(len(train) == 12 and len(development) == 3 and not train & development, "Speaker leakage")
    require(len(overrides) == 1 and overrides[0]["id"] == "lip_J_1_F_05_C057_A_005_005", "User choice scope")
    mapping = {row["id"]: row for row in overrides}
    for row, selected, clip in zip(document["records"], source["records"], media["clips"]):
        same(selected["expected_frames"], clip["timing"]["student_frames"], "Actual student frame count")
        check_target_record(row, selected, mapping)
        if row["selection_evidence"]:
            evidence = row["selection_evidence"]["evidence"]
            same(evidence["source_plan"], plan["source_plan"], "User evidence source")
            same(evidence["media_manifest"], plan["media_manifest"], "User evidence media")
            same(evidence["raw_source_text"], selected["source_text"], "User evidence raw text")
            require(evidence["audio"] in (clip["audio"], clip["context_audio"]), "User listened to another audio")
            verify(evidence["audio"])
    counts = {"records": 120, "train": 96, "val": 24, "speakers": 15,
              "ordinary_tokens_identical": sum(row["ordinary_tokens_identical"] is True for row in document["records"]),
              "canonical_targets_available": sum(row["canonical_target"] is not None for row in document["records"]),
              "target_status": dict(Counter(row["target_status"] for row in document["records"]))}
    same(document["counts"], counts, "Target summary")
    require(counts["ordinary_tokens_identical"] == 119 and counts["canonical_targets_available"] == 120
            and counts["target_status"] == {"unchanged_ordinary_target": 119, "resolved_alternative_choice": 1}, "Expected target transition")
    changes = [{key: row[key] for key in ("id", "previous_target", "canonical_target", "canonical_tokens", "ctc_min_frames", "student_frames", "selection_evidence")}
               for row in document["records"] if row["canonical_tokens"] != row["previous_target"]["tokens"]]
    same(document["target_changes"], changes, "Exact change inventory")
    require(len(changes) == 1 and changes[0]["id"] == overrides[0]["id"], "Unexpected changed label")
    require(document["source_labels_modified"] is False and document["asr_used_to_choose_targets"] is False
            and document["training_ready"] is False and document["full_transcript_human_approvals_added"] == 0, "Target approval scope")
    expected_meta = {"path": str(ROOT / "backend/models/tokenizer.py"), "sha256": TOKENIZER_SHA, **tokenizer_metadata()}
    same(document["tokenizer"], expected_meta, "Frozen target tokenizer")
    verify(expected_meta)
    return counts, changes


def check_context_audio(spec):
    path = verify(spec)
    require(path.is_relative_to(STEP6 / "results/media") and path.name == "context_audio.wav", "Wrong context file")
    with wave.open(str(path), "rb") as stream:
        require((stream.getframerate(), stream.getnchannels(), stream.getsampwidth(), stream.getcomptype()) ==
                (16000, 1, 2, "NONE"), "Context PCM header")
        require(stream.getnframes() == spec["frames"] > 0, "Context sample count")
        require(len(stream.readframes(spec["frames"])) == spec["frames"] * 2, "Truncated context PCM")
    require(spec["sample_rate"] == 16000 and spec["channels"] == 1 and spec["sample_width_bytes"] == 2
            and spec["clock"] == "original_label_wav" and spec["source_role"] == "audio", "Context metadata")
    close(spec["duration_seconds"], spec["frames"] / 16000, "Context duration")


def audit(expected_label_plan, expected_context_plan, output):
    started = time.perf_counter()
    require(not output.exists(), "Preserve prior Stage7 audit")
    refs = {name: bind(BASE / name) for name in ("label_plan.json", "context_plan.json", "results/label_targets.json", "results/context_asr_results.json", "label_overrides.json", "user_alternative_choice.json")}
    require(refs["label_plan.json"]["sha256"] == expected_label_plan and refs["context_plan.json"]["sha256"] == expected_context_plan, "Expected plan SHA")
    labels, context = read(refs["label_plan.json"]["path"]), read(refs["context_plan.json"]["path"])
    for ref in context["code"] + context["implementation"] + list(context["inputs"].values()) + context["context_audio"]:
        verify(ref)
    for ref in labels["code"] + labels["choice_evidence"] + [labels["source_plan"], labels["media_manifest"], labels["overrides"]]:
        verify(ref)
    require(labels["schema"] == "lipnet-transcript-step07-label-plan-v1" and labels["frozen_before_target_artifact_generation"] is True, "Label plan schema")
    require(context["schema"] == "step07-context-asr-plan-v1" and context["status"] == "frozen_before_transcription", "Context plan schema")
    expected = dict(old.OPTIONS, word_timestamps=True)
    same(context["transcribe_options"], expected, "Only word_timestamps changed")
    require(context["same_model_as_step6"] is True and context["independent_asr_cross_validation"] is False
            and context["reference_supplied_to_recognizer"] is False and context["automatic_approval"] is False
            and context["human_listening_confirmed_count"] == 0 and context["training_executed"] is False
            and context["labels_modified"] is False, "Context plan scope")
    same([context["word_timing_rules"][key] for key in ("context_padding_seconds", "near_boundary_seconds", "timestamp_tolerance_seconds")], [.5, .12, .02], "Frozen numeric rules")
    prior_ref = context["inputs"]["baseline_audit"]
    require(prior_ref["sha256"] == "33465ef8768e1416eb5998266ccdb6f133a7d00df2a1231348b5a3064ea5fc33", "Frozen Step6 audit")
    prior = read(prior_ref["path"])
    require(prior["status"] == "passed", "Baseline audit")
    for context_key, prior_key in (("media_manifest", "results/media_manifest.json"), ("source_plan", "source_plan.json"), ("baseline_asr", "results/asr_results.json"), ("baseline_plan", "asr_plan.json")):
        same(context["inputs"][context_key], prior["inputs"][prior_key], "Baseline linkage " + context_key)
    same(context["inputs"]["setup"], prior["asr_setup"], "Setup linkage")
    for key in ("source_plan", "media_manifest"):
        same(labels[key], context["inputs"][key], "Label/context " + key)
    same(labels["overrides"], refs["label_overrides.json"], "Override binding")
    same(labels["choice_evidence"], [refs["user_alternative_choice.json"]], "Choice evidence inventory")
    same(refs["user_alternative_choice.json"]["sha256"], "100da7dc7f3012bff2fc852e3dc1753d883042a22aa125cd44b1eb366539b931", "User evidence frozen SHA")
    source = read(context["inputs"]["source_plan"]["path"])
    media = read(context["inputs"]["media_manifest"]["path"])
    baseline = read(context["inputs"]["baseline_asr"]["path"])
    baseline_plan = read(context["inputs"]["baseline_plan"]["path"])
    setup = read(context["inputs"]["setup"]["path"])
    same(context["model"], baseline_plan["model"], "Same model")
    same(context["implementation"], baseline_plan["implementation"], "Same implementation")
    old.verify_model(baseline_plan, setup)
    decoder = old.ByteTokens(read(Path(setup["model_directory"]) / "tokenizer.json"))
    ids = [row["id"] for row in source["records"]]
    same(context["ids"], ids, "Context plan order")
    same(labels["ids"], ids, "Label plan order")
    same(context["split_counts"], {"train": 96, "val": 24}, "Context plan split")
    require(context["record_count"] == 120, "Context count")
    same(context["intervals"], [{"id": row["id"], **interval(row)} for row in media["clips"]], "Frozen interval index")
    same(context["context_audio"], [{"id": row["id"], "path": row["context_audio"]["path"], "sha256": row["context_audio"]["sha256"]} for row in media["clips"]], "Frozen context index")
    targets = read(refs["results/label_targets.json"]["path"])
    same(targets["label_plan"], refs["label_plan.json"], "Target plan linkage")
    for key in ("source_plan", "media_manifest", "overrides"):
        same(targets[key], labels[key], "Target provenance " + key)
    counts, changes = check_targets(targets, labels, source, media, read(refs["label_overrides.json"]["path"]))
    result = read(refs["results/context_asr_results.json"]["path"])
    require(result["schema"] == "step07-context-asr-results-v1" and result["status"] == "completed" and result["record_count"] == 120, "Context result schema")
    same(result["context_plan"], refs["context_plan.json"], "Context result plan")
    same(result["inputs"], context["inputs"], "Context result inputs")
    same(result["ids"], ids, "Context result ids")
    same([row["id"] for row in result["records"]], ids, "Context result row order")
    require(result["same_model_as_step6"] is True and result["independent_asr_cross_validation"] is False
            and all(result[key] is False for key in ("audio_uploaded", "automatic_approval", "labels_modified", "training_executed"))
            and result["human_listening_confirmed_count"] == 0, "Context result scope")
    same(result["limits"], context["limits"], "Context limits")
    require(old.finite(result["elapsed_seconds_this_run"], "elapsed") >= 0, "Context elapsed")
    directory = BASE / "results/context_records"
    require({path.name for path in directory.glob("*.json")} == {identity + ".json" for identity in ids}, "Context orphan/missing files")
    record_refs = [bind(directory / (identity + ".json")) for identity in ids]
    same(result["record_files"], record_refs, "Context record files index")
    same(result["progress"], bind(BASE / "results/context_progress.json"), "Context progress SHA")
    progress = read(result["progress"]["path"])
    same(progress, {"schema": "step07-context-asr-progress-v1", "context_plan": refs["context_plan.json"],
                    "records": [{"id": identity, **ref} for identity, ref in zip(ids, record_refs)]}, "Context progress contents")
    for row, clip, prior_row, ref in zip(result["records"], media["clips"], baseline["records"], record_refs):
        same(read(ref["path"]), row, "Context per-record and aggregate equality")
        check_context_audio(clip["context_audio"])
        check_context(row, clip, prior_row, refs["context_plan.json"], decoder)
    same(result["summary"], context_summary(result["records"]), "Independent context totals")
    research_path = BASE / "research/official_transcription_policy.json"
    research = read(research_path)
    require(research["status"] == "dataset_specific_convention_not_verified", "Do not infer official alternative semantics")
    for ref in list(refs.values()) + record_refs + [prior_ref, result["progress"]]:
        verify(ref)
    artifact = {"schema": "lipnet-transcript-step07-independent-audit-v1", "status": "passed",
        "created_utc": datetime.now(timezone.utc).isoformat(), "code": bind(__file__),
        "tests": bind(BASE / "test_audit_stage07.py"), "frozen_independent_helpers": bind(HELPER),
        "inputs": refs, "baseline_independent_audit": prior_ref, "official_policy_research": bind(research_path),
        "target_counts": counts, "target_change": {key: changes[0][key] for key in ("id", "canonical_target", "ctc_min_frames", "student_frames")},
        "previous_changed_target_ctc_min": changes[0]["previous_target"]["ctc_min_frames"],
        "context_summary": result["summary"],
        "context_raw_segments_checked": sum(len(row["raw_segments"]) for row in result["records"]),
        "context_words_checked": sum(len(row["words"]) for row in result["records"]),
        "context_audio_samples_checked": sum(row["context_audio"]["frames"] for row in media["clips"]),
        "human_alternative_choices": 1, "full_transcript_human_approvals_added": 0,
        "training_ready": False, "model_executed": False, "training_executed": False, "original_labels_modified": False,
        "elapsed_seconds": time.perf_counter() - started,
        "runtime": {"executable": sys.executable, "python": sys.version, "standard_library_only": True},
        "limits": ["Saved evidence consistency is checked without rerunning recognition or listening to speech",
                   "One user-confirmed alternative is not approval of a full transcript or its temporal boundaries",
                   "Official dataset538 left/right alternative semantics and repetition/partial-speech policies remain unverified",
                   "Same-model context sensitivity is not independent ASR cross-validation or a lipreading performance measure",
                   "Word timestamps are estimates and boundary cues do not prove clipping; token timestamps may differ after word alignment",
                   "No training or corrected source-label overwrite was performed"]}
    with output.open("x", encoding="utf8", newline="\n") as stream:
        json.dump(artifact, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return artifact


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-plan-sha256", required=True)
    parser.add_argument("--context-plan-sha256", required=True)
    parser.add_argument("--output", type=Path, default=BASE / "results/independent_stage07_audit.json")
    args = parser.parse_args()
    result = audit(args.label_plan_sha256, args.context_plan_sha256, args.output.resolve())
    print(json.dumps({"status": result["status"], "target_counts": result["target_counts"],
                      "context_summary": result["context_summary"], "output": bind(args.output)}, ensure_ascii=False))
