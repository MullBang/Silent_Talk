"""Same-model context/word-timing diagnosis, never a transcript approval gate.

All audio-only recognition finishes before reference comparison or midpoint word
selection. The original Step6 artifacts and labels are immutable inputs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import tempfile
import time

import numpy as np

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
STEP6 = BASE.parent / "new_data_step06_20261007"
_spec = importlib.util.spec_from_file_location("step6_asr_readonly", STEP6 / "asr_audit.py")
audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit)
PLAN = BASE / "context_plan.json"
RESULT = BASE / "results/context_asr_results.json"
PROGRESS = BASE / "results/context_progress.json"
RECORDS = BASE / "results/context_records"
OPTIONS = copy.deepcopy(audit.OPTIONS)
OPTIONS["word_timestamps"] = True
RULES = {
    "context_padding_seconds": 0.5,
    "near_boundary_seconds": 0.12,
    "timestamp_tolerance_seconds": 0.02,
    "word_selection": "start <= (word.start+word.end)/2 < end; start/end are original label times minus context WAV start",
    "crossing": "word.start < boundary < word.end, strictly; no tolerance or rounding changes this diagnostic cue",
    "near_boundary": "min(abs(word.start-boundary),abs(word.end-boundary)) <= 0.12 seconds, including words outside original interval",
    "selected_text": "Concatenate selected word strings unchanged in recognizer segment/word order, then compare using frozen Step6 NFC/punctuation/whitespace normalization",
    "time_basis": "All segment and word times are relative to the context WAV; original interval is [0.5,0.5+original_label_duration) seconds",
    "interpretation": "Model-estimated word times are approximate review cues, not forced-alignment truth or proof of complete/clipped speech; no new approval gate",
}
require, read, bind, verify, save_new = audit.require, audit.read, audit.bind, audit.verify, audit.save_new


def fixed_inputs():
    return {"media_manifest": STEP6 / "results/media_manifest.json", "baseline_asr": STEP6 / "results/asr_results.json",
            "baseline_plan": STEP6 / "asr_plan.json", "baseline_audit": STEP6 / "results/independent_asr_audit.json",
            "source_plan": STEP6 / "source_plan.json", "setup": STEP6 / "asr_setup.json"}


def load_inputs():
    paths = fixed_inputs()
    docs = {name: read(path) for name, path in paths.items()}
    media, old = docs["media_manifest"], docs["baseline_asr"]
    clips = audit.validate_media(media)
    require(media["status"] == "completed_all_planned_sources" and old["status"] == "completed", "Incomplete prior evidence")
    previous_audit = docs["baseline_audit"]
    require(previous_audit["status"] == "passed", "Baseline independent audit did not pass")
    for name, key in (("baseline_asr", "results/asr_results.json"), ("media_manifest", "results/media_manifest.json"),
                      ("source_plan", "source_plan.json"), ("baseline_plan", "asr_plan.json")):
        require(previous_audit["inputs"][key] == bind(paths[name]), "Baseline audit input binding differs: " + name)
    require(previous_audit["asr_setup"] == bind(paths["setup"]), "Baseline audit setup binding differs")
    require(media["source_plan"] == bind(paths["source_plan"]), "Changed source selection")
    require(old["media_manifest"] == bind(paths["media_manifest"]) and old["asr_plan"] == bind(paths["baseline_plan"]),
            "Baseline evidence bindings differ")
    require(old["ids"] == [clip["id"] for clip in clips] == [row["id"] for row in old["records"]], "Baseline cohort order differs")
    for clip, row in zip(clips, old["records"]):
        audit.validate_saved_record(row, clip, old["asr_plan"])
    for ref in docs["baseline_plan"]["code"] + docs["baseline_plan"]["implementation"]:
        verify(ref)
    audit.verify_setup(docs["setup"])
    return docs, clips, old["records"]


def interval_for(clip):
    context, original = clip["context_audio"], clip["audio"]
    require(clip["timing_status"] == "verified" and clip["artifacts_ready"] is True,
            "Only completed verified Step6 artifacts are inputs")
    require(context is not None and original is not None and context["clock"] == original["clock"] == "original_label_wav"
            and context["source_sha256"] == original["source_sha256"], "Context/original clock or source mismatch")
    start = clip["start_time"] - context["start_seconds"]
    end = clip["end_time"] - context["start_seconds"]
    duration = context["frames"] / 16000
    tolerance = 2 / 16000
    require(abs(start - RULES["context_padding_seconds"]) < 1e-8
            and abs(context["end_seconds"] - clip["end_time"] - RULES["context_padding_seconds"]) < 1e-8,
            "Expected pre-existing half-second context on each side")
    require(0 < start < end < duration and abs(end - start - original["frames"] / 16000) <= tolerance
            and abs(duration - (context["end_seconds"] - context["start_seconds"])) <= tolerance,
            "Context/label sample duration mismatch")
    return {"start_seconds": start, "end_seconds": end, "context_duration_seconds": duration,
            "original_duration_seconds": clip["end_time"] - clip["start_time"]}


def validate_info(info, sample_count):
    require(isinstance(info, dict), "Missing context transcription metadata")
    require(info["language"] == "ko", "Changed recognizer language")
    actual = info["transcription_options"]
    for name in ("beam_size", "condition_on_previous_text", "initial_prompt", "prefix", "hotwords", "word_timestamps"):
        require(actual[name] == OPTIONS[name], "Changed context recognizer option: " + name)
    require(actual["temperatures"] == [0.0], "Unexpected temperature fallback")
    require(abs(info["duration"] - sample_count / 16000) <= 1 / 16000 + 1e-6
            and abs(info["duration_after_vad"] - info["duration"]) <= 1 / 16000 + 1e-6,
            "Context duration or VAD changed")


def infer_audio_only(model, samples):
    """Neither the function nor its model call accepts a reference transcript."""
    require(isinstance(samples, np.ndarray) and samples.dtype == np.float32 and samples.ndim == 1
            and len(samples) > 0 and np.isfinite(samples).all(), "Invalid context audio")
    segments, info = model.transcribe(samples, **copy.deepcopy(OPTIONS))
    segments = [audit.json_value(segment) for segment in segments]
    info = audit.json_value(info)
    validate_info(info, len(samples))
    return "".join(segment["text"] for segment in segments), segments, info


def interpret_words(segments, interval):
    """Fixed audio-time-only selection after inference; never inspect labels."""
    start, end, duration = (interval[key] for key in ("start_seconds", "end_seconds", "context_duration_seconds"))
    near, tolerance = RULES["near_boundary_seconds"], RULES["timestamp_tolerance_seconds"]
    words, timing_flags = [], set()
    cue_indices = {key: [] for key in ("crosses_label_start", "crosses_label_end", "near_label_start", "near_label_end")}
    previous_end = -math.inf
    for segment_index, segment in enumerate(segments):
        require(isinstance(segment.get("words"), list), "Word timestamps missing from segment")
        for segment_word_index, word in enumerate(segment["words"]):
            require(isinstance(word.get("word"), str), "Invalid word text")
            for key in ("start", "end", "probability"):
                require(type(word.get(key)) in (int, float) and math.isfinite(word[key]), "Invalid word timestamp/probability")
            left, right = word["start"], word["end"]
            require(right >= left and 0 <= word["probability"] <= 1, "Inverted word or invalid probability")
            if left < -tolerance or right > duration + tolerance:
                timing_flags.add("word_timestamp_outside_context")
            if left < previous_end - tolerance:
                timing_flags.add("word_timestamps_overlap_or_out_of_order")
            if left == right:
                timing_flags.add("zero_duration_word")
            previous_end = right
            midpoint = (left + right) / 2
            row = {"index": len(words), "segment_index": segment_index, "segment_word_index": segment_word_index,
                   "word": word["word"], "start": left, "end": right, "probability": word["probability"],
                   "midpoint": midpoint, "selected_by_midpoint": start <= midpoint < end,
                   "crosses_label_start": left < start < right, "crosses_label_end": left < end < right,
                   "near_label_start": min(abs(left - start), abs(right - start)) <= near,
                   "near_label_end": min(abs(left - end), abs(right - end)) <= near}
            for key in cue_indices:
                if row[key]:
                    cue_indices[key].append(row["index"])
            words.append(row)
    if not words:
        timing_flags.add("no_word_timestamps")
    selected = [word["index"] for word in words if word["selected_by_midpoint"]]
    if not selected:
        timing_flags.add("no_midpoint_selected_words")
    return {"words": words, "selected_word_indices": selected,
            "selected_text": "".join(words[index]["word"] for index in selected),
            "all_word_text": "".join(word["word"] for word in words),
            "boundary_cues": cue_indices, "word_timing_flags": sorted(timing_flags)}


def make_record(model, clip, baseline, plan_ref, audio_loader=audit.load_audio):
    interval = interval_for(clip)
    record = {key: clip[key] for key in ("id", "split", "speaker_id", "source_stem", "reference", "start_time", "end_time")}
    record.update(schema="step07-context-asr-record-v1", context_plan=plan_ref, context_audio=clip["context_audio"],
                  original_audio=clip["audio"], original_interval_in_context=interval,
                  baseline={key: baseline[key] for key in ("status", "asr_text", "metrics", "boundary_flags", "confidence_flags")},
                  status="asr_failed", full_context_text="", raw_segments=[], transcription_info=None,
                  error=None, context_segment_boundary_flags=[], decoder_confidence_flags=[], inference_seconds=0.,
                  human_listening_confirmed=False, automatic_approval=False, label_modified=False, training_ready=False)
    record.update(interpret_words([], interval))
    started = time.perf_counter()
    # Integrity/header failures stop this run; retain previous immutable records.
    samples = audio_loader(clip["context_audio"])
    try:
        text, segments, info = infer_audio_only(model, samples)
        interpreted = interpret_words(segments, interval)
        boundaries, confidence = audit.assess_segments(segments, len(samples) / 16000)
        if not audit.normalize_for_comparison(text):
            confidence = sorted(set(confidence + ["empty_normalized_transcription"]))
        record.update(status="ok", full_context_text=text, raw_segments=segments, transcription_info=info,
                      context_segment_boundary_flags=boundaries, decoder_confidence_flags=confidence, **interpreted)
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    record["inference_seconds"] = time.perf_counter() - started
    # These are the first calls that compare recognized output with the reference.
    record["full_context_comparison"] = audit.compare_text(clip["reference"], record["full_context_text"])
    record["selected_comparison"] = audit.compare_text(clip["reference"], record["selected_text"])
    return record


def validate_record(record, clip, baseline, plan_ref):
    fields = {"id", "split", "speaker_id", "source_stem", "reference", "start_time", "end_time", "schema", "context_plan",
              "context_audio", "original_audio", "original_interval_in_context", "baseline", "status", "full_context_text",
              "raw_segments", "transcription_info", "error", "context_segment_boundary_flags", "decoder_confidence_flags",
              "inference_seconds", "human_listening_confirmed", "automatic_approval", "label_modified", "training_ready",
              "words", "selected_word_indices", "selected_text", "all_word_text", "boundary_cues", "word_timing_flags",
              "full_context_comparison", "selected_comparison"}
    require(set(record) == fields and record["schema"] == "step07-context-asr-record-v1", "Record fields differ")
    require(record["context_plan"] == plan_ref, "Wrong context plan")
    for name in ("id", "split", "speaker_id", "source_stem", "reference", "start_time", "end_time", "context_audio"):
        require(record[name] == clip[name], "Changed original input: " + name)
    require(record["original_audio"] == clip["audio"] and record["original_interval_in_context"] == interval_for(clip), "Changed audio clock")
    require(record["baseline"] == {key: baseline[key] for key in record["baseline"]}
            and set(record["baseline"]) == {"status", "asr_text", "metrics", "boundary_flags", "confidence_flags"}, "Changed baseline")
    require(all(record[key] is False for key in ("human_listening_confirmed", "automatic_approval", "label_modified", "training_ready")),
            "Unwarranted approval or training claim")
    require(type(record["inference_seconds"]) in (int, float) and math.isfinite(record["inference_seconds"])
            and record["inference_seconds"] >= 0, "Invalid inference runtime")
    require(record["status"] in ("ok", "asr_failed"), "Unknown record status")
    if record["status"] == "ok":
        require(record["error"] is None and isinstance(record["transcription_info"], dict), "Malformed successful record")
        validate_info(record["transcription_info"], clip["context_audio"]["frames"])
        require(record["full_context_text"] == "".join(segment["text"] for segment in record["raw_segments"]), "Changed context text")
        boundary, confidence = audit.assess_segments(record["raw_segments"], clip["context_audio"]["frames"] / 16000)
        if not audit.normalize_for_comparison(record["full_context_text"]):
            confidence = sorted(set(confidence + ["empty_normalized_transcription"]))
        require(record["context_segment_boundary_flags"] == boundary and record["decoder_confidence_flags"] == confidence,
                "Changed segment cues")
    else:
        require(record["full_context_text"] == "" and record["raw_segments"] == [] and record["transcription_info"] is None
                and isinstance(record["error"], str) and record["error"]
                and record["context_segment_boundary_flags"] == record["decoder_confidence_flags"] == [], "Malformed failed record")
    expected = interpret_words(record["raw_segments"], record["original_interval_in_context"])
    require(all(record[key] == value for key, value in expected.items()), "Changed word selection or timing cues")
    require(record["full_context_comparison"] == audit.compare_text(clip["reference"], record["full_context_text"])
            and record["selected_comparison"] == audit.compare_text(clip["reference"], record["selected_text"]), "Changed comparison")


def save_progress(value, path=PROGRESS):
    path = Path(path)
    require(path.resolve().is_relative_to(BASE), "Progress must stay within Stage7")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=".context_progress.", suffix=".tmp", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write((json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)


def load_progress(clips, baselines, plan_ref, path=PROGRESS, directory=RECORDS):
    path, directory = Path(path), Path(directory)
    progress = read(path) if path.exists() else {"schema": "step07-context-asr-progress-v1", "context_plan": plan_ref, "records": []}
    require(set(progress) == {"schema", "context_plan", "records"} and progress["schema"] == "step07-context-asr-progress-v1"
            and progress["context_plan"] == plan_ref, "Wrong progress plan/schema")
    refs = progress["records"]
    require(isinstance(refs, list) and len(refs) <= len(clips) and [ref["id"] for ref in refs] ==
            [clip["id"] for clip in clips[:len(refs)]], "Resume records must be fixed ordered prefix")
    require(not directory.exists() or {p.name for p in directory.glob("*.json")} == {ref["id"] + ".json" for ref in refs},
            "Orphan/missing resumed record; preserve and investigate")
    records = []
    for ref, clip, baseline in zip(refs, clips, baselines):
        require(set(ref) == {"id", "path", "sha256"} and Path(ref["path"]).resolve() == (directory / (clip["id"] + ".json")).resolve(),
                "Wrong resumed path")
        verify(ref)
        record = read(ref["path"])
        validate_record(record, clip, baseline, plan_ref)
        records.append(record)
    return progress, records


def freeze():
    require(not PLAN.exists(), "Context plan already frozen")
    docs, clips, _ = load_inputs()
    intervals = []
    for clip in clips:
        intervals.append({"id": clip["id"], **interval_for(clip)})
        audit.load_audio(clip["context_audio"])
    plan = {"schema": "step07-context-asr-plan-v1", "status": "frozen_before_transcription",
            "created_utc": datetime.now(timezone.utc).isoformat(), "ids": [clip["id"] for clip in clips],
            "record_count": 120, "split_counts": {"train": 96, "val": 24},
            "inputs": {name: bind(path) for name, path in fixed_inputs().items()},
            "code": [bind(Path(__file__)), bind(BASE / "test_context_asr.py"), bind(STEP6 / "asr_audit.py")],
            "implementation": audit.implementation_bindings(),
            "context_audio": [{"id": clip["id"], **bind(clip["context_audio"]["path"])} for clip in clips],
            "intervals": intervals, "transcribe_options": OPTIONS, "word_timing_rules": RULES,
            "model": copy.deepcopy(docs["baseline_plan"]["model"]),
            "same_model_as_step6": True, "independent_asr_cross_validation": False,
            "reference_supplied_to_recognizer": False, "automatic_approval": False,
            "human_listening_confirmed_count": 0, "training_executed": False, "labels_modified": False,
            "failure_policy": "Preserve120 IDs and split roles; inference failures use empty prediction and are recorded; input/hash/header evidence failure aborts; no silent omission or retry setting change",
            "limits": ["Same-model context sensitivity diagnosis, not independent recognizer corroboration",
                       "Words with midpoint inside original interval may overlap its boundaries; original audio and context must be listened to for final interpretation",
                       "Word timings and decoder probabilities are model estimates, not verified acoustic boundaries",
                       "Neither boundary cues nor improved text agreement approve or relabel a sentence",
                       "Full-context text may contain neighboring speech; selected text is a deterministic diagnostic subset only",
                       "Raw source annotation alternatives are preserved; CER is not a corrected-label or lipreading performance metric"]}
    require(plan["model"]["compute_type"] == "int8_float16" and plan["model"]["clips_per_call"] == 1, "Changed model policy")
    save_new(PLAN, plan)
    return plan


def summarize(records):
    require(len(records) == 120 and len({record["id"] for record in records}) == 120, "Incomplete diagnostic cohort")
    comparisons = {}
    for name in ("full_context_comparison", "selected_comparison"):
        metrics = [record[name]["metrics"] for record in records]
        edits, chars = sum(row["character_edits"] for row in metrics), sum(row["reference_characters"] for row in metrics)
        no_space_edits = sum(row["without_spaces_character_edits"] for row in metrics)
        no_space_chars = sum(row["without_spaces_reference_characters"] for row in metrics)
        comparisons[name] = {"exact": sum(row["exact"] for row in metrics), "without_spaces_exact": sum(row["without_spaces_exact"] for row in metrics),
                             "character_edits": edits, "reference_characters": chars, "pooled_cer_percent": 100. * edits / chars,
                             "without_spaces_character_edits": no_space_edits, "without_spaces_reference_characters": no_space_chars,
                             "without_spaces_pooled_cer_percent": 100. * no_space_edits / no_space_chars}
    return {"records": 120, "status_counts": dict(Counter(record["status"] for record in records)),
            "comparison_metrics": comparisons,
            "cue_sentence_counts": {key: sum(bool(record["boundary_cues"][key]) for record in records)
                                    for key in ("crosses_label_start", "crosses_label_end", "near_label_start", "near_label_end")},
            "word_timing_flag_counts": dict(Counter(flag for record in records for flag in record["word_timing_flags"])),
            "context_segment_boundary_flag_counts": dict(Counter(flag for record in records for flag in record["context_segment_boundary_flags"])),
            "decoder_confidence_flag_counts": dict(Counter(flag for record in records for flag in record["decoder_confidence_flags"])),
            "selected_text_equal_to_baseline_normalized": sum(audit.normalize_for_comparison(record["selected_text"]) ==
                                                            audit.normalize_for_comparison(record["baseline"]["asr_text"]) for record in records),
            "human_listening_confirmed_count": 0, "automatic_approval_count": 0, "labels_modified": False, "training_executed": False}


def run(expected_plan_sha256):
    require(not RESULT.exists(), "Context diagnosis completed; preserve results")
    plan_ref = bind(PLAN)
    require(isinstance(expected_plan_sha256, str) and len(expected_plan_sha256) == 64 and
            plan_ref["sha256"] == expected_plan_sha256.lower(), "Explicit context plan SHA mismatch")
    plan = read(PLAN)
    require(plan["status"] == "frozen_before_transcription" and plan["transcribe_options"] == OPTIONS
            and plan["word_timing_rules"] == RULES, "Changed diagnosis plan/options")
    refs = plan["code"] + plan["implementation"] + list(plan["inputs"].values()) + plan["context_audio"] + [plan_ref]
    for ref in refs:
        verify(ref)
    docs, clips, baselines = load_inputs()
    require(plan["ids"] == [clip["id"] for clip in clips] and plan["intervals"] ==
            [{"id": clip["id"], **interval_for(clip)} for clip in clips], "Changed context cohort/time mapping")
    progress, records = load_progress(clips, baselines, plan_ref)
    if not PROGRESS.exists():
        save_progress(progress)
    setup = docs["setup"]
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HOME"] = str(ROOT / "models/asr_audit/hf_home")
    dll_handle = os.add_dll_directory(setup["cuda_dll_directory"])
    os.environ["PATH"] = setup["cuda_dll_directory"] + os.pathsep + os.environ["PATH"]
    from faster_whisper import WhisperModel
    model = WhisperModel(plan["model"]["directory"], device="cuda", device_index=0, compute_type="int8_float16",
                         cpu_threads=4, num_workers=1, local_files_only=True)
    require(model.model.compute_type == "int8_float16" and model.model.device == "cuda", "Changed runtime")
    started = time.perf_counter()
    for index, (clip, baseline) in enumerate(zip(clips[len(records):], baselines[len(records):]), len(records) + 1):
        record = make_record(model, clip, baseline, plan_ref)
        validate_record(record, clip, baseline, plan_ref)
        path = RECORDS / (clip["id"] + ".json")
        save_new(path, record)
        progress["records"].append({"id": clip["id"], **bind(path)})
        save_progress(progress)
        records.append(record)
        print(json.dumps({"completed": index, "total": 120, "id": clip["id"], "status": record["status"]}), flush=True)
    for ref in refs + setup["model_files"]:
        verify(ref)
    progress_checked, records_checked = load_progress(clips, baselines, plan_ref)
    require(progress_checked == progress and records_checked == records, "Records changed during diagnosis")
    result = {"schema": "step07-context-asr-results-v1", "status": "completed", "created_utc": datetime.now(timezone.utc).isoformat(),
              "context_plan": plan_ref, "inputs": plan["inputs"], "ids": plan["ids"], "record_count": 120,
              "records": records, "summary": summarize(records), "elapsed_seconds_this_run": time.perf_counter() - started,
              "progress": bind(PROGRESS), "record_files": [bind(RECORDS / (clip["id"] + ".json")) for clip in clips],
              "same_model_as_step6": True, "independent_asr_cross_validation": False, "audio_uploaded": False,
              "automatic_approval": False, "human_listening_confirmed_count": 0, "labels_modified": False,
              "training_executed": False, "limits": plan["limits"]}
    save_new(RESULT, result)
    del model
    dll_handle.close()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", action="store_true")
    group.add_argument("--run", action="store_true")
    parser.add_argument("--expected-plan-sha256")
    args = parser.parse_args()
    value = freeze() if args.freeze else run(args.expected_plan_sha256)
    print(json.dumps({"status": value["status"], "summary": value.get("summary")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
