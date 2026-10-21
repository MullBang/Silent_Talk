"""Local audio-only auxiliary transcript audit; never edits labels or trains.

The recognizer receives only audio samples and fixed unprompted Korean options.
References are compared after transcription. Exact ASR agreement provides
automatic support, never human verification or proof of visual synchronization.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
import unicodedata
import wave

import numpy as np

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
MEDIA = BASE / "results/media_manifest.json"
PLAN = BASE / "asr_plan.json"
SETUP = BASE / "asr_setup.json"
RESULT = BASE / "results/asr_results.json"
RECORDS = BASE / "results/asr_records"
PROGRESS = BASE / "results/asr_progress.json"
OPTIONS = {"language": "ko", "task": "transcribe", "beam_size": 5, "best_of": 5,
    "patience": 1.0, "length_penalty": 1.0, "repetition_penalty": 1.0, "no_repeat_ngram_size": 0,
    "temperature": 0.0, "compression_ratio_threshold": 2.4, "log_prob_threshold": -1.0,
    "no_speech_threshold": 0.6, "condition_on_previous_text": False, "prompt_reset_on_temperature": 0.5,
    "initial_prompt": None, "prefix": None, "hotwords": None, "suppress_blank": True, "suppress_tokens": [-1],
    "without_timestamps": False, "max_initial_timestamp": 1.0, "word_timestamps": False,
    "prepend_punctuations": "\"'“¿([{-", "append_punctuations": "\"'.。,，!！?？:：”)]}、",
    "multilingual": False, "vad_filter": False, "vad_parameters": None, "max_new_tokens": None,
    "chunk_length": None, "clip_timestamps": "0", "hallucination_silence_threshold": None,
    "language_detection_threshold": 0.5, "language_detection_segments": 1, "log_progress": False}
FLAGS = {"boundary_margin_seconds": 0.08, "timestamp_outside_tolerance_seconds": 0.02,
         "minimum_average_log_probability": -1.0, "maximum_no_speech_probability": 0.6,
         "maximum_compression_ratio": 2.4}


class AudioEvidenceError(ValueError):
    """Input evidence changed or is malformed; do not silently downgrade it."""


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def bind(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path)}


def verify(spec):
    require(sha(spec["path"]) == spec["sha256"], f"Changed ASR audit evidence: {spec['path']}")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_new(path, value):
    path = Path(path)
    require(not path.exists(), f"Refusing to overwrite audit evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix="." + path.name + ".", suffix=".tmp", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write((json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)


def save_progress(path, value):
    """Atomically update only this run's mutable per-record checksum index."""
    path = Path(path)
    require(path.resolve().is_relative_to(BASE.resolve()), "Progress must remain inside this experiment")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix="." + path.name + ".", suffix=".tmp", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write((json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)


def normalize_for_comparison(text):
    """NFC, punctuation to spaces, whitespace collapse; no spelling/number edits."""
    require(isinstance(text, str), "Transcript must be text")
    return " ".join("".join(" " if unicodedata.category(char).startswith("P") else char
                            for char in unicodedata.normalize("NFC", text)).split())


def distance(reference, prediction):
    previous = list(range(len(prediction) + 1))
    for i, left in enumerate(reference, 1):
        current = [i]
        for j, right in enumerate(prediction, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (left != right)))
        previous = current
    return previous[-1]


def compare_text(reference, prediction):
    ref, pred = normalize_for_comparison(reference), normalize_for_comparison(prediction)
    require(ref and ref.replace(" ", ""), "Empty normalized reference")
    edits, chars = distance(ref, pred), len(ref)
    without_ref, without_pred = ref.replace(" ", ""), pred.replace(" ", "")
    without_edits = distance(without_ref, without_pred)
    return {"normalized_reference": ref, "normalized_asr": pred,
            "metrics": {"character_edits": edits, "reference_characters": chars,
                        "cer_percent": 100. * edits / chars, "exact": ref == pred,
                        "without_spaces_character_edits": without_edits,
                        "without_spaces_reference_characters": len(without_ref),
                        "without_spaces_cer_percent": 100. * without_edits / len(without_ref),
                        "without_spaces_exact": without_ref == without_pred}}


def json_value(value):
    if is_dataclass(value):
        return json_value(asdict(value))
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, np.generic):
        return json_value(value.item())
    require(value is None or type(value) in (str, int, float, bool), "Unsupported ASR metadata type")
    require(not isinstance(value, float) or math.isfinite(value), "Nonfinite ASR metadata")
    return value


def assess_segments(segments, duration):
    """Flag uncertain boundaries/decoder statistics, never declare real clipping."""
    boundary, confidence = [], []
    if not segments:
        confidence.append("no_segments")
        return boundary, confidence
    margin, tolerance = FLAGS["boundary_margin_seconds"], FLAGS["timestamp_outside_tolerance_seconds"]
    if segments[0]["start"] <= margin:
        boundary.append("first_asr_segment_near_audio_start")
    if segments[-1]["end"] >= duration - margin:
        boundary.append("last_asr_segment_near_audio_end")
    previous_end = -math.inf
    for segment in segments:
        start, end = segment["start"], segment["end"]
        if start < -tolerance or end > duration + tolerance or end < start or start < previous_end - tolerance:
            boundary.append("asr_segment_timestamp_outside_or_nonmonotonic")
        previous_end = end
        if segment["avg_logprob"] < FLAGS["minimum_average_log_probability"]:
            confidence.append("low_segment_average_log_probability")
        if segment["no_speech_prob"] >= FLAGS["maximum_no_speech_probability"]:
            confidence.append("high_segment_no_speech_probability")
        if segment["compression_ratio"] > FLAGS["maximum_compression_ratio"]:
            confidence.append("high_segment_compression_ratio")
    return sorted(set(boundary)), sorted(set(confidence))


def load_audio(spec):
    """Only read generated local PCM, rejecting changed/truncated input."""
    try:
        return _load_audio(spec)
    except (OSError, ValueError, KeyError, wave.Error) as exc:
        raise AudioEvidenceError(str(exc)) from exc


def _load_audio(spec):
    path = Path(spec["path"]).resolve()
    require(path.is_relative_to(BASE.resolve()) and path.suffix.lower() == ".wav", "Only this audit's local WAV excerpts are allowed")
    verify(spec)
    with wave.open(str(path), "rb") as stream:
        require((stream.getframerate(), stream.getnchannels(), stream.getsampwidth(), stream.getcomptype()) ==
                (16000, 1, 2, "NONE"), "ASR needs mono PCM16 at 16000Hz")
        require(spec["sample_rate"] == 16000 and spec["channels"] == 1 and spec["sample_width_bytes"] == 2
                and stream.getnframes() == spec["frames"] > 0, "Audio manifest/header mismatch")
        data = stream.readframes(stream.getnframes())
    require(len(data) == spec["frames"] * 2, "Truncated audio samples")
    samples = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.
    verify(spec)
    return samples


def infer_audio_only(model, samples):
    """No reference argument exists at this recognizer boundary."""
    require(isinstance(samples, np.ndarray) and samples.dtype == np.float32 and samples.ndim == 1
            and len(samples) > 0 and np.isfinite(samples).all(), "Invalid audio samples")
    segments, info = model.transcribe(samples, **copy.deepcopy(OPTIONS))
    segments = [json_value(segment) for segment in segments]
    info = json_value(info)
    require(info["language"] == "ko", "Recognizer language differs from plan")
    actual = info["transcription_options"]
    for name in ("beam_size", "condition_on_previous_text", "initial_prompt", "prefix", "hotwords", "word_timestamps"):
        require(actual[name] == OPTIONS[name], f"Recognizer changed option: {name}")
    require(actual["temperatures"] == [0.0], "Unexpected temperature fallback")
    require(abs(info["duration"] - len(samples) / 16000) <= 1 / 16000 + 1e-6
            and abs(info["duration_after_vad"] - info["duration"]) <= 1 / 16000 + 1e-6,
            "Recognizer input duration/VAD changed")
    return "".join(segment["text"] for segment in segments), segments, info


def audit_clip(model, clip, plan_ref, audio_loader=load_audio):
    record = {key: clip[key] for key in ("id", "split", "speaker_id", "source_stem", "reference", "start_time", "end_time", "timing_status")}
    record.update({"schema": "step06-local-asr-record-v1", "asr_plan": plan_ref, "audio": clip.get("audio"),
        "status": "not_transcribed", "asr_text": "", "raw_segments": [], "transcription_info": None,
        "error": None, "boundary_flags": [], "confidence_flags": [], "review_reasons": [],
        "review_status": "human_listening_required", "human_listening_confirmed": False,
        "label_modified": False, "training_ready": False, "inference_seconds": 0.0})
    started = time.perf_counter()
    eligible = clip.get("timing_status") == "verified" and clip.get("artifacts_ready") is True and clip.get("audio") is not None
    try:
        require(eligible, "Media timing/artifact gate not satisfied")
        samples = audio_loader(clip["audio"])
        # Reference text is not supplied to this call or the recognizer options.
        text, segments, info = infer_audio_only(model, samples)
        boundary, confidence = assess_segments(segments, len(samples) / 16000)
        record.update(status="ok", asr_text=text, raw_segments=segments, transcription_info=info,
                      boundary_flags=boundary, confidence_flags=confidence)
        if not normalize_for_comparison(text):
            record["confidence_flags"] = sorted(set(confidence + ["empty_normalized_transcription"]))
    except AudioEvidenceError:
        raise
    except Exception as exc:
        record["status"] = "asr_failed" if eligible else "not_transcribed"
        record["error"] = f"{type(exc).__name__}: {exc}"
    record["inference_seconds"] = time.perf_counter() - started
    record.update(compare_text(clip["reference"], record["asr_text"]))
    reasons = []
    if record["status"] != "ok":
        reasons.append("asr_not_completed")
    if clip["timing_status"] != "verified":
        reasons.append("time_mapping_not_verified")
    if not record["metrics"]["exact"]:
        reasons.append("normalized_transcript_mismatch")
    reasons.extend(record["boundary_flags"] + record["confidence_flags"])
    record["review_reasons"] = sorted(set(reasons))
    if not reasons:
        record["review_status"] = "automatic_support_only"
    return record


def validate_saved_record(record, clip, plan_ref):
    """Verify resumed content after its separate progress-index SHA is checked."""
    required = {"id", "split", "speaker_id", "source_stem", "reference", "start_time", "end_time", "timing_status",
        "schema", "asr_plan", "audio", "status", "asr_text", "raw_segments", "transcription_info", "error",
        "boundary_flags", "confidence_flags", "review_reasons", "review_status", "human_listening_confirmed",
        "label_modified", "training_ready", "inference_seconds", "normalized_reference", "normalized_asr", "metrics"}
    require(set(record) == required and record["schema"] == "step06-local-asr-record-v1", "Malformed resumed record fields")
    require(record["asr_plan"] == plan_ref, "Resumed record belongs to another ASR plan")
    for name in ("id", "split", "speaker_id", "source_stem", "reference", "start_time", "end_time", "timing_status", "audio"):
        require(record[name] == clip[name], "Resumed immutable input differs: " + name)
    require(record["human_listening_confirmed"] is False and record["label_modified"] is False
            and record["training_ready"] is False, "Unwarranted human/label/training claim in resumed record")
    require(isinstance(record["inference_seconds"], (int, float)) and math.isfinite(record["inference_seconds"])
            and record["inference_seconds"] >= 0, "Malformed resumed runtime")
    for name, value in compare_text(clip["reference"], record["asr_text"]).items():
        require(record[name] == value, "Resumed text metrics changed")
    require(record["status"] in ("ok", "asr_failed", "not_transcribed"), "Invalid resumed status")
    if record["status"] == "ok":
        require(clip["timing_status"] == "verified" and clip["artifacts_ready"] is True and clip["audio"] is not None
                and record["error"] is None and isinstance(record["transcription_info"], dict), "Resumed success gate mismatch")
        require(record["asr_text"] == "".join(row["text"] for row in record["raw_segments"]), "Resumed raw segment text changed")
        boundary, confidence = assess_segments(record["raw_segments"], clip["audio"]["frames"] / 16000)
        if not normalize_for_comparison(record["asr_text"]):
            confidence = sorted(set(confidence + ["empty_normalized_transcription"]))
        require(boundary == record["boundary_flags"] and confidence == record["confidence_flags"], "Resumed review flags changed")
    else:
        require(record["asr_text"] == "" and record["raw_segments"] == [] and record["transcription_info"] is None
                and isinstance(record["error"], str) and record["error"]
                and record["boundary_flags"] == record["confidence_flags"] == [], "Malformed resumed failure")
        eligible = clip["timing_status"] == "verified" and clip["artifacts_ready"] is True and clip["audio"] is not None
        require(record["status"] == ("asr_failed" if eligible else "not_transcribed"), "Resumed failure category mismatch")
    reasons = ([] if record["status"] == "ok" else ["asr_not_completed"])
    if clip["timing_status"] != "verified":
        reasons.append("time_mapping_not_verified")
    if not record["metrics"]["exact"]:
        reasons.append("normalized_transcript_mismatch")
    reasons = sorted(set(reasons + record["boundary_flags"] + record["confidence_flags"]))
    require(record["review_reasons"] == reasons and record["review_status"] ==
            ("human_listening_required" if reasons else "automatic_support_only"), "Resumed support classification changed")


def load_progress(path, clips, plan_ref, record_directory=RECORDS):
    path, record_directory = Path(path), Path(record_directory)
    if path.exists():
        value = read(path)
        require(value.get("schema") == "step06-local-asr-progress-v1" and value.get("asr_plan") == plan_ref,
                "Progress plan/schema mismatch")
        refs = value["records"]
        require(isinstance(refs, list) and len(refs) <= len(clips)
                and [ref["id"] for ref in refs] == [clip["id"] for clip in clips[:len(refs)]],
                "Progress records must form the fixed ordered prefix")
    else:
        value = {"schema": "step06-local-asr-progress-v1", "asr_plan": plan_ref, "records": []}
        refs = []
    require(not record_directory.exists() or {p.name for p in record_directory.glob("*.json")} ==
            {ref["id"] + ".json" for ref in refs}, "Unindexed or missing resumed record; preserve and investigate")
    records = []
    for ref, clip in zip(refs, clips):
        require(set(ref) == {"id", "path", "sha256"} and Path(ref["path"]).resolve() ==
                (record_directory / (clip["id"] + ".json")).resolve(), "Unexpected resumed record path")
        verify(ref)
        record = read(ref["path"])
        validate_saved_record(record, clip, plan_ref)
        records.append(record)
    return value, records


def validate_media(media):
    clips = media["clips"]
    require(len(clips) == 120 and len({row["id"] for row in clips}) == 120, "Expected all 120 unique fixed clips")
    require(Counter(row["split"] for row in clips) == {"train": 96, "val": 24}, "Changed train96/val24 roles")
    require(len({row["source_stem"] for row in clips}) == 30, "Expected 30 source recordings")
    for row in clips:
        require(row["id"] and all(char.isalnum() or char == "_" for char in row["id"]), "Unsafe clip ID")
        require(row.get("training_ready") is False and row.get("human_transcript_review") == "not_reviewed",
                "Unexpected pre-existing training/human verification claims")
        require(isinstance(row["reference"], str) and normalize_for_comparison(row["reference"]), "Empty reference")
    return clips


def runtime_packages():
    return {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()}


def implementation_bindings():
    site = Path(sys.prefix) / "Lib/site-packages"
    paths = sorted((site / "faster_whisper").glob("*.py"))
    for package in ("ctranslate2", "tokenizers", "numpy/_core"):
        paths.extend(sorted((site / package).glob("*.pyd")))
    require(paths and any(path.suffix == ".pyd" for path in paths), "ASR implementation files absent")
    return [bind(path) for path in paths]


def verify_setup(setup):
    require(setup["status"] == "ready" and setup["model_loaded_successfully"]
            and setup["loaded_device"] == "cuda" and setup["loaded_compute_type"] == "int8_float16", "ASR setup is not ready")
    require(Path(sys.executable).resolve() == Path(setup["executable"]).resolve()
            and runtime_packages() == setup["packages"], "Use the frozen independent ASR environment")
    for spec in setup["model_files"] + setup["cuda_dlls"] + [setup["repository"], setup["installation_report"], setup["setup_code"]]:
        verify(spec)


def freeze_plan():
    require(not PLAN.exists(), "ASR plan already frozen")
    setup, media = read(SETUP), read(MEDIA)
    verify_setup(setup)
    clips = validate_media(media)
    available = []
    for row in clips:
        if row.get("audio") is not None:
            load_audio(row["audio"])
            available.append({"id": row["id"], **bind(row["audio"]["path"])})
    value = {"schema": "step06-local-asr-audit-plan-v1", "status": "frozen_before_transcription",
        "created_utc": datetime.now(timezone.utc).isoformat(), "media_manifest": bind(MEDIA), "setup": bind(SETUP),
        "ids": [row["id"] for row in clips], "record_count": 120, "split_counts": {"train": 96, "val": 24},
        "code": [bind(Path(__file__)), bind(BASE / "test_asr_audit.py")], "implementation": implementation_bindings(),
        "available_audio": available, "transcribe_options": OPTIONS, "review_thresholds": FLAGS,
        "model": {"repository": "Systran/faster-whisper-large-v3", "revision": setup["revision"],
                  "directory": setup["model_directory"], "device": "cuda", "device_index": 0,
                  "compute_type": "int8_float16", "cpu_threads": 4, "num_workers": 1, "clips_per_call": 1},
        "normalization": "NFC; every Unicode punctuation character becomes a space; collapse whitespace; preserve letters/numbers/unsupported symbols",
        "matching": "Exact match after normalization supports the label automatically only when time gate verified and no confidence/boundary flags",
        "boundary_policy": "Flag segment at/before 0.08s, at/after duration-0.08s, beyond audio +/-0.02s, inverted or overlapping timestamps; these are review cues, not proven clipped words",
        "confidence_policy": "Flag any avg_logprob<-1.0, no_speech_prob>=0.6, compression_ratio>2.4, no segments, or normalized empty output; values are uncalibrated decoder statistics",
        "failure_policy": "Retain all120 rows; absent/quarantined media not transcribed; ASR failure or mismatch needs human listening; no label replacement",
        "network_during_inference": "HF_HUB_OFFLINE=1; local_files_only=True; no audio upload, external API or account",
        "reference_supplied_to_asr": False, "human_listening_confirmed_count": 0,
        "labels_modified": False, "training_executed": False,
        "limits": ["ASR agreement is automatic corroboration, not manual transcript verification or proof of lip/audio synchronization",
                   "Forced Korean language probability is not independent language-detection confidence",
                   "Whitespace/number spelling differences can require review despite matching spoken content",
                   "No model outputs are allowed to replace source labels automatically"]}
    save_new(PLAN, value)
    return value


def summarize(records):
    require(len(records) == 120 and len({row["id"] for row in records}) == 120, "Incomplete ASR result cohort")
    edits = sum(row["metrics"]["character_edits"] for row in records)
    chars = sum(row["metrics"]["reference_characters"] for row in records)
    return {"records": 120, "asr_status_counts": dict(Counter(row["status"] for row in records)),
            "review_status_counts": dict(Counter(row["review_status"] for row in records)),
            "normalized_exact_matches": sum(row["metrics"]["exact"] for row in records),
            "without_spaces_exact_matches": sum(row["metrics"]["without_spaces_exact"] for row in records),
            "character_edits": edits, "reference_characters": chars, "pooled_cer_percent_including_failed_empty": 100. * edits / chars,
            "boundary_flagged_sentences": sum(bool(row["boundary_flags"]) for row in records),
            "confidence_flagged_sentences": sum(bool(row["confidence_flags"]) for row in records),
            "human_listening_confirmed_count": 0, "labels_modified": False,
            "interpretation": "Auxiliary ASR/reference agreement only, not lipreading accuracy or manual review completion"}


def run_audit(expected_plan_sha256):
    require(not RESULT.exists(), "ASR audit already completed; preserve evidence")
    plan_ref = bind(PLAN)
    require(isinstance(expected_plan_sha256, str) and len(expected_plan_sha256) == 64
            and plan_ref["sha256"] == expected_plan_sha256.lower(), "Explicit expected ASR plan SHA mismatch")
    plan = read(PLAN)
    verify(plan_ref)
    require(plan["status"] == "frozen_before_transcription" and plan["transcribe_options"] == OPTIONS
            and plan["review_thresholds"] == FLAGS, "Changed ASR plan/options")
    for spec in plan["code"] + plan["implementation"] + [plan["media_manifest"], plan["setup"]] + plan["available_audio"]:
        verify(spec)
    setup, media = read(SETUP), read(MEDIA)
    verify_setup(setup)
    clips = validate_media(media)
    require(plan["ids"] == [row["id"] for row in clips], "Changed fixed cohort order")
    progress, records = load_progress(PROGRESS, clips, plan_ref)
    if not PROGRESS.exists():
        save_progress(PROGRESS, progress)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HOME"] = str(ROOT / "models/asr_audit/hf_home")
    dll_dir = setup["cuda_dll_directory"]
    dll_handle = os.add_dll_directory(dll_dir)
    os.environ["PATH"] = dll_dir + os.pathsep + os.environ["PATH"]
    from faster_whisper import WhisperModel
    model = WhisperModel(plan["model"]["directory"], device="cuda", device_index=0, compute_type="int8_float16",
                         cpu_threads=4, num_workers=1, local_files_only=True)
    require(model.model.compute_type == "int8_float16" and model.model.device == "cuda", "Changed actual ASR runtime")
    started = time.perf_counter()
    for index, clip in enumerate(clips[len(records):], len(records) + 1):
        path = RECORDS / (clip["id"] + ".json")
        record = audit_clip(model, clip, plan_ref)
        validate_saved_record(record, clip, plan_ref)
        save_new(path, record)
        progress["records"].append({"id": clip["id"], **bind(path)})
        save_progress(PROGRESS, progress)
        records.append(record)
        print(json.dumps({"completed": index, "total": 120, "id": clip["id"], "status": record["status"],
                          "review_status": record["review_status"]}, ensure_ascii=False), flush=True)
    for spec in plan["code"] + plan["implementation"] + [plan_ref, plan["media_manifest"], plan["setup"]] + plan["available_audio"]:
        verify(spec)
    for spec in setup["model_files"]:
        verify(spec)
    progress_checked, records_checked = load_progress(PROGRESS, clips, plan_ref)
    require(progress_checked == progress and records_checked == records, "Progress changed during transcription")
    value = {"schema": "step06-local-asr-audit-results-v1", "status": "completed", "created_utc": datetime.now(timezone.utc).isoformat(),
        "asr_plan": plan_ref, "media_manifest": plan["media_manifest"], "setup": plan["setup"],
        "record_count": 120, "ids": plan["ids"], "records": records, "summary": summarize(records),
        "elapsed_seconds_this_run": time.perf_counter() - started,
        "record_files": [bind(RECORDS / (clip["id"] + ".json")) for clip in clips], "progress": bind(PROGRESS),
        "audio_uploaded": False, "label_modified": False, "training_executed": False,
        "human_listening_confirmed_count": 0, "limits": plan["limits"]}
    save_new(RESULT, value)
    del model
    dll_handle.close()
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", action="store_true")
    group.add_argument("--run", action="store_true")
    parser.add_argument("--expected-plan-sha256", help="Required with --run; exact reviewed asr_plan.json SHA-256")
    args = parser.parse_args()
    result = freeze_plan() if args.freeze else run_audit(args.expected_plan_sha256)
    print(json.dumps({"status": result["status"], "summary": result.get("summary")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
