"""Independent CPU-only audit of saved Step6 ASR evidence; no model imports.

Recomputes comparisons, gates and ByteLevel token decoding with the standard
library. Reads only existing local excerpts, model bytes and evidence files.
This cannot authenticate inference or prove a transcript by listening.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import difflib
import hashlib
import importlib.metadata
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
PINS = {
    "asr_plan.json": "a6864c0ebf2b59f8db8a8d0baa1c02b3c241a499e18159bb81079580b8c99e8c",
    "source_plan.json": "5187f0ccbf84ea099d2f52d0a0e8d94db11d75deed50da1ad093e1a032465929",
    "results/media_manifest.json": "fc74e5664fbfc835a9b10173334e397be4abe51dfc7dac6d04c871ba30cdfdf8",
    "results/independent_media_audit.json": "a331e258d58973f4b966bb4699716f09bca46ab24225cbfc31adee6d6f08db2e",
    "results/asr_results.json": "db3983ad00d2a3ddec267009a2904b6e3522bba649adb10f55ceeb545c105018",
    "results/asr_difference_review.json": "f40792d71016c8c2739d17d89d990b4ead737c6a08bd2f814dbadc4249611724",
}
OPTIONS = {
    "language": "ko", "task": "transcribe", "beam_size": 5, "best_of": 5,
    "patience": 1.0, "length_penalty": 1.0, "repetition_penalty": 1.0,
    "no_repeat_ngram_size": 0, "temperature": 0.0,
    "compression_ratio_threshold": 2.4, "log_prob_threshold": -1.0,
    "no_speech_threshold": .6, "condition_on_previous_text": False,
    "prompt_reset_on_temperature": .5, "initial_prompt": None, "prefix": None,
    "hotwords": None, "suppress_blank": True, "suppress_tokens": [-1],
    "without_timestamps": False, "max_initial_timestamp": 1.0,
    "word_timestamps": False, "prepend_punctuations": "\"'“¿([{-",
    "append_punctuations": "\"'.。,，!！?？:：”)]}、", "multilingual": False,
    "vad_filter": False, "vad_parameters": None, "max_new_tokens": None,
    "chunk_length": None, "clip_timestamps": "0",
    "hallucination_silence_threshold": None, "language_detection_threshold": .5,
    "language_detection_segments": 1, "log_progress": False,
}
THRESHOLDS = {"boundary_margin_seconds": .08, "timestamp_outside_tolerance_seconds": .02,
              "minimum_average_log_probability": -1., "maximum_no_speech_probability": .6,
              "maximum_compression_ratio": 2.4}
SEGMENT_FIELDS = set("id seek start end text tokens avg_logprob compression_ratio no_speech_prob words temperature".split())
RECORD_FIELDS = set("id split speaker_id source_stem reference start_time end_time timing_status schema asr_plan audio status asr_text raw_segments transcription_info error boundary_flags confidence_flags review_reasons review_status human_listening_confirmed label_modified training_ready inference_seconds normalized_reference normalized_asr metrics".split())


def require(value, message):
    if not value:
        raise ValueError(message)


def finite(value, name):
    require(type(value) in (int, float) and math.isfinite(value), "Nonfinite/non-numeric " + name)
    return value


def same(actual, expected, name):
    # JSON true must not silently compare equal to integer 1.
    require(json.dumps(actual, sort_keys=True, ensure_ascii=False, allow_nan=False) ==
            json.dumps(expected, sort_keys=True, ensure_ascii=False, allow_nan=False), "Mismatch: " + name)


def close(actual, expected, name, tolerance=1e-10):
    require(abs(finite(actual, name) - expected) <= tolerance, "Numeric mismatch: " + name)


def read(path):
    def bad(value):
        raise ValueError("Nonfinite JSON: " + value)
    def pairs(values):
        result = {}
        for key, value in values:
            require(key not in result, "Duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=bad, object_pairs_hook=pairs)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def bind(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path)}


def verify(spec):
    path = Path(spec["path"]).resolve()
    require(path.is_relative_to(ROOT), "Refusing non-project payload: " + str(path))
    require(sha(path) == spec["sha256"], "Changed bytes: " + str(path))
    if "bytes" in spec:
        require(path.stat().st_size == spec["bytes"], "Changed length: " + str(path))
    return path


def normalize(text):
    require(type(text) is str, "Transcript must be string")
    out = []
    for char in unicodedata.normalize("NFC", text):
        out.append(" " if unicodedata.category(char)[0] == "P" else char)
    return " ".join("".join(out).split())


def edit_distance(a, b):
    # Full independent table rather than the producer's rolling row.
    costs = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        costs[i][0] = i
    for j in range(len(b) + 1):
        costs[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            costs[i][j] = min(costs[i-1][j] + 1, costs[i][j-1] + 1,
                              costs[i-1][j-1] + int(a[i-1] != b[j-1]))
    return costs[-1][-1]


def comparison(reference, prediction):
    ref, pred = normalize(reference), normalize(prediction)
    without_ref, without_pred = ref.replace(" ", ""), pred.replace(" ", "")
    require(without_ref, "Empty reference")
    edits, without_edits = edit_distance(ref, pred), edit_distance(without_ref, without_pred)
    return {"normalized_reference": ref, "normalized_asr": pred, "metrics": {
        "character_edits": edits, "reference_characters": len(ref),
        "cer_percent": edits * 100. / len(ref), "exact": ref == pred,
        "without_spaces_character_edits": without_edits,
        "without_spaces_reference_characters": len(without_ref),
        "without_spaces_cer_percent": without_edits * 100. / len(without_ref),
        "without_spaces_exact": without_ref == without_pred}}


class ByteTokens:
    """Pure Python GPT2 ByteLevel decode and limited punctuation-only BPE encode.

    Encoding is only used for the fixed non-speech punctuation suppression set,
    whose strings are single ByteLevel regex groups. It is not a general encoder.
    """
    def __init__(self, spec):
        require(spec["model"]["type"] == "BPE" and spec["decoder"]["type"] == "ByteLevel",
                "Unexpected tokenizer format")
        self.vocab = spec["model"]["vocab"]
        self.inverse = {value: key for key, value in self.vocab.items()}
        require(len(self.inverse) == len(self.vocab), "Duplicate vocabulary ID")
        self.special = {entry["content"]: entry["id"] for entry in spec["added_tokens"]}
        self.ids = set(self.inverse) | set(self.special.values())
        self.eot = self.special["<|endoftext|>"]
        self.timestamp = self.special["<|notimestamps|>"] + 1
        base = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        self.byte_chars = {value: chr(value) for value in base}
        missing = [value for value in range(256) if value not in self.byte_chars]
        self.byte_chars.update({value: chr(256 + index) for index, value in enumerate(missing)})
        self.char_bytes = {char: value for value, char in self.byte_chars.items()}
        self.ranks = {tuple(pair.split(" ") if isinstance(pair, str) else pair): i
                      for i, pair in enumerate(spec["model"]["merges"])}

    def decode(self, tokens):
        require(all(type(token) is int and token in self.ids for token in tokens), "Unknown raw token")
        chars = "".join(self.inverse[token] for token in tokens if token < self.eot)
        return bytes(self.char_bytes[char] for char in chars).decode("utf-8", errors="replace")

    def punctuation_tokens(self, text):
        require(text.lstrip(" ") and len(text) - len(text.lstrip(" ")) <= 1 and
                all(not char.isspace() and not char.isalnum() for char in text.lstrip(" ")),
                "Only one punctuation regex group is supported")
        parts = [self.byte_chars[value] for value in text.encode("utf-8")]
        while len(parts) > 1:
            pairs = list(zip(parts, parts[1:]))
            best = min(pairs, key=lambda pair: self.ranks.get(pair, math.inf))
            if best not in self.ranks:
                break
            merged, i = [], 0
            while i < len(parts):
                if i+1 < len(parts) and (parts[i], parts[i+1]) == best:
                    merged.append(parts[i] + parts[i+1]); i += 2
                else:
                    merged.append(parts[i]); i += 1
            parts = merged
        return [self.vocab[part] for part in parts]

    def suppressed(self):
        symbols = list('"#()*+/:;<=>@[\\]^_`{|}~「」『』')
        symbols += "<< >> <<< >>> -- --- -( -[ (' (\" (( )) ((( ))) [[ ]] {{ }} ♪♪ ♪♪♪".split()
        music = set("♩♪♫♬♭♮♯")
        ids = {self.punctuation_tokens(" -")[0], self.punctuation_tokens(" '")[0]}
        for symbol in symbols + sorted(music):
            for version in (symbol, " " + symbol):
                encoded = self.punctuation_tokens(version)
                if len(encoded) == 1 or symbol in music:
                    ids.add(encoded[0])
        ids.update(self.special[token] for token in ("<|transcribe|>", "<|translate|>",
            "<|startoftranscript|>", "<|startofprev|>", "<|startoflm|>", "<|nospeech|>"))
        return sorted(ids)


def check_info(info, duration, decoder):
    require(set(info) == set("language language_probability duration duration_after_vad all_language_probs transcription_options vad_options".split()), "Info fields")
    require(info["language"] == "ko" and info["language_probability"] == 1 and
            info["all_language_probs"] is None and info["vad_options"] is None, "Changed language/VAD")
    close(info["duration"], duration, "input duration", 1/16000 + 1e-6)
    close(info["duration_after_vad"], duration, "duration after VAD", 1/16000 + 1e-6)
    excluded = {"language", "task", "temperature", "vad_filter", "vad_parameters", "chunk_length",
                "language_detection_threshold", "language_detection_segments", "log_progress"}
    expected = {key: value for key, value in OPTIONS.items() if key not in excluded}
    expected.update(temperatures=[0.0], clip_timestamps=[0.0], suppress_tokens=decoder.suppressed())
    same(info["transcription_options"], expected, "actual recognizer options")


def segment_flags(segments, duration):
    boundary, confidence = set(), set()
    if not segments:
        return [], ["no_segments"]
    if segments[0]["start"] <= .08:
        boundary.add("first_asr_segment_near_audio_start")
    if segments[-1]["end"] >= duration - .08:
        boundary.add("last_asr_segment_near_audio_end")
    for i, row in enumerate(segments):
        if (row["start"] < -.02 or row["end"] > duration + .02 or row["end"] < row["start"] or
                (i and row["start"] < segments[i-1]["end"] - .02)):
            boundary.add("asr_segment_timestamp_outside_or_nonmonotonic")
        if row["avg_logprob"] < -1.:
            confidence.add("low_segment_average_log_probability")
        if row["no_speech_prob"] >= .6:
            confidence.add("high_segment_no_speech_probability")
        if row["compression_ratio"] > 2.4:
            confidence.add("high_segment_compression_ratio")
    return sorted(boundary), sorted(confidence)


def check_segments(segments, decoder):
    for index, row in enumerate(segments, 1):
        require(set(row) == SEGMENT_FIELDS, "Unexpected segment fields")
        require(type(row["id"]) is int and row["id"] == index and type(row["seek"]) is int
                and row["seek"] >= 0, "Raw segment ID/seek")
        require(row["words"] is None and row["temperature"] == 0.0, "Changed word/temperature policy")
        for key in ("start", "end", "avg_logprob", "compression_ratio", "no_speech_prob"):
            finite(row[key], key)
        require(0 <= row["no_speech_prob"] <= 1 and row["compression_ratio"] >= 0, "Invalid raw statistics")
        require(row["tokens"] and type(row["text"]) is str, "Empty raw token evidence")
        same(decoder.decode(row["tokens"]), row["text"], "raw tokens decoded text")
        require(row["tokens"][0] >= decoder.timestamp and row["tokens"][-1] >= decoder.timestamp,
                "Missing timestamp token evidence")
        for field, token in (("start", row["tokens"][0]), ("end", row["tokens"][-1])):
            close(row[field], row["seek"] * .01 + (token - decoder.timestamp) * .02,
                  "timestamp token " + field)


def check_audio(spec):
    path = verify(spec)
    require(path.is_relative_to(BASE / "results/media") and path.name == "label_audio.wav", "Wrong audio kind")
    require(spec["clock"] == "original_label_wav" and spec["source_role"] == "audio", "Wrong source clock")
    with wave.open(str(path), "rb") as stream:
        require((stream.getframerate(), stream.getnchannels(), stream.getsampwidth(), stream.getcomptype()) ==
                (16000, 1, 2, "NONE"), "Wrong audio header")
        frames = stream.getnframes()
        require(frames == spec["frames"] > 0 and len(stream.readframes(frames)) == frames * 2, "Truncated PCM")
    require(spec["sample_rate"] == 16000 and spec["channels"] == 1 and spec["sample_width_bytes"] == 2,
            "Wrong audio specification")
    close(spec["duration_seconds"], frames / 16000, "PCM duration")
    return frames


def check_record(row, clip, plan_ref, decoder):
    require(set(row) == RECORD_FIELDS and row["schema"] == "step06-local-asr-record-v1", "Record schema")
    same(row["asr_plan"], plan_ref, "record plan")
    for key in ("id", "split", "speaker_id", "source_stem", "reference", "start_time", "end_time", "timing_status", "audio"):
        same(row[key], clip[key], "immutable clip " + key)
    require(all(row[key] is False for key in ("human_listening_confirmed", "label_modified", "training_ready")), "Unwarranted review claim")
    require(finite(row["inference_seconds"], "inference_seconds") >= 0, "Negative runtime")
    require(row["status"] in ("ok", "asr_failed", "not_transcribed"), "Unknown status")
    for key, value in comparison(clip["reference"], row["asr_text"]).items():
        same(row[key], value, "comparison " + key)
    eligible = clip["timing_status"] == "verified" and clip["artifacts_ready"] is True and clip["audio"] is not None
    if row["status"] == "ok":
        require(eligible and row["error"] is None, "Success gate")
        check_info(row["transcription_info"], clip["audio"]["frames"] / 16000, decoder)
        check_segments(row["raw_segments"], decoder)
        same(row["asr_text"], "".join(segment["text"] for segment in row["raw_segments"]), "segment concatenation")
        boundary, confidence = segment_flags(row["raw_segments"], clip["audio"]["frames"] / 16000)
        if not normalize(row["asr_text"]):
            confidence = sorted(set(confidence) | {"empty_normalized_transcription"})
    else:
        require(row["status"] == ("asr_failed" if eligible else "not_transcribed"), "Failure classification")
        require(row["asr_text"] == "" and row["raw_segments"] == [] and row["transcription_info"] is None
                and type(row["error"]) is str and row["error"], "Failure evidence")
        boundary, confidence = [], []
    same(row["boundary_flags"], boundary, "boundary flags")
    same(row["confidence_flags"], confidence, "confidence flags")
    reasons = boundary + confidence
    if row["status"] != "ok":
        reasons += ["asr_not_completed"]
    if clip["timing_status"] != "verified":
        reasons += ["time_mapping_not_verified"]
    if not row["metrics"]["exact"]:
        reasons += ["normalized_transcript_mismatch"]
    reasons = sorted(set(reasons))
    same(row["review_reasons"], reasons, "review reasons")
    same(row["review_status"], "human_listening_required" if reasons else "automatic_support_only", "review classification")


def summary(records):
    edits = sum(row["metrics"]["character_edits"] for row in records)
    chars = sum(row["metrics"]["reference_characters"] for row in records)
    return {"records": len(records), "asr_status_counts": dict(Counter(row["status"] for row in records)),
            "review_status_counts": dict(Counter(row["review_status"] for row in records)),
            "normalized_exact_matches": sum(row["metrics"]["exact"] for row in records),
            "without_spaces_exact_matches": sum(row["metrics"]["without_spaces_exact"] for row in records),
            "character_edits": edits, "reference_characters": chars,
            "pooled_cer_percent_including_failed_empty": 100. * edits / chars,
            "boundary_flagged_sentences": sum(bool(row["boundary_flags"]) for row in records),
            "confidence_flagged_sentences": sum(bool(row["confidence_flags"]) for row in records),
            "human_listening_confirmed_count": 0, "labels_modified": False,
            "interpretation": "Auxiliary ASR/reference agreement only, not lipreading accuracy or manual review completion"}


def check_plan(plan):
    require(plan["schema"] == "step06-local-asr-audit-plan-v1" and plan["status"] == "frozen_before_transcription", "ASR plan schema")
    same(plan["transcribe_options"], OPTIONS, "planned options")
    same(plan["review_thresholds"], THRESHOLDS, "planned thresholds")
    require(plan["reference_supplied_to_asr"] is False and plan["labels_modified"] is False and
            plan["training_executed"] is False and plan["human_listening_confirmed_count"] == 0, "Plan scope")
    model = plan["model"]
    expected = {"repository": "Systran/faster-whisper-large-v3", "revision": "edaa852ec7e145841d8ffdb056a99866b5f0a478",
                "directory": str(ROOT / "models/asr_audit/faster-whisper-large-v3"), "device": "cuda",
                "device_index": 0, "compute_type": "int8_float16", "cpu_threads": 4, "num_workers": 1, "clips_per_call": 1}
    same(model, expected, "model plan")


def verify_model(plan, setup):
    require(setup["status"] == "ready" and setup["model_loaded_successfully"] is True
            and setup["loaded_device"] == "cuda" and setup["loaded_compute_type"] == "int8_float16", "Setup readiness")
    require(setup["revision"] == plan["model"]["revision"] and setup["model_directory"] == plan["model"]["directory"], "Setup model identity")
    for key in ("audio_read", "transcription_executed", "source_audio_uploaded", "original_environments_modified"):
        require(setup[key] is False, "Setup scope " + key)
    for spec in setup["cuda_dlls"] + [setup["repository"], setup["installation_report"], setup["setup_code"]]:
        verify(spec)
    repository = read(setup["repository"]["path"])
    require(repository["repo_id"] == plan["model"]["repository"] and repository["revision"] == setup["revision"], "Public revision")
    public = {spec["name"]: spec for spec in repository["files"]}
    require(len(public) == len(setup["model_files"]) == 6, "Model inventory")
    total = 0
    for spec in setup["model_files"]:
        path = verify(spec)
        require(path.parent == Path(setup["model_directory"]) and spec["public_digest_verified"] is True, "Model path/digest claim")
        public_spec = public[path.name]
        require(spec["bytes"] == public_spec["size"], "Public model length")
        if public_spec["lfs_sha256"]:
            require(spec["sha256"] == public_spec["lfs_sha256"], "Public LFS hash")
        else:
            content = path.read_bytes()
            digest = hashlib.sha1(b"blob " + str(len(content)).encode("ascii") + b"\0" + content).hexdigest()
            require(digest == public_spec["git_blob_id"], "Public Git blob hash")
        total += spec["bytes"]
    require(total == setup["model_bytes"], "Model total bytes")
    env = ROOT / ".venv-audio-audit"
    require(Path(setup["executable"]).resolve() == env / "Scripts/python.exe", "ASR environment path")
    site = env / "Lib/site-packages"
    packages = {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions(path=[str(site)])}
    same(packages, setup["packages"], "Installed ASR package metadata")
    paths = sorted((site / "faster_whisper").glob("*.py"))
    for package in ("ctranslate2", "tokenizers", "numpy/_core"):
        paths.extend(sorted((site / package).glob("*.pyd")))
    same(plan["implementation"], [bind(path) for path in paths], "Implementation inventory/hashes")
    return total


def number_forms(digits):
    """Independent conservative enumeration for saved triage, not a correction."""
    if not re.fullmatch("[0-9]{1,4}", digits):
        return set()
    numeral = "영일이삼사오육칠팔구"
    result = {"".join(numeral[int(c)] for c in digits), "".join("공" if c == "0" else numeral[int(c)] for c in digits)}
    n = int(digits)
    if n == 0:
        result.update(("영", "공", "제로"))
    if len(digits) == 1 or digits[0] != "0":
        spelled = ""
        for position, digit in enumerate(reversed(digits)):
            if digit == "0":
                continue
            spelled = (("" if digit == "1" and position else numeral[int(digit)]) + ["", "십", "백", "천"][position]) + spelled
        if spelled:
            result.add(spelled)
        if 0 < n < 100:
            tens = ["", "열", "스물", "서른", "마흔", "쉰", "예순", "일흔", "여든", "아흔"][n // 10]
            for forms in (["", "하나", "둘", "셋", "넷", "다섯", "여섯", "일곱", "여덟", "아홉"],
                          ["", "한", "두", "세", "네", "다섯", "여섯", "일곱", "여덟", "아홉"]):
                result.add(tens + forms[n % 10])
            if n == 20:
                result.add("스무")
    return result


def category(row):
    if row["status"] != "ok":
        return "asr_unavailable_requires_listening", []
    if row["normalized_reference"] == row["normalized_asr"]:
        return "normalized_match", []
    left, right = (row[key].replace(" ", "") for key in ("normalized_reference", "normalized_asr"))
    if left == right:
        return "spacing_only_difference", []
    spans = []
    for op, a, b, c, d in difflib.SequenceMatcher(a=left, b=right, autojunk=False).get_opcodes():
        if op != "equal":
            spans.append({"operation": op, "reference_part": left[a:b], "asr_part": right[c:d],
                          "reference_span_without_spaces": [a, b], "asr_span_without_spaces": [c, d]})
    numbers = all(span["asr_part"] in number_forms(span["reference_part"]) or
                  span["reference_part"] in number_forms(span["asr_part"]) for span in spans)
    numbers = numbers and not re.search(r"[0-9][.,][0-9]", row["reference"] + " " + row["asr_text"])
    return ("possible_number_expression_difference" if numbers else "possible_content_difference"), spans


def check_triage(triage, results, result_ref):
    require(triage["schema"] == "step06-asr-listening-triage-v1" and triage["status"] == "completed", "Triage schema")
    same(triage["asr_results"], result_ref, "Triage source")
    same(triage["ids"], results["ids"], "Triage order")
    require(triage["record_count"] == len(triage["records"]) == len(results["records"]), "Triage count")
    for key in ("code", "tests"):
        verify(triage[key])
    for item, row in zip(triage["records"], results["records"]):
        for key in ("id", "split", "speaker_id", "source_stem", "reference", "asr_text", "normalized_reference", "normalized_asr", "audio", "boundary_flags", "confidence_flags", "metrics"):
            same(item[key], row[key], "Triage immutable " + key)
        same(item["asr_status"], row["status"], "Triage ASR status")
        same(item["original_asr_review_status"], row["review_status"], "Triage gate retained")
        expected, spans = category(row)
        same(item["category"], expected, "Triage category")
        same(item["difference_spans"], spans, "Triage difference spans")
        priority = 1 if row["boundary_flags"] or expected in ("possible_content_difference", "asr_unavailable_requires_listening") else 2 if expected == "possible_number_expression_difference" else 3 if expected == "spacing_only_difference" else 4
        same(item["suggested_listening_priority"], priority, "Triage priority")
        require(all(item[key] is False for key in ("human_listening_confirmed", "training_ready", "label_modified")), "Triage review claim")
    same(triage["category_counts"], dict(Counter(row["category"] for row in triage["records"])), "Triage totals")
    same(triage["mismatches_in_original_order"], [row for row in triage["records"] if row["category"] != "normalized_match"], "Triage mismatch order")
    require(triage["human_listening_confirmed_count"] == 0 and triage["labels_modified"] is False and triage["model_executed"] is False, "Triage scope")


def audit(output):
    started = time.perf_counter()
    require(not output.exists(), "Preserve existing independent audit")
    documents, refs = {}, {}
    for name, expected in PINS.items():
        ref = bind(BASE / name)
        require(ref["sha256"] == expected, "Frozen artifact changed: " + name)
        refs[name] = ref
        documents[name] = read(ref["path"])
    plan, results, media, source, prior = [documents[name] for name in (
        "asr_plan.json", "results/asr_results.json", "results/media_manifest.json", "source_plan.json", "results/independent_media_audit.json")]
    check_plan(plan)
    require(prior["status"] == "passed", "Prior media audit did not pass")
    same(prior["media_manifest"], refs["results/media_manifest.json"], "Prior media binding")
    same(prior["source_plan"], refs["source_plan.json"], "Prior source binding")
    for key in ("code", "tests"):
        verify(prior[key])
    same(plan["media_manifest"], refs["results/media_manifest.json"], "Plan media")
    for spec in plan["code"] + [plan["setup"]]:
        verify(spec)
    same([Path(spec["path"]).name for spec in plan["code"]], ["asr_audit.py", "test_asr_audit.py"], "Production code inventory")
    setup = read(plan["setup"]["path"])
    model_bytes = verify_model(plan, setup)
    decoder = ByteTokens(read(Path(setup["model_directory"]) / "tokenizer.json"))
    ids = [row["id"] for row in source["records"]]
    require(len(ids) == len(set(ids)) == 120, "Fixed original cohort")
    clips = media["clips"]
    for name, actual in (("plan", plan["ids"]), ("results", results["ids"]), ("media", [row["id"] for row in clips]), ("records", [row["id"] for row in results["records"]])):
        same(actual, ids, name + " fixed order")
    same(dict(Counter(row["split"] for row in clips)), {"train": 96, "val": 24}, "Cohort roles")
    same(plan["split_counts"], {"train": 96, "val": 24}, "Plan split counts")
    require(results["schema"] == "step06-local-asr-audit-results-v1" and results["status"] == "completed" and
            results["record_count"] == plan["record_count"] == 120, "ASR results schema/count")
    for key, expected in (("asr_plan", refs["asr_plan.json"]), ("media_manifest", plan["media_manifest"]), ("setup", plan["setup"])):
        same(results[key], expected, "Results " + key)
    require(all(results[key] is False for key in ("audio_uploaded", "label_modified", "training_executed")) and
            results["human_listening_confirmed_count"] == 0, "Result scope claims")
    same(results["limits"], plan["limits"], "Preserved limits")
    require(finite(results["elapsed_seconds_this_run"], "elapsed") >= 0, "Negative elapsed")
    same(plan["available_audio"], [{"id": clip["id"], "path": clip["audio"]["path"], "sha256": clip["audio"]["sha256"]} for clip in clips], "Audio index")
    verify(results["progress"])
    require(Path(results["progress"]["path"]).resolve() == BASE / "results/asr_progress.json", "Progress path")
    progress = read(results["progress"]["path"])
    require(progress["schema"] == "step06-local-asr-progress-v1", "Progress schema")
    same(progress["asr_plan"], refs["asr_plan.json"], "Progress plan")
    directory = BASE / "results/asr_records"
    require({path.name for path in directory.glob("*.json")} == {identity + ".json" for identity in ids}, "Missing/orphan record files")
    expected_refs = [bind(directory / (identity + ".json")) for identity in ids]
    same(results["record_files"], expected_refs, "Result record index")
    same(progress["records"], [{"id": identity, **ref} for identity, ref in zip(ids, expected_refs)], "Progress record index")
    frames = 0
    audited_rows = []
    for selected, clip, row, ref in zip(source["records"], clips, results["records"], expected_refs):
        same(read(ref["path"]), row, "Individual record vs aggregate")
        for key in ("id", "split", "speaker_id", "start_time", "end_time"):
            same(clip[key], selected[key], "Selection " + key)
        same(clip["reference"], selected["source_text"], "Original label text")
        require(clip["training_ready"] is False and clip["human_transcript_review"] == "not_reviewed", "Media review scope")
        frames += check_audio(clip["audio"])
        check_record(row, clip, refs["asr_plan.json"], decoder)
        audited_rows.append({"id": row["id"], "record": ref, "audio_sha256": row["audio"]["sha256"],
                             "raw_segments": len(row["raw_segments"]), "raw_token_text_checked": True,
                             "comparison_and_flags_checked": True})
    same(results["summary"], summary(results["records"]), "Full summary")
    check_triage(documents["results/asr_difference_review.json"], results, refs["results/asr_results.json"])
    # Confirm all small source artifacts stayed unchanged during this audit.
    for ref in list(refs.values()) + expected_refs + [results["progress"], plan["setup"]] + plan["code"] + plan["available_audio"]:
        verify(ref)
    value = {"schema": "lipnet-new120-independent-asr-audit-v1", "status": "passed",
        "created_utc": datetime.now(timezone.utc).isoformat(), "code": bind(__file__),
        "tests": bind(BASE / "test_verify_asr.py"), "inputs": refs,
        "asr_setup": plan["setup"], "asr_progress": results["progress"],
        "summary": summary(results["records"]), "split_counts": {"train": 96, "val": 24},
        "boundary_flag_counts": dict(Counter(flag for row in results["records"] for flag in row["boundary_flags"])),
        "confidence_flag_counts": dict(Counter(flag for row in results["records"] for flag in row["confidence_flags"])),
        "triage_category_counts": documents["results/asr_difference_review.json"]["category_counts"],
        "checked_audio_frames": frames, "checked_model_bytes": model_bytes,
        "raw_segments_checked": sum(row["raw_segments"] for row in audited_rows),
        "implementation_files_checked": len(plan["implementation"]), "records": audited_rows,
        "model_executed": False, "training_executed": False, "labels_modified": False,
        "human_listening_confirmed_count": 0, "training_ready": False,
        "audit_runtime": {"python": sys.version, "executable": sys.executable, "standard_library_only": True},
        "elapsed_seconds": time.perf_counter() - started,
        "limits": ["This independently checks saved evidence consistency, not fresh model inference or the truth of spoken content",
                   "Boundary flags are conservative timestamp review cues and do not prove actual audio clipping",
                   "ASR/reference agreement and decoder statistics are not human listening or lip synchronization proof",
                   "Public model bytes are checked against recorded repository digests; no network freshness lookup was performed",
                   "Producer code and actual option metadata are checked; saved records cannot independently prove runtime isolation",
                   "Difference categories are listening hypotheses and do not replace labels or change the frozen gates"]}
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=BASE / "results/independent_asr_audit.json")
    args = parser.parse_args()
    result = audit(args.output.resolve())
    print(json.dumps({"status": result["status"], "summary": result["summary"], "output": bind(args.output)}, ensure_ascii=False))
