"""Independently verify saved review proposals and PCM slices using stdlib only."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
from typing import Any
import wave

BASE = Path(__file__).resolve().parent
OLD = BASE.parent / "offline_alignment_step02_20261012"


def load(path: Path) -> Any:
    """Read evidence without importing any producer code."""
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Cannot read {path}") from exc


def sha(path: Path) -> str:
    """Hash an entire evidence file."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise RuntimeError(f"Cannot hash {path}") from exc


def check(condition: bool, message: str) -> None:
    """Raise even when Python assertions have been disabled."""
    try:
        if not condition:
            raise ValueError(message)
    except ValueError as exc:
        raise RuntimeError(f"Independent audit failed: {message}") from exc


def pcm(path: Path) -> tuple[int, bytes]:
    """Validate and return mono 16 kHz PCM frames."""
    try:
        with wave.open(str(path), "rb") as handle:
            check((handle.getnchannels(), handle.getsampwidth(), handle.getframerate(), handle.getcomptype())
                  == (1, 2, 16000, "NONE"), f"PCM contract: {path}")
            count = handle.getnframes()
            raw = handle.readframes(count)
            check(len(raw) == count * 2, f"Truncated PCM: {path}")
            return count, raw
    except (OSError, wave.Error) as exc:
        raise RuntimeError(f"Cannot read PCM {path}") from exc


def rational_round(numerator: int, denominator: int) -> int:
    """Compute nearest integer with ties to even using integer division."""
    try:
        check(numerator >= 0 and denominator > 0, "Nonnegative rounding operands")
        q, r = divmod(numerator, denominator)
        return q + int(2 * r > denominator or (2 * r == denominator and q % 2 == 1))
    except (ValueError, ZeroDivisionError) as exc:
        raise RuntimeError("Integer rounding failed") from exc


def tokens(text: str) -> list[int]:
    """Recreate fixed 69-token labels directly from Unicode Hangul arithmetic."""
    try:
        result: list[int] = []
        for char in text:
            if char == " ":
                result.append(68)
            else:
                value = ord(char) - 0xAC00
                check(0 <= value < 11172, "Expected canonical modern Korean syllable")
                lead, rest = divmod(value, 21 * 28)
                vowel, tail = divmod(rest, 28)
                result.extend((1 + lead, 20 + vowel))
                if tail:
                    result.append(40 + tail)
        return result
    except (ValueError, TypeError) as exc:
        raise RuntimeError("Independent Hangul arithmetic failed") from exc


def main() -> None:
    """Verify all 410 candidates, provenance links, and sixteen PCM outputs."""
    try:
        manifest_path = BASE / "review_manifest.json"
        batch_path = BASE / "results/listening_batch.json"
        manifest, batch = load(manifest_path), load(batch_path)
        canonical_path = OLD / "results/revision2/canonical_alignment_results.json"
        media_path = OLD / "results/media_manifest.json"
        canonical = load(canonical_path)
        media = {r["id"]: r for r in load(media_path)["records"]}
        frozen = load(OLD / "results/completion_manifest.json")
        for entry in frozen["artifacts"]:
            check(sha(Path(entry["path"])) == entry["sha256"], "Frozen Stage2 artifact identity")
        for path, expected in manifest["sources"].items():
            check(sha(Path(path)) == expected, f"Manifest source identity: {path}")
        check(manifest["training_ready"] is False and manifest["automatic_approval"] is False
              and manifest["default_decision"] is None, "No automatic approval")
        check(batch["training_ready"] is False and batch["review_manifest_sha256"] == sha(manifest_path), "Batch binding")
        check([r["id"] for r in manifest["records"]] == [r["id"] for r in canonical["records"]], "All parent identities/order")
        check(len(manifest["records"]) == 16 and len(set(r["id"] for r in manifest["records"])) == 16, "Sixteen unique parents")
        count: Counter[str] = Counter()
        rows, recommended, source_pcm = {}, {}, {}
        max_float_residual_samples = 0.0
        for row, original in zip(manifest["records"], canonical["records"], strict=True):
            identity = row["id"]
            m = media[identity]
            rows[identity] = row
            check(row["text"] == original["text"], "Canonical sentence unchanged")
            check(row["review_target_sha256"] == hashlib.sha256(row["text"].encode()).hexdigest(), "Text digest")
            check(row["speaker_id"] == m["speaker_id"], "Speaker unchanged")
            source_count, raw = pcm(Path(m["context_audio"]["path"]))
            source_pcm[identity] = raw
            check(row["context_samples_16k"] == source_count and row["context_duration_seconds"] == source_count / 16000, "Context extent")
            for audio_key in ("context_audio", "sentence_audio"):
                a = m[audio_key]
                check(sha(Path(a["path"])) == a["sha256"], "Old WAV hash")
                check(abs(a["source_first_sample"] / a["source_sample_rate"] - a["actual_source_start_seconds"]) < 1e-10, "Native source origin")
            for key, name, field in (("full_preview", "context_video.mp4", "context_video_url"),
                                     ("roi_preview", "context_lips.mp4", "roi_video_url")):
                v = m["video"][key]
                check(sha(Path(v["path"])) == v["sha256"] == manifest["sources"][v["path"]], "Old MP4 hash binding")
                check(row[field] == f"/media/{identity}/{name}", "Media URL identity")
            check(row["context_audio_url"] == f"/media/{identity}/context.wav", "Audio URL identity")
            sentence, context = m["sentence_audio"], m["context_audio"]
            check(sentence["source_sample_rate"] == context["source_sample_rate"], "Native rates equal")
            origin = Fraction(sentence["source_first_sample"] - context["source_first_sample"], sentence["source_sample_rate"]) * 16000
            check(origin.denominator == 1, "Exact context sample origin")
            origin = int(origin)
            end = origin + sentence["samples"]
            check(row["sentence_start_sample_16k"] == origin and row["sentence_end_sample_16k"] == end <= source_count, "Sentence sample bounds")
            check(row["sentence_start_in_context"] == origin / 16000 and row["sentence_end_in_context"] == end / 16000, "Sentence time display")
            source_candidates = [("word", s) for s in original["words"]] + [("phrase", s) for s in original["phrases"]]
            check(len(row["segments"]) == len(source_candidates), "All source proposals retained")
            check(len({s["segment_id"] for s in row["segments"]}) == len(row["segments"]), "Unique local segment IDs")
            for seg, (kind, old) in zip(row["segments"], source_candidates, strict=True):
                first = old["index"] if kind == "word" else old["first_word_index"]
                length = 1 if kind == "word" else old["word_count"]
                check(seg["segment_id"] == f"{kind}:{first}:{length}" and seg["kind"] == kind, "Stable candidate identity")
                check(seg["first_word_index"] == first and seg["word_count"] == length, "Word extent")
                check(seg["text"] == old["text"] == " ".join(row["text"].split()[first:first+length]), "Contiguous canonical target")
                for label in ("start", "end"):
                    local_samples = old[f"local_audio_{label}"] * 16000
                    nearest = int(local_samples + .5)
                    max_float_residual_samples = max(max_float_residual_samples, abs(local_samples-nearest))
                    check(abs(local_samples-nearest) < 1e-6, "Exact local sample grid")
                    check(seg[f"{label}_sample_16k"] == origin + nearest, "Local-to-context mapping")
                    check(seg[f"{label}_in_context"] == (origin + nearest) / 16000, "Context display time")
                    check(abs(old[f"wav_{label}"] - (context["source_first_sample"] / context["source_sample_rate"] + seg[f"{label}_sample_16k"] / 16000)) < 1e-10, "Absolute WAV origin")
                    check(abs(old[f"video_{label}"] - old[f"wav_{label}"] - m["offset_seconds"]) < 1e-10, "One video offset")
                left, right = seg["start_sample_16k"], seg["end_sample_16k"]
                check(origin <= left < right <= end <= source_count, "Proposal within sentence and context")
                frames = rational_round((right-left)*25, 16000)
                labels = tokens(seg["text"])
                minimum = len(labels) + sum(a == b for a, b in zip(labels, labels[1:]))
                check(labels == old["visual_token_ids"], "Independent 69-token encoding")
                check(frames == seg["frames_25fps"] == old["proposed_frames_25fps"], "Independent rational frame count")
                check(minimum == seg["ctc_min_frames"] == old["visual_ctc_min_frames"], "Independent CTC length")
                check(seg["ctc_feasible"] == old["ctc_feasible_proposal"] == (frames >= minimum), "CTC feasibility preserved")
                native = [rational_round(*Fraction(str((old["video_start"] + t/25) * m["video"]["native_fps"])).as_integer_ratio()) for t in range(frames)]
                check(native == old["proposed_native_indices"], "Prior candidate native indices retained in source")
                check(all(old[k] is False for k in ("actual_training_clip_created", "human_transcript_approved", "human_audio_boundary_approved", "human_visual_context_approved", "training_ready")), "No prior approval promotion")
                if kind == "word":
                    flags = set(old["flags"])
                else:
                    flags = {f for w in original["words"][first:first+length] for f in w["flags"] if f != "insufficient_frames_for_visual_ctc"}
                    if frames < minimum:
                        flags.add("insufficient_frames_for_visual_ctc")
                check(set(seg["flags"]) == flags and len(flags) == len(seg["flags"]), "Warnings preserved")
                if seg["recommended"]:
                    recommended[(identity, seg["segment_id"])] = seg
                else:
                    check(seg["recommendation_rank"] is None, "No hidden selection rank")
                count[kind] += 1
                count[kind + "_ctc_feasible"] += frames >= minimum
                count["zero_frames"] += frames == 0
        check(count["word"] == 125 and count["phrase"] == 285 and len(recommended) == 8, "Expected retained counts")
        check([b["rank"] for b in batch["records"]] == list(range(1, 9)), "Eight ordered questions")
        check(len({b["speaker_id"] for b in batch["records"]}) == 8, "One question per speaker")
        check(set(recommended) == {(b["id"], b["segment"]["segment_id"]) for b in batch["records"]}, "Question selection equality")
        for q in batch["records"]:
            identity, seg = q["id"], q["segment"]
            check(seg == recommended[(identity, seg["segment_id"])] and seg["recommendation_rank"] == q["rank"], "Question segment full binding")
            check(q["speaker_id"] == rows[identity]["speaker_id"] and q["human_decision"] is None, "Question human state")
            for field, margin in (("exact_audio", 0), ("context_audio", 3200)):
                item = q[field]
                left = max(0, seg["start_sample_16k"] - margin)
                right = min(rows[identity]["context_samples_16k"], seg["end_sample_16k"] + margin)
                check((item["start_sample_16k"], item["end_sample_16k"], item["samples"]) == (left, right, right-left), "Exact or +/- 0.2 second boundaries")
                n, raw = pcm(Path(item["path"]))
                check(n == right-left and raw == source_pcm[identity][2*left:2*right], "Exact source PCM byte slice")
                check(sha(Path(item["path"])) == item["sha256"] and hashlib.sha256(raw).hexdigest() == item["pcm_sha256"], "Question WAV/PCM hashes")
                check(item["review_only"] is True, "Review-only audio")
                count["verified_question_wavs"] += 1
        report = {
            "schema": "offline-boundary-review-independent-audit-v1",
            "status": "passed_saved_candidates_sample_geometry_and_pcm_identity_audit",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "auditor_sha256": sha(Path(__file__)),
            "review_manifest_sha256": sha(manifest_path), "listening_batch_sha256": sha(batch_path),
            "checked_source_hashes": len(manifest["sources"]), "frozen_stage2_artifacts_verified": len(frozen["artifacts"]),
            "records": 16, "speakers": len({r["speaker_id"] for r in manifest["records"]}),
            "counts": dict(count), "maximum_local_float_residual_samples": max_float_residual_samples,
            "production_functions_imported": False, "neural_calls": 0, "training_updates": 0,
            "human_approvals_created": 0, "training_ready": False,
            "limits": ["This verifies saved proposal arithmetic and review-media identity, not actual word boundaries.",
                       "No human transcript, audio boundary, or visual boundary approval was inferred.",
                       "Eight questions are illustrative choices and not representative accuracy samples.",
                       "Original source TARs and MP4 streams were not decoded again; saved provenance and file hashes were checked.",
                       "Browser playback timing, UI behavior, and export importer are outside this artifact audit."]}
        (BASE / "results/independent_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False))
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        raise RuntimeError("Independent saved-artifact audit did not pass") from exc


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
