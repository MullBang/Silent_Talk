"""Bind existing Stage2 candidates to exact PCM samples for human review only."""
from __future__ import annotations

import hashlib
import json
import sys
import wave
from fractions import Fraction
from pathlib import Path
from typing import Any

BASE = Path(__file__).resolve().parent
OLD = BASE.parent / "offline_alignment_step02_20261012"
PICKS = [
    ("C002_A_009_045", "못 꿀 것 같아"),
    ("C017_A_008_004", "못살게 구는 것 같아"),
    ("C312_A_010_010", "정말 잘 돼 있어"),
    ("C004_A_011_014", "야근을 많이 해서"),
    ("C032_A_012_019", "보는 것 같아"),
    ("C282_A_011_039", "먹게 되는 것 같아"),
    ("C293_A_011_018", "정말 잘 돼 있어"),
    ("C297_A_006_006", "너무 많이 막혀"),
]


def digest(path: Path) -> str:
    """Hash bytes without altering a source artifact."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ValueError(f"Cannot hash {path}") from exc


def read(path: Path) -> Any:
    """Read a local UTF-8 JSON artifact."""
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read {path}") from exc


def write(path: Path, value: Any) -> None:
    """Write deterministic JSON into this new review version."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot write {path}") from exc


def sample(seconds: float) -> int:
    """Reject timestamps that are not exact 16 kHz sample locations."""
    try:
        value = seconds * 16000
        result = round(value)
        if abs(value - result) > 1e-6:
            raise ValueError("Timestamp does not map to an integer PCM sample")
        return result
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Invalid sample time {seconds}") from exc


def cut_pcm(source: Path, output: Path, left: int, right: int) -> dict[str, Any]:
    """Write an exact PCM byte slice, never a timer-stopped preview."""
    try:
        with wave.open(str(source), "rb") as handle:
            if (handle.getframerate(), handle.getnchannels(), handle.getsampwidth()) != (16000, 1, 2):
                raise ValueError("Expected mono 16 kHz signed 16-bit PCM")
            if not 0 <= left < right <= handle.getnframes():
                raise ValueError("PCM slice outside source")
            handle.setpos(left)
            raw = handle.readframes(right - left)
        output.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(output), "wb") as handle:
            handle.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            handle.writeframes(raw)
        return {"path": str(output), "sha256": digest(output), "start_sample_16k": left,
                "end_sample_16k": right, "samples": right - left,
                "pcm_sha256": hashlib.sha256(raw).hexdigest(), "review_only": True}
    except (OSError, ValueError, wave.Error) as exc:
        raise ValueError(f"Cannot cut review audio {output}") from exc


def main() -> None:
    """Keep frozen sources intact and expose all 410 provisional segments."""
    try:
        completed = read(OLD / "results/completion_manifest.json")
        for item in completed["artifacts"]:
            if digest(Path(item["path"])) != item["sha256"]:
                raise ValueError(f"Frozen Stage2 artifact changed: {item['path']}")
        canonical_path = OLD / "results/revision2/canonical_alignment_results.json"
        media_path = OLD / "results/media_manifest.json"
        previous_path = OLD / "results/revision2/review_manifest.json"
        canonical = read(canonical_path)
        media = {row["id"]: row for row in read(media_path)["records"]}
        previous = {row["id"]: row for row in read(previous_path)["records"]}
        source_paths = [canonical_path, media_path, previous_path, OLD / "results/completion_manifest.json", Path(__file__)]
        sources = {str(path): digest(path) for path in source_paths}
        rows, questions = [], []
        selected = {tail: (rank + 1, text) for rank, (tail, text) in enumerate(PICKS)}
        for record in canonical["records"]:
            identity = record["id"]
            m, old = media[identity], previous[identity]
            audio = Path(m["context_audio"]["path"])
            if digest(audio) != m["context_audio"]["sha256"]:
                raise ValueError("Context audio differs")
            sources[str(audio)] = digest(audio)
            with wave.open(str(audio), "rb") as handle:
                frames = handle.getnframes()
                if handle.getframerate() != 16000:
                    raise ValueError("Unexpected sample rate")
            native_rate = m["sentence_audio"]["source_sample_rate"]
            origin = Fraction(m["sentence_audio"]["source_first_sample"] - m["context_audio"]["source_first_sample"], native_rate) * 16000
            if origin.denominator != 1:
                raise ValueError("Sentence/context origins not on the same 16 kHz grid")
            start = int(origin)
            end = start + m["sentence_audio"]["samples"]
            row = {key: old[key] for key in ["id", "text", "source_text", "context_audio_url", "context_video_url", "roi_video_url", "automatic_flags"]}
            row.update(speaker_id=m["speaker_id"], review_target_sha256=hashlib.sha256(record["text"].encode()).hexdigest(),
                       context_duration_seconds=frames / 16000, context_samples_16k=frames,
                       sentence_start_in_context=start / 16000, sentence_end_in_context=end / 16000,
                       sentence_start_sample_16k=start, sentence_end_sample_16k=end, segments=[])
            for filename in ("context_video.mp4", "context_lips.mp4"):
                video = audio.parent / filename
                sources[str(video)] = digest(video)
            for kind, candidates in [("word", record["words"]), ("phrase", record["phrases"])]:
                for candidate in candidates:
                    first = candidate.get("first_word_index", candidate.get("index"))
                    count = candidate.get("word_count", 1)
                    left, right = start + sample(candidate["local_audio_start"]), start + sample(candidate["local_audio_end"])
                    if not start <= left < right <= end <= frames:
                        raise ValueError(f"Candidate outside parent: {identity}")
                    num_frames = round(Fraction((right - left) * 25, 16000))
                    if num_frames != candidate["proposed_frames_25fps"]:
                        raise ValueError("r2 frame count changed")
                    selected_info = next((v for tail, v in selected.items() if identity.endswith(tail) and candidate["text"] == v[1] and kind == "phrase"), None)
                    flags = sorted(set(candidate.get("flags", [])))
                    if kind == "phrase":
                        flags = sorted({flag for word in record["words"][first:first+count] for flag in word["flags"] if flag != "insufficient_frames_for_visual_ctc"})
                    if not candidate["ctc_feasible_proposal"] and "insufficient_frames_for_visual_ctc" not in flags:
                        flags.append("insufficient_frames_for_visual_ctc")
                    seg = {"segment_id": f"{kind}:{first}:{count}", "kind": kind, "text": candidate["text"],
                           "first_word_index": first, "word_count": count, "start_sample_16k": left, "end_sample_16k": right,
                           "start_in_context": left / 16000, "end_in_context": right / 16000,
                           "frames_25fps": num_frames, "ctc_min_frames": candidate["visual_ctc_min_frames"],
                           "ctc_feasible": candidate["ctc_feasible_proposal"], "flags": flags,
                           "recommended": selected_info is not None, "recommendation_rank": selected_info[0] if selected_info else None}
                    row["segments"].append(seg)
                    if selected_info:
                        rank = selected_info[0]
                        exact = cut_pcm(audio, BASE / f"questions/{rank:02d}_exact.wav", left, right)
                        context = cut_pcm(audio, BASE / f"questions/{rank:02d}_context.wav", max(0, left - 3200), min(frames, right + 3200))
                        questions.append({"rank": rank, "id": identity, "speaker_id": m["speaker_id"], "segment": seg,
                                          "exact_audio": exact, "context_audio": context, "human_decision": None})
            rows.append(row)
        questions.sort(key=lambda item: item["rank"])
        if len(questions) != 8 or len({item["speaker_id"] for item in questions}) != 8:
            raise ValueError("Expected eight illustrative phrases from eight speakers")
        manifest = {"schema": "lipnet-offline-boundary-review-manifest-v1", "sources": sources, "records": rows,
                    "selection_policy": "All 125 word + 285 phrase candidates retained unchanged. Eight explicitly chosen illustrative phrases (one per speaker) only set the review order, not quality or training eligibility.",
                    "default_decision": None, "automatic_approval": False, "training_ready": False}
        write(BASE / "review_manifest.json", manifest)
        write(BASE / "results/listening_batch.json", {"schema": "offline-boundary-listening-batch-v1", "review_manifest_sha256": digest(BASE / "review_manifest.json"), "records": questions, "training_ready": False})
        segments = [seg for row in rows for seg in row["segments"]]
        write(BASE / "results/preparation_summary.json", {"status": "prepared_exact_audio_and_phrase_review_human_pending", "records": len(rows),
              "word_candidates": sum(s["kind"] == "word" for s in segments), "phrase_candidates": sum(s["kind"] == "phrase" for s in segments),
              "recommended_phrases": len(questions), "frozen_stage2_artifacts_verified": len(completed["artifacts"]),
              "human_approvals": 0, "new_neural_calls": 0, "training_updates": 0, "training_ready": False})
        print(json.dumps(read(BASE / "results/preparation_summary.json"), ensure_ascii=False))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise RuntimeError("Review preparation failed; no approval inferred") from exc


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
