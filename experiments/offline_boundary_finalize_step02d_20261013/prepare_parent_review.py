"""Prepare exact parent audio and eight review records from frozen evidence."""
from __future__ import annotations

import hashlib
import json
import wave
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
PREVIOUS = BASE.parent / "offline_boundary_review_step02c_20261013"
MEDIA = BASE.parent / "offline_alignment_step02_20261012" / "media"


def sha(path: Path) -> str:
    """Hash a local evidence file."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise RuntimeError(f"Cannot hash {path}") from exc


def main() -> None:
    """Preserve prior decisions and prepare parent-only questions."""
    try:
        source = PREVIOUS / "review_manifest_r1.json"
        frozen = json.loads((PREVIOUS / "results/completion_manifest.json").read_text(encoding="utf-8"))
        for artifact in frozen["artifacts"]:
            if sha(Path(artifact["path"])) != artifact["sha256"]:
                raise ValueError(f"Prior evidence changed: {artifact['path']}")
        source_data = json.loads(source.read_text(encoding="utf-8"))
        questions = BASE / "questions"
        questions.mkdir(parents=True, exist_ok=True)
        rows = []
        for parent in source_data["records"]:
            for segment in parent["segments"]:
                if not segment.get("recommended"):
                    continue
                if segment["external_audio_evidence"]["decision"] != "acceptable":
                    raise ValueError("Current phrase lacks actual audio confirmation")
                context = MEDIA / parent["id"] / "context.wav"
                start = parent["sentence_start_sample_16k"]
                end = parent["sentence_end_sample_16k"]
                with wave.open(str(context), "rb") as reader:
                    if (reader.getnchannels(), reader.getsampwidth(), reader.getframerate()) != (1, 2, 16000):
                        raise ValueError("Unexpected context audio format")
                    if not 0 <= start < end <= reader.getnframes():
                        raise ValueError("Parent outside context")
                    reader.setpos(start)
                    pcm = reader.readframes(end - start)
                rank = segment["recommendation_rank"]
                output = questions / f"{rank:02d}_parent_exact.wav"
                if output.exists():
                    with wave.open(str(output), "rb") as reader:
                        if reader.readframes(reader.getnframes()) != pcm:
                            raise ValueError("Refusing to replace different question PCM")
                else:
                    with wave.open(str(output), "wb") as writer:
                        writer.setnchannels(1)
                        writer.setsampwidth(2)
                        writer.setframerate(16000)
                        writer.writeframes(pcm)
                row = {key: parent[key] for key in (
                    "id", "speaker_id", "text", "context_audio_url", "context_video_url", "roi_video_url",
                    "sentence_start_sample_16k", "sentence_end_sample_16k", "context_samples_16k",
                )}
                row.update(
                    rank=rank,
                    phrase={key: segment[key] for key in ("segment_id", "text", "start_sample_16k", "end_sample_16k")},
                    audio_evidence={"decision": "acceptable", "scope": "phrase_only", "source": segment["external_audio_evidence"]},
                    parent_exact_audio_path=str(output),
                    parent_exact_audio_sha256=sha(output),
                    context_audio_sha256=sha(context),
                    parent_transcript_decision=None,
                    parent_audio_boundary_decision=None,
                    phrase_visual_boundary_decision=None,
                    training_ready=False,
                )
                rows.append(row)
        rows.sort(key=lambda row: row["rank"])
        if len(rows) != 8 or len({row["id"] for row in rows}) != 8:
            raise ValueError("Expected eight distinct parents")
        result = {
            "schema": "offline-parent-and-visual-review-manifest-v1",
            "source_manifest": str(source), "source_manifest_sha256": sha(source),
            "records": rows, "training_ready": False, "stage2_complete": False,
            "audio_contract": "Exact 16kHz integer slices of context.wav. Separate historical sentence.wav is not asserted PCM-identical.",
            "prior_frozen_artifacts_verified": len(frozen["artifacts"]),
        }
        (BASE / "review_manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"parents": len(rows), "previous_frozen_files": len(frozen["artifacts"]), "new_training_clips": 0}))
    except (OSError, ValueError, KeyError, TypeError, wave.Error) as exc:
        raise RuntimeError("Parent review preparation failed") from exc


if __name__ == "__main__":
    main()
