"""Prepare exact parent-only listening evidence for the remaining eight rows."""
from __future__ import annotations

import hashlib
import json
import wave
from pathlib import Path

BASE = Path(__file__).resolve().parent
PARENT = BASE.parent / "offline_boundary_finalize_step02d_20261013"
SOURCE = BASE.parent / "offline_boundary_review_step02c_20261013" / "review_manifest_r1.json"
MEDIA = BASE.parent / "offline_alignment_step02_20261012" / "media"


def main() -> None:
    """Write context PCM slices and preserve unanswered review state."""
    try:
        for name in ("completion_manifest.json", "completion_manifest_r2.json"):
            frozen = json.loads((PARENT / "results" / name).read_text(encoding="utf-8"))
            for item in frozen["artifacts"]:
                if hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest() != item["sha256"]:
                    raise ValueError(f"Prior evidence changed: {item['path']}")
        source = json.loads(SOURCE.read_text(encoding="utf-8"))
        index = {row["id"]: row for row in source["records"]}
        audit = json.loads((PARENT / "results/parent_metadata_audit.json").read_text(encoding="utf-8"))
        ids = audit["remaining_eight_parent_ids_outside_this_review"]
        questions = BASE / "questions"
        questions.mkdir(parents=True, exist_ok=True)
        (BASE / "results").mkdir(exist_ok=True)
        rows = []
        for rank, identifier in enumerate(ids, 9):
            row = index[identifier]
            context = MEDIA / identifier / "context.wav"
            start, end = row["sentence_start_sample_16k"], row["sentence_end_sample_16k"]
            with wave.open(str(context), "rb") as reader:
                if (reader.getframerate(), reader.getnchannels(), reader.getsampwidth()) != (16000, 1, 2):
                    raise ValueError("Unexpected context PCM format")
                if not 0 <= start < end <= reader.getnframes():
                    raise ValueError("Invalid parent sample bounds")
                reader.setpos(start)
                pcm = reader.readframes(end - start)
            output = questions / f"{rank:02d}_parent_exact.wav"
            if output.exists():
                raise ValueError("Refusing to replace existing listening evidence")
            with wave.open(str(output), "wb") as writer:
                writer.setnchannels(1)
                writer.setsampwidth(2)
                writer.setframerate(16000)
                writer.writeframes(pcm)
            rows.append({
                "rank": rank, "id": identifier, "speaker_id": row["speaker_id"], "text": row["text"],
                "start_sample_16k": start, "end_sample_16k": end,
                "context_audio_path": str(context), "context_audio_sha256": hashlib.sha256(context.read_bytes()).hexdigest(),
                "question_audio_path": str(output), "question_audio_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "automatic_flags": row["automatic_flags"], "alignment_candidate_count": len(row["segments"]),
                "parent_transcript_decision": None, "parent_audio_boundary_decision": None,
                "phrase_audio_decision": None, "visual_decision": None, "training_ready": False,
            })
        result = {"schema": "offline-remaining-eight-parent-review-v1", "source_manifest": str(SOURCE),
                  "source_manifest_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(), "records": rows,
                  "scope": "Only remaining parent sentence listening; no short-phrase or visual approvals.",
                  "training_ready": False, "stage2_complete": False}
        (BASE / "remaining_parent_manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"remaining_parents": len(rows), "question_wavs": len(rows), "new_training_clips": 0}))
    except (OSError, KeyError, TypeError, ValueError, wave.Error) as exc:
        raise RuntimeError("Remaining parent review preparation failed") from exc


if __name__ == "__main__":
    main()
