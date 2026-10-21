"""Create three review-only variants in response to actual reports of clipped audio."""
from __future__ import annotations

import json
import sys
import wave
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

from prepare_phrase_media import BASE, load, sha, write_pcm


def main() -> None:
    """Preserve originals and produce new, still-unapproved ranges."""
    try:
        output = BASE / "phrase_variants_r1.json"
        if output.exists():
            raise ValueError("Refusing to replace variants")
        original = load(BASE / "phrase_review_manifest.json")
        by_rank = {r["rank"]: r for r in original["records"]}
        evidence_paths = [BASE / "user_evidence/001_phrase_audio_answers.json", BASE / "user_evidence/002_phrase_audio_answers.json"]
        replies = {r["rank"]: r for p in evidence_paths for r in load(p)["records"]}
        rows = []
        for rank in (9, 11, 15):
            row, evidence = by_rank[rank], replies[rank]
            if evidence["phrase_audio_approved"]:
                raise ValueError("A repair must not claim an approved phrase was rejected")
            start, end = row["start_sample_16k"], row["end_sample_16k"]
            if rank == 9:
                new_start, new_end = start-1600, end
                reason = "Actual suspected onset clipping. Trial 100ms earlier start; not a measured boundary error; listen for any previous-word contamination."
            else:
                new_start, new_end = start, row["parent_end_sample_16k"]
                reason = "Actual final sound clipping. Include the previously human-confirmed parent end; target is the final phrase of this parent. New phrase still needs confirmation."
            if not row["parent_start_sample_16k"] <= new_start < new_end <= row["parent_end_sample_16k"]:
                raise ValueError("Variant exceeds confirmed parent")
            context = Path(row["source_context_path"])
            if sha(context) != row["source_context_sha256"]:
                raise ValueError("Changed source PCM")
            with wave.open(str(context), "rb") as reader:
                pcm = reader.readframes(reader.getnframes())
            exact = write_pcm(pcm, new_start, new_end, BASE / "media" / f"{rank:02d}_phrase_r1_exact.wav")
            expanded = write_pcm(pcm, max(row["parent_start_sample_16k"], new_start-3200), min(row["parent_end_sample_16k"], new_end+3200), BASE / "media" / f"{rank:02d}_phrase_r1_context.wav")
            new_frames = round(Fraction(new_end-new_start, 16000)*25)
            if new_frames < row["ctc_min_frames"]:
                raise ValueError("CTC infeasible variant")
            rows.append({**row, "prior_segment_id": row["segment_id"], "segment_id": f"review-r1:{row['segment_id']}",
                         "start_sample_16k": new_start, "end_sample_16k": new_end,
                         "duration_seconds": (new_end-new_start)/16000, "frames_25fps": new_frames,
                         "frame_count_policy": "Historical half-even estimate only; final extraction policy pending.",
                         "prior_sample_range": [start,end], "delta_start_samples": new_start-start, "delta_end_samples": new_end-end,
                         "parent_end_minus_phrase_end_ms": (row["parent_end_sample_16k"]-new_end)/16,
                         "existing_candidate_unchanged": False, "reason": reason, "actual_trigger": evidence,
                         "exact_audio": exact, "context_audio": expanded, "phrase_audio_approval": None,
                         "visual_approval": None, "training_ready": False})
        document = {"schema": "stage2g-human-triggered-phrase-variants-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
                    "status": "new_variants_prepared_actual_listening_pending", "records": rows,
                    "source_sha256": {str(p): sha(p) for p in [BASE / "phrase_review_manifest.json", *evidence_paths, Path(__file__)]},
                    "originals_preserved": True, "training_ready": False, "stage2_complete": False,
                    "new_neural_calls": 0, "new_TAR_reads": 0, "new_downloads": 0,
                    "warning": "Extension amounts are trial crop edits, not measured phonetic timing errors."}
        output.write_text(json.dumps(document, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        print(json.dumps({"new_variants": len(rows), "ranks": [r["rank"] for r in rows], "training_ready": False}))
    except (OSError, ValueError, KeyError, TypeError, wave.Error) as exc:
        raise RuntimeError("Variant preparation failed") from exc


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
