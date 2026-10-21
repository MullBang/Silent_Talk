"""Propose wider natural phrases after reports of preceding speech contamination."""
from __future__ import annotations

import importlib.util
import json
import sys
import wave
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

from prepare_phrase_media import BASE, SOURCE, load, sha, write_pcm


def main() -> None:
    """Change text and range together while leaving all earlier versions intact."""
    try:
        target = BASE / "phrase_variants_r2.json"
        if target.exists():
            raise ValueError("Refusing existing variant replacement")
        tokenizer_path = BASE.parents[1] / "backend/models/tokenizer.py"
        spec = importlib.util.spec_from_file_location("pure_tokenizer", tokenizer_path)
        tokenizer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tokenizer)
        original = load(BASE / "phrase_review_manifest.json")
        by_rank = {r["rank"]: r for r in original["records"]}
        source = {r["id"]: r for r in load(SOURCE)["records"]}
        rows = []
        for rank, first_word, count, evidence_file in [(9, 1, 5, "004"), (10, 4, 4, "003")]:
            row = by_rank[rank]
            canonical = source[row["parent_id"]]
            words = sorted((s for s in canonical["segments"] if s["kind"] == "word"), key=lambda s:s["first_word_index"])
            selected = words[first_word:first_word+count]
            text = " ".join(w["text"] for w in selected)
            assert text in canonical["text"] and len(selected) == count
            # Trial boundaries between adjacent alignment spans; these are not measured phoneme boundaries.
            start = (words[first_word-1]["end_sample_16k"] + selected[0]["start_sample_16k"]) // 2
            if first_word+count == len(words):
                end = row["parent_end_sample_16k"]
                end_basis = "Previously confirmed parent end; phrase reaches final word."
            else:
                end = (selected[-1]["end_sample_16k"] + words[first_word+count]["start_sample_16k"]) // 2
                end_basis = "Trial midpoint to next aligned word; human check required."
            assert row["parent_start_sample_16k"] <= start < end <= row["parent_end_sample_16k"]
            token_ids = tokenizer.encode_text(text)
            ctc_min = tokenizer.ctc_min_input_len(token_ids)
            frames = round(Fraction(end-start, 16000)*25)
            assert frames >= ctc_min
            context = Path(row["source_context_path"])
            assert sha(context) == row["source_context_sha256"]
            with wave.open(str(context), "rb") as reader:
                pcm = reader.readframes(reader.getnframes())
            exact = write_pcm(pcm, start, end, BASE / "media" / f"{rank:02d}_phrase_r2_exact.wav")
            expanded = write_pcm(pcm, max(row["parent_start_sample_16k"],start-3200), min(row["parent_end_sample_16k"],end+3200), BASE / "media" / f"{rank:02d}_phrase_r2_context.wav")
            evidence_path = BASE / f"user_evidence/{evidence_file}_phrase_audio_answers.json"
            actual = next(r for r in load(evidence_path)["records"] if r["rank"] == rank)
            rows.append({**row, "prior_segment_id": row["segment_id"], "segment_id": f"review-r2:words:{first_word}:{count}",
                         "text": text, "first_word_index": first_word, "word_count": count,
                         "start_sample_16k": start, "end_sample_16k": end, "duration_seconds": (end-start)/16000,
                         "frames_25fps": frames, "label": token_ids, "ctc_min_frames": ctc_min,
                         "frame_count_policy": "Historical half-even estimate; final sampling policy pending.",
                         "parent_end_minus_phrase_end_ms": (row["parent_end_sample_16k"]-end)/16,
                         "existing_candidate_unchanged": False,
                         "reason": "Joint text and crop change to include a natural preceding word; old target was not approved.",
                         "start_basis": "Trial midpoint from preceding aligned word end to new first-word onset, not an established phonetic boundary.",
                         "end_basis": end_basis, "actual_trigger": actual,
                         "exact_audio": exact, "context_audio": expanded, "phrase_audio_approval": None,
                         "visual_approval": None, "training_ready": False})
        document = {"schema": "stage2g-joint-text-and-range-variants-v2", "created_utc": datetime.now(timezone.utc).isoformat(),
                    "status": "new_natural_phrase_variants_listening_pending", "records": rows,
                    "source_sha256": {str(p):sha(p) for p in [SOURCE, BASE / "phrase_review_manifest.json", tokenizer_path, Path(__file__), BASE / "user_evidence/003_phrase_audio_answers.json", BASE / "user_evidence/004_phrase_audio_answers.json"]},
                    "scope": "Separate review proposals. The five-word C002 phrase is explicitly wider than the original 2-4-word candidate pool; no old candidate or label is overwritten.",
                    "originals_preserved": True, "training_ready": False, "new_TAR_reads": 0, "new_neural_calls": 0,
                    "new_downloads": 0, "training_updates": 0}
        target.write_text(json.dumps(document,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        for row in rows:
            print(row["rank"],row["text"],row["start_sample_16k"],row["end_sample_16k"],row["frames_25fps"],row["ctc_min_frames"])
    except (OSError, ValueError, KeyError, TypeError, AssertionError, wave.Error) as exc:
        raise RuntimeError("Natural phrase variant preparation failed") from exc


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
