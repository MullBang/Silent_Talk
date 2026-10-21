"""Prepare seven exact phrase questions from frozen, already available context PCM."""
from __future__ import annotations

import hashlib
import json
import sys
import wave
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parent
PREVIOUS = BASE.parent / "offline_boundary_remaining_step02f_20261013"
SOURCE = BASE.parent / "offline_boundary_review_step02c_20261013/review_manifest_r1.json"


def sha(path: Path) -> str:
    """Return the local file SHA-256."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise RuntimeError(f"Cannot hash {path}") from exc


def load(path: Path) -> dict:
    """Read a source JSON document."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Cannot load {path}") from exc


def write_pcm(source: bytes, start: int, end: int, path: Path) -> dict:
    """Write one exact sample slice, refusing existing evidence replacement."""
    try:
        if path.exists() or not 0 <= start < end <= len(source) // 2:
            raise ValueError(f"Existing output or invalid bounds: {path}")
        pcm = source[start * 2:end * 2]
        with wave.open(str(path), "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(16000)
            writer.writeframes(pcm)
        return {"path": str(path), "sha256": sha(path), "pcm_sha256": hashlib.sha256(pcm).hexdigest(),
                "start_sample_16k": start, "end_sample_16k": end, "samples": end - start,
                "duration_seconds": (end - start) / 16000}
    except (OSError, ValueError, wave.Error) as exc:
        raise RuntimeError(f"Failed PCM slice: {path}") from exc


def main() -> None:
    """Preserve old snapshots and prepare review-only exact/context pairs."""
    try:
        manifest_path = BASE / "phrase_review_manifest.json"
        if manifest_path.exists():
            raise ValueError("A manifest already exists; create a new revision")
        completion = PREVIOUS / "results/completion_manifest.json"
        frozen = load(completion)
        for item in frozen["artifacts"]:
            if sha(Path(item["path"])) != item["sha256"]:
                raise ValueError(f"Changed previous artifact: {item['path']}")
        proposal_path = PREVIOUS / "results/remaining_phrase_proposal.json"
        proposal = load(proposal_path)
        if sha(SOURCE) != proposal["source_review_manifest_sha256"]:
            raise ValueError("Source review manifest hash mismatch")
        parents_path = PREVIOUS / "remaining_parent_manifest.json"
        parents = {r["id"]: r for r in load(parents_path)["records"]}
        source = {r["id"]: r for r in load(SOURCE)["records"]}
        handoff_path = PREVIOUS / "results/parent_audio_review_handoff.json"
        confirmed = {r["parent_id"]: r for r in load(handoff_path)["records"]}
        for folder in ("media", "results", "user_evidence"):
            (BASE / folder).mkdir(parents=True, exist_ok=True)
        records = []
        for row in proposal["records"]:
            identifier = row["parent_id"]
            parent, canonical = parents[identifier], source[identifier]
            segment = next(s for s in canonical["segments"] if s["segment_id"] == row["segment_id"])
            for key in ("text", "start_sample_16k", "end_sample_16k", "frames_25fps", "ctc_min_frames"):
                if segment[key] != row[key]:
                    raise ValueError(f"Proposal mismatch: {identifier}/{key}")
            if not (confirmed[identifier]["parent_transcript_confirmed"] and confirmed[identifier]["parent_audio_boundary_confirmed"]):
                raise ValueError("Parent audio not confirmed")
            context = Path(parent["context_audio_path"])
            if sha(context) != parent["context_audio_sha256"]:
                raise ValueError(f"Context hash mismatch: {identifier}")
            with wave.open(str(context), "rb") as reader:
                if (reader.getframerate(), reader.getnchannels(), reader.getsampwidth()) != (16000, 1, 2):
                    raise ValueError("Unexpected PCM format")
                pcm = reader.readframes(reader.getnframes())
            start, end = row["start_sample_16k"], row["end_sample_16k"]
            parent_start, parent_end = parent["start_sample_16k"], parent["end_sample_16k"]
            if not parent_start <= start < end <= parent_end:
                raise ValueError("Phrase leaves its confirmed parent")
            exact = write_pcm(pcm, start, end, BASE / "media" / f"{row['rank']:02d}_phrase_exact.wav")
            expanded = write_pcm(pcm, max(parent_start, start-3200), min(parent_end, end+3200),
                                 BASE / "media" / f"{row['rank']:02d}_phrase_context.wav")
            records.append({**row, "parent_text": canonical["text"],
                            "parent_start_sample_16k": parent_start, "parent_end_sample_16k": parent_end,
                            "parent_audio_confirmed": True, "parent_confirmation_evidence": confirmed[identifier]["evidence_path"],
                            "source_context_path": str(context), "source_context_sha256": sha(context),
                            "exact_audio": exact, "context_audio": expanded,
                            "context_policy": "Up to 0.20 seconds each side, clamped to confirmed parent; may contain adjacent words of the same parent.",
                            "phrase_audio_approval": None, "visual_approval": None,
                            "parent_human_approval_transferred": False, "training_ready": False})
        manifest = {"schema": "offline-stage2g-seven-phrase-review-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
                    "status": "exact_phrase_media_prepared_human_review_pending", "records": records,
                    "held": proposal["held"], "source_sha256": {str(p): sha(p) for p in [completion, proposal_path, SOURCE, parents_path, handoff_path, Path(__file__)]},
                    "previous_frozen_artifacts_verified": len(frozen["artifacts"]), "question_wavs": len(records)*2,
                    "new_TAR_reads": 0, "new_downloads": 0, "neural_calls": 0, "training_updates": 0,
                    "training_ready": False, "stage2_complete": False,
                    "scope": "Parent confirmation remains valid; exact short phrase approval requires a new actual reply. Context playback does not approve the exact range."}
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"phrases": len(records), "wav_pairs": len(records), "frozen_preserved": len(frozen["artifacts"]), "training_ready": False}))
    except (OSError, ValueError, KeyError, TypeError, StopIteration, wave.Error) as exc:
        raise RuntimeError("Phrase media preparation failed") from exc


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
