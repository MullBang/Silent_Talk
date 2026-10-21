"""Audit appended boundary evidence and the single audio repair without producer imports."""
from __future__ import annotations

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


def read(path: Path) -> Any:
    """Read a recorded JSON artifact."""
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Cannot read {path}") from exc


def digest(path: Path) -> str:
    """Return an exact byte hash."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise RuntimeError(f"Cannot hash {path}") from exc


def require(value: bool, reason: str) -> None:
    """Reject any failed invariant even under optimized Python."""
    try:
        if not value:
            raise ValueError(reason)
    except ValueError as exc:
        raise RuntimeError(f"Repair audit failed: {reason}") from exc


def audio(path: Path) -> bytes:
    """Read only validated 16 kHz mono integer PCM."""
    try:
        with wave.open(str(path), "rb") as handle:
            require((handle.getframerate(), handle.getnchannels(), handle.getsampwidth(), handle.getcomptype()) == (16000, 1, 2, "NONE"), "PCM format")
            raw = handle.readframes(handle.getnframes())
            require(len(raw) == handle.getnframes() * 2, "Complete PCM stream")
            return raw
    except (OSError, wave.Error) as exc:
        raise RuntimeError(f"Cannot inspect PCM {path}") from exc


def main() -> None:
    """Check the immutable base, narrow evidence scopes, and appended variant."""
    try:
        paths = {name: BASE / name for name in (
            "review_manifest.json", "review_manifest_r1.json", "results/listening_batch.json",
            "results/independent_audit.json", "results/boundary_repair_proposal.json",
            "user_evidence/001_phrase_audio_answers.json", "user_evidence/002_repaired_audio_answer.json")}
        base, revised = read(paths["review_manifest.json"]), read(paths["review_manifest_r1.json"])
        batch = read(paths["results/listening_batch.json"])
        previous = read(paths["results/independent_audit.json"])
        proposal = read(paths["results/boundary_repair_proposal.json"])
        evidence1 = read(paths["user_evidence/001_phrase_audio_answers.json"])
        evidence2 = read(paths["user_evidence/002_repaired_audio_answer.json"])
        require(previous["review_manifest_sha256"] == digest(paths["review_manifest.json"]), "Previously audited manifest unchanged")
        require(previous["listening_batch_sha256"] == digest(paths["results/listening_batch.json"]), "Previously audited audio batch unchanged")
        for path, expected in revised["sources"].items():
            require(digest(Path(path)) == expected, f"Source hash {path}")
        require(all(revised["sources"][key] == value for key, value in base["sources"].items()), "All original source links preserved")
        require(revised["automatic_approval"] is False and revised["training_ready"] is False and revised["default_decision"] is None, "No automatic promotion")
        require(evidence1["review_manifest_sha256"] == digest(paths["review_manifest.json"]), "Initial evidence manifest binding")
        expected_answers = ["온전히 들림", "온전히 들림", '맨 끝에 "어" 부분이 살짝 잘린 것 같음']
        changed: dict[tuple[str, str], dict[str, Any]] = {}
        preserved = 0
        appended: list[tuple[dict[str, Any], dict[str, Any]]] = []
        require(len(base["records"]) == len(revised["records"]) == 16, "All sixteen parent rows")
        for before, after in zip(base["records"], revised["records"], strict=True):
            require({k: v for k, v in before.items() if k != "segments"} == {k: v for k, v in after.items() if k != "segments"}, "Parent metadata and time extent unchanged")
            require(len({s["segment_id"] for s in after["segments"]}) == len(after["segments"]), "Unique revision IDs")
            require(len(after["segments"]) >= len(before["segments"]), "No source proposal dropped")
            for original, current in zip(before["segments"], after["segments"]):
                ignored = {"recommended", "recommendation_rank", "external_audio_evidence"}
                require({k: v for k, v in original.items() if k not in ignored} == {k: v for k, v in current.items() if k not in ignored}, "Original proposal text, samples, geometry and flags unchanged")
                if current != original:
                    changed[(after["id"], current["segment_id"])] = current
                preserved += 1
            appended.extend((after, item) for item in after["segments"][len(before["segments"]):])
        require(preserved == 410 and len(changed) == 3 and len(appended) == 1, "Exactly 410 preserved, three evidence annotations, one appended variant")
        require(len(evidence1["records"]) == 3, "Three initial actual replies")
        for i, item in enumerate(evidence1["records"]):
            question = batch["records"][i]
            require(json.loads(item["question_item_id"]) == ["request_user_input_async", "call_8e462bd93d1d4147b4465c019e0970d2", i], "Initial question call binding")
            require(item["answer_verbatim"] == expected_answers[i], "Actual answer text supplied by parent task")
            require(item["id"] == question["id"] and item["segment_id"] == question["segment"]["segment_id"] and item["text"] == question["segment"]["text"], "Actual reply target binding")
            require(item["exact_audio"] == question["exact_audio"], "Actual reply played audio binding")
            require(item["audio_boundary_decision"] == ("acceptable" if i < 2 else "cut_suspected"), "Narrow audio interpretation")
            require(item["segment_transcript_decision"] == ("match" if i < 2 else None), "Transcript uncertainty retained")
            current = changed[(item["id"], item["segment_id"])]
            linked = current["external_audio_evidence"]
            require(linked["answer_verbatim"] == item["answer_verbatim"] and linked["decision"] == item["audio_boundary_decision"], "Displayed evidence agrees")
            require(linked["evidence_path"] == "user_evidence/001_phrase_audio_answers.json", "Displayed evidence source")
        parent, repair = appended[0]
        require(parent["id"] == proposal["id"] == evidence2["id"] == "lip_J_1_F_03_C312_A_010_010", "Repair parent")
        require(repair["segment_id"] == "phrase:3:4:end_plus120ms" and proposal["base_segment_id"] == evidence2["base_segment_id"] == "phrase:3:4", "New variant identity")
        require(repair["text"] == proposal["text"] == evidence2["text"] == "정말 잘 돼 있어", "Repair target unchanged")
        require(proposal["cause_evidence_sha256"] == digest(paths["user_evidence/001_phrase_audio_answers.json"]), "Repair cause binding")
        require(proposal["original_exact_audio"] == batch["records"][2]["exact_audio"], "Original failed audio preserved")
        require(proposal["proposed_exact_audio"] == evidence2["exact_audio"], "Repaired audio linked to actual answer")
        require(json.loads(evidence2["question_item_id"]) == ["request_user_input_async", "call_e21a71987b8c4f55bcc517edbacbf90a", 0], "Repair question binding")
        require(evidence2["answer_verbatim"] == "이제 온전히 들림" and evidence2["audio_boundary_decision"] == "acceptable" and evidence2["segment_transcript_decision"] == "match", "Repair actual answer and scope")
        require(repair["external_audio_evidence"]["answer_verbatim"] == evidence2["answer_verbatim"] and repair["external_audio_evidence"]["decision"] == "acceptable" and repair["external_audio_evidence"]["evidence_path"] == "user_evidence/002_repaired_audio_answer.json", "Repaired UI evidence")
        for item in [*evidence1["records"], evidence2]:
            require(item["visual_context_decision"] is None and item["full_parent_transcript_approved"] is False and item["training_ready"] is False, "No full-parent or visual promotion")
        require(proposal["human_response"] is None and proposal["visual_context_approved"] is False and proposal["training_ready"] is False, "Historical repair proposal stays historical")
        old_audio, new_audio = proposal["original_exact_audio"], proposal["proposed_exact_audio"]
        require(new_audio["start_sample_16k"] == old_audio["start_sample_16k"] == repair["start_sample_16k"] == 30920, "Repair start unchanged")
        require(new_audio["end_sample_16k"] == repair["end_sample_16k"] == 50120 and new_audio["end_sample_16k"] - old_audio["end_sample_16k"] == 1920, "Exactly 120ms appended")
        require(parent["sentence_start_sample_16k"] <= 30920 < 50120 <= parent["sentence_end_sample_16k"] == 53154, "Repaired interval remains within parent")
        require(repair["start_in_context"] == 30920/16000 and repair["end_in_context"] == 50120/16000, "Repaired time display")
        require(repair["frames_25fps"] == round(Fraction((50120-30920)*25,16000)) == 30 and repair["ctc_min_frames"] == 19 and repair["ctc_feasible"] is True, "Repaired CTC geometry")
        old_record = next(r for r in read(OLD / "results/media_manifest.json")["records"] if r["id"] == parent["id"])
        source = audio(Path(old_record["context_audio"]["path"]))
        raw = audio(Path(new_audio["path"]))
        require(raw == source[2*30920:2*50120] and len(raw) == new_audio["samples"]*2 == 38400, "Repair exact source PCM bytes")
        require(digest(Path(new_audio["path"])) == new_audio["sha256"] and hashlib.sha256(raw).hexdigest() == new_audio["pcm_sha256"] and new_audio["review_only"] is True, "Repair WAV and PCM hash binding")
        report = {
            "schema": "offline-boundary-repair-independent-audit-v1", "status": "passed_preservation_evidence_scope_and_repair_pcm_audit",
            "created_utc": datetime.now(timezone.utc).isoformat(), "auditor_sha256": digest(Path(__file__)),
            "inputs": {str(path): digest(path) for path in paths.values()},
            "preserved_original_candidates": 410, "appended_repair_candidates": 1, "current_candidates": 411,
            "source_hashes_verified": len(revised["sources"]), "human_phrase_audio_acceptable": 3,
            "original_phrase_cut_suspected": 1, "new_phrase_end_extension_samples_16k": 1920,
            "repaired_frames_25fps": 30, "repaired_ctc_min_frames": 19,
            "full_parent_transcript_approvals": 0, "human_visual_boundary_approvals": 0,
            "neural_calls": 0, "training_updates": 0, "training_ready": False,
            "provenance_note": "Question IDs and answer contents were supplied as actual current-chat replies by the parent task. JSON reviewer/source declarations are not independently treated as human authentication.",
            "limits": ["120ms is the tested repair increment, not a measured ground-truth boundary error.",
                       "Three phrase audio judgments are not full-parent or visual-context approvals.",
                       "The historical proposal remains unedited; the newer separately linked evidence records the repair answer.",
                       "This artifact audit does not test UI rendering or browser playback synchronization."]}
        (BASE / "results/repair_independent_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        print(json.dumps({k:v for k,v in report.items() if k != "inputs"}, ensure_ascii=False))
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        raise RuntimeError("Independent repair audit did not pass") from exc


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
