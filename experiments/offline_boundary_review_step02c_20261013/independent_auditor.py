"""Independently check retained candidates, actual reply bindings and PCM crops.

Only stdlib is used; no producer, tokenizer, model or UI implementation imports.
This is a data/provenance audit, not listening or visual-boundary approval.
"""
from __future__ import annotations

import hashlib
import json
import sys
import wave
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent.parent
PARENT = BASE.parent / "offline_boundary_review_step02b_20261013"
MEDIA = BASE.parent / "offline_alignment_step02_20261012/media"


def sha(path: Path) -> str:
    """Hash one file without loading production modules."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise RuntimeError(f"Cannot hash {path}") from exc


def read(path: Path) -> dict[str, Any]:
    """Load one UTF-8 JSON artifact."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Cannot read {path}") from exc


def check(condition: bool, label: str) -> None:
    """Fail closed with a reproducible audit description."""
    try:
        if not condition:
            raise ValueError(label)
    except ValueError as exc:
        raise RuntimeError(f"Audit check failed: {label}") from exc


def frames(samples: int) -> int:
    """Compute exact 25 Hz ties-to-even rounding with integer divmod."""
    try:
        q, r = divmod(samples * 25, 16000)
        return q + int(2 * r > 16000 or (2 * r == 16000 and q % 2 == 1))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Invalid sample duration") from exc


def minimum_ctc_frames(text: str) -> int:
    """Independently decompose Hangul into the existing 69-position vocabulary."""
    try:
        ids: list[int] = []
        for char in text:
            if char == " ":
                ids.append(68)
            else:
                code = ord(char) - 0xAC00
                check(0 <= code < 11172, "non-Hangul target")
                lead, remainder = divmod(code, 588)
                vowel, tail = divmod(remainder, 28)
                ids.extend((1 + lead, 20 + vowel))
                if tail:
                    ids.append(40 + tail)
        return len(ids) + sum(a == b for a, b in zip(ids, ids[1:]))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Cannot compute independent CTC requirement") from exc


def pcm(path: Path) -> bytes:
    """Read uncompressed mono PCM16 at 16 kHz, rejecting other formats."""
    try:
        with wave.open(str(path), "rb") as handle:
            check((handle.getframerate(), handle.getnchannels(), handle.getsampwidth(), handle.getcomptype()) == (16000, 1, 2, "NONE"), f"PCM format {path}")
            result = handle.readframes(handle.getnframes())
            check(len(result) == handle.getnframes() * 2, f"full PCM length {path}")
            return result
    except (OSError, wave.Error) as exc:
        raise RuntimeError(f"Cannot read PCM {path}") from exc


def run() -> dict[str, Any]:
    """Audit frozen sources, all geometry, all actual replies and latest handoff."""
    try:
        old = read(PARENT / "review_manifest_r1.json")
        current = read(BASE / "review_manifest_r1.json")
        handoff = read(BASE / "results/current_audio_confirmed_phrases.json")
        completion = read(PARENT / "results/completion_manifest.json")
        for item in completion["artifacts"]:
            check(sha(Path(item["path"])) == item["sha256"], f"frozen parent {item['path']}")
        check(len(completion["artifacts"]) == 41, "41 frozen parent artifacts")
        for path, digest in current["sources"].items():
            check(sha(Path(path)) == digest, f"source hash {path}")
        check(current["training_ready"] is False and current["automatic_approval"] is False and current["default_decision"] is None, "manifest no automatic approval")
        check(handoff["training_ready"] is False, "handoff no training promotion")
        check(handoff["review_manifest_sha256"] == sha(BASE / "review_manifest_r1.json"), "handoff manifest binding")

        old_by_id = {row["id"]: row for row in old["records"]}
        by_id = {row["id"]: row for row in current["records"]}
        check(list(old_by_id) == list(by_id) and len(by_id) == 16, "16 preserved parent IDs and order")
        old_segments: dict[tuple[str, str], dict[str, Any]] = {}
        segments: dict[tuple[str, str], dict[str, Any]] = {}
        mutable = {"external_audio_evidence", "recommended", "recommendation_rank"}
        selected: list[dict[str, Any]] = []
        for parent_id, row in by_id.items():
            prior = old_by_id[parent_id]
            check({k: v for k, v in row.items() if k != "segments"} == {k: v for k, v in prior.items() if k != "segments"}, f"parent metadata {parent_id}")
            mapping = {s["segment_id"]: s for s in row["segments"]}
            check(len(mapping) == len(row["segments"]), f"unique segment IDs {parent_id}")
            for source in prior["segments"]:
                target = mapping[source["segment_id"]]
                check({k: v for k, v in source.items() if k not in mutable} == {k: v for k, v in target.items() if k not in mutable}, f"retained geometry {parent_id}/{source['segment_id']}")
                old_segments[(parent_id, source["segment_id"])] = source
            for segment in row["segments"]:
                key = (parent_id, segment["segment_id"])
                segments[key] = segment
                start, end = segment["start_sample_16k"], segment["end_sample_16k"]
                check(type(start) is int and type(end) is int and 0 <= start < end <= row["context_samples_16k"], f"integer range {key}")
                check(segment["start_in_context"] == start / 16000 and segment["end_in_context"] == end / 16000, f"seconds {key}")
                count = frames(end - start)
                minimum = minimum_ctc_frames(segment["text"])
                check(segment["frames_25fps"] == count and segment["ctc_min_frames"] == minimum and segment["ctc_feasible"] == (count >= minimum), f"CTC geometry {key}")
                first, length = segment["first_word_index"], segment["word_count"]
                check(segment["text"] == " ".join(row["text"].split()[first:first + length]), f"contiguous canonical words {key}")
                if segment["recommended"]:
                    selected.append(dict(parent_id=parent_id, speaker_id=row["speaker_id"], **segment))
        check(len(old_segments) == 411 and len(segments) == 415, "411 preserved and 415 total candidates")
        additions = set(segments) - set(old_segments)
        check(additions == {
            ("lip_J_1_F_02_C004_A_011_014", "phrase:5:3:end_plus80ms"),
            ("lip_J_1_F_02_C004_A_011_014", "phrase:6:4:sentence_tail"),
            ("lip_J_1_F_03_C293_A_011_018", "phrase:3:4:sentence_tail"),
            ("lip_J_1_F_03_C297_A_006_006", "phrase:4:3:sentence_tail")}, "four explicit new variants")

        expected = {
            ("call_8e462bd93d1d4147b4465c019e0970d2", 0): ("온전히 들림", "acceptable"),
            ("call_8e462bd93d1d4147b4465c019e0970d2", 1): ("온전히 들림", "acceptable"),
            ("call_8e462bd93d1d4147b4465c019e0970d2", 2): ('맨 끝에 "어" 부분이 살짝 잘린 것 같음', "cut_suspected"),
            ("call_e21a71987b8c4f55bcc517edbacbf90a", 0): ("이제 온전히 들림", "acceptable"),
            ("call_104d766a989b42c1b45268bf23e1ca19", 0): ('"해"까지만 들리고 뒤에 "서"는 안들림', "cut_suspected"),
            ("call_104d766a989b42c1b45268bf23e1ca19", 1): ("온전히 들림", "acceptable"),
            ("call_104d766a989b42c1b45268bf23e1ca19", 2): ("온전히 들림", "acceptable"),
            ("call_b2664f73217745718e7b7374a911e561", 0): ("아직 ‘서’가 잘림", "cut_suspected"),
            ("call_b2664f73217745718e7b7374a911e561", 1): ('"어"부분이 잘림', "cut_suspected"),
            ("call_b2664f73217745718e7b7374a911e561", 2): ('뒤에 "혀"부분이 잘려있음', "cut_suspected"),
            ("call_cfecd7ed27bb4e9980a5e7723f1c1f25", 0): ("온전히 들림", "acceptable"),
            ("call_cfecd7ed27bb4e9980a5e7723f1c1f25", 1): ("온전히 들림", "acceptable"),
            ("call_cfecd7ed27bb4e9980a5e7723f1c1f25", 2): ("온전히 들림", "acceptable"),
        }
        expected_targets = {
            "call_8e462bd93d1d4147b4465c019e0970d2": [
                ("lip_J_1_F_02_C002_A_009_045", "phrase:6:4", "못 꿀 것 같아"),
                ("lip_J_1_F_02_C017_A_008_004", "phrase:7:4", "못살게 구는 것 같아"),
                ("lip_J_1_F_03_C312_A_010_010", "phrase:3:4", "정말 잘 돼 있어")],
            "call_e21a71987b8c4f55bcc517edbacbf90a": [
                ("lip_J_1_F_03_C312_A_010_010", "phrase:3:4:end_plus120ms", "정말 잘 돼 있어")],
            "call_104d766a989b42c1b45268bf23e1ca19": [
                ("lip_J_1_F_02_C004_A_011_014", "phrase:5:3", "야근을 많이 해서"),
                ("lip_J_1_F_02_C032_A_012_019", "phrase:12:3", "보는 것 같아"),
                ("lip_J_1_F_03_C282_A_011_039", "phrase:4:4", "먹게 되는 것 같아")],
            "call_b2664f73217745718e7b7374a911e561": [
                ("lip_J_1_F_02_C004_A_011_014", "phrase:5:3:end_plus80ms", "야근을 많이 해서"),
                ("lip_J_1_F_03_C293_A_011_018", "phrase:3:4", "정말 잘 돼 있어"),
                ("lip_J_1_F_03_C297_A_006_006", "phrase:4:3", "너무 많이 막혀")],
            "call_cfecd7ed27bb4e9980a5e7723f1c1f25": [
                ("lip_J_1_F_02_C004_A_011_014", "phrase:6:4:sentence_tail", "많이 해서 잠이 부족해"),
                ("lip_J_1_F_03_C293_A_011_018", "phrase:3:4:sentence_tail", "정말 잘 돼 있어"),
                ("lip_J_1_F_03_C297_A_006_006", "phrase:4:3:sentence_tail", "너무 많이 막혀")],
        }
        evidence_paths = sorted((PARENT / "user_evidence").glob("*.json")) + sorted((BASE / "user_evidence").glob("*.json"))
        seen: set[tuple[str, int]] = set()
        evidence_count = 0
        audio_hashes: dict[str, str] = {}
        evidence_by_segment: dict[tuple[str, str], dict[str, Any]] = {}
        for evidence_path in evidence_paths:
            data = read(evidence_path)
            for evidence in data.get("records", [data]):
                source, call, index = json.loads(evidence["question_item_id"])
                question = (call, index)
                check(source == "request_user_input_async" and question not in seen, "unique actual question binding")
                seen.add(question)
                check((evidence["answer_verbatim"], evidence["audio_boundary_decision"]) == expected[question], f"actual answer {question}")
                check(evidence["visual_context_decision"] is None and evidence["full_parent_transcript_approved"] is False and evidence["training_ready"] is False, f"limited approval scope {question}")
                segment_id = evidence.get("segment_id", "phrase:3:4:end_plus120ms")
                key = (evidence["id"], segment_id)
                check((*key, evidence["text"]) == expected_targets[call][index], f"actual question target identity {question}")
                check(evidence["segment_transcript_decision"] == ("match" if expected[question][1] == "acceptable" else None), f"only accepted target wording confirmed {question}")
                segment = segments[key]
                check(evidence["text"] == segment["text"], f"evidence target {question}")
                audio = evidence["exact_audio"]
                path = Path(audio["path"])
                audio_hashes[str(path)] = sha(path)
                check(audio_hashes[str(path)] == audio["sha256"], f"audio SHA {question}")
                waveform = pcm(path)
                context = pcm(MEDIA / evidence["id"] / "context.wav")
                start, end = audio["start_sample_16k"], audio["end_sample_16k"]
                check(start == segment["start_sample_16k"] and end == segment["end_sample_16k"], f"played geometry {question}")
                check(len(waveform) == audio["samples"] * 2 == (end - start) * 2 and waveform == context[start * 2:end * 2], f"exact parent PCM slice {question}")
                check(hashlib.sha256(waveform).hexdigest() == audio["pcm_sha256"], f"PCM hash {question}")
                attached = segment["external_audio_evidence"]
                check(Path(attached["evidence_path"]).resolve() == evidence_path.resolve(), f"attached evidence file {question}")
                check(attached["decision"] == evidence["audio_boundary_decision"] and attached["answer_verbatim"] == evidence["answer_verbatim"], f"attached answer {question}")
                evidence_by_segment[key] = evidence
                evidence_count += 1
        check(seen == set(expected) and evidence_count == 13, "13 actual replies including prior4 and new9; duplicate freeform not recounted")
        selected.sort(key=lambda s: s["recommendation_rank"])
        check(selected == handoff["records"] and len(selected) == 8, "latest eight handoff exact")
        check([s["recommendation_rank"] for s in selected] == list(range(1, 9)), "rank uniqueness")
        check(len({s["speaker_id"] for s in selected}) == 8 and len({s["text"] for s in selected}) == 7, "8 speakers and 7 distinct texts")
        check(all(s["external_audio_evidence"]["decision"] == "acceptable" for s in selected), "latest eight audio confirmations")
        check(selected[2]["text"] == selected[6]["text"] == "정말 잘 돼 있어", "C293/C312 shared text group")
        check(selected[3]["text"] == "많이 해서 잠이 부족해" and selected[3]["first_word_index"] == 6, "rank4 genuinely different target and start")
        rank4old = segments[(selected[3]["parent_id"], "phrase:5:3")]
        rank4trial = segments[(selected[3]["parent_id"], "phrase:5:3:end_plus80ms")]
        check(rank4old["text"] == rank4trial["text"] == "야근을 많이 해서" and rank4trial["end_sample_16k"] - rank4old["end_sample_16k"] == 1280, "failed +80ms trial preserved")
        check(rank4old["start_sample_16k"] != selected[3]["start_sample_16k"], "rank4 start changed")
        for rank in (4, 7, 8):
            target = selected[rank - 1]
            check(target["end_sample_16k"] == by_id[target["parent_id"]]["sentence_end_sample_16k"], f"parent end tail {rank}")
        original = read(PARENT / "results/listening_batch.json")["records"]
        decisions = Counter(evidence_by_segment[(r["id"], r["segment"]["segment_id"])]["audio_boundary_decision"] for r in original)
        check(decisions == {"acceptable": 4, "cut_suspected": 4}, "original8 descriptive count")
        receipt = read(BASE / "results/build_receipt.json")
        check(receipt["manifest_sha256"] == sha(BASE / "review_manifest_r1.json") and receipt["template_sha256"] == sha(BASE / "review_template.html") and receipt["html_sha256"] == sha(BASE / "index.html"), "UI build binding")
        check(receipt["agent_created_ui_decisions"] == 0 and receipt["training_ready"] is False, "UI no agent approval")
        artifact_paths = [BASE / "review_manifest_r1.json", BASE / "results/current_audio_confirmed_phrases.json", BASE / "results/build_receipt.json", BASE / "review_template.html", BASE / "index.html", Path(__file__), *evidence_paths]
        return {
            "schema": "offline-boundary-step02c-independent-audit-v1",
            "status": "passed_independent_geometry_pcm_and_actual_reply_audit",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "producer_imports": 0,
            "frozen_parent_artifacts_verified": len(completion["artifacts"]),
            "source_hashes_verified": len(current["sources"]),
            "preserved_candidates": len(old_segments),
            "all_candidate_geometries_checked": len(segments),
            "new_variants": len(additions),
            "actual_reply_bindings": evidence_count,
            "exact_pcm_audio_crops_verified": len(audio_hashes),
            "latest_audio_confirmed_phrases": 8,
            "latest_distinct_speakers": 8,
            "latest_distinct_target_texts": 7,
            "original_eight_audio_decisions": dict(decisions),
            "original_eight_count_interpretation": "Descriptive count only for deliberately chosen illustrative phrases, not an alignment error-rate estimate.",
            "changed_target_rank4": True,
            "human_full_parent_transcript_approvals": 0,
            "human_visual_boundary_approvals": 0,
            "training_ready": False,
            "measured_boundary_error_ms": None,
            "limitations": ["No auditor listening or visual-boundary judgment performed.", "Audio confirmations do not imply parent-sentence or visual-boundary approval.", "CTC minimum length is a geometry gate, not trained recognition quality.", "Extension magnitudes are candidate edits, not measured reference boundary error."],
            "audited_artifact_hashes": {str(path): sha(path) for path in artifact_paths},
            "audited_audio_hashes": audio_hashes,
        }
    except (KeyError, TypeError, ValueError, OSError) as exc:
        raise RuntimeError("Independent boundary continuation audit failed") from exc


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        result = run()
        output = BASE / "results/independent_audit.json"
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({k: v for k, v in result.items() if k not in {"audited_artifact_hashes", "audited_audio_hashes"}}, ensure_ascii=False, indent=2))
    except Exception:
        raise
