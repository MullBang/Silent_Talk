"""Preserve previous candidates and bind actual phrase feedback to new variants."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

BASE = Path(__file__).resolve().parent
PARENT = BASE.parent / "offline_boundary_review_step02b_20261013"
ROOT = BASE.parent.parent
sys.path.insert(0, str(PARENT))
from prepare_review import digest, read, write
from validate_review import manifest_index


def geometry(segment: dict[str, Any]) -> None:
    """Compute visual CTC feasibility using the unchanged positional-jamo codec."""
    try:
        tokenizer_path = ROOT / "backend/models/tokenizer.py"
        spec = importlib.util.spec_from_file_location("review_tokenizer", tokenizer_path)
        if spec is None or spec.loader is None:
            raise ValueError("Cannot load tokenizer")
        codec = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(codec)
        left, right = segment["start_sample_16k"], segment["end_sample_16k"]
        segment["start_in_context"], segment["end_in_context"] = left / 16000, right / 16000
        segment["frames_25fps"] = round(Fraction((right - left) * 25, 16000))
        segment["ctc_min_frames"] = codec.ctc_min_input_len(codec.encode_text(segment["text"]))
        segment["ctc_feasible"] = segment["frames_25fps"] >= segment["ctc_min_frames"]
    except (OSError, TypeError, KeyError, ValueError) as exc:
        raise ValueError("Invalid candidate geometry") from exc


def main() -> None:
    """Generate review artifacts only; never turn chat audio judgments into training."""
    try:
        for item in read(PARENT / "results/completion_manifest.json")["artifacts"]:
            if digest(Path(item["path"])) != item["sha256"]:
                raise ValueError(f"Previous artifact changed: {item['path']}")
        manifest = read(PARENT / "review_manifest_r1.json")
        index = {row["id"]: row for row in manifest["records"]}
        for row in manifest["records"]:
            for segment in row["segments"]:
                evidence = segment.get("external_audio_evidence")
                if evidence:
                    evidence["evidence_path"] = str(PARENT / evidence["evidence_path"])
        trial = read(BASE / "results/repair04_proposal.json")
        row = index[trial["id"]]
        original = next(s for s in row["segments"] if s["segment_id"] == trial["base_segment_id"])
        short_variant = copy.deepcopy(original)
        short_variant.update(segment_id=original["segment_id"] + ":end_plus80ms",
                             end_sample_16k=trial["proposed_audio"]["end_sample_16k"],
                             recommended=False, recommendation_rank=None,
                             boundary_revision="끝 +0.08초 비교 후보. 사용자 청취에서 여전히 ‘서’가 잘린 것으로 보고되어 보류.")
        geometry(short_variant)
        row["segments"].append(short_variant)
        variants = read(BASE / "results/sentence_tail_proposals.json")["records"]
        for proposal in variants:
            row = index[proposal["id"]]
            original = next(s for s in row["segments"] if s["segment_id"] == proposal["base_segment_id"])
            original["recommended"] = False
            original["recommendation_rank"] = None
            candidate = copy.deepcopy(original)
            for field in ("segment_id", "text", "first_word_index", "word_count", "start_sample_16k", "end_sample_16k"):
                candidate[field] = proposal[field]
            candidate.update(recommended=True, recommendation_rank=proposal["rank"],
                             boundary_revision="원래 문장 끝까지 포함한 별도 후보. ④는 ‘많이 해서 잠이 부족해’로 구간과 정답을 함께 변경함.")
            geometry(candidate)
            if not row["sentence_start_sample_16k"] <= candidate["start_sample_16k"] < candidate["end_sample_16k"] <= row["sentence_end_sample_16k"]:
                raise ValueError("Candidate outside parent sentence")
            row["segments"].append(candidate)
        for evidence_path in sorted((BASE / "user_evidence").glob("*.json")):
            for evidence in read(evidence_path)["records"]:
                row = index[evidence["id"]]
                segment = next(s for s in row["segments"] if s["segment_id"] == evidence["segment_id"])
                if segment["text"] != evidence["text"]:
                    raise ValueError("Evidence transcript mismatch")
                audio = evidence["exact_audio"]
                if segment["start_sample_16k"] != audio["start_sample_16k"] or segment["end_sample_16k"] != audio["end_sample_16k"] or digest(Path(audio["path"])) != audio["sha256"]:
                    raise ValueError("Evidence audio mismatch")
                segment["external_audio_evidence"] = {
                    "decision": evidence["audio_boundary_decision"], "answer_verbatim": evidence["answer_verbatim"],
                    "evidence_path": str(evidence_path), "scope": "이 구절의 음성 범위만 확인; 문장 전체와 입술 동작 경계는 미확인"}
        sources = manifest["sources"]
        for source in [PARENT / "review_manifest_r1.json", PARENT / "results/completion_manifest.json", PARENT / "review_template.html",
                       ROOT / "backend/models/tokenizer.py", Path(__file__), BASE / "results/repair04_proposal.json",
                       BASE / "results/sentence_tail_proposals.json", *sorted((BASE / "user_evidence").glob("*.json")),
                       *sorted((BASE / "questions").glob("*.wav"))]:
            sources[str(source)] = digest(source)
        for path, expected in sources.items():
            if digest(Path(path)) != expected:
                raise ValueError(f"Source changed: {path}")
        manifest["revision_note"] = "Preserve410 original+prior C312 repair. Add failed C004+80ms trial and3 sentence-tail variants. All8 latest recommended phrase audio ranges confirmed; no full-parent or visual approval."
        manifest_index(manifest)
        selected = [dict(parent_id=row["id"], speaker_id=row["speaker_id"], **segment) for row in manifest["records"] for segment in row["segments"] if segment["recommended"]]
        selected.sort(key=lambda segment: segment["recommendation_rank"])
        if len(selected) != 8 or any(s["external_audio_evidence"]["decision"] != "acceptable" for s in selected):
            raise ValueError("Eight confirmed audio targets expected")
        write(BASE / "review_manifest_r1.json", manifest)
        write(BASE / "results/current_audio_confirmed_phrases.json", {"schema": "offline-boundary-audio-handoff-v1", "review_manifest_sha256": digest(BASE / "review_manifest_r1.json"), "records": selected, "training_ready": False})
        template = (PARENT / "review_template.html").read_text(encoding="utf-8")
        template = template.replace("segment.segment_id==='phrase:3:4:end_plus120ms'", "Boolean(segment.boundary_revision)", 1)
        template = template.replace("끝 경계 수정 후보 ${repairCount}개", "추가 검수 후보 ${repairCount}개")
        template = template.replace("  if(segment.segment_id==='phrase:3:4:end_plus120ms')panel.append(el('p','끝을 0.12초 늘린 별도 수정 후보입니다. “후보 그대로 재생”은 위에 표시한 수정 구간만 재생하며, 추가 문맥을 붙이지 않습니다. 원래 후보는 별도로 보존됩니다.','muted'));", "  if(segment.boundary_revision)panel.append(el('p',segment.boundary_revision,'muted'));")
        template = template.replace("    if(evidence.evidence_path)banner.append(el('p','근거 기록: '+evidence.evidence_path,'muted'));", "")
        write(BASE / "results/template_derivation.json", {"source": str(PARENT / "review_template.html"), "source_sha256": digest(PARENT / "review_template.html"), "changes": ["count all explicit variants", "show variant-specific scope", "hide internal evidence paths from human review UI"]})
        (BASE / "review_template.html").write_text(template, encoding="utf-8")
        payload = copy.deepcopy(manifest)
        payload["review_manifest_sha256"] = digest(BASE / "review_manifest_r1.json")
        html = template.replace("__REVIEW_DATA__", json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c"))
        (BASE / "index.html").write_text(html, encoding="utf-8")
        write(BASE / "results/build_receipt.json", {"manifest_sha256": payload["review_manifest_sha256"], "template_sha256": digest(BASE / "review_template.html"), "html_sha256": digest(BASE / "index.html"), "agent_created_ui_decisions": 0, "training_ready": False})
        print(json.dumps({"candidate_count": sum(len(row["segments"]) for row in manifest["records"]), "confirmed_current_phrase_audio": len(selected), "training_ready": False}))
    except (OSError, ValueError, KeyError, StopIteration, TypeError) as exc:
        raise RuntimeError("Continuation review build failed") from exc


if __name__ == "__main__":
    main()
