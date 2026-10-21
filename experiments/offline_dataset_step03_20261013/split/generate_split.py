"""Generate a metadata-only, paired parent/phrase engine-pilot split.

Reads only the JSON allowlist frozen in selection_protocol.json. Never opens
media, archives, arrays, weights, or another experiment's output for writing.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import random
import re
import sys
import unicodedata
from typing import Any

BASE = Path(__file__).resolve().parent


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def text_key(text: str) -> str:
    return "".join(unicodedata.normalize("NFC", text).split())


def recording_key(stem: str) -> str:
    """Preserve filename identity except the explicit camera-view letter."""
    if not re.fullmatch(r"lip_.+_C\d+_[ABC]_\d+", stem):
        raise ValueError(f"Unsupported recording stem: {stem}")
    return re.sub(r"_[ABC]_(\d+)$", r"_\1", stem)


def tags(row: dict[str, Any]) -> set[tuple[str, str]]:
    result = {(key, row[key]) for key in (
        "source_stem", "angle_independent_recording_key", "video_sha256", "label_sha256")}
    result.update(("full_text", value) for value in row["full_text_keys"])
    return result


def components(rows: list[dict[str, Any]]) -> tuple[list[list[str]], list[dict[str, Any]]]:
    """Union all matching identity/text keys; retain the exact grouping reasons."""
    parent = {r["parent_id"]: r["parent_id"] for r in rows}
    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item
    owners: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in rows:
        for tag in tags(row):
            owners[tag].append(row["parent_id"])
    links = []
    for (key, value), ids in sorted(owners.items()):
        ids = sorted(ids)
        if len(ids) > 1:
            links.append({"key": key, "value": value, "parent_ids": ids})
            for item in ids[1:]:
                a, b = find(ids[0]), find(item)
                parent[max(a, b)] = min(a, b)
    groups: dict[str, list[str]] = defaultdict(list)
    for item in sorted(parent):
        groups[find(item)].append(item)
    return sorted(groups.values()), links


def collision_report(rows: list[dict[str, Any]], protected: list[dict[str, Any]]) -> dict[str, Any]:
    by_tag: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in protected:
        for tag in tags(row):
            by_tag[tag].append(row["parent_id"])
    matches = []
    for row in rows:
        for tag in sorted(tags(row)):
            if tag in by_tag:
                matches.append({"parent_id": row["parent_id"], "key": tag[0],
                                "value": tag[1], "protected_ids": sorted(by_tag[tag])})
    return {"protected_ids_count": len(protected), "matches": matches,
            "matching_parent_count": len({r["parent_id"] for r in matches})}


def generate() -> dict[str, Any]:
    protocol_path = BASE / "selection_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["seed"] != 42 or protocol["status"] != "fixed_before_selection_execution":
        raise ValueError("Unrecognized fixed selection protocol")
    docs = {}
    for name, ref in protocol["inputs"].items():
        path = Path(ref["path"])
        if path.suffix != ".json" or sha(path) != ref["sha256"]:
            raise ValueError(f"Metadata input hash/type mismatch: {name}")
        docs[name] = json.loads(path.read_text(encoding="utf-8"))
    # Verify the inherited metadata chain; hashes are read from JSON, no payloads.
    refs_by_path = {str(Path(r["path"]).resolve()): r["sha256"]
                    for r in protocol["inputs"].values()}
    bound_checks = []
    for ref in docs["stage1_completion"]["artifacts"]:
        key = str(Path(ref["path"]).resolve())
        if key in refs_by_path:
            assert refs_by_path[key] == ref["sha256"], key
            bound_checks.append(key)
    for path, expected in docs["stage1_summary"]["source_files"].items():
        key = str(Path(path).resolve())
        assert refs_by_path[key] == expected, key
        bound_checks.append(key)
    for name, other in [("handoff", "historical_handoff"), ("old_data_plan", "old_plan"),
                        ("new_data_plan", "new_plan")]:
        ref = docs["step15_plan"]["source_refs"][name]
        assert protocol["inputs"][other] == ref
        bound_checks.append(ref["path"])
    media = {r["id"]: r for r in docs["media_manifest"]["records"]}
    historical = {r["id"]: r for r in docs["old_plan"]["records"] + docs["new_plan"]["records"]}
    historical_entries = {r["id"]: r for r in docs["historical_handoff"]["records"]}
    assert len(historical) == len(historical_entries) == 632
    for sid, row in historical.items():
        assert digest(row) == historical_entries[sid]["logical_record_sha256"], sid
    source_members = {r["stem"]: r["members"] for r in docs["d_source_plan"]["sources"]}
    source_members.update({r["stem"]: r["members"] for r in docs["new_plan"]["sources"]})
    pool = {r["id"]: r for r in docs["analysis293"]}
    queue = {r["id"]: r for r in docs["queue16"]}
    assert len(pool) == 293 and len(queue) == 16 and set(queue) <= set(pool)
    parent_bounds = {}
    for row in docs["parent8"]["records"]:
        parent_bounds[row["id"]] = (row["sentence_start_sample_16k"], row["sentence_end_sample_16k"], row["text"])
    for row in docs["parent_remaining8"]["records"]:
        parent_bounds[row["id"]] = (row["start_sample_16k"], row["end_sample_16k"], row["text"])
    held = protocol["held_parent"]
    handoff = docs["phrase_handoff"]
    assert {r["parent_id"] for r in handoff["held_short_alignment"]} == {held}
    assert handoff["training_ready"] is False

    def identity(sid: str, extra_texts: list[str] | None = None) -> dict[str, Any]:
        row = historical[sid]
        members = source_members[row["source_stem"]]
        assert members["label"]["sha256"] == row["label_sha256"]
        return {"parent_id": sid, "speaker_id": row["speaker_id"],
                "source_stem": row["source_stem"],
                "angle_independent_recording_key": recording_key(row["source_stem"]),
                "video_sha256": members["video"]["sha256"],
                "label_sha256": members["label"]["sha256"],
                "full_text_keys": sorted({text_key(t) for t in [row["text"]] + (extra_texts or [])})}

    rows = []
    for phrase in sorted(handoff["records"], key=lambda r: r["parent_id"]):
        sid = phrase["parent_id"]
        assert sid in pool and sid in queue and sid != held
        assert phrase["audio_boundary_confirmed"] and phrase["visual_no_obvious_cut_reported"]
        assert phrase["new_original_grid_approved"] is False
        assert historical_entries[sid]["role"] == "old_train"
        record = identity(sid, [phrase["parent_text"], phrase["text"]])
        sm = media[sid]
        assert sm["source_stem"] == phrase["recording_group"] == record["source_stem"]
        for kind in ("video", "label"):
            expected = record[kind + "_sha256"]
            assert sm["source_members"][kind]["sha256"] == expected
            assert queue[sid]["d_source_members"][kind]["sha256"] == expected
            assert pool[sid][kind + "_sha256"] == expected
        ps, pe, parent_text = parent_bounds[sid]
        assert parent_text == phrase["parent_text"]
        fs, fe = phrase["start_sample_16k"], phrase["end_sample_16k"]
        assert 0 <= ps <= fs < fe <= pe <= sm["context_audio"]["samples"]
        record.update({"rank": phrase["rank"], "parent_text": parent_text,
            "historical_parent_text": historical[sid]["text"],
            "historical_role": "old_train", "split": "unassigned",
            "sentence_start_sample_16k": ps, "sentence_end_sample_16k": pe,
            "context_audio_sha256": sm["context_audio"]["sha256"],
            "context_samples_16k": sm["context_audio"]["samples"],
            "context_source_first_sample": sm["context_audio"]["source_first_sample"],
            "context_source_sample_rate": sm["context_audio"]["source_sample_rate"],
            "offset_seconds": sm["offset_seconds"],
            "phrase": {key: phrase[key] for key in (
                "segment_id", "text", "start_sample_16k", "end_sample_16k")},
            "review_scope": {"phrase_audio_confirmed": True,
                "phrase_no_obvious_visual_cut_reported": True,
                "exact_visual_boundary_confirmed": False, "new_original_grid_approved": False},
            "training_ready": False})
        rows.append(record)
    assert len(rows) == 15 and len({r["parent_id"] for r in rows}) == 15
    assert {r["parent_id"] for r in rows} == set(queue) - {held}
    groups, links = components(rows)
    row_by_id = {r["parent_id"]: r for r in rows}
    speakers = {r["speaker_id"] for r in rows}
    assert len(speakers) == 8
    # Enumerate all legal component choices, then make exactly one seeded draw.
    valid = []
    for count in range(1, len(groups) + 1):
        for indices in itertools.combinations(range(len(groups)), count):
            chosen = {sid for i in indices for sid in groups[i]}
            if len(chosen) != 3:
                continue
            if len({row_by_id[sid]["speaker_id"] for sid in chosen}) != 3:
                continue
            if {r["speaker_id"] for r in rows if r["parent_id"] not in chosen} != speakers:
                continue
            valid.append(indices)
    valid.sort()
    assert valid, "No legal 12/3 split; do not relax the protocol silently"
    selected_index = random.Random(42).randrange(len(valid))
    selected = valid[selected_index]
    dev_ids = sorted(sid for i in selected for sid in groups[i])
    train_ids = sorted(set(row_by_id) - set(dev_ids))
    for record in rows:
        record["split"] = "dev" if record["parent_id"] in dev_ids else "train"
    train = [r for r in rows if r["split"] == "train"]
    dev = [r for r in rows if r["split"] == "dev"]
    cross = collision_report(train, dev)
    assert not cross["matches"]

    summary1 = docs["stage1_summary"]
    protection = {}
    for name, key in [("olddev95_and_held27", "protected_historical_ids"),
                      ("prior_probe14", "protected_probe_ids"),
                      ("previous_dev16", "protected_dev16_ids")]:
        protected = [identity(sid) for sid in summary1[key]]
        check = collision_report(rows, protected)
        check["id_overlap"] = sorted(set(row_by_id) & set(summary1[key]))
        assert not check["matches"] and not check["id_overlap"], name
        protection[name] = check
    blocked_targets = set(docs["step15_plan"]["pool_profile"]["blocked_normalized_targets"])
    current_blocked = [(r["parent_id"], key) for r in rows
                       for key in r["full_text_keys"] if key in blocked_targets]
    assert not current_blocked

    clips = []
    for record in rows:
        for role, start, end, text in [
            ("phrase", record["phrase"]["start_sample_16k"], record["phrase"]["end_sample_16k"], record["phrase"]["text"]),
            ("full_sentence", record["sentence_start_sample_16k"], record["sentence_end_sample_16k"], record["parent_text"])]:
            clips.append({"clip_id": record["parent_id"] + "::" + role,
                          "parent_id": record["parent_id"], "role": role,
                          "segment_id": record["phrase"]["segment_id"] if role == "phrase" else None,
                          "split": record["split"], "text": text,
                          "start_sample_16k": start, "end_sample_16k": end,
                          "clock": "existing 16kHz context.wav, immutable [start,end)",
                          "training_ready": False})
    # Expansion is only a metadata possibility, with transitive dev protection.
    expansion_rows = [identity(sid, [pool[sid]["text"]]) for sid in sorted(pool)]
    for index, item in enumerate(expansion_rows):
        if item["parent_id"] in row_by_id:
            expansion_rows[index] = row_by_id[item["parent_id"]]
    expansion_groups, _ = components(expansion_rows)
    blocked_component_ids = set()
    for group in expansion_groups:
        if set(group) & set(dev_ids):
            blocked_component_ids.update(group)
    existing_train_bridges = sorted(blocked_component_ids & set(train_ids))
    outside_queue = set(pool) - set(queue)
    remaining = sorted(outside_queue - blocked_component_ids)
    excluded = sorted(outside_queue & blocked_component_ids)
    train_stems = {r["source_stem"] for r in train}
    possible_rows = [pool[sid] for sid in remaining]
    by_speaker = dict(sorted(Counter(r["speaker_id"] for r in possible_rows).items()))
    expansion = {
        "schema": "lipnet-offline-dataset-stage3-expansion-metadata-only-v1",
        "status": "proposal_only_no_new_candidate_review_or_split_assignment",
        "source_pool_parents": 293, "current_reviewed_parents": 15, "held_parents": [held],
        "additional_parent_metadata_before_new_dev_protection": len(outside_queue),
        "excluded_additional_metadata_due_to_transitive_new_dev_identity_or_text": len(excluded),
        "additional_parent_metadata_remaining": len(remaining),
        "remaining_speakers": len(by_speaker),
        "remaining_recordings": len({r["source_stem"] for r in possible_rows}),
        "remaining_counts_by_speaker": by_speaker,
        "remaining_on_current_train_recordings": sum(r["source_stem"] in train_stems for r in possible_rows),
        "remaining_on_other_recordings": sum(r["source_stem"] not in train_stems for r in possible_rows),
        "transitive_dev_component_ids": sorted(blocked_component_ids),
        "current_train_ids_reachable_only_after_adding_unreviewed_pool": existing_train_bridges,
        "bridge_interpretation": "If nonempty, expanding through these unreviewed bridges would change the current leakage graph; redesign or exclude the bridges before admission. Current15 use their directly observed graph only.",
        "excluded_additional_ids": excluded, "remaining_metadata_ids": remaining,
        "remaining_records": [{"id": r["id"], "speaker_id": r["speaker_id"],
                               "source_stem": r["source_stem"], "historical_text": r["text"],
                               "reviewed_for_this_curriculum": False, "split": "unassigned",
                               "training_ready": False} for r in possible_rows],
        "required_before_use": ["Review canonical transcript and immutable audio/visual bounds for each new phrase and parent.",
            "Recheck revised text and all-angle recording/source identity against all protected sets and frozen dev.",
            "Verify native PTS, D grid, crop and neighboring-label scope; audit exact source/content hashes.",
            "Use a separately frozen expansion plan; do not count these metadata rows as approved clips."],
        "verified_new_phrases": 0, "new_cache_clips": 0, "new_training_updates": 0,
        "training_ready": False, "unseen_speaker_source_available_in_this293": False}
    counts = {}
    for split_name, subset in [("train", train), ("dev", dev)]:
        counts[split_name] = {"parents": len(subset), "phrases": len(subset),
            "planned_clips": len(subset) * 2,
            "speakers": len({r["speaker_id"] for r in subset}),
            "recordings": len({r["source_stem"] for r in subset}),
            "parents_by_speaker": dict(sorted(Counter(r["speaker_id"] for r in subset).items())),
            "unique_parent_texts": len({text_key(r["parent_text"]) for r in subset}),
            "unique_phrase_texts": len({text_key(r["phrase"]["text"]) for r in subset}),
            "parent_samples_16k": sum(r["sentence_end_sample_16k"]-r["sentence_start_sample_16k"] for r in subset),
            "phrase_samples_16k": sum(r["phrase"]["end_sample_16k"]-r["phrase"]["start_sample_16k"] for r in subset)}
    plan = {
        "schema": "lipnet-offline-dataset-stage3-split-plan-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "fixed_metadata_split_for_small_engine_pilot_not_training_approval",
        "selection_protocol": {"path": str(protocol_path), "sha256": sha(protocol_path)},
        "generator": {"path": str(Path(__file__).resolve()), "sha256": sha(Path(__file__))},
        "inputs": protocol["inputs"],
        "selection": {"seed": 42, "legal_partition_count": len(valid),
            "selected_zero_based_index": selected_index, "selected_component_indices": list(selected),
            "all_legal_partitions_sha256": digest(valid), "random_draws": 1,
            "model_scores_used": False, "duration_optimization_used": False},
        "leakage_components": groups, "grouping_links": links,
        "parent_to_split": {r["parent_id"]: r["split"] for r in rows},
        "train_parent_ids": train_ids, "dev_parent_ids": dev_ids,
        "records": rows, "planned_clips": clips, "counts": counts,
        "held_parent_excluded_from_both_roles": held,
        "cross_split_collision_check": cross, "historical_protection_checks": protection,
        "canonical_parent_or_phrase_vs_inherited_blocked_targets": current_blocked,
        "metadata_chain_bindings_verified": len(bound_checks), "historical_logical_rows_verified": 632,
        "inherited_protection_scope": [
            "293 pool inherited olddev95/held27 ID and full-transcript protection, not a blanket whole-recording claim.",
            "Probe14 and previous dev16 were protected by full recording/stem/videoSHA/labelSHA and full transcript.",
            "This15 additionally rechecked all listed identities and current/historical full-parent/full-phrase text against these groups: zero intersections.",
            "Filename-based angle-independent keys group camera letters A/B/C; only recorded front A sources are present here. Unseen angle files were not inspected."],
        "interpretation": [
            "15 parents x (one phrase + one full sentence) =30 nested planned clips, not30 independent utterances.",
            "12train/3dev is a small data/engine pilot. Dev speakers all appear in train; no unseen-speaker or final-test claim.",
            "These parents were historical training candidates; prior trained checkpoints may already know the dev material.",
            "For a future new-split learning comparison require fresh student initialization; pretrained-teacher overlap remains unestablished.",
            "Common eojeols and substrings are allowed; full parent/phrase equality after NFC/whitespace normalization is grouped.",
            "No validation metric may be reported as a robust generalization estimate from three dev parents.",
            "Native extraction, CTC feasibility, crop/boundary QC and training authorization remain separate gates."],
        "training_ready": False, "split_metadata_ready": True,
        "new_source_media_reads": 0, "new_media_decodes": 0, "new_model_calls": 0,
        "new_cache_clips_generated_by_split_generator": 0, "new_training_updates": 0,
        "service_weights_changed": False, "expansion": expansion}
    # Recheck every permitted input after planning; all outputs remain under split/.
    for ref in protocol["inputs"].values():
        assert sha(Path(ref["path"])) == ref["sha256"]
    return plan


def report(plan: dict[str, Any]) -> str:
    c, e = plan["counts"], plan["expansion"]
    lines = ["녹화형 3단계: 15부모·15구절 분할 계획", "",
        "상태: 메타데이터 분할 고정. 원본 추출·CTC·새 픽셀 검수·학습 승인은 별도입니다.",
        f"학습 {c['train']['parents']}부모/{c['train']['planned_clips']}예정 클립, 개발 {c['dev']['parents']}부모/{c['dev']['planned_clips']}예정 클립입니다.",
        f"학습에는 기존8화자 전원, 개발에는 그중3화자가 포함됩니다. 미지 화자 평가나 최종시험이 아닙니다.",
        "부모 문장과 부분 구절은 같은 영상을 공유하므로 30개 독립 발화로 세지 않습니다.", "",
        "선정 규칙", "사전 고정 selection_protocol.json: seed42, 동일 녹화·영상 SHA·라벨 SHA·완전 문장·완전 구절을 union.",
        "전사는 NFC 후 모든 공백을 제거해 비교하며, 과거 부모 전사 별칭도 보호합니다. 공통 어절·부분 문자열은 허용합니다.",
        "카메라 문자 A/B/C를 제거한 녹화키도 묶었습니다. 현재15자료는 정면 A이며 다른 각도 파일을 새로 검사한 것은 아닙니다.",
        f"12/3과 화자 조건을 만족하는 {plan['selection']['legal_partition_count']}개 분할 중 seed42로 단1회 추첨했습니다.",
        f"선택 인덱스(0부터): {plan['selection']['selected_zero_based_index']}. 모델 점수·오류율·길이 최적화·재추첨은 사용하지 않았습니다.",
        "C293_A011_018과 C312_A010_010의 동일 전사/구절은 함께 학습에 두었습니다.", "", "개발 부모:"]
    for sid in plan["dev_parent_ids"]:
        row = next(r for r in plan["records"] if r["parent_id"] == sid)
        lines.append(f"- {sid}: {row['parent_text']} / 구절: {row['phrase']['text']}")
    lines += ["", "보호와 한계",
        "학습/개발 간 부모녹화·영상/라벨 SHA·완전 전사/구절 교집합0. C293_A012_019 ‘탔’ OOV 부모는 문장·구절 모두 제외했습니다.",
        "과거 olddev95/held27의 계승 규칙은 ID·완전 전사이며, Probe14·이전dev16은 녹화와 완전 전사까지 보호합니다.",
        "이번15에는 과거 보호152행(122+14+16, 그룹 간 중복 가능)에 대해 ID·녹화키·SHA·현재/과거 전사와 구절을 다시 비교했습니다.",
        "실제 검사 그룹 크기는 JSON의 historical_protection_checks를 기준으로 보십시오. 현재15 교집합은 모두0입니다.",
        "모든 부모는 과거 학습 후보이므로 기존 모델을 이어 학습하면 개발 자료가 이미 학습되었을 수 있습니다.",
        "새 분할의 학습 비교는 새 학생 초기화가 필요하며 교사 사전학습의 데이터 중복은 여전히 미확인입니다.",
        "개발3부모는 실행기 검증용으로 작습니다. 일반화/실용 인식 성능을 증명하지 않습니다.", "", "추가 메타데이터 가능성",
        f"기존 보호293 중 현재15와 OOV1을 빼면 {e['additional_parent_metadata_before_new_dev_protection']}부모 메타데이터가 남습니다.",
        f"새 개발 분할에 녹화/SHA/완전전사로 직접·간접 연결되는 추가 {e['excluded_additional_metadata_due_to_transitive_new_dev_identity_or_text']}개를 제외하면 {e['additional_parent_metadata_remaining']}개가 후속 검수 가능 범위입니다.",
        f"남은 범위는 {e['remaining_speakers']}화자/{e['remaining_recordings']}녹화이며 승인된 새 클립0입니다.",
        "확대 후보는 아직 전사·구절 경계·새 원본 픽셀 검수와 분할이 미완료입니다. 문장 메타데이터 수를 학습 가능한 구절 수로 환산하지 않습니다.",
        "이번 생성기는 JSON만 읽었습니다. TAR/영상/음성/배열/모델/다운로드/학습/서비스 변경0."]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true", help="Recompute and compare existing output, without writing")
    args = parser.parse_args()
    plan = generate()
    destination = BASE / "split_plan.json"
    if args.verify:
        stored = json.loads(destination.read_text(encoding="utf-8"))
        plan["created_utc"] = stored["created_utc"]
        assert plan == stored, "Recomputed split differs from fixed output"
        assert (BASE / "split_report.txt").read_text(encoding="utf-8") == report(stored)
        assert json.loads((BASE / "expansion_metadata_proposal.json").read_text(encoding="utf-8")) == plan["expansion"]
        print(json.dumps({"verified": True, "split_sha256": sha(destination), "counts": plan["counts"]}, ensure_ascii=False))
        return
    for name in ("split_plan.json", "split_report.txt", "expansion_metadata_proposal.json"):
        if (BASE / name).exists():
            raise FileExistsError(f"Refuse to replace fixed output: {name}")
    destination.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (BASE / "split_report.txt").write_text(report(plan), encoding="utf-8")
    (BASE / "expansion_metadata_proposal.json").write_text(json.dumps(plan["expansion"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"split_sha256": sha(destination), "counts": plan["counts"],
                      "dev_ids": plan["dev_parent_ids"], "expansion_remaining": plan["expansion"]["additional_parent_metadata_remaining"]}, ensure_ascii=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    try:
        main()
    except Exception as exc:
        print(f"Split preparation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
