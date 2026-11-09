"""병실 명령 채점·기준 맞추기·평가 보고 (병실 기획 9.5·10.2). 결과 행은 ward_service_batch.py, 등록 거리는 ward_enroll.py apply.

  metrics     : 결과 행에 기준값 한 벌을 적용한 지표 (조정 자료 점검용, 기본 임시값 0.5 / -3.40)
  calibrate   : P01·P03 조정 자료(병실 명령 + 목록 밖 문장 촬영 클립)로 기준값·축소 목록·등록 거리 D를 맞춰
                동결 초안 JSON을 쓴다. 봉인 자료(P02·ward_eval)는 어떤 옵션으로도 받지 않는다. 초안은 사람이 검토해 커밋한다.
                --sealed-folder로 봉인 폴더의 목록 지문(해시만, 메타데이터·영상은 열지 않음)을 초안에 넣는다.
  report      : 동결 파일 한 벌을 결과 행(+ 등록 거리)에 그대로 적용해 사전 등록 지표 표(한국어)를 낸다.
                봉인 자료는 --allow-sealed와 커밋된(status frozen) 동결 파일, 동결 뒤에 만든 결과 파일, 동결한 폴더 지문이 필요하다.
  fingerprint : 폴더 목록 지문 (해시만, 봉인 폴더에도 쓸 수 있다: 8단계 문서 2절과 같은 형식)
  validate    : 동결 파일 형식 검사

판정 규칙과 지표 정의는 scripts/ward_tools/metrics.py 머리말에 있다. 서버 accepted는 쓰지 않고 저장된 margin·score로 다시 판정하며,
서버 accepted와 다시 판정한 값이 다른 클립 수를 보고서에 적는다(서버가 동결 기준값으로 켜졌는지 확인).
긴급 가능성 점수 하한은 화면에 없으므로 기본으로 쓰지 않는다(calibrate --floor로만 초안에 넣고, 화면에 넣기 전에는 동결할 수 없다).

사용 예:
  py -3.11 scripts/ward_eval_tools.py calibrate \
      --tune-rows experiments/ward_tune_20261111/P01_ward_tune.json --tune-rows experiments/ward_tune_20261111/P03_ward_tune.json \
      --negative-rows experiments/ward_negatives_20261110/P01_sentences_as_ward.json \
      --negative-rows experiments/ward_negatives_20261110/P03_sentences_as_ward.json \
      --enroll experiments/ward_tune_20261111/P01_enroll.json --enroll experiments/ward_tune_20261111/P03_enroll.json \
      --sealed-folder "<P02 ward_eval 폴더>" --out experiments/ward_tune_20261111/ward_freeze_draft.json
  py -3.11 scripts/ward_eval_tools.py report --freeze docs/ward_freeze.json --rows <P02 행> --allow-sealed --out <결과.json>
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from ward_tools import calibrate as cal  # noqa: E402
from ward_tools import common  # noqa: E402
from ward_tools import metrics as mt  # noqa: E402
from ward_tools import report as rp  # noqa: E402

# 동결 파일에 지문을 남길 코드·목록 파일 (저장소 기준 경로). 화면 판정 코드도 넣는다(판정 규칙이 화면과 같아야 한다).
CODE_FILES = (
    "scripts/ward_eval_tools.py", "scripts/ward_service_batch.py", "scripts/ward_enroll.py",
    "scripts/ward_tools/__init__.py", "scripts/ward_tools/common.py", "scripts/ward_tools/metrics.py",
    "scripts/ward_tools/calibrate.py", "scripts/ward_tools/enroll.py", "scripts/ward_tools/report.py",
    "scripts/olkavs_transcribe.py", "scripts/pilot_evaluate.py", "backend/config.py", "backend/api/record.py",
    "backend/services/olkavs_service.py", "backend/services/olkavs_worker.py",
    "frontend/src/ward/commands.json", "frontend/src/capture/ward_prompts.json",
    "frontend/src/ward/wardMachine.js", "frontend/src/ward/wardResult.js",
)
CONFLICT_RULE = ("글자 규칙을 통과하면 등록 거리와 관계없이 글자 1등을 고른다. 통과하지 못했고 등록 1등 거리가 D 이하일 때만 "
                 "글자 1등과 등록 1등이 같은지 보고(require_agreement), 같으면 등록 1등을 고른다. 고르지 못했을 때 글자 1·2등에 "
                 "긴급이 있거나 거리 D 이하인 등록 1등이 긴급이면 긴급 가능성이다.")


def code_fingerprints() -> dict[str, str]:
    """CODE_FILES의 SHA-256 (없는 파일은 뺀다)."""
    out = {}
    for rel in CODE_FILES:
        path = ROOT / rel
        if path.is_file():
            out[rel] = common.file_sha256(path)
    return out


def git(*args: str) -> str | None:
    """저장소에서 git 명령 (실패하면 None)."""
    return common.git(*args)


def freeze_commit_state(path: Path) -> dict[str, Any]:
    """동결 파일이 git에 커밋됐고 그 뒤 바뀌지 않았는지: {tracked, clean, commit, commit_time}."""
    return common.freeze_commit_state(path)


def sealed_ok_for_report(freeze: dict[str, Any], state: dict[str, Any]) -> list[str]:
    """봉인 자료 보고 전 조건: 병실 동결 파일, status frozen, git에 커밋됨, 커밋 뒤 수정 없음. 문제 목록."""
    return common.sealed_ok(freeze, state)


# ---------------------------------------------------------------------------
# 입력 읽기
# ---------------------------------------------------------------------------

def load_enroll_extras(paths: list[str], commands: dict[str, Any] | None = None
                       ) -> tuple[dict[tuple[str, str], dict[str, Any]], list[dict[str, Any]]]:
    """ward_enroll.py apply 출력 → {(session, file): {text_scores, tmpl_dists, is_template}}, 파일 정보.

    Raises:
        ValueError: apply 출력이 아니거나, commands를 주었는데 명령 목록 SHA-256이 다를 때
    """
    extras: dict[tuple[str, str], dict[str, Any]] = {}
    info = []
    for p in paths:
        data = common.read_json(p)
        if not isinstance(data, dict) or data.get("schema") != "ward-enroll-v1":
            raise ValueError(f"ward_enroll.py apply 출력이 아닙니다: {p}")
        enr_cmd = (data.get("commands") or {}).get("sha256")
        if commands is not None and enr_cmd is not None and enr_cmd != commands["sha256"]:
            raise ValueError(f"{p}: 등록 파일의 명령 목록이 지금과 다릅니다(commands.json SHA-256 불일치)")
        clips = data.get("clips") or []
        for c in clips:
            extras[(str(c.get("session")), str(c.get("file")))] = {
                "text_scores": c.get("text_scores"), "tmpl_dists": c.get("tmpl_dists"), "is_template": c.get("is_template")}
        info.append({"path": str(p), "sha256": common.file_sha256(p), "template": data.get("template"),
                     "feature": data.get("feature"), "distance": data.get("distance"), "commands_sha256": enr_cmd,
                     "n_clips": len(clips), "participants": sorted({str(c.get("participant")) for c in clips}),
                     "sessions": sorted({str(c.get("session")) for c in clips}), "synthetic": bool(data.get("synthetic")),
                     "created": data.get("created"), "git_head": data.get("git_head"), "freeze": data.get("freeze"),
                     "inputs_listing": data.get("inputs_listing"),
                     "sealed": bool(common.sealed_reasons([p] + list(data.get("inputs") or []), clips))})
    return extras, info


def load_bases(rows_paths: list[str], commands: dict[str, Any], extras: dict[tuple[str, str], dict[str, Any]],
               phrase_set: str = "ward") -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """결과 행 파일들 → 판정 전 기록, 파일 정보(지문 포함).

    Raises:
        ValueError: 묶음이 phrase_set이 아니거나 phrase.set이 다른(없는) 행이 있을 때, 결과 행 파일의 명령 목록 SHA-256이
            지금과 다를 때, 같은 (묶음, 파일)이 두 번 나올 때
    """
    bases: list[dict[str, Any]] = []
    info = []
    seen: set[tuple[str, str]] = set()
    for p in rows_paths:
        head, rows = common.read_rows(p)
        if head.get("phrase_set") not in (None, phrase_set):
            raise ValueError(f"{p}: 묶음이 {head.get('phrase_set')}입니다({phrase_set} 결과가 필요)")
        head_cmd = (head.get("commands") or {}).get("sha256") if isinstance(head.get("commands"), dict) else None
        if head_cmd is not None and head_cmd != commands["sha256"]:
            raise ValueError(f"{p}: 실행 때 명령 목록이 지금과 다릅니다(commands.json SHA-256 불일치)")
        for r in rows:
            key = (str(r.get("session")), str(r.get("file")))
            if key in seen:
                raise ValueError(f"같은 클립이 두 번 들어왔습니다: {key}")
            seen.add(key)
            b = mt.prepare(r, commands, extras.get(key))
            raw = mt.row_phrase(r) if r.get("status") == "done" else None
            if raw is not None and raw.get("set") != phrase_set:
                raise ValueError(f"{p}: {r.get('file')}의 phrase.set이 {raw.get('set')!r}입니다({phrase_set} 결과가 아님)")
            b["synthetic"] = bool(head.get("synthetic") or r.get("synthetic"))
            bases.append(b)
        listing = (head.get("clips_listing") or {}).get("listing_sha256")
        info.append({"path": str(p), "sha256": common.file_sha256(p), "schema": head.get("schema"),
                     "phrase_set": head.get("phrase_set"), "clips_dir": head.get("clips_dir"),
                     "clips_listing_sha256": listing, "n_rows": len(rows),
                     "participants": sorted({str(r.get("participant")) for r in rows}),
                     "sessions": sorted({str(r.get("session")) for r in rows}),
                     "server_thresholds": head.get("thresholds_from_health"), "git_head_at_run": head.get("git_head"),
                     "git_head": head.get("git_head"), "created": head.get("created"), "freeze": head.get("freeze"),
                     "synthetic": bool(head.get("synthetic")),
                     "phrase_problems": sum(1 for b in bases[-len(rows):] if b.get("phrase_problem")) if rows else 0,
                     "sealed": bool(common.sealed_reasons([p, head.get("clips_dir")], rows))})
    return bases, info


def folder_check(info: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """결과 행 파일의 촬영 폴더 지문을 지금 다시 계산해 실행 때와 비교한다 (폴더가 없으면 건너뜀)."""
    out = []
    for f in info:
        d = f.get("clips_dir")
        if not d or not Path(d).is_dir():
            out.append({"clips_dir": d, "checked": False})
            continue
        now = common.folder_fingerprint(d)
        out.append({"clips_dir": d, "checked": True, "listing_sha256": now["listing_sha256"], "files": now["files"],
                    "bytes": now["bytes"], "same_as_run": now["listing_sha256"] == f.get("clips_listing_sha256")})
    return out


def rule_from_args(args: argparse.Namespace, commands: dict[str, Any]) -> mt.Rule:
    """명령줄 기준값 → 규칙."""
    return mt.Rule(min_margin=args.margin, min_score=args.score, emergency_ids=frozenset(commands["emergency"]),
                   max_distance=args.D, require_agreement=not args.no_agreement, possible_min_score=args.possible_floor)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def cmd_metrics(args: argparse.Namespace) -> int:
    """기준값 한 벌의 지표 (조정 자료 점검). 봉인 자료는 --allow-sealed --freeze(커밋된 동결 파일)가 있어야 한다."""
    try:
        gate = common.freeze_gate(args.freeze, args.allow_sealed)
        common.require_unsealed(list(args.rows) + list(args.enroll or []), [], args.allow_sealed, "지표 계산")
        commands = common.load_commands()
        extras, _einfo = load_enroll_extras(args.enroll or [], commands)
        bases, info = load_bases(args.rows, commands, extras)
        common.require_unsealed([f.get("clips_dir") for f in info], bases, args.allow_sealed, "지표 계산")
    except (common.SealedDataError, ValueError, OSError) as exc:
        print(f"시작하지 않았습니다: {exc}", file=sys.stderr)
        return 2
    rule = rule_from_args(args, commands)
    sel = [b for b in bases if b.get("tmpl_dists")] if rule.max_distance is not None else bases
    summary = mt.summarize_groups(mt.evaluate_all(sel, rule), rule, commands)
    md = rp.table(summary["by_group"], f"기준 {rule.min_margin} / {rule.min_score}"
                  + (f", D {rule.max_distance}" if rule.max_distance is not None else "")
                  + (f", 긴급 가능성 하한 {rule.possible_min_score}" if rule.possible_min_score is not None else ""))
    print(md)
    if args.out:
        common.atomic_write_json(args.out, {"schema": "ward-metrics-v1", "created": datetime.now(timezone.utc).isoformat(),
                                            "rule": rule.describe(), "inputs": info, "freeze": gate, "summary": summary})
    return 0


# ---------------------------------------------------------------------------
# calibrate
# ---------------------------------------------------------------------------

def enrollment_block(bases: list[dict[str, Any]], rule: mt.Rule, enroll_info: list[dict[str, Any]],
                     opts: dict[str, Any], notes: list[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    """동결 초안의 enrollment 칸과 근거. 템플릿 방식·특징·거리 이름이 등록 파일마다 같아야 D를 정한다."""
    block: dict[str, Any] = {"max_distance": None, "template": None, "feature": common.FEATURE_NAME,
                             "distance": common.DISTANCE_NAME, "require_agreement": True, "conflict_rule": CONFLICT_RULE,
                             "unbounded": None, "min_negatives_with_dist": opts.get("min_negatives", cal.MIN_NEGATIVES_FOR_D),
                             "negatives_with_dist": None, "chosen_by": "등록 거리 자료가 없어 정하지 않음"}
    if not any(b.get("tmpl_dists") for b in bases):
        if enroll_info:
            notes.append("등록 파일에 등록 거리가 있는 클립이 없어(병실 묶음의 등록 템플릿 없음) D를 정하지 않았다.")
        return block, {}
    templates = sorted({str(e.get("template")) for e in enroll_info})
    names = sorted({(str(e.get("feature")), str(e.get("distance"))) for e in enroll_info})
    lopo = cal.lopo_distance(bases, rule)
    block["negatives_with_dist"] = {k: v["fit"].get("n_negatives_with_dist") for k, v in lopo.items()}
    block["unbounded"] = any(v["fit"].get("D") is not None and v["fit"].get("unbounded") for v in lopo.values())
    if len(templates) != 1 or templates[0] not in common.ENROLL_TEMPLATES:
        block["chosen_by"] = f"등록 파일의 템플릿 방식이 하나가 아니어서 D를 정하지 않음: {templates}"
        notes.append(block["chosen_by"])
        return block, lopo
    if names != [(common.FEATURE_NAME, common.DISTANCE_NAME)]:
        block["chosen_by"] = f"등록 파일의 특징·거리 이름이 ward_enroll.py와 다르거나 섞여 있어 D를 정하지 않음: {names}"
        notes.append(block["chosen_by"])
        return block, lopo
    pick = cal.choose_distance(lopo, opts.get("d_rule", "min"), opts.get("min_negatives", cal.MIN_NEGATIVES_FOR_D))
    block.update(template=templates[0], chosen_by=pick["chosen_by"],
                 max_distance=None if pick["D"] is None else round(pick["D"], 4))
    if pick["D"] is None:
        notes.append(f"등록 거리 D: {pick['chosen_by']}")
    return block, lopo


def build_draft(bases: list[dict[str, Any]], commands: dict[str, Any], rows_info: list[dict[str, Any]],
                enroll_info: list[dict[str, Any]], opts: dict[str, Any]) -> dict[str, Any]:
    """조정 기록 → 동결 초안 (CLI와 테스트가 같이 쓴다).

    opts: target_size, keep, max_loss, delta, d_rule, min_negatives, floor_clearance, use_floor(기본 False), deltas,
          folder_checks, sealed_folders
    """
    emergency = commands["emergency"]
    prov_rule = cal.rule_at(*cal.PROVISIONAL, emergency)
    notes: list[str] = []
    lopo = cal.lopo_thresholds(bases, emergency)
    pooled = cal.pooled_thresholds(bases, emergency, opts.get("deltas", cal.DEFAULT_DELTAS))
    prop = cal.choose_proposal(pooled, opts.get("max_loss", 0.10), opts.get("delta"))
    rule = cal.rule_at(prop["M"], prop["S"], emergency)
    # 긴급 가능성 점수 하한: 화면(wardMachine.js)에 없으므로 근거로만 계산하고, --floor일 때만 초안에 넣는다
    floor_block: dict[str, Any] = {"possible_min_score": None, "floor_implemented_in_ui": False,
                                   "chosen_by": "기획 9.3·화면 그대로(점수 하한 없음: 고르지 못했을 때 1·2등에 긴급이면 긴급 가능성)"}
    floor_lopo = cal.lopo_possible_floor(bases, rule, opts.get("floor_clearance", 0.2))
    if opts.get("use_floor", False):
        floors = {k: (v["fit"]["floor"] if k != "pooled" else v["floor"]) for k, v in floor_lopo.items()}
        floors = {k: v for k, v in floors.items() if v is not None}
        if floors:
            key = min(floors, key=lambda k: floors[k])
            floor_block.update(possible_min_score=round(floors[key], 3),
                               chosen_by=f"긴급 명령 클립을 하나도 놓치지 않는 하한에서 여유 {opts.get('floor_clearance', 0.2)}를 "
                                         f"뺀 값 중 가장 낮은 값({key})")
            notes.append("긴급 가능성 점수 하한을 넣었지만 화면(wardMachine.js)은 하한을 쓰지 않는다. 화면에 먼저 넣고 "
                         "decision_rule.floor_implemented_in_ui를 true로 바꾸기 전에는 형식 검사가 동결을 막는다.")
        else:
            notes.append("긴급 명령 클립 중 '1·2등에 긴급' 길로만 긴급에 닿는 클립이 없어 긴급 가능성 점수 하한을 정하지 않았다.")
    # 축소 목록
    reduced = cal.select_reduced_list(bases, commands["ids"], opts.get("target_size", 12),
                                      list(emergency) + list(opts.get("keep") or []), (prop["M"], prop["S"]),
                                      emergency=emergency)
    red_block: dict[str, Any] = {**{k: reduced[k] for k in ("ids", "size", "protected_ids", "method", "exact", "trace")},
                                 "min_margin": None, "min_score": None}
    red_evidence: dict[str, Any] = {}
    if reduced["exact"]:
        rp_pooled = cal.pooled_thresholds(bases, emergency, opts.get("deltas", cal.DEFAULT_DELTAS), ids=reduced["ids"])
        rprop = cal.choose_proposal(rp_pooled, opts.get("max_loss", 0.10), opts.get("delta"))
        red_block.update(min_margin=rprop["M"], min_score=rprop["S"], chosen_by=rprop["chosen_by"])
        red_evidence = {"lopo": cal.lopo_thresholds(bases, emergency, ids=reduced["ids"]), "pooled": rp_pooled}
    else:
        notes.append("후보별 점수(ward_enroll.py 결과)가 모든 클립에 있지 않아 축소 목록은 서비스 1·2등만으로 근사했고 "
                     "축소 목록 기준값을 정하지 않았다(null).")
    # 등록 거리
    enr_block, d_evidence = enrollment_block(bases, rule, enroll_info, opts, notes)
    synthetic = any(f.get("synthetic") for f in rows_info + enroll_info)
    if synthetic:
        notes.insert(0, "합성(가짜) 자료가 섞인 초안이다. 동결에 쓸 수 없다(파이프라인 시험용).")
    sealed_folders = opts.get("sealed_folders") or []
    if not sealed_folders:
        notes.append("봉인 폴더 지문(--sealed-folder)이 없다. 봉인 자료 보고(report)는 동결한 폴더 지문과 결과 행의 지문이 "
                     "같아야 하므로, 동결 전에 P02 ward_eval 폴더 지문을 넣어야 한다.")
    return {
        "schema": cal.FREEZE_SCHEMA, "status": "draft", "created": datetime.now(timezone.utc).isoformat(),
        "phrase_set": "ward", "synthetic": synthetic,
        "commands": {"path": "frontend/src/ward/commands.json", "version": commands["version"], "sha256": commands["sha256"],
                     "ids": commands["ids"], "emergency_ids": emergency},
        "full_list": {"min_margin": prop["M"], "min_score": prop["S"], "delta": prop["delta"], "chosen_by": prop["chosen_by"],
                      "env": {"LIPNET_OLKAVS_WARD_MIN_MARGIN": str(prop["M"]), "LIPNET_OLKAVS_WARD_MIN_SCORE": str(prop["S"])}},
        "reduced_list": red_block,
        "enrollment": enr_block,
        "decision_rule": {"table": "docs/병실무음도우미_기획_및_개발계획_2026-11-09.md 9.3 (scripts/ward_tools/metrics.py)",
                          "possible_ranks": 2, **floor_block,
                          "fit_rule": "맞추는 사람 자료에서 표시 중 오답 0, 함정·말하지 않기·문장 촬영 클립의 명령 표시 0, "
                                      "짧은 긴급 표현의 일반 명령 실행 0, 그다음 표시 최대"},
        "sealed_folders": sealed_folders,
        "evidence": {"provisional": cal.headline(bases, prov_rule), "lopo": lopo, "pooled": pooled,
                     "proposal_on_all": cal.headline(bases, rule), "possible_floor": floor_lopo,
                     "reduced": red_evidence, "distance": d_evidence, "repro": mt.repro_check(bases)},
        "data": {"rows_files": rows_info, "enroll_files": enroll_info, "folder_checks": opts.get("folder_checks", []),
                 "listing_format": common.LISTING_FORMAT},
        "code": {"git_head": git("rev-parse", "HEAD"), "files": code_fingerprints()},
        "sealed_check": {"participants_used": sorted({str(b.get("participant")) for b in bases}),
                         "sessions_used": sorted({str(b.get("session")) for b in bases}), "sealed_data_read": False,
                         "sealed_folders_hash_only": True},
        "notes": notes,
    }


def sealed_folder_fingerprints(folders: list[str]) -> list[dict[str, Any]]:
    """봉인 폴더의 목록 지문 (파일 이름·크기·SHA-256만, 메타데이터 JSON과 영상은 해석하지 않는다)."""
    out = []
    for f in folders:
        if not Path(f).is_dir():
            raise ValueError(f"봉인 폴더가 없습니다: {f}")
        fp = common.folder_fingerprint(f)
        out.append({k: fp[k] for k in ("path", "files", "videos", "bytes", "listing_sha256")})
    return out


def cmd_calibrate(args: argparse.Namespace) -> int:
    """조정 자료로 동결 초안을 만든다. 봉인 자료는 받지 않는다(옵션 없음). 봉인 폴더는 지문만 계산한다."""
    try:
        commands = common.load_commands()
        paths = (args.tune_rows or []) + (args.negative_rows or [])
        common.require_unsealed(paths + (args.enroll or []), [], False, "기준 맞추기")
        extras, einfo = load_enroll_extras(args.enroll or [], commands)
        bases, info = load_bases(paths, commands, extras)
        common.require_unsealed([f.get("clips_dir") for f in info], bases, False, "기준 맞추기")
        neg_files = {str(Path(p)) for p in args.negative_rows or []}
        for f in info:
            f["role"] = "negatives" if str(Path(f["path"])) in neg_files else "tune"
        if len(cal.persons(bases)) < 2:
            raise ValueError("한 사람으로 정해 다른 사람에게 적용하려면 참가자가 2명 이상 필요합니다")
        sealed_folders = sealed_folder_fingerprints(args.sealed_folder or [])
    except (common.SealedDataError, ValueError, OSError) as exc:
        print(f"시작하지 않았습니다: {exc}", file=sys.stderr)
        return 2
    opts = {"target_size": args.target_size, "keep": args.keep or [], "max_loss": args.max_loss, "delta": args.delta,
            "d_rule": args.d_rule, "min_negatives": args.min_negatives, "floor_clearance": args.floor_clearance,
            "use_floor": bool(args.floor), "sealed_folders": sealed_folders,
            "folder_checks": [] if args.skip_folder_check else folder_check(info)}
    draft = build_draft(bases, commands, info, einfo, opts)
    problems = cal.validate_freeze(draft)
    draft["validation_problems"] = problems
    common.atomic_write_json(args.out, draft)
    md = rp.calibration_markdown(draft)
    Path(args.out).with_suffix(".md").write_text(md, encoding="utf-8")
    print(md)
    print(f"초안: {args.out}" + (f" (형식 문제 {len(problems)}개: {problems})" if problems else " (형식 검사 통과)"))
    return 0 if not problems else 1


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def run_report(freeze: dict[str, Any], bases: list[dict[str, Any]], commands: dict[str, Any]) -> dict[str, Any]:
    """동결 규칙을 기록에 적용한 결과 (전체 목록, 등록, 축소 목록).

    주 결과는 화면이 실제로 쓰는 규칙(긴급 가능성 점수 하한 없음)이다. 동결 파일에 하한이 있으면 하한을 적용한 결과를
    full_floored(보조)로 따로 낸다.
    """
    full = cal.rule_from_freeze(freeze, "full")
    out: dict[str, Any] = {"full": mt.summarize_groups(mt.evaluate_all(bases, full), full, commands)}
    if (freeze.get("decision_rule") or {}).get("possible_min_score") is not None:
        ff = cal.rule_from_freeze(freeze, "full", apply_floor=True)
        out["full_floored"] = mt.summarize_groups(mt.evaluate_all(bases, ff), ff, commands)
    tests = [b for b in bases if b.get("tmpl_dists")]
    if tests and (freeze.get("enrollment") or {}).get("max_distance") is not None:
        er = cal.rule_from_freeze(freeze, "full", with_enrollment=True)
        out["enroll"] = mt.summarize_groups(mt.evaluate_all(tests, er), er, commands)
        out["enroll_text_only"] = mt.summarize_groups(mt.evaluate_all(tests, full), full, commands)
    red = freeze.get("reduced_list") or {}
    if red.get("min_margin") is not None and any(b.get("text_scores") for b in bases):
        rr = cal.rule_from_freeze(freeze, "reduced")
        ids = set(rr.ids or ())
        # 축소 목록을 쓰는 사람은 목록에서 뺀 명령을 말하지 않으므로 그 명령 클립은 뺀다
        sel = [b for b in bases if b["role"] != mt.ROLE_COMMAND or b["truth"] in ids]
        out["reduced"] = mt.summarize_groups(mt.evaluate_all(sel, rr), rr, commands)
    out["repro"] = mt.repro_check(bases)
    return out


def enroll_consistency(freeze: dict[str, Any], einfo: list[dict[str, Any]]) -> list[str]:
    """등록 파일이 동결 파일의 등록 설정(템플릿 방식·특징·거리 이름·명령 목록)과 같은지 (문제 목록)."""
    enr = freeze.get("enrollment") or {}
    if enr.get("max_distance") is None:
        return []
    problems = []
    for e in einfo:
        if e.get("template") != enr.get("template"):
            problems.append(f"{e['path']}: 템플릿 방식 {e.get('template')!r}이(가) 동결 값 {enr.get('template')!r}과 다릅니다")
        if e.get("feature") != enr.get("feature") or e.get("distance") != enr.get("distance"):
            problems.append(f"{e['path']}: 특징·거리 이름이 동결 값과 다릅니다")
        if e.get("commands_sha256") != (freeze.get("commands") or {}).get("sha256"):
            problems.append(f"{e['path']}: 명령 목록 SHA-256이 동결 값과 다릅니다")
    return problems


def sealed_input_problems(freeze: dict[str, Any], freeze_sha: str, state: dict[str, Any], info: list[dict[str, Any]],
                          einfo: list[dict[str, Any]], is_ancestor: Any = None) -> list[str]:
    """봉인 자료 보고 전 입력 조건 (문제 목록): 봉인 결과 파일이 이 동결 파일로, 동결 커밋 뒤에 만들어졌고,
    촬영 폴더 지문이 동결 파일의 sealed_folders 값과 같아야 한다."""
    sealed_rows = [f for f in info if f.get("sealed")]
    sealed_enroll = [e for e in einfo if e.get("sealed")]
    problems = common.order_problems(sealed_rows + sealed_enroll, freeze_sha, state.get("commit"), state.get("commit_time"),
                                     is_ancestor)
    frozen = {sf.get("listing_sha256") for sf in freeze.get("sealed_folders") or [] if isinstance(sf, dict)}
    for f in sealed_rows:
        if f.get("clips_listing_sha256") not in frozen:
            problems.append(f"{f['path']}: 촬영 폴더 지문({f.get('clips_listing_sha256')})이 동결 파일 sealed_folders에 없습니다")
    for e in sealed_enroll:
        listings = [x.get("listing_sha256") for x in e.get("inputs_listing") or [] if isinstance(x, dict)]
        if not listings or any(x not in frozen for x in listings):
            problems.append(f"{e['path']}: 등록 입력 폴더 지문이 동결 파일 sealed_folders에 없습니다: {listings}")
    return problems


def cmd_report(args: argparse.Namespace) -> int:
    """동결 파일로 채점한 사전 등록 지표 표."""
    try:
        freeze_path = Path(args.freeze).resolve()
        freeze = common.read_json(freeze_path)
        problems = cal.validate_freeze(freeze, allow_synthetic=args.allow_synthetic)
        if problems:
            raise ValueError("동결 파일 문제: " + "; ".join(problems))
        commands = common.load_commands()
        if commands["sha256"] != freeze["commands"]["sha256"]:
            raise ValueError("명령 목록이 동결 때와 다릅니다(commands.json SHA-256 불일치)")
        state = freeze_commit_state(freeze_path)
        freeze_sha = common.file_sha256(freeze_path)
        path_reasons = common.sealed_reasons(list(args.rows) + list(args.enroll or []))
        if path_reasons:
            if not args.allow_sealed:
                raise common.SealedDataError("봉인 자료입니다. 동결 파일을 커밋한 뒤 --allow-sealed를 붙이세요: "
                                             + "; ".join(path_reasons))
            blockers = sealed_ok_for_report(freeze, state)
            if blockers:
                raise common.SealedDataError("봉인 자료 보고 조건이 맞지 않습니다: " + "; ".join(blockers))
        extras, einfo = load_enroll_extras(args.enroll or [], commands)
        bases, info = load_bases(args.rows, commands, extras)
        if any(b.get("synthetic") for b in bases) and not args.allow_synthetic:
            raise ValueError("합성 자료가 섞여 있습니다(--allow-synthetic은 파이프라인 시험에만)")
        reasons = common.sealed_reasons([f["path"] for f in info] + [f.get("clips_dir") for f in info]
                                        + (args.enroll or []), bases)
        if reasons:
            if not args.allow_sealed:
                raise common.SealedDataError("봉인 자료입니다. 동결 파일을 커밋한 뒤 --allow-sealed를 붙이세요: " + "; ".join(reasons))
            blockers = sealed_ok_for_report(freeze, state) + sealed_input_problems(freeze, freeze_sha, state, info, einfo)
            if blockers:
                raise common.SealedDataError("봉인 자료 보고 조건이 맞지 않습니다: " + "; ".join(blockers))
        mismatch = enroll_consistency(freeze, einfo)
        if mismatch:
            raise ValueError("등록 파일이 동결 설정과 다릅니다: " + "; ".join(mismatch))
    except (common.SealedDataError, ValueError, OSError) as exc:
        print(f"시작하지 않았습니다: {exc}", file=sys.stderr)
        return 2
    now_code = code_fingerprints()
    changed = sorted(k for k, v in (freeze.get("code") or {}).get("files", {}).items() if now_code.get(k) != v)
    result = run_report(freeze, bases, commands)
    full = freeze["full_list"]
    check = result["full"]["all"].get("server_accepted_check") or {}
    result.update({
        "schema": "ward-report-v1", "created": datetime.now(timezone.utc).isoformat(),
        "freeze": {"path": str(freeze_path), "status": freeze.get("status"), "commit": state.get("commit"),
                   "sha256": freeze_sha,
                   "full": {"min_margin": full["min_margin"], "min_score": full["min_score"],
                            "possible_min_score": (freeze.get("decision_rule") or {}).get("possible_min_score"),
                            "floor_implemented_in_ui": (freeze.get("decision_rule") or {}).get("floor_implemented_in_ui")},
                   "max_distance": (freeze.get("enrollment") or {}).get("max_distance"),
                   "template": (freeze.get("enrollment") or {}).get("template"),
                   "reduced_ids": (freeze.get("reduced_list") or {}).get("ids")},
        "inputs": {"rows_files": [f["path"] for f in info], "rows_info": info, "enroll_files": einfo,
                   "folder_checks": [] if args.skip_folder_check else folder_check(info)},
        "code_changed": changed, "sealed_data": bool(reasons),
        "server_threshold_mismatch": int(check.get("mismatch") or 0)})
    md = rp.markdown_report(result)
    if args.out:
        common.atomic_write_json(args.out, result)
        Path(args.out).with_suffix(".md").write_text(md, encoding="utf-8")
    print(md)
    if result["server_threshold_mismatch"] and reasons:
        print(f"주의: 서버 accepted가 동결 기준값과 다른 클립이 {result['server_threshold_mismatch']}개입니다. 봉인 자료 결과를 "
              "주 결과로 쓰기 전에 서버를 full_list.env로 켰는지 확인하세요.", file=sys.stderr)
        return 1
    return 0


def cmd_fingerprint(args: argparse.Namespace) -> int:
    """폴더 목록 지문 (해시만). 봉인 폴더에도 쓸 수 있다."""
    try:
        out = sealed_folder_fingerprints(args.folders)
    except (ValueError, OSError) as exc:
        print(f"지문을 계산하지 못했습니다: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"format": common.LISTING_FORMAT, "folders": out}, ensure_ascii=False, indent=1))
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """동결 파일 형식 검사."""
    try:
        problems = cal.validate_freeze(common.read_json(args.freeze))
    except ValueError as exc:
        problems = [str(exc)]
    print(json.dumps({"ok": not problems, "problems": problems}, ensure_ascii=False, indent=1))
    return 0 if not problems else 1


def main(argv: list[str] | None = None) -> int:
    """명령줄 진입점."""
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("metrics", help="기준값 한 벌의 지표")
    p.add_argument("--rows", action="append", required=True)
    p.add_argument("--enroll", action="append")
    p.add_argument("--margin", type=float, default=cal.PROVISIONAL[0])
    p.add_argument("--score", type=float, default=cal.PROVISIONAL[1])
    p.add_argument("--possible-floor", type=float, default=None, help="긴급 가능성 점수 하한 (기본 없음 = 기획 9.3·화면)")
    p.add_argument("--D", type=float, default=None, help="등록 거리 D (주면 거리 있는 시험 클립만)")
    p.add_argument("--no-agreement", action="store_true")
    p.add_argument("--allow-sealed", action="store_true", help="봉인 자료도 계산 (--freeze 필요)")
    p.add_argument("--freeze", help="커밋된 동결 파일 (--allow-sealed와 함께 필수)")
    p.add_argument("--out")
    p.set_defaults(func=cmd_metrics)
    p = sub.add_parser("calibrate", help="조정 자료로 동결 초안")
    p.add_argument("--tune-rows", action="append", required=True, help="P01·P03 ward_tune 결과 행")
    p.add_argument("--negative-rows", action="append", help="문장 촬영 클립을 phrase_set=ward로 채점한 결과 행")
    p.add_argument("--enroll", action="append", help="ward_enroll.py apply 출력 (후보별 점수·등록 거리)")
    p.add_argument("--sealed-folder", action="append", help="봉인 폴더 (목록 지문만 계산해 초안 sealed_folders에 넣음)")
    p.add_argument("--out", required=True)
    p.add_argument("--target-size", type=int, default=12)
    p.add_argument("--keep", action="append", help="축소 목록에서 빼지 않을 명령 ID (긴급 명령은 늘 지킴)")
    p.add_argument("--max-loss", type=float, default=0.10)
    p.add_argument("--delta", type=float, default=None, help="합친 맞추기 여유 δ를 직접 고름")
    p.add_argument("--d-rule", choices=("min", "pooled"), default="min")
    p.add_argument("--min-negatives", type=int, default=cal.MIN_NEGATIVES_FOR_D,
                   help="D를 정하려면 사람마다 필요한, 등록 거리가 있는 목록 밖 클립 수")
    p.add_argument("--floor-clearance", type=float, default=0.2)
    p.add_argument("--floor", action="store_true",
                   help="긴급 가능성 점수 하한을 초안에 넣음 (화면에 없으므로 floor_implemented_in_ui=true 전에는 동결 불가)")
    p.add_argument("--skip-folder-check", action="store_true", help="촬영 폴더 지문 재계산을 건너뜀")
    p.set_defaults(func=cmd_calibrate)
    p = sub.add_parser("report", help="동결 파일로 평가")
    p.add_argument("--freeze", required=True)
    p.add_argument("--rows", action="append", required=True)
    p.add_argument("--enroll", action="append")
    p.add_argument("--allow-sealed", action="store_true")
    p.add_argument("--allow-synthetic", action="store_true", help="합성 자료로 파이프라인만 시험")
    p.add_argument("--skip-folder-check", action="store_true", help="촬영 폴더 지문 재계산을 건너뜀")
    p.add_argument("--out")
    p.set_defaults(func=cmd_report)
    p = sub.add_parser("fingerprint", help="폴더 목록 지문 (해시만)")
    p.add_argument("folders", nargs="+")
    p.set_defaults(func=cmd_fingerprint)
    p = sub.add_parser("validate", help="동결 파일 형식 검사")
    p.add_argument("freeze")
    p.set_defaults(func=cmd_validate)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
