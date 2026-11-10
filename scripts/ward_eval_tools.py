"""병실 명령 채점·기준 맞추기·평가 보고 (병실 기획 9.5·10.2). 결과 행은 ward_service_batch.py, 등록 거리는 ward_enroll.py apply.

  metrics     : 결과 행에 기준값 한 벌을 적용한 지표 (조정 자료 점검용, 기본 임시값 0.5 / -3.40)
  calibrate   : P01·P03 조정 자료(병실 명령 + 목록 밖 문장 촬영 클립)로 기준값·축소 목록·등록 거리 D를 맞춰
                동결 초안 JSON을 쓴다. 봉인 자료(P02·ward_eval)는 어떤 옵션으로도 받지 않는다. 초안은 사람이 검토해 커밋한다.
                --sealed-folder로 봉인 폴더의 목록 지문(해시만, 메타데이터·영상은 열지 않음)을 초안에 넣는다.
  report      : 동결 파일 한 벌을 결과 행(+ 등록 거리)에 그대로 적용해 사전 등록 지표 표(한국어)를 낸다.
                주 결과는 앱과 같은 판정이다: 글자 규칙 + 긴급 가능성 하한(동결 값) + 결과 행에 서버 등록 결과가 있으면 서버 등록
                거리(ward_service_batch.py --enroll-profile, 등록 영상(store) 행은 뺌). 오프라인 등록(--enroll)은 보조 결과다.
                봉인 자료는 --allow-sealed와 커밋된(status frozen) 동결 파일, 동결 뒤에 만든 결과 파일, 동결한 폴더 지문이 필요하다.
  fingerprint : 폴더 목록 지문 (해시만, 봉인 폴더에도 쓸 수 있다: 8단계 문서 2절과 같은 형식)
  validate    : 동결 파일 형식 검사 (full_list.env가 기준값 4개와 같은지 포함)
  env         : 동결 파일의 서버 환경 변수 4개를 PowerShell 줄로 출력

판정 규칙과 지표 정의는 scripts/ward_tools/metrics.py 머리말에 있다. 서버 accepted는 쓰지 않고 저장된 margin·score로 다시 판정하며,
서버 accepted·서버가 적은 기준값(phrase.rule)·서버 등록 판정(enroll.within·accepted)과 다시 판정한 값이 다른 클립 수를 보고서에
적는다(서버가 동결 기준값으로 켜졌는지 확인). 긴급 가능성 점수 하한은 화면이 서버 값(phrase.rule)으로 쓴다. 기본은 null이고
calibrate --floor일 때만 초안에 넣는다. 등록 거리 출처(서버 등록 / 오프라인 등록)는 섞지 않고 --enroll-source로 고른다.

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
from dataclasses import replace
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
    "scripts/olkavs_transcribe.py", "scripts/pilot_evaluate.py", "backend/config.py", "backend/main.py",
    "backend/api/record.py", "backend/api/ward_profiles.py",
    "backend/services/olkavs_service.py", "backend/services/olkavs_worker.py",
    "backend/services/ward_profiles.py", "backend/services/ward_templates.py",
    "frontend/src/ward/commands.json", "frontend/src/capture/ward_prompts.json",
    "frontend/src/ward/wardMachine.js", "frontend/src/ward/wardResult.js", "frontend/src/ward/wardSettings.js",
    "frontend/src/pages/WardPage.jsx", "frontend/src/pages/NursePage.jsx", "frontend/src/pages/EnrollPage.jsx",
    "frontend/src/api/client.js", "backend/api/routes.py", "backend/api/infer.py", "backend/schemas/models.py",
)
# 화면 코드는 폴더째 지문을 남긴다(등록 화면·설정·결과 해석 파일 이름이 늘어도 빠지지 않게). 저장소 기준 glob.
CODE_GLOBS = ("frontend/src/ward/*.js", "frontend/src/ward/*.jsx", "frontend/src/**/*[Ee]nroll*.js",
              "frontend/src/**/*[Ee]nroll*.jsx")
CONFLICT_RULE = ("글자 규칙을 통과하면 등록 거리와 관계없이 글자 1등을 고른다. 통과하지 못했고 등록 1등 거리가 D 이하일 때만 "
                 "글자 1등과 등록 1등이 같은지 보고(require_agreement), 같으면 등록 1등을 고른다. 고르지 못했을 때 글자 1·2등에 "
                 "긴급이 있거나 거리 D 이하인 등록 1등이 긴급이면 긴급 가능성이다.")


def code_paths() -> list[str]:
    """지문을 남길 코드 파일 (CODE_FILES + CODE_GLOBS에 맞는 파일, 저장소 기준 경로, 겹침 없이 정렬)."""
    rels = {rel for rel in CODE_FILES if (ROOT / rel).is_file()}
    try:
        for pattern in CODE_GLOBS:
            rels.update(p.relative_to(ROOT).as_posix() for p in ROOT.glob(pattern) if p.is_file())
    except (OSError, ValueError) as exc:
        raise OSError(f"코드 파일 목록을 만들지 못했습니다: {exc}") from exc
    return sorted(rels)


def code_fingerprints() -> dict[str, str]:
    """코드 파일(code_paths)의 SHA-256 (없는 파일은 뺀다)."""
    return {rel: common.file_sha256(ROOT / rel) for rel in code_paths()}


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
        mine = bases[-len(rows):] if rows else []
        enr_head = head.get("enroll") if isinstance(head.get("enroll"), dict) else None
        info.append({"path": str(p), "sha256": common.file_sha256(p), "schema": head.get("schema"),
                     "enroll": ({k: enr_head.get(k) for k in ("profile", "mode", "run_id", "profile_sha256", "deleted",
                                                              "status_before", "status_after")} if enr_head else None),
                     "service_enroll_rows": {m: sum(1 for b in mine if b.get("svc_enroll_mode") == m)
                                             for m in ("store", "match")},
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


def resolve_enroll_source(choice: str, bases: list[dict[str, Any]], n_enroll_files: int) -> str:
    """등록 거리 출처를 하나로 정한다 (서버 등록과 오프라인 등록을 섞지 않는다).

    auto: 결과 행에 서버 등록 결과가 있으면 service, --enroll 파일이 있으면 offline, 둘 다 있으면 고르라고 거부, 둘 다 없으면 none.

    Raises:
        ValueError: 둘 다 있는데 auto이거나, 고른 출처의 자료가 없을 때
    """
    svc = mt.has_service_enroll(bases)
    off = n_enroll_files > 0
    if choice == "auto":
        if svc and off:
            raise ValueError("결과 행에 서버 등록 결과가 있고 --enroll(오프라인 등록) 파일도 있습니다. 두 출처를 섞지 않으므로 "
                             "--enroll-source service 또는 offline을 고르세요")
        return "service" if svc else ("offline" if off else "none")
    if choice == "service" and not svc:
        raise ValueError("--enroll-source service인데 결과 행에 서버 등록 결과(ward_service_batch.py --enroll-profile)가 없습니다")
    if choice == "offline" and not off:
        raise ValueError("--enroll-source offline인데 --enroll 파일이 없습니다")
    if choice not in mt.ENROLL_SOURCES:
        raise ValueError(f"알 수 없는 등록 거리 출처: {choice}")
    return choice


ENROLL_SOURCE_KO = {"service": "서버 등록(앱 경로)", "offline": "오프라인 등록(ward_enroll.py)", "none": "등록 없음"}


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
        source = resolve_enroll_source(args.enroll_source, bases, len(args.enroll or []))
        if args.D is not None and source == "none":
            raise ValueError("--D를 주었지만 등록 거리가 없습니다(서버 등록 결과 행 또는 --enroll 파일이 필요)")
    except (common.SealedDataError, ValueError, OSError) as exc:
        print(f"시작하지 않았습니다: {exc}", file=sys.stderr)
        return 2
    rule = rule_from_args(args, commands)
    for problem in mt.profile_problems(bases):
        print(f"주의: {problem}", file=sys.stderr)
    bases = mt.use_enroll_source(bases, source)
    sel = mt.enroll_population(bases, source) if rule.max_distance is not None else bases
    summary = mt.summarize_groups(mt.evaluate_all(sel, rule), rule, commands)
    used = summary["all"]["enroll_sources"]
    md = rp.table(summary["by_group"], f"기준 {rule.min_margin} / {rule.min_score}"
                  + (f", D {rule.max_distance} (등록 거리 출처: {ENROLL_SOURCE_KO[source]}, 등록 1등을 본 클립 "
                     f"서버 {used['service']} · 오프라인 {used['offline']} · 없음 {used['none']})"
                     if rule.max_distance is not None else "")
                  + (f", 긴급 가능성 하한 {rule.possible_min_score}" if rule.possible_min_score is not None else ""))
    print(md)
    if args.out:
        common.atomic_write_json(args.out, {"schema": "ward-metrics-v1", "created": datetime.now(timezone.utc).isoformat(),
                                            "rule": rule.describe(), "enroll_source": source, "inputs": info,
                                            "freeze": gate, "profile_problems": mt.profile_problems(bases),
                                            "summary": summary})
    return 0


# ---------------------------------------------------------------------------
# calibrate
# ---------------------------------------------------------------------------

def enrollment_block(bases: list[dict[str, Any]], rule: mt.Rule, enroll_info: list[dict[str, Any]],
                     opts: dict[str, Any], notes: list[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    """동결 초안의 enrollment 칸과 근거. 템플릿 방식·특징·거리 이름이 등록 파일마다 같아야 D를 정한다.

    opts['enroll_source']: 'service'(서버 등록 거리, bases는 use_enroll_source(.., 'service')를 거친 기록),
    'offline'(ward_enroll.py 파일, 기본), 'none'. 서버 등록은 템플릿 방식 voiced, 특징·거리 이름은 ward_enroll.py와 같다.
    """
    source = opts.get("enroll_source") or ("offline" if enroll_info else "none")
    block: dict[str, Any] = {"max_distance": None, "template": None, "feature": common.FEATURE_NAME,
                             "distance": common.DISTANCE_NAME, "source": None if source == "none" else source,
                             "require_agreement": True, "conflict_rule": CONFLICT_RULE,
                             "unbounded": None, "min_negatives_with_dist": opts.get("min_negatives", cal.MIN_NEGATIVES_FOR_D),
                             "negatives_with_dist": None, "chosen_by": "등록 거리 자료가 없어 정하지 않음"}
    if source == "none" or not any(b.get("tmpl_dists") for b in bases):
        if source == "service":
            notes.append("서버 등록 결과에 거리가 있는 클립이 없어(등록 미완성 등) D를 정하지 않았다.")
        elif enroll_info:
            notes.append("등록 파일에 등록 거리가 있는 클립이 없어(병실 묶음의 등록 템플릿 없음) D를 정하지 않았다.")
        return block, {}
    if source == "service":
        templates = [common.SERVICE_TEMPLATE]
        names = [(common.FEATURE_NAME, common.DISTANCE_NAME)]
    else:
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
          folder_checks, sealed_folders, enroll_source('service'·'offline'·'none', 기본은 등록 파일이 있으면 offline)
    full_list.env에는 서버 환경 변수 4개(M·S·긴급 가능성 하한·등록 거리 D, 없으면 'none')를 쓴다.
    """
    emergency = commands["emergency"]
    prov_rule = cal.rule_at(*cal.PROVISIONAL, emergency)
    notes: list[str] = []
    lopo = cal.lopo_thresholds(bases, emergency)
    pooled = cal.pooled_thresholds(bases, emergency, opts.get("deltas", cal.DEFAULT_DELTAS))
    prop = cal.choose_proposal(pooled, opts.get("max_loss", 0.10), opts.get("delta"))
    rule = cal.rule_at(prop["M"], prop["S"], emergency)
    # 긴급 가능성 점수 하한: 화면이 서버 값(phrase.rule.possible_min_score)을 쓴다. 근거로는 늘 계산하고 --floor일 때만 넣는다
    floor_block: dict[str, Any] = {"possible_min_score": None, "floor_implemented_in_ui": True,
                                   "chosen_by": "기본: 기획 9.3 그대로(점수 하한 없음: 고르지 못했을 때 1·2등에 긴급이면 긴급 가능성)"}
    floor_lopo = cal.lopo_possible_floor(bases, rule, opts.get("floor_clearance", 0.2))
    if opts.get("use_floor", False):
        floors = {k: (v["fit"]["floor"] if k != "pooled" else v["floor"]) for k, v in floor_lopo.items()}
        floors = {k: v for k, v in floors.items() if v is not None}
        if floors:
            key = min(floors, key=lambda k: floors[k])
            floor_block.update(possible_min_score=round(floors[key], 3),
                               chosen_by=f"긴급 명령 클립을 하나도 놓치지 않는 하한에서 여유 {opts.get('floor_clearance', 0.2)}를 "
                                         f"뺀 값 중 가장 낮은 값({key})")
            notes.append("긴급 가능성 점수 하한을 넣었다. 서버는 full_list.env의 LIPNET_OLKAVS_WARD_POSSIBLE_MIN_SCORE로 켜고, "
                         "화면은 서버가 결과에 적는 하한(phrase.rule.possible_min_score)으로 판정한다.")
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
    expected = {"min_margin": prop["M"], "min_score": prop["S"], "possible_min_score": floor_block["possible_min_score"],
                "enroll_max_distance": enr_block["max_distance"]}
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
                      "env": common.server_env(expected),
                      "env_note": "서버 환경 변수. 기준값·하한·D를 고치면 env도 같게 고친다(validate가 확인, ward_eval_tools.py env로 출력)"},
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
        source = resolve_enroll_source(args.enroll_source, bases, len(args.enroll or []))
        problems = mt.profile_problems(bases)
        if problems and source == "service":
            raise ValueError("서버 등록 결과가 사람마다 등록 하나·같은 등록 상태가 아닙니다: " + "; ".join(problems))
        bases = mt.use_enroll_source(bases, source)
        sealed_folders = sealed_folder_fingerprints(args.sealed_folder or [])
    except (common.SealedDataError, ValueError, OSError) as exc:
        print(f"시작하지 않았습니다: {exc}", file=sys.stderr)
        return 2
    opts = {"target_size": args.target_size, "keep": args.keep or [], "max_loss": args.max_loss, "delta": args.delta,
            "d_rule": args.d_rule, "min_negatives": args.min_negatives, "floor_clearance": args.floor_clearance,
            "use_floor": bool(args.floor), "sealed_folders": sealed_folders, "enroll_source": source,
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
    """동결 규칙을 기록에 적용한 결과 (주 결과, 비교, 오프라인 등록, 축소 목록).

    주 결과(full)는 앱과 같은 판정이다: 글자 규칙 + 동결 파일의 긴급 가능성 하한(있으면) + 결과 행에 서버 등록 결과가 있고
    동결 D가 있으면 서버 등록 거리(등록 영상 = store 행은 뺀다, 거리가 없는 클립은 앱처럼 글자 규칙만).
    비교: full_text_only(같은 클립, 등록 없이), full_no_floor(하한 없이), all_rows_text(등록 영상까지 모든 클립, 글자 규칙).
    오프라인 등록(ward_enroll.py, --enroll 파일)은 enroll·enroll_text_only(보조)로 따로 내며 서버 등록과 섞지 않는다.
    """
    app = cal.rule_from_freeze(freeze, "full")  # 글자 규칙 + 하한(동결 값)
    D = (freeze.get("enrollment") or {}).get("max_distance")
    floor = (freeze.get("decision_rule") or {}).get("possible_min_score")
    svc = mt.has_service_enroll(bases)
    sb = mt.use_enroll_source(bases, "service" if svc else "none")
    pop = mt.enroll_population(sb, "service") if svc else sb
    main_rule = cal.rule_from_freeze(freeze, "full", with_enrollment=True) if (svc and D is not None) else app
    main_evals = mt.evaluate_all(pop, main_rule)
    out: dict[str, Any] = {"full": mt.summarize_groups(main_evals, main_rule, commands)}
    out["main"] = {"rule": main_rule.describe(), "enrollment_source": "service" if (svc and D is not None) else None,
                   "service_enroll_rows": svc, "templates_excluded": len(sb) - len(pop), "n": len(pop),
                   "profile_problems": mt.profile_problems(bases)}
    out["server_rule_check"] = mt.rule_check(main_evals, main_rule, common.expected_server_rule(freeze))
    if main_rule.max_distance is not None:
        out["full_text_only"] = mt.summarize_groups(mt.evaluate_all(pop, app), app, commands)
    if floor is not None:
        nf = replace(main_rule, possible_min_score=None)
        out["full_no_floor"] = mt.summarize_groups(mt.evaluate_all(pop, nf), nf, commands)
    if len(pop) < len(sb):
        out["all_rows_text"] = mt.summarize_groups(mt.evaluate_all(sb, app), app, commands)
    ob = mt.use_enroll_source(bases, "offline")
    tests = mt.enroll_population(ob, "offline")
    if tests and D is not None:
        er = cal.rule_from_freeze(freeze, "full", with_enrollment=True)
        out["enroll"] = mt.summarize_groups(mt.evaluate_all(tests, er), er, commands)
        out["enroll_text_only"] = mt.summarize_groups(mt.evaluate_all(tests, app), app, commands)
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
        prof = mt.profile_problems(bases)
        if prof:
            raise ValueError("서버 등록 결과가 사람마다 등록 하나·같은 등록 상태가 아닙니다: " + "; ".join(prof))
    except (common.SealedDataError, ValueError, OSError) as exc:
        print(f"시작하지 않았습니다: {exc}", file=sys.stderr)
        return 2
    now_code = code_fingerprints()
    changed = sorted(k for k, v in (freeze.get("code") or {}).get("files", {}).items() if now_code.get(k) != v)
    result = run_report(freeze, bases, commands)
    full = freeze["full_list"]
    check = result["full"]["all"].get("server_accepted_check") or {}
    echeck = result["full"]["all"].get("server_enroll_check") or {}
    rcheck = result.get("server_rule_check") or {}
    result.update({
        "schema": "ward-report-v1", "created": datetime.now(timezone.utc).isoformat(),
        "freeze": {"path": str(freeze_path), "status": freeze.get("status"), "commit": state.get("commit"),
                   "sha256": freeze_sha,
                   "full": {"min_margin": full["min_margin"], "min_score": full["min_score"],
                            "possible_min_score": (freeze.get("decision_rule") or {}).get("possible_min_score"),
                            "floor_implemented_in_ui": (freeze.get("decision_rule") or {}).get("floor_implemented_in_ui")},
                   "max_distance": (freeze.get("enrollment") or {}).get("max_distance"),
                   "template": (freeze.get("enrollment") or {}).get("template"),
                   "enroll_source": (freeze.get("enrollment") or {}).get("source"),
                   "env": (freeze.get("full_list") or {}).get("env"),
                   "reduced_ids": (freeze.get("reduced_list") or {}).get("ids")},
        "inputs": {"rows_files": [f["path"] for f in info], "rows_info": info, "enroll_files": einfo,
                   "folder_checks": [] if args.skip_folder_check else folder_check(info)},
        "code_changed": changed, "sealed_data": bool(reasons),
        "server_threshold_mismatch": int(check.get("mismatch") or 0),
        "server_rule_mismatch": int(rcheck.get("mismatch") or 0),
        "server_enroll_mismatch": int(echeck.get("mismatch") or 0) + int(echeck.get("max_distance_mismatch") or 0)
        + int(echeck.get("top_inconsistent") or 0)})
    md = rp.markdown_report(result)
    if args.out:
        common.atomic_write_json(args.out, result)
        Path(args.out).with_suffix(".md").write_text(md, encoding="utf-8")
    print(md)
    bad = {k: result[k] for k in ("server_threshold_mismatch", "server_rule_mismatch", "server_enroll_mismatch") if result[k]}
    if bad and reasons:
        print(f"주의: 서버가 동결 기준값과 다르게 판정·기록한 클립이 있습니다({bad}). 봉인 자료 결과를 주 결과로 쓰기 전에 "
              "서버를 full_list.env로 켰는지 확인하세요.", file=sys.stderr)
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


def cmd_env(args: argparse.Namespace) -> int:
    """동결 파일의 서버 환경 변수 4개(full_list.env와 같아야 함)를 PowerShell 줄로 출력한다."""
    try:
        freeze = common.read_json(args.freeze)
        problems = cal.validate_freeze(freeze, allow_synthetic=True)
        env = cal.freeze_env(freeze)
    except ValueError as exc:
        print(f"환경 변수를 만들지 못했습니다: {exc}", file=sys.stderr)
        return 2
    for name, value in env.items():
        print(f'$env:{name} = "{value}"')
    if problems:
        print("동결 파일 형식 문제: " + "; ".join(problems), file=sys.stderr)
        return 1
    return 0


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
    p.add_argument("--D", type=float, default=None,
                   help="등록 거리 D (서버 등록: 등록 영상 행만 뺀 모든 클립, 오프라인 등록: 거리 있는 시험 클립만)")
    p.add_argument("--enroll-source", choices=("auto",) + mt.ENROLL_SOURCES, default="auto",
                   help="등록 거리 출처 (auto: 서버 등록 결과 행이면 service, --enroll 파일이면 offline, 둘 다면 거부)")
    p.add_argument("--no-agreement", action="store_true")
    p.add_argument("--allow-sealed", action="store_true", help="봉인 자료도 계산 (--freeze 필요)")
    p.add_argument("--freeze", help="커밋된 동결 파일 (--allow-sealed와 함께 필수)")
    p.add_argument("--out")
    p.set_defaults(func=cmd_metrics)
    p = sub.add_parser("calibrate", help="조정 자료로 동결 초안")
    p.add_argument("--tune-rows", action="append", required=True, help="P01·P03 ward_tune 결과 행")
    p.add_argument("--negative-rows", action="append", help="문장 촬영 클립을 phrase_set=ward로 채점한 결과 행")
    p.add_argument("--enroll", action="append", help="ward_enroll.py apply 출력 (후보별 점수·등록 거리)")
    p.add_argument("--enroll-source", choices=("auto",) + mt.ENROLL_SOURCES, default="auto",
                   help="D를 정할 등록 거리 출처 (auto: 서버 등록 결과 행이면 service, --enroll 파일이면 offline, 둘 다면 거부)")
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
                   help="긴급 가능성 점수 하한을 초안에 넣음 (기본 없음. 서버 환경 변수로 켜고 화면은 서버 값을 씀)")
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
    p = sub.add_parser("env", help="동결 파일의 서버 환경 변수 (PowerShell)")
    p.add_argument("freeze")
    p.set_defaults(func=cmd_env)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
