"""팀원 웹캠 파일럿 클립을 OLKAVS로 일괄 평가한다 (3단계).

입력 폴더에는 CapturePage가 저장한 `{가명}_{조건}_{문장ID}_{회차}.webm|mp4` 와 같은 이름의 `.json`(정답
문장·조건·프레임 로그)이 있다. 두 가상환경을 서브프로세스로 묶는다.

  check   : 아무 Python      모델 없이 폴더 점검(문장 목록 일치, 촬영 묶음 섞임, 참가자별 클립 수, 자유 문장 받아쓰기)
  roi-all : .venv-olkavs-roi  모든 클립 → 입술 프레임 NPZ (실패 사유 기록)
  score   : .venv-olkavs      모델 1회 로드 → CTC/hybrid 텍스트, 지정 20문장 후보 점수, 집계표
  run     : 위 둘을 순서대로 실행

촬영 묶음(클립 .json의 session)이 ward_로 시작하면(병실 명령 ward_tune·ward_eval) 문장 목록은
frontend/src/capture/ward_prompts.json(명령은 frontend/src/ward/commands.json과 같아야 함)을 쓰고, 그 밖은 prompts.json을 쓴다.
병실 명령 묶음은 check·roi-all까지 지원한다. score·run(병실 명령 후보 채점)은 아직 지원하지 않으며 입술 추출 전에 거부한다.
병실 명령 평가용(ward_eval)은 봉인 자료라 동결 파일을 만들기 전에는 check만 돌린다. roi-all은 --allow-sealed가 있어야 실행한다.

사용 예:
  py -3.11 scripts/pilot_evaluate.py check --clips D:/silent_talk_webcam/eval
  python scripts/pilot_evaluate.py run --clips D:/silent_talk_webcam/pilot --out experiments/webcam_pilot_20261022 --device cpu

보고 지표(조건별·문장 묶음별): NFC 문자 CER, 완전 일치, 지정 문장 후보 top-1(길이만 보는 분류기와 비교),
무발화 오수락률, 입술 검출 성공률, 처리 시간. 자유 문장은 .json 에 `spoken`(실제 말한 문장)이 있을 때만 CER을 낸다.
이 수치는 튜닝 참가자 파일럿이며 최종 평가가 아니다.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import olkavs_transcribe as ot  # noqa: E402

VIDEO_SUFFIXES = {".webm", ".mp4"}
PROMPTS_PATH = ROOT / "frontend" / "src" / "capture" / "prompts.json"
WARD_PROMPTS_PATH = ROOT / "frontend" / "src" / "capture" / "ward_prompts.json"
WARD_COMMANDS_PATH = ROOT / "frontend" / "src" / "ward" / "commands.json"
WARD_KINDS = ("fixed", "decoy", "short", "nonspeech")
# 동결 파일 전에는 check 외에 아무것도 돌리지 않는 봉인 묶음 (병실 명령 평가용, 기획 문서 10.1)
SEALED_SESSIONS = ("ward_eval",)
WARD_SCORE_UNSUPPORTED = "병실 명령 묶음(ward_*) 채점은 아직 지원하지 않습니다. check·roi-all만 실행할 수 있습니다."


def load_clips(clips_dir: Path) -> list[dict[str, Any]]:
    """영상과 짝이 맞는 메타 JSON을 읽어 클립 목록을 만든다."""
    rows = []
    for video in sorted(clips_dir.iterdir()):
        # macOS가 USB(exFAT/FAT32)에 만드는 '._이름' 보조 파일은 클립이 아니다.
        if video.suffix.lower() not in VIDEO_SUFFIXES or video.name.startswith("._"):
            continue
        meta_path = video.with_suffix(".json")
        if not meta_path.exists():
            rows.append({"file": video.name, "status": "missing_meta"})
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            rows.append({"file": video.name, "status": "bad_meta", "error": str(exc)})
            continue
        rows.append({"file": video.name, "path": str(video), "status": "ok", "participant": meta.get("participant"),
                     "condition": meta.get("condition"), "sentence_id": meta.get("sentenceId"), "kind": meta.get("kind"),
                     "rep": meta.get("rep"), "text": meta.get("text"), "spoken": meta.get("spoken"),
                     "duration_ms": meta.get("durationMs"), "frame_callbacks": meta.get("frameCallbacks"),
                     "estimated_fps": meta.get("estimatedFps"), "track": meta.get("trackSettings"),
                     "session": meta.get("session"), "repeats": meta.get("repeats"),
                     "prompts_version": meta.get("promptsVersion"), "posture": meta.get("posture")})
    return rows


def is_ward_session(session: Any) -> bool:
    """병실 명령 촬영 묶음(ward_tune·ward_eval 등 ward_로 시작)인지."""
    return isinstance(session, str) and session.startswith("ward_")


def load_ward_prompts() -> dict[str, Any]:
    """ward_prompts.json을 읽고 명령(fixed)의 id·text가 commands.json과 같은지 확인한다.

    촬영 화면은 명령 문장을 commands.json에서 직접 읽으므로, 두 파일이 어긋나면 촬영 당시 문장과 점검 기준이 달라진다.
    """
    try:
        ward = json.loads(WARD_PROMPTS_PATH.read_text(encoding="utf-8"))
        commands = json.loads(WARD_COMMANDS_PATH.read_text(encoding="utf-8"))
        same = [(p["id"], p["text"]) for p in ward["fixed"]] == [(p["id"], p["text"]) for p in commands["fixed"]]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"병실 명령 목록을 읽지 못했습니다: {exc}") from exc
    if not same:
        raise ValueError("ward_prompts.json의 명령(fixed)이 frontend/src/ward/commands.json과 다릅니다. 두 파일을 맞추세요.")
    return ward


def prompts_for_rows(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], str]:
    """클립의 촬영 묶음으로 문장 목록 파일을 고른다 (ward_* → ward_prompts.json, 그 밖 → prompts.json).

    병실 명령 클립과 문장 촬영 클립(묶음 기록이 없는 v1 포함)이 한 폴더에 있으면 어느 목록으로도 점검할 수 없으므로 거부한다.
    Returns:
        (문장 목록, 파일 이름)
    """
    try:
        sessions = {r.get("session") or "pilot" for r in rows if r.get("status") == "ok"}
    except AttributeError as exc:
        raise ValueError(f"클립 목록 형식 오류: {exc}") from exc
    if any(is_ward_session(s) for s in sessions):
        if len(sessions) > 1:
            raise ValueError(f"촬영 묶음이 섞여 있습니다({', '.join(sorted(sessions))}). 묶음별 폴더를 따로 평가하세요.")
        return load_ward_prompts(), WARD_PROMPTS_PATH.name
    return json.loads(PROMPTS_PATH.read_text(encoding="utf-8")), PROMPTS_PATH.name


def check_clip_prompts(rows: list[dict[str, Any]], fixed: list[dict[str, Any]],
                       others: list[dict[str, Any]] | None = None, source: str = "prompts.json") -> None:
    """클립을 찍을 때의 문장이 지금 목록(문장 촬영은 prompts.json, 병실 명령은 ward_prompts.json)과 같은지,
    촬영 묶음이 섞이지 않았는지 확인한다.

    후보 점수는 지금 목록의 문장으로 매기므로, 문장 목록이 바뀐 뒤의 옛 클립을 섞으면 조용히 틀린 표가 나온다.
    파일럿·조정용과 최종 평가용(봉인)을 한 번에 돌리면 조정 자료와 평가 자료가 섞이므로 거부한다.
    문장 촬영은 fixed에 prompts.json의 지정 문장을 준다. 병실 명령 묶음은 fixed에 명령(ward_prompts.json의 fixed),
    others에 그 묶음의 함정·짧은 말 목록, source에 ward_prompts.json을 준다(_check_prompts가 고른다).
    """
    try:
        fixed_text = {p["id"]: p["text"] for p in fixed}
        other_text = {p["id"]: p["text"] for p in (others or [])}
        stale = sorted(r["file"] for r in rows
                       if r.get("kind") == "fixed" and fixed_text.get(r.get("sentence_id")) != r.get("text"))
        stale += sorted(r["file"] for r in rows
                        if r.get("kind") in ("decoy", "short") and other_text.get(r.get("sentence_id")) != r.get("text"))
        sessions = sorted({r["session"] for r in rows if r.get("session")})
    except (KeyError, TypeError) as exc:
        raise ValueError(f"클립 목록 형식 오류: {exc}") from exc
    if stale:
        raise ValueError(f"촬영 당시 지정 문장이 지금 {source}과 다른 클립 {len(stale)}개: {stale[:5]}")
    if len(sessions) > 1:
        raise ValueError(f"촬영 묶음이 섞여 있습니다({', '.join(sessions)}). 묶음별 폴더를 따로 평가하세요.")


def ward_other_prompts(ward: dict[str, Any], session: str | None) -> list[dict[str, Any]]:
    """병실 명령 묶음에서 찍는 함정·짧은 말 목록 (조정용은 decoys.tune, 평가용은 decoys.eval).

    묶음 구성(sessions)에 없는 묶음이면 빈 목록이며, 그 묶음의 함정·짧은 말 클립은 문장 불일치로 걸린다.
    """
    spec = (ward.get("sessions") or {}).get(session or "") or {}
    decoys = (ward.get("decoys") or {}).get(spec.get("decoys"), []) if spec.get("decoys") else []
    return list(decoys) + (list(ward.get("short") or []) if spec.get("short") else [])


def ward_session_of(rows: list[dict[str, Any]]) -> str | None:
    """클립 목록에서 처음 보이는 병실 명령 묶음 ID (없으면 None = 문장 촬영 폴더)."""
    try:
        return next((r.get("session") for r in rows if is_ward_session(r.get("session"))), None)
    except AttributeError as exc:
        raise ValueError(f"클립 목록 형식 오류: {exc}") from exc


def _check_prompts(rows: list[dict[str, Any]], prompts: dict[str, Any], ward_session: str | None) -> None:
    """묶음에 맞는 목록으로 check_clip_prompts를 부른다 (check_rows·folder_report 공용).

    문장 촬영(ward_session None)은 prompts.json의 지정 문장, 병실 명령 묶음은 명령과 그 묶음의 함정·짧은 말로 본다.
    """
    try:
        if ward_session is None:
            check_clip_prompts(rows, prompts["fixed"])
        else:
            check_clip_prompts(rows, prompts["fixed"], ward_other_prompts(prompts, ward_session), WARD_PROMPTS_PATH.name)
    except (KeyError, TypeError) as exc:
        raise ValueError(f"문장 목록 형식 오류: {exc}") from exc


def check_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """폴더 클립의 묶음에 맞는 목록을 골라 check_clip_prompts로 검사하고 그 목록을 돌려준다."""
    prompts, _ = prompts_for_rows(rows)
    _check_prompts(rows, prompts, ward_session_of(rows))
    return prompts


WardKey = tuple[str, str, str, int, str]  # (종류, 조건, 문장 ID, 회차, 자세)


def ward_clip_key(row: dict[str, Any]) -> tuple[Any, ...]:
    """클립 기록의 (종류, 조건, 문장 ID, 회차, 자세). ward_plan_keys의 항목과 비교한다."""
    try:
        return (row.get("kind"), row.get("condition"), row.get("sentence_id"), row.get("rep"), row.get("posture"))
    except AttributeError as exc:
        raise ValueError(f"클립 목록 형식 오류: {exc}") from exc


def ward_plan_keys(ward: dict[str, Any], session: str) -> list[WardKey]:
    """병실 명령 묶음 한 사람의 촬영 계획 (종류, 조건, 문장 ID, 회차, 자세) 목록, 촬영 순서대로.

    촬영 화면(frontend/src/capture/plan.js의 buildWardPlan)과 같은 구성이다: 명령 × 조건별 반복(서 있는 자세)
    → 함정·짧은 말(소리 없이) → 말하지 않는 동작(조건 none) → 구성에 reclined가 있으면 기댄 자세 명령(조건 reclined).
    Raises:
        ValueError: ward_prompts.json에 그 묶음 구성이 없거나 형식이 잘못됐을 때
    """
    try:
        spec = ward["sessions"][session]
        keys: list[WardKey] = [("fixed", str(cond), p["id"], rep, "upright")
                               for cond, reps in spec["repeats"].items() for rep in range(1, int(reps) + 1)
                               for p in ward["fixed"]]
        extras: list[tuple[str, str, list[dict[str, Any]]]] = []
        if spec.get("decoys"):
            extras.append(("decoy", "silent", ward["decoys"][spec["decoys"]]))
        if spec.get("short"):
            extras.append(("short", "silent", ward.get("short") or []))
        if spec.get("nonspeech"):
            extras.append(("nonspeech", "none", ward.get("nonspeech") or []))
        keys += [(kind, cond, p["id"], 1, "upright") for kind, cond, items in extras for p in items]
        if spec.get("reclined"):
            fixed_ids = {p["id"] for p in ward["fixed"]}
            unknown = [cid for cid in ward.get("reclined") or [] if cid not in fixed_ids]
            if unknown:
                raise ValueError(f"기댄 자세 명령 {unknown}가 명령 목록에 없습니다")
            keys += [("fixed", "reclined", cid, 1, "reclined") for cid in ward.get("reclined") or []]
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise ValueError(f"ward_prompts.json에 촬영 묶음 '{session}'의 구성이 없거나 잘못됐습니다: {exc}") from exc
    return keys


def ward_expected(ward: dict[str, Any], session: str) -> dict[str, int]:
    """병실 명령 묶음 한 사람의 종류별 기대 클립 수 (ward_plan_keys의 촬영 계획으로 센다).

    명령(fixed) = 명령 수 × 조건별 반복 합 + 기댄 자세 명령 수(조정용), 함정·짧은 말·말하지 않기는 목록 길이.
    Raises:
        ValueError: 묶음 구성이 없거나 구성으로 센 클립 수가 sessions의 clips와 다를 때
    """
    keys = ward_plan_keys(ward, session)
    counts_by_kind = {kind: sum(1 for k in keys if k[0] == kind) for kind in WARD_KINDS}
    try:
        clips = ward["sessions"][session].get("clips")
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f"ward_prompts.json에 촬영 묶음 '{session}'의 구성이 없거나 잘못됐습니다: {exc}") from exc
    if sum(counts_by_kind.values()) != clips:
        raise ValueError(f"ward_prompts.json의 '{session}' 구성으로 센 클립 {sum(counts_by_kind.values())}개가 "
                         f"clips {clips}와 다릅니다")
    return counts_by_kind


def _ward_person_report(person: str, mine: list[dict[str, Any]], plan_keys: list[WardKey] | None,
                        expected_by_kind: dict[str, int] | None) -> tuple[dict[str, Any], list[str]]:
    """병실 명령 묶음 한 사람의 클립 수·종류·자세를 촬영 계획과 비교한다. (참가자 요약, 경고 목록)을 돌려준다."""
    try:
        by_kind = {k: sum(1 for r in mine if r.get("kind") == k) for k in WARD_KINDS}
        by_posture = {p: sum(1 for r in mine if r.get("posture") == p) for p in ("upright", "reclined")}
        present = {ward_clip_key(r) for r in mine}
        missing = [f"{person}_{k[1]}_{k[2]}_{k[3]}" for k in (plan_keys or []) if k not in present]
    except (TypeError, AttributeError) as exc:
        raise ValueError(f"클립 목록 형식 오류: {exc}") from exc
    summary = {"clips": len(mine), "expected": sum(expected_by_kind.values()) if expected_by_kind else None,
               "by_kind": by_kind, "expected_by_kind": expected_by_kind, "by_posture": by_posture,
               "missing": len(missing) if plan_keys is not None else None}
    warnings: list[str] = []
    if expected_by_kind is not None and by_kind != expected_by_kind:
        warnings.append(f"{person}: 종류별 클립 수가 촬영 계획과 다릅니다(지금 {by_kind}, 계획 {expected_by_kind})")
    if plan_keys is not None:
        reclined_expected = sum(1 for k in plan_keys if k[4] == "reclined")
        if by_posture["reclined"] != reclined_expected:
            warnings.append(f"{person}: 기댄 자세 클립 {by_posture['reclined']}/{reclined_expected}개")
        if missing:
            warnings.append(f"{person}: 빠진 클립 {len(missing)}개: {missing[:5]}")
    return summary, warnings


def folder_report(rows: list[dict[str, Any]], prompts: dict[str, Any]) -> dict[str, Any]:
    """모델 없이 촬영 폴더를 점검한 결과. problems가 있으면 평가에 쓸 수 없고, warnings는 채워야 할 것이다.

    병실 명령 묶음(ward_*)이면 prompts는 ward_prompts.json이며, 기대 클립 수는 그 묶음 구성(93·103 등)으로 센다.
    촬영 계획에 없는 (종류, 조건, 문장 ID, 회차, 자세)의 클립은 problems, 빠진 클립·종류·기댄 자세 수 차이는 warnings다.
    """
    problems: list[str] = []
    ward_session = ward_session_of(rows)
    try:
        _check_prompts(rows, prompts, ward_session)
    except ValueError as exc:
        problems.append(str(exc))
    ok = [r for r in rows if r.get("status") == "ok"]
    problems += [f"메타 JSON 없음/깨짐: {r['file']}" for r in rows if r.get("status") != "ok"]
    keys: dict[tuple, list[str]] = defaultdict(list)
    for r in ok:
        keys[(r.get("participant"), r.get("condition"), r.get("sentence_id"), r.get("rep"))].append(r["file"])
    problems += [f"같은 클립이 여러 개: {files}" for files in keys.values() if len(files) > 1]
    expected_by_kind: dict[str, int] | None = None
    plan_keys: list[WardKey] | None = None
    if ward_session is not None:
        try:
            plan_keys = ward_plan_keys(prompts, ward_session)
            expected_by_kind = ward_expected(prompts, ward_session)
        except ValueError as exc:
            problems.append(str(exc))
        if plan_keys is not None:
            allowed = set(plan_keys)
            off_plan = sorted(r["file"] for r in ok if ward_clip_key(r) not in allowed)
            if off_plan:
                problems.append(f"'{ward_session}' 촬영 계획에 없는 클립 {len(off_plan)}개(종류·조건·회차·자세·문장 ID 확인): "
                                f"{off_plan[:5]}")
    nonspeech = prompts.get("nonspeech") or 0
    n_non = len(nonspeech) if isinstance(nonspeech, list) else int(nonspeech)
    per_person: dict[str, dict[str, Any]] = {}
    ward_warnings: list[str] = []
    for person in sorted({str(r.get("participant")) for r in ok}):
        mine = [r for r in ok if str(r.get("participant")) == person]
        if ward_session is not None:
            per_person[person], person_warnings = _ward_person_report(person, mine, plan_keys, expected_by_kind)
            ward_warnings += person_warnings
            continue
        reps = max((int(r.get("repeats") or 2) for r in mine), default=2)
        expected = len(prompts["fixed"]) * 2 * reps + len(prompts["free"]) + n_non
        per_person[person] = {"clips": len(mine), "expected": expected,
                              "by_kind": {k: sum(1 for r in mine if r.get("kind") == k) for k in ("fixed", "free", "nonspeech")}}
    warnings = [f"{p}: {v['clips']}/{v['expected']}클립" for p, v in per_person.items() if v["clips"] != v["expected"]]
    warnings += ward_warnings
    no_spoken = sorted(r["file"] for r in ok if r.get("kind") == "free" and not r.get("spoken"))
    if no_spoken:
        warnings.append(f"자유 문장 받아쓰기(spoken) 없음 {len(no_spoken)}개: {no_spoken[:5]}")
    return {"ok": not problems, "clips": len(rows), "sessions": sorted({r["session"] for r in ok if r.get("session")}),
            "prompts_version": prompts.get("version"), "participants": per_person, "problems": problems, "warnings": warnings}


def cmd_check(args: argparse.Namespace) -> int:
    """모델 없이 촬영 폴더를 점검한다. 최종 평가용 폴더 동결(11/01) 때 인식 없이 확인하는 용도."""
    try:
        clips_dir = Path(args.clips).resolve()
        rows = load_clips(clips_dir)
        try:
            prompts, _ = prompts_for_rows(rows)  # 묶음에 맞는 목록 (ward_* → ward_prompts.json)
        except ValueError as exc:
            report = {"ok": False, "clips": len(rows), "sessions": sorted({r["session"] for r in rows if r.get("session")}),
                      "problems": [str(exc)], "warnings": []}
        else:
            report = folder_report(rows, prompts)
        nested = sorted(d.name for d in clips_dir.iterdir()
                        if d.is_dir() and any(f.suffix.lower() in VIDEO_SUFFIXES for f in d.iterdir()))
        if nested:
            report["warnings"].append(f"하위 폴더 안의 클립은 세지 않았습니다(압축을 풀 때 생긴 폴더?): {nested}. 안의 파일을 이 폴더로 옮기세요.")
    except (OSError, ValueError, KeyError) as exc:
        report = {"ok": False, "problems": [f"점검 실패: {exc}"]}
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0 if report["ok"] else 1


def sealed_refusal(rows: list[dict[str, Any]], allow_sealed: bool) -> str | None:
    """봉인 묶음(ward_eval) 클립이 있고 --allow-sealed가 없으면 거부 문구, 아니면 None.

    기획 문서 10.1: 기준값·축소 목록·커밋을 적은 동결 파일을 만들기 전에는 병실 명령 평가용 폴더에 check 외에는 돌리지 않는다.
    """
    try:
        sealed = sorted({str(r.get("session")) for r in rows if r.get("session") in SEALED_SESSIONS})
    except AttributeError as exc:
        raise ValueError(f"클립 목록 형식 오류: {exc}") from exc
    if not sealed or allow_sealed:
        return None
    return (f"봉인된 평가용 폴더입니다({', '.join(sealed)}). 동결 파일을 만들기 전에는 check만 실행합니다. "
            "동결 파일을 만든 뒤에만 --allow-sealed를 붙여 입술 추출을 실행하세요.")


def cmd_roi_all(args: argparse.Namespace) -> int:
    """모든 클립의 입술 프레임을 추출한다 (.venv-olkavs-roi). 봉인 묶음은 --allow-sealed가 있어야 한다."""
    clips_dir, out = Path(args.clips).resolve(), Path(args.out).resolve()
    rows = load_clips(clips_dir)
    refusal = sealed_refusal(rows, getattr(args, "allow_sealed", False))
    if refusal:
        print(refusal, file=sys.stderr)
        return 2
    (out / "roi").mkdir(parents=True, exist_ok=True)
    check_rows(rows)  # 묶음에 맞는 목록으로 검사 (병실 명령 묶음도 입술 추출은 같다)
    results = []
    for row in rows:
        if row["status"] != "ok":
            results.append(row)
            continue
        stem = Path(row["file"]).stem
        clip_out = out / "roi" / stem
        ns = argparse.Namespace(video=row["path"], out=str(clip_out), start=None, end=None, resample=False)
        t0 = time.perf_counter()
        rc = ot.run_roi(ns)
        info = json.loads((clip_out / "roi.json").read_text(encoding="utf-8"))
        row = {**row, "roi_rc": rc, "roi_seconds": time.perf_counter() - t0, "roi_dir": str(clip_out)}
        if rc == 0:
            row.update(frames=info["selected_frames"]["count"], roi_status="ok",
                       detector=info["detector_stats"], video_info=info["video"], selection=info["selected_frames"])
        else:
            row.update(roi_status="failed", roi_error=info.get("error"))
        results.append(row)
        print(json.dumps({"file": row["file"], "roi": row.get("roi_status"), "frames": row.get("frames"),
                          "error": (row.get("roi_error") or {}).get("message")}, ensure_ascii=False), flush=True)
    ot.save_json(out / "roi_index.json", {"clips_dir": str(clips_dir), "n": len(results), "rows": results})
    return 0


def text_to_olkavs_tokens(text: str, bundle: dict[str, Any]) -> list[int]:
    """문장 → OLKAVS 72토큰 ID. 서비스 지정 문장 모드와 같은 변환을 쓴다."""
    return ot.phrase_tokens(text, bundle)


ctc_forced_score = ot.ctc_forced_score  # 후보 문장 CTC 강제 정렬 점수 (olkavs_transcribe 공용 구현)


def cmd_score(args: argparse.Namespace) -> int:
    """입술 프레임을 인식하고 지정 문장 후보 점수·집계표를 만든다 (.venv-olkavs)."""
    out = Path(args.out).resolve()
    index = json.loads((out / "roi_index.json").read_text(encoding="utf-8"))
    # 병실 명령 묶음을 문장 목록(prompts.json)으로 잘못 채점하지 않게, 모델을 불러오기 전에 거부한다.
    if any(is_ward_session(r.get("session")) for r in index["rows"]):
        print(WARD_SCORE_UNSUPPORTED, file=sys.stderr)
        return 2
    import numpy as np
    import torch
    sys.path.insert(0, str(ot.OLKAVS))
    from evaluation_metrics import counts, aggregate
    prompts = json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))
    fixed = prompts["fixed"]
    check_clip_prompts(index["rows"], fixed)
    device = args.device if (not args.device.startswith("cuda") or torch.cuda.is_available()) else "cpu"
    if device == "cpu":
        torch.set_num_threads(max(1, args.threads))
    bundle = ot.load_olkavs(device)
    model = bundle["model"]
    cand_tokens = {p["id"]: text_to_olkavs_tokens(p["text"], bundle) for p in fixed}
    records = []
    for row in index["rows"]:
        rec = dict(row)
        if row.get("roi_status") != "ok":
            records.append(rec)
            continue
        frames = ot.read_frames(Path(row["roi_dir"]) / "roi.npz")
        res = ot.transcribe_frames(frames, device, ("ctc_greedy", "hybrid"))
        rec["outputs"] = res["outputs"]
        rec["timing"] = res["timing_seconds"]
        # CTC 로그확률로 지정 문장 후보 점수
        video, _ = ot.normalize_frames(frames)
        with torch.inference_mode():
            feats = model.medium(model.encoder(video.to(device), torch.tensor([video.shape[2]], device=device), None, None))
            logits = model.decoder.ctcdecoder(inputs=feats)[0].double().cpu()
        lp = torch.log_softmax(logits, dim=-1)
        scores = {cid: ctc_forced_score(lp, toks, blank=3) for cid, toks in cand_tokens.items()}
        feasible = {k: v for k, v in scores.items() if v["status"] == "ok"}
        ranked = sorted(feasible, key=lambda k: feasible[k]["per_token"], reverse=True)
        ranked_total = sorted(feasible, key=lambda k: feasible[k]["total"], reverse=True)
        # 길이만 보는 기준선: 프레임 수와 후보 토큰 수의 비율이 학습 데이터 평균(약 2.6프레임/토큰, 30fps)에 가장 가까운 후보
        n = len(frames)
        length_pick = min(cand_tokens, key=lambda k: abs(n / max(1, len(cand_tokens[k])) - args.frames_per_token))
        rec["candidates"] = {"ranked_per_token": ranked[:5], "ranked_total": ranked_total[:5], "length_only_pick": length_pick,
                             "margin_per_token": (feasible[ranked[0]]["per_token"] - feasible[ranked[1]]["per_token"]) if len(ranked) > 1 else None,
                             "scores": scores}
        ref = row.get("text") if row.get("kind") == "fixed" else row.get("spoken")
        if ref:
            for m, o in rec["outputs"].items():
                o["metrics"] = counts(ref, o.get("text", ""))
        records.append(rec)
        print(json.dumps({"file": row["file"], "cond": row.get("condition"), "ref": ref,
                          "hybrid": rec["outputs"]["hybrid"].get("text"), "ctc": rec["outputs"]["ctc_greedy"].get("text"),
                          "top1": ranked[0] if ranked else None}, ensure_ascii=False), flush=True)

    summary = summarize(records, fixed)
    session = next((r["session"] for r in index["rows"] if r.get("session")), None)
    who = "최종 평가용(조정에 참여하지 않은 참가자)" if session == "eval" else "튜닝 참가자 파일럿, 최종 평가 아님"
    summary.update({"device": device, "torch": torch.__version__, "checkpoint_sha256": ot.CHECKPOINT_SHA256,
                    "prompts_version": prompts.get("version"), "frames_per_token_baseline": args.frames_per_token,
                    "session": session,
                    "note": f"{who}. AI Hub 문장(A)은 OLKAVS 학습 문구일 수 있으므로 N과 나눠 본다."})
    ot.save_json(out / "pilot_results.json", {"summary": summary, "records": records})
    ot.save_json(out / "pilot_summary.json", summary)
    print(json.dumps(summary["tables"], ensure_ascii=False, indent=1))
    return 0


def summarize(records: list[dict[str, Any]], fixed: list[dict[str, Any]]) -> dict[str, Any]:
    """조건·문장 묶음별 CER/정확/후보 top-1/오수락/검출률 표."""
    sys.path.insert(0, str(ot.OLKAVS))
    from evaluation_metrics import aggregate
    subset = {p["id"]: ("aihub" if p["id"].startswith("A") else "new") for p in fixed}
    ok = [r for r in records if r.get("roi_status") == "ok" and "outputs" in r]
    tables: dict[str, Any] = {}
    groups: dict[str, list] = defaultdict(list)
    for r in ok:
        if r.get("kind") == "fixed":
            groups[f"fixed/{r['condition']}/all"].append(r)
            groups[f"fixed/{r['condition']}/{subset.get(r['sentence_id'], '?')}"].append(r)
        elif r.get("kind") == "free" and r.get("spoken"):
            groups[f"free/{r['condition']}"].append(r)
    for key, rows in sorted(groups.items()):
        entry: dict[str, Any] = {"clips": len(rows)}
        for m in ("hybrid", "ctc_greedy"):
            scored = [{"id": r["file"], "status": "ok", "metrics": r["outputs"][m]["metrics"]} for r in rows if "metrics" in r["outputs"][m]]
            if scored:
                a = aggregate(scored)
                entry[m] = {"cer_percent": round(a["cer_percent"], 2), "exact": a["exact_sentences"], "n": len(scored)}
        if key.startswith("fixed"):
            top1 = sum(1 for r in rows if r["candidates"]["ranked_per_token"][:1] == [r["sentence_id"]])
            top1_total = sum(1 for r in rows if r["candidates"]["ranked_total"][:1] == [r["sentence_id"]])
            top3 = sum(1 for r in rows if r["sentence_id"] in r["candidates"]["ranked_per_token"][:3])
            length = sum(1 for r in rows if r["candidates"]["length_only_pick"] == r["sentence_id"])
            entry["candidate_top1_per_token"] = f"{top1}/{len(rows)}"
            entry["candidate_top1_total"] = f"{top1_total}/{len(rows)}"
            entry["candidate_top3_per_token"] = f"{top3}/{len(rows)}"
            entry["length_only_top1"] = f"{length}/{len(rows)}"
            entry["chance"] = f"1/{len(fixed)}"
        tables[key] = entry
    non = [r for r in records if r.get("kind") == "nonspeech"]
    non_ok = [r for r in non if r.get("roi_status") == "ok" and "outputs" in r]
    tables["nonspeech"] = {"clips": len(non), "lip_detected": len(non_ok),
                           "hybrid_nonempty": sum(1 for r in non_ok if r["outputs"]["hybrid"].get("text")),
                           "ctc_nonempty": sum(1 for r in non_ok if r["outputs"]["ctc_greedy"].get("text"))}
    det_total = [r for r in records if r.get("status") == "ok"]
    tables["detection"] = {"clips": len(det_total), "roi_ok": sum(1 for r in det_total if r.get("roi_status") == "ok"),
                           "failures": sorted({(r.get("roi_error") or {}).get("message", "")[:60] for r in det_total if r.get("roi_status") != "ok"})}
    times = [r["timing"]["encoder"] + r["outputs"]["hybrid"].get("seconds", 0) for r in ok if "timing" in r]
    tables["timing_seconds"] = {"encoder_plus_hybrid_median": sorted(times)[len(times) // 2] if times else None,
                                "roi_median": sorted(r["roi_seconds"] for r in ok)[len(ok) // 2] if ok else None}
    return {"tables": tables, "clips_total": len(records)}


def cmd_run(args: argparse.Namespace) -> int:
    """roi-all → score 를 각 가상환경으로 실행한다.

    병실 명령 묶음은 score가 아직 채점하지 않으므로, 몇 분 걸리는 입술 추출(roi-all)을 시작하기 전에 거부한다.
    """
    try:
        rows = load_clips(Path(args.clips).resolve())
    except OSError as exc:
        print(f"클립 폴더를 읽지 못했습니다: {exc}", file=sys.stderr)
        return 1
    if ward_session_of(rows) is not None:
        print(WARD_SCORE_UNSUPPORTED, file=sys.stderr)
        return 2
    script = str(Path(__file__).resolve())
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    rc = subprocess.run([str(ot.ROI_PYTHON), script, "roi-all", "--clips", args.clips, "--out", args.out], env=env).returncode
    if rc != 0:
        return rc
    return subprocess.run([str(ot.INFER_PYTHON), script, "score", "--out", args.out, "--device", args.device,
                           "--threads", str(args.threads), "--frames-per-token", str(args.frames_per_token)], env=env).returncode


def main(argv: list[str] | None = None) -> int:
    """명령줄 진입점."""
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("check"); p.add_argument("--clips", required=True); p.set_defaults(func=cmd_check)
    p = sub.add_parser("roi-all"); p.add_argument("--clips", required=True); p.add_argument("--out", required=True)
    p.add_argument("--allow-sealed", action="store_true",
                   help="봉인된 병실 명령 평가용(ward_eval) 폴더도 입술 추출 (동결 파일을 만든 뒤에만)")
    p.set_defaults(func=cmd_roi_all)
    p = sub.add_parser("score"); p.add_argument("--out", required=True); p.add_argument("--device", default="cuda:0")
    p.add_argument("--threads", type=int, default=4); p.add_argument("--frames-per-token", type=float, default=2.6); p.set_defaults(func=cmd_score)
    p = sub.add_parser("run"); p.add_argument("--clips", required=True); p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda:0"); p.add_argument("--threads", type=int, default=4)
    p.add_argument("--frames-per-token", type=float, default=2.6); p.set_defaults(func=cmd_run)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
