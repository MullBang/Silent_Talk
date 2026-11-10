"""병실 화면 판정표 검사용 고정 사례(ward_decision_cases.json)를 만든다.

같은 서버 결과 모양(results)을 scripts/ward_tools/metrics.py의 판정표(decide·row_phrase·phrase_issue·window_kind)에 넣어
기대 결과를 적는다. 화면 쪽 검사(frontend/tests/wardDecision.test.js)는 같은 results를 classifyWardResults·decideWard에
넣어 이 기대 결과와 같은지 본다. 값은 모두 손으로 정한 합성 값이며 사람·영상 자료는 쓰지 않는다.

서버 결과의 phrase.accepted는 서버와 같은 규칙(1·2등 차이 ≥ M 그리고 1등 점수 ≥ S)으로, enroll(match)의 within·agree·accepted는
계약(상태 ok이고 D가 있고 거리 ≤ D, 1등 ID가 같음)대로 채운다. 일부 사례는 일부러 서로 맞지 않는 값(상태가 ok가 아닌데 within=true 등)을
넣어 화면이 그런 결과를 쓰지 않는지 본다.

실행: 저장소 루트에서 py -3.11 frontend/tests/fixtures/make_ward_decision_cases.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
OUT_PATH = Path(__file__).resolve().parent / "ward_decision_cases.json"
sys.path.insert(0, str(ROOT / "scripts"))

from ward_tools import common, metrics  # noqa: E402  (scripts 경로를 넣은 뒤 import)

SCHEMA = "ward-decision-cases-v1"
M = 0.5     # 글자 규칙 1·2등 차이 (서버 임시값과 같은 크기)
S = -3.40   # 글자 규칙 1등 점수
D = 0.27    # 등록 거리 임시값
FLOOR = -3.6  # 긴급 가능성 점수 하한 사례용


def rule_block(floor: float | None = None, dist: float | None = D) -> dict[str, Any]:
    """서버 phrase.rule 모양."""
    return {"min_margin": M, "min_score": S, "possible_min_score": floor, "enroll_max_distance": dist}


def phrase(commands: dict[str, Any], top: str, runner: str | None, margin: float, score: float,
           rule: dict[str, Any] | None, phrase_set: str = "ward") -> dict[str, Any]:
    """서버 phrase 모양 (accepted는 서버와 같은 글자 규칙으로 계산)."""
    try:
        out: dict[str, Any] = {
            "id": top, "text": commands["texts"].get(top, "목록 밖 문장"), "score": score, "margin": margin,
            "runner_up_id": runner, "accepted": bool(margin >= M and score >= S), "set": phrase_set,
        }
        if rule is not None:
            out["rule"] = rule
        return out
    except (KeyError, TypeError) as exc:
        raise ValueError(f"phrase 만들기 실패: {exc}") from exc


def enroll_match(phrase_id: str | None, status: str = "ok", top: str | None = None, dist: float | None = None,
                 max_distance: float | None = D, count: int = 20, **override: Any) -> dict[str, Any]:
    """서버 enroll(match) 모양. status가 ok일 때만 거리를 채운다. override로 일부러 틀린 값을 넣을 수 있다."""
    try:
        ok = status == "ok"
        top_id = top if ok else None
        distance = dist if ok else None
        within = bool(ok and max_distance is not None and distance is not None and distance <= max_distance)
        agree = bool(ok and top_id is not None and top_id == phrase_id)
        out: dict[str, Any] = {
            "mode": "match", "profile": "demo01", "status": status, "count": count,
            "top_id": top_id, "distance": distance,
            "runner_up_id": ("W12" if top_id != "W12" else "W13") if ok else None,
            "runner_up_distance": (round(distance + 0.05, 6) if distance is not None else None),
            "distances": ({top_id: distance} if ok and top_id else {}),
            "max_distance": max_distance, "within": within, "agree": agree, "accepted": bool(within and agree),
        }
        out.update(override)
        return out
    except (TypeError, ValueError) as exc:
        raise ValueError(f"enroll 만들기 실패: {exc}") from exc


def window(text: str, ph: dict[str, Any] | None = None, enroll: dict[str, Any] | None = None) -> dict[str, Any]:
    """결과 창 하나 (5초 녹화)."""
    item: dict[str, Any] = {"start_ms": 0, "end_ms": 5000, "text": text, "confidence": 0.8 if ph else None,
                            "raw_score": -1.0 if ph else None}
    if ph is not None:
        item["phrase"] = ph
    if enroll is not None:
        item["enroll"] = enroll
    return item


def build_cases(commands: dict[str, Any]) -> list[dict[str, Any]]:
    """사례 목록 (이름, results)."""
    try:
        r = rule_block()
        rf = rule_block(floor=FLOOR)
        rn = rule_block(dist=None)
        P = lambda *a, **k: phrase(commands, *a, **k)  # noqa: E731
        cases: list[tuple[str, list[dict[str, Any]]]] = [
            # 글자 규칙 통과
            ("text_pass_emergency", [window("x", P("W02", "W05", 1.2, -1.0, r))]),
            ("text_pass_request", [window("x", P("W05", "W06", 1.2, -1.0, r))]),
            ("text_pass_request_enroll_emergency_ignored",
             [window("x", P("W05", "W06", 1.2, -1.0, r), enroll_match("W05", top="W03", dist=0.1))]),
            ("text_pass_emergency_with_enroll_agree",
             [window("x", P("W04", "W02", 0.9, -2.0, r), enroll_match("W04", top="W04", dist=0.1))]),
            # 등록 거리로 고름
            ("enroll_accepted_emergency",
             [window("x", P("W03", "W05", 0.2, -3.0, r), enroll_match("W03", top="W03", dist=0.2))]),
            ("enroll_accepted_emergency_runner_emergency",
             [window("x", P("W10", "W02", 0.1, -3.8, r), enroll_match("W10", top="W10", dist=0.27))]),
            ("enroll_accepted_request",
             [window("x", P("W11", "W13", 0.2, -3.0, r), enroll_match("W11", top="W11", dist=0.2))]),
            ("enroll_accepted_request_low_score",
             [window("x", P("W14", "W12", 0.6, -3.9, r), enroll_match("W14", top="W14", dist=0.05))]),
            # 등록으로 고른 일반 명령 + 글자 2등 긴급: 등록 없이도 나던 긴급 가능성을 없애지 않는다
            ("enroll_accepted_request_runner_emergency",
             [window("x", P("W05", "W04", 0.1, -3.5, r), enroll_match("W05", top="W05", dist=0.2))]),
            ("enroll_accepted_request_runner_emergency_near_text_rule",
             [window("x", P("W08", "W03", 0.434, -2.982, r), enroll_match("W08", top="W08", dist=0.136))]),
            ("enroll_accepted_request_runner_emergency_floor_above",
             [window("x", P("W08", "W03", 0.2, -3.3, rf), enroll_match("W08", top="W08", dist=0.15))]),
            ("enroll_accepted_request_runner_emergency_floor_equal",
             [window("x", P("W09", "W10", 0.25, -3.35, rf), enroll_match("W09", top="W09", dist=0.15))]),
            ("enroll_accepted_request_runner_emergency_floor_below",
             [window("x", P("W08", "W03", 0.4, -3.3, rf), enroll_match("W08", top="W08", dist=0.15))]),
            ("enroll_accepted_cancel_w20",
             [window("x", P("W20", "W12", 0.3, -3.2, r), enroll_match("W20", top="W20", dist=0.15))]),
            # 글자 1등 ≠ 등록 1등 (고르지 않음)
            ("conflict_not_chosen",
             [window("x", P("W11", "W13", 0.2, -3.0, r), enroll_match("W11", top="W13", dist=0.2))]),
            ("conflict_enroll_alarm",
             [window("x", P("W11", "W13", 0.2, -3.0, r), enroll_match("W11", top="W10", dist=0.25))]),
            ("conflict_enroll_alarm_and_text_alarm",
             [window("x", P("W03", "W11", 0.2, -3.0, r), enroll_match("W03", top="W10", dist=0.25))]),
            ("conflict_enroll_alarm_same_as_runner",
             [window("x", P("W11", "W02", 0.2, -3.0, r), enroll_match("W11", top="W02", dist=0.1))]),
            # 등록 거리 밖·D 없음
            ("enroll_outside_distance",
             [window("x", P("W11", "W13", 0.2, -3.0, r), enroll_match("W11", top="W11", dist=0.4))]),
            ("enroll_outside_distance_emergency_top",
             [window("x", P("W11", "W13", 0.2, -3.0, r), enroll_match("W11", top="W02", dist=0.3))]),
            ("enroll_distance_off",
             [window("x", P("W11", "W13", 0.2, -3.0, rn), enroll_match("W11", top="W11", dist=0.1, max_distance=None))]),
            ("enroll_distance_off_emergency_top",
             [window("x", P("W11", "W13", 0.2, -3.0, rn), enroll_match("W11", top="W03", dist=0.1, max_distance=None))]),
            # 글자 긴급 가능성: 하한 없음
            ("text_alarm_top_no_floor", [window("x", P("W03", "W05", 0.1, -3.9, r))]),
            ("text_alarm_runner_no_floor", [window("x", P("W05", "W04", 0.1, -3.9, r))]),
            ("text_alarm_runner_very_low_no_floor", [window("x", P("W05", "W04", 2.0, -5.0, r))]),
            ("not_chosen_plain", [window("x", P("W11", "W13", 0.1, -3.9, r))]),
            # 글자 긴급 가능성: 하한 −3.6 (1등 = score, 2등 = score − margin)
            ("floor_top_above", [window("x", P("W03", "W05", 0.1, -3.5, rf))]),
            ("floor_top_below", [window("x", P("W03", "W05", 0.1, -3.7, rf))]),
            ("floor_top_equal", [window("x", P("W03", "W05", 0.1, -3.6, rf))]),
            ("floor_runner_above", [window("x", P("W05", "W04", 0.25, -3.3, rf))]),
            ("floor_runner_below", [window("x", P("W05", "W04", 0.35, -3.3, rf))]),
            ("floor_runner_boundary_arith", [window("x", P("W05", "W04", 0.2, -3.4, rf))]),
            ("floor_runner_boundary_arith2", [window("x", P("W05", "W04", 0.3, -3.3, rf))]),
            ("floor_top_just_below_runner_below", [window("x", P("W02", "W04", 0.05, -3.62, rf))]),
            ("floor_both_below", [window("x", P("W02", "W04", 0.3, -3.7, rf))]),
            ("floor_top_below_enroll_alarm",
             [window("x", P("W03", "W11", 0.2, -3.7, rf), enroll_match("W03", top="W10", dist=0.2))]),
            ("floor_enroll_alarm_only",
             [window("x", P("W11", "W13", 0.2, -3.0, rf), enroll_match("W11", top="W02", dist=0.2))]),
            ("floor_top_below_enroll_accepted",
             [window("x", P("W03", "W05", 0.2, -3.9, rf), enroll_match("W03", top="W03", dist=0.2))]),
            ("floor_text_pass_emergency", [window("x", P("W02", "W05", 1.0, -3.0, rf))]),
            # 옛 서버 (rule·enroll 없음): 오늘과 같은 동작
            ("old_server_text_alarm_runner", [window("x", P("W05", "W04", 0.1, -3.9, None))]),
            ("old_server_text_alarm_low", [window("x", P("W10", "W05", 0.1, -6.0, None))]),
            ("old_server_request", [window("x", P("W05", "W06", 1.2, -1.0, None))]),
            ("old_server_emergency", [window("x", P("W04", "W06", 1.2, -1.0, None))]),
            ("old_server_not_chosen", [window("x", P("W11", "W13", 0.1, -3.9, None))]),
            # 등록 상태가 ok가 아님 (거리 없음)
            ("enroll_incomplete",
             [window("x", P("W11", "W13", 0.2, -3.0, r), enroll_match("W11", status="incomplete", count=12))]),
            ("enroll_stale", [window("x", P("W11", "W13", 0.2, -3.0, r), enroll_match("W11", status="stale"))]),
            ("enroll_missing",
             [window("x", P("W03", "W13", 0.2, -3.0, r), enroll_match("W03", status="missing", count=0))]),
            ("enroll_error", [window("x", P("W11", "W13", 0.2, -3.0, r), enroll_match("W11", status="error"))]),
            ("enroll_no_phrase_status",
             [window("x", P("W11", "W13", 0.2, -3.0, r), enroll_match("W11", status="no_phrase"))]),
            # 일부러 틀린 서버 값: 화면은 쓰지 않는다
            ("bad_stale_but_within",
             [window("x", P("W11", "W13", 0.2, -3.0, r),
                     enroll_match("W11", status="stale", top_id="W11", distance=0.1, within=True, agree=True,
                                  accepted=True))]),
            ("bad_stale_but_within_emergency",
             [window("x", P("W11", "W13", 0.2, -3.0, r),
                     enroll_match("W11", status="stale", top_id="W02", distance=0.1, within=True, agree=False,
                                  accepted=False))]),
            ("bad_accepted_but_other_top",
             [window("x", P("W11", "W13", 0.2, -3.0, r),
                     enroll_match("W11", top="W13", dist=0.2, agree=True, accepted=True))]),
            ("bad_unknown_enroll_top",
             [window("x", P("W11", "W13", 0.2, -3.0, r),
                     enroll_match("W11", top="W11", dist=0.2, top_id="A01", agree=False, accepted=False))]),
            ("store_mode_ignored",
             [window("x", P("W11", "W13", 0.2, -3.0, r),
                     {"mode": "store", "profile": "demo01", "command": "W11", "stored": True, "reason": None,
                      "text_top": "W11", "text_top_matches": True, "count": 5, "complete": False})]),
            # 2등 정리 (목록 밖·1등과 같음)
            ("runner_same_as_top", [window("x", P("W03", "W03", 0.1, -3.9, r))]),
            ("runner_unknown_top_emergency", [window("x", P("W03", "A01", 0.1, -3.9, r))]),
            ("runner_unknown_enroll_accepted",
             [window("x", P("W11", "A01", 0.1, -3.9, r), enroll_match("W11", top="W11", dist=0.2))]),
            # 지정 문장 결과 없음·발화 없음
            ("multi_window_with_enroll",
             [window("가", P("W03", "W05", 0.1, -3.9, r), enroll_match("W03", top="W03", dist=0.1)), window("나")]),
            ("partial_detection", [window("가"), window("[검출 실패 구간]")]),
            ("no_phrase_unaligned", [window("가")]),
            ("other_set_sentences", [window("x", P("W03", "W05", 0.1, -3.9, r, phrase_set="sentences"))]),
            ("unknown_top_id", [window("x", P("A01", "W05", 0.1, -3.9, r))]),
            ("no_speech", [window("[발화 없음]")]),
            ("no_speech_two_windows", [window("[발화 없음]"), window("[발화 없음]")]),
            ("lip_fail", [window("[검출 실패 구간]")]),
            ("lip_fail_mixed", [window("[발화 없음]"), window("[검출 실패 구간]")]),
        ]
        return [{"name": name, "results": results} for name, results in cases]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"사례 만들기 실패: {exc}") from exc


def expected(results: list[dict[str, Any]], commands: dict[str, Any]) -> dict[str, Any]:
    """metrics 판정표로 기대 결과를 구한다 (require_agreement=True, possible_ranks=2)."""
    try:
        row = {"status": "done", "results": results}
        raw = metrics.row_phrase(row)
        issue = metrics.phrase_issue(raw, commands)
        none = {"chosen_id": None, "via": None, "text_pass": False, "conflict": False, "enroll_alarm": False}
        if raw is None or issue is not None:
            return {"outcome": metrics.OUT_NO_RESULT if issue else metrics.window_kind(row), **none}
        rule = raw.get("rule") or {}
        runner = raw.get("runner_up_id")
        if runner is not None and (runner not in commands["texts"] or runner == raw["id"]):
            runner = None
        enroll = results[0].get("enroll") if isinstance(results[0].get("enroll"), dict) else {}
        tmpl_top = tmpl_dist = None
        if (enroll.get("mode") == "match" and enroll.get("status") == "ok" and enroll.get("top_id") in commands["texts"]
                and isinstance(enroll.get("distance"), (int, float))):
            tmpl_top, tmpl_dist = enroll["top_id"], float(enroll["distance"])
        ward_rule = metrics.Rule(
            min_margin=float(rule.get("min_margin", M)), min_score=float(rule.get("min_score", S)),
            emergency_ids=frozenset(commands["emergency"]), max_distance=rule.get("enroll_max_distance"),
            require_agreement=True, possible_ranks=2, possible_min_score=rule.get("possible_min_score"))
        d = metrics.decide(raw["id"], runner, raw["margin"], raw["score"], ward_rule, tmpl_top, tmpl_dist)
        return {k: d[k] for k in ("outcome", "chosen_id", "via", "text_pass", "conflict", "enroll_alarm")}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"기대 결과 계산 실패: {exc}") from exc


def main() -> int:
    """사례 파일을 쓴다."""
    try:
        commands = common.load_commands()
        cases = build_cases(commands)
        names = [c["name"] for c in cases]
        if len(set(names)) != len(names):
            raise ValueError("사례 이름이 겹칩니다")
        for case in cases:
            case["expected"] = expected(case["results"], commands)
            # 서버 accepted가 metrics 글자 규칙과 같은지 (사례 자체 검사)
            ph = case["results"][0].get("phrase") if len(case["results"]) == 1 else None
            if ph and "rule" in ph and ph.get("set") == "ward" and ph["id"] in commands["texts"]:
                if bool(ph["accepted"]) != case["expected"]["text_pass"]:
                    raise ValueError(f"{case['name']}: accepted와 글자 규칙이 다릅니다")
        doc = {
            "schema": SCHEMA,
            "generator": "frontend/tests/fixtures/make_ward_decision_cases.py",
            "decide": "scripts/ward_tools/metrics.py decide (require_agreement=True, possible_ranks=2)",
            "commands_sha256": commands["sha256"],
            "emergency_ids": commands["emergency"],
            "constants": {"min_margin": M, "min_score": S, "enroll_max_distance": D, "possible_min_score": FLOOR},
            "cases": cases,
        }
        OUT_PATH.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
        counts: dict[str, int] = {}
        for case in cases:
            counts[case["expected"]["outcome"]] = counts.get(case["expected"]["outcome"], 0) + 1
        sys.stdout.reconfigure(encoding="utf-8")
        print(f"{len(cases)}개 사례 → {OUT_PATH.relative_to(ROOT).as_posix()} {counts}")
        return 0
    except (OSError, ValueError) as exc:
        print(f"실패: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
