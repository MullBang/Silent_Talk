"""병실 명령 채점 도구 검사 (모델·GPU·서버 불필요).

scripts/ward_tools(판정표·지표·0건 상한·격자 맞추기·LOPO·축소 목록·등록 거리·긴급 가능성 하한·동결 파일 형식·봉인 확인·지문)와
scripts/ward_service_batch.py·ward_enroll.py·ward_eval_tools.py의 순수 함수·봉인 거부를 가짜 자료로 확인한다.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "backend")]

import ward_enroll  # noqa: E402
import ward_eval_tools as wet  # noqa: E402
import ward_service_batch as wsb  # noqa: E402
from ward_tools import calibrate as cal  # noqa: E402
from ward_tools import common  # noqa: E402
from ward_tools import enroll  # noqa: E402
from ward_tools import metrics as mt  # noqa: E402
from ward_tools import report as rp  # noqa: E402

CMDS = common.load_commands()
IDS = CMDS["ids"]
EM = frozenset(CMDS["emergency"])
RULE = mt.Rule(0.5, -3.40, EM)


@pytest.fixture(autouse=True)
def coarse_grid(monkeypatch: pytest.MonkeyPatch) -> None:
    """격자를 성기게 해 검사를 빠르게 한다 (선택 규칙은 같다)."""
    monkeypatch.setattr(cal, "M_GRID", np.round(np.arange(0.0, 2.0 + 1e-9, 0.05), 3))
    monkeypatch.setattr(cal, "S_GRID", np.round(np.arange(-6.0, -2.0 + 1e-9, 0.05), 3))


def scores(top: str, s_top: float, margin: float, runner: str | None = None, rest: float = -7.0) -> dict[str, float]:
    """명령 20개 후보 점수: top이 s_top, runner(없으면 top이 아닌 첫 명령)가 s_top - margin, 나머지는 rest."""
    runner = runner or next(i for i in IDS if i != top)
    out = {i: rest for i in IDS}
    out[top] = s_top
    out[runner] = s_top - margin
    return out


def svc_row(file: str, kind: str, cond: str, sid: str, sc: dict[str, float] | None = None, *, session: str = "ward_tune",
            participant: str = "P01", rep: int = 1, posture: str | None = None, results: list | None = None,
            end_ms: int = 4000, duration_sec: float = 5.0, seconds: float = 1.5) -> dict[str, Any]:
    """서비스 일괄 실행 결과 행 (sc를 주면 창 하나 + phrase)."""
    if results is None:
        r = mt.rank_scores(sc, IDS)
        results = [{"start_ms": 300, "end_ms": end_ms, "text": "", "confidence": 0.8,
                    "phrase": {"id": r["id"], "text": CMDS["texts"][r["id"]], "score": r["score"], "margin": r["margin"],
                               "runner_up_id": r["runner_up_id"], "accepted": r["margin"] >= 0.5 and r["score"] >= -3.4,
                               "set": "ward"}}]
    return {"file": file, "participant": participant, "kind": kind, "condition": cond, "sentence_id": sid, "rep": rep,
            "posture": posture or ("reclined" if cond == "reclined" else "upright"), "session": session,
            "phrase_set": "ward", "status": "done", "results": results, "seconds": seconds, "duration_sec": duration_sec}


def base_of(row: dict[str, Any], extras: dict[str, Any] | None = None) -> dict[str, Any]:
    """행 → 판정 전 기록."""
    return mt.prepare(row, CMDS, extras)


# ---------------------------------------------------------------------------
# 명령 목록·상한
# ---------------------------------------------------------------------------

def test_commands_file_matches_plan_emergency_and_short_commands() -> None:
    assert CMDS["emergency"] == ["W02", "W03", "W04", "W10"]
    assert len(IDS) == 20
    nine = [i for i, t in CMDS["texts"].items() if mt.syllables(t) <= 9]
    assert nine == ["W05", "W06", "W09", "W12", "W15"]  # 기획 10.2 '9음절 명령'


def test_upper_bound_matches_plan_examples() -> None:
    assert common.upper95(23) == pytest.approx(0.122, abs=0.001)  # 기획 10.2 '0/23 (상한 약 12%)'
    assert common.upper95(20) == pytest.approx(0.139, abs=0.001)  # 8단계 문서 '0/20 → 약 14%'
    assert common.upper95(0) is None
    assert common.frac(0, 12) == "0/12 (한쪽 95% 상한 22.1%)"
    assert common.frac(3, 12) == "3/12"
    assert mt.count(0, 0)["upper95_if_zero"] is None


# ---------------------------------------------------------------------------
# 판정표
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("top,runner,margin,score,outcome,chosen", [
    ("W02", "W05", 0.8, -3.0, mt.OUT_EMERGENCY, "W02"),       # 통과 + 긴급 → 빨간 카드
    ("W05", "W02", 0.8, -3.0, mt.OUT_REQUEST, "W05"),         # 통과 + 일반 → 요청
    ("W10", "W05", 0.2, -3.0, mt.OUT_POSSIBLE, None),         # 미통과, 1등 긴급
    ("W05", "W04", 0.2, -3.0, mt.OUT_POSSIBLE, None),         # 미통과, 2등 긴급
    ("W05", "W06", 0.2, -3.0, mt.OUT_NOT_CHOSEN, None),       # 미통과, 긴급 없음
    ("W05", "W06", 0.8, -3.6, mt.OUT_NOT_CHOSEN, None),       # 점수 미달
    ("W05", "W06", 0.5, -3.40, mt.OUT_REQUEST, "W05"),        # 경계값은 통과(≥)
    (None, None, None, None, mt.OUT_NO_RESULT, None),
])
def test_decision_table(top: str | None, runner: str | None, margin: float | None, score: float | None,
                        outcome: str, chosen: str | None) -> None:
    d = mt.decide(top, runner, margin, score, RULE)
    assert (d["outcome"], d["chosen_id"]) == (outcome, chosen)


def test_possible_ranks_and_floor_use_runner_up_score() -> None:
    one = replace(RULE, possible_ranks=1)
    assert mt.decide("W05", "W04", 0.2, -3.0, one)["outcome"] == mt.OUT_NOT_CHOSEN
    floor = replace(RULE, possible_min_score=-3.5)
    # 2등 W04 점수 = -3.0 - 0.4 = -3.4 ≥ -3.5 → 긴급 가능성, -3.0 - 0.6 = -3.6 < -3.5 → 고르지 않음
    assert mt.decide("W05", "W04", 0.4, -3.0, floor)["outcome"] == mt.OUT_POSSIBLE
    assert mt.decide("W05", "W04", 0.6, -3.6, floor)["outcome"] == mt.OUT_NOT_CHOSEN
    assert mt.possible_score("W10", "W04", 0.3, -3.2, RULE) == pytest.approx(-3.2)


def test_enrollment_rules() -> None:
    r = replace(RULE, max_distance=0.27)
    # 글자 미통과 + 등록 거리 통과 + 1등 일치: 일반은 요청, 긴급은 '긴급 가능성'(명령 있음)
    assert mt.decide("W05", "W06", 0.1, -3.9, r, "W05", 0.2) == {
        "outcome": mt.OUT_REQUEST, "chosen_id": "W05", "via": "enroll", "text_pass": False, "conflict": False,
        "enroll_alarm": False}
    d = mt.decide("W02", "W06", 0.1, -3.9, r, "W02", 0.2)
    assert (d["outcome"], d["chosen_id"], d["via"]) == (mt.OUT_POSSIBLE, "W02", "enroll")
    # 거리 초과 → 고르지 않음
    assert mt.decide("W05", "W06", 0.1, -3.9, r, "W05", 0.3)["outcome"] == mt.OUT_NOT_CHOSEN
    # 글자 규칙 실패 + 거리 통과 + 글자 1등과 등록 1등이 다름 → 고르지 않음(conflict), 글자 1등이 긴급이면 긴급 가능성
    d = mt.decide("W05", "W06", 0.1, -3.9, r, "W06", 0.1)
    assert (d["outcome"], d["chosen_id"], d["conflict"]) == (mt.OUT_NOT_CHOSEN, None, True)
    assert mt.decide("W03", "W06", 0.1, -3.9, r, "W06", 0.1)["outcome"] == mt.OUT_POSSIBLE
    # 일치를 요구하지 않으면 '글자 규칙 또는 거리'
    loose = replace(r, require_agreement=False)
    assert mt.decide("W05", "W06", 0.1, -3.9, loose, "W06", 0.2)["chosen_id"] == "W06"
    assert mt.decide("W05", "W06", 0.9, -3.0, loose, "W06", 0.2)["chosen_id"] == "W05"
    # 등록 정보가 없으면 글자 규칙만
    assert mt.decide("W05", "W06", 0.9, -3.0, r)["via"] == "text"


def test_text_rule_pass_is_kept_when_template_top_differs() -> None:
    """글자 규칙을 통과한 클립은 등록 1등이 달라도 그대로 고른다 (일치 확인은 등록 거리로 고를 때만)."""
    r = replace(RULE, max_distance=0.27)
    red = mt.decide("W02", "W06", 0.9, -3.0, r, "W06", 0.1)
    assert (red["outcome"], red["chosen_id"], red["via"], red["conflict"]) == (mt.OUT_EMERGENCY, "W02", "text", False)
    req = mt.decide("W05", "W06", 0.9, -3.0, r, "W07", 0.1)
    assert (req["outcome"], req["chosen_id"], req["via"]) == (mt.OUT_REQUEST, "W05", "text")


def test_template_top_emergency_within_d_raises_emergency_possible() -> None:
    """글자 1·2등이 일반 명령이어도 거리 D 이하인 등록 1등이 긴급이면 긴급 가능성 (기획 11.3)."""
    r = replace(RULE, max_distance=0.27)
    d = mt.decide("W05", "W06", 0.1, -3.9, r, "W10", 0.2)
    assert (d["outcome"], d["chosen_id"], d["conflict"], d["enroll_alarm"]) == (mt.OUT_POSSIBLE, None, True, True)
    assert mt.decide("W05", "W06", 0.1, -3.9, r, "W10", 0.3)["outcome"] == mt.OUT_NOT_CHOSEN  # D 초과
    assert mt.decide("W05", "W06", 0.1, -3.9, RULE, "W10", 0.2)["outcome"] == mt.OUT_NOT_CHOSEN  # 등록 안 씀
    # 글자 2등이 긴급이면 글자 길로 긴급 가능성(enroll_alarm은 등록 때문에만 생긴 경우)
    assert mt.decide("W05", "W04", 0.1, -3.9, r, "W10", 0.2)["enroll_alarm"] is False


def test_rank_scores_matches_service_ranking_rules() -> None:
    assert mt.rank_scores({"A": -3.0, "B": -3.0, "C": -4.0}, ["A", "B", "C"])["id"] == "A"  # 동점은 목록 앞
    assert mt.rank_scores({"A": -3.0, "B": -3.0}, ["B", "A"])["id"] == "B"
    assert mt.rank_scores({"A": -3.0, "B": None}, ["A", "B"]) is None  # 하나라도 정렬 불가면 결과 없음
    r = mt.rank_scores({"A": 0.3, "B": 0.2, "C": 0.5}, ["A", "B", "C"], higher_is_better=False)
    assert (r["id"], r["runner_up_id"], r["margin"]) == ("B", "A", pytest.approx(0.1))


def test_role_and_window_kinds() -> None:
    assert mt.role_of({"session": "ward_tune", "kind": "fixed"}) == mt.ROLE_COMMAND
    assert mt.role_of({"session": "ward_eval", "kind": "short"}) == mt.ROLE_SHORT
    assert mt.role_of({"session": "pilot", "kind": "fixed"}) == mt.ROLE_SENT_SPEECH
    assert mt.role_of({"session": "eval", "kind": "free"}) == mt.ROLE_SENT_SPEECH
    assert mt.role_of({"session": "pilot", "kind": "nonspeech"}) == mt.ROLE_SENT_NONSPEECH
    with pytest.raises(ValueError):
        mt.role_of({"session": "ward_tune", "kind": "free"})
    no_speech = svc_row("a", "nonspeech", "none", "X01", results=[{"start_ms": 0, "end_ms": 5000, "text": common.NO_SPEECH_TEXT}])
    lip = svc_row("b", "fixed", "silent", "W01", results=[{"start_ms": 0, "end_ms": 5000, "text": common.FAIL_TEXT}])
    multi = svc_row("c", "fixed", "silent", "W01", results=[{"text": common.FAIL_TEXT}, {"text": "가나"}])
    err = {**svc_row("d", "fixed", "silent", "W01", scores("W01", -3, 1)), "status": "error"}
    outs = [mt.evaluate(base_of(r), RULE)["outcome"] for r in (no_speech, lip, multi, err)]
    assert outs == [mt.OUT_NO_SPEECH, mt.OUT_LIP_FAIL, mt.OUT_NO_RESULT, mt.OUT_ERROR]  # 처리 실패 행은 판정하지 않는다
    assert mt.window_kind({"status": "error"}) == mt.OUT_ERROR


def test_reduced_list_reranks_with_full_scores() -> None:
    sc = scores("W06", -3.0, 0.1, runner="W05")
    sc["W07"] = -3.7
    b = base_of(svc_row("a", "fixed", "silent", "W05", sc), {"text_scores": sc})
    full = mt.evaluate(b, RULE)
    assert (full["top"], full["chosen_id"]) == ("W06", None)
    red = mt.evaluate(b, replace(RULE, ids=("W02", "W05", "W07")))
    assert (red["top"], red["runner_up"], red["chosen_id"]) == ("W05", "W07", "W05")
    assert red["margin"] == pytest.approx(0.6)
    no_scores = mt.evaluate(base_of(svc_row("b", "fixed", "silent", "W05", sc)), replace(RULE, ids=("W02", "W05")))
    assert no_scores["outcome"] == mt.OUT_UNSCORABLE


# ---------------------------------------------------------------------------
# 지표
# ---------------------------------------------------------------------------

def small_set() -> list[dict[str, Any]]:
    """한 사람의 작은 조정 묶음 + 문장 촬영 목록 밖 클립."""
    rows = [
        svc_row("P01_silent_W02_1", "fixed", "silent", "W02", scores("W02", -3.0, 0.9)),          # 빨간 카드
        svc_row("P01_silent_W03_1", "fixed", "silent", "W03", scores("W05", -3.0, 0.1, "W03")),   # 긴급 가능성(2등)
        svc_row("P01_silent_W04_1", "fixed", "silent", "W04", scores("W06", -3.0, 0.9)),          # 오답 표시 → 놓침
        svc_row("P01_reclined_W10_1", "fixed", "reclined", "W10", scores("W10", -3.6, 0.9)),      # 점수 미달, 1등 긴급
        svc_row("P01_silent_W05_1", "fixed", "silent", "W05", scores("W05", -3.0, 0.9), end_ms=4980),  # 요청, 녹화 끝
        svc_row("P01_silent_W07_1", "fixed", "silent", "W07", scores("W07", -3.0, 0.1, "W10")),   # 거짓 긴급 가능성
        svc_row("P01_silent_D01_1", "decoy", "silent", "D01", scores("W02", -3.1, 0.7)),          # 함정 → 빨간 카드
        svc_row("P01_silent_E01_1", "short", "silent", "E01", scores("W02", -3.8, 0.2)),          # 짧은 긴급 → 가능성
        svc_row("P01_none_X01_1", "nonspeech", "none", "X01",
                results=[{"start_ms": 0, "end_ms": 5000, "text": common.NO_SPEECH_TEXT}]),
        svc_row("P01_voiced_A01_1", "fixed", "voiced", "A01", scores("W11", -3.2, 0.8), session="pilot", end_ms=4990),
    ]
    return [base_of(r) for r in rows]


def test_summary_metrics_on_small_set() -> None:
    s = mt.summarize(mt.evaluate_all(small_set(), RULE), RULE, CMDS)
    assert s["commands"]["n"] == 6
    assert s["commands"]["shown"]["k"] == 3  # W02, W04(오답), W05
    assert s["commands"]["shown_wrong"]["k"] == 1 and s["commands"]["shown_wrong_clips"] == ["P01_silent_W04_1->W06"]
    er = s["emergency_recall"]
    assert er["all"]["reached"]["text"] == "3/4" and er["condition_silent"]["reached"]["k"] == 2
    assert er["condition_reclined"]["possible"] == 1
    assert er["all"]["missed_clips"] == ["P01_silent_W04_1:request"]
    fa = s["false_alarms"]
    assert fa["command_non_emergency"]["possible"] == 1 and fa["decoy"]["red"] == 1
    assert fa["total"]["k"] == 2 and fa["total"]["n"] == 2 + 1 + 1 + 1  # 짧은 긴급 표현은 넣지 않는다
    neg = s["negatives_shown"]
    assert neg["total"]["k"] == 2 and neg[mt.ROLE_SENT_SPEECH]["clips"] == ["P01_voiced_A01_1->W11"]
    assert s["short"]["reached_alarm"]["k"] == 1 and s["short"]["possible"] == 1
    assert s["posture"]["upright"]["shown"]["text"] == "3/5" and s["posture"]["reclined"]["shown"]["k"] == 0
    # 녹화 끝: 병실 묶음만, 꼬리 20ms(W05)는 가까움, 1000ms는 아님, 문장 촬영(A01)은 세지 않음
    re_ = s["recording_end"]
    assert re_["near_end"]["k"] == 1 and re_["clips"] == ["P01_silent_W05_1:20"]
    assert re_["near_end"]["n"] == 8
    assert s["window_length"]["short_ids"] == ["W05", "W06", "W09", "W12", "W15"]
    assert s["latency_seconds"]["median"] == 1.5
    assert s["server_accepted_check"]["mismatch"] == 0


def test_summarize_groups_and_markdown() -> None:
    bases = small_set() + [base_of(svc_row("P03_silent_W02_1", "fixed", "silent", "W02", scores("W02", -3.0, 0.9),
                                           participant="P03"))]
    g = mt.summarize_groups(mt.evaluate_all(bases, RULE), RULE, CMDS)
    assert set(g["by_group"]) == {"P01/ward_tune", "P01/pilot", "P03/ward_tune"}
    assert set(g["by_person"]) == {"P01", "P03"}
    md = rp.table(g["by_group"], "시험")
    assert "실행된 명령 중 오답" in md and "P03/ward_tune" in md


def test_repro_check_detects_difference() -> None:
    sc = scores("W05", -3.0, 0.6)
    same = base_of(svc_row("a", "fixed", "silent", "W05", sc), {"text_scores": sc})
    other = dict(sc)
    other["W05"] = -3.1
    diff = base_of(svc_row("b", "fixed", "silent", "W05", sc), {"text_scores": other})
    r = mt.repro_check([same, diff])
    assert r["n_compared"] == 2 and r["n_different"] == 1 and r["max_score_abs_diff"] == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# 기준 맞추기
# ---------------------------------------------------------------------------

def item(m: float, s: float, cls: str, file: str = "x") -> dict[str, Any]:
    return {"m": m, "s": s, "cls": cls, "file": file, "participant": "P01", "top": "W05"}


def test_grid_fit_zero_bad_max_good_and_clearance() -> None:
    items = [item(1.0, -3.0, "good"), item(0.9, -3.2, "good"), item(0.3, -3.0, "block"), item(1.2, -3.8, "wrong")]
    g = cal.grid_fit(items, m_grid=np.round(np.arange(0, 1.5001, 0.05), 3), s_grid=np.round(np.arange(-4.0, -2.4999, 0.05), 3))
    assert g["feasible"] and g["max_correct_shown"] == 2
    assert 0.3 < g["M"] <= 0.9 and -3.8 < g["S"] <= -3.2
    shown = [i for i in items if i["m"] >= g["M"] and i["s"] >= g["S"]]
    assert {i["cls"] for i in shown} == {"good"}
    # 여유 0.7: 나쁜 클립과 한 축에서 0.7 이상 떨어져야 하므로 정답 하나만 남는다
    g2 = cal.grid_fit(items, 0.7, m_grid=np.round(np.arange(0, 1.5001, 0.05), 3),
                      s_grid=np.round(np.arange(-4.0, -2.4999, 0.05), 3))
    assert g2["feasible"] and g2["max_correct_shown"] == 1
    assert not cal.grid_fit([item(1.0, -3.0, "block")], m_grid=np.array([0.0]), s_grid=np.array([-6.0]))["feasible"]
    assert cal.grid_fit([])["feasible"] is False


def two_person_bases() -> list[dict[str, Any]]:
    """P01·P03: 정답 명령(차이 크게), 오답 명령(차이 작게), 목록 밖 문장."""
    rows = []
    for pid, neg_margin in (("P01", 0.3), ("P03", 0.75)):
        for k, cid in enumerate(IDS):
            rows.append(svc_row(f"{pid}_silent_{cid}_1", "fixed", "silent", cid, scores(cid, -3.0 - 0.01 * k, 1.0),
                                participant=pid))
            rows.append(svc_row(f"{pid}_voiced_{cid}_1", "fixed", "voiced", cid, scores(cid, -2.9, 0.6 + 0.02 * k),
                                participant=pid))
        other = "W06" if pid == "P01" else "W07"
        rows.append(svc_row(f"{pid}_silent_W05_2", "fixed", "silent", "W05", scores(other, -3.0, 0.2, "W05"), participant=pid,
                            rep=2))
        rows.append(svc_row(f"{pid}_silent_A01_1", "fixed", "silent", "A01", scores("W11", -3.0, neg_margin), participant=pid,
                            session="pilot"))
    return [base_of(r, {"text_scores": None}) for r in rows]


def test_lopo_applies_fit_thresholds_to_other_person() -> None:
    bases = two_person_bases()
    lopo = cal.lopo_thresholds(bases, EM)
    p01, p03 = lopo["fit_P01"], lopo["fit_P03"]
    assert p01["fit"]["feasible"] and p03["fit"]["feasible"]
    assert p01["on_fit_person"]["negatives_shown_k"] == 0 and p03["on_fit_person"]["negatives_shown_k"] == 0
    assert p01["fit"]["M"] > 0.3 and p03["fit"]["M"] > 0.75
    if p01["fit"]["M"] <= 0.75:  # P01로 정한 기준은 P03의 목록 밖 문장(차이 0.75)을 막지 못할 수 있다
        assert p01["on_P03"]["negatives_shown_k"] == 1
    assert p03["on_P01"]["negatives_shown_k"] == 0


def test_choose_proposal_rule() -> None:
    pooled = {"0.00": {"fit": {"feasible": True, "M": 0.4, "S": -3.5, "max_correct_shown": 100}},
              "0.05": {"fit": {"feasible": True, "M": 0.45, "S": -3.45, "max_correct_shown": 95}},
              "0.10": {"fit": {"feasible": True, "M": 0.5, "S": -3.4, "max_correct_shown": 80}},
              "0.15": {"fit": {"feasible": False}}}
    p = cal.choose_proposal(pooled, max_loss=0.10)
    assert (p["M"], p["S"], p["delta"]) == (0.45, -3.45, 0.05)
    assert cal.choose_proposal(pooled, delta=0.1)["M"] == 0.5
    with pytest.raises(ValueError):
        cal.choose_proposal(pooled, delta=0.15)
    p = cal.choose_proposal({"0.00": {"fit": {"feasible": False}}})
    assert (p["M"], p["S"], p["delta"]) == (*cal.PROVISIONAL, None)


def test_reduced_list_keeps_emergency_and_drops_confused() -> None:
    rows = []
    for cid in IDS:
        sc = scores(cid, -3.0, 1.0)
        rows.append(svc_row(f"P01_silent_{cid}_1", "fixed", "silent", cid, sc))
    for k in range(3):  # W05 클립이 W06으로, W06이 W05로 헷갈림 + W10이 목록 밖 문장을 끌어당김
        rows.append(svc_row(f"P01_silent_W05_{k + 2}", "fixed", "silent", "W05", scores("W06", -3.0, 0.3, "W05"), rep=k + 2))
        rows.append(svc_row(f"P01_silent_W06_{k + 2}", "fixed", "silent", "W06", scores("W05", -3.0, 0.3, "W06"), rep=k + 2))
        rows.append(svc_row(f"P01_silent_A0{k}_1", "fixed", "silent", f"A0{k}", scores("W10", -3.0, 0.9), session="pilot"))
    exact_bases = [base_of(r, {"text_scores": r["results"][0]["phrase"] and _scores_of(r)}) for r in rows]
    red = cal.select_reduced_list(exact_bases, IDS, 18, list(EM), (0.5, -3.4))
    assert red["exact"] and red["size"] == 18 and set(EM) <= set(red["ids"])
    # 먼저 서로 헷갈리는 W05·W06 중 하나를 빼고, 남은 혼동이 없으면 정답 차이가 작은 명령 → ID가 큰 명령(W20) 순
    assert red["trace"][0]["dropped"] in ("W05", "W06") and red["trace"][1]["dropped"] == "W20"
    assert "W10" in red["ids"]  # 목록 밖 문장을 끌어당겨도 긴급 명령은 빼지 않는다
    approx = cal.select_reduced_list([base_of(r) for r in rows], IDS, 19, list(EM), (0.5, -3.4))
    assert not approx["exact"] and approx["trace"][0]["dropped"] in ("W05", "W06")
    with pytest.raises(ValueError):
        cal.select_reduced_list(exact_bases, IDS, 3, list(EM), (0.5, -3.4))


def test_reduced_headline_leaves_out_dropped_commands() -> None:
    sc = scores("W05", -3.0, 0.9)
    rows = [svc_row("a", "fixed", "silent", "W05", sc), svc_row("b", "fixed", "silent", "W06", scores("W06", -3.0, 0.9))]
    bases = [base_of(r, {"text_scores": _scores_of(r)}) for r in rows]
    h = cal.headline(bases, cal.rule_at(0.5, -3.4, EM, ids=["W02", "W03", "W04", "W10", "W05", "W07"]))
    assert h["commands_n"] == 1 and h["shown_wrong_k"] == 0  # W06 클립은 축소 목록 사용자가 말하지 않는다


def _scores_of(row: dict[str, Any]) -> dict[str, float]:
    """svc_row가 만든 phrase를 다시 20개 점수로 (scores()와 같은 모양)."""
    ph = row["results"][0]["phrase"]
    return scores(ph["id"], ph["score"], ph["margin"], ph["runner_up_id"])


def test_fit_distance_midpoint_unbounded_and_none() -> None:
    def b(file: str, role_kind: str, sid: str, top: str, dist_top: float, text_margin: float = 0.1) -> dict[str, Any]:
        sc = scores(top, -3.9, text_margin)
        d = {i: 0.4 for i in IDS}
        d[top] = dist_top
        return base_of(svc_row(file, role_kind, "silent", sid, sc), {"text_scores": sc, "tmpl_dists": d})
    good = [b("g1", "fixed", "W05", "W05", 0.20), b("g2", "fixed", "W06", "W06", 0.25)]
    bad = [b("n1", "decoy", "D01", "W07", 0.27), b("w1", "fixed", "W08", "W09", 0.30)]
    f = cal.fit_distance(good + bad, RULE)
    assert f["blocking_dist"] == pytest.approx(0.27) and f["D"] == pytest.approx(0.26) and not f["unbounded"]
    ev = mt.evaluate_all(good + bad, replace(RULE, max_distance=f["D"]))
    assert [e["chosen_id"] for e in ev] == ["W05", "W06", None, None]
    f2 = cal.fit_distance(good, RULE)
    assert f2["unbounded"] and f2["D"] == pytest.approx(0.25)
    assert cal.fit_distance(bad, RULE)["D"] == pytest.approx(0.27 - 1e-6)
    assert cal.fit_distance([], RULE)["D"] is None
    lopo = cal.lopo_distance(good + bad, RULE)
    assert lopo["pooled"]["fit"]["D"] == pytest.approx(0.26)
    assert lopo["pooled"]["fit"]["n_negatives"] == 1 and lopo["pooled"]["fit"]["n_negatives_with_dist"] == 1
    many = {"D": 0.3, "n_negatives_with_dist": 60}
    assert cal.choose_distance({"a": {"fit": many}, "pooled": {"fit": {"D": 0.28}}})["D"] == 0.28
    assert cal.choose_distance({"a": {"fit": many}, "pooled": {"fit": {"D": 0.28}}}, "pooled")["D"] == 0.28


def test_choose_distance_refuses_unbounded_or_too_few_negatives() -> None:
    """막는 거리를 모르는 D(나쁜 클립 없음)나 거리가 있는 목록 밖 클립이 적은 사람이 있으면 D를 정하지 않는다."""
    unb = cal.choose_distance({"fit_P01": {"fit": {"D": 0.25, "unbounded": True, "n_negatives_with_dist": 90}},
                               "pooled": {"fit": {"D": 0.25, "unbounded": True}}})
    assert unb["D"] is None and unb["unbounded"] and "막는 거리" in unb["chosen_by"]
    few = cal.choose_distance({"fit_P01": {"fit": {"D": 0.25, "n_negatives_with_dist": 49}},
                               "fit_P03": {"fit": {"D": 0.26, "n_negatives_with_dist": 80}},
                               "pooled": {"fit": {"D": 0.25}}})
    assert few["D"] is None and "fit_P01(49)" in few["chosen_by"]
    ok = cal.choose_distance({"fit_P01": {"fit": {"D": 0.25, "n_negatives_with_dist": 50}},
                              "fit_P03": {"fit": {"D": 0.26, "n_negatives_with_dist": 80}},
                              "pooled": {"fit": {"D": 0.255}}})
    assert ok["D"] == 0.25
    # 실제 문장 촬영 목록 밖 클립에 거리가 없으면(등록 템플릿 없음) 나쁜 클립이 없어 unbounded → D 없음
    def b(file: str, kind: str, sid: str, top: str, dist: float | None, session: str = "ward_tune") -> dict[str, Any]:
        sc = scores(top, -3.9, 0.1)
        d = None if dist is None else {**{i: 0.4 for i in IDS}, top: dist}
        return base_of(svc_row(file, kind, "silent", sid, sc, session=session), {"text_scores": sc, "tmpl_dists": d})
    bases = [b("g1", "fixed", "W05", "W05", 0.2)] + [b(f"n{k}", "fixed", f"A{k:02d}", "W07", None, "pilot") for k in range(60)]
    lopo = cal.lopo_distance(bases, RULE)
    assert lopo["fit_P01"]["fit"]["unbounded"] and lopo["fit_P01"]["fit"]["n_negatives_with_dist"] == 0
    assert cal.choose_distance(lopo)["D"] is None


def test_enroll_candidates_count_template_alarms_and_short_requests() -> None:
    """등록 1등이 긴급이라 긴급 가능성이 되는 목록 밖 클립, 등록 거리로 일반 명령이 된 짧은 긴급 표현은 D를 막는다."""
    def b(file: str, kind: str, sid: str, top: str, tmpl_top: str, dist: float) -> dict[str, Any]:
        sc = scores(top, -3.9, 0.1)
        d = {i: 0.45 for i in IDS}
        d[tmpl_top] = dist
        return base_of(svc_row(file, kind, "silent", sid, sc), {"text_scores": sc, "tmpl_dists": d})
    rows = [b("g", "fixed", "W05", "W05", "W05", 0.20), b("alarm", "decoy", "D01", "W07", "W10", 0.24),
            b("short_req", "short", "E01", "W08", "W08", 0.22), b("short_em", "short", "E02", "W02", "W02", 0.15),
            b("em_cmd", "fixed", "W03", "W06", "W03", 0.18)]
    cands = {c["file"]: c for c in cal.enroll_only_candidates(rows, RULE)}
    assert cands["g"]["cls"] == "good" and cands["alarm"] == {**cands["alarm"], "cls": "bad", "effect": "alarm"}
    assert cands["short_req"]["cls"] == "bad" and "short_em" not in cands
    assert "em_cmd" not in cands  # 긴급 명령 클립이 다른 1등 때문에 긴급 가능성 → 바라는 결과(나쁜 클립 아님)
    f = cal.fit_distance(rows, RULE)
    assert f["blocking_dist"] == pytest.approx(0.22) and f["D"] == pytest.approx(0.21)


def test_possible_floor_keeps_every_emergency_clip() -> None:
    rows = [svc_row("e1", "fixed", "silent", "W02", scores("W02", -3.6, 0.9)),           # 1등 긴급 -3.6
            svc_row("e2", "fixed", "silent", "W03", scores("W05", -3.2, 0.3, "W03")),    # 2등 긴급 -3.5
            svc_row("n1", "fixed", "silent", "A01", scores("W10", -4.5, 0.2), session="pilot"),
            svc_row("n2", "fixed", "silent", "A02", scores("W10", -3.65, 0.2), session="pilot")]
    f = cal.fit_possible_floor([base_of(r) for r in rows], RULE, clearance=0.0)
    assert f["needed"] == pytest.approx(-3.6) and f["floor"] == pytest.approx(-3.6) and f["n_relying_on_possible"] == 2
    assert f["without_floor"]["false_alarms_k"] == 2 and f["with_floor"]["false_alarms_k"] == 0
    assert f["with_floor"]["emergency_recall"] == "2/2"
    f2 = cal.fit_possible_floor([base_of(r) for r in rows], RULE, clearance=0.1)
    assert f2["floor"] == pytest.approx(-3.7) and f2["with_floor"]["false_alarms_k"] == 1
    assert cal.fit_possible_floor([base_of(rows[2])], RULE)["floor"] is None


# ---------------------------------------------------------------------------
# 동결 파일
# ---------------------------------------------------------------------------

def write_rows(path: Path, rows: list[dict[str, Any]], **head: Any) -> Path:
    path.write_text(json.dumps({"schema": wsb.SCHEMA, "phrase_set": "ward", **head, "rows": rows}, ensure_ascii=False),
                    encoding="utf-8")
    return path


def real_draft(tmp_path: Path) -> dict[str, Any]:
    """가짜 두 사람 자료로 만든 동결 초안 (합성 표시 없음 = 형식 검사용)."""
    rows = two_person_bases()
    raw = [r for r in rows]
    files = []
    for pid in ("P01", "P03"):
        mine = [{k: v for k, v in r.items()} for r in raw if r["participant"] == pid]
        files.append(write_rows(tmp_path / f"{pid}.json", [_row_back(b) for b in mine]))
    bases, info = wet.load_bases([str(f) for f in files], CMDS, {})
    return wet.build_draft(bases, CMDS, info, [], {"deltas": (0.0, 0.1), "target_size": 12})


def _row_back(b: dict[str, Any]) -> dict[str, Any]:
    """판정 전 기록 → 서비스 행 (two_person_bases 행을 파일로 쓰기 위해)."""
    sc = scores(b["top"], b["score"], b["margin"], b["runner_up"])
    return svc_row(b["file"], b["kind"], b["condition"], b["sentence_id"], sc, session=b["session"],
                   participant=b["participant"], rep=b["rep"])


def test_freeze_draft_schema_and_rule_roundtrip(tmp_path: Path) -> None:
    d = real_draft(tmp_path)
    assert cal.validate_freeze(d) == []
    assert d["status"] == "draft" and d["sealed_check"]["sealed_data_read"] is False
    assert set(EM) <= set(d["reduced_list"]["ids"]) and d["reduced_list"]["size"] == 12
    assert d["reduced_list"]["min_margin"] is None  # 후보별 점수가 없으면 축소 목록 기준값은 정하지 않는다
    assert all(len(v) == 64 for v in d["code"]["files"].values()) and "scripts/ward_tools/metrics.py" in d["code"]["files"]
    assert d["full_list"]["env"]["LIPNET_OLKAVS_WARD_MIN_MARGIN"] == str(d["full_list"]["min_margin"])
    rule = cal.rule_from_freeze(d)
    assert (rule.min_margin, rule.min_score, rule.ids) == (d["full_list"]["min_margin"], d["full_list"]["min_score"], None)
    with pytest.raises(ValueError, match="D가 없습니다"):
        cal.rule_from_freeze(d, with_enrollment=True)
    with pytest.raises(ValueError, match="축소 목록에 기준값"):
        cal.rule_from_freeze(d, "reduced")
    json.dumps(d, ensure_ascii=False)  # 그대로 저장할 수 있다


@pytest.mark.parametrize("mutate,expect", [
    (lambda d: d.update(schema="x"), "schema"),
    (lambda d: d.update(status="final"), "status"),
    (lambda d: d["full_list"].update(min_score=0.5), "min_score"),
    (lambda d: d["reduced_list"].update(ids=["W01", "W05"], size=2), "긴급 명령"),
    (lambda d: d["reduced_list"].update(size=3), "size"),
    (lambda d: d["enrollment"].update(max_distance=-1), "max_distance"),
    (lambda d: d["decision_rule"].update(possible_min_score=1.0), "possible_min_score"),
    (lambda d: d["commands"].update(sha256="abc"), "sha256"),
    (lambda d: d["data"]["rows_files"][0].update(path="C:/웹캠 영상 P02/p02최종/x.json"), "봉인"),
    (lambda d: d["data"]["rows_files"][0].update(participants=["P02"]), "봉인"),
    (lambda d: d["sealed_check"].update(sealed_data_read=True), "sealed_data_read"),
    (lambda d: d.update(synthetic=True), "합성"),
    (lambda d: d["code"]["files"].update({"x.py": "nothex"}), "code.files"),
])
def test_freeze_validation_catches_problems(tmp_path: Path, mutate: Any, expect: str) -> None:
    d = real_draft(tmp_path)
    mutate(d)
    problems = cal.validate_freeze(d)
    assert any(expect in p for p in problems), problems


def test_synthetic_allowed_only_for_pipeline_tests(tmp_path: Path) -> None:
    d = real_draft(tmp_path)
    d["synthetic"] = True
    assert cal.validate_freeze(d, allow_synthetic=True) == []
    assert cal.rule_from_freeze(d).min_margin == d["full_list"]["min_margin"]


# ---------------------------------------------------------------------------
# 봉인 확인
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path,sealed", [
    ("C:/Users/me/Desktop/웹캠 영상 모음/웹캠 영상 P02/p02최종", True),
    ("D:/x/silent_talk_P02", True), ("C:/x/p02최종", True), ("C:/x/P02_silent_W01_1.webm", True),
    ("C:/x/웹캠 영상 P01", False), ("C:/x/P021", False), ("C:/x/AP02x", False), ("C:/x/P03/ward_tune", False),
])
def test_sealed_path_detection(path: str, sealed: bool) -> None:
    assert common.is_sealed_path(path) is sealed


def test_sealed_reasons_and_require() -> None:
    assert common.sealed_reasons([], [{"participant": "p02", "session": "pilot"}])
    assert common.sealed_reasons([], [{"participant": "P01", "session": "ward_eval"}])
    assert common.sealed_reasons(["C:/x/P01"], [{"participant": "P01", "session": "ward_tune"}]) == []
    with pytest.raises(common.SealedDataError):
        common.require_unsealed([], [{"session": "ward_eval"}])
    common.require_unsealed([], [{"session": "ward_eval"}], allow_sealed=True)


def make_clip_folder(folder: Path, participant: str, session: str = "ward_tune") -> Path:
    """빈 영상 + 클립 기록 하나 (모델·서버 없이 거부 경로만 시험)."""
    folder.mkdir(parents=True, exist_ok=True)
    name = f"{participant}_silent_W01_1"
    (folder / f"{name}.webm").write_bytes(b"")
    meta = {"participant": participant, "condition": "silent", "sentenceId": "W01", "kind": "fixed", "rep": 1,
            "text": CMDS["texts"]["W01"], "session": session, "posture": "upright", "file": f"{name}.webm"}
    (folder / f"{name}.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    return folder


def test_service_batch_refuses_sealed_folders_before_contacting_server(tmp_path: Path) -> None:
    out = tmp_path / "out.json"
    for folder in (make_clip_folder(tmp_path / "a", "P02"), make_clip_folder(tmp_path / "b", "P05", "ward_eval")):
        rc = wsb.main(["--clips", str(folder), "--out", str(out), "--server", "http://127.0.0.1:9", "--wait-ready", "0"])
        assert rc == 2 and not out.exists()


def test_enroll_refuses_sealed_folders(tmp_path: Path) -> None:
    folder = make_clip_folder(tmp_path / "c", "P02")
    assert ward_enroll.main(["roi", "--clips", str(folder), "--out", str(tmp_path / "o")]) == 2
    assert not (tmp_path / "o" / "roi_index.json").exists()
    idx = tmp_path / "f"
    idx.mkdir()
    (idx / "features_index.json").write_text(json.dumps({"clips": [{"participant": "P05", "session": "ward_eval"}]}),
                                             encoding="utf-8")
    assert ward_enroll.main(["apply", "--enroll-dir", str(idx), "--out", str(tmp_path / "x.json")]) == 2


def test_calibrate_never_accepts_sealed_rows(tmp_path: Path) -> None:
    rows = [svc_row("P02_silent_W01_1", "fixed", "silent", "W01", scores("W01", -3, 1), participant="P02")]
    f = write_rows(tmp_path / "rows.json", rows)
    g = write_rows(tmp_path / "rows2.json", [svc_row("x", "fixed", "silent", "W01", scores("W01", -3, 1),
                                                     session="ward_eval", participant="P05")])
    for path in (f, g):
        assert wet.main(["calibrate", "--tune-rows", str(path), "--out", str(tmp_path / "d.json")]) == 2
    assert not (tmp_path / "d.json").exists()


def test_report_on_sealed_rows_needs_flag_and_committed_frozen_file(tmp_path: Path) -> None:
    d = real_draft(tmp_path)
    fz = tmp_path / "freeze.json"
    fz.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    rows = write_rows(tmp_path / "sealed.json", [svc_row("P05_silent_W01_1", "fixed", "silent", "W01", scores("W01", -3, 1),
                                                         session="ward_eval", participant="P05")])
    assert wet.main(["report", "--freeze", str(fz), "--rows", str(rows)]) == 2
    assert wet.main(["report", "--freeze", str(fz), "--rows", str(rows), "--allow-sealed"]) == 2  # 초안·커밋 안 됨
    frozen = {"schema": cal.FREEZE_SCHEMA, "phrase_set": "ward", "status": "frozen"}
    assert wet.sealed_ok_for_report(frozen, {"tracked": True, "clean": True, "commit": "a" * 40}) == []
    assert len(wet.sealed_ok_for_report({**frozen, "status": "draft"},
                                        {"tracked": False, "clean": False, "commit": None})) == 2
    assert wet.sealed_ok_for_report({"status": "frozen"}, {"tracked": True, "clean": True, "commit": "a" * 40})  # 형식 아님
    # 봉인이 아닌 자료는 초안으로도 보고할 수 있다
    ok_rows = write_rows(tmp_path / "ok.json", [svc_row("P01_silent_W01_9", "fixed", "silent", "W01", scores("W01", -3, 1))])
    assert wet.main(["report", "--freeze", str(fz), "--rows", str(ok_rows), "--out", str(tmp_path / "r.json")]) == 0
    assert (tmp_path / "r.md").read_text(encoding="utf-8").startswith("# 병실 명령 평가 결과")


# ---------------------------------------------------------------------------
# 지문·파일
# ---------------------------------------------------------------------------

def test_listing_fingerprint_format(tmp_path: Path) -> None:
    (tmp_path / "b.webm").write_bytes(b"bb")
    (tmp_path / "a.json").write_bytes(b"{}")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.webm").write_bytes(b"c")
    fp = common.folder_fingerprint(tmp_path)
    text = "\n".join(f"{n}\t{len(b)}\t{hashlib.sha256(b).hexdigest()}" for n, b in (("a.json", b"{}"), ("b.webm", b"bb")))
    assert fp["listing_sha256"] == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert (fp["files"], fp["videos"], fp["bytes"]) == (2, 1, 4)


def test_read_rows_accepts_legacy_list_and_object(tmp_path: Path) -> None:
    (tmp_path / "l.json").write_text(json.dumps([{"file": "a"}]), encoding="utf-8")
    head, rows = common.read_rows(tmp_path / "l.json")
    assert head["schema"] == "legacy-row-list" and rows == [{"file": "a"}]
    common.atomic_write_json(tmp_path / "o.json", {"schema": "s", "rows": [{"file": "b"}]})
    head, rows = common.read_rows(tmp_path / "o.json")
    assert head["schema"] == "s" and rows[0]["file"] == "b"
    with pytest.raises(ValueError):
        (tmp_path / "bad.json").write_text("{}", encoding="utf-8")
        common.read_rows(tmp_path / "bad.json")


def test_load_bases_rejects_wrong_phrase_set_and_duplicates(tmp_path: Path) -> None:
    r = svc_row("a", "fixed", "silent", "W01", scores("W01", -3, 1))
    with pytest.raises(ValueError, match="묶음"):
        wet.load_bases([str(write_rows(tmp_path / "s.json", [r], phrase_set="sentences"))], CMDS, {})
    bad = json.loads(json.dumps(r))
    bad["results"][0]["phrase"]["set"] = "sentences"
    with pytest.raises(ValueError, match="phrase.set"):
        wet.load_bases([str(write_rows(tmp_path / "p.json", [bad]))], CMDS, {})
    f = str(write_rows(tmp_path / "d.json", [r]))
    with pytest.raises(ValueError, match="두 번"):
        wet.load_bases([f, f], CMDS, {})


def test_service_roi_settings_match_backend_config() -> None:
    import config
    assert common.SERVICE_ROI["segment_seconds"] == config.OLKAVS_SEGMENT_SECONDS
    assert common.SERVICE_ROI["min_window_seconds"] == config.OLKAVS_MIN_WINDOW_SECONDS
    assert common.SERVICE_ROI["cut_search_seconds"] == config.OLKAVS_CUT_SEARCH_SECONDS
    import olkavs_transcribe as ot
    assert common.VAD_POST_PAD_MS == ot.VAD_POST_PAD_SECONDS * 1000
    from services import olkavs_service as svc
    assert (common.FAIL_TEXT, common.NO_SPEECH_TEXT, common.UNREC_TEXT) == (svc.FAIL_TEXT, svc.NO_SPEECH_TEXT, svc.UNREC_TEXT)


# ---------------------------------------------------------------------------
# 서비스 일괄 실행 순수 함수
# ---------------------------------------------------------------------------

def test_multipart_body_and_helpers(tmp_path: Path) -> None:
    v = tmp_path / "clip.webm"
    v.write_bytes(b"\x1a\x45\xdf\xa3data")
    body, ctype = wsb.multipart_body(v, "ward", boundary="BND")
    assert ctype == "multipart/form-data; boundary=BND"
    assert b'name="phrase_set"\r\n\r\nward\r\n' in body and b'filename="recording.webm"' in body
    assert b"\x1a\x45\xdf\xa3data\r\n--BND--\r\n" in body
    assert wsb.phrase_set_mismatch([{"phrase": {"set": "sentences"}}, {"text": "x"}], "ward") == ["sentences"]
    assert wsb.phrase_set_mismatch([{"phrase": {"set": "ward"}}], "ward") == []
    kept = wsb.kept_rows([{"file": "a", "status": "done"}, {"file": "b", "status": "error"}], redo=False)
    assert list(kept) == ["a"] and wsb.kept_rows([{"file": "a", "status": "done"}], redo=True) == {}
    assert wsb.thresholds_from_health({"model": {"phrase_sets": {"ward": {"min_margin": 0.5}}}}, "ward") == {"min_margin": 0.5}
    assert wsb.thresholds_from_health({"model": {}}, "ward") is None


def test_health_problems() -> None:
    good = {"status": "ok", "model_backend": "olkavs",
            "model": {"roi_worker": "ready", "infer_worker": "ready", "checkpoint_sha256": wsb.EXPECTED_CHECKPOINT,
                      "ward_phrases": {"enabled": True, "count": 20, "version": CMDS["version"]}}}
    assert wsb.health_problems(good, "ward", CMDS) == []
    assert wsb.health_problems(good, "sentences", None) == []
    bad = json.loads(json.dumps(good))
    bad["model"]["infer_worker"] = "stopped"
    bad["model"]["ward_phrases"]["count"] = 0
    probs = wsb.health_problems(bad, "ward", CMDS)
    assert any("infer_worker" in p for p in probs) and any("20개" in p for p in probs)
    health, probs = wsb.wait_ready("http://x", "ward", CMDS, 0.0, fetch=lambda url: bad, sleep=lambda s: None)
    assert probs and health is bad


# ---------------------------------------------------------------------------
# 등록 거리
# ---------------------------------------------------------------------------

def test_dtw_distance_properties() -> None:
    rng = np.random.default_rng(0)
    a = enroll.unit_rows(rng.normal(size=(12, 8)))
    assert enroll.dtw_distance(a, a) == pytest.approx(0.0, abs=1e-12)
    stretched = enroll.unit_rows(np.repeat(np.asarray(a), 2, axis=0))
    assert enroll.dtw_distance(a, stretched) == pytest.approx(0.0, abs=1e-12)  # 시간 늘림은 거리 0
    b = enroll.unit_rows(rng.normal(size=(9, 8)))
    assert enroll.dtw_distance(a, b) == pytest.approx(enroll.dtw_distance(b, a), abs=1e-12)
    # 한 칸짜리: 코사인 거리 그대로
    x, y = enroll.unit_rows([[1.0, 0.0]]), enroll.unit_rows([[0.0, 1.0]])
    assert enroll.dtw_distance(x, y) == pytest.approx(1.0)
    d = enroll.template_distances(a, {"W01": [a, b], "W02": [b], "W03": []})
    assert d["W01"] == pytest.approx(0.0, abs=1e-12) and d["W02"] > 0 and d["W03"] is None
    with pytest.raises(ValueError):
        enroll.unit_rows(np.zeros((3, 4)))


def test_enroll_templates_and_distances_per_person() -> None:
    def rec(pid: str, cond: str, sid: str, rep: int, kind: str = "fixed", session: str = "ward_tune",
            posture: str = "upright") -> dict[str, Any]:
        return {"file": f"{pid}_{cond}_{sid}_{rep}.webm", "participant": pid, "session": session, "kind": kind,
                "condition": cond, "sentence_id": sid, "rep": rep, "posture": posture, "status": "ok"}
    recs = [rec("P01", "voiced", "W01", 1), rec("P01", "voiced", "W02", 1), rec("P01", "silent", "W01", 1),
            rec("P01", "silent", "W02", 2), rec("P01", "reclined", "W01", 1, posture="reclined"),
            rec("P01", "voiced", "A01", 1, session="pilot"), rec("P03", "silent", "W01", 1)]
    assert ward_enroll.is_template(recs[0], "voiced") and not ward_enroll.is_template(recs[2], "voiced")
    assert ward_enroll.is_template(recs[2], "silent1") and ward_enroll.is_template(recs[2], "both")
    assert not ward_enroll.is_template(recs[4], "both") and not ward_enroll.is_template(recs[5], "both")
    feats = {"W01": [[1.0, 0.0]] * 3, "W02": [[0.0, 1.0]] * 3}
    out = ward_enroll.enroll_distances(recs, "voiced", ["W01", "W02"], load=lambda r: np.asarray(feats.get(
        r["sentence_id"], [[1.0, 1.0]] * 3)))
    by = {o["file"]: o for o in out}
    assert by["P01_voiced_W01_1.webm"]["tmpl_dists"] is None and by["P01_voiced_W01_1.webm"]["is_template"]
    assert by["P01_silent_W01_1.webm"]["tmpl_dists"]["W01"] == pytest.approx(0.0, abs=1e-12)
    assert by["P01_silent_W02_2.webm"]["tmpl_dists"]["W01"] == pytest.approx(1.0)
    assert by["P01_voiced_A01_1.webm"]["tmpl_dists"]["W02"] == pytest.approx(1 - math.sqrt(0.5))
    assert by["P03_silent_W01_1.webm"]["tmpl_dists"] is None  # P03 템플릿 없음(다른 사람 템플릿은 쓰지 않는다)


def test_pseudo_row_feeds_metrics() -> None:
    rank = {"id": "W05", "score": -3.0, "margin": 0.7, "runner_up_id": "W06"}
    rec = {"file": "a", "participant": "P01", "session": "ward_tune", "kind": "fixed", "condition": "silent",
           "sentence_id": "W05", "rep": 2, "posture": "upright", "status": "ok", "start_ms": 100, "end_ms": 3000,
           "text_rank": rank}
    e = mt.evaluate(base_of(ward_enroll.pseudo_row(rec)), RULE)
    assert (e["outcome"], e["chosen_id"]) == (mt.OUT_REQUEST, "W05")
    for status, outcome in (("no_speech", mt.OUT_NO_SPEECH), ("lip_fail", mt.OUT_LIP_FAIL), ("no_result", mt.OUT_NO_RESULT),
                            ("failed", mt.OUT_ERROR)):
        e = mt.evaluate(base_of(ward_enroll.pseudo_row({**rec, "status": status})), RULE)
        assert e["outcome"] == outcome
    assert ward_enroll.segment_status({"status": "ok", "single": False, "segments": [{"status": "failed"}, {"status": "ok"}]}) \
        == "no_result"


def test_tools_have_no_sentence_mode_side_effects() -> None:
    # 문장 모드 기준값·목록 파일은 읽기만 한다: 동결 지문 대상에 문장 목록·VAD 코드가 들어 있고, 병실 목록은 commands.json이다
    assert "scripts/olkavs_transcribe.py" in wet.CODE_FILES and "frontend/src/ward/commands.json" in wet.CODE_FILES
    assert os.path.normcase(str(common.COMMANDS_PATH)).endswith(os.path.normcase(os.path.join("ward", "commands.json")))
    # 화면 판정 코드도 동결 지문에 들어간다(판정 규칙이 화면과 같아야 한다)
    assert {"frontend/src/ward/wardMachine.js", "frontend/src/ward/wardResult.js"} <= set(wet.CODE_FILES)


# ---------------------------------------------------------------------------
# 화면과 같은 결과 해석 (phrase.set·목록 밖 ID·표시만 있는 창)
# ---------------------------------------------------------------------------

def test_phrase_without_ward_set_or_unknown_id_is_no_result_like_ui() -> None:
    sc = scores("W05", -3.0, 1.2)
    legacy = svc_row("a", "fixed", "silent", "W05", sc)
    del legacy["results"][0]["phrase"]["set"]
    sentence = json.loads(json.dumps(legacy))
    sentence["results"][0]["phrase"].update(id="A01", set="sentences")
    unknown = svc_row("b", "fixed", "silent", "W05", sc)
    unknown["results"][0]["phrase"]["id"] = "ZZ9"
    for row, why in ((legacy, "other_set"), (sentence, "other_set"), (unknown, "unknown_id")):
        b = base_of(row)
        e = mt.evaluate(b, RULE)
        assert (b["phrase_problem"], e["outcome"], e["chosen_id"]) == (why, mt.OUT_NO_RESULT, None)
    # 2등 ID가 목록 밖이면 2등 없음 → 1등이 일반이면 긴급 가능성도 아님
    odd = svc_row("c", "fixed", "silent", "W05", scores("W05", -3.0, 0.1))
    odd["results"][0]["phrase"]["runner_up_id"] = "ZZ9"
    b = base_of(odd)
    assert b["runner_up"] is None and mt.evaluate(b, RULE)["outcome"] == mt.OUT_NOT_CHOSEN


def test_load_bases_rejects_legacy_rows_and_other_command_lists(tmp_path: Path) -> None:
    r = svc_row("a", "fixed", "silent", "W01", scores("W01", -3, 1))
    legacy = json.loads(json.dumps(r))
    del legacy["results"][0]["phrase"]["set"]
    with pytest.raises(ValueError, match="phrase.set"):
        wet.load_bases([str(write_rows(tmp_path / "l.json", [legacy]))], CMDS, {})
    (tmp_path / "list.json").write_text(json.dumps([legacy], ensure_ascii=False), encoding="utf-8")  # 예전 행 목록 형식
    with pytest.raises(ValueError, match="phrase.set"):
        wet.load_bases([str(tmp_path / "list.json")], CMDS, {})
    with pytest.raises(ValueError, match="명령 목록"):
        wet.load_bases([str(write_rows(tmp_path / "c.json", [r], commands={"sha256": "0" * 64}))], CMDS, {})


def test_mixed_marker_windows_are_lip_fail_like_ui() -> None:
    mixed = svc_row("a", "fixed", "silent", "W01", results=[{"text": common.NO_SPEECH_TEXT}, {"text": common.FAIL_TEXT}])
    assert mt.evaluate(base_of(mixed), RULE)["outcome"] == mt.OUT_LIP_FAIL
    allns = svc_row("b", "fixed", "silent", "W01", results=[{"text": common.NO_SPEECH_TEXT}, {"text": common.NO_SPEECH_TEXT}])
    assert mt.evaluate(base_of(allns), RULE)["outcome"] == mt.OUT_NO_SPEECH


# ---------------------------------------------------------------------------
# 짧은 긴급 표현이 일반 명령으로 실행되는 경우
# ---------------------------------------------------------------------------

def test_short_phrase_shown_as_normal_command_blocks_fit_and_is_headlined() -> None:
    rows = [svc_row("e2", "short", "silent", "E02", scores("W07", -3.0, 0.9)),          # 일반 명령 1등 → 막아야 함
            svc_row("e1", "short", "silent", "E01", scores("W02", -3.0, 0.9)),          # 긴급 1등 → 바라는 결과
            svc_row("g", "fixed", "silent", "W05", scores("W05", -3.0, 1.5))]
    bases = [base_of(r) for r in rows]
    items = cal.fit_items(bases, emergency=EM)
    assert {i["file"]: i["cls"] for i in items} == {"e2": "block", "g": "good"}
    h = cal.headline(bases, RULE)
    assert h["short_as_request_k"] == 1 and h["short_as_request_clips"] == ["e2->W07"] and h["negatives_shown_k"] == 0
    g = cal.grid_fit(items)
    assert g["feasible"] and g["M"] > 0.9  # 짧은 긴급 표현(차이 0.9)을 막는 기준
    s = mt.summarize(mt.evaluate_all(bases, RULE), RULE, CMDS)
    rows_md = {name: (crit, v) for name, crit, v in rp.headline_rows(s)}
    assert rows_md["짧은 긴급 표현 → 일반 요청으로 실행 (틀린 요청)"] == ("0", "1/2")
    assert "e2->W07" in rp.detail(s)


# ---------------------------------------------------------------------------
# 긴급 가능성 점수 하한: 화면에 없으므로 기본 끔
# ---------------------------------------------------------------------------

def test_floor_is_opt_in_and_needs_ui_flag(tmp_path: Path) -> None:
    d = real_draft(tmp_path)
    assert d["decision_rule"]["possible_min_score"] is None and d["decision_rule"]["floor_implemented_in_ui"] is False
    d["decision_rule"]["possible_min_score"] = -3.8
    assert any("화면" in p for p in cal.validate_freeze(d))
    with pytest.raises(ValueError, match="화면"):
        cal.rule_from_freeze(d)
    d["decision_rule"]["floor_implemented_in_ui"] = True
    assert cal.validate_freeze(d) == []
    assert cal.rule_from_freeze(d).possible_min_score is None  # 기본: 화면 규칙(하한 없음)
    assert cal.rule_from_freeze(d, apply_floor=True).possible_min_score == -3.8


def test_build_draft_with_floor_option_is_not_freezable(tmp_path: Path) -> None:
    rows = [svc_row("P01_silent_W03_1", "fixed", "silent", "W03", scores("W05", -3.2, 0.3, "W03")),
            svc_row("P01_silent_W05_1", "fixed", "silent", "W05", scores("W05", -3.0, 1.0)),
            svc_row("P03_silent_W03_1", "fixed", "silent", "W03", scores("W06", -3.3, 0.2, "W03"), participant="P03"),
            svc_row("P03_silent_W05_1", "fixed", "silent", "W05", scores("W05", -3.0, 1.0), participant="P03")]
    files = [write_rows(tmp_path / f"{p}.json", [r for r in rows if r["participant"] == p]) for p in ("P01", "P03")]
    bases, info = wet.load_bases([str(f) for f in files], CMDS, {})
    plain = wet.build_draft(bases, CMDS, info, [], {"deltas": (0.0,), "target_size": 12})
    assert plain["decision_rule"]["possible_min_score"] is None
    floored = wet.build_draft(bases, CMDS, info, [], {"deltas": (0.0,), "target_size": 12, "use_floor": True})
    assert floored["decision_rule"]["possible_min_score"] is not None
    assert any("화면" in p for p in cal.validate_freeze(floored))


def test_report_primary_has_no_floor_and_floored_is_secondary(tmp_path: Path) -> None:
    d = real_draft(tmp_path)
    d["decision_rule"].update(possible_min_score=-3.5, floor_implemented_in_ui=True)
    rows = [svc_row("P01_silent_A01_1", "fixed", "silent", "A01", scores("W05", -3.9, 0.2, "W10"), session="pilot")]
    bases = [base_of(r) for r in rows]
    res = wet.run_report(d, bases, CMDS)
    assert res["full"]["all"]["false_alarms"]["total"]["k"] == 1        # 화면 규칙: 2등 긴급 → 긴급 가능성
    assert res["full_floored"]["all"]["false_alarms"]["total"]["k"] == 0  # 하한 적용(보조)


def test_markdown_report_shows_server_accepted_mismatch(tmp_path: Path) -> None:
    d = real_draft(tmp_path)
    fz = tmp_path / "freeze.json"
    fz.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    r = svc_row("P01_silent_W01_9", "fixed", "silent", "W01", scores("W01", -3.0, 5.0))
    r["results"][0]["phrase"]["accepted"] = False  # 서버가 다른 기준값으로 켜짐
    rows = write_rows(tmp_path / "rows.json", [r])
    assert wet.main(["report", "--freeze", str(fz), "--rows", str(rows), "--out", str(tmp_path / "r.json"),
                     "--skip-folder-check"]) == 0
    md = (tmp_path / "r.md").read_text(encoding="utf-8")
    assert "주의: 서버 accepted가 동결 기준값과 다른 클립이 1개" in md and "다름 1개 (P01_silent_W01_9)" in md


# ---------------------------------------------------------------------------
# 봉인: --allow-sealed는 커밋된 동결 파일과 함께만
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pid,session,sealed", [
    ("P02", "ward_tune", True), (" p02 ", "ward_tune", True), ("P2", "ward_tune", True), ("P002", "pilot", True),
    ("P09", "WARD_EVAL", True), ("P09", " ward_eval2 ", True), ("P20", "ward_tune", False), ("P12", "eval", False),
])
def test_sealed_participant_and_session_normalisation(pid: str, session: str, sealed: bool) -> None:
    assert bool(common.sealed_reasons([], [{"participant": pid, "session": session}])) is sealed


def frozen_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, commit_time: str = "2026-11-11T20:00:00+09:00",
                **extra: Any) -> Path:
    """status frozen 동결 파일 + git 커밋 상태를 흉내 낸다 (검사에서 실제 커밋을 만들지 않는다)."""
    d = real_draft(tmp_path)
    d.update(status="frozen", **extra)
    fz = tmp_path / "frozen.json"
    fz.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(common, "freeze_commit_state", lambda path: {"tracked": True, "clean": True, "commit": "c" * 40,
                                                                     "commit_time": commit_time, "path": str(path)})
    return fz


def test_freeze_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert common.freeze_gate(None, False) is None
    with pytest.raises(common.SealedDataError, match="--freeze"):
        common.freeze_gate(None, True)
    draft = tmp_path / "draft.json"
    draft.write_text(json.dumps(real_draft(tmp_path), ensure_ascii=False), encoding="utf-8")
    with pytest.raises(common.SealedDataError, match="frozen"):
        common.freeze_gate(draft, True)
    info = common.freeze_gate(draft, False)  # 봉인이 아닌 자료에는 초안도 정보로만 쓴다
    assert info["sealed_ok"] is False and info["sha256"] == common.file_sha256(draft)
    fz = frozen_file(tmp_path, monkeypatch)
    ok = common.freeze_gate(fz, True)
    assert ok["sealed_ok"] and ok["commit"] == "c" * 40 and ok["full_list"]["min_margin"] is not None
    assert common.freeze_commit_state.__name__ == "<lambda>"


def test_freeze_commit_state_outside_repo_is_not_committed(tmp_path: Path) -> None:
    s = wet.freeze_commit_state(tmp_path / "x.json")
    assert s["tracked"] is False and s["commit"] is None


def test_sealed_tools_need_freeze_with_allow_sealed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    folder = make_clip_folder(tmp_path / "w", "P05", "ward_eval")
    out = tmp_path / "out.json"
    base = ["--clips", str(folder), "--out", str(out), "--server", "http://127.0.0.1:9", "--wait-ready", "0"]
    assert wsb.main(base + ["--allow-sealed"]) == 2 and not out.exists()   # 동결 파일 없음
    draft = tmp_path / "draft.json"
    draft.write_text(json.dumps(real_draft(tmp_path), ensure_ascii=False), encoding="utf-8")
    assert wsb.main(base + ["--allow-sealed", "--freeze", str(draft)]) == 2  # 초안
    # 클립 기록을 읽기 전에 거부한다: 폴더가 없어도 동결 파일 문제로 먼저 멈춘다
    assert wsb.main(["--clips", str(tmp_path / "none"), "--out", str(out), "--allow-sealed", "--wait-ready", "0"]) == 2
    assert ward_enroll.main(["roi", "--clips", str(folder), "--out", str(tmp_path / "o"), "--allow-sealed"]) == 2
    assert ward_enroll.main(["features", "--out", str(tmp_path / "o"), "--allow-sealed"]) == 2
    assert ward_enroll.main(["run", "--clips", str(folder), "--out", str(tmp_path / "o"), "--allow-sealed"]) == 2
    assert ward_enroll.main(["apply", "--enroll-dir", str(tmp_path / "o"), "--out", str(tmp_path / "x.json"),
                             "--allow-sealed"]) == 2
    rows = write_rows(tmp_path / "rows.json", [svc_row("x", "fixed", "silent", "W01", scores("W01", -3, 1), session="ward_eval",
                                                      participant="P05")])
    assert wet.main(["metrics", "--rows", str(rows), "--allow-sealed"]) == 2
    assert wet.main(["metrics", "--rows", str(rows)]) == 2
    # 커밋된 frozen 동결 파일이면 봉인 확인을 지나 서버 확인까지 간다(서버가 없어 3)
    fz = frozen_file(tmp_path, monkeypatch)
    assert wsb.main(base + ["--allow-sealed", "--freeze", str(fz)]) == 3
    assert wet.main(["metrics", "--rows", str(rows), "--allow-sealed", "--freeze", str(fz)]) == 0


def test_enroll_features_of_sealed_data_must_come_from_same_freeze(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fz = frozen_file(tmp_path, monkeypatch)
    idx = tmp_path / "f"
    idx.mkdir()
    clip = {"participant": "P05", "session": "ward_eval", "file": "a.webm", "status": "no_speech"}
    (idx / "features_index.json").write_text(json.dumps({"clips": [clip]}), encoding="utf-8")
    out = tmp_path / "x.json"
    assert ward_enroll.main(["apply", "--enroll-dir", str(idx), "--out", str(out), "--allow-sealed", "--freeze", str(fz)]) == 2
    sha = common.file_sha256(fz)
    (idx / "features_index.json").write_text(json.dumps({"clips": [clip], "freeze": {"sha256": sha},
                                                         "clips_listing": {"listing_sha256": "a" * 64}}), encoding="utf-8")
    assert ward_enroll.main(["apply", "--enroll-dir", str(idx), "--out", str(out), "--allow-sealed", "--freeze", str(fz)]) == 0
    res = json.loads(out.read_text(encoding="utf-8"))
    assert res["freeze"]["sha256"] == sha and res["inputs_listing"][0]["listing_sha256"] == "a" * 64 and res["git_head"]


def test_apply_takes_template_from_freeze_and_refuses_other(tmp_path: Path) -> None:
    d = real_draft(tmp_path)
    d["enrollment"].update(max_distance=0.27, template="voiced", unbounded=False)
    assert cal.validate_freeze(d) == []
    fz = tmp_path / "fz.json"
    fz.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    idx = tmp_path / "f"
    idx.mkdir()
    (idx / "features_index.json").write_text(json.dumps({"clips": []}), encoding="utf-8")
    out = tmp_path / "x.json"
    assert ward_enroll.main(["apply", "--enroll-dir", str(idx), "--out", str(out), "--freeze", str(fz), "--template", "both"]) == 2
    assert ward_enroll.main(["apply", "--enroll-dir", str(idx), "--out", str(out), "--freeze", str(fz)]) == 0
    res = json.loads(out.read_text(encoding="utf-8"))
    assert res["template"] == "voiced" and res["rule"]["max_distance"] == 0.27
    # report도 등록 파일의 템플릿·이름·명령 목록을 동결 값과 대조한다
    einfo = [{"path": "e.json", "template": "both", "feature": common.FEATURE_NAME, "distance": common.DISTANCE_NAME,
              "commands_sha256": CMDS["sha256"]}]
    assert any("템플릿" in p for p in wet.enroll_consistency(d, einfo))
    einfo[0]["template"] = "voiced"
    assert wet.enroll_consistency(d, einfo) == []
    einfo[0]["commands_sha256"] = "0" * 64
    assert any("명령 목록" in p for p in wet.enroll_consistency(d, einfo))


@pytest.mark.parametrize("mutate,expect", [
    (lambda d: d["enrollment"].update(max_distance=0.27, template="voiced", unbounded=True), "unbounded"),
    (lambda d: d["enrollment"].update(max_distance=0.27, template="mixed", unbounded=False), "template"),
    (lambda d: d["enrollment"].update(max_distance=0.27, template="voiced", feature="x"), "feature"),
    (lambda d: d.update(sealed_folders=[{"path": "x", "files": 3}]), "sealed_folders"),
])
def test_freeze_validation_enrollment_and_sealed_folders(tmp_path: Path, mutate: Any, expect: str) -> None:
    d = real_draft(tmp_path)
    mutate(d)
    assert any(expect in p for p in cal.validate_freeze(d))


def test_order_problems_requires_same_freeze_later_commit_and_time() -> None:
    sha = "f" * 64
    good = {"path": "rows.json", "freeze": {"sha256": sha}, "git_head": "h" * 40, "created": "2026-11-12T10:00:00+00:00"}
    yes = lambda a, b: True  # noqa: E731
    assert common.order_problems([good], sha, "c" * 40, "2026-11-11T20:00:00+09:00", yes) == []
    assert any("freeze.sha256" in p for p in common.order_problems([{**good, "freeze": None}], sha, "c" * 40,
                                                                   "2026-11-11T20:00:00+09:00", yes))
    assert any("git HEAD" in p for p in common.order_problems([good], sha, "c" * 40, "2026-11-11T20:00:00+09:00",
                                                              lambda a, b: False))
    early = {**good, "created": "2026-11-11T10:00:00+00:00"}  # 동결 커밋(11:00 UTC)보다 이르다
    assert any("만든 시각" in p for p in common.order_problems([early], sha, "c" * 40, "2026-11-11T20:00:00+09:00", yes))
    assert common.order_problems([{**good, "created": None}], sha, "c" * 40, "2026-11-11T20:00:00+09:00", yes)
    # 실제 git: HEAD는 HEAD의 조상(같은 커밋)이다
    head = common.git("rev-parse", "HEAD")
    assert head and common.git_is_ancestor(head, head) and not common.git_is_ancestor("0" * 40, head)


def sealed_rows_file(tmp_path: Path, folder: Path, freeze_sha: str | None, created: str, git_head: str) -> Path:
    """봉인(ward_eval) 결과 행 파일 + 머리(폴더 지문, 동결 파일, 만든 시각, git HEAD)."""
    fp = common.folder_fingerprint(folder)
    rows = [svc_row("P05_silent_W01_1", "fixed", "silent", "W01", scores("W01", -3, 1), session="ward_eval",
                    participant="P05")]
    return write_rows(tmp_path / "sealed_rows.json", rows, clips_dir=str(folder), clips_listing=fp,
                      freeze={"sha256": freeze_sha} if freeze_sha else None, created=created, git_head=git_head)


def test_report_on_sealed_rows_checks_order_and_folder_fingerprint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    folder = make_clip_folder(tmp_path / "ward_eval_P05", "P05", "ward_eval")
    listing = common.folder_fingerprint(folder)["listing_sha256"]
    fz = frozen_file(tmp_path, monkeypatch, sealed_folders=[{"path": "x", "files": 2, "listing_sha256": listing}])
    sha = common.file_sha256(fz)
    monkeypatch.setattr(common, "git_is_ancestor", lambda a, b: True)
    out = tmp_path / "rep.json"
    args = ["report", "--freeze", str(fz), "--allow-sealed", "--out", str(out)]
    # 동결 전에 만든 결과(만든 시각이 이르다) → 거부
    rows = sealed_rows_file(tmp_path, folder, sha, "2026-11-11T10:00:00+00:00", "h" * 40)
    assert wet.main(args + ["--rows", str(rows)]) == 2 and not out.exists()
    # 동결 파일 기록 없음 → 거부
    rows = sealed_rows_file(tmp_path, folder, None, "2026-11-12T10:00:00+00:00", "h" * 40)
    assert wet.main(args + ["--rows", str(rows)]) == 2
    # 조건이 모두 맞으면 보고한다
    rows = sealed_rows_file(tmp_path, folder, sha, "2026-11-12T10:00:00+00:00", "h" * 40)
    assert wet.main(args + ["--rows", str(rows)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["sealed_data"] is True
    # 폴더 지문이 동결 값과 다르면(클립이 바뀜) 거부
    (folder / "extra.txt").write_bytes(b"x")
    rows = sealed_rows_file(tmp_path, folder, sha, "2026-11-12T10:00:00+00:00", "h" * 40)
    assert wet.main(args + ["--rows", str(rows)]) == 2


def test_calibrate_records_sealed_folder_fingerprint_hash_only(tmp_path: Path) -> None:
    sealed = make_clip_folder(tmp_path / "웹캠 영상 P02" / "ward_eval", "P02", "ward_eval")
    rows = two_person_bases()
    files = [write_rows(tmp_path / f"{p}.json", [_row_back(b) for b in rows if b["participant"] == p]) for p in ("P01", "P03")]
    out = tmp_path / "draft.json"
    rc = wet.main(["calibrate", "--tune-rows", str(files[0]), "--tune-rows", str(files[1]), "--sealed-folder", str(sealed),
                   "--out", str(out), "--skip-folder-check"])
    assert rc == 0
    d = json.loads(out.read_text(encoding="utf-8"))
    assert d["sealed_folders"][0]["listing_sha256"] == common.folder_fingerprint(sealed)["listing_sha256"]
    assert d["sealed_folders"][0]["files"] == 2 and d["sealed_check"]["sealed_data_read"] is False
    assert wet.main(["fingerprint", str(sealed)]) == 0


def test_batch_helpers_resume_and_threshold_check() -> None:
    rows = [{"file": "a", "status": "done", "results": [{"phrase": {"set": "ward"}}]},
            {"file": "b", "status": "done", "results": [{"phrase": {"set": "sentences"}}]},
            {"file": "c", "status": "done", "results": [{"phrase": None, "text": common.NO_SPEECH_TEXT}]}]
    assert sorted(wsb.kept_rows(rows, False, "ward")) == ["a", "c"]  # 묶음이 다른 행은 다시 보낸다
    full = {"min_margin": 0.5, "min_score": -3.4}
    ok = [{"phrase": {"id": "W01", "margin": 0.6, "score": -3.0, "accepted": True}}]
    bad = [{"phrase": {"id": "W01", "margin": 0.45, "score": -3.0, "accepted": True}}]
    assert wsb.threshold_mismatch(ok, full) == [] and wsb.threshold_mismatch(bad, None) == []
    assert wsb.threshold_mismatch(bad, full)[0]["expected"] is False
    assert wsb.threshold_mismatch([{"phrase": None}], full) == []


def test_pseudo_row_marks_ward_set() -> None:
    rec = {"file": "a", "participant": "P01", "session": "ward_tune", "kind": "fixed", "condition": "silent",
           "sentence_id": "W05", "rep": 2, "status": "ok", "text_rank": {"id": "W05", "score": -3.0, "margin": 0.7,
                                                                           "runner_up_id": "W06"}}
    assert ward_enroll.pseudo_row(rec)["results"][0]["phrase"]["set"] == "ward"
    assert ward_enroll.pseudo_row({**rec, "text_rank": None})["results"][0]["phrase"] is None
