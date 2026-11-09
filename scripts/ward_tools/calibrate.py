"""병실 명령 기준값 맞추기 (numpy). 세션 임시 스크립트 p03/calib/step1_phrase.py·step2c_or_lopo.py를 옮겨 일반화했다.

맞추기 규칙(맞추는 사람 자료에서): 표시 중 오답 0, 목록 밖 입력(함정·말하지 않기·문장 촬영 클립)이 명령으로 표시 0,
짧은 긴급 표현이 일반 명령으로 실행 0을 지키고 명령 표시 수를 최대로 한다(짧은 긴급 표현이 긴급으로 가는 것은 막지 않는다). 같은 최대가 여럿이면 가장 가까운 '뒤집힐 클립'과의 여유(safety)가 가장 큰 점, 다시 그 최적 칸의 가운데를
고른다. safety = min(막은 나쁜 클립: max(M - m, S - s), 남긴 정답: min(m - M, s - S)) — 두 축 모두 토큰당 점수 단위다.
clearance(δ)를 주면 나쁜 클립은 한 축에서 δ 이상 떨어져야 막은 것으로 본다(합친 맞추기의 여유 선택지).

축소 목록: P01·P03 혼동만으로 미리 정한다. 긴급 명령(과 --keep)은 지키고, 혼동에 가장 많이 얽힌 명령부터 하나씩 뺀다.
등록 거리 D: 글자 규칙을 통과하지 못한 클립 중 등록 거리만으로 표시되거나 긴급 가능성이 될 나쁜 클립의 최소 거리(막는 값)와,
그보다 작은 정답 거리 중 최대값의 가운데. 막는 값을 모르거나(나쁜 클립 없음) 거리가 있는 목록 밖 클립이 사람마다 MIN_NEGATIVES_FOR_D개
미만이면 D를 정하지 않는다. 세 함수 모두 P02·ward_eval 자료를 받지 않는다(호출하는 CLI가 막는다).
긴급 가능성 점수 하한은 화면(wardMachine.js)에 없으므로 참고로만 계산하고, 동결 초안에는 --floor를 줄 때만 넣는다.
"""
from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Iterable

import numpy as np

from ward_tools import common
from ward_tools import metrics as mt

M_GRID = np.round(np.arange(0.0, 3.0 + 1e-9, 0.005), 3)
S_GRID = np.round(np.arange(-6.0, -1.0 + 1e-9, 0.005), 3)
DEFAULT_DELTAS = (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3)
PROVISIONAL = (0.5, -3.40)  # backend/config.py OLKAVS_WARD_MIN_MARGIN·OLKAVS_WARD_MIN_SCORE 임시값
FREEZE_SCHEMA = common.FREEZE_SCHEMA
MIN_NEGATIVES_FOR_D = 50  # D를 정하려면 사람마다 등록 거리가 있는 목록 밖 클립이 이만큼 있어야 한다


# ---------------------------------------------------------------------------
# 맞추기 항목
# ---------------------------------------------------------------------------

def reranked(bases: Iterable[dict[str, Any]], ids: Iterable[str] | None) -> list[dict[str, Any]]:
    """ids(축소 목록) 안에서 후보별 점수로 다시 순위를 매긴 기록. ids가 None이면 그대로. 점수가 없거나 정렬할 수 없으면 top=None."""
    if ids is None:
        return [dict(b) for b in bases]
    ids = list(ids)
    out = []
    for b in bases:
        c = dict(b)
        r = mt.rank_scores(b["text_scores"], ids) if (b.get("top") is not None and b.get("text_scores")) else None
        if r is None:
            c.update(top=None, runner_up=None, margin=None, score=None)
        else:
            c.update(top=r["id"], runner_up=r["runner_up_id"], margin=r["margin"], score=r["score"])
        out.append(c)
    return out


def fit_items(bases: Iterable[dict[str, Any]], block_roles: Iterable[str] = mt.NEGATIVE_ROLES,
              conditions: Iterable[str] | None = None, ids: Iterable[str] | None = None,
              emergency: Iterable[str] = ()) -> list[dict[str, Any]]:
    """맞추기 항목: 글자 순위가 있는 클립만 {m, s, cls, file, participant}. cls = good(명령 1등 정답) / wrong(명령 1등 오답) /
    block(막아야 할 목록 밖 입력, 1등이 긴급이 아닌 짧은 긴급 표현 = 표시되면 틀린 요청이 실행됨).
    그 밖(1등이 긴급인 짧은 긴급 표현, 목록에서 뺀 명령, conditions 밖의 명령)은 넣지 않는다.
    """
    block_roles = tuple(block_roles)
    conds = set(conditions) if conditions is not None else None
    ids_set = set(ids) if ids is not None else None
    em = set(emergency)
    out = []
    for b in bases:
        if b.get("top") is None:
            continue
        if b["role"] == mt.ROLE_COMMAND:
            if conds is not None and b["condition"] not in conds:
                continue
            if ids_set is not None and b["truth"] not in ids_set:
                continue  # 축소 목록에서 뺀 명령의 클립은 그 목록을 쓰는 사람이 말하지 않는다
            cls = "good" if b["top"] == b["truth"] else "wrong"
        elif b["role"] in block_roles:
            cls = "block"
        elif b["role"] == mt.ROLE_SHORT and b["top"] not in em:
            cls = "block"  # 짧은 긴급 표현이 일반 명령으로 실행되면 틀린 요청 (긴급 1등은 막지 않는다)
        else:
            continue
        out.append({"m": float(b["margin"]), "s": float(b["score"]), "cls": cls, "file": b["file"],
                    "participant": b.get("participant"), "top": b["top"]})
    return out


def grid_fit(items: list[dict[str, Any]], delta: float = 0.0, m_grid: np.ndarray | None = None,
             s_grid: np.ndarray | None = None, chunk: int = 16) -> dict[str, Any]:
    """격자 위에서 나쁜 클립 0, 정답 표시 최대, 여유 최대 점을 고른다 (step1_phrase.grid_eval과 같은 선택).

    m_grid/s_grid를 주지 않으면 모듈의 M_GRID(0~3, 0.005 간격)·S_GRID(-6~-1, 0.005 간격)를 부를 때 읽는다.
    Returns:
        feasible False 또는 {M, S, max_correct_shown, safety, safety_vs_bad, safety_vs_kept, optimal_*_range, n_optimal_grid_points}
    """
    m_grid = M_GRID if m_grid is None else np.asarray(m_grid, dtype=np.float64)
    s_grid = S_GRID if s_grid is None else np.asarray(s_grid, dtype=np.float64)
    if not items:
        return {"feasible": False, "reason": "항목 없음"}
    try:
        m = np.array([i["m"] for i in items], dtype=np.float64)
        s = np.array([i["s"] for i in items], dtype=np.float64)
        good = np.array([i["cls"] == "good" for i in items])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"맞추기 항목 형식 오류: {exc}") from exc
    bad = ~good
    big = 99.0
    nm, ns = len(m_grid), len(s_grid)
    n_good = np.zeros((nm, ns), dtype=np.int32)
    bad_viol = np.zeros((nm, ns), dtype=np.int32)
    safety_bad = np.full((nm, ns), big)
    safety_good = np.full((nm, ns), big)
    S = s_grid[None, :, None]
    for lo in range(0, nm, max(1, chunk)):
        M = m_grid[lo:lo + chunk][:, None, None]
        excl = np.maximum(M - m[None, None, :], S - s[None, None, :])   # > 0 이면 막힘
        keep = np.minimum(m[None, None, :] - M, s[None, None, :] - S)   # >= 0 이면 표시
        kept = keep >= 0
        n_good[lo:lo + chunk] = (kept & good).sum(-1)
        not_excluded = (excl < delta) if delta > 0 else (excl <= 0)
        bad_viol[lo:lo + chunk] = (not_excluded & bad).sum(-1)
        safety_bad[lo:lo + chunk] = np.where(bad, excl, big).min(-1)
        safety_good[lo:lo + chunk] = np.where(good & kept, keep, big).min(-1)
    safety = np.minimum(safety_bad, safety_good)
    feas = bad_viol == 0
    if not feas.any():
        return {"feasible": False, "delta": delta, "reason": "격자 안에 나쁜 클립을 모두 막는 점이 없음"}
    best_good = int(n_good[feas].max())
    opt = feas & (n_good == best_good)
    idx = np.argwhere(opt)
    saf = safety[opt]
    k = int(np.argmax(saf))
    i, j = (int(x) for x in idx[k])
    col = np.where(opt[:, j])[0]
    ic = int(round((col.min() + col.max()) / 2))
    if opt[ic, j]:  # 같은 여유의 끝점 대신 최적 칸의 가운데 (step1과 같은 2차 동점 처리)
        rowc = np.where(opt[ic, :])[0]
        jc = int(round((rowc.min() + rowc.max()) / 2))
        if opt[ic, jc]:
            i, j = ic, jc
    col = np.where(opt[:, j])[0]
    row = np.where(opt[i, :])[0]
    return {"feasible": True, "delta": delta, "max_correct_shown": best_good, "M": float(m_grid[i]), "S": float(s_grid[j]),
            "max_safety": float(saf[k]), "safety": float(safety[i, j]), "safety_vs_bad": float(safety_bad[i, j]),
            "safety_vs_kept": float(safety_good[i, j]),
            "optimal_M_range_at_S": [float(m_grid[col.min()]), float(m_grid[col.max()])],
            "optimal_S_range_at_M": [float(s_grid[row.min()]), float(s_grid[row.max()])],
            "n_optimal_grid_points": int(opt.sum()), "n_items": len(items),
            "n_good": int(good.sum()), "n_bad": int(bad.sum())}


def nearest(items: list[dict[str, Any]], M: float, S: float, k: int = 4) -> dict[str, Any]:
    """기준점에서 가장 가까운 나쁜 클립(막은 여유)과 남긴 정답(남긴 여유)."""
    bad = sorted(((max(M - i["m"], S - i["s"]), i) for i in items if i["cls"] != "good"), key=lambda x: x[0])
    kept = sorted(((min(i["m"] - M, i["s"] - S), i) for i in items if i["cls"] == "good" and i["m"] >= M and i["s"] >= S),
                  key=lambda x: x[0])

    def fmt(slack: float, i: dict[str, Any]) -> dict[str, Any]:
        return {"file": i["file"], "cls": i["cls"], "top": i["top"], "margin": round(i["m"], 3), "score": round(i["s"], 3),
                "slack": round(slack, 3)}

    return {"bad_closest": [fmt(a, b) for a, b in bad[:k]], "kept_good_closest": [fmt(a, b) for a, b in kept[:k]]}


# ---------------------------------------------------------------------------
# 규칙 적용 요약
# ---------------------------------------------------------------------------

def headline(bases: list[dict[str, Any]], rule: mt.Rule) -> dict[str, Any]:
    """한 규칙의 핵심 수치 (사람·묶음을 합친 값). 축소 목록이면 목록에서 뺀 명령의 클립은 넣지 않는다(그 명령은 말하지 않는다)."""
    if rule.ids is not None:
        keep = set(rule.ids)
        bases = [b for b in bases if b["role"] != mt.ROLE_COMMAND or b["truth"] in keep]
    ev = mt.evaluate_all(bases, rule)
    cmd = mt.command_table(ev)
    neg = mt.negatives_shown(ev)
    fa = mt.false_alarms(ev, rule)
    er = mt.emergency_recall(ev, rule)
    sh = mt.short_table(ev)
    return {"rule": {"min_margin": rule.min_margin, "min_score": rule.min_score, "max_distance": rule.max_distance,
                     "ids": len(rule.ids) if rule.ids is not None else None, "possible_min_score": rule.possible_min_score},
            "commands_shown": cmd["shown"]["text"], "commands_shown_k": cmd["shown"]["k"], "commands_n": cmd["n"],
            "shown_wrong": cmd["shown_wrong"]["text"], "shown_wrong_k": cmd["shown_wrong"]["k"],
            "shown_wrong_clips": cmd["shown_wrong_clips"],
            "negatives_shown": neg["total"]["text"], "negatives_shown_k": neg["total"]["k"],
            "negatives_shown_clips": [c for r in mt.NEGATIVE_ROLES for c in neg[r]["clips"]],
            "false_alarms": fa["total"]["text"], "false_alarms_k": fa["total"]["k"],
            "emergency_recall": er["all"]["reached"]["text"], "short_alarm": sh["reached_alarm"]["text"],
            "short_as_request": sh["shown_as_request"]["text"], "short_as_request_k": sh["shown_as_request"]["k"],
            "short_as_request_clips": sh["as_request_clips"], "via_enroll": cmd["via_enroll"]}


def rule_at(M: float, S: float, emergency: Iterable[str], ids: Iterable[str] | None = None,
            max_distance: float | None = None, require_agreement: bool = True) -> mt.Rule:
    """기준값으로 규칙을 만든다."""
    return mt.Rule(min_margin=float(M), min_score=float(S), emergency_ids=frozenset(emergency),
                   ids=tuple(ids) if ids is not None else None, max_distance=max_distance,
                   require_agreement=require_agreement)


def persons(bases: Iterable[dict[str, Any]]) -> list[str]:
    """기록에 있는 참가자 ID (정렬)."""
    return sorted({str(b.get("participant")) for b in bases})


def lopo_thresholds(bases: list[dict[str, Any]], emergency: Iterable[str], block_roles: Iterable[str] = mt.NEGATIVE_ROLES,
                    conditions: Iterable[str] | None = None, ids: Iterable[str] | None = None) -> dict[str, Any]:
    """한 사람으로 맞춘 기준을 다른 사람에게 그대로 적용한다 (모든 순서쌍)."""
    pids = persons(bases)
    rb = reranked(bases, ids)  # 맞추기 항목용 (축소 목록이면 그 목록 안의 1·2등)

    def of(rows: list[dict[str, Any]], pid: str) -> list[dict[str, Any]]:
        return [b for b in rows if str(b.get("participant")) == pid]

    out: dict[str, Any] = {}
    emergency = list(emergency)
    for fit_pid in pids:
        g = grid_fit(fit_items(of(rb, fit_pid), block_roles, conditions, ids, emergency))
        entry: dict[str, Any] = {"fit": g}
        if g.get("feasible"):
            rule = rule_at(g["M"], g["S"], emergency, ids)
            entry["on_fit_person"] = headline(of(bases, fit_pid), rule)
            for other in pids:
                if other == fit_pid:
                    continue
                entry[f"on_{other}"] = headline(of(bases, other), rule)
                entry[f"nearest_on_{other}"] = nearest(fit_items(of(rb, other), block_roles, conditions, ids, emergency),
                                                       g["M"], g["S"])
        out[f"fit_{fit_pid}"] = entry
    return out


def pooled_thresholds(bases: list[dict[str, Any]], emergency: Iterable[str], deltas: Iterable[float] = DEFAULT_DELTAS,
                      block_roles: Iterable[str] = mt.NEGATIVE_ROLES, conditions: Iterable[str] | None = None,
                      ids: Iterable[str] | None = None) -> dict[str, Any]:
    """사람을 합쳐 맞춘다. 여유 δ마다 결과를 낸다."""
    rb = reranked(bases, ids)
    emergency = list(emergency)
    items = fit_items(rb, block_roles, conditions, ids, emergency)
    out: dict[str, Any] = {}
    for d in deltas:
        g = grid_fit(items, float(d))
        entry: dict[str, Any] = {"fit": g}
        if g.get("feasible"):
            rule = rule_at(g["M"], g["S"], emergency, ids)
            entry["all"] = headline(bases, rule)
            entry.update({f"on_{p}": headline([b for b in bases if str(b.get("participant")) == p], rule)
                          for p in persons(bases)})
            entry["nearest"] = nearest(items, g["M"], g["S"])
        out[f"{float(d):.2f}"] = entry
    return out


def choose_proposal(pooled: dict[str, Any], max_loss: float = 0.10, delta: float | None = None) -> dict[str, Any]:
    """동결 초안에 넣을 기준 (미리 정한 고르는 규칙).

    delta를 주면 그 여유의 합친 맞추기. 아니면 합친 맞추기 중 정답 표시 수가 δ=0의 (1 - max_loss)배 이상인 가장 큰 δ.
    맞출 수 있는 δ가 없으면 임시값(0.5, -3.40)을 그대로 두고 그렇게 적는다.
    """
    feas = {k: v["fit"] for k, v in pooled.items() if v["fit"].get("feasible")}
    if not feas:
        return {"M": PROVISIONAL[0], "S": PROVISIONAL[1], "delta": None,
                "chosen_by": "합친 맞추기가 불가능해 임시값을 그대로 둠 (사람이 다시 정해야 함)"}
    if delta is not None:
        key = f"{float(delta):.2f}"
        if key not in feas:
            raise ValueError(f"여유 {key}의 합친 맞추기가 없거나 불가능합니다")
        g = feas[key]
        return {"M": g["M"], "S": g["S"], "delta": float(delta), "chosen_by": f"사람이 정한 여유 δ={key}"}
    base = feas.get("0.00")
    floor = (1.0 - max_loss) * base["max_correct_shown"] if base else 0.0
    ok = [(float(k), g) for k, g in feas.items() if g["max_correct_shown"] >= floor]
    d, g = max(ok, key=lambda pair: pair[0])
    return {"M": g["M"], "S": g["S"], "delta": d,
            "chosen_by": f"합친 맞추기 중 정답 표시가 δ=0의 {100 * (1 - max_loss):.0f}% 이상인 가장 큰 여유 δ={d:.2f}"}


# ---------------------------------------------------------------------------
# 축소 목록
# ---------------------------------------------------------------------------

def _rank_for(b: dict[str, Any], ids: list[str], exact: bool) -> tuple[str, str | None, float, float] | None:
    """목록 ids 안에서 클립의 (1등, 2등, 1등 점수, 차이). exact면 후보별 점수, 아니면 서비스 1·2등만으로 근사."""
    if b.get("top") is None:
        return None
    if exact:
        r = mt.rank_scores(b["text_scores"], ids) if b.get("text_scores") else None
        return None if r is None else (r["id"], r["runner_up_id"], r["score"], r["margin"])
    if b["top"] not in ids:
        return None  # 1등이 목록에서 빠지면 새 1등을 알 수 없다(근사에서는 뺀다)
    runner = b["runner_up"] if b["runner_up"] in ids else None
    return b["top"], runner, float(b["score"]), float(b["margin"])


def involvement(bases: list[dict[str, Any]], ids: list[str], ref: tuple[float, float], exact: bool,
                block_roles: Iterable[str] = mt.NEGATIVE_ROLES,
                emergency: Iterable[str] = ()) -> tuple[dict[str, float], dict[str, list[float]]]:
    """목록 안 명령별 혼동 관여 점수와 정답 1등일 때의 차이(margin) 목록.

    명령 클립(정답이 목록 안): 1등 오답이면 정답·오답 명령 각 +1, 1등 정답이지만 기준 미달이면 정답·2등 각 +0.5.
    막아야 할 입력과 1등이 긴급이 아닌 짧은 긴급 표현: 기준(ref)을 넘어 명령으로 표시되면 그 1등 명령 +1.
    """
    inv = {i: 0.0 for i in ids}
    margins: dict[str, list[float]] = {i: [] for i in ids}
    block_roles = tuple(block_roles)
    em = set(emergency)
    M, S = ref
    for b in bases:
        r = _rank_for(b, ids, exact)
        if r is None:
            continue
        top, runner, score, margin = r
        if b["role"] == mt.ROLE_COMMAND:
            if b["truth"] not in inv:
                continue
            if top != b["truth"]:
                inv[b["truth"]] += 1.0
                inv[top] += 1.0
            else:
                margins[top].append(margin)
                if not (margin >= M and score >= S):
                    inv[top] += 0.5
                    if runner in inv:
                        inv[runner] += 0.5
        elif (b["role"] in block_roles or (b["role"] == mt.ROLE_SHORT and top not in em)) and margin >= M and score >= S:
            inv[top] += 1.0
    return inv, margins


def select_reduced_list(bases: list[dict[str, Any]], all_ids: list[str], target: int, protected: Iterable[str],
                        ref: tuple[float, float], exact: bool | None = None,
                        block_roles: Iterable[str] = mt.NEGATIVE_ROLES, emergency: Iterable[str] = ()) -> dict[str, Any]:
    """혼동에 가장 많이 얽힌 명령부터 하나씩 빼서 target개 목록을 만든다 (보호 명령은 빼지 않는다).

    동점이면 정답 1등일 때 차이의 중앙값이 작은(약한) 명령, 그다음 ID가 큰 명령을 뺀다. exact가 None이면 모든 1등 클립에
    후보별 점수가 있을 때만 정확 모드(매번 다시 순위)다.
    """
    protected = [p for p in protected if p in all_ids]
    if target < len(protected) or target > len(all_ids) or target < 2:
        raise ValueError(f"축소 목록 크기 {target}이(가) 보호 명령 {len(protected)}개~전체 {len(all_ids)}개 범위 밖입니다")
    if exact is None:
        exact = all(b.get("text_scores") for b in bases if b.get("top") is not None)
    ids = list(all_ids)
    trace = []
    while len(ids) > target:
        inv, margins = involvement(bases, ids, ref, exact, block_roles, emergency)

        def weakness(cid: str) -> tuple[float, float, str]:
            med = float(np.median(margins[cid])) if margins[cid] else -math.inf
            return (inv[cid], -med, cid)

        cands = [i for i in ids if i not in protected]
        drop = max(cands, key=weakness)
        trace.append({"dropped": drop, "involvement": round(inv[drop], 3),
                      "median_correct_margin": (round(float(np.median(margins[drop])), 4) if margins[drop] else None),
                      "list_size_after": len(ids) - 1,
                      "involvement_all": {k: round(v, 3) for k, v in sorted(inv.items(), key=lambda kv: -kv[1]) if v > 0}})
        ids = [i for i in ids if i != drop]
    return {"ids": ids, "size": len(ids), "protected_ids": sorted(protected), "exact": bool(exact), "ref": list(ref),
            "trace": trace,
            "method": ("목록 안 명령별 혼동 관여 점수(1등 오답 시 정답·오답 각 +1, 정답 1등이지만 기준 미달이면 정답·2등 각 +0.5, "
                       "막아야 할 입력이나 1등이 긴급이 아닌 짧은 긴급 표현이 기준을 넘으면 그 1등 +1)가 가장 큰 비보호 명령을 "
                       "하나씩 뺌. 동점은 정답 1등 차이의 중앙값이 "
                       "작은 명령, 그다음 ID가 큰 명령. " + ("매 단계 후보별 점수로 다시 순위를 매김." if exact else
                                                         "후보별 점수가 없어 서비스 1·2등만으로 근사(뺀 명령이 1등인 클립은 제외)."))}


# ---------------------------------------------------------------------------
# 등록 거리 D
# ---------------------------------------------------------------------------

def enroll_only_candidates(bases: list[dict[str, Any]], rule: mt.Rule,
                           block_roles: Iterable[str] = mt.NEGATIVE_ROLES) -> list[dict[str, Any]]:
    """글자 규칙으로는 표시되지 않지만 등록 거리 때문에 표시되거나 긴급 가능성이 될 수 있는 클립
    {file, participant, dist, cls(good/bad), effect(shown/alarm)}.

    shown: 등록 1등이 고른 명령(정답 명령이면 good, 오답 명령·목록 밖 입력·1등이 일반인 짧은 긴급 표현이면 bad).
    alarm: 고르지 않았는데 등록 1등이 긴급이라 긴급 가능성이 되는 클립(긴급이 아닌 명령·목록 밖 입력이면 bad,
           긴급 명령·짧은 긴급 표현은 바라는 결과라 넣지 않는다).
    """
    block_roles = tuple(block_roles)
    inf_rule = replace(rule, max_distance=math.inf)
    out = []
    for e in mt.evaluate_all(bases, inf_rule):
        if e.get("via") == "enroll":
            effect = "shown"
            if e["role"] == mt.ROLE_COMMAND:
                cls = "good" if e["chosen_id"] == e["truth"] else "bad"
            elif e["role"] in block_roles:
                cls = "bad"
            elif e["role"] == mt.ROLE_SHORT and e["chosen_id"] not in rule.emergency_ids:
                cls = "bad"
            else:
                continue
        elif e.get("enroll_alarm"):
            effect = "alarm"
            if e["role"] == mt.ROLE_COMMAND and e["truth"] not in rule.emergency_ids:
                cls = "bad"
            elif e["role"] in block_roles:
                cls = "bad"
            else:
                continue
        else:
            continue
        out.append({"file": e["file"], "participant": e.get("participant"), "dist": float(e["tmpl_dist"]), "cls": cls,
                    "chosen_id": e["chosen_id"] or e.get("tmpl_top"), "role": e["role"], "effect": effect})
    return out


def negatives_with_distances(bases: Iterable[dict[str, Any]], block_roles: Iterable[str] = mt.NEGATIVE_ROLES) -> dict[str, int]:
    """목록 밖 입력(block_roles) 수와 그중 등록 거리가 있는 클립 수 {n_negatives, n_negatives_with_dist}."""
    block_roles = tuple(block_roles)
    neg = [b for b in bases if b["role"] in block_roles]
    return {"n_negatives": len(neg),
            "n_negatives_with_dist": sum(1 for b in neg if any(v is not None for v in (b.get("tmpl_dists") or {}).values()))}


def fit_distance(bases: list[dict[str, Any]], rule: mt.Rule, block_roles: Iterable[str] = mt.NEGATIVE_ROLES) -> dict[str, Any]:
    """나쁜 클립을 하나도 표시하거나 긴급 가능성으로 보내지 않는 가장 큰 쪽의 D (막는 값과 그 아래 가장 큰 정답 거리의 가운데).

    나쁜 클립이 없으면 막는 값을 모르므로 D = 가장 큰 정답 거리로 두고 unbounded=True로 적는다(동결에는 쓰지 않는다,
    choose_distance 참조). 정답도 없으면 None.
    """
    cands = enroll_only_candidates(bases, rule, block_roles)
    bad = sorted(c["dist"] for c in cands if c["cls"] == "bad")
    block = bad[0] if bad else math.inf
    need = sorted(c["dist"] for c in cands if c["cls"] == "good" and c["dist"] < block)
    if math.isfinite(block):
        D = (block + need[-1]) / 2 if need else block - 1e-6
    else:
        D = need[-1] if need else None
    return {"D": D, "blocking_dist": block if math.isfinite(block) else None,
            "largest_needed_dist": need[-1] if need else None, "unbounded": not math.isfinite(block),
            "n_enroll_only_good": sum(c["cls"] == "good" for c in cands), "n_enroll_only_bad": len(bad),
            "good_gained": len(need), "closest_bad": sorted((c for c in cands if c["cls"] == "bad"),
                                                            key=lambda c: c["dist"])[:4],
            **negatives_with_distances(bases, block_roles)}


def lopo_distance(bases: list[dict[str, Any]], rule: mt.Rule, block_roles: Iterable[str] = mt.NEGATIVE_ROLES
                  ) -> dict[str, Any]:
    """D를 한 사람으로 맞춰 다른 사람에게 적용, 그리고 합친 맞추기. 등록 판정이 가능한 클립(거리 있음)만 쓴다."""
    out: dict[str, Any] = {}
    pids = persons(bases)
    for fit_pid in pids:
        mine = [b for b in bases if str(b.get("participant")) == fit_pid]
        f = fit_distance(mine, rule, block_roles)
        entry: dict[str, Any] = {"fit": f}
        if f["D"] is not None:
            r = replace(rule, max_distance=f["D"])
            entry["on_fit_person"] = headline([b for b in mine if b.get("tmpl_dists")], r)
            for other in pids:
                if other != fit_pid:
                    theirs = [b for b in bases if str(b.get("participant")) == other and b.get("tmpl_dists")]
                    entry[f"on_{other}"] = headline(theirs, r)
        out[f"fit_{fit_pid}"] = entry
    f = fit_distance(bases, rule, block_roles)
    out["pooled"] = {"fit": f}
    if f["D"] is not None:
        r = replace(rule, max_distance=f["D"])
        out["pooled"].update({f"on_{p}": headline([b for b in bases if str(b.get("participant")) == p and b.get("tmpl_dists")], r)
                              for p in pids})
    return out


def choose_distance(lopo: dict[str, Any], how: str = "min", min_negatives: int = MIN_NEGATIVES_FOR_D) -> dict[str, Any]:
    """동결할 D: 'min' = 사람별·합친 맞추기 D 중 가장 작은 값(보수적), 'pooled' = 합친 맞추기 값.

    D를 정하지 않는 경우(D None, 사유를 chosen_by에): 사람별·합친 맞추기 중 하나라도 막는 거리를 모를 때(unbounded),
    사람별 맞추기 중 등록 거리가 있는 목록 밖 클립이 min_negatives개 미만일 때.
    """
    fits = {k: v["fit"] for k, v in lopo.items()}
    unbounded = sorted(k for k, f in fits.items() if f.get("D") is not None and f.get("unbounded"))
    if unbounded:
        return {"D": None, "unbounded": True,
                "chosen_by": f"막는 거리를 모르는 맞추기가 있어(나쁜 클립 없음: {', '.join(unbounded)}) D를 정하지 않음"}
    few = sorted(f"{k}({f.get('n_negatives_with_dist', 0)})" for k, f in fits.items()
                 if k != "pooled" and int(f.get("n_negatives_with_dist") or 0) < min_negatives)
    if few:
        return {"D": None, "unbounded": False,
                "chosen_by": f"등록 거리가 있는 목록 밖 클립이 {min_negatives}개 미만인 사람이 있어 D를 정하지 않음: {', '.join(few)}"}
    values = {k: f["D"] for k, f in fits.items() if f.get("D") is not None}
    if not values:
        return {"D": None, "chosen_by": "등록 거리로 표시할 수 있는 정답이 없어 D를 정하지 않음"}
    if how == "pooled":
        if "pooled" not in values:
            return {"D": None, "chosen_by": "합친 맞추기 D가 없음"}
        return {"D": values["pooled"], "chosen_by": "합친 맞추기 D"}
    key = min(values, key=lambda k: values[k])
    return {"D": values[key], "chosen_by": f"사람별·합친 맞추기 D 중 가장 작은 값({key})"}


# ---------------------------------------------------------------------------
# 긴급 가능성 점수 하한
# ---------------------------------------------------------------------------

def fit_possible_floor(bases: list[dict[str, Any]], rule: mt.Rule, clearance: float = 0.2,
                       conditions: Iterable[str] | None = None) -> dict[str, Any]:
    """긴급 명령 클립을 하나도 긴급(빨간 카드·가능성)에서 빼지 않는 가장 높은 긴급 가능성 점수 하한에서 clearance를 뺀 값.

    글자 규칙으로 고르지 못해 '1·2등에 긴급' 길로만 긴급에 닿는 긴급 명령 클립의 긴급 후보 점수 중 최솟값이 필요한 하한이다.
    그런 클립이 없으면 하한을 정할 근거가 없어 floor=None이다. conditions를 주면 그 조건의 긴급 명령 클립만 본다.
    """
    conds = set(conditions) if conditions is not None else None
    plain = replace(rule, possible_min_score=None)
    ev = mt.evaluate_all(bases, plain)
    em = [e for e in ev if e["role"] == mt.ROLE_COMMAND and e["truth"] in rule.emergency_ids
          and (conds is None or e["condition"] in conds)]
    rely = [(ps, e["file"]) for e in em if e.get("chosen_id") is None and e["outcome"] == mt.OUT_POSSIBLE
            and (ps := mt.possible_score(e["top"], e["runner_up"], e["margin"], e["score"], plain)) is not None]
    needed = min(rely)[0] if rely else None
    floor = (needed - clearance) if needed is not None else None
    out: dict[str, Any] = {"floor": floor, "needed": needed, "needed_clip": min(rely)[1] if rely else None,
                           "clearance": clearance, "n_emergency_clips": len(em), "n_relying_on_possible": len(rely),
                           "without_floor": headline(bases, plain)}
    if floor is not None:
        out["with_floor"] = headline(bases, replace(rule, possible_min_score=floor))
    return out


def lopo_possible_floor(bases: list[dict[str, Any]], rule: mt.Rule, clearance: float = 0.2,
                        conditions: Iterable[str] | None = None) -> dict[str, Any]:
    """긴급 가능성 점수 하한을 한 사람으로 정해 다른 사람에게 적용, 그리고 합친 값."""
    out: dict[str, Any] = {}
    pids = persons(bases)
    for fit_pid in pids:
        mine = [b for b in bases if str(b.get("participant")) == fit_pid]
        f = fit_possible_floor(mine, rule, clearance, conditions)
        entry: dict[str, Any] = {"fit": {k: v for k, v in f.items() if k not in ("without_floor", "with_floor")}}
        if f["floor"] is not None:
            r = replace(rule, possible_min_score=f["floor"])
            entry.update({f"on_{p}": headline([b for b in bases if str(b.get("participant")) == p], r)
                          for p in pids if p != fit_pid})
        out[f"fit_{fit_pid}"] = entry
    out["pooled"] = fit_possible_floor(bases, rule, clearance, conditions)
    return out


# ---------------------------------------------------------------------------
# 동결 파일
# ---------------------------------------------------------------------------

_HEX64 = set("0123456789abcdef")


def _is_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX64


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def validate_freeze(d: Any, allow_synthetic: bool = False) -> list[str]:
    """동결 파일(초안 포함) 형식 검사. 문제 목록(없으면 빈 목록).

    합성(가짜) 자료로 만든 초안은 allow_synthetic(파이프라인 시험)이 아니면 문제로 본다.
    """
    problems: list[str] = []
    if not isinstance(d, dict):
        return ["동결 파일이 JSON 객체가 아닙니다"]
    data_files = ((d.get("data") or {}).get("rows_files") or []) + ((d.get("data") or {}).get("enroll_files") or [])
    if not allow_synthetic and (d.get("synthetic") or any(isinstance(f, dict) and f.get("synthetic") for f in data_files)):
        problems.append("합성(가짜) 자료로 만든 초안은 동결할 수 없습니다")
    if d.get("schema") != FREEZE_SCHEMA:
        problems.append(f"schema가 {FREEZE_SCHEMA}가 아닙니다: {d.get('schema')!r}")
    if d.get("status") not in ("draft", "frozen"):
        problems.append(f"status는 draft 또는 frozen이어야 합니다: {d.get('status')!r}")
    if d.get("phrase_set") != "ward":
        problems.append("phrase_set이 ward가 아닙니다")
    cmd = d.get("commands") or {}
    ids = cmd.get("ids") if isinstance(cmd.get("ids"), list) else []
    emergency = cmd.get("emergency_ids") if isinstance(cmd.get("emergency_ids"), list) else []
    if len(ids) < 2 or len(set(ids)) != len(ids):
        problems.append("commands.ids가 2개 이상의 서로 다른 ID 목록이 아닙니다")
    if not emergency or not set(emergency) <= set(ids):
        problems.append("commands.emergency_ids가 비었거나 명령 목록 밖의 ID가 있습니다")
    if not _is_sha(cmd.get("sha256")):
        problems.append("commands.sha256이 SHA-256이 아닙니다")
    full = d.get("full_list") or {}
    if not (_finite(full.get("min_margin")) and float(full["min_margin"]) >= 0):
        problems.append("full_list.min_margin은 0 이상의 유한한 수여야 합니다")
    if not (_finite(full.get("min_score")) and float(full["min_score"]) <= 0):
        problems.append("full_list.min_score는 0 이하의 유한한 수여야 합니다")
    red = d.get("reduced_list")
    if red is not None:
        rids = red.get("ids") if isinstance(red, dict) and isinstance(red.get("ids"), list) else None
        if rids is None or len(rids) < 2 or len(set(rids)) != len(rids) or not set(rids) <= set(ids):
            problems.append("reduced_list.ids가 명령 목록 안의 서로 다른 ID 2개 이상이 아닙니다")
        elif not set(emergency) <= set(rids):
            problems.append("reduced_list에 긴급 명령이 모두 들어 있지 않습니다")
        elif red.get("size") != len(rids):
            problems.append("reduced_list.size가 ids 개수와 다릅니다")
        for key, check in (("min_margin", lambda v: v >= 0), ("min_score", lambda v: v <= 0)):
            v = red.get(key) if isinstance(red, dict) else None
            if v is not None and not (_finite(v) and check(float(v))):
                problems.append(f"reduced_list.{key} 값이 잘못됐습니다: {v!r}")
    enr = d.get("enrollment")
    if enr is not None:
        D = enr.get("max_distance") if isinstance(enr, dict) else "?"
        if D is not None and not (_finite(D) and 0 < float(D) < 2):
            problems.append(f"enrollment.max_distance는 0~2 사이 수 또는 null이어야 합니다: {D!r}")
        elif D is not None:
            if enr.get("unbounded"):
                problems.append("enrollment.max_distance가 막는 거리 없이(unbounded) 정한 값입니다")
            if enr.get("template") not in common.ENROLL_TEMPLATES:
                problems.append(f"enrollment.template은 {common.ENROLL_TEMPLATES} 중 하나여야 합니다: {enr.get('template')!r}")
            if enr.get("feature") != common.FEATURE_NAME or enr.get("distance") != common.DISTANCE_NAME:
                problems.append("enrollment.feature·distance 이름이 ward_enroll.py와 다릅니다")
    rule = d.get("decision_rule") or {}
    if rule.get("possible_ranks") not in (1, 2):
        problems.append("decision_rule.possible_ranks는 1 또는 2여야 합니다")
    floor = rule.get("possible_min_score")
    if floor is not None and not (_finite(floor) and float(floor) <= 0):
        problems.append(f"decision_rule.possible_min_score는 0 이하의 수 또는 null이어야 합니다: {floor!r}")
    elif floor is not None and rule.get("floor_implemented_in_ui") is not True:
        problems.append("decision_rule.possible_min_score가 있지만 화면(wardMachine.js)에 하한이 없습니다"
                        "(floor_implemented_in_ui가 true가 아님). 화면에 먼저 넣거나 null로 두세요")
    for sf in d.get("sealed_folders") or []:
        if not isinstance(sf, dict) or not _is_sha(sf.get("listing_sha256")) or not isinstance(sf.get("files"), int):
            problems.append(f"sealed_folders 항목에 listing_sha256·files가 없습니다: {sf!r}"[:200])
    data = d.get("data") or {}
    rows_files = data.get("rows_files") if isinstance(data.get("rows_files"), list) else []
    if not rows_files:
        problems.append("data.rows_files가 비었습니다")
    for rf in rows_files:
        if not isinstance(rf, dict) or not _is_sha(rf.get("sha256")):
            problems.append(f"data.rows_files 항목에 sha256이 없습니다: {rf!r}"[:200])
        elif common.sealed_reasons([rf.get("path"), rf.get("clips_dir")], [{"participant": p, "session": s}
                                    for p in rf.get("participants") or [] for s in rf.get("sessions") or []]):
            problems.append(f"기준 맞추기 자료에 봉인 자료가 있습니다: {rf.get('path')}")
    code = (d.get("code") or {}).get("files") or {}
    if not code or not all(_is_sha(v) for v in code.values()):
        problems.append("code.files에 SHA-256이 없는 파일이 있습니다")
    sealed = d.get("sealed_check") or {}
    if sealed.get("sealed_data_read") is not False:
        problems.append("sealed_check.sealed_data_read가 false가 아닙니다")
    return problems


def rule_from_freeze(freeze: dict[str, Any], which: str = "full", with_enrollment: bool = False,
                     apply_floor: bool = False) -> mt.Rule:
    """동결 파일 → 판정 규칙. which = 'full'(전체 목록, 서비스 phrase) 또는 'reduced'(축소 목록, 후보별 점수 필요).

    긴급 가능성 점수 하한은 apply_floor일 때만 넣는다(기본은 화면이 실제로 쓰는 규칙 = 하한 없음).

    Raises:
        ValueError: 동결 파일 형식이 틀렸거나 요청한 목록·D가 없을 때
    """
    problems = validate_freeze(freeze, allow_synthetic=True)  # 합성 자료 여부는 부르는 쪽(CLI)이 막는다
    if problems:
        raise ValueError("동결 파일 문제: " + "; ".join(problems))
    emergency = freeze["commands"]["emergency_ids"]
    enr = freeze.get("enrollment") or {}
    D = enr.get("max_distance") if with_enrollment else None
    if with_enrollment and D is None:
        raise ValueError("동결 파일에 등록 거리 D가 없습니다")
    agree = bool(enr.get("require_agreement", True))
    ranks = int((freeze.get("decision_rule") or {}).get("possible_ranks", 2))
    floor = (freeze.get("decision_rule") or {}).get("possible_min_score") if apply_floor else None
    if which == "full":
        block = freeze["full_list"]
        ids = None
    elif which == "reduced":
        block = freeze.get("reduced_list") or {}
        if block.get("min_margin") is None or block.get("min_score") is None:
            raise ValueError("동결 파일의 축소 목록에 기준값이 없습니다")
        ids = block["ids"]
    else:
        raise ValueError(f"알 수 없는 목록: {which}")
    return mt.Rule(min_margin=float(block["min_margin"]), min_score=float(block["min_score"]),
                   emergency_ids=frozenset(emergency), ids=tuple(ids) if ids is not None else None,
                   max_distance=float(D) if D is not None else None, require_agreement=agree, possible_ranks=ranks,
                   possible_min_score=float(floor) if floor is not None else None)
