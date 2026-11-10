"""병실 명령 판정 규칙(기획 9.3)과 사람·묶음별 지표(기획 10.2). 표준 라이브러리만 쓰는 순수 함수다.

판정표 (글자 규칙 = 1·2등 차이 ≥ M 그리고 1등 점수 ≥ S, 등록 거리 규칙 = 내 등록 영상과의 거리 ≤ D):
  글자 규칙 통과 + 긴급 명령        → emergency          (빨간 카드 즉시)
  글자 규칙 통과 + 일반 명령        → request            (확인 카운트다운 뒤 전송)
  글자 규칙 실패, 등록 거리 통과, 글자 1등 = 등록 1등 + 긴급 → emergency_possible (긴급 가능성 카드, chosen_id 있음)
  글자 규칙 실패, 등록 거리 통과, 글자 1등 = 등록 1등 + 일반, 글자 2등이 긴급(하한이 있으면 그 점수 이상)
                                    → emergency_possible (chosen_id 있음. 등록 없이도 나던 알림을 등록이 없애지 않는다)
  글자 규칙 실패, 등록 거리 통과, 글자 1등 = 등록 1등 + 일반 (그 밖) → request
  글자 규칙 실패, 등록 거리 통과, 글자 1등 ≠ 등록 1등(require_agreement) → 고르지 않음 (conflict)
  고르지 않음 + (글자 1·2등에 긴급 또는 거리 ≤ D인 등록 1등이 긴급) → emergency_possible (chosen_id 없음)
  고르지 않음                       → not_chosen         (스캐닝 선택 + 확인 필요)
  지정 문장 결과 없음(여러 창·일부 검출 실패·정렬 불가·병실 묶음이 아닌 결과·목록 밖 ID) → no_result (분류 스캐닝)
  모든 창이 [발화 없음]             → no_speech ; 모든 창이 표시(발화 없음·검출 실패)이고 검출 실패가 있음 → lip_fail
  서버 처리 실패 → error
'글자 1등 ≠ 등록 1등' 행은 글자 규칙을 통과하지 못해 등록 거리로 고를 때에만 본다(글자 규칙 통과는 그대로 둔다).
등록 1등이 긴급이면 고르지 않더라도 긴급 가능성으로 올린다(기획 11.3: 놓치는 쪽보다 확인하는 쪽).
'표시(shown)'는 화면에 명령 하나가 뜬 것(chosen_id가 있음)이다. 서버 accepted는 쓰지 않고 저장된 margin·score로 다시 판정한다.
화면(frontend/src/ward/wardResult.js)과 같게 phrase.set이 ward가 아니거나 1등 ID가 명령 목록 밖이면 결과 없음, 2등 ID가 목록
밖이면 2등 없음으로 본다. 긴급 가능성 점수 하한(possible_min_score)은 화면이 서버의 phrase.rule.possible_min_score로 쓴다.
기본은 None(기획 9.3 그대로: 점수와 관계없이 1·2등에 긴급이면 긴급 가능성)이다.

등록 거리의 출처는 둘이고 섞지 않는다(use_enroll_source로 하나를 고른다):
  service : 서버 등록(POST /api/record의 enroll_profile, 결과 results[0].enroll의 match 결과). 앱이 실제로 쓰는 경로.
            tmpl_top/tmpl_dist = 서버 top_id/distance. store(등록 영상) 행은 등록 템플릿이다(svc_is_template).
  offline : ward_enroll.py apply 출력(--enroll 파일)의 tmpl_dists. 보조 결과.
판정 기록의 enroll_from이 그 클립의 등록 판정에 쓴 출처('service'·'offline'·None)다. 서버의 enroll.within·accepted와
phrase.rule(서버가 쓴 기준값)은 판정에 쓰지 않고, 다시 판정한 값과 비교만 한다(enroll_check·rule_check).

입력 역할(role):
  command(병실 명령 W01~W20), decoy(목록 밖 함정 문장), short(목록 밖 짧은 긴급 표현 E01~E03), nonspeech(말하지 않는 동작),
  sentence_speech(문장 촬영 폴더의 지정·자유 문장 = 병실 목록 밖 말), sentence_nonspeech(문장 촬영 폴더의 말하지 않기)
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Iterable

from ward_tools import common

ROLE_COMMAND = "command"
ROLE_DECOY = "decoy"
ROLE_SHORT = "short"
ROLE_NONSPEECH = "nonspeech"
ROLE_SENT_SPEECH = "sentence_speech"
ROLE_SENT_NONSPEECH = "sentence_nonspeech"
ROLES = (ROLE_COMMAND, ROLE_DECOY, ROLE_SHORT, ROLE_NONSPEECH, ROLE_SENT_SPEECH, ROLE_SENT_NONSPEECH)
# 명령으로 실행되면 안 되는 입력 (기획 10.2 '함정·비발화 입력이 명령으로 실행된 수'). 짧은 긴급 표현은 따로 본다.
NEGATIVE_ROLES = (ROLE_DECOY, ROLE_NONSPEECH, ROLE_SENT_SPEECH, ROLE_SENT_NONSPEECH)
SPEECH_ROLES = (ROLE_COMMAND, ROLE_DECOY, ROLE_SHORT, ROLE_SENT_SPEECH)

OUT_EMERGENCY = "emergency"
OUT_REQUEST = "request"
OUT_POSSIBLE = "emergency_possible"
OUT_NOT_CHOSEN = "not_chosen"
OUT_NO_RESULT = "no_result"
OUT_NO_SPEECH = "no_speech"
OUT_LIP_FAIL = "lip_fail"
OUT_ERROR = "error"
OUT_UNSCORABLE = "unscorable"  # 축소 목록·등록 판정에 필요한 후보별 점수가 없는 클립
OUTCOMES = (OUT_EMERGENCY, OUT_REQUEST, OUT_POSSIBLE, OUT_NOT_CHOSEN, OUT_NO_RESULT, OUT_NO_SPEECH, OUT_LIP_FAIL,
            OUT_ERROR, OUT_UNSCORABLE)
ALARM_OUTCOMES = (OUT_EMERGENCY, OUT_POSSIBLE)  # 간호사실에 긴급(확정 또는 가능성)으로 바로 가는 결과

NEAR_END_MS = 500.0     # 말 끝이 녹화 끝 이 시간 안이면 '녹화가 짧았을 수 있음' (기획 10.2 녹화 길이)
NEAR_END_TOL_MS = 50.0  # 프레임 시각 반올림 여유

ENROLL_SOURCES = ("service", "offline", "none")
ENROLL_DIST_TOL = 1e-9  # 서버 top_id·distance와 distances 최솟값이 같다고 볼 차이


@dataclass(frozen=True)
class Rule:
    """판정 기준 한 벌.

    min_margin/min_score: 글자 규칙 M·S. emergency_ids: 긴급 명령 ID. ids: 후보 목록(None = 서비스 전체 목록, 서비스 phrase 사용;
    목록을 주면 후보별 점수 text_scores로 그 목록 안에서 다시 순위를 매긴다 = 축소 목록). max_distance: 등록 거리 D(None = 등록 안 씀).
    require_agreement: 글자 규칙을 통과하지 못해 등록 거리로 고를 때 글자 1등과 등록 1등이 다르면 고르지 않는다(기획 9.3).
    possible_ranks: 긴급 가능성을 볼 글자 순위 수.
    possible_min_score: 고르지 못했을 때 긴급 가능성으로 볼 긴급 후보의 토큰당 점수 하한(1등은 score, 2등은 score - margin).
        None(기본)이면 기획 9.3·화면 그대로 점수와 관계없이 1·2등에 긴급이 있으면 긴급 가능성이다.
    """

    min_margin: float
    min_score: float
    emergency_ids: frozenset[str] = field(default_factory=frozenset)
    ids: tuple[str, ...] | None = None
    max_distance: float | None = None
    require_agreement: bool = True
    possible_ranks: int = 2
    possible_min_score: float | None = None

    def describe(self) -> dict[str, Any]:
        """JSON용 요약."""
        return {"min_margin": self.min_margin, "min_score": self.min_score, "emergency_ids": sorted(self.emergency_ids),
                "ids": list(self.ids) if self.ids is not None else None, "max_distance": self.max_distance,
                "require_agreement": self.require_agreement, "possible_ranks": self.possible_ranks,
                "possible_min_score": self.possible_min_score}


def possible_score(top: str | None, runner_up: str | None, margin: float | None, score: float | None,
                   rule: Rule) -> float | None:
    """고르지 못한 클립의 글자 1·2등(possible_ranks개) 가운데 긴급 후보의 가장 높은 점수. 긴급 가능성이 아니면 None.

    1등 점수는 score, 2등 점수는 score - margin이다. rule.possible_min_score가 있으면 그 이상인 긴급 후보만 센다.
    """
    try:
        cands = [(top, float(score)), (runner_up, float(score) - float(margin))][:max(1, rule.possible_ranks)]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"긴급 가능성 입력 오류: {exc}") from exc
    found = [s for cid, s in cands if cid and cid in rule.emergency_ids
             and (rule.possible_min_score is None or s >= rule.possible_min_score)]
    return max(found) if found else None


def decide(top: str | None, runner_up: str | None, margin: float | None, score: float | None, rule: Rule,
           tmpl_top: str | None = None, tmpl_dist: float | None = None) -> dict[str, Any]:
    """글자 순위(와 등록 1등)로 판정표를 적용한다. top이 None이면 no_result.

    글자 규칙을 통과하면 등록 거리와 관계없이 글자 1등을 고른다. 통과하지 못했고 등록 1등 거리가 D 이하이면
    (require_agreement일 때 글자 1등과 같을 때만) 등록 1등을 고른다. 등록으로 고른 명령이 일반이어도 글자 1·2등에
    긴급(하한 이상)이 있으면 긴급 가능성이다(등록은 알림을 없애지 않는다). 고르지 못했을 때 글자 1·2등에 긴급이 있거나
    거리 D 이하인 등록 1등이 긴급이면 긴급 가능성이다.

    Returns:
        {outcome, chosen_id, via('text'|'enroll'|None), text_pass, conflict, enroll_alarm}
    """
    try:
        if top is None or margin is None or score is None:
            return {"outcome": OUT_NO_RESULT, "chosen_id": None, "via": None, "text_pass": False, "conflict": False,
                    "enroll_alarm": False}
        text_pass = float(margin) >= rule.min_margin and float(score) >= rule.min_score
        chosen: str | None = None
        via: str | None = None
        conflict = False
        within = (rule.max_distance is not None and tmpl_top is not None and tmpl_dist is not None
                  and float(tmpl_dist) <= rule.max_distance)
        if text_pass:
            chosen, via = top, "text"
        elif within:
            if rule.require_agreement and tmpl_top != top:
                conflict = True
            else:
                chosen, via = tmpl_top, "enroll"
        enroll_alarm = False
        if chosen is not None:
            if chosen in rule.emergency_ids:
                outcome = OUT_EMERGENCY if via == "text" else OUT_POSSIBLE
            elif via == "enroll" and possible_score(top, runner_up, margin, score, rule) is not None:
                # 등록으로 고른 일반 명령이라도 글자 2등(하한 이상)이 긴급이면 등록이 없을 때처럼 긴급 가능성으로 둔다.
                # 등록은 알림을 더하기만 하고 없애지 않는다(고른 명령은 chosen_id로 남아 후보 맨 앞에 온다).
                outcome = OUT_POSSIBLE
            else:
                outcome = OUT_REQUEST
        else:
            text_alarm = possible_score(top, runner_up, margin, score, rule) is not None
            enroll_alarm = bool(within and tmpl_top in rule.emergency_ids)
            outcome = OUT_POSSIBLE if (text_alarm or enroll_alarm) else OUT_NOT_CHOSEN
            enroll_alarm = enroll_alarm and not text_alarm  # 등록 1등 때문에만 긴급 가능성이 된 경우
    except (TypeError, ValueError) as exc:
        raise ValueError(f"판정 입력 오류: {exc}") from exc
    return {"outcome": outcome, "chosen_id": chosen, "via": via, "text_pass": text_pass, "conflict": conflict,
            "enroll_alarm": enroll_alarm}


def rank_scores(scores: dict[str, float | None], ids: Iterable[str], higher_is_better: bool = True
                ) -> dict[str, Any] | None:
    """후보별 점수를 ids 안에서 순위 매긴다 (서비스 rank_phrases와 같은 규칙: 하나라도 None이면 None, 같으면 목록 앞 우선).

    Returns:
        {id, score, margin(1등-2등의 차이, 거리면 2등-1등), runner_up_id} 또는 None
    """
    try:
        ids = list(ids)
        if len(ids) < 2 or any(scores.get(i) is None for i in ids):
            return None
        usable = [(float(scores[i]), i) for i in ids]
        usable.sort(key=lambda pair: pair[0], reverse=higher_is_better)
        (best, best_id), (second, second_id) = usable[0], usable[1]
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"후보 점수 형식 오류: {exc}") from exc
    return {"id": best_id, "score": best, "margin": abs(best - second), "runner_up_id": second_id}


def role_of(row: dict[str, Any]) -> str:
    """결과 행의 입력 역할. 병실 묶음(session ward_*)은 kind로, 문장 촬영 폴더는 모두 병실 목록 밖 입력이다."""
    kind = row.get("kind")
    session = row.get("session")
    if isinstance(session, str) and session.startswith("ward_"):
        mapping = {"fixed": ROLE_COMMAND, "decoy": ROLE_DECOY, "short": ROLE_SHORT, "nonspeech": ROLE_NONSPEECH}
    else:
        mapping = {"fixed": ROLE_SENT_SPEECH, "free": ROLE_SENT_SPEECH, "nonspeech": ROLE_SENT_NONSPEECH}
    if kind not in mapping:
        raise ValueError(f"알 수 없는 클립 종류: session={session!r} kind={kind!r} ({row.get('file')})")
    return mapping[kind]


def row_phrase(row: dict[str, Any]) -> dict[str, Any] | None:
    """녹화 한 건의 지정 문장 결과 (창이 하나이고 phrase가 있을 때만, 서비스와 같은 조건)."""
    results = row.get("results") or []
    if len(results) != 1 or not isinstance(results[0], dict):
        return None
    phrase = results[0].get("phrase")
    return phrase if isinstance(phrase, dict) else None


def window_kind(row: dict[str, Any]) -> str:
    """phrase가 없을 때의 결과 종류: error / no_speech / lip_fail / no_result.

    화면(wardResult.js classifyWardResults)과 같게, 모든 창이 표시([발화 없음]·[검출 실패 구간])이면 입이 안 보인 결과다:
    전부 [발화 없음]이면 no_speech, 검출 실패가 하나라도 있으면 lip_fail.
    """
    if row.get("status") != "done":
        return OUT_ERROR
    results = [w for w in row.get("results") or [] if isinstance(w, dict)]
    if not results:
        return OUT_ERROR
    texts = [str(w.get("text") or "").strip() for w in results]
    markers = (common.NO_SPEECH_TEXT, common.FAIL_TEXT)
    if all(t in markers for t in texts):
        return OUT_NO_SPEECH if all(t == common.NO_SPEECH_TEXT for t in texts) else OUT_LIP_FAIL
    return OUT_NO_RESULT


def _float(value: Any) -> float | None:
    """숫자면 float, 아니면 None."""
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def phrase_issue(phrase: dict[str, Any] | None, commands: dict[str, Any], phrase_set: str = "ward") -> str | None:
    """화면이 병실 명령으로 해석하지 않는 phrase의 사유 (wardResult.js와 같다). 문제가 없거나 phrase가 없으면 None.

    'other_set': set이 없거나(옛 서버·문장 모드 결과) phrase_set이 아님, 'unknown_id': 1등 ID가 명령 목록 밖,
    'bad_phrase': id가 문자열이 아니거나 margin·score가 숫자가 아님.
    """
    if phrase is None:
        return None
    if not isinstance(phrase.get("id"), str):
        return "bad_phrase"
    if phrase.get("set") != phrase_set:
        return "other_set"
    if phrase["id"] not in commands["texts"]:
        return "unknown_id"
    if _float(phrase.get("margin")) is None or _float(phrase.get("score")) is None:
        return "bad_phrase"
    return None


def server_rule_of(phrase: dict[str, Any] | None) -> dict[str, float | None] | None:
    """phrase.rule(서버가 쓴 기준값) → {min_margin, min_score, possible_min_score, enroll_max_distance}. 없으면 None (옛 서버)."""
    rule = phrase.get("rule") if isinstance(phrase, dict) else None
    if not isinstance(rule, dict):
        return None
    return {k: (_float(rule.get(k)) if rule.get(k) is not None else None) for k in common.SERVER_RULE_KEYS}


def row_enroll(row: dict[str, Any], commands: dict[str, Any]) -> dict[str, Any]:
    """서비스 결과 행의 서버 등록 결과(results[0].enroll) → 판정 전 기록의 svc_* 칸.

    match 결과의 status가 ok이고 top_id가 명령 목록 안이고 distance가 숫자일 때만(화면이 쓰는 조건과 같다)
    svc_tmpl_top·svc_tmpl_dist·svc_tmpl_dists를 채운다(그 밖은 None = 앱도 등록으로 고르지 않는다). svc_tmpl_dists는
    distances 중 명령 목록 안의 숫자 값(서버는 20개 모두 준다)이며 1등은 늘 들어간다. 처리 실패 행(status가 done이 아님)은 읽지 않는다.
    """
    empty = {"svc_enroll_mode": None, "svc_enroll": None, "svc_enroll_status": None, "svc_is_template": False,
             "svc_profile": None, "svc_tmpl_dists": None, "svc_tmpl_top": None, "svc_tmpl_dist": None,
             "svc_within": None, "svc_agree": None, "svc_accepted": None, "svc_max_distance": None,
             "svc_profile_sha256": row.get("profile_sha256")}
    if row.get("status") != "done":
        return empty
    results = row.get("results") or []
    first = results[0] if results and isinstance(results[0], dict) else {}
    enr = first.get("enroll")
    if not isinstance(enr, dict) or enr.get("mode") not in ("store", "match"):
        return empty
    out = dict(empty)
    out.update(svc_enroll_mode=enr["mode"], svc_profile=enr.get("profile"),
               svc_enroll={k: enr.get(k) for k in ("mode", "profile", "command", "stored", "reason", "status", "count",
                                                    "top_id", "distance", "runner_up_id", "runner_up_distance",
                                                    "max_distance", "within", "agree", "accepted", "text_top",
                                                    "text_top_matches", "complete")})
    if enr["mode"] == "store":
        out.update(svc_is_template=True, svc_enroll_status="stored" if enr.get("stored") else
                   f"not_stored:{enr.get('reason')}")
        return out
    out.update(svc_enroll_status=enr.get("status"), svc_within=enr.get("within"), svc_agree=enr.get("agree"),
               svc_accepted=enr.get("accepted"),
               svc_max_distance=_float(enr.get("max_distance")) if enr.get("max_distance") is not None else None)
    raw = enr.get("distances") if isinstance(enr.get("distances"), dict) else {}
    top, dist = enr.get("top_id"), _float(enr.get("distance"))
    if enr.get("status") == "ok" and isinstance(top, str) and top in commands["texts"] and dist is not None:
        dists = {i: float(raw[i]) for i in commands["ids"] if _float(raw.get(i)) is not None}
        dists.setdefault(top, dist)
        out.update(svc_tmpl_dists=dists, svc_tmpl_top=top, svc_tmpl_dist=dist)
    return out


def prepare(row: dict[str, Any], commands: dict[str, Any], extras: dict[str, Any] | None = None) -> dict[str, Any]:
    """결과 행 → 판정 전 기록 (규칙과 무관한 정보).

    extras(선택, ward_enroll.py 출력의 같은 클립): text_scores{명령 ID: 토큰당 점수}, tmpl_dists{명령 ID: 등록 거리}, is_template.
    phrase가 병실 묶음(set 'ward')이 아니거나 1등 ID가 명령 목록 밖이면 화면처럼 지정 문장 결과 없음(phrase_problem에 사유)이고,
    2등 ID가 목록 밖이거나 1등과 같으면 2등 없음으로 본다.
    서버 등록 결과(results[0].enroll)는 svc_* 칸에 따로 둔다. tmpl_dists는 오프라인(extras) 값이며 enroll_source는
    'offline'(거리 있음) 또는 None이다. 서버 등록 거리로 판정하려면 use_enroll_source(bases, 'service')를 거친다.
    """
    try:
        role = role_of(row)
        raw_phrase = row_phrase(row) if row.get("status") == "done" else None  # 처리 실패 행은 결과가 있어도 판정하지 않는다
        phrase_problem = phrase_issue(raw_phrase, commands)
        phrase = raw_phrase if phrase_problem is None else None
        runner = phrase.get("runner_up_id") if phrase else None
        if runner is not None and (runner not in commands["texts"] or runner == phrase.get("id")):
            runner = None
        results = [w for w in row.get("results") or [] if isinstance(w, dict)]
        single_ok = len(results) == 1 and results[0].get("text") not in (common.NO_SPEECH_TEXT, common.FAIL_TEXT)
        duration_ms = (_float(row.get("duration_sec")) * 1000.0 if _float(row.get("duration_sec")) is not None
                       else _float(row.get("duration_ms")))
        start_ms = _float(results[0].get("start_ms")) if single_ok else None
        end_ms = _float(results[0].get("end_ms")) if single_ok else None
        tail_gap = (duration_ms - end_ms) if (duration_ms is not None and end_ms is not None) else None
        has_extras = bool(extras)
        extras = extras or {}
        truth = str(row.get("sentence_id")) if role == ROLE_COMMAND else None
        if truth is not None and truth not in commands["texts"]:
            raise ValueError(f"명령 목록에 없는 명령 ID: {truth} ({row.get('file')})")
        return {
            "file": row.get("file"), "participant": row.get("participant"), "session": row.get("session"),
            "role": role, "kind": row.get("kind"), "truth": truth, "sentence_id": row.get("sentence_id"),
            "condition": row.get("condition"), "posture": row.get("posture") or ("reclined" if row.get("condition") == "reclined"
                                                                                    else "upright"),
            "rep": row.get("rep"), "status": row.get("status"),
            "window_kind": None if phrase else (OUT_NO_RESULT if phrase_problem else window_kind(row)),
            "phrase_problem": phrase_problem,
            "top": phrase.get("id") if phrase else None, "runner_up": runner,
            "margin": _float(phrase.get("margin")) if phrase else None, "score": _float(phrase.get("score")) if phrase else None,
            "server_accepted": phrase.get("accepted") if phrase else None,
            "phrase_set": raw_phrase.get("set") if raw_phrase else None,
            "seconds": _float(row.get("seconds")), "duration_ms": duration_ms, "start_ms": start_ms, "end_ms": end_ms,
            "window_ms": (end_ms - start_ms) if (start_ms is not None and end_ms is not None) else None,
            "tail_gap_ms": tail_gap,
            "text_scores": extras.get("text_scores"), "tmpl_dists": extras.get("tmpl_dists"),
            "is_template": bool(extras.get("is_template")), "has_extras": has_extras,
            "enroll_source": "offline" if extras.get("tmpl_dists") else None,
            "phrase_top": phrase.get("id") if phrase else None,
            "server_rule": server_rule_of(phrase),
            **row_enroll(row, commands),
        }
    except (AttributeError, KeyError, TypeError) as exc:
        raise ValueError(f"결과 행 형식 오류({row.get('file') if isinstance(row, dict) else row!r}): {exc}") from exc


def evaluate(base: dict[str, Any], rule: Rule) -> dict[str, Any]:
    """판정 전 기록에 규칙을 적용한다. rule.ids가 있으면 후보별 점수로 그 목록 안에서 다시 순위를 매긴다."""
    out = dict(base)
    tmpl_top = tmpl_dist = None
    dists = base.get("tmpl_dists")
    if rule.max_distance is not None and dists:
        if (base.get("enroll_source") == "service" and rule.ids is None and base.get("svc_tmpl_top") in dists
                and base.get("svc_tmpl_dist") is not None):
            # 서버 등록: 앱과 같게 서버의 top_id·distance를 그대로 쓴다 (distances 최솟값과 같은지는 enroll_check가 본다)
            tmpl_top, tmpl_dist = base["svc_tmpl_top"], float(base["svc_tmpl_dist"])
        else:
            # 등록 템플릿이 없는(등록 영상 검출 실패) 명령은 빼고 남은 템플릿 안에서 가장 가까운 명령을 본다
            avail = [i for i in (rule.ids or list(dists)) if dists.get(i) is not None]
            tr = rank_scores(dists, avail, higher_is_better=False) if len(avail) >= 2 else None
            if tr is not None:
                tmpl_top, tmpl_dist = tr["id"], tr["score"]
    out["tmpl_top"], out["tmpl_dist"] = tmpl_top, tmpl_dist
    out["enroll_from"] = (base.get("enroll_source") or "offline") if tmpl_top is not None else None
    if rule.ids is None:
        top, runner, margin, score = base["top"], base["runner_up"], base["margin"], base["score"]
        if top is None:
            out.update(outcome=base["window_kind"] or OUT_NO_RESULT, chosen_id=None, via=None, text_pass=False, conflict=False,
                       enroll_alarm=False)
            return out
    else:
        if base["top"] is None:
            out.update(outcome=base["window_kind"] or OUT_NO_RESULT, chosen_id=None, via=None, text_pass=False, conflict=False,
                       enroll_alarm=False)
            return out
        if not base.get("text_scores"):
            out.update(outcome=OUT_UNSCORABLE, chosen_id=None, via=None, text_pass=False, conflict=False,
                       enroll_alarm=False)
            return out
        ranked = rank_scores(base["text_scores"], rule.ids)
        if ranked is None:
            out.update(outcome=OUT_NO_RESULT, chosen_id=None, via=None, text_pass=False, conflict=False,
                       enroll_alarm=False)
            return out
        top, runner, margin, score = ranked["id"], ranked["runner_up_id"], ranked["margin"], ranked["score"]
        out.update(top=top, runner_up=runner, margin=margin, score=score)
    out.update(decide(top, runner, margin, score, rule, tmpl_top, tmpl_dist))
    return out


def evaluate_all(bases: Iterable[dict[str, Any]], rule: Rule) -> list[dict[str, Any]]:
    """기록 전체에 같은 규칙을 적용한다."""
    return [evaluate(b, rule) for b in bases]


# ---------------------------------------------------------------------------
# 등록 거리 출처 (서버 등록 / 오프라인 등록)
# ---------------------------------------------------------------------------

def has_service_enroll(bases: Iterable[dict[str, Any]]) -> bool:
    """서버 등록 결과(store 또는 match)가 있는 행이 있는가."""
    return any(b.get("svc_enroll_mode") in ("store", "match") for b in bases)


def has_offline_enroll(bases: Iterable[dict[str, Any]]) -> bool:
    """오프라인 등록 파일(ward_enroll.py apply)에서 읽은 행이 있는가."""
    return any(b.get("has_extras") for b in bases)


def use_enroll_source(bases: Iterable[dict[str, Any]], source: str) -> list[dict[str, Any]]:
    """판정 전 기록의 등록 거리 출처를 하나로 정한 복사본 (두 출처를 섞지 않는다).

    service: tmpl_dists = 서버 match 결과의 distances(status ok일 때만), is_template = 서버 store(등록 영상) 행.
    offline: tmpl_dists·is_template = ward_enroll.py 출력 그대로(prepare 기본값).
    none   : 등록 거리를 쓰지 않는다.

    Raises:
        ValueError: 알 수 없는 출처
    """
    if source not in ENROLL_SOURCES:
        raise ValueError(f"알 수 없는 등록 거리 출처: {source!r} ({ENROLL_SOURCES})")
    out = []
    for b in bases:
        c = dict(b)
        if source == "service":
            c["tmpl_dists"] = b.get("svc_tmpl_dists")
            c["is_template"] = bool(b.get("svc_is_template"))
            c["enroll_source"] = "service" if c["tmpl_dists"] else None
        elif source == "offline":
            c["enroll_source"] = "offline" if c.get("tmpl_dists") else None
        else:
            c["tmpl_dists"] = None
            c["enroll_source"] = None
        out.append(c)
    return out


def enroll_population(bases: Iterable[dict[str, Any]], source: str) -> list[dict[str, Any]]:
    """등록 규칙을 적용할 시험 클립.

    service: 서버 store(등록 영상) 행만 빼고 모두. 등록 거리가 없는 클립(등록 미완성·창 여럿 등)은 앱처럼 글자 규칙만 받는다.
    offline: 등록 템플릿을 뺀, 거리가 있는 클립만(ward_enroll.py 오프라인 평가와 같다).
    """
    bases = list(bases)
    if source == "service":
        return [b for b in bases if not b.get("is_template")]
    if source == "offline":
        return [b for b in bases if b.get("tmpl_dists")]
    return bases


def profile_problems(bases: Iterable[dict[str, Any]]) -> list[str]:
    """서버 등록 결과가 사람마다 등록 하나·같은 등록 상태로 채점됐는지 (문제 목록).

    한 사람의 클립이 서로 다른 등록 ID나 서로 다른 등록 상태(profile_sha256: ward_service_batch.py가 match 단계 시작 때 잰
    등록 지문)로 채점됐거나, 한 등록 ID를 두 사람 이상이 썼으면 문제다(다른 사람의 등록으로 채점하면 등록 효과가 아니다).
    """
    ids_by_pid: dict[str, set[str]] = {}
    state_by_pid: dict[str, set[Any]] = {}
    pids_by_profile: dict[str, set[str]] = {}
    for b in bases:
        mode = b.get("svc_enroll_mode")
        if mode not in ("store", "match"):
            continue
        pid, prof = str(b.get("participant")), str(b.get("svc_profile"))
        ids_by_pid.setdefault(pid, set()).add(prof)
        pids_by_profile.setdefault(prof, set()).add(pid)
        if mode == "match":
            state_by_pid.setdefault(pid, set()).add(b.get("svc_profile_sha256"))
    problems = [f"{pid}: 등록 ID가 여럿입니다({', '.join(sorted(v))})" for pid, v in sorted(ids_by_pid.items()) if len(v) > 1]
    problems += [f"{pid}: 서로 다른 등록 상태로 채점한 클립이 섞여 있습니다(profile_sha256 {len(v)}가지)"
                 for pid, v in sorted(state_by_pid.items()) if len(v) > 1]
    problems += [f"등록 {prof}을(를) 여러 사람이 썼습니다({', '.join(sorted(v))})"
                 for prof, v in sorted(pids_by_profile.items()) if len(v) > 1]
    return problems


def enroll_source_counts(evals: Iterable[dict[str, Any]]) -> dict[str, int]:
    """판정에서 등록 1등을 본 출처별 클립 수 {service, offline, none}."""
    out = {"service": 0, "offline": 0, "none": 0}
    for e in evals:
        out[e.get("enroll_from") or "none"] += 1
    return out


def _close(a: float | None, b: float | None, tol: float) -> bool:
    """둘 다 None이거나 차이가 tol 이하."""
    if a is None or b is None:
        return a is None and b is None
    return abs(float(a) - float(b)) <= tol


def enroll_check(evals: list[dict[str, Any]], rule: Rule) -> dict[str, Any]:
    """서버 등록 match 결과의 within·agree·accepted가 이 규칙의 D로 다시 판정한 값과 같은지 (서버 기준값·계산 확인용).

    다시 판정: within = status ok이고 D가 있고 거리 ≤ D, agree = 등록 1등 = 서버 글자 1등, accepted = within and agree.
    서버가 적은 max_distance가 D와 다른 클립, top_id·distance가 distances의 최솟값과 다른 클립도 센다.
    """
    sel = [e for e in evals if e.get("svc_enroll_mode") == "match"]
    mism: list[str] = []
    dmism: list[str] = []
    incons: list[str] = []
    statuses: dict[str, int] = {}
    D = rule.max_distance
    for e in sel:
        st = str(e.get("svc_enroll_status"))
        statuses[st] = statuses.get(st, 0) + 1
        ok = st == "ok" and e.get("svc_tmpl_dist") is not None
        within = bool(ok and D is not None and float(e["svc_tmpl_dist"]) <= D)
        agree = bool(ok and e.get("svc_tmpl_top") == e.get("phrase_top") and e.get("phrase_top") is not None)
        want = {"within": within, "agree": agree, "accepted": within and agree}
        got = {"within": bool(e.get("svc_within")), "agree": bool(e.get("svc_agree")), "accepted": bool(e.get("svc_accepted"))}
        if want != got:
            mism.append(f"{e['file']}:{','.join(k for k in want if want[k] != got[k])}")
        if not _close(e.get("svc_max_distance"), D, common.RULE_TOL):
            dmism.append(e["file"])
        dists = e.get("svc_tmpl_dists")
        if dists:
            best = min(dists.values())
            if not _close(dists.get(e.get("svc_tmpl_top")), best, ENROLL_DIST_TOL) or \
                    not _close(e.get("svc_tmpl_dist"), best, ENROLL_DIST_TOL):
                incons.append(e["file"])
    return {"n": len(sel), "statuses": statuses, "mismatch": len(mism), "mismatch_files": sorted(mism)[:20],
            "max_distance_mismatch": len(dmism), "max_distance_mismatch_files": sorted(dmism)[:20],
            "top_inconsistent": len(incons), "top_inconsistent_files": sorted(incons)[:20], "rule_max_distance": D}


def rule_check(evals: list[dict[str, Any]], rule: Rule, expected: dict[str, float | None] | None = None) -> dict[str, Any]:
    """서버가 phrase.rule에 적은 기준값이 기대 값과 같은지 (서버를 같은 기준값으로 켰는지 확인용, 전체 목록만).

    expected를 주지 않으면 이 규칙의 값(등록 거리는 rule.max_distance)과 비교한다. 동결 파일로 볼 때는
    common.expected_server_rule(동결 파일)을 준다(등록 결과가 없는 행도 서버는 동결 D로 켜져 있어야 한다).
    """
    want = expected if expected is not None else {
        "min_margin": rule.min_margin, "min_score": rule.min_score, "possible_min_score": rule.possible_min_score,
        "enroll_max_distance": rule.max_distance}
    with_rule = [e for e in evals if isinstance(e.get("server_rule"), dict)]
    mism: list[str] = []
    fields: dict[str, int] = {}
    for e in with_rule:
        bad = [k for k in common.SERVER_RULE_KEYS if not _close(e["server_rule"].get(k), want[k], common.RULE_TOL)]
        for k in bad:
            fields[k] = fields.get(k, 0) + 1
        if bad:
            mism.append(e["file"])
    values = sorted({tuple(e["server_rule"].get(k) for k in common.SERVER_RULE_KEYS) for e in with_rule},
                    key=lambda t: str(t))
    return {"n": len(with_rule), "mismatch": len(mism), "mismatch_files": sorted(mism)[:20], "fields": fields,
            "server_values": [dict(zip(common.SERVER_RULE_KEYS, v)) for v in values][:5], "expected": want}


# ---------------------------------------------------------------------------
# 지표
# ---------------------------------------------------------------------------

def quantile(values: list[float], q: float) -> float:
    """선형 보간 분위수 (값이 하나면 그 값)."""
    s = sorted(values)
    if not s:
        raise ValueError("빈 목록의 분위수")
    pos = q * (len(s) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def distribution(values: Iterable[float | None]) -> dict[str, Any] | None:
    """min/p10/중앙값/p90/max 요약 (값이 없으면 None)."""
    v = [float(x) for x in values if x is not None]
    if not v:
        return None
    return {"n": len(v), "min": round(min(v), 3), "p10": round(quantile(v, 0.1), 3), "median": round(statistics.median(v), 3),
            "p90": round(quantile(v, 0.9), 3), "max": round(max(v), 3)}


def count(k: int, n: int) -> dict[str, Any]:
    """{k, n, text} — 0건이면 한쪽 95% 상한을 붙인다."""
    return {"k": k, "n": n, "upper95_if_zero": common.upper95(n) if k == 0 and n > 0 else None, "text": common.frac(k, n)}


def outcome_counts(evals: list[dict[str, Any]]) -> dict[str, int]:
    """결과 종류별 개수 (없는 종류는 뺀다)."""
    out: dict[str, int] = {}
    for e in evals:
        out[e["outcome"]] = out.get(e["outcome"], 0) + 1
    return out


def command_table(evals: list[dict[str, Any]]) -> dict[str, Any]:
    """명령 클립 묶음 하나: 1등 정답, 표시, 표시 중 오답, 결과 종류."""
    cmd = [e for e in evals if e["role"] == ROLE_COMMAND]
    ranked = [e for e in cmd if e["top"] is not None]
    shown = [e for e in cmd if e.get("chosen_id") is not None]
    wrong = [e for e in shown if e["chosen_id"] != e["truth"]]
    return {"n": len(cmd), "ranked": len(ranked), "top1_correct": sum(e["top"] == e["truth"] for e in ranked),
            "shown": count(len(shown), len(cmd)), "shown_wrong": count(len(wrong), len(shown)),
            "shown_wrong_clips": sorted(f"{e['file']}->{e['chosen_id']}" for e in wrong),
            "via_enroll": sum(e.get("via") == "enroll" for e in shown),
            "via_enroll_wrong": sum(e.get("via") == "enroll" for e in wrong), "outcomes": outcome_counts(cmd)}


def emergency_recall(evals: list[dict[str, Any]], rule: Rule) -> dict[str, Any]:
    """긴급 명령 클립이 빨간 카드 또는 긴급 가능성으로 간 수 (조건별, 전체)."""
    em = [e for e in evals if e["role"] == ROLE_COMMAND and e["truth"] in rule.emergency_ids]

    def block(sel: list[dict[str, Any]]) -> dict[str, Any]:
        hit = [e for e in sel if e["outcome"] in ALARM_OUTCOMES]
        return {"reached": count(len(hit), len(sel)), "red": sum(e["outcome"] == OUT_EMERGENCY for e in sel),
                "possible": sum(e["outcome"] == OUT_POSSIBLE for e in sel),
                "missed_clips": sorted(f"{e['file']}:{e['outcome']}" for e in sel if e["outcome"] not in ALARM_OUTCOMES)}

    conds = sorted({str(e["condition"]) for e in em})
    return {"all": block(em), **{f"condition_{c}": block([e for e in em if e["condition"] == c]) for c in conds}}


def false_alarms(evals: list[dict[str, Any]], rule: Rule) -> dict[str, Any]:
    """긴급이 아닌 입력이 빨간 카드·긴급 가능성으로 간 수 (역할별). 머리 수치에는 짧은 긴급 표현을 넣지 않는다."""
    groups = {"command_non_emergency": [e for e in evals if e["role"] == ROLE_COMMAND and e["truth"] not in rule.emergency_ids],
              **{r: [e for e in evals if e["role"] == r] for r in NEGATIVE_ROLES}}
    out: dict[str, Any] = {}
    total_k = total_n = 0
    for name, sel in groups.items():
        hit = [e for e in sel if e["outcome"] in ALARM_OUTCOMES]
        out[name] = {"alarm": count(len(hit), len(sel)), "red": sum(e["outcome"] == OUT_EMERGENCY for e in sel),
                     "possible": sum(e["outcome"] == OUT_POSSIBLE for e in sel),
                     "clips": sorted(f"{e['file']}:{e['outcome']}:{e.get('chosen_id') or e.get('top')}" for e in hit)}
        total_k += len(hit)
        total_n += len(sel)
    out["total"] = count(total_k, total_n)
    return out


def negatives_shown(evals: list[dict[str, Any]]) -> dict[str, Any]:
    """명령으로 실행되면 안 되는 입력이 명령으로 표시된 수 (역할별, 합계)."""
    out: dict[str, Any] = {}
    total_k = total_n = 0
    for r in NEGATIVE_ROLES:
        sel = [e for e in evals if e["role"] == r]
        hit = [e for e in sel if e.get("chosen_id") is not None]
        out[r] = {"shown": count(len(hit), len(sel)), "clips": sorted(f"{e['file']}->{e['chosen_id']}" for e in hit),
                  "outcomes": outcome_counts(sel)}
        total_k += len(hit)
        total_n += len(sel)
    out["total"] = count(total_k, total_n)
    return out


def short_table(evals: list[dict[str, Any]]) -> dict[str, Any]:
    """짧은 긴급 표현(목록 밖): 긴급(빨간 카드·가능성)으로 간 수(참고), 일반 명령으로 실행된 수(실행된 틀린 요청, 합격 기준 0)."""
    sel = [e for e in evals if e["role"] == ROLE_SHORT]
    alarm = [e for e in sel if e["outcome"] in ALARM_OUTCOMES]
    as_request = [e for e in sel if e["outcome"] == OUT_REQUEST]
    return {"reached_alarm": count(len(alarm), len(sel)), "possible": sum(e["outcome"] == OUT_POSSIBLE for e in sel),
            "red": sum(e["outcome"] == OUT_EMERGENCY for e in sel), "shown_as_request": count(len(as_request), len(sel)),
            "as_request_clips": sorted(f"{e['file']}->{e['chosen_id']}" for e in as_request),
            "clips": sorted(f"{e['file']}:{e['outcome']}:{e.get('chosen_id') or e.get('top')}" for e in sel)}


def posture_table(evals: list[dict[str, Any]]) -> dict[str, Any]:
    """소리 없는 명령 클립의 자세별 표시율 (서 있는 자세 = 조건 silent, 기댄 자세 = 조건 reclined)."""
    out = {}
    for posture, cond in (("upright", "silent"), ("reclined", "reclined")):
        sel = [e for e in evals if e["role"] == ROLE_COMMAND and e["condition"] == cond]
        out[posture] = {"shown": count(sum(e.get("chosen_id") is not None for e in sel), len(sel)),
                        "lip_or_no_result": sum(e["outcome"] in (OUT_LIP_FAIL, OUT_NO_RESULT, OUT_NO_SPEECH) for e in sel),
                        "top1_correct": sum(e["top"] == e["truth"] for e in sel if e["top"] is not None)}
    return out


def recording_end_table(evals: list[dict[str, Any]], near_ms: float = NEAR_END_MS,
                        post_pad_ms: float = common.VAD_POST_PAD_MS, tol_ms: float = NEAR_END_TOL_MS) -> dict[str, Any]:
    """말 끝이 녹화 끝 near_ms 안에 닿은 클립 (병실 묶음 = 5초 자동 정지 녹화의 말소리 클립 중 창 하나).

    결과 end_ms = 마지막 활성 프레임 + 뒤 여유(post_pad_ms, 녹화 끝에서 잘림)이므로 말 끝 추정 = end_ms - post_pad_ms.
    (녹화 길이 - 말 끝 추정) ≤ near_ms + tol_ms이면 가깝다고 센다. 뒤 여유가 near_ms와 같으면 end_ms가 녹화 끝에 닿은 클립이다.
    문장 촬영 클립은 손으로 멈춘 녹화라 녹화 길이 판단에 쓰지 않는다.
    """
    sel = [e for e in evals if e["role"] in SPEECH_ROLES and e.get("tail_gap_ms") is not None
           and str(e.get("session") or "").startswith("ward_")]
    near = [e for e in sel if e["tail_gap_ms"] + post_pad_ms <= near_ms + tol_ms]
    return {"near_end": count(len(near), len(sel)), "near_ms": near_ms, "post_pad_ms": post_pad_ms,
            "tail_gap_ms": distribution(e["tail_gap_ms"] for e in sel),
            "clips": sorted(f"{e['file']}:{e['tail_gap_ms']:.0f}" for e in near)}


def syllables(text: str) -> int:
    """한글 음절 수."""
    return sum(1 for ch in str(text) if "가" <= ch <= "힣")


def window_length_table(evals: list[dict[str, Any]], commands: dict[str, Any], max_syllables: int = 9) -> dict[str, Any]:
    """짧은 명령(max_syllables 음절 이하)과 나머지의 발화 창 길이(ms, 앞 0.4초·뒤 0.5초 여유 포함)."""
    short_ids = sorted(i for i, t in commands["texts"].items() if syllables(t) <= max_syllables)
    cmd = [e for e in evals if e["role"] == ROLE_COMMAND and e.get("window_ms") is not None]
    return {"short_ids": short_ids, "short": distribution(e["window_ms"] for e in cmd if e["truth"] in short_ids),
            "others": distribution(e["window_ms"] for e in cmd if e["truth"] not in short_ids)}


def latency_table(evals: list[dict[str, Any]]) -> dict[str, Any] | None:
    """업로드 시작~최종 상태 벽시계 초."""
    return distribution(e.get("seconds") for e in evals)


def accepted_check(evals: list[dict[str, Any]], rule: Rule) -> dict[str, Any]:
    """서버 accepted가 이 규칙의 글자 판정과 같은지 (서버 기준값 확인용, 전체 목록만)."""
    with_flag = [e for e in evals if e.get("server_accepted") is not None and e.get("margin") is not None]
    mism = [e["file"] for e in with_flag
            if bool(e["server_accepted"]) != (e["margin"] >= rule.min_margin and e["score"] >= rule.min_score)]
    return {"n": len(with_flag), "mismatch": len(mism), "mismatch_files": sorted(mism)[:20]}


def summarize(evals: list[dict[str, Any]], rule: Rule, commands: dict[str, Any]) -> dict[str, Any]:
    """기획 10.2 지표 한 벌 (사람·묶음 하나 또는 합친 것)."""
    return {
        "n": len(evals), "roles": {r: sum(e["role"] == r for e in evals) for r in ROLES if any(e["role"] == r for e in evals)},
        "commands": command_table(evals),
        "commands_by_condition": {c: command_table([e for e in evals if e["condition"] == c])
                                  for c in sorted({str(e["condition"]) for e in evals if e["role"] == ROLE_COMMAND})},
        "emergency_recall": emergency_recall(evals, rule),
        "false_alarms": false_alarms(evals, rule),
        "negatives_shown": negatives_shown(evals),
        "short": short_table(evals),
        "posture": posture_table(evals),
        "recording_end": recording_end_table(evals),
        "window_length": window_length_table(evals, commands),
        "latency_seconds": latency_table(evals),
        "server_accepted_check": accepted_check(evals, rule) if rule.ids is None else None,
        "server_rule_check": rule_check(evals, rule) if rule.ids is None else None,
        "server_enroll_check": enroll_check(evals, rule) if rule.ids is None else None,
        "enroll_sources": enroll_source_counts(evals),
    }


def group_key(e: dict[str, Any]) -> str:
    """사람/촬영 묶음 키 (예: 'P01/ward_tune')."""
    return f"{e.get('participant')}/{e.get('session')}"


def summarize_groups(evals: list[dict[str, Any]], rule: Rule, commands: dict[str, Any]) -> dict[str, Any]:
    """사람·묶음별, 사람별(묶음 합침), 전체 지표."""
    out: dict[str, Any] = {"by_group": {}, "by_person": {}}
    for key in sorted({group_key(e) for e in evals}):
        out["by_group"][key] = summarize([e for e in evals if group_key(e) == key], rule, commands)
    for pid in sorted({str(e.get("participant")) for e in evals}):
        out["by_person"][pid] = summarize([e for e in evals if str(e.get("participant")) == pid], rule, commands)
    out["all"] = summarize(evals, rule, commands)
    return out


def repro_check(bases: list[dict[str, Any]], tol: float = 1e-6) -> dict[str, Any]:
    """서비스 phrase(전체 목록)와 오프라인 후보별 점수(text_scores) 순위가 같은지 (같은 입술 추출·같은 모델이면 같아야 한다)."""
    both = [b for b in bases if b.get("top") is not None and b.get("text_scores")]
    # 같은 클립의 오프라인 기록은 있는데 서비스만 지정 문장 결과가 있는 경우(입술 창 나누기가 다름)
    only_service = [b["file"] for b in bases if b.get("top") is not None and b.get("has_extras") and not b.get("text_scores")]
    diffs = []
    max_ds = max_dm = 0.0
    for b in both:
        r = rank_scores(b["text_scores"], list(b["text_scores"]))
        if r is None:
            diffs.append({"file": b["file"], "problem": "offline infeasible"})
            continue
        ds, dm = abs(r["score"] - b["score"]), abs(r["margin"] - b["margin"])
        max_ds, max_dm = max(max_ds, ds), max(max_dm, dm)
        if r["id"] != b["top"] or r["runner_up_id"] != b["runner_up"] or ds > tol or dm > tol:
            diffs.append({"file": b["file"], "service": [b["top"], b["runner_up"], b["score"], b["margin"]],
                          "offline": [r["id"], r["runner_up_id"], r["score"], r["margin"]]})
    return {"n_compared": len(both), "n_different": len(diffs), "max_score_abs_diff": max_ds,
            "max_margin_abs_diff": max_dm, "service_phrase_without_offline_scores": sorted(only_service)[:20],
            "examples": diffs[:10]}
