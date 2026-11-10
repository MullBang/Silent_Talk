"""채점 결과를 사람이 읽는 한국어 표(Markdown)로 만든다 (표준 라이브러리만)."""
from __future__ import annotations

from typing import Any

from ward_tools import metrics as mt

OUTCOME_KO = {mt.OUT_EMERGENCY: "빨간 카드", mt.OUT_REQUEST: "일반 요청", mt.OUT_POSSIBLE: "긴급 가능성",
              mt.OUT_NOT_CHOSEN: "고르지 않음", mt.OUT_NO_RESULT: "지정 문장 결과 없음", mt.OUT_NO_SPEECH: "발화 없음",
              mt.OUT_LIP_FAIL: "입술 검출 실패", mt.OUT_ERROR: "처리 오류", mt.OUT_UNSCORABLE: "점수 없음"}
ROLE_KO = {mt.ROLE_COMMAND: "명령", "command_non_emergency": "긴급 아닌 명령", mt.ROLE_DECOY: "함정 문장",
           mt.ROLE_SHORT: "짧은 긴급 표현", mt.ROLE_NONSPEECH: "말하지 않는 동작", mt.ROLE_SENT_SPEECH: "문장 촬영 말소리",
           mt.ROLE_SENT_NONSPEECH: "문장 촬영 말하지 않기"}


def _dist(d: dict[str, Any] | None, unit: str = "") -> str:
    """분포 요약 한 칸."""
    if not d:
        return "-"
    return f"중앙값 {d['median']}{unit} · p90 {d['p90']}{unit} · 최대 {d['max']}{unit} (n={d['n']})"


def _outcomes(o: dict[str, int]) -> str:
    """결과 종류 개수 한 칸."""
    return ", ".join(f"{OUTCOME_KO.get(k, k)} {v}" for k, v in sorted(o.items(), key=lambda kv: -kv[1])) or "-"


def _c(c: dict[str, Any] | None) -> str:
    """count 결과 한 칸 (분모가 0이면 '-')."""
    if not c or not c.get("n"):
        return "-"
    return c["text"]


def headline_rows(s: dict[str, Any]) -> list[tuple[str, str, str]]:
    """summarize 결과 → (지표, 합격 기준, 값) 행 (기획 10.2 순서)."""
    cmd = s["commands"]
    by_cond = s["commands_by_condition"]
    silent = by_cond.get("silent")
    er = s["emergency_recall"]
    er_silent = er.get("condition_silent", {}).get("reached")
    rows = [
        ("실행된 명령 중 오답 (표시 중 오답)", "0", _c(cmd["shown_wrong"])),
        ("긴급 명령(소리 없이) → 빨간 카드·긴급 가능성", "전부", _c(er_silent)),
        ("긴급 명령(전체 조건) → 빨간 카드·긴급 가능성", "참고", _c(er["all"]["reached"])),
        ("긴급 아닌 명령·함정·말하지 않기·문장 촬영 → 긴급", "0", _c(s["false_alarms"]["total"])),
        ("함정·말하지 않기·문장 촬영 → 명령 실행", "0", _c(s["negatives_shown"]["total"])),
        ("표시 비율 (명령 전체)", "참고 70% 이상", _c(cmd["shown"])),
        ("표시 비율 (소리 없이)", "참고", _c(silent["shown"]) if silent else "-"),
        ("1등 정답 (명령 전체)", "참고", f"{cmd['top1_correct']}/{cmd['n']}" if cmd["n"] else "-"),
        ("등록 거리로 고른 표시 (그중 오답)", "참고", f"{cmd.get('via_enroll', 0)} ({cmd.get('via_enroll_wrong', 0)})"
         if cmd["n"] else "-"),
        ("짧은 긴급 표현 → 긴급(빨간 카드·가능성)", "참고", _c(s["short"]["reached_alarm"])),
        ("짧은 긴급 표현 → 일반 요청으로 실행 (틀린 요청)", "0", _c(s["short"]["shown_as_request"])),
        ("기댄 자세 표시 비율", "참고", _c(s["posture"]["reclined"]["shown"])),
        ("말 끝이 녹화 끝 0.5초 안 (병실 5초 녹화)", "참고", _c(s["recording_end"]["near_end"])),
        ("짧은 명령(9음절 이하) 발화 창 길이(ms, 여유 0.9초 포함)", "참고", _dist(s["window_length"]["short"])),
        ("처리 시간(업로드~결과, 초)", "녹화 끝→표시 3초 이내", _dist(s["latency_seconds"], "초")),
    ]
    return rows


def table(groups: dict[str, dict[str, Any]], title: str) -> str:
    """여러 사람·묶음의 지표를 열로 놓은 표."""
    keys = list(groups)
    if not keys:
        return f"### {title}\n\n(자료 없음)\n"
    per = {k: headline_rows(groups[k]) for k in keys}
    lines = [f"### {title}", "", "| 지표 | 합격 기준 | " + " | ".join(keys) + " |", "|---|---|" + "---|" * len(keys)]
    for i, (name, crit, _v) in enumerate(per[keys[0]]):
        lines.append(f"| {name} | {crit} | " + " | ".join(per[k][i][2] for k in keys) + " |")
    lines.append("")
    return "\n".join(lines)


def detail(s: dict[str, Any]) -> str:
    """역할별 결과 종류와 문제 클립."""
    lines = ["| 입력 | 결과 종류 |", "|---|---|"]
    lines.append(f"| 명령 | {_outcomes(s['commands']['outcomes'])} |")
    for r in mt.NEGATIVE_ROLES:
        if r in s["negatives_shown"] and s["negatives_shown"][r]["shown"]["n"]:
            lines.append(f"| {ROLE_KO[r]} | {_outcomes(s['negatives_shown'][r]['outcomes'])} |")
    out = "\n".join(lines) + "\n"
    wrong = s["commands"]["shown_wrong_clips"]
    if wrong:
        out += f"\n표시 중 오답: {', '.join(wrong)}\n"
    neg = [c for r in mt.NEGATIVE_ROLES for c in s["negatives_shown"].get(r, {}).get("clips", [])]
    if neg:
        out += f"\n목록 밖 입력이 명령으로 표시: {', '.join(neg)}\n"
    short_req = s["short"].get("as_request_clips") or []
    if short_req:
        out += f"\n짧은 긴급 표현이 일반 요청으로 실행: {', '.join(short_req)}\n"
    missed = s["emergency_recall"]["all"]["missed_clips"]
    if missed:
        out += f"\n긴급을 놓친 클립: {', '.join(missed)}\n"
    return out


def accepted_line(check: dict[str, Any] | None) -> str:
    """서버 accepted와 동결 기준값으로 다시 판정한 값의 비교 한 줄."""
    if not check:
        return "- 서버 accepted 대조: 자료 없음"
    files = check.get("mismatch_files") or []
    return (f"- 서버 accepted와 동결 기준값으로 다시 판정한 값: 비교 {check.get('n')}개 중 다름 {check.get('mismatch')}개"
            + (f" ({', '.join(files)})" if files else ""))


def rule_line(check: dict[str, Any] | None) -> str:
    """서버가 결과에 적은 기준값(phrase.rule)과 동결 값의 비교 한 줄."""
    if not check or not check.get("n"):
        return "- 서버 기준값(phrase.rule) 대조: 기준값을 적은 결과가 없음(옛 서버)"
    fields = ", ".join(f"{k} {v}개" for k, v in (check.get("fields") or {}).items())
    return (f"- 서버가 결과에 적은 기준값(phrase.rule)과 동결 값: 비교 {check['n']}개 중 다름 {check['mismatch']}개"
            + (f" ({fields})" if fields else ""))


def enroll_line(check: dict[str, Any] | None) -> str:
    """서버 등록 판정(enroll.within·agree·accepted)과 동결 D로 다시 판정한 값의 비교 한 줄."""
    if not check or not check.get("n"):
        return "- 서버 등록 판정 대조: 서버 등록(match) 결과 없음"
    st = ", ".join(f"{k} {v}" for k, v in sorted((check.get("statuses") or {}).items()))
    return (f"- 서버 등록 판정(within·agree·accepted)과 D {check.get('rule_max_distance')}로 다시 판정한 값: 비교 {check['n']}개 중 "
            f"다름 {check['mismatch']}개, 서버 max_distance 다름 {check.get('max_distance_mismatch', 0)}개, 1등 거리 불일치 "
            f"{check.get('top_inconsistent', 0)}개 (등록 상태: {st})")


def _src(s: dict[str, Any] | None) -> str:
    """표 제목에 붙일 등록 거리 출처 개수."""
    used = (s or {}).get("enroll_sources") or {}
    if not used.get("service") and not used.get("offline"):
        return "등록 1등을 본 클립 없음"
    parts = [f"{name} {used[k]}개" for k, name in (("service", "서버 등록"), ("offline", "오프라인 등록")) if used.get(k)]
    return "등록 거리 출처: " + ", ".join(parts) + f", 거리 없음 {used.get('none', 0)}개"


def markdown_report(result: dict[str, Any]) -> str:
    """ward_eval_tools report 결과 → Markdown. 표마다 등록 거리 출처(서버 등록·오프라인 등록)를 제목에 적는다."""
    f = result["freeze"]
    main = result.get("main") or {}
    all_main = result["full"].get("all") or {}
    check = all_main.get("server_accepted_check")
    echeck = all_main.get("server_enroll_check")
    rcheck = result.get("server_rule_check")
    floor = f["full"].get("possible_min_score")
    lines = ["# 병실 명령 평가 결과 (동결 파일 기준)", ""]
    if check and check.get("mismatch"):
        lines += [f"> **주의: 서버 accepted가 동결 기준값과 다른 클립이 {check['mismatch']}개입니다.** 서버가 동결 기준값"
                  "(full_list.env)으로 켜지지 않았을 수 있습니다. 화면은 서버 accepted를 쓰므로 이 표와 시연 동작이 다를 수 있습니다.", ""]
    if rcheck and rcheck.get("mismatch"):
        lines += [f"> **주의: 서버가 결과에 적은 기준값(phrase.rule)이 동결 값과 다른 클립이 {rcheck['mismatch']}개입니다.** "
                  "서버를 full_list.env의 환경 변수 4개로 켰는지 확인하세요.", ""]
    if echeck and (echeck.get("mismatch") or echeck.get("max_distance_mismatch") or echeck.get("top_inconsistent")):
        lines += ["> **주의: 서버 등록 판정이 동결 D로 다시 판정한 값과 다른 클립이 있습니다.** 화면은 서버 enroll.accepted·within을 "
                  "쓰므로 이 표와 시연 동작이 다를 수 있습니다.", ""]
    if main.get("enrollment_source") == "service":
        enroll_desc = (f"서버 등록 거리 D {f.get('max_distance')} (앱 경로, 등록 영상 {main.get('templates_excluded')}개는 시험에서 뺌, "
                       "거리가 없는 클립은 글자 규칙만)")
    elif main.get("service_enroll_rows"):
        enroll_desc = "등록 거리 쓰지 않음(동결 D 없음, 등록 영상은 시험에서 뺌)"
    else:
        enroll_desc = "등록 없음(결과 행에 서버 등록 결과가 없음)"
    lines += [f"- 동결 파일: `{f['path']}` (상태 {f['status']}, 커밋 {f.get('commit') or '확인 안 함'})",
              f"- 주 결과(앱과 같은 판정): 1·2등 차이 ≥ {f['full']['min_margin']}, 1등 점수 ≥ {f['full']['min_score']}, "
              + (f"긴급 가능성 하한 {floor}" if floor is not None
                 else "긴급 가능성 하한 없음(고르지 못했을 때 1·2등에 긴급이면 긴급 가능성)") + f", {enroll_desc}",
              f"- 동결 등록 거리 D: {f.get('max_distance')} (템플릿 {f.get('template')}, D를 정한 출처 {f.get('enroll_source')})"
              f"  · 축소 목록: {f.get('reduced_ids') or '없음'}",
              f"- 결과 행 파일: {', '.join(result['inputs']['rows_files'])}",
              accepted_line(check), rule_line(rcheck), enroll_line(echeck),
              "- 0건은 표본 수와 한쪽 95% 상한(1 − 0.05^(1/n))을 함께 적었다. 처음 보는 사람이 1명뿐이면 그 한계도 함께 밝힌다.",
              ""]
    for rf in (result.get("inputs") or {}).get("rows_info") or []:
        e = rf.get("enroll")
        if e:
            counts = rf.get("service_enroll_rows") or {}
            lines.append(f"- 서버 등록 `{e.get('profile')}` ({e.get('mode')}): `{rf['path']}` 등록 영상 {counts.get('store', 0)}개, "
                         f"시험 {counts.get('match', 0)}개, 실행 전 {e.get('status_before')}, 실행 뒤 {e.get('status_after')}, "
                         f"지움 {e.get('deleted')}")
    lines.append("")
    for fc in (result.get("inputs") or {}).get("folder_checks") or []:
        if fc.get("checked") and not fc.get("same_as_run"):
            lines += [f"> 주의: 촬영 폴더가 실행 뒤 바뀌었습니다: `{fc['clips_dir']}`", ""]
    if result.get("code_changed"):
        lines += [f"> 주의: 동결 뒤 바뀐 코드 파일 {len(result['code_changed'])}개: {', '.join(result['code_changed'])}", ""]
    lines.append(table(result["full"]["by_group"], f"1. 앱과 같은 판정 (주 결과, 사람·촬영 묶음별; {_src(all_main)})"))
    if len(result["full"]["by_person"]) < len(result["full"]["by_group"]):
        lines.append(table(result["full"]["by_person"], "1-1. 사람별 (촬영 묶음 합침)"))
    if result.get("full_text_only"):
        lines.append(table(result["full_text_only"]["by_group"], "1-2. 같은 클립, 등록 없이 글자 규칙만 (등록 효과 비교)"))
    if result.get("full_no_floor"):
        lines.append(table(result["full_no_floor"]["by_group"], f"1-3. 같은 클립, 긴급 가능성 하한 {floor} 없이 (비교)"))
    if result.get("all_rows_text"):
        lines.append(table(result["all_rows_text"]["by_group"],
                           "1-4. 등록 영상까지 모든 클립, 글자 규칙만 (참고: 등록 없이 쓸 때와 같은 묶음)"))
    for key, s in result["full"]["by_group"].items():
        lines += [f"#### {key} 세부", "", detail(s)]
    if result.get("enroll"):
        lines.append(table(result["enroll"]["by_group"],
                           f"2. 오프라인 등록(ward_enroll.py) 거리 규칙 (보조, 등록 템플릿을 뺀 시험 클립만; "
                           f"{_src(result['enroll'].get('all'))})"))
        lines.append(table(result["enroll_text_only"]["by_group"], "2-1. 같은 시험 클립, 글자 규칙만 (비교)"))
    if result.get("reduced"):
        lines.append(table(result["reduced"]["by_group"], "3. 축소 목록 (보조 결과, 오프라인 후보별 점수로 다시 순위)"))
    if result.get("repro"):
        r = result["repro"]
        lines += [f"- 서비스 1·2등·점수와 오프라인 후보별 점수의 일치: 비교 {r['n_compared']}개 중 다름 {r['n_different']}개", ""]
    return "\n".join(lines)


def calibration_markdown(draft: dict[str, Any]) -> str:
    """기준 맞추기 초안 → 사람이 검토할 요약."""
    ev = draft.get("evidence") or {}
    lines = ["# 병실 명령 기준 맞추기 초안 (사람이 검토한 뒤 커밋)", "",
             f"- 제안: 1·2등 차이 ≥ {draft['full_list']['min_margin']}, 1등 점수 ≥ {draft['full_list']['min_score']} "
             f"({draft['full_list']['chosen_by']})",
             f"- 긴급 가능성 점수 하한: {draft['decision_rule'].get('possible_min_score')} ({draft['decision_rule'].get('chosen_by')}; "
             f"화면 구현 {draft['decision_rule'].get('floor_implemented_in_ui')})",
             f"- 등록 거리 D: {(draft.get('enrollment') or {}).get('max_distance')} "
             f"({(draft.get('enrollment') or {}).get('chosen_by')}; 출처 {(draft.get('enrollment') or {}).get('source')}, "
             f"템플릿 {(draft.get('enrollment') or {}).get('template')}, "
             f"거리가 있는 목록 밖 클립 {(draft.get('enrollment') or {}).get('negatives_with_dist')})",
             "- 서버 환경 변수(full_list.env): " + (", ".join(f"{k}={v}" for k, v in
                                                        ((draft.get("full_list") or {}).get("env") or {}).items()) or "없음"),
             "- 봉인 폴더 지문: " + (", ".join(f"{sf.get('path')} {sf.get('files')}개 {str(sf.get('listing_sha256'))[:12]}…"
                                            for sf in draft.get("sealed_folders") or []) or "없음"),
             f"- 축소 목록({(draft.get('reduced_list') or {}).get('size')}개): {(draft.get('reduced_list') or {}).get('ids')}",
             f"  기준 {(draft.get('reduced_list') or {}).get('min_margin')} / {(draft.get('reduced_list') or {}).get('min_score')}",
             ""]
    lines += ["## 한 사람으로 정해 다른 사람에게 적용", "", "| 맞춘 사람 | M | S | 맞춘 사람 결과 | 다른 사람 결과 |", "|---|---|---|---|---|"]
    for key, v in (ev.get("lopo") or {}).items():
        g = v["fit"]
        if not g.get("feasible"):
            lines.append(f"| {key} | - | - | 맞출 수 없음 | - |")
            continue
        others = [k for k in v if k.startswith("on_") and k != "on_fit_person"]
        lines.append(f"| {key} | {g['M']} | {g['S']} | {_hl(v['on_fit_person'])} | "
                     + " / ".join(f"{k[3:]}: {_hl(v[k])}" for k in others) + " |")
    lines += ["", "## 합친 맞추기 (여유 δ별)", "", "| δ | M | S | 결과 |", "|---|---|---|---|"]
    for key, v in (ev.get("pooled") or {}).items():
        g = v["fit"]
        lines.append(f"| {key} | {g.get('M', '-')} | {g.get('S', '-')} | {_hl(v['all']) if 'all' in v else '맞출 수 없음'} |")
    prov = ev.get("provisional") or {}
    if prov:
        lines += ["", f"## 임시값 {prov.get('rule', {}).get('min_margin')} / {prov.get('rule', {}).get('min_score')}", "",
                  _hl(prov)]
    if ev.get("proposal_on_all"):
        lines += ["", "## 제안 기준(점수 하한·등록 거리 없이)을 조정 자료 전체에 적용", "", _hl(ev["proposal_on_all"])]
    floor = ev.get("possible_floor") or {}
    if floor:
        lines += ["", "## 참고: 긴급 가능성 점수 하한 (한 사람으로 정해 다른 사람에게 적용, --floor일 때만 초안에 넣음)", "",
                  "| 맞춘 사람 | 하한 | 필요한 값(클립) | 다른 사람 결과 |", "|---|---|---|---|"]
        for key, v in floor.items():
            f = v["fit"] if "fit" in v else v
            others = [k for k in v if k.startswith("on_")]
            fl = "-" if f.get("floor") is None else round(f["floor"], 3)
            need = "-" if f.get("needed") is None else f"{round(f['needed'], 3)} ({f.get('needed_clip')})"
            res = " / ".join(f"{k[3:]}: {_hl(v[k])}" for k in others) if others else (
                _hl(v["with_floor"]) if v.get("with_floor") else "-")
            lines.append(f"| {key} | {fl} | {need} | {res} |")
    dist = ev.get("distance") or {}
    if dist:
        lines += ["", "## 등록 거리 D (한 사람으로 정해 다른 사람에게 적용)", "",
                  "| 맞춘 사람 | D | 막는 거리 | 필요한 가장 큰 정답 거리 | 거리가 있는 목록 밖 클립 | 결과 |",
                  "|---|---|---|---|---|---|"]
        for key, v in dist.items():
            f = v["fit"]
            others = [k for k in v if k.startswith("on_")]
            blocking = f.get("blocking_dist") if f.get("blocking_dist") is not None else "모름(나쁜 클립 없음)"
            lines.append(f"| {key} | {f.get('D')} | {blocking} | {f.get('largest_needed_dist')} | "
                         f"{f.get('n_negatives_with_dist')}/{f.get('n_negatives')} | "
                         + " / ".join(f"{k[3:]}: {_hl(v[k])}" for k in others) + " |")
    red = draft.get("reduced_list") or {}
    if red.get("trace"):
        lines += ["", "## 축소 목록에서 뺀 순서", "",
                  ", ".join(f"{t['dropped']}({t['involvement']})" for t in red["trace"]), "", str(red.get("method"))]
    repro = ev.get("repro") or {}
    if repro:
        lines += ["", f"- 서비스 1·2등·점수와 오프라인 후보별 점수 일치: 비교 {repro.get('n_compared')}개 중 다름 "
                      f"{repro.get('n_different')}개, 최대 차이 {repro.get('max_score_abs_diff')}"]
    lines += ["", "## 자료", ""]
    for rf in (draft.get("data") or {}).get("rows_files", []):
        lines.append(f"- `{rf['path']}` ({rf.get('participants')}, {rf.get('sessions')}, 행 {rf.get('n_rows')}, "
                     f"{'합성' if rf.get('synthetic') else '실제'})")
    for note in draft.get("notes") or []:
        lines.append(f"- {note}")
    return "\n".join(lines) + "\n"


def _hl(h: dict[str, Any]) -> str:
    """headline 한 줄."""
    return (f"표시 {h['commands_shown']}, 오답 {h['shown_wrong']}, 목록 밖 표시 {h['negatives_shown']}, "
            f"짧은 긴급→요청 {h.get('short_as_request', '-')}, 거짓 긴급 {h['false_alarms']}, 긴급 도달 {h['emergency_recall']}")
