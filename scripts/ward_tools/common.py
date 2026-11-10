"""병실 명령 채점 도구 공용 부분 (표준 라이브러리만 쓴다: 서버 3.11·입술 추출·인식 가상환경 어디서나 import 가능).

- 명령 목록(frontend/src/ward/commands.json)과 긴급 명령 ID
- 봉인 확인: 처음 보는 사람(P02)과 병실 명령 평가용(ward_eval) 자료는 동결 파일을 만들기 전에는 읽지 않는다(기획 10.1).
  --allow-sealed는 --freeze(git에 커밋된 status frozen 동결 파일)와 함께만 받는다(freeze_gate). 지문 계산(해시만)은 예외다.
- 파일·폴더 지문: 폴더 목록 지문은 '파일명<TAB>크기<TAB>SHA-256' 줄을 이름순으로 '\\n'으로 이은 UTF-8 글의 SHA-256
- 0건의 한쪽 95% 상한 1 - 0.05^(1/n), 결과 행 파일 읽기·쓰기
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

ROOT = Path(__file__).resolve().parents[2]
COMMANDS_PATH = ROOT / "frontend" / "src" / "ward" / "commands.json"
WARD_PROMPTS_PATH = ROOT / "frontend" / "src" / "capture" / "ward_prompts.json"
PROMPTS_PATH = ROOT / "frontend" / "src" / "capture" / "prompts.json"

# 동결 파일 전에는 열지 않는 자료 (기획 문서 10.1, 메모: 최종 평가 대상 P02)
SEALED_SESSIONS = ("ward_eval",)
SEALED_PARTICIPANTS = ("P02",)
# 경로 조각에서 봉인 참가자를 찾는다 ('웹캠 영상 P02', 'p02최종', 'silent_talk_P02' 모두 걸린다)
_SEALED_PATH = re.compile(r"(?i)(?<![0-9a-z])p02(?![0-9])")
# 클립 기록의 참가자 ID ('P02', ' p02 ', 'P2', 'P002' 모두 봉인 참가자로 본다)
_SEALED_PID = re.compile(r"(?i)^p0*2$")

FREEZE_SCHEMA = "ward-freeze-v1"
# 등록 평가의 특징·거리·템플릿 이름 (ward_enroll.py 출력과 동결 파일 enrollment가 같아야 한다)
FEATURE_NAME = "olkavs_medium_hidden_256"
DISTANCE_NAME = "dtw_cosine_unit_steps_over_path_length"
ENROLL_TEMPLATES = ("voiced", "silent1", "both")
# 서버 등록(내 입 모양 등록, backend/services/ward_profiles.py)이 쓰는 템플릿 방식: 명령마다 소리 내어 한 번
SERVICE_TEMPLATE = "voiced"
# 서버 등록 ID 규칙 (backend/services/ward_profiles.py와 같다)
PROFILE_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,32}")
"""서버 등록 이름 규칙 (fullmatch로 쓴다: '$'는 끝의 줄바꿈도 받아들이므로 쓰지 않는다)."""
WINDOWS_RESERVED_NAMES = frozenset({"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(10)),
                                    *(f"LPT{i}" for i in range(10))})
"""서버(backend/services/ward_profiles.py)가 거부하는 Windows 장치 이름 (대소문자 무관)."""
# 서버가 병실 결과 phrase.rule에 적는 기준값 키와 그 환경 변수 (backend/config.py)
SERVER_RULE_KEYS = ("min_margin", "min_score", "possible_min_score", "enroll_max_distance")
SERVER_RULE_ENV = {"min_margin": "LIPNET_OLKAVS_WARD_MIN_MARGIN", "min_score": "LIPNET_OLKAVS_WARD_MIN_SCORE",
                   "possible_min_score": "LIPNET_OLKAVS_WARD_POSSIBLE_MIN_SCORE",
                   "enroll_max_distance": "LIPNET_OLKAVS_WARD_ENROLL_MAX_DISTANCE"}
ENV_NONE = "none"  # 환경 변수로 '쓰지 않음'을 줄 때의 값 (backend/config.py가 ''·none을 None으로 읽는다)
RULE_TOL = 1e-9    # 서버가 쓴 기준값과 동결 값을 같다고 볼 차이

VIDEO_SUFFIXES = (".webm", ".mp4")
LISTING_FORMAT = ("UTF-8 text, one line per regular file 'name<TAB>size_bytes<TAB>sha256_hex', lines sorted by name "
                  "and joined with '\\n' (no trailing newline); the listing fingerprint is the SHA-256 of that text")

# 서버가 결과 창 text에 넣는 고정 표시 (backend/services/olkavs_service.py와 같다)
FAIL_TEXT = "[검출 실패 구간]"
UNREC_TEXT = "[인식 불가]"
NO_SPEECH_TEXT = "[발화 없음]"

# 녹화 모드 입술 추출 설정 (backend/config.py의 OLKAVS_SEGMENT_SECONDS·OLKAVS_MIN_WINDOW_SECONDS·OLKAVS_CUT_SEARCH_SECONDS,
# backend/services/olkavs_service.py의 utterance=True 녹화 요청과 같다. 테스트가 config 값과 같은지 확인한다)
SERVICE_ROI = {"segment_seconds": 12.0, "min_window_seconds": 1.0, "cut_search_seconds": 6.0, "resample": False,
               "utterance": True}
# 녹화 모드의 발화 구간 뒤 여유(scripts/olkavs_transcribe.py VAD_POST_PAD_SECONDS). 결과 end_ms는 마지막 활성 프레임 + 이 값이다.
VAD_POST_PAD_MS = 500.0


class SealedDataError(ValueError):
    """봉인 자료(P02·ward_eval)를 허락 없이 읽으려 할 때."""


def load_commands(path: Path = COMMANDS_PATH) -> dict[str, Any]:
    """명령 목록을 읽는다.

    Returns:
        {version, ids(순서대로), texts{id: 인식용 문장}, display{id: 화면 문장}, emergency(긴급 ID 정렬 목록), sha256, path}

    Raises:
        ValueError: 파일이 없거나 형식이 틀렸을 때, 명령이 2개 미만이거나 ID가 겹칠 때
    """
    try:
        raw = path.read_bytes()
        data = json.loads(raw.decode("utf-8"))
        fixed = data["fixed"]
        ids = [str(p["id"]) for p in fixed]
        texts = {str(p["id"]): str(p["text"]) for p in fixed}
        display = {str(p["id"]): str(p.get("display") or p["text"]) for p in fixed}
        emergency = sorted(str(p["id"]) for p in fixed if p.get("emergency") is True)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"명령 목록을 읽지 못했습니다({path}): {exc}") from exc
    if len(ids) < 2 or len(set(ids)) != len(ids):
        raise ValueError(f"명령 목록이 2개 미만이거나 ID가 겹칩니다: {ids}")
    return {"version": data.get("version"), "ids": ids, "texts": texts, "display": display, "emergency": emergency,
            "sha256": hashlib.sha256(raw).hexdigest(), "path": str(path)}


def upper95(n: int) -> float | None:
    """0/n일 때 한쪽 95% 상한 = 1 - 0.05^(1/n). n이 0 이하면 None."""
    try:
        return None if n <= 0 else 1.0 - 0.05 ** (1.0 / n)
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"상한 계산 실패(n={n!r}): {exc}") from exc


def frac(k: int, n: int) -> str:
    """'k/n' 문자열. k가 0이고 n이 1 이상이면 한쪽 95% 상한을 붙인다(예: '0/12 (한쪽 95% 상한 22.1%)')."""
    try:
        if n <= 0:
            return f"{k}/{n}"
        if k == 0:
            return f"0/{n} (한쪽 95% 상한 {100 * upper95(n):.1f}%)"
        return f"{k}/{n}"
    except (TypeError, ValueError) as exc:
        raise ValueError(f"비율 표기 실패({k!r}/{n!r}): {exc}") from exc


def is_sealed_path(path: str | os.PathLike[str]) -> bool:
    """경로에 봉인 참가자(P02) 조각이 있는가 (대소문자 무시, 'P021' 같은 다른 번호는 아님)."""
    try:
        return bool(_SEALED_PATH.search(str(path)))
    except TypeError as exc:
        raise ValueError(f"경로 형식 오류: {path!r}") from exc


def is_sealed_participant(pid: Any) -> bool:
    """참가자 ID가 봉인 참가자(P02)인가 (앞뒤 공백·대소문자·앞자리 0 무시: 'P02', ' p02 ', 'P2')."""
    return bool(_SEALED_PID.match(str(pid or "").strip()))


def is_sealed_session(session: Any) -> bool:
    """촬영 묶음이 봉인 묶음(ward_eval)인가 (앞뒤 공백·대소문자 무시, 'ward_eval2'처럼 뒤에 붙은 것 포함)."""
    s = str(session or "").strip().lower()
    return any(s.startswith(x) for x in SEALED_SESSIONS)


def sealed_reasons(paths: Iterable[str | os.PathLike[str]], rows: Iterable[dict[str, Any]] = ()) -> list[str]:
    """봉인 자료라고 볼 이유 목록 (없으면 빈 목록). 경로의 P02 조각, 행의 participant P02, session ward_eval을 본다.

    Args:
        paths: 입력 폴더·파일 경로
        rows: 클립 기록·결과 행 (participant, session 키를 본다)
    """
    reasons: list[str] = []
    try:
        for p in paths:
            if p is not None and is_sealed_path(p):
                reasons.append(f"경로에 봉인 참가자(P02)가 있습니다: {p}")
        participants: set[str] = set()
        sessions: set[str] = set()
        for r in rows:
            if not isinstance(r, dict):
                continue
            if is_sealed_participant(r.get("participant")):
                participants.add(str(r.get("participant")).strip())
            if is_sealed_session(r.get("session")):
                sessions.add(str(r.get("session")).strip())
    except (TypeError, AttributeError) as exc:
        raise ValueError(f"봉인 확인 입력 형식 오류: {exc}") from exc
    for pid in sorted(participants):
        reasons.append(f"봉인 참가자({pid})의 클립이 있습니다")
    for s in sorted(sessions):
        reasons.append(f"봉인 촬영 묶음({s})의 클립이 있습니다")
    return reasons


def require_unsealed(paths: Iterable[str | os.PathLike[str]], rows: Iterable[dict[str, Any]] = (),
                     allow_sealed: bool = False, what: str = "이 작업") -> None:
    """봉인 자료면 allow_sealed 없이는 SealedDataError를 낸다 (allow_sealed는 freeze_gate를 통과한 뒤에만 True로 준다).

    Raises:
        SealedDataError: 봉인 이유가 있고 allow_sealed가 False일 때
    """
    reasons = sealed_reasons(list(paths), list(rows))
    if reasons and not allow_sealed:
        raise SealedDataError(f"{what}은(는) 봉인 자료에 실행하지 않습니다. 동결 파일을 커밋한 뒤에만 "
                              f"--allow-sealed --freeze <동결 파일>을 붙이세요. 이유: {'; '.join(reasons)}")


# ---------------------------------------------------------------------------
# 동결 파일 확인 (git)
# ---------------------------------------------------------------------------

def git(*args: str) -> str | None:
    """저장소에서 git 명령을 실행해 표준 출력을 돌려준다 (실패하면 None)."""
    try:
        res = subprocess.run(["git", *args], cwd=str(ROOT), capture_output=True, text=True, timeout=20)
        return res.stdout.strip() if res.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def git_is_ancestor(ancestor: str, descendant: str) -> bool:
    """ancestor 커밋이 descendant 커밋의 조상(또는 같은 커밋)인가 (git merge-base --is-ancestor)."""
    try:
        res = subprocess.run(["git", "merge-base", "--is-ancestor", ancestor, descendant], cwd=str(ROOT),
                             capture_output=True, text=True, timeout=20)
        return res.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def freeze_commit_state(path: str | os.PathLike[str]) -> dict[str, Any]:
    """동결 파일이 git에 커밋됐고 그 뒤 바뀌지 않았는지: {tracked, clean, commit, commit_time, path}."""
    try:
        rel = Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return {"tracked": False, "clean": False, "commit": None, "commit_time": None, "reason": "저장소 밖 파일"}
    tracked = git("ls-files", "--error-unmatch", rel) is not None
    status = git("status", "--porcelain", "--", rel)
    commit = git("log", "-1", "--format=%H", "--", rel) if tracked else None
    commit_time = git("log", "-1", "--format=%cI", commit) if commit else None
    return {"tracked": tracked, "clean": tracked and status == "", "commit": commit or None,
            "commit_time": commit_time or None, "path": rel}


def sealed_ok(freeze: Any, state: dict[str, Any]) -> list[str]:
    """봉인 자료를 처리하기 전 조건: 병실 동결 파일 형식, status frozen, 합성 아님, git에 커밋됨, 커밋 뒤 수정 없음. 문제 목록."""
    if not isinstance(freeze, dict) or freeze.get("schema") != FREEZE_SCHEMA or freeze.get("phrase_set") != "ward":
        return [f"병실 명령 동결 파일({FREEZE_SCHEMA}, phrase_set ward)이 아닙니다"]
    problems = []
    if freeze.get("status") != "frozen":
        problems.append("동결 파일 status가 frozen이 아닙니다(초안)")
    if freeze.get("synthetic"):
        problems.append("합성(가짜) 자료로 만든 동결 파일입니다")
    if not state.get("tracked") or not state.get("commit"):
        problems.append("동결 파일이 git에 커밋되지 않았습니다")
    elif not state.get("clean"):
        problems.append("동결 파일이 커밋 뒤 수정됐습니다")
    return problems


def freeze_gate(freeze_path: str | os.PathLike[str] | None, allow_sealed: bool) -> dict[str, Any] | None:
    """--allow-sealed를 받기 전 확인: --freeze가 git에 커밋된 status frozen 동결 파일이어야 한다.

    봉인 자료를 읽거나 서버·모델을 부르기 전에 부른다. --freeze 없이 --allow-sealed만 주면 거부한다.
    --freeze만 주고 --allow-sealed를 주지 않으면 정보만 돌려준다(초안도 됨, 봉인 자료는 여전히 거부된다).

    Returns:
        {path, sha256, status, commit, commit_time, sealed_ok, full_list} 또는 None(동결 파일 없음)
    Raises:
        SealedDataError: --allow-sealed인데 --freeze가 없거나 조건이 맞지 않을 때, 동결 파일을 읽지 못할 때
    """
    if not freeze_path:
        if allow_sealed:
            raise SealedDataError("--allow-sealed에는 --freeze <커밋된 동결 파일>이 필요합니다(기획 10.1)")
        return None
    path = Path(freeze_path).resolve()
    try:
        freeze = read_json(path)
    except ValueError as exc:
        raise SealedDataError(f"동결 파일을 읽지 못했습니다: {exc}") from exc
    state = freeze_commit_state(path)
    problems = sealed_ok(freeze, state)
    if allow_sealed and problems:
        raise SealedDataError("봉인 자료 처리 조건이 맞지 않습니다: " + "; ".join(problems))
    fz = freeze if isinstance(freeze, dict) else {}
    return {"path": str(path), "sha256": file_sha256(path),
            "status": fz.get("status"),
            "commit": state.get("commit"), "commit_time": state.get("commit_time"), "sealed_ok": not problems,
            "full_list": fz.get("full_list"),
            "decision_rule": fz.get("decision_rule") if isinstance(fz.get("decision_rule"), dict) else None,
            "enrollment": fz.get("enrollment") if isinstance(fz.get("enrollment"), dict) else None}


# ---------------------------------------------------------------------------
# 서버 기준값 (phrase.rule·/health thresholds)과 동결 값
# ---------------------------------------------------------------------------

def is_template(row: dict[str, Any], template: str) -> bool:
    """등록 템플릿 클립인가: 병실 묶음의 서 있는 자세 명령 중 소리 내어 1회(voiced), 소리 없이 1회(silent1), 또는 둘 다(both).

    ward_enroll.py(오프라인 등록)와 ward_service_batch.py --enroll-from voiced(서버 등록)가 같은 규칙으로 등록 영상을 고른다.

    Raises:
        ValueError: 알 수 없는 템플릿 방식
    """
    if template not in ENROLL_TEMPLATES:
        raise ValueError(f"알 수 없는 템플릿 방식: {template}")
    try:
        if not (isinstance(row.get("session"), str) and row["session"].startswith("ward_")) or row.get("kind") != "fixed":
            return False
        if (row.get("posture") or "upright") != "upright" or row.get("rep") != 1:
            return False
        cond = row.get("condition")
    except AttributeError as exc:
        raise ValueError(f"클립 기록 형식 오류: {row!r}") from exc
    return (template in ("voiced", "both") and cond == "voiced") or (template in ("silent1", "both") and cond == "silent")


def valid_profile_id(value: Any) -> bool:
    """서버 등록 ID 규칙(영문·숫자·_·- 1~32자, Windows 장치 이름 제외)에 맞는가 (서버 is_valid_profile_id와 같다)."""
    try:
        return (isinstance(value, str) and PROFILE_ID_RE.fullmatch(value) is not None
                and value.upper() not in WINDOWS_RESERVED_NAMES)
    except TypeError:
        return False


def _num_or_none(value: Any) -> float | None | str:
    """숫자면 float, None이면 None, 그 밖은 '?'(형식 오류 표시)."""
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return "?"


def expected_server_rule(freeze: dict[str, Any] | None) -> dict[str, float | None] | None:
    """동결 파일(또는 freeze_gate 결과) → 서버가 써야 할 병실 기준값 {min_margin, min_score, possible_min_score,
    enroll_max_distance}. 동결 파일이 없거나 full_list가 없으면 None.

    긴급 가능성 하한은 decision_rule.possible_min_score, 등록 거리는 enrollment.max_distance다(없으면 None = 쓰지 않음).
    """
    try:
        if not isinstance(freeze, dict) or not isinstance(freeze.get("full_list"), dict):
            return None
        full = freeze["full_list"]
        rule = freeze.get("decision_rule") if isinstance(freeze.get("decision_rule"), dict) else {}
        enr = freeze.get("enrollment") if isinstance(freeze.get("enrollment"), dict) else {}
        out = {"min_margin": _num_or_none(full.get("min_margin")), "min_score": _num_or_none(full.get("min_score")),
               "possible_min_score": _num_or_none(rule.get("possible_min_score")),
               "enroll_max_distance": _num_or_none(enr.get("max_distance"))}
    except (AttributeError, TypeError) as exc:
        raise ValueError(f"동결 파일 기준값 형식 오류: {exc}") from exc
    if any(v == "?" for v in out.values()):
        raise ValueError(f"동결 파일 기준값이 숫자가 아닙니다: {out}")
    return out  # type: ignore[return-value]


def rule_values_problems(server: Any, expected: dict[str, float | None] | None,
                         keys: Iterable[str] = SERVER_RULE_KEYS) -> list[str]:
    """서버가 쓴 기준값(phrase.rule 또는 /health thresholds)이 기대 값과 다른 항목 (문제 목록, 없으면 빈 목록).

    expected가 None이면 확인하지 않는다. 서버 값이 없으면(옛 서버): 기대 값 중 하한·등록 거리가 None이 아니면 문제다
    (옛 서버는 하한·등록을 쓰지 않으므로 글자 규칙만 같으면 같은 판정이다).
    """
    if expected is None:
        return []
    keys = tuple(keys)
    if not isinstance(server, dict):
        missing = [k for k in ("possible_min_score", "enroll_max_distance") if k in keys and expected.get(k) is not None]
        return [f"서버가 기준값(rule)을 알려 주지 않습니다(옛 서버). 동결 값 {', '.join(f'{k}={expected[k]}' for k in missing)}"
                "을 쓰는지 알 수 없습니다"] if missing else []
    problems = []
    for k in keys:
        want, got = expected.get(k), _num_or_none(server.get(k))
        if got == "?":
            problems.append(f"{k}: 서버 값이 숫자가 아닙니다({server.get(k)!r})")
        elif (want is None) != (got is None) or (want is not None and abs(float(want) - float(got)) > RULE_TOL):
            problems.append(f"{k}: 서버 {got} ≠ 동결 {want}")
    return problems


def env_value(value: float | None) -> str:
    """기준값 → 환경 변수 문자열 (None이면 'none')."""
    if value is None:
        return ENV_NONE
    try:
        return repr(float(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"환경 변수 값으로 바꿀 수 없습니다: {value!r}") from exc


def parse_env_value(text: Any) -> float | None:
    """환경 변수 문자열 → 기준값 (''·none은 None).

    Raises:
        ValueError: 숫자가 아닐 때
    """
    s = str(text if text is not None else "").strip()
    if s == "" or s.lower() == ENV_NONE:
        return None
    try:
        return float(s)
    except ValueError as exc:
        raise ValueError(f"환경 변수 값이 숫자가 아닙니다: {text!r}") from exc


def server_env(expected: dict[str, float | None]) -> dict[str, str]:
    """기준값 4개 → 서버 환경 변수 {LIPNET_OLKAVS_WARD_*: 문자열} (동결 파일 full_list.env)."""
    try:
        return {SERVER_RULE_ENV[k]: env_value(expected.get(k)) for k in SERVER_RULE_KEYS}
    except (AttributeError, KeyError) as exc:
        raise ValueError(f"기준값 형식 오류: {exc}") from exc


def parse_time(value: Any) -> datetime | None:
    """ISO 8601 시각 문자열 → 시간대가 있는 datetime (없거나 형식이 틀리면 None)."""
    try:
        t = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    return t if t.tzinfo is not None else None


def order_problems(heads: Iterable[dict[str, Any]], freeze_sha: str, freeze_commit: str | None,
                   commit_time: str | None, is_ancestor: Callable[[str, str], bool] | None = None) -> list[str]:
    """봉인 자료 결과 파일이 이 동결 파일 뒤에 만들어졌는지 (문제 목록, 없으면 빈 목록).

    각 머리(결과 행·등록 파일)에 이 동결 파일의 SHA-256(freeze.sha256)이 있어야 하고, 실행 때 git HEAD(git_head)가
    동결 커밋과 같거나 그 뒤여야 하며, 만든 시각(created)이 동결 커밋 시각보다 뒤여야 한다.

    Args:
        heads: [{path, freeze, git_head, created}]
        is_ancestor: (조상, 자손) → bool. 기본은 git merge-base --is-ancestor (검사에서 바꿔 넣는다)
    """
    is_ancestor = is_ancestor or git_is_ancestor
    problems: list[str] = []
    ct = parse_time(commit_time)
    for h in heads:
        name = h.get("path")
        fz = h.get("freeze") if isinstance(h.get("freeze"), dict) else {}
        if fz.get("sha256") != freeze_sha:
            problems.append(f"{name}: 이 동결 파일로 실행한 기록이 없습니다(freeze.sha256 {fz.get('sha256')!r})")
        head = h.get("git_head")
        if not freeze_commit or not head or not is_ancestor(freeze_commit, head):
            problems.append(f"{name}: 실행 때 git HEAD({head})가 동결 커밋({freeze_commit})과 같거나 그 뒤가 아닙니다")
        created = parse_time(h.get("created"))
        if ct is None or created is None or not created > ct:
            problems.append(f"{name}: 만든 시각({h.get('created')})이 동결 커밋 시각({commit_time})보다 뒤가 아닙니다")
    return problems


def file_sha256(path: str | os.PathLike[str]) -> str:
    """파일 SHA-256 (1MiB 블록).

    Raises:
        OSError: 파일을 읽을 수 없을 때
    """
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
    except OSError as exc:
        raise OSError(f"해시를 계산할 수 없습니다: {path}: {exc}") from exc
    return digest.hexdigest()


def listing_text(entries: Iterable[tuple[str, int, str]]) -> str:
    """(파일명, 크기, SHA-256) 목록 → 지문용 글 (이름순 정렬, '\\n'으로 잇고 끝 줄바꿈 없음)."""
    try:
        return "\n".join(f"{name}\t{int(size)}\t{sha}" for name, size, sha in sorted(entries, key=lambda e: e[0]))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"목록 형식 오류: {exc}") from exc


def folder_fingerprint(folder: str | os.PathLike[str]) -> dict[str, Any]:
    """폴더 안 일반 파일 전체(하위 폴더 제외)의 목록 지문 (기획 10.1 봉인 절차·8단계 문서 2절과 같은 성격).

    Returns:
        {path, files, videos, bytes, listing_sha256, format}
    """
    try:
        base = Path(folder)
        entries = [(p.name, p.stat().st_size, file_sha256(p)) for p in sorted(base.iterdir(), key=lambda p: p.name)
                   if p.is_file()]
    except OSError as exc:
        raise OSError(f"폴더 지문을 계산할 수 없습니다: {folder}: {exc}") from exc
    text = listing_text(entries)
    return {"path": str(folder), "files": len(entries),
            "videos": sum(1 for n, _s, _h in entries if n.lower().endswith(VIDEO_SUFFIXES) and not n.startswith("._")),
            "bytes": sum(s for _n, s, _h in entries), "listing_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "format": LISTING_FORMAT}


def atomic_write_json(path: str | os.PathLike[str], obj: Any) -> None:
    """JSON을 임시 파일에 쓴 뒤 바꿔 넣는다 (중간에 끊겨도 이전 파일이 깨지지 않는다)."""
    target = Path(path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=str(target.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(obj, handle, ensure_ascii=False, indent=1)
            os.replace(tmp, target)
        except BaseException:
            try:
                os.remove(tmp)
            except OSError:
                pass
            raise
    except (OSError, TypeError, ValueError) as exc:
        raise OSError(f"JSON을 쓰지 못했습니다: {target}: {exc}") from exc


def read_json(path: str | os.PathLike[str]) -> Any:
    """UTF-8 JSON 파일을 읽는다.

    Raises:
        ValueError: 읽을 수 없거나 JSON이 아닐 때
    """
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"JSON을 읽지 못했습니다: {path}: {exc}") from exc


def read_rows(path: str | os.PathLike[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """서비스 일괄 실행 결과 파일 → (머리 정보, 행 목록).

    ward_service_batch.py가 쓰는 {schema, ..., rows:[...]} 형식과, 예전 세션 스크립트가 쓴 행 목록(list) 형식을 모두 받는다.
    """
    data = read_json(path)
    if isinstance(data, list):
        return {"schema": "legacy-row-list", "path": str(path)}, data
    if isinstance(data, dict) and isinstance(data.get("rows"), list):
        head = {k: v for k, v in data.items() if k != "rows"}
        head["path"] = str(path)
        return head, data["rows"]
    raise ValueError(f"결과 행 파일 형식이 아닙니다: {path}")
