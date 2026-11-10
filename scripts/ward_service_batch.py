"""촬영 폴더의 클립을 실행 중인 서버(POST /api/record)로 하나씩 보내 결과 행을 모은다 (병실 기획 9.5).

세션 임시 스크립트(eval_service_phrases.py·eval_service_p02.py)를 옮겨 일반화한 것이다. 서비스 코드·기준값은 바꾸지 않는다.
  - 한 번에 한 클립: 앞 클립의 최종 상태를 받은 뒤 다음 클립을 보낸다. 매 클립 뒤 출력 파일을 다시 쓴다(중간에 끊겨도 안전).
  - 이어 하기: 출력 파일이 있으면 status가 done인 행은 그대로 두고 나머지만 다시 보낸다(--redo면 모두 다시).
  - --phrase-set ward|sentences를 폼 필드 phrase_set으로 보낸다. 결과 phrase.set이 다르면(묶음을 모르는 서버) 멈춘다.
    이어 하기에서도 phrase.set이 다른 행은 다시 보낸다.
  - 봉인 자료(처음 보는 사람 P02, 병실 명령 평가용 ward_eval)는 --allow-sealed --freeze <동결 파일>이 있어야 보낸다.
    동결 파일이 git에 커밋된 status frozen 파일인지 클립 목록을 읽기 전에 확인하고, 그 SHA-256·커밋을 출력 머리에 남긴다.
  - --freeze를 주면(병실 묶음) 클립마다 서버 accepted가 동결 기준값(full_list)으로 다시 판정한 값과 같은지, 서버가 결과에 적은
    기준값(phrase.rule: M·S·긴급 가능성 하한·등록 거리 D)이 동결 값과 같은지, 서버 등록 판정(enroll.within·accepted)이
    동결 D로 다시 판정한 값과 같은지 보고, 다르면 서버가 동결 기준값으로 켜지지 않은 것이므로 멈춘다.
    /health가 기준값을 알려 주면 클립을 보내기 전에 먼저 비교한다.
  - 시작·끝의 서버 /health(워커 상태, 체크포인트 SHA, ward_phrases 등)를 출력 파일 머리에 남긴다.

내 입 모양 등록(서버 등록, 앱이 실제로 쓰는 경로)으로 채점하기 (phrase_set ward만):
  --enroll-profile ID --enroll-from voiced : 먼저 소리 내어 1회 찍은 명령 영상(병실 묶음, 서 있는 자세, 1회차)을 모두
        enroll_command=<명령 ID>로 보내 등록(store)하고, 그다음 나머지 클립을 등록 거리와 함께 채점(match)한다.
        새 출력 파일이면 서버에 그 ID의 등록이 없어야 한다. 이어 하기는 서버 등록이 이 실행이 저장한 것과 같을 때만 한다
        (다른 실행이 만들거나 바꾼 등록 위에 다시 저장하지 않는다). --redo는 그 ID의 등록을 지우고 처음부터 한다.
  --enroll-profile ID --enroll-match-only : 이미 완성된(20/20, ok) 등록으로 모든 클립을 채점만 한다(목록 밖 문장 폴더 등).
  --enroll-delete-after : 모든 클립이 done이면 끝에 서버 등록을 지운다(끝나지 않았으면 이어 하기를 위해 남긴다).
  등록 상태(서버 GET /api/ward/profiles/ID)를 실행 전·뒤에 머리 enroll에 남기고, match 행마다 채점 때의 등록 지문
  (profile_sha256)을 남긴다. 이어 하기에서 등록 지문이 바뀐 match 행은 다시 보낸다.

행 형식: {file, participant, kind, condition, sentence_id, rep, posture, session, phrase_set, status, results, model,
         seconds(업로드 직전~최종 상태 벽시계 초), post_seconds, duration_sec(서버가 잰 녹화 길이), duration_ms(촬영 기록),
         error, spoken, (등록 때) enroll_request{mode, profile, command}, profile_sha256(match), template_created(store)}

종료 코드: 0 모두 done, 1 done이 아닌 클립이 남음, 2 시작하지 않음(인자·봉인·파일), 3 서버 준비·기준값 문제,
          4 서버가 요청한 묶음·등록을 처리하지 않음, 5 서버 기준값·등록 판정이 동결 값과 다름, 6 서버 등록 상태 문제.

사용 예 (서버는 backend/에서 py -3.11 -m uvicorn main:app --host 127.0.0.1 --port 8001):
  py -3.11 scripts/ward_service_batch.py --clips "<P01 문장 촬영 폴더>" \
      --out experiments/ward_negatives_20261110/P01_sentences_as_ward.json --server http://127.0.0.1:8001 --phrase-set ward
  py -3.11 scripts/ward_service_batch.py --clips "<P01 ward_tune 폴더>" --out experiments/ward_tune_20261111/P01_ward_tune.json \
      --enroll-profile P01 --enroll-from voiced
  (P02, 동결 뒤) ... --clips "<P02 ward_eval 폴더>" --allow-sealed --freeze docs/병실명령_동결_2026-11-11.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import pilot_evaluate as pe  # noqa: E402
from ward_tools import common  # noqa: E402

SCHEMA = "ward-service-rows-v1"
PHRASE_SETS = ("ward", "sentences")
TERMINAL = ("done", "failed", "error", "timeout")
POLL_SECONDS = 0.1
CLIENT_TIMEOUT_SECONDS = 180.0
UPLOAD_TIMEOUT_SECONDS = 60.0
EXPECTED_CHECKPOINT = "f41d742380af11b72f9cafe178cd59d0a9016c02f5a976145f5f7595d44326a5"
PROFILE_API = "/api/ward/profiles"
ENROLL_FROM = (common.SERVICE_TEMPLATE,)
MODE_FROM = "from_voiced"
MODE_MATCH_ONLY = "match_only"


def multipart_body(video: Path, phrase_set: str, boundary: str | None = None,
                   fields: dict[str, str] | None = None) -> tuple[bytes, str]:
    """POST /api/record 멀티파트 본문 (file + phrase_set + 추가 폼 필드). 파일 이름은 서버가 믿지 않으므로 고정 이름을 쓴다.

    Args:
        fields: 추가 폼 필드 (예: {"enroll_profile": "P01", "enroll_command": "W05"})
    Returns:
        (본문 바이트, Content-Type 헤더 값)
    """
    try:
        data = video.read_bytes()
    except OSError as exc:
        raise OSError(f"영상을 읽지 못했습니다: {video}: {exc}") from exc
    boundary = boundary or f"----wardbatch{uuid.uuid4().hex}"
    suffix = video.suffix.lower().lstrip(".") or "webm"
    mime = "video/mp4" if suffix == "mp4" else "video/webm"
    parts = [f"--{boundary}\r\nContent-Disposition: form-data; name=\"phrase_set\"\r\n\r\n{phrase_set}\r\n".encode("utf-8")]
    for name, value in (fields or {}).items():
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode("utf-8"))
    parts += [
        (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"recording.{suffix}\"\r\n"
         f"Content-Type: {mime}\r\n\r\n").encode("utf-8"),
        data, b"\r\n", f"--{boundary}--\r\n".encode("utf-8"),
    ]
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def http_json(url: str, data: bytes | None = None, content_type: str | None = None,
              timeout: float = 10.0, method: str | None = None) -> dict[str, Any]:
    """GET(또는 data가 있으면 POST, method를 주면 그 방식) → JSON dict. HTTP 오류는 본문 앞부분을 붙인 RuntimeError."""
    request = urllib.request.Request(url, data=data, method=method or ("POST" if data is not None else "GET"))
    if content_type:
        request.add_header("Content-Type", content_type)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")[:300]
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise RuntimeError(f"{type(exc).__name__}: {exc}") from exc


def health_problems(health: dict[str, Any], phrase_set: str, commands: dict[str, Any] | None,
                    enroll: bool = False) -> list[str]:
    """서버 /health가 이 일괄 실행에 맞는지 (문제 목록, 없으면 빈 목록).

    워커 두 개가 ready이고 체크포인트 SHA가 서비스 고정값이어야 한다. ward면 ward_phrases의 개수·버전이 명령 목록과 같아야 한다.
    enroll이면 ward_phrases.enroll(등록 기능)이 켜져 있고 등록 폴더에 쓸 수 있어야 한다.
    """
    problems: list[str] = []
    try:
        model = health.get("model") or {}
        if health.get("status") != "ok":
            problems.append(f"서버 상태가 ok가 아닙니다: {health.get('status')!r}")
        if health.get("model_backend") != "olkavs":
            problems.append(f"공식 모델(olkavs) 서버가 아닙니다: {health.get('model_backend')!r}")
        for key in ("roi_worker", "infer_worker"):
            if model.get(key) != "ready":
                problems.append(f"{key}가 준비되지 않았습니다: {model.get(key)!r}")
        if model.get("checkpoint_sha256") != EXPECTED_CHECKPOINT:
            problems.append(f"체크포인트 SHA가 다릅니다(또는 인식 워커 기동 전): {model.get('checkpoint_sha256')!r}")
        if phrase_set == "ward":
            ward = model.get("ward_phrases") or {}
            want = len(commands["ids"]) if commands else None
            if not ward.get("enabled") or ward.get("count") != want:
                problems.append(f"서버의 병실 명령 목록이 {want}개가 아닙니다: {ward}")
            if commands and ward.get("version") != commands.get("version"):
                problems.append(f"서버의 병실 명령 목록 버전이 다릅니다: {ward.get('version')!r} != {commands.get('version')!r}")
            if enroll:
                en = ward.get("enroll") if isinstance(ward.get("enroll"), dict) else {}
                if en.get("enabled") is not True:
                    problems.append(f"서버의 내 입 모양 등록 기능이 켜져 있지 않습니다(옛 서버일 수 있음): {en or None}")
                elif en.get("profile_dir_writable") is not True:
                    problems.append("서버 등록 폴더(LIPNET_WARD_PROFILE_DIR)에 쓸 수 없습니다")
    except (AttributeError, TypeError, KeyError) as exc:
        problems.append(f"/health 형식 오류: {exc}")
    return problems


def thresholds_from_health(health: dict[str, Any], phrase_set: str) -> dict[str, Any] | None:
    """/health가 묶음별 기준값을 알려 주면 그 값, 아니면 None.

    ward는 model.ward_phrases.thresholds({min_margin, min_score, possible_min_score, enroll_max_distance})를 먼저 본다.
    """
    try:
        model = health.get("model") or {}
        if phrase_set == "ward":
            ward = model.get("ward_phrases") or {}
            if isinstance(ward.get("thresholds"), dict):
                return ward["thresholds"]
        for key in ("phrase_sets", "thresholds"):
            block = model.get(key) or health.get(key)
            if isinstance(block, dict) and isinstance(block.get(phrase_set), dict):
                return block[phrase_set]
    except AttributeError:
        return None
    return None


def wait_ready(server: str, phrase_set: str, commands: dict[str, Any] | None, wait_seconds: float,
               fetch: Callable[[str], dict[str, Any]] | None = None,
               sleep: Callable[[float], None] = time.sleep, enroll: bool = False) -> tuple[dict[str, Any], list[str]]:
    """워커가 준비될 때까지 /health를 기다린다. (마지막 health, 남은 문제 목록)."""
    fetch = fetch or (lambda url: http_json(url))
    deadline = time.monotonic() + max(0.0, wait_seconds)
    health: dict[str, Any] = {}
    problems: list[str] = ["/health를 받지 못했습니다"]
    while True:
        try:
            health = fetch(f"{server}/health")
            problems = health_problems(health, phrase_set, commands, enroll)
        except RuntimeError as exc:
            problems = [f"/health 요청 실패: {exc}"]
        if not problems or time.monotonic() >= deadline:
            return health, problems
        sleep(2.0)


def run_clip(server: str, video: Path, phrase_set: str, fields: dict[str, str] | None = None) -> dict[str, Any]:
    """클립 하나를 보내고 최종 상태까지 폴링한다.

    Returns:
        {status, results, model, error, warnings, seconds, post_seconds, duration_sec}
    """
    t0 = time.perf_counter()
    body, ctype = multipart_body(video, phrase_set, fields=fields)
    start = http_json(f"{server}/api/record", data=body, content_type=ctype, timeout=UPLOAD_TIMEOUT_SECONDS)
    post_seconds = time.perf_counter() - t0
    sid = start.get("session_id")
    if not sid:
        raise RuntimeError(f"업로드 응답에 session_id가 없습니다: {str(start)[:200]}")
    result: dict[str, Any] = {"status": "client_timeout"}
    deadline = t0 + CLIENT_TIMEOUT_SECONDS
    while time.perf_counter() < deadline:
        result = http_json(f"{server}/api/result/{sid}")
        if result.get("status") in TERMINAL:
            break
        time.sleep(POLL_SECONDS)
    return {"status": result.get("status"), "results": result.get("results"), "model": result.get("model"),
            "error": result.get("error_message"), "warnings": result.get("warnings"),
            "seconds": round(time.perf_counter() - t0, 3), "post_seconds": round(post_seconds, 3),
            "duration_sec": start.get("duration_sec")}


def row_base(clip: dict[str, Any], phrase_set: str) -> dict[str, Any]:
    """pilot_evaluate.load_clips의 클립 기록 → 결과 행의 클립 정보 부분."""
    return {"file": clip.get("file"), "participant": clip.get("participant"), "kind": clip.get("kind"),
            "condition": clip.get("condition"), "sentence_id": clip.get("sentence_id"), "rep": clip.get("rep"),
            "posture": clip.get("posture"), "session": clip.get("session"), "phrase_set": phrase_set,
            "duration_ms": clip.get("duration_ms"), "spoken": clip.get("spoken")}


def phrase_set_mismatch(results: list[dict[str, Any]] | None, phrase_set: str) -> list[Any]:
    """결과 창의 phrase.set 가운데 요청한 묶음과 다른 값 목록 (phrase가 없는 창은 보지 않는다)."""
    try:
        return [w["phrase"].get("set") for w in (results or [])
                if isinstance(w, dict) and isinstance(w.get("phrase"), dict) and w["phrase"].get("set") != phrase_set]
    except (AttributeError, TypeError):
        return ["형식 오류"]


def kept_rows(existing: list[dict[str, Any]], redo: bool, phrase_set: str | None = None,
              want: dict[str, dict[str, Any] | None] | None = None) -> dict[str, dict[str, Any]]:
    """이어 하기에서 그대로 둘 행 (파일 이름 → 행). status가 done이고 phrase.set이 요청 묶음과 같은 행만, --redo면 없음.

    want(파일 이름 → 등록 요청 {mode, profile, command} 또는 None)를 주면 행의 enroll_request가 같은 행만 둔다
    (등록 없이 만든 행을 등록 실행에 섞지 않는다).
    """
    if redo:
        return {}
    return {r["file"]: r for r in existing
            if isinstance(r, dict) and r.get("status") == "done" and r.get("file")
            and not (phrase_set and phrase_set_mismatch(r.get("results"), phrase_set))
            and (want is None or r.get("enroll_request") == want.get(r["file"]))}


def threshold_mismatch(results: list[dict[str, Any]] | None, full_list: dict[str, Any] | None) -> list[dict[str, Any]]:
    """서버 phrase.accepted가 동결 기준값(1·2등 차이 ≥ M 그리고 1등 점수 ≥ S)으로 다시 판정한 값과 다른 창 목록."""
    try:
        if not full_list:
            return []
        M, S = float(full_list["min_margin"]), float(full_list["min_score"])
        out = []
        for w in results or []:
            ph = w.get("phrase") if isinstance(w, dict) else None
            if not isinstance(ph, dict) or not isinstance(ph.get("margin"), (int, float)) or \
                    not isinstance(ph.get("score"), (int, float)):
                continue
            want = float(ph["margin"]) >= M and float(ph["score"]) >= S
            if bool(ph.get("accepted")) != want:
                out.append({"id": ph.get("id"), "margin": ph["margin"], "score": ph["score"], "accepted": ph.get("accepted"),
                            "expected": want})
        return out
    except (KeyError, TypeError, ValueError) as exc:
        return [{"problem": f"형식 오류: {exc}"}]


def rule_mismatch(results: list[dict[str, Any]] | None, expected: dict[str, float | None] | None) -> list[str]:
    """병실 phrase마다 서버가 적은 기준값(phrase.rule)이 동결 값과 다른 항목 (문제 목록). expected가 None이면 확인 안 함."""
    if expected is None:
        return []
    out: list[str] = []
    try:
        for w in results or []:
            ph = w.get("phrase") if isinstance(w, dict) else None
            if isinstance(ph, dict) and ph.get("set") == "ward":
                out += [f"{ph.get('id')}: {p}" for p in common.rule_values_problems(ph.get("rule"), expected)]
    except (AttributeError, TypeError) as exc:
        out.append(f"형식 오류: {exc}")
    return out


def result_enroll(results: list[dict[str, Any]] | None) -> dict[str, Any] | None:
    """결과의 등록 정보(results[0].enroll, 첫 창에만 붙는다). 없으면 None."""
    if isinstance(results, list) and results and isinstance(results[0], dict) and isinstance(results[0].get("enroll"), dict):
        return results[0]["enroll"]
    return None


def enroll_mismatch(results: list[dict[str, Any]] | None, request: dict[str, Any] | None,
                    expected: dict[str, float | None] | None = None) -> list[str]:
    """등록 요청을 서버가 그대로 처리했는지, (expected를 주면) match 판정이 동결 D로 다시 판정한 값과 같은지 (문제 목록).

    요청이 없으면 결과에 enroll이 없어야 한다. store는 mode·profile·command, match는 mode·profile을 본다.
    match 다시 판정: within = status ok이고 D가 있고 distance ≤ D, accepted = within이고 top_id = 첫 창 phrase.id.
    """
    enr = result_enroll(results)
    if request is None:
        return [] if enr is None else ["등록을 요청하지 않았는데 결과에 enroll이 있습니다"]
    if enr is None:
        return ["서버가 등록 요청을 처리하지 않았습니다(결과에 enroll이 없음, 옛 서버일 수 있음)"]
    problems = []
    if enr.get("mode") != request.get("mode") or enr.get("profile") != request.get("profile"):
        problems.append(f"요청 {request.get('mode')}/{request.get('profile')}과 결과 {enr.get('mode')}/{enr.get('profile')}가 다릅니다")
    if request.get("mode") == "store" and enr.get("command") != request.get("command"):
        problems.append(f"등록 명령이 다릅니다: 요청 {request.get('command')} 결과 {enr.get('command')}")
    if expected is None or request.get("mode") != "match" or problems:
        return problems
    D = expected.get("enroll_max_distance")
    md = enr.get("max_distance")
    if (D is None) != (md is None) or (D is not None and abs(float(md) - float(D)) > common.RULE_TOL):
        problems.append(f"서버 등록 거리 기준 {md} ≠ 동결 D {D}")
    dist = enr.get("distance")
    ok = enr.get("status") == "ok" and isinstance(dist, (int, float)) and not isinstance(dist, bool)
    within = bool(ok and D is not None and float(dist) <= float(D))
    first = results[0] if results else {}
    phrase_id = (first.get("phrase") or {}).get("id") if isinstance(first.get("phrase"), dict) else None
    accepted = within and enr.get("top_id") == phrase_id and phrase_id is not None
    if bool(enr.get("within")) != within or bool(enr.get("accepted")) != accepted:
        problems.append(f"서버 등록 판정 within={enr.get('within')} accepted={enr.get('accepted')}이(가) 동결 D로 다시 판정한 "
                        f"within={within} accepted={accepted}과 다릅니다")
    return problems


# ---------------------------------------------------------------------------
# 서버 등록 (GET·DELETE /api/ward/profiles/ID)
# ---------------------------------------------------------------------------

def profile_url(server: str, profile: str) -> str:
    """등록 하나의 API 주소."""
    return f"{server}{PROFILE_API}/{urllib.parse.quote(profile, safe='')}"


def get_profile(server: str, profile: str, fetch: Callable[..., dict[str, Any]] | None = None) -> dict[str, Any]:
    """서버 등록 상태. 없으면(HTTP 404) {status: 'missing', count: 0, templates: {}}.

    Raises:
        RuntimeError: 그 밖의 요청 실패
    """
    fetch = fetch or http_json
    try:
        data = fetch(profile_url(server, profile))
    except RuntimeError as exc:
        if str(exc).startswith("HTTP 404"):
            return {"profile": profile, "status": "missing", "count": 0, "complete": False, "templates": {}}
        raise
    if not isinstance(data, dict):
        raise RuntimeError(f"등록 상태 응답 형식 오류: {str(data)[:200]}")
    return data


def delete_profile(server: str, profile: str, fetch: Callable[..., dict[str, Any]] | None = None) -> bool:
    """서버 등록을 지운다. 지웠거나 원래 없으면 True.

    Raises:
        RuntimeError: 요청 실패(404 제외)
    """
    fetch = fetch or http_json
    try:
        res = fetch(profile_url(server, profile), method="DELETE")
    except RuntimeError as exc:
        if str(exc).startswith("HTTP 404"):
            return True
        raise
    return bool(isinstance(res, dict) and res.get("deleted"))


def profile_snapshot(profile: dict[str, Any]) -> dict[str, Any]:
    """서버 등록 상태 → 머리에 남길 요약과 지문 {status, count, complete, templates{명령: {created, text_top,
    text_top_matches}}, sha256}. sha256은 status·count·templates의 정렬 JSON의 SHA-256(등록이 바뀌면 달라진다)."""
    try:
        temps = profile.get("templates") if isinstance(profile.get("templates"), dict) else {}
        templates = {str(cid): {k: (t or {}).get(k) for k in ("created", "text_top", "text_top_matches")}
                     for cid, t in sorted(temps.items())}
        core = {"status": profile.get("status"), "count": profile.get("count"), "templates": templates}
        sha = hashlib.sha256(json.dumps(core, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"등록 상태 형식 오류: {exc}") from exc
    return {**core, "complete": profile.get("complete"), "sha256": sha}


def brief(snapshot: dict[str, Any] | None) -> dict[str, Any] | None:
    """등록 요약 한 줄용 {status, count, complete, sha256}."""
    if not snapshot:
        return None
    return {k: snapshot.get(k) for k in ("status", "count", "complete", "sha256")}


def enroll_plan(clips: list[dict[str, Any]], commands: dict[str, Any], profile: str | None, mode: str | None
                ) -> tuple[dict[str, dict[str, Any] | None], list[str]]:
    """클립마다 보낼 등록 요청과 등록(store) 순서.

    from_voiced: 등록 영상(common.is_template(voiced): 병실 묶음, 서 있는 자세, 소리 내어 1회차 명령)은
    {mode: store, profile, command: 명령 ID}, 나머지는 {mode: match, profile}. match_only: 모두 match. 등록 없음: 모두 None.

    Raises:
        ValueError: 같은 명령의 등록 영상이 둘 이상이거나 명령 목록 밖 ID일 때
    """
    want: dict[str, dict[str, Any] | None] = {}
    store: list[str] = []
    seen: dict[str, str] = {}
    for c in clips:
        f = c["file"]
        if not profile or not mode:
            want[f] = None
            continue
        if mode == MODE_FROM and c.get("status") == "ok" and common.is_template(c, common.SERVICE_TEMPLATE):
            cid = str(c.get("sentence_id"))
            if cid not in commands["texts"]:
                raise ValueError(f"등록 영상의 명령 ID가 목록 밖입니다: {f} ({cid})")
            if cid in seen:
                raise ValueError(f"같은 명령의 등록 영상이 둘입니다: {seen[cid]}, {f} ({cid})")
            seen[cid] = f
            want[f] = {"mode": "store", "profile": profile, "command": cid}
            store.append(f)
        else:
            want[f] = {"mode": "match", "profile": profile}
    return want, store


def store_conflicts(snapshot: dict[str, Any], recorded: dict[str, Any], pending: set[str], resuming: bool) -> list[str]:
    """서버 등록이 이 실행이 저장한 것과 맞는지 (문제 목록). 다른 실행이 만든 등록 위에 다시 저장하지 않기 위해서다.

    새 실행(resuming False): 서버에 등록이 없어야 한다(템플릿 0개).
    이어 하기: 이 실행이 저장한 템플릿(recorded: 명령 → created)이 서버에 그대로 있어야 하고, 서버에만 있는 템플릿은
    이번에 다시 보낼 등록 영상(pending, 저장 직후 끊긴 경우)이어야 한다.
    """
    server = snapshot.get("templates") or {}
    if not resuming:
        return [f"서버에 등록 {len(server)}개가 이미 있습니다(다른 실행이 만든 등록). --redo로 지우고 다시 하거나 다른 ID를 쓰세요"] \
            if server else []
    problems = []
    for cid, created in sorted(recorded.items()):
        if cid not in server:
            problems.append(f"{cid}: 이 실행이 저장한 등록이 서버에 없습니다(지워졌거나 다른 실행이 다시 만듦)")
        elif server[cid].get("created") != created:
            problems.append(f"{cid}: 서버 등록 시각 {server[cid].get('created')}이(가) 이 실행이 저장한 {created}과 다릅니다")
    for cid in sorted(set(server) - set(recorded) - set(pending)):
        problems.append(f"{cid}: 이 실행이 저장하지 않은 등록이 서버에 있습니다")
    return problems


def git_head() -> str | None:
    """저장소 HEAD 커밋 (git이 없으면 None)."""
    return common.git("rev-parse", "HEAD")


def check_enroll_args(args: argparse.Namespace) -> str | None:
    """등록 인자 확인. 등록 방식(from_voiced·match_only) 또는 None.

    Raises:
        ValueError: 인자 조합이 틀렸을 때
    """
    profile = args.enroll_profile
    if profile is None:
        if args.enroll_from or args.enroll_match_only or args.enroll_delete_after:
            raise ValueError("--enroll-from·--enroll-match-only·--enroll-delete-after에는 --enroll-profile이 필요합니다")
        return None
    if not common.valid_profile_id(profile):
        raise ValueError(f"등록 ID는 영문·숫자·_·- 1~32자여야 합니다: {profile!r}")
    if args.phrase_set != "ward":
        raise ValueError("등록은 --phrase-set ward에서만 씁니다")
    if bool(args.enroll_from) == bool(args.enroll_match_only):
        raise ValueError("--enroll-profile에는 --enroll-from voiced 또는 --enroll-match-only 중 하나를 주세요")
    return MODE_FROM if args.enroll_from else MODE_MATCH_ONLY


def main(argv: list[str] | None = None) -> int:
    """명령줄 진입점."""
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", required=True, help="촬영 폴더 (영상 + 같은 이름의 .json)")
    ap.add_argument("--out", required=True, help="결과 행 JSON (있으면 이어 한다)")
    ap.add_argument("--server", default="http://127.0.0.1:8000")
    ap.add_argument("--phrase-set", choices=PHRASE_SETS, default="ward")
    ap.add_argument("--allow-sealed", action="store_true",
                    help="봉인 자료(P02·ward_eval)도 보냄 (--freeze에 커밋된 status frozen 동결 파일이 있어야 함)")
    ap.add_argument("--freeze", help="동결 파일. 봉인 자료에는 필수, 주면 서버 기준값·판정이 동결 값과 같은지 클립마다 확인")
    ap.add_argument("--redo", action="store_true", help="이미 done인 행도 다시 보냄 (--enroll-from이면 그 ID의 서버 등록을 지우고 처음부터)")
    ap.add_argument("--retries", type=int, default=1, help="실패한 클립을 바로 다시 보내는 횟수")
    ap.add_argument("--limit", type=int, default=0, help="이번에 보낼 최대 클립 수 (0 = 전부)")
    ap.add_argument("--expect-clips", type=int, default=0, help="폴더의 클립 수가 이 값이 아니면 멈춤 (0 = 확인 안 함)")
    ap.add_argument("--wait-ready", type=float, default=240.0, help="워커 준비를 기다리는 최대 초")
    ap.add_argument("--enroll-profile", help="서버 등록 ID (영문·숫자·_·- 1~32자, 보통 P번호)")
    ap.add_argument("--enroll-from", choices=ENROLL_FROM,
                    help="voiced: 소리 내어 1회 찍은 명령 영상으로 먼저 등록하고 나머지를 등록 거리와 함께 채점")
    ap.add_argument("--enroll-match-only", action="store_true", help="이미 완성된 등록으로 모든 클립을 채점만 함")
    ap.add_argument("--enroll-delete-after", action="store_true", help="모든 클립이 done이면 끝에 서버 등록을 지움")
    args = ap.parse_args(argv)
    server = args.server.rstrip("/")
    clips_dir = Path(args.clips).resolve()
    out = Path(args.out).resolve()
    try:
        # 봉인 확인은 클립 기록을 읽기 전에: --allow-sealed면 동결 파일부터, 아니면 경로부터 본다
        gate = common.freeze_gate(args.freeze, args.allow_sealed)
        mode = check_enroll_args(args)
        common.require_unsealed([clips_dir, out], [], args.allow_sealed, "서비스 일괄 실행")
        clips = pe.load_clips(clips_dir)
        common.require_unsealed([clips_dir], clips, args.allow_sealed, "서비스 일괄 실행")
        pe.check_rows(clips)  # 촬영 묶음에 맞는 목록과 문장이 같은지, 묶음이 섞이지 않았는지
        if args.expect_clips and len(clips) != args.expect_clips:
            raise ValueError(f"클립 {len(clips)}개, 기대 {args.expect_clips}개")
        commands = common.load_commands() if args.phrase_set == "ward" else None
        full_list = (gate or {}).get("full_list") if args.phrase_set == "ward" else None
        if gate is not None and args.phrase_set == "ward" and not isinstance(full_list, dict):
            raise ValueError("동결 파일에 full_list 기준값이 없습니다")
        expected = common.expected_server_rule(gate) if (gate is not None and args.phrase_set == "ward") else None
        want, store_files = enroll_plan(clips, commands or {"texts": {}}, args.enroll_profile, mode)
        if mode == MODE_FROM and not store_files:
            raise ValueError("등록 영상(병실 묶음, 서 있는 자세, 소리 내어 1회차 명령)이 폴더에 없습니다")
        head_prev, rows_prev = ({}, [])
        if out.exists():
            head_prev, rows_prev = common.read_rows(out)
            if head_prev.get("phrase_set") not in (None, args.phrase_set) or \
                    (head_prev.get("clips_dir") and Path(head_prev["clips_dir"]).resolve() != clips_dir):
                raise ValueError(f"기존 출력 파일의 묶음·폴더가 다릅니다: {head_prev.get('phrase_set')} {head_prev.get('clips_dir')}")
            prev_fz = (head_prev.get("freeze") or {}).get("sha256") if isinstance(head_prev.get("freeze"), dict) else None
            if prev_fz and gate and prev_fz != gate["sha256"]:
                raise ValueError("기존 출력 파일을 만든 동결 파일과 지금 동결 파일이 다릅니다(새 출력 파일을 쓰세요)")
            if common.sealed_reasons([clips_dir], clips) and not prev_fz:
                raise ValueError("기존 출력 파일에 동결 파일 기록이 없습니다(동결 전 결과일 수 있음). 새 출력 파일을 쓰세요")
            prev_enroll = head_prev.get("enroll") if isinstance(head_prev.get("enroll"), dict) else None
            prev_key = (prev_enroll.get("profile"), prev_enroll.get("mode")) if prev_enroll else None
            if not args.redo and prev_key != ((args.enroll_profile, mode) if mode else None):
                raise ValueError(f"기존 출력 파일의 등록 설정({prev_key})이 지금({(args.enroll_profile, mode) if mode else None})과 "
                                 "다릅니다. 새 출력 파일을 쓰거나 --redo로 모두 다시 보내세요")
    except (common.SealedDataError, ValueError, OSError) as exc:
        print(f"시작하지 않았습니다: {exc}", file=sys.stderr)
        return 2
    health, problems = wait_ready(server, args.phrase_set, commands, args.wait_ready, enroll=mode is not None)
    if problems:
        print("서버가 준비되지 않았습니다: " + "; ".join(problems), file=sys.stderr)
        return 3
    th = thresholds_from_health(health, args.phrase_set)
    early = common.rule_values_problems(th, expected) if (expected is not None and th is not None) else []
    if early:
        print("서버 기준값이 동결 값과 다릅니다(/health): " + "; ".join(early)
              + ". 서버를 동결 파일 full_list.env 환경 변수로 다시 켜세요(ward_eval_tools.py env).", file=sys.stderr)
        return 3
    keep = kept_rows(rows_prev, args.redo, args.phrase_set, want)
    prev_enroll = head_prev.get("enroll") if (isinstance(head_prev.get("enroll"), dict) and not args.redo) else None
    enroll_head: dict[str, Any] | None = None
    if mode is not None:
        profile = args.enroll_profile
        try:
            snap0 = profile_snapshot(get_profile(server, profile))
            redo_deleted = False
            if mode == MODE_FROM:
                if args.redo and snap0.get("templates"):
                    redo_deleted = delete_profile(server, profile)
                    snap_now = profile_snapshot(get_profile(server, profile))
                    if snap_now.get("templates"):
                        raise RuntimeError(f"--redo: 서버 등록 {profile}을(를) 지우지 못했습니다")
                else:
                    snap_now = snap0
                recorded = {keep[f]["enroll_request"]["command"]: keep[f].get("template_created") for f in store_files
                            if f in keep and ((result_enroll(keep[f].get("results")) or {}).get("stored"))}
                pending = {want[f]["command"] for f in store_files if f not in keep}
                conflicts = store_conflicts(snap_now, recorded, pending, resuming=prev_enroll is not None)
                if conflicts:
                    print(f"서버 등록 {profile}이(가) 이 출력 파일의 실행과 맞지 않아 멈춥니다(다른 실행이 만든 등록 위에 다시 "
                          "저장하지 않습니다). --redo로 지우고 처음부터 하거나 다른 ID를 쓰세요: " + "; ".join(conflicts),
                          file=sys.stderr)
                    return 6
            else:
                snap_now = snap0
                if snap0.get("status") != "ok":
                    print(f"서버 등록 {profile}이(가) 완성되지 않았습니다(status {snap0.get('status')}, {snap0.get('count')}개). "
                          "--enroll-match-only는 완성된(ok) 등록에만 씁니다", file=sys.stderr)
                    return 6
        except (RuntimeError, ValueError) as exc:
            print(f"서버 등록 상태를 확인하지 못했습니다: {exc}", file=sys.stderr)
            return 6
        enroll_head = {"profile": profile, "mode": mode, "template_mode": common.SERVICE_TEMPLATE,
                       "run_id": (prev_enroll or {}).get("run_id") or uuid.uuid4().hex,
                       "status_before": (prev_enroll or {}).get("status_before") or brief(snap0),
                       "status_at_this_start": brief(snap0), "redo_deleted": redo_deleted,
                       "store_files": store_files, "snapshot_after_store": None, "profile_sha256": None,
                       "status_after": None, "snapshot_after": None, "delete_after": bool(args.enroll_delete_after),
                       "deleted": False, "status_after_delete": None}
    head = {"schema": SCHEMA, "created": head_prev.get("created") or datetime.now(timezone.utc).isoformat(),
            "updated": None, "server": server, "phrase_set": args.phrase_set, "clips_dir": str(clips_dir),
            "clips_listing": common.folder_fingerprint(clips_dir), "n_clips": len(clips),
            "health_start": health, "health_end": None, "thresholds_from_health": th,
            "server_rules_seen": [],
            "commands": ({k: commands[k] for k in ("version", "sha256", "emergency")} if commands else None),
            "tool_sha256": common.file_sha256(Path(__file__)), "git_head": git_head(),
            "allow_sealed": bool(args.allow_sealed),
            "freeze": ({k: gate[k] for k in ("path", "sha256", "status", "commit", "commit_time", "sealed_ok")}
                       if gate else None),
            "freeze_thresholds_checked": bool(full_list), "freeze_expected_rule": expected,
            "enroll": enroll_head}
    rows_by_file = dict(keep)
    order = [c["file"] for c in clips]
    by_file = {c["file"]: c for c in clips}

    def save() -> None:
        head["updated"] = datetime.now(timezone.utc).isoformat()
        common.atomic_write_json(out, {**head, "rows": [rows_by_file[f] for f in order if f in rows_by_file]})

    def note_rule(results: Any) -> None:
        for w in results or []:
            rule = (w.get("phrase") or {}).get("rule") if isinstance(w, dict) and isinstance(w.get("phrase"), dict) else None
            if isinstance(rule, dict) and rule not in head["server_rules_seen"] and len(head["server_rules_seen"]) < 5:
                head["server_rules_seen"].append(rule)

    sent = 0
    t_start = time.time()

    def send(clip: dict[str, Any], request: dict[str, Any] | None, profile_sha: str | None) -> int | None:
        """클립 하나를 보내고 확인한다. 멈춰야 하면 종료 코드, 아니면 None."""
        nonlocal sent
        row = row_base(clip, args.phrase_set)
        if request is not None:
            row["enroll_request"] = request
        if clip.get("status") != "ok":
            row.update(status=clip.get("status"), error=clip.get("error"))
            rows_by_file[clip["file"]] = row
            return None
        fields = None
        if request is not None:
            fields = {"enroll_profile": request["profile"]}
            if request["mode"] == "store":
                fields["enroll_command"] = request["command"]
        attempts = max(0, args.retries) + 1
        for attempt in range(attempts):
            try:
                row.update(run_clip(server, Path(clip["path"]), args.phrase_set, fields))
            except (RuntimeError, OSError) as exc:
                row.update(status="error", error=f"{type(exc).__name__}: {exc}")
            row["attempts"] = attempt + 1
            if row.get("status") == "done":
                break
            if attempt + 1 < attempts:
                time.sleep(2.0)  # 다른 서버가 입술 추출 임시 폴더를 비운 경우 등: 잠깐 뒤 다시 보낸다
        enr = result_enroll(row.get("results")) if row.get("status") == "done" else None
        if request is not None and request["mode"] == "match":
            row["profile_sha256"] = profile_sha
        if request is not None and request["mode"] == "store" and enr and enr.get("stored"):
            try:
                snap = profile_snapshot(get_profile(server, request["profile"]))
                row["template_created"] = (snap["templates"].get(request["command"]) or {}).get("created")
            except (RuntimeError, ValueError) as exc:
                row["template_created"] = None
                row["error"] = f"등록 시각 확인 실패: {exc}"
        note_rule(row.get("results"))
        rows_by_file[clip["file"]] = row
        sent += 1
        save()
        print(json.dumps({"n": len(rows_by_file), "file": row["file"], "status": row.get("status"), "s": row.get("seconds"),
                          "phrase": [(w.get("phrase") or {}).get("id") for w in row.get("results") or []
                                     if isinstance(w, dict)],
                          "enroll": ({k: enr.get(k) for k in ("mode", "stored", "reason", "status", "top_id", "distance",
                                                              "accepted", "count") if k in enr} if enr else None),
                          "elapsed": round(time.time() - t_start)}, ensure_ascii=False), flush=True)
        mismatch = phrase_set_mismatch(row.get("results"), args.phrase_set)
        if mismatch:
            print(f"서버가 묶음 {args.phrase_set}을(를) 쓰지 않았습니다(phrase.set={mismatch}). 멈춥니다.", file=sys.stderr)
            return 4
        if row.get("status") == "done":
            structural = enroll_mismatch(row.get("results"), request)
            if structural:
                print(f"{row['file']}: {'; '.join(structural)}. 멈춥니다.", file=sys.stderr)
                return 4
        wrong = threshold_mismatch(row.get("results"), full_list)
        if wrong:
            print(f"서버 accepted가 동결 기준값({full_list.get('min_margin')}, {full_list.get('min_score')})과 다릅니다: {wrong}. "
                  "서버를 동결 파일 full_list.env 환경 변수로 다시 켜세요. 멈춥니다.", file=sys.stderr)
            return 5
        wrong_rule = rule_mismatch(row.get("results"), expected)
        wrong_enroll = enroll_mismatch(row.get("results"), request, expected) if (row.get("status") == "done"
                                                                                 and expected is not None) else []
        if wrong_rule or wrong_enroll:
            print(f"{row['file']}: 서버 기준값·등록 판정이 동결 값과 다릅니다: {'; '.join(wrong_rule + wrong_enroll)}. "
                  "서버를 동결 파일 full_list.env 환경 변수로 다시 켜세요(ward_eval_tools.py env). 멈춥니다.", file=sys.stderr)
            return 5
        return None

    # 1단계: 등록 영상 (from_voiced)
    for f in store_files:
        if f in keep:
            continue
        if args.limit and sent >= args.limit:
            break
        rc = send(by_file[f], want[f], None)
        if rc is not None:
            return rc
    profile_sha: str | None = None
    if mode is not None:
        pending_store = [f for f in store_files if rows_by_file.get(f, {}).get("status") != "done"]
        if pending_store:
            save()
            print(f"등록 영상 {len(pending_store)}개가 아직 done이 아니라 시험 단계를 시작하지 않습니다. 다시 실행하면 이어 합니다.",
                  file=sys.stderr)
            return 1
        try:
            snap = profile_snapshot(get_profile(server, args.enroll_profile))
        except (RuntimeError, ValueError) as exc:
            print(f"서버 등록 상태를 확인하지 못했습니다: {exc}", file=sys.stderr)
            return 6
        profile_sha = snap["sha256"]
        head["enroll"].update(snapshot_after_store=snap, profile_sha256=profile_sha)
        if snap.get("status") != "ok":
            print(f"주의: 서버 등록 {args.enroll_profile}이(가) 완성되지 않았습니다(status {snap.get('status')}, "
                  f"{snap.get('count')}개). 앱처럼 등록 없이(글자 규칙만) 채점됩니다.", file=sys.stderr)
        stale = [f for f, r in keep.items() if (want.get(f) or {}).get("mode") == "match"
                 and r.get("profile_sha256") != profile_sha]
        for f in stale:  # 등록이 바뀐 뒤의 시험 행은 다시 보낸다
            keep.pop(f, None)
            rows_by_file.pop(f, None)
        if stale:
            print(f"등록 상태가 이전 실행과 달라 시험 행 {len(stale)}개를 다시 보냅니다.", flush=True)
        save()
    # 2단계: 나머지 클립 (등록 없음 또는 match)
    for clip in clips:
        f = clip["file"]
        if f in keep or f in store_files:
            continue
        if args.limit and sent >= args.limit:
            break
        rc = send(clip, want[f], profile_sha)
        if rc is not None:
            return rc
    try:
        head["health_end"] = http_json(f"{server}/health")
    except RuntimeError as exc:
        head["health_end"] = {"error": str(exc)}
    done = sum(1 for r in rows_by_file.values() if r.get("status") == "done")
    all_done = done == len(clips)
    if mode is not None:
        try:
            snap_end = profile_snapshot(get_profile(server, args.enroll_profile))
            head["enroll"].update(status_after=brief(snap_end), snapshot_after=snap_end)
            if args.enroll_delete_after and all_done:
                head["enroll"]["deleted"] = delete_profile(server, args.enroll_profile)
                head["enroll"]["status_after_delete"] = brief(profile_snapshot(get_profile(server, args.enroll_profile)))
                print(f"서버 등록 {args.enroll_profile}을(를) 지웠습니다: {head['enroll']['deleted']}", flush=True)
            elif args.enroll_delete_after:
                print(f"아직 done이 아닌 클립이 있어 서버 등록 {args.enroll_profile}을(를) 남겼습니다(이어 하기용). 끝난 뒤 다시 "
                      "실행하면 지웁니다.", file=sys.stderr)
        except (RuntimeError, ValueError) as exc:
            head["enroll"]["status_after"] = {"error": str(exc)}
    save()
    print(f"끝: 행 {len(rows_by_file)}/{len(clips)}개, done {done}개 → {out}", flush=True)
    return 0 if all_done else 1


if __name__ == "__main__":
    sys.exit(main())
