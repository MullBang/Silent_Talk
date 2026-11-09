"""촬영 폴더의 클립을 실행 중인 서버(POST /api/record)로 하나씩 보내 결과 행을 모은다 (병실 기획 9.5).

세션 임시 스크립트(eval_service_phrases.py·eval_service_p02.py)를 옮겨 일반화한 것이다. 서비스 코드·기준값은 바꾸지 않는다.
  - 한 번에 한 클립: 앞 클립의 최종 상태를 받은 뒤 다음 클립을 보낸다. 매 클립 뒤 출력 파일을 다시 쓴다(중간에 끊겨도 안전).
  - 이어 하기: 출력 파일이 있으면 status가 done인 행은 그대로 두고 나머지만 다시 보낸다(--redo면 모두 다시).
  - --phrase-set ward|sentences를 폼 필드 phrase_set으로 보낸다. 결과 phrase.set이 다르면(묶음을 모르는 서버) 멈춘다.
    이어 하기에서도 phrase.set이 다른 행은 다시 보낸다.
  - 봉인 자료(처음 보는 사람 P02, 병실 명령 평가용 ward_eval)는 --allow-sealed --freeze <동결 파일>이 있어야 보낸다.
    동결 파일이 git에 커밋된 status frozen 파일인지 클립 목록을 읽기 전에 확인하고, 그 SHA-256·커밋을 출력 머리에 남긴다.
  - --freeze를 주면(병실 묶음) 클립마다 서버 accepted가 동결 기준값(full_list)으로 다시 판정한 값과 같은지 보고,
    다르면 서버가 동결 기준값으로 켜지지 않은 것이므로 멈춘다.
  - 시작·끝의 서버 /health(워커 상태, 체크포인트 SHA, ward_phrases 등)를 출력 파일 머리에 남긴다.
    /health가 기준값을 알려 주지 않으면 thresholds는 null이며, 채점 도구는 저장된 margin·score로 기준을 다시 적용한다.

행 형식: {file, participant, kind, condition, sentence_id, rep, posture, session, phrase_set, status, results, model,
         seconds(업로드 직전~최종 상태 벽시계 초), post_seconds, duration_sec(서버가 잰 녹화 길이), duration_ms(촬영 기록),
         error, spoken}

사용 예 (서버는 backend/에서 py -3.11 -m uvicorn main:app --host 127.0.0.1 --port 8001):
  py -3.11 scripts/ward_service_batch.py --clips "<P01 문장 촬영 폴더>" \
      --out experiments/ward_negatives_20261110/P01_sentences_as_ward.json --server http://127.0.0.1:8001 --phrase-set ward
  (P02, 동결 뒤) ... --clips "<P02 ward_eval 폴더>" --allow-sealed --freeze docs/병실명령_동결_2026-11-11.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
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


def multipart_body(video: Path, phrase_set: str, boundary: str | None = None) -> tuple[bytes, str]:
    """POST /api/record 멀티파트 본문 (file + phrase_set). 파일 이름은 서버가 믿지 않으므로 고정 이름을 쓴다.

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
    parts = [
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"phrase_set\"\r\n\r\n{phrase_set}\r\n".encode("utf-8"),
        (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"recording.{suffix}\"\r\n"
         f"Content-Type: {mime}\r\n\r\n").encode("utf-8"),
        data, b"\r\n", f"--{boundary}--\r\n".encode("utf-8"),
    ]
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def http_json(url: str, data: bytes | None = None, content_type: str | None = None,
              timeout: float = 10.0) -> dict[str, Any]:
    """GET(또는 data가 있으면 POST) → JSON dict. HTTP 오류는 본문 앞부분을 붙인 RuntimeError."""
    request = urllib.request.Request(url, data=data, method="POST" if data is not None else "GET")
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


def health_problems(health: dict[str, Any], phrase_set: str, commands: dict[str, Any] | None) -> list[str]:
    """서버 /health가 이 일괄 실행에 맞는지 (문제 목록, 없으면 빈 목록).

    워커 두 개가 ready이고 체크포인트 SHA가 서비스 고정값이어야 한다. ward면 ward_phrases의 개수·버전이 명령 목록과 같아야 한다.
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
    except (AttributeError, TypeError, KeyError) as exc:
        problems.append(f"/health 형식 오류: {exc}")
    return problems


def thresholds_from_health(health: dict[str, Any], phrase_set: str) -> dict[str, Any] | None:
    """/health가 묶음별 기준값을 알려 주면 그 값, 아니면 None (지금 서버는 알려 주지 않는다)."""
    try:
        model = health.get("model") or {}
        for key in ("phrase_sets", "thresholds"):
            block = model.get(key) or health.get(key)
            if isinstance(block, dict) and isinstance(block.get(phrase_set), dict):
                return block[phrase_set]
    except AttributeError:
        return None
    return None


def wait_ready(server: str, phrase_set: str, commands: dict[str, Any] | None, wait_seconds: float,
               fetch: Callable[[str], dict[str, Any]] | None = None,
               sleep: Callable[[float], None] = time.sleep) -> tuple[dict[str, Any], list[str]]:
    """워커가 준비될 때까지 /health를 기다린다. (마지막 health, 남은 문제 목록)."""
    fetch = fetch or (lambda url: http_json(url))
    deadline = time.monotonic() + max(0.0, wait_seconds)
    health: dict[str, Any] = {}
    problems: list[str] = ["/health를 받지 못했습니다"]
    while True:
        try:
            health = fetch(f"{server}/health")
            problems = health_problems(health, phrase_set, commands)
        except RuntimeError as exc:
            problems = [f"/health 요청 실패: {exc}"]
        if not problems or time.monotonic() >= deadline:
            return health, problems
        sleep(2.0)


def run_clip(server: str, video: Path, phrase_set: str) -> dict[str, Any]:
    """클립 하나를 보내고 최종 상태까지 폴링한다.

    Returns:
        {status, results, model, error, warnings, seconds, post_seconds, duration_sec}
    """
    t0 = time.perf_counter()
    body, ctype = multipart_body(video, phrase_set)
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


def kept_rows(existing: list[dict[str, Any]], redo: bool, phrase_set: str | None = None) -> dict[str, dict[str, Any]]:
    """이어 하기에서 그대로 둘 행 (파일 이름 → 행). status가 done이고 phrase.set이 요청 묶음과 같은 행만, --redo면 없음."""
    if redo:
        return {}
    return {r["file"]: r for r in existing
            if isinstance(r, dict) and r.get("status") == "done" and r.get("file")
            and not (phrase_set and phrase_set_mismatch(r.get("results"), phrase_set))}


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


def git_head() -> str | None:
    """저장소 HEAD 커밋 (git이 없으면 None)."""
    return common.git("rev-parse", "HEAD")


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
    ap.add_argument("--freeze", help="동결 파일. 봉인 자료에는 필수, 주면 서버 accepted가 동결 기준값과 같은지 클립마다 확인")
    ap.add_argument("--redo", action="store_true", help="이미 done인 행도 다시 보냄")
    ap.add_argument("--retries", type=int, default=1, help="실패한 클립을 바로 다시 보내는 횟수")
    ap.add_argument("--limit", type=int, default=0, help="이번에 보낼 최대 클립 수 (0 = 전부)")
    ap.add_argument("--expect-clips", type=int, default=0, help="폴더의 클립 수가 이 값이 아니면 멈춤 (0 = 확인 안 함)")
    ap.add_argument("--wait-ready", type=float, default=240.0, help="워커 준비를 기다리는 최대 초")
    args = ap.parse_args(argv)
    server = args.server.rstrip("/")
    clips_dir = Path(args.clips).resolve()
    out = Path(args.out).resolve()
    try:
        # 봉인 확인은 클립 기록을 읽기 전에: --allow-sealed면 동결 파일부터, 아니면 경로부터 본다
        gate = common.freeze_gate(args.freeze, args.allow_sealed)
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
    except (common.SealedDataError, ValueError, OSError) as exc:
        print(f"시작하지 않았습니다: {exc}", file=sys.stderr)
        return 2
    health, problems = wait_ready(server, args.phrase_set, commands, args.wait_ready)
    if problems:
        print("서버가 준비되지 않았습니다: " + "; ".join(problems), file=sys.stderr)
        return 3
    keep = kept_rows(rows_prev, args.redo, args.phrase_set)
    head = {"schema": SCHEMA, "created": head_prev.get("created") or datetime.now(timezone.utc).isoformat(),
            "updated": None, "server": server, "phrase_set": args.phrase_set, "clips_dir": str(clips_dir),
            "clips_listing": common.folder_fingerprint(clips_dir), "n_clips": len(clips),
            "health_start": health, "health_end": None, "thresholds_from_health": thresholds_from_health(health, args.phrase_set),
            "commands": ({k: commands[k] for k in ("version", "sha256", "emergency")} if commands else None),
            "tool_sha256": common.file_sha256(Path(__file__)), "git_head": git_head(),
            "allow_sealed": bool(args.allow_sealed),
            "freeze": ({k: gate[k] for k in ("path", "sha256", "status", "commit", "commit_time", "sealed_ok")}
                       if gate else None),
            "freeze_thresholds_checked": bool(full_list)}
    rows_by_file = dict(keep)
    order = [c["file"] for c in clips]

    def save() -> None:
        head["updated"] = datetime.now(timezone.utc).isoformat()
        common.atomic_write_json(out, {**head, "rows": [rows_by_file[f] for f in order if f in rows_by_file]})

    sent = 0
    t_start = time.time()
    for clip in clips:
        if clip["file"] in keep:
            continue
        if args.limit and sent >= args.limit:
            break
        row = row_base(clip, args.phrase_set)
        if clip.get("status") != "ok":
            row.update(status=clip.get("status"), error=clip.get("error"))
            rows_by_file[clip["file"]] = row
            continue
        attempts = max(0, args.retries) + 1
        for attempt in range(attempts):
            try:
                row.update(run_clip(server, Path(clip["path"]), args.phrase_set))
            except (RuntimeError, OSError) as exc:
                row.update(status="error", error=f"{type(exc).__name__}: {exc}")
            row["attempts"] = attempt + 1
            if row.get("status") == "done":
                break
            if attempt + 1 < attempts:
                time.sleep(2.0)  # 다른 서버가 입술 추출 임시 폴더를 비운 경우 등: 잠깐 뒤 다시 보낸다
        mismatch = phrase_set_mismatch(row.get("results"), args.phrase_set)
        rows_by_file[clip["file"]] = row
        sent += 1
        save()
        print(json.dumps({"n": len(rows_by_file), "file": row["file"], "status": row.get("status"), "s": row.get("seconds"),
                          "phrase": [(w.get("phrase") or {}).get("id") for w in row.get("results") or []],
                          "elapsed": round(time.time() - t_start)}, ensure_ascii=False), flush=True)
        if mismatch:
            print(f"서버가 묶음 {args.phrase_set}을(를) 쓰지 않았습니다(phrase.set={mismatch}). 멈춥니다.", file=sys.stderr)
            return 4
        wrong = threshold_mismatch(row.get("results"), full_list)
        if wrong:
            print(f"서버 accepted가 동결 기준값({full_list.get('min_margin')}, {full_list.get('min_score')})과 다릅니다: {wrong}. "
                  "서버를 동결 파일 full_list.env 환경 변수로 다시 켜세요. 멈춥니다.", file=sys.stderr)
            return 5
    try:
        head["health_end"] = http_json(f"{server}/health")
    except RuntimeError as exc:
        head["health_end"] = {"error": str(exc)}
    save()
    done = sum(1 for r in rows_by_file.values() if r.get("status") == "done")
    print(f"끝: 행 {len(rows_by_file)}/{len(clips)}개, done {done}개 → {out}", flush=True)
    return 0 if done == len(clips) else 1


if __name__ == "__main__":
    sys.exit(main())
