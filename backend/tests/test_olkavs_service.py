"""OLKAVS 상주 워커 클라이언트 검사.

실제 공식 모델 대신 같은 JSON 줄 규약을 따르는 가짜 워커(현재 인터프리터로 실행)를 써서
기동·진행률·오류·시간 초과·비정상 종료·재기동·사용자 문구를 검증한다. 실제 olkavs_worker.serve()
루프는 가짜 코어를 주입해 서브프로세스로 검증한다. 실제 두 가상환경 통합 검사는
test_olkavs_integration.py (선택 실행).
"""
from __future__ import annotations

import os
import sys
import textwrap
import threading
import time

import pytest

_BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

from services import olkavs_service as svc  # noqa: E402

FAKE_WORKER = textwrap.dedent(r'''
    import json, os, sys, time
    mode = sys.argv[1]
    proto = os.fdopen(os.dup(1), "wb", buffering=0)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    print("noise on stdout must not break the protocol")
    print("SECRET-LOG-LINE C:/Users/someone/other_session_file.mp4", file=sys.stderr, flush=True)
    def send(m):
        proto.write((json.dumps(m, ensure_ascii=True) + "\n").encode("ascii")); proto.flush()
    if mode == "fatal":
        send({"type": "fatal", "error": {"type": "RuntimeError", "message": "no model at C:/secret/path"}}); sys.exit(1)
    if mode == "silent":
        time.sleep(30); sys.exit(0)
    send({"type": "ready", "info": {"mode": mode, "pid": os.getpid()}})
    proto.write(b"[1, 2, 3]\n"); proto.flush()   # 규약 밖 출력(객체가 아닌 JSON)은 무시되어야 한다
    for line in sys.stdin:
        req = json.loads(line)
        rid, cmd = req["id"], req.get("cmd")
        if cmd == "shutdown":
            send({"type": "result", "id": rid, "ok": True}); break
        if cmd == "echo":
            for k in range(3):
                send({"type": "progress", "id": rid, "fraction": (k + 1) / 3})
            send({"type": "result", "id": rid, "ok": True, "echo": req.get("value"), "pid": os.getpid()})
        elif cmd == "fail":
            send({"type": "result", "id": rid, "ok": False,
                  "error": {"type": "ValueError", "message": req.get("message", "bad input C:/secret/x.mp4")}})
        elif cmd == "poison":
            send({"type": "result", "id": rid, "ok": False, "worker_poisoned": True,
                  "error": {"type": "AcceleratorError", "message": "CUDA error: device-side assert"}})
        elif cmd == "sleep":
            time.sleep(float(req.get("seconds", 10)))
            send({"type": "result", "id": rid, "ok": True})
        elif cmd == "die":
            os._exit(7)
''')


@pytest.fixture()
def fake_script(tmp_path):
    """가짜 워커 스크립트 경로."""
    path = tmp_path / "fake_worker.py"
    path.write_text(FAKE_WORKER, encoding="utf-8")
    return str(path)


def _worker(fake_script: str, tmp_path, mode: str = "ok", start_timeout: float = 20.0, **kw) -> svc.WorkerProcess:
    return svc.WorkerProcess(f"fake-{mode}", [sys.executable, "-u", fake_script, mode],
                             str(tmp_path / f"{mode}.log"), start_timeout, **kw)


def _assert_safe(message: str) -> None:
    """사용자 문구에 로그·경로·상세가 섞이지 않았는지."""
    assert "SECRET" not in message and "C:/" not in message and "C:\\" not in message and "|" not in message


def test_worker_ready_request_progress_and_close(fake_script, tmp_path) -> None:
    """기동 → ready 정보, 진행 메시지 콜백, 표준출력 잡음·규약 밖 JSON 무시, 정상 종료."""
    worker = _worker(fake_script, tmp_path)
    try:
        info = worker.start()
        assert info["mode"] == "ok" and worker.state() == "ready"
        seen = []
        out = worker.request({"cmd": "echo", "value": "한글 경로 D:\\009.립리딩"}, timeout=10, on_progress=seen.append)
        assert out["echo"] == "한글 경로 D:\\009.립리딩"
        assert [m["fraction"] for m in seen] == pytest.approx([1 / 3, 2 / 3, 1.0])
        assert worker.request({"cmd": "echo", "value": 2}, timeout=10)["pid"] == info["pid"]  # 같은 프로세스 재사용
    finally:
        worker.close()
    assert worker.state() == "stopped"
    assert "noise on stdout" in (tmp_path / "ok.log").read_text(encoding="utf-8", errors="replace")


def test_worker_error_result_keeps_process_and_uses_fixed_message(fake_script, tmp_path) -> None:
    """처리 실패는 고정 문구의 WorkerError(상세는 detail)로 올리되 워커는 계속 쓴다."""
    worker = _worker(fake_script, tmp_path)
    try:
        pid = worker.start()["pid"]
        with pytest.raises(svc.WorkerError) as info:
            worker.request({"cmd": "fail"}, timeout=10)
        assert str(info.value) == svc.MSG_FAILED and "bad input" in info.value.detail
        _assert_safe(str(info.value))
        with pytest.raises(svc.WorkerError) as info2:
            worker.request({"cmd": "fail", "message": "영상을 열 수 없습니다: abc_홍길동.mp4"}, timeout=10)
        assert str(info2.value) == svc.MSG_VIDEO_UNREADABLE and "홍길동" not in str(info2.value)
        assert worker.last_error is None  # 요청 하나의 실패는 워커 상태 오류가 아니다
        assert worker.request({"cmd": "echo", "value": 1}, timeout=10)["pid"] == pid
    finally:
        worker.close()


def test_worker_restarts_after_consecutive_failures(fake_script, tmp_path) -> None:
    """같은 워커가 연속 N회 실패하면 종료해 다음 요청에서 새로 띄운다."""
    worker = _worker(fake_script, tmp_path, max_consecutive_failures=2)
    try:
        pid = worker.start()["pid"]
        for _ in range(2):
            with pytest.raises(svc.WorkerError):
                worker.request({"cmd": "fail"}, timeout=10)
        assert worker.state() == "stopped"
        assert worker.request({"cmd": "echo", "value": 1}, timeout=10)["pid"] != pid
    finally:
        worker.close()


def test_worker_poisoned_result_triggers_restart(fake_script, tmp_path) -> None:
    """CUDA 컨텍스트 손상(worker_poisoned)이면 한 번에 재기동한다."""
    worker = _worker(fake_script, tmp_path)
    try:
        pid = worker.start()["pid"]
        with pytest.raises(svc.WorkerError) as info:
            worker.request({"cmd": "poison"}, timeout=10)
        _assert_safe(str(info.value))
        assert worker.state() == "stopped"
        assert worker.request({"cmd": "echo", "value": 1}, timeout=10)["pid"] != pid
    finally:
        worker.close()


def test_worker_timeout_kills_and_next_request_restarts(fake_script, tmp_path) -> None:
    """시간 초과면 워커를 종료하고, 다음 요청에서 새 프로세스로 다시 띄운다."""
    worker = _worker(fake_script, tmp_path)
    try:
        pid = worker.start()["pid"]
        with pytest.raises(svc.WorkerError) as info:
            worker.request({"cmd": "sleep", "seconds": 30}, timeout=1.5)
        assert str(info.value) == svc.MSG_TIMEOUT
        assert worker.state() == "stopped"
        assert worker.request({"cmd": "echo", "value": 1}, timeout=10)["pid"] != pid
    finally:
        worker.close()


def test_worker_crash_message_has_no_log_tail(fake_script, tmp_path) -> None:
    """요청 중 프로세스가 죽으면 고정 문구로 알리고(로그 꼬리는 detail에만), 다음 요청은 재기동한다."""
    worker = _worker(fake_script, tmp_path)
    try:
        worker.start()
        with pytest.raises(svc.WorkerError) as info:
            worker.request({"cmd": "die"}, timeout=10)
        assert str(info.value) == svc.MSG_CRASHED
        _assert_safe(str(info.value))
        assert "SECRET-LOG-LINE" in info.value.detail  # 서버 로그용 상세에는 남는다
        # /health에 보이는 상태는 짧은 코드와 시각뿐 (로그 꼬리·경로 없음)
        assert worker.last_error.startswith("crashed at ")
        _assert_safe(worker.last_error)
        assert "SECRET" not in worker.last_error and str(tmp_path) not in worker.last_error
        assert worker.request({"cmd": "echo", "value": 1}, timeout=10)["echo"] == 1
    finally:
        worker.close()


def test_worker_fatal_start_cooldown_and_missing_interpreter(fake_script, tmp_path) -> None:
    """기동 실패(fatal)는 고정 문구, 대기 시간 동안은 다시 띄우지 않는다. 기동 시간 초과·인터프리터 없음도 WorkerError."""
    fatal = _worker(fake_script, tmp_path, "fatal", fatal_cooldown=60)
    with pytest.raises(svc.WorkerError) as info:
        fatal.start()
    assert str(info.value) == svc.MSG_START_FAILED and "no model" in info.value.detail
    _assert_safe(str(info.value))
    began = time.monotonic()
    with pytest.raises(svc.WorkerError):
        fatal.start()  # 대기 시간 안: 프로세스를 띄우지 않고 바로 실패
    assert time.monotonic() - began < 1.0 and fatal.state() == "stopped"
    silent = _worker(fake_script, tmp_path, "silent", start_timeout=1.5)
    with pytest.raises(svc.WorkerError) as info:
        silent.start()
    assert str(info.value) == svc.MSG_START_FAILED and silent.state() == "stopped"
    missing = svc.WorkerProcess("missing", [str(tmp_path / "nope.exe")], str(tmp_path / "m.log"), 5)
    with pytest.raises(svc.WorkerError) as info:
        missing.start()
    assert "인터프리터가 없습니다" in info.value.detail
    _assert_safe(str(info.value))


# ---------------------------------------------------------------------------
# 실제 olkavs_worker.serve() 루프 (가짜 코어 주입)
# ---------------------------------------------------------------------------
SERVE_DRIVER = textwrap.dedent(r'''
    import sys, types
    sys.path.insert(0, sys.argv[1])
    from services import olkavs_worker
    core = types.SimpleNamespace(
        read_frames=lambda p: [0] * 5,
        recognize=lambda frames, device, decoder: {"text": "안녕", "confidence": 0.5, "raw_score": -1.0,
                                                   "decoder": decoder, "token_ids": [1], "frames": len(frames)},
        extract_segments=lambda video, out_dir, progress=None, **kw: (
            progress({"stage": "roi", "fraction": 0.5}) or {"segments": [], "kw": sorted(kw),
                                                           "hint": kw.get("duration_hint_seconds")}),
    )
    def init(c):
        import pdb
        return {"mode": sys.argv[2], "device": "cpu", "pdb_stdin_is_devnull": sys.stdin.read() == ""}
    sys.exit(olkavs_worker.serve(sys.argv[2], "cpu", "hybrid", False, core=core, init=init))
''')


def test_real_worker_serve_protocol(tmp_path) -> None:
    """실제 serve(): ready, ping, 진행률, 잘못된 JSON(id=None 응답 후 계속), 모드 밖 명령 오류, shutdown."""
    driver = tmp_path / "driver.py"
    driver.write_text(SERVE_DRIVER, encoding="utf-8")
    for mode in ("infer", "roi"):
        worker = svc.WorkerProcess(mode, [sys.executable, "-u", str(driver), _BACKEND_ROOT, mode],
                                   str(tmp_path / f"{mode}.log"), 30)
        try:
            info = worker.start()
            assert info["mode"] == mode and info["pdb_stdin_is_devnull"] is True
            assert worker.request({"cmd": "ping"}, timeout=10)["info"]["mode"] == mode
            worker._proc.stdin.write(b"not json\n")
            worker._proc.stdin.flush()
            if mode == "infer":
                out = worker.request({"cmd": "infer", "npz": "x.npz"}, timeout=10)
                assert out["text"] == "안녕" and "token_ids" not in out
                with pytest.raises(svc.WorkerError):
                    worker.request({"cmd": "roi", "video": "v", "out_dir": "o"}, timeout=10)
            else:
                seen = []
                out = worker.request({"cmd": "roi", "video": "v.mp4", "out_dir": str(tmp_path)}, timeout=10,
                                     on_progress=seen.append)
                assert out["segments"] == [] and seen and seen[0]["fraction"] == 0.5
                base = {"cut_search_seconds", "min_window_seconds", "resample", "segment_seconds"}
                # 6단계: 워커가 utterance 플래그·길이 힌트를 넘길 수 있다(없으면 5단계와 같다).
                assert base <= set(out["kw"]) <= base | {"utterance", "duration_hint_seconds"}
                assert out["hint"] is None  # duration_seconds를 보내지 않으면 힌트 없음
                hinted = worker.request({"cmd": "roi", "video": "v.mp4", "out_dir": str(tmp_path),
                                         "duration_seconds": 8.5}, timeout=10)
                assert hinted["hint"] == 8.5
                for bad in (-1.0, 0, "8", True):
                    with pytest.raises(svc.WorkerError):  # 유한한 양수·null이 아니면 ok:false
                        worker.request({"cmd": "roi", "video": "v.mp4", "out_dir": str(tmp_path),
                                        "duration_seconds": bad}, timeout=10)
            assert worker.state() == "ready"
        finally:
            worker.close()
        assert worker.state() == "stopped"


# ---------------------------------------------------------------------------
# 서비스: 구간 → 결과, FIFO, 중단
# ---------------------------------------------------------------------------
class _StubWorker:
    """request()/start()만 흉내 내는 대체 워커."""

    def __init__(self, handler, name="stub"):
        self.handler = handler
        self.calls = []
        self.info = None
        self.last_error = None
        self.name = name
        self.started = 0

    def start(self):
        self.started += 1
        return {}

    def request(self, payload, timeout, on_progress=None):
        self.calls.append(payload)
        return self.handler(payload, on_progress)

    def state(self):
        return "ready"

    def close(self):
        pass


def _roi_result(segments):
    def handler(payload, on_progress):
        assert payload["cmd"] == "roi"
        if on_progress:
            on_progress({"type": "progress", "fraction": 0.5})
            on_progress({"type": "progress", "fraction": None})
        return {"ok": True, "segments": segments, "selection": {"mode": "pts_grid", "reason": "x"}}
    return handler


def test_transcribe_video_maps_segments_to_results(tmp_path, monkeypatch) -> None:
    """정상 창은 인식, 실패 구간은 FAIL_TEXT, 빈 결과는 UNREC_TEXT. 진행률은 0→100 단조. 워커를 먼저 확인.

    구간 대응만 보므로 confidence 문턱은 끈다(문턱 동작은 별도 테스트).
    """
    monkeypatch.setattr(svc.config, "OLKAVS_MIN_CONFIDENCE", 0.0)
    segments = [
        {"index": 0, "start_ms": 0, "end_ms": 10000, "status": "ok", "npz": "a.npz"},
        {"index": 1, "start_ms": 10000, "end_ms": 20000, "status": "failed", "npz": None, "reason": "no_face"},
        {"index": 2, "start_ms": 20000, "end_ms": 21500, "status": "ok", "npz": "c.npz"},
    ]
    texts = {"a.npz": {"text": "안녕하세요", "confidence": 0.7, "raw_score": -4.0, "search_cap_reached": True},
             "c.npz": {"text": "", "confidence": 0.2, "raw_score": -9.0}}
    roi = _StubWorker(_roi_result(segments))
    infer = _StubWorker(lambda payload, _p: {"ok": True, **texts[payload["npz"]]})
    service = svc.OlkavsService(roi, infer)
    progress = []
    results = service.transcribe_video(str(tmp_path / "v.mp4"), str(tmp_path / "work"), on_progress=progress.append)
    assert results == [
        {"start_ms": 0, "end_ms": 10000, "text": "안녕하세요", "confidence": 0.7, "raw_score": -4.0},
        {"start_ms": 10000, "end_ms": 20000, "text": svc.FAIL_TEXT, "confidence": None, "raw_score": None},
        {"start_ms": 20000, "end_ms": 21500, "text": svc.UNREC_TEXT, "confidence": None, "raw_score": None},
    ]
    assert [c["npz"] for c in infer.calls] == ["a.npz", "c.npz"]  # 실패 구간은 인식하지 않는다
    assert progress == sorted(progress) and progress[-1] == 100.0 and all(0 <= p <= 100 for p in progress)
    assert roi.calls[0]["segment_seconds"] > 0 and os.path.isabs(roi.calls[0]["video"])
    assert infer.started == 1 and roi.started == 1


def test_transcribe_video_fails_fast_when_infer_worker_cannot_start(tmp_path) -> None:
    """인식 워커가 못 뜨면 입술 추출을 하지 않고 바로 실패한다."""
    roi = _StubWorker(_roi_result([]))

    class Broken(_StubWorker):
        def start(self):
            raise svc.WorkerError(svc.MSG_START_FAILED, "no weights")

    service = svc.OlkavsService(roi, Broken(lambda p, _p: {}))
    with pytest.raises(svc.WorkerError, match="시작하지 못했습니다"):
        service.transcribe_video("v.mp4", str(tmp_path))
    assert roi.calls == []


def test_transcribe_video_abort_and_empty(tmp_path) -> None:
    """중단 신호면 JobAborted(입술 추출 전에도 확인), 구간이 없으면 WorkerError."""
    segments = [{"index": 0, "start_ms": 0, "end_ms": 1000, "status": "ok", "npz": "a.npz"}]
    roi = _StubWorker(_roi_result(segments))
    service = svc.OlkavsService(roi, _StubWorker(lambda payload, _p: {"ok": True, "text": "x"}))
    with pytest.raises(svc.JobAborted):
        service.transcribe_video("v.mp4", str(tmp_path), should_abort=lambda: True)
    assert roi.calls == []
    empty = svc.OlkavsService(_StubWorker(_roi_result([])), _StubWorker(lambda p, _p: {}))
    with pytest.raises(svc.WorkerError) as info:
        empty.transcribe_video("v.mp4", str(tmp_path))
    assert str(info.value) == svc.MSG_NO_FRAMES


def test_fifo_gate_orders_jobs_and_skips_aborted() -> None:
    """작업은 들어온 순서대로 하나씩, 대기 중 중단된 작업은 건너뛴다."""
    gate = svc.FifoGate()
    first = gate.acquire()
    order, threads = [], []
    abort_second = threading.Event()

    def job(name, abort=None):
        try:
            ticket = gate.acquire(should_abort=abort, poll=0.05)
        except svc.JobAborted:
            order.append(f"{name}-aborted")
            return
        order.append(name)
        gate.release(ticket)

    for name, abort in (("second", abort_second.is_set), ("third", None), ("fourth", None)):
        t = threading.Thread(target=job, args=(name, abort))
        t.start()
        threads.append(t)
        time.sleep(0.15)  # 들어온 순서를 고정
    assert gate.waiting() == 4
    abort_second.set()
    time.sleep(0.3)
    assert gate.waiting() == 3  # 차례 전에 취소된 작업은 대기 수에서 빠진다
    gate.release(first)
    for t in threads:
        t.join(5)
    assert order == ["second-aborted", "third", "fourth"]
    assert gate.waiting() == 0


def test_transcribe_video_progress_never_goes_backwards(tmp_path) -> None:
    """ROI 진행률이 앞뒤로 흔들려도 사용자 진행률은 줄지 않는다."""
    segments = [{"index": 0, "start_ms": 0, "end_ms": 1000, "status": "ok", "npz": "a.npz"}]

    def roi_handler(payload, on_progress):
        for fraction in (0.6, 0.3, 0.9, 0.1):
            on_progress({"type": "progress", "fraction": fraction})
        return {"ok": True, "segments": segments, "selection": {}}

    service = svc.OlkavsService(_StubWorker(roi_handler), _StubWorker(lambda p, _p: {"ok": True, "text": "가"}))
    progress = []
    service.transcribe_video("v.mp4", str(tmp_path), on_progress=progress.append)
    assert progress == sorted(progress) and progress[-1] == 100.0


def test_transcribe_video_admission_control(tmp_path) -> None:
    """남은 Job 시간이 예상 처리 시간보다 짧으면 입술 추출을 시작하지 않고 MSG_BUSY로 끝낸다."""
    segments = [{"index": 0, "start_ms": 0, "end_ms": 1000, "status": "ok", "npz": "a.npz"}]
    roi = _StubWorker(_roi_result(segments))
    infer = _StubWorker(lambda p, _p: {"ok": True, "text": "가"})
    service = svc.OlkavsService(roi, infer)
    expected = svc.expected_processing_seconds(180.0)
    assert expected > 180.0
    with pytest.raises(svc.JobAborted) as info:
        service.transcribe_video("v.mp4", str(tmp_path), remaining_seconds=lambda: expected - 1,
                                 video_seconds=180.0)
    assert info.value.user_message == svc.MSG_BUSY and roi.calls == [] and infer.started == 0
    assert service.gate.waiting() == 0  # 거절한 작업도 차례를 넘긴다
    # 시간이 충분하거나 길이를 모르면 그대로 처리한다.
    assert service.transcribe_video("v.mp4", str(tmp_path), remaining_seconds=lambda: expected + 1,
                                    video_seconds=180.0)[0]["text"] == "가"
    assert service.transcribe_video("v.mp4", str(tmp_path), remaining_seconds=lambda: 1.0,
                                    video_seconds=None)[0]["text"] == "가"


def test_transcribe_video_low_fps_warning(tmp_path) -> None:
    """ROI 워커가 낮은 프레임률을 알리면 사용자 경고를 덧붙인다."""
    segments = [{"index": 0, "start_ms": 0, "end_ms": 1000, "status": "ok", "npz": "a.npz"}]

    def roi_handler(payload, on_progress):
        return {"ok": True, "segments": segments,
                "selection": {"mode": "pts_grid", "reason": "25fps", "low_fps": True, "source_fps": 25.0}}

    service = svc.OlkavsService(_StubWorker(roi_handler), _StubWorker(lambda p, _p: {"ok": True, "text": "가"}))
    warnings = []
    service.transcribe_video("v.mp4", str(tmp_path), warnings=warnings)
    assert warnings == [svc.MSG_LOW_FPS.format(fps=25.0)] and "25fps" in warnings[0]
    normal = svc.OlkavsService(_StubWorker(_roi_result(segments)), _StubWorker(lambda p, _p: {"ok": True, "text": "가"}))
    quiet = []
    normal.transcribe_video("v.mp4", str(tmp_path), warnings=quiet)
    assert quiet == []


def test_status_does_not_start_workers(tmp_path) -> None:
    """상태 조회는 워커를 띄우지 않는다."""
    roi = svc.WorkerProcess("roi", [str(tmp_path / "nope.exe")], str(tmp_path / "r.log"), 5)
    infer = svc.WorkerProcess("infer", [str(tmp_path / "nope.exe")], str(tmp_path / "i.log"), 5)
    status = svc.OlkavsService(roi, infer).status()
    assert status["roi_worker"] == "stopped" and status["infer_worker"] == "stopped"
    assert status["jobs_in_queue"] == 0
    assert status["last_error"] == {"roi": None, "infer": None}


def test_worker_env_strips_server_paths(monkeypatch) -> None:
    """워커 환경에는 서버의 PYTHONPATH 등이 섞이지 않는다."""
    monkeypatch.setenv("PYTHONPATH", "C:/server/site")
    env = svc._worker_env()
    assert "PYTHONPATH" not in env and env["PYTHONNOUSERSITE"] == "1" and env["PYTHONIOENCODING"] == "utf-8"


# ---------------------------------------------------------------------------
# 6단계: 발화 단위(utterance) 요청, 무발화·낮은 확신 표시, 모델 정보, 설정 검사
# ---------------------------------------------------------------------------
def test_transcribe_video_forwards_utterance_flag(tmp_path) -> None:
    """utterance 인자는 ROI 요청의 'utterance'로 그대로 간다 (기본 False)."""
    segments = [{"index": 0, "start_ms": 0, "end_ms": 1000, "status": "ok", "npz": "a.npz"}]
    roi = _StubWorker(_roi_result(segments))
    service = svc.OlkavsService(roi, _StubWorker(lambda p, _p: {"ok": True, "text": "가"}))
    service.transcribe_video("v.mp4", str(tmp_path))
    service.transcribe_video("v.mp4", str(tmp_path), utterance=True)
    assert [c["utterance"] for c in roi.calls] == [False, True]
    assert all(c["segment_seconds"] == svc.config.OLKAVS_SEGMENT_SECONDS for c in roi.calls)


def test_transcribe_video_no_speech_segment_is_not_inferred(tmp_path) -> None:
    """status 'no_speech' 창은 인식하지 않고 '[발화 없음]'(점수 없음)으로 표시한다."""
    segments = [
        {"index": 0, "start_ms": 0, "end_ms": 4000, "status": "no_speech", "npz": None,
         "speech": {"active_seconds": 0.0, "ratio_coverage": 1.0}},
        {"index": 1, "start_ms": 4000, "end_ms": 9000, "status": "ok", "npz": "b.npz",
         "speech": {"active_seconds": 2.5, "ratio_coverage": 1.0}},
    ]

    def roi_handler(payload, on_progress):
        return {"ok": True, "segments": segments, "selection": {},
                "speech": {"enabled": True, "detected": True, "start_ms": 4200, "end_ms": 8700,
                           "active_seconds": 2.5, "trimmed_head_ms": 0, "trimmed_tail_ms": 0},
                "vad": {"threshold": 0.1}}

    infer = _StubWorker(lambda p, _p: {"ok": True, "text": "안녕", "confidence": 0.9, "raw_score": -1.0})
    service = svc.OlkavsService(_StubWorker(roi_handler), infer)
    progress = []
    results = service.transcribe_video("v.mp4", str(tmp_path), on_progress=progress.append)
    assert svc.NO_SPEECH_TEXT == "[발화 없음]"
    assert results == [
        {"start_ms": 0, "end_ms": 4000, "text": svc.NO_SPEECH_TEXT, "confidence": None, "raw_score": None},
        {"start_ms": 4000, "end_ms": 9000, "text": "안녕", "confidence": 0.9, "raw_score": -1.0},
    ]
    assert [c["npz"] for c in infer.calls] == ["b.npz"]
    assert progress == sorted(progress) and progress[-1] == 100.0


def test_transcribe_video_low_confidence_becomes_unrecognized_with_candidate(tmp_path, monkeypatch) -> None:
    """confidence < OLKAVS_MIN_CONFIDENCE 이면 '[인식 불가]'로, 점수와 모델 문장(candidate)은 남긴다."""
    segments = [{"index": i, "start_ms": i * 1000, "end_ms": (i + 1) * 1000, "status": "ok", "npz": f"{i}.npz"}
                for i in range(4)]
    outputs = {"0.npz": {"text": "낮은 확신", "confidence": 0.29, "raw_score": -20.0},
               "1.npz": {"text": "경계값", "confidence": 0.3, "raw_score": -5.0},
               "2.npz": {"text": "", "confidence": 0.01, "raw_score": -50.0},
               "3.npz": {"text": "점수 없음", "confidence": None, "raw_score": None}}
    monkeypatch.setattr(svc.config, "OLKAVS_MIN_CONFIDENCE", 0.3)
    service = svc.OlkavsService(_StubWorker(_roi_result(segments)),
                                _StubWorker(lambda p, _p: {"ok": True, **outputs[p["npz"]]}))
    results = service.transcribe_video("v.mp4", str(tmp_path))
    assert results == [
        {"start_ms": 0, "end_ms": 1000, "text": svc.UNREC_TEXT, "confidence": 0.29, "raw_score": -20.0,
         "candidate": "낮은 확신"},
        {"start_ms": 1000, "end_ms": 2000, "text": "경계값", "confidence": 0.3, "raw_score": -5.0},
        {"start_ms": 2000, "end_ms": 3000, "text": svc.UNREC_TEXT, "confidence": None, "raw_score": None},
        {"start_ms": 3000, "end_ms": 4000, "text": "점수 없음", "confidence": None, "raw_score": None},
    ]


def test_transcribe_video_default_threshold_keeps_low_confidence_text(tmp_path, monkeypatch) -> None:
    """기본값 0.0(판정 끔)이면 확신이 낮아도 모델 문장을 그대로 보여 준다 (5단계 동작)."""
    monkeypatch.setattr(svc.config, "OLKAVS_MIN_CONFIDENCE", 0.0)
    segments = [{"index": 0, "start_ms": 0, "end_ms": 1000, "status": "ok", "npz": "a.npz"}]
    service = svc.OlkavsService(_StubWorker(_roi_result(segments)),
                                _StubWorker(lambda p, _p: {"ok": True, "text": "가", "confidence": 0.0001,
                                                           "raw_score": -99.0}))
    assert service.transcribe_video("v.mp4", str(tmp_path)) == [
        {"start_ms": 0, "end_ms": 1000, "text": "가", "confidence": 0.0001, "raw_score": -99.0}]


def test_model_info_without_and_with_started_workers(tmp_path) -> None:
    """model_info는 워커를 띄우지 않고, 기동 정보가 있으면 SHA·커밋·디코더·ROI 프로필을 알려 준다."""
    roi = svc.WorkerProcess("roi", [str(tmp_path / "nope.exe")], str(tmp_path / "r.log"), 5)
    infer = svc.WorkerProcess("infer", [str(tmp_path / "nope.exe")], str(tmp_path / "i.log"), 5)
    service = svc.OlkavsService(roi, infer)
    assert service.model_info() == {"backend": "olkavs", "decoder": svc.config.OLKAVS_DECODER,
                                    "checkpoint_sha256": None, "upstream_commit": None, "roi_profile": None}
    assert roi.state() == "stopped" and infer.state() == "stopped"
    infer.info = {"decoder": "ctc", "checkpoint_sha256": "f41d7423", "upstream_commit": "abc123", "device": "cpu"}
    roi.info = {"detector_profile": "stable_context_square"}
    assert service.model_info() == {"backend": "olkavs", "decoder": "ctc", "checkpoint_sha256": "f41d7423",
                                    "upstream_commit": "abc123", "roi_profile": "stable_context_square"}


@pytest.mark.parametrize("raw, expected", [(None, 0.0), ("", 0.0), ("0.35", 0.35), ("0", 0.0), ("1", 1.0)])
def test_min_confidence_env_accepts_unit_range(monkeypatch, raw, expected) -> None:
    """LIPNET_OLKAVS_MIN_CONFIDENCE: 미설정·빈 값은 기본값, 0~1은 그대로."""
    import config

    if raw is None:
        monkeypatch.delenv("LIPNET_OLKAVS_MIN_CONFIDENCE", raising=False)
    else:
        monkeypatch.setenv("LIPNET_OLKAVS_MIN_CONFIDENCE", raw)
    assert config._env_unit_float("LIPNET_OLKAVS_MIN_CONFIDENCE", 0.0) == expected


@pytest.mark.parametrize("raw", ["nan", "NaN", "-0.1", "1.5", "inf", "abc"])
def test_min_confidence_env_rejects_invalid(monkeypatch, raw) -> None:
    """NaN·무한대·범위 밖·숫자 아님은 기동 시 거부한다."""
    import config

    monkeypatch.setenv("LIPNET_OLKAVS_MIN_CONFIDENCE", raw)
    with pytest.raises(ValueError):
        config._env_unit_float("LIPNET_OLKAVS_MIN_CONFIDENCE", 0.0)


def test_record_config_limits() -> None:
    """녹화 접수 기준값: 30초 미만·0.5초 이상, 크기 상한, 기본 확신 기준(판정 끔)이 0~1 범위."""
    import config

    assert config.RECORD_MAX_SECONDS == 30.0 and config.RECORD_MIN_SECONDS == 0.5
    assert config.RECORD_MAX_SIZE_MB > 0 and config.RECORD_MAX_FRAMES > 0
    assert 0.0 <= config.OLKAVS_MIN_CONFIDENCE <= 1.0


# ---------------------------------------------------------------------------
# 6단계 보완: ROI 요청의 길이 힌트(duration_seconds), 차례가 오면 바로 'processing'
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("video_seconds, expected", [
    (8.5, 8.5), (3, 3.0), (None, None), (0.0, None), (-1.0, None), (float("nan"), None), (float("inf"), None),
    (True, None),
])
def test_transcribe_video_sends_duration_hint_only_when_valid(tmp_path, video_seconds, expected) -> None:
    """유한한 양수 video_seconds만 ROI 요청의 'duration_seconds'로 보내고, 아니면 필드를 빼 둔다."""
    segments = [{"index": 0, "start_ms": 0, "end_ms": 1000, "status": "ok", "npz": "a.npz"}]
    roi = _StubWorker(_roi_result(segments))
    service = svc.OlkavsService(roi, _StubWorker(lambda p, _p: {"ok": True, "text": "가", "confidence": 0.9}))
    service.transcribe_video("v.mp4", str(tmp_path), video_seconds=video_seconds)
    if expected is None:
        assert "duration_seconds" not in roi.calls[0]
    else:
        assert roi.calls[0]["duration_seconds"] == expected and isinstance(roi.calls[0]["duration_seconds"], float)


def test_transcribe_video_moves_job_to_processing_when_gate_acquired(tmp_path) -> None:
    """FIFO 대기 중에는 'pending'(0%), 차례가 오면 ROI 진행률이 없어도(fraction None) 바로 'processing' 0%.

    ROI 단계 동안 진행률은 _ROI_SHARE를 넘지 않고 끝까지 단조 증가한다.
    """
    from services import job_manager as jm

    session_id = "7c1e2d3f-4a5b-4c6d-8e9f-0a1b2c3d4e5f"
    segments = [{"index": 0, "start_ms": 0, "end_ms": 1000, "status": "ok", "npz": "a.npz"}]
    seen_during_roi: list[tuple[str, float]] = []

    def roi_handler(payload, on_progress):
        record = jm.get_job(session_id)
        seen_during_roi.append((record.status, record.progress))
        for _ in range(3):
            on_progress({"type": "progress", "fraction": None})  # 메타데이터 없는 녹화
        record = jm.get_job(session_id)
        seen_during_roi.append((record.status, record.progress))
        return {"ok": True, "segments": segments, "selection": {}}

    service = svc.OlkavsService(_StubWorker(roi_handler),
                                _StubWorker(lambda p, _p: {"ok": True, "text": "가", "confidence": 0.9}))
    progress: list[float] = []

    def on_progress(value: float) -> None:
        progress.append(value)
        jm.update_progress(session_id, value)

    jm.create_job(session_id)
    holder = service.gate.acquire()
    try:
        worker = threading.Thread(target=service.transcribe_video, args=("v.mp4", str(tmp_path)),
                                  kwargs={"on_progress": on_progress, "video_seconds": 3.0})
        worker.start()
        time.sleep(0.3)
        record = jm.get_job(session_id)
        assert (record.status, record.progress) == (jm.STATUS_PENDING, 0.0)  # 아직 차례가 아니다
        assert progress == []
        service.gate.release(holder)
        holder = None
        worker.join(10)
        assert not worker.is_alive()
        assert seen_during_roi == [(jm.STATUS_PROCESSING, 0.0), (jm.STATUS_PROCESSING, 0.0)]
        assert progress[0] == 0.0 and progress == sorted(progress) and progress[-1] == 100.0
        assert all(p <= svc._ROI_SHARE for p in progress[:-1])
    finally:
        if holder is not None:
            service.gate.release(holder)
        jm.clear_job(session_id)
