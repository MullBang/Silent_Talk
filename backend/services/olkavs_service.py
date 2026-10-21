"""OLKAVS 상주 워커 관리와 업로드 영상 인식 (서버 프로세스 쪽 클라이언트).

서버(FastAPI)는 torch·mediapipe 공식 모델 환경을 직접 import 하지 않는다. 대신
services/olkavs_worker.py 를 두 가상환경에서 서브프로세스로 띄우고 JSON 줄 규약으로 요청한다.

  roi   워커 (.venv-olkavs-roi): 영상 → 입술 NPZ 창들 (tmp/roi/<session_id>/)
  infer 워커 (.venv-olkavs)    : NPZ → 텍스트·confidence·raw_score

- 워커는 한 번 띄우면 모델·검출기를 메모리에 둔 채 계속 쓴다. 죽었으면 다음 요청 때 다시 띄운다.
- 요청마다 시간 상한을 둔다(넘으면 워커를 종료해 다음에 새로 띄움).
- 영상 작업은 FIFO 순서로 하나씩 처리한다(단일 GPU·직렬 워커에서 먼저 온 작업이 먼저 끝나게).
- 사용자에게 보내는 오류 문구는 고정 문구만 쓴다. 로그 꼬리·경로 같은 상세는 서버 로그에만 남긴다.
- 결과 표시 규약(6단계): 입술 미검출 구간 FAIL_TEXT, 말하지 않은 구간 NO_SPEECH_TEXT(인식하지 않음),
  빈 출력 또는 confidence < config.OLKAVS_MIN_CONFIDENCE 이면 UNREC_TEXT(후자는 모델 문장을 candidate로 보존).
"""
from __future__ import annotations

import json
import logging
import math
import os
import queue
import subprocess
import sys
import threading
import time
from typing import Any, Callable

_BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

import config  # noqa: E402

logger = logging.getLogger(__name__)

FAIL_TEXT = "[검출 실패 구간]"
UNREC_TEXT = "[인식 불가]"
NO_SPEECH_TEXT = "[발화 없음]"

# 진행률 배분: 입술 추출 0~40%, 인식 40~100%
_ROI_SHARE = 40.0

MSG_START_FAILED = "인식 모델 워커를 시작하지 못했습니다. 서버 설정(가상환경·가중치)을 확인하세요."
MSG_TIMEOUT = "인식 처리 시간이 초과되었습니다. 더 짧은 영상으로 다시 시도해 주세요."
MSG_CRASHED = "인식 워커가 비정상 종료되었습니다. 다시 시도해 주세요."
MSG_VIDEO_UNREADABLE = "영상을 읽을 수 없습니다. 다른 파일로 다시 시도해 주세요."
MSG_FAILED = "영상을 처리하지 못했습니다. 다시 시도해 주세요."
MSG_NO_FRAMES = "영상에서 처리할 프레임을 찾지 못했습니다."
MSG_ABORTED = "작업 시간이 초과되어 중단했습니다."
MSG_BUSY = "앞선 작업이 많아 제한 시간 안에 처리할 수 없습니다. 잠시 후 다시 시도해 주세요."
MSG_LOW_FPS = ("영상 프레임률이 낮아(약 {fps:.0f}fps) 인식 정확도가 떨어질 수 있습니다. "
               "30fps로 녹화한 영상을 권장합니다.")


class WorkerError(RuntimeError):
    """워커 기동·통신·처리 실패.

    str(exc)와 user_message는 사용자에게 보여도 되는 고정 문구이고, detail은 서버 로그용 상세다.
    """

    def __init__(self, user_message: str, detail: str = ""):
        super().__init__(user_message)
        self.user_message = user_message
        self.detail = detail


class JobAborted(RuntimeError):
    """작업이 외부(타임아웃 등)에서 중단됨."""

    def __init__(self, message: str = MSG_ABORTED):
        super().__init__(message)
        self.user_message = message


def _user_message_for(error: dict[str, Any]) -> str:
    """워커가 보낸 오류를 사용자용 고정 문구로 바꾼다 (경로·파일명을 내보내지 않는다)."""
    message = str(error.get("message") or "")
    if "영상을 열 수 없습니다" in message or "프레임을 읽지 못했습니다" in message:
        return MSG_VIDEO_UNREADABLE
    if "PTS가 단조 증가하지 않습니다" in message:
        return MSG_VIDEO_UNREADABLE
    return MSG_FAILED


class WorkerProcess:
    """JSON 줄 규약으로 통신하는 상주 서브프로세스 하나.

    표준출력은 별도 스레드가 읽어 큐에 넣는다(시간 상한을 걸기 위해). 표준오류는 로그 파일로 보낸다.
    """

    def __init__(self, name: str, command: list[str], log_path: str, start_timeout: float,
                 cwd: str | None = None, env: dict[str, str] | None = None,
                 fatal_cooldown: float = config.OLKAVS_FATAL_COOLDOWN_SEC,
                 max_consecutive_failures: int = config.OLKAVS_MAX_CONSECUTIVE_FAILURES):
        self.name = name
        self.command = command
        self.log_path = log_path
        self.start_timeout = start_timeout
        self.cwd = cwd
        self.env = env
        self.fatal_cooldown = fatal_cooldown
        self.max_consecutive_failures = max_consecutive_failures
        self.info: dict[str, Any] | None = None
        self.last_error: str | None = None
        self._fatal_at: float | None = None
        self._consecutive_failures = 0
        self._proc: subprocess.Popen | None = None
        self._queue: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._next_id = 0
        self._log_handle = None

    # -- 수명주기 ---------------------------------------------------------------
    def alive(self) -> bool:
        """프로세스가 살아 있고 기동을 마쳤는지."""
        return self._proc is not None and self._proc.poll() is None and self.info is not None

    def state(self) -> str:
        """stopped | ready | dead."""
        if self._proc is None:
            return "stopped"
        return "ready" if self.alive() else "dead"

    def start(self) -> dict[str, Any]:
        """프로세스를 띄우고 ready 메시지를 기다린다. 이미 살아 있으면 그대로 둔다."""
        with self._lock:
            return self._start_locked()

    def _fail(self, user_message: str, detail: str, code: str | None) -> WorkerError:
        """상세는 서버 로그에만 남기고, 사용자용 고정 문구의 예외를 만든다.

        code가 있으면(워커 자체의 상태 오류) /health에 보일 짧은 상태 코드와 시각만 last_error에 둔다.
        요청 하나의 처리 실패(code=None)는 워커 상태가 아니므로 last_error를 바꾸지 않는다.
        """
        if code is not None:
            self.last_error = f"{code} at {time.strftime('%Y-%m-%d %H:%M:%S')}"
        logger.error("%s 워커: %s", self.name, detail)
        return WorkerError(user_message, detail)

    def _start_locked(self) -> dict[str, Any]:
        if self.alive():
            return self.info or {}
        if self._fatal_at is not None and time.monotonic() - self._fatal_at < self.fatal_cooldown:
            raise WorkerError(MSG_START_FAILED, f"{self.name} 기동 실패 후 재시도 대기 중 ({self.last_error})")
        self._kill_locked()
        if not os.path.exists(self.command[0]):
            self._fatal_at = time.monotonic()
            raise self._fail(MSG_START_FAILED, f"인터프리터가 없습니다: {self.command[0]}", "start_failed")
        try:
            os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
            self._log_handle = open(self.log_path, "ab")  # noqa: SIM115 - 프로세스 수명 동안 유지
            self._log_handle.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} start {self.name} =====\n".encode())
            self._log_handle.flush()
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            self._queue = queue.Queue()
            self._proc = subprocess.Popen(  # noqa: S603 - 고정된 프로젝트 인터프리터와 스크립트만 실행
                self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log_handle,
                cwd=self.cwd, env=self.env, creationflags=creationflags,
            )
        except OSError as exc:
            self._kill_locked()
            self._fatal_at = time.monotonic()
            raise self._fail(MSG_START_FAILED, f"프로세스 생성 실패: {exc}", "start_failed") from exc
        threading.Thread(target=self._reader, args=(self._proc, self._queue), name=f"{self.name}-reader",
                         daemon=True).start()
        try:
            message = self._wait_for(lambda m: m.get("type") in ("ready", "fatal"), self.start_timeout, None,
                                     MSG_START_FAILED, "start_failed")
        except WorkerError:
            self._fatal_at = time.monotonic()
            raise
        if message.get("type") == "fatal":
            error = message.get("error") or {}
            self._kill_locked()
            self._fatal_at = time.monotonic()
            raise self._fail(MSG_START_FAILED, f"기동 실패 — {error.get('type')}: {error.get('message')}",
                             "start_failed")
        self.info = message.get("info") or {}
        self.last_error = None
        self._fatal_at = None
        self._consecutive_failures = 0
        logger.info("%s 워커 준비됨: %s", self.name, self.info)
        return self.info

    @staticmethod
    def _reader(proc: subprocess.Popen, out: queue.Queue) -> None:
        """표준출력 줄을 읽어 큐에 넣는다. EOF면 None을 넣는다. 객체가 아닌 JSON은 규약 밖 출력으로 본다."""
        try:
            for raw in iter(proc.stdout.readline, b""):
                try:
                    message = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, ValueError):
                    message = None
                if not isinstance(message, dict):
                    message = {"type": "garbage", "raw": raw[:200].decode("utf-8", "replace")}
                out.put(message)
        except (OSError, ValueError):
            pass
        finally:
            out.put(None)

    def _wait_for(self, accept: Callable[[dict], bool], timeout: float,
                  on_progress: Callable[[dict], None] | None, timeout_message: str = MSG_TIMEOUT,
                  timeout_code: str = "timeout") -> dict[str, Any]:
        """조건에 맞는 메시지가 올 때까지 기다린다. 시간 초과·종료 시 워커를 정리하고 예외."""
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._kill_locked()
                raise self._fail(timeout_message, f"{timeout:.0f}초 안에 응답이 없어 워커를 종료했습니다", timeout_code)
            try:
                message = self._queue.get(timeout=min(remaining, 1.0))
            except queue.Empty:
                continue
            if message is None:
                code = self._proc.poll() if self._proc else None
                tail = self._log_tail()
                self._kill_locked()
                raise self._fail(MSG_CRASHED, f"워커가 예기치 않게 종료됨(code={code}). 로그 끝: {tail}", "crashed")
            if message.get("type") == "progress":
                if on_progress is not None:
                    try:
                        on_progress(message)
                    except Exception:  # noqa: BLE001 - 진행률 콜백 오류가 요청을 깨지 않게 한다
                        logger.exception("진행률 콜백 오류")
                continue
            if message.get("type") == "garbage":
                logger.warning("%s 워커의 규약 밖 출력: %s", self.name, message.get("raw"))
                continue
            if accept(message):
                return message

    def request(self, payload: dict[str, Any], timeout: float,
                on_progress: Callable[[dict], None] | None = None) -> dict[str, Any]:
        """요청 하나를 보내고 결과를 기다린다. 워커가 없으면 먼저 띄운다."""
        with self._lock:
            self._start_locked()
            self._next_id += 1
            rid = self._next_id
            line = json.dumps({**payload, "id": rid}, ensure_ascii=True) + "\n"
            try:
                self._proc.stdin.write(line.encode("ascii"))
                self._proc.stdin.flush()
            except (OSError, ValueError) as exc:
                self._kill_locked()
                raise self._fail(MSG_CRASHED, f"요청 전송 실패: {exc}", "send_failed") from exc
            message = self._wait_for(lambda m: m.get("type") == "result" and m.get("id") == rid, timeout, on_progress)
            if message.get("ok"):
                self._consecutive_failures = 0
                return message
            error = message.get("error") or {}
            self._consecutive_failures += 1
            detail = f"처리 실패 — {error.get('type')}: {error.get('message')}"
            code = None
            if message.get("worker_poisoned") or self._consecutive_failures >= self.max_consecutive_failures:
                # CUDA 컨텍스트 손상 등으로 계속 실패할 워커는 종료해 다음 요청에서 새로 띄운다.
                code = "poisoned_restart" if message.get("worker_poisoned") else "repeated_failures_restart"
                detail += f" (워커 재기동: {code}, 연속 실패 {self._consecutive_failures}회)"
                self._kill_locked()
                self._consecutive_failures = 0
            raise self._fail(_user_message_for(error), detail, code)

    def close(self, timeout: float = 5.0) -> None:
        """정상 종료를 요청하고, 응답이 없으면 강제 종료한다."""
        with self._lock:
            proc = self._proc
            if proc is not None and proc.poll() is None:
                try:
                    proc.stdin.write(b'{"id": 0, "cmd": "shutdown"}\n')
                    proc.stdin.flush()
                    proc.stdin.close()
                    proc.wait(timeout=timeout)
                except (OSError, ValueError, subprocess.TimeoutExpired):
                    pass
            self._kill_locked()

    def _kill_locked(self) -> None:
        """프로세스와 로그 핸들을 정리한다 (잠금 보유 상태에서 호출)."""
        proc, self._proc = self._proc, None
        self.info = None
        if proc is not None:
            try:
                if proc.poll() is None:
                    _kill_tree(proc)
                proc.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                pass
            for stream in (proc.stdin, proc.stdout):
                try:
                    if stream is not None:
                        stream.close()
                except OSError:
                    pass
        if self._log_handle is not None:
            try:
                self._log_handle.close()
            except OSError:
                pass
            self._log_handle = None

    def _log_tail(self, lines: int = 8) -> str:
        """서버 로그용 워커 로그 마지막 몇 줄 (사용자에게 보내지 않는다)."""
        try:
            with open(self.log_path, "rb") as handle:
                handle.seek(0, os.SEEK_END)
                size = handle.tell()
                handle.seek(max(0, size - 4000))
                text = handle.read().decode("utf-8", "replace")
            return " | ".join(text.strip().splitlines()[-lines:])
        except OSError:
            return f"로그를 읽을 수 없습니다: {self.log_path}"


def _kill_tree(proc: subprocess.Popen) -> None:
    """프로세스와 자식을 함께 종료한다.

    Windows의 venv Scripts/python.exe는 실제 인터프리터를 자식 프로세스로 띄우는 런처다.
    런처의 작업 객체가 보통 자식도 정리하지만, 확실히 하려고 taskkill /T 로 트리 전체를 끝낸다.
    """
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],  # noqa: S603,S607 - 고정 명령
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if proc.poll() is None:
            proc.kill()
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("워커 프로세스 종료 실패(pid=%s): %s", proc.pid, exc)


def _worker_env() -> dict[str, str]:
    """워커용 환경변수: 서버 쪽 경로 설정이 섞이지 않게 정리한다."""
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV")}
    env.update(PYTHONIOENCODING="utf-8", PYTHONNOUSERSITE="1", PYTHONUNBUFFERED="1",
               TF_CPP_MIN_LOG_LEVEL="2", GLOG_minloglevel="2")
    return env


class FifoGate:
    """먼저 온 작업부터 하나씩 통과시키는 대기열. 기다리는 동안 중단 신호를 확인한다."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._next_ticket = 0
        self._serving = 0
        self._cancelled: set[int] = set()

    def waiting(self) -> int:
        """현재 처리 중·대기 중인 작업 수."""
        with self._cond:
            # 차례가 오기 전에 취소된 티켓은 실제 대기 작업이 아니다.
            return self._next_ticket - self._serving - len(self._cancelled)

    def acquire(self, should_abort: Callable[[], bool] | None = None, poll: float = 0.5) -> int:
        """차례가 올 때까지 기다린다. 중단되면 순서를 넘기고 JobAborted."""
        with self._cond:
            ticket = self._next_ticket
            self._next_ticket += 1
            while self._serving != ticket:
                if should_abort is not None and should_abort():
                    self._skip(ticket)
                    raise JobAborted()
                self._cond.wait(timeout=poll)
            return ticket

    def release(self, ticket: int) -> None:
        """차례를 다음 작업에 넘긴다."""
        with self._cond:
            self._skip(ticket)

    def _skip(self, ticket: int) -> None:
        """티켓을 끝낸다. 차례가 아직 안 온 취소 티켓은 표시해 두었다가 건너뛴다 (잠금 보유 상태)."""
        if self._serving == ticket:
            self._serving += 1
            while self._serving in self._cancelled:
                self._cancelled.discard(self._serving)
                self._serving += 1
        else:
            self._cancelled.add(ticket)
        self._cond.notify_all()


def _duration_hint(video_seconds: float | None) -> float | None:
    """ROI 요청의 'duration_seconds' 값: 유한한 양수 길이(초)면 float, 아니면 None(필드를 보내지 않는다)."""
    try:
        if video_seconds is None or isinstance(video_seconds, bool):
            return None
        value = float(video_seconds)
        return value if math.isfinite(value) and value > 0.0 else None
    except (TypeError, ValueError):
        return None


def expected_processing_seconds(video_seconds: float) -> float:
    """영상 길이로 예상 처리 시간을 어림한다 (입장 제어용, 실측: 180초 영상 약 215초)."""
    return (config.OLKAVS_EXPECTED_OVERHEAD_SEC
            + config.OLKAVS_EXPECTED_SECONDS_PER_VIDEO_SECOND * max(0.0, float(video_seconds)))


class OlkavsService:
    """두 워커를 묶어 업로드 영상 한 개를 시간 구간별 텍스트로 바꾼다."""

    def __init__(self, roi_worker: WorkerProcess | None = None, infer_worker: WorkerProcess | None = None):
        log_dir = os.path.join(config.LOG_DIR, "olkavs")
        self.roi = roi_worker or WorkerProcess(
            "roi", [config.OLKAVS_ROI_PYTHON, "-u", config.OLKAVS_WORKER_SCRIPT, "--mode", "roi"],
            os.path.join(log_dir, "roi_worker.log"), config.OLKAVS_START_TIMEOUT_SEC,
            cwd=config.PROJECT_ROOT, env=_worker_env())
        self.infer = infer_worker or WorkerProcess(
            "infer", [config.OLKAVS_INFER_PYTHON, "-u", config.OLKAVS_WORKER_SCRIPT, "--mode", "infer",
                      "--device", config.OLKAVS_DEVICE, "--decoder", config.OLKAVS_DECODER, "--warmup"],
            os.path.join(log_dir, "infer_worker.log"), config.OLKAVS_START_TIMEOUT_SEC,
            cwd=config.PROJECT_ROOT, env=_worker_env())
        self.gate = FifoGate()
        self._warm_thread: threading.Thread | None = None

    def start(self) -> dict[str, Any]:
        """두 워커를 띄운다(이미 떠 있으면 그대로). 기동 정보를 돌려준다."""
        return {"roi": self.roi.start(), "infer": self.infer.start()}

    def warm_start_async(self) -> None:
        """서버 시작을 막지 않도록 백그라운드에서 워커를 띄운다. 실패는 로그만 남기고 다음 요청 때 재시도."""
        def run() -> None:
            try:
                self.start()
            except WorkerError as exc:
                logger.error("OLKAVS 워커 사전 기동 실패: %s", exc.detail or exc)
        if self._warm_thread is None or not self._warm_thread.is_alive():
            self._warm_thread = threading.Thread(target=run, name="olkavs-warm-start", daemon=True)
            self._warm_thread.start()

    def status(self) -> dict[str, Any]:
        """/health 용 상태. 워커를 새로 띄우지 않는다. 상세 오류 대신 짧은 상태 코드만 보인다."""
        infer_info = self.infer.info or {}
        return {
            "roi_worker": self.roi.state(), "infer_worker": self.infer.state(),
            "device": infer_info.get("device"), "cuda_device": infer_info.get("cuda_device"),
            "decoder": infer_info.get("decoder", config.OLKAVS_DECODER),
            "checkpoint_sha256": infer_info.get("checkpoint_sha256"),
            "upstream_commit": infer_info.get("upstream_commit"),
            "roi_profile": (self.roi.info or {}).get("detector_profile"),
            "jobs_in_queue": self.gate.waiting(),
            "last_error": {"roi": self.roi.last_error, "infer": self.infer.last_error},
        }

    def model_info(self) -> dict[str, Any]:
        """결과와 함께 보여 줄 인식 모델 정보. 워커를 띄우지 않으며, 기동 전이면 값이 None일 수 있다.

        Returns:
            {backend, decoder, checkpoint_sha256, upstream_commit, roi_profile}.
        """
        infer_info = self.infer.info or {}
        roi_info = self.roi.info or {}
        return {
            "backend": "olkavs",
            "decoder": infer_info.get("decoder", config.OLKAVS_DECODER),
            "checkpoint_sha256": infer_info.get("checkpoint_sha256"),
            "upstream_commit": infer_info.get("upstream_commit"),
            "roi_profile": roi_info.get("detector_profile"),
        }

    def shutdown(self) -> None:
        """두 워커를 종료한다."""
        for worker in (self.roi, self.infer):
            try:
                worker.close()
            except Exception:  # noqa: BLE001 - 종료 중 오류는 무시
                logger.exception("%s 워커 종료 중 오류", worker.name)

    def transcribe_video(self, video_path: str, work_dir: str,
                         on_progress: Callable[[float], None] | None = None,
                         should_abort: Callable[[], bool] | None = None,
                         remaining_seconds: Callable[[], float | None] | None = None,
                         video_seconds: float | None = None,
                         warnings: list[str] | None = None,
                         utterance: bool = False) -> list[dict[str, Any]]:
        """영상 → [{start_ms, end_ms, text, confidence, raw_score[, candidate]}] (시간 순서).

        - 얼굴(입술) 미검출 구간은 FAIL_TEXT, 빈 인식 결과는 UNREC_TEXT(점수 없음)로 표시한다.
        - 입 움직임으로 판정한 무발화 창(status 'no_speech')은 인식하지 않고 NO_SPEECH_TEXT로 표시한다.
        - confidence가 config.OLKAVS_MIN_CONFIDENCE보다 낮으면 UNREC_TEXT로 바꾸고 점수와 모델 문장(candidate)을 남긴다.
        - utterance=True(웹캠 녹화)는 영상 전체를 한 발화로 보고 ROI 워커가 앞뒤 무발화 구간을 잘라 낸다.
          12초 모델 창은 그대로라 12초를 넘는 녹화는 여러 창으로 나뉠 수 있다.
        - 작업은 FIFO로 하나씩 처리하며, 대기 중·입술 추출 전·창 사이마다 중단 신호를 확인한다.
        - 차례가 왔을 때 남은 Job 시간(remaining_seconds)이 예상 처리 시간(video_seconds 기준)보다 짧으면
          입술 추출을 시작하지 않고 바로 안내 문구로 끝내 다음 작업에 워커를 넘긴다.
        - warnings에는 사용자에게 보여 줄 품질 경고(낮은 프레임률 등)를 덧붙인다.
        - video_seconds가 유한한 양수면 ROI 요청에 'duration_seconds'로 보내 메타데이터 없는 영상의 진행률 분모로 쓴다.
        - FIFO 차례가 오면 바로 진행률 0을 알려 Job을 'processing'으로 옮긴다(진행률은 단조 증가).
        """
        last = {"progress": 0.0}

        def check_abort() -> None:
            if should_abort is not None and should_abort():
                raise JobAborted()

        def report(value: float) -> None:
            value = max(last["progress"], max(0.0, min(100.0, value)))  # 진행률은 뒤로 가지 않는다
            last["progress"] = value
            if on_progress is not None:
                on_progress(value)

        def roi_progress(message: dict) -> None:
            fraction = message.get("fraction")
            if isinstance(fraction, (int, float)):
                report(_ROI_SHARE * float(fraction))

        ticket = self.gate.acquire(should_abort)
        try:
            check_abort()
            # 차례가 왔다: Job을 'pending'에서 'processing'(0%)으로 바로 옮긴다. ROI 진행률이 오기 전에는
            # 0%를 넘지 않는다(길이 메타데이터 없는 녹화도 대기 끝을 알 수 있게).
            report(0.0)
            if remaining_seconds is not None and video_seconds is not None:
                remaining = remaining_seconds()
                expected = expected_processing_seconds(video_seconds)
                if remaining is not None and expected > remaining:
                    logger.warning("남은 Job 시간 %.0f초 < 예상 처리 %.0f초 → 시작하지 않음", remaining, expected)
                    raise JobAborted(MSG_BUSY)
            # 인식 워커가 준비되지 않았으면 영상 전체 입술 추출을 하기 전에 바로 실패한다.
            self.infer.start()
            self.roi.start()
            check_abort()
            roi_request: dict[str, Any] = {"cmd": "roi", "video": os.path.abspath(video_path),
                                           "out_dir": os.path.abspath(work_dir),
                                           "segment_seconds": config.OLKAVS_SEGMENT_SECONDS,
                                           "min_window_seconds": config.OLKAVS_MIN_WINDOW_SECONDS,
                                           "cut_search_seconds": config.OLKAVS_CUT_SEARCH_SECONDS,
                                           "utterance": bool(utterance)}
            duration_hint = _duration_hint(video_seconds)
            if duration_hint is not None:
                # 컨테이너에 길이 메타데이터가 없을 때(MediaRecorder WebM) ROI 진행률의 분모로 쓴다.
                roi_request["duration_seconds"] = duration_hint
            roi = self.roi.request(roi_request, timeout=config.OLKAVS_ROI_TIMEOUT_SEC, on_progress=roi_progress)
            segments = roi.get("segments") or []
            if not segments:
                raise WorkerError(MSG_NO_FRAMES, "입술 추출 결과 구간이 없습니다")
            selection = roi.get("selection") or {}
            if selection.get("reason"):
                logger.warning("프레임 선택: %s (%s)", selection.get("mode"), selection.get("reason"))
            if selection.get("low_fps") and warnings is not None:
                warnings.append(MSG_LOW_FPS.format(fps=float(selection.get("source_fps") or 0.0)))
            speech = roi.get("speech")
            if isinstance(speech, dict) and speech.get("enabled"):
                logger.info("발화 구간 판정(utterance=%s): detected=%s %s~%sms active=%.2fs trim=%s/%sms",
                            bool(utterance), speech.get("detected"), speech.get("start_ms"), speech.get("end_ms"),
                            float(speech.get("active_seconds") or 0.0), speech.get("trimmed_head_ms"),
                            speech.get("trimmed_tail_ms"))
            min_confidence = float(config.OLKAVS_MIN_CONFIDENCE)
            report(_ROI_SHARE)
            results: list[dict[str, Any]] = []
            for i, seg in enumerate(segments):
                check_abort()
                item: dict[str, Any] = {"start_ms": int(seg["start_ms"]), "end_ms": int(seg["end_ms"]),
                                        "text": FAIL_TEXT, "confidence": None, "raw_score": None}
                if seg.get("status") == "no_speech":
                    item["text"] = NO_SPEECH_TEXT  # 말하지 않은 구간은 인식하지 않는다
                elif seg.get("status") == "ok" and seg.get("npz"):
                    out = self.infer.request({"cmd": "infer", "npz": seg["npz"]},
                                             timeout=config.OLKAVS_INFER_TIMEOUT_SEC)
                    if out.get("search_cap_reached"):
                        logger.warning("hybrid 탐색 상한 도달 — 창 끝부분이 잘렸을 수 있음 (%s~%sms)",
                                       item["start_ms"], item["end_ms"])
                    confidence = out.get("confidence")
                    if not out.get("text"):
                        item["text"] = UNREC_TEXT  # 글자가 나오지 않으면 점수도 붙이지 않는다
                    elif isinstance(confidence, (int, float)) and float(confidence) < min_confidence:
                        # 확신이 낮으면 결과처럼 보이지 않게 '인식 불가'로 두고 모델 문장은 candidate로 남긴다.
                        item.update(text=UNREC_TEXT, confidence=confidence, raw_score=out.get("raw_score"),
                                    candidate=out["text"])
                    else:
                        item.update(text=out["text"], confidence=confidence, raw_score=out.get("raw_score"))
                results.append(item)
                report(_ROI_SHARE + (100.0 - _ROI_SHARE) * (i + 1) / len(segments))
            return results
        finally:
            self.gate.release(ticket)


_service: OlkavsService | None = None
_service_lock = threading.Lock()


def get_service() -> OlkavsService:
    """프로세스 전역 서비스 인스턴스 (지연 생성)."""
    global _service
    with _service_lock:
        if _service is None:
            _service = OlkavsService()
        return _service


def peek_service() -> OlkavsService | None:
    """이미 만들어진 인스턴스만 돌려준다 (/health가 워커를 띄우지 않도록)."""
    return _service


def shutdown_service() -> None:
    """서버 종료 시 워커 정리."""
    global _service
    with _service_lock:
        service, _service = _service, None
    if service is not None:
        service.shutdown()
