"""OLKAVS 상주 워커 프로세스 진입점 (서버가 서브프로세스로 실행한다).

torch와 mediapipe를 한 프로세스에 함께 올리면 segfault가 날 수 있고 두 가상환경의 라이브러리
버전도 다르므로, 서버(FastAPI) 옆에 워커 두 개를 따로 띄운다.

  --mode roi   : .venv-olkavs-roi  (mediapipe 0.10.21)  영상 → 구간별 입술 96x96 BGR NPZ
  --mode infer : .venv-olkavs      (torch)              NPZ → 텍스트·confidence·raw_score

이 파일은 backend 패키지(config 등)를 import 하지 않는다. 워커 가상환경에는 fastapi 등이 없다.
공용 코어는 scripts/olkavs_transcribe.py 를 경로로 불러온다.

통신 규약(한 줄 = JSON 하나, UTF-8, ASCII escape):
  서버 → 워커  {"id": n, "cmd": "ping"|"roi"|"infer"|"shutdown", ...}
  워커 → 서버  {"type": "ready", "info": {...}}                      (기동 완료 1회)
               {"type": "fatal", "error": {...}}                     (기동 실패 후 종료)
               {"type": "progress", "id": n, ...}                    (진행 상황, 0회 이상)
               {"type": "result", "id": n, "ok": true|false, ...}    (요청당 정확히 1회)
표준출력은 규약 전용이다. 라이브러리가 찍는 출력은 모두 표준오류(서버가 로그 파일로 보냄)로 돌린다.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, BinaryIO

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _claim_protocol_stdout() -> BinaryIO:
    """원래 표준출력을 규약 전용 스트림으로 떼어 내고, fd 1을 표준오류로 돌린다."""
    try:
        sys.stdout.flush()
        proto = os.fdopen(os.dup(1), "wb", buffering=0)
        os.dup2(2, 1)
        sys.stdout = sys.stderr
        return proto
    except OSError as exc:
        raise RuntimeError(f"규약 표준출력 분리 실패: {exc}") from exc


class Channel:
    """규약 메시지를 한 줄씩 쓰는 출력 채널."""

    def __init__(self, stream: BinaryIO):
        self.stream = stream

    def send(self, message: dict[str, Any]) -> None:
        """JSON 한 줄을 쓴다 (ASCII escape로 인코딩 문제를 피한다)."""
        try:
            self.stream.write((json.dumps(message, ensure_ascii=True) + "\n").encode("ascii"))
            self.stream.flush()
        except (OSError, ValueError) as exc:
            # 서버 쪽 파이프가 닫혔으면 더 할 일이 없다.
            print(f"[worker] 규약 출력 실패: {exc}", file=sys.stderr, flush=True)
            raise SystemExit(3) from exc


def _error(exc: BaseException) -> dict[str, Any]:
    """예외를 규약 오류 객체로 바꾼다 (스택은 표준오류 로그에만 남긴다)."""
    traceback.print_exc(file=sys.stderr)
    return {"type": type(exc).__name__, "message": str(exc)}


def _load_core() -> Any:
    """scripts/olkavs_transcribe.py 를 공용 코어로 불러온다."""
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        import olkavs_transcribe as core  # noqa: PLC0415
        return core
    except ImportError as exc:
        raise RuntimeError(f"공용 코어를 불러오지 못했습니다: {exc}") from exc


def _init_roi(core: Any) -> dict[str, Any]:
    """입술 추출 워커 기동 확인: mediapipe 버전과 검출기 생성."""
    try:
        import cv2
        import mediapipe as mp
        sys.path.insert(0, str(core.OLKAVS))
        from automatic_roi import PROFILE_ID
        if mp.__version__ != "0.10.21":
            raise RuntimeError(f"입술 추출 규약은 mediapipe 0.10.21이 필요합니다: {mp.__version__}")
        probe = core.vad_tracker_class()()  # 서비스가 실제로 쓰는 추적기(입 벌림 비율 포함)로 기동 확인
        probe.close()
        return {"mode": "roi", "mediapipe": mp.__version__, "cv2": cv2.__version__, "detector_profile": PROFILE_ID,
                "geometry_arm": core.GEOMETRY_ARM, "vad_profile": core.vad_profile(),
                "python": sys.version.split()[0]}
    except ImportError as exc:
        raise RuntimeError(f"입술 추출 환경이 아닙니다(.venv-olkavs-roi 필요): {exc}") from exc


def _init_infer(core: Any, device: str, decoder: str, warmup: bool) -> dict[str, Any]:
    """인식 워커 기동: 장치 결정, 고정 commit·가중치 SHA 검사 후 모델 로드, 예열 추론."""
    try:
        import numpy as np
        import torch
    except ImportError as exc:
        raise RuntimeError(f"인식 환경이 아닙니다(.venv-olkavs 필요): {exc}") from exc
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    elif device.startswith("cuda") and not torch.cuda.is_available():
        print("[worker] CUDA를 쓸 수 없어 cpu로 실행합니다", file=sys.stderr, flush=True)
        device = "cpu"
    if device == "cpu":
        torch.set_num_threads(max(1, min(8, os.cpu_count() or 4)))
    started = time.perf_counter()
    core.load_olkavs(device)
    load_seconds = time.perf_counter() - started
    warm_seconds = None
    if warmup:
        dummy = (np.random.RandomState(0).rand(30, 96, 96, 3) * 255).astype(np.uint8)
        started = time.perf_counter()
        core.recognize(dummy, device, "ctc")
        warm_seconds = time.perf_counter() - started
    return {"mode": "infer", "device": device, "decoder": decoder,
            "cuda_device": torch.cuda.get_device_name(0) if device.startswith("cuda") else None,
            "torch": torch.__version__, "checkpoint_sha256": core.CHECKPOINT_SHA256,
            "upstream_commit": core.EXPECTED_COMMIT, "model_load_seconds": load_seconds,
            "warmup_seconds": warm_seconds, "python": sys.version.split()[0]}


def _duration_seconds(request: dict[str, Any]) -> float | None:
    """요청의 선택 항목 "duration_seconds"(서버가 잰 영상 길이, 초). 없거나 null이면 None.

    유한한 양수(bool 제외)가 아니면 ValueError로 요청을 거부한다(ok:false).
    """
    value = request.get("duration_seconds")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("duration_seconds는 양수 실수 또는 null이어야 합니다")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError("duration_seconds는 유한한 양수여야 합니다")
    return number


def _handle_roi(core: Any, request: dict[str, Any], channel: Channel) -> dict[str, Any]:
    """영상 → 구간별 입술 NPZ. 구간이 끝날 때마다 진행 메시지를 보낸다.

    "utterance"(기본 False): True면 녹화 한 건을 한 발화로 판정해 앞뒤 무발화 부분을 잘라 낸다. 모델 창
    (segment_seconds, 기본 12초)은 그대로라 12초보다 긴 녹화는 여러 창으로 나뉠 수 있다.
    "duration_seconds"(선택, 유한한 양수 또는 null): 컨테이너에 길이 정보가 없을 때(MediaRecorder WebM)
    진행률 분모로 쓴다. extract_segments(duration_hint_seconds=...)로 넘긴다.
    """
    video = Path(str(request["video"]))
    out_dir = Path(str(request["out_dir"]))
    rid = request["id"]
    utterance = request.get("utterance", False)
    if not isinstance(utterance, bool):
        raise ValueError("utterance는 true/false여야 합니다")
    duration_seconds = _duration_seconds(request)

    def progress(info: dict[str, Any]) -> None:
        channel.send({"type": "progress", "id": rid, **info})

    return core.extract_segments(video, out_dir,
                                 segment_seconds=float(request.get("segment_seconds", 12.0)),
                                 min_window_seconds=float(request.get("min_window_seconds", 1.0)),
                                 cut_search_seconds=float(request.get("cut_search_seconds", 6.0)),
                                 progress=progress, resample=bool(request.get("resample", False)),
                                 utterance=utterance, duration_hint_seconds=duration_seconds)


def _cuda_poisoned(device: str) -> bool:
    """요청 실패 뒤 CUDA 컨텍스트가 계속 쓸 수 있는지 확인한다.

    드라이버 리셋(TDR)·illegal memory access·device-side assert 뒤에는 컨텍스트가 영구히 손상되어
    이후 모든 요청이 실패한다. 그 경우 워커를 끝내 서버가 새로 띄우게 한다.
    """
    if not device.startswith("cuda"):
        return False
    try:
        import torch
        torch.cuda.synchronize()
        return float(torch.zeros(1, device=device).add_(1).item()) != 1.0
    except Exception:  # noqa: BLE001 - 어떤 오류든 컨텍스트 손상으로 본다
        return True


def _handle_infer(core: Any, request: dict[str, Any], device: str, default_decoder: str) -> dict[str, Any]:
    """NPZ 한 구간 → 텍스트·confidence·raw_score."""
    frames = core.read_frames(Path(str(request["npz"])))
    decoder = str(request.get("decoder") or default_decoder)
    result = core.recognize(frames, device, decoder)
    result.pop("token_ids", None)
    return result


def serve(mode: str, device: str, decoder: str, warmup: bool, core: Any = None, init: Any = None) -> int:
    """기동 후 표준입력 요청을 하나씩 처리한다. 표준입력이 닫히면(서버 종료) 끝난다.

    core/init은 테스트용 주입 지점이다(기본: 실제 공용 코어와 _init_roi/_init_infer).
    """
    channel = Channel(_claim_protocol_stdout())
    stdin = sys.stdin.buffer
    # 요청 파이프는 규약 전용이다. 라이브러리 코드(예: upstream의 pdb.set_trace)가 sys.stdin을 읽으면
    # 요청을 삼키거나 멈추므로, sys.stdin은 빈 입력으로 바꿔 즉시 EOF가 나게 한다.
    sys.stdin = open(os.devnull, encoding="utf-8")  # noqa: SIM115 - 프로세스 수명 동안 유지
    try:
        core = core or _load_core()
        if init is not None:
            info = init(core)
        else:
            info = _init_roi(core) if mode == "roi" else _init_infer(core, device, decoder, warmup)
    except Exception as exc:  # noqa: BLE001 - 기동 실패는 규약으로 알리고 종료
        channel.send({"type": "fatal", "error": _error(exc)})
        return 1
    device = info.get("device", device)
    channel.send({"type": "ready", "info": info})
    while True:
        line = stdin.readline()
        if not line:
            return 0  # 서버가 파이프를 닫았다
        try:
            request = json.loads(line.decode("utf-8"))
            if not isinstance(request, dict) or "id" not in request:
                raise ValueError("요청은 id가 있는 JSON 객체여야 합니다")
        except (UnicodeDecodeError, ValueError) as exc:
            channel.send({"type": "result", "id": None, "ok": False, "error": _error(exc)})
            continue
        rid, cmd = request["id"], request.get("cmd")
        started = time.perf_counter()
        try:
            if cmd == "ping":
                payload: dict[str, Any] = {"info": info}
            elif cmd == "shutdown":
                channel.send({"type": "result", "id": rid, "ok": True})
                return 0
            elif cmd == "roi" and mode == "roi":
                payload = _handle_roi(core, request, channel)
            elif cmd == "infer" and mode == "infer":
                payload = _handle_infer(core, request, device, decoder)
            else:
                raise ValueError(f"이 워커({mode})가 처리할 수 없는 명령: {cmd}")
            channel.send({"type": "result", "id": rid, "ok": True, "elapsed_seconds": time.perf_counter() - started,
                          **payload})
        except Exception as exc:  # noqa: BLE001 - 요청 하나의 실패가 워커를 죽이지 않게 한다
            poisoned = mode == "infer" and _cuda_poisoned(device)
            channel.send({"type": "result", "id": rid, "ok": False, "error": _error(exc),
                          "worker_poisoned": poisoned})
            if poisoned:
                print("[worker] CUDA 컨텍스트 손상 — 종료합니다(서버가 다시 띄움)", file=sys.stderr, flush=True)
                return 2


def main(argv: list[str] | None = None) -> int:
    """명령줄 진입점."""
    parser = argparse.ArgumentParser(description="OLKAVS 상주 워커")
    parser.add_argument("--mode", choices=("roi", "infer"), required=True)
    parser.add_argument("--device", default="auto", help="infer 전용: auto|cpu|cuda:0")
    parser.add_argument("--decoder", default="hybrid", choices=("hybrid", "ctc"), help="infer 전용 기본 디코더")
    parser.add_argument("--warmup", action="store_true", help="infer 전용: 기동 시 예열 추론 1회")
    args = parser.parse_args(argv)
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    return serve(args.mode, args.device, args.decoder, args.warmup)


if __name__ == "__main__":
    sys.exit(main())
