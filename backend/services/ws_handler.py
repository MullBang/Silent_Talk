"""웹캠 프로토콜 v2: 명시된 프레임을 모두 받은 청크만 한 번 추론한다.

SESSION_START → SESSION_READY → (CHUNK_START + frame_count개의 binary) → PARTIAL.
frame_count=0인 final은 누적 발화를 확정한다. SESSION_END도 남은 발화를 확정한다.
모델 작업은 별도 스레드에서 직렬화하며 한 발화의 메모리/길이를 제한한다.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import json
import logging
import math
import os
import threading

import numpy as np
from starlette.websockets import WebSocketDisconnect

from config import (
    CORS_ORIGINS, FPS, MODEL_BACKEND, MODEL_WEIGHTS_PATH, NORMALIZE_MEAN, NORMALIZE_STD,
    ROI_SIZE, WS_CHUNK_FRAMES, WS_MAX_UTTERANCE_FRAMES, WS_PROTOCOL_VERSION,
)

EXPECTED_ROI_BYTES = ROI_SIZE[0] * ROI_SIZE[1] * 3
HEADER_BYTES = 8
MSG_TYPE_FRAME_DATA = 0x01
FRAME_MS = 1000 // FPS
_model = None
_model_lock = threading.Lock()
_infer_lock = threading.Lock()
logger = logging.getLogger(__name__)


class ProtocolError(ValueError):
    """누적 상태를 재사용하지 않고 연결을 종료해야 하는 규격 오류."""
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass
class StreamSession:
    """연결별 한 발화 버퍼 및 단조 증가 식별자."""
    session_id: str | None = None
    chunk: dict | None = None
    frames: list[np.ndarray] = field(default_factory=list)
    utterance_id: int | None = None
    last_utterance_id: int = 0
    last_chunk_id: int = 0
    next_ms: int | None = None
    last_end_ms: int = 0
    total_segments: int = 0
    total_frames: int = 0

    def clear_utterance(self) -> None:
        """확정/중단된 발화의 영상 메모리만 해제한다."""
        self.frames.clear()
        self.utterance_id = None
        self.next_ms = None


def _get_model():
    """호환되는 학습 가중치만 한 번 로드한다. 무작위 가중치로 대체하지 않는다."""
    global _model
    with _model_lock:
        if _model is None:
            from models.baseline import LipNetBaseline
            if not os.path.exists(MODEL_WEIGHTS_PATH):
                raise FileNotFoundError(
                    f"v2 가중치가 없습니다. 새 토큰 사전으로 학습하세요: {MODEL_WEIGHTS_PATH}"
                )
            _model = LipNetBaseline.load_from_checkpoint(MODEL_WEIGHTS_PATH)
    return _model


def _infer(frames: list[np.ndarray], with_confidence: bool) -> tuple[str, float | None]:
    """별도 스레드에서 정규화/추론한다. 동시 모델 실행의 메모리 폭증을 막는다."""
    import torch
    from models.decoder import decode
    with _infer_lock:
        arr = np.stack(frames).astype(np.float32) / 255.0
        arr = (arr - np.asarray(NORMALIZE_MEAN, dtype=np.float32)) / np.asarray(NORMALIZE_STD, dtype=np.float32)
        tensor = torch.from_numpy(arr).permute(3, 0, 1, 2).unsqueeze(0)
        with torch.no_grad():
            log_probs = _get_model()(tensor)
        text, confidence, _raw = decode(log_probs)
        return text, confidence if with_confidence else None


def _integer(message: dict, key: str, minimum: int, maximum: int) -> int:
    """bool/실수/음수 등 잘못된 ID 및 프레임 수를 거부한다."""
    value = message.get(key)
    if type(value) is not int or not minimum <= value <= maximum:
        raise ProtocolError("PROTOCOL", f"잘못된 {key}")
    return value


async def _predict(websocket, state: StreamSession, chunk: dict) -> None:
    """완성 청크 또는 세션 종료의 누적 발화를 한 번만 추론한다."""
    final = chunk["is_final"]
    text, confidence = "", None
    if state.frames:
        text, confidence = await asyncio.to_thread(_infer, list(state.frames), final)
    await websocket.send_json({
        "type": "PARTIAL", "session_id": state.session_id,
        "utterance_id": state.utterance_id, "chunk_id": chunk["chunk_id"],
        "text": text, "confidence": confidence, "is_final": final,
        "frame_count": len(state.frames), "window_end_ms": state.next_ms,
    })
    if final:
        state.total_segments += bool(state.frames)
        state.clear_utterance()


async def handle_control_message(websocket, raw: str, state: StreamSession) -> bool:
    """제어 메시지를 검증한다. 반환 True이면 정상 종료한 연결이다."""
    try:
        message = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProtocolError("BAD_JSON", "제어 메시지 파싱 실패") from exc
    if not isinstance(message, dict):
        raise ProtocolError("BAD_JSON", "제어 메시지는 JSON 객체여야 합니다.")
    kind = message.get("type")
    if kind == "SESSION_START":
        if state.session_id is not None:
            raise ProtocolError("PROTOCOL", "이미 시작한 세션입니다.")
        if message.get("protocol_version") != WS_PROTOCOL_VERSION:
            raise ProtocolError("PROTOCOL_VERSION", "웹캠 화면과 서버를 함께 업데이트하세요.")
        sid = message.get("session_id")
        timestamp = message.get("client_ts")
        if not isinstance(sid, str) or not sid or len(sid) > 128:
            raise ProtocolError("PROTOCOL", "올바른 session_id가 필요합니다.")
        if type(timestamp) not in (int, float) or not math.isfinite(timestamp) or timestamp < 0:
            raise ProtocolError("PROTOCOL", "올바른 client_ts가 필요합니다.")
        try:
            await asyncio.to_thread(_get_model)
        except FileNotFoundError as exc:
            if MODEL_BACKEND == "olkavs":
                raise ProtocolError("MODEL_NOT_READY", "웹캠 실시간 인식은 공식 모델에서 아직 지원하지 않습니다. "
                                    "영상을 녹화해 업로드하면 공식 모델로 분석합니다.") from exc
            raise ProtocolError("MODEL_NOT_READY", "학습된 v2 모델이 아직 없습니다. 모델 학습을 완료한 뒤 웹캠 인식을 시작하세요.") from exc
        state.session_id = sid
        await websocket.send_json({"type": "SESSION_READY", "session_id": sid,
                                   "protocol_version": WS_PROTOCOL_VERSION, "fps": FPS,
                                   "chunk_frames": WS_CHUNK_FRAMES,
                                   "max_utterance_frames": WS_MAX_UTTERANCE_FRAMES})
        return False
    if state.session_id is None:
        raise ProtocolError("PROTOCOL", "SESSION_START 선행 필요")
    if state.chunk is not None:
        raise ProtocolError("INCOMPLETE_CHUNK", "선언한 청크의 모든 프레임을 먼저 보내세요.")
    if kind == "CHUNK_START":
        chunk_id = _integer(message, "chunk_id", 1, 2**32 - 1)
        utterance_id = _integer(message, "utterance_id", 1, 2**32 - 1)
        count = _integer(message, "frame_count", 0, WS_CHUNK_FRAMES)
        start = _integer(message, "window_start_ms", 0, 2**53 - 1)
        end = _integer(message, "window_end_ms", 0, 2**53 - 1)
        final = message.get("is_final")
        if type(final) is not bool or (count == 0 and not final):
            raise ProtocolError("PROTOCOL", "빈 청크는 발화 종료에만 사용합니다.")
        if chunk_id != state.last_chunk_id + 1:
            raise ProtocolError("CHUNK_ORDER", "중복 또는 순서가 바뀐 chunk_id")
        if start % FRAME_MS or end - start != count * FRAME_MS:
            raise ProtocolError("TIMELINE", "청크 시간과 25fps 프레임 수가 일치하지 않습니다.")
        if state.utterance_id is None:
            if utterance_id <= state.last_utterance_id or start < state.last_end_ms or count == 0:
                raise ProtocolError("UTTERANCE_ORDER", "새 발화 ID/시작 시간 또는 빈 종료가 올바르지 않습니다.")
            state.utterance_id = utterance_id
            state.last_utterance_id = utterance_id
        elif utterance_id != state.utterance_id or start != state.next_ms:
            raise ProtocolError("TIMELINE", "발화 중 프레임 누락·중복 또는 다른 발화 혼입")
        if len(state.frames) + count > WS_MAX_UTTERANCE_FRAMES:
            raise ProtocolError("BUFFER_LIMIT", "한 발화는 8초 이내로 보내세요.")
        state.last_chunk_id = chunk_id
        state.chunk = {**message, "received": 0}
        if count == 0:
            await _predict(websocket, state, state.chunk)
            state.chunk = None
        return False
    if kind == "UTTERANCE_ABORT":
        utterance_id = _integer(message, "utterance_id", 1, 2**32 - 1)
        if state.utterance_id != utterance_id:
            raise ProtocolError("UTTERANCE_ORDER", "중단할 발화 ID가 일치하지 않습니다.")
        state.clear_utterance()
        await websocket.send_json({"type": "ABORTED", "utterance_id": utterance_id})
        return False
    if kind == "SESSION_END":
        if state.frames:
            await _predict(websocket, state, {"is_final": True, "chunk_id": state.last_chunk_id})
        await websocket.send_json({"type": "SESSION_END", "total_segments": state.total_segments,
                                   "total_frames": state.total_frames})
        await websocket.close(code=1000)
        return True
    raise ProtocolError("UNKNOWN_TYPE", f"알 수 없는 제어 타입: {kind}")


async def handle_binary_chunk(websocket, data: bytes, state: StreamSession) -> None:
    """binary 헤더/청크 ID를 확인하고 마지막 프레임에서만 추론한다."""
    chunk = state.chunk
    if state.session_id is None or chunk is None:
        raise ProtocolError("PROTOCOL", "SESSION_START/CHUNK_START 선행 필요")
    if len(data) < HEADER_BYTES:
        raise ProtocolError("BAD_HEADER", "헤더 길이 부족")
    if data[4] != MSG_TYPE_FRAME_DATA:
        raise ProtocolError("BAD_MSG_TYPE", "미지원 바이너리 메시지 종류")
    if data[5:8] != b"\x00\x00\x00":
        raise ProtocolError("BAD_HEADER", "예약 헤더는 0이어야 합니다.")
    if int.from_bytes(data[:4], "big") != chunk["chunk_id"]:
        raise ProtocolError("CHUNK_ID", "제어 메시지와 바이너리 chunk_id 불일치")
    if len(data) != HEADER_BYTES + EXPECTED_ROI_BYTES:
        raise ProtocolError("BAD_SIZE", f"ROI는 RGB Uint8 {EXPECTED_ROI_BYTES}바이트여야 합니다.")
    frame = np.frombuffer(data, dtype=np.uint8, offset=HEADER_BYTES).reshape(*ROI_SIZE, 3)
    state.frames.append(frame.copy())
    state.total_frames += 1
    chunk["received"] += 1
    if chunk["received"] == chunk["frame_count"]:
        state.next_ms = chunk["window_end_ms"]
        state.last_end_ms = state.next_ms
        await _predict(websocket, state, chunk)
        state.chunk = None


async def handle_connection(websocket) -> None:
    """Origin 확인 후 세션을 처리하고 오류/종료에서 모든 버퍼를 해제한다."""
    origin = websocket.headers.get("origin")
    if origin is not None and origin not in CORS_ORIGINS:
        await websocket.close(code=1008)
        return
    await websocket.accept()
    state = StreamSession()
    try:
        while True:
            receive = websocket.receive()
            message = await asyncio.wait_for(receive, timeout=30) if state.chunk else await receive
            if message.get("type") == "websocket.disconnect":
                break
            if message.get("text") is not None:
                if await handle_control_message(websocket, message["text"], state):
                    break
            elif message.get("bytes") is not None:
                await handle_binary_chunk(websocket, message["bytes"], state)
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        code = exc.code if isinstance(exc, ProtocolError) else "INFERENCE_FAILED"
        if isinstance(exc, asyncio.TimeoutError):
            code, detail = "CHUNK_TIMEOUT", "청크 프레임 수신 시간이 초과되었습니다."
        elif isinstance(exc, ProtocolError):
            detail = str(exc)
        else:
            from models.tokenizer import TokenizerMismatchError
            if isinstance(exc, TokenizerMismatchError):
                code, detail = "MODEL_INCOMPATIBLE", "모델의 한글 사전이 현재 코드와 다릅니다. v2 모델을 학습하세요."
            else:
                detail = "모델 처리에 실패했습니다. 서버 상태와 가중치를 확인하세요."
                logger.exception("Webcam inference failed")
        try:
            await websocket.send_json({"type": "ERROR", "code": code, "message": detail, "fatal": True})
            await websocket.close(code=1008 if isinstance(exc, ProtocolError) else 1011)
        except (RuntimeError, WebSocketDisconnect):
            pass
    finally:
        state.clear_utterance()
        state.chunk = None
