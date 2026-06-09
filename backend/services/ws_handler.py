"""WebSocket 연결 핸들러 (설계문서 4절 — 실시간 스트리밍, 2차 목표).

업링크 규격:
  - JSON 제어 메시지 선행 → Binary ArrayBuffer.
  - 제어: SESSION_START / CHUNK_START / SESSION_END.
  - 바이너리 프레임: [0~3]=chunk_id(Uint32 BE) [4]=0x01(FRAME_DATA) [5~7]=reserved
                     [8~]=ROI 픽셀 Uint8 27,648 bytes (96×96×3).
다운링크: JSON {type:'PARTIAL'|'ERROR'|'SESSION_END', ...}.

세션별 프레임을 누적(슬라이딩 윈도우)하고, is_final=True 청크에서 전체 누적
프레임으로 최종 추론 후 버퍼를 비운다.

정규화는 파일 추론 파이프라인과 동일하게 normalize_roi(÷255 + ImageNet)를
적용하여 학습·추론 전처리 일관성을 유지한다(설계 4.2의 '[0,1]'은 단순 표기).
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

# backend 루트 부트스트랩
_BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

from config import CORS_ORIGINS, MODEL_WEIGHTS_PATH, ROI_SIZE  # noqa: E402

# 허용 바이너리 ROI 페이로드 크기 (Uint8). Float32(110,592) 직접 전송은 거부.
EXPECTED_ROI_BYTES: int = ROI_SIZE[0] * ROI_SIZE[1] * 3  # 27,648
HEADER_BYTES: int = 8
MSG_TYPE_FRAME_DATA: int = 0x01

# 정책 위반(허용 외 Origin) 종료 코드 — WebSocket의 403 대응
_WS_POLICY_VIOLATION = 1008

# 지연 로딩 모델 캐시
_model = None


def _get_model():
    """모델을 지연 로딩한다 (가중치 있으면 로드, 없으면 미학습 모델 폴백).

    Returns:
        eval 모드 LipNetBaseline 인스턴스.
    """
    global _model
    if _model is None:
        from models.baseline import LipNetBaseline

        if os.path.exists(MODEL_WEIGHTS_PATH):
            _model = LipNetBaseline.load_from_checkpoint(MODEL_WEIGHTS_PATH)
        else:
            _model = LipNetBaseline().eval()  # 미학습 폴백 (구조 검증용)
    return _model


def _infer(frames: list[np.ndarray], with_confidence: bool) -> tuple[str, float | None]:
    """누적된 ROI 프레임으로 추론한다.

    Args:
        frames: (96, 96, 3) uint8 ROI 프레임 리스트 (가변 길이 T).
        with_confidence: True면 confidence 산출(최종), False면 None(중간 예측).

    Returns:
        (transcript, confidence | None).
    """
    import torch

    from models.decoder import decode
    from preprocessing.roi_extractor import normalize_roi, to_model_tensor

    arr = np.stack(frames, axis=0)  # (T, 96, 96, 3)
    tensor = to_model_tensor(normalize_roi(arr))  # (1, 3, T, 96, 96)
    model = _get_model()
    with torch.no_grad():
        log_probs = model(tensor)  # (T, 1, num_classes)
    text, confidence, _raw = decode(log_probs)
    return text, (confidence if with_confidence else None)


async def handle_connection(websocket) -> None:
    """WebSocket 연결 수명주기를 처리한다.

    Origin 화이트리스트 검사 → accept → 제어/바이너리 메시지 루프.

    Args:
        websocket: 수신된 WebSocket 연결.
    """
    # ① Origin 화이트리스트 검사 (브라우저 교차출처 방어; Origin 없으면 비브라우저로 허용)
    origin = websocket.headers.get("origin")
    if origin is not None and origin not in CORS_ORIGINS:
        await websocket.close(code=_WS_POLICY_VIOLATION)  # ≈ 403
        return

    await websocket.accept()

    # 세션 상태 / 누적 버퍼
    state: dict = {"session_id": None, "client_ts": None, "chunk": None}
    buffers: dict[str, list[np.ndarray]] = {}

    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break
            if message.get("text") is not None:
                closed = await handle_control_message(
                    websocket, message["text"], state, buffers
                )
                if closed:
                    break
            elif message.get("bytes") is not None:
                await handle_binary_chunk(websocket, message["bytes"], state, buffers)
    except Exception:  # noqa: BLE001 - 연결 종료/오류 시 정리
        pass
    finally:
        sid = state.get("session_id")
        if sid is not None:
            buffers.pop(sid, None)


async def handle_control_message(websocket, raw: str, state: dict, buffers: dict) -> bool:
    """JSON 제어 메시지를 처리한다.

    Args:
        websocket: WebSocket 연결.
        raw: JSON 텍스트.
        state: 세션 상태 dict.
        buffers: 세션별 프레임 버퍼.

    Returns:
        연결을 종료해야 하면 True (SESSION_END).
    """
    try:
        message = json.loads(raw)
    except json.JSONDecodeError:
        await websocket.send_json(
            {"type": "ERROR", "code": "BAD_JSON", "message": "제어 메시지 파싱 실패"}
        )
        return False

    mtype = message.get("type")

    if mtype == "SESSION_START":
        sid = message.get("session_id")
        state["session_id"] = sid
        state["client_ts"] = message.get("client_ts")
        if sid is not None:
            buffers.setdefault(sid, [])
        return False

    if mtype == "CHUNK_START":
        state["chunk"] = {
            "chunk_id": message.get("chunk_id"),
            "window_start_ms": message.get("window_start_ms"),
            "window_end_ms": message.get("window_end_ms"),
            "is_final": bool(message.get("is_final", False)),
        }
        return False

    if mtype == "SESSION_END":
        sid = state.get("session_id")
        total = len(buffers.get(sid, [])) if sid else 0
        await websocket.send_json({"type": "SESSION_END", "total_segments": total})
        if sid is not None:
            buffers.pop(sid, None)
        await websocket.close()
        return True

    await websocket.send_json(
        {"type": "ERROR", "code": "UNKNOWN_TYPE", "message": f"알 수 없는 제어 타입: {mtype}"}
    )
    return False


async def handle_binary_chunk(websocket, data: bytes, state: dict, buffers: dict) -> None:
    """바이너리 ROI 프레임을 검증/누적하고 추론 결과를 전송한다.

    Args:
        websocket: WebSocket 연결.
        data: [8B 헤더 + 27,648B Uint8] 바이너리 프레임.
        state: 세션 상태.
        buffers: 세션별 프레임 버퍼.
    """
    chunk = state.get("chunk")
    sid = state.get("session_id")
    if sid is None or chunk is None:
        await websocket.send_json(
            {"type": "ERROR", "code": "PROTOCOL", "message": "SESSION_START/CHUNK_START 선행 필요"}
        )
        return

    # 헤더 검증
    if len(data) < HEADER_BYTES:
        await websocket.send_json(
            {"type": "ERROR", "code": "BAD_HEADER", "message": "헤더 길이 부족"}
        )
        return
    chunk_id = int.from_bytes(data[0:4], "big")
    msg_type = data[4]
    if msg_type != MSG_TYPE_FRAME_DATA:
        await websocket.send_json(
            {"type": "ERROR", "code": "BAD_MSG_TYPE", "message": f"미지원 msg_type: {msg_type}"}
        )
        return

    payload = data[HEADER_BYTES:]
    # Uint8 27,648 bytes만 허용 (Float32 110,592 등 거부)
    if len(payload) != EXPECTED_ROI_BYTES:
        await websocket.send_json(
            {
                "type": "ERROR",
                "code": "BAD_SIZE",
                "message": f"ROI 페이로드는 {EXPECTED_ROI_BYTES}B여야 합니다 (수신 {len(payload)}B)",
            }
        )
        return

    # Uint8 → (96,96,3) 프레임 누적
    frame = np.frombuffer(payload, dtype=np.uint8).reshape(ROI_SIZE[0], ROI_SIZE[1], 3)
    buffers.setdefault(sid, []).append(frame.copy())

    is_final = chunk["is_final"]
    try:
        text, confidence = _infer(buffers[sid], with_confidence=is_final)
    except Exception as exc:  # noqa: BLE001
        await websocket.send_json(
            {"type": "ERROR", "code": "DETECTION_FAILED", "message": str(exc)}
        )
        return

    await websocket.send_json(
        {
            "type": "PARTIAL",
            "text": text,
            "confidence": confidence,
            "is_final": is_final,
            "chunk_id": chunk_id,
        }
    )

    if is_final:
        # 최종 추론 완료 → 버퍼 클리어, 청크 리셋
        buffers[sid].clear()
        state["chunk"] = None
