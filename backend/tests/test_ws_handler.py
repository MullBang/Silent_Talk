"""ws_handler 단위/통합 테스트 (설계문서 4절).

제어 메시지·바이너리 프레임 파싱·누적·PARTIAL 응답·ERROR 처리·Origin 검사를
TestClient WebSocket으로 검증한다. 모델 추론은 monkeypatch로 대체한다.
"""

from __future__ import annotations

import os
import sys

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

# backend 패키지 루트를 import 경로에 추가
_BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

from config import CORS_ORIGINS  # noqa: E402
from main import app  # noqa: E402
from services import ws_handler  # noqa: E402

client = TestClient(app)

ROI_BYTES = ws_handler.EXPECTED_ROI_BYTES  # 27,648


def _frame(chunk_id: int, payload_len: int = ROI_BYTES, msg_type: int = 0x01,
           fill: int = 128) -> bytes:
    """[8B 헤더 + payload] 바이너리 프레임을 만든다."""
    header = chunk_id.to_bytes(4, "big") + bytes([msg_type]) + b"\x00\x00\x00"
    return header + bytes([fill]) * payload_len


@pytest.fixture(autouse=True)
def _patch_infer(monkeypatch):
    """무거운 모델 추론을 가벼운 더미로 대체."""
    monkeypatch.setattr(
        ws_handler,
        "_infer",
        lambda frames, with_confidence: (
            ("안녕", 0.91) if with_confidence else ("안", None)
        ),
    )


def test_partial_then_final_flow() -> None:
    """SESSION_START → CHUNK_START(중간) → 바이너리 → PARTIAL, 이어 최종까지 검증."""
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_json({"type": "SESSION_START", "session_id": "s1", "client_ts": 1718})

        # 중간 청크
        ws.send_json({
            "type": "CHUNK_START", "chunk_id": 1,
            "window_start_ms": 0, "window_end_ms": 480, "is_final": False,
        })
        ws.send_bytes(_frame(1))
        m = ws.receive_json()
        assert m["type"] == "PARTIAL"
        assert m["is_final"] is False
        assert m["confidence"] is None
        assert m["chunk_id"] == 1
        assert m["text"] == "안"

        # 최종 청크
        ws.send_json({
            "type": "CHUNK_START", "chunk_id": 8,
            "window_start_ms": 2800, "window_end_ms": 3200, "is_final": True,
        })
        ws.send_bytes(_frame(8))
        m2 = ws.receive_json()
        assert m2["is_final"] is True
        assert m2["confidence"] == 0.91
        assert m2["chunk_id"] == 8
        assert m2["text"] == "안녕"

        # 세션 종료
        ws.send_json({"type": "SESSION_END"})
        end = ws.receive_json()
        assert end["type"] == "SESSION_END"


def test_binary_before_control_errors() -> None:
    """SESSION_START/CHUNK_START 없이 바이너리 → PROTOCOL 에러."""
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_bytes(_frame(1))
        m = ws.receive_json()
        assert m["type"] == "ERROR"
        assert m["code"] == "PROTOCOL"


def test_bad_payload_size_errors() -> None:
    """잘못된 ROI 페이로드 크기 → BAD_SIZE 에러 (Float32 등 거부)."""
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_json({"type": "SESSION_START", "session_id": "s2", "client_ts": 1})
        ws.send_json({"type": "CHUNK_START", "chunk_id": 1, "is_final": False})
        ws.send_bytes(_frame(1, payload_len=100))  # 27,648이 아님
        m = ws.receive_json()
        assert m["type"] == "ERROR"
        assert m["code"] == "BAD_SIZE"


def test_bad_msg_type_errors() -> None:
    """헤더 msg_type이 0x01이 아니면 BAD_MSG_TYPE 에러."""
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_json({"type": "SESSION_START", "session_id": "s3", "client_ts": 1})
        ws.send_json({"type": "CHUNK_START", "chunk_id": 1, "is_final": False})
        ws.send_bytes(_frame(1, msg_type=0x02))
        m = ws.receive_json()
        assert m["type"] == "ERROR"
        assert m["code"] == "BAD_MSG_TYPE"


def test_origin_allowed_connects() -> None:
    """화이트리스트 Origin은 연결되고 정상 동작한다."""
    with client.websocket_connect(
        "/ws/stream", headers={"origin": CORS_ORIGINS[0]}
    ) as ws:
        ws.send_json({"type": "SESSION_START", "session_id": "s4", "client_ts": 1})
        ws.send_json({"type": "SESSION_END"})
        end = ws.receive_json()
        assert end["type"] == "SESSION_END"


def test_origin_rejected() -> None:
    """미허용 Origin은 핸드셰이크가 거부된다 (정책 위반 close)."""
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(
            "/ws/stream", headers={"origin": "http://evil.example.com"}
        ) as ws:
            ws.receive_json()
