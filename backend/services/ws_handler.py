"""WebSocket 연결 핸들러.

업링크 규격(CLAUDE.md §WebSocket 규격):
  - JSON 제어 메시지 선행 → Binary ArrayBuffer (Uint8, 27,648 bytes).
  - Float32 직접 전송 금지 (110,592 bytes — MVP 미사용).
  - 제어 메시지: SESSION_START, CHUNK_START.
"""

from __future__ import annotations

from fastapi import WebSocket

EXPECTED_CHUNK_BYTES: int = 27_648
"""허용 바이너리 청크 크기 (Uint8). Float32(110,592) 전송은 거부한다."""


async def handle_connection(websocket: WebSocket) -> None:
    """WebSocket 연결 수명주기를 처리한다.

    제어 메시지(JSON)를 먼저 수신하여 세션/청크 상태를 설정한 뒤, 이어지는
    바이너리 청크를 검증/누적한다.

    Args:
        websocket: 수락된 WebSocket 연결.
    """
    pass


async def handle_control_message(message: dict) -> None:
    """JSON 제어 메시지를 처리한다.

    type: 'SESSION_START' {session_id, client_ts} |
          'CHUNK_START' {chunk_id, window_start_ms, is_final}.

    Args:
        message: 파싱된 제어 메시지 딕셔너리.
    """
    pass


async def handle_binary_chunk(data: bytes) -> None:
    """바이너리 ArrayBuffer 청크를 검증/누적한다.

    EXPECTED_CHUNK_BYTES(27,648) Uint8만 허용하고 Float32 페이로드는 거부한다.

    Args:
        data: 수신된 바이너리 청크.
    """
    pass
