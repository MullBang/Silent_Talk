"""실제 Node WebSocket → FastAPI 송수신 검사. ML 모델만 SHA-256 계산으로 대체.

프로덕션 가중치나 학습 자료를 수정하지 않는다. 합성 RGB 프레임만 로컬 연결로 보낸다.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))


def main() -> None:
    """임시 루프백 서버를 띄워 프론트 전송 클래스로 검증 후 서버를 종료한다."""
    import uvicorn
    from fastapi import FastAPI
    from starlette.websockets import WebSocket
    from services import ws_handler

    calls = []

    def probe_infer(frames, final):
        digest = hashlib.sha256(b"".join(f.tobytes() for f in frames)).hexdigest()
        calls.append({"frames": len(frames), "final": final, "sha256": digest})
        return digest, 1.0 if final else None

    # 이 검증 프로세스 안에서만 추론을 대체한다.
    ws_handler._get_model = lambda: object()
    ws_handler._infer = probe_infer
    app = FastAPI()

    async def endpoint(ws):
        await ws_handler.handle_connection(ws)
    endpoint.__annotations__["ws"] = WebSocket
    app.websocket("/ws/stream")(endpoint)
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", ws="websockets-sansio"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not server.started:
            if time.monotonic() > deadline:
                raise TimeoutError("검증 서버 시작 실패")
            time.sleep(0.02)
        node = shutil.which("node")
        if node is None:
            raise FileNotFoundError("Node.js가 필요합니다.")
        result = subprocess.run([node, str(ROOT / "frontend/tests/webcamWire.mjs"),
                                 f"ws://127.0.0.1:{port}/ws/stream"],
                                check=True, capture_output=True, text=True, encoding="utf-8", timeout=30)
        evidence = json.loads(result.stdout)
        evidence["server_inference_calls"] = calls
        target = ROOT / "tmp/step3-wire-result.json"
        target.parent.mkdir(exist_ok=True)
        target.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(evidence, indent=2))
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        listener.close()


if __name__ == "__main__":
    main()
