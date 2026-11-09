"""빌드한 화면(frontend/dist)을 http://127.0.0.1:5173 으로 보여 주는 작은 정적 파일 서버.

Node 없이 파이썬 표준 라이브러리만 쓴다. python -m http.server 대신 이 파일을 쓰는 이유:
  - Windows 레지스트리에 .js가 text/plain으로 잘못 등록된 PC가 있다. 그러면 브라우저가 화면 스크립트를 막는다.
    여기서는 파일 종류(MIME)를 직접 정한다(.js → text/javascript, .wasm → application/wasm).
  - '/'는 index.html을 준다(#ward·#nurse는 주소의 # 뒤라 서버에 오지 않는다). 폴더 목록은 보여 주지 않는다.
  - 같은 포트를 두 번 열지 않는다(Windows의 SO_REUSEADDR는 이미 쓰는 포트에도 붙을 수 있어 끈다).
  - 요청마다 콘솔에 찍지 않는다(창을 클릭해 출력이 멈춰도 화면 제공이 멈추지 않게).
CORS 화이트리스트(backend/config.py)가 http://localhost:5173, http://127.0.0.1:5173 이므로 포트는 5173이어야 한다.
"""
from __future__ import annotations

import argparse
import http.server
import socket
import socketserver
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bundle_common as bc  # noqa: E402

MIME_TYPES: dict[str, str] = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".wasm": "application/wasm",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".ico": "image/x-icon",
    ".woff2": "font/woff2",
    ".txt": "text/plain; charset=utf-8",
    ".task": "application/octet-stream",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
}


class DistHandler(http.server.SimpleHTTPRequestHandler):
    """dist 폴더 안의 파일만 주는 처리기."""

    extensions_map = {"": "application/octet-stream", **MIME_TYPES}

    def guess_type(self, path: Any) -> str:
        """확장자로만 MIME을 정한다(레지스트리 값을 쓰지 않는다)."""
        try:
            return MIME_TYPES.get(Path(str(path)).suffix.lower(), "application/octet-stream")
        except (TypeError, ValueError):
            return "application/octet-stream"

    def list_directory(self, path: Any) -> None:  # type: ignore[override]
        """폴더 목록은 보여 주지 않는다."""
        self.send_error(404, "Not Found")
        return None

    def end_headers(self) -> None:
        """새로 빌드한 화면이 바로 보이도록 캐시하지 않게 한다."""
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        super().end_headers()

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - 부모 서명 유지
        """오류(4xx·5xx)만 남긴다."""
        try:
            if len(args) >= 2 and str(args[1])[:1] in ("4", "5"):
                sys.stderr.write(f"[화면] {self.address_string()} {format % args}\n")
        except (TypeError, ValueError, OSError):
            pass


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    """요청마다 스레드를 쓰는 서버. 이미 쓰는 포트에는 붙지 않는다."""

    daemon_threads = True
    allow_reuse_address = False

    def server_bind(self) -> None:
        """포트를 독점해 연다(Windows SO_EXCLUSIVEADDRUSE). 이름 역조회(getfqdn)는 오프라인에서 느려 생략한다."""
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        socketserver.TCPServer.server_bind(self)
        host, port = self.server_address[:2]
        self.server_name, self.server_port = str(host), int(port)


def serve(root: Path, host: str, port: int) -> int:
    """정적 서버를 연다. Ctrl+C 또는 창 닫기로 끝난다.

    Returns:
        종료 코드 (포트 사용 중·폴더 없음이면 1).
    """
    if not (root / "index.html").is_file():
        print(f"화면 파일이 없습니다: {root / 'index.html'}. 묶음을 다시 풀어 주세요.", flush=True)
        return 1
    handler = partial(DistHandler, directory=str(root))
    try:
        httpd = Server((host, port), handler)
    except OSError as exc:
        print(f"{port}번 포트를 열지 못했습니다({exc}). 다른 프로그램이 쓰고 있으면 끄고 다시 실행하세요.", flush=True)
        return 1
    print(f"화면 서버: http://{host}:{port}/  (폴더 {root})", flush=True)
    print("이 창을 닫으면 화면이 꺼집니다.", flush=True)
    try:
        httpd.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


def main(argv: list[str] | None = None) -> int:
    """명령줄 진입점."""
    bc.setup_console()
    bc.disable_quick_edit()
    parser = argparse.ArgumentParser(description="빌드한 화면 정적 서버")
    parser.add_argument("--root", type=Path, default=bc.FRONTEND_DIST)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=bc.FRONTEND_PORT)
    args = parser.parse_args(argv)
    try:
        return serve(args.root.resolve(), args.host, args.port)
    except Exception as exc:  # noqa: BLE001 - 창에 이유를 남기고 끝낸다
        print(f"화면 서버 오류: {type(exc).__name__}: {exc}", flush=True)
        time.sleep(2)
        return 1


if __name__ == "__main__":
    sys.exit(main())
