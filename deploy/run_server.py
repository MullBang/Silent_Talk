"""시연 서버 실행기: backend/ 에서 `uvicorn main:app --host 127.0.0.1 --port 8000`과 같게 서버를 켠다.

uvicorn을 바로 부르지 않고 이 파일을 거치는 이유:
  - 콘솔 '빠른 편집'을 끈다. 켜져 있으면 창을 클릭해 글자가 선택되는 동안 서버 출력이 멈추고, 그동안 요청도 멈춘다.
  - 작업 폴더·import 경로를 backend/로 맞춘다(기록은 backend/logs, 임시 파일은 backend/tmp에 생긴다).
환경변수(LIPNET_*, git 대체 명령 PATH)는 launch.py가 넘긴다.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bundle_common as bc  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """서버를 켠다. 끝나면(창 닫기·Ctrl+C) 0."""
    bc.setup_console()
    bc.disable_quick_edit()
    parser = argparse.ArgumentParser(description="Silent Talk 시연 서버")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=bc.BACKEND_PORT)
    args = parser.parse_args(argv)
    try:
        os.chdir(bc.BACKEND)
        sys.path.insert(0, str(bc.BACKEND))
        import uvicorn  # noqa: PLC0415

        print(f"Silent Talk 서버: http://{args.host}:{args.port}  (이 창을 닫으면 서버가 꺼집니다)", flush=True)
        uvicorn.run("main:app", host=args.host, port=args.port, log_level="info")
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as exc:  # noqa: BLE001 - 창에 이유를 남긴다
        print(f"서버를 켜지 못했습니다: {type(exc).__name__}: {exc}", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
