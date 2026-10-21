"""Serve only the new Q01 review resources on loopback."""
from __future__ import annotations

from http.server import ThreadingHTTPServer
import importlib.util
from pathlib import Path
from prepare_q01_revision import BASE, bind, read


def main() -> None:
    """Keep the frozen previous review UI intact."""
    try:
        plan = read(BASE / 'server_plan_r1.json')
        for key in ('range_handler', 'manifest', 'server'):
            assert bind(Path(plan[key]['path'])) == plan[key]
        routes = {}
        for route, binding in plan['routes'].items():
            path = Path(binding['path'])
            assert path.resolve().is_relative_to(BASE)
            assert bind(path) == binding
            routes[route] = path
        spec = importlib.util.spec_from_file_location('stage5g_review_ranges', plan['range_handler']['path'])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.ReviewHandler.routes = routes
        with ThreadingHTTPServer(('127.0.0.1', 8774), module.ReviewHandler) as server:
            print('Q01 revision: http://127.0.0.1:8774/', flush=True)
            server.serve_forever()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        raise RuntimeError('Q01 revision server failed') from exc


if __name__ == '__main__':
    main()
