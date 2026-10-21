"""Serve only bound stage5f review resources on loopback with byte ranges."""
from __future__ import annotations
import argparse
from http.server import ThreadingHTTPServer
import importlib.util
from pathlib import Path
from typing import Any

from prepare_policy import BASE, ROOT, binding, load


def prepare_handler() -> Any:
    """Reuse the frozen range handler with this stage's explicit media allowlist."""
    try:
        helper = ROOT / 'experiments/offline_boundary_review_step02b_20261013/serve_review.py'
        expected = load(BASE / 'server_plan.json')['range_handler']
        assert binding(helper) == expected
        spec = importlib.util.spec_from_file_location('stage5f_range_handler', helper)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        manifest = load(BASE / 'parent_review_manifest.json')
        rows = manifest['records']
        expected_ids = load(BASE / 'execution_policy.json')['record_ids']
        assert [row['parent_id'] for row in rows] == expected_ids
        routes = {'/': BASE / 'index.html', '/index.html': BASE / 'index.html'}
        for row in rows:
            for key, item in row['media'].items():
                path = Path(item['path']).resolve()
                assert path.is_relative_to(BASE / 'media')
                assert binding(path)['sha256'] == item['sha256']
                routes[row['urls'][key]] = path
        module.ReviewHandler.routes = routes
        return module.ReviewHandler
    except Exception as exc:
        raise RuntimeError('Review route binding failed') from exc


def main() -> None:
    """Start a read-only local review server; answers remain in the chat."""
    try:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--port', type=int, default=8772)
        arguments = parser.parse_args()
        handler = prepare_handler()
        with ThreadingHTTPServer(('127.0.0.1', arguments.port), handler) as server:
            print(f'Review: http://127.0.0.1:{arguments.port}/', flush=True)
            server.serve_forever()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        raise RuntimeError('Local parent review server stopped unexpectedly') from exc


if __name__ == '__main__':
    main()
