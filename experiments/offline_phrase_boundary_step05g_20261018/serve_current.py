"""Serve only manifest-bound review resources; refresh on explicit UI revision."""
from __future__ import annotations

from http.server import ThreadingHTTPServer
import importlib.util
from pathlib import Path
from threading import Lock

from prepare_q01_revision import BASE, PREVIOUS, bind, read


def main() -> None:
    """Expose the local current review page without recording human approvals."""
    try:
        handler_binding = read(PREVIOUS / 'phrase_server_plan.json')['range_handler']
        assert bind(Path(handler_binding['path'])) == handler_binding
        spec = importlib.util.spec_from_file_location('stage5g_current_ranges', handler_binding['path'])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        class Handler(module.ReviewHandler):
            """Update the exact allowlist only when its bound manifest changes."""
            version = None
            update_lock = Lock()

            def do_GET(self) -> None:
                """Resolve a small explicit manifest pointer before read-only serving."""
                try:
                    pointer = read(BASE / 'ui/current.json')['manifest']
                    with self.update_lock:
                        if type(self).version != pointer['sha256']:
                            path = Path(pointer['path'])
                            assert path.resolve().is_relative_to(BASE / 'ui') and bind(path) == pointer
                            manifest = read(path)
                            assert manifest['range_handler'] == handler_binding
                            routes = {}
                            for route, item in manifest['routes'].items():
                                media = Path(item['path']).resolve()
                                assert media.is_relative_to(BASE) or media.is_relative_to(PREVIOUS)
                                assert bind(media) == item
                                routes[route] = media
                            type(self).routes = routes
                            type(self).version = pointer['sha256']
                    super().do_GET()
                except Exception:
                    self.send_error(500, 'Review evidence is unavailable or changed')

        with ThreadingHTTPServer(('127.0.0.1', 8775), Handler) as server:
            print('Current review: http://127.0.0.1:8775/', flush=True)
            server.serve_forever()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        raise RuntimeError('Current review server failed') from exc


if __name__ == '__main__':
    main()
