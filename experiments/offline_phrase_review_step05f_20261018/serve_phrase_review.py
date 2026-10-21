"""Serve the exact allowlisted phrase resources locally with HTTP byte ranges."""
from __future__ import annotations

from http.server import ThreadingHTTPServer
import importlib.util
from pathlib import Path

from prepare_policy import BASE, binding, load


def main() -> None:
    """Expose only the reviewed page and declared audio/video files."""
    try:
        plan = load(BASE / 'phrase_server_plan.json')
        for key in ('range_handler', 'server', 'manifest', 'page'):
            assert binding(Path(plan[key]['path'])) == plan[key]
        spec = importlib.util.spec_from_file_location('stage5f_phrase_ranges', plan['range_handler']['path'])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        manifest = load(Path(plan['manifest']['path']))
        routes = {'/': Path(plan['page']['path']), '/index.html': Path(plan['page']['path'])}
        for row in manifest['records']:
            for key, media in row['media'].items():
                path = Path(media['path']).resolve()
                assert path.is_relative_to(Path(manifest['media_root']))
                assert path.is_relative_to(BASE)
                assert binding(path)['sha256'] == media['sha256']
                routes[row['urls'][key]] = path
        module.ReviewHandler.routes = routes
        with ThreadingHTTPServer(('127.0.0.1', 8773), module.ReviewHandler) as server:
            print('Phrase review: http://127.0.0.1:8773/', flush=True)
            server.serve_forever()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        raise RuntimeError('Local phrase review server failed') from exc


if __name__ == '__main__':
    main()
