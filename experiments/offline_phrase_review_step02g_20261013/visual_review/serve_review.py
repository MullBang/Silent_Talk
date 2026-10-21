"""Serve only this review page and declared media on the local loopback."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import BinaryIO
from urllib.parse import unquote, urlsplit

BASE = Path(__file__).resolve().parent


class ReviewHandler(SimpleHTTPRequestHandler):
    """Expose exact declared routes without exposing other experiment files."""

    routes: dict[str, Path] = {}
    range_length: int | None = None

    def send_head(self) -> BinaryIO | None:
        """Support byte ranges so browsers can seek the MP4 to the target time."""
        try:
            target = Path(self.translate_path(self.path))
            self.range_length = None
            if not target.is_file():
                self.send_error(404, "Not a declared review resource")
                return None
            size = target.stat().st_size
            left, right = 0, size - 1
            requested_range = self.headers.get("Range")
            if requested_range:
                match = re.fullmatch(r"bytes=(\d*)-(\d*)", requested_range)
                if not match or not any(match.groups()):
                    self.send_error(416, "Unsupported byte range")
                    return None
                a, b = match.groups()
                if a:
                    left = int(a)
                    right = min(int(b), size - 1) if b else size - 1
                else:
                    left = max(0, size - int(b))
                if left >= size or right < left:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return None
            source = target.open("rb")
            source.seek(left)
            self.range_length = right - left + 1
            self.send_response(206 if requested_range else 200)
            self.send_header("Content-Type", self.guess_type(str(target)))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(self.range_length))
            if requested_range:
                self.send_header("Content-Range", f"bytes {left}-{right}/{size}")
            self.end_headers()
            return source
        except OSError:
            self.send_error(500, "Cannot read review resource")
            return None

    def copyfile(self, source: BinaryIO, outputfile: BinaryIO) -> None:
        """Copy exactly the requested bytes, tolerating cancelled media requests."""
        try:
            remaining = self.range_length
            if remaining is None:
                raise ValueError("Missing response length")
            while remaining:
                chunk = source.read(min(65536, remaining))
                if not chunk:
                    break
                outputfile.write(chunk)
                remaining -= len(chunk)
        except (OSError, ValueError):
            self.close_connection = True

    def translate_path(self, path: str) -> str:
        """Resolve an allowlisted URL; deny all unspecified file paths."""
        try:
            requested = unquote(urlsplit(path).path)
            return str(self.routes.get(requested, BASE / ".not-served"))
        except (TypeError, ValueError):
            return str(BASE / ".not-served")

    def end_headers(self) -> None:
        """Keep review media local and prevent stale page caching."""
        try:
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            super().end_headers()
        except OSError:
            self.close_connection = True


def main() -> None:
    """Start a local-only read-only media server; no submission endpoint."""
    try:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--port", type=int, default=8772)
        args = parser.parse_args()
        manifest = json.loads((BASE / "visual_review_manifest.json").read_text(encoding="utf-8"))
        media_root = (BASE.parents[2] / "experiments/offline_alignment_step02_20261012/media").resolve()
        ReviewHandler.routes = {"/": BASE / "index.html", "/index.html": BASE / "index.html"}
        for url,spec in manifest["routes"].items():
            target=Path(spec["path"]).resolve()
            if not re.fullmatch(r"/media/[0-9]+/(roi|full)\.mp4",url) or not target.is_relative_to(media_root) or target.name not in ("context_lips.mp4","context_video.mp4"):
                raise ValueError("Undeclared media path")
            if hashlib.sha256(target.read_bytes()).hexdigest()!=spec["sha256"]:
                raise ValueError("Preview hash changed")
            ReviewHandler.routes[url]=target
        with ThreadingHTTPServer(("127.0.0.1", args.port), ReviewHandler) as server:
            print(f"Review: http://127.0.0.1:{args.port}/index.html", flush=True)
            server.serve_forever()
    except KeyboardInterrupt:
        pass
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise RuntimeError("Review server failed") from exc


if __name__ == "__main__":
    main()
