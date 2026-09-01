#!/usr/bin/env python3
"""HTTP server for the visualizer: gzip for JS/HTML, ignore dropped clients."""
from __future__ import annotations

import argparse
import gzip
import os
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.abspath(__file__))
GZIP_SUFFIXES = (".js", ".html", ".css", ".json", ".svg", ".xml")
GZIP_MAX = 64 * 1024 * 1024  # skip the full 467MB data.js


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=ROOT, **kwargs)

    def end_headers(self):
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def copyfile(self, source, outputfile):
        try:
            super().copyfile(source, outputfile)
        except (ConnectionResetError, BrokenPipeError, TimeoutError, ConnectionAbortedError):
            pass

    def do_GET(self):
        path = self.translate_path(self.path.split("?", 1)[0])
        accept = self.headers.get("Accept-Encoding", "")
        if (
            "gzip" in accept
            and os.path.isfile(path)
            and path.endswith(GZIP_SUFFIXES)
            and os.path.getsize(path) <= GZIP_MAX
        ):
            try:
                with open(path, "rb") as fh:
                    raw = fh.read()
                gz = gzip.compress(raw, compresslevel=5)
            except OSError:
                return super().do_GET()
            try:
                self.send_response(200)
                self.send_header("Content-Type", self.guess_type(path))
                self.send_header("Content-Encoding", "gzip")
                self.send_header("Content-Length", str(len(gz)))
                self.send_header("Vary", "Accept-Encoding")
                self.end_headers()
                self.wfile.write(gz)
            except (ConnectionResetError, BrokenPipeError, TimeoutError, ConnectionAbortedError):
                pass
            return
        return super().do_GET()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8888)
    ap.add_argument("--bind", default="0.0.0.0")
    args = ap.parse_args()
    httpd = ThreadingHTTPServer((args.bind, args.port), Handler)
    print(f"serving {ROOT} at http://{args.bind}:{args.port}/visualizer.html", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
