"""Local dev server for dist/.

    python serve.py            # http://localhost:8767
    python serve.py 9000

IT HAS TO SPEAK HTTP RANGE, which is the only reason this is not two lines of
`http.server`.  A .pmtiles archive is read by the browser through range requests: pmtiles.js
asks for the header, then the directory, then individual tile byte ranges.  Python's
SimpleHTTPRequestHandler ignores `Range` and answers 200 with the whole file, and pmtiles.js
treats a 200 without a Content-Range as a broken server, so the map comes up with a basemap
and no rail on it -- which looks exactly like a bad build.

Bound to 127.0.0.1 deliberately; do not bind "" or 0.0.0.0.
"""
import functools
import http.server
import os
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE / "dist"
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8767

RANGE = re.compile(r"^bytes=(\d*)-(\d*)$")


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".pmtiles": "application/octet-stream",
        ".json": "application/json",
    }

    def do_GET(self):
        rng = self.headers.get("Range")
        if rng:
            served = self.serve_range(rng)
            if served:
                return
        super().do_GET()

    def serve_range(self, rng):
        m = RANGE.match(rng.strip())
        if not m:
            return False
        path = self.translate_path(self.path)
        if not os.path.isfile(path):
            return False
        size = os.path.getsize(path)
        lo, hi = m.group(1), m.group(2)
        if lo == "" and hi == "":
            return False
        if lo == "":                       # bytes=-N, the last N bytes
            start, end = max(0, size - int(hi)), size - 1
        else:
            start = int(lo)
            end = min(int(hi), size - 1) if hi else size - 1
        if start > end or start >= size:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return True
        with open(path, "rb") as f:
            f.seek(start)
            body = f.read(end - start + 1)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()          # adds Accept-Ranges and Cache-Control for every response
        self.wfile.write(body)
        return True

    def end_headers(self):
        # A cached tile archive looks exactly like a rebuild that silently did nothing.
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt, *args):
        pass


def main():
    if not (ROOT / "index.html").exists():
        sys.exit(f"{ROOT}/index.html missing")
    handler = functools.partial(Handler, directory=str(ROOT))
    with http.server.ThreadingHTTPServer(("127.0.0.1", PORT), handler) as httpd:
        print(f"serving {ROOT} at http://localhost:{PORT}/  (ctrl-c to stop)", flush=True)
        httpd.serve_forever()


if __name__ == "__main__":
    main()
