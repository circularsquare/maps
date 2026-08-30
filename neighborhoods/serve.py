"""
Local dev server for the neighbourhood browser.

Needed because index.html fetches data/base.json and browsers block fetch() on file://
URLs. Static only — there is no write API here yet, unlike citybrowser's.

Bound to 127.0.0.1 deliberately; do NOT bind "" or 0.0.0.0.

    python serve.py            # http://localhost:8766
    python serve.py 9000       # other port
"""

import functools
import http.server
import pathlib
import sys
import webbrowser

HERE = pathlib.Path(__file__).parent
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8766


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # base.json is rebuilt constantly during development and a cached copy looks
        # exactly like a build that silently did nothing.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt, *args):
        if "base.json" in (args[0] if args else ""):
            super().log_message(fmt, *args)


def main():
    if not (HERE / "data" / "base.json").exists():
        sys.exit("data/base.json missing — run `python build.py` first")
    handler = functools.partial(Handler, directory=str(HERE))
    with http.server.ThreadingHTTPServer(("127.0.0.1", PORT), handler) as httpd:
        url = f"http://localhost:{PORT}/"
        print(f"serving {HERE.name} at {url}  (ctrl-c to stop)")
        webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")


if __name__ == "__main__":
    main()
