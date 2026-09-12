"""Static server that refuses to be cached.

`python -m http.server` sends no cache headers, so the browser keeps its own
copy of index.html and an edit can look like it did nothing. That cost two
rounds of "it still does the old thing" here, and a fresh port did not clear
it. This sends no-store on everything.

    python serve.py [port]
"""

import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from functools import partial
import os

HERE = os.path.dirname(os.path.abspath(__file__))


class NoCache(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def log_message(self, fmt, *args):
        pass  # the tile flood is not worth reading


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8971
    handler = partial(NoCache, directory=HERE)
    print(f"serving {HERE} at http://127.0.0.1:{port}/  (no-store)")
    HTTPServer(("127.0.0.1", port), handler).serve_forever()


if __name__ == "__main__":
    main()
