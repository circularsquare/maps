"""
Serve the nudge editor and let it write nudges.json back.

A static page cannot save to disk, and the alternative — download the file and
move it into place by hand after every session — is exactly the kind of friction
that stops a tool getting used.

    python nudge/serve.py        # then open http://127.0.0.1:8799/nudge/

Serves the poster directory, so the page at /nudge/ can reach
/build/nudge_data.json alongside it. POST /nudge/save writes nudges.json.
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
NUDGES = HERE / "nudges.json"


class Handler(SimpleHTTPRequestHandler):
    def do_POST(self):
        if self.path.rstrip("/") != "/nudge/save":
            self.send_error(404)
            return
        n = int(self.headers.get("content-length", 0))
        try:
            data = json.loads(self.rfile.read(n) or b"{}")
            if not isinstance(data, dict):
                raise ValueError('expected {"move": {...}, "add": [...], "drop": [...]}')
            text = json.dumps(data, indent=0, sort_keys=True)
            # Keep the version being replaced. Hand-placing is slow work and the
            # ways to lose it are all cheap: a stale tab saving over a good file,
            # a mis-click, or somebody tidying up test edits without looking at
            # what is in there first. One level of file-side undo costs nothing.
            if NUDGES.exists():
                old = NUDGES.read_text(encoding="utf-8")
                if old.strip() and old != text:
                    NUDGES.with_name("nudges.backup.json").write_text(
                        old, encoding="utf-8")
            # write via a temp file so a half-written save can never replace a
            # good one (see reference_wb_truncates in the notes)
            tmp = NUDGES.with_suffix(".json.tmp")
            tmp.write_text(text, encoding="utf-8")
            tmp.replace(NUDGES)
        except Exception as e:                       # noqa: BLE001
            self.send_error(400, str(e))
            return
        body = json.dumps({"ok": True, "n": len(data)}).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        print(f"saved {len(data.get('move') or {})} moved, "
              f"{len(data.get('add') or [])} added, "
              f"{len(data.get('drop') or [])} deleted")

    def end_headers(self):
        self.send_header("cache-control", "no-store")
        super().end_headers()

    def log_message(self, *a):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8799)
    args = ap.parse_args()
    handler = partial(Handler, directory=str(ROOT))
    with ThreadingHTTPServer(("127.0.0.1", args.port), handler) as httpd:
        print(f"nudge editor: http://127.0.0.1:{args.port}/nudge/")
        print(f"writing to {NUDGES}")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
