"""One local server for every map, each under its own path.

    python serve.py            # http://localhost:8800/
    python serve.py 9000

    python serve.py list
    python serve.py add noritetsu noritetsu/dist "Global rail map and ridden-lines tracker"
    python serve.py remove noritetsu

The front page links to each registered map.  The register is served.json beside this file:
the path a map is served under, the folder holding its index.html (relative to maps/), and a
few words for the front page.  Register a map while it is being worked on and remove it when
it is not.  The running server re-reads served.json on every request, so add and remove take
effect on the next page load without a restart.

It speaks HTTP Range, which plain `python -m http.server` does not.  A .pmtiles archive is
read through range requests, and without them the map loads with a basemap and nothing on
it.  It also sends Cache-Control: no-store, so a rebuild shows up on refresh.

The maps' own servers still work (noritetsu/serve.py on 8767, `npx serve` in religiondots,
helper1m/serve.sh).  A different port is a different origin, so localStorage does not carry
over between them: settings saved on one port start fresh on another.

Bound to 127.0.0.1 deliberately; do not bind "" or 0.0.0.0.
"""
import html
import http.server
import json
import os
import pathlib
import re
import sys
import time
import urllib.parse

HERE = pathlib.Path(__file__).resolve().parent
REGISTER = HERE / "served.json"
PORT = 8800

RANGE = re.compile(r"^bytes=(\d*)-(\d*)$")
NAME = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def load():
    """[(name, folder, about)] from served.json; [] if it is missing or half-written."""
    try:
        rows = json.loads(REGISTER.read_text(encoding="utf-8"))
        return [(r["name"], r["folder"], r.get("about", "")) for r in rows]
    except (OSError, ValueError, KeyError, TypeError):
        return []


class Locked:
    """Agents in different sessions register at once; without this one add can undo another."""
    path = HERE / "served.json.lock"

    def __enter__(self):
        for _ in range(100):
            try:
                os.close(os.open(self.path, os.O_CREAT | os.O_EXCL))
                return
            except FileExistsError:
                try:
                    if time.time() - self.path.stat().st_mtime > 30:    # left by a crash
                        self.path.unlink(missing_ok=True)
                except FileNotFoundError:
                    pass
                time.sleep(0.1)
        sys.exit(f"{self.path} has been held for 10 s; delete it if nothing is registering")

    def __exit__(self, *exc):
        self.path.unlink(missing_ok=True)


def save(maps):
    rows = [{"name": n, "folder": f, "about": a} for n, f, a in maps]
    tmp = REGISTER.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for _ in range(20):
        try:
            return os.replace(tmp, REGISTER)
        except PermissionError:         # Windows: the server is reading it this instant
            time.sleep(0.05)
    os.replace(tmp, REGISTER)


def front_page():
    rows = []
    for name, folder, desc in load():
        missing = "" if (HERE / folder / "index.html").exists() else \
            f' <span class="miss">(no {html.escape(folder)}/index.html)</span>'
        rows.append(f'<li><a href="/{name}/">{html.escape(name)}</a>'
                    f'<span class="desc">{html.escape(desc)}</span>{missing}</li>')
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>maps</title>
<style>
  body {{ font: 16px/1.5 system-ui, sans-serif; margin: 40px auto; max-width: 560px;
          padding: 0 16px; color: #222; background: #fff; }}
  h1 {{ font-size: 18px; font-weight: 600; margin: 0 0 16px; }}
  ul {{ list-style: none; padding: 0; margin: 0; }}
  li {{ padding: 8px 0; border-top: 1px solid #eee; }}
  a {{ color: #1a5fb4; font-weight: 600; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  .desc {{ display: block; color: #666; font-size: 14px; }}
  .miss {{ color: #b00; font-size: 14px; }}
</style></head>
<body><h1>maps on localhost:{PORT}</h1><ul>
{chr(10).join(rows)}
</ul></body></html>
"""


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".pmtiles": "application/octet-stream",
        ".json": "application/json",
        ".geojson": "application/geo+json",
        ".js": "text/javascript",
    }

    def translate_path(self, path):
        # /noritetsu/data/x.pmtiles -> <maps>/noritetsu/dist/data/x.pmtiles
        parts = urllib.parse.urlsplit(path).path.split("/")
        want = urllib.parse.unquote(parts[1]) if len(parts) > 1 else ""
        folder = next((f for n, f, _ in load() if n == want), None)
        if folder is None:
            return str(HERE / "served.json.tmp" / "no-such-map")      # never a file: a 404
        self.directory = str(HERE / folder)
        return super().translate_path("/" + "/".join(parts[2:]))

    def send_front_page(self, body_too):
        body = front_page().encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body_too:
            self.wfile.write(body)

    def do_HEAD(self):
        if urllib.parse.urlsplit(self.path).path == "/":
            return self.send_front_page(False)
        super().do_HEAD()

    def do_GET(self):
        if urllib.parse.urlsplit(self.path).path == "/":
            return self.send_front_page(True)
        rng = self.headers.get("Range")
        if rng and self.serve_range(rng):
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
        self.end_headers()
        self.wfile.write(body)
        return True

    def end_headers(self):
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt, *args):
        pass


class Server(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        # The browser drops tile requests it no longer needs whenever the map moves;
        # on Windows each one would otherwise print a ConnectionAbortedError traceback.
        if isinstance(sys.exc_info()[1], (ConnectionError, TimeoutError)):
            return
        super().handle_error(request, client_address)


def cmd_list():
    maps = load()
    if not maps:
        print("nothing registered")
    for name, folder, about in maps:
        print(f"{name:16} {folder:24} {about}")


def cmd_add(name, folder, about=""):
    if not NAME.match(name):
        sys.exit(f"name {name!r}: lowercase letters, digits, - and _ only")
    folder = folder.replace("\\", "/").strip("/")
    if not (HERE / folder / "index.html").is_file():
        sys.exit(f"no index.html in {HERE / folder}")
    with Locked():
        maps = [m for m in load() if m[0] != name]
        maps.append((name, folder, about))
        save(maps)
    print(f"registered {name} -> {folder}  (http://localhost:{PORT}/{name}/)")


def cmd_remove(name):
    with Locked():
        maps = load()
        kept = [m for m in maps if m[0] != name]
        if len(kept) == len(maps):
            sys.exit(f"{name} is not registered")
        save(kept)
    print(f"removed {name}")


def serve(port):
    try:
        httpd = Server(("127.0.0.1", port), Handler)
    except OSError as e:
        sys.exit(f"port {port} is taken ({e.strerror}); is serve.py already running? "
                 f"Or pick another: python serve.py {port + 1}")
    global PORT
    PORT = port
    with httpd:
        print(f"maps at http://localhost:{port}/  (ctrl-c to stop)", flush=True)
        for name, _, _ in load():
            print(f"  http://localhost:{port}/{name}/", flush=True)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass


def main():
    args = sys.argv[1:]
    if not args:
        return serve(PORT)
    if args[0].isdigit() and len(args) == 1:
        return serve(int(args[0]))
    if args[0] == "list" and len(args) == 1:
        return cmd_list()
    if args[0] == "add" and len(args) in (3, 4):
        return cmd_add(*args[1:])
    if args[0] == "remove" and len(args) == 2:
        return cmd_remove(args[1])
    sys.exit(__doc__)


if __name__ == "__main__":
    main()
