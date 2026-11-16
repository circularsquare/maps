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

A page can also start one of the scripts listed in ACTIONS below (diskmap's rescan button),
and append to one of the INBOXES or save to one of the PADS (the trainworld board's).

It speaks HTTP Range, which plain `python -m http.server` does not.  A .pmtiles archive is
read through range requests, and without them the map loads with a basemap and nothing on
it.  It also sends Cache-Control: no-store, so a rebuild shows up on refresh.

The maps' own servers still work (noritetsu/serve.py on 8767, `npx serve` in religiondots,
helper1m/serve.sh).  A different port is a different origin, so localStorage does not carry
over between them: settings saved on one port start fresh on another.

Bound to 127.0.0.1 deliberately; do not bind "" or 0.0.0.0.
"""
import calendar
import html
import http.server
import json
import os
import pathlib
import re
import subprocess
import sys
import threading
import time
import urllib.parse

HERE = pathlib.Path(__file__).resolve().parent
REGISTER = HERE / "served.json"
PORT = 8800

RANGE = re.compile(r"^bytes=(\d*)-(\d*)$")
NAME = re.compile(r"^[a-z0-9][a-z0-9_-]*$")

# Scripts a page may start: POST /<name>/_run/<action> starts one, GET on the same path gives
# {"running", "line" (its latest output line), "code" (exit code once done)}.  Only these run.
# The POST must carry an X-Maps-Run header, which a page on another site cannot send without
# a CORS preflight this server never grants.
ACTIONS = {("diskmap", "rescan"): "diskmap/scan.py",
           ("twboard", "rebuild"): "trainworld/board/build.py"}
RUNS = {}           # (name, action) -> {"proc", "line", "code"}
RUNS_LOCK = threading.Lock()

# Append-only files a page may write to (the trainworld board's inbox for the orchestrator).
# POST /<name>/_inbox with a JSON object appends it as one line, with "at" added, and "id" too
# unless it is a {"handled": <id>} line.  Nothing here ever rewrites or truncates the file.
# GET gives {"pending": [actions with no handled line, send-now markers left out],
# "last_at", "sent_at"}.  The POST needs X-Maps-Run, as for ACTIONS.
INBOXES = {"twboard": "trainworld/notes/board-inbox.jsonl"}
INBOX_LOCK = threading.Lock()
INBOX_MAX = 64 * 1024
INBOX_LAST_ID = [0]
INBOX_RECENT_DAYS = 14

# A free-text file a page keeps saved as its person types (the board's scratch pad).
# GET /<name>/_pad gives {"text", "version"}.  PUT {"text", "version"} (X-Maps-Run, at most 256 KB)
# replaces the file if "version" is the one last read, else 409 with the current {"text", "version"}.
PADS = {"twboard": "trainworld/notes/board-pad.md"}
PAD_LOCK = threading.Lock()
PAD_MAX = 256 * 1024


def inbox_read(path):
    """The inbox's lines as dicts; a half-written or bad line is skipped."""
    rows = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except ValueError:
                    continue
                if isinstance(obj, dict):
                    rows.append(obj)
    except FileNotFoundError:
        pass
    return rows


def inbox_state(path):
    rows = inbox_read(path)
    handled = {str(r["handled"]) for r in rows if "handled" in r}
    pending, last_at, sent_at = [], None, None
    for r in rows:
        if "handled" in r or "id" not in r:
            continue
        if r.get("type") == "send-now":
            sent_at = r.get("at")
            continue
        last_at = r.get("at")
        if str(r["id"]) not in handled:
            pending.append(r)
    # her pad-send actions from the last INBOX_RECENT_DAYS days, handled or not (withdrawn ones left out)
    cutoff = time.time() - INBOX_RECENT_DAYS * 86400
    marks = {str(r["handled"]): r for r in rows if "handled" in r}
    pads = []
    for r in rows:
        if r.get("type") != "pad-send" or "id" not in r:
            continue
        try:
            when = calendar.timegm(time.strptime(r.get("at", ""), "%Y-%m-%dT%H:%M:%SZ"))
        except ValueError:
            continue
        mark = marks.get(str(r["id"]))
        if when < cutoff or (mark and mark.get("note") == "withdrawn"):
            continue
        pads.append({"id": r["id"], "at": r["at"], "text": r.get("text", ""),
                     "handled": {"at": mark.get("at"), "by": mark.get("by"), "note": mark.get("note", "")} if mark else None})
    return {"pending": pending, "last_at": last_at, "sent_at": sent_at, "pads": pads}


def inbox_append(path, obj):
    """Append one line; returns the stored object."""
    now = time.time()
    obj = dict(obj)
    obj["at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now))
    if "handled" not in obj:
        ms = max(int(now * 1000), INBOX_LAST_ID[0] + 1)     # unique and increasing
        INBOX_LAST_ID[0] = ms
        obj["id"] = f"a{ms}"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    return obj


def pad_state(path):
    """{"text", "version"}; version is the file's mtime in microseconds (small enough for a JS number), 0 while there is no file."""
    try:
        st = os.stat(path)
        return {"text": path.read_text(encoding="utf-8"), "version": st.st_mtime_ns // 1000}
    except FileNotFoundError:
        return {"text": "", "version": 0}


def pad_write(path, text):
    """Replace the file whole (temp file, then os.replace); returns the new state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    for _ in range(20):
        try:
            os.replace(tmp, path)
            break
        except PermissionError:         # Windows: a reader has it open this instant
            time.sleep(0.05)
    else:
        os.replace(tmp, path)
    return pad_state(path)


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

    def action(self):
        """(name, action) if this request is for /<name>/_run/<action>, else None."""
        parts = urllib.parse.urlsplit(self.path).path.split("/")
        if len(parts) == 4 and parts[2] == "_run":
            return parts[1], parts[3]
        return None

    def send_json(self, code, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def run_state(self, key):
        run = RUNS.get(key)
        if not run:
            return {"running": False, "line": "", "code": None}
        return {"running": run["code"] is None, "line": run["line"], "code": run["code"]}

    def inbox(self):
        """The inbox file if this request is for /<name>/_inbox of a listed name, else None."""
        parts = urllib.parse.urlsplit(self.path).path.split("/")
        if len(parts) == 3 and parts[2] == "_inbox" and parts[1] in INBOXES:
            return HERE / INBOXES[parts[1]]
        return None

    def post_inbox(self, path):
        if not self.headers.get("X-Maps-Run"):
            return self.send_error(403)
        try:
            size = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return self.send_error(400)
        if size > INBOX_MAX:
            return self.send_error(413)
        try:
            obj = json.loads(self.rfile.read(size).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return self.send_error(400, "not JSON")
        if not isinstance(obj, dict) or not obj:
            return self.send_error(400, "not a JSON object")
        with INBOX_LOCK:
            stored = inbox_append(path, obj)
        self.send_json(200, stored)

    def pad(self):
        """The pad file if this request is for /<name>/_pad of a listed name, else None."""
        parts = urllib.parse.urlsplit(self.path).path.split("/")
        if len(parts) == 3 and parts[2] == "_pad" and parts[1] in PADS:
            return HERE / PADS[parts[1]]
        return None

    def do_PUT(self):
        path = self.pad()
        if not path:
            return self.send_error(404)
        if not self.headers.get("X-Maps-Run"):
            return self.send_error(403)
        try:
            size = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return self.send_error(400)
        if size > PAD_MAX:
            return self.send_error(413)
        try:
            obj = json.loads(self.rfile.read(size).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return self.send_error(400, "not JSON")
        if not isinstance(obj, dict) or not isinstance(obj.get("text"), str) or not isinstance(obj.get("version"), int):
            return self.send_error(400, "need text and version")
        if len(obj["text"].encode("utf-8")) > PAD_MAX:
            return self.send_error(413)
        with PAD_LOCK:
            cur = pad_state(path)
            if cur["version"] != obj["version"]:
                return self.send_json(409, cur)
            self.send_json(200, pad_write(path, obj["text"]))

    def do_POST(self):
        path = self.inbox()
        if path:
            return self.post_inbox(path)
        key = self.action()
        if key not in ACTIONS:
            return self.send_error(404)
        if not self.headers.get("X-Maps-Run"):
            return self.send_error(403)
        with RUNS_LOCK:
            run = RUNS.get(key)
            if not run or run["code"] is not None:
                proc = subprocess.Popen(
                    [sys.executable, "-u", str(HERE / ACTIONS[key])], cwd=HERE,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                    text=True, encoding="utf-8", errors="replace")
                run = RUNS[key] = {"proc": proc, "line": "starting", "code": None}
                threading.Thread(target=follow, args=(run,), daemon=True).start()
            state = self.run_state(key)
        self.send_json(202, state)

    def do_GET(self):
        if urllib.parse.urlsplit(self.path).path == "/":
            return self.send_front_page(True)
        path = self.pad()
        if path:
            with PAD_LOCK:
                state = pad_state(path)
            return self.send_json(200, state)
        path = self.inbox()
        if path:
            with INBOX_LOCK:
                state = inbox_state(path)
            return self.send_json(200, state)
        key = self.action()
        if key in ACTIONS:
            with RUNS_LOCK:
                state = self.run_state(key)
            return self.send_json(200, state)
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


def follow(run):
    """Keep the latest output line of a started script, then its exit code."""
    for line in run["proc"].stdout:
        if line.strip():
            run["line"] = line.strip()
    code = run["proc"].wait()
    with RUNS_LOCK:
        run["code"] = code


class Server(http.server.ThreadingHTTPServer):
    daemon_threads = True
    # Windows refuses (not delays) connections past a full backlog; the default 5 lost 60% of
    # a 200-connection burst, as when a game opens many packs and tiles at once (trainworld T-490).
    request_queue_size = 128
    # On Windows SO_REUSEADDR lets a second serve.py bind 8800 beside the first, and requests
    # then go to either one at random; without it the second exits with "port is taken".
    allow_reuse_address = sys.platform != "win32"

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
