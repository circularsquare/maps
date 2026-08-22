"""
Local dev server for citybrowser: static files + the edit-mode write API.

Needed because index.html fetches data/*.json and browsers block fetch() on
file:// URLs. It also owns the only path that writes overrides.json.

Edit mode is local by design — curation lives in git, no hosting, no auth, and
nothing to lose. Bound to 127.0.0.1 deliberately; do NOT bind "" or 0.0.0.0,
which would expose an unauthenticated write API to the network.

API (see SCHEMA.md):
    PATCH  /api/city/<key>   {"field":"name","value":"Tokyo"}   set one field
                             {"field":"name","value":null}      clear an override
    DELETE /api/city/<key>                                      tombstone a city
    POST   /api/city         {"lat":..,"lon":..,"name":..}      create one
    POST   /api/build                                           rebuild cities.json

    python serve.py            # http://localhost:8765
    python serve.py 9000       # other port
"""

import http.server
import json
import os
import pathlib
import sys
import threading
import datetime

HERE = pathlib.Path(__file__).parent
DATA = HERE / "data"
BASE = DATA / "base.json"
OVERRIDES = DATA / "overrides.json"
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8765

# The server is threaded, so every read-modify-write of overrides.json must be
# serialised or two quick edits will lose one of themselves.
LOCK = threading.Lock()
_base_cache = None
_alts_cache = None


def alts():
    """qid -> alt candidate list, loaded once.

    A first version scanned all 1,238 stage-3 cache files per request, which is
    ~1,238 file reads to answer one lookup. Loading them once costs a couple of
    seconds and about 12 MB of process memory, and every lookup after is free.
    """
    global _alts_cache
    if _alts_cache is None:
        _alts_cache = {}
        d = HERE / "cache" / "entities"
        if d.exists():
            for f in sorted(d.glob("*.json")):
                for q, r in json.loads(f.read_text(encoding="utf-8")).items():
                    _alts_cache[q] = r.get("alt") or []
    return _alts_cache


def base():
    global _base_cache
    if _base_cache is None:
        _base_cache = json.loads(BASE.read_text(encoding="utf-8")) if BASE.exists() else {}
    return _base_cache


def read_overrides():
    if not OVERRIDES.exists():
        return {}
    return json.loads(OVERRIDES.read_text(encoding="utf-8"))


def write_overrides(ov):
    DATA.mkdir(exist_ok=True)
    tmp = OVERRIDES.with_suffix(".tmp")
    tmp.write_text(json.dumps(ov, ensure_ascii=False, indent=1, sort_keys=True),
                   encoding="utf-8")
    tmp.replace(OVERRIDES)   # atomic: an interrupt can never truncate curation


def today():
    return datetime.date.today().isoformat()


class Handler(http.server.SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"   # Chrome expects keep-alive
    timeout = 10                    # NEVER block forever on an idle socket

    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(HERE), **kw)

    # --- helpers ----------------------------------------------------------
    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def _json(self, obj, code=200):
        b = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _key(self, prefix):
        if not self.path.startswith(prefix):
            return None
        return self.path[len(prefix):].strip("/") or None

    # --- API --------------------------------------------------------------
    def do_PATCH(self):
        key = self._key("/api/city/")
        if not key:
            return self._json({"error": "no key"}, 400)
        try:
            body = self._body()
            field, value = body["field"], body.get("value")
        except Exception as e:
            return self._json({"error": f"bad body: {e}"}, 400)
        if field.startswith("_"):
            return self._json({"error": "cannot patch reserved field"}, 400)

        with LOCK:
            ov = read_overrides()
            rec = ov.setdefault(key, {})
            if value is None:
                rec.pop(field, None)        # clearing reverts to the base value
                if not rec:
                    ov.pop(key, None)
            else:
                # `was` is captured ONCE, from base, at first edit. Re-editing
                # must not overwrite it with our own previous value, or the
                # staleness check silently stops working.
                was = rec.get(field, {}).get("was", base().get(key, {}).get(field))
                rec[field] = {"value": value, "was": was, "at": today()}
            write_overrides(ov)
        return self._json({"ok": True, "key": key, "field": field, "value": value})

    def do_GET(self):
        # Full alt-name candidate list for one city, straight from the stage-3
        # cache. Kept OUT of base.json because inlining all 14 per city added
        # 12 MB to every page load for data only the edit panel ever reads.
        if self.path.startswith("/api/alts/"):
            key = self.path[len("/api/alts/"):].strip("/")
            import assemble_base as ab
            name = base().get(key, {}).get("name")
            out = [x for x in alts().get(key, []) if ab._useful_alt(name, x[1])]
            return self._json({"key": key, "alt": out})
        return super().do_GET()

    def do_DELETE(self):
        key = self._key("/api/city/")
        if not key:
            return self._json({"error": "no key"}, 400)
        with LOCK:
            ov = read_overrides()
            ov.setdefault(key, {})["_deleted"] = True
            write_overrides(ov)
        return self._json({"ok": True, "key": key, "deleted": True})

    def do_POST(self):
        if self.path.rstrip("/") == "/api/build":
            import importlib
            with LOCK:
                bm = importlib.import_module("build")
                importlib.reload(bm)
                cities, counts = bm.merge(bm.load(bm.BASE, {}), read_overrides())
                bm.OUT.write_text(
                    json.dumps(cities, ensure_ascii=False, separators=(",", ":")),
                    encoding="utf-8")
            return self._json({"ok": True, "cities": len(cities), **counts})

        if self.path.rstrip("/") != "/api/city":
            return self._json({"error": "unknown endpoint"}, 404)
        try:
            body = self._body()
            lat, lon = float(body["lat"]), float(body["lon"])
        except Exception as e:
            return self._json({"error": f"bad body: {e}"}, 400)

        with LOCK:
            ov = read_overrides()
            n = 1
            while f"x{n:04d}" in ov or f"x{n:04d}" in base():
                n += 1                      # never reuse a synthetic key
            key = f"x{n:04d}"
            ov[key] = {"_created": {"lat": lat, "lon": lon,
                                    "name": body.get("name") or "Untitled",
                                    "pop": body.get("pop")}}
            write_overrides(ov)
        return self._json({"ok": True, "key": key})

    # --- static -----------------------------------------------------------
    def end_headers(self):
        # No caching: during curation a reload must show the new build.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def copyfile(self, source, outputfile):
        # A browser navigating away mid-download aborts the socket, which is
        # ConnectionAbortedError on Windows and BrokenPipeError on POSIX. Normal
        # client behaviour, not a fault — do not spew a traceback per reload.
        try:
            super().copyfile(source, outputfile)
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            pass

    def log_message(self, fmt, *args):
        # Quiet successful requests. Match the status as its own token — an
        # earlier `"200" not in fmt % args` also swallowed any URL containing
        # "200", which is exactly the kind of thing you lose an hour to.
        try:
            if str(args[1]) == "200":
                return
        except (IndexError, TypeError):
            pass
        super().log_message(fmt, *args)


# ThreadingHTTPServer, NOT socketserver.TCPServer.
#
# TCPServer handles ONE connection at a time, and BaseHTTPRequestHandler has no
# read timeout by default. So a client that opens a socket and sends nothing
# blocks the accept loop FOREVER. Not hypothetical: VSCode parks an idle socket
# on the port, which froze the server solid (0.28s CPU, socket set unchanged for
# ten minutes) while the default backlog of 5 filled with Chrome's connections —
# after which every new connection got ECONNREFUSED, from a process that was
# alive and listening. Hence all three of: threads, a handler read timeout, and
# a much larger backlog.
class Server(http.server.ThreadingHTTPServer):
    # allow_reuse_address is FALSE on Windows, deliberately.
    #
    # On POSIX SO_REUSEADDR only skips the TIME_WAIT wait. On Windows it lets a
    # second process bind a port another process is already listening on — both
    # end up in the listen table and connections go to whichever the OS picks.
    # That happened here: a restarted server appeared to work while a stale one
    # kept answering, so new endpoints 404'd with no error anywhere. Binding
    # exclusively turns that silent split-brain into a loud "port in use".
    allow_reuse_address = (os.name != "nt")
    daemon_threads = True       # ctrl-c should not hang on a stalled transfer
    request_queue_size = 128    # 5 is far too small when a client idles


try:
    httpd = Server(("127.0.0.1", PORT), Handler)
except OSError as e:
    sys.exit(f"port {PORT} is already in use ({e}). "
             f"Another citybrowser server is running -- stop that one first, "
             f"or start this on a different port: python serve.py {PORT + 1}")

with httpd:
    print(f"citybrowser -> http://localhost:{PORT}/")
    print("(threaded; edit API at /api/city; ctrl-c to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
