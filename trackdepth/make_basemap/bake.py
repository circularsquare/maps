"""Drive render.html in headless Chrome over CDP and write the basemap tiles.

usage:
  bake.py test  <z> <x0> <y0> <nx> <ny> <outdir>    render one block, write PNG + WEBP into outdir
  bake.py run   <outroot> <zmin> <zmax> [fmt]       bake every tile of the extent (resumable)
env: CDP_PORT (default 9391), WEBP_Q (default 82)

Rebuild recipe (the tiles in data/basemap are gitignored; the 2026-11-02 bake was z12-16, 17,148 tiles, 212 MB):
  1. curl https://tiles.openfreemap.org/styles/dark -> dark.json ; python make_style.py dark.json style.json
  2. maplibre-gl 4.7.1 UMD (cdnjs.cloudflare.com/ajax/libs/maplibre-gl/4.7.1/maplibre-gl.js) saved as maplibre-gl.js
     beside render.html; style.json and bake.py go beside them too
  3. headless Chrome with its own --user-data-dir, --remote-debugging-port=9391 --remote-debugging-address=127.0.0.1
     --remote-allow-origins=* --allow-file-access-from-files
  4. python bake.py run <trackdepth>/data/basemap 12 16 webp   (resumable; about 30 min on a GPU)
Land colour is Stadia's raw (51,51,51), water (34,34,34): make_terrain.py's BASEMAP assumes the same chain.
"""
import base64, io, json, math, os, sys, time, urllib.request
import websocket
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = int(os.environ.get("CDP_PORT", "9391"))
WEBP_Q = int(os.environ.get("WEBP_Q", "82"))
EXTENT = [-74.26, 40.508, -73.645, 40.975]
BLOCK = 4


def tile_range(z):
    n = 2 ** z
    def tx(lon): return (lon + 180) / 360 * n
    def ty(lat): return (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n
    x0, x1 = int(tx(EXTENT[0])), int(tx(EXTENT[2]))
    y0, y1 = int(ty(EXTENT[3])), int(ty(EXTENT[1]))
    return x0, x1, y0, y1


class Cdp:
    def __init__(self):
        targets = json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json/list"))
        page = [t for t in targets if t["type"] == "page"][0]
        self.ws = websocket.create_connection(page["webSocketDebuggerUrl"], max_size=None, timeout=600)
        self.id = 0
        self.logs = []
        self.call("Runtime.enable")
        self.call("Page.enable")
        self.call("Emulation.setDeviceMetricsOverride", width=2400, height=2400, deviceScaleFactor=1, mobile=False)

    def call(self, method, **params):
        self.id += 1
        mid = self.id
        self.ws.send(json.dumps({"id": mid, "method": method, "params": params}))
        while True:
            m = json.loads(self.ws.recv())
            if m.get("method") == "Runtime.consoleAPICalled":
                self.logs.append(" ".join(str(a.get("value", a.get("description"))) for a in m["params"]["args"]))
            elif m.get("method") == "Runtime.exceptionThrown":
                self.logs.append("EXC " + str(m["params"]["exceptionDetails"]))
            if m.get("id") == mid:
                if "error" in m:
                    raise RuntimeError(f"{method}: {m['error']}")
                return m["result"]

    def eval(self, expr):
        r = self.call("Runtime.evaluate", expression=expr, awaitPromise=True, returnByValue=True)
        if "exceptionDetails" in r:
            raise RuntimeError(str(r["exceptionDetails"])[:800])
        return r["result"].get("value")


def setup(cdp):
    cdp.call("Page.navigate", url="file:///" + HERE.replace("\\", "/") + "/render.html?v=" + str(int(time.time())))
    for _ in range(100):
        time.sleep(0.2)
        if cdp.eval("typeof init === 'function' && typeof maplibregl !== 'undefined'"):
            break
    style = open(os.path.join(HERE, "style.json")).read()
    size = cdp.eval(f"init({style}, {BLOCK}, {BLOCK})")
    print("canvas", size, "gl:", cdp.eval(
        "(()=>{const g=document.createElement('canvas').getContext('webgl');const e=g.getExtension('WEBGL_debug_renderer_info');return g.getParameter(e.UNMASKED_RENDERER_WEBGL)})()"))


def render(cdp, z, x0, y0, want):
    res = cdp.eval(f"renderBlock({z},{x0},{y0},{BLOCK},{BLOCK},{json.dumps(want)})")
    out = {}
    for k, v in res.items():
        out[k] = Image.open(io.BytesIO(base64.b64decode(v.split(",", 1)[1]))).convert("RGB")
    return out


def save(im, root, z, x, y, fmt):
    d = os.path.join(root, str(z), str(x))
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, f"{y}.{fmt}")
    tmp = p + ".tmp"
    if fmt == "webp":
        im.save(tmp, format="WEBP", quality=WEBP_Q, method=6)
    else:
        im.save(tmp, format="PNG", optimize=True)
    os.replace(tmp, p)
    return os.path.getsize(p)


def main():
    cdp = Cdp()
    setup(cdp)
    if sys.argv[1] == "test":
        z, x0, y0, nx, ny = map(int, sys.argv[2:7])
        out = sys.argv[7]
        t = time.time()
        want = [f"{x0+i},{y0+j}" for j in range(ny) for i in range(nx)]
        tiles = render(cdp, z, x0, y0, want)
        print("rendered", len(tiles), "in", round(time.time() - t, 2), "s")
        for k, im in tiles.items():
            x, y = map(int, k.split(","))
            s1 = save(im, out + "/png", z, x, y, "png")
            s2 = save(im, out + "/webp", z, x, y, "webp")
            print(k, "png", s1, "webp", s2)
        print("\n".join(cdp.logs[:30]))
        return
    root, zmin, zmax = sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    fmt = sys.argv[5] if len(sys.argv) > 5 else "webp"
    total = sum((b - a + 1) * (d - c + 1) for a, b, c, d in (tile_range(z) for z in range(zmin, zmax + 1)))
    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(2)
    done = 0
    nbytes = 0
    t0 = time.time()
    for z in range(zmin, zmax + 1):
        xa, xb, ya, yb = tile_range(z)
        for by in range(ya // BLOCK * BLOCK, yb + 1, BLOCK):
            for bx in range(xa // BLOCK * BLOCK, xb + 1, BLOCK):
                want = [f"{x},{y}" for y in range(by, by + BLOCK) for x in range(bx, bx + BLOCK)
                        if xa <= x <= xb and ya <= y <= yb]
                todo = [k for k in want if not os.path.exists(os.path.join(root, str(z), k.split(",")[0], k.split(",")[1] + "." + fmt))]
                if not todo:
                    done += len(want)
                    continue
                tiles = render(cdp, z, bx, by, todo)
                futs = []
                for k, im in tiles.items():
                    x, y = map(int, k.split(","))
                    futs.append(pool.submit(save, im, root, z, x, y, fmt))
                for f in futs:
                    nbytes += f.result()
                done += len(want)
                el = time.time() - t0
                print(f"z{z} block {bx},{by}: {done}/{total} tiles  {nbytes/1e6:.1f} MB  {el:.0f}s  "
                      f"eta {el/max(done,1)*(total-done):.0f}s", flush=True)
                if cdp.logs:
                    print("  console:", cdp.logs[:5], flush=True)
                    cdp.logs.clear()
    print("DONE", done, "tiles", f"{nbytes/1e6:.1f} MB", f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
