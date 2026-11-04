"""T-2257: fixed links in OSM (bridges, tunnels, causeways, fords) that join one land part to another, for
a look for false splits; and the packs whose parts the current rule (routes.way_flags) changes.

    python checks/t2257_links.py [<id> ... | --all] [--workers N] [--downloads N] [--min N]
    python checks/t2257_links.py --report [<id> ... | --all] [--workers N] [--min N]

Every pack with two parts or more (or the named ones). Per extract of theirs (a .osm.pbf kept in
data/raw/osm, or downloaded into data/raw/osm_t2257/ and deleted once read), in reader processes:
  - the water pixels under the ways that join land by the current rule, as parts_run.py has them
    -> data/work/parts/t2257/px/<id>/<extract>.npy;
  - the link ways near the pack: any highway or railway tagged bridge, tunnel, embankment, ford or
    causeway, and roads closed to motor traffic -> data/work/parts/t2257/links/<id>/<extract>.pkl.
Then per pack (--report does only this): its parts from parts_run's pixels (data/work/parts/px) and
from the new ones, `changed` when they differ (parts_run.py --redo with the new pixels makes them);
and on the new parts' land, each link way that crosses water, drawn with its ends carried 60 m on,
joins the labels it touches. A part other than the first is reported as joined
    car   by ways some car may use (a near miss of the 50 m drawing, or a way outside the pack's extracts)
    path  only by footways, paths, cycleways and roads closed to cars (a split as the rule means it)
    rail  only by railways
and the rest have no fixed link in OSM. Prints the changed packs, every car join, and path and rail
joins of parts with MIN people or jobs or more (default 1000); all of it in
../artifacts/trainworld/t2257/links.json. Delete data/work/parts/t2257 once the packs are rerun.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import glob  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import pickle  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from concurrent.futures import ThreadPoolExecutor, as_completed  # noqa: E402

import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
import routes  # noqa: E402
import parts_run  # noqa: E402

ART = os.path.join(ROOT, "..", "artifacts", "trainworld", "t2257")
WORK = os.path.join(parts_run.OUT, "t2257")
DL = os.path.join(ROOT, "data", "raw", "osm_t2257")
CLS = set(routes.DRIVE_CLASSES) | {"service", "track"}
RAIL = {"rail", "light_rail", "subway", "tram", "narrow_gauge", "monorail", "funicular", "preserved"}
LINK_TAGS = ("bridge", "tunnel", "embankment", "ford", "causeway")
KEEP_TAGS = ("name", "access", "motor_vehicle", "motorcar", "vehicle", "bus", "psv", "foot", "bicycle", "bridge",
             "tunnel", "embankment", "ford", "causeway", "service", "area")
PLANNED = {"construction", "proposed"}
LINK_PAD_DEG = 0.02  # link ways with a point this near the pack's grid are kept for it


def new_px(cid, e):
    return os.path.join(WORK, "px", cid, parts_run.slug(e) + ".npy")


def links_path(cid, e):
    return os.path.join(WORK, "links", cid, parts_run.slug(e) + ".pkl")


def other_tags(o):
    """GDAL's other_tags text as a dict."""
    out = {}
    for kv in o.split('","'):
        k, _, v = kv.strip('"').partition('"=>"')
        if k:
            out[k] = v
    return out


# ---------------------------------------------------------------- readers

_big = None


def _init(sem):
    global _big
    _big = sem
    routes.OSM = DL


def fetch(e, tries=3):
    kept = os.path.join(ROOT, "data", "raw", "osm", parts_run.slug(e) + ".osm.pbf")
    if os.path.exists(kept):
        return kept
    for k in range(tries):
        while shutil.disk_usage(ROOT).free < parts_run.MIN_FREE:
            time.sleep(30)
        try:
            return routes.download(e)
        except Exception:
            part = routes.pbf_path(e) + ".part"
            if os.path.exists(part):
                os.remove(part)
            if k == tries - 1:
                raise
            time.sleep(60)


def read_job(e, ids, path):
    import pyogrio
    import shapely
    import water
    from read_pack import load_pack
    t0 = time.time()
    try:
        big = os.path.getsize(path) > parts_run.BIG_PBF
        if big:
            _big.acquire()
        try:
            df = pyogrio.read_dataframe(path, layer="lines", columns=["osm_id", "highway", "railway", "other_tags"],
                                        where="highway IS NOT NULL OR railway IS NOT NULL")
        finally:
            if big:
                _big.release()
        if os.path.dirname(os.path.abspath(path)) == os.path.abspath(DL):
            os.remove(path)
        t_read = time.time() - t0
        hw = df["highway"].fillna("").to_numpy()
        rw = df["railway"].fillna("").to_numpy()
        oth = df["other_tags"].fillna("").to_numpy()
        n = len(df)
        cls = np.full(n, 255, np.uint8)
        flags = np.zeros(n, np.uint8)
        link = np.zeros(n, bool)
        cat = np.zeros(n, np.uint8)   # 0 car, 1 path, 2 rail
        old = np.zeros(n, bool)       # a car way by T-2213's rule
        tag = routes.tag
        for i in range(n):
            h, o = hw[i], oth[i]
            lt = any(tag(o, k) not in ("", "no") for k in LINK_TAGS)
            if h:
                cls[i], flags[i] = routes.way_flags(h, o)
                access = tag(o, "access")
                motor = tag(o, "motor_vehicle") or tag(o, "motorcar")
                closed = h in CLS and (access == "no" or motor == "no")
                link[i] = lt or closed
                cat[i] = 0 if flags[i] & routes.PART_FLAGS else 1
                old[i] = h in CLS and tag(o, "area") != "yes" and access != "no" and motor != "no"
            elif rw[i] in RAIL:
                link[i] = lt
                cat[i] = 2
        geoms = df.geometry.to_numpy()
        ids_osm = df["osm_id"].to_numpy()
        del df
        keep = flags > 0
        coords, idx = shapely.get_coordinates(geoms[keep], return_index=True)
        off = np.cumsum(np.r_[0, np.bincount(idx, minlength=int(keep.sum()))])
        ways = dict(cls=cls[keep], flags=flags[keep], off=off, lon=coords[:, 0].astype(np.float64),
                    lat=coords[:, 1].astype(np.float64))
        del coords, idx
        n_seg, n_px = parts_run.write_pixels(e, ways, ids, dest=new_px)
        del ways
        li = np.nonzero(link)[0]
        coords, idx = shapely.get_coordinates(geoms[li], return_index=True)
        cnt = np.bincount(idx, minlength=len(li))
        off = np.cumsum(np.r_[0, cnt])
        lon, lat = coords[:, 0], coords[:, 1]
        st = off[:-1][cnt > 0]
        has = np.nonzero(cnt > 0)[0]
        bx = np.full((len(li), 4), np.nan)
        if len(st):
            bx[has, 0] = np.minimum.reduceat(lon, st)
            bx[has, 1] = np.minimum.reduceat(lat, st)
            bx[has, 2] = np.maximum.reduceat(lon, st)
            bx[has, 3] = np.maximum.reduceat(lat, st)
        n_links = 0
        for cid in ids:
            h_, a_ = load_pack(cid, parts_run.PACKS)
            bar, origin = routes.water50(h_, a_)
            H, W = bar.shape
            lo, la = water.to_lonlat(np.array([origin[0], origin[0] + W * routes.WALK_CELL_M]),
                                     np.array([origin[1], origin[1] + H * routes.WALK_CELL_M]),
                                     h_["origin"]["lon"], h_["origin"]["lat"])
            p = LINK_PAD_DEG
            sel = np.nonzero((bx[:, 2] >= lo[0] - p) & (bx[:, 0] <= lo[1] + p) & (bx[:, 3] >= la[0] - p) & (bx[:, 1] <= la[1] + p))[0]
            parts = [np.arange(off[j], off[j + 1]) for j in sel]
            pidx = np.concatenate(parts) if parts else np.zeros(0, np.int64)
            rec = {"cat": cat[li[sel]], "old": old[li[sel]], "osm_id": [str(ids_osm[li[j]]) for j in sel],
                   "off": np.cumsum(np.r_[0, cnt[sel]]), "lon": lon[pidx], "lat": lat[pidx],
                   "hw": [hw[li[j]] or rw[li[j]] for j in sel],
                   "tags": [{k: v for k, v in other_tags(oth[li[j]]).items() if k in KEEP_TAGS} for j in sel]}
            q = links_path(cid, e)
            os.makedirs(os.path.dirname(q), exist_ok=True)
            with open(q + ".part", "wb") as f:
                pickle.dump(rec, f)
            os.replace(q + ".part", q)
            n_links += len(sel)
        return e, True, (f"{e}: {n:,} ways read in {t_read:.0f} s; {len(ids)} packs, {n_px:,} water pixels under roads, "
                         f"{n_links:,} link ways ({time.time() - t0:.0f} s)")
    except Exception:
        return e, False, f"{e}: FAILED\n{traceback.format_exc()}"


# ---------------------------------------------------------------- per pack

class UF:
    def __init__(self):
        self.p = {}

    def find(self, a):
        p = self.p
        p.setdefault(a, a)
        while p[a] != a:
            p[a] = p[p[a]]
            a = p[a]
        return a

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


_wf = None


def pack_job(cid, exs):
    global _wf
    import world_field
    try:
        _wf = _wf or world_field.load()
        h, a = routes.res9_pack(cid)
        bar, origin = routes.water50(h, a)
        H, W = bar.shape
        lon0, lat0 = h["origin"]["lon"], h["origin"]["lat"]
        old_px = np.unique(np.concatenate([np.load(parts_run.px_path(cid, e)) for e in exs]))
        new = np.unique(np.concatenate([np.load(new_px(cid, e)) for e in exs]))
        lab_o, edge_o, info_o = routes.land_labels(bar, old_px)
        part_o, block_o, _ = routes.land_parts(h, a, lab_o, edge_o, origin, info_o, exs, _wf)
        del lab_o
        lab, edge, info = routes.land_labels(bar, new)
        part, block, lab_part = routes.land_parts(h, a, lab, edge, origin, info, exs, _wf)
        changed = not np.array_equal(part_o, part)
        edge_set = set(int(v) for v in edge)
        land = lab > 0
        ways = []
        for e in exs:
            with open(links_path(cid, e), "rb") as f:
                rec = pickle.load(f)
            off = rec["off"]
            for j in range(len(off) - 1):
                if off[j + 1] - off[j] < 2:
                    continue
                x, y = routes.local_xy(rec["lon"][off[j]:off[j + 1]], rec["lat"][off[j]:off[j + 1]], lon0, lat0)
                idx = routes.line_pixels(x, y, np.array([0, len(x)]), origin[0], origin[1], routes.WALK_CELL_M, W, H,
                                         extend_m=routes.BRIDGE_EXTEND_M, step=routes.WALK_CELL_M / 2)
                if not len(idx):
                    continue
                on = land.flat[idx]
                if on.all():
                    continue
                labs = set(int(v) for v in np.unique(lab.flat[idx][on]))
                if len(labs) < 2:
                    continue
                c = 3 if rec["hw"][j] in PLANNED else int(rec["cat"][j])
                ways.append((c, bool(rec["old"][j]), rec["osm_id"][j], rec["hw"][j], rec["tags"][j], labs))
        names = ("car", "path", "rail", "planned")
        uf = [UF() for _ in names]
        for c, _o, _i, _h, _t, labs in ways:
            ls = sorted(labs)
            for k in range(c, 4):
                for L in ls[1:]:
                    uf[k].union(ls[0], L)

        def node(L):
            p = int(lab_part[L]) if L < len(lab_part) else -1
            return p if p >= 0 else ("edge" if L in edge_set else None)
        groups = []
        for c in range(4):
            g = {}
            for L in list(uf[c].p):
                g.setdefault(uf[c].find(L), set()).add(node(L))
            groups.append(g)
        joined = []
        for k in range(1, block["parts"]):
            mine = set(int(L) for L in np.nonzero(lab_part == k)[0] if L > 0)
            for c in range(4):
                reached = set()
                for L in mine:
                    if L in uf[c].p:
                        reached |= groups[c][uf[c].find(L)]
                reached -= {k, None}
                if reached:
                    links = [{"cat": names[cw], "car_before": ow, "way": wid, "highway": hwv, "tags": t}
                             for cw, ow, wid, hwv, t, labs in ways if labs & mine and cw <= c]
                    joined.append({"part": k, "people": block["people"][k], "jobs": block["jobs"][k], "cells": block["cells"][k],
                                   "open": block["open"][k], "lon": block["lon"][k], "lat": block["lat"][k], "by": names[c],
                                   "to": sorted(str(v) for v in reached), "links": links[:10]})
                    break
        keys = ("people", "jobs", "cells", "open", "lon", "lat")
        top = [dict({kk: block[kk][i] for kk in keys}, part=i) for i in range(1, min(block["parts"], 60))]
        return {"id": cid, "changed": changed, "parts_before": block_o["parts"], "parts": block["parts"],
                "off_first_before": float(sum(block_o["people"][1:])), "off_first": float(sum(block["people"][1:])),
                "jobs_off_first_before": float(sum(block_o["jobs"][1:])), "jobs_off_first": float(sum(block["jobs"][1:])),
                "joined": joined, "top": top}
    except Exception:
        return {"id": cid, "error": traceback.format_exc()}


def _pack_star(t):
    return pack_job(*t)


# ---------------------------------------------------------------- driver

def main(args):
    sys.stdout.reconfigure(encoding="utf-8")
    workers = int(args.pop(args.index("--workers") + 1)) if "--workers" in args else 3
    downloads = int(args.pop(args.index("--downloads") + 1)) if "--downloads" in args else 4
    mn = float(args.pop(args.index("--min") + 1)) if "--min" in args else 1000.0
    report_only = "--report" in args
    ids = [x for x in args if not x.startswith("--")]
    if not ids:
        ids = []
        for p in glob.glob(os.path.join(parts_run.OUT, "*.json")):
            with open(p, encoding="utf-8") as f:
                s = json.load(f)
            if s["parts"] > 1:
                ids.append(s["id"])
    plan = parts_run.plan(sorted(ids))
    os.makedirs(ART, exist_ok=True)
    t0 = time.time()
    if not report_only:
        by_ex = {}
        for c, ex in plan.items():
            for e in ex:
                if not (os.path.exists(new_px(c, e)) and os.path.exists(links_path(c, e))):
                    by_ex.setdefault(e, []).append(c)
        print(f"links: {len(plan)} packs, {len(by_ex)} extracts to read, {workers} readers, {downloads} downloads", flush=True)
        os.makedirs(DL, exist_ok=True)
        routes.OSM = DL
        mgr = mp.Manager()
        sem = mgr.BoundedSemaphore(2)
        ahead = threading.BoundedSemaphore(downloads + workers)
        lock = threading.Lock()
        count = [0]

        def get(e):
            ahead.acquire()
            try:
                return e, fetch(e), None
            except Exception as x:
                return e, None, f"{type(x).__name__}: {x}"

        def done(res):
            e, ok, line = res
            ahead.release()
            with lock:
                count[0] += 1
                print(f"[{count[0]}/{len(by_ex)} {time.time() - t0:.0f}s] {line}", flush=True)

        order = sorted(by_ex, key=lambda e: (-len(by_ex[e]), e))
        with ThreadPoolExecutor(downloads) as tp, mp.Pool(workers, initializer=_init, initargs=(sem,), maxtasksperchild=1) as pool:
            for f in as_completed([tp.submit(get, e) for e in order]):
                e, path, err = f.result()
                if err:
                    done((e, False, f"{e}: download FAILED ({err})"))
                    continue
                pool.apply_async(read_job, (e, by_ex[e], path), callback=done,
                                 error_callback=lambda x, e=e: done((e, False, f"{e}: FAILED ({x})")))
            pool.close()
            pool.join()
    ready = [(c, ex) for c, ex in plan.items()
             if all(os.path.exists(new_px(c, e)) and os.path.exists(links_path(c, e)) and os.path.exists(parts_run.px_path(c, e))
                    for e in ex)]
    print(f"links: {len(ready)} of {len(plan)} packs ready ({time.time() - t0:.0f} s)", flush=True)
    out = []
    with mp.Pool(workers) as pool:
        for k, r in enumerate(pool.imap_unordered(_pack_star, ready)):
            out.append(r)
            if "error" in r:
                print(f"[{k + 1}/{len(ready)}] {r['id']}: FAILED\n{r['error']}", flush=True)
                continue
            if r["changed"]:
                print(f"[{k + 1}/{len(ready)}] {r['id']}: CHANGED, {r['parts_before']} -> {r['parts']} parts, people off the first "
                      f"{r['off_first_before']:,.0f} -> {r['off_first']:,.0f}, jobs {r['jobs_off_first_before']:,.0f} -> "
                      f"{r['jobs_off_first']:,.0f}", flush=True)
            for j in r["joined"]:
                if j["by"] == "car" or j["open"] or max(j["people"], j["jobs"]) >= mn:
                    ex = "; ".join(f"{l['cat']} w{l['way']} {l['highway']} {l['tags']}" for l in j["links"][:3])
                    print(f"[{k + 1}/{len(ready)}] {r['id']} part {j['part']} ({j['people']:,.0f} people, {j['jobs']:,.0f} jobs, "
                          f"{j['lon']},{j['lat']}): {j['by']} to {j['to']}: {ex}", flush=True)
    out.sort(key=lambda r: r["id"])
    with open(os.path.join(ART, "links.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    ok = [r for r in out if "error" not in r]
    big = [(r["id"], t) for r in ok for t in r["top"] if not t["open"] and max(t["people"], t["jobs"]) >= mn]
    jk = {(r["id"], j["part"]): j["by"] for r in ok for j in r["joined"]}
    nb = {c: sum(1 for i, t in big if jk.get((i, t["part"])) == c) for c in ("car", "path", "rail", "planned")}
    print(f"links: done in {time.time() - t0:.0f} s; {len(ok)} packs, {sum(r['changed'] for r in ok)} changed "
          f"({' '.join(r['id'] for r in ok if r['changed'])}), {len(out) - len(ok)} failed; "
          f"{len(big)} interior parts with {mn:,.0f}+ people or jobs: joined by car {nb['car']}, only by path {nb['path']}, "
          f"only by rail {nb['rail']}, only by a road or bridge being built or proposed {nb['planned']}, "
          f"no fixed link {len(big) - sum(nb.values())}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
