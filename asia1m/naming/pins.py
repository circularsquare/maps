"""Pins for candidates that aren't current counties or cities -> batches/<name>/pins.json

    python asia1m/naming/pins.py <batch name>        (after candidates.json is written)

A rivers, mountain, plain or historical name gets a point on the region's thumbnail, from the
first of: lat/lon in the candidate itself; the CHGIS seat of that name nearest the region (for
historical prefectures, commanderies, counties); Chinese Wikipedia's coordinates for the name;
Wikidata's. A river's point is wherever its article puts it, often the mouth. Points outside the
thumbnail are kept and marked, and the viewer draws them on the edge.
"""
import os, sys, json, time, urllib.request, urllib.parse
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "regionnaming"))
import numpy as np
import pyogrio
from geo import lonlat_to_px
sys.stdout.reconfigure(encoding="utf-8")

name = sys.argv[1]
B = os.path.join(HERE, "batches", name)
facts = {r["n"]: r for r in json.load(open(f"{B}/facts.json", encoding="utf-8"))["regions"]}
cands = json.load(open(f"{B}/candidates.json", encoding="utf-8"))
UA = {"User-Agent": "asia1m-naming/0.1 (personal map project)"}
ADMIN = {"county", "city"}

CH = os.path.join(HERE, "..", "..", "data", "asia1m", "chgis")
seats = defaultdict(list)
for z, shp in [("v6_time_pref_pts_utf_wgs84.zip", "v6_time_pref_pts_utf_wgs84.shp"),
               ("v6_time_cnty_pts_utf_wgs84.zip", "v6_time_cnty_pts_utf_wgs84.shp")]:
    df = pyogrio.read_dataframe(f"zip://{CH}/{z}!{shp}", read_geometry=False, columns=["NAME_CH", "X_COOR", "Y_COOR"])
    for nm, x, y in zip(df["NAME_CH"], df["X_COOR"], df["Y_COOR"]):
        if nm and x == x and y == y: seats[nm].append((float(x), float(y)))

def get(url):
    for wait in [1, 3, 8, None]:
        try:
            return json.load(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30))
        except Exception:
            if wait is None: return None
            time.sleep(wait)

def wiki_coord(title):
    r = get("https://zh.wikipedia.org/w/api.php?" + urllib.parse.urlencode(dict(
        action="query", format="json", titles=title, redirects=1, prop="coordinates|pageprops", ppprop="wikibase_item")))
    if not r: return None
    p = next(iter(r["query"]["pages"].values()))
    if "missing" in p: return None
    c = (p.get("coordinates") or [None])[0]
    if c: return (c["lon"], c["lat"], "zh-wiki")
    q = p.get("pageprops", {}).get("wikibase_item")
    if q:
        w = get(f"https://www.wikidata.org/w/api.php?action=wbgetclaims&format=json&entity={q}&property=P625")
        cl = (w or {}).get("claims", {}).get("P625")
        if cl:
            v = cl[0]["mainsnak"].get("datavalue", {}).get("value")
            if v: return (v["longitude"], v["latitude"], "wikidata")
    return None

OVERPASS = "https://overpass-api.de/api/interpreter"
OUA = {"User-Agent": "asia1m-naming/0.1 (personal map project; region names for a population map)"}
def osm_river(local, f):
    """OSM ways named `local` (waterway=*) around the region, in map pixels; ways only, since a
    relation's `out geom` repeats every member and times out."""
    t = f["thumb"]
    from geo import px_to_lonlat
    xs = [t["x0"], t["x0"] + t["w"] / t["k"]]; ys = [t["y0"], t["y0"] + t["h"] / t["k"]]
    lo, la = px_to_lonlat(np.array([xs[0], xs[1], xs[0], xs[1]]), np.array([ys[0], ys[0], ys[1], ys[1]]))
    s, w, n, e = min(la) - 0.5, min(lo) - 0.5, max(la) + 0.5, max(lo) + 0.5
    q = f'[out:json][timeout:60];way["waterway"]["name"="{local}"]({s},{w},{n},{e});out geom;'
    for wait in [5, 15, None]:
        try:
            req = urllib.request.Request(OVERPASS, data=urllib.parse.urlencode({"data": q}).encode(), headers=OUA)
            r = json.load(urllib.request.urlopen(req, timeout=90))
            break
        except Exception:
            if wait is None: return None
            time.sleep(wait)
    lines = []
    for el in r.get("elements", []):
        g = el.get("geometry")
        if not g: continue
        px, py = lonlat_to_px(np.array([p["lon"] for p in g]), np.array([p["lat"] for p in g]))
        lines.append(np.c_[px, py])
    return lines or None

RN = os.path.join(HERE, "..", "regionnaming")
_twn = None
def township(local, f):
    """A township (街道/镇/乡) of that name inside the thumbnail: its pixel centroid, map pixels."""
    global _twn, _tnames
    if _twn is None:
        _twn = np.load(f"{RN}/adm_twn.npy", mmap_mode="r")
        _tnames = json.load(open(f"{RN}/adm_twn_names.json", encoding="utf-8"))
    t = f["thumb"]
    x0, y0 = int(t["x0"]), int(t["y0"])
    win = np.asarray(_twn[y0:y0 + int(t["h"] / t["k"]) + 1, x0:x0 + int(t["w"] / t["k"]) + 1])
    for g in np.unique(win):
        nm = (_tnames[g] or {}).get("town") if g else None
        if nm and nm.startswith(local) and len(nm) - len(local) <= 3:
            ys, xs = np.nonzero(win == g)
            return (x0 + xs.mean(), y0 + ys.mean())
    return None

def wiki_title(url):
    if url and "wikipedia.org/wiki/" in url:
        return urllib.parse.unquote(url.split("/wiki/", 1)[1]).split("#")[0]

def nearest_seat(local, lon, lat):
    best = None
    for key in [local, local + "州", local + "郡", local + "县", local + "府"]:
        for x, y in seats.get(key, []):
            d = (x - lon) ** 2 + (y - lat) ** 2
            if best is None or d < best[0]: best = (d, x, y)
    if best and best[0] < 9:          # within ~3 degrees
        return (best[1], best[2], "CHGIS")
    return None

out = {}
for reg in cands["regions"]:
    f = facts.get(reg["n"])
    if not f or "thumb" not in f: continue
    t = f["thumb"]; pins = []
    for i, c in enumerate(reg["candidates"]):
        if c.get("origin") in ADMIN: continue
        local = (c.get("local") or "").strip()
        pt = None; d = None
        if c.get("origin") == "river" and local:
            lines = None
            for nm in dict.fromkeys([local] + [local + s for s in ("河", "江") if not local.endswith(("河", "江", "水"))]):
                lines = osm_river(nm, f)
                time.sleep(1)
                if lines: break
            if lines:
                tl = [(ln - [t["x0"], t["y0"]]) * t["k"] for ln in lines]
                d = "".join("M" + "L".join(f"{x:.1f},{y:.1f}" for x, y in ln) for ln in tl)
                allp = np.vstack(tl)
                ins = allp[(allp[:, 0] >= 0) & (allp[:, 0] <= t["w"]) & (allp[:, 1] >= 0) & (allp[:, 1] <= t["h"])]
                cx, cy = t["w"] / 2, t["h"] / 2
                use = ins if len(ins) else allp
                x, y = use[np.argmin(((use - [cx, cy]) ** 2).sum(1))]
                pins.append(dict(i=i, name=c["name"], x=round(float(x), 1), y=round(float(y), 1),
                                 inside=bool(len(ins)), src="OSM", d=d))
                print(f"  {reg['n']} {c['name']}: OSM line, {len(lines)} ways")
                continue
        mp = None                         # a point already in map pixels
        if c.get("lat") is not None and c.get("lon") is not None:
            pt = (float(c["lon"]), float(c["lat"]), "agent")
        if not pt and wiki_title(c.get("source_url")):
            pt = wiki_coord(wiki_title(c["source_url"])); time.sleep(0.3)
        if not pt and local and c.get("origin") in ("historical", "prefecture"):
            pt = nearest_seat(local, f["lon"], f["lat"])
        if not pt and local:
            tw = township(local, f)
            if tw: mp = (tw[0], tw[1], "township")
        if not pt and not mp and local and c.get("origin") == "other":
            pt = nearest_seat(local, f["lon"], f["lat"])
        if not pt and not mp and local:
            suffix = {"river": ["河", "江", "水"], "mountain": ["山", "山脉"], "plain": ["原", "塬", "平原"],
                      "lake": ["湖"], "historical": ["宫", "郡", "州"], "other": ["宫", "山", "寺"]}.get(c.get("origin"), [])
            for title in [local] + [local + s for s in suffix if not local.endswith(s)]:
                pt = wiki_coord(title)
                time.sleep(0.3)
                if pt: break
        if not pt and not mp and local and c.get("origin") not in ("historical", "prefecture"):
            pt = nearest_seat(local, f["lon"], f["lat"])
        if not pt and not mp:
            print(f"  {reg['n']} {c['name']}: no point"); continue
        if mp:
            px, py, src = mp
            pt = (None, None, src)
        else:
            px, py = lonlat_to_px(pt[0], pt[1])
        x = (float(px) - t["x0"]) * t["k"]; y = (float(py) - t["y0"]) * t["k"]
        inside = 0 <= x <= t["w"] and 0 <= y <= t["h"]
        pins.append(dict(i=i, name=c["name"], x=round(x, 1), y=round(y, 1), inside=inside, src=pt[2]))
        print(f"  {reg['n']} {c['name']}: {pt[2]}{'' if inside else ' (outside the picture)'}")
    out[reg["n"]] = pins
json.dump(out, open(f"{B}/pins.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"wrote {B}/pins.json: {sum(len(v) for v in out.values())} pins")
