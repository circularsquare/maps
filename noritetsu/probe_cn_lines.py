"""Stage-1 measurements for mainland China, per named line.

    python probe_cn_lines.py > data/raw/cn/probe_lines.txt

For every heavy-rail line name on OSM track (the way's own `name`, or its infrastructure
relation's where the way has none): track km, the share on highspeed=yes ways, whether an
infrastructure relation of that name carries a national line code, and the OSM stations
within STATION_M of its track -- how many of them are passenger stations (their name is in
12306's ticketing list, data/raw/cn/12306_station_name.js) and how many are not. That last
split is what separates passenger lines from freight-only ones (大秦线, 朔黄线), which OSM's
route relations cannot do in China: only 134 route=train relations exist.

Also: the OSM stations against 12306's list overall, and Wikidata's line chains (by name,
铁路 read as 线) against the OSM line names.
"""
import json
import pickle
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
D = ROOT / "data" / "proc" / "cn"
RAW = ROOT / "data" / "raw" / "cn"
STATION_M = 300
CODE = re.compile(r"^\s*(\d{4})\b")


def nkey(s):
    s = unicodedata.normalize("NFKC", s or "")
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[(（].*?[)）]", "", s)
    for suf in ("火车站", "站"):
        if s.endswith(suf) and len(s) > len(suf) + 1:
            s = s[: -len(suf)]
            break
    return s


def load_12306():
    txt = (RAW / "12306_station_name.js").read_text("utf-8")
    out = {}
    for rec in txt.split("@")[1:]:
        f = rec.split("|")
        if len(f) > 3:
            out[f[1]] = f[2]
    return out


def main():
    import build_tiles as bt
    from scipy.spatial import cKDTree
    with open(D / "ways.pkl", "rb") as f:
        ways = pickle.load(f)
    with open(D / "infra.pkl", "rb") as f:
        infra = pickle.load(f)
    with open(D / "stops.pkl", "rb") as f:
        stops = pickle.load(f)
    c = np.load(D / "coords.npz")
    cid, cx, cy = c["id"], c["x"], c["y"]
    pax = load_12306()
    paxk = {nkey(n) for n in pax}
    print(f"12306 list: {len(pax)} stations")

    rel_name, rel_code = {}, {}
    code_names = defaultdict(set)
    for rid, (tags, members) in infra.items():
        nm = (tags.get("name") or "").strip()
        m = CODE.match(tags.get("ref") or "")
        if nm and m:
            code_names[nm].add(m.group(1))
        for ty, ref, _r in members:
            if ty == "w":
                rel_name.setdefault(ref, nm)

    km = Counter()
    hs = Counter()
    pts = defaultdict(list)
    for wid, (tags, nodes) in ways.items():
        if tags["railway"] not in ("rail", "narrow_gauge", "preserved"):
            continue
        if bt.rank_of(bt.KIND[tags["railway"]], tags) >= 2:
            continue
        name = (tags.get("name") or "").strip() or rel_name.get(wid, "")
        if not name:
            continue
        pos = np.searchsorted(cid, nodes)
        np.clip(pos, 0, cid.size - 1, out=pos)
        pos = pos[cid[pos] == nodes]
        if pos.size < 2:
            continue
        lon, lat = cx[pos] / 1e7, cy[pos] / 1e7
        k = float(np.hypot(np.diff(lon) * np.cos(np.radians(lat[:-1])) * 111.32,
                           np.diff(lat) * 110.57).sum())
        for part in name.split(";"):
            km[part] += k
            if tags.get("highspeed") == "yes":
                hs[part] += k
            pts[part].append(np.column_stack([lon, lat]))

    # stations
    st = []
    for nid, (tags, lon, lat) in stops.items():
        if tags.get("railway") in ("station", "halt") and tags.get("name"):
            if tags.get("station") in ("subway", "light_rail", "monorail") and tags.get("train") != "yes":
                continue
            if any(tags.get(m) == "yes" for m in ("subway", "light_rail", "tram", "monorail")) \
                    and tags.get("train") != "yes":
                continue
            st.append((lon, lat, tags["name"]))
    st_in = sum(1 for s in st if nkey(s[2]) in paxk)
    found = {nkey(s[2]) for s in st} & paxk
    print(f"OSM heavy-rail stations (railway=station/halt, named): {len(st)}; "
          f"{st_in} have a 12306 name; {len(found)} of 12306's {len(paxk)} names found in OSM")
    missing = sorted(paxk - {nkey(s[2]) for s in st})
    print(f"  12306 names with no OSM station: {len(missing)}: {' '.join(missing[:120])}")
    lat0 = 35.0
    kx = 111.32 * np.cos(np.radians(lat0))
    sxy = np.array([[s[0] * 111.32 * np.cos(np.radians(s[1])), s[1] * 110.57] for s in st])
    tree = cKDTree(sxy)

    rows = []
    for name, k in km.items():
        if k < 5:
            continue
        a = np.vstack(pts[name])
        xy = np.column_stack([a[:, 0] * 111.32 * np.cos(np.radians(a[:, 1])), a[:, 1] * 110.57])
        near = set()
        for lst in tree.query_ball_point(xy, STATION_M / 1000):
            near.update(lst)
        n_p = sum(1 for i in near if nkey(st[i][2]) in paxk)
        rows.append((name, k, hs[name] / k, n_p, len(near) - n_p,
                     ",".join(sorted(code_names.get(name, ())))))
    rows.sort(key=lambda r: -r[1])
    tot = sum(r[1] for r in rows)
    print(f"\n{len(rows)} line names with 5+ km of main/branch track, {tot:,.0f} km of track "
          f"(double track counted twice)")
    hs_lines = [r for r in rows if r[2] >= 0.5]
    print(f"  high-speed (half or more of its track highspeed=yes): {len(hs_lines)} lines, "
          f"{sum(r[1] for r in hs_lines):,.0f} km")
    coded = [r for r in rows if r[5]]
    print(f"  with a national line code on an infrastructure relation of the same name: "
          f"{len(coded)} lines, {sum(r[1] for r in coded):,.0f} km")
    for lo, hi in ((0, 0.001), (0.001, 0.2), (0.2, 0.5), (0.5, 1.01)):
        sel = [r for r in rows if lo <= r[3] / max(1, r[3] + r[4]) < hi]
        print(f"  passenger share of stations on track {lo:.1f}-{hi:.1f}: {len(sel)} lines, "
              f"{sum(r[1] for r in sel):,.0f} km")
    print("\n     km   hs%  pax  other  code  name")
    for name, k, h, n_p, n_o, code in rows[:400]:
        print(f"  {k:7.0f} {100 * h:4.0f}  {n_p:4d}  {n_o:4d}  {code:>5}  {name}")

    # Wikidata chains by name
    tail = lambda u: u.rsplit("/", 1)[-1]
    lines = json.loads((RAW / "wd_lines.json").read_text("utf-8"))["rows"]
    adj = json.loads((RAW / "wd_adjacency.json").read_text("utf-8"))["rows"]
    lab = {}
    for r in lines:
        if r.get("lab") and r["lang"] in ("zh-cn", "zh-hans", "zh"):
            lab.setdefault(tail(r["x"]), r["lab"])
    per = defaultdict(set)
    for r in adj:
        per[tail(r["line"])].add(tail(r["s"]))
    names = set(km)
    hit = Counter()
    for q, s in per.items():
        nm = lab.get(q, "")
        cands = {nm, nm.replace("高速铁路", "高速线").replace("客运专线", "客专线")
                 .replace("城际铁路", "城际线").replace("铁路", "线")}
        if cands & names:
            hit["matched"] += 1
            hit["stations"] += len(s)
        else:
            hit["not"] += 1
    print(f"\nWikidata lines with adjacency whose label (铁路 -> 线) is an OSM line name: {dict(hit)}")


if __name__ == "__main__":
    main()
