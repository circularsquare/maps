"""T-509 stage 3: Tokyo (`tyo`): real network and calibration targets.

    python pipeline/calib_tyo.py trips     # census lines -> data/work/realnet/tyo_trips.csv
    python pipeline/realnet.py net tyo     # the network (CITIES["tyo"] reads that CSV)
    python pipeline/calib_tyo.py areas     # pack cell -> municipality -> data/work/realnet/tyo_cell_muni.u16
    python pipeline/calib.py targets tyo   # calls targets() here

The network is the 2015 Metropolitan Transportation Census's lines (MLIT, via maps/riders/
tokyoriders: data/processed/segments_kanto.json and .geojson, 141 lines, station to station),
not OSM: every JR, Tokyo Metro, Toei, private and third-sector line in the Tokyo area with its
stations in order. The three Shinkansen are dropped; limited expresses and rapid services are not
in it (every train stops everywhere here). Lines opened since 2015 are missing (Sotetsu-JR and
Sotetsu-Tokyu links, Takanawa Gateway). A census line with branches becomes one line per branch
(the longest path first, then each branch from its junction); the Yamanote is a loop.

Frequencies are defaults by kind of line and its busiest census segment (TIERS), with the
peak-to-midday shape of realnet.OSM_TPH; run times from distance x OSM_DETOUR at OSM_KMH by kind.
Rough, as agreed for Tokyo (no timetable source without an ODPT account).

Stations: census stations of one name within 500 m are one station; each is joined to the MLIT
S12 station group (S12_001g: stations of one name within 300 m, every operator) of the same name
within 500 m, whose id becomes its `target`.

Targets:
- stations: S12 boardings and alightings per group, FY2019 (pre-COVID, S12_041), halved. One
  factor of 1.0 for the year and the day: about 0.9 for FY2024 against FY2019 (MLIT rail
  statistics, Tokyo area, rough) times about 1.1 for a weekday against the year's average day.
  A group with records of more than one operator counts riders changing between them, so it is a
  hub.
- operators: the same FY2019 S12 figures summed per operator and halved (an operator's count at a
  station includes riders arriving from other operators), scored for operators whose stations in
  the network nearly all have a count.
- areas: 2020 census table 17-2 (e-Stat statInfId 000032214690, e17_02.xlsx): commuters and
  students 15+ by municipality of residence using rail, alone or with another means, over those
  whose means is known.
"""
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import realnet as rn  # noqa: E402

TR = os.path.join(rn.MAPS, "riders", "tokyoriders", "data", "processed")
S12 = os.path.join(rn.MAPS, "riders", "japanriders", "data", "S12-22", "UTF-8", "S12-22_NumberOfPassengers.geojson")
RAW = os.path.join(rn.RAW, "t509", "tyo")
TRIPS = os.path.join(rn.WORK, "tyo_trips.csv")
ROUTES = os.path.join(rn.WORK, "tyo_routes.json")
CELL_MUNI = os.path.join(rn.WORK, "tyo_cell_muni.u16")
E17 = os.path.join(RAW, "e17_02.xlsx")
E17_URL = "https://www.e-stat.go.jp/stat-search/file-download?statInfId=000032214690&fileKind=0"
PREFS = ["08", "11", "12", "13", "14"]
# Tohoku, Joetsu and Tokaido Shinkansen; the Seibu Yamaguchi line, whose two stops the census
# puts at one point (a zero-length hop the game's fares refuse)
DROP = {"041", "042", "043", "410"}
S12_FY2019 = "S12_041"
YEAR_DAY_FACTOR = 1.0

OPERATORS = {"東日本旅客鉄道": "jr_east", "東海旅客鉄道": "jr_central", "東京地下鉄": "tokyo_metro", "東京都交通局": "toei",
             "東京都": "toei", "横浜市交通局": "yokohama", "横浜市": "yokohama", "京浜急行電鉄": "keikyu", "小田急電鉄": "odakyu",
             "京王電鉄": "keio", "東京急行電鉄": "tokyu", "東急電鉄": "tokyu", "西武鉄道": "seibu", "東武鉄道": "tobu",
             "京成電鉄": "keisei", "相模鉄道": "sotetsu", "新京成電鉄": "shin_keisei", "北総鉄道": "hokuso",
             "東京臨海高速鉄道": "rinkai", "首都圏新都市鉄道": "tx", "東葉高速鉄道": "toyo", "埼玉高速鉄道": "saitama_rr",
             "横浜高速鉄道": "minatomirai", "東京モノレール": "tokyo_monorail", "ゆりかもめ": "yurikamome",
             "多摩都市モノレール": "tama_monorail", "千葉都市モノレール": "chiba_monorail"}
# metres per train (the game's cars are 20 m): 10-car EMUs as the default; the small-profile
# subways, monorails, guideways and trams by line
TRAIN_M = {"_": 200, "010": 220, "150": 96, "151": 108, "152": 54, "053": 132, "050": 144, "051": 160, "054": 54, "055": 13,
           "354": 25, "880": 54, "860": 54, "980": 26, "510": 30, "410": 30, "870": 60, "820": 30, "821": 30, "960": 90,
           "970": 40, "850": 40, "900": 50, "101": 60, "100": 108}
# trains an hour each way at the peak by kind and the line's busiest census segment. The census
# figure ranks lines (the Yamanote's busiest segment 338k, the Toei Arakawa tram's 1.4k, so it is
# not a full day's count); the peaks are judgement against Tokyo's known headways (2-3 minutes on
# the trunk subways and JR lines, 5-10 on branches)
TIERS = {"metro": [(80_000, 24), (30_000, 18), (0, 8)],
         "commuter": [(150_000, 24), (80_000, 18), (30_000, 12), (10_000, 8), (0, 4)],
         "_": [(30_000, 12), (0, 8)]}


def kind_of(op, line):
    if line in ("荒川線", "世田谷線"):
        return "tram"
    if "モノレール" in op or "モノレール" in line:
        return "monorail"
    if line in ("日暮里・舎人ライナー", "東京臨海新交通臨海線", "金沢シーサイド線", "伊奈線", "ユーカリが丘線", "山口線"):
        return "light_rail"
    if op in ("東京地下鉄", "東京都交通局", "横浜市交通局"):
        return "metro"
    return "commuter"


def norm(name):
    name = re.sub(r"[（(].*?[)）]", "", str(name)).replace("駅", "").replace(" ", "").replace("　", "")
    return name.replace("ケ", "ヶ").replace("ヵ", "ヶ").replace("ッ", "ツ")


def hours_tph(peak):
    """24 hourly trains each way: OSM_TPH's commuter shape scaled to this peak."""
    p, m, f = rn.OSM_TPH["commuter"]
    mid, first = peak * m / p, peak * f / p
    tph = [0.0] * 5 + [first] + [mid] * 16 + [mid / 2] * 2
    for h in (7, 8, 17, 18):
        tph[h] = peak
    return tph


def paths(edges, riders):
    """A line's segments (undirected station pairs) as service paths: repeatedly the longest
    simple path of the edges left, until none. A loop (the Yamanote) is cut at its least used
    segment, since the game's lines do not close (a repeated end stop breaks its fares)."""
    left = {frozenset(e) for e in edges if e[0] != e[1]}
    use = defaultdict(float)
    for e, r in zip(edges, riders):
        use[frozenset(e)] += r
    out = []
    while left:
        adj = defaultdict(set)
        for e in left:
            a, b = tuple(e)
            adj[a].add(b)
            adj[b].add(a)
        best = []
        starts = [n for n in adj if len(adj[n]) == 1] or list(adj)
        for s in starts:
            stack = [(s, [s])]
            n_seen = 0
            while stack and n_seen < 200_000:
                n, p = stack.pop()
                n_seen += 1
                if len(p) > len(best):
                    best = p
                for m in adj[n]:
                    if m not in p:
                        stack.append((m, p + [m]))
        closing = frozenset((best[0], best[-1]))
        if len(best) > 2 and closing in left:
            ring = best + [best[0]]
            cut = min(range(len(best)), key=lambda i: use[frozenset((ring[i], ring[i + 1]))])
            left.discard(frozenset((ring[cut], ring[cut + 1])))
            best = ring[cut + 1:] + ring[1:cut + 1]
        for i in range(len(best) - 1):
            left.discard(frozenset((best[i], best[i + 1])))
        out.append(best)
    return out


def load_s12():
    d = json.load(open(S12, encoding="utf-8"))
    rows = []
    for f in d["features"]:
        p = f["properties"]
        g = f["geometry"]
        pts = g["coordinates"] if g["type"] == "LineString" else [q for part in g["coordinates"] for q in part]
        lon, lat = np.mean([q[0] for q in pts]), np.mean([q[1] for q in pts])
        rows.append({"name": norm(p["S12_001"]), "group": p.get("S12_001g") or p.get("S12_001c") or f"n{p['S12_001']}@{lon:.3f},{lat:.3f}", "op": p["S12_002"],
                     "line": p["S12_003"], "n2019": float(p.get(S12_FY2019) or 0), "lon": float(lon), "lat": float(lat)})
    return pd.DataFrame(rows)


def cmd_trips():
    seg = json.load(open(os.path.join(TR, "segments_kanto.json"), encoding="utf-8"))
    geo = json.load(open(os.path.join(TR, "segments_kanto.geojson"), encoding="utf-8"))
    # census station code -> position, from the segment lines' ends
    pos = {}
    by_names = {}
    for f in geo["features"]:
        p = f["properties"]
        c = f["geometry"]["coordinates"]
        by_names[(p["line_code"], p["from"], p["to"])] = (c[0], c[-1])
    for s in seg["segments"]:
        ends = by_names.get((s["line_code"], s["from_name"], s["to_name"]))
        if ends:
            pos.setdefault(s["from_code"], ends[0])
            pos.setdefault(s["to_code"], ends[1])
    st = seg["stations"]
    o = rn.origin("tyo")
    # stations: one name within 500 m
    key_of, keys = {}, defaultdict(list)  # name -> [(key, x, y)]
    for code in sorted(pos):
        nm = norm(st[code]["name"])
        x, y = rn.to_xy(*pos[code], o)
        k = next((kk for kk, xx, yy in keys[nm] if (xx - x) ** 2 + (yy - y) ** 2 < 500 ** 2), None)
        if k is None:
            k = f"{nm}@{code}"
            keys[nm].append((k, float(x), float(y)))
        key_of[code] = k
    # S12 join: the group of the same name nearest within 500 m
    s12 = load_s12()
    sx, sy = rn.to_xy(s12["lon"].values, s12["lat"].values, o)
    s12["x"], s12["y"] = sx, sy
    by_nm = {nm: g for nm, g in s12.groupby("name")}
    target, kpos = {}, {}
    for nm, ks in keys.items():
        for k, x, y in ks:
            kpos[k] = (x, y)
            g = by_nm.get(nm)
            if g is None:
                continue
            d = np.hypot(g["x"].values - x, g["y"].values - y)
            i = int(np.argmin(d))
            if d[i] < 500:
                target[k] = g["group"].iloc[i]
    n_st = sum(len(v) for v in keys.values())
    print(f"{len(pos)} census station codes -> {n_st} stations, {len(target)} joined to S12")
    unjoined = sorted(k.split("@")[0] for k in kpos if k not in target)
    print("not in S12:", " ".join(unjoined[:60]), "..." if len(unjoined) > 60 else "")
    # lines
    by_line = defaultdict(list)
    for s in seg["segments"]:
        if s["line_code"] in DROP or s["from_code"] not in pos or s["to_code"] not in pos:
            continue
        by_line[s["line_code"]].append(s)
    rows, meta = [], {}
    lon0, lat0 = o
    for lc, ss in sorted(by_line.items()):
        info = seg["lines"][lc]
        kind = kind_of(info["op_name"], info["line_name"])
        peak = next(t for v, t in TIERS.get(kind, TIERS["_"]) if max(s["riders"] for s in ss) >= v)
        tph = hours_tph(peak)
        edges = [(key_of[s["from_code"]], key_of[s["to_code"]]) for s in ss]
        for j, path in enumerate(paths(edges, [s["riders"] for s in ss])):
            if len(path) < 2:
                continue
            route = f"{lc}" if j == 0 else f"{lc}-{j}"
            meta[route] = {"name": info["line_name"], "op": OPERATORS.get(info["op_name"], "other"), "kind": kind,
                           "train_m": TRAIN_M.get(lc, TRAIN_M["_"]), "peak": peak}
            xy = [kpos[k] for k in path]
            hop = [float(np.hypot(xy[i + 1][0] - xy[i][0], xy[i + 1][1] - xy[i][1])) * rn.OSM_DETOUR / (rn.OSM_KMH[kind] / 3.6)
                   for i in range(len(path) - 1)]
            at = np.concatenate([[0.0], np.cumsum(hop)])
            for dr, seq in (("0", list(range(len(path)))), ("1", list(range(len(path) - 1, -1, -1)))):
                t_rel = at if dr == "0" else at[-1] - at
                acc = 0.0
                for h, n in enumerate(tph):
                    acc += n
                    m = int(acc)
                    acc -= m
                    for i in range(m):
                        dep = h * 3600 + (i + 0.5) * 3600 / m
                        trip = f"{route}_{dr}_{h}_{i}"
                        for q in seq:
                            k = path[q]
                            x, y = xy[q]
                            lon = lon0 + np.degrees(x / (rn.R_EARTH * np.cos(np.radians(lat0))))
                            lat = lat0 + np.degrees(y / rn.R_EARTH)
                            rows.append((route, dr, trip, target.get(k, "x" + k), k.split("@")[0], round(lon, 6), round(lat, 6),
                                         int(dep + t_rel[q]), info["line_name"]))
    df = pd.DataFrame(rows, columns=["route", "direction", "trip", "stop", "name", "lon", "lat", "time", "route_name"])
    df.to_csv(TRIPS, index=False, encoding="utf-8")
    json.dump(meta, open(ROUTES, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"{len(meta)} lines, {df['trip'].nunique():,} trips -> {TRIPS}")


def cmd_areas():
    """One u16 per pack cell: its municipality's index in the sorted list of those with a cell."""
    import japan
    import read_pack
    from shapely import STRtree, points
    header, a = read_pack.load_pack("tyo")
    lon0, lat0 = header["origin"]["lon"], header["origin"]["lat"]
    x, y = a["x_m"].astype(float), a["y_m"].astype(float)
    lon = lon0 + np.degrees(x / (rn.R_EARTH * np.cos(np.radians(lat0))))
    lat = lat0 + np.degrees(y / rn.R_EARTH)
    ms = japan.municipalities(PREFS)
    tree = STRtree([m[3] for m in ms])
    pi, gi = tree.query(points(lon, lat), predicate="within")
    codes = sorted({ms[g][0] for g in gi})
    idx = {c: i for i, c in enumerate(codes)}
    out = np.full(len(x), 65535, np.uint16)
    out[pi] = [idx[ms[g][0]] for g in gi]
    out.tofile(CELL_MUNI)
    names = {m[0]: m[1] for m in ms}
    json.dump([[c, names[c]] for c in codes], open(CELL_MUNI + ".ids.json", "w", encoding="utf-8"), ensure_ascii=False)
    print(f"{len(x)} cells, {(out != 65535).sum()} in {len(codes)} municipalities -> {CELL_MUNI}")


def e17():
    """{municipality code: (rail, known)} from table 17-2 by residence."""
    if not os.path.exists(E17):
        import requests
        os.makedirs(RAW, exist_ok=True)
        r = requests.get(E17_URL, headers={"User-Agent": "trainworld-pipeline/1.0"}, timeout=300)
        r.raise_for_status()
        open(E17, "wb").write(r.content)
    x = pd.read_excel(E17, header=None, dtype=str)
    d = x.iloc[10:, [2, 3, 4]]
    d.columns = ["area", "mode", "n"]
    d["code"] = d["area"].str[:5]
    d["n"] = pd.to_numeric(d["n"].str.replace(",", ""), errors="coerce").fillna(0.0)
    m = d["mode"].str.split("_").str[0]
    rail_modes = {"21", "31", "32", "33", "34", "35", "4"}
    out = {}
    for code, g in d.assign(m=m).groupby("code"):
        tot = g.loc[g.m == "0", "n"].sum()
        unk = g.loc[g.m == "5", "n"].sum()
        rail = g.loc[g.m.isin(rail_modes), "n"].sum()
        out[code] = (float(rail), float(tot - unk), float(g.loc[g.m == "1", "n"].sum()))
    return out


def targets():
    net = json.load(open(rn.CITIES["tyo"]["out"], encoding="utf-8"))
    in_net = {s["target"] for s in net["stations"] if s.get("target")}
    s12 = load_s12()
    s12 = s12[s12["group"].isin(in_net)]
    g = s12.groupby("group")
    st = []
    hubs = []
    for gid, r in g:
        st.append({"target": gid, "name": r["name"].iloc[0], "entries": float(r["n2019"].sum()) / 2.0 * YEAR_DAY_FACTOR})
        if r.loc[r["n2019"] > 0, "op"].nunique() > 1 or r["op"].nunique() > 1:
            hubs.append(gid)
    # operators: S12 sums for operators with a count at nearly every one of their stations
    ops, info = {}, {}
    op_of_line = defaultdict(set)
    for gid, r in g:
        for op in r["op"]:
            op_of_line[OPERATORS.get(op, "other")].add(gid)
    for op, gids in op_of_line.items():
        if op == "other":
            continue
        rr = s12[s12["op"].map(lambda o: OPERATORS.get(o, "other")) == op]
        have = rr.groupby("group")["n2019"].sum()
        cover = float((have > 0).mean()) if len(have) else 0.0
        v = float(rr["n2019"].sum()) / 2.0 * YEAR_DAY_FACTOR
        (ops if cover >= 0.9 and op not in ("jr_east",) else info)[op] = v
        print(f"  {op:15s} {len(have):4d} groups, {cover:.0%} with a count, {v:,.0f} a day")
    ids = json.load(open(CELL_MUNI + ".ids.json", encoding="utf-8"))
    e = e17()
    # a municipality with no station of the network (beyond the census lines' reach: northern
    # Ibaraki, Choshi) weighs nothing: its rail riders use lines the network lacks
    import japan
    from shapely import STRtree, points
    ms = japan.municipalities(PREFS)
    tree = STRtree([m[3] for m in ms])
    _, gi = tree.query(points([s["lon"] for s in net["stations"]], [s["lat"] for s in net["stations"]]), predicate="within")
    served = {ms[g][0] for g in gi}
    areas = []
    for code, name in ids:
        rail, known, walk = e.get(code, (0.0, 0.0, 0.0))
        areas.append({"name": f"{code} {name}", "rail_share": rail / known if known else 0.0, "commuters": known if code in served else 0.0,
                      "walk_share": walk / known if known else 0.0})
    print(f"{len(served & {c for c, _ in ids})} of {len(ids)} municipalities have a station")
    miss = [c for c, _ in ids if c not in e]
    if miss:
        print("municipalities not in table 17-2:", miss)
    return {
        "city": "tyo", "net": rn.CITIES["tyo"]["out"], "areas_file": CELL_MUNI,
        "station_min": 2000.0, "hub_m": 300.0, "stations": st, "hubs": hubs,
        "operators": ops, "operators_info": info, "areas": areas,
        "sources": {
            "stations": "MLIT S12 station passengers (boardings and alightings) FY2019 per S12 station group, halved; groups of more than one operator as hubs",
            "operators": "the same S12 FY2019 figures summed per operator and halved; operators with a count at under 90% of their station groups, and JR East (one record per station carries all its lines, internal changes not counted), not scored",
            "areas": "2020 census table 17-2 by municipality of residence: commuters and students 15+ using rail (alone, with one other means, or three or more means) over those whose means is known",
        },
    }


if __name__ == "__main__":
    {"trips": cmd_trips, "areas": cmd_areas}[sys.argv[1]]()
