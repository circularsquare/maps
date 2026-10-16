"""T-509 stage 3: Sendai's calibration targets (the network: `realnet.py net sdj`, OSM).

    python pipeline/realnet.py net sdj     # the network
    python pipeline/calib.py targets sdj   # calls targets() here: joins the network's stations to
                                           # S12 (writes their `target` into sdj_real.json), writes
                                           # cell_muni_sdj.u8 and the targets

Sources, all open and national, so any Japanese city reuses them:
- MLIT KSJ S12 (FY2011-21, maps/riders/japanriders/data/S12-22): passengers a day per station
  and operator, boardings plus alightings, so entries are half; FY2019 (pre-COVID). A row whose
  figure is counted on another row of its group (JR's Sendai lines, Tozai at Sendai) has 0. One
  target per S12 station group and operator: an operator's gates count riders changing in from
  another operator, so those stations are reported apart (hubs), as are the Shinkansen stations.
- 2020 census table 18 (e-Stat file e18.xlsx, statInfId 000032214691): commuters and students
  15+ by municipality of residence and means of transport, more than one answer allowed; rail
  share = those using rail over all with a known means.
"""
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import realnet as rn  # noqa: E402

S12 = os.path.join(rn.MAPS, "riders", "japanriders", "data", "S12-22", "UTF-8", "S12-22_NumberOfPassengers.geojson")
E18 = os.path.join(rn.RAW, "t509", "census2020_e18.xlsx")
AREAS = os.path.join(rn.WORK, "cell_muni_sdj.u8")
PREFS = ("04", "06", "07")
FY2019 = "S12_041"
OPERATOR = {"東日本旅客鉄道": "jr", "仙台空港鉄道": "jr", "仙台市": "subway", "阿武隈急行": "abukuma", "福島交通": "iizaka"}
SHINKANSEN = {"仙台", "古川", "福島", "白石蔵王", "くりこま高原", "一ノ関"}
MATCH_M = 800.0
HUB_M = 400.0


def norm(s):
    return re.sub(r"[（(].*?[)）]", "", s or "").strip().removesuffix("駅")


def s12_targets(o):
    """{target: (name, operator, entries a day, x, y)} for the operators the network has."""
    feats = json.load(open(S12, encoding="utf-8"))["features"]
    acc = defaultdict(lambda: [None, None, 0.0, [], []])
    for f in feats:
        p = f["properties"]
        op = OPERATOR.get(p["S12_002"])
        if not op:
            continue
        c = np.array(f["geometry"]["coordinates"], float).reshape(-1, 2).mean(axis=0)
        x, y = rn.to_xy(c[0], c[1], o)
        a = acc[f"{p['S12_001g']}:{op}"]
        a[0], a[1] = p["S12_001"], op
        a[2] += float(p.get(FY2019) or 0) / 2
        a[3].append(float(x))
        a[4].append(float(y))
    return {k: (v[0], v[1], v[2], float(np.mean(v[3])), float(np.mean(v[4]))) for k, v in acc.items()}


def join(net, tg):
    """Each network station to the S12 target of one of its lines' operators: the same name
    within MATCH_M, else none. A target claimed by two stations keeps the nearer."""
    ops = defaultdict(set)
    for l in net["lines"]:
        for s in l["stops"]:
            ops[s].add(l["operator"])
    best = {}
    for i, s in enumerate(net["stations"]):
        s["target"] = None
        nm = norm(s["name"].split(" / ")[0])
        for k, (tn, op, _, x, y) in tg.items():
            d = float(np.hypot(x - s["x"], y - s["y"]))
            if op in ops[i] and norm(tn) == nm and d < MATCH_M and (k not in best or d < best[k][1]):
                best[k] = (i, d)
    for k, (i, _) in best.items():
        if net["stations"][i]["target"] is None:
            net["stations"][i]["target"] = k
    return sum(1 for s in net["stations"] if s["target"]), len(net["stations"])


def cell_areas(o):
    """One byte per pack cell: the index of its municipality (N03 2025) among those listed."""
    import japan
    import read_pack
    from shapely import contains_xy
    header, a = read_pack.load_pack("sdj")
    lon0, lat0 = header["origin"]["lon"], header["origin"]["lat"]
    x, y = a["x_m"].astype(float), a["y_m"].astype(float)
    lon = lon0 + np.degrees(x / (rn.R_EARTH * np.cos(np.radians(lat0))))
    lat = lat0 + np.degrees(y / rn.R_EARTH)
    munis = [m for p in PREFS for m in japan.municipalities([p])]
    idx = np.full(len(x), -1)
    for j, (code, name, _, g) in enumerate(munis):
        m = (idx < 0) & contains_xy(g, lon, lat)
        idx[m] = j
    used = sorted(set(idx[idx >= 0].tolist()))
    assert len(used) < 255
    out = np.full(len(x), 255, np.uint8)
    for k, j in enumerate(used):
        out[idx == j] = k
    out.tofile(AREAS)
    pop = np.asarray(a["pop"], float)
    return [(munis[j][0], munis[j][1], float(pop[idx == j].sum())) for j in used]


def census_rail(codes):
    """{code: (rail share, commuters with a known means)} from table 18."""
    import pandas as pd
    x = pd.read_excel(E18, header=None, dtype=str)
    x = x[x[2].astype(str).str[:5].isin(codes)]
    out = {}
    for code, g in x.groupby(x[2].str[:5]):
        v = {r[3][:2]: float(str(r[4]).replace(",", "").replace("-", "0") or 0) for r in g.itertuples(index=False)}
        known = v["00"] - v.get("10", 0.0)
        if known > 0:
            out[code] = (v["02"] / known, known)
    return out


def targets():
    net_path = rn.CITIES["sdj"]["out"]
    net = json.load(open(net_path))
    o = rn.origin("sdj")
    tg = s12_targets(o)
    n_joined, n_st = join(net, tg)
    json.dump(net, open(net_path, "w"))
    used = {s["target"] for s in net["stations"] if s["target"]}
    stations = [{"target": k, "name": tg[k][0], "operator": tg[k][1], "entries": tg[k][2]} for k in sorted(used) if tg[k][2] > 0]
    # hubs: Shinkansen stations, and stations with another operator's gates within HUB_M (riders
    # changing in tap in again and count as entries)
    hubs = []
    for k in used:
        n, op, _, x, y = tg[k]
        if n in SHINKANSEN or any(o2 != op and np.hypot(x2 - x, y2 - y) < HUB_M for k2, (_, o2, _, x2, y2) in tg.items() if k2 in used):
            hubs.append(k)
    sub = sum(s["entries"] for s in stations if s["operator"] == "subway")
    jr = sum(s["entries"] for s in stations if s["operator"] == "jr")
    munis = cell_areas(o)
    rail = census_rail([m[0] for m in munis])
    # areas mostly inside the pack (pack residents at least 1.5 x the census commuters, about 3/4
    # of the residents, since about half of residents commute)
    areas = []
    for code, name, pop in munis:
        if code in rail and pop >= 1.5 * rail[code][1]:
            areas.append({"name": name, "code": code, "rail_share": rail[code][0], "commuters": rail[code][1]})
        else:
            areas.append({"name": name, "code": code, "rail_share": 0.0, "commuters": 0.0})
    print(f"{n_joined} of {n_st} stations joined to S12; {sum(1 for a in areas if a['commuters'])} of {len(areas)} municipalities scored")
    return {
        "city": "sdj", "net": net_path, "areas_file": AREAS, "station_min": 1000.0, "hub_m": 300.0,
        "stations": stations, "hubs": hubs,
        "operators": {"subway": sub},
        "operators_info": {"jr": jr},
        "areas": areas,
        "sources": {
            "stations": "MLIT KSJ S12-22, FY2019 passengers a day (boardings plus alightings) halved, by S12 station group and operator; not scaled to 2024",
            "operators": "Sendai subway: its S12 entries summed (every subway trip passes a subway gate once); JR info only (includes Shinkansen riders)",
            "areas": "2020 census table 18 (e18.xlsx), commuters and students 15+ by municipality of residence using rail, over those with a known means; "
                     "municipalities mostly outside the pack carry no weight"},
    }
