"""Put data/ports_<year>.csv on the map using 国土数値情報 C02 港湾 (2014).

Joins on prefecture + port name. The port statistics disambiguate same-named
ports in one prefecture with a bracket — either the manager
(桜島(鹿児島県管理)) or the municipality (小用(江田島市)) — so a bracket is
matched against C02's manager name first, then against OVERRIDES.
Writes data/ports_<year>.geojson and prints every port it could not place.
"""
import csv
import json
import os
import re
import sys
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "4")
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import geopandas as gpd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")
DATA = os.path.join(HERE, "data")
YEAR = 2024

PREFS = [
    "北海道", "青森", "岩手", "宮城", "秋田", "山形", "福島", "茨城", "栃木", "群馬",
    "埼玉", "千葉", "東京", "神奈川", "新潟", "富山", "石川", "福井", "山梨", "長野",
    "岐阜", "静岡", "愛知", "三重", "滋賀", "京都", "大阪", "兵庫", "奈良", "和歌山",
    "鳥取", "島根", "岡山", "広島", "山口", "徳島", "香川", "愛媛", "高知", "福岡",
    "佐賀", "長崎", "熊本", "大分", "宮崎", "鹿児島", "沖縄",
]
PREF_CODE = {name: f"{i + 1:02d}" for i, name in enumerate(PREFS)}

# (prefecture, port as written in the statistics) -> C02 port code, for
# brackets that name a municipality C02 doesn't carry
OVERRIDES = {
    ("鹿児島", "宮之浦(屋久島町)"): "46010",  # muni 46505 屋久島町, not 46404
    ("沖縄", "水納(本部町)"): "47044",  # muni 47308 本部町, not 47375 多良間村
    ("広島", "小用(江田島市)"): "34028",  # muni 34215 江田島市, not the 呉市 one
}


def norm(s):
    s = s or ""
    for a, b in (("奧", "奥"), ("ヶ", "ケ"), ("ヵ", "カ"), ("　", ""), (" ", "")):
        s = s.replace(a, b)
    return s


def split_bracket(port):
    m = re.match(r"^(.*?)[（(](.*?)[)）]$", port)
    if m:
        return m.group(1), m.group(2)
    return port, ""


def main():
    c02 = gpd.read_file(
        os.path.join(RAW, "C02-14", "C02-14_GML", "C02-14-g_PortAndHarbor.shp"), encoding="cp932"
    )
    by_key = {}
    for row in c02.itertuples():
        by_key.setdefault((row.C02_004[:2], norm(row.C02_005)), []).append(row)
    by_code = {row.C02_004: row for row in c02.itertuples()}

    with open(os.path.join(DATA, f"ports_{YEAR}.csv"), encoding="utf-8") as f:
        ports = list(csv.DictReader(f))

    features, missing, ambiguous = [], [], []
    for p in ports:
        pref_code = PREF_CODE.get(p["prefecture"])
        base, bracket = split_bracket(p["port"])
        hit = None
        if (p["prefecture"], p["port"]) in OVERRIDES:
            hit = by_code[OVERRIDES[(p["prefecture"], p["port"])]]
        else:
            cands = by_key.get((pref_code, norm(base)), [])
            if bracket and len(cands) > 1:
                manager = bracket.removesuffix("管理")
                cands = [c for c in cands if norm(c.C02_007) == norm(manager)] or cands
            if len(cands) == 1:
                hit = cands[0]
            elif len(cands) > 1:
                ambiguous.append((p, cands))
        if hit is None:
            if not any(a[0] is p for a in ambiguous):
                missing.append(p)
            continue
        props = dict(p)
        props["c02_code"] = hit.C02_004
        props["c02_manager"] = hit.C02_007
        for k in list(props):
            if k.startswith(("dom_", "intl_")):
                props[k] = int(props[k])
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [round(hit.geometry.x, 5), round(hit.geometry.y, 5)]},
            "properties": props,
        })

    path = os.path.join(DATA, f"ports_{YEAR}.geojson")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f, ensure_ascii=False)

    placed = sum(ft["properties"]["dom_total"] for ft in features)
    total = sum(int(p["dom_total"]) for p in ports)
    print(f"placed {len(features)}/{len(ports)} ports, {placed / total:.1%} of domestic passengers -> {path}")
    for p, cands in ambiguous:
        opts = ", ".join(f"{c.C02_004} {c.C02_005} ({c.C02_007}, muni {c.C02_003})" for c in cands)
        print(f"  ambiguous: {p['prefecture']} {p['port']} {int(p['dom_total']):,} -> {opts}")
    for p in sorted(missing, key=lambda p: -int(p["dom_total"])):
        print(f"  missing: {p['prefecture']} {p['port']} {int(p['dom_total']):,} ({p['table']})")
        pref_code = PREF_CODE.get(p["prefecture"])
        near = [f"{r.C02_004} {r.C02_005}" for r in c02.itertuples()
                if r.C02_004[:2] == pref_code and set(norm(r.C02_005)) & set(norm(p["port"]))]
        print(f"    C02 names sharing a character: {', '.join(near) or 'none'}")


if __name__ == "__main__":
    main()
