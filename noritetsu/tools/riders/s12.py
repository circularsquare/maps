"""Japan: 国土数値情報 S12 駅別乗降客数 (MLIT), edition S12-25, FY2011-FY2024.

One record per station GROUP (S12_001g, the same name within 300 m across operators, the same
code system as N02's N02_005g that jp's station ids are made of), as riders/japanriders'
build_stations.py does it:

- the figure is 乗降客数, boardings + alightings per day (JR East's are its 乗車人員 doubled
  by MLIT), for the latest fiscal year in which ANY record of the group is positive, and only
  that year: until FY2018 every operator of a jointly run station reported the same joint
  figure (寄居: Chichibu, JR East and Tobu each 7,476), from FY2019 one operator carries the
  joint figure with an "X を含む" remark and the others report 0. One year per group keeps a
  group inside one convention;
- the operators' records for that year are added (新宿: JR East + Odakyu + Keio + Tokyo Metro,
  Toei's inside Keio's, as the remarks say). Adding operators counts a passenger changing
  between them twice, which is the usual way Japanese station totals are quoted.

A record with no group code is its own group.
"""
import json
import zipfile

KEY = "s12"
CC = "jp"
FOLDER = "s12"
RADIUS_KM = 1.0
COMBINE = "sum"     # only if two S12 groups land on one jp station (logged as SEVERAL)
META = {
    "label": "Station counts from Japan's transport ministry (MLIT)",
    "name": "国土数値情報 駅別乗降客数データ (S12-25), MLIT",
    "url": "https://nlftp.mlit.go.jp/ksj/gml/datalist/KsjTmplt-S12-2025.html",
    "licence": "国土数値情報 terms: free use with attribution (CC BY 4.0 compatible)",
    "counts": "boardings + alightings per day, all operators at the station added",
    "note": "fiscal year (April-March) by its first year",
}
ZIP = "S12-25_GML.zip"
MEMBER = "S12-25_GML/UTF-8/S12-25_NumberOfPassengers.geojson"
FIRST_YEAR = 2011


def midpoint(geom):
    if geom["type"] == "LineString":
        pts = geom["coordinates"]
    elif geom["type"] == "MultiLineString":
        pts = [p for part in geom["coordinates"] for p in part]
    else:
        return geom["coordinates"]
    return [sum(a) / len(pts) for a in zip(*pts)]


def records(raw):
    with zipfile.ZipFile(raw / ZIP) as z:
        feats = json.load(z.open(MEMBER))["features"]
    nblocks = (max(int(k[4:7]) for k in feats[0]["properties"]
                   if k[4:7].isdigit()) - 5) // 4
    years = [FIRST_YEAR + i for i in range(nblocks)]

    groups = {}
    for f in feats:
        p = f["properties"]
        per = {}
        for i, y in enumerate(years):
            c = p.get(f"S12_{6 + i * 4 + 3:03d}")
            if c and c > 0:
                per[y] = int(c)
        if not per:
            continue
        g = p.get("S12_001g") or ("solo", p["S12_001"], p["S12_002"], p["S12_003"],
                                  tuple(midpoint(f["geometry"])))
        groups.setdefault(g, []).append((p, per, midpoint(f["geometry"])))

    out = []
    for g, items in groups.items():
        year = max(y for _, per, _ in items for y in per)
        recs = sorted(((per[year], p, xy) for p, per, xy in items if per.get(year)),
                      key=lambda t: -t[0])
        total = sum(t[0] for t in recs)
        _, p0, xy0 = recs[0]
        codes = []
        if isinstance(g, str):
            codes.append(g)
        for _, p, _ in items:
            if p.get("S12_001c") and p["S12_001c"] not in codes:
                codes.append(p["S12_001c"])
        out.append({
            "name": p0["S12_001"], "x": xy0[0], "y": xy0[1],
            "sids": [f"g{c}" for c in codes],
            "n": total, "year": year,
            "ops": sorted({p["S12_002"] for _, p, _ in recs}),
        })
    return out
