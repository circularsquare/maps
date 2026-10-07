"""Northern Cyprus, 2011 census (TRNC State Planning Organisation / Statistics Institute), drawn
as Turkish by village. Anita's ruling on ask 015 (2026-10-05): the census asked no language
question; everyone it counted is drawn on Turkish, tier `derived`.

    python sources/cy_north.py --fetch    3 census .xls + one Overpass query (OSM boundaries)
    python sources/cy_north.py            rebuild from data/raw/cy/north2011/

Writes
  data/normalized/cy_north.csv   one row per drawn unit: the 2011 usual-resident population
                                 (Tablo 3, mahalle level, summed onto OSM village polygons)
  data/geo/cy/cy_north_units.gpkg  the north's units (unit, name, ilce, pop2011, geometry)
  data/geo/cy/cy_hexes.gpkg      the WHOLE ISLAND's placement layer: religiondots' south hexes
                                 (read-only copy) minus any whose centroid is in the north, plus
                                 the north's Kontur hexes keyed to the units above

THE LINE. The south is drawn on GISCO LAU 2021's 396 enumerated communities, as religiondots
draws it. GISCO's Lefkosia municipality (1000) is the pre-1974 municipality and runs across the
Green Line into the walled city's north half, so religiondots' hexes for 1000 include north
Nicosia. The north here is OSM relation 2514541 (Northern Cyprus, the de facto line) minus OSM's
UN Buffer Zone (3263909). Any south hex whose centroid is inside that mask is dropped from the
south's layer, and no north hex is taken from outside it, so the two never share a hex and
nothing is placed in the buffer zone. Every number is printed.

UNITS. The census's finest level is the mahalle (quarter or village), 250 of them, with Turkish
names and no codes. OSM maps the TRNC's villages (admin_level 8) and many town quarters (9).
Each census row goes to one or more admin-8 villages: by its name within its ilce, else via an
admin-9 quarter of that name, else by a fixed list below, else to the village of its belediye
(the town quarters of Lefkosa, Gazimagusa, Girne and Guzelyurt that OSM has no polygon for).
Rows that share a village are summed; villages joined by one row are merged.

PYLA (Pile, 479 people) is in the UN buffer zone and is the one row not drawn: it would sit on
the buffer zone or on the south's Pyla community. It is in `gap`.

Citizenship (Tablo 5) is printed and recorded; it does not change the language drawn.
"""
import argparse
import json
import os
import re
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "cy", "north2011")
GEO = os.path.join(ROOT, "data", "geo", "cy")
NORM = os.path.join(ROOT, "data", "normalized", "cy_north.csv")
RD = os.path.join(os.path.dirname(ROOT), "religiondots")
RD_HEXES = os.path.join(RD, "data", "geo", "cy", "cy_hexes.gpkg")
RD_LAU = os.path.join(RD, "data", "geo", "cy", "cy_lau.gpkg")
KONTUR = os.path.join(RD, "data", "raw", "cy", "kontur_population_CY_20231101.gpkg")

BASE = "https://istatistik.gov.ct.tr/Portals/39/"
XLS = ["Tablo-1-IlceCinsiyet.xls", "Tablo-3-Mahalle_Cinsiyet.xls", "Tablo-5-Citizenship.xls"]
OSM_JSON = os.path.join(RAW, "osm_boundaries.json")
UA_WEB = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36"
UA_OVERPASS = "languagedots-research/0.1 (python-requests)"
NORTH_REL, BUFFER_REL = 2514541, 3263909
OVERPASS_Q = f"""[out:json][timeout:300];
area(id:{3600000000 + NORTH_REL})->.n;
(
 rel({NORTH_REL});
 rel({BUFFER_REL});
 rel["boundary"="administrative"]["admin_level"~"^(5|8|9)$"](area.n);
);
out geom;"""

TOTAL_2011 = 286257
NOT_DRAWN = {"PİLE"}          # buffer zone
# census name -> OSM admin-8 names, where the name rules do not reach
MANUAL = {
    "MALATYA - İNCESU": ["Malatya", "İncesu"],
    "ZEYTİNLİK KESİM": ["Zeytinlik"],
    "ZEYTİNLİK KÖY": ["Zeytinlik"],
    "EDREMİT": ["Erdemit"],
    "KILIÇARSLAN": ["Kılıçaslan"],
    "KARAMAN (YUKARI KARMİ)": ["Karaman"],
    "YUKARI GİRNE": ["Girne"],
    "KAPALI MARAŞ": ["Gazimağusa"],
}
# census name -> (lon, lat) of its OSM place node, for a village with no polygon of its own
MANUAL_POINT = {"KANTARA": (33.8984756, 35.387272)}     # OSM node 6029501335
# 2011 ilce -> OSM ilce names (Lefke was split from Guzelyurt in 2016)
ILCE = {"Lefkoşa": ["Lefkoşa ilçesi"], "Gazimağusa": ["Gazimağusa ilçesi"],
        "Girne": ["Girne ilçesi"], "Güzelyurt": ["Güzelyurt ilçesi", "Lefke ilçesi"],
        "İskele": ["İskele ilçesi"]}


def tr_key(s):
    s = str(s).replace("\xa0", " ").replace("i", "İ").replace("ı", "I").upper()
    s = re.sub(r"\bMAHALLES[İI]\b", "", s)
    return re.sub(r"[^A-ZÇĞİÖŞÜ]", "", s)


def fetch():
    import requests
    import time
    os.makedirs(RAW, exist_ok=True)
    for f in XLS:
        r = requests.get(BASE + f, headers={"User-Agent": UA_WEB}, timeout=120)
        r.raise_for_status()
        if r.content[:4] != b"\xd0\xcf\x11\xe0":
            raise SystemExit(f"{f}: not an .xls")
        open(os.path.join(RAW, f), "wb").write(r.content)
        print(f"  {f} {len(r.content):,} bytes")
    eps = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter"]
    for i in range(6):
        ep = eps[i % 2]
        try:
            r = requests.post(ep, data={"data": OVERPASS_Q}, headers={"User-Agent": UA_OVERPASS},
                              timeout=400)
            print(f"  {ep} {r.status_code}")
            if r.status_code == 200:
                open(OSM_JSON, "w", encoding="utf-8").write(r.text)
                return
        except Exception as e:  # noqa: BLE001
            print(f"  {ep} {e}")
        time.sleep(15)
    raise SystemExit("Overpass failed")


def read_census():
    import pandas as pd
    x = pd.read_excel(os.path.join(RAW, XLS[1]), header=None)
    rows, ilce, bucak, beled, beltot = [], None, None, None, {}
    for i in range(5, len(x)):
        c0, c1, c2, c3, n = (x.iloc[i, j] for j in range(5))
        if pd.notna(c0):
            ilce = str(c0).strip()
        if pd.notna(c1) and str(c1).strip() != "İlçe Toplam":
            bucak = str(c1).strip()
        if pd.notna(c2) and str(c2).strip() != "Bucak Toplamı":
            beled = str(c2).strip()
            if pd.isna(c3):
                beltot[(bucak, beled)] = int(n)
        if pd.notna(c3) and pd.notna(n):
            rows.append(dict(ilce=ilce, bucak=bucak, belediye=beled,
                             mahalle=str(c3).strip(), pop=int(n)))
    df = pd.DataFrame(rows)
    if df["pop"].sum() != TOTAL_2011:
        raise SystemExit(f"mahalles sum to {df['pop'].sum():,}, expected {TOTAL_2011:,}")
    t1 = pd.read_excel(os.path.join(RAW, XLS[0]), header=None)
    t1 = {str(t1.iloc[i, 0]).strip(): int(t1.iloc[i, 1]) for i in range(5, 10)}
    got = df.groupby("ilce")["pop"].sum().to_dict()
    if got != t1:
        raise SystemExit(f"ilce sums {got} != Tablo 1 {t1}")
    bel = df.groupby(["bucak", "belediye"])["pop"].sum().to_dict()
    bad = {k: (v, bel.get(k)) for k, v in beltot.items() if bel.get(k) != v}
    if bad:
        raise SystemExit(f"belediye totals disagree: {bad}")
    print(f"census: {len(df)} mahalles, {df['pop'].sum():,} people; ilce sums = Tablo 1; "
          f"{len(beltot)} belediye totals = their mahalles")
    return df, t1


def read_citizenship(t1):
    import pandas as pd
    x = pd.read_excel(os.path.join(RAW, XLS[2]), header=None)
    out = {}
    for i in range(6, 20):
        lab = x.iloc[i, 0] if pd.notna(x.iloc[i, 0]) else x.iloc[i, 1]
        out[str(lab).strip()] = int(x.iloc[i, 2])
    if out["GENEL TOPLAM"] != TOTAL_2011:
        raise SystemExit("Tablo 5 total")
    kktc = out["KKTC TOPLAM"]
    if out["YALNIZ KKTC"] + out["KKTC - Türkiye"] + out["KKTC - Diğer"] != kktc:
        raise SystemExit("Tablo 5 KKTC parts")
    rest = [k for k in out if k not in ("GENEL TOPLAM", "KKTC TOPLAM", "YALNIZ KKTC",
                                        "KKTC - Türkiye", "KKTC - Diğer")]
    if kktc + sum(out[k] for k in rest) != TOTAL_2011:
        raise SystemExit("Tablo 5 rows do not sum")
    print("citizenship (Tablo 5): " + ", ".join(f"{k} {v:,}" for k, v in out.items()))
    return out


def osm_polygons():
    import geopandas as gpd
    from shapely.geometry import LineString
    from shapely.ops import polygonize, unary_union
    js = json.load(open(OSM_JSON, encoding="utf-8"))
    recs = []
    for e in js["elements"]:
        if e["type"] != "relation":
            continue
        parts = {"outer": [], "inner": []}
        for m in e.get("members", []):
            if m["type"] == "way" and "geometry" in m:
                role = "inner" if m.get("role") == "inner" else "outer"
                parts[role].append(LineString([(p["lon"], p["lat"]) for p in m["geometry"]]))
        outer = unary_union(list(polygonize(unary_union(parts["outer"])))) if parts["outer"] else None
        if outer is None or outer.is_empty:
            print(f"  !! relation {e['id']} {e['tags'].get('name')} has no polygon")
            continue
        if parts["inner"]:
            outer = outer.difference(unary_union(list(polygonize(unary_union(parts["inner"])))))
        t = e["tags"]
        recs.append(dict(rel=e["id"], level=t.get("admin_level"), name=t.get("name"),
                         geometry=outer))
    g = gpd.GeoDataFrame(recs, geometry="geometry", crs=4326)
    return g


def main():
    import geopandas as gpd
    import numpy as np
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    if ap.parse_args().fetch:
        fetch()

    df, t1 = read_census()
    cit = read_citizenship(t1)

    osm = osm_polygons()
    eq = osm.to_crs(6933)          # equal-area for km2
    osm["km2"] = eq.area / 1e6
    north = osm[osm["rel"] == NORTH_REL].geometry.iloc[0]
    buffer = osm[osm["rel"] == BUFFER_REL].geometry.iloc[0]
    mask = north.difference(buffer)
    km = lambda geom: float(gpd.GeoSeries([geom], crs=4326).to_crs(6933).area.iloc[0] / 1e6)  # noqa: E731
    print(f"OSM north {km(north):,.0f} km2, buffer zone {km(buffer):,.0f} km2, "
          f"north minus buffer {km(mask):,.0f} km2, north within buffer {km(north.intersection(buffer)):,.1f} km2")

    lau = gpd.read_file(RD_LAU).to_crs(4326)
    ov = lau.geometry.intersection(mask)
    ovkm = gpd.GeoSeries(ov, crs=4326).to_crs(6933).area / 1e6
    big = lau.assign(km2=ovkm.to_numpy())[ovkm.to_numpy() > 0.05].sort_values("km2", ascending=False)
    print(f"south communities overlapping the north mask: {len(big)}, "
          f"{ovkm.sum():.1f} km2 in all: " + ", ".join(f"{r.unit} {r['name']} {r.km2:.2f}"
                                                      for _, r in big.head(8).iterrows()))

    ilces = osm[osm["level"] == "5"]
    a8 = osm[osm["level"] == "8"].copy()
    a9 = osm[osm["level"] == "9"].copy()
    rp = a8.geometry.representative_point()
    a8 = a8[rp.within(mask).to_numpy()].copy()      # villages the TRNC claims in the south drop out
    def ilce_of(geoms):
        pts = gpd.GeoDataFrame(geometry=geoms.representative_point(), crs=4326)
        j = gpd.sjoin(pts, ilces[["name", "geometry"]], how="left", predicate="within")
        return j[~j.index.duplicated()]["name"].reindex(pts.index)
    a8["ilce"] = ilce_of(a8.geometry)
    a9["ilce"] = ilce_of(a9.geometry)
    a8["key"] = a8["name"].map(tr_key)
    a9["key"] = a9["name"].map(tr_key)
    # admin-9 -> the admin-8 that holds it
    p9 = gpd.GeoDataFrame(a9[["rel"]], geometry=a9.geometry.representative_point(), crs=4326)
    j9 = gpd.sjoin(p9, a8[["rel", "geometry"]].rename(columns={"rel": "a8"}), how="left",
                   predicate="within")
    a9["a8"] = j9[~j9.index.duplicated()]["a8"].reindex(a9.index)
    print(f"OSM: {len(a8)} villages (admin 8) inside the mask, {len(a9)} quarters (admin 9)")

    def find8(key, ilce):
        c = a8[(a8["key"] == key) & a8["ilce"].isin(ILCE[ilce])]
        return list(c["rel"])

    def find9(key, ilce):
        c = a9[(a9["key"] == key) & a9["ilce"].isin(ILCE[ilce]) & a9["a8"].notna()]
        return sorted(set(int(v) for v in c["a8"]))

    links, how = [], {}
    for r in df.itertuples():
        if r.mahalle in NOT_DRAWN:
            links.append([]); how[r.mahalle] = "not drawn"; continue
        key = tr_key(r.mahalle)
        if r.mahalle in MANUAL:
            t = [int(a8[(a8["name"] == n) & a8["ilce"].isin(ILCE[r.ilce])]["rel"].iloc[0])
                 for n in MANUAL[r.mahalle]]
            src = "manual"
        elif r.mahalle in MANUAL_POINT:
            from shapely.geometry import Point
            t = list(a8[a8.geometry.contains(Point(*MANUAL_POINT[r.mahalle]))]["rel"])
            src = "manual point"
            if len(t) != 1:
                raise SystemExit(f"!! {r.mahalle}'s point is in {t}")
        elif len(find8(key, r.ilce)) == 1:
            t, src = find8(key, r.ilce), "village"
        elif len(find9(key, r.ilce)) == 1:
            t, src = find9(key, r.ilce), "quarter"
        else:
            bk = tr_key(re.sub(r"\(.*\)", "", r.belediye))
            t, src = find8(bk, r.ilce), "belediye"
            if len(t) != 1:
                raise SystemExit(f"!! no polygon for {r.mahalle} ({r.belediye}, {r.ilce}): {t}")
        links.append(t)
        how[r.mahalle + "|" + r.belediye] = src
    df["a8"] = links
    srcs = pd.Series(how).value_counts().to_dict()
    print(f"census rows matched: {srcs}")
    for r in df.itertuples():
        if how.get(r.mahalle + "|" + r.belediye) in ("belediye",):
            print(f"    by belediye: {r.mahalle} ({r.pop:,}) -> {r.belediye}")

    # union-find: villages a single row spans are one unit
    parent = {int(v): int(v) for v in a8["rel"]}
    def f(v):
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v
    for t in df["a8"]:
        for v in t[1:]:
            parent[f(v)] = f(t[0])
    df["unit"] = [("N" + str(f(t[0]))) if t else None for t in df["a8"]]
    a8["unit"] = ["N" + str(f(int(v))) for v in a8["rel"]]
    drawn = df[df["unit"].notna()]
    upop = drawn.groupby("unit")["pop"].sum()
    # Villages with no 2011 row. One mostly inside the mask is ground the census counted under a
    # neighbour (a village formed or split off after 2011, or uninhabited land): it joins the
    # used unit of the same ilce it shares the longest border with, so its hexes are placed. One
    # mostly outside the mask is a village the TRNC lists in the south or the buffer zone: left out.
    a8["mkm2"] = gpd.GeoSeries(a8.geometry.intersection(mask), crs=4326).to_crs(6933).area / 1e6
    a8["share"] = a8["mkm2"] / a8["km2"]
    for _ in range(3):
        used_ids = set(upop.index)
        for i, r in a8[~a8["unit"].isin(used_ids) & (a8["share"] >= 0.5)].iterrows():
            nb = a8[a8["unit"].isin(used_ids) & (a8["ilce"] == r["ilce"])]
            shared = gpd.GeoSeries(nb.geometry.intersection(r.geometry.boundary),
                                   crs=4326).to_crs(6933).length
            if len(shared) and shared.max() > 0:
                a8.at[i, "unit"] = nb.loc[shared.idxmax(), "unit"]
                print(f"    {r['name']} ({r['ilce']}, no 2011 row) joins "
                      f"{nb.loc[shared.idxmax(), 'name']}")
    used = a8[a8["unit"].isin(upop.index)]
    unused = a8[~a8["unit"].isin(upop.index)]
    print(f"units: {len(upop)} from {len(used)} villages; {len(unused)} OSM villages left out "
          f"(mostly south of the line, or no neighbour): "
          + ", ".join(f"{n} {s:.0%}" for n, s in zip(unused["name"], unused["share"])))

    units = used.dissolve(by="unit", aggfunc={"name": lambda s: " / ".join(sorted(s)),
                                              "ilce": "first"}).reset_index()
    units["pop2011"] = units["unit"].map(upop).astype(int)
    units["geometry"] = units.geometry.intersection(mask)
    if int(units["pop2011"].sum()) != TOTAL_2011 - int(df[df["unit"].isna()]["pop"].sum()):
        raise SystemExit("unit populations do not add up")
    # no two units overlap, and none reaches the south's enumerated communities beyond the mask
    ua = units.to_crs(6933)
    over = float(ua.area.sum() - ua.union_all().area) / 1e6 if hasattr(ua, "union_all") \
        else float(ua.area.sum() - ua.unary_union.area) / 1e6
    print(f"units cover {ua.area.sum() / 1e6:,.0f} km2; overlap between units {over:.2f} km2")
    if over > 1.0:
        raise SystemExit("!! units overlap")
    os.makedirs(GEO, exist_ok=True)
    units[["unit", "name", "ilce", "pop2011", "geometry"]].to_file(
        os.path.join(GEO, "cy_north_units.gpkg"), driver="GPKG", layer="units")

    # ---- placement layer, whole island ----
    rd = gpd.read_file(RD_HEXES)
    rdc = gpd.GeoSeries(rd.to_crs(3857).geometry.centroid, crs=3857).to_crs(4326)
    inn = rdc.within(mask).to_numpy()
    lost = rd[inn].groupby("unit")["pop"].agg(["size", "sum"]).sort_values("sum", ascending=False)
    print(f"religiondots south hexes with centroid in the north mask: {int(inn.sum())} "
          f"({rd.loc[inn, 'pop'].sum():,.0f} Kontur people), dropped from: "
          + ", ".join(f"{u} {int(r['size'])} hexes/{r['sum']:,.0f}" for u, r in lost.iterrows()))
    south = rd[~inn].copy()
    gone = sorted(set(rd["unit"]) - set(south["unit"]))
    if gone:
        raise SystemExit(f"!! south communities left with no hex: {gone}")

    k = gpd.read_file(KONTUR)
    popcol = next(c for c in k.columns if c.lower() == "population")
    pts = gpd.GeoDataFrame({"pop": k[popcol].to_numpy(dtype=float)},
                           geometry=k.geometry.centroid, crs=k.crs).to_crs(4326)
    j = gpd.sjoin(pts, units[["unit", "geometry"]], how="left", predicate="within")
    j = j[~j.index.duplicated()].reindex(pts.index)
    in_mask = pts.within(mask).to_numpy()
    nounit = in_mask & j["unit"].isna().to_numpy()
    print(f"Kontur: {pts['pop'].sum():,.0f} on the island; {pts.loc[in_mask, 'pop'].sum():,.0f} "
          f"in the north mask; {pts.loc[nounit, 'pop'].sum():,.0f} of those in no unit "
          f"({int(nounit.sum())} hexes)")
    keep = j["unit"].notna().to_numpy()
    nh = gpd.GeoDataFrame({"unit": j.loc[keep, "unit"].to_numpy(), "pop": pts.loc[keep, "pop"].to_numpy()},
                          geometry=k.geometry[keep].to_crs(4326).to_numpy(), crs=4326)
    nh = nh[nh["pop"] > 0]
    per = nh.groupby("unit")["pop"].sum()
    empty = sorted(set(units["unit"]) - set(per.index))
    if empty:
        fill = units[units["unit"].isin(empty)][["unit", "pop2011", "geometry"]].rename(
            columns={"pop2011": "pop"})
        nh = gpd.GeoDataFrame(pd.concat([nh, fill], ignore_index=True), geometry="geometry", crs=4326)
        print(f"  {len(empty)} units with no populated hex get their own polygon: "
              + ", ".join(f"{u} {units.set_index('unit').loc[u, 'name']}" for u in empty))
    ratio = per.sum() / units["pop2011"].sum()
    norm = (per.reindex(units["unit"]).fillna(0).to_numpy() / units["pop2011"].to_numpy() / ratio)
    print(f"Kontur / 2011 census in the north {ratio:.3f}; per unit p10 {np.percentile(norm, 10):.2f} "
          f"median {np.median(norm):.2f} p90 {np.percentile(norm, 90):.2f}")
    lc = np.log(units["pop2011"].to_numpy())
    lk = np.log(np.maximum(per.reindex(units["unit"]).fillna(1).to_numpy(), 1))
    r = np.corrcoef(lc, lk)[0, 1]
    rng = np.random.default_rng(0)
    best = max(abs(np.corrcoef(lc, rng.permutation(lk))[0, 1]) for _ in range(500))
    o = np.argsort(norm)
    nm = units["name"].to_numpy()
    pp = units["pop2011"].to_numpy()
    print("  lowest: " + ", ".join(f"{nm[i]} {pp[i]:,} {norm[i]:.2f}" for i in o[:6]))
    print("  highest: " + ", ".join(f"{nm[i]} {pp[i]:,} {norm[i]:.2f}" for i in o[-6:]))
    big = pp >= 3000
    print(f"  units of 3,000+: {int(big.sum())}, normalised ratio "
          + ", ".join(f"{nm[i]} {norm[i]:.2f}" for i in np.where(big)[0]))
    print(f"log correlation r = {r:.3f}, best of 500 shuffles {best:.3f}")
    if r <= best:
        raise SystemExit("!! the join carries no information")

    # Hexes are 400 m across and a hex on the line straddles it: cut each side's hexes at the
    # line so a dot can never land across it (north hexes also lose any buffer-zone part).
    south = south.to_crs(4326)[["unit", "pop", "geometry"]]
    cross = south.intersects(mask).to_numpy()
    south.loc[cross, "geometry"] = south.loc[cross].geometry.difference(mask)
    nh["geometry"] = nh.geometry.intersection(mask)
    south = south[~south.geometry.is_empty]
    nh = nh[~nh.geometry.is_empty]
    print(f"  cut at the line: {int(cross.sum())} south hexes, north hexes clipped to the mask")
    allh = gpd.GeoDataFrame(pd.concat([south, nh[["unit", "pop", "geometry"]]], ignore_index=True),
                            geometry="geometry", crs=4326)
    allh.to_file(os.path.join(GEO, "cy_hexes.gpkg"), driver="GPKG", layer="hexes")
    print(f"wrote {os.path.join(GEO, 'cy_hexes.gpkg')}: {len(south):,} south + {len(nh):,} north hexes")

    out = pd.DataFrame({"geo_id": units["unit"], "geo_level": "north_village",
                        "geo_name": units["name"], "source_category": "Population (no language question)",
                        "count": units["pop2011"], "tier": "derived", "year": 2011,
                        "source_id": "trnc_census2011_tablo3"})
    out.to_csv(NORM, index=False, encoding="utf-8")
    print(f"wrote {NORM}: {len(out)} units, {out['count'].sum():,} people "
          f"(Pile {int(df[df['unit'].isna()]['pop'].sum()):,} not drawn)")


if __name__ == "__main__":
    main()
