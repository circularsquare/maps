"""North Macedonia: the placement layer, Kontur 400 m hexes keyed to the 80 municipalities and,
inside them, to the census's settlements.

    python sources/mk_geo.py --fetch   SSO's ethnicity-by-settlement table (two PxWeb POSTs) and
                                       the OSM settlement points (from a Geofabrik extract, which
                                       is deleted once the points are read)
    python sources/mk_geo.py           -> data/geo/mk/mk_hexes.gpkg, data/geo/mk/mk_settlements.csv

UNITS. religiondots' data/geo/mk/mk_opstini.gpkg (GISCO LAU 2021, 80 polygons keyed by `kod`) and
mk_lookup.csv (SSO's four-digit PxWeb code -> `kod`), read only. religiondots places North
Macedonia on the bare polygons; this map weights inside them, so the hexes are new here. Kontur MK
is not in religiondots' kontur folder, so _grid.py downloads it into languagedots'.

HEXES OUTSIDE EVERY UNIT. North Macedonia is landlocked; a hex whose centroid falls outside the
GISCO polygons is either across a land border (Tetovo's and Gostivar's neighbours in Kosovo,
Gevgelija's Greek twin Evzonoi, Kosovo's Hani i Elezit beside Djeneral Jankovikj) or on the
shores of Ohrid, Prespa and Dojran, where GISCO's generalised lake shore cuts off lakeside hexes.
Each is classed by its centroid against Natural Earth 10m (religiondots' copy, read only): inside
Natural Earth's North Macedonia and more than NB_M from a neighbour -> snapped to the nearest unit
within SNAP_M; otherwise dropped.

SETTLEMENTS (the within-unit placement; AGENT_BRIEF section 4.4). The mother-tongue table is by
municipality only, but SSO publishes ethnicity by settlement (T1503P21: 1,780-odd settlements,
the same census). Inside a municipality each language's dots follow the settlements of the
matching ethnicity (countries/mk.py says which); the counts drawn stay the municipal table's. To
use that, each hex needs a settlement:
  1. Settlement points are OSM place nodes (city, town, village, hamlet, suburb...) from
     Geofabrik's macedonia-latest.osm.pbf, by their Macedonian name (`name`, else `name:mk`).
  2. A census settlement `<name> (<municipality>)` is matched to the node of the same folded
     Cyrillic name inside its municipality's polygon buffered by BUF_M (GISCO's border is
     generalised and a village node can sit just across it). Exactly one candidate is required:
     of two, the one inside the polygon itself wins, then the higher place rank (a village
     node over a hamlet or locality of the same name); a tie is left unmatched.
  3. The urban settlements of the City of Skopje (`Скопје - Аеродром` and nine more) have no node
     of their own. Each is given the Kontur-population-weighted centroid of its municipality's
     hexes, which lands in the built-up part; so is any other settlement left unmatched that
     holds more than half its municipality.
  4. Each hex goes to the nearest placed settlement OF ITS OWN MUNICIPALITY.
Unmatched settlements are listed with their population; their people still draw (the hexes near
them weigh by their nearest matched neighbour's mix), so a miss blurs the mix, it loses no one.

CHECKS: the settlements sum to every municipality's total in T1503P21 exactly, and those totals
equal the mother-tongue table's (the same census); the share of each municipality's population
whose settlement was placed; Kontur against the census per unit with a shuffled-join control.
"""
import json
import math
import os
import random
import sys
import unicodedata
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
from rdlink import RD_GEO  # noqa: E402
from _grid import kontur_path  # noqa: E402

UNITS = RD_GEO / "mk" / "mk_opstini.gpkg"
LOOKUP = RD_GEO / "mk" / "mk_lookup.csv"
NE = RD_GEO / "ne_10m_admin_0_countries.geojson"
RAW = ROOT / "data" / "raw" / "mk"
NORM = ROOT / "data" / "normalized" / "mk.csv"
OUT = ROOT / "data" / "geo" / "mk" / "mk_hexes.gpkg"
SETTLEMENTS = ROOT / "data" / "geo" / "mk" / "mk_settlements.csv"
PBF = RAW / "macedonia-latest.osm.pbf"
PLACES = RAW / "osm_places.csv"
PBF_URL = "https://download.geofabrik.de/europe/macedonia-latest.osm.pbf"
T1503 = ("https://makstat.stat.gov.mk/pxweb/api/v1/{lang}/MakStat/Popisi/Popis2021/"
         "NaselenieVkupno/PodatociNaselenie/T1503P21.px")

N_UNITS = 80
NEIGHBOURS = ["SRB", "KOS", "BGR", "GRC", "ALB"]
SNAP_M = 1000
NB_M = 2000
BUF_M = 1500
CRS_M = 6316                     # MGI 1901 / Balkans zone 7
PLACE_TAGS = {"city", "town", "village", "hamlet", "suburb", "neighbourhood", "quarter",
              "isolated_dwelling", "locality"}
ETHN = {"0": "total", "1": "macedonians", "3": "albanians", "6": "turks", "5": "roma",
        "4": "vlachs", "27": "serbs", "10": "bosniaks", "37": "other", "38": "undeclared",
        "99": "unknown", "88": "admin"}


def fetch():
    import requests
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    h = {"User-Agent": "Mozilla/5.0"}
    RAW.mkdir(parents=True, exist_ok=True)
    for lang in ("en", "mk"):
        dest = RAW / f"T1503P21_{lang}.json"
        if dest.exists() and dest.stat().st_size > 100_000:
            print("already have", dest)
            continue
        url = T1503.format(lang=lang)
        meta = requests.get(url, timeout=120, verify=False, headers=h).json()
        q = {"query": [{"code": v["code"], "selection": {"filter": "all", "values": ["*"]}}
                       for v in meta["variables"]], "response": {"format": "json-stat2"}}
        r = requests.post(url, json=q, timeout=300, verify=False, headers=h)
        r.raise_for_status()
        doc = r.json()
        if "value" not in doc:
            raise SystemExit(f"{url}: not a json-stat2 cube")
        dest.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
        print(f"  {dest} {dest.stat().st_size:,} bytes")
    if PLACES.exists():
        print("already have", PLACES)
        return
    if not PBF.exists():
        r = requests.get(PBF_URL, timeout=1200, stream=True, headers=h)
        r.raise_for_status()
        with open(PBF, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
    import csv
    import osmium

    class H(osmium.SimpleHandler):
        def __init__(self):
            super().__init__()
            self.rows = []

        def node(self, n):
            p = n.tags.get("place")
            if p in PLACE_TAGS:
                self.rows.append({"osm_id": n.id, "place": p, "name": n.tags.get("name", ""),
                                  "name_mk": n.tags.get("name:mk", ""),
                                  "lat": n.location.lat, "lon": n.location.lon})
    hd = H()
    hd.apply_file(str(PBF))
    with open(PLACES, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(hd.rows[0]))
        w.writeheader()
        w.writerows(hd.rows)
    print(f"  {PLACES}: {len(hd.rows):,} place nodes")
    PBF.unlink()                 # 30 MB, read once; the points are all that is kept
    print(f"  deleted {PBF.name}")


def fold(s):
    s = unicodedata.normalize("NFKD", str(s).lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return "".join(c for c in s if c.isalnum())


def settlements():
    """T1503P21 -> one row per settlement: code, name, municipality name, ethnic counts."""
    import pandas as pd
    out = {}
    for lang in ("en", "mk"):
        doc = json.loads((RAW / f"T1503P21_{lang}.json").read_text(encoding="utf-8"))
        ids, sizes = doc["id"], doc["size"]
        cats = {}
        for d in ids:
            idx = doc["dimension"][d]["category"]["index"]
            order = sorted(idx, key=lambda k: idx[k]) if isinstance(idx, dict) else list(idx)
            cats[d] = (order, doc["dimension"][d]["category"]["label"])
        geo = next(d for d, n in zip(ids, sizes) if n > 1000)
        eth = next(d for d, n in zip(ids, sizes) if n == 12)
        if ids != [geo, eth]:
            raise SystemExit(f"T1503P21: unexpected dimension order {ids}")
        if set(cats[eth][0]) != set(ETHN):
            raise SystemExit(f"T1503P21: ethnicity codes changed: {cats[eth][0]}")
        ne = len(cats[eth][0])
        rows = []
        for i, g in enumerate(cats[geo][0]):
            r = {"code": g, "label": cats[geo][1][g]}
            for j, e in enumerate(cats[eth][0]):
                r[ETHN[e]] = int(doc["value"][i * ne + j] or 0)
            rows.append(r)
        out[lang] = pd.DataFrame(rows)
    mk, en = out["mk"], out["en"]
    if list(mk["code"]) != list(en["code"]):
        raise SystemExit("T1503P21: the two language editions differ in settlements")
    mk["label_en"] = en["label"].to_numpy()
    mk = mk[~mk["code"].isin(["000000", "499994"])].copy()     # the country; City of Skopje
    parts = mk["label"].str.extract(r"^(.*)\s+\((.*)\)\s*$")
    if parts.isna().any().any():
        raise SystemExit(f"T1503P21: labels without '(municipality)': "
                         f"{mk.loc[parts[0].isna(), 'label'].head().tolist()}")
    mk["name"], mk["opstina"] = parts[0].str.strip(), parts[1].str.strip()
    return mk


def main():
    if "--fetch" in sys.argv:
        fetch()

    import geopandas as gpd
    import numpy as np
    import pandas as pd

    units = gpd.read_file(UNITS)[["kod", "name", "geometry"]].rename(columns={"kod": "unit"})
    if len(units) != N_UNITS or units["unit"].nunique() != N_UNITS:
        raise SystemExit(f"{UNITS}: {len(units)} polygons")
    lut = pd.read_csv(LOOKUP, dtype=str)
    norm = pd.read_csv(NORM, dtype={"geo_id": str})
    tot = norm[(norm["geo_level"] == "municipality") & (norm["source_category"] == "Total")]
    tot = tot.assign(unit=tot["geo_id"].map(dict(zip(lut["geo_id"], lut["kod"]))))
    if tot["unit"].isna().any() or set(tot["unit"]) != set(units["unit"]):
        raise SystemExit("mk.csv's municipalities and the polygons disagree")
    census = tot.set_index("unit")["count"].astype(float)
    name_to_unit = dict(zip(tot["geo_name"].map(fold), tot["unit"]))
    print(f"  {N_UNITS} polygons and {N_UNITS} census municipalities, matched both ways by "
          "religiondots' lookup")

    # --- settlements: the census table, checked against the municipal table
    st = settlements()
    st["unit"] = st["opstina"].map(fold).map(name_to_unit)
    if st["unit"].isna().any():
        raise SystemExit(f"settlements naming no known municipality: "
                         f"{st.loc[st['unit'].isna(), 'opstina'].unique()[:8]}")
    by_unit = st.groupby("unit")["total"].sum()
    off = [u for u in census.index if by_unit.get(u) != census[u]]
    print(f"  {'OK ' if not off else 'BAD'} {len(st):,} settlements sum to every municipality's "
          f"total in the mother-tongue table ({len(off)} off)")
    if off:
        raise SystemExit(f"settlement sums disagree: {off[:5]}")
    eth_cols = [c for c in ETHN.values() if c != "total"]
    bad = (st[eth_cols].sum(axis=1) != st["total"]).sum()
    print(f"  {'OK ' if not bad else 'BAD'} the 11 ethnic categories partition every settlement")
    if bad:
        raise SystemExit("ethnic categories do not partition the settlements")

    # --- settlement points
    pl = pd.read_csv(PLACES, dtype=str, keep_default_na=False)
    pl["nm"] = np.where(pl["name"].str.contains(r"[Ѐ-ӿ]"), pl["name"], pl["name_mk"])
    pl = pl[pl["nm"] != ""]
    pts = gpd.GeoDataFrame(pl, geometry=gpd.points_from_xy(pl["lon"].astype(float),
                                                            pl["lat"].astype(float)),
                           crs=4326).to_crs(CRS_M)
    pts["key"] = pts["nm"].map(fold)
    um = units.to_crs(CRS_M).set_index("unit")
    rank = {"city": 0, "town": 1, "village": 2, "suburb": 3, "hamlet": 4, "quarter": 5,
            "neighbourhood": 6, "isolated_dwelling": 7, "locality": 8}
    pts["rank"] = pts["place"].map(rank)

    xs, ys, how = [], [], []
    for _, s in st.iterrows():
        poly = um.geometry[s["unit"]]
        cand = pts[(pts["key"] == fold(s["name"]))]
        cand = cand[cand.geometry.within(poly.buffer(BUF_M))]
        if len(cand) > 1:
            inside = cand[cand.geometry.within(poly)]
            cand = inside if len(inside) else cand
        if len(cand) > 1:
            cand = cand.sort_values("rank").head(1) if cand["rank"].nunique() > 1 else cand
        if len(cand) == 1:
            g = cand.geometry.iloc[0]
            xs.append(g.x); ys.append(g.y); how.append("osm")
        else:
            xs.append(np.nan); ys.append(np.nan)
            how.append("ambiguous" if len(cand) > 1 else "none")
    st["x"], st["y"], st["how"] = xs, ys, how

    # --- hexes
    hexes = gpd.read_file(kontur_path("mk"))
    popcol = next(c for c in hexes.columns if c.lower() == "population")
    hp = gpd.GeoDataFrame({"pop": hexes[popcol].to_numpy(dtype=float)},
                          geometry=hexes.geometry.centroid, crs=hexes.crs).to_crs(CRS_M)
    um_r = um.reset_index()
    j = gpd.sjoin(hp, um_r[["unit", "geometry"]], how="left", predicate="within")
    j = j[~j.index.duplicated(keep="first")].reindex(hp.index)
    outside = j["unit"].isna()
    print(f"  Kontur MK: {len(hexes):,} hexes, {hp['pop'].sum():,.0f} people; "
          f"{int(outside.sum()):,} hexes ({hp.loc[outside, 'pop'].sum():,.0f} people) outside every unit")
    o = hp[outside]
    ne = gpd.read_file(NE)
    a3 = "ADM0_A3"
    ne = ne[ne[a3].isin(["MKD"] + NEIGHBOURS)][[a3, "geometry"]].to_crs(CRS_M)
    oc = gpd.sjoin(o, ne, how="left", predicate="within")
    oc = oc[~oc.index.duplicated(keep="first")].reindex(o.index)
    where = oc[a3].fillna("none")
    d_nb = o.geometry.distance(ne[ne[a3] != "MKD"].geometry.union_all())
    snap_ok = (where == "MKD") & (d_nb > NB_M)
    for k, s in (("in NE's MKD, clear of a border", snap_ok),
                 ("in NE's MKD, near a border", (where == "MKD") & ~snap_ok),
                 ("in a neighbour", where.isin(NEIGHBOURS)), ("in no country", where == "none")):
        print(f"    outside, {k:<32} {int(s.sum()):>5,} hexes {o.loc[s, 'pop'].sum():>9,.0f} people")
    cand = o[snap_ok]
    near = gpd.sjoin_nearest(cand, um_r[["unit", "geometry"]], how="left", max_distance=SNAP_M,
                             distance_col="dist")
    near = near[~near.index.duplicated(keep="first")]
    sn = near["unit"].notna()
    j.loc[near.index[sn], "unit"] = near.loc[sn, "unit"]
    print(f"  snapped {int(sn.sum()):,} hexes ({near.loc[sn, 'pop'].sum():,.0f} people) within "
          f"{SNAP_M} m; dropped the rest ({o['pop'].sum() - near.loc[sn, 'pop'].sum():,.0f} people)")

    keep = j["unit"].notna().to_numpy()
    lay = gpd.GeoDataFrame({"unit": j.loc[keep, "unit"].astype(str).to_numpy(),
                            "pop": hp.loc[keep, "pop"].to_numpy()},
                           geometry=hp.geometry[keep].to_numpy(), crs=CRS_M)

    # unmatched settlements holding more than half their municipality (Skopje's urban
    # settlements, and any town OSM spells differently) sit at the unit's population centroid
    cx = lay.assign(wx=lay.geometry.x * lay["pop"], wy=lay.geometry.y * lay["pop"]) \
        .groupby("unit")[["wx", "wy", "pop"]].sum()
    big = (st["how"] != "osm") & (st["total"] > 0.5 * st["unit"].map(census))
    for i in st.index[big]:
        u = st.at[i, "unit"]
        st.at[i, "x"] = cx.at[u, "wx"] / cx.at[u, "pop"]
        st.at[i, "y"] = cx.at[u, "wy"] / cx.at[u, "pop"]
        st.at[i, "how"] = "centroid"
    placed = st["how"].isin(["osm", "centroid"])
    print(f"  settlements placed: {int((st['how'] == 'osm').sum()):,} on an OSM node, "
          f"{int((st['how'] == 'centroid').sum())} at their municipality's population centroid; "
          f"{int((~placed).sum())} not placed ({int((st['how'] == 'ambiguous').sum())} ambiguous), "
          f"holding {st.loc[~placed, 'total'].sum():,} people "
          f"({100 * st.loc[~placed, 'total'].sum() / census.sum():.2f}%)")
    miss = st[~placed].sort_values("total", ascending=False).head(12)
    print("    largest not placed: " + ", ".join(f"{a} ({b}) {c:,}" for a, b, c in
                                                zip(miss["label_en"], miss["how"], miss["total"])))
    share = st[placed].groupby("unit")["total"].sum() / census
    low = share.sort_values().head(6)
    print("    lowest share of a municipality placed: " + ", ".join(
        f"{um.loc[u, 'name']} {v:.0%}" for u, v in low.items()))

    # each hex to the nearest placed settlement of its own municipality
    lay["settlement"] = ""
    sp = st[placed]
    for u, g in lay.groupby("unit"):
        s = sp[sp["unit"] == u]
        if s.empty:
            raise SystemExit(f"no placed settlement in {u}")
        sx, sy = s["x"].to_numpy(), s["y"].to_numpy()
        hx, hy = g.geometry.x.to_numpy()[:, None], g.geometry.y.to_numpy()[:, None]
        k = ((hx - sx) ** 2 + (hy - sy) ** 2).argmin(axis=1)
        lay.loc[g.index, "settlement"] = s["code"].to_numpy()[k]
    # a settlement with people and no hex weighs nothing; say how many people that is
    got = set(lay.loc[lay["pop"] > 0, "settlement"])
    nohex = sp[~sp["code"].isin(got) & (sp["total"] > 0)]
    print(f"  {len(nohex)} placed settlements own no populated hex ({nohex['total'].sum():,} people;"
          " their neighbours' hexes carry them)")

    # Kontur against the census, per unit
    per = lay.groupby("unit")["pop"].sum()
    empty = sorted(set(units["unit"]) - set(per.index[per > 0]))
    if empty:
        raise SystemExit(f"units with no populated hex: {empty}")
    rows = [(u, census[u], float(per.get(u, 0.0))) for u in census.index]
    ratio = sum(k for _, _, k in rows) / sum(c for _, c, _ in rows)
    nrm = sorted(((k / c / ratio), u) for u, c, k in rows)
    nm = dict(zip(units["unit"], units["name"]))
    print(f"  Kontur / census nationally {ratio:.3f}; per unit, normalised: p10 "
          f"{nrm[len(nrm) // 10][0]:.2f}  median {nrm[len(nrm) // 2][0]:.2f}  p90 "
          f"{nrm[9 * len(nrm) // 10][0]:.2f}")
    print("  lowest: " + ", ".join(f"{nm[u]} {r:.2f}" for r, u in nrm[:6]))
    print("  highest: " + ", ".join(f"{nm[u]} {r:.2f}" for r, u in nrm[-6:]))
    lc = [math.log(c) for _, c, _ in rows]
    lk = [math.log(max(k, 1.0)) for _, _, k in rows]

    def pear(a, b):
        ma, mb = sum(a) / len(a), sum(b) / len(b)
        num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
        return num / math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    r = pear(lc, lk)
    rng = random.Random(0)
    best = max(abs(pear(lc, rng.sample(lk, len(lk)))) for _ in range(500))
    print(f"  log correlation r = {r:.3f} against a best of {best:.3f} over 500 shuffles")
    if r <= best:
        raise SystemExit("the join is not carrying information")

    out = gpd.GeoDataFrame(lay[["unit", "settlement", "pop"]],
                           geometry=hexes.geometry[keep].to_numpy(), crs=hexes.crs).to_crs(4326)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_file(OUT, layer="hexes", driver="GPKG")
    print(f"  wrote {OUT} ({len(out):,} hexes, {out['pop'].sum():,.0f} people)")
    cols = ["code", "name", "label_en", "opstina", "unit", "how", "x", "y", "total"] + eth_cols
    st[cols].to_csv(SETTLEMENTS, index=False, encoding="utf-8")
    print(f"  wrote {SETTLEMENTS} ({len(st):,} settlements; x, y in EPSG:{CRS_M})")


if __name__ == "__main__":
    main()
