"""Montenegro: the placement layer, Kontur 400 m hexes keyed to the 25 municipalities of the 2023
census and, inside them, to the census's settlements.

    python sources/me_geo.py --fetch   Geofabrik's montenegro-latest.osm.pbf -> the 25 municipality
                                       polygons and the OSM place nodes (the .pbf is deleted after)
    python sources/me_geo.py           -> data/geo/me/me_hexes.gpkg, data/geo/me/me_settlements.csv

UNITS. Not religiondots' layer: that is geoBoundaries ADM1, 23 polygons, with Tuzi (2018) and Zeta
(2022) folded back into Podgorica, and Tuzi is Montenegro's second Albanian municipality, which a
language map cannot average into the capital. OSM has all 25 as admin_level=6 relations
("Opstina Tuzi", "Opstina Zeta", "Glavni grad Podgorica", "Prijestolnica Cetinje"...), assembled
here with pyosmium's area builder and joined to the census by name after stripping the title
("Opstina", "Glavni grad", "Prijestolnica", Ulcinj's Albanian half). The join must be 25 of 25
both ways.

HEXES OUTSIDE EVERY UNIT. OSM's municipalities stop at the shore; Montenegro's border with the
sea and the lakes (Skadar, Plav) is where most outside hexes are. As in mk_geo.py, each is classed
against Natural Earth 10m (religiondots' copy, read only): inside Natural Earth's Montenegro and
more than NB_M from a neighbour -> snapped to the nearest unit within SNAP_M; otherwise dropped.

SETTLEMENTS (the within-unit placement; AGENT_BRIEF section 4.4). MONSTAT publishes mother tongue
itself by settlement (1,462), with small cells suppressed (`z`). Inside a municipality each
language's dots follow the settlements where the census counted it (countries/me.py); the counts
drawn stay the municipal table's. Each hex needs a settlement:
  1. Settlement points are OSM place nodes (city, town, village, hamlet, suburb...). A node's
     names are `name`, `name:sr-Latn`, `name:cnr`, `name:sr` (Cyrillic, transliterated) and
     `name:sq`, each folded (diacritics off, letters only).
  2. A census settlement is matched to a node with the same folded name inside its municipality's
     polygon buffered by BUF_M. Of several, the one inside the polygon itself wins, then the
     higher place rank; a tie is left unmatched.
  3. An unmatched settlement holding more than half its municipality sits at the Kontur-weighted
     centroid of the municipality's hexes.
  4. Each hex goes to the nearest placed settlement OF ITS OWN MUNICIPALITY.
Unmatched settlements are listed with their population; their people still draw (the hexes near
them weigh by their nearest matched neighbour's mix), so a miss blurs the mix, it loses no one.

CHECKS: 25 polygons joined 25/25 both ways; the share of each municipality's population whose
settlement was placed; Kontur against the census per unit with a shuffled-join control.
"""
import math
import os
import random
import re
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

NE = RD_GEO / "ne_10m_admin_0_countries.geojson"
RAW = ROOT / "data" / "raw" / "me"
NORM = ROOT / "data" / "normalized" / "me.csv"
GEO = ROOT / "data" / "geo" / "me"
UNITS = GEO / "me_opstine.gpkg"
OUT = GEO / "me_hexes.gpkg"
SETTLEMENTS = GEO / "me_settlements.csv"
PBF = RAW / "montenegro-latest.osm.pbf"
PLACES = RAW / "osm_places.csv"
PBF_URL = "https://download.geofabrik.de/europe/montenegro-latest.osm.pbf"
WD = RAW / "wikidata_places.csv"
WD_QUERY = """
SELECT ?item ?coord ?lab_sh ?lab_srel ?lab_en ?lab_sr ?lab_sq ?admin WHERE {
  VALUES ?t { wd:Q486972 wd:Q532 wd:Q3957 wd:Q515 wd:Q188509 wd:Q123705 wd:Q5084 }
  ?item wdt:P17 wd:Q236 ; wdt:P625 ?coord ; wdt:P31 ?t .
  OPTIONAL { ?item rdfs:label ?lab_sh FILTER(lang(?lab_sh)="sh") }
  OPTIONAL { ?item rdfs:label ?lab_srel FILTER(lang(?lab_srel)="sr-el") }
  OPTIONAL { ?item rdfs:label ?lab_en FILTER(lang(?lab_en)="en") }
  OPTIONAL { ?item rdfs:label ?lab_sr FILTER(lang(?lab_sr)="sr") }
  OPTIONAL { ?item rdfs:label ?lab_sq FILTER(lang(?lab_sq)="sq") }
  OPTIONAL { ?item wdt:P131 ?admin }
}"""
WD_NAMES = ["sh", "sr_el", "en", "sr", "sq"]
ST_XLSX = RAW / "naselja_maternji_jezik_2023.xlsx"

N_UNITS = 25
NEIGHBOURS = ["SRB", "KOS", "BIH", "HRV", "ALB"]
SNAP_M = 1500
NB_M = 2000
BUF_M = 1500
CRS_M = 32634                    # UTM 34N
PLACE_TAGS = {"city", "town", "village", "hamlet", "suburb", "neighbourhood", "quarter",
              "isolated_dwelling", "locality"}
NAME_TAGS = ["name", "name:sr-Latn", "name:cnr", "name:sr", "name:sq"]

CYR = dict(zip("абвгдђежзијклљмнњопрстћуфхцчџш",
               ["a", "b", "v", "g", "d", "dj", "e", "z", "z", "i", "j", "k", "l", "lj", "m", "n",
                "nj", "o", "p", "r", "s", "t", "c", "u", "f", "h", "c", "c", "dz", "s"]))


def fetch():
    import csv

    import osmium
    import requests
    import shapely.wkb
    RAW.mkdir(parents=True, exist_ok=True)
    GEO.mkdir(parents=True, exist_ok=True)
    if not WD.exists():
        r = requests.get("https://query.wikidata.org/sparql",
                         params={"query": WD_QUERY, "format": "json"}, timeout=300,
                         headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        r.raise_for_status()
        with open(WD, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["qid", "lon", "lat"] + WD_NAMES + ["admin"])
            for b in r.json()["results"]["bindings"]:
                c = b["coord"]["value"]
                if not c.startswith("Point("):
                    continue
                lon, lat = c[6:-1].split()
                g = lambda k: b.get(k, {}).get("value", "")  # noqa: E731
                w.writerow([g("item").rsplit("/", 1)[-1], lon, lat, g("lab_sh"), g("lab_srel"),
                            g("lab_en"), g("lab_sr"), g("lab_sq"), g("admin").rsplit("/", 1)[-1]])
        print(f"  {WD}: Wikidata settlement items")
    if PLACES.exists() and UNITS.exists():
        print("already have", PLACES, "and", UNITS)
        return
    if not PBF.exists():
        r = requests.get(PBF_URL, timeout=1200, stream=True, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        with open(PBF, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
    wkb = osmium.geom.WKBFactory()

    class H(osmium.SimpleHandler):
        def __init__(self):
            super().__init__()
            self.rows, self.areas = [], []

        def node(self, n):
            p = n.tags.get("place")
            if p in PLACE_TAGS:
                r = {"osm_id": n.id, "place": p, "lat": n.location.lat, "lon": n.location.lon}
                for t in NAME_TAGS:
                    r[t] = n.tags.get(t, "")
                self.rows.append(r)

        def area(self, a):
            if (a.tags.get("boundary") == "administrative" and a.tags.get("admin_level") == "6"
                    and a.from_way() is False):
                try:
                    g = shapely.wkb.loads(wkb.create_multipolygon(a), hex=True)
                except Exception as e:           # an incomplete relation of a neighbour
                    print("  skip", a.tags.get("name"), e)
                    return
                self.areas.append({"osm_rel": a.orig_id(), "osm_name": a.tags.get("name", ""),
                                   "name_en": a.tags.get("name:en", ""), "geometry": g})
    h = H()
    h.apply_file(str(PBF), locations=True)
    with open(PLACES, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(h.rows[0]))
        w.writeheader()
        w.writerows(h.rows)
    print(f"  {PLACES}: {len(h.rows):,} place nodes")
    import geopandas as gpd
    g = gpd.GeoDataFrame(h.areas, crs=4326)
    g.to_file(UNITS, layer="admin6", driver="GPKG")
    print(f"  {UNITS}: {len(g)} admin_level=6 areas (all of them; main() picks Montenegro's)")
    PBF.unlink()                 # 34 MB, read once; the points and polygons are all that is kept
    print(f"  deleted {PBF.name}")


def fold(s):
    s = str(s).lower()
    s = "".join(CYR.get(c, c) for c in s)
    s = s.replace("đ", "dj")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return "".join(c for c in s if c.isalnum())


def unit_name(osm_name):
    """'Opština Ulcinj - Komuna e Ulqinit' -> 'Ulcinj'; 'Glavni grad Podgorica' -> 'Podgorica'."""
    s = re.sub(r"^(Opština|Glavni grad|Prijestolnica)\s+", "", osm_name.strip())
    return s.split(" - ")[0].strip()


def settlements():
    """The census settlements with their mother-tongue cells (None where suppressed)."""
    from me_census import read_settlements
    st, cats = read_settlements()
    return st, cats


def main():
    if "--fetch" in sys.argv:
        fetch()

    import geopandas as gpd
    import numpy as np
    import pandas as pd

    norm = pd.read_csv(NORM)
    tot = norm[(norm["geo_level"] == "municipality") & (norm["source_category"] == "Ukupno")]
    census = tot.set_index("geo_id")["count"].astype(float)

    a = gpd.read_file(UNITS)
    a["unit"] = a["osm_name"].map(unit_name)
    a = a[a["unit"].map(fold).isin(set(census.index.map(fold)))].copy()
    fk = dict(zip(census.index.map(fold), census.index))
    a["unit"] = a["unit"].map(fold).map(fk)
    if len(a) != N_UNITS or a["unit"].nunique() != N_UNITS or set(a["unit"]) != set(census.index):
        raise SystemExit(f"the OSM municipalities do not join 25/25: {sorted(a['unit'])}, "
                         f"missing {sorted(set(census.index) - set(a['unit']))}")
    print(f"  {N_UNITS} OSM admin_level=6 polygons and {N_UNITS} census municipalities, matched "
          "both ways by name")
    units = a[["unit", "osm_rel", "osm_name", "geometry"]]
    um = units.to_crs(CRS_M).set_index("unit")
    print(f"    area {um.area.sum() / 1e6:,.0f} km² (Montenegro's land area is 13,812 km²)")

    # --- settlements
    st, cats = settlements()
    st["unit"] = st["opstina"]
    st["code"] = [f"{i:04d}" for i in range(len(st))]

    pl = pd.read_csv(PLACES, dtype=str, keep_default_na=False)
    pts = gpd.GeoDataFrame(pl, geometry=gpd.points_from_xy(pl["lon"].astype(float),
                                                            pl["lat"].astype(float)),
                           crs=4326).to_crs(CRS_M)
    rank = {"city": 0, "town": 1, "village": 2, "suburb": 3, "hamlet": 4, "quarter": 5,
            "neighbourhood": 6, "isolated_dwelling": 7, "locality": 8}
    pts["rank"] = pts["place"].map(rank)

    wd = pd.read_csv(WD, dtype=str, keep_default_na=False).drop_duplicates("qid")
    wdp = gpd.GeoDataFrame(wd, geometry=gpd.points_from_xy(wd["lon"].astype(float),
                                                           wd["lat"].astype(float)),
                           crs=4326).to_crs(CRS_M)
    wdp["rank"] = 0

    def index(df, tags):
        keys = {}
        for i, r in df.iterrows():
            for t in tags:
                if r[t]:
                    keys.setdefault(fold(r[t]), set()).add(i)
        return keys

    sources = [("osm", pts, index(pts, NAME_TAGS)), ("wikidata", wdp, index(wdp, WD_NAMES))]

    def match(s):
        """(x, y, how) for one census settlement: OSM first, Wikidata where OSM has no answer."""
        poly = um.geometry[s["unit"]]
        res = "none"
        for tag, df, keys in sources:
            cand = df.loc[sorted(keys.get(fold(s["naselje"]), ()))]
            cand = cand[cand.geometry.within(poly.buffer(BUF_M))]
            if len(cand) > 1:
                inside = cand[cand.geometry.within(poly)]
                cand = inside if len(inside) else cand
            if len(cand) > 1 and cand["rank"].nunique() > 1:
                cand = cand[cand["rank"] == cand["rank"].min()]
            if len(cand) == 1:
                g = cand.geometry.iloc[0]
                return g.x, g.y, tag
            if len(cand) > 1:
                res = "ambiguous"
        return np.nan, np.nan, res

    m = [match(s) for _, s in st.iterrows()]
    st["x"], st["y"], st["how"] = [a for a, _, _ in m], [b for _, b, _ in m], [c for _, _, c in m]
    print(f"  settlement points: {int((st['how'] == 'osm').sum()):,} from OSM place nodes, "
          f"{int((st['how'] == 'wikidata').sum()):,} more from Wikidata ({len(wdp):,} items)")

    # --- hexes
    hexes = gpd.read_file(kontur_path("me"))
    popcol = next(c for c in hexes.columns if c.lower() == "population")
    hp = gpd.GeoDataFrame({"pop": hexes[popcol].to_numpy(dtype=float)},
                          geometry=hexes.geometry.centroid, crs=hexes.crs).to_crs(CRS_M)
    um_r = um.reset_index()
    j = gpd.sjoin(hp, um_r[["unit", "geometry"]], how="left", predicate="within")
    j = j[~j.index.duplicated(keep="first")].reindex(hp.index)
    outside = j["unit"].isna()
    print(f"  Kontur ME: {len(hexes):,} hexes, {hp['pop'].sum():,.0f} people; "
          f"{int(outside.sum()):,} hexes ({hp.loc[outside, 'pop'].sum():,.0f} people) outside every unit")
    o = hp[outside]
    ne = gpd.read_file(NE)
    a3 = "ADM0_A3"
    ne = ne[ne[a3].isin(["MNE"] + NEIGHBOURS)][[a3, "geometry"]].to_crs(CRS_M)
    oc = gpd.sjoin(o, ne, how="left", predicate="within")
    oc = oc[~oc.index.duplicated(keep="first")].reindex(o.index)
    where = oc[a3].fillna("none")
    d_nb = o.geometry.distance(ne[ne[a3] != "MNE"].geometry.union_all())
    snap_ok = (where.isin(["MNE", "none"])) & (d_nb > NB_M)
    for k, s in (("in NE's MNE, clear of a border", (where == "MNE") & snap_ok),
                 ("in the sea, clear of a border", (where == "none") & snap_ok),
                 ("near a border", where.isin(["MNE", "none"]) & ~snap_ok),
                 ("in a neighbour", where.isin(NEIGHBOURS))):
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

    cx = lay.assign(wx=lay.geometry.x * lay["pop"], wy=lay.geometry.y * lay["pop"]) \
        .groupby("unit")[["wx", "wy", "pop"]].sum()
    big = (~st["how"].isin(["osm", "wikidata"])) & \
        (st["Ukupno"].fillna(0) > 0.5 * st["unit"].map(census))
    for i in st.index[big]:
        u = st.at[i, "unit"]
        st.at[i, "x"] = cx.at[u, "wx"] / cx.at[u, "pop"]
        st.at[i, "y"] = cx.at[u, "wy"] / cx.at[u, "pop"]
        st.at[i, "how"] = "centroid"
    placed = st["how"].isin(["osm", "wikidata", "centroid"])
    stot = st["Ukupno"].fillna(0)
    print(f"  settlements placed: {int(st['how'].isin(['osm', 'wikidata']).sum()):,} on a point, "
          f"{int((st['how'] == 'centroid').sum())} at their municipality's population centroid; "
          f"{int((~placed).sum())} not placed ({int((st['how'] == 'ambiguous').sum())} ambiguous), "
          f"holding {stot[~placed].sum():,.0f} people "
          f"({100 * stot[~placed].sum() / census.sum():.2f}%)")
    miss = st[~placed].assign(t=stot).sort_values("t", ascending=False).head(12)
    print("    largest not placed: " + ", ".join(f"{a} [{b}] ({c}) {t:,.0f}" for a, b, c, t in
                                                zip(miss["naselje"], miss["unit"], miss["how"], miss["t"])))
    share = st[placed].assign(t=stot).groupby("unit")["t"].sum() / census
    low = share.sort_values().head(6)
    print("    lowest share of a municipality placed: " + ", ".join(
        f"{u} {v:.0%}" for u, v in low.items()))

    lay["settlement"] = ""
    sp = st[placed]
    for u, g in lay.groupby("unit"):
        s = sp[sp["unit"] == u]
        if s.empty:
            raise SystemExit(f"no placed settlement in {u}")
        sx, sy = s["x"].to_numpy(dtype=float), s["y"].to_numpy(dtype=float)
        hx, hy = g.geometry.x.to_numpy()[:, None], g.geometry.y.to_numpy()[:, None]
        k = ((hx - sx) ** 2 + (hy - sy) ** 2).argmin(axis=1)
        lay.loc[g.index, "settlement"] = s["code"].to_numpy()[k]
    got = set(lay.loc[lay["pop"] > 0, "settlement"])
    nohex = sp[~sp["code"].isin(got) & (stot[sp.index] > 0)]
    print(f"  {len(nohex)} placed settlements own no populated hex "
          f"({stot[nohex.index].sum():,.0f} people; their neighbours' hexes carry them)")

    per = lay.groupby("unit")["pop"].sum()
    empty = sorted(set(units["unit"]) - set(per.index[per > 0]))
    if empty:
        raise SystemExit(f"units with no populated hex: {empty}")
    rows = [(u, census[u], float(per.get(u, 0.0))) for u in census.index]
    ratio = sum(k for _, _, k in rows) / sum(c for _, c, _ in rows)
    nrm = sorted(((k / c / ratio), u) for u, c, k in rows)
    print(f"  Kontur / census nationally {ratio:.3f}; per unit, normalised: p10 "
          f"{nrm[len(nrm) // 10][0]:.2f}  median {nrm[len(nrm) // 2][0]:.2f}  p90 "
          f"{nrm[9 * len(nrm) // 10][0]:.2f}")
    print("  lowest: " + ", ".join(f"{u} {r:.2f}" for r, u in nrm[:6]))
    print("  highest: " + ", ".join(f"{u} {r:.2f}" for r, u in nrm[-6:]))
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
    out.to_file(OUT, layer="hexes", driver="GPKG")
    print(f"  wrote {OUT} ({len(out):,} hexes, {out['pop'].sum():,.0f} people)")
    cols = ["code", "naselje", "opstina", "unit", "how", "x", "y", "Ukupno"] + cats
    st[cols].to_csv(SETTLEMENTS, index=False, encoding="utf-8")
    print(f"  wrote {SETTLEMENTS} ({len(st):,} settlements; x, y in EPSG:{CRS_M}; "
          "empty cells are MONSTAT's `z`)")


if __name__ == "__main__":
    main()
