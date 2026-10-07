"""Hong Kong: the 1,746 Large Subunit Groups, and Kontur's hexes cut to them.

    python sources/hk_geo.py --fetch    the boundary GeoJSON from the CSDI portal (~48 MB)
    python sources/hk_geo.py            build data/geo/hk/

Writes
    data/geo/hk/hk_units.gpkg      1,746 LSUG polygons: `unit` (the LSUG code), `dc` (main district)
    data/geo/hk/hk_lookup.csv      unit, dc, w: each LSUG's districts and their weights
    data/geo/hk/hk_cut.gpkg        Kontur 2023 hexes cut to the LSUGs (and districts): `unit`, `pop`

THE UNITS. C&SD publishes the boundaries of its own release on the CSDI portal (dataset
censtatd_rcd_1635933282224_58228, layer LSUG_21C), keyed by the same `lsbg` code as the table,
so the join is a code join, asserted both ways.

THE DISTRICT. Each LSUG needs its District Council district for the language mix
(countries/hk.py). The release carries none, and LSUGs do NOT nest in districts: 80 cross a
district line (more than 0.5% of their Kontur people on the far side). The districts are the
census's own, the union of its 452 constituency areas (CSDI dataset censtatd_rcd_1635933003339_9920,
the DCCA code's first two digits are the district); HAD's district file gives the same answer.
Each LSUG's weight in each district is the share of its Kontur people there (its area where
Kontur has nobody). The witness is the census's district table: the LSUGs' five groups, shared
by these weights, miss DC_21C by 40,355 people of 7.18 million (0.6%), against 45,289 with each
LSUG wholly in its main district; the script stops above 1%. The miss is the LSUG polygons and
the district polygons disagreeing about a few big straddlers (one LSUG's 20,600 Cantonese
speakers counted in Kwun Tong, its polygon mostly in Sai Kung), not a join error.

PLACEMENT. Hong Kong is below the Kontur floor (religiondots playbooks/geography.md, Malta): the
median LSUG is 0.06 km2, a twelfth of one 0.74 km2 hex, so a centroid join would leave most of
them with no hex. The hexes are cut by the units and each hex's people shared over its pieces by
area, divided by the WHOLE hex (South Africa's rule; see the comment at the cut for why not
Malta's). Slivers under 50 m2 are dropped. A unit with no populated piece gets its own polygon at
weight 0 (equal shares inside it). Kontur only moves dots inside an LSUG; the census decides how
many each gets. The grid-floor warning the scatter prints is accepted (sources/hk.md).

KONTUR'S CAP. religiondots reviewed Hong Kong's blocks at the cap and marked every one `real`
(religiondots/kontur_cap.csv, 2026-09-14), which changes no weight. This script runs that check on
the UNCUT hexes and stops if any block is unregistered or would be changed. The cut layer is named
`hk_cut.gpkg`, not `*_hexes.gpkg`, ON PURPOSE: kontur_cap.py recognises Kontur layers by that suffix,
and on cut pieces it splits the 62 reviewed blocks into 249 fragments (its block-finder spaces by
the median distance between neighbouring pieces, a fraction of a hex here), each of which would
need its own registry row saying the same thing. The scatter's grid-floor warning goes with it.
"""
import gzip
import math
import os
import random
import shutil
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RAW = ROOT / "data" / "raw" / "hk"
GEO = ROOT / "data" / "geo" / "hk"
NORM = ROOT / "data" / "normalized" / "hk.csv"
RD = ROOT.parent / "religiondots"
RD_KONTUR_GZ = RD / "data" / "raw" / "hk" / "kontur_population_HK_20231101.gpkg.gz"
KONTUR = ROOT / "data" / "geo" / "kontur" / "kontur_population_HK_20231101.gpkg"
BOUNDS = RAW / "LSUG_21C.geojson"
BOUNDS_URL = ("https://portal.csdi.gov.hk/csdi-webpage/file-api?"
              "dataset_id=censtatd_rcd_1635933282224_58228&format=geojson&layer_name=LSUG_21C")
DCCA = RAW / "DCCA_21C.geojson"
DCCA_URL = ("https://portal.csdi.gov.hk/csdi-webpage/file-api?"
            "dataset_id=censtatd_rcd_1635933003339_9920&format=geojson&layer_name=DCCA_21C")
HK_GRID = 2326          # Hong Kong 1980 Grid, metres
N_UNITS = 1746
SLIVER_M2 = 50
# the head of a DCCA code (IDDS's AREA code) -> the census's district letter (DC_21C `dc_class`)
DC_CODE = {"11": "A", "12": "B", "13": "C", "14": "D", "27": "E", "23": "F", "24": "G", "25": "H",
           "26": "J", "31": "S", "32": "K", "33": "L", "34": "M", "35": "N", "36": "P", "37": "R",
           "38": "Q", "39": "T"}


def fetch():
    import requests
    RAW.mkdir(parents=True, exist_ok=True)
    for dest, url in ((BOUNDS, BOUNDS_URL), (DCCA, DCCA_URL)):
        if dest.exists() and dest.stat().st_size > 10_000_000:
            print("already have", dest.name)
            continue
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=600)
        r.raise_for_status()
        dest.with_suffix(".part").write_bytes(r.content)
        os.replace(dest.with_suffix(".part"), dest)
        print(f"got {dest.name}: {len(r.content):,} bytes")


def kontur():
    if KONTUR.exists() and KONTUR.stat().st_size > 10_000:
        return KONTUR
    KONTUR.parent.mkdir(parents=True, exist_ok=True)
    tmp = KONTUR.with_suffix(".part")
    with gzip.open(RD_KONTUR_GZ, "rb") as src, open(tmp, "wb") as dst:
        shutil.copyfileobj(src, dst)
    with open(tmp, "rb") as fh:
        if fh.read(4) != b"SQLi":
            raise SystemExit(f"{tmp} is not a GeoPackage")
    os.replace(tmp, KONTUR)
    return KONTUR


def pear(a, b):
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    return num / math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))


def main():
    import geopandas as gpd
    import numpy as np
    import pandas as pd
    import rdlink

    if "--fetch" in sys.argv:
        fetch()

    norm = pd.read_csv(NORM, dtype={"geo_id": str})
    lsg = norm[norm["geo_level"] == "lsg"]
    dc5 = norm[norm["geo_level"] == "dc"].groupby(["geo_id", "group"])["count"].sum()

    # ---- units, joined on the release's own code -------------------------------------------
    u = gpd.read_file(BOUNDS)[["lsbg", "t_pop", "geometry"]].rename(columns={"lsbg": "unit"})
    u["unit"] = u["unit"].astype(str)
    if len(u) != N_UNITS or u["unit"].duplicated().any():
        raise SystemExit(f"!! boundary file: {len(u)} features, {u['unit'].duplicated().sum()} repeated codes")
    w, s, e, n = u.total_bounds
    if not (0.1 < e - w < 1.5 and 0.1 < n - s < 1.5):
        raise SystemExit("!! Hong Kong does not fit in a degree; geometry is torn")
    tab = set(lsg["geo_id"])
    if tab - set(u["unit"]):
        raise SystemExit(f"!! {len(tab - set(u['unit']))} table LSUGs have no polygon")
    nolang = set(u["unit"]) - tab
    print(f"LSUG_21C boundaries: {len(u):,} polygons, every table code matched both ways"
          + (f" ({len(nolang)} polygons with no language rows: {sorted(nolang)[:5]})" if nolang else ""))
    um = u.to_crs(HK_GRID)
    um["geometry"] = um.geometry.make_valid()
    area = um.geometry.area / 1e6
    print(f"  area: median {area.median():.3f} km2, p10 {area.quantile(.1):.3f}, "
          f"p90 {area.quantile(.9):.2f}, total {area.sum():,.0f} km2")

    # ---- districts: the census's own, as the union of its 452 constituency areas -------------
    dcca = gpd.read_file(DCCA)[["dcca", "geometry"]]
    dcca["dc"] = dcca["dcca"].astype(str).str[:2].map(DC_CODE)
    if len(dcca) != 452 or dcca["dc"].isna().any() or dcca["dc"].nunique() != 18:
        raise SystemExit("!! DCCA_21C: expected 452 areas in 18 districts")
    d = dcca.to_crs(HK_GRID)
    d["geometry"] = d.geometry.make_valid()
    d = d.dissolve("dc").reset_index()[["dc", "geometry"]]
    # identity, not intersection: the slivers of an LSUG outside every district (the two files'
    # coastlines differ) stay in the placement layer with no district
    fr = gpd.overlay(um[["unit", "geometry"]], d, how="identity", keep_geom_type=True)
    fr["a"] = fr.geometry.area
    fr = fr[fr["a"] > 0].reset_index(drop=True)
    fr["fid"] = np.arange(len(fr))

    # ---- Kontur, cap check on the uncut hexes ------------------------------------------------
    hx = gpd.read_file(kontur())
    if len(hx) == 0:
        raise SystemExit("Kontur HK extract has ZERO features")
    popcol = next(c for c in hx.columns if c.lower() == "population")
    hx = gpd.GeoDataFrame({"hid": np.arange(len(hx)), "pop": hx[popcol].to_numpy(dtype=float)},
                          geometry=hx.geometry, crs=hx.crs)
    print(f"Kontur HK: {len(hx):,} hexes, {hx['pop'].sum():,.0f} people")
    kc = rdlink.module("kontur_cap")
    raw = hx.to_crs(4326)
    cj = gpd.sjoin(gpd.GeoDataFrame(geometry=hx.geometry.centroid, crs=hx.crs).to_crs(4326),
                   u[["unit", "geometry"]], how="left", predicate="within")
    cj = cj[~cj.index.duplicated()].reindex(raw.index)
    raw["unit"] = cj["unit"].fillna("outside").astype(str)
    capped = kc.apply(raw.copy(), "hk", "hk_hexes.gpkg")
    if not np.allclose(capped["pop"].to_numpy(), raw["pop"].to_numpy()):
        raise SystemExit("!! the cap registry now changes Hong Kong's weights; review before cutting")
    print("  cap check on the uncut hexes: every block at the cap is registered `real`; no weight changes")

    # ---- cut ---------------------------------------------------------------------------------
    hm = hx.to_crs(HK_GRID)
    pieces = gpd.overlay(hm, fr[["unit", "dc", "geometry"]], how="intersection", keep_geom_type=True)
    pieces["a"] = pieces.geometry.area
    land = pieces.groupby("hid")["a"].sum()
    # Divided by the WHOLE hex, not its land (South Africa's rule, not Malta's): Kontur gives
    # harbour hexes that are mostly water the full density of the waterfront beside them, and
    # sharing that over the land alone put 78% of Central's 12102L on a 7,357 m2 strip of
    # harbourfront (5.3 million/km2). Whole-hex sharing also keeps every piece at or below
    # Kontur's own density.
    hexa = hm.set_index("hid").geometry.area
    pieces["pop"] = pieces["pop"] * pieces["a"] / pieces["hid"].map(hexa)
    lost = hx.loc[~hx["hid"].isin(land.index), "pop"].sum()
    print(f"  cut: {len(pieces):,} pieces; {int((~hx['hid'].isin(land.index)).sum())} hexes "
          f"({lost:,.0f} people) touch no LSUG and are dropped; each hex's people shared over its "
          "pieces by area, divided by the whole hex")
    # ---- each LSUG's district weights: its Kontur people in each district part ---------------
    # LSUGs do not nest in districts: 2021's subunit groups were drawn on planning units, and 76
    # of them cross a district line by more than 1% of their area. Each LSUG's mix is the
    # weighted mean of its districts' mixes, weighted by where Kontur puts its people (by area
    # where Kontur puts nobody).
    kp = pieces.dropna(subset=["dc"]).groupby(["unit", "dc"])["pop"].sum()
    ka = fr.dropna(subset=["dc"]).groupby(["unit", "dc"])["a"].sum()
    wts = pd.DataFrame({"pop": kp, "a": ka}).fillna(0.0).reset_index()
    tp = wts.groupby("unit")["pop"].transform("sum")
    ta = wts.groupby("unit")["a"].transform("sum")
    wts["w"] = np.where(tp > 0, wts["pop"] / tp.where(tp > 0, 1), wts["a"] / ta)
    wts = wts[wts["w"] >= 0.005]
    wts["w"] = wts["w"] / wts.groupby("unit")["w"].transform("sum")
    if set(wts["unit"]) != set(um["unit"]):
        raise SystemExit("!! some LSUGs touch no district")
    multi = wts.groupby("unit").size()
    print(f"  district weights: {int((multi > 1).sum())} LSUGs span two or more districts "
          f"(Kontur people, 0.5% floor); the rest sit in one")
    best = wts.sort_values("w").groupby("unit").tail(1).set_index("unit")["dc"]
    got = lsg.assign(dc=lsg["geo_id"].map(best)).groupby(["dc", "group"])["count"].sum()
    diff = got.sub(dc5, fill_value=0)
    soft = lsg.merge(wts, left_on="geo_id", right_on="unit")
    soft = (soft["count"] * soft["w"]).groupby([soft["dc"], soft["group"]]).sum()
    sdiff = soft.sub(dc5, fill_value=0)
    print(f"  witness, DC_21C's five groups by district: each LSUG in its main district misses by "
          f"{diff.abs().sum() / 2:,.0f} people over 90 cells; shared by these weights, by "
          f"{sdiff.abs().sum() / 2:,.0f} (of {dc5.sum():,.0f})")
    if sdiff.abs().sum() / 2 > 0.01 * dc5.sum():
        raise SystemExit("!! the district weights miss the census's district table by over 1%")

    GEO.mkdir(parents=True, exist_ok=True)
    um[["unit", "geometry"]].assign(dc=um["unit"].map(best)).to_crs(4326).to_file(
        GEO / "hk_units.gpkg", layer="units", driver="GPKG")
    wts[["unit", "dc", "w"]].to_csv(GEO / "hk_lookup.csv", index=False, float_format="%.6f")

    # slivers where two boundary files meet (overlay artefacts, some of zero area) go; a unit left
    # with none gets its own polygon below
    sliver = pieces["a"] < SLIVER_M2
    print(f"  {int(sliver.sum()):,} slivers under {SLIVER_M2} m2 dropped "
          f"({pieces.loc[sliver, 'pop'].sum():,.0f} Kontur people)")
    pieces = pieces[~sliver & (pieces["pop"] > 0)]
    per = pieces.groupby("unit")["pop"].sum()
    empty = sorted(set(um["unit"]) - set(per.index))
    add = um[um["unit"].isin(empty)][["unit", "geometry"]].assign(pop=0.0)
    if len(add):
        cen = lsg[lsg["geo_id"].isin(empty)].groupby("geo_id")["count"].sum()
        print(f"  {len(add)} LSUGs have no populated piece and are placed on their own polygon "
              f"({cen.sum():,.0f} people aged 5+ in them; largest {cen.max() if len(cen) else 0:,.0f})")
    layer = pd.concat([pieces[["unit", "pop", "geometry"]], add], ignore_index=True)
    layer = gpd.GeoDataFrame(layer, geometry="geometry", crs=HK_GRID).to_crs(4326)
    npc = layer.groupby("unit").size()
    print(f"  pieces per LSUG: median {npc.median():.0f}, p10 {npc.quantile(.1):.0f}, "
          f"{int((npc == 1).sum())} with one")

    # ---- Kontur against the census, per LSUG -------------------------------------------------
    census = dict(zip(u["unit"], pd.to_numeric(u["t_pop"], errors="coerce").fillna(0)))
    rows = [(k, census[k], float(per.get(k, 0.0))) for k in census if census[k] > 0]
    ratio = sum(x for _, _, x in rows) / sum(c for _, c, _ in rows)
    normd = sorted((x / c / ratio, k) for k, c, x in rows)
    q = lambda p: normd[int(p * (len(normd) - 1))][0]
    print(f"  Kontur / census nationally {ratio:.3f}; per LSUG, normalised: p10 {q(.1):.2f}, "
          f"median {q(.5):.2f}, p90 {q(.9):.2f}; {sum(1 for r, _ in normd if not 1/3 <= r <= 3)} "
          f"of {len(normd)} outside a factor of 3")
    pos = [(c, x) for _, c, x in rows if x > 0]
    lc, lx = [math.log(c) for c, _ in pos], [math.log(x) for _, x in pos]
    r = pear(lc, lx)
    rng = random.Random(0)
    bestr = max(abs(pear(lc, rng.sample(lx, len(lx)))) for _ in range(200))
    print(f"  log correlation r = {r:.3f} against a best of {bestr:.3f} over 200 shuffles")
    if r <= bestr:
        raise SystemExit("!! the cut grid carries no information about the LSUGs")

    out = GEO / "hk_cut.gpkg"
    tmp = GEO / "hk_cut.part.gpkg"
    layer.to_file(tmp, layer="hexes", driver="GPKG")
    os.replace(tmp, out)
    print(f"wrote {out} ({len(layer):,} pieces), hk_units.gpkg, hk_lookup.csv")


if __name__ == "__main__":
    main()
