"""Iceland: the two NUTS 3 units, and Kontur's population grid, calibrated to each municipality.

Writes:
    data/geo/is/is_units.gpkg        IS001 and IS002, dissolved from GISCO LAU 2021's 69 municipalities
    data/geo/is/is_grid_400m.gpkg    Kontur H3 r8 hexes with `unit`, `muni` and `pop`   (`place`)
    data/geo/is/hag_man02005_2021.json   Hagstofa's population by municipality, 1 January 2021

Usage:
    python sources/is_geo.py --fetch   # Kontur's Iceland extract and Hagstofa MAN02005
    python sources/is_geo.py           # build both layers

THE COUNTING UNIT IS NUTS 3, TWO OF THEM, because that is ESS's `region` for Iceland in every
round and where the 2021 census counts citizenship (Eurostat `cens_21ctz_r3`). IS001
Höfuðborgarsvæði is the capital area, seven municipalities; IS002 Landsbyggð is the other 62.

THE MUNICIPALITIES ARE ALREADY ON DISK, AND THE JOIN IS BY CODE. GISCO LAU 2021 carries Iceland's
69 municipalities with Hagstofa's own municipality numbers, whose first digit is the statistical
region: 0 and 1 are the capital area, 2-8 the regions outside it. `CAPITAL` lists the seven codes;
the build asserts the digit rule and the list agree, and witnesses the rule against Eurostat's own
IS001 and IS002 with Hagstofa's population register, which neither the codes nor the polygons
determine.

KONTUR ALONE PUTS THE COUNTRYSIDE ON SUMMER HOUSES, SO IT IS CALIBRATED TO EACH MUNICIPALITY.
Uncalibrated, Kontur holds 0.89 of the capital area's census share, and outside it reads the
cottage and farm municipalities several times over their registered population (sources/is.md §5
has the table): the dots for Landsbyggð would go to Grímsnes, Bláskógabyggð and Skorradalur rather
than to Akureyri, Reykjanesbær and Selfoss. Iran's and the Faroes' lesson (playbooks/geography.md):
where the office publishes population below the counting unit, share that count over the unit's
hexes. Each hex keeps Kontur's share of its municipality, and the municipality's total is
Hagstofa's for 1 January 2021 (MAN02005, which is on the 1 January 2026 municipality map, so seven
2021 municipalities are merged forward by `MERGE_2026`).

The LAU shapefile's POP_2021 is zero for every Icelandic municipality (Iceland is not in GISCO's
EU-27 correspondence workbook).
"""

import json
import os
import sys
import urllib.request

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
GEO = os.path.join(ROOT, "data", "geo", "is")
LAU = os.path.join(ROOT, "data", "geo", "lau2021", "shp4326", "LAU_RG_01M_2021_4326.shp")

KONTUR_DIR = os.path.join(ROOT, "data", "geo", "kontur")
KONTUR_URL = ("https://geodata-eu-central-1-kontur-public.s3.eu-central-1.amazonaws.com/"
              "kontur_datasets/kontur_population_IS_20231101.gpkg.gz")
KONTUR_GZ = os.path.join(KONTUR_DIR, "kontur_population_IS_20231101.gpkg.gz")
KONTUR = os.path.join(KONTUR_DIR, "kontur_population_IS_20231101.gpkg")

HAG_POP = ("https://px.hagstofa.is/pxis/api/v1/is/Ibuar/mannfjoldi/2_byggdir/sveitarfelog/"
           "MAN02005.px")
POP_YEAR = "2021"
POP_JSON = os.path.join(GEO, f"hag_man02005_{POP_YEAR}.json")

UNITS_OUT = os.path.join(GEO, "is_units.gpkg")
# Named `_grid_<n>m` so kontur_cap.py and the grid-floor check recognise it as a Kontur layer.
GRID_OUT = os.path.join(GEO, "is_grid_400m.gpkg")

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) religiondots"}

N_LAU = 69
N_MUNI_2026 = 62
CAPITAL = {"0000": "Reykjavíkurborg", "1000": "Kópavogsbær", "1100": "Seltjarnarnesbær",
           "1300": "Garðabær", "1400": "Hafnarfjarðarkaupstaður", "1604": "Mosfellsbær",
           "1606": "Kjósarhreppur"}
# 2021 municipality -> its 1 January 2026 municipality, for the seven merged since. Every other
# code is its own. Asserted: the result is MAN02005's 62 codes, and no merge crosses a region digit.
MERGE_2026 = {
    "3710": "3716", "3711": "3716",                 # Helgafellssveit + Stykkishólmsbær
    "4604": "4604", "4607": "4604",                 # Tálknafjarðarhreppur + Vesturbyggð
    "5604": "5613", "5611": "5613", "5612": "5613",  # Blönduós, Skagabyggð, Húnavatnshreppur
    "5200": "5716", "5706": "5716",                 # Skagafjörður + Akrahreppur
    "6607": "6613", "6612": "6613",                 # Skútustaðahreppur + Þingeyjarsveit
    "6706": "6710", "6709": "6710",                 # Svalbarðshreppur + Langanesbyggð
}
# cens_21ctz_r3, 1 January 2021, TOTAL. The same figures sources/is.py asserts.
CENSUS = {"IS001": 230_657, "IS002": 128_465}
ISN93 = 3057                  # Iceland's national projected CRS, metres
SNAP_M = 500                  # vu_grid.py's cap for a hex centre just offshore
REG_BAND = (0.98, 1.02)       # each unit's register/census ratio over the national one


def fetch():
    import gzip
    import shutil

    import requests

    os.makedirs(KONTUR_DIR, exist_ok=True)
    if os.path.exists(KONTUR) and os.path.getsize(KONTUR) > 100_000:
        print("already have", KONTUR)
    else:
        if not os.path.exists(KONTUR_GZ) or os.path.getsize(KONTUR_GZ) < 10_000:
            print("GET", KONTUR_URL)
            r = requests.get(KONTUR_URL, timeout=600, headers={"User-Agent": "religiondots/1.0"})
            r.raise_for_status()
            if r.content[:2] != b"\x1f\x8b":
                raise SystemExit(f"not gzip: first bytes {r.content[:40]!r}")
            with open(KONTUR_GZ + ".tmp", "wb") as fh:
                fh.write(r.content)
            os.replace(KONTUR_GZ + ".tmp", KONTUR_GZ)
            print(f"  {os.path.getsize(KONTUR_GZ):,} bytes")
        with gzip.open(KONTUR_GZ, "rb") as src, open(KONTUR + ".tmp", "wb") as dst:
            shutil.copyfileobj(src, dst)
        os.replace(KONTUR + ".tmp", KONTUR)
        print(f"  decompressed to {KONTUR}  {os.path.getsize(KONTUR):,} bytes")

    os.makedirs(GEO, exist_ok=True)
    if os.path.exists(POP_JSON):
        print("already have", POP_JSON)
        return
    meta = json.load(urllib.request.urlopen(urllib.request.Request(HAG_POP, headers=UA), timeout=120))
    pick = {"Aldur": ["Alls"], "Ár": [POP_YEAR], "Kyn": ["Alls"]}
    query = []
    for v in meta["variables"]:
        if v["text"] in pick:
            codes = [c for c, t in zip(v["values"], v["valueTexts"]) if t in pick[v["text"]]]
            if len(codes) != 1:
                raise SystemExit(f"MAN02005 `{v['text']}`: {codes} for {pick[v['text']]}")
        else:
            codes = list(v["values"])
        query.append({"code": v["code"], "selection": {"filter": "item", "values": codes}})
    body = json.dumps({"query": query, "response": {"format": "json-stat2"}}).encode()
    req = urllib.request.Request(HAG_POP, data=body, headers={"Content-Type": "application/json", **UA})
    d = json.load(urllib.request.urlopen(req, timeout=300))
    with open(POP_JSON + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(d, fh, ensure_ascii=False)
    os.replace(POP_JSON + ".tmp", POP_JSON)
    print(f"  MAN02005 {POP_YEAR}: {os.path.getsize(POP_JSON):,} bytes")


def read_register():
    """{2026 municipality code: population on 1 January 2021}, the national row checked."""
    d = json.load(open(POP_JSON, encoding="utf-8"))
    ids, sizes = d["id"], d["size"]
    muni = [x for x in ids if d["dimension"][x]["label"] == "Sveitarfélag"
            or x == "Sveitarfélag"][0]
    if any(s != 1 for x, s in zip(ids, sizes) if x != muni):
        raise SystemExit(f"MAN02005: more than one value in a non-municipality dimension {sizes}")
    idx = d["dimension"][muni]["category"]["index"]
    vals = d["value"]
    out = {code: float(vals[i]) for code, i in idx.items()}
    total = out.pop("9999")
    if abs(sum(out.values()) - total) > 0.5:
        raise SystemExit(f"MAN02005: municipalities sum to {sum(out.values()):,.0f}, Alls {total:,.0f}")
    if len(out) != N_MUNI_2026:
        raise SystemExit(f"MAN02005: {len(out)} municipalities, expected {N_MUNI_2026}")
    print(f"  Hagstofa MAN02005, 1 January {POP_YEAR}: {len(out)} municipalities, {total:,.0f} people")
    return out


def build_units(register):
    import geo_checks

    g = geo_checks.read_layer(LAU, "GISCO LAU 2021, Iceland", where="CNTR_CODE='IS'").to_crs(4326)
    g["lau"] = g["LAU_ID"].astype(str).str.strip().str.zfill(4)
    print(f"  GISCO LAU 2021, Iceland: {len(g)} municipalities")
    if len(g) != N_LAU:
        raise SystemExit(f"expected {N_LAU} municipalities, got {len(g)}")
    if g["lau"].duplicated().any():
        raise SystemExit("duplicate municipality codes")

    by_digit = set(g.loc[g["lau"].str[0].isin(["0", "1"]), "lau"])
    if by_digit != set(CAPITAL):
        raise SystemExit(f"the region digit and CAPITAL disagree: digit-only "
                         f"{sorted(by_digit - set(CAPITAL))}, list-only {sorted(set(CAPITAL) - by_digit)}")
    names = dict(zip(g["lau"], g["LAU_NAME"]))
    wrong = {c: names.get(c) for c, n in CAPITAL.items() if names.get(c) != n}
    if wrong:
        raise SystemExit(f"capital-area codes carry other names: {wrong}")
    g["unit"] = g["lau"].map(lambda c: "IS001" if c in CAPITAL else "IS002")
    g["muni"] = g["lau"].map(lambda c: MERGE_2026.get(c, c))
    stray = sorted(set(MERGE_2026) - set(g["lau"]))
    if stray:
        raise SystemExit(f"MERGE_2026 names codes the LAU file lacks: {stray}")
    if any(k[0] != v[0] for k, v in MERGE_2026.items()):
        raise SystemExit("a MERGE_2026 entry crosses a region digit")
    if set(g["muni"]) != set(register):
        raise SystemExit(f"LAU merged to 2026 against MAN02005: LAU-only "
                         f"{sorted(set(g['muni']) - set(register))}, register-only "
                         f"{sorted(set(register) - set(g['muni']))}")
    print(f"    IS001 {int((g['unit'] == 'IS001').sum())} municipalities, IS002 "
          f"{int((g['unit'] == 'IS002').sum())}; the digit rule and the list agree; the 69 merge to "
          f"MAN02005's {len(register)} exactly")

    # ---- witness: the code rule against Eurostat's own NUTS 3 totals, through the register ----
    reg_unit = {"IS001": sum(v for k, v in register.items() if k in CAPITAL),
                "IS002": sum(v for k, v in register.items() if k not in CAPITAL)}
    nat = sum(reg_unit.values()) / sum(CENSUS.values())
    print(f"    register {POP_YEAR} over census 2021: national {nat:.4f}")
    for u in sorted(CENSUS):
        r = reg_unit[u] / CENSUS[u] / nat
        print(f"      {u}  register {reg_unit[u]:>9,.0f}  census {CENSUS[u]:>9,}  relative {r:.4f}")
        if not REG_BAND[0] <= r <= REG_BAND[1]:
            raise SystemExit(f"{u}: register/census {r:.4f} of the national ratio, outside {REG_BAND}; "
                             "the code-to-NUTS-3 rule is not Eurostat's")

    minx, miny, maxx, maxy = g.total_bounds
    print(f"    bbox {minx:.2f},{miny:.2f} .. {maxx:.2f},{maxy:.2f}")
    # Bjargtangar is the west point near 24.5W, Gerpir the east near 13.5W, Surtsey the south at
    # 63.3N, Grímsey the north at 66.55N.
    if not (-25.2 < minx < -24.0 and -14.0 < maxx < -13.0 and 63.1 < miny < 63.5
            and 66.3 < maxy < 66.8):
        raise SystemExit("bbox is not Iceland's; check the CRS and the country filter")

    units = g.dissolve(by="unit", as_index=False)[["unit", "geometry"]]
    units["area_km2"] = units.to_crs(ISN93).area.values / 1e6
    for r in units.itertuples():
        print(f"    {r.unit}  {r.area_km2:,.0f} km²")
    os.makedirs(GEO, exist_ok=True)
    units.to_file(UNITS_OUT, layer="units", driver="GPKG")
    print(f"  wrote {UNITS_OUT}")
    return g[["lau", "muni", "unit", "LAU_NAME", "geometry"]], units


def build_grid(lau, units, register):
    import geopandas as gpd
    import numpy as np
    import pyogrio
    import shapely

    import geo_checks

    if not os.path.exists(KONTUR):
        raise SystemExit(f"missing {KONTUR}; run sources/is_geo.py --fetch first")
    layers = list(pyogrio.list_layers(KONTUR)[:, 0])
    layer = "population" if "population" in layers else layers[0]
    hexes = geo_checks.read_layer(KONTUR, "Kontur IS", layer=layer).to_crs(4326)
    total = float(hexes["population"].sum())
    print(f"\n  Kontur IS: {len(hexes):,} hexes, {total:,.0f} people")

    centres = gpd.GeoDataFrame(geometry=hexes.geometry.representative_point(), crs=4326)
    hit = gpd.sjoin(centres, lau[["lau", "geometry"]], how="left", predicate="within")
    hit = hit[~hit.index.duplicated(keep="first")]
    hexes["lau"] = hit["lau"].to_numpy()

    out = hexes["lau"].isna()
    print(f"    {int(out.sum())} hex centres outside every municipality "
          f"({hexes.loc[out, 'population'].sum():,.0f} people)")
    if out.any():
        near = gpd.sjoin_nearest(centres[out.to_numpy()].to_crs(ISN93),
                                 lau[["lau", "geometry"]].to_crs(ISN93),
                                 how="left", max_distance=SNAP_M, distance_col="dist")
        near = near[~near.index.duplicated(keep="first")]
        got = near["lau"].notna()
        hexes.loc[near.index[got], "lau"] = near.loc[got, "lau"].to_numpy()
        print(f"    {int(got.sum())} snapped within {SNAP_M} m "
              f"({hexes.loc[near.index[got], 'population'].sum():,.0f} people)")
    lost = hexes["lau"].isna()
    lost_people = float(hexes.loc[lost, "population"].sum())
    print(f"    {int(lost.sum())} left in no municipality ({lost_people:,.0f} people, "
          f"{100 * lost_people / total:.3f}%)")
    if lost_people / total > 0.005:
        raise SystemExit("over 0.5% of Kontur's people fall in no municipality; check the layer")
    hexes = hexes[~lost].copy()
    info = lau.set_index("lau")
    hexes["muni"] = hexes["lau"].map(info["muni"])
    hexes["unit"] = hexes["lau"].map(info["unit"])

    # ---- Kontur's own error, measured: per unit and per municipality against the register -----
    kon_u = hexes.groupby("unit")["population"].sum()
    nat_c = kon_u.sum() / sum(CENSUS.values())
    print(f"\n  Kontur 2023 alone, over census 2021 (national {nat_c:.3f}): " + ", ".join(
        f"{u} {kon_u[u] / CENSUS[u] / nat_c:.3f}" for u in sorted(CENSUS)) + " of the national ratio")
    kon_m = hexes.groupby("muni")["population"].sum()
    empty = sorted(m for m in register if register[m] > 0 and kon_m.get(m, 0.0) <= 0)
    if empty:
        raise SystemExit(f"municipalities with people and no Kontur hex: {empty}")
    nat_m = kon_m.sum() / sum(register.values())
    rel = {m: kon_m[m] / register[m] / nat_m for m in register}
    name = info.groupby("muni")["LAU_NAME"].agg(lambda s: " + ".join(sorted(s)))
    order = sorted(rel, key=lambda m: -rel[m])
    print(f"    per municipality, Kontur over the register {POP_YEAR} relative to the national "
          f"{nat_m:.3f}; the ten highest and the five lowest:")
    for m in order[:10] + order[-5:]:
        print(f"      {m}  {name[m][:44]:<46}{kon_m[m]:>9,.0f}{register[m]:>9,.0f}  {rel[m]:6.2f}")
    over = sum(max(0.0, kon_m[m] / nat_m - register[m]) for m in register if m not in CAPITAL)
    print(f"    Kontur holds {over:,.0f} more people than the register in the municipalities it "
          "over-reads outside the capital area (after the national scale)")

    # ---- calibrate: each hex keeps its share of its municipality, which takes the register count
    factor = {m: register[m] / kon_m[m] for m in register}
    hexes["pop"] = hexes["population"] * hexes["muni"].map(factor)
    check = hexes.groupby("muni")["pop"].sum()
    if max(abs(check[m] - register[m]) for m in register) > 0.5:
        raise SystemExit("calibration did not reproduce the register")
    print(f"    calibrated: {hexes['pop'].sum():,.0f} people; factors from "
          f"{min(factor.values()):.3f} ({name[min(factor, key=factor.get)]}) to "
          f"{max(factor.values()):.3f} ({name[max(factor, key=factor.get)]})")

    # ---- clip each hex to its unit where its centre was inside; a snapped hex keeps whatever of
    # it lies in the unit, or its whole shape when nothing does (water.clip removes the sea later)
    print("\n  clipping hexes to their unit…")
    poly = units.set_index("unit")["geometry"]
    geom = hexes.geometry.to_numpy()
    who = hexes["unit"].to_numpy()
    new = geom.copy()
    n_clipped = 0
    for unit, parent in poly.items():
        idx = np.flatnonzero(who == unit)
        shapely.prepare(parent)
        inside = shapely.contains_properly(parent, geom[idx])
        edge = idx[~inside]
        if edge.size:
            cut = shapely.intersection(parent, geom[edge])
            keep = ~shapely.is_empty(cut)
            new[edge[keep]] = cut[keep]
            n_clipped += int(keep.sum())
    hexes["geometry"] = gpd.GeoSeries(new, crs=4326, index=hexes.index)
    print(f"    {n_clipped:,} edge hexes clipped, {len(hexes) - n_clipped:,} left whole")

    hexes = hexes[["unit", "muni", "pop", "geometry"]]
    per = hexes.groupby("unit").agg(hexes=("pop", "size"), pop=("pop", "sum"))
    for u, r in per.iterrows():
        print(f"    {u}  {r['hexes']:,} hexes, {r['pop']:,.0f} people")
    if set(per.index) != set(CENSUS):
        raise SystemExit(f"units in the grid {sorted(per.index)}")
    hexes.to_file(GRID_OUT, layer="grid", driver="GPKG")
    print(f"  wrote {GRID_OUT}  {len(hexes):,} hexes")


def main():
    if "--fetch" in sys.argv:
        fetch()
    register = read_register()
    lau, units = build_units(register)
    build_grid(lau, units, register)


if __name__ == "__main__":
    main()
