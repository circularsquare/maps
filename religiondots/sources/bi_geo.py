"""Burundi — the 17 provinces of the 2008 census, their 2008 populations, and the grid.

Writes:
    data/geo/bi/bi_units.gpkg         the 17 units (`units`)
    data/geo/bi/bi_hexes.gpkg         Kontur 400 m hexes with `unit`, `commune` and `pop` (`place`)
    data/geo/bi/bi_lookup.csv         unit -> name, 2008 population (resident, ordinary households,
                                      collective households, urban, rural), Kontur population
    data/geo/bi/bi_communes.csv       the 116 rural-province communes and Bujumbura Mairie, each
                                      with its 2008 province, for `sources/bi.py`'s check of the
                                      survey's commune column against its province

Usage:
    python sources/bi_geo.py --fetch    geoBoundaries ADM1 and ADM2 (~10 MB), Kontur (~3 MB),
                                        the 2008 census's chapter 1 workbook from the Wayback Machine
    python sources/bi_geo.py            rebuild from data/raw/bi/

## WHY THE 17 PROVINCES OF 2008

The religion table is the 2008 census's (`sources/bi.py`), and the survey that places it is the
Afrobarometer's rounds 5 (2012) and 6 (2014), both fielded on the same 17 provinces. Rumonge was
made a province in 2015 out of three Bururi communes (Burambi, Buyengero, Rumonge) and two of
Bujumbura Rural (Bugarama, Muhuta), and the 2025 reform (loi 2023) replaced all 18 with five
provinces and 42 communes. COD-AB Burundi version 02 (July 2026) is that five-province map, with no
older edition on HDX, so it cannot rebuild 2008's provinces.

## BOUNDARIES: geoBoundaries gbOpen BDI, ADM1 (18 provinces) AND ADM2 (119 communes)

ADM2 is the pre-2025 commune map: the 116 communes outside Bujumbura Mairie, which are the 2008
census's own (Tableau 1.5 lists them under their provinces), and the capital as the three communes
of 2014 where the census had 13 urban communes. geoBoundaries carries no parent column, so each
commune's 2008 province is read from the census table by name and witnessed by geography: the
commune's representative point must fall in the ADM1 province of the same name, or, for the five
communes of Rumonge province, in Rumonge. Two names repeat inside the census table (Butaganzwa in
Kayanza and in Ruyigi; Kanyosha in Bujumbura Rural and in the Mairie) and are matched inside their
province only. The census prints Makamba twice in Makamba province and omits Mabanda. The FIRST
row (45,836, urban 3,249) is Mabanda and the second (93,558, urban 9,396, the provincial seat) is
Makamba: every province's communes are listed alphabetically, and Kontur 2023 reads 1.34x and 1.20x
that way round against 0.66x and 2.46x the other way (national 1.65x). Measured 2026-10-03.

geoBoundaries' ADM2 is a pre-2014 shapefile and its names stop at ten letters; `GB_ALIAS` pairs the
four that differ otherwise. Kontur per commune is the join's witness: p10 1.24, p90 2.11 of the
2008 count, nothing outside `KONTUR_COMMUNE_BAND`.

## POPULATION: THE 2008 CENSUS ITSELF, WITH ITS URBAN AND RURAL SPLIT

ISTEEBU's chapter 1 workbook, re-read on every run: Tableau 1.4 (ordinary and collective
households by commune) and Tableau 1.5 (urban and rural by commune). 8,053,574 residents, of whom
7,964,078 are in ordinary households, the universe of the religion table, and 89,496 in collective
ones. The units are drawn at the census's own count; the map shows Burundi as it was counted in
August 2008, not today's 12-13 million (RGPHAE 2024 publishes only on the five new provinces).

## PLACEMENT: KONTUR, CALIBRATED TO EACH COMMUNE'S 2008 COUNT

Kontur (2023) is a picture of fifteen years later. Each hex's weight is its Kontur people scaled so
its commune sums to the 2008 census count (the Mairie as one), so dots inside a province follow the
2008 distribution between communes and Kontur only inside a commune. Hexes are cut to Burundi's
outline so no dot lands in Lake Tanganyika (water.py clips sea, not lakes); slivers under 20% land
are dropped first. Kontur's per-commune ratio is wide (Giteranyi 3.47x, Nyamurenza 0.63x against
1.65x nationally) and is not investigated further: the calibration takes it out at the commune.
"""

import os
import re
import sys

os.environ.setdefault("OMP_NUM_THREADS", "6")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "bi")
GEO = os.path.join(ROOT, "data", "geo", "bi")
OUT_UNITS = os.path.join(GEO, "bi_units.gpkg")
OUT_HEXES = os.path.join(GEO, "bi_hexes.gpkg")
OUT_LOOKUP = os.path.join(GEO, "bi_lookup.csv")
OUT_COMMUNES = os.path.join(GEO, "bi_communes.csv")

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36")

GB = "https://github.com/wmgeolab/geoBoundaries/raw/9469f09/releaseData/gbOpen/BDI"
KONTUR_GPKG = "kontur_population_BI_20231101.gpkg"
CHAP1 = "rgph2008_chapitre1.xlsx"
DOWNLOADS = {
    "geoBoundaries-BDI-ADM1.geojson": (f"{GB}/ADM1/geoBoundaries-BDI-ADM1.geojson", b"{", 100_000),
    "geoBoundaries-BDI-ADM2.geojson": (f"{GB}/ADM2/geoBoundaries-BDI-ADM2.geojson", b"{", 100_000),
    KONTUR_GPKG + ".gz": (
        "https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
        "kontur_population_BI_20231101.gpkg.gz", b"\x1f\x8b", 500_000),
    # The old ISTEEBU WordPress (isteebu.bi is now an unrelated site; sources/bi.md §1).
    CHAP1: ("https://web.archive.org/web/20220327050202id_/https://www.isteebu.bi/wp-content/"
            "uploads/2020/10/Chapitre-1.xlsx", b"PK", 50_000),
}

N_UNITS = 17
N_COMMUNES_RURAL = 116          # the communes outside Bujumbura Mairie, 2008
CENSUS_2008 = 8_053_574
ORDINARY_2008 = 7_964_078
URBAN_2008 = 811_866

MAIRIE = "BI17"
# 2008 province (Tableau 1.4/1.5 spelling, folded) -> unit id and display name. Ids in the
# census table's own order, the Mairie last as the table prints it.
PROVINCES = {
    "bubanza": ("BI01", "Bubanza"),
    "bujumburarural": ("BI02", "Bujumbura Rural"),
    "bururi": ("BI03", "Bururi"),
    "cankuzo": ("BI04", "Cankuzo"),
    "cibitoke": ("BI05", "Cibitoke"),
    "gitega": ("BI06", "Gitega"),
    "karuzi": ("BI07", "Karuzi"),
    "kayanza": ("BI08", "Kayanza"),
    "kirundo": ("BI09", "Kirundo"),
    "makamba": ("BI10", "Makamba"),
    "muramvya": ("BI11", "Muramvya"),
    "muyinga": ("BI12", "Muyinga"),
    "mwaro": ("BI13", "Mwaro"),
    "ngozi": ("BI14", "Ngozi"),
    "rutana": ("BI15", "Rutana"),
    "ruyigi": ("BI16", "Ruyigi"),
    "bujumburamairie": (MAIRIE, "Bujumbura Mairie"),
}
MAIRIE_KEY = "bujumburamairie"

# The five communes of Rumonge province (2015) and the 2008 province each came from.
RUMONGE = {"bugarama": "bujumburarural", "muhuta": "bujumburarural",
           "burambi": "bururi", "buyengero": "bururi", "rumonge": "bururi"}
# Census commune spellings that geoBoundaries spells otherwise (folded both sides).
COMMUNE_ALIAS = {}
# The census prints `MAKAMBA` twice in Makamba province; the first is Mabanda (see docstring).
MAKAMBA_FIRST = "mabanda"

# Kontur 2023 against the 2008 census: about 1.6x nationally, after fifteen years of growth. The
# band is wide on purpose; it catches a wrong join (a commune at 0.2x or 5x), not growth.
KONTUR_COMMUNE_BAND = (0.6, 4.0)


def fold(s):
    import unicodedata
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "", s.casefold())


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    for name, (url, magic, min_size) in DOWNLOADS.items():
        dst = os.path.join(RAW, name)
        if os.path.exists(dst) and os.path.getsize(dst) > min_size:
            print(f"  have {name} ({os.path.getsize(dst):,} bytes)")
            continue
        r = requests.get(url, headers={"User-Agent": UA}, timeout=900)
        r.raise_for_status()
        if not r.content.lstrip().startswith(magic):              # §5a: a 200 is not a download
            raise SystemExit(f"{name} starts {r.content[:16]!r}, not {magic!r}")
        if len(r.content) < min_size:
            raise SystemExit(f"{name} is {len(r.content):,} bytes, expected over {min_size:,}")
        with open(dst + ".part", "wb") as f:
            f.write(r.content)
        os.replace(dst + ".part", dst)
        print(f"  got  {name} ({len(r.content):,} bytes)")


def unpack():
    import gzip
    import shutil

    gz = os.path.join(RAW, KONTUR_GPKG + ".gz")
    gpkg = os.path.join(RAW, KONTUR_GPKG)
    if not os.path.exists(gpkg):
        if not os.path.exists(gz):
            raise SystemExit(f"missing {gz}; run with --fetch first")
        with gzip.open(gz, "rb") as src, open(gpkg + ".part", "wb") as dst:
            shutil.copyfileobj(src, dst)
        os.replace(gpkg + ".part", gpkg)
    with open(gpkg, "rb") as fh:
        if fh.read(4) != b"SQLi":
            raise SystemExit(f"{gpkg} is not a GeoPackage")
    return gpkg


def _rows(ws):
    return [r for r in ws.iter_rows(values_only=True) if any(c is not None for c in r)]


def _n(v):
    return 0 if v in (None, "-", "") else int(v)


def read_census():
    """Tableaux 1.4 and 1.5 by commune, every row checked against its province and the nation.

    Returns a DataFrame, one row per 2008 commune: province key, commune key, resident, ordinary,
    collective, urban, rural."""
    import openpyxl
    import pandas as pd

    path = os.path.join(RAW, CHAP1)
    if not os.path.exists(path):
        raise SystemExit(f"missing {path}; run with --fetch")
    wb = openpyxl.load_workbook(path, data_only=True, read_only=True)

    def walk(sheet, title, cols):
        rows = _rows(wb[sheet])
        if not str(rows[0][0]).startswith(title):
            raise SystemExit(f"sheet {sheet} is not {title}: {rows[0][0]!r}")
        nat, provs, comms, prov = None, {}, [], None
        for r in rows[3:]:
            label = str(r[0])
            if not label.strip():
                continue
            indent = len(label) - len(label.lstrip(" "))
            vals = tuple(_n(r[i]) for i in cols)
            k = fold(label)
            if k == "burundi":
                nat = vals
            elif indent == 4:
                prov = k
                provs[k] = vals
            elif indent == 8:
                comms.append((prov, k, vals))
            else:
                raise SystemExit(f"{sheet}: row {label!r} at indent {indent}")
        return nat, provs, comms

    # 1.4: ordinary total (col 3), collective total (col 6), total (col 9)
    n4, p4, c4 = walk("ETA4", "Tableau 1.4", (3, 6, 9))
    # 1.5: urban (1), rural (2), total (3)
    n5, p5, c5 = walk("ETA5", "Tableau 1.5", (1, 2, 3))
    if n4 != (ORDINARY_2008, CENSUS_2008 - ORDINARY_2008, CENSUS_2008) or n5 != (URBAN_2008, CENSUS_2008 - URBAN_2008, CENSUS_2008):
        raise SystemExit(f"national rows are not the census: 1.4 {n4}, 1.5 {n5}")
    if set(p4) != set(PROVINCES) or set(p5) != set(PROVINCES):
        raise SystemExit(f"province rows differ from PROVINCES: {sorted(set(p4) ^ set(PROVINCES))}")
    for p, v in p4.items():
        if v[0] + v[1] != v[2]:
            raise SystemExit(f"1.4 {p}: ordinary + collective != total")
        if sum(x[2][2] for x in c4 if x[0] == p) != v[2] or sum(x[2][0] for x in c4 if x[0] == p) != v[0]:
            raise SystemExit(f"1.4 {p}: communes do not sum to the province")
        if p5[p][2] != v[2] or p5[p][0] + p5[p][1] != v[2]:
            raise SystemExit(f"1.5 {p}: does not agree with 1.4 or does not close")
        if sum(x[2][0] for x in c5 if x[0] == p) != p5[p][0]:
            raise SystemExit(f"1.5 {p}: urban communes do not sum to the province")
    if [(a, b) for a, b, _ in c4] != [(a, b) for a, b, _ in c5]:
        raise SystemExit("1.4 and 1.5 list different communes")
    if sum(v[2] for v in p4.values()) != CENSUS_2008:
        raise SystemExit("provinces do not sum to the nation")

    rows = []
    seen_makamba = 0
    for (p, c, v4), (_p, _c, v5) in zip(c4, c5):
        if v4[2] != v5[2]:
            raise SystemExit(f"{p}/{c}: 1.4 total {v4[2]} against 1.5 {v5[2]}")
        if p == "makamba" and c == "makamba":
            seen_makamba += 1
            if seen_makamba == 1:
                c = MAKAMBA_FIRST
        rows.append(dict(province=p, commune=COMMUNE_ALIAS.get(c, c), resident=v4[2], ordinary=v4[0],
                         collective=v4[1], urban=v5[0], rural=v5[1]))
    if seen_makamba != 2:
        raise SystemExit(f"expected Makamba twice in Makamba province, found {seen_makamba}")
    df = pd.DataFrame(rows)
    dup = df[df.duplicated(["province", "commune"], keep=False)]
    if len(dup):
        raise SystemExit(f"a commune repeats inside its province: {dup.to_dict('records')}")
    if (df["province"] != MAIRIE_KEY).sum() != N_COMMUNES_RURAL:
        raise SystemExit(f"{(df['province'] != MAIRIE_KEY).sum()} communes outside the Mairie, "
                         f"expected {N_COMMUNES_RURAL}")
    if (df.loc[df["province"] == MAIRIE_KEY, "rural"] != 0).any():
        raise SystemExit("the Mairie has rural population")
    print(f"census 2008 re-read: Tableaux 1.4 and 1.5, {len(df)} communes in 17 provinces, "
          f"{CENSUS_2008:,} residents ({ORDINARY_2008:,} in ordinary households, "
          f"{URBAN_2008:,} urban); every commune sums to its province and every province to the nation")
    return df


def main():
    import geopandas as gpd
    import pandas as pd

    import geo_checks

    if "--fetch" in sys.argv:
        fetch()
    gpkg = unpack()
    cen = read_census()

    a2 = join_communes(cen)
    units, communes = build_units(a2, cen)
    hexes = build_hexes(gpkg, a2, cen, units)

    # ---- write ----
    os.makedirs(GEO, exist_ok=True)
    keep_cols = ["geo_id", "unit", "name", "pop", "ordinary", "collective", "urban", "rural",
                 "kontur_pop", "hexes", "area_sqkm", "geometry"]
    units[keep_cols].to_file(OUT_UNITS, layer="units", driver="GPKG")
    pd.DataFrame(units[keep_cols].drop(columns="geometry")).to_csv(OUT_LOOKUP, index=False,
                                                                  encoding="utf-8")
    hexes.to_file(OUT_HEXES, layer="hexes", driver="GPKG")
    communes.to_csv(OUT_COMMUNES, index=False, encoding="utf-8")
    print(f"\nwrote {OUT_UNITS}\nwrote {OUT_LOOKUP}\nwrote {OUT_HEXES} ({len(hexes):,} hexes)\n"
          f"wrote {OUT_COMMUNES}")


# geoBoundaries' ADM2 names come from a shapefile and stop at ten characters (`bukinanyan`,
# `nyabitsind`), so a census name that starts with a ten-character geoBoundaries name matches it.
# These four differ otherwise; each pairs the only unmatched name on both sides in its province.
GB_ALIAS = {
    "kanyosha1": "kanyosha",        # Bujumbura Rural's Kanyosha; the Mairie's is a 2008 quartier
    "mpingakay": "mpinga",          # Mpinga-Kayove, Rutana; the census prints MPINGA
    "buhiga": "buhuga",             # Karuzi; the census prints BUHUGA, the commune is Buhiga
    "nyanrusang": "nyarusange",     # Gitega
}
MAIRIE_GB = {"muha", "mukaza", "ntahangwa"}     # the Mairie's three communes since 2014


def join_communes(cen):
    """Every geoBoundaries commune gets its 2008 province and census commune, one to one."""
    import geopandas as gpd

    a1 = gpd.read_file(os.path.join(RAW, "geoBoundaries-BDI-ADM1.geojson"))
    a2 = gpd.read_file(os.path.join(RAW, "geoBoundaries-BDI-ADM2.geojson"))
    if (len(a1), len(a2)) != (18, N_COMMUNES_RURAL + len(MAIRIE_GB)):
        raise SystemExit(f"geoBoundaries BDI has {len(a1)} ADM1 and {len(a2)} ADM2, expected 18 and "
                         f"{N_COMMUNES_RURAL + len(MAIRIE_GB)}")
    print(f"geoBoundaries BDI: {len(a1)} provinces (2015, Rumonge included), {len(a2)} communes, "
          f"crs={a1.crs}")
    a1["k"] = a1["shapeName"].map(fold)
    a2["k"] = a2["shapeName"].map(fold)
    if set(a1["k"]) != set(PROVINCES) | {"rumonge"}:
        raise SystemExit(f"ADM1 names are not the 17 provinces and Rumonge: {sorted(a1['k'])}")
    # the witness the name does not decide: where the commune is
    pts = a2[["k", "geometry"]].copy()
    pts["geometry"] = a2.geometry.representative_point()
    j = gpd.sjoin(pts, a1[["k", "geometry"]].rename(columns={"k": "adm1"}), how="left",
                  predicate="within")
    j = j[~j.index.duplicated(keep="first")]
    a2["adm1"] = j["adm1"].reindex(a2.index)
    if a2["adm1"].isna().any():
        raise SystemExit(f"communes in no ADM1 province: {sorted(a2.loc[a2['adm1'].isna(), 'k'])}")
    a2["province"] = a2["adm1"]
    rum = a2["adm1"] == "rumonge"
    if set(a2.loc[rum, "k"]) != set(RUMONGE):
        raise SystemExit(f"Rumonge's communes are {sorted(a2.loc[rum, 'k'])}, not {sorted(RUMONGE)}")
    a2.loc[rum, "province"] = a2.loc[rum, "k"].map(RUMONGE)
    mairie = a2["province"] == MAIRIE_KEY
    if set(a2.loc[mairie, "k"]) != MAIRIE_GB:
        raise SystemExit(f"the Mairie's communes are {sorted(a2.loc[mairie, 'k'])}")

    # census commune per geoBoundaries commune, inside the province
    a2["commune"] = None
    for prov, sub in a2[~mairie].groupby("province"):
        names = set(cen.loc[cen["province"] == prov, "commune"])
        for i, k in sub["k"].items():
            k2 = GB_ALIAS.get(k, k)
            hit = [n for n in names if n == k2] or [n for n in names if len(k2) == 10 and n.startswith(k2)]
            if len(hit) != 1:
                raise SystemExit(f"{prov}: geoBoundaries commune {k} matches {hit} in the census")
            a2.loc[i, "commune"] = hit[0]
        got = list(a2.loc[sub.index, "commune"])
        if sorted(got) != sorted(names):
            raise SystemExit(f"{prov}: census communes {sorted(names - set(got))} have no polygon, "
                             f"or one is matched twice")
    a2.loc[mairie, "commune"] = MAIRIE_KEY
    used = [k for k in GB_ALIAS if k in set(a2["k"])]
    if len(used) != len(GB_ALIAS):
        raise SystemExit(f"GB_ALIAS names not in geoBoundaries: {set(GB_ALIAS) - set(used)}")
    print(f"  {N_COMMUNES_RURAL} communes matched one to one with the census inside their 2008 "
          f"province ({len(GB_ALIAS)} by alias, the rest by name or its first ten letters); "
          f"Rumonge's five returned to Bururi and Bujumbura Rural; the Mairie's three are one unit")
    return a2


def build_units(a2, cen):
    import pandas as pd

    a2 = a2.copy()
    a2["unit"] = a2["province"].map(lambda p: PROVINCES[p][0])
    units = a2[["unit", "geometry"]].dissolve(by="unit").reset_index()
    if len(units) != N_UNITS:
        raise SystemExit(f"{len(units)} units after the dissolve, expected {N_UNITS}")
    units["area_sqkm"] = units.to_crs("EPSG:6933").area / 1e6
    prov = cen.groupby("province")[["resident", "ordinary", "collective", "urban", "rural"]].sum()
    prov.index = prov.index.map(lambda p: PROVINCES[p][0])
    units = units.join(prov, on="unit")
    units = units.rename(columns={"resident": "pop"})
    units["geo_id"] = units["unit"]
    units["name"] = units["unit"].map({v[0]: v[1] for v in PROVINCES.values()})
    if int(units["pop"].sum()) != CENSUS_2008:
        raise SystemExit("units do not sum to the census")
    communes = cen.assign(unit=cen["province"].map(lambda p: PROVINCES[p][0]))
    communes = communes[["unit", "province", "commune", "resident", "ordinary", "urban", "rural"]]
    return units, communes


def build_hexes(gpkg, a2, cen, units):
    """Kontur hexes, each in a commune, weighted to the commune's 2008 count."""
    import geopandas as gpd
    import numpy as np
    import pandas as pd

    import geo_checks

    hexes = geo_checks.read_layer(gpkg, "Kontur BI")
    popcol = next(c for c in hexes.columns if c.lower() == "population")
    hexes = hexes.to_crs(a2.crs)
    pts = gpd.GeoDataFrame({"pop": hexes[popcol].to_numpy(dtype=float)},
                           geometry=hexes.to_crs("EPSG:32735").geometry.centroid,
                           crs="EPSG:32735").to_crs(a2.crs)
    cells = a2[["commune", "province", "geometry"]].copy()
    j = gpd.sjoin(pts, cells, how="left", predicate="within")
    j = j[~j.index.duplicated(keep="first")].reindex(pts.index)
    outside = j["commune"].isna()
    print(f"\nKontur BI: {len(hexes):,} hexes, {pts['pop'].sum():,.0f} people; {int(outside.sum())} "
          f"centroids outside every commune ({pts.loc[outside, 'pop'].sum():,.0f} people)")
    commune, province = j["commune"].copy(), j["province"].copy()
    if outside.any():
        ne = gpd.read_file(os.path.join(ROOT, "data", "geo", "ne_10m_admin_0_countries.geojson"))
        others = ne[ne["ADM0_A3"] != "BDI"][["ADM0_A3", "geometry"]].to_crs(a2.crs)
        sub = pts[outside]
        hit = gpd.sjoin(sub, others, how="left", predicate="within")
        hit = hit[~hit.index.duplicated(keep="first")].reindex(sub.index)
        foreign = hit["ADM0_A3"].notna()
        metric = "EPSG:32735"
        nn = gpd.sjoin_nearest(sub.to_crs(metric), cells.to_crs(metric), distance_col="d")
        nn = nn[~nn.index.duplicated(keep="first")].reindex(sub.index)
        by = hit.loc[foreign].groupby("ADM0_A3")["pop"].sum().round()
        print(f"    in a neighbour by Natural Earth (dropped): {int(foreign.sum())} hexes, "
              f"{sub.loc[foreign, 'pop'].sum():,.0f} people ({', '.join(f'{k} {v:,.0f}' for k, v in by.items())})")
        snap = ~foreign
        far = snap & (nn["d"] > SNAP_KM * 1000)
        if far.any():
            raise SystemExit(f"{int(far.sum())} hexes in no country lie over {SNAP_KM} km from a "
                             f"commune ({sub.loc[far, 'pop'].sum():,.0f} people)")
        print(f"    in no country (the lake shore; snapped to the nearest commune within {SNAP_KM} "
              f"km): {int(snap.sum())} hexes, {sub.loc[snap, 'pop'].sum():,.0f} people")
        commune.loc[sub.index[snap.to_numpy()]] = nn.loc[snap, "commune"]
        province.loc[sub.index[snap.to_numpy()]] = nn.loc[snap, "province"]
    keep = commune.notna()

    def ck(p, c):
        return MAIRIE_KEY if p == MAIRIE_KEY else f"{p}/{c}"

    # Each hex is cut to Burundi's outline (the union of the communes), so a dot cannot land in
    # Lake Tanganyika (the lake-shore hexes snapped above are mostly water) or across the border.
    # Not to its commune: a hex whose centroid is just inside a commune line would keep its whole
    # weight on a sliver, and scatter.py read 113,785 people/km2 on the first such build. A snapped
    # hex with no land at all is dropped before the calibration, so its people go to the commune's
    # other hexes.
    import shapely
    land = shapely.union_all(a2.geometry.to_numpy())
    geoms = hexes.geometry.to_numpy()
    clipped = np.array([None] * len(geoms), dtype=object)
    clipped[keep.to_numpy()] = shapely.intersection(geoms[keep.to_numpy()], land)
    area0 = shapely.area(geoms[keep.to_numpy()]).sum()
    # A hex left with under MIN_LAND_FRACTION of its area is dropped too: slivers, mostly hexes across
    # the Rusizi and the Rwandan border that Natural Earth's coarser line left in no country, read up
    # to 113,785 people/km2 before. Built 2026-10-03: 71 hexes, 18,634 Kontur people, dropped.
    frac = np.array([0.0 if g is None else shapely.area(g) for g in clipped]) / \
        np.where(keep.to_numpy(), shapely.area(geoms), 1)
    empty = keep.to_numpy() & (frac < MIN_LAND_FRACTION)
    print(f"  hexes cut to the national outline: {100 * (1 - shapely.area(clipped[keep.to_numpy() & ~empty]).sum() / area0):.2f}% "
          f"of hex area removed (lake and borders); {int(empty.sum())} hexes with under "
          f"{MIN_LAND_FRACTION:.0%} of their area on land dropped ({pts.loc[empty, 'pop'].sum():,.0f} "
          f"Kontur people)")
    if empty.sum() > MAX_EMPTY_AFTER_CUT:
        raise SystemExit("more hexes with no land after the cut than the first build saw")
    keep = keep & ~pd.Series(empty, index=keep.index)

    # calibrate each commune (the Mairie as one) to its 2008 count
    target = cen.assign(k=[ck(p, c) for p, c in zip(cen["province"], cen["commune"])]) \
        .groupby("k")["resident"].sum()
    key = [ck(p, c) for p, c in zip(province[keep], commune[keep])]
    kp = pts.loc[keep, "pop"].groupby(key).sum()
    missing = sorted(set(target.index) - set(kp.index), key=str)
    if missing:
        raise SystemExit(f"communes with no populated hex: {missing}")
    ratio = kp / target.reindex(kp.index)
    national = kp.sum() / target.sum()
    print(f"\n  Kontur 2023 against the 2008 census, per commune: national {national:.3f}x; "
          f"p10 {ratio.quantile(.1):.2f}, median {ratio.median():.2f}, p90 {ratio.quantile(.9):.2f}")
    for k_, r in ratio.sort_values().iloc[[0, 1, 2, -3, -2, -1]].items():
        print(f"    {str(k_):<40}{r:5.2f}x  (2008 {int(target[k_]):,})")
    lo, hi = KONTUR_COMMUNE_BAND
    bad = ratio[(ratio < lo) | (ratio > hi)]
    if len(bad):
        raise SystemExit(f"communes outside the Kontur band {KONTUR_COMMUNE_BAND}: {bad.round(2).to_dict()}")
    scale = (target.reindex(kp.index) / kp)
    w = pts.loc[keep, "pop"].to_numpy() * np.array([scale[k_] for k_ in key])
    unit = [PROVINCES[p][0] for p in province[keep]]
    out = gpd.GeoDataFrame({"unit": unit, "commune": [c for c in commune[keep]],
                            "kontur_pop": pts.loc[keep, "pop"].to_numpy(), "pop": w},
                           geometry=list(clipped[keep.to_numpy()]), crs=a2.crs)
    per = out.groupby("unit").agg(hexes=("pop", "size"), pop=("pop", "sum"), kontur=("kontur_pop", "sum"))
    want = units.set_index("unit")["pop"]
    if (abs(per["pop"] - want.reindex(per.index)) > 1).any():
        raise SystemExit("calibrated weights do not sum to each unit's 2008 count")
    units["kontur_pop"] = units["unit"].map(per["kontur"]).round().astype("int64")
    units["hexes"] = units["unit"].map(per["hexes"]).astype("int64")
    print("\n  per unit: 2008 census, Kontur 2023 (the witness before calibration), urban share:")
    for r in units.sort_values("pop", ascending=False).itertuples():
        print(f"    {r.unit:<6}{r.name:<18}{r.pop:>10,}  Kontur {r.kontur_pop:>10,} "
              f"{r.kontur_pop / r.pop:5.2f}x  urban {100 * r.urban / r.pop:5.1f}%  {r.hexes:>6,} hexes "
              f"{r.area_sqkm:>7,.0f} km2")
    dense = out.assign(d=out["pop"] / 0.737).sort_values("d").iloc[-1]
    print(f"  densest calibrated hex: {dense['d']:,.0f} people/km2 in {dense['commune']}")
    return out


SNAP_KM = 3
MIN_LAND_FRACTION = 0.2
MAX_EMPTY_AFTER_CUT = 100       # first build, 2026-10-03: 71 (2 with no land at all)


if __name__ == "__main__":
    main()
