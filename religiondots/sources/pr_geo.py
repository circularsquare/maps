"""Puerto Rico — the six World Values Survey regions, built from the 78 municipios.

Writes data/geo/pr/pr_regions.gpkg, data/geo/pr/pr_lookup.csv and
data/geo/pr/pr_municipios.csv. `sources/pr.md` is this country's record.

## THE REGIONS ARE THE SURVEY TEAM'S OWN, AND THE ONLY DEFINITION IS A MAP IN THEIR REPORT

The WVS file codes six regions in `N_REGION_WVS` (codebook annex: 630001 Norte, 630002 Sur,
630003 Oeste, 630004 Este, 630006 Centro, 630007 Metropolitana; no 630005). No list of the
municipios inside them is published anywhere found. The survey team's report, *Encuesta Mundial
de Valores para Puerto Rico 2018* (Instituto de Estadísticas de Puerto Rico, 17 June 2019,
`estadisticas.pr/files/Publicaciones/Encuesta_Mundial_de_Valores_para_Puerto Rico_20190617.pdf`),
prints on p.16 a map of all 78 municipios coloured by region with the three sampled municipios
and their interview counts in each. `REGION_OF` below is that map, read municipio by municipio.

**Vieques and Culebra are not on the map.** They are assigned to Este, the region of Fajardo and
Ceiba, which is where their ferries run from; together 8,506 people in 2024. That is this build's
call, not the survey's, and `sources/pr.md` says so.

The map is checked three ways that do not use it: the 18 sampled municipios' regions in the
microdata (`sources/pr.py`), the report's own interview counts per region (asserted there too),
and here, geography: Oeste must hold the westernmost municipio centroid, Este the easternmost
one on the main island, Norte must sit north of Sur, and Metropolitana must be the densest.

## THE POPULATION IS THE CENSUS BUREAU'S, TWO FILES FOR TWO JOBS

* **Vintage 2024 estimates** (`PRM-EST2024-POP`, July 1 2024) are the drawn population, the
  newest official figure per municipio.
* **The 2020 census redistricting file** (`pr2020.pl.zip`, P1 total and P3 18 and over) gives
  adults, which is what the survey sampled in proportion to (the report, p.16), for the
  held-out check in `sources/pr.py`. Its total is asserted against the estimates base.

Usage:
    python sources/pr_geo.py --fetch    two Census Bureau files (15 MB); the county boundaries
                                        are the shared data/geo/cb_2020_us_county_500k.zip
    python sources/pr_geo.py            rebuild from data/raw/pr/
"""

import io
import os
import sys
import unicodedata
import urllib.request
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import geopandas as gpd
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "pr")
OUT_DIR = os.path.join(ROOT, "data", "geo", "pr")
OUT = os.path.join(OUT_DIR, "pr_regions.gpkg")
LOOKUP = os.path.join(OUT_DIR, "pr_lookup.csv")
MUNIS = os.path.join(OUT_DIR, "pr_municipios.csv")
COUNTIES = os.path.join(ROOT, "data", "geo", "cb_2020_us_county_500k.zip")
PL_ZIP = os.path.join(RAW, "pr2020.pl.zip")
PEP_XLSX = os.path.join(RAW, "prm-est2024-pop.xlsx")

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) religiondots/1.0"}
DOWNLOADS = {
    PL_ZIP: "https://www2.census.gov/programs-surveys/decennial/2020/data/"
            "01-Redistricting_File--PL_94-171/Puerto_Rico/pr2020.pl.zip",
    PEP_XLSX: "https://www2.census.gov/programs-surveys/popest/tables/2020-2024/municipios/"
              "totals/prm-est2024-pop.xlsx",
}

N_MUNI = 78
PR_2020 = 3_285_874          # 2020 census, P1; also the estimates base
PR_2020_ADULTS = 2_724_903   # 2020 census, P3
PR_2024 = 3_203_295          # Vintage 2024, July 1 2024

# N_REGION_WVS, verbatim from the WVS-7 codebook annex (Region country-specific).
REGIONS = {
    "630001": "Norte", "630002": "Sur", "630003": "Oeste", "630004": "Este",
    "630006": "Centro", "630007": "Metropolitana",
}
N, S, O, E, C, M = "630001", "630002", "630003", "630004", "630006", "630007"

# The report's p.16 map, read municipio by municipio. Vieques and Culebra are not drawn on it
# and are this build's assignment (see the docstring).
REGION_OF = {
    # Oeste (16)
    "Aguadilla": O, "Isabela": O, "Aguada": O, "Rincón": O, "Moca": O, "San Sebastián": O,
    "Añasco": O, "Las Marías": O, "Mayagüez": O, "Maricao": O, "Hormigueros": O,
    "San Germán": O, "Cabo Rojo": O, "Lajas": O, "Sabana Grande": O, "Guánica": O,
    # Norte (12)
    "Quebradillas": N, "Camuy": N, "Hatillo": N, "Arecibo": N, "Barceloneta": N, "Florida": N,
    "Manatí": N, "Vega Baja": N, "Vega Alta": N, "Dorado": N, "Toa Baja": N, "Toa Alta": N,
    # Centro (14)
    "Lares": C, "Utuado": C, "Adjuntas": C, "Jayuya": C, "Ciales": C, "Morovis": C,
    "Orocovis": C, "Corozal": C, "Naranjito": C, "Barranquitas": C, "Comerío": C,
    "Aguas Buenas": C, "Aibonito": C, "Cayey": C,
    # Sur (12)
    "Yauco": S, "Guayanilla": S, "Peñuelas": S, "Ponce": S, "Villalba": S, "Juana Díaz": S,
    "Coamo": S, "Santa Isabel": S, "Salinas": S, "Guayama": S, "Arroyo": S, "Patillas": S,
    # Metropolitana (6)
    "Cataño": M, "Bayamón": M, "Guaynabo": M, "San Juan": M, "Trujillo Alto": M, "Carolina": M,
    # Este (16 on the map, plus the two islands)
    "Loíza": E, "Canóvanas": E, "Río Grande": E, "Luquillo": E, "Fajardo": E, "Ceiba": E,
    "Naguabo": E, "Humacao": E, "Las Piedras": E, "Juncos": E, "Gurabo": E, "Caguas": E,
    "San Lorenzo": E, "Yabucoa": E, "Maunabo": E, "Cidra": E,
    "Vieques": E, "Culebra": E,
}
NOT_ON_MAP = {"Vieques", "Culebra"}
MAP_COUNTS = {O: 16, N: 12, C: 14, S: 12, M: 6, E: 16}

# N_REGION_ISO, verbatim from the codebook annex: the 18 sampled municipios. The code is
# 630000 plus the municipio's place in the alphabetical list of 78, which `main()` asserts.
WVS_MUNICIPIO = {
    630009: "Barceloneta", 630015: "Canóvanas", 630017: "Cataño", 630018: "Cayey",
    630024: "Corozal", 630030: "Guayama", 630035: "Hormigueros", 630040: "Juncos",
    630051: "Moca", 630054: "Naranjito", 630057: "Peñuelas", 630061: "Río Grande",
    630064: "San Germán", 630065: "San Juan", 630070: "Toa Baja", 630071: "Trujillo Alto",
    630074: "Vega Baja", 630078: "Yauco",
}


def fold(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return "".join(ch for ch in s.lower() if ch.isalnum())


def fetch():
    os.makedirs(RAW, exist_ok=True)
    for dst, url in DOWNLOADS.items():
        if os.path.exists(dst) and os.path.getsize(dst) > 10_000:
            print(f"  have {os.path.basename(dst)} ({os.path.getsize(dst):,} bytes)")
            continue
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=900) as r, open(dst + ".part", "wb") as f:
            f.write(r.read())
        os.replace(dst + ".part", dst)
        print(f"  got  {os.path.basename(dst)} ({os.path.getsize(dst):,} bytes)")


def read_pl():
    """2020 census P1 (total) and P3 (18 and over) per municipio, off the redistricting file."""
    if not os.path.exists(PL_ZIP):
        raise SystemExit(f"{PL_ZIP} missing, run with --fetch")
    with zipfile.ZipFile(PL_ZIP) as z:
        geo = [ln.split("|") for ln in z.read("prgeo2020.pl").decode("latin-1").splitlines()]
        seg1 = {p[4]: p for p in (ln.split("|") for ln in
                                  z.read("pr000012020.pl").decode("latin-1").splitlines())}
        seg2 = {p[4]: p for p in (ln.split("|") for ln in
                                  z.read("pr000022020.pl").decode("latin-1").splitlines())}
    state = [p for p in geo if p[2] == "040" and p[4] == "00"]
    cty = [p for p in geo if p[2] == "050" and p[4] == "00"]
    if len(state) != 1 or len(cty) != N_MUNI:
        raise SystemExit(f"redistricting geo file: {len(state)} state rows, {len(cty)} municipio "
                         f"rows, expected 1 and {N_MUNI}")
    name_i = next(i for i, v in enumerate(cty[0]) if v.endswith(" Municipio"))
    if any(not p[name_i].endswith(" Municipio") for p in cty):
        raise SystemExit("the NAME field is not in the same column on every municipio row")
    rows = []
    for p in cty:
        lr = p[7]
        rows.append({"geoid": p[9], "pl_name": p[name_i][: -len(" Municipio")],
                     "pop_2020": int(seg1[lr][5]), "adults_2020": int(seg2[lr][5])})
    df = pd.DataFrame(rows)
    lr = state[0][7]
    st_total, st_adult = int(seg1[lr][5]), int(seg2[lr][5])
    if (st_total, st_adult) != (PR_2020, PR_2020_ADULTS):
        raise SystemExit(f"state row P1/P3 {st_total:,}/{st_adult:,}, expected "
                         f"{PR_2020:,}/{PR_2020_ADULTS:,}")
    if (int(df["pop_2020"].sum()), int(df["adults_2020"].sum())) != (st_total, st_adult):
        raise SystemExit("municipio P1 or P3 does not sum to the state row")
    print(f"  2020 census: {N_MUNI} municipios, P1 {PR_2020:,}, P3 (18+) {PR_2020_ADULTS:,}, "
          "both summing to the state row")
    return df


def read_pep():
    """Vintage 2024 estimates per municipio."""
    if not os.path.exists(PEP_XLSX):
        raise SystemExit(f"{PEP_XLSX} missing, run with --fetch")
    x = pd.read_excel(PEP_XLSX, header=None)
    head = x.iloc[4]
    if str(head[0]).strip() != "Puerto Rico" or int(head[6]) != PR_2024 or int(head[1]) != PR_2020:
        raise SystemExit(f"PRM-EST2024-POP's Puerto Rico row reads {list(head)}")
    body = x.iloc[5:5 + N_MUNI]
    if not body[0].astype(str).str.endswith(" Municipio, Puerto Rico").all():
        raise SystemExit("PRM-EST2024-POP's municipio rows are not where they were")
    df = pd.DataFrame({
        "pep_name": body[0].astype(str).str.lstrip(".").str.replace(" Municipio, Puerto Rico", "",
                                                                     regex=False),
        "base_2020": body[1].astype("int64").to_numpy(),
        "pop_2024": body[6].astype("int64").to_numpy(),
    })
    if int(df["pop_2024"].sum()) != PR_2024 or int(df["base_2020"].sum()) != PR_2020:
        raise SystemExit("PRM-EST2024-POP's municipios do not sum to its Puerto Rico row")
    print(f"  Vintage 2024: {N_MUNI} municipios summing to {PR_2024:,} (July 1 2024)")
    return df


def main():
    if "--fetch" in sys.argv:
        fetch()

    pl = read_pl()
    pep = read_pep()

    # ---- names: the estimates table, the redistricting file and REGION_OF must agree 78 of 78
    pep["key"] = pep["pep_name"].map(fold)
    pl["key"] = pl["pl_name"].map(fold)
    reg = {fold(k): v for k, v in REGION_OF.items()}
    if len(reg) != N_MUNI or len(set(pep["key"])) != N_MUNI or len(set(pl["key"])) != N_MUNI:
        raise SystemExit("municipio names are not 78 unique keys in all three lists")
    if set(pep["key"]) != set(pl["key"]) or set(pep["key"]) != set(reg):
        raise SystemExit(f"name sets differ: estimates-only {set(pep['key']) - set(pl['key'])}, "
                         f"census-only {set(pl['key']) - set(pep['key'])}, "
                         f"map-only {set(reg) - set(pep['key'])}")
    got = pd.Series(REGION_OF).value_counts().to_dict()
    want = {k: v + (2 if k == E else 0) for k, v in MAP_COUNTS.items()}
    if got != want:
        raise SystemExit(f"REGION_OF has {got} municipios per region, the map has {want}")
    m = pl.merge(pep, on="key", validate="one_to_one")
    m["region"] = m["key"].map(reg)
    m["name"] = m["key"].map({fold(k): k for k in REGION_OF})
    drift = (m["base_2020"] - m["pop_2020"]).abs()
    print(f"  names: estimates, census and the report's map agree on all {N_MUNI}; estimates "
          f"base against census count differs by up to {int(drift.max())} people "
          f"({m.loc[drift.idxmax(), 'name']})")

    # ---- the WVS municipio code is 630000 + alphabetical place, which ties the codebook's
    # names to these rows independently of spelling
    order = sorted(m["name"], key=lambda n: fold(n))
    for code, nm in WVS_MUNICIPIO.items():
        if order.index(nm) + 1 != code - 630000:
            raise SystemExit(f"{nm} is number {order.index(nm) + 1} alphabetically, its WVS code "
                             f"says {code - 630000}")
    print(f"  the 18 WVS municipio codes are each 630000 + alphabetical place among the 78")

    # ---- polygons: the 2020 cartographic boundary counties, state 72
    if not os.path.exists(COUNTIES):
        raise SystemExit(f"{COUNTIES} missing (shared with the US build)")
    g = gpd.read_file(f"zip://{COUNTIES}")
    g = g[g["STATEFP"] == "72"].copy()
    if len(g) != N_MUNI:
        raise SystemExit(f"{len(g)} Puerto Rico features in the county file, expected {N_MUNI}")
    g = g.to_crs(4326)
    g = g.merge(m[["geoid", "name", "region", "pop_2024", "pop_2020", "adults_2020"]],
                left_on="GEOID", right_on="geoid", how="left", validate="one_to_one")
    if g["name"].isna().any():
        raise SystemExit(f"county GEOIDs with no census row: {sorted(g.loc[g['name'].isna(), 'GEOID'])}")
    minx, miny, maxx, maxy = g.total_bounds
    # Mayagüez takes in Mona Island at 67.9 W, so the west edge is well past the main island's.
    if not (-68.1 < minx < -67.0 and -65.4 < maxx < -65.1 and 17.7 < miny < 18.0
            and 18.4 < maxy < 18.6):
        raise SystemExit(f"Puerto Rico's bounding box is {g.total_bounds}, which is not the island")
    print(f"  county file: {N_MUNI} municipios joined on GEOID, bbox "
          f"{minx:.2f},{miny:.2f},{maxx:.2f},{maxy:.2f}")

    # ---- witnesses that use neither the names nor the codes
    cen = g.to_crs(32161).geometry.centroid.to_crs(4326)
    g["lon"], g["lat"] = cen.x, cen.y
    main_island = g[~g["name"].isin(NOT_ON_MAP)]
    west = main_island.loc[main_island["lon"].idxmin()]
    east = main_island.loc[main_island["lon"].idxmax()]
    if west["region"] != O or east["region"] != E:
        raise SystemExit(f"westernmost {west['name']} is in {west['region']}, easternmost "
                         f"{east['name']} in {east['region']}; the region map is misread")
    r = g.dissolve(by="region", aggfunc={"pop_2024": "sum", "pop_2020": "sum",
                                         "adults_2020": "sum", "ALAND": "sum"}).reset_index()
    rc = r.to_crs(32161).geometry.centroid.to_crs(4326)
    r["lat"] = rc.y
    lat = dict(zip(r["region"], r["lat"]))
    if not lat[N] > lat[S]:
        raise SystemExit("Norte's centroid is not north of Sur's")
    r["density"] = r["pop_2024"] / (r["ALAND"] / 1e6)
    if r.loc[r["density"].idxmax(), "region"] != M:
        raise SystemExit("Metropolitana is not the densest region")
    print(f"  witnesses: {west['name']} (westernmost) in Oeste, {east['name']} (easternmost on the "
          f"main island) in Este, Norte north of Sur, Metropolitana densest "
          f"({r['density'].max():,.0f}/km2)")

    r["unit"] = r["region"]
    r["geo_id"] = r["region"]
    r["name"] = r["region"].map(REGIONS)
    r["pop"] = r["pop_2024"].astype("int64")
    os.makedirs(OUT_DIR, exist_ok=True)
    r[["unit", "name", "geo_id", "pop", "geometry"]].to_file(OUT, layer="regions", driver="GPKG")
    print(f"\nwrote {OUT} ({len(r)} regions)")

    n_muni = g.groupby("region").size()
    lut = pd.DataFrame({
        "geo_id": r["region"], "unit": r["region"], "name": r["name"],
        "pop_2024": r["pop_2024"].astype("int64"), "pop_2020": r["pop_2020"].astype("int64"),
        "adults_2020": r["adults_2020"].astype("int64"),
        "n_municipios": r["region"].map(n_muni).astype(int),
        "area_sqkm": (r["ALAND"] / 1e6).round(1),
    }).sort_values("geo_id")
    lut.to_csv(LOOKUP, index=False, encoding="utf-8")
    for _, row in lut.iterrows():
        print(f"    {row['geo_id']} {row['name']:<14} {row['n_municipios']:>3} municipios  "
              f"{row['pop_2024']:>9,} (2024)  {row['adults_2020']:>9,} adults (2020)")
    print(f"wrote {LOOKUP}")

    wvs_of = {nm: code for code, nm in WVS_MUNICIPIO.items()}
    mm = g.drop(columns="geometry").sort_values("GEOID")
    pd.DataFrame({
        "geoid": mm["GEOID"], "name": mm["name"], "region": mm["region"],
        "region_name": mm["region"].map(REGIONS),
        "wvs_code": mm["name"].map(wvs_of).astype("Int64"),
        "on_report_map": ~mm["name"].isin(NOT_ON_MAP),
        "pop_2024": mm["pop_2024"], "pop_2020": mm["pop_2020"], "adults_2020": mm["adults_2020"],
    }).to_csv(MUNIS, index=False, encoding="utf-8")
    print(f"wrote {MUNIS} ({N_MUNI} municipios)")


if __name__ == "__main__":
    main()
