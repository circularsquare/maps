"""Kiribati — boundaries for the 24 inhabited islands.

Writes data/geo/ki/ki_islands.gpkg and data/geo/ki/ki_lookup.csv.

OCHA COD-AB Kiribati (`cod-ab-kir`, 2020), the **ADM2 island** shapefile, on §12's Chile rule.
36 polygons for the country's 33 islands and atolls plus its separately-councilled parts, and
the census's 24 inhabited islands join **24/24 on the name**, with five contractions the report
uses for compass-point pairs.

**THE TWELVE THAT DO NOT JOIN ARE UNINHABITED AND THAT IS THE RIGHT ANSWER**: Flint, Malden,
Millenium (Caroline), Starburk (Starbuck) and Vostok in the Line Islands, and Birnie,
Enderbury, Mackean, Manra, Nikumaroro, Orona and Rawaki in the Phoenix group. Kanton is the
only inhabited Phoenix island and the census gives it 20 people.

**KIRIBATI STRADDLES THE ANTIMERIDIAN AND THE USUAL CHECK IS WRONG HERE.** Every other country
on this map gets a bounding-box width assertion, because a 180-crossing polygon reprojected
without care comes out spanning most of the globe ([[reference_antimeridian]]). Kiribati's own
bounding box **legitimately spans 351 degrees**: the Gilberts sit at 173 E and Kiritimati at
157 W, 4,000 km apart, so a country-level width check fires on correct data and would have to
be switched off — which is exactly how a real tear gets missed later.

**So the check is per-polygon instead.** No individual island may span more than
`MAX_ISLAND_SPAN_DEG`; every atoll here is a few kilometres across, so a torn one is
unmissable, and the country is allowed to be as wide as it really is. Fiji (§9bd) needed the
opposite treatment because its *provinces* genuinely cross 180; none of Kiribati's islands
does.

Usage:
    python sources/ki_geo.py --fetch    one ~6 MB zip from HDX
    python sources/ki_geo.py            rebuild from data/raw/ki/
"""

import csv
import os
import re
import sys
import unicodedata

os.environ.setdefault("OMP_NUM_THREADS", "6")          # [[feedback_cap_cpu]]
os.environ.setdefault("OPENBLAS_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "ki")
OUT_DIR = os.path.join(ROOT, "data", "geo", "ki")
OUT = os.path.join(OUT_DIR, "ki_islands.gpkg")
LOOKUP = os.path.join(OUT_DIR, "ki_lookup.csv")
NORM = os.path.join(ROOT, "data", "normalized", "ki.csv")

ZIP_URL = ("https://data.humdata.org/dataset/156a0966-1e09-455c-9e4f-e68676aa2fbf/resource/"
           "30c02bf1-d705-46ad-b828-b4b9bbed46e6/download/kir_adm_2020_shp.zip")
ZIP_NAME = "kir_adm_2020_SHP.zip"
SHP = "kir_admbnda_adm2_2020.shp"

EXPECTED_POLYGONS = 36
EXPECTED_UNITS = 24

# Per-polygon, not per-country: see the module docstring. The widest real island here is
# Kiritimati at about 0.7 degrees.
MAX_ISLAND_SPAN_DEG = 3.0
# The country legitimately reaches from Banaba (169.5 E) to Millenium (150.2 W).
EXPECTED_LAT = (-12.5, 5.5)

# The report contracts five names for the compass-point pairs; everything else joins verbatim.
NAME_ALIAS = {
    "ntarawa": "north tarawa",
    "starawa": "south tarawa",
    "ntabiteuea": "north tabiteuea",
    "stabiteuea": "south tabiteuea",
    "teeraina": "teraina",
}


def fold(s):
    """Casefold, strip accents and punctuation — for COMPARING names, never for storing."""
    s = unicodedata.normalize("NFKD", str(s).strip().lower())
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def key(name):
    k = fold(name).replace(" ", "")
    return NAME_ALIAS.get(k, fold(name)).replace(" ", "")


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    dest = os.path.join(RAW, ZIP_NAME)
    if os.path.exists(dest) and os.path.getsize(dest) > 1_000_000:
        print("already have", dest)
        return
    print("GET", ZIP_URL)
    r = requests.get(ZIP_URL, timeout=1800, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    if not r.content.startswith(b"PK"):
        raise SystemExit(f"HDX returned something that is not a zip ({len(r.content):,} bytes)")
    tmp = dest + ".part"                                  # [[reference_wb_truncates]]
    with open(tmp, "wb") as fh:
        fh.write(r.content)
    os.replace(tmp, dest)
    print(f"  {os.path.getsize(dest):,} bytes")


def build():
    import geopandas as gpd
    import pandas as pd

    src = os.path.join(RAW, ZIP_NAME)
    if not os.path.exists(src):
        raise SystemExit(f"{src} is missing — run `python sources/ki_geo.py --fetch`")
    g = gpd.read_file("zip://" + src + "!" + SHP)
    if len(g) != EXPECTED_POLYGONS:
        raise SystemExit(f"{len(g)} ADM2 polygons, expected {EXPECTED_POLYGONS} — "
                         "COD reissued the layer")
    g = g.to_crs(4326)

    # --- the antimeridian check, per polygon. See the docstring for why not per country.
    spans = [(r["ADM2_EN"], r.geometry.bounds[2] - r.geometry.bounds[0]) for _, r in g.iterrows()]
    torn = [(n, s) for n, s in spans if s > MAX_ISLAND_SPAN_DEG]
    if torn:
        raise SystemExit(f"{len(torn)} island(s) span more than {MAX_ISLAND_SPAN_DEG} degrees "
                         f"of longitude and are torn across 180: {torn} "
                         "[[reference_antimeridian]]")
    miny, maxy = g.total_bounds[1], g.total_bounds[3]
    if not (EXPECTED_LAT[0] <= miny and maxy <= EXPECTED_LAT[1]):
        raise SystemExit(f"latitude range {miny:.2f}..{maxy:.2f} is not Kiribati")
    print(f"  {len(g)} polygons, widest island {max(s for _, s in spans):.2f}° — none torn")
    print(f"  the country itself spans {g.total_bounds[2] - g.total_bounds[0]:.0f}° of "
          "longitude, which is correct: it straddles the antimeridian")

    # --- the census islands.
    if not os.path.exists(NORM):
        raise SystemExit(f"{NORM} is missing — run `python sources/ki.py` first")
    df = pd.read_csv(NORM, dtype=str, keep_default_na=False)
    df["count"] = df["count"].astype(int)
    islands = df[["geo_id", "geo_name"]].drop_duplicates()
    if len(islands) != EXPECTED_UNITS:
        raise SystemExit(f"{len(islands)} islands in ki.csv, expected {EXPECTED_UNITS}")

    by = {}
    for idx, r in g.iterrows():
        by.setdefault(key(r["ADM2_EN"]), []).append((idx, r["ADM2_PCODE"], r["ADM1_EN"]))
    dup = {k: v for k, v in by.items() if len(v) > 1}
    if dup:
        raise SystemExit(f"COD has repeated island names: {dup}")

    rows, claimed = [], {}
    for _, r in islands.iterrows():
        k = key(r["geo_name"])
        hit = by.get(k)
        if not hit:
            raise SystemExit(f"no COD polygon for census island {r['geo_name']!r} (key {k!r}) "
                             "— add it to NAME_ALIAS")
        idx, pcode, div = hit[0]
        if pcode in claimed:
            raise SystemExit(f"{pcode} claimed twice")
        claimed[pcode] = r["geo_id"]
        rows.append((idx, r["geo_id"], r["geo_name"], pcode, div))
    print(f"  joined {len(rows)}/{EXPECTED_UNITS} on the name "
          f"({sum(1 for _, _, n, _, _ in rows if key(n) != fold(n).replace(' ', ''))} "
          "via a contraction)")

    out = g.loc[[i for i, *_ in rows]].copy()
    out["unit"] = [u for _, u, *_ in rows]
    out["name"] = [n for _, _, n, _, _ in rows]
    out = out[["unit", "name", "ADM2_PCODE", "ADM1_EN", "geometry"]]

    os.makedirs(OUT_DIR, exist_ok=True)
    tmp = OUT[:-5] + ".part.gpkg"
    if os.path.exists(tmp):
        os.remove(tmp)
    out.to_file(tmp, driver="GPKG", layer="islands")
    os.replace(tmp, OUT)

    with open(LOOKUP + ".part", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["geo_id", "unit", "island", "adm2_pcode", "division"])
        for _, u, n, pcode, div in rows:
            w.writerow([u, u, n, pcode, div])
    os.replace(LOOKUP + ".part", LOOKUP)

    per = df.groupby("geo_id")["count"].sum()
    div_of = {u: d for _, u, _, _, d in rows}
    name_of = {u: n for _, u, n, _, _ in rows}
    print(f"\n    {'island':<18} {'division':<17} {'people':>8}")
    for u in sorted(per.index, key=lambda u: -per[u]):
        print(f"    {name_of[u]:<18} {div_of[u]:<17} {per[u]:>8,}")
    print(f"    {'':<18} {'':<17} {per.sum():>8,}")

    unclaimed = [f"{r['ADM2_PCODE']} {r['ADM1_EN']}/{r['ADM2_EN']}"
                 for _, r in g.iterrows() if r["ADM2_PCODE"] not in claimed]
    print(f"\nwrote {OUT} ({len(out)} units)")
    print(f"wrote {LOOKUP}")
    print(f"  {len(unclaimed)} COD polygons have no census row, all uninhabited:")
    for u in unclaimed:
        print("     ", u)


if __name__ == "__main__":
    if "--fetch" in sys.argv:
        fetch()
    else:
        build()
