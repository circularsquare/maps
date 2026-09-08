"""Paraguay — boundaries for the 224 districts the 2002 census counted.

Writes data/geo/py/py_distritos.gpkg and data/geo/py/py_lookup.csv.

OCHA COD-AB Paraguay (`cod-ab-pry`), the **DGEEC 2020 shapefile** bundle. DGEEC is the office
that ran the 2002 census and is INE's former name, so `ADM2_PCODE` is `PY` plus the census's
own four-digit `DDdd` district code and the join is free: **223 of the census's codes appear
in the boundary file unchanged.**

**THE VINTAGE GAP IS THE WHOLE OF THE WORK HERE, AND IT GOES BOTH WAYS.**

    census 2002 has 229 districts        COD-AB 2020 has 250
      6 of them are ASUNCION's barrios     26 of them were created after 2002
      and are ONE polygon in COD-AB        and are inside a 2002 district's extent

**ASUNCION IS AGGREGATED, 6 into 1.** The census tabulates the capital as six districts (La
Encarnacion, Catedral, San Roque, Lambare, Recoleta, Santisima Trinidad); every boundary
source treats Asuncion as a single unit. Their counts are summed onto `0000`. That is a real
loss of resolution over 512,000 people and it is stated in `countries.py`'s `grain`.

**THE 26 POST-2002 DISTRICTS ARE DISSOLVED BACK INTO A 2002 PARENT, CHOSEN BY LONGEST SHARED
BOUNDARY WITHIN THE SAME DEPARTMENT.** A district created after 2002 was, in 2002, part of a
district that kept its code, so its territory has to be given back or the parent's dots have
nowhere to go. Two of the assignments are not a judgement at all:

  * **Boqueron had exactly ONE district in 2002** (`1602` Mcal. Jose F. Estigarribia), so
    Filadelfia and Loma Plata, the two Mennonite colony towns, can only have come from it.
  * **Alto Paraguay's two children** share 322 km and 308 km with one parent and **zero** with
    the other.

**ELEVEN OF THE REMAINING TWENTY-FOUR ARE CLOSE CALLS** — the runner-up boundary is more than
60% as long as the winner. They are listed in `PARENT` with their margin and named again in
`sources/py.md` §5. **No count depends on any of them**: the counts are per 2002 district
whatever polygon they are drawn in, and a wrong call moves dots between two adjacent districts
of the same department. It is a placement error with a bounded blast radius, not a data error,
and it is the reason `countries.py` carries a `fill`.

**AND ONE OF THEM IS WORTH NAMING BECAUSE THE WITNESS FLAGS IT.** `1503` Pto. Pinasco counted
2,702 people aged 10+ in 2002 and the reconstructed unit holds 40,056 in Kontur 2023, a ratio
of 14.8 against a national 1.80. Either the central Chaco really did fill up that fast, which
it partly did, or `1507` Tte. Irala Fernandez was carved from somewhere else. The unit is
drawn as assigned and the ratio is printed by `check()` so it stays visible.

Usage:
    python sources/py_geo.py --fetch    one ~22 MB zip from HDX
    python sources/py_geo.py            rebuild from data/raw/py/
"""

import argparse
import csv
import os
import sys
import zipfile

os.environ.setdefault("OMP_NUM_THREADS", "6")

import geopandas as gpd
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "py")
GEO = os.path.join(ROOT, "data", "geo", "py")
NORM = os.path.join(ROOT, "data", "normalized", "py.csv")

ZIP_URL = ("https://data.humdata.org/dataset/212cd82f-bcb8-445f-8b32-aa194387f6c3/"
           "resource/7b2e1608-ca01-4271-ba60-dcde5c73433f/download/"
           "pry_adm_dgeec_2020_shp.zip")
ZIP_NAME = "pry_adm_dgeec_2020_shp.zip"
SHP = "pry_admbnda_adm2_DGEEC_2020.shp"

# Paraguay is entirely west of the antimeridian and entirely in one UTM zone; 32721
# (WGS 84 / UTM 21S) is used for every length and area here.
METRIC = 32721

# The capital's six census districts, aggregated onto the single COD-AB polygon 0000.
ASUNCION = [f"001{i}" for i in range(6)]
ASUNCION_CODE = "0000"

# post-2002 district -> the 2002 district it is dissolved into.
# Derived by longest shared boundary within the department; the comment is
# (winner km, runner-up km). `weak` marks a runner-up above 60% of the winner.
PARENT = {
    "0105": "0106",   # San Carlos del Apa  ->  San Lazaro          ( 26.9,  0.0)
    "0108": "0103",   # Azote'y             ->  Horqueta            ( 65.3, 47.9) weak
    "0109": "0107",   # Sgto. Jose F. Lopez ->  Yby Ya'u            (  0.0,  0.0) weak, no shared edge
    "0110": "0106",   # San Alfredo         ->  San Lazaro          ( 63.4, 42.4) weak
    "0111": "0104",   # Paso Barreto        ->  Loreto              ( 65.8, 45.0) weak
    "0219": "0216",   # Yrybucua            ->  Guajayvi            ( 44.8, 23.5)
    "0220": "0214",   # Liberacion          ->  Gral. I. Resquin    ( 63.0, 50.5) weak
    "0418": "0404",   # Tebicuary           ->  Coronel Martinez    ( 19.5, 16.5) weak
    "0521": "0514",   # Tembiapora          ->  Raul A. Oviedo      ( 39.3, 22.0)
    "0522": "0520",   # Nueva Toledo        ->  Vaqueria            ( 55.1, 39.2) weak
    "0611": "0607",   # 3 de Mayo           ->  San Juan Nepomuceno ( 46.7, 43.5) weak
    "1020": "1017",   # Santa Fe del Parana ->  Mbaracayu           ( 60.1, 46.2) weak
    "1021": "1013",   # Tavapy              ->  Santa Rita          ( 30.7, 27.4) weak
    "1022": "1014",   # Dr. Raul Pena       ->  Naranjal            ( 59.0, 23.0)
    "1304": "1301",   # Zanja Pyta          ->  Pedro Juan Caballero( 97.8,  4.8)
    "1305": "1303",   # Karapai             ->  Capitan Bado        ( 77.3, 17.7)
    "1411": "1403",   # Yasy Cany           ->  Villa Curuguaty     ( 87.3,  0.0)
    "1412": "1402",   # Ybyrarobana         ->  Corpus Christi      ( 52.4, 15.2)
    "1413": "1404",   # Yby Pyta            ->  Villa Ygatimi       ( 64.5, 49.0) weak
    "1507": "1503",   # Tte. Irala Fernandez->  Pto. Pinasco        (176.1, 64.5) see docstring
    "1508": "1504",   # Tte. Esteban Martinez-> Villa Hayes         ( 59.3,  0.0)
    "1509": "1502",   # Gral. J. M. Bruguez ->  Benjamin Aceval     (106.2, 87.3) weak
    "1604": "1602",   # Filadelfia          ->  Mcal. Estigarribia  (272.6,  0.0) sole 2002 district
    "1605": "1602",   # Loma Plata          ->  Mcal. Estigarribia  ( 26.5,  0.0) sole 2002 district
    "1704": "1701",   # Bahia Negra         ->  Fuerte Olimpo       (322.0,  0.0)
    "1705": "1702",   # Carmelo Peralta     ->  La Victoria         (308.3,  0.0)
}

N_UNITS = 224          # 223 districts outside the capital, plus Asuncion as one


def fetch():
    import requests
    import urllib3
    urllib3.disable_warnings()
    os.makedirs(RAW, exist_ok=True)
    dest = os.path.join(RAW, ZIP_NAME)
    if os.path.exists(dest) and os.path.getsize(dest) > 1_000_000:
        print(f"  have {dest}")
    else:
        r = requests.get(ZIP_URL, headers={"User-Agent": "religiondots"},
                         timeout=900, verify=False)
        r.raise_for_status()
        with open(dest, "wb") as f:
            f.write(r.content)
        print(f"  {len(r.content):,} bytes -> {dest}")
    if not zipfile.is_zipfile(dest):
        raise SystemExit(f"{dest} is not a zip -- HDX served something else")
    with zipfile.ZipFile(dest) as z:
        z.extractall(os.path.join(RAW, "shp"))
    print(f"  extracted to {os.path.join(RAW, 'shp')}")


def census_codes():
    """The district codes the census actually tabulated, with Asuncion aggregated."""
    codes = {}
    with open(NORM, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            codes[row["geo_id"]] = row["geo_name"]
    for c in ASUNCION:
        codes.pop(c, None)
    codes[ASUNCION_CODE] = "ASUNCION"
    return codes


def build():
    path = os.path.join(RAW, "shp", SHP)
    if not os.path.exists(path):
        raise SystemExit(f"missing {path}; run with --fetch")

    # fiona rather than pyogrio: §12's rule, pyogrio is the engine that has silently
    # returned zero features from an HDX bundle.
    g = gpd.read_file(path, engine="fiona")
    if len(g) == 0:
        raise SystemExit("ADM2 read returned ZERO features")
    g["code"] = g["ADM2_PCODE"].str.replace("^PY", "", regex=True)
    g["unit"] = g["code"].map(lambda c: PARENT.get(c, c))

    want = census_codes()
    unknown = sorted(set(g["unit"]) - set(want))
    missing = sorted(set(want) - set(g["unit"]))
    assert not unknown, f"polygons resolving to no census district: {unknown}"
    assert not missing, f"census districts with no polygon: {missing}"

    units = g.dissolve(by="unit", as_index=False)[["unit", "geometry"]]
    units["name"] = units["unit"].map(want)
    units = units.rename(columns={"unit": "code"}).to_crs(4326)
    assert len(units) == N_UNITS, f"{len(units)} units, expected {N_UNITS}"

    # A torn or mis-projected country shows up as an implausible bbox. Paraguay spans
    # roughly 62.6W..54.2W and 27.6S..19.3S.
    minx, miny, maxx, maxy = units.total_bounds
    assert -63.0 < minx < -62.0 and -55.0 < maxx < -54.0, f"lon bounds {minx},{maxx}"
    assert -28.0 < miny < -27.0 and -20.0 < maxy < -19.0, f"lat bounds {miny},{maxy}"

    os.makedirs(GEO, exist_ok=True)
    out = os.path.join(GEO, "py_distritos.gpkg")
    units.to_file(out, driver="GPKG", layer="distritos")
    # geo_id -> unit, the shape every other country's lookup uses. The only rows where
    # the two differ are Asuncion's six census districts, which share one polygon.
    seen = {}
    with open(NORM, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            seen[row["geo_id"]] = row["geo_name"]
    with open(os.path.join(GEO, "py_lookup.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["geo_id", "unit", "geo_name", "dept"])
        for gid in sorted(seen):
            unit = ASUNCION_CODE if gid in ASUNCION else gid
            w.writerow([gid, unit, seen[gid], gid[:2]])
    print(f"  {len(units)} units -> {out}")
    check(units, g)


def check(units, g):
    """Print the growth-ratio witness, so a bad parent stays visible."""
    kon = os.path.join(GEO, "py_kontur_by_unit.csv")
    if not os.path.exists(kon):
        print("  (no kontur totals yet; run sources/py_grid.py for the witness)")
        return
    pop = {}
    with open(NORM, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            c = ASUNCION_CODE if row["geo_id"] in ASUNCION else row["geo_id"]
            pop[c] = pop.get(c, 0) + int(row["count"])
    k = {r["code"]: int(r["kontur_2023"]) for r in
         csv.DictReader(open(kon, encoding="utf-8"))}
    rows = sorted(((k.get(c, 0) / p, c) for c, p in pop.items() if p), reverse=True)
    nat = sum(k.values()) / sum(pop.values())
    print(f"  growth ratio (kontur 2023 all ages / census 2002 aged 10+): "
          f"national {nat:.2f}")
    for r, c in rows[:3]:
        print(f"    highest {c} {r:.1f}x")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fetch", action="store_true")
    a = ap.parse_args()
    if a.fetch:
        fetch()
    build()


if __name__ == "__main__":
    main()
