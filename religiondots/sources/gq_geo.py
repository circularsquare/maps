"""Equatorial Guinea: the 7 provinces of the 2015 census, COD-AB polygons with the census's own
definitive count.

Writes data/geo/gq/gq_provinces.gpkg and data/geo/gq/gq_lookup.csv. `sources/gq.md` §4 is the
record.

  * **boundaries**: COD-AB `cod-ab-gnq` v01 (OCHA ROWCA; valid from 2021-07-15, reviewed
    2025-10-30), `gnq_admin1.geojson`: GQ198-GQ204, the seven provinces the 2015 census counted.
    Djibloho, a province since 2017, is not a unit in it; its Ciudad Nueva Oyala is a municipality
    of Wele-Nzas there, which is where the 2015 census counted it.
  * **population**: INEGE, *Resultados definitivos del IV Censo General de Población y Viviendas
    2015* (`data/raw/gq/RESULTADOS-DEFINITIVOS-...-2015.pdf`, a phone scan with no text layer),
    Tabla 2.1 *Población total por provincia* (printed p.18): 1,225,377. Transcribed into
    `CENSUS_2015` and checked three ways on every run, since the scan cannot be parsed: the
    provinces sum to Tabla 1.1's region rows and national total; Tabla 3.1's 18 districts
    (printed p.19) sum to each province; and the *Resultados preliminares* (September 2015, text
    layer absent too, `PRELIM_2015`) are within `PRELIM_TOL` of each province.
  * **not used**: there is no COD-PS for Equatorial Guinea on HDX (`cod-ps-gnq` returns 404,
    2026-10-03), and the census is the newest count.

THE JOIN is by name over a fixed list of seven, folded. The witness neither key decides is area:
the preliminary volume's Tabla 6.1 prints density per province, so its population over its density
is the census's area, to the rounding of a whole-number density, and each COD polygon must be within
`AREA_TOL` of it.

Usage:
    python sources/gq_geo.py --fetch    COD-AB geojson zip into data/raw/gq/
    python sources/gq_geo.py            rebuild from data/raw/gq/
"""

import io
import os
import re
import sys
import unicodedata
import urllib.request
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("OMP_NUM_THREADS", "6")

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "gq")
GEO = os.path.join(ROOT, "data", "geo", "gq")
OUT = os.path.join(GEO, "gq_provinces.gpkg")
LOOKUP = os.path.join(GEO, "gq_lookup.csv")

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/128.0 Safari/537.36")
COD_AB_URL = ("https://data.humdata.org/dataset/0c1a3093-c97d-48a2-a992-e652ebcbf05c/resource/"
              "6b004317-faaa-4176-b5f4-eb6cd4ed46a6/download/gnq_admin_boundaries.geojson.zip")
COD_AB = os.path.join(RAW, "gnq_admin_boundaries.geojson.zip")
DEFINITIVOS = os.path.join(RAW, "RESULTADOS-DEFINITIVOS-DEL-IV-CENSO-GENERAL-DE-POBLACION-Y-VIVIENDAS-2015.pdf")
DEFINITIVOS_BYTES = 18_947_093          # sources.md §gq-2026-09-15 has its SHA-256

NATIONAL_2015 = 1_225_377
INSULAR, CONTINENTAL = 340_362, 885_015              # Tabla 1.1
# Tabla 2.1, by COD-AB p-code. Read off the scan 2026-10-03.
CENSUS_2015 = {
    "GQ199": ("Bioko Norte", 300_374), "GQ200": ("Bioko Sur", 34_674),
    "GQ198": ("Annobón", 5_314), "GQ203": ("Litoral", 367_348),
    "GQ201": ("Centro Sur", 141_986), "GQ204": ("Wele-Nzas", 192_017),
    "GQ202": ("Kié-Ntem", 183_664),
}
INSULAR_CODES = {"GQ198", "GQ199", "GQ200"}
# Tabla 3.1, the 18 districts, by province. Used only as a check on the transcription.
DISTRICTS_2015 = {
    "GQ199": {"Malabo": 271_008, "Baney": 29_366},
    "GQ200": {"Luba": 26_331, "Riaba": 8_343},
    "GQ198": {"Annobón": 5_314},
    "GQ203": {"Bata": 309_345, "Mbini": 28_662, "Cogo": 29_341},
    "GQ201": {"Evinayong": 56_664, "Niefang": 61_708, "Acurenam": 23_614},
    "GQ204": {"Mongomo": 88_326, "Añisok": 62_924, "Nsork": 16_421, "Aconibe": 24_346},
    "GQ202": {"Ebibeyín": 94_019, "Micomiseng": 51_725, "Nsok Nsomo": 37_920},
}
# Resultados preliminares (data/raw/gq/RESULTADOS-PRELIMINARES-DEL-IV-CENSO-DE-POBLACION-2015.pdf),
# Tabla 2.1 population and Tabla 6.1 density, by p-code: 1,222,442 in all.
PRELIM_2015 = {
    "GQ199": (299_836, 452), "GQ200": (34_627, 27), "GQ198": (5_232, 258),
    "GQ203": (366_130, 53), "GQ201": (141_903, 18), "GQ204": (191_383, 29),
    "GQ202": (183_331, 52),
}
PRELIM_NATIONAL = 1_222_442
PRELIM_TOL = 0.02           # Annobón 5,314 against 5,232 (+1.57%); the six others within 0.35%
AREA_TOL = 0.15             # measured 0.885 (Bioko Norte) to 1.067 (Centro Sur); sources/gq.md §4
METRIC_AREA = "ESRI:54034"


def fold(s):
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower()
    return re.sub(r"[^a-z]", "", s)


def fetch():
    os.makedirs(RAW, exist_ok=True)
    if os.path.exists(COD_AB) and os.path.getsize(COD_AB) > 10_000:
        print(f"  have {os.path.basename(COD_AB)} ({os.path.getsize(COD_AB):,} bytes)")
        return
    req = urllib.request.Request(COD_AB_URL, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=600) as r:
        data = r.read()
    if not data.startswith(b"PK"):
        raise SystemExit(f"{COD_AB_URL} is not a zip (starts {data[:16]!r})")
    with open(COD_AB + ".part", "wb") as fh:
        fh.write(data)
    os.replace(COD_AB + ".part", COD_AB)
    print(f"  got  {os.path.basename(COD_AB)} ({len(data):,} bytes)")


def check_transcription():
    if not os.path.exists(DEFINITIVOS) or os.path.getsize(DEFINITIVOS) != DEFINITIVOS_BYTES:
        raise SystemExit(f"{DEFINITIVOS} is missing or not the {DEFINITIVOS_BYTES:,}-byte scan the "
                         "figures were read from")
    tot = sum(p for _n, p in CENSUS_2015.values())
    ins = sum(CENSUS_2015[c][1] for c in INSULAR_CODES)
    if tot != NATIONAL_2015 or ins != INSULAR or tot - ins != CONTINENTAL:
        raise SystemExit(f"Tabla 2.1 as transcribed sums to {tot:,} ({ins:,} insular); Tabla 1.1 "
                         f"says {NATIONAL_2015:,} ({INSULAR:,})")
    for c, d in DISTRICTS_2015.items():
        if sum(d.values()) != CENSUS_2015[c][1]:
            raise SystemExit(f"{CENSUS_2015[c][0]}: Tabla 3.1's districts sum to {sum(d.values()):,}, "
                             f"Tabla 2.1 says {CENSUS_2015[c][1]:,}")
    if sum(len(d) for d in DISTRICTS_2015.values()) != 18:
        raise SystemExit("Tabla 3.1 has 18 districts")
    if sum(p for p, _d in PRELIM_2015.values()) != PRELIM_NATIONAL:
        raise SystemExit("the preliminary provinces do not sum to 1,222,442")
    worst = max(abs(CENSUS_2015[c][1] / PRELIM_2015[c][0] - 1) for c in CENSUS_2015)
    if worst > PRELIM_TOL:
        raise SystemExit(f"a province's definitive count is {worst:.2%} off the preliminary one")
    print(f"  census 2015 (definitive), 7 provinces: {NATIONAL_2015:,}; Tabla 1.1's regions and "
          f"Tabla 3.1's 18 districts agree; preliminary within {worst:.2%} per province "
          f"({PRELIM_NATIONAL:,} nationally)")


def main():
    import geopandas as gpd

    if "--fetch" in sys.argv or not os.path.exists(COD_AB):
        fetch()
    check_transcription()

    with zipfile.ZipFile(COD_AB) as z:
        g = gpd.read_file(io.BytesIO(z.read("gnq_admin1.geojson")))
    if len(g) != 7 or set(g["adm1_pcode"]) != set(CENSUS_2015):
        raise SystemExit(f"COD-AB admin1 is not GQ198-GQ204: {sorted(g['adm1_pcode'])}")
    bad = [(p, n) for p, n in zip(g["adm1_pcode"], g["adm1_name"]) if fold(n) != fold(CENSUS_2015[p][0])]
    if bad:
        raise SystemExit(f"COD-AB names that are not the pinned province for their p-code: {bad}")

    g["unit"] = g["adm1_pcode"]
    g["name"] = [CENSUS_2015[p][0] for p in g["unit"]]
    g["pop"] = [CENSUS_2015[p][1] for p in g["unit"]]
    area = g.to_crs(METRIC_AREA).area / 1e6
    print("\n  province, census 2015, COD km2 / census km2 (preliminary population over density):")
    worst = 0.0
    for (_i, r), a in sorted(zip(g.iterrows(), area), key=lambda t: t[0][1]["unit"]):
        pp, dens = PRELIM_2015[r["unit"]]
        ckm2 = pp / dens
        rel = a / ckm2
        worst = max(worst, abs(rel - 1))
        print(f"      {r['unit']}  {r['name']:<12} {r['pop']:>9,}  {a:>8,.0f} / {ckm2:>8,.0f} = {rel:5.3f}")
    if worst > AREA_TOL:
        raise SystemExit(f"a province's COD area is {worst:.1%} off the census's; the join or the polygon is wrong")
    print(f"  area witness: every province within {worst:.1%} of the census's area (bar {AREA_TOL:.0%})")

    os.makedirs(GEO, exist_ok=True)
    g[["unit", "name", "pop", "geometry"]].to_file(OUT, layer="provinces", driver="GPKG")
    lut = g[["unit", "name", "pop"]].rename(columns={"unit": "geo_id"})
    lut["unit"] = lut["geo_id"]
    lut.sort_values("geo_id").to_csv(LOOKUP, index=False, encoding="utf-8")
    print(f"\nwrote {OUT} and {LOOKUP} (7 provinces, {int(g['pop'].sum()):,} people)")


if __name__ == "__main__":
    main()
