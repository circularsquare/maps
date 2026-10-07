"""Gabon: the 9 provinces, COD-AB polygons, with the 2026 census count split into Gabonese citizens
and foreign residents.

Writes data/geo/ga/ga_provinces.gpkg and data/geo/ga/ga_lookup.csv. `sources/ga.md` §4 is the
record.

  * **boundaries**: COD-AB `cod-ab-gab` v01 (INC Gabon via OCHA; valid from 2026-06-01),
    `gab_admin1.geojson`: GA01-GA09, the nine provinces.
  * **population**: the RGPL 2026 province counts the Constitutional Court certified on 27 August
    2026, 3,518,621 in all. No table from INSTAT is out; the nine figures are as Gabon Media Time
    printed them (`RGPL_2026_URL`, read 2026-10-03), and they sum to the certified total to the
    person. Estuaire, Haut-Ogooué and Ogooué-Maritime are in a second article of the same outlet,
    and Estuaire in Gabonactu and Economie Gabon+. The Minister of Planning gave the national split
    on Gabon 1ère on 1 September 2026: 2,318,365 Gabonese and 1,200,256 foreign residents (34.1%).
  * **who is drawn**: the religion source is the Afrobarometer, which samples Gabonese citizens of
    18 and over only. So the dots are the Gabonese; foreign residents are not drawn (Anita
    2026-09-15, Libya ruling: estimate them and put them in the not-drawn part of the bar).
  * **Gabonese per province is an estimate.** 2026 gives the split nationally only. RGPL 2013
    printed foreigners per province (*Résultats globaux*, Tableau 24, against Tableau 5's
    totals), so the 9 x 2 table is fitted (IPF) to the 2026 province totals and the 2026 national
    split, seeded with the 2013 province mix: every province's odds of a resident being foreign
    move by the same factor since 2013. 2013's 65,236 people of undeclared nationality are on the
    Gabonese side of the seed (Tableau 23 counts them apart; not per province).
  * **not used**: COD-PS 2022 (`cod-ps-gab`, a DGS projection from 2013) is older than the
    count. The 2013 count is the witness (`check_2013`).

THE JOIN is by name over a fixed list of nine, folded. The witness neither key decides is area:
RGPL 2013 Tableau 5 prints density per province, so its population over density is the census's
area, and each COD polygon must be within `AREA_TOL` of it.

Usage:
    python sources/ga_geo.py --fetch    COD-AB geojson zip and the RGPL 2013 volume into data/raw/ga/
    python sources/ga_geo.py            rebuild from data/raw/ga/
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

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "ga")
GEO = os.path.join(ROOT, "data", "geo", "ga")
OUT = os.path.join(GEO, "ga_provinces.gpkg")
LOOKUP = os.path.join(GEO, "ga_lookup.csv")

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/128.0 Safari/537.36")
COD_AB_URL = ("https://data.humdata.org/dataset/d8d992f3-5374-448f-a3d5-bed7c1925a11/resource/"
              "72ab3a59-84a4-461f-ac4d-d49bb79574b9/download/gab_admin_boundaries.geojson.zip")
COD_AB = os.path.join(RAW, "gab_admin_boundaries.geojson.zip")
RGPL_2013_URL = "https://gabon.unfpa.org/sites/default/files/pub-pdf/Resultats%20Globaux%20RGPL(1).pdf"
RGPL_2013 = os.path.join(RAW, "rgpl2013_resultats_globaux.pdf")
RGPL_2026_URL = ("https://gabonmediatime.com/gabon-62-de-la-population-concentree-dans-lestuaire-"
                 "lurgence-de-repeupler-linterieur-du-pays/")

# RGPL 2026, certified 2026-08-27, by COD-AB p-code.
CENSUS_2026 = {
    "GA01": ("Estuaire", 2_184_908), "GA02": ("Haut-Ogooué", 297_100),
    "GA03": ("Moyen-Ogooué", 105_703), "GA04": ("Ngounié", 159_404),
    "GA05": ("Nyanga", 68_215), "GA06": ("Ogooué-Ivindo", 93_482),
    "GA07": ("Ogooué-Lolo", 75_128), "GA08": ("Ogooué-Maritime", 292_652),
    "GA09": ("Woleu-Ntem", 242_029),
}
NATIONAL_2026 = 3_518_621
GABONESE_2026, FOREIGN_2026 = 2_318_365, 1_200_256

# RGPL 2013 Résultats globaux: Tableau 5 (total, density; PDF page 33) and Tableau 24 (foreign
# residents; PDF page 50), by p-code. Re-read from the PDF's text on every run.
CENSUS_2013 = {
    "GA01": (895_689, "43,2", 181_416), "GA02": (250_799, "6,9", 34_861),
    "GA03": (69_287, "3,7", 5_502), "GA04": (100_838, "2,7", 5_388),
    "GA05": (52_854, "2,5", 3_808), "GA06": (63_293, "1,4", 2_233),
    "GA07": (65_771, "2,6", 2_617), "GA08": (157_562, "6,9", 24_401),
    "GA09": (154_986, "4,0", 27_153),
}
NATIONAL_2013, FOREIGN_2013, GABONESE_2013 = 1_811_079, 287_379, 1_458_464
PAGE_T5, PAGE_T24 = 33, 50
AREA_TOL = 0.12              # measured 0.890 (Moyen-Ogooué) to 1.056 (Nyanga), less Ogooué-Lolo
# Ogooué-Lolo reads 1.151 (29,118 km2 against 25,297; the density's rounding, 2.55-2.65, allows
# 1.13-1.17), and its neighbour Haut-Ogooué 0.926: COD draws their shared line further east than
# the 2013 densities imply, about 2,700-3,800 km2 of forest. Whether it moves people is the
# per-province Kontur check in `sources/ga_grid.py`. Pinned here so a different miss still stops.
AREA_PINNED = {"GA07": (1.10, 1.20)}
METRIC_AREA = "ESRI:54034"


def fold(s):
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower()
    return re.sub(r"[^a-z]", "", s)


def _get(url, path, magic):
    if os.path.exists(path) and os.path.getsize(path) > 10_000:
        print(f"  have {os.path.basename(path)} ({os.path.getsize(path):,} bytes)")
        return
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=600) as r:
        data = r.read()
    if not data.startswith(magic):
        raise SystemExit(f"{url} does not start {magic!r} (starts {data[:16]!r})")
    with open(path + ".part", "wb") as fh:
        fh.write(data)
    os.replace(path + ".part", path)
    print(f"  got  {os.path.basename(path)} ({len(data):,} bytes)")


def fetch():
    os.makedirs(RAW, exist_ok=True)
    _get(COD_AB_URL, COD_AB, b"PK")
    _get(RGPL_2013_URL, RGPL_2013, b"%PDF")


def fmt(n):
    """The volume prints thousands with a space in Tableau 5 and without one in Tableau 24."""
    return f"{n:,}".replace(",", " ")


def check_2013():
    """Re-read the 2013 figures off the PDF's text layer and check their sums."""
    import fitz

    doc = fitz.open(RGPL_2013)
    t5 = doc[PAGE_T5 - 1].get_text()
    t24 = doc[PAGE_T24 - 1].get_text()
    if "Tableau 5" not in t5 or "Tableau 24" not in t24:
        raise SystemExit("the 2013 volume's pages have moved; Tableau 5 or 24 is not where expected")
    toks5, toks24 = t5.split("\n"), [s.strip() for s in t24.split("\n")]
    for pc, (tot, dens, fgn) in CENSUS_2013.items():
        if f"{fmt(tot)} " not in toks5 and fmt(tot) not in [s.strip() for s in toks5]:
            raise SystemExit(f"{CENSUS_2026[pc][0]}: Tableau 5's {tot:,} is not on PDF page {PAGE_T5}")
        if str(fgn) not in toks24:
            raise SystemExit(f"{CENSUS_2026[pc][0]}: Tableau 24's {fgn:,} is not on PDF page {PAGE_T24}")
    if sum(v[0] for v in CENSUS_2013.values()) != NATIONAL_2013:
        raise SystemExit("Tableau 5's provinces do not sum to 1,811,079")
    if sum(v[2] for v in CENSUS_2013.values()) != FOREIGN_2013:
        raise SystemExit("Tableau 24's provinces do not sum to 287,379")
    print(f"  RGPL 2013: Tableau 5 and Tableau 24 re-read from the PDF; {NATIONAL_2013:,} residents, "
          f"{FOREIGN_2013:,} foreign ({FOREIGN_2013 / NATIONAL_2013:.1%})")


def check_2026():
    tot = sum(p for _n, p in CENSUS_2026.values())
    if tot != NATIONAL_2026 or GABONESE_2026 + FOREIGN_2026 != NATIONAL_2026:
        raise SystemExit(f"RGPL 2026 as transcribed: provinces {tot:,}, split "
                         f"{GABONESE_2026 + FOREIGN_2026:,}, certified {NATIONAL_2026:,}")
    print(f"  RGPL 2026: nine provinces sum to the certified {NATIONAL_2026:,}; Gabonese "
          f"{GABONESE_2026:,}, foreign {FOREIGN_2026:,} ({FOREIGN_2026 / NATIONAL_2026:.1%})")


def split_citizens():
    """The 9 x 2 IPF: rows the 2026 province totals, columns the 2026 national split, seeded with
    each province's 2013 foreign share. Returns integer Gabonese and foreign counts per province
    that meet both margins exactly."""
    codes = sorted(CENSUS_2026)
    rows = np.array([CENSUS_2026[c][1] for c in codes], dtype=float)
    f13 = np.array([CENSUS_2013[c][2] / CENSUS_2013[c][0] for c in codes])
    m = np.column_stack([1 - f13, f13]) * rows[:, None]
    cols = np.array([GABONESE_2026, FOREIGN_2026], dtype=float)
    for _ in range(2000):
        m *= (rows / m.sum(axis=1))[:, None]
        m *= cols / m.sum(axis=0)
        if np.abs(m.sum(axis=1) - rows).max() < 1e-6:
            break
    else:
        raise SystemExit("the citizen split did not converge")
    fgn = np.floor(m[:, 1]).astype("int64")
    short = FOREIGN_2026 - int(fgn.sum())
    fgn[np.argsort(-(m[:, 1] - fgn))[:short]] += 1
    gab = rows.astype("int64") - fgn
    out = pd.DataFrame({"geo_id": codes, "total_2026": rows.astype("int64"), "gabonese": gab,
                        "foreign": fgn})
    if int(out["foreign"].sum()) != FOREIGN_2026 or int(out["gabonese"].sum()) != GABONESE_2026:
        raise SystemExit("the rounded split misses a national margin")
    odds = (m[:, 1] / m[:, 0]) / (f13 / (1 - f13))
    print("\n  Gabonese and foreign residents by province, 2026 (fitted) beside 2013 (counted):")
    print(f"    {'':<16}{'2026 total':>11}{'Gabonese':>11}{'foreign':>10}{'share':>7}"
          f"{'2013 share':>11}{'Gabonese 2013->2026':>21}")
    for i, c in enumerate(codes):
        g13 = CENSUS_2013[c][0] - CENSUS_2013[c][2]
        r = out.iloc[i]
        print(f"    {CENSUS_2026[c][0]:<16}{r.total_2026:>11,}{r.gabonese:>11,}{r.foreign:>10,}"
              f"{r.foreign / r.total_2026:7.1%}{f13[i]:11.1%}{r.gabonese / g13:20.2f}x")
    print(f"    every province's foreign odds x{odds[0]:.3f} against 2013 (one factor, by construction)")
    return out


def main():
    import geopandas as gpd

    if "--fetch" in sys.argv or not os.path.exists(COD_AB) or not os.path.exists(RGPL_2013):
        fetch()
    check_2026()
    check_2013()
    split = split_citizens()

    with zipfile.ZipFile(COD_AB) as z:
        g = gpd.read_file(io.BytesIO(z.read("gab_admin1.geojson")))
    if len(g) != 9 or set(g["adm1_pcode"]) != set(CENSUS_2026):
        raise SystemExit(f"COD-AB admin1 is not GA01-GA09: {sorted(g['adm1_pcode'])}")
    bad = [(p, n) for p, n in zip(g["adm1_pcode"], g["adm1_name"]) if fold(n) != fold(CENSUS_2026[p][0])]
    if bad:
        raise SystemExit(f"COD-AB names that are not the pinned province for their p-code: {bad}")

    g["unit"] = g["adm1_pcode"]
    g["name"] = [CENSUS_2026[p][0] for p in g["unit"]]
    gab = dict(zip(split["geo_id"], split["gabonese"]))
    g["pop"] = [int(gab[p]) for p in g["unit"]]
    area = g.to_crs(METRIC_AREA).area / 1e6
    print("\n  province, COD km2 / census 2013 km2 (Tableau 5 population over density):")
    worst = 0.0
    for (_i, r), a in sorted(zip(g.iterrows(), area), key=lambda t: t[0][1]["unit"]):
        tot, dens, _f = CENSUS_2013[r["unit"]]
        ckm2 = tot / float(dens.replace(",", "."))
        rel = a / ckm2
        print(f"      {r['unit']}  {r['name']:<16} {a:>8,.0f} / {ckm2:>8,.0f} = {rel:5.3f}")
        if r["unit"] in AREA_PINNED:
            lo, hi = AREA_PINNED[r["unit"]]
            if not lo <= rel <= hi:
                raise SystemExit(f"{r['name']} reads {rel:.3f}, outside its pinned {lo}-{hi}")
            continue
        worst = max(worst, abs(rel - 1))
    if worst > AREA_TOL:
        raise SystemExit(f"a province's COD area is {worst:.1%} off the census's; the join or the polygon is wrong")
    print(f"  area witness: every province within {worst:.1%} of the census's area (bar {AREA_TOL:.0%})")

    os.makedirs(GEO, exist_ok=True)
    g[["unit", "name", "pop", "geometry"]].to_file(OUT, layer="provinces", driver="GPKG")
    lut = g[["unit", "name", "pop"]].rename(columns={"unit": "geo_id"})
    lut["unit"] = lut["geo_id"]
    lut = lut.merge(split[["geo_id", "total_2026", "foreign"]], on="geo_id")
    lut.sort_values("geo_id").to_csv(LOOKUP, index=False, encoding="utf-8")
    print(f"\nwrote {OUT} and {LOOKUP} (9 provinces, {int(g['pop'].sum()):,} Gabonese drawn, "
          f"{FOREIGN_2026:,} foreign residents not)")


if __name__ == "__main__":
    main()
