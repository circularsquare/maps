"""Cabo Verde — boundaries for the 22 concelhos.

Writes data/geo/cv/cv_concelhos.gpkg and data/geo/cv/cv_lookup.csv.

OCHA COD-AB Cabo Verde (`cod-ab-cpv`), the **shapefile** bundle rather than the geodatabase
on §12's Chile rule. `cpv_admin1.shp` is the 22 concelhos, which is exactly the tier INE's
census workbooks publish religion at; `cpv_admin2.shp` is the 32 freguesias, which nothing
here has religion for, and `cpv_islands.shp` is the nine inhabited islands.

## THREE PAIRS OF CONCELHOS SHARE A NAME AND THE CENSUS SHEETS USE THE SHORT FORM

    Ribeira Grande  (Santo Antão)   vs  Ribeira Grande de Santiago
    Santa Catarina  (Santiago)      vs  Santa Catarina do Fogo
    Tarrafal        (Santiago)      vs  Tarrafal de São Nicolau

COD-AB prints the short name for the first of each pair. So does the table title inside each
INE workbook — *Tabela 1 - População residente no concelho de Ribeira Grande*, with nothing
to say which island. **A name join on the sheet title would pair each of those three with a
coin flip**, and [[reference_name_join_wrong_neighbour]] is that a wrong pairing survives
every total: the country still sums to 491,233 either way.

The pairing therefore lives in `sources/cv.py`'s `PCODE`, keyed on INE's own content id from
its site API, whose listing titles ARE unambiguous (*Ribeira Grande de Santo Antão*, *Santa
Catarina de Santiago*, *Tarrafal de Santiago*). This file proves it three ways rather than
counting matched rows:

  1. **nineteen of the twenty-two names fold to a COD name exactly**, and the three that do
     not are precisely the ambiguous ones, which is the shape the trap has;
  2. **populations rank together across all 22 units** — Spearman against COD-PS, which is a
     projection off the 2010 census and shares no lineage with the 2021 workbooks;
  3. **each of the three ambiguous swaps is tested and must fail.** Swapping Ribeira Grande
     for Ribeira Grande de Santiago takes a −1.5% population difference to −43% and +92%.
     That assertion is the point of the file: if COD ever renames these, it stops here.

COD-PS 2022 is **16% above the 2021 census nationally** (569,523 against 491,233) and Boa
Vista is +72%, because it projects the 2010 census forward and Boa Vista's tourist boom
outran it. So it is used for RANK and never for level, and `sources/cv_grid.py` carries the
independent check on magnitude.

Usage:
    python sources/cv_geo.py --fetch    one ~17 MB zip from HDX, plus a 7 KB CSV
    python sources/cv_geo.py            rebuild from data/raw/cv/
"""

import os
import sys
import unicodedata
import urllib.request
import zipfile

os.environ.setdefault("OMP_NUM_THREADS", "6")          # [[feedback_cap_cpu]]
os.environ.setdefault("OPENBLAS_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import geopandas as gpd
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

RAW = os.path.join(ROOT, "data", "raw", "cv")
SHP_DIR = os.path.join(RAW, "shp")
OUT_DIR = os.path.join(ROOT, "data", "geo", "cv")
OUT = os.path.join(OUT_DIR, "cv_concelhos.gpkg")
LOOKUP = os.path.join(OUT_DIR, "cv_lookup.csv")

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) religiondots/1.0"}

DOWNLOADS = {
    "cpv_admin_boundaries.shp.zip":
        "https://data.humdata.org/dataset/ce7df560-f77b-4f55-9175-4719ea21f50b/resource/"
        "ed06ede5-0c7a-4b93-a257-d03636c347bf/download/cpv_admin_boundaries.shp.zip",
    "cpv_admpop_adm1_2022B.csv":
        "https://data.humdata.org/dataset/f8075c2a-7c31-49c1-9aa8-925a271d4de2/resource/"
        "ac138668-3b09-4c5d-96fd-c747a5ff632b/download/cpv_admpop_adm1_2022b.csv",
}

EXPECTED_UNITS = 22

# The three pairs whose first member COD-AB and the census sheets both call by a short name.
AMBIGUOUS = [("CV09", "CV10"), ("CV12", "CV13"), ("CV21", "CV22")]

# COD-AB's own spelling for each of the three, so a rename is caught rather than absorbed.
COD_SHORT = {"CV09": "Ribeira Grande", "CV12": "Santa Catarina", "CV21": "Tarrafal"}

# How close the true pairing runs, and how far a swapped one must be pushed. COD-PS is a
# projection off the 2010 census, so the band is on the loose side deliberately; the swaps
# it has to reject miss by 40 to 900 per cent.
TRUE_BAND = 0.80          # |log(codps / census)| for a correctly paired concelho
SWAP_FLOOR = 0.35         # the smaller of the two swapped errors must exceed this
MIN_SPEARMAN = 0.95


def fold(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return "".join(ch for ch in s.lower() if ch.isalnum())


def fetch():
    os.makedirs(RAW, exist_ok=True)
    for name, url in DOWNLOADS.items():
        dst = os.path.join(RAW, name)
        if os.path.exists(dst):
            print(f"  have {name} ({os.path.getsize(dst):,} bytes)")
            continue
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=300) as r, open(dst + ".part", "wb") as f:
            f.write(r.read())
        os.replace(dst + ".part", dst)
        print(f"  got  {name} ({os.path.getsize(dst):,} bytes)")


def check_swaps(census, codps, names):
    """Each ambiguous pair, swapped, must break the population agreement.

    This is the assertion the file exists for. A swap preserves the national total, every
    category total and every row count, so nothing else in the pipeline can see it.
    """
    print(f"\n  witness 3 — the {len(AMBIGUOUS)} ambiguous pairings, and what a swap costs:")
    import math
    for a, b in AMBIGUOUS:
        true_a = abs(math.log(codps[a] / census[a]))
        true_b = abs(math.log(codps[b] / census[b]))
        swap_a = abs(math.log(codps[b] / census[a]))
        swap_b = abs(math.log(codps[a] / census[b]))
        print(f"      {names[a]:<30}{census[a]:>8,} census vs {codps[a]:>8,} COD-PS "
              f"({100 * (math.exp(true_a if codps[a] > census[a] else -true_a) - 1):+6.1f}%)")
        print(f"      {names[b]:<30}{census[b]:>8,} census vs {codps[b]:>8,} COD-PS "
              f"({100 * (math.exp(true_b if codps[b] > census[b] else -true_b) - 1):+6.1f}%)")
        print(f"        swapped: {math.exp(swap_a) - 1:+.1%} and {math.exp(swap_b) - 1:+.1%}")
        if max(true_a, true_b) >= TRUE_BAND:
            raise SystemExit(f"{a}/{b}: the pairing this file asserts is already outside "
                             f"the band ({true_a:.2f}, {true_b:.2f}) — check PCODE")
        if min(swap_a, swap_b) <= SWAP_FLOOR:
            raise SystemExit(f"{a}/{b}: swapping them is NOT distinguishable "
                             f"({swap_a:.2f}, {swap_b:.2f}) — this witness has stopped "
                             "working and the join needs another one. Do not delete it.")


def main():
    if "--fetch" in sys.argv:
        fetch()

    import cv as cvmod

    zpath = os.path.join(RAW, "cpv_admin_boundaries.shp.zip")
    if not os.path.exists(zpath):
        raise SystemExit(f"{zpath} missing — run with --fetch")
    os.makedirs(SHP_DIR, exist_ok=True)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(SHP_DIR)

    shp = os.path.join(SHP_DIR, "cpv_admin1.shp")
    g = gpd.read_file(shp, engine="fiona")
    if len(g) != EXPECTED_UNITS:
        raise SystemExit(f"{len(g)} ADM1 features, expected {EXPECTED_UNITS} — COD has "
                         "re-cut Cabo Verde")
    if g.crs is None or g.crs.to_epsg() != 4326:
        raise SystemExit(f"unexpected CRS {g.crs}")
    minx, miny, maxx, maxy = g.total_bounds
    if not (-26.0 < minx and maxx < -22.0 and 14.0 < miny and maxy < 18.0):
        raise SystemExit(f"the layer's bbox {g.total_bounds} is not Cabo Verde")
    print(f"read {shp}: {len(g)} concelhos, {g.crs}")

    g["pcode"] = g["adm1_pcode"].astype(str).str.strip()
    ine = {p: n for p, n in cvmod.PCODE.values()}
    if sorted(g["pcode"]) != sorted(ine):
        raise SystemExit(f"COD's pcodes and cv.py's PCODE differ: "
                         f"{sorted(set(g['pcode']) ^ set(ine))}")

    # ---- witness 1: which names match, and the three that cannot ----
    cod_by_fold = dict(zip(g["adm1_name"].map(fold), g["pcode"]))
    exact = {p: n for p, n in ine.items() if cod_by_fold.get(fold(n)) == p}
    unmatched = sorted(set(ine) - set(exact))
    print(f"  witness 1 — {len(exact)} of {EXPECTED_UNITS} INE names fold to their own COD "
          f"name exactly; the {len(unmatched)} that do not are {unmatched}")
    if sorted(unmatched) != sorted(COD_SHORT):
        raise SystemExit(f"the unmatched names are {unmatched}, not {sorted(COD_SHORT)} — "
                         "COD has renamed something and the ambiguity has moved")
    for p, short in COD_SHORT.items():
        got = g.loc[g["pcode"] == p, "adm1_name"].iloc[0]
        if fold(got) != fold(short):
            raise SystemExit(f"{p}: COD now calls it {got!r}, not {short!r}")
        twin = next(q for a, b in AMBIGUOUS for q in (a, b)
                    if q != p and p in (a, b))
        print(f"      {p} COD {short!r} / INE {ine[p]!r}, whose twin is "
              f"{twin} {ine[twin]!r}")

    # ---- witness 2: populations, joined on the pcode, ranked ----
    pop = pd.read_csv(os.path.join(RAW, "cpv_admpop_adm1_2022B.csv"), encoding="utf-8-sig")
    pop["pcode"] = pop["ADM1_PCODE"].astype(str).str.strip()
    if len(pop) != EXPECTED_UNITS or set(pop["pcode"]) != set(g["pcode"]):
        raise SystemExit("COD-PS ADM1 does not cover the same 22 pcodes as COD-AB")
    codps = dict(zip(pop["pcode"], pop["T_TL"].astype("int64")))

    t = cvmod.read()
    census = {p: t[c]["population"] for c, (p, _) in cvmod.PCODE.items()}
    s = pd.Series(codps).corr(pd.Series(census), method="spearman")
    print(f"\n  witness 2 — COD-PS 2022 ({sum(codps.values()):,}) against the 2021 census "
          f"({sum(census.values()):,}): Spearman {s:.4f} over {EXPECTED_UNITS} concelhos")
    if s < MIN_SPEARMAN:
        raise SystemExit(f"the population ranks agree at only {s:.3f} — the join is "
                         "probably permuted")
    top = sorted(census, key=census.get, reverse=True)[:2]
    if top != sorted(codps, key=codps.get, reverse=True)[:2]:
        raise SystemExit("the two largest concelhos differ between the census and COD-PS")
    print(f"      largest two agree: {ine[top[0]]} and {ine[top[1]]}")

    check_swaps(census, codps, ine)

    g["unit"] = g["pcode"]
    g["geo_id"] = g["pcode"]
    g["name"] = g["pcode"].map(ine)          # INE's unambiguous name, not COD's short one
    g["cod_name"] = g["adm1_name"]
    g["island"] = g["island"]
    g["pop"] = g["pcode"].map(census).astype("int64")
    g["pop15"] = g["pcode"].map({p: t[c]["total15"] for c, (p, _) in cvmod.PCODE.items()})

    os.makedirs(OUT_DIR, exist_ok=True)
    out = g[["unit", "name", "cod_name", "island", "pcode", "geo_id", "pop", "pop15",
             "geometry"]]
    out.to_file(OUT, layer="concelhos", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out)} polygons, {out['pop'].sum():,} people)")

    lut = out.drop(columns="geometry").sort_values("pcode")
    lut.to_csv(LOOKUP, index=False, encoding="utf-8")
    print(f"wrote {LOOKUP} ({len(lut)} rows)")
    for _, r in lut.iterrows():
        print(f"    {r['pcode']}  {r['name']:<30}{r['island']:<14}{r['pop']:>8,}")


if __name__ == "__main__":
    main()
