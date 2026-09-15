"""Nauru — 2021 Population and Housing Census, table G-7, religious affiliation.

    python sources/nr.py --fetch     the census workbook + the Kontur extract, ~85 KB
    python sources/nr.py             normalise + build the placement layer

Writes:
    data/normalized/nr.csv           one row per religion, geo_level `country`
    data/geo/nr/nr_hexes.gpkg        Kontur 400 m population hexagons, unit = NR

**THE MICROSTATE TIER, BUILT FROM THE OFFICE RATHER THAN THE YEARBOOK.** Nauru is 11,680
people, so at spec §4's 1 dot = 1,000 it draws twelve dots and §9bf's reasoning applies
unchanged: where inside the island a dot lands asserts nothing, so a national table is a
complete source rather than a coarse one. What is different from `micro.py`'s nine is the
instrument. **Nauru is not in UNSD's table 28 at all** (`oracle.py --list`, 117 countries,
2026-09-11), so the source here is the Nauru Bureau of Statistics' own census workbook, and
it is deeper than the Yearbook is for anyone in that tier: **nineteen categories**, against
four for the Marshall Islands and nine for Palau.

**THE OFFICE MOVED AND THE FILES ARE BEHIND A PLUGIN, NOT BEHIND A WALL.** `stats.gov.nr` is
the Nauru Bureau of Statistics; `nauru.prism.spc.int`, the host queue.md carried, 301-redirects
to it. It runs **WP File Download**, the same plugin as Fiji (§9bd) and PNG (§11ab), and its
AJAX route is unauthenticated in the same way — `id=0` returns the whole library, 131 files,
ten to a page. The 2021 census sits in category 49. `sources/nr.md` has the sweep.

**THE TABLE IS DEEPER THAN THE QUESTIONNAIRE'S OWN CODE LIST, AND THAT IS THE INTERESTING
FACT ABOUT IT.** Question 307 of the 2021 questionnaire offers **ten** pre-coded answers — No
Religion, Nauruan Congregational, Catholic, Assemblies of God, Nauru Independent, Pacific Light
House, Seven Day Adventist, Baptist, Do not wish to answer, Other religion — plus a free-text
`307_oth` reached only when the answer is `Other religion`. Table G-7 prints **nineteen** rows.
The extra nine (Protestant, Shalosh Pentecostal, Fishers of Men, Brethren, FOM Pentecostal,
Christ Embassy, Hinduism, Fundamental Christian, Methodist) are the office's back-coding of
those write-ins, and the 98 still filed as `Other religion` are what it did not code.
Reading the questionnaire is what establishes this ([[reference_census_questionnaire]]); the
table alone reads like a nineteen-way pre-coded list, which it is not.

**THE PARTITION IS EXACT.** Nineteen categories sum to 11,680, which is the census's own total
population, with a difference of zero. Asserted below, so a re-publication that revises a
figure fails the build rather than moving the map quietly.

**WHAT IS NOT DRAWN: 57 people, 0.49%.** `Do not wish to answer` is the refusal cell, and the
office itself relabels it `Not stated` in the analytical report's table 25. It is excluded in
`taxonomy/nr2021.py` rather than dropped here, so `gap_share.py` can see it.

**THERE IS A DISTRICT TABLE AND IT IS DELIBERATELY NOT USED.** The Person Tables volume's
table 19 gives religion by district — but only for the 11,215 Nauruan citizens and dual
citizens, and with the five coded write-in bodies folded back into `Other religion`. Two
reasons not to take it: the country draws twelve dots over fifteen districts, so no placement
could express it; and it would trade the whole population and nineteen categories for 96% of
the population and twelve. `sources/nr.md` records the district shares, which are real and
striking, for whoever wants them. The two tables reconcile to the person, which is how the
citizen table is known to be a clean subset of G-7.
"""

import csv
import gzip
import os
import shutil
import ssl
import sys
import urllib.request

os.environ.setdefault("OMP_NUM_THREADS", "6")          # [[feedback_cap_cpu]]
os.environ.setdefault("OPENBLAS_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "nr")

# WP File Download's download route: /download/<catid>/<catslug>/<fileid>/<anything>.<ext>.
# The trailing slug is free; the category id, the category slug and the file id are not.
WORKBOOK_URL = ("https://stats.gov.nr/download/49/2021/182/"
                "population-housing-census-2021-tables-vol1.xlsx")
WORKBOOK = "population-housing-census-2021-tables-vol1.xlsx"
SHEET = "G-7"                  # Total population by religious affiliation and sex
YEAR = 2021

KONTUR_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
              "kontur_population_NR_20231101.gpkg.gz")
KONTUR_VINTAGE = "2023-11-01"

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

# Pinned from the workbook as published, 2026-09-11. The point of pinning is spec §12's:
# a re-publication that revises a figure should fail the build, not move the map quietly.
EXPECTED_TOTAL = 11_680
EXPECTED_CATEGORIES = 19

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"}


def _ctx():
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _get(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"  have {os.path.basename(dest)}")
        return
    body = urllib.request.urlopen(urllib.request.Request(url, headers=UA),
                                  timeout=180, context=_ctx()).read()
    tmp = dest + ".part"
    with open(tmp, "wb") as fh:
        fh.write(body)
    os.replace(tmp, dest)                                   # [[reference_wb_truncates]]
    print(f"  got {os.path.basename(dest)} ({len(body):,} bytes)")


def fetch():
    os.makedirs(RAW, exist_ok=True)
    _get(WORKBOOK_URL, os.path.join(RAW, WORKBOOK))
    gz = os.path.join(RAW, "kontur_population_NR_20231101.gpkg.gz")
    _get(KONTUR_URL, gz)
    gpkg = gz[:-3]
    if not (os.path.exists(gpkg) and os.path.getsize(gpkg) > 0):
        with gzip.open(gz, "rb") as src, open(gpkg + ".part", "wb") as dst:
            shutil.copyfileobj(src, dst)
        os.replace(gpkg + ".part", gpkg)


def read_g7():
    """Sheet G-7 -> [(category, total), ...] in the workbook's own order.

    The sheet's shape, which is every G-sheet's shape in this workbook:

        row 2   Table G-7: Total population by religious affiliation and sex: 2021 PHC Nauru
        row 3   |          | Sex
        row 4   |          | Total | Male | Female
        row 5   TOTAL      | 11680 |  5893 |  5787
        row 6   Religion                                 <- the stub header, no figures
        row 7+  <category>  | total | male | female

    Anchored on the `TOTAL` and `Religion` labels rather than on row numbers, because this
    office renumbers its sheets between editions the way Korail renames its files
    ([[reference_korail_yearbook]]).
    """
    import openpyxl

    path = os.path.join(RAW, WORKBOOK)
    if not os.path.exists(path):
        raise SystemExit(f"missing {path} — run with --fetch")
    wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    if SHEET not in wb.sheetnames:
        raise SystemExit(f"no sheet {SHEET!r}; workbook has {wb.sheetnames}")
    rows = [[c for c in r] for r in wb[SHEET].iter_rows(values_only=True)]
    wb.close()

    title = next((str(c) for r in rows for c in r
                  if c is not None and str(c).startswith("Table G-7")), "")
    if "religious affiliation" not in title.lower():
        raise SystemExit(f"{SHEET} is not the religion table any more: {title!r}")

    stated = None
    cats, started = [], False
    for r in rows:
        label = "" if not r or r[0] is None else str(r[0]).strip()
        value = r[1] if len(r) > 1 else None
        if label.upper() == "TOTAL" and isinstance(value, (int, float)):
            stated = int(value)
            continue
        if label.lower() == "religion":
            started = True                    # the stub header; categories follow it
            continue
        if started and label and isinstance(value, (int, float)):
            cats.append((label, int(value)))

    if stated is None:
        raise SystemExit(f"{SHEET}: no TOTAL row")
    if not cats:
        raise SystemExit(f"{SHEET}: no categories under the `Religion` stub")
    return title, stated, cats


def normalise():
    title, stated, cats = read_g7()

    if stated != EXPECTED_TOTAL:
        raise SystemExit(f"nr: the workbook now says {stated:,}, this build expects "
                         f"{EXPECTED_TOTAL:,}")
    if len(cats) != EXPECTED_CATEGORIES:
        raise SystemExit(f"nr: {len(cats)} categories, expected {EXPECTED_CATEGORIES}")
    drift = sum(n for _, n in cats) - stated
    if drift:
        raise SystemExit(f"nr: categories sum to {sum(n for _, n in cats):,} against a "
                         f"stated {stated:,}, a difference of {drift:+,}")
    names = [c for c, _ in cats]
    if len(set(names)) != len(names):
        raise SystemExit(f"nr: duplicate category labels in {SHEET}: {names}")

    print(f"  {title}")
    print(f"  {len(cats)} categories, {stated:,} people, partition exact")
    for cat, n in sorted(cats, key=lambda kv: -kv[1]):
        print(f"      {cat:<32}{n:>7,}  {100 * n / stated:5.2f}%")

    rows = [dict(geo_id="NR", geo_level="country", geo_name="Nauru",
                 source_category=cat, count=n, basis="self_id", year=YEAR,
                 source_id=f"nbs_phc2021_g7_{YEAR}",
                 note="Nauru Bureau of Statistics, 2021 Population and Housing Census, "
                      "Tables Vol 1, sheet G-7")
            for cat, n in sorted(cats, key=lambda kv: -kv[1])]

    out = os.path.join(ROOT, "data", "normalized", "nr.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".part"
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, out)
    return stated


def geometry(census_total):
    """Kontur hexes -> data/geo/nr/nr_hexes.gpkg, every hex on the one unit.

    THERE IS NO JOIN. `unit` is `NR` on every row of both files, so §12's first two shapes of
    failure — a name matched to the wrong twin, a code matched to the wrong vintage — cannot
    arise here at all.
    """
    import geopandas as gpd

    gpkg = os.path.join(RAW, "kontur_population_NR_20231101.gpkg")
    if not os.path.exists(gpkg):
        raise SystemExit(f"missing {gpkg} — run with --fetch")
    hexes = gpd.read_file(gpkg)
    popcol = next((c for c in hexes.columns if c.lower() == "population"), None)
    if popcol is None:
        raise SystemExit(f"nr: no population column in {list(hexes.columns)}")
    hexes = hexes.rename(columns={popcol: "pop"})
    hexes = hexes[hexes["pop"] > 0].copy()
    if hexes.empty:
        raise SystemExit("nr: Kontur has no populated hexes")
    hexes["unit"] = "NR"
    hexes["cellcode"] = "NR:" + hexes.index.astype(str)
    hexes = hexes.to_crs("EPSG:4326")

    # Nauru is one island and does not go near the antimeridian, but it is 166 deg E and the
    # Pacific is where this bites ([[reference_antimeridian]], §9bd), so assert the bbox.
    w, s, e, n = hexes.total_bounds
    if not (160 < w < 170 and 160 < e < 170 and -1.0 < s < 0 and -1.0 < n < 0):
        raise SystemExit(f"nr: Kontur bbox {w:.3f},{s:.3f},{e:.3f},{n:.3f} is not Nauru")

    geo = os.path.join(ROOT, "data", "geo", "nr")
    os.makedirs(geo, exist_ok=True)
    out = os.path.join(geo, "nr_hexes.gpkg")
    hexes[["cellcode", "unit", "pop", "geometry"]].to_file(out, layer="hexes", driver="GPKG")

    k = float(hexes["pop"].sum())
    ratio = k / census_total
    flag = "" if 0.5 <= ratio <= 2.0 else "   <-- CHECK"
    print(f"      {len(hexes):>6,} hexes, Kontur {k:>9,.0f} vs census {census_total:>9,} "
          f"({KONTUR_VINTAGE}, ratio {ratio:.2f}){flag}")


def main():
    if "--fetch" in sys.argv:
        fetch()
    total = normalise()
    geometry(total)


if __name__ == "__main__":
    main()
