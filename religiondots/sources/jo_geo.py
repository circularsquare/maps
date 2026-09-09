"""Jordan — boundaries and populations for the 12 governorates.

Writes data/geo/jo/jo_governorates.gpkg and data/geo/jo/jo_lookup.csv.

Two sources, and the second one is the reason this country does not need a COD file:

  * **boundaries** — geoBoundaries gbOpen `JOR/ADM1`, pinned to commit `9469f09`, twelve
    features carrying `shapeISO` = the ISO 3166-2 subdivision code.
  * **populations and areas** — **the Department of Statistics' own governorate estimate**,
    `PopulationEstimates.xlsx` on dosweb.dos.gov.jo, Table 2.2 (population at end-2025) and
    Table 2.6 (area in km2 and density) in the same workbook.

## THERE IS NO COD FILE FOR JORDAN, WHICH IS WHY THIS IS DOS'S OWN

`cod-ab-jor`, `cod-ps-jor` and `cod-em-jor` all 404 on HDX and the Jordan group there carries
no administrative or population COD of any kind — Kontur's aggregation, WhosOnFirst and
Facebook's density rasters are the whole list. So the usual fallback is not available and the
question of whether to prefer the office's own figures over COD-PS's (§9bn's Ecuador, §9bz's
Egypt) does not arise: DOS is the only published source, and it is the right one anyway. It
is dated **end-2025, released January 2026**, which is the freshest denominator on this map.

## WHAT THE 11,937,000 COUNTS, WHICH IS NOT WHAT THE SURVEY SAMPLES

DOS's estimate is of the **whole resident population**, Jordanian and not. Jordan's 2015
census counted 9,531,712 people of whom 2,918,125 were non-Jordanian, so roughly three in ten
residents are not citizens — Syrians above all, then Egyptians, Palestinians without
citizenship, Iraqis and others. `sources/jo.py` is where that is handled and `sources/jo.md`
§4 has the measurement; nothing about it changes which polygons or which totals belong here.

## THE JOIN, AND WHY THE NAMES ARE NOT THE KEY

geoBoundaries and DOS romanise three of the twelve differently — `Jerash`/`Jarash`,
`Ajloun`/`Ajlun`, `Tafilah`/`Tafiela` — and neither is wrong, so the English is not a key
(§9bz's Egypt reached the same conclusion with nine of twenty-seven disagreeing). The key is
an authored table of ISO 3166-2 codes, and it is then checked three ways:

  1. the authored table covers geoBoundaries' twelve `shapeISO` values exactly;
  2. **DOS's published area against the area of geoBoundaries' polygon**, per governorate.
     Jordan's run from 410 km2 (Jarash) to 32,832 km2 (Ma'an), a factor of 80, so a permuted
     pairing cannot survive it. This is a genuinely separate signal: DOS publishes the statute
     area and geoBoundaries measures its own geometry;
  3. **the three desert governorates must come out the three sparsest.** Ma'an, Mafraq and
     Aqaba are 74.7% of Jordan's land and 9.5% of its people; everyone else lives on the
     western highlands and in the Jordan valley. DOS's own density column says the same, so
     what is tested is that the pairing reproduces it.

And a fourth that costs nothing and is worth printing: nine of the twelve names agree
character for character on a letters-only fold, and the three that do not are named above. A
permutation would break that in nine places at once.

Usage:
    python sources/jo_geo.py --fetch    one 84 KB geojson, one 88 KB xlsx
    python sources/jo_geo.py            rebuild from data/raw/jo/
"""

import os
import re
import ssl
import sys
import urllib.request

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "jo")
OUT_DIR = os.path.join(ROOT, "data", "geo", "jo")
OUT = os.path.join(OUT_DIR, "jo_governorates.gpkg")
LOOKUP = os.path.join(OUT_DIR, "jo_lookup.csv")

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) religiondots/1.0"}

GB_URL = ("https://github.com/wmgeolab/geoBoundaries/raw/9469f09/releaseData/gbOpen/JOR/"
          "ADM1/geoBoundaries-JOR-ADM1.geojson")
ADM1 = os.path.join(RAW, "geoBoundaries-JOR-ADM1.geojson")

# DOS publishes this at the end of every year; the file name does not carry the vintage, so
# the cover sheet's date and the national total are both asserted below rather than trusted.
DOS_URL = ("https://dosweb.dos.gov.jo/databank/Population/Population_Estimares/"
           "PopulationEstimates.xlsx")
DOS_XLSX = os.path.join(RAW, "PopulationEstimates.xlsx")
DOS_TOTAL = 11_937_000          # Table 2.2's own `Total` row, end-2025
DOS_AREA_TOTAL = 88_793.512     # Table 2.6's own `Total` row, km2

# ISO 3166-2 code -> (the p-code this map mints, the name it prints, DOS's English spelling).
#
# **The p-code is Jordan's OWN governorate numbering** — 11-14 for the central region, 21-24
# for the north, 31-34 for the south — and it is not decoration. Arab Barometer waves V, VII
# and VIII code `Q1` as 800 followed by exactly this number (80011 Amman ... 80034 Aqaba),
# which is how `sources/jo.py` checks its label harmonisation against something that is not a
# name. Waves IV and VI use a different arbitrary 1..12 order, which is why the code can never
# be the pooling key; see `sources/jo.py`'s `NORM`.
GOVERNORATES = {
    "JO-AM": ("JO11", "Amman",    "Amman"),
    "JO-BA": ("JO12", "Balqa",    "Balqa"),
    "JO-AZ": ("JO13", "Zarqa",    "Zarqa"),
    "JO-MD": ("JO14", "Madaba",   "Madaba"),
    "JO-IR": ("JO21", "Irbid",    "Irbid"),
    "JO-MA": ("JO22", "Mafraq",   "Mafraq"),
    "JO-JA": ("JO23", "Jerash",   "Jarash"),
    "JO-AJ": ("JO24", "Ajloun",   "Ajlun"),
    "JO-KA": ("JO31", "Karak",    "Karak"),
    "JO-AT": ("JO32", "Tafilah",  "Tafiela"),
    "JO-MN": ("JO33", "Ma'an",    "Ma'an"),
    "JO-AQ": ("JO34", "Aqaba",    "Aqaba"),
}

# The three that geoBoundaries and DOS spell differently. Named so that a FOURTH disagreement
# is a failure rather than a shrug: three known romanisation splits is a fact about two
# gazetteers, and a fourth would mean one of them has re-cut something.
SPELLING_SPLITS = {"JO-JA", "JO-AJ", "JO-AT"}

# 74.7% of Jordan's land, 9.5% of its people. Everyone else is on the highlands and in the
# valley. DOS's own density column puts these three at 6.0, 25.9 and 36.3 per km2 against a
# next-sparsest of 54.4, so this is a measured fact and not a plausible belief (§9bz's
# witness-4 lesson: an assertion that encodes a guess fails on correct data).
DESERT = {"JO33", "JO22", "JO34"}

UTM = 32637      # UTM 37N, 36-42E: Jordan sits inside it


def fold(s):
    return re.sub(r"[^a-z]", "", str(s).lower())


def _ctx():
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _get(url, dst, magic, minsize):
    if os.path.exists(dst) and os.path.getsize(dst) > minsize:
        print(f"  have {os.path.basename(dst)} ({os.path.getsize(dst):,} bytes)")
        return
    print("  GET", url)
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=900, context=_ctx()) as r:
        data = r.read()
    # §5a: a 200 is not a download.
    if not data.startswith(magic):
        raise SystemExit(f"{url} did not return the expected file — starts {data[:24]!r}")
    with open(dst + ".part", "wb") as f:
        f.write(data)
    os.replace(dst + ".part", dst)
    print(f"  got  {os.path.basename(dst)} ({os.path.getsize(dst):,} bytes)")


def fetch():
    os.makedirs(RAW, exist_ok=True)
    _get(GB_URL, ADM1, b"{", 40_000)
    _get(DOS_URL, DOS_XLSX, b"PK", 40_000)


def dos_tables():
    """Table 2.2 (population) and Table 2.6 (area) out of DOS's workbook, by English name.

    ## ANCHOR THE HEADER AND BOUND THE TABLE, DO NOT READ BY ROW NUMBER

    Both sheets carry more than one table and DOS re-lays the workbook every year, so each
    block is bounded by its own header and its own `Total` row rather than by an offset. Sheet
    `2.7` is the case that proves it is needed: read as "every row with a name in column 7 and
    numbers in columns 3 and 4", it returns **27 rows rather than 12**, because Table 2.5's
    age-group percentages sit above Table 2.6 in the same columns and a float renders as a
    perfectly good name. `[[reference_pdf_table_geometry]]` is about PDFs and the rule is the
    same one.
    """
    xl = pd.ExcelFile(DOS_XLSX)

    def block(sheet, name_col, value_cols, header):
        df = xl.parse(sheet, header=None)
        names = df[name_col].astype(str).str.strip()
        starts = list(names[names == header].index)
        if len(starts) != 1:
            raise SystemExit(f"sheet {sheet!r} column {name_col} has {len(starts)} rows "
                             f"reading {header!r}, expected exactly 1")
        ends = [i for i in names[names == "Total"].index if i > starts[0]]
        if not ends:
            raise SystemExit(f"sheet {sheet!r} has no Total row under {header!r}")
        rows = {}
        for i in range(starts[0] + 1, ends[0] + 1):
            r = df.loc[i]
            nm = names[i]
            if nm in ("nan", ""):
                continue
            vals = [r[c] for c in value_cols]
            if any(not isinstance(v, (int, float, np.integer, np.floating)) or pd.isna(v)
                   for v in vals):
                continue
            rows[nm] = [float(v) for v in vals]
        return rows

    # sheet 2.3: col 0 Arabic name, col 3 total, col 5 English name.
    pop = block("2.3", 5, [3], "Governorate")
    # sheet 2.7: col 1 Arabic name, col 3 population, col 4 area km2, col 7 English name.
    area = block("2.7", 7, [3, 4], "Governorate")

    if "Total" not in pop or abs(pop["Total"][0] - DOS_TOTAL) > 0.5:
        raise SystemExit(f"Table 2.2's Total row reads {pop.get('Total')}, not {DOS_TOTAL:,} — "
                         "DOS has published a new vintage; read it before re-pinning")
    if "Total" not in area or abs(area["Total"][1] - DOS_AREA_TOTAL) > 0.5:
        raise SystemExit(f"Table 2.6's Total row reads {area.get('Total')}, not "
                         f"{DOS_AREA_TOTAL} km2 — DOS has re-laid the workbook")
    pop.pop("Total")
    area.pop("Total")
    if len(pop) != 12 or len(area) != 12:
        raise SystemExit(f"read {len(pop)} population rows and {len(area)} area rows, "
                         "expected 12 of each")
    if abs(sum(v[0] for v in pop.values()) - DOS_TOTAL) > 0.5:
        raise SystemExit("the twelve population rows do not sum to DOS's own Total")
    if abs(sum(v[1] for v in area.values()) - DOS_AREA_TOTAL) > 0.5:
        raise SystemExit("the twelve area rows do not sum to DOS's own Total")
    # The two tables are separate blocks in separate sheets; that they agree on every
    # governorate's population is the cheapest possible check that neither was misread.
    for nm, (p,) in pop.items():
        if nm not in area:
            raise SystemExit(f"{nm!r} is in Table 2.2 and not in Table 2.6")
        if abs(area[nm][0] - p) > 0.5:
            raise SystemExit(f"{nm}: Table 2.2 says {p:,.0f} and Table 2.6 says "
                             f"{area[nm][0]:,.0f}")
    print(f"  DOS Table 2.2 and Table 2.6 agree on all 12 governorates; "
          f"{DOS_TOTAL:,} people over {DOS_AREA_TOTAL:,.0f} km2")
    return ({nm: int(round(v[0])) for nm, v in pop.items()},
            {nm: v[1] for nm, v in area.items()})


def main():
    if "--fetch" in sys.argv:
        fetch()
    for p in (ADM1, DOS_XLSX):
        if not os.path.exists(p):
            raise SystemExit(f"{p} missing — run with --fetch")

    g = gpd.read_file(ADM1)
    if len(g) != 12:
        raise SystemExit(f"{len(g)} ADM1 features, expected 12 — geoBoundaries has re-cut "
                         "Jordan")
    if g.crs is None or g.crs.to_epsg() != 4326:
        raise SystemExit(f"unexpected CRS {g.crs}")
    print(f"read {ADM1}: {len(g)} governorates, {g.crs}")

    pop, area = dos_tables()

    # ---- witness 1: the authored table covers geoBoundaries' twelve ISO codes exactly ----
    off = sorted(set(g["shapeISO"].astype(str)) ^ set(GOVERNORATES))
    if off:
        raise SystemExit(f"geoBoundaries' shapeISO values and GOVERNORATES do not agree: {off}")
    if sorted(v[2] for v in GOVERNORATES.values()) != sorted(pop):
        raise SystemExit("GOVERNORATES' DOS spellings and Table 2.2's names do not agree: "
                         f"{sorted(set(v[2] for v in GOVERNORATES.values()) ^ set(pop))}")
    print("  witness 1 — the authored table covers all 12 shapeISO values and all 12 DOS names")

    g["iso"] = g["shapeISO"].astype(str)
    g["geo_id"] = g["iso"].map(lambda i: GOVERNORATES[i][0])
    g["name"] = g["iso"].map(lambda i: GOVERNORATES[i][1])
    g["dos_name"] = g["iso"].map(lambda i: GOVERNORATES[i][2])
    g["pop"] = g["dos_name"].map(pop).astype("int64")
    g["dos_area"] = g["dos_name"].map(area)
    g["gb_area"] = g.to_crs(UTM).geometry.area / 1e6

    # ---- witness 2: DOS's published area against geoBoundaries' own geometry ----
    rho = stats.spearmanr(g["dos_area"], g["gb_area"]).statistic
    rng = np.random.default_rng(0)
    a, b = g["dos_area"].to_numpy(), g["gb_area"].to_numpy()
    perm = np.array([stats.spearmanr(a, rng.permutation(b)).statistic for _ in range(5000)])
    beaten = int((perm >= rho).sum())
    print(f"  witness 2 — DOS area vs geoBoundaries area over 12: rho = {rho:+.3f}, and "
          f"{beaten} of 5,000 random pairings reach it (best random {perm.max():+.3f})")
    g["area_ratio"] = g["gb_area"] / g["dos_area"]
    worst = g.reindex((g["area_ratio"] - 1).abs().sort_values(ascending=False).index).head(3)
    for _i, r in worst.iterrows():
        print(f"      {r['name']:<10} DOS {r['dos_area']:>10,.0f} km2   geoBoundaries "
              f"{r['gb_area']:>10,.0f} km2   {r['area_ratio']:.2f}x")
    if beaten:
        raise SystemExit("DOS's areas and geoBoundaries' do not pin this pairing — a "
                         "permutation of the codes would do as well. STOP.")

    # ---- witness 3: and the shape of the population on the ground ----
    g["density"] = g["pop"] / g["gb_area"]
    order = g.sort_values("density")
    print("  witness 3 — sparsest three: "
          + ", ".join(f"{n} {d:.1f}/km2"
                      for n, d in zip(order["name"][:3], order["density"][:3]))
          + f"; next is {order['name'].iloc[3]} at {order['density'].iloc[3]:.1f}/km2")
    if set(order["geo_id"][:3]) != DESERT:
        raise SystemExit(f"the three sparsest governorates are {sorted(order['name'][:3])}, "
                         "not the three desert ones — the population join is permuted")
    print(f"    densest {order['name'].iloc[-1]!r} at {order['density'].iloc[-1]:,.0f}/km2")

    # ---- witness 4: the names, which are NOT the key and are therefore free evidence ----
    same = {i for i in GOVERNORATES
            if fold(g.loc[g["iso"] == i, "shapeName"].iloc[0]) == fold(GOVERNORATES[i][2])}
    differ = sorted(set(GOVERNORATES) - same)
    print(f"  witness 4 — {len(same)} of 12 names agree letter for letter; the {len(differ)} "
          f"that do not are "
          + ", ".join(f"{GOVERNORATES[i][1]} "
                      f"({g.loc[g['iso'] == i, 'shapeName'].iloc[0]}/{GOVERNORATES[i][2]})"
                      for i in differ))
    if set(differ) != SPELLING_SPLITS:
        raise SystemExit(f"the names that disagree are {differ}, not the three known "
                         f"romanisation splits {sorted(SPELLING_SPLITS)} — read them before "
                         "trusting the pairing")

    g["unit"] = g["geo_id"]
    os.makedirs(OUT_DIR, exist_ok=True)
    out = g[["unit", "name", "iso", "geo_id", "pop", "geometry"]]
    out.to_file(OUT, layer="governorates", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out)} polygons)")

    lut = (g[["geo_id", "unit", "name", "iso", "pop", "dos_area"]]
           .sort_values("geo_id").reset_index(drop=True))
    lut.to_csv(LOOKUP, index=False, encoding="utf-8")
    print(f"wrote {LOOKUP} ({len(lut)} rows, {lut['pop'].sum():,} people)")
    print(lut.sort_values("pop", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
