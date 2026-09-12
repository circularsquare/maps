"""Iraq — boundaries and populations for the 18 governorates.

Writes data/geo/iq/iq_governorates.gpkg and data/geo/iq/iq_lookup.csv.

Two sources, and both of them are the state's own:

  * **boundaries** — OCHA's **COD-AB for Iraq** on HDX, `irq_admin1.shp` inside
    `irq_admin_boundaries.shp.zip`, eighteen features carrying the official `IQG01`-`IQG18`
    p-codes and each governorate's Arabic name. The layer is the Central Statistical
    Organisation's own, valid from 2019-06-03.
  * **populations** — the **2024 census**, enumerated 20-21 November 2024, 46,118,793 people,
    the first full count since 1987. COSIT prints it as Table 3/2 of the Annual Statistical
    Abstract's census chapter, `cosit.gov.iq/documents/AAS2024/02.pdf`, by governorate crossed
    with urban/rural and sex.
  * **areas**, used only as a witness on the join — Table 1/1 A of the same abstract's first
    chapter, sourced by COSIT to the Ministry of Water Resources' General Survey Authority.

## WHY NOT geoBoundaries, WHICH IS WHAT JORDAN AND MOST OF THIS MAP USE

`gbOpen/IRQ/ADM1` also has eighteen features with ISO 3166-2 codes and is a much easier
download, and its **Baghdad polygon is 912 km2 against Iraq's own published 4,555 and COD-AB's
5,100**. It is drawing something close to the built-up city and handing the rest of the
governorate to Babil, Diyala and Salah al-Din, whose polygons are correspondingly 1.5x, 1.06x
and 1.08x their statute areas. Baghdad is **21.2% of Iraq's population**, so that is not a
tidy-up: it is a fifth of the country's dots crammed into a fifth of the right polygon, with
the overflow drawn in three neighbours it does not belong to. COD-AB reproduces every
published area to a few per cent apart from the Najaf/Anbar desert boundary, which is a real
disagreement between two Iraqi authorities rather than a defect in either file.

## THE 2024 CENSUS IS THE DENOMINATOR, NOT COD-PS

COD-PS for Iraq is a projection series; the census is a count taken fourteen months before
this was written, and COSIT publishes it by governorate itself. §9bn's Ecuador rule and §9bz's
Egypt one both point the same way when the office's own figure is the fresher measurement.

## HALABJA IS THE NINETEENTH GOVERNORATE AND IS NOT DRAWN SEPARATELY

Iraq's parliament made Halabja a governorate in its own right in 2014, splitting it out of
Sulaymaniyah. **Neither of the two files here has it**: COD-AB's ADM1 is eighteen features and
the 2024 census tabulates eighteen governorates, with Halabja's people inside
Al-Sulaymaniyah's 2,401,724. The Arab Barometer's `Q1` is also eighteen. So all three tiers
agree and nothing is being lost; what would be wrong is drawing a nineteenth unit that no
input carries.

Usage:
    python sources/iq_geo.py --fetch    one 1.2 MB shapefile zip, two COSIT PDFs
    python sources/iq_geo.py            rebuild from data/raw/iq/
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
RAW = os.path.join(ROOT, "data", "raw", "iq")
OUT_DIR = os.path.join(ROOT, "data", "geo", "iq")
OUT = os.path.join(OUT_DIR, "iq_governorates.gpkg")
LOOKUP = os.path.join(OUT_DIR, "iq_lookup.csv")

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) religiondots/1.0"}

COD_URL = ("https://data.humdata.org/dataset/488bb3cd-3ce9-49d3-862a-3ce7975c63e1/resource/"
           "1d1ed1f3-a295-47b6-800d-356dc1036731/download/irq_admin_boundaries.shp.zip")
COD_ZIP = os.path.join(RAW, "irq_admin_boundaries.shp.zip")
COD_LAYER = "irq_admin1.shp"

# The Annual Statistical Abstract 2024-2025, chapter by chapter. 01 is the physical-features
# chapter and carries the areas; 02 is the census chapter and carries the population.
AAS_BASE = "https://cosit.gov.iq/documents/AAS2024/"
AREA_PDF = os.path.join(RAW, "AAS2024_01.pdf")
POP_PDF = os.path.join(RAW, "AAS2024_02.pdf")

# COSIT's own totals, asserted so a re-issue is a failure here rather than a quiet re-levelling.
CENSUS_TOTAL = 46_118_793          # Table 3/2's own Total row, and Table 2/2's
IRAQ_AREA = 435_052.0              # Table 1/1 A's `Total of Iraq`, including territorial waters
TERRITORIAL_WATERS = 924.0         # its own row, which is not a governorate

# **THE AREA TABLE IS PRINTED THREE TIMES IN THREE VINTAGES**, on consecutive pages: `FOR
# 2024`, `FOR 2022`, `FOR 2023`, in that order. Reading "the area table" by page number or by
# the first caption match gets one of them at random, and the 2022 and 2023 sheets differ from
# 2024 in a few governorates. [[reference_pdf_table_geometry]]: anchor the header, and the
# header here has to include the year.
AREA_CAPTION = "AREA OF GOVERNORATES AND NUMBER OF DISTRICTS AND SUB-DISTRICTS"
AREA_YEAR = "FOR 2024"
POP_CAPTION = "POPULATION BY GOVERNORATE, ENVIRONMENT AND SEX ACCORDING TO GENERAL CENSUS 2024"

# p-code -> (the name this map prints, COSIT's spelling in the CENSUS table, COSIT's spelling
# in the AREA table).
#
# **The two COSIT tables do not spell six of the eighteen the same way**, in the same
# publication: `maysan`/`Missan`, `Basrah`/`Al-Basrah`, `Duhok`/`Dohouk`, `Qadisiya`/`Al
# -Qadisiya`, `Al-Anbar`/`Al -Anbar`, `Al-Najaf`/`Al -Najaf`. Romanising Arabic is not a key
# and this file never treats it as one; the key is the p-code, and both spellings are written
# out here so that a re-issue that changes one is a failure rather than a silent drop.
#
# The printed names follow COD-AB's English where it is a normal English rendering and the
# census's where COD's is idiosyncratic (`Wassit` -> `Wasit`).
GOVERNORATES = {
    "IQG01": ("Anbar",         "Al-Anbar",       "Al -Anbar"),
    "IQG02": ("Basra",         "Basrah",         "Al-Basrah"),
    "IQG03": ("Muthanna",      "Al-Muthanna",    "Al -Muthanna"),
    "IQG04": ("Najaf",         "Al-Najaf",       "Al -Najaf"),
    "IQG05": ("Qadisiyyah",    "Qadisiya",       "Al -Qadisiya"),
    "IQG06": ("Sulaymaniyah",  "AL-Sulaimaniya", "Al-Sulaimaniya"),
    "IQG07": ("Babil",         "Babylon",        "Babylon"),
    "IQG08": ("Baghdad",       "Baghdad",        "Baghdad"),
    "IQG09": ("Duhok",         "Duhok",          "Dohouk"),
    "IQG10": ("Diyala",        "Diala",          "Diala"),
    "IQG11": ("Erbil",         "Erbil",          "Erbil"),
    "IQG12": ("Karbala",       "Kerbela",        "Kerbela"),
    "IQG13": ("Kirkuk",        "Kirkuk",         "Kirkuk"),
    "IQG14": ("Maysan",        "maysan",         "Missan"),
    "IQG15": ("Nineveh",       "Ninevah",        "Ninevah"),
    "IQG16": ("Salah al-Din",  "Salah AL-Deen",  "Salah AL-Deen"),
    "IQG17": ("Dhi Qar",       "Thi Qar",        "Thi-Qar"),
    "IQG18": ("Wasit",         "Wasit",          "Wasit"),
}

# The three sparsest, and they are sparse because they are the western and southern desert:
# Anbar and Muthanna are 40% of Iraq's land between them, and Najaf's polygon runs from the
# shrine city out to the Saudi border. Measured on the census and COD-AB below, 16, 20 and 49
# people per km2 against a next-sparsest of 71. Baghdad is the densest by a factor of four.
DESERT = {"IQG01", "IQG03", "IQG04"}
DENSEST = "IQG08"

# Lambert azimuthal equal-area on Iraq's own centre. Iraq straddles UTM 37N and 38N, so a UTM
# area is wrong at one edge or the other by enough to matter in a table this file then
# rank-correlates against a published one.
LAEA = "+proj=laea +lat_0=33 +lon_0=44 +datum=WGS84 +units=m +no_defs"


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
    _get(COD_URL, COD_ZIP, b"PK", 400_000)
    _get(AAS_BASE + "01.pdf", AREA_PDF, b"%PDF", 400_000)
    _get(AAS_BASE + "02.pdf", POP_PDF, b"%PDF", 400_000)


def _page(path, caption, also=None):
    """The one page carrying `caption` (and `also`), as a list of lines.

    [[reference_pdf_truncated_at_source]]: a PyMuPDF open that returns page_count=0 is what a
    damaged download looks like, so the count is asserted rather than the absence of an
    exception.
    """
    import fitz

    doc = fitz.open(path)
    if doc.page_count == 0:
        raise SystemExit(f"{path} opened with zero pages — the download is damaged")
    hits = []
    for i in range(doc.page_count):
        flat = " ".join(doc[i].get_text().split())
        if caption in flat and (also is None or also in flat):
            hits.append(i)
    if len(hits) != 1:
        raise SystemExit(f"{os.path.basename(path)} has {len(hits)} pages carrying "
                         f"{caption!r}{'' if also is None else ' + ' + repr(also)}: {hits}. "
                         "The abstract prints the area table three times in three vintages; "
                         "if this is that table, the year anchor is missing or has moved.")
    print(f"  {os.path.basename(path)} page {hits[0]}: {caption[:52]}...")
    return doc[hits[0]].get_text().splitlines()


_NUM = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*$")


def _rows(lines, names, want):
    """Walk a table page and hand back {English name: [the numbers printed before it]}.

    The abstract's tables extract as a run of value lines followed by the row's English name,
    with the Arabic name glued to the FRONT of the first value (`دهوك641816`), so the row is
    read backwards from its English label rather than forwards from its Arabic one. Anything
    between two labels that does not end in a number is dropped, which is what removes the
    column headers and the footnotes.

    `want` is how many values each row must have, and the row takes the LAST `want` of
    whatever has accumulated. That is not laziness: the area table has three subtotal rows
    inside it (non-Kurdistan, Kurdistan, Iraq) whose values would otherwise be prepended to
    the next governorate's, and taking the tail drops them. A row that has FEWER than `want`
    raises, because a short row is the tell that the extraction has merged two cells and the
    numbers are then not the columns they look like.

    **A LINE IS ONLY A ROW LABEL IF IT CARRIES NO ARABIC AND NO DIGITS.** Without that the
    census page's column header, which extracts as `Total الذكور/`, folds to `total` and is
    read as the table's Total row with one value in front of it. Every real label on these
    pages is a bare Latin string on a line of its own and every real value line either ends in
    a digit or is glued to the Arabic name.
    """
    out, buf = {}, []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if not re.search(r"[؀-ۿ\d]", s) and fold(s) in names:
            name = names[fold(s)]
            if name in out:
                raise SystemExit(f"{name!r} appears twice on this page")
            if len(buf) < want:
                raise SystemExit(f"{name!r} has {len(buf)} values before it, not {want}: {buf}")
            out[name] = buf[-want:]
            buf = []
            continue
        m = _NUM.search(s)
        buf = buf + [float(m.group(1).replace(",", ""))] if m else buf
    return out


def census_population():
    """Table 3/2: the 2024 census, by governorate, urban/rural x sex, nine values a row."""
    names = {fold(v[1]): k for k, v in GOVERNORATES.items()}
    if len(names) != len(GOVERNORATES):
        raise SystemExit("two governorates share a census spelling")
    rows = _rows(_page(POP_PDF, POP_CAPTION), names, want=9)
    missing = sorted(set(GOVERNORATES) - set(rows))
    if missing:
        raise SystemExit(f"the census table has no row for {missing} — COSIT has re-spelled a "
                         "governorate, or the extraction has changed")
    pop = {k: int(rows[k][8]) for k in GOVERNORATES}
    urban = {k: int(rows[k][2]) for k in GOVERNORATES}
    # The table's own Total row is NOT read back off the page: it extracts as the bare string
    # `Total`, which the census page also uses three times inside its column header, and a
    # label that ambiguous is worth less than the constant. `CENSUS_TOTAL` is that Total row,
    # and the abstract states the same 46,118,793 twice more — Table 2/2's age-group total on
    # the previous page, and the 1989-2024 series' last row in thousands.
    if sum(pop.values()) != CENSUS_TOTAL:
        raise SystemExit(f"the eighteen governorates sum to {sum(pop.values()):,}, not "
                         f"COSIT's own {CENSUS_TOTAL:,}")
    # Urban + rural = total, per governorate, which is the cheapest check that the nine
    # columns were read as the nine columns and not shifted by one.
    for k in GOVERNORATES:
        if int(rows[k][2]) + int(rows[k][5]) != pop[k]:
            raise SystemExit(f"{k}: urban {rows[k][2]:,.0f} + rural {rows[k][5]:,.0f} is not "
                             f"the printed total {pop[k]:,}")
    print(f"  census Table 3/2: 18 governorates, {CENSUS_TOTAL:,} people, urban+rural checks "
          "out on every one")
    return pop, urban


def statute_area():
    """Table 1/1 A, the 2024 vintage: area km2, % of Iraq, districts, sub-districts."""
    names = {fold(v[2]): k for k, v in GOVERNORATES.items()}
    if len(names) != len(GOVERNORATES):
        raise SystemExit("two governorates share an area-table spelling")
    rows = _rows(_page(AREA_PDF, AREA_CAPTION, also=AREA_YEAR), names, want=4)
    missing = sorted(set(GOVERNORATES) - set(rows))
    if missing:
        raise SystemExit(f"the area table has no row for {missing}")
    area = {k: rows[k][0] for k in GOVERNORATES}
    total = sum(area.values()) + TERRITORIAL_WATERS
    if abs(total - IRAQ_AREA) > 1.0:
        raise SystemExit(f"the eighteen areas plus {TERRITORIAL_WATERS:,.0f} km2 of "
                         f"territorial waters come to {total:,.0f} km2, not COSIT's own "
                         f"{IRAQ_AREA:,.0f}")
    # The table prints each governorate's share of Iraq as well, to one decimal, so the two
    # columns check each other without any outside number.
    for k in GOVERNORATES:
        if abs(rows[k][1] - 100.0 * area[k] / IRAQ_AREA) > 0.06:
            raise SystemExit(f"{k}: printed share {rows[k][1]}% against "
                             f"{100.0 * area[k] / IRAQ_AREA:.2f}% computed from its own area")
    print(f"  area Table 1/1 A (2024): 18 governorates summing to {IRAQ_AREA - TERRITORIAL_WATERS:,.0f} "
          f"km2 of land, and every printed % agrees with its own km2")
    return area


def main():
    if "--fetch" in sys.argv:
        fetch()
    for p in (COD_ZIP, AREA_PDF, POP_PDF):
        if not os.path.exists(p):
            raise SystemExit(f"{p} missing — run with --fetch")

    g = gpd.read_file(f"zip://{COD_ZIP}!{COD_LAYER}")
    if len(g) != 18:
        raise SystemExit(f"{len(g)} ADM1 features, expected 18 — OCHA has re-cut Iraq, most "
                         "likely by finally splitting Halabja out of Sulaymaniyah")
    if g.crs is None or g.crs.to_epsg() != 4326:
        raise SystemExit(f"unexpected CRS {g.crs}")
    print(f"read {COD_LAYER}: {len(g)} governorates, {g.crs}, valid_on "
          f"{sorted(set(g['valid_on'].astype(str)))}")

    pop, urban = census_population()
    area = statute_area()

    # ---- witness 1: the authored table covers COD-AB's eighteen p-codes exactly ----
    off = sorted(set(g["adm1_pcode"].astype(str)) ^ set(GOVERNORATES))
    if off:
        raise SystemExit(f"COD-AB's p-codes and GOVERNORATES do not agree: {off}")
    print("  witness 1 — the authored table covers all 18 p-codes and both COSIT name sets")

    g["geo_id"] = g["adm1_pcode"].astype(str)
    g["name"] = g["geo_id"].map(lambda k: GOVERNORATES[k][0])
    g["name_ar"] = g["adm1_name1"].astype(str)
    g["pop"] = g["geo_id"].map(pop).astype("int64")
    g["urban"] = g["geo_id"].map(urban).astype("int64")
    g["cosit_area"] = g["geo_id"].map(area)
    g["cod_area"] = g.to_crs(LAEA).geometry.area / 1e6

    # ---- witness 2: COSIT's published area against COD-AB's own geometry ----
    # Two authorities and two operations: the Ministry of Water Resources surveyed these, and
    # this is the area of the polygon the Central Statistical Organisation drew. They span a
    # factor of 30, from Baghdad's 4,555 km2 to Anbar's 137,808, so a permuted pairing has
    # nowhere to hide.
    rho = stats.spearmanr(g["cosit_area"], g["cod_area"]).statistic
    rng = np.random.default_rng(0)
    a, b = g["cosit_area"].to_numpy(), g["cod_area"].to_numpy()
    perm = np.array([stats.spearmanr(a, rng.permutation(b)).statistic for _ in range(5000)])
    beaten = int((perm >= rho).sum())
    print(f"  witness 2 — COSIT area vs COD-AB geometry over 18: rho = {rho:+.3f}, and "
          f"{beaten} of 5,000 random pairings reach it (best random {perm.max():+.3f})")
    g["area_ratio"] = g["cod_area"] / g["cosit_area"]
    worst = g.reindex((g["area_ratio"] - 1).abs().sort_values(ascending=False).index).head(3)
    for _i, r in worst.iterrows():
        print(f"      {r['name']:<14} COSIT {r['cosit_area']:>10,.0f} km2   COD-AB "
              f"{r['cod_area']:>10,.0f} km2   {r['area_ratio']:.2f}x")
    if beaten:
        raise SystemExit("the two area tables do not pin this pairing — a permutation of the "
                         "p-codes would do as well. STOP.")

    # ---- witness 3: the shape of the population on the ground ----
    g["density"] = g["pop"] / g["cod_area"]
    order = g.sort_values("density")
    print("  witness 3 — sparsest three: "
          + ", ".join(f"{n} {d:.0f}/km2"
                      for n, d in zip(order["name"][:3], order["density"][:3]))
          + f"; next is {order['name'].iloc[3]} at {order['density'].iloc[3]:.0f}/km2")
    if set(order["geo_id"][:3]) != DESERT:
        raise SystemExit(f"the three sparsest governorates are {sorted(order['name'][:3])}, "
                         "not the three desert ones — the population join is permuted")
    if order["geo_id"].iloc[-1] != DENSEST:
        raise SystemExit(f"the densest governorate is {order['name'].iloc[-1]}, not Baghdad")
    print(f"    densest {order['name'].iloc[-1]!r} at {order['density'].iloc[-1]:,.0f}/km2, "
          f"against {order['name'].iloc[-2]!r} at {order['density'].iloc[-2]:,.0f}/km2")

    # ---- witness 4: the names, which are NOT the key and are therefore free evidence ----
    # Weak on its own and printed as such: Iraqi governorate names romanise badly and only
    # about half of them survive a letters-only fold between two English renderings of the
    # same Arabic. What it would catch is a wholesale permutation, which would leave none.
    same = sorted(k for k in GOVERNORATES
                  if fold(g.loc[g["geo_id"] == k, "adm1_name"].iloc[0])
                  == fold(GOVERNORATES[k][1]))
    print(f"  witness 4 — {len(same)} of 18 COD-AB names match COSIT's census spelling letter "
          f"for letter; a permutation would leave 0 or 1")
    if len(same) < 6:
        raise SystemExit(f"only {len(same)} names agree, which is too few to be two "
                         "romanisations of one list — read them before trusting the pairing")

    g["unit"] = g["geo_id"]
    os.makedirs(OUT_DIR, exist_ok=True)
    out = g[["unit", "name", "name_ar", "geo_id", "pop", "geometry"]]
    out.to_file(OUT, layer="governorates", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out)} polygons)")

    lut = (g[["geo_id", "unit", "name", "name_ar", "pop", "urban", "cosit_area"]]
           .sort_values("geo_id").reset_index(drop=True))
    lut.to_csv(LOOKUP, index=False, encoding="utf-8")
    print(f"wrote {LOOKUP} ({len(lut)} rows, {lut['pop'].sum():,} people)")
    print(lut.sort_values("pop", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
