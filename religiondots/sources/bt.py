"""Bhutan: the census stopped publishing religion; three surveys of the Centre for Bhutan & GNH Studies did not.

Reads data/geo/bt/bt_lookup.csv (written by `sources/bt_geo.py`), data/raw/bt/ (the Centre's
reports) and the shared UN DESA and Pew files; writes data/normalized/bt.csv (Bhutanese, by
dzongkhag and religion) and data/normalized/bt_foreign.csv (non-Bhutanese, already on nodes).
`sources/bt.md` is the record in prose.

## WHAT IS ASKED, AND WHAT IS PUBLISHED

  * **The 2005 census asked everyone their religion** (form PHCB-2C, item "Religion": 1 Buddhism,
    2 Hinduism, 3 other; IHSN catalogue 1374's DDI, variable `q4ca`, in data/raw/bt/) **and never
    published it**: no table in the 502-page report, the factsheet, the indicators volume or the
    online table set. Its microdata are released only on a written application with a signed
    undertaking (the DDI's access conditions). The 2017 census did not ask.
  * **The Gross National Happiness surveys ask** ("Q12. What is your religion?", Buddhism, Hinduism,
    Christianity, others; 2010, 2015, 2022; Bhutanese aged 15 and over, a sample designed to be
    representative by dzongkhag). The Centre publishes the national figure only. Its weighted
    estimate from 2010 is the one population figure: "Eighty-one per cent of Bhutanese are
    Buddhists, 18% are Hindus, and 1.2% are Christians" (*An Extensive Analysis of GNH Index*, 2012,
    p.153). The 2015 and 2022 reports print only the unweighted sample (Hindu 14.5% and 12.2%,
    against 13.1% unweighted in 2010, which weighting raised to 18%), printed here as witnesses.
  * **The 2015 survey published mother tongue by dzongkhag**, weighted: Table A1.5 of *A Compass
    Towards a Just and Harmonious Society* (2016), pdf pp.300-301. Nepali (Lhotshamkha) is 18.69%
    nationally (Table 65) and 56.25% in Samtse, 3.63% in Gasa.

## THE MODEL: HINDUS PLACED BY NEPALI MOTHER TONGUE (spec §14.12, §14.10)

Bhutanese in each dzongkhag are the 2017 census's count (Table 2.6). The national level is the
Centre's 2010 estimate, 18% Hindu and 1.2% Christian, the rest Buddhist (81% printed; 80.8% is the
remainder). The Hindus are placed in proportion to the 2015 survey's Nepali mother-tongue share:

    hindu_d = 0.18 * N_d * L_d / L      L = sum(N_d * L_d) / sum(N_d)

so the national total is the Centre's, and only where it falls comes from language. Christians are
drawn at 1.2% everywhere; Buddhists are the remainder in each dzongkhag. `HINDU_PLACEMENT =
"national"` draws the Hindus at 18% everywhere instead, which is the reversal ask 046 offers.

**Why language, and what it cannot see.** Hindus in Bhutan are Lhotshampa, the Nepali-speaking
people of the southern foothills. The national figures agree: 18% Hindu (2010, weighted) and
18.69% Nepali mother tongue (2015, weighted); the 2015 report also puts the population, in passing,
at 83% Buddhist "and the remainder Hindu" (printed p.9). But the coefficient is an assumed
identity, never measured: no source crosses religion with language. It misses Lhotshampa who are
Buddhist (Tamang, Gurung and Sherpa families, many of whom answer a language other than Nepali and
sit in the `Others` column, 24% of Tsirang and 20% of Dagana) and Hindus who give another mother
tongue. **There is no independent check by dzongkhag** (spec §14.12 condition 3): the census's
2005 religion item is the only thing that could be one, and it is unpublished.

**Children.** The survey's universe is 15 and over; its shares are applied to every age.

## NON-BHUTANESE, BY NATIONALITY (Mauritania's construction, `sources/mr.py`)

The census counts 45,425 non-Bhutanese in households (Table 2.8, by dzongkhag), mostly workers on
the hydropower projects, but tabulates no nationality. UN DESA's 2020 migrant stock for Bhutan names
the origins of 49,514 of 53,612 (India 46,974); the named origins are taken as the mix and each
goes through Pew Research Center's 2020 row for that country (`taxonomy/origin_religion.py`), the
Muslim branches folded to `islam`. DESA's `Others` (4,098) are assumed to be like the named ones.
Not drawn and not residents: the 8,408 tourists and others found in hotels, and the 16,057 day
workers who cross from India each morning.

Usage:
    python sources/bt.py            rebuild data/normalized/bt.csv and bt_foreign.csv
"""

import io
import os
import re
import sys
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "taxonomy"))
os.environ.setdefault("OMP_NUM_THREADS", "6")

import pandas as pd

from afrobarometer import round_within_rows
import origin_religion as origin

RAW = os.path.join(ROOT, "data", "raw", "bt")
GNH2015 = os.path.join(RAW, "gnh_2015_compass_report.pdf")
GNH2015_URL = ("http://web.archive.org/web/2018id_/http://www.grossnationalhappiness.com/"
               "wp-content/uploads/2017/01/Final-GNH-Report-jp-21.3.17-ilovepdf-compressed.pdf")
GNH2010 = os.path.join(RAW, "gnh_extensive_analysis_2012.pdf")
GNH2010_URL = ("https://bhutanstudies.org.bt/wp-content/uploads/2025/01/"
               "An-Extensive-Analysis-of-GNH-Index.pdf")
GNH2022 = os.path.join(RAW, "gnh_2022_report.pdf")
GNH2022_URL = ("https://bhutanstudies.org.bt/wp-content/uploads/2025/01/"
               "2022-GNH-Survey-Report_compressed-1.pdf")
DESA = os.path.join(ROOT, "data", "raw", "mr",
                    "undesa_pd_2024_ims_stock_by_sex_destination_and_origin.xlsx")
PEW = os.path.join(ROOT, "data", "raw", "estimates", "pew.zip")
LOOKUP = os.path.join(ROOT, "data", "geo", "bt", "bt_lookup.csv")
OUT = os.path.join(ROOT, "data", "normalized", "bt.csv")
OUT_FOREIGN = os.path.join(ROOT, "data", "normalized", "bt_foreign.csv")

HINDU_PLACEMENT = "language"            # or "national"; ask 046
OTHER_NODE = "other.bt"

# The Centre's 2010 estimate, An Extensive Analysis of GNH Index (2012), pdf p.158 (printed p.153)
LEVEL_PAGE = 158
LEVEL_SENTENCE = ("Eighty-one per cent of Bhutanese are Buddhists, 18% are Hindus, and 1.2% are "
                  "Christians, according to GNH 2010 survey.")
HINDU, CHRISTIAN = 0.18, 0.012

# A Compass Towards a Just and Harmonious Society (2016), Table A1.5, pdf pp.300-301
A15_PAGES = (300, 301)
A15_COLUMNS = ["Dzongkha", "Cho-cha nga-chakha", "Tshangla", "Bumthangkha", "Khengkha", "Kurtop",
               "Nyenkha", "Dzala", "Dakpa", "Chali kha", "Monpakha", "Brokpa", "Lakha", "Bokha",
               "Nepali", "Lhokpu", "Gongduk", "Lepcha", "Layap", "English", "Others", "Total"]
NEPALI_NATIONAL = 18.69                 # Table 65 and A1.5's Bhutan row
# p.187's prose, the five southern dzongkhags it names; a witness on the table read
PROSE = {"Samtse": 56.25, "Tsirang": 43.75, "Sarpang": 39.80, "Chukha": 38.74, "Dagana": 34.23}

# The unweighted samples (2022 report Table 1, pdf p.44; 2015 report Table 3, pdf p.60)
SAMPLES = {2010: dict(Buddhism=6123, Hinduism=933, Christianity=83),
           2015: dict(Buddhism=5945, Hinduism=1039, Christianity=146, Others=15, None_=7),
           2022: dict(Buddhism=9392, Hinduism=1352, Christianity=230, Others=18, None_=60)}

DESA_YEAR = "2020.0"
DESA_ORIGINS = {"India": "IN", "Nepal": "NP", "China": "CN", "Japan": "JP", "Republic of Korea": "KR",
                "Bangladesh": "BD", "Pakistan": "PK", "Sri Lanka": "LK", "Myanmar": "MM",
                "Philippines": "PH", "Singapore": "SG", "Thailand": "TH", "Denmark": "DK",
                "Sweden": "SE", "United Kingdom": "GB", "Italy": "IT", "France": "FR",
                "Germany": "DE", "Netherlands": "NL", "Switzerland": "CH", "Canada": "CA",
                "United States of America": "US", "Australia": "AU", "New Zealand": "NZ"}
DESA_UNNAMED = {"Others"}
DESA_WORLD_2020 = 53_612


def pdf_lines(path, pages):
    import fitz

    with open(path, "rb") as fh:
        if b"%%EOF" not in fh.read()[-4096:]:
            raise SystemExit(f"{path} has no %%EOF trailer; the download is truncated")
    doc = fitz.open(path)
    out = []
    for p in pages:
        out += [ln.strip() for ln in doc[p - 1].get_text().splitlines() if ln.strip()]
    return out


def read_level():
    text = " ".join(" ".join(pdf_lines(GNH2010, [LEVEL_PAGE])).split())
    if LEVEL_SENTENCE not in text:
        raise SystemExit(f"p.{LEVEL_PAGE} of {GNH2010} no longer carries the 2010 sentence")
    print(f"GNH 2010 (Centre for Bhutan Studies, 2012, p.153): {HINDU:.0%} Hindu, {CHRISTIAN:.1%} "
          f"Christian, the rest Buddhist ({1 - HINDU - CHRISTIAN:.1%}; 81% printed)")


def read_a15(gnh_names):
    """{GNH dzongkhag name: {column: percent}} from Table A1.5, every row summing to 100."""
    lines = pdf_lines(GNH2015, A15_PAGES)
    if not any(" ".join(ln.split()).startswith("Table A1.5: Distribution of Dzongkhag population by "
                                               "mother tongue") for ln in lines):
        raise SystemExit("Table A1.5 is not on its pages")
    out = {}
    for name in list(gnh_names) + ["Bhutan"]:
        hits = [i for i, ln in enumerate(lines) if ln == name]
        if len(hits) != 1:
            raise SystemExit(f"A1.5: {name!r} occurs {len(hits)} times")
        vals, j = [], hits[0] + 1
        while len(vals) < len(A15_COLUMNS):
            toks = lines[j].split()
            if not all(re.fullmatch(r"[0-9]+(\.[0-9]+)?", t) for t in toks):
                raise SystemExit(f"A1.5: {name} reached {lines[j]!r} after {len(vals)} numbers")
            vals += [float(t) for t in toks]
            j += 1
        if len(vals) != len(A15_COLUMNS):
            raise SystemExit(f"A1.5: {name} has {len(vals)} numbers")
        row = dict(zip(A15_COLUMNS, vals))
        if row["Total"] != 100 or abs(sum(vals[:-1]) - 100) > 0.1:
            raise SystemExit(f"A1.5: {name}'s row sums to {sum(vals[:-1]):.2f}")
        out[name] = row
    if out["Bhutan"]["Nepali"] != NEPALI_NATIONAL:
        raise SystemExit(f"A1.5's national Nepali is {out['Bhutan']['Nepali']}")
    bad = {k: (out[k]["Nepali"], v) for k, v in PROSE.items() if out[k]["Nepali"] != v}
    if bad:
        raise SystemExit(f"A1.5 disagrees with p.187's prose: {bad}")
    print(f"GNH 2015 Table A1.5: 20 dzongkhags and Bhutan, every row sums to 100; Nepali "
          f"{NEPALI_NATIONAL}% nationally, and the five southern dzongkhags match p.187's prose")
    return out


def read_samples():
    lines = pdf_lines(GNH2022, [44])
    i = lines.index("Religion")
    seq = [ln.replace(",", "") for ln in lines[i:i + 40]]
    for label, key in (("Buddhism", "Buddhism"), ("Hinduism", "Hinduism"),
                       ("Christianity", "Christianity")):
        k = seq.index(label)
        got = (int(seq[k + 1]), int(seq[k + 3]))
        want = (SAMPLES[2022][key], SAMPLES[2015][key])
        if got != want:
            raise SystemExit(f"GNH 2022 Table 1 {label}: {got}, pinned {want}")
    for yr, s in SAMPLES.items():
        n = sum(s.values())
        print(f"  witness, GNH {yr} unweighted sample: {n:,} respondents, Hindu "
              f"{s['Hinduism'] / n:.1%}, Christian {s['Christianity'] / n:.1%}")


def desa_mix():
    with zipfile.ZipFile(DESA) as z:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as out:
            for n in z.namelist():
                if n != "xl/styles.xml":            # openpyxl is slow on the stylesheet
                    out.writestr(n, z.read(n))
    buf.seek(0)
    df = pd.read_excel(buf, sheet_name="Table 1", header=None, engine="openpyxl")
    hdr = next(i for i in range(2, 20)
               if any("of destination" in str(x) for x in df.iloc[i])
               and any("of origin" in str(x) for x in df.iloc[i]))
    cols = [str(x).strip() for x in df.iloc[hdr]]
    dcol = next(i for i, x in enumerate(cols) if "of destination" in x)
    ocol = next(i for i, x in enumerate(cols) if "of origin" in x)
    ccol = next(i for i, x in enumerate(cols) if x == "Location code of origin")
    ycol = cols.index(DESA_YEAR)                     # the first 2020 column: both sexes
    body = df.iloc[hdr + 1:]
    m = body[body[dcol].astype(str).str.strip().str.rstrip("*").str.strip() == "Bhutan"]
    name = m[ocol].astype(str).str.strip().str.rstrip("*").str.strip()
    world = int(pd.to_numeric(m.loc[name == "World", ycol]).iloc[0])
    if world != DESA_WORLD_2020:
        raise SystemExit(f"DESA 2020 world stock for Bhutan is {world:,}")
    code = pd.to_numeric(m[ccol], errors="coerce")
    ctry = m[(code < 900) | name.isin(DESA_UNNAMED)]
    stock = dict(zip(ctry[ocol].astype(str).str.strip().str.rstrip("*").str.strip(),
                     pd.to_numeric(ctry[ycol]).astype(int)))
    if set(stock) != set(DESA_ORIGINS) | DESA_UNNAMED:
        raise SystemExit(f"DESA's origins for Bhutan changed: {sorted(set(stock) ^ (set(DESA_ORIGINS) | DESA_UNNAMED))}")
    if sum(stock.values()) != world:
        raise SystemExit(f"DESA's origins sum to {sum(stock.values()):,}, world {world:,}")
    named = {DESA_ORIGINS[k]: v for k, v in stock.items() if k in DESA_ORIGINS}
    print(f"  UN DESA 2020: {world:,} migrants in Bhutan; named origins {sum(named.values()):,} "
          f"(India {named['IN']:,}, {named['IN'] / sum(named.values()):.1%} of the named), "
          f"`Others` {stock['Others']:,} taken to be like them")
    return named


def foreign_composition(named):
    with zipfile.ZipFile(PEW) as z:
        pname = [n for n in z.namelist() if n.endswith("(unrounded counts).csv")][0]
        t = pd.read_csv(io.BytesIO(z.read(pname)), thousands=",")
    pew = t[t["Year"] == 2020].set_index("Country")
    tot = float(sum(named.values()))
    out = {}
    for iso, w in named.items():
        pn = origin.PEW_BY_ISO.get(iso)
        if pn is None or pn not in pew.index:
            raise SystemExit(f"Pew has no row {pn!r} for {iso}")
        row = {f: float(pew.loc[pn, f]) for f in origin.FAMILIES}
        for node, s in origin.composition(iso, row, OTHER_NODE).items():
            node = "islam" if node.startswith("islam") else node
            out[node] = out.get(node, 0.0) + (w / tot) * s
    return out, pew


def main():
    lut = pd.read_csv(LOOKUP, dtype={"geo_id": str}).set_index("geo_id")
    if len(lut) != 20:
        raise SystemExit(f"{LOOKUP} has {len(lut)} dzongkhags; re-run sources/bt_geo.py")
    read_level()
    a15 = read_a15(lut["gnh_name"])
    read_samples()

    # ---- Bhutanese: the Centre's level, Hindus placed by Nepali mother tongue ----
    nat = lut["bhutanese"].astype(float)
    lang = lut["gnh_name"].map(lambda g: a15[g]["Nepali"] / 100.0)
    lbar = float((nat * lang).sum() / nat.sum())
    print(f"\n  Nepali mother tongue over the 2017 census's Bhutanese: {lbar:.2%} "
          f"(the survey's own weighted national figure {NEPALI_NATIONAL}%)")
    if HINDU_PLACEMENT == "language":
        hshare = HINDU * lang / lbar
    elif HINDU_PLACEMENT == "national":
        hshare = pd.Series(HINDU, index=lut.index)
    else:
        raise SystemExit(f"HINDU_PLACEMENT {HINDU_PLACEMENT!r}")
    if (hshare + CHRISTIAN > 1).any():
        raise SystemExit("a dzongkhag's Hindu and Christian shares exceed its people")
    m = pd.DataFrame({"Hindu": nat * hshare, "Christian": nat * CHRISTIAN})
    m["Buddhist"] = nat - m["Hindu"] - m["Christian"]
    m = round_within_rows(m[["Buddhist", "Hindu", "Christian"]])
    if (m.sum(axis=1) != lut["bhutanese"]).any():
        raise SystemExit("rounding broke a dzongkhag's total")
    print(f"  placement: {HINDU_PLACEMENT}; Hindu share by dzongkhag, as drawn:")
    for g in m.sort_values("Hindu", ascending=False).index:
        print(f"      {g} {lut.loc[g, 'name']:<17} Nepali {lang[g]:6.2%}  Hindu "
              f"{m.loc[g, 'Hindu'] / lut.loc[g, 'bhutanese']:6.2%}  {int(m.loc[g, 'Hindu']):>7,} of "
              f"{int(lut.loc[g, 'bhutanese']):>7,}")
    tot_n = int(m.values.sum())
    print(f"  Bhutanese: {tot_n:,}; " + ", ".join(f"{c} {int(m[c].sum()):,} "
                                                  f"({m[c].sum() / tot_n:.2%})" for c in m.columns))

    rows = m.stack().rename("count").reset_index()
    rows.columns = ["geo_id", "source_category", "count"]
    rows = rows[rows["count"] > 0].copy()
    rows["geo_level"] = "dzongkhag"
    rows["geo_name"] = rows["geo_id"].map(lut["name"])
    rows["basis"] = "survey_modelled"
    rows["year"] = 2017
    rows["source_id"] = "gnh2010_level_x_gnh2015_mother_tongue_x_phcb2017"
    rows["note"] = ("GNH 2010's national shares; Hindus placed by GNH 2015's Nepali mother tongue "
                    "by dzongkhag (sources/bt.py); on the 2017 census's Bhutanese")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    rows[["geo_id", "geo_level", "geo_name", "source_category", "count", "basis", "year",
          "source_id", "note"]].to_csv(OUT, index=False, encoding="utf-8")

    # ---- non-Bhutanese ----
    named = desa_mix()
    comp, pew = foreign_composition(named)
    top = sorted(comp.items(), key=lambda kv: -kv[1])
    print("  their mix through Pew 2020: " + ", ".join(f"{k} {v:.2%}" for k, v in top))
    nodes = [k for k, _v in top]
    f = pd.DataFrame({k: lut["non_bhutanese"].astype(float) * v for k, v in comp.items()})[nodes]
    f = round_within_rows(f)
    if (f.sum(axis=1) != lut["non_bhutanese"]).any():
        raise SystemExit("rounding broke a dzongkhag's non-Bhutanese total")
    ext = f.stack().rename("count").reset_index()
    ext.columns = ["geo_id", "node", "count"]
    ext = ext[ext["count"] > 0].copy()
    ext["geo_level"] = "dzongkhag"
    ext["geo_name"] = ext["geo_id"].map(lut["name"])
    ext["tier"] = "modelled"
    ext["basis"] = "nationality_derived"
    ext["year"] = 2017
    ext["source_id"] = "phcb2017_non_bhutanese_x_desa2020_x_pew2020"
    ext[["geo_id", "geo_level", "geo_name", "node", "count", "tier", "basis", "year",
         "source_id"]].to_csv(OUT_FOREIGN, index=False, encoding="utf-8")

    # ---- the level against Pew's figure for everyone living in Bhutan ----
    total = tot_n + int(f.values.sum())
    hindu_all = int(m["Hindu"].sum()) + int(f.get("hinduism", pd.Series(0)).sum())
    b = pew.loc["Bhutan"]
    print(f"\nwrote {OUT} ({tot_n:,} Bhutanese) and {OUT_FOREIGN} ({int(f.values.sum()):,} "
          f"non-Bhutanese, {len(nodes)} nodes); {total:,} people")
    print(f"  drawn Hindu {hindu_all / total:.2%} of everyone; Pew 2020 for everyone living in "
          f"Bhutan {b['Hindus'] / b['Population']:.2%} Hindu, {b['Buddhists'] / b['Population']:.2%} "
          f"Buddhist, {b['Christians'] / b['Population']:.2%} Christian")


if __name__ == "__main__":
    main()
