"""Hong Kong — 2021 census ethnicity by district, and one published survey of religion.

Reads (or fetches) data/raw/hk/ and writes TWO files:
  data/normalized/hk.csv          ethnicity by District Council district, 2021 census
  data/normalized/hk_survey.csv   religion shares for the whole territory, one survey

**HONG KONG HAS NEVER ASKED ABOUT RELIGION IN A CENSUS EITHER.** The 2021 Population Census
publishes its 46 topics and religion is not among them, and Hong Kong is absent from UNSD
table 28. So this is mainland China's problem again and it gets mainland China's answer, in
the same two layers: an ethnic derivation where the category and a religion are the same
object (spec §14.5), and a self-identification survey carved out of the grey that is left
(§14.16). `taxonomy/hk2021.py` argues each category.

WHAT IS DELIBERATELY NOT USED, AND IT IS THE NUMBER EVERYONE ELSE QUOTES.
`gov.hk`'s *Hong Kong: The Facts -- Religion* gives over 1 million Buddhists, over 1 million
Taoists, 1,040,000 Protestants, 390,000 Catholics, 300,000 Muslims, 100,000 Hindus and 15,000
Sikhs. Every one of those is the religious body's own claim -- the sheet says so for Islam
("according to the Incorporated Trustees of the Islamic Community Fund") and for Sikhism --
and the set fails the only external check available, twice:

  * The same office put Protestants at **480,000** in July 2022 (quoted in the US State
    Department's 2023 IRF report from SAR government statistics) and at **1,040,000** in
    January 2026. Over the same period the churches' own eighth Hong Kong Church Survey
    counted 255,091 congregants and 197,935 weekly worshippers, **down 26% in five years**.
    The government's figure more than doubled while the only measurement fell by a quarter.
  * Its 300,000 Muslims and 100,000 Hindus are each about twice what the census's own ethnic
    counts and the survey below independently agree on. See `sources/hk.md` §4.

A body's estimate of its own adherents is not a basis this map has anywhere else, and spec
§3.1 forbids mixing it with the two that are here.

THE THREE SOURCES, ALL OPEN AND NONE GATED.

  1. **`DC_21C.CSV`** -- 2021 Population Census Statistics by District Council District,
     `census2021.gov.hk/doc/DC_21C.zip`, linked from data.gov.hk. Exact counts per district
     for total population and for ethnicity as Chinese / Filipino / Indonesian / White /
     Others. Row `Z`, *Land total*, is the 18 summed and is skipped.
  2. **Thematic Report: Ethnic Minorities**, `census2021.gov.hk/doc/pub/21c-ethnic-minorities.pdf`.
     Table 3.1 is each ethnicity's territory total; Table 8.1 is each ethnicity's percentage
     distribution over the 18 districts. Together these split `DC_21C`'s lumped `Others` into
     Indian, Nepalese, Pakistani, Other South Asian, Thai, Japanese and Korean.
  3. **The Hong Kong Political Culture Survey 2021**, Cai and Hung, published as Table 1 of
     *Religion and Trust in Hong Kong*, The China Quarterly 257 (2024), open access. 3,744
     respondents aged 16+, May-September 2021. The published table IS the tabulation, so no
     microdata is needed -- Guatemala's LAPOP shares are used the same way.

TABLE 8.1 IS PARSED FROM THE PDF AND THEN CHECKED TWO WAYS, because a column misassignment in
a ten-column table with the names printed after the numbers would be silent (§12, and cn.py's
"the column order is verified, not assumed"):

  * **Its Filipino and Indonesian columns must reproduce `DC_21C`'s exact counts.** Those two
    ethnicities appear in both sources, so the percentages are recomputable from the CSV.
    Worst disagreement over 36 comparisons: 0.05 percentage points.
  * **Its South Asian `Overall` column must equal the weighted mean of its own four South
    Asian columns**, using Table 3.1's totals as weights. Worst over 18 districts: 0.07 pp.

Neither check is a tolerance. They are two independent tables agreeing, and if the parse ever
slips a column both go wrong at once and loudly.

Usage:
    python sources/hk.py --fetch    one 20 KB zip and one 6 MB PDF
    python sources/hk.py            normalise from data/raw/hk/
"""

import argparse
import csv
import io
import os
import re
import sys
import zipfile

os.environ.setdefault("OMP_NUM_THREADS", "6")          # [[feedback_cap_cpu]]

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "hk")
OUT = os.path.join(ROOT, "data", "normalized", "hk.csv")
OUT_SURVEY = os.path.join(ROOT, "data", "normalized", "hk_survey.csv")

DC_URL = "https://www.census2021.gov.hk/doc/DC_21C.zip"
DC_NAME = "DC_21C.zip"
EM_URL = "https://www.census2021.gov.hk/doc/pub/21c-ethnic-minorities.pdf"
EM_NAME = "21c-ethnic-minorities.pdf"

SOURCE_ID = "hk_census_2021_ethnicity"
YEAR = 2021
BASIS = "ethnicity_derived"
COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

# Published totals, 2021, Thematic Report Table 3.1 (page 20). Used as the weights in the
# `Overall` check and as the multiplier on Table 8.1's percentages.
TOTALS_3_1 = {
    "Filipino": 201_291, "Indonesian": 142_065, "Indian": 42_569, "Nepalese": 29_701,
    "Pakistani": 24_385, "OtherSouthAsian": 5_314, "SouthAsianOverall": 101_969,
    "Thai": 12_972, "Japanese": 10_291, "Korean": 8_700,
}
# Table 8.1's ten columns, in the order they are printed.
T81_COLS = ["Filipino", "Indonesian", "Indian", "Nepalese", "Pakistani",
            "OtherSouthAsian", "SouthAsianOverall", "Thai", "Japanese", "Korean"]
# The seven that come out of DC_21C's lumped `Others`; Filipino and Indonesian are already
# exact in the CSV and `SouthAsianOverall` is a subtotal, not a category.
FROM_OTHERS = ["Indian", "Nepalese", "Pakistani", "OtherSouthAsian",
               "Thai", "Japanese", "Korean"]

DISTRICTS = [
    "Central and Western", "Wan Chai", "Eastern", "Southern",
    "Yau Tsim Mong", "Sham Shui Po", "Kowloon City", "Wong Tai Sin", "Kwun Tong",
    "Kwai Tsing", "Tsuen Wan", "Tuen Mun", "Yuen Long", "North", "Tai Po",
    "Sha Tin", "Sai Kung", "Islands",
]

CENSUS_POPULATION = 7_413_070      # 2021 census, whole territory
LAND_TOTAL = 7_411_945             # DC_21C row Z; the difference is the marine population

# ---------------------------------------------------------------------------------
# The survey, transcribed from one published table
# ---------------------------------------------------------------------------------
#
# Cai, Yongshun and Sin Yu Hung, "Religion and Trust in Hong Kong", The China Quarterly 257
# (2024), 609-628, Table 1, column "Believers (2021)" / "Frequency (%)". n = 3,740 of 3,744
# interviewed. **The eight shares sum to exactly 100.00**, so the table is a partition and
# needs no residual of its own.
#
# THE DESIGN, WHICH IS WHY THE SMALL CELLS ARE NOT TRUSTED FOR GEOGRAPHY. Two-stage cluster
# sample: 72 of Hong Kong's 452 electoral districts drawn at random, then 52 residents in
# each, quota-matched on the 2016 census's sex, age, education, income and housing type.
# 30-minute tablet interviews, May to September 2021, aged 16 and over. The authors
# themselves write that "the number of followers of Islam, Hinduism, Sikhism and other
# religions was limited" -- 89, 22 and 2 respondents.
SURVEY = {
    "Buddhism":   (13.72, 513),
    "Taoism":     (4.04, 151),
    "Hinduism":   (0.59, 22),
    "Sikhism":    (0.05, 2),
    "Islam":      (2.38, 89),
    "Protestant": (9.11, 341),
    "Catholic":   (4.28, 160),
    "NoReligion": (65.83, 2462),
}
SURVEY_N = 3_740
SURVEY_ID = "hk_political_culture_survey_2021"
# Of the 65.83% with no religious affiliation, the same table splits off those who
# nonetheless practise folk religion: 2,097 respondents, 56.07% of the whole sample. That is
# a PRACTICE measure and spec §3.1 forbids mixing it with the naming ones, so it is recorded
# here and in note_public and is not drawn -- exactly what China does with the same gap.
SURVEY_FOLK_PRACTISING = (56.07, 2097)


def _curl(url, dest):
    import requests
    print("GET", url)
    r = requests.get(url, timeout=900, stream=True, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    with open(dest + ".part", "wb") as fh:
        for chunk in r.iter_content(1 << 20):
            fh.write(chunk)
    os.replace(dest + ".part", dest)
    print(f"  {os.path.getsize(dest):,} bytes")


def fetch():
    os.makedirs(RAW, exist_ok=True)
    for url, name, floor in ((DC_URL, DC_NAME, 5_000), (EM_URL, EM_NAME, 1_000_000)):
        dest = os.path.join(RAW, name)
        if os.path.exists(dest) and os.path.getsize(dest) > floor:
            print("already have", dest)
            continue
        _curl(url, dest)
    print("\nBoundaries come from sources/hk_geo.py --fetch, the grid from hk_grid.py.")


def read_dc():
    """{district: {t_pop, chi, phi, ind, wh, oth}} from DC_21C.CSV, exact counts."""
    z = zipfile.ZipFile(os.path.join(RAW, DC_NAME))
    txt = z.read("DC_21C.CSV").decode("utf-8-sig")
    rows = list(csv.reader(io.StringIO(txt)))
    ix = {c: i for i, c in enumerate(rows[4])}
    out, land_total = {}, None
    for r in rows[5:]:
        if not r or not r[ix["dc_eng"]].strip():
            continue
        name = r[ix["dc_eng"]].strip()
        rec = {"letter": r[ix["dc_class"]].strip(),
               "t_pop": int(r[ix["t_pop"]]), "chi": int(r[ix["ethn_chi"]]),
               "phi": int(r[ix["ethn_phi"]]), "ind": int(r[ix["ethn_ind"]]),
               "wh": int(r[ix["ethn_wh"]]), "oth": int(r[ix["ethn_oth"]])}
        if rec["letter"] == "Z":                 # 'Land total', the 18 summed
            land_total = rec
            continue
        out[name] = rec
    if land_total is None:
        raise SystemExit("!! DC_21C.CSV has no 'Land total' row; the shape changed")
    if land_total["t_pop"] != LAND_TOTAL:
        raise SystemExit(f"!! land total is {land_total['t_pop']:,}, expected {LAND_TOTAL:,}")
    return out, land_total


def read_t81():
    """{district: {ethnicity: percent}} from Thematic Report Table 8.1.

    The PDF prints each row as ten numbers followed by the district's English name, so the
    parse keys on the name that FOLLOWS a run of exactly ten one-decimal numbers. `Sub-total`
    and `Overall` rows have the same shape and are discarded by not being district names.
    """
    import fitz

    doc = fitz.open(os.path.join(RAW, EM_NAME))
    pages = [i for i in range(doc.page_count)
             if "Table 8.1" in doc[i].get_text() or "表8.1" in doc[i].get_text()]
    pages = [i for i in pages if "Yau Tsim Mong" in doc[i].get_text()
             or "Central and Western" in doc[i].get_text()]
    if not pages:
        raise SystemExit("!! Table 8.1 not found in the thematic report")
    lines = []
    for i in range(min(pages), min(pages) + 4):
        lines += [l.strip() for l in doc[i].get_text().splitlines()]

    num = re.compile(r"^\d+\.\d$")
    out, i = {}, 0
    while i < len(lines):
        vals, j = [], i
        while j < len(lines) and len(vals) < 10:
            s = lines[j]
            if num.match(s):
                vals.append(float(s))
            elif s and vals:
                break
            j += 1
        if len(vals) == 10:
            for k in range(j, min(j + 6, len(lines))):
                if lines[k] in DISTRICTS:
                    out[lines[k]] = dict(zip(T81_COLS, vals))
                    break
                if lines[k] in ("Sub-total", "Overall"):
                    break
            i = j
        else:
            i += 1
    missing = [d for d in DISTRICTS if d not in out]
    if missing:
        raise SystemExit(f"!! Table 8.1 parse missed {len(missing)} districts: {missing}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    args = ap.parse_args()
    if args.fetch:
        fetch()
        return

    dc, land = read_dc()
    print(f"DC_21C: {len(dc)} districts, land total {land['t_pop']:,} "
          f"(census {CENSUS_POPULATION:,}; the {CENSUS_POPULATION - land['t_pop']:,} "
          f"difference is the marine population, which has no district)")
    if len(dc) != 18:
        raise SystemExit(f"!! expected 18 districts, got {len(dc)}")

    # DC_21C's five ethnicity columns must partition each district's population.
    for name, r in dc.items():
        got = r["chi"] + r["phi"] + r["ind"] + r["wh"] + r["oth"]
        if abs(got - r["t_pop"]) > 1:
            raise SystemExit(f"!! {name}: ethnicity columns sum to {got:,} against a "
                             f"population of {r['t_pop']:,}")
    print("  every district's five ethnicity columns sum to its population, to the person")

    t81 = read_t81()
    print(f"\nTable 8.1 parsed: {len(t81)} districts x {len(T81_COLS)} columns")

    # ---- check 1: Table 8.1's Filipino/Indonesian against DC_21C's exact counts --------
    tot_phi = sum(r["phi"] for r in dc.values())
    tot_ind = sum(r["ind"] for r in dc.values())
    worst1, where1 = 0.0, ""
    for name in DISTRICTS:
        for col, key, tot in (("Filipino", "phi", tot_phi), ("Indonesian", "ind", tot_ind)):
            d = abs(t81[name][col] - 100.0 * dc[name][key] / tot)
            if d > worst1:
                worst1, where1 = d, f"{name} {col}"
    print(f"  CHECK 1  Filipino/Indonesian columns vs DC_21C's exact counts: "
          f"worst {worst1:.2f} pp ({where1})")
    if worst1 > 0.15:
        raise SystemExit("!! Table 8.1's columns do not reproduce the CSV — the parse has "
                         "slipped a column, or the report and the CSV disagree")

    # ---- check 2: the South Asian subtotal against its own four parts -----------------
    sa = ["Indian", "Nepalese", "Pakistani", "OtherSouthAsian"]
    worst2, where2 = 0.0, ""
    for name in DISTRICTS:
        want = (sum(t81[name][c] * TOTALS_3_1[c] for c in sa)
                / TOTALS_3_1["SouthAsianOverall"])
        d = abs(t81[name]["SouthAsianOverall"] - want)
        if d > worst2:
            worst2, where2 = d, name
    print(f"  CHECK 2  South Asian subtotal vs the weighted mean of its four parts: "
          f"worst {worst2:.2f} pp ({where2})")
    if worst2 > 0.2:
        raise SystemExit("!! Table 8.1's South Asian columns are not internally consistent")

    # ---- split DC_21C's lumped `Others` ----------------------------------------------
    #
    # `Others` is everything but Chinese, Filipino, Indonesian and White, so it holds the
    # seven named groups below plus Other Asian, Mixed and a residual. Each named group's
    # district count is its Table 3.1 territory total times its Table 8.1 share; whatever is
    # left of `Others` is written as `OtherEthnicity` and claims nothing.
    rows, neg = [], []
    for name in DISTRICTS:
        r = dc[name]
        named = {}
        for c in FROM_OTHERS:
            named[c] = TOTALS_3_1[c] * t81[name][c] / 100.0
        rest = r["oth"] - sum(named.values())
        if rest < -0.5:
            neg.append((name, rest, r["oth"]))
        cats = {"Chinese": float(r["chi"]), "Filipino": float(r["phi"]),
                "Indonesian": float(r["ind"]), "White": float(r["wh"])}
        cats.update(named)
        cats["OtherEthnicity"] = max(rest, 0.0)
        rows.append((r["letter"], name, r["t_pop"], cats))
    if neg:
        print(f"\n  !! {len(neg)} districts where the named groups exceed DC_21C's `Others`:")
        for name, rest, oth in neg:
            print(f"     {name}: over by {-rest:,.0f} of {oth:,}")
        raise SystemExit("the two sources disagree about `Others`; do not paper over it")
    print("  the seven named groups fit inside DC_21C's `Others` in every district")

    # ---- write -----------------------------------------------------------------------
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    written = 0
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(COLUMNS)
        for letter, name, pop, cats in rows:
            for cat, n in sorted(cats.items()):
                if n < 0.5:
                    continue
                note = ("2021 census; exact count" if cat in
                        ("Chinese", "Filipino", "Indonesian", "White")
                        else "2021 census; Table 3.1 total x Table 8.1 district share")
                w.writerow([letter, "district", name, cat, int(round(n)),
                            BASIS, YEAR, SOURCE_ID, note])
                written += 1
            w.writerow([letter, "district", name, "Total", pop, BASIS, YEAR, SOURCE_ID,
                        "unit population, not a category"])
            written += 1
    print(f"\nwrote {OUT}\n  {written:,} rows over {len(rows)} districts")

    # ---- and the survey ---------------------------------------------------------------
    total_pct = sum(v[0] for v in SURVEY.values())
    if abs(total_pct - 100.0) > 0.01:
        raise SystemExit(f"!! the survey shares sum to {total_pct}, not 100")
    n_sum = sum(v[1] for v in SURVEY.values())
    if n_sum != SURVEY_N:
        raise SystemExit(f"!! the survey counts sum to {n_sum}, not {SURVEY_N}")
    print(f"  the survey's {len(SURVEY)} categories sum to {total_pct:.2f}% "
          f"and to {n_sum:,} respondents — a partition, both ways")
    with open(OUT_SURVEY, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["category", "share", "respondents", "n", "source_id", "year"])
        for cat, (pct, n) in SURVEY.items():
            w.writerow([cat, f"{pct / 100.0:.6f}", n, SURVEY_N, SURVEY_ID, YEAR])
    print(f"wrote {OUT_SURVEY}")

    # ---- what the two layers say about each other -------------------------------------
    #
    # THIS IS THE WHOLE REASON THE ETHNIC LAYER EXISTS BESIDE THE SURVEY. Two sources with no
    # lineage in common -- a census of 7.4 million and a survey of 3,740 -- are asked the same
    # question about Hong Kong's smallest religions, and the government's published figures
    # are asked it too.
    from_eth = {
        "islam": (TOTALS_3_1["Indonesian"] * 0.8751 + TOTALS_3_1["Pakistani"] * 0.9647),
        "hinduism": (TOTALS_3_1["Nepalese"] + TOTALS_3_1["Indian"]),
        "christianity.catholic": TOTALS_3_1["Filipino"] * 0.7888,
    }
    print("\nWHAT THE CENSUS'S ETHNIC COUNTS AND THE SURVEY SAY ABOUT EACH OTHER:")
    print(f"  {'':<22}{'from ethnicity':>16}{'from the survey':>18}{'gov.hk':>12}")
    for label, eth, sur, gov in (
            ("Muslims", from_eth["islam"], SURVEY["Islam"][0] / 100 * CENSUS_POPULATION,
             300_000),
            ("Hindus (upper bound)", from_eth["hinduism"],
             SURVEY["Hinduism"][0] / 100 * CENSUS_POPULATION, 100_000),
            ("Catholics", from_eth["christianity.catholic"],
             SURVEY["Catholic"][0] / 100 * CENSUS_POPULATION, 390_000)):
        print(f"  {label:<22}{eth:>16,.0f}{sur:>18,.0f}{gov:>12,}")
    print("  the Hindu row is an UPPER bound: it is every Nepalese and Indian resident, and")
    print("  neither nationality is religiously uniform, which is why neither is derived.")


if __name__ == "__main__":
    main()
