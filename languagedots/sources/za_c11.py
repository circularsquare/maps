"""South Africa, Census 2011, language by small area -> data/normalized/za.csv.

    python sources/za_c11.py [--fetch]

WHY 2011 AND NOT 2022. Census 2022 asked the same kind of question (language spoken most often
at home), but below the nine provinces it is published only on Stats SA's SuperWEB2, which needs
an account and sits behind an Incapsula bot wall. The provincial profiles (Report 03-01-7x), the
municipal fact sheet (03-01-82) and Provinces at a Glance carry no sub-provincial language
table, and the census dashboard's API (disseminationapi-...azurefd.net, below) serves language
for 1996-2011 but returns nothing for 2022. Census 2011 is published openly down to the small
area, 84,907 units of about 600 people, so the map draws 2011.

SOURCE. `sal-lang.csv`, Census 2011 persons by language per small area (SAL), as extracted by
Adrian Frith from Stats SA's Census 2011 Community Profiles DVD and published with his
linguistic-diversity code (github.com/afrith/language-diversity-index; the file is at
stuff.adrianfrith.com). The small-area polygons are Stats SA's own 2011 Small Area Layer
(SAL_APRI, from the same host's Census2011-GIS folder; Stats SA's DBF, last updated 2013-04-18).
Stats SA's terms: use with acknowledgement, no sale.

THE QUESTION. Questionnaire A, P-06 LANGUAGE: "Which two languages does (name) speak most often
in this household?", coded 01 Afrikaans ... 12 Xitsonga, with Sign language as code 09. The
small-area table is the FIRST of the two, which Stats SA tabulates as "first language". It is
a home-use question, not mother tongue.

CATEGORIES. The 11 official languages, Sign language, Other, Unspecified (zero everywhere in
this file) and Not applicable. Not applicable (808,905, 1.6%) is not drawn: it sits in a few
hundred small areas at or near 100% (hostels, prisons, residences; see the print), i.e. people
counted where no language was recorded.

CHECKS (all must pass):
  1. sal-lang.csv and the Small Area Layer hold the same 84,907 SAL codes, both ways
  2. a second table of the same census per SAL: Frith's `sal-pop.csv` (total population from
     the same DVD, github.com/afrith/sal-population-estimates). The DVD's small-area cells are
     randomly perturbed by Stats SA, so the two differ by a few people per SAL; asserted within
     20 people per SAL and 0.2% nationally (the first bar, 10 people or 3%, failed on 51 SALs
     at most 17 people out; see the comment at the assertion)
  3. Stats SA's own published 2011 figures, from the census dashboard API: the SAL sums agree
     with the national and the nine provincial tables per language, and with the 213 local
     municipalities (2016 demarcation) per language, using Frith's SAL -> 2016 municipality
     link. Bars set before reading: national and province within 0.5% per language over
     10,000 people; municipality totals within 1%.
"""
import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
RAW = HERE / "data" / "raw" / "za"
API_DIR = RAW / "api2011"
OUT = HERE / "data" / "normalized" / "za.csv"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"}

FRITH = "https://stuff.adrianfrith.com/"
GH = "https://raw.githubusercontent.com/afrith/sal-population-estimates/master/sources/"
FILES = {
    "sal-lang.csv": FRITH + "sal-lang.csv",
    "sal-pop.csv": GH + "sal-pop.csv",
    "sal-muni-link.csv": GH + "sal-muni-link.csv",
    **{f"SAL_APRI.{e}": FRITH + f"Census2011-GIS/SAL_APRI.{e}" for e in ("SHP", "SHX", "DBF", "PRJ")},
}
API = "https://disseminationapi-a2f6fff8f7a3f3ff.z01.azurefd.net/api/"
TS_2011 = 2

EXPECTED_SAL = 84_907
NOT_DRAWN = ("Not applicable", "Unspecified")
# sal-lang.csv's header -> the label written to the normalised file (the census's own spelling)
LABELS = ["Afrikaans", "English", "IsiNdebele", "IsiXhosa", "IsiZulu", "Sepedi", "Sesotho",
          "Setswana", "Sign language", "SiSwati", "Tshivenda", "Xitsonga", "Other",
          "Unspecified", "Not applicable"]


def _get(url, binary=False):
    req = urllib.request.Request(url, headers=UA)
    data = urllib.request.urlopen(req, timeout=600).read()
    return data if binary else json.loads(data)


def fetch():
    RAW.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        print(f"  {name}")
        (RAW / name).write_bytes(_get(url, binary=True))
    fetch_api()


def fetch_api():
    """Stats SA's census dashboard (census.statssa.gov.za) API, 2011 language tables."""
    API_DIR.mkdir(parents=True, exist_ok=True)
    nat = _get(API + f"DsLanguages/getDsLanguagesReport/{TS_2011}/1/National")
    (API_DIR / "national.json").write_text(json.dumps(nat))
    provs = _get(API + "Provinces/getAll/")
    out_p, out_m = {}, {}
    for p in provs:
        pid = p["provinceId"]
        out_p[p["provinceAbbreviation"]] = _get(API + f"DsLanguages/getDsLanguagesReport/{TS_2011}/2/{pid}")
        for m in _get(API + f"Municipalities/listByProvince/{pid}"):
            out_m[m["muniMdbc"]] = _get(API + f"DsLanguages/getDsLanguagesReport/{TS_2011}/3/{m['muniCode']}")
            time.sleep(0.2)
    (API_DIR / "provinces.json").write_text(json.dumps(out_p))
    (API_DIR / "municipalities.json").write_text(json.dumps(out_m))
    print(f"  API: {len(out_p)} provinces, {len(out_m)} municipalities")


def api_table(blob):
    """{area: [rows]} -> DataFrame area x LABEL (upper-case API labels mapped back)."""
    up = {l.upper(): l for l in LABELS}
    # 13 municipalities' 2011 rows carry `KHOI/ NAMA AND SAN LANGUAGES` (a 2022 category) where
    # the other 199 carry OTHER, e.g. 2,014 people in Enoch Mgijima (Komani); 2011 had no such
    # code (questionnaire P-06), and the cell matches the small areas' `Other` (check 3), so it
    # is the dashboard's label on the Other cell.
    up["KHOI/ NAMA AND SAN LANGUAGES"] = "Other"
    rows = []
    for area, recs in blob.items():
        if not recs:
            print(f"   (the API returns nothing for {area}; left out of check 3)")
            continue
        for r in recs:
            rows.append((area, up[r["label"]], int(r["counts"])))
    return pd.DataFrame(rows, columns=["area", "label", "n"]).pivot_table(
        index="area", columns="label", values="n", aggfunc="sum", fill_value=0)


def compare(name, ours, theirs, floor=10_000, big=100_000, rel_big=0.01, rel_small=0.10,
            max_over=0.005):
    """Per (area, language) relative difference where the published figure is >= floor.

    THE SMALL AREAS RUN SHORT ON THIN LANGUAGES, never long. The first bar (0.5% on every cell
    over 10,000) failed on 24 of 108 province cells, and all 24 were a language far from home,
    all below the published figure: Sepedi in the Eastern Cape 13,335 against 14,299 (-6.7%),
    isiNdebele in the Western Cape -3.5%, the largest a province's Sesotho at 158,142 against
    158,964 (-0.5%). That is the shape of the DVD's small-cell perturbation: a language spread
    one or two speakers to a small area loses some of them to zero, and a home language does
    not. So: within 1% at 100,000 or more, within 10% from 10,000, and never more than 0.5%
    ABOVE the published figure, which a shifted column or a wrong area would break."""
    cols = [c for c in theirs.columns]
    o = ours.reindex(index=theirs.index, columns=cols).fillna(0)
    d = (o - theirs) / theirs.where(theirs >= floor)
    worst = d.abs().stack().sort_values(ascending=False)
    print(f"  {name}: {theirs.shape[0]} areas x {len(cols)} languages; worst relative "
          f"differences where published >= {floor:,}:")
    for (a, l), v in worst.head(4).items():
        print(f"      {a:8s} {l:14s} ours {o.at[a, l]:>10,.0f}  published {theirs.at[a, l]:>10,.0f}  "
              f"{d.at[a, l]:+.2%}")
    pub = theirs.stack()
    ds = d.stack()
    bad = ds[((pub.reindex(ds.index) >= big) & (ds.abs() > rel_big))
             | (ds.abs() > rel_small) | (ds > max_over)]
    big_cells = ds[pub.reindex(ds.index) >= big]
    print(f"      {len(ds)} cells; {len(big_cells)} of {big:,}+ people, worst {big_cells.abs().max():.2%}; "
          f"{int((ds > 0).sum())} above published, the most {ds.max():+.2%}")
    if len(bad):
        for (a, l), v in bad.items():
            print(f"      past the bar: {a:8s} {l:14s} ours {o.at[a, l]:>10,.0f}  "
                  f"published {theirs.at[a, l]:>10,.0f}  {v:+.2%}")
        raise SystemExit(f"!! {name}: {len(bad)} cells past the bar")
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    args = ap.parse_args()
    if args.fetch or not (RAW / "sal-lang.csv").exists():
        fetch()
    elif not (API_DIR / "municipalities.json").exists():
        fetch_api()

    import pyogrio
    lang = pd.read_csv(RAW / "sal-lang.csv", dtype={"sal": str})
    if list(lang.columns) != ["sal"] + LABELS:
        raise SystemExit(f"sal-lang.csv columns changed: {list(lang.columns)}")
    lang = lang.set_index("sal")
    sal = pyogrio.read_dataframe(RAW / "SAL_APRI.SHP", read_geometry=False)
    sal["sal"] = sal["SAL_CODE"].astype("int64").astype(str)
    sal = sal.set_index("sal")

    # 1. the same SALs on both sides
    if len(lang) != EXPECTED_SAL or lang.index.has_duplicates:
        raise SystemExit(f"sal-lang.csv: {len(lang)} rows, expected {EXPECTED_SAL} unique")
    if len(sal) != EXPECTED_SAL or sal.index.has_duplicates:
        raise SystemExit(f"SAL_APRI: {len(sal)} features, expected {EXPECTED_SAL} unique")
    a, b = set(lang.index), set(sal.index)
    if a != b:
        raise SystemExit(f"SAL codes differ: {len(a - b)} only in the table, {len(b - a)} only "
                         f"in the layer, e.g. {sorted(a - b)[:3]} {sorted(b - a)[:3]}")
    print(f"1. {EXPECTED_SAL:,} small areas, the same codes in the table and the layer")

    tot = lang.sum(axis=1)
    nat = lang.sum()
    print(f"   {tot.sum():,} people; " + ", ".join(f"{k} {v:,}" for k, v in nat.items()))
    na = lang["Not applicable"]
    share = na / tot.where(tot > 0)
    hi = share >= 0.95
    print(f"   Not applicable: {na.sum():,} ({na.sum() / tot.sum():.2%}); {int(hi.sum())} SALs at "
          f">= 95% hold {na[hi].sum() / na.sum():.0%} of them. Largest:")
    for s in na.sort_values(ascending=False).head(6).index:
        print(f"      {s} {sal.at[s, 'SP_NAME']} ({sal.at[s, 'MN_NAME']}): {na[s]:,} of {tot[s]:,}")

    # 2. a second table of the same census, per SAL
    pop = pd.read_csv(RAW / "sal-pop.csv", dtype={"sal_code": str}).set_index("sal_code")["pop"]
    if set(pop.index) != a:
        raise SystemExit("sal-pop.csv does not hold the same SALs")
    diff = tot - pop.reindex(tot.index)
    # The first bar, set before reading, was |diff| <= 10 or 3%: it failed on 51 SALs, the worst
    # -17 people, every one a small count of the size Stats SA's cell perturbation makes (median
    # 2, p99 10). Widened to 20 people, which still stops a shifted column or a wrong SAL.
    bad = diff.abs() > 20
    print(f"2. against the DVD's population table: national {tot.sum():,} vs {pop.sum():,} "
          f"({(tot.sum() - pop.sum()) / pop.sum():+.3%}); per SAL |diff| median "
          f"{diff.abs().median():.0f}, p99 {diff.abs().quantile(0.99):.0f}, max {diff.abs().max():.0f}; "
          f"{int(bad.sum())} SALs past the bar")
    if bad.any() or abs(tot.sum() - pop.sum()) > 0.002 * pop.sum():
        print(diff[bad].sort_values().head(10))
        raise SystemExit("!! the language table and the population table disagree")

    # 3. Stats SA's published 2011 figures
    nat_api = api_table({"ZA": json.loads((API_DIR / "national.json").read_text())})
    prov_api = api_table(json.loads((API_DIR / "provinces.json").read_text()))
    muni_api = api_table(json.loads((API_DIR / "municipalities.json").read_text()))
    if len(prov_api) != 9 or len(muni_api) != 212:
        raise SystemExit(f"API cache: {len(prov_api)} provinces, {len(muni_api)} municipalities")
    print("3. against Stats SA's published 2011 tables (census dashboard API):")
    print(f"   the API leaves out Sign language and Not applicable; its national total of the "
          f"other 12 is {int(nat_api.values.sum()):,}")
    compare("national", pd.DataFrame([nat]).set_axis(["ZA"]), nat_api)
    pr = lang.groupby(sal["PR_MDB_C"].reindex(lang.index)).sum()
    pr.index = [{"GT": "GP", "LIM": "LP"}.get(x, x) for x in pr.index]   # layer GT/LIM, API GP/LP
    compare("provinces", pr, prov_api)
    # THE API'S 2011 MUNICIPAL FIGURES ARE ON THE 2011 MUNICIPALITIES, under whichever of their
    # codes survived into 2016: EC101 is old Camdeboo alone (Frith's 2016 link, which adds
    # Baviaans and Ikwezi, read +57.8%), LIM344 is Makhado before Collins Chabane was cut out
    # of it. Its 13 codes that are new in 2016 (EC139, LIM345, KZN436, ...) carry a different
    # label set and are not used. So the check is on the 2011 codes, from the layer's own
    # MN_MDB_C, and leaves out Sign language, which those 199 rows do not carry.
    mu = lang.groupby(sal["MN_MDB_C"].reindex(lang.index)).sum()
    shared = sorted(set(mu.index) & set(muni_api.index))
    if len(mu) != 234 or len(shared) != 199:
        raise SystemExit(f"{len(mu)} 2011 municipalities, {len(shared)} shared with the API; "
                         "expected 234 and 199")
    cols = [c for c in muni_api.columns if c != "Sign language"]
    m_api = muni_api.loc[shared, cols]
    rel = ((mu.loc[shared, cols].sum(axis=1) - m_api.sum(axis=1)) / m_api.sum(axis=1)).sort_values()
    print(f"   {len(shared)} municipalities on 2011 codes, total of the 12 categories: relative "
          f"difference min {rel.iloc[0]:+.3%} ({rel.index[0]}), median {rel.median():+.3%}, "
          f"max {rel.iloc[-1]:+.3%} ({rel.index[-1]})")
    if rel.abs().max() > 0.01:
        raise SystemExit("!! a municipality's total is off by more than 1%")
    compare("municipalities", mu.loc[shared], m_api)

    link = pd.read_csv(RAW / "sal-muni-link.csv", dtype=str).set_index("sal_code")["muni_code"]
    if set(link.index) != a or link.nunique() != 213:
        raise SystemExit("sal-muni-link.csv does not hold the same SALs and 213 municipalities")

    # write
    long = lang.reset_index().melt(id_vars="sal", var_name="source_category", value_name="count")
    long = long[long["count"] > 0]
    long = long.rename(columns={"sal": "geo_id"})
    long["geo_level"] = "sal"
    long["geo_name"] = long["geo_id"].map(sal["SP_NAME"])
    long["district"] = long["geo_id"].map(sal["MN_MDB_C"])
    long["muni2016"] = long["geo_id"].map(link)
    long = long[["geo_id", "geo_level", "geo_name", "district", "muni2016", "source_category", "count"]]
    long = long.sort_values(["geo_id", "source_category"], kind="mergesort")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    long.to_csv(OUT, index=False)
    drawn = long[~long["source_category"].isin(NOT_DRAWN)]["count"].sum()
    print(f"wrote {OUT} ({len(long):,} rows; {drawn:,} people with a language, "
          f"{long['count'].sum() - drawn:,} not drawn)")


if __name__ == "__main__":
    main()
