"""Central African Republic 2003 census (RGPH03), language commonly spoken, by commune
-> data/normalized/cf.csv and data/normalized/cf_under3.csv.

    python sources/cf_uscb.py [--fetch]

SOURCE. The U.S. Census Bureau's "Central African Republic Subnational Population and Housing
Data Tables with Administrative Boundaries" on HDX (CC BY), the workbook religiondots draws CAR's
religion from (religiondots/sources/cf.py). Its `Language` sheet is ICASEES's RGPH03 tabulation
"Langue couramment parlée" (language commonly spoken), taken by USCB from ICASEES's REDATAM
server (http://108.60.219.85/redbin/RpWebEngine.exe/Portal?BASE=RGPH03FRA, which timed out from
here on 2026-10-03 and again on 2026-10-04), for the country, 17 prefectures, 72 sous-prefectures
and 177 communes. One answer per person: the 78 columns partition the total to rounding.

LABELS. USCB renamed every column to an ISO 639 name "where applicable", and several are wrong
(the census's Mandjia became Mangbetu of DR Congo, Issongo became Manza, Mondjombo became
Mbangala of Angola, Aka became the Aka of Sudan's Nuba hills, Kara became the Central Sudanic
Kara where the census's Kara sits 76% in Bocaranga, which is Gbaya Kara). The Data Dictionary
keeps ICASEES's own field name for every column ('Original field name: "Gbaya."'), so
`source_category` is the census's French and USCB's name is carried only for reference.
taxonomy/cf2003.py maps from the French.

THE FULFULDE COLUMN HOLDS THE CHILDREN UNDER THREE. USCB's metadata: "the data collected on
language spoken excluded household members under the age of 3", yet the language total is 95.7%
of the census, not the ~85% that excluding them would leave. They are in `Fulfuldé`. Fulfulde is
9-12% of EVERY commune, including the 35 where under 2% of people are Muslim and under 1% are of
Haoussa/Muslim ethnicity (every Fulfulde speaker in CAR is Peul or Mbororo, and Muslim), and in
the five non-Muslim communes of Ouham and Ouham-Pende where the language table collapsed (34-61%
of the census answered) Fulfulde is still 10.6-11.6% OF THE WHOLE CENSUS POPULATION, a share of
people who were counted whether or not anyone asked them a language. Nationally the column holds
457,091 people (12.3%); about 382,000 of them are the under-threes, 61% of the census's 627,118
aged 0-4 ("La RCA en chiffres", ICASEES 2005), where three single years of five less infant
deaths would be about 62%.

So each commune's under-threes are estimated as a floor share of its census population, the
floor measured on the 35 communes with no Muslim or Haoussa presence (median FUL/census, Bangui's
arrondissements apart because a city's fertility is lower), and Fulfulde is the column minus that,
never below zero. The estimate and its checks print on every run; cf_under3.csv carries it per
commune, and countries/cf.py draws Fulfulde from it as `derived`.

CHECKS (all must pass):
  1. 78 language columns, each with one census name in the dictionary, all distinct
  2. no negative cells; 1 country, 17 prefectures, 72 sous-prefectures, 177 communes
  3. every row's categories sum to its total within 7 (USCB: "small summation errors that were
     not corrected"), and communes sum to the national figure per category within 12
  4. the Ethnicity sheet's national total is the published RGPH03 population, 3,895,139
  5. the Fulfulde floor: at least 30 clean communes, their spread printed; the communes whose
     language table collapsed sit inside the clean communes' range (the test that the column's
     floor scales with the census, not with who answered); the implied under-threes are 55-70% of
     the census's 0-4 year olds
"""
import argparse
import re
import sys
import urllib.request
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent.parent
RAW = HERE / "data" / "raw" / "cf"
XLSX = RAW / "central_african_republic_uscb_202303.xlsx"
URL = ("https://data.humdata.org/dataset/4f41a2c7-167f-4e60-8e90-dd282383dd90/resource/"
       "1670d3a6-a89f-40cd-80c9-bf580b27e08e/download/central_african_republic_uscb_202303.xlsx")
OUT = HERE / "data" / "normalized" / "cf.csv"
OUT_U3 = HERE / "data" / "normalized" / "cf_under3.csv"

CENSUS = 3_895_139            # RGPH03, the Ethnicity sheet's total and "La RCA en chiffres"
AGE_0_4 = 627_118             # "La RCA en chiffres" (ICASEES, 2005), §2.9.1
N_COLS = 78
SUM_BOUND = 7                 # categories minus total, per row, as measured
NAT_BOUND = 12                # communes minus national, per category, as measured
BANGUI = "BANGUI"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"}


def fetch():
    RAW.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(URL, headers=UA)
    data = urllib.request.urlopen(req, timeout=300).read()
    if data[:4] != b"PK\x03\x04" or len(data) < 300_000:
        raise SystemExit(f"{XLSX.name}: not an xlsx ({len(data):,} bytes, starts {data[:16]!r})")
    XLSX.write_bytes(data)
    print(f"wrote {XLSX} ({len(data):,} bytes)")


def sheet(name):
    # row 0 is the field code, row 1 the field's English description
    return pd.read_excel(XLSX, sheet_name=name, header=0, skiprows=[1])


def census_names():
    d = pd.read_excel(XLSX, sheet_name="Data Dictionary", header=None)
    out, in_lang = {}, False
    for r in d.itertuples(index=False):
        v = [str(x) for x in r if pd.notna(x)]
        if not v:
            continue
        if v[0].startswith("Language ("):
            in_lang = True
            continue
        if in_lang and v[0].startswith("LNG_") and v[0] != "LNG_BTOTL":
            m = re.search(r'Original field name: "(.*?)\.?"', v[2])
            if not m:
                raise SystemExit(f"{v[0]}: no original field name in the dictionary")
            if v[0] in out:
                raise SystemExit(f"{v[0]} twice in the dictionary")
            out[v[0]] = (m.group(1).strip(), v[1].strip())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    args = ap.parse_args()
    if args.fetch or not XLSX.exists():
        fetch()

    ok = True

    def report(good, msg):
        nonlocal ok
        ok &= bool(good)
        print(f"  {'OK ' if good else 'BAD'} {msg}")

    print("Central African Republic, RGPH03 2003, language commonly spoken (USCB tabulation)\n")
    names = census_names()
    lang = sheet("Language")
    cols = [c for c in lang.columns if c.startswith("LNG_") and c != "LNG_BTOTL"]
    fr = [names[c][0] for c in cols]
    report(len(cols) == N_COLS and set(cols) == set(names) and len(set(fr)) == len(fr),
           f"{len(cols)} language columns, each with one census name, all distinct")

    for c in cols + ["LNG_BTOTL"]:
        lang[c] = pd.to_numeric(lang[c], errors="raise")
    neg = int((lang[cols] < 0).sum().sum())
    lv = lang["ADM_LEVEL"].value_counts().to_dict()
    report(neg == 0 and (lv.get(0), lv.get(1), lv.get(2), lv.get(3)) == (1, 17, 72, 177),
           f"{neg} negative cells; levels {dict(sorted(lv.items()))}")

    resid = (lang[cols].sum(axis=1) - lang["LNG_BTOTL"]).astype(int)
    report(resid.abs().max() <= SUM_BOUND,
           f"categories minus total per row in [{resid.min()}, {resid.max()}] "
           f"(bound {SUM_BOUND}); distribution {dict(sorted(resid.value_counts().items()))}")

    nat = lang[lang["ADM_LEVEL"] == 0].iloc[0]
    com = lang[lang["ADM_LEVEL"] == 3].copy()
    off = {c: int(com[c].sum() - nat[c]) for c in cols}
    worst = max(abs(v) for v in off.values())
    report(worst <= NAT_BOUND, f"communes minus national per category: largest {worst} "
                               f"(bound {NAT_BOUND}); total {int(com['LNG_BTOTL'].sum() - nat['LNG_BTOTL'])}")

    eth = sheet("Ethnicity")
    eth_nat = int(eth.loc[eth["ADM_LEVEL"] == 0, "ETH_BTOTL"].iloc[0])
    report(eth_nat == CENSUS, f"the Ethnicity sheet's national total is {eth_nat:,} "
                              f"(RGPH03 population {CENSUS:,})")
    e3 = eth[eth["ADM_LEVEL"] == 3].set_index("GEO_MATCH")
    rel = sheet("Religion")
    r3 = rel[rel["ADM_LEVEL"] == 3].set_index("GEO_MATCH")
    com = com.set_index("GEO_MATCH")
    report(set(com.index) == set(e3.index) == set(r3.index),
           "the same 177 commune ids in the Language, Ethnicity and Religion sheets")

    cov = com["LNG_BTOTL"] / e3["ETH_BTOTL"]
    print(f"\n  language total {int(nat['LNG_BTOTL']):,} = {nat['LNG_BTOTL'] / CENSUS:.1%} of the "
          f"census; per commune min {cov.min():.1%}, median {cov.median():.1%}")
    low = cov[cov < 0.9].sort_values()
    print("  communes below 90%: " + ", ".join(f"{com.loc[k, 'AREA_NAME']} ({v:.0%})"
                                              for k, v in low.items()))

    # ---- the Fulfulde floor (the children under three) ----
    print("\n  THE FULFULDE COLUMN AND THE UNDER-THREES")
    t = pd.DataFrame({
        "name": com["AREA_NAME"], "prefecture": com["ADM1_NAME"],
        "census_pop": e3["ETH_BTOTL"].astype(int), "language_total": com["LNG_BTOTL"].astype(int),
        "fulfulde_published": com["LNG_FUL"].astype(int),
        "muslim": r3["RLG_MUS"] / r3["RLG_BTOTL"], "haoussa": e3["ETH_HAMU"] / e3["ETH_BTOTL"]})
    t["ful_share"] = t["fulfulde_published"] / t["census_pop"]
    t["stratum"] = (t["prefecture"] == BANGUI).map({True: "bangui", False: "rest"})
    clean = t[(t["muslim"] < 0.02) & (t["haoussa"] < 0.01)]
    floor = clean.groupby("stratum")["ful_share"].median().to_dict()
    for s, g in clean.groupby("stratum"):
        q = g["ful_share"].quantile([0, .25, .5, .75, 1]).round(4).tolist()
        print(f"     {s}: {len(g)} clean communes, Fulfulde as a share of the census "
              f"min/q1/median/q3/max {q}")
    report(len(clean) >= 30 and set(floor) == {"bangui", "rest"},
           f"{len(clean)} communes with under 2% Muslims and under 1% Haoussa/Muslim ethnicity "
           f"set the floor: Bangui {floor.get('bangui', 0):.4f}, elsewhere {floor.get('rest', 0):.4f}")
    band = clean.loc[clean["stratum"] == "rest", "ful_share"]
    collapsed = t[(cov < 0.65) & (t["muslim"] < 0.02)]
    inside = collapsed["ful_share"].between(band.min(), band.max())
    report(len(collapsed) >= 4 and inside.all(),
           f"the {len(collapsed)} non-Muslim communes where under 65% answered the language "
           f"question still have Fulfulde at {collapsed['ful_share'].min():.3f}-"
           f"{collapsed['ful_share'].max():.3f} of the census, inside the clean range: the floor "
           "scales with who was counted, not with who answered")

    t["under3_est"] = (t["stratum"].map(floor) * t["census_pop"]).round().astype(int)
    t["under3_est"] = t[["under3_est", "fulfulde_published"]].min(axis=1)
    t["fulfulde_est"] = t["fulfulde_published"] - t["under3_est"]
    u = int(t["under3_est"].sum())
    report(0.55 <= u / AGE_0_4 <= 0.70,
           f"{u:,} under-threes implied = {u / AGE_0_4:.0%} of the census's {AGE_0_4:,} aged 0-4 "
           f"(three of five single years, less infant deaths, is about 62%)")
    over = t[t["fulfulde_est"] > t["muslim"] * t["census_pop"]]
    print(f"  -- Fulfulde left: {int(t['fulfulde_est'].sum()):,} (from {int(t['fulfulde_published'].sum()):,} "
          f"published); zero in {int((t['fulfulde_est'] == 0).sum())} communes; above the commune's "
          f"Muslims (a sign of floor noise) in {len(over)}, by {int((over['fulfulde_est'] - over['muslim'] * over['census_pop']).sum()):,} people in all")

    if not ok:
        raise SystemExit("\nreconciliation FAILED; nothing written")

    long = com.reset_index().melt(id_vars=["GEO_MATCH", "AREA_NAME", "ADM1_NAME"],
                                  value_vars=cols, var_name="uscb_field", value_name="count")
    long = long[long["count"] > 0].copy()
    long["count"] = long["count"].astype("int64")
    long["source_category"] = long["uscb_field"].map(lambda c: names[c][0])
    long["uscb_name"] = long["uscb_field"].map(lambda c: names[c][1])
    long["geo_level"] = "commune"
    out = long.rename(columns={"GEO_MATCH": "geo_id", "AREA_NAME": "geo_name",
                               "ADM1_NAME": "prefecture"})
    out = out[["geo_id", "geo_level", "geo_name", "prefecture", "source_category", "count",
               "uscb_field", "uscb_name"]].sort_values(["geo_id", "count"], ascending=[True, False])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False, encoding="utf-8")
    u3 = t.reset_index().rename(columns={"GEO_MATCH": "geo_id"})
    u3[["geo_id", "name", "prefecture", "stratum", "census_pop", "language_total",
        "fulfulde_published", "under3_est", "fulfulde_est"]].to_csv(OUT_U3, index=False, encoding="utf-8")
    print(f"\nwrote {OUT}: {len(out):,} rows, {out['geo_id'].nunique()} communes, "
          f"{out['count'].sum():,} people, {out['source_category'].nunique()} languages with speakers")
    print(f"wrote {OUT_U3}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
