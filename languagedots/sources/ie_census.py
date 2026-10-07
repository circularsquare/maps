"""Ireland: Census of Population 2022 (CSO), by Small Area.

    python sources/ie_census.py --fetch    the SAPS Small Area CSV (40 MB) and PxStat F5029, if missing
    python sources/ie_census.py            normalise from data/raw/ie/

-> data/normalized/ie.csv: unit (SA GUID), source_category, count, tier. One row per (Small Area,
   language), 18,919 Small Areas, every resident drawn.

THERE IS NO FIRST-LANGUAGE QUESTION. The 2022 form asks two language questions of everyone:
  * Q14 "Can you speak Irish?" and, if yes, how often (daily within the education system, daily /
    weekly / less often / never outside it). Most of the 1.87M "yes" learned Irish at school.
  * Q15 "Do you speak a language other than English or Irish at home?" and, if yes, which
    (one write-in). 751,507 people answered with a language.
English is never asked about as a home language. So the map is built from three parts, all on
the Small Area Population Statistics (SAPS) themes 1-3:

  1. Languages other than English or Irish (Q15), measured: SAPS T2_5 names Polish, French and
     Spanish per Small Area plus `Other (incl. not stated)`. The Other part is shared among the
     64 further labels of PxStat F5029 (language spoken at home x county, 2022) in its county's
     proportions, tier `derived`.
  2. Irish, `derived`: the Irish speakers who speak it daily outside the education system
     (T3_2DIDOT "daily within and daily outside" + T3_2DOEST "daily only outside", 3+). The
     ability answer is drawn on English: Irish learned at school is a second language (AGENT_BRIEF
     §2), and daily use outside school is the census's nearest thing to Irish as a home language.
     IRISH_DAILY = False draws nobody as Irish.
  3. English, `derived`: everyone else in the Small Area (T1_1AGETT, all ages), clipped at zero.

THE CHECKS. Per Small Area: PL+FR+ES+OTH == T2_5T; the frequency rows sum to T3_2ALLT and T3_2ALLT
== T3_1YES. The SA->county crosswalk (CSO's 34-county cut on the boundary layer, religiondots'
copy, read-only) matches all 18,919 Small Areas both ways. Per F5029 county: the 66 labels sum to
`All languages`; SAPS summed over the county's Small Areas equals F5029 for Polish, French, Spanish
and the total. State totals against CSO's headline figures.
"""

import json
import os
import sys
import urllib.request
from itertools import product

os.environ.setdefault("OMP_NUM_THREADS", "2")
import pandas as pd  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from rdlink import RD_GEO  # noqa: E402

RAW = os.path.join(ROOT, "data", "raw", "ie")
OUT = os.path.join(ROOT, "data", "normalized", "ie.csv")
SA_SHP = os.path.join(str(RD_GEO), "ie", "smallareas2022", "SMALL_AREA_2022.shp")

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
FILES = {
    "SAPS_2022_Small_Area_UR_171024.csv":
        "https://www.cso.ie/en/media/csoie/census/census2022/SAPS_2022_Small_Area_UR_171024.csv",
    "F5029.json":
        "https://ws.cso.ie/public/api.restful/PxStat.Data.Cube_API.ReadDataset/F5029/JSON-stat/2.0/en",
}

IRISH_DAILY = True
IRISH_LABEL = "Irish (speaks it daily outside the education system)"
ENGLISH_LABEL = "English (everyone not counted under another language)"

# boundary layer's COUNTY_ENGLISH (34) -> F5029's county of usual residence (31)
COUNTY = {
    "CARLOW": "Carlow", "CAVAN": "Cavan", "CLARE": "Clare",
    "CORK": "Cork City and Cork County", "CORK CITY": "Cork City and Cork County",
    "DONEGAL": "Donegal", "DUBLIN CITY": "Dublin City",
    "DUN LAOGHAIRE/RATHDOWN": "Dún Laoghaire-Rathdown", "FINGAL": "Fingal",
    "GALWAY": "Galway County", "GALWAY CITY": "Galway City", "KERRY": "Kerry",
    "KILDARE": "Kildare", "KILKENNY": "Kilkenny", "LAOIS": "Laois", "LEITRIM": "Leitrim",
    "LIMERICK": "Limerick City and County", "LIMERICK CITY": "Limerick City and County",
    "LONGFORD": "Longford", "LOUTH": "Louth", "MAYO": "Mayo", "MEATH": "Meath",
    "MONAGHAN": "Monaghan", "NORTH TIPPERARY": "Tipperary", "SOUTH TIPPERARY": "Tipperary",
    "OFFALY": "Offaly", "ROSCOMMON": "Roscommon", "SLIGO": "Sligo",
    "SOUTH DUBLIN": "South Dublin", "WATERFORD": "Waterford City and County",
    "WATERFORD CITY": "Waterford City and County", "WESTMEATH": "Westmeath",
    "WEXFORD": "Wexford", "WICKLOW": "Wicklow",
}

SAPS_NAMED = {"T2_5PL": "Polish", "T2_5FR": "French", "T2_5ES": "Spanish"}


def fetch():
    os.makedirs(RAW, exist_ok=True)
    for name, url in FILES.items():
        path = os.path.join(RAW, name)
        if os.path.exists(path):
            continue
        print(f"fetching {name}")
        raw = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300).read()
        with open(path + ".tmp", "wb") as f:
            f.write(raw)
        os.replace(path + ".tmp", path)


def read_f5029():
    d = json.load(open(os.path.join(RAW, "F5029.json"), encoding="utf-8"))
    cats = []
    for k in d["id"]:
        idx = d["dimension"][k]["category"]["index"]
        cats.append(idx if isinstance(idx, list) else sorted(idx, key=idx.get))
    labs = [d["dimension"][k]["category"]["label"] for k in d["id"]]
    rows = []
    for i, combo in enumerate(product(*cats)):
        rows.append([labs[j][c] for j, c in enumerate(combo)] + [d["value"][i]])
    df = pd.DataFrame(rows, columns=["stat", "year", "sex", "county", "language", "count"])
    df = df[(df["year"] == "2022") & (df["sex"] == "Both sexes")]
    return df.pivot(index="county", columns="language", values="count").fillna(0)


def main():
    if "--fetch" in sys.argv:
        fetch()
    sa = pd.read_csv(os.path.join(RAW, "SAPS_2022_Small_Area_UR_171024.csv"), encoding="latin-1")
    # the file ends with the State row, GUID "IE0": every column must equal the Small Areas' sum
    state = sa[sa["GUID"] == "IE0"]
    sa = sa[sa["GUID"] != "IE0"].copy()
    assert len(state) == 1 and len(sa) == 18919 and sa["GUID"].is_unique, len(sa)
    num = [c for c in sa.columns if c.startswith("T")]
    bad = [c for c in num if sa[c].sum() != state[c].iloc[0]]
    assert not bad, f"Small Areas do not sum to the State row in {bad[:5]}"

    # per-SA consistency
    assert (sa[["T2_5PL", "T2_5FR", "T2_5ES", "T2_5OTH"]].sum(axis=1) == sa["T2_5T"]).all()
    freq = [c for c in sa.columns if c.startswith("T3_2") and c.endswith("T") and c != "T3_2ALLT"]
    assert len(freq) == 10, freq
    assert (sa[freq].sum(axis=1) == sa["T3_2ALLT"]).all()
    assert (sa["T3_2ALLT"] == sa["T3_1YES"]).all()

    # SA -> county, both ways
    import pyogrio
    g = pyogrio.read_dataframe(SA_SHP, read_geometry=False, columns=["SA_GUID__1", "COUNTY_ENG"])
    assert set(g["COUNTY_ENG"]) == set(COUNTY), set(g["COUNTY_ENG"]) ^ set(COUNTY)
    cty = dict(zip(g["SA_GUID__1"], g["COUNTY_ENG"].map(COUNTY)))
    miss_a = set(sa["GUID"]) - set(cty)
    miss_b = set(cty) - set(sa["GUID"])
    assert not miss_a and not miss_b, (len(miss_a), len(miss_b))
    sa["county"] = sa["GUID"].map(cty)

    f = read_f5029()
    langs = [c for c in f.columns if c != "All languages"]
    assert len(langs) == 66, len(langs)
    d = (f[langs].sum(axis=1) - f["All languages"]).abs().max()
    assert d == 0, f"F5029 labels do not sum to All languages (max diff {d})"
    counties = [c for c in f.index if c != "State"]
    assert sorted(counties) == sorted(set(COUNTY.values())), set(counties) ^ set(COUNTY.values())
    assert (f.loc[counties].sum() == f.loc["State"]).all()

    by_c = sa.groupby("county")[["T2_5PL", "T2_5FR", "T2_5ES", "T2_5T"]].sum()
    print("SAPS summed by county against F5029 (max abs difference over 31 counties):")
    for col, lab in [("T2_5PL", "Polish"), ("T2_5FR", "French"), ("T2_5ES", "Spanish"),
                     ("T2_5T", "All languages")]:
        diff = (by_c[col] - f.loc[by_c.index, lab]).abs()
        rel = (diff / f.loc[by_c.index, lab]).max()
        print(f"  {lab:14s} SAPS {by_c[col].sum():>9,}  F5029 {f.loc['State', lab]:>9,.0f}  "
              f"max county diff {diff.max():,.0f} ({100 * rel:.2f}%), "
              f"sum of abs diffs {diff.sum():,.0f}")
        # State totals are exact; counties differ by a few dozen at most (sources/ie.md says why)
        assert by_c[col].sum() == f.loc["State", lab], lab
        assert diff.max() <= 100 and diff.sum() <= 0.005 * by_c[col].sum(), (lab, diff[diff > 0])

    # the county mix of everything SAPS leaves under Other
    rest = [c for c in langs if c not in ("Polish", "French", "Spanish")]
    mix = f.loc[counties, rest]
    mix = mix.div(mix.sum(axis=1), axis=0)

    out = []
    for col, lab in SAPS_NAMED.items():
        t = sa[["GUID", col]].rename(columns={"GUID": "unit", col: "count"})
        t["source_category"] = lab
        t["tier"] = "measured"
        out.append(t)
    oth = sa[sa["T2_5OTH"] > 0]
    for c, grp in oth.groupby("county"):
        for lab in rest:
            share = mix.loc[c, lab]
            if share == 0:
                continue
            out.append(pd.DataFrame({"unit": grp["GUID"], "count": grp["T2_5OTH"] * share,
                                     "source_category": lab, "tier": "derived"}))
    irish = (sa["T3_2DIDOT"] + sa["T3_2DOEST"]) if IRISH_DAILY else sa["T3_2DIDOT"] * 0
    out.append(pd.DataFrame({"unit": sa["GUID"], "count": irish,
                             "source_category": IRISH_LABEL, "tier": "derived"}))
    eng_raw = sa["T1_1AGETT"] - sa["T2_5T"] - irish
    clipped = eng_raw.clip(lower=0)
    out.append(pd.DataFrame({"unit": sa["GUID"], "count": clipped,
                             "source_category": ENGLISH_LABEL, "tier": "derived"}))
    df = pd.concat(out, ignore_index=True)
    df = df[df["count"] > 0]

    tot = sa["T1_1AGETT"].sum()
    print(f"population {tot:,} (CSO 2022: 5,149,139)")
    print(f"other-than-English-or-Irish at home {sa['T2_5T'].sum():,} (F5029 State 751,507)")
    print(f"Irish ability yes {sa['T3_1YES'].sum():,}; daily within and outside education "
          f"{sa['T3_2DIDOT'].sum():,}; daily only outside {sa['T3_2DOEST'].sum():,}; "
          f"drawn as Irish {irish.sum():,}")
    neg = eng_raw[eng_raw < 0]
    print(f"Small Areas where foreign + Irish exceed the population: {len(neg)} "
          f"({-neg.sum():,} people over; English clipped to 0 there)")
    print(f"drawn {df['count'].sum():,.0f} people, {df['unit'].nunique():,} Small Areas, "
          f"{df['source_category'].nunique()} labels")
    nat = df.groupby("source_category")["count"].sum().sort_values(ascending=False)
    for k, v in nat.head(12).items():
        print(f"  {v:>12,.0f}  {k}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df[["unit", "source_category", "count", "tier"]].to_csv(OUT + ".tmp", index=False)
    os.replace(OUT + ".tmp", OUT)
    print(f"wrote {OUT} ({len(df):,} rows)")


if __name__ == "__main__":
    main()
