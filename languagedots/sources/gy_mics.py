"""Guyana: 2012 census ethnic groups by region, with the language of each region's Amerindians
(and the few Spanish, Portuguese and other speakers) taken from MICS6 2019-20 microdata
-> data/normalized/gy_mics.csv.

    python sources/gy_mics.py           (needs data/normalized/gy.csv from sources/gy_census.py)

SOURCE. Guyana Multiple Indicator Cluster Survey 2019-20 (MICS6; Bureau of Statistics, Ministry
of Health, UNICEF), SPSS files from mics.unicef.org (Anita's UNICEF account, 2026-10-09),
unzipped to data/raw/gy/mics_2019/ (gitignored; research use, no redistribution). 8,285
households in 10 regions (HH7), 7,072 interviewed (hhweight > 0), 26,209 household members.

ITEM. HC1B, "Language of household head" (English / Indigenous language / Spanish / Portuguese /
Other language), MICS's household mother-tongue item, read as every member's (hl.sav members x
hhweight). HH16 (respondent's native language) was not used: of 264 households whose head's
language is indigenous, the respondent answered English in 190 and indigenous in 73, and every
questionnaire was English (HH14) and 7,063 of 7,072 interviews (HH15). WM14 (women 15-49) has
only English / Other and puts 193 of 216 women in indigenous-headed households on English.

METHOD: MICS AS RETENTION. The census counts Amerindians per region exactly; MICS's own Amerindian
sample per region is small (17 to 435 households). So the census groups stay, and MICS gives,
per region, the share of Amerindian-headed persons whose head's language is indigenous (also
Spanish, Portuguese, other). That replaces the flat 20% of the first build. The rest of the
population (census African, East Indian, Mixed, Chinese, Portuguese, Other) gets the region's
shares among non-Amerindian-headed persons; white Guyanese stay on English. "English" answers
are drawn as Guyanese Creole, the first build's choice. Region totals are the census's.
"Other language" with an Amerindian head (19 households, 12 of them in three Upper Mazaruni
clusters beside the indigenous answers, 4 with an indigenous-speaking respondent) is read as an
unnamed indigenous language (AM_OTHER); with any other head it stays "other" (3 of them are
Chinese-headed households in East Berbice).

INTERVIEWER CHECK. HC1B depends on who asked. Within the same cluster, some interviewers record
an indigenous language for most Amerindian heads and others for none (Potaro-Siparuni cluster
358: interviewer 834 6 of 6, 833 0 of 6; 833 found none in 34 Amerindian households). For each
interviewer, E = the indigenous households expected from the rate their cluster-mates recorded
in the same clusters, O = what they recorded; an interviewer with E >= MIN_E and Poisson
P(X <= O | E) < ALPHA is a non-recorder and their households are left out of the rate. The
rate is then built per cluster from the remaining interviewers' households and averaged over
clusters weighted by all Amerindian-headed persons there, so dropping an interviewer does not
shift the region towards other clusters. A cluster where every interviewer was dropped keeps
its raw rate. This raises the interior regions' rates; it is printed beside the raw rate.

SHRINKAGE. Amerindian households cluster in villages, so the effective sample is closer to the
number of clusters than of households. Each region's rate is shrunk towards its pool's rate
(interior: Regions 7, 8, 9; coast and north-west: the rest) as (c * r + K * pool) / (c + K),
c = clusters with an Amerindian-headed household, K = 10.

CHECKS. Every member matches a household; HH7 covers exactly the ten units; the census regions
are religiondots' units 1-10; national retention within 10-30% (the IDB 2013 survey's 20% of
households fluent, sources/gy.md); every region total equals gy.csv's.
"""
import csv
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
RAW = HERE / "data" / "raw" / "gy" / "mics_2019"
CENSUS = HERE / "data" / "normalized" / "gy.csv"
OUT = HERE / "data" / "normalized" / "gy_mics.csv"
SOURCE_ID = "mics6_2019_hc1b"
N_HH, N_HL, N_IV = 8_285, 26_209, 7_072
CENSUS_TOTAL = 746_955

# MICS HH7 -> census region / religiondots unit
REGION = {"BARIMA-WAINI": "1", "POMEROON-SUPENAAM": "2", "ESSEQUIBO ISLANDS-WEST DEMERARA": "3",
          "DEMERARA-MAHAICA": "4", "MAHAICA-BERBICE": "5", "EAST BERBICE-CORENTYNE": "6",
          "CUYUNI-MAZARUNI": "7", "POTARO-SIPARUNI": "8", "UPPER TAKUTU-UPPER ESSEQUIBO": "9",
          "UPPER DEMERARA-BERBICE": "10"}
NAME = {v: k.title() for k, v in REGION.items()}
INTERIOR = {"7", "8", "9"}
# HC1B answer -> label written to gy_mics.csv (taxonomy/gy2019.py maps these)
LABEL = {"ENGLISH": "Creole (English answer)", "INDIGENOUS LANGUAGE": "Indigenous language",
         "SPANISH": "Spanish", "PORTUGUESE": "Portuguese", "OTHER LANGUAGE": "Other language"}
WHITE = "English (white Guyanese)"
# "Other language" with an Amerindian head: 19 households, 12 of them in three Upper Mazaruni
# clusters (319, 322, 323) beside that region's indigenous answers, 4 with the respondent
# answering an indigenous language (HH16). Read as an unnamed indigenous language.
AM_OTHER = "Other language (Amerindian head)"
SURVEY_LABELS = [v for v in LABEL.values() if v != LABEL["ENGLISH"]] + [AM_OTHER]
MIN_E, ALPHA, K = 3.0, 0.05, 10.0
IDB_RANGE = (10.0, 30.0)


def poisson_cdf(o, e):
    return sum(math.exp(-e) * e ** k / math.factorial(k) for k in range(int(o) + 1))


def shares(d):
    """{label: share} of person weight in d"""
    s = d.groupby("label")["hhweight"].sum()
    return (s / s.sum()).to_dict() if s.sum() else {}


def main():
    import pandas as pd
    import pyreadstat
    hh, _ = pyreadstat.read_sav(str(RAW / "hh.sav"), apply_value_formats=True,
                                usecols=["HH1", "HH2", "HH3", "HH7", "HH16", "HC1B", "HC2",
                                         "hhweight"])
    hl, _ = pyreadstat.read_sav(str(RAW / "hl.sav"), usecols=["HH1", "HH2", "HL1", "hhweight"])
    if len(hh) != N_HH or len(hl) != N_HL:
        raise SystemExit(f"hh {len(hh)} / hl {len(hl)} rows, expected {N_HH} / {N_HL}")
    iv = hh[hh["hhweight"] > 0].copy()
    if len(iv) != N_IV:
        raise SystemExit(f"{len(iv)} interviewed households, expected {N_IV}")
    if set(iv["HH7"].astype(str)) != set(REGION):
        raise SystemExit(f"HH7 regions {sorted(set(iv['HH7'].astype(str)))} != REGION")
    iv["unit"] = iv["HH7"].astype(str).map(REGION)
    iv["label"] = iv["HC1B"].astype(str).map(LABEL)
    if iv["label"].isna().any():
        raise SystemExit(f"HC1B answers with no label: {set(iv.loc[iv.label.isna(), 'HC1B'])}")
    iv["am"] = iv["HC2"].astype(str) == "AMERINDIAN"
    iv.loc[iv["am"] & (iv["label"] == "Other language"), "label"] = AM_OTHER
    iv["ind"] = (iv["label"] == "Indigenous language").astype(int)
    persons = hl.merge(iv.drop(columns="hhweight"), on=["HH1", "HH2"], how="left",
                       indicator=True)
    if persons.groupby("_merge", observed=True).size().get("right_only", 0):
        raise SystemExit("interviewed households with no members")
    persons = persons[persons["_merge"] == "both"]
    print(f"  {len(hh):,} households, {len(iv):,} interviewed, {len(persons):,} members in them")

    x = pd.crosstab(iv["HC1B"], iv["HH16"])
    ih = x.loc["INDIGENOUS LANGUAGE"]
    print(f"  HC1B indigenous {int(ih.sum())} households: HH16 English {int(ih['ENGLISH'])}, "
          f"indigenous {int(ih['INDIGENOUS LANGUAGE'])} (HH16 not used)")

    # --- interviewer check, Amerindian-headed households
    am = iv[iv["am"]]
    cl = am.groupby(["HH1", "HH3"]).agg(n=("ind", "size"), o=("ind", "sum")).reset_index()
    tot = cl.groupby("HH1")[["n", "o"]].sum()
    cl = cl.join(tot, on="HH1", rsuffix="_c")
    cl["mate_n"] = cl["n_c"] - cl["n"]
    cl["e"] = (cl["o_c"] - cl["o"]) / cl["mate_n"].where(cl["mate_n"] > 0) * cl["n"]
    cmp_ = cl[cl["mate_n"] > 0]
    per = cmp_.groupby("HH3")[["n", "o", "e"]].sum()
    per["p"] = [poisson_cdf(r.o, r.e) for r in per.itertuples()]
    drop = set(per[(per["e"] >= MIN_E) & (per["p"] < ALPHA)].index)
    region_of = am.groupby("HH3")["unit"].first()
    print(f"  interviewers recording far fewer indigenous heads than their cluster-mates "
          f"(E >= {MIN_E:.0f}, P < {ALPHA}):")
    for i, r in per[per["e"] >= MIN_E].sort_values("p").iterrows():
        print(f"    {int(i):>4} region {region_of[i]:>2}: recorded {int(r.o):>2} of {int(r.n):>2}, "
              f"cluster-mates' rate implies {r.e:4.1f}, P {r.p:.4f}"
              f"{'  DROPPED' if i in drop else ''}")

    # --- per region rates among Amerindian-headed persons
    pa = persons[persons["am"]]
    cw = pa.groupby(["unit", "HH1"])["hhweight"].sum()
    raw, corr, nclu = {}, {}, {}
    for u, d in pa.groupby("unit"):
        raw[u] = shares(d)
        nclu[u] = d["HH1"].nunique()
        acc = {}
        for c, dc in d.groupby("HH1"):
            keep = dc[~dc["HH3"].isin(drop)]
            s = shares(keep if len(keep) else dc)
            for k, v in s.items():
                acc[k] = acc.get(k, 0) + v * cw[(u, c)]
        t = sum(acc.values())
        corr[u] = {k: v / t for k, v in acc.items()}
    if set(raw) != set(REGION.values()):
        raise SystemExit("a region has no Amerindian-headed household")
    pool = {}
    for name, units in (("interior", INTERIOR), ("coast", set(REGION.values()) - INTERIOR)):
        acc = {}
        for u in units:
            w = cw[u].sum()
            for k, v in corr[u].items():
                acc[k] = acc.get(k, 0) + v * w
        t = sum(acc.values())
        pool[name] = {k: v / t for k, v in acc.items()}
    final = {}
    for u in REGION.values():
        p = pool["interior" if u in INTERIOR else "coast"]
        c = nclu[u]
        labs = set(corr[u]) | set(p)
        final[u] = {k: (c * corr[u].get(k, 0) + K * p.get(k, 0)) / (c + K) for k in labs}

    # --- rest of the population: non-Amerindian-headed persons, raw
    rest = {u: shares(d) for u, d in persons[~persons["am"]].groupby("unit")}

    # --- census groups
    cen = pd.read_csv(CENSUS, dtype={"geo_id": str})
    if int(cen["count"].sum()) != CENSUS_TOTAL or set(cen["geo_id"]) != set(REGION.values()):
        raise SystemExit("gy.csv is not the 2012 census table: rerun sources/gy_census.py")
    g = cen.pivot_table(index="geo_id", columns="source_category", values="count",
                        aggfunc="sum", fill_value=0)
    ret_natl = sum(g.loc[u, "Amerindian"] * final[u].get("Indigenous language", 0)
                   for u in g.index) / g["Amerindian"].sum() * 100
    raw_natl = sum(g.loc[u, "Amerindian"] * raw[u].get("Indigenous language", 0)
                   for u in g.index) / g["Amerindian"].sum() * 100

    print("  Amerindian-headed persons with an indigenous-language head, % (weighted):")
    print("    region                              hh  clusters   raw  checked  drawn | other  es+pt")
    for u in sorted(REGION.values(), key=int):
        n = int(am[am["unit"] == u].shape[0])
        f = final[u]
        print(f"    {u:>2} {NAME[u]:<31} {n:>4} {nclu[u]:>6} {raw[u].get('Indigenous language', 0) * 100:7.1f}"
              f" {corr[u].get('Indigenous language', 0) * 100:7.1f} {f.get('Indigenous language', 0) * 100:7.1f}"
              f" | {f.get(AM_OTHER, 0) * 100:5.1f} {(f.get('Spanish', 0) + f.get('Portuguese', 0)) * 100:5.1f}")
    print(f"  national (census Amerindians x rate): raw {raw_natl:.1f}%, drawn {ret_natl:.1f}% "
          f"(IDB 2013: 20% of households fluent)")
    if not IDB_RANGE[0] <= ret_natl <= IDB_RANGE[1]:
        raise SystemExit(f"national retention {ret_natl:.1f}% outside {IDB_RANGE}")

    out, natl = [], {}
    for u in sorted(g.index, key=int):
        pop = int(g.loc[u].sum())
        amer = g.loc[u, "Amerindian"]
        white = g.loc[u, "White"]
        others = pop - amer - white
        rawc = {WHITE: float(white)}
        for k, v in final[u].items():
            rawc[k] = rawc.get(k, 0) + amer * v
        for k, v in rest[u].items():
            rawc[k] = rawc.get(k, 0) + others * v
        cnt = {k: int(v) for k, v in rawc.items()}
        for k in sorted(rawc, key=lambda k: rawc[k] - cnt[k], reverse=True)[:pop - sum(cnt.values())]:
            cnt[k] += 1
        assert sum(cnt.values()) == pop, u
        for lab in sorted(cnt, key=cnt.get, reverse=True):
            if not cnt[lab]:
                continue
            natl[lab] = natl.get(lab, 0) + cnt[lab]
            survey = lab in SURVEY_LABELS
            note = ("MICS6 HC1B shares (Amerindian-headed persons, interviewer-checked, shrunk; "
                    "others raw) x 2012 census groups" if survey else
                    "2012 census ethnic groups read as language, less the MICS6 shares")
            out.append(dict(geo_id=u, geo_level="region", geo_name=NAME[u], source_category=lab,
                            count=cnt[lab], tier="modelled" if survey else "derived",
                            source_id=SOURCE_ID, year=2019, note=note))
    if sum(natl.values()) != CENSUS_TOTAL:
        raise SystemExit("national total moved")
    print("  Amerindian language drawn per region (first build: 20% of census Amerindians):")
    for u in sorted(g.index, key=int):
        rows = {r["source_category"]: r["count"] for r in out if r["geo_id"] == u}
        now = rows.get("Indigenous language", 0) + rows.get(AM_OTHER, 0)
        pop = int(g.loc[u].sum())
        print(f"    {u:>2} {NAME[u]:<31} before {round(g.loc[u, 'Amerindian'] * 0.2):>6,}  "
              f"after {now:>6,}  ({now / pop * 100:4.1f}% of {pop:,}); Spanish "
              f"{rows.get('Spanish', 0):,}, Portuguese {rows.get('Portuguese', 0):,}, other "
              f"{rows.get('Other language', 0):,}")

    tmp = OUT.with_suffix(".part")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["geo_id", "geo_level", "geo_name", "source_category",
                                          "count", "tier", "source_id", "year", "note"])
        w.writeheader()
        w.writerows(out)
    tmp.replace(OUT)
    print(f"  wrote {OUT.relative_to(HERE)}: {len(out)} rows, {sum(natl.values()):,} people")
    for lab, n in sorted(natl.items(), key=lambda kv: -kv[1]):
        print(f"      {lab:<26} {n:>9,}  {n / CENSUS_TOTAL * 100:6.2f}%")


if __name__ == "__main__":
    main()
