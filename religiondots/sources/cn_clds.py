"""China — Chinese Labor-force Dynamics Survey 2016, self-identified religion by province.

Reads data/raw/cn/clds/ and writes data/normalized/cn_clds.csv, a table of COEFFICIENTS
(province -> share) in the same schema as cn_cgss.csv. See sources/cn_clds.md for the
provenance, the licence and what the archive contained.

**NOTHING HERE IS DRAWN, AND THAT IS THE DECISION RATHER THAN A GAP** — spec §14.20, Anita,
2026-09-08. `_cn_counts` applies cn_cgss.csv alone and should keep doing so. This module runs
the checks that make the CGSS layer defensible; the CSV it writes is evidence, not input.
It also cannot add geography: `COUNTY` is randomised within the city by the depositor, and
only 11 of 157 prefectures carry ten Protestant respondents.

    magnitude     the `unknown` residual by county   cn.csv, census 2000 x 2010
    coefficients  share(religion | province)         CLDS 2016, weighted by `wpp`
    output        province x node                    tier `modelled`, basis `self_id`

## WHY THIS SURVEY IS WORTH THE TROUBLE: IT IS AN INDEPENDENT SECOND OPINION

Every coloured Han dot in China rests on ONE survey's provincial gradient. §14.16 drew it and
said so; §14.10's fifth condition asks for stability, and the only stability CGSS could report
was across its own waves, which is not the same thing as across instruments. CLDS is a
different survey, a different fieldwork house (Sun Yat-sen, not Renmin), a different sample
and a different questionnaire, asking the same §3.1 `self_id` question: `I7_1 宗教信仰`.

21,086 respondents, 29 provinces, 402 communities. That is two thirds of the pooled CGSS
sample in a single wave.

**Its answer set is RICHER than CGSS's in one place and POORER in another**, and both matter:

| | CGSS | CLDS |
|---|---|---|
| Buddhism | 佛教 only | **佛教 and 藏传佛教 separately** |
| folk religion | 民间信仰（拜妈祖、关公等） | **absent** |
| instrument | multi-select 2012/2017, single 2021 | single choice |

## WHAT IT SAYS, AND THE THREE FINDINGS

**1. The national level lands exactly where CGSS's trend predicts, which is the strongest
corroboration available.** CGSS has any-religion at 14.47% (2012), 10.61% (2017), 7.50%
(2021). CLDS 2016, an entirely separate survey, returns **12.20%** — between the 2012 and 2017
CGSS readings, in the right year, on the right slope. §14.16 attributed that decline partly to
an instrument change at 2021; a fourth point from a different instrument agreeing with the
2012-2017 segment says the decline is mostly real.

**2. The Protestant layer is replicated, and this is what the source was chased for.**
§14.16 drew Protestantism flagged, on a 2012<->2021 rank correlation of **+0.17** that its own
docstring calls *"a failure to demonstrate signal"*. Across surveys the answer is different:
**Spearman +0.595** between CLDS 2016 and CGSS pooled over 29 provinces, and CLDS independently
puts **Henan first at 10.9%** with the largest Protestant cell in either survey (97
respondents). Two surveys that share no fieldwork agreeing on the ordering is the evidence
§14.10's fifth condition was actually asking for; CGSS's wave-to-wave wobble measures the
temporal instability of reporting, not the absence of a geography.

**3. Buddhism agrees on SHAPE and disagrees on LEVEL, by a factor that is not folk religion.**
Spearman +0.596; Zhejiang and Fujian are first and second in both. But CLDS puts Zhejiang at
36.1% against CGSS's 14.8%, and Fujian at 32.1% against 11.5%. The obvious explanation is that
CLDS offers no folk-religion option, so Mazu and Guandi worshippers pick 佛教 — and **it is
wrong**: the gap correlates with CGSS's folk share at only **+0.10**, adding folk to CGSS
*lowers* agreement with CLDS (Pearson +0.662 -> +0.492), and the two provinces with the
largest gaps have among the LOWEST folk shares in CGSS (Zhejiang 0.24%) while Guangdong, with
the highest (22.5%), has no gap at all. Recorded so it is not re-proposed.

Nor is it the universe. CLDS's weight sums to 965M against a 15-64 population of ~1.00bn and
it under-represents the over-65s, who are usually assumed more religious — but in CLDS they
are not: any-religion runs 12.1 / 13.0 / 12.0 / 11.7% across the 15-29, 30-44, 45-59 and 60+
bands. **Restricting CLDS to CGSS's 18+ universe moves Buddhism from 6.50% to 6.51%.** The
level difference is between the surveys, not between their populations.

## THE PSU COUNT IS THE THING TO CONDITION ON, AND CLDS PROVES IT BOTH WAYS

§14.16 found that CGSS's provincial cut of Islam was a lottery, because a concentrated
minority is either in the sampled communities or not. CLDS fails the same test in the opposite
direction, which turns a suspicion into a rule:

| | census, Muslim nationalities | CGSS pooled | CLDS 2016 | CLDS communities |
|---|---|---|---|---|
| Xinjiang | 58.3% | 92.0% | **61.3%** | 17 |
| Ningxia | 34.5% | 90.4% | **1.1%** | **4** |
| Qinghai | 16.9% | 1.1% | 20.6% | **4** |
| Yunnan | 1.5% | 10.4% | 0.2% | 12 |

**Ningxia's four communities returned four Muslims between them** — one community at 5%, three
at 0% — in a province a third of whose people are Hui. Those same four communities are where
its **18.4% Buddhist** figure comes from, the largest disagreement with CGSS anywhere in the
country and pure sampling accident.

**Where the community count is high the survey reproduces the census margin almost exactly.**
Xinjiang's 17 communities run from 97% Muslim to 3%, which is the real north-south structure
of the province, and the weighted total lands within three points of the census. Zhejiang's
36% is likewise spread over all 17 of its communities (74, 65, 62, 59, 52, 49, 41, 38, 35, 23,
22, 16, 16, 9, 8, 3, 3%) rather than concentrated in a few, so the level is a real property of
the sample and not an artefact.

**So the rule is about design, not about which survey is better:** a province drawn from four
communities carries no usable religion estimate for a spatially clustered group, whichever
survey drew it. `check()` reports the community count beside every province for exactly this
reason, and `MIN_PSU` is where a threshold would go if one is ever wanted.
"""

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw", "cn", "clds")
OUT = os.path.join(ROOT, "data", "normalized", "cn_clds.csv")

SOURCE_ID = "cn_clds_2016"

# The individual file, inside the archive's own Chinese directory names.
DTA = os.path.join(
    RAW, "2016", "中国劳动力动态调查2016",
    "CLDS2016全部数据(STATA)171106",
    "CLDS2016individual_STATA_171106.dta")

# I7_1's codes, from the file's own `I7_1a` value-label set. Note 藏传佛教 is a separate
# answer here, which CGSS does not offer.
REL = {
    1: "Catholic", 2: "Protestant", 3: "Buddhism", 4: "TibetanBuddhism",
    5: "Daoism", 6: "Islam", 7: "Orthodox", 8: "OtherReligion", 9: "none",
}

# The two categories that survive §14.10, matching cn_cgss.py's DRAWN so the tables can be
# compared row for row. Islam is deliberately absent for the reason in the docstring.
DRAWN = {
    "Buddhism": "buddhism.mahayana",
    "Protestant": "christianity.protestant",
}

# GB/T 2260 province code -> the English names cn.csv's `note` column carries.
PROV = {
    11: "Beijing", 12: "Tianjin", 13: "Hebei", 14: "Shanxi", 15: "Inner Mongolia",
    21: "Liaoning", 22: "Jilin", 23: "Heilongjiang", 31: "Shanghai", 32: "Jiangsu",
    33: "Zhejiang", 34: "Anhui", 35: "Fujian", 36: "Jiangxi", 37: "Shandong",
    41: "Henan", 42: "Hubei", 43: "Hunan", 44: "Guangdong", 45: "Guangxi",
    46: "Hainan", 50: "Chongqing", 51: "Sichuan", 52: "Guizhou", 53: "Yunnan",
    54: "Xizang", 61: "Shaanxi", 62: "Gansu", 63: "Qinghai", 64: "Ningxia",
    65: "Xinjiang",
}

# Below this many sampled communities a province's estimate for a spatially clustered group
# is a lottery rather than a measurement — Ningxia and Qinghai are the demonstration. Nothing
# is filtered on it yet; check() reports it.
MIN_PSU = 8


def fix(s):
    """The .dta carries GB18030 label bytes that pandas hands back decoded as latin-1.

    Every label is mojibake until it is round-tripped, which is why a naive grep for 宗教
    over the variable labels finds NOTHING and the file looks like it has no religion
    question at all. It has one.
    """
    if not s:
        return ""
    for enc in ("gb18030", "utf-8"):
        try:
            return s.encode("latin-1").decode(enc)
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
    return s


def load():
    """One row per respondent: province, community, weight, single religion category."""
    if not os.path.exists(DTA):
        raise SystemExit(f"missing {DTA} — see sources/cn_clds.md for the archive layout")
    d = pd.read_stata(
        DTA, columns=["PROV2016", "CITY", "CID2016", "birthyear", "I7_1", "wpp"],
        convert_categoricals=False)
    d["province"] = d["PROV2016"].map(PROV)
    unmapped = sorted(set(d["PROV2016"].dropna()) - set(PROV))
    if unmapped:
        raise SystemExit(f"unmapped CLDS province codes: {unmapped}")
    d["rel"] = d["I7_1"].map(REL)
    d["w"] = pd.to_numeric(d["wpp"], errors="coerce").fillna(0.0)
    by = pd.to_numeric(d["birthyear"], errors="coerce")
    d["age"] = 2016 - by.where(by.between(1900, 2016))
    return d


def shares(d):
    """Province x drawn category -> weighted share, with the unweighted cells beside it."""
    rows = []
    for province, g in d.dropna(subset=["province"]).groupby("province"):
        W = g["w"].sum()
        for cat, node in DRAWN.items():
            k = int((g["rel"] == cat).sum())
            rows.append({
                "province": province,
                "node": node,
                "share": g.loc[g["rel"] == cat, "w"].sum() / W if W else np.nan,
                "n": int(len(g)),
                "k": k,
                "psu": int(g["CID2016"].nunique()),
                "basis": "self_id",
                "source_id": SOURCE_ID,
                "note": f"CLDS 2016; {k} of {len(g)} respondents over "
                        f"{g['CID2016'].nunique()} communities",
            })
    out = pd.DataFrame(rows).sort_values(["node", "share"], ascending=[True, False])
    return out.reset_index(drop=True)


def census_muslim_share():
    """Province -> share of population in the nationalities §14.5 sends to `islam`.

    This is the margin the survey is tested against. It is the only external check either
    survey has at province level, and it is the one that exposes a four-community sample.
    """
    sys.path[:0] = [os.path.join(ROOT, "taxonomy")]
    import cn2000
    cn = pd.read_csv(os.path.join(ROOT, "data", "normalized", "cn.csv"))
    prov = cn["note"].str.extract(r"province=([^;]+)")[0]
    cn = cn.assign(province=prov)
    tot = cn[cn["source_category"] == "Total"].groupby("province")["count"].sum()
    isl = {}
    for cat, g in cn[cn["source_category"] != "Total"].groupby("source_category"):
        share = sum(s for node, s, _ in cn2000.shares(cat) if node == "islam")
        if share:
            for p, c in g.groupby("province")["count"].sum().items():
                isl[p] = isl.get(p, 0.0) + c * share
    return pd.Series(isl) / tot


def check(d, out):
    """The four things that decide whether this source may be used at all."""
    ok = True
    W = d["w"].sum()

    # 1. the national level, against CGSS's own three readings
    any_ = 100 * (1 - d.loc[d["rel"] == "none", "w"].sum() / W)
    bud = 100 * d.loc[d["rel"] == "Buddhism", "w"].sum() / W
    prot = 100 * d.loc[d["rel"] == "Protestant", "w"].sum() / W
    isl = 100 * d.loc[d["rel"] == "Islam", "w"].sum() / W
    print(f"  national, weighted: any {any_:.2f}%  Buddhism {bud:.2f}%  "
          f"Protestant {prot:.2f}%  Islam {isl:.2f}%")
    print(f"    CGSS reads any 14.47 (2012) / 10.61 (2017) / 7.50 (2021); "
          f"CLDS is 2016 and should sit between the first two")
    if not 7.50 <= any_ <= 14.47:
        print("    !! outside the CGSS range — the weight or the recode is wrong")
        ok = False

    # 2. the Islam margin, which is the lottery test and the reason Islam is not drawn
    marg = census_muslim_share()
    obs = d[d["rel"].notna()].groupby("province").apply(
        lambda g: g.loc[g["rel"] == "Islam", "w"].sum() / g["w"].sum(),
        include_groups=False)
    psu = d.groupby("province")["CID2016"].nunique()
    cmp = pd.DataFrame({"census": marg, "clds": obs, "psu": psu}).dropna()
    cmp = cmp[cmp["census"] > 0.01].sort_values("census", ascending=False)
    print("\n  Islam against the census margin, the check no other source can give:")
    print(f"    {'province':16s} {'census':>8s} {'CLDS':>8s} {'ratio':>7s} {'PSU':>4s}")
    for p, r in cmp.iterrows():
        ratio = r["clds"] / r["census"] if r["census"] else np.nan
        flag = "  <-- lottery" if r["psu"] < MIN_PSU else ""
        print(f"    {p:16s} {100*r['census']:7.1f}% {100*r['clds']:7.1f}% "
              f"{ratio:6.2f}x {r['psu']:4.0f}{flag}")

    # 3. thin provinces, named rather than counted
    thin = out[out["psu"] < MIN_PSU]["province"].unique()
    print(f"\n  provinces sampled from fewer than {MIN_PSU} communities: "
          f"{', '.join(sorted(thin)) if len(thin) else 'none'}")

    # 4. shares must be shares, and coverage must be what the archive claims
    print(f"  provinces: {out['province'].nunique()} of 31    n = {len(d):,}    "
          f"communities = {d['CID2016'].nunique()}")
    if out["province"].nunique() < 25:
        print("    !! fewer than 25 provinces — the file probably failed to load")
        ok = False
    bad = out[(out["share"] < 0) | (out["share"] > 1) | out["share"].isna()]
    if len(bad):
        print(f"    !! {len(bad)} shares outside [0,1]")
        ok = False
    return ok


def venues():
    """Do the BUILDINGS agree with the ANSWERS? — the check no other source here can give.

    The community questionnaire records, for each of the 402 sampled communities, whether the
    interviewer found a church (C68), a temple (C69), a mosque (C70), a Daoist temple (C71)
    and an ancestral hall (C67). §14.15 ruled out the registered-venue registry as a MAGNITUDE
    source and was right — §2.6 is the Thailand mistake and China's registry omits house
    churches. This is a different object: an observation of a place, used as a COHERENCE
    check. If the people who say they are Protestant live in the communities that have
    churches, the self-report is measuring something locally real rather than a mood.

    **The yes/no coding is 1 = 没有 and 2 = 有, which is the reverse of the obvious guess**,
    and taking the guess turns every ratio below into its reciprocal and the whole finding
    upside down. Read it off the converted categoricals, never off the raw code.
    """
    com_path = os.path.join(
        RAW, "2016", "中国劳动力动态调查2016",
        "CLDS2016全部数据(STATA)171106",
        "CLDS2016community_(STATA)_170908.dta")
    if not os.path.exists(com_path):
        print("\n  community file absent — venue coherence check skipped")
        return
    pairs = [("C68", "church", "Protestant"), ("C69", "temple", "Buddhism"),
             ("C70", "mosque", "Islam"), ("C67", "ancestral hall", "Buddhism")]
    com = pd.read_stata(com_path, columns=["CID2016"] + [c for c, _, _ in pairs],
                        convert_categoricals=True)
    for c, _, _ in pairs:
        com[c] = com[c].astype(object).map(
            lambda s: fix(s) if isinstance(s, str) else s)

    ind = pd.read_stata(DTA, columns=["CID2016", "I7_1"], convert_categoricals=False)
    ind["rel"] = ind["I7_1"].map(REL)
    per = ind[ind["rel"].notna()].groupby("CID2016").apply(
        lambda g: pd.Series({k: (g["rel"] == k).mean()
                             for k in ["Protestant", "Buddhism", "Islam"]}),
        include_groups=False)
    j = per.join(com.set_index("CID2016"), how="inner")

    print("\n  places of worship the interviewer found, against what people said:")
    print(f"    {'venue':16s} {'has':>4s} {'hasnt':>6s}   share of the matching self-report")
    for col, lab, rel in pairs:
        y = j[j[col] == "有"][rel]
        n = j[(j[col] != "有") & j[col].notna()][rel]
        if not len(y) or not len(n) or not n.mean():
            continue
        print(f"    {lab:16s} {len(y):4d} {len(n):6d}   {100*y.mean():6.2f}%  vs "
              f"{100*n.mean():6.2f}%   {y.mean()/n.mean():5.1f}x")


def agree(out):
    """Cross-survey agreement with CGSS — the whole point of the source."""
    cg_path = os.path.join(ROOT, "data", "normalized", "cn_cgss.csv")
    if not os.path.exists(cg_path):
        print("\n  cn_cgss.csv absent — run sources/cn_cgss.py for the comparison")
        return
    cg = pd.read_csv(cg_path)
    print("\n  agreement with CGSS pooled, over the provinces both cover:")
    for node in sorted(DRAWN.values()):
        a = out[out["node"] == node].set_index("province")["share"]
        b = cg[cg["node"] == node].set_index("province")["share"]
        j = pd.DataFrame({"clds": a, "cgss": b}).dropna()
        sp = j["clds"].rank().corr(j["cgss"].rank())
        print(f"    {node:26s} Spearman {sp:+.3f} over {len(j)} provinces "
              f"(CGSS's own 2012<->2021 was +0.63 Buddhism, +0.17 Protestant)")


def main():
    d = load()
    out = shares(d)
    print(f"CLDS 2016: {len(d):,} respondents, {d['province'].nunique()} provinces, "
          f"{d['CID2016'].nunique()} communities")
    if not check(d, out):
        raise SystemExit("checks failed")
    venues()
    agree(out)
    out.to_csv(OUT, index=False, encoding="utf-8")
    print(f"\nwrote {OUT}  ({len(out)} rows)")
    for node in sorted(DRAWN.values()):
        sub = out[out["node"] == node].nlargest(5, "share")
        top = ", ".join(f"{r.province} {r.share*100:.1f}%" for r in sub.itertuples())
        print(f"  {node:26s} top: {top}")


if __name__ == "__main__":
    sys.exit(main())
