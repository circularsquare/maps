"""China — Chinese General Social Survey, self-identified religion by province.

Reads data/raw/cn/cgss/*.dta and writes data/normalized/cn_cgss.csv, which is a table of
COEFFICIENTS (province -> share) rather than counts. countries.py::_cn_counts applies it.
See sources/cn_cgss.md for the mirror hunt, the licences and the re-fetch commands.

**THIS IS THE FIRST TIME ANYBODY'S RELIGION IN CHINA IS DRAWN FROM SOMETHING THEY SAID ABOUT
THEMSELVES.** Everything else in this country is spec §14.5's derivation from the census
nationality column — a claim the map makes about people, not a claim they made. CGSS asks
*您的宗教信仰是什么* (which religion do you belong to), which is §3.1's `self_id`, the same
basis Vietnam's census, Korea's, Russia's Arena and France's ESS are drawn on.

**AND THE REASON IT IS CGSS RATHER THAN CFPS IS THE WHOLE OF §14.15's REFRAMING.** CFPS asks
about *belief in Buddha or a bodhisattva* and returns 33% where CGSS returns 4% on the same
population in the same year. §3.1 forbids mixing bases; a CFPS layer could not have sat beside
any other country here. The 2026-09-07 refusal of the CFPS application cost nothing.

    magnitude     the `unknown` residual by county   cn.csv, census 2000 x 2010
    coefficients  share(religion | province)         CGSS 2012 + 2017 + 2021, weighted
    output        religion x county                  tier `modelled`, basis `self_id`

## WHAT IS DRAWN, AND THE TWO THAT ARE NOT

Pooled n = 32,495 over 29 of 31 provinces — 99.2% of China's population. Only Hainan and
Xizang are uncovered; CGSS has never sampled Tibet and §14.5 already draws it from ethnicity.

| | respondents | provinces, cell <10 | rank stability 2012<->2021 | |
|---|---|---|---|---|
| Buddhism -> `buddhism.mahayana` | 1,592 | 3 / 29 | +0.63 | drawn |
| Protestantism -> `christianity.protestant` | 585 | 13 / 29 | **+0.17** | drawn, flagged |
| Islam | 698 | 21 / 29 | +0.64 | **NOT drawn — see below** |
| folk | 681 | 18 / 29 | +0.45 | not drawn |
| Daoism | 80 | 28 / 29 | +0.52 | not drawn |
| Catholicism | 65 | **29 / 29** | +0.16 | not drawn |

**Buddhism passes §14.10 cleanly.** chi-square homogeneity across provinces p = 4e-184;
Zhejiang 15.7% (CI 14.0-17.5) against Anhui 0.9% (0.4-1.4), nowhere near overlapping. The
pattern is the southeastern coastal belt the literature describes — Zhejiang 14.8, Fujian 11.5,
Jiangxi 9.0, Shanghai 7.0 — against Shanxi 1.0, Shandong 1.1, Anhui 1.1, Chongqing 1.4. That is
a gradient, not sampling noise, which is exactly what §14.13 said the 2021 wave alone could not
deliver.

**Protestantism is drawn and it is the weakest thing in this country.** Anita's call,
2026-09-08, on the argument that its SPATIAL variation is highly significant (chi-square
p = 1.3e-84) even though its TEMPORAL stability is poor. Henan comes out top at 6.4%
(CI 5.2-7.6) — China's Protestant heartland, found by the data unaided — with Heilongjiang,
Jiangsu, Zhejiang and Jilin behind it. But the 2012<->2021 rank correlation is **+0.17**, so
the map is less sure this is the CURRENT geography than that it is A geography.
§14.12's disclosure rule says name the weakest drawn cell rather than declaring the country
modelled, and `note_public` does.

**A rank correlation of +0.17 on 19 overlapping provinces is a FAILURE TO DEMONSTRATE SIGNAL,
not a demonstration of noise** — its confidence interval includes zero and also reaches past
+0.55. Some of the instability is likely real: reported Protestantism fell 2.31% -> 1.03%
across the period, and if enforcement varied by province the ordering SHOULD move.

### ISLAM IS NOT DRAWN FROM THIS SOURCE, AND FINDING OUT WHY IS WORTH MORE THAN THE LAYER

CGSS's provincial subsamples come from a handful of PSUs, so where a minority is concentrated
*within* a province the sample either lands on it or misses it entirely:

| | census, Muslim nationalities | CGSS pooled self-id Islam | ratio |
|---|---|---|---|
| Qinghai | 16.9% | 1.1% | **0.07x** |
| Gansu | 7.4% | 0.6% | 0.1x |
| Xinjiang | 58.3% | 92.0% | 1.6x |
| Ningxia | 34.5% | 93.1% | **2.7x** |

**Off by fourteenfold one way and nearly threefold the other. That is not a bias a weight can
correct, it is a lottery over which PSUs were drawn.** §14.5's county-level derivation is
straightforwardly better and stays.

**Nationally, though, the two agree, and this is the first external check §14.5 has ever had:**
CGSS self-id Islam runs 1.87% (2021) to 2.56% (2012); the derivation's 23.07M over 1.259bn is
**1.83%**. The survey finds at least as many Muslims as the derivation predicts. The check is
valid at national level and only there.

**The same finding is why CGSS cannot answer whether the Tibetan coefficient should be under
1.0** — Anita asked, 2026-09-08. Qinghai is the only covered province with a large Tibetan
population (24.4%): expected Buddhism at coefficient 1.0 is 27.3%, observed is 11.8%, which
naively argues for ~0.4. But the same sample found 7% of Qinghai's Muslims, so it is a Han and
urban sample that missed the Tibetans for the same reason it missed the Hui. **The test fails
its own control.** Recorded so it is not re-run.

## THE LEVELS MOVE DOWN MONOTONICALLY AND POOLING CANNOT FIX IT

Weighted national shares, % of adults:

| | 2012 | 2017 | 2021 |
|---|---|---|---|
| **any religion** | **14.47** | **10.61** | **7.50** |
| Buddhism | 6.02 | 4.66 | 3.76 |
| Protestantism | 2.31 | 1.41 | 1.03 |
| folk | 3.43 | 2.11 | **0.27** |
| Islam | 2.56 | 2.23 | 1.87 |

Every category except Catholicism falls, Islam by 27% in a population whose Muslim
nationalities grew. Two mechanisms, not exclusive: real decline in willingness to report a
religion across the 2018 Regulations on Religious Affairs; and a multi-select-to-single-choice
instrument change at 2021, which reliably lowers affirmatives. **The 2012->2017 fall happens
with the instrument held constant**, so the instrument cannot explain all of it.

**THE LEVEL TAKEN HERE IS THE POOLED ONE, AND THE ARGUMENT IS ABOUT THE DENOMINATOR RATHER
THAN ABOUT RELIGION.** The map's China is a 2010-magnitude country: cn.csv carries 2000 county
structure scaled to 2010 provincial totals, so the 1.259 billion people being coloured are the
2010 census's people. Pooled and n-weighted, CGSS's centre of mass is about 2015; the 2021 wave
alone is eleven years after the population it would be describing. **Pooling is the closer fit
to the denominator, not merely the larger sample.** The cost is stated plainly: this layer is
about half again larger than the 2021 wave alone would draw, and if the decline is real rather
than reporting, it runs high.

    Buddhism      pooled 55.4M / 55,407 dots      2021-only 31.0M
    Protestantism pooled 20.4M / 20,372 dots      2021-only 10.4M

For scale, all colour on China before this was 31.5M. **The Buddhist layer alone is larger than
everything previously drawn.**

## TWO IMPLEMENTATION CALLS

**The share is applied to the `unknown` residual, not to the province's whole population.**
Otherwise Qinghai and Gansu would count their Tibetans twice — once as `buddhism.vajrayana`
from ethnicity and again as generic Buddhism here. Anita spotted this. Its cost is in the other
direction and is named in `note_public`: in the four provinces where the derived population is
large, some of CGSS's Buddhist respondents WERE those minority people, so re-spreading the
share over the residual alone runs slightly high there. Everywhere else the derived share is
under 2% and the distinction does not matter.

**The multi-select in 2012/2017 is collapsed to a single category, and that was measured rather
than assumed.** Those waves ask religion as a checkbox block (您的宗教信仰（多选）); 2021 asks a
single-choice A5. Of respondents naming any religion in 2012/2017, **only 2.5% name more than
one** (73 of 2,961 name two, 2 name three). Collapsing is safe and the three waves are
comparable on that axis. This is the §3.1a check — two sources sharing a basis can still differ
in their answer sets — done rather than waved at.
"""

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw", "cn", "cgss")
OUT = os.path.join(ROOT, "data", "normalized", "cn_cgss.csv")

SOURCE_ID = "cn_cgss_2012_2017_2021"

# CGSS province labels -> the English names cn.csv's `note` column carries.
CN2EN = {
    "北京市": "Beijing", "天津市": "Tianjin", "河北省": "Hebei", "山西省": "Shanxi",
    "内蒙古自治区": "Inner Mongolia", "辽宁省": "Liaoning", "吉林省": "Jilin",
    "黑龙江省": "Heilongjiang", "上海市": "Shanghai", "江苏省": "Jiangsu",
    "浙江省": "Zhejiang", "安徽省": "Anhui", "福建省": "Fujian", "江西省": "Jiangxi",
    "山东省": "Shandong", "河南省": "Henan", "湖北省": "Hubei", "湖南省": "Hunan",
    "广东省": "Guangdong", "广西壮族自治区": "Guangxi", "海南省": "Hainan",
    "重庆市": "Chongqing", "四川省": "Sichuan", "贵州省": "Guizhou", "云南省": "Yunnan",
    "西藏自治区": "Xizang", "陕西省": "Shaanxi", "甘肃省": "Gansu", "青海省": "Qinghai",
    "宁夏回族自治区": "Ningxia", "新疆维吾尔自治区": "Xinjiang",
}

# The two categories that survive the §14.10 test, and the nodes they land on. `佛教` is
# generic Buddhism in CGSS's answer set; in the 29 covered provinces it is Han Mahayana
# practice, because the Vajrayana and Theravada populations are drawn from ethnicity and
# Xizang is not sampled at all.
DRAWN = {
    "Buddhism": "buddhism.mahayana",
    "Protestant": "christianity.protestant",
}

# Multi-select waves: one binary per religion. 2021 is single-choice `A5`.
MULTI = {
    "a511": "Buddhism", "a512": "Daoism", "a513": "folk", "a514": "Islam",
    "a515": "Catholic", "a516": "Protestant", "a517": "Orthodox",
    "a518": "otherChristian", "a519": "Judaism", "a520": "Hindu", "a521": "other",
}
SINGLE = {
    "佛教": "Buddhism", "道教": "Daoism", "民间信仰（拜妈祖、关公等）": "folk",
    "回教/伊斯兰教": "Islam", "天主教": "Catholic", "基督教": "Protestant",
    "东正教": "Orthodox", "其他基督教": "otherChristian", "犹太教": "Judaism",
    "印度教": "Hindu", "其他": "other", "不信仰宗教": "none",
}
CATS = list(dict.fromkeys(list(MULTI.values()) + ["none"]))

WAVES = [
    # file, the "no religion" column, the weight column, year
    ("cgss2012_14.dta", "a501", "weight", 2012),
    ("cgss2017.dta", "a51", "weight", 2017),
    ("CGSS2021.dta", None, "weight_raking", 2021),
]


def load():
    """The three waves, harmonised to one row per respondent with binary category flags."""
    frames = []
    for fn, nonecol, wcol, year in WAVES:
        path = os.path.join(RAW, fn)
        if not os.path.exists(path):
            raise SystemExit(f"missing {path} — see sources/cn_cgss.md for the fetch")
        if year == 2021:
            s = pd.read_stata(path, columns=["provinces", "A5", wcol],
                              convert_categoricals=True)
            d = pd.DataFrame({
                "prov": s["provinces"].astype(str).values,
                "w": pd.to_numeric(s[wcol], errors="coerce").fillna(1.0).values,
                "wave": year,
            })
            lab = s["A5"].astype(str).map(SINGLE)
            for c in CATS:
                d[c] = (lab == c).astype(int).values
        else:
            num = pd.read_stata(path, columns=[nonecol] + list(MULTI) + [wcol],
                                convert_categoricals=False)
            # s41 has to be read separately: converting categoricals for the whole frame
            # turns the religion flags into "是"/"否" labels and silently zeroes every count.
            cat = pd.read_stata(path, columns=["s41"], convert_categoricals=True)
            d = pd.DataFrame({
                "prov": cat["s41"].astype(str).values,
                "w": pd.to_numeric(num[wcol], errors="coerce").fillna(1.0).values,
                "wave": year,
            })
            for col, name in MULTI.items():
                d[name] = (pd.to_numeric(num[col], errors="coerce") == 1).astype(int).values
            d["none"] = (pd.to_numeric(num[nonecol], errors="coerce") == 1).astype(int).values
        frames.append(d)

    df = pd.concat(frames, ignore_index=True)
    df["province"] = df["prov"].map(CN2EN)
    unmapped = sorted(set(df["prov"][df["province"].isna()]))
    if unmapped:
        raise SystemExit(f"unmapped CGSS province labels: {unmapped}")
    return df


def shares(df):
    """Province x drawn category -> weighted share, with the unweighted cells beside it."""
    rows = []
    for province, g in df.groupby("province"):
        W = g["w"].sum()
        for cat, node in DRAWN.items():
            rows.append({
                "province": province,
                "node": node,
                "share": (g[cat] * g["w"]).sum() / W if W else np.nan,
                "n": int(len(g)),
                "k": int(g[cat].sum()),
                "waves": int(g["wave"].nunique()),
                "basis": "self_id",
                "source_id": SOURCE_ID,
                "note": f"CGSS pooled 2012/2017/2021; {int(g[cat].sum())} of {len(g)} "
                        f"respondents; {int(g['wave'].nunique())} waves",
            })
    out = pd.DataFrame(rows).sort_values(["node", "share"], ascending=[True, False])
    return out.reset_index(drop=True)


def check(df, out):
    """The three things that would silently break this file."""
    ok = True

    # 1. the multi-select collapse (§3.1a) — overlap must stay negligible
    rel = [c for c in CATS if c != "none"]
    mw = df[df["wave"] != 2021]
    k = mw[rel].sum(axis=1)
    overlap = (k >= 2).sum() / max(1, (k >= 1).sum())
    print(f"  multi-select overlap 2012+2017: {overlap*100:.1f}% of religious respondents "
          f"name >1 religion")
    if overlap > 0.05:
        print("    !! over 5% — the collapse to a single category is no longer safe")
        ok = False

    # 2. the national Islam reconciliation against §14.5's derivation (1.83%)
    for year, g in df.groupby("wave"):
        W = g["w"].sum()
        isl = (g["Islam"] * g["w"]).sum() / W * 100
        print(f"  {year}: self-id Islam {isl:5.2f}%   (derivation says 1.83%)")

    # 3. coverage
    print(f"  provinces: {out['province'].nunique()} of 31    pooled n = {len(df):,}")
    if out["province"].nunique() < 25:
        print("    !! fewer than 25 provinces — a wave probably failed to load")
        ok = False

    # 4. the drawn shares must be shares
    bad = out[(out["share"] < 0) | (out["share"] > 1) | out["share"].isna()]
    if len(bad):
        print(f"    !! {len(bad)} shares outside [0,1]")
        ok = False
    return ok


def main():
    df = load()
    out = shares(df)
    print(f"CGSS pooled: {len(df):,} respondents, {df['wave'].nunique()} waves")
    if not check(df, out):
        raise SystemExit("checks failed")
    out.to_csv(OUT, index=False, encoding="utf-8")
    print(f"\nwrote {OUT}  ({len(out)} rows)")
    for node in sorted(DRAWN.values()):
        sub = out[out["node"] == node].nlargest(5, "share")
        top = ", ".join(f"{r.province} {r.share*100:.1f}%" for r in sub.itertuples())
        print(f"  {node:26s} top: {top}")


if __name__ == "__main__":
    sys.exit(main())
