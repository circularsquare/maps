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

**Five waves since 2026-09-08: 2010, 2012, 2013, 2017, 2021.** Pooled n = 55,637 over 30 of 31
provinces. 2010 and 2013 came from the open replication mirror `doi:10.7910/DVN/R1S5RP` rather
than from CNSDA, whose every download needs a reviewed application; `sources/cn_cgss_fetch.py`
pulls them and `cn_cgss.md` is the record. **2011 is deliberately not pooled** — it is small
(5,620) and unweighted, and Anita's call was to keep it back as an independent check.

**Hainan is now covered**, from CGSS 2010, which is the only wave that samples all 31. Xizang
is sampled by that wave too and is **dropped on purpose**: see DROP_PROVINCES.

| | respondents | provinces, cell <10 | median rank stability | worst pair | |
|---|---|---|---|---|---|
| Buddhism -> `buddhism.mahayana` | 2,793 | 3 / 30 | **+0.711** | +0.633 | drawn |
| Islam | 1,224 | 19 / 30 | +0.756 | +0.590 | **NOT drawn — see below** |
| folk -> `chinesefolk` | 1,173 | 16 / 30 | +0.565 | +0.446 | **drawn, 2026-09-08** |
| Protestantism -> `christianity.protestant` | 1,016 | 9 / 30 | **+0.559** | +0.166 | drawn |
| Daoism | 143 | 25 / 30 | +0.408 | +0.064 | not drawn |
| Catholicism | 129 | 28 / 30 | +0.229 | +0.059 | not drawn |

**Folk religion is drawn on all five waves and is the second largest colour in China.**
Anita's call, 2026-09-08, on the observation that China had no colour a Chinese reader would
recognise as Chinese. It clears the bar Protestantism cleared on most axes: more respondents
(1,173 against 1,016), better rank stability (+0.565 against +0.559), and a gradient the
literature would predict unaided — **Fujian and Guangdong around 19%**, which is the Mazu and
Guandi coast the answer option literally names, against under 1% across the north.

**IT IS ALSO THE SOFTEST NUMBER ON THIS MAP AND THE DISCLOSURE MATTERS MORE THAN THE LAYER.**
Three things, all in `note_public`:

- **Its size moves 16.6x with the question format**, 4.40% to 0.27% across waves, against
  Buddhism's 1.5x and Protestantism's 1.8x. No other drawn category is close to this.
- **Sixteen of thirty provinces hold fewer than ten folk respondents**, against
  Protestantism's nine. They contribute only 1.7M of the drawn total, about 4%, so this is a
  smaller problem than it sounds, but Jilin's entire folk population rests on one respondent.
- **About 8.5% of folk respondents also named Buddhism** (75 people), and because the
  multi-select flags are independent those people are drawn twice, once on each node. In
  Fujian that is 3.6 points of the province. Bounded, disclosed, not corrected.

**And the gap the layer cannot show is the whole point of it.** The Spiritual Life Study of
Chinese Residents (2007, ARDA, `sources/cn_slsc.md`) asks both questions of the same 7,021
people: **15.8% say they have a religious belief, and only 37.6% say they never worship a god,
spirit or ancestor.** Four times as many people practise as name it. This layer draws the
naming.

**Daoism and Catholicism stay out on cell size and nothing else.** 143 and 129 respondents,
25 and 28 provinces under ten. Worth knowing before anyone reopens it: the familiar
"hundreds of millions of Daoists" figures are BELIEF measures. By self-identification, which
is the basis this map is drawn on, Daoism really is about 0.3% of China, so the small number
is the finding rather than a failure to find.

**Confucianism cannot be drawn from this source at any sample size, because it is not an
answer.** Neither CGSS's list nor CLDS's offers 儒教. Korea and Thailand carry Confucian dots
because their own censuses ask; China's survey does not, and no processing invents it.

**Rank stability is now a median over TEN wave pairs rather than one**, and that changes the
reading of the whole file. See the Protestantism section.

**Buddhism passes §14.10 cleanly.** chi-square homogeneity across provinces p = 4e-184;
Zhejiang 15.7% (CI 14.0-17.5) against Anhui 0.9% (0.4-1.4), nowhere near overlapping. The
pattern is the southeastern coastal belt the literature describes — Zhejiang 14.8, Fujian 11.5,
Jiangxi 9.0, Shanghai 7.0 — against Shanxi 1.0, Shandong 1.1, Anhui 1.1, Chongqing 1.4. That is
a gradient, not sampling noise, which is exactly what §14.13 said the 2021 wave alone could not
deliver.

**Protestantism is drawn, and the case for it got much stronger on 2026-09-08.** It was drawn
on Anita's call on the argument that its SPATIAL variation is highly significant (chi-square
p = 1.3e-84) even though its TEMPORAL stability looked poor. Henan comes out top at 6.8% —
China's Protestant heartland, found by the data unaided — with Heilongjiang, Zhejiang, Jiangsu
and Jilin behind it.

***THE +0.17 WAS ONE UNLUCKY PAIR, NOT THE TRUTH, AND THIS IS THE LESSON OF THE WHOLE FILE.***
With three waves there was exactly ONE pair to correlate, 2012<->2021, and it returned +0.17.
Five waves give ten pairs:

    2010 vs 2017  +0.857      2012 vs 2013  +0.702      2013 vs 2021  +0.409
    2010 vs 2012  +0.742      2012 vs 2017  +0.682      2017 vs 2021  +0.320
    2012 vs 2013  +0.702      2013 vs 2017  +0.572      2010 vs 2021  +0.214
    2010 vs 2013  +0.545                                2012 vs 2021  +0.166

**Median +0.559, and the pair the old conclusion rested on is the WORST of the ten.** Every
weak pair involves 2021 — the wave with 19 provinces and the smallest cells — so what was read
as an unstable geography is mostly one unstable WAVE. An independent survey agrees: CLDS 2016
ranks the provinces at +0.595 against this pool (`sources/cn_clds.md`).

**A single correlation is a sample of size one.** §14.10's fifth condition was applied to one
pair and read as a property of the data; it was a property of the pair. Where a condition is
computed from a pairwise statistic, count how many pairs there are before believing it.

Some real instability remains: reported Protestantism fell 2.31% -> 1.03% across the period,
and if enforcement varied by province the ordering SHOULD move. §14.12's disclosure rule still
applies and `note_public` still names this as the layer to trust least.

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

SOURCE_ID = "cn_cgss_pooled"

# Xizang is READ and then DROPPED, and the reason is a wrong LABEL rather than a thin cell.
# CGSS 2010 is the only wave that samples Tibet, 79 respondents, 53 of whom answer 佛教. Its
# answer set has no separate 藏传佛教 row (CLDS's does), so those 53 are Tibetan Buddhists
# about to be written to `buddhism.mahayana` at 58.5% — in the one province where §14.5
# already draws Vajrayana from ethnicity. This module's own docstring justifies calling CGSS
# 佛教 "Han Mahayana practice" on the premise that "Xizang is not sampled at all"; 2010
# breaks that premise, so the premise is enforced here instead of assumed.
DROP_PROVINCES = {"Xizang": "CGSS cannot separate Vajrayana; §14.5 draws Tibet from ethnicity"}

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
    "folk": "chinesefolk",
}

# A category MAY be drawn from fewer waves than the pool. Nothing currently is, and the empty
# dict is the record of a decision that was taken and then reversed the same day.
#
# **民间信仰 does collapse in 2021 and only in 2021**: 2.90 -> 3.43 -> 1.91 -> 2.11 -> 0.27
# per cent, while Buddhism moves 4.66 -> 3.76 and everything else drifts. 2021 is the
# SINGLE-CHOICE wave, and it was excluded for folk on the argument that a respondent who
# tends a Mazu shrine AND calls themselves Buddhist must pick one, so folk loses to Buddhism.
#
# **THAT ARGUMENT IS WRONG AND THE DATA SAYS SO.** It predicts Buddhism RISES in 2021 as it
# absorbs the folk answers. Buddhism falls, 4.66 -> 3.76, and `none` gains 3.1 points. The
# folk respondents went to NO RELIGION, not to Buddhism. What single choice actually does is
# make people who tend a shrine say they have no religion, because they do not consider it a
# 宗教 — so the multi-select waves measure a PERMISSIVE threshold and 2021 a STRICT one, and
# excluding 2021 was choosing the permissive one.
#
# **And it bought almost nothing**: 3.05% pooled over five waves against 3.31% over four,
# 37.2M dots against 40.4M. A per-category vintage inconsistency for 8% more dots. Anita's
# call to reverse it, 2026-09-08. *A wave that disagrees is evidence about the question, and
# dropping it is a claim about which asking is right — which needs a mechanism that survives
# being tested, not just one that sounds plausible.*
EXCLUDE_WAVES = {}

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
    # 2010 and 2013 added 2026-09-08 from the open replication mirror; 2011 is deliberately
    # NOT here and is Anita's call, see cn_cgss.md. `weight=None` means the open copy of that
    # wave carries no weight column and the wave is pooled UNWEIGHTED, which is measured
    # rather than waved at: check() reports what it costs.
    dict(file="cgss2010.dta", year=2010, kind="single",
         prov="s41", rel="a5", weight="WEIGHT"),
    dict(file="cgss2012_14.dta", year=2012, kind="multi",
         prov="s41", none="a501", weight="weight"),
    dict(file="cgss2013.dta", year=2013, kind="multi",
         prov="s41", none="a501", weight=None),
    dict(file="cgss2017.dta", year=2017, kind="multi",
         prov="s41", none="a51", weight="weight"),
    dict(file="CGSS2021.dta", year=2021, kind="single",
         prov="provinces", rel="A5", weight="weight_raking"),
]


def _single(label):
    """One answer string to a category name, across two different single-choice sets.

    2021 writes the answers bare (`佛教`). **2010 prefixes every religious answer with
    `信仰宗教-`** (`信仰宗教-佛教`) while leaving `不信仰宗教` bare, so a straight lookup in
    SINGLE silently matches nothing but the no-religion row and the wave loads as 100%
    irreligious. Strip the prefix, then look up.
    """
    s = str(label).strip()
    if s in SINGLE:
        return SINGLE[s]
    for sep in ("-", "－", "—", "–"):
        if sep in s:
            tail = s.split(sep, 1)[1].strip()
            if tail in SINGLE:
                return SINGLE[tail]
    return None


def load():
    """Every pooled wave, harmonised to one row per respondent with binary category flags."""
    frames = []
    for w in WAVES:
        path = os.path.join(RAW, w["file"])
        if not os.path.exists(path):
            raise SystemExit(f"missing {path} — see sources/cn_cgss.md for the fetch")
        wcol = w["weight"]

        if w["kind"] == "single":
            s = pd.read_stata(path, columns=[w["prov"], w["rel"]],
                              convert_categoricals=True)
            d = pd.DataFrame({
                "prov": s[w["prov"]].astype(str).values,
                "wave": w["year"],
            })
            lab = s[w["rel"]].map(_single)
            for c in CATS:
                d[c] = (lab == c).astype(int).values
        else:
            num = pd.read_stata(path, columns=[w["none"]] + list(MULTI),
                                convert_categoricals=False)
            # the province column has to be read separately: converting categoricals for the
            # whole frame turns the religion flags into "是"/"否" and silently zeroes every
            # count.
            cat = pd.read_stata(path, columns=[w["prov"]], convert_categoricals=True)
            d = pd.DataFrame({
                "prov": cat[w["prov"]].astype(str).values,
                "wave": w["year"],
            })
            for col, name in MULTI.items():
                d[name] = (pd.to_numeric(num[col], errors="coerce") == 1).astype(int).values
            d["none"] = (pd.to_numeric(num[w["none"]], errors="coerce") == 1).astype(int).values

        if wcol:
            wv = pd.read_stata(path, columns=[wcol], convert_categoricals=False)
            d["w"] = pd.to_numeric(wv[wcol], errors="coerce").fillna(1.0).values
        else:
            d["w"] = 1.0
        d["weighted"] = bool(wcol)
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
    for province, gall in df.groupby("province"):
        if province in DROP_PROVINCES:
            continue
        for cat, node in DRAWN.items():
            drop = EXCLUDE_WAVES.get(cat, set())
            g = gall[~gall["wave"].isin(drop)] if drop else gall
            if not len(g):
                continue
            W = g["w"].sum()
            years = "+".join(str(y) for y in sorted(g["wave"].unique()))
            unw = sorted({int(y) for y in g.loc[~g["weighted"], "wave"].unique()})
            rows.append({
                "province": province,
                "node": node,
                "share": (g[cat] * g["w"]).sum() / W if W else np.nan,
                "n": int(len(g)),
                "k": int(g[cat].sum()),
                "waves": int(g["wave"].nunique()),
                "basis": "self_id",
                "source_id": SOURCE_ID,
                "note": f"CGSS {years}; {int(g[cat].sum())} of {len(g)} respondents; "
                        f"{int(g['wave'].nunique())} waves"
                        + (f"; {','.join(str(u) for u in unw)} unweighted" if unw else ""),
            })
    out = pd.DataFrame(rows).sort_values(["node", "share"], ascending=[True, False])
    return out.reset_index(drop=True)


def check(df, out):
    """The three things that would silently break this file."""
    ok = True

    # 1. the multi-select collapse (§3.1a) — overlap must stay negligible.
    # Measured PER WAVE and on the multi-select waves only. Testing `wave != 2021` was right
    # when 2021 was the only single-choice wave; 2010 is single-choice too, and pooling it in
    # would dilute the overlap toward zero and hide a wave that had drifted.
    rel = [c for c in CATS if c != "none"]
    multi_years = [w["year"] for w in WAVES if w["kind"] == "multi"]
    for year in multi_years:
        mw = df[df["wave"] == year]
        k = mw[rel].sum(axis=1)
        overlap = (k >= 2).sum() / max(1, (k >= 1).sum())
        flag = "  !! over 5%, the collapse is no longer safe" if overlap > 0.05 else ""
        print(f"  multi-select overlap {year}: {overlap*100:4.1f}% of religious "
              f"respondents name >1 religion{flag}")
        if overlap > 0.05:
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
