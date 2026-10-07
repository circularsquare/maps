# China. 2020 census nationality (minzu) by county, from chinaethnicity (sources/cn_ethnic.py),
# each nationality read as its language for the share that speaks it (taxonomy/cn2020.py), the
# rest on Chinese. Placed on religiondots' 3 km grid re-keyed to chinaethnicity's counties
# (sources/cn_geo.py). In chinaethnicity's 15 estimated provinces each county is put on its 2020
# census total (sources/cn_totals.py, inside cn_ethnic.py). Everyone on Chinese is then moved onto the county's main dialect group
# (sources/cn_dialect.py, cn2020.DIALECT); the share of each county registered in another province
# goes on the origin provinces' dialect mixes instead (sources/cn_migrants.py, _migrants below).
# In ten southern provinces CLDS 2016's Putonghua share then moves onto Mandarin (_putonghua).
# Dialect groups only MCPDict's located points show in a southern county get a small share, and
# every group's dots lean towards its points (sources/cn_mcpdict.py; its Placer is the weighter).
# Dialect groups only MCPDict's located points show in a southern county get a small share, and every
# group's dots lean towards its points (sources/cn_mcpdict.py, its Placer is the weighter).
# The record is sources/cn.md.
from _shared import *  # noqa: F401,F403
import sys
import numpy as np


def _counts():
    import cn2020
    df = pd.read_csv(NORM / "cn.csv", dtype={"geo_id": str})
    unknown = sorted(set(df["source_category"]) - set(cn2020.NAMES))
    if unknown:
        raise SystemExit(f"cn.csv nationalities with no mapping: {unknown}")
    before = df["count"].sum()
    out = []
    for (cat, prov), g in df.groupby(["source_category", "prov"]):
        for node, share in cn2020.shares(cat, int(prov)):
            out.append(pd.DataFrame({"unit": g["geo_id"], "node": node,
                                     "count": g["count"] * share}))
    res = pd.concat(out, ignore_index=True)
    # Chinese -> the county's main dialect group (Language Atlas of China, sources/cn_dialect.py)
    # cn_dialect_dlac.csv is cn_dialect.csv with the counties the atlas's polygons split between two
    # groups split by people (sources/cn_dlac.py); cn_dialect_mcp.csv adds, in six southern
    # provinces, groups only MCPDict's dialect points show in a county (sources/cn_mcpdict.py)
    dia = pd.read_csv(NORM / "cn_dialect_mcp.csv", dtype={"unit": str})
    dia["dnode"] = [cn2020.dialect_node(u, g, s) for u, g, s in zip(dia["unit"], dia["group"], dia["subgroup"])]
    zh = res[res["node"] == cn2020.CHINESE]
    if set(zh["unit"]) - set(dia["unit"]):
        raise SystemExit(f"cn: counties with no dialect group: {sorted(set(zh['unit']) - set(dia['unit']))}")
    ztot = zh.groupby("unit")["count"].sum()
    zh = zh.drop(columns="node").merge(dia[["unit", "dnode", "share"]], on="unit")
    zh = pd.DataFrame({"unit": zh["unit"], "node": zh["dnode"], "count": zh["count"] * zh["share"]})
    zh = zh.groupby(["unit", "node"], as_index=False)["count"].sum()
    zh = _migrants(zh, ztot)
    zh = _putonghua(zh, ztot)
    res["tier"] = "derived"
    res = pd.concat([res[res["node"] != cn2020.CHINESE], zh], ignore_index=True)
    if abs(res["count"].sum() - before) > 1:
        raise SystemExit(f"cn: split lost people ({res['count'].sum():,.0f} of {before:,.0f})")
    return res.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()


MIGRANT_CAP = 0.9     # no county is near it (Bao'an 56%, the highest); a guard against a bad base


def _migrants(local, ztot):
    """Residents registered in another province (sources/cn_migrants.py) speak their origin
    province's Chinese, not the county's. local: (unit, node, count) on the county's own dialect
    groups. Each county keeps its Chinese total; the share f of it that the census finds registered
    in another province (people from other provinces / the county's population) is moved onto each
    origin province's own mix (its locals' dialect groups, summed over the province), weighted by
    where the county's migrants come from. Locals stay `derived`; the moved people are `modelled`."""
    mig = pd.read_csv(NORM / "cn_migrants.csv", dtype={"unit": str, "origin": str})
    f = (mig.groupby("unit")["count"].sum() / mig.groupby("unit")["base"].first()).clip(upper=MIGRANT_CAP)
    org = mig.groupby(["unit", "origin"])["count"].sum()
    org = (org / org.groupby(level=0).transform("sum")).rename("o").reset_index()
    prov = local.assign(p=local["unit"].str[:2]).groupby(["p", "node"])["count"].sum()
    mix = (prov / prov.groupby(level=0).transform("sum")).rename("m").reset_index()
    fu = local["unit"].map(f).fillna(0.0)
    out = [local.assign(count=local["count"] * (1 - fu), tier="derived")]
    m = org.merge(mix, left_on="origin", right_on="p")
    m["w"] = m["o"] * m["m"]
    m = m.groupby(["unit", "node"], as_index=False)["w"].sum()
    m["count"] = m["unit"].map(ztot).fillna(0.0) * m["unit"].map(f).fillna(0.0) * m["w"]
    out.append(m[["unit", "node", "count"]].assign(tier="modelled"))
    res = pd.concat(out, ignore_index=True)
    res = res[res["count"] > 0]
    got = res.groupby("unit")["count"].sum()
    bad = (got - ztot.reindex(got.index)).abs()
    if bad.max() > 1e-6 * max(1.0, ztot.max()) or set(ztot.index[ztot > 0]) - set(got.index):
        raise SystemExit(f"cn: migrant mixing changed a county's Chinese total ({bad.idxmax()})")
    return res


PUTONGHUA_PROVS = {"31", "32", "33", "34", "35", "36", "43", "44", "45", "46"}


def _putonghua(zh, ztot):
    """In the ten provinces with non-Mandarin speech (sources/cn_putonghua.py), the share of people
    who use Putonghua after work in CLDS 2016 is moved from each non-Mandarin dialect row onto
    Mandarin: the `settled` share (locals and people from elsewhere in the province) from the
    county's own groups (the `derived` rows), the `inter` share from people from other provinces'
    home groups (the `modelled` rows). The prefecture's share where CLDS sampled it, else the
    province's (Hainan: the pool of every sampled non-Mandarin city). Mandarin rows are untouched;
    the moved people are `modelled`. Each county's Chinese total is unchanged."""
    import cn2020
    mand = f"{cn2020.SI}.mandarin"
    sh = pd.read_csv(NORM / "cn_putonghua.csv", dtype={"code": str})
    look = {(r.level, r.code, r.group): r.p for r in sh.itertuples()}

    def p(unit, group):
        pref = unit[:2] + "0000" if unit[:2] == "31" else unit[:4] + "00"
        v = look.get(("city", pref, group))
        return v if v is not None else look[("province", str(int(unit[:2])), group)]

    zh = zh.copy()
    on = zh["unit"].str[:2].isin(PUTONGHUA_PROVS) & (zh["node"] != mand)
    grp = np.where(zh["tier"] == "modelled", "inter", "settled")
    share = np.array([p(u, g) if o else 0.0 for u, g, o in zip(zh["unit"], grp, on)])
    moved = zh["count"] * share
    zh["count"] = zh["count"] - moved
    add = zh.assign(node=mand, count=moved, tier="modelled")[moved > 0]
    res = pd.concat([zh, add], ignore_index=True)
    res = res.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()
    res = res[res["count"] > 0]
    got = res.groupby("unit")["count"].sum()
    bad = (got - ztot.reindex(got.index)).abs()
    if bad.max() > 1e-6 * max(1.0, ztot.max()):
        raise SystemExit(f"cn: the Putonghua step changed a county's Chinese total ({bad.idxmax()})")
    return res


def _placer(place):
    """sources/cn_mcpdict.py's Placer: in a county with several local dialect groups each group's
    dots lean towards its MCPDict points and its 1987 atlas area; everything else follows population.
    Placement only: the counts are counts()'s."""
    sys.path.insert(0, str(ROOT / "sources"))
    import cn_mcpdict
    return cn_mcpdict.Placer(place)


ENTRY = dict(
    name="China",
    source=("Seventh National Population Census 2020, provincial census yearbooks table 1-4 "
            "(population by nationality), as compiled for chinaethnicity; county totals in the "
            "15 provinces without a county table from Dong and Wang's 2020 county census panel; "
            "speaker shares from the "
            "National Language Resource Monitoring and Research Center, Minority Languages (2024); "
            "dialect groups from the Language Atlas of China (2012; 1987 map digitised by "
            "Lawrence Crissman); residents registered in another province from the 2020 census "
            "yearbooks' tables 1-3 and 7-3 (national, provincial, Shenzhen and Guangzhou) and "
            "city census bulletins; Putonghua use from the China Labor-force Dynamics Survey "
            "2016 (Center for Social Survey, Sun Yat-sen University); located dialect points "
            "from MCPDict (osfans/MCPDict, MIT licence); Guangxi and Guangdong dialect totals "
            "from their provincial gazetteers' dialect volumes (1998, 2004)"),
    how=("census, 2020, nationality; each nationality drawn as its own language for the share "
         "that speaks it, the rest and all Han as Chinese, on each county's main dialect group "
         "in the Language Atlas of China; residents registered in another province drawn on "
         "their home province's dialect groups; in ten southern provinces the share using "
         "Putonghua after work in a 2016 labour survey drawn as Mandarin; in six southern "
         "provinces a small share given to dialect groups that documented dialect points place "
         "inside a county, and each group's dots placed towards its points"),
    parts=[
        dict(covers="Minority languages",
             source="2020 census, nationality, times the share of each nationality that speaks "
                    "its language (National Language Resource Monitoring and Research Center)",
             people=71_817_105),
        dict(covers="Chinese, people from other provinces",
             source="2020 census, residents registered in another province (county, district or "
                    "prefecture tables and city bulletins) times the county's Chinese speakers, "
                    "drawn on their home provinces' dialect groups",
             people=112_012_287),
        dict(covers="Chinese, Putonghua in the non-Mandarin south",
             source="China Labor-force Dynamics Survey 2016, language used after work, by "
                    "prefecture and migrant status: the share using Putonghua moved onto Mandarin "
                    "in ten provinces (7.4 million of them from other provinces)",
             people=43_662_165),
        dict(covers="Chinese, everyone else",
             source="2020 census, Han and the rest of each nationality, drawn on the county's "
                    "main dialect group in the Language Atlas of China", rest=True),
    ],
    grain=("2,846 counties, 495,000 people on average; in 15 provinces each county's "
           "nationality mix is estimated from 2000"),
    gap=("2 million serving military, who are counted nationally only; speakers of a dialect "
         "group other than their county's main one, including people who moved from elsewhere "
         "in the same province"),
    view=[73.5, 18.0, 135.1, 53.6],
    counts=_counts,
    mappings=["cn2020"],
    place=ROOT / "data" / "geo" / "cn" / "cn_grid_3km.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=_placer,
    note_public=(
        "China's census asks nationality, not language, so this map reads each nationality "
        "as its language. Only the share of each nationality that speaks its language is drawn "
        "on it, using the National Language Resource Monitoring and Research Center's figures "
        "for each nationality; the rest are drawn as Chinese. Hui and Manchu speak Chinese, and "
        "only a few percent of Tujia, Gelao and She speak their languages. Those shares are "
        "national, so within a nationality every county gets the same split. The census does not "
        "ask which variety of Chinese people speak. Everyone drawn as Chinese is shown on the "
        "main dialect group of their county in the Language Atlas of China, so most counties "
        "appear as a single group; 125 counties that the atlas divides between two groups are "
        "split by where people live on each side of its line. In Guangdong, Guangxi, Fujian, "
        "Jiangxi, Hunan and Zhejiang, where a county holds documented village or township points "
        "of another group in the MCPDict dialect database, that group is given up to a quarter "
        "of the county's Chinese speakers (less for isolated pockets), and inside every county "
        "each group's dots are placed nearer its points. These shares are estimates: the points "
        "say where a variety is spoken, not by how many. Guangxi's and Guangdong's totals are "
        "kept moving only towards the dialect figures in their provincial gazetteers. "
        "This shows where Mandarin, Wu, "
        "Min, Yue, Hakka and the other groups are spoken, but not how many speak them. In ten "
        "provinces where the local speech is not Mandarin (Guangdong, Guangxi, Hainan, Fujian, "
        "Zhejiang, Shanghai, Jiangsu, Jiangxi, Hunan, Anhui), the share of people who said they "
        "mainly use Putonghua after work in the China Labor-force Dynamics Survey 2016 is drawn "
        "as Mandarin, by city where the survey sampled it and separately for locals and people "
        "from other provinces. People living outside their home province "
        "(about 125 million in the 2020 census, 47% of Shenzhen and 42% of Shanghai) are drawn "
        "on their home province's dialect groups in the same proportions as the people who "
        "stayed there; the census counts them by province of registration, and for most "
        "counties outside the big destinations only by province or prefecture. People who "
        "moved within their province are drawn on the local dialect. Where a nationality speaks several "
        "languages, as Yi, Miao, Tibetans and Dai do, the group is drawn without naming one. "
        "16 provinces are drawn from their own 2020 county tables by nationality. The other 15 "
        "did not publish one openly; there each county has its 2020 census population, from "
        "Dong and Wang's county census panel, and its nationality mix comes from the 2000 "
        "census, adjusted to match each province's 2020 nationality totals."),
)
