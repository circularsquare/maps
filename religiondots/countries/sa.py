# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


# Nodes allowed inside Mecca's haram. Alevis are Turkish Muslims by Pew's count and travel on
# Turkish passports; Druze and Yazidis are not treated as Muslims by the Saudi state.
_SA_HARAM_ALLOWED = ("islam", "alevism")


class _SaHaramWeighter(_KonturHexWeighter):
    """Kontur hex population, except that non-Muslims are kept out of Mecca's haram.

    Non-Muslims may not enter Mecca's sacred boundary (US State Department, 2023 religious freedom
    report; the boundary is OSM relation 19590020, sources/sa_grid.py --haram). Every region's
    foreigners take one national mix, so without this Makkah region's non-Muslims fell in Mecca at
    the region's rate (review cb8b206e-rev9, sources/sa.md §8). A hex is inside when any of it is.

    In a unit that touches the haram, a non-Muslim row weighs 0 inside it and hex population
    outside. A Muslim row weighs hex population inside and `pop * (1 - s * P / P_out)` outside,
    where `s` is the unit's non-Muslim share of its counts, `P` its Kontur people and `P_out` those
    outside the haram. Every hex then still draws dots in proportion to its people, the haram's
    all Muslim; only placement inside the region moves, never a count (§4.1).
    """

    def __init__(self, place, built_by, inside, nonmuslim_share):
        super().__init__(place, built_by)
        self.inside = inside
        self.unit = place["unit"].astype(str).to_numpy()
        self.share = nonmuslim_share
        self.n_kept_out = 0
        self.n_raised = 0

    def weights(self, node, idx, count, plain=False):
        pop = self.pop[idx]
        m = self.inside[idx]
        if not m.any() or pop[m].sum() <= 0:
            return super().weights(node, idx, count, plain)
        w = pop.copy()
        if node.startswith(_SA_HARAM_ALLOWED):
            s = self.share[self.unit[idx[0]]]
            p_all, p_out = pop.sum(), pop[~m].sum()
            f = 1.0 - s * p_all / p_out
            if not 0.0 < f <= 1.0:
                raise SystemExit(f"sa haram: Muslim weight factor {f:.3f} outside (0, 1]")
            w[~m] *= f
            self.n_raised += 1
        else:
            w[m] = 0.0
            self.n_kept_out += 1
        self.n_pop += 1
        return w

    def summary(self):
        return (super().summary() + f"; Mecca's haram: {self.n_kept_out:,} non-Muslim rows kept "
                f"out, {self.n_raised:,} Muslim rows filling it (countries/sa.py)")


def _sa_place_weight(place):
    """countries.py hook. `place` is the 400 m hex layer scatter.py has read.

    Saudi Arabia is 1.9 million km2 and five cities hold half its people; Kontur decides where
    people are inside each region, uncalibrated, since nothing finer than the region is published
    with Saudis and non-Saudis (sources/sa_grid.py). Non-Muslims are kept out of Mecca's haram
    (`_SaHaramWeighter`).
    """
    if "pop" not in place.columns:
        return _kontur_place_weight(place, "sa_hexes.gpkg", "sources/sa_grid.py")
    import geopandas as gpd

    path = HERE / "data" / "geo" / "sa" / "sa_haram.gpkg"
    if not path.exists():
        raise SystemExit(f"missing {path}; run python sources/sa_grid.py --haram")
    haram = gpd.read_file(path).to_crs(place.crs).geometry.iloc[0]
    # Any overlap counts: a dot is sampled anywhere in its hex, so a hex astride the boundary
    # with its centre outside put 2 non-Muslim dots inside it on the first run.
    inside = place.geometry.intersects(haram).to_numpy()
    df = _sa_counts()
    allowed = df["node"].str.startswith(_SA_HARAM_ALLOWED)
    tot = df.groupby("unit")["count"].sum()
    share = (df[~allowed].groupby("unit")["count"].sum().reindex(tot.index, fill_value=0)
             / tot).to_dict()
    units = sorted(set(place.loc[inside, "unit"].astype(str)))
    if units != ["SA02"]:
        raise SystemExit(f"sa haram hexes fall in {units}, expected only SA02 (Makkah)")
    print(f"  Mecca's haram: {int(inside.sum()):,} hexes, {place.loc[inside, 'pop'].sum():,.0f} "
          f"Kontur people; Makkah non-Muslim share {share['SA02']:.2%}")
    return _SaHaramWeighter(place, "sources/sa_grid.py", inside, share)


def _sa_counts():
    """Census 2022 citizens on Islam, non-Saudis by nationality and sex: 13 regions.

    EVERY ROW IS `modelled` (§7b). No source asks religion in Saudi Arabia, so every citizen is
    drawn on Islam; each region's non-Saudi men and women take the census's national nationality mix
    for their sex through Pew 2020, with Burma's nationals (the Rohingya) on Islam and India's Hindu
    share set by Pew's Saudi estimate. Both halves are the same census's counts per region, so they
    partition each unit. sources/sa.py and sources/sa.md.
    """
    from sa2022 import resolve

    nat = pd.read_csv(HERE / "data" / "normalized" / "sa.csv", dtype={"geo_id": str},
                      keep_default_na=False, na_values=[""])
    nat["node"] = nat["source_category"].map(resolve)
    unmapped = sorted(nat.loc[nat["node"].isna(), "source_category"].unique())
    if unmapped:
        raise SystemExit(f"sa.csv categories with no node: {unmapped}")
    ext = pd.read_csv(HERE / "data" / "normalized" / "sa_foreign.csv", dtype={"geo_id": str})
    df = pd.concat([nat[["geo_id", "node", "count"]], ext[["geo_id", "node", "count"]]],
                   ignore_index=True)
    lut = pd.read_csv(HERE / "data" / "geo" / "sa" / "sa_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    missing = sorted(df.loc[df["unit"].isna(), "geo_id"].unique())
    if missing:
        raise SystemExit(f"sa rows with no unit: {missing}; re-run sources/sa_geo.py")
    if df["unit"].nunique() != 13:
        raise SystemExit(f"{df['unit'].nunique()} regions, expected 13")
    df = df[df["count"] > 0]
    df["tier"] = "modelled"
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations", "tier"]]


ENTRY = {
    "sa": dict(
        name="Saudi Arabia",
        source="No census or survey asks; GASTAT's 2022 census count of Saudis and non-Saudis in each "
               "region and of non-Saudis by nationality and sex (as mirrored by the Gulf Labour "
               "Markets, Migration and Population programme), through Pew Research Center's 2020 "
               "estimates",
        basis="citizens drawn as Muslim, which nobody asked; foreign residents by nationality and sex",
        note_public=(
            "**Nobody in Saudi Arabia is asked their religion.** The 2022 census has no question on "
            "it, and the Arab Barometer's Saudi interviews leave it blank. Saudi citizens are Muslim "
            "by law, so the **18,792,262** citizens the census counted are all drawn as Muslim. The "
            "US State Department puts citizens at 85 to 90% Sunni and 10 to 12% Shia, most of the "
            "Shia in the Eastern Province, with Ismailis in Najran; the map does not split them, "
            "because Shia mosques in Qatif, Dammam and Najran were bombed in 2015. Nobody counted "
            "these dots, so they disappear when inferred dots are turned off. "
            "**Every non-Muslim drawn is a foreign resident.** The census counted **13,382,962** "
            "non-Saudis, 41.6% of the people living in the country, in every region, but their "
            "nationality only for the whole country, by sex. Men and women come from different "
            "places: there are 1,181 Bangladeshi men for every 100 Bangladeshi women, and 61 "
            "Filipino men for every 100 Filipino women. So each region's foreign men are drawn at "
            "the national mix of foreign men and its foreign women at the mix of foreign women, "
            "using the census's count of foreign men per 100 foreign women in each region, from "
            "264 in Makkah to 510 in Asir. Each nationality is drawn at Pew Research Center's 2020 "
            "estimate for its home country, which cannot see anyone who converted or stopped "
            "practising. "
            "**Two nationalities are not drawn at their home country's figure.** The 163,717 people "
            "from Myanmar are Rohingya, who are Muslim. Indians in the Gulf are mostly Muslim, which "
            "Pew says of its own migration estimates, so Indians are drawn at the Hindu share that "
            "gives Pew's figure for Hindus in Saudi Arabia (2.6% of everyone): 19.7% of Indians, "
            "not India's 79%. "
            "That puts **2,461,948** people on religions other than Islam: 1,337,918 Christians, "
            "843,672 Hindus, 120,949 Buddhists and 60,679 with no religion. Pew's estimate for "
            "everyone living in Saudi Arabia is 92.7% Muslim and 4.4% Christian; this map draws "
            "92.3% and 4.2%. Its Buddhists are about five times Pew's figure."),
        how="no source asks; citizens drawn as Muslim, foreign residents by nationality and sex",
        grain="regions, 2,475,000 people on average",
        gap="Saudi citizens who are not Muslim, whom no source counts; and foreign residents the 2022 "
            "census missed, whom nobody has counted",
        counts=_sa_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "sa" / "sa_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_sa_place_weight,
        note="BUILT ON ANITA'S MAGHREB AND MAURITANIA RULINGS (ask/RULINGS.md 2026-09-15 and 2026-09-16: "
             "a near-uniformly Muslim country on a compiler's figure, foreigners by region). "
             "sources/sa.md is the record. CITIZENS: nothing asks (census 2022 no item; AB II Q1012 "
             "empty, AB V no rows; ministry counts mosques); all 18,792,262 on islam; no Sunni/Shia "
             "split (spec §14, ask 040). NON-SAUDIS: 13,382,962 per region (GLMM's copy of the census "
             "table, checked against the census report's Figure 11), split by sex with Figure 12's "
             "ratios raked to the national sexes (Riyadh within 0.013% of RCRC's count); national "
             "nationality by sex from GLMM's four tables (continent shares reproduce the report's "
             "prose), each sex's mix applied in every region; Pew 2020 per nationality, Muslim "
             "branches folded to islam; Burma on islam (Rohingya); India's Hindu share 19.69% so the "
             "layer's Hindus equal Pew's Saudi 2.622%; remainder (42,140, the Americas) on Pew's "
             "North America and Latin America rows. WITNESS: Christians 0.95 of Pew's Saudi share; "
             "Buddhists 5.1x and unaffiliated 1.7x, not corrected. GEOGRAPHY: COD-AB ADM1 13 regions. "
             "PLACEMENT: Kontur SA uncalibrated (1.148x the census; 0.89-1.29 per region); no block at "
             "the cap; non-Muslims kept out of Mecca's haram (OSM relation 19590020), Muslims filling "
             "it, sources/sa.md §9.",
    ),
}
