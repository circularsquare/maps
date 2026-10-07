# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _tz_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read.

    30 units over 947,000 km2 is about 31,600 km2 a unit, coarser than Nigeria's states. Tabora
    is 75,000 km2 and Dar es Salaam 1,600, and most regions hold their people in a few belts:
    the lake shore, the Kilimanjaro and Meru slopes, the southern highlands, the coast
    (sources/tz_grid.py).
    """
    return _kontur_place_weight(place, "tz_hexes.gpkg", "sources/tz_grid.py")


def _tz_counts():
    """Five pooled Afrobarometer rounds on the 2022 census's region counts, the mainland's
    Christians split into five churches with the GFS 2023: 10 nodes, 30 units, and EVERY ROW IS
    `modelled` IN §7.

    TANZANIA HAS NOT ASKED RELIGION SINCE THE 1967 CENSUS, so as in Nigeria there is no margin
    to fit and no measured tier. Each unit is drawn at the mix its own respondents gave
    (Christian, Muslim and None each pass the split-half) and at its census population;
    Traditional and Other are flat at the national share under spec §12's small-category rule.
    sources/tz.py has the construction, why round 5 is left out and how round 8 is decoded.

    MBEYA AND SONGWE ARE ONE UNIT (`TZ12`), because the survey filed Songwe's districts under
    Mbeya until 2021. ZANZIBAR'S FIVE UNITS SHARE ONE CHRISTIAN SHARE (0.93%), pooled across
    their 1,072 respondents on Anita's ruling of 2026-09-14 (sources/tz.md §8); region by region
    three had none and one rested on a single respondent.
    """
    from tz2022 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "tz.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    lut = pd.read_csv(HERE / "data" / "geo" / "tz" / "tz_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    missing = sorted(df.loc[df["unit"].isna(), "geo_id"].unique())
    if missing:
        raise SystemExit(f"tz.csv units with no polygon: {missing} -- re-run "
                         "sources/tz_geo.py, the lookup is stale")
    if df["unit"].nunique() != 30:
        raise SystemExit(f"{df['unit'].nunique()} units, expected 30")

    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(df.loc[df["node"].isna(), "source_category"].unique())
    if unmapped:
        raise SystemExit(f"tz.csv categories with no node: {unmapped}")
    df = df[df["count"] > 0]
    # EVERY row, without exception -- there is no measured tier in this country (§7).
    df["tier"] = "modelled"
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations", "tier"]]


ENTRY = {
    "tz": dict(
        name="Tanzania",
        source="five pooled rounds of the Afrobarometer, 2008 to 2022, with the Global "
               "Flourishing Study 2023 for the churches, on the region counts of the 2022 "
               "Population and Housing Census (National Bureau of Statistics)",
        basis="self-identification, whole census population",
        note_public=(
            "**Tanzania has not counted religion since the 1967 census.** The question was "
            "left off the 1978 census without a public reason and has not been asked in any "
            "census since, including the one in 2022. The 1967 count found mainland Tanzania "
            "32% Christian, 30% Muslim and 37% following traditional religions. "
            "**So this map's Tanzania comes from a survey.** Five rounds of the Afrobarometer "
            "are pooled, **10,745** adults interviewed between June 2008 and November 2022, "
            "and nobody counted these dots, so they disappear when inferred dots are turned off. "
            "Each region is drawn at "
            "the mix its own respondents gave and at its 2022 census population, which puts "
            "the country at **64.1%** Christian, **29.5%** Muslim and 5.2% with no religion. "
            "The Global Flourishing Study, a separate survey of 9,075 Tanzanians in 2023, "
            "comes out at 64.6% Christian and 32.6% Muslim when its regions are weighted the "
            "same way, and ranks the regions in nearly the same order. "
            "**The line runs between the coast and the interior.** Zanzibar's five regions are "
            "98% Muslim, and Lindi, Mtwara, Pwani and Tanga on the mainland coast 75 to "
            "86%; Dar es Salaam is **58.8%** Muslim. Inland it turns over, and Njombe, Iringa "
            "and Rukwa are each more than 94% Christian. Split the five rounds into two halves "
            "and the regions rank the same way in both, at **+0.97** for Islam and +0.92 for "
            "Christianity. "
            "**Having no religion is regional too**: 24.0% of Shinyanga and 23.5% of "
            "Simiyu, and almost nobody on the coast. That is the answer these people gave, on "
            "a card that also offered traditional religion, which 18 people chose in five "
            "rounds. Traditional religion (0.20%) and other religions (0.93%) are drawn at the "
            "same share everywhere, because the survey is too thin to place them. "
            "**Mbeya and Songwe are drawn as one unit**, because the survey filed Songwe's "
            "districts under Mbeya until 2021. Zanzibar's Christians are drawn at one share "
            "across its five regions, **0.9%**, from the 13 Christians among 1,072 people "
            "interviewed there, because region by region three of the five would show none "
            "and one would rest on a single person. "
            "**The churches take a second survey.** The Afrobarometer names them, but between a "
            "tenth and a third of its Christians answer just Christian, depending on the round. "
            "The Global Flourishing Study asked which church people most identify with and "
            "almost all of them named one, so it sets each church's national share, and the two "
            "surveys together place them; for all five churches they rank the mainland regions "
            "alike. Catholics are **27.6%** of Tanzania, Pentecostals 13.9%, Lutherans 8.5%, "
            "Anglicans 6.3% and Adventists 3.9%. Catholics are four in five Christians in "
            "Ruvuma, Lutherans half of Arusha's and a third of Kilimanjaro's and Njombe's, "
            "Anglicans more than half of Dodoma's, and Adventists a fifth of Mara's and "
            "Simiyu's. The two surveys disagree most about Pentecostals, a fifth of Christians "
            "in the Flourishing Study and under a tenth in the Afrobarometer, which names a "
            "church less often; the map follows the first. Zanzibar's Christians are too few to "
            "split by church. Islam is one colour, because the share naming Sunni rather than "
            "just Muslim moves from round to round."),
        how="two pooled surveys, 2008 to 2023, on 2022 census region populations",
        grain="regions, 2.1 million people on average",
        counts=_tz_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "tz" / "tz_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_tz_place_weight,
        note="TANZANIA HAS NOT ASKED SINCE 1967: the question was removed from the 1978 "
             "questionnaire (C.K. Omari, Tanzanian Affairs, July 1983) and Tanzania is ABSENT "
             "from the UNSD oracle. sources/tz.md lists what was searched. Drawn on Anita's "
             "Nigeria ruling (ask/answered/010-ng): the regional picture from a pooled survey, "
             "every row modelled, the national level computed rather than fitted. "
             "ROUND 5 IS LEFT OUT and ROUND 4 IS PLACED BY DISTRICT: both use the 26 regions "
             "of before March 2012, and only round 4 has a district column. ROUND 8 HAS NO "
             "REGION LABELS for Tanzania in the merged file and is decoded by code, checked "
             "against rounds 7 and 9. Mbeya and Songwe are one unit. Christian, Muslim and "
             "None carry their own unit shares on the split-half; Traditional and Other are "
             "flat under spec §12's 2x rule. The GFS 2023 is a witness to the Christian and "
             "Muslim shares, and since 2026-10-03 sets the mainland churches' levels and, with "
             "Afrobarometer rounds 4, 6, 8 and 9, their unit shares (sources/tz.md §9). "
             "ZANZIBAR'S CHRISTIANS ARE ONE POOLED SHARE across its five regions (Anita's "
             "ruling 2026-09-14, sources/tz.md §8, sources.md §tz-2026-10-03).",
    ),
}
