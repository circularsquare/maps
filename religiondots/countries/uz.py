# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _uz_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read.

    Karakalpakstan and Navoi are 278,000 km2 between them, mostly the Kyzylkum and the dry
    Aral seabed, with their people along the Amu Darya and in a few mining towns. An equal
    share of dots per polygon puts that colour on sand (sources/uz_grid.py).
    """
    return _kontur_place_weight(place, "uz_hexes.gpkg", "sources/uz_grid.py")


def _uz_counts():
    """Central Asia Barometer, waves 1-6 (2017-2019), religion WITHIN ETHNIC GROUP, applied to
    the 2026 census's ethnic groups by region: 7 answers, 14 units, and EVERY ROW IS
    `modelled` IN §7.

    No Uzbek census has ever asked about religion, the 2026 one included (sources/uz.md §1,
    against the printed questionnaire). The survey went to telephone at wave 7 and stopped
    asking Uzbeks the question, so 2017-2019 is the whole pool.

    DRAWN BY GROUP SINCE 2026-09-14. The 2026 census counts eight nationalities per region, and
    a region-share build put Tashkent city at 21.7% Christian from a sample 23% Russian against
    a census 9.29%. Three groups (Uzbek and other Central Asian, Russian, other); sources/uz.py
    has the tests and sources/uz.md §9 and sources.md §9dp the write-up.

    THE PUBLIC FILE DOES CARRY THE SAMPLING POINT (`SamPt`); the first build said it did not.
    The big group's split-half stays on waves; the two small groups are tested Tashkent city
    against the rest with sampling points shuffled.

    THE POPULATION IS THE 2026 CENSUS'S, 15 January 2026, 39,047,321, and its region rows run
    in the office order that SOATO 17NN -> UZNN and the barometer's codes already follow.
    """
    from uz2019 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "uz.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    lut = pd.read_csv(HERE / "data" / "geo" / "uz" / "uz_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    missing = sorted(df.loc[df["unit"].isna(), "geo_id"].unique())
    if missing:
        raise SystemExit(f"uz.csv regions with no polygon: {missing} -- re-run "
                         "sources/uz_geo.py, the lookup is stale")
    if df["unit"].nunique() != 14:
        raise SystemExit(f"{df['unit'].nunique()} regions, expected 14")
    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(df.loc[df["node"].isna(), "source_category"].unique())
    if unmapped:
        raise SystemExit(f"uz.csv answers with no node: {unmapped}")
    df = df[df["count"] > 0]
    df["tier"] = "modelled"
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations", "tier"]]


ENTRY = {
    "uz": dict(
        name="Uzbekistan",
        source="Central Asia Barometer, waves 1 to 6, 2017 to 2019, applied to the National "
               "Statistics Committee's 2026 census counts by ethnic group",
        basis="self-identification, adults 18 and over",
        view=[55.9, 37.1, 73.2, 45.6],
        note_public=(
            "**No Uzbek census has ever asked about religion, the 2026 one included, so this "
            "is a survey standing where a census would be.** The answers come from the Central "
            "Asia Barometer: **9,000 people** interviewed face to face across all fourteen "
            "regions in six rounds from 2017 to 2019. The dots are drawn desaturated to say "
            "that. "
            "**Religion is read by ethnic group.** Most of Uzbekistan's Christians are "
            "Russian, and the survey's Tashkent city sample was 23% Russian where the 2026 "
            "census counts **9.3%**. So the answers of each group (Uzbeks and the other Central "
            "Asian peoples, Russians, and everyone else) are applied to the number of people "
            "the census counted in that group in each region. Tashkent city comes out at about "
            "300,000 Christians, just over half of those on this map. "
            "The survey has one Christian answer and one Muslim answer, so Orthodox and "
            "Protestant Christians cannot be told apart, and neither can Sunni and Shia "
            "Muslims."),
        how="survey, six rounds of 1,500 interviews from 2017 to 2019, read by ethnic group",
        grain="regions, 2.8 million people on average",
        gap="0.75% of the survey, who did not know or would not say",
        gap_share=0.007517,
        counts=_uz_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "uz" / "uz_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_uz_place_weight,
        note="A SURVEY READ WITHIN ETHNIC GROUP, ON THE CENSUS'S ETHNIC GROUPS, EVERY ROW "
             "`modelled` (§7b). Redrawn 2026-09-14 on Anita's ruling after the review found "
             "Tashkent city drawn 21.7% Christian from a sample 23% Russian against a census "
             "9.29%; the precedent is the citizenship splits (be, se, no, dk) and Latvia's "
             "non-citizens. The Central Asia Barometer's `Religion_M`, pooled over the six "
             "face-to-face waves, gives religion within three groups keyed on `Ethnic_M` "
             "(central: Uzbek, Karakalpak, Kazakh, Tajik, Kyrgyz; russian; other: Tatar and "
             "Other (vol.)). The 2026 census compilation's ethnic composition by region "
             "(stat.uz, printed twice, both parsed and cross-checked) gives each group's people "
             "per region, Turkmens counted in central. CENTRAL: Sweden's split-half on waves "
             "plus the spatial chi-square at 14 regions places Christian (p=0.0425 on 17 "
             "respondents, the likeliest false pass), non-believer and no particular faith; "
             "Muslim is the residual; the five DHS 1996 regions place nothing extra. RUSSIAN "
             "AND OTHER cannot take a split-half (most wave x region cells are empty) and are "
             "tested Tashkent city against the rest, shuffling sampling points (`SamPt`, which "
             "IS in the public file) within wave, plus the chi-square: Russian Muslim is placed "
             "(city 1.3%, rest 26.4%) with Christian the residual; other's Christian, "
             "non-believer and no particular faith are placed with Muslim the residual. "
             "`Other (vol.)` stays refused in OVERRIDE (one interviewer, sources/uz.md §8) and "
             "fails anyway. QUOTA TEST: all 15 wave pairs, worst 2 vs 6, p=0.848. JOIN: the "
             "census's region rows run in the office order the barometer's codes 4001-4014 and "
             "uz_lookup.csv follow; census over SIAT is 0.97 to 1.07 per region except "
             "Tashkent region at 1.19. sources/uz.py the build, sources/uz.md §9 and "
             "sources.md §9dp the write-up.",
    ),
}
