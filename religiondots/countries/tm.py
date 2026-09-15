# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _tm_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read.

    Six units over 491,200 km2, most of it the Karakum. Balkan is 150,000 km2 with its people
    on the Caspian coast and along the railway; an equal share of dots per polygon puts that
    colour on sand (sources/tm_grid.py).
    """
    return _kontur_place_weight(place, "tm_hexes.gpkg", "sources/tm_grid.py")


def _tm_counts():
    """Central Asia Barometer, waves 4-6 (2018-2019), at velayat: 6 answers, 6 units, and
    EVERY ROW IS `modelled` IN §7.

    No Turkmen census since independence has published religion. The survey's frame and
    weights are 1995 figures and its sample is three times as Russian as the 2022 census, so
    the pooled regional shares are not drawn: Christian is each nationality's national answer
    share applied to the 2022 census's nationality counts by velayat, Muslim is the residual,
    and the four small answers share it at their national proportions. sources/tm.py has the
    construction and the tests, sources/tm.md the record.
    """
    from tm2019 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "tm.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    lut = pd.read_csv(HERE / "data" / "geo" / "tm" / "tm_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    missing = sorted(df.loc[df["unit"].isna(), "geo_id"].unique())
    if missing:
        raise SystemExit(f"tm.csv velayats with no polygon: {missing} -- re-run "
                         "sources/tm_geo.py, the lookup is stale")
    if df["unit"].nunique() != 6:
        raise SystemExit(f"{df['unit'].nunique()} velayats, expected 6")
    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(df.loc[df["node"].isna(), "source_category"].unique())
    if unmapped:
        raise SystemExit(f"tm.csv answers with no node: {unmapped}")
    df = df[df["count"] > 0]
    df["tier"] = "modelled"
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations", "tier"]]


ENTRY = {
    "tm": dict(
        name="Turkmenistan",
        source="Central Asia Barometer, waves 4 to 6, 2018 and 2019, against the 2022 census's "
               "population and nationality counts by velayat",
        basis="self-identification, adults 18 and over",
        view=[52.4, 35.1, 66.7, 42.8],
        note_public=(
            "No Turkmen census since independence has published a count of religion, so this "
            "map is drawn from a survey. The Central Asia Barometer interviewed **4,500** people "
            "face to face across all six velayats in 2018 and 2019. "
            "**The survey's sample was drawn on 1995 population figures.** Russians, "
            "Ukrainians and Armenians are 6.4% of it and 1.9% of the 2022 census, and almost "
            "every Christian interviewed was one of them, so the survey's own regional shares "
            "would draw far too many Christians. The map instead takes how each nationality "
            "answered and applies it to where the 2022 census counts that nationality. "
            "**Most of the difference is Ashgabat.** The survey reads it at 35% Christian and "
            "the map draws **7.3%**, against 1.9% nationally; 57% of the Christians on this "
            "map are there. "
            "The survey has one Christian answer and one Muslim answer, so Orthodox and "
            "Protestant Christians cannot be told apart, and Christians of Turkmen nationality "
            "are drawn at the same low rate in every velayat."),
        how="survey, three rounds of 1,500 interviews in 2018 and 2019, applied to census "
            "nationality counts",
        grain="velayats, 1.2 million people on average",
        counts=_tm_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "tm" / "tm_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_tm_place_weight,
        note="A SURVEY ON A CENSUS AND EVERY ROW IS `modelled` (§7b). The Central Asia "
             "Barometer's `Religion_M`, waves 4-6 (4,500 face-to-face interviews), gives each "
             "nationality's answer shares nationally; the 2022 census's tables 4.3-4.8 give "
             "each velayat's nationality counts and table 1.3 its population. Christian is "
             "composed from those, Muslim is the residual, the four small answers share it at "
             "their post-stratified national proportions. sources/tm.py the build, sources/tm.md "
             "the record, sources.md §9do the write-up. WHY NOT THE SURVEY'S REGIONAL SHARES: "
             "its frame and weights are 1995 figures, it is 6.4% Russian/Ukrainian/Armenian "
             "against the census's 1.87%, and it reads Ashgabat 35.3% Christian where the "
             "census counts 7.71% of those three nationalities. THE SPLIT-HALF HAS NO POWER "
             "(Muslim and Christian +0.600 against a null of +0.600 on six units); Anita "
             "approved the build knowing that, 2026-09-14. Witness on the drawn ordering: wave "
             "14 (2023, phone) rho +1.000, exact p 0.0014. THE DECODE: names, polygon "
             "positions, and the wave 4 methods report's PSU allocation per named region "
             "(six different counts, matched exactly). Boundaries are Kontur's OSM extract of "
             "2022-04-07; there is no COD-AB.",
    ),
}
