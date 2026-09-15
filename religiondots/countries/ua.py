# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _ua_counts():
    """Ukraine at COD-AB's 27 first-level units, from ESS rounds 2-6 (2005-2013).

    One part, everyone: ESS samples residents regardless of citizenship (0.47% are not citizens) and
    Ukraine has had no census since 2001 to build a foreign half on. Sevastopol, which ESS never names,
    takes Crimea's composition. The population is Ukrstat's present population at 1 January 2022, and
    at 1 January 2014 for Crimea and Sevastopol. The Orthodox answer is one node, not a jurisdiction
    (sources/ua.md §7, ask 016).
    """
    from ua2013 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "ua.csv", dtype={"geo_id": str},
                     keep_default_na=False, na_values=[""])
    df = df[df["geo_level"] == "oblast"].copy()
    if df["geo_id"].nunique() != 27:
        raise SystemExit(f"{df['geo_id'].nunique()} units in ua.csv, expected 27 -- re-run sources/ua.py")
    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(set(df.loc[df["node"].isna(), "source_category"]))
    if unmapped:
        raise SystemExit(f"ua.csv has unmapped source categories: {unmapped}")
    df = df.rename(columns={"geo_id": "unit"})
    df = df.groupby(["unit", "node"], as_index=False)["count"].sum()
    df["congregations"] = 0
    # A survey: nothing here is `measured`.
    df["tier"] = "modelled"
    return df[["unit", "node", "count", "congregations", "tier"]]


def _ua_place_weight(place):
    """countries.py hook. `place` is the 400m Kontur hex layer scatter.py has read."""
    return _kontur_place_weight(place, "ua_grid_400m.gpkg", "sources/ua_geo.py")


ENTRY = {
    "ua": dict(
        name="Ukraine",
        name_in="Ukraine",
        source="European Social Survey rounds 2 to 6 (2005 to 2013); population estimates, State "
               "Statistics Service of Ukraine",
        basis="self-identification, sample survey",
        note_public=(
            "**Ukraine's 2001 census did not ask about religion and there has been no census since, "
            "so this map is drawn from a survey.** In the European Social Survey's five Ukrainian "
            "rounds from 2005 to 2013, **58%** named an Orthodox church, 30% said they belong to no "
            "religion and 8% named the Catholic Church, almost all of them Greek Catholic. "
            "**Three western oblasts are majority Greek Catholic.** Ivano-Frankivsk is **72%** Greek "
            "Catholic, Lviv 63% and Ternopil 59%. "
            "**The map does not divide the Orthodox by church.** Until 2013 the survey asked about the "
            "Moscow and Kyiv patriarchates, and the Kyiv Patriarchate has since merged into the "
            "Orthodox Church of Ukraine. In the survey's 2023-24 round, 75% of Orthodox respondents "
            "named the Orthodox Church of Ukraine and 10% the Moscow Patriarchate. "
            "**What this cannot show.** Every oblast is drawn at the State Statistics Service's last "
            "estimate before the full-scale invasion, 1 January 2022, and Crimea and Sevastopol at 1 "
            "January 2014, with the answers people gave from 2005 to 2013. The map shows neither the "
            "war's displacement nor any change under occupation. In the oblasts the 2023-24 round "
            "could reach, 34% said they belong to no religion, against 28% on the map."),
        how="survey, 9,641 people, 2005 to 2013",
        grain="oblasts, 1.6 million people on average",
        gap="3.5% of Ukraine: people who were asked about religion and declined or named no "
            "denomination",
        gap_share=0.034762,
        counts=_ua_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "ua" / "ua_grid_400m.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_ua_place_weight,
        note="THE OFFICE COUNTS ORGANISATIONS, NOT PEOPLE: DESS Form 1 (registered communities by "
             "oblast) is a witness, not drawn. ESS rounds 2-6 at 26 survey units by label (`regionua` "
             "in 2-4, `region` in 5-6); **ROUND 11 IS NOT POOLED** (it cannot sample Crimea, Donetsk or "
             "Luhansk, places the displaced where they live now, and has a different card) and is not "
             "used to rescale although No religion sits 6.3 points higher on its 23 oblasts (ask 016). "
             "Split-half on the 19 oblasts sampled in all five rounds, then the 8 groups of ESS's own "
             "region codes. **THE ORTHODOX ANSWER IS ONE NODE** (ask 016). Catholic split Eastern/Latin "
             "on rounds 4-6 and 11. sources/ua.md, sources.md §9dw.",
    ),
}
