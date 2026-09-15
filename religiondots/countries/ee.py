# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _ee_counts():
    """Statistics Estonia 2021 at the finest unit it publishes: 21 categories, branch level.

    TWO LEVELS, AND THEY ARE ALTERNATIVES. The 8 Tallinn linnaosad REPLACE Tallinn; they do
    not nest under it for drawing. Tallinn as one municipality is 33.07% of the 15+
    population in a single 159 km² polygon, which would have been the worst capital case on
    the map by a factor of three. Statistics Estonia publishes RL21452 for the districts as
    well, so this is measured, not allocated — the same situation as Czechia.

    THE KEY IS A SLICE of the 14-character PxWeb place code, which concatenates EHAK codes:
    a municipality is `code[4:8]` and a city district is `code[8:12]`. sources/ee_geo.py
    checks the two namespaces do not collide, and re-keys four polygons whose EHAK code
    changed between the census and the 2024 boundary release.

    EVERYTHING IS ROUNDED TO BASE 10 (spec §3.8), so nothing reconciles exactly and is not
    meant to. The universe is persons aged 15 and over — no Estonian child is drawn.
    """
    from ee2021 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "ee.csv",
                     dtype={"geo_id": str}, low_memory=False)
    df = df[df["geo_level"].isin(("municipality", "city_district"))].copy()
    df["unit"] = [c[4:8] if lv == "municipality" else c[8:12]
                  for c, lv in zip(df["geo_id"], df["geo_level"])]

    replaced = set(pd.read_csv(HERE / "data" / "geo" / "ee" / "ee_replaced.csv",
                               dtype=str)["kod"])
    df = df[~((df["geo_level"] == "municipality") & (df["unit"].isin(replaced)))]

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations"]]


ENTRY = {
    "ee": dict(
        name="Estonia",
        source="Rahvaloendus 2021 (Statistics Estonia)",
        basis="self-identification, voluntary question, persons aged 15+",
        view=[21.5, 57.4, 28.3, 59.8],
        gap=("13.3% of those aged 15 and over: refusals, and a smaller cell the census could "
             "not establish at all"),
        gap_share=0.1331,
        note_public=(
            "The least religious country on this map: 58% of Estonians aged 15 and over "
            "say they feel no affiliation to any religion, and a further 11% declined the "
            "question. Only children are missing for a different reason — the question is "
            "asked from age 15, so no Estonian child is drawn at all. Among those who do "
            "report a religion, Orthodoxy is larger than Lutheranism, which is the "
            "opposite of the country's history and follows the Russian-speaking "
            "population of Ida-Viru and Tallinn. Two things here are enumerated nowhere "
            "else on earth: Maausk and Taarausk, the Estonian native faith, counted as "
            "themselves; and the Old Believers of Lake Peipus."),
        how="census, 2021, voluntary, ages 15 and over",
        grain="municipalities, 11,000 people on average",
        counts=_ee_counts,
        # Tallinn is replaced by its 8 linnaosad, which Statistics Estonia publishes
        # religion for. Without that one polygon would be a third of the country.
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "ee" / "ee_finest.gpkg",
        place_unit=lambda g: g["kod"].astype(str),
        note="Counts are rounded to base 10 (spec §3.8) and the universe is persons aged "
             "15+, not the whole population (sources/ee.md §2).",
    ),
}
