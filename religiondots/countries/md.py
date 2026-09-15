# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _md_counts():
    """BNS RPL 2024 at the UAT, the finest unit it publishes religion for.

    ONE LEVEL, AND THE UNIT KEY IS THE COUNT'S OWN CODE. md.csv also carries the 35 raions
    and a country row; both are cross-check levels written by sources/md.py and neither is
    drawn, so this filters to `uat` and the 901 rows that leaves are the map.

    THE FIVE CHIŞINĂU SECTORS ARE UATs HERE AND ARE NOT ELSEWHERE. BNS publishes religion
    for Botanica, Buiucani, Centru, Ciocana and Rîşcani, and sources/md_geo.py draws them
    by clipping OpenStreetMap's sector boundaries to the city. Without that, oraşul
    Chişinău would be one polygon holding 567,038 people, 23.5% of the country, which
    would have been the worst capital case on the map after Romania's Bucharest. The same
    situation as Estonia's Tallinn linnaosad and Czechia's Prague: measured, not allocated.
    """
    from md2024 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "md.csv",
                     dtype={"geo_id": str}, low_memory=False)
    df = df[df["geo_level"] == "uat"].copy()
    df["unit"] = df["geo_id"]

    lut = pd.read_csv(HERE / "data" / "geo" / "md" / "md_uat_lookup.csv",
                      dtype={"geo_id": str, "kod": str})
    unknown = sorted(set(df["unit"]) - set(lut["kod"]))
    if unknown:
        raise SystemExit(f"{len(unknown)} md.csv UATs have no polygon: {unknown[:5]} -- "
                         "re-run sources/md_geo.py, the lookup is stale")

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations"]]


ENTRY = {
    "md": dict(
        name="Moldova",
        source="Recensământul Populaţiei şi al Locuinţelor 2024 (Biroul Naţional de "
               "Statistică)",
        basis="self-identification",
        note_public=(
            "Moldova is **94.3%** Orthodox, and the census does not divide that: the "
            "Metropolis of Chişinău under the Moscow Patriarchate and the Metropolis of "
            "Bessarabia under Bucharest are two churches on the same ground competing for "
            "the same parishes, and they are one answer here. Only **0.75%** of people "
            "left the question blank, which is low for a religion question anywhere. "
            "What the fine geography buys is the north. Briceni raion is 80.5% Orthodox "
            "where Şoldăneşti is 98.7%, and the difference is Jehovah's Witnesses at 7.7% "
            "of Briceni and Pentecostals at 4.2%; in the village of Caracuşenii Vechi the "
            "Witnesses are 822 of 2,629 people. The Old Believers who settled Bessarabia "
            "after the Nikonian reforms are **4,053** people, and nearly half of them are "
            "in two villages: Cunicea, in Florești, has 982, and Pocrovca, in Dondușeni, "
            "is 921 Old Believers out of 940 residents. Two categories the 2014 census "
            "had are gone from this one. It "
            "names no Jewish answer and no Lutheran one, so both sit inside `Alte "
            "religii`, which is **4,720** people in all, and no dot on this map is drawn "
            "for either."),
        how="census, 2024, direct question; unanswered by 0.75%",
        grain="towns and communes, 2,700 people on average",
        gap=("0.8% who did not answer the religion question; and the left bank of the Nistru "
             "and Bender, never enumerated"),
        gap_share=0.007514,
        counts=_md_counts,
        # UATs are the count layer and the placement layer: BNS publishes religion at no
        # finer unit, and at a median of 1,251 people each they are finer than Romania's
        # communes next door, so no population grid is read (spec §8.2).
        #
        # Chişinău is the exception and it is handled rather than accepted: the five city
        # sectors are drawn separately, because BNS counts them separately. See
        # _md_counts and sources/md_geo.py.
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "md" / "md_uat.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        note="Nothing is suppressed and nothing is rounded: the 901 UATs reconcile to the "
             "published national figure in every one of the fifteen category columns, and "
             "to each of the 35 raion rows (sources/md.md §3). The polygons are BNS's own commune "
             "layer, joined on the CUATM statistical code and confirmed by the population "
             "the layer carries, which matches the census total for all 896 of them.",
    ),
}
