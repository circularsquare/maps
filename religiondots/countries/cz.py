# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _cz_counts():
    """ČSÚ 2021 at the finest available unit: 78 categories, mapped at branch level.

    The only country so far that needs no allocate.py step — it publishes its finest
    categories at its finest geography, so nothing here is derived and every row may ring
    (spec §3.9/§3.10, sources.md §9b).
    """
    from cz2021 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "cz.csv",
                     dtype={"geo_id": str}, low_memory=False)

    # ONE level per place, and the two levels are alternatives. The file carries the whole
    # territorial hierarchy — obec, city district, ORP, okres, kraj, NUTS2, country — so
    # reading it as delivered counts the country eight times over.
    #
    # City districts subdivide 8 statutory cities and REPLACE them: Prague is 1,301,432
    # people in one obec, 12.4% of the country in a single polygon, and 57 city districts
    # instead. sources/cz_geo.py derives which 8 obce are replaced (spatially, from the
    # polygons) and writes the list; the matching polygon layer is cz_finest.gpkg. Both
    # levels are published by ČSÚ, so this is measured data and not an allocation.
    replaced = set(pd.read_csv(HERE / "data" / "geo" / "cz" / "cz_replaced.csv",
                               dtype=str)["kod"])
    df = df[((df["geo_level"] == "municipality") & ~df["geo_id"].isin(replaced))
            | (df["geo_level"] == "city_district")]

    # DROP THE EXPLICIT ZEROS, and they are most of the file. Because ČSÚ publishes a
    # complete partition it emits a row for every category in every municipality whether
    # anyone is there or not: 417,083 of 494,066 municipal rows are zeros, 84% of the file.
    # ASARB and the other sources list only what they found, so nothing before Czechia had
    # to think about this.
    #
    # Keeping them costs nothing in dots — zero people is zero dots — but a ring asserts
    # PRESENCE (spec §4.3), and every zero would become a ring claiming a body is in a
    # village it is not in. Left in, Czechia drew 277,987 rings against 7,346 dots, which
    # is a near-solid mask of false claims.
    df = df[df["count"] > 0]

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna()]
    df["congregations"] = 0
    return df.rename(columns={"geo_id": "unit"})[
        ["unit", "node", "count", "congregations"]]


ENTRY = {
    "cz": dict(
        name="Czechia",
        source="Sčítání 2021 (Czech Statistical Office)",
        basis="self-identification, voluntary question",
        gap="30.1% of the country, who did not answer a voluntary question",
        gap_share=0.3005,
        note_public=(
            "The religion question was voluntary and 30% of the country did not answer. "
            "Those people are not drawn at all, so this map shows 7.4 million of 10.5 "
            "million — and the share who answered runs from 11% to 81% between "
            "municipalities, so it is not an even haircut. What is drawn is unusually "
            "good: 78 categories published at municipality level with no rounding and no "
            "suppression, down to bodies with a single adherent. Jedi is the thirteenth "
            "largest answer, ahead of Jehovah's Witnesses, and is drawn as what it is."),
        how="census, 2021, voluntary question",
        grain="municipalities, 1,150 people on average",
        counts=_cz_counts,
        # Czechia is Ireland's case: the counts are already ON the finest unit, so there is
        # no separate placement layer and no allocation inside a unit. Czech obce have a
        # median population of 435 — finer than a US census tract (3,424) and about the
        # size of an Australian SA1 — so an equal share per polygon is a good population
        # weighting almost everywhere (spec §8.2), and in the 8 statutory cities the city
        # districts carry it the rest of the way.
        #
        # cz_finest.gpkg is built by sources/cz_geo.py: 6,250 obce + 142 city districts,
        # which is the finest complete cover of the country ČSÚ publishes religion for.
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "cz" / "cz_finest.gpkg",
        place_unit=lambda g: g["kod"].astype(str),
        note="ČSÚ is self_id on a voluntary question; the 30% who did not answer are "
             "excluded rather than drawn (spec §3.5).",
    ),
}
