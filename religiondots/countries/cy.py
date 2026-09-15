# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _cy_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read.

    396 communities over 5,846 km2 is 14.8 km2 a unit, so this is nowhere near Kenya's
    problem and the grid still earns its place. What it fixes is the Troodos and the Pafos
    hinterland, where a community is a mountain valley with everybody in one village at the
    bottom of it. Cyprus draws about 920 dots in total, so a single dot landing on a ridge
    instead of a village is a visible error rather than a rounding one (sources/cy_grid.py).

    It is a POPULATION weight and not a religion one, and here that is doubly true: nothing
    measures where a community's Muslims sit inside it, and nothing measures where Cyprus's
    Muslims are at all. See _cy_counts().
    """
    return _kontur_place_weight(place, "cy_hexes.gpkg", "sources/cy_grid.py")


def _cy_counts():
    """CYSTAT Census 2021 at municipality/community: 12 categories, 396 units, EVERY ROW
    `derived` -- and it is the GEOGRAPHY that is derived, not the category.

    THE COUNTS ARE EXACT AND THEY ARE NATIONAL. CYSTAT-DB tabulates religion by citizenship
    group, by country-of-birth group and by sex, and by nothing spatial; it tabulates
    LANGUAGE by district three separate ways. The twelve national figures match UNSD
    Demographic Yearbook table 28 to the person on all twelve. What sources/cy.py infers is
    only where those people live, from

        count(religion, community) = SUM over the four citizenship groups of
                                     P(religion | group) x N(group, community)

    with P from table 1891632E and N from 1891213E -- same census, same office, same
    universe, same day, and no outside coefficient anywhere. The national totals come back to
    0.001 of a person. This is a stronger footing than the nationality derivations in _gr, _es,
    _fr and _it, which have to assume what a Romanian resident believes; Cyprus counted.

    WHAT IT CANNOT DO. It moves religion around the island only as far as the citizenship mix
    moves. Islam runs 0.9% to 7.8% across communities and Buddhism 0.3% to 3.6%, which is real
    geography; the Armenian church comes out at 0.21-0.23% everywhere and the Maronite church
    at 0.40-0.56%, because both sit inside the `Cypriots` group and that group has one
    profile. Those two are 6,511 people, about six dots, and the only way to give them a
    geography was to invent a coefficient off the language table. taxonomy/cy2021.py argues it.

    SO `inferred dots: not shown` EMPTIES CYPRUS, and that is the honest answer rather than a
    defect: no religion was counted at any Cypriot unit, so tools/check_rollup.py's case B
    applies and there is deliberately no COLUMNS dict. Same shape as China and as
    Switzerland's canton spread.
    """
    from cy2021 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "cy.csv",
                     dtype={"geo_id": str}, low_memory=False)
    if df["geo_id"].nunique() != 396:
        raise SystemExit(f"{df['geo_id'].nunique()} communities, expected 396 -- re-run "
                         "`python sources/cy.py`")

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df = df.rename(columns={"geo_id": "unit"})
    df["congregations"] = 0
    df["tier"] = "derived"
    return (df.groupby(["unit", "node"], as_index=False)
              .agg(count=("count", "sum"), congregations=("congregations", "max"),
                   tier=("tier", "first")))


ENTRY = {
    "cy": dict(
        name="Cyprus",
        source="Census of Population and Housing 2021 (CYSTAT)",
        basis="self-identification, voluntary question, whole enumerated population",
        # The whole island, not the drawn bbox. The census covers the government-controlled
        # area only, and cropping to the dots would tidy the north out of the picture
        # instead of showing it empty, which is the one thing the reader most needs to see.
        view=[32.15, 34.50, 34.70, 35.78],
        note_public=(
            "**Cyprus asks about religion and publishes the answer for the country as a "
            "whole, and for nothing smaller.** The 2021 census tabulates religion by "
            "citizenship, by country of birth and by sex; it tabulates language by district "
            "three separate ways, and religion by district not at all. So the twelve counts "
            "here are exact national figures, and where each of them is drawn is worked out "
            "from the citizenship mix of the 396 municipalities and communities. That does "
            "real work for the religions immigration brought. Across the 102 communities of "
            "a thousand people or more, Islam runs from 0.9% of the answers given up to "
            "**10.2%** at Pegeia on the Pafos coast and Buddhism from 0.2% to 4.8%; the "
            "smaller villages reach further both ways, but the widest of them hold single "
            "figures of people and should not be read as geography. It does "
            "nothing at all for the Armenian and Maronite churches, which come out flat "
            "across the island when both are concentrated in Lefkosia and Lemesos. "
            "**The question was optional, and 159,835 people, 17.3% of the country, did not "
            "answer it.** No other census on this map made religion optional, and the "
            "refusals are not spread evenly: 13.7% of Cypriot citizens skipped it against "
            "29.3% of non-EU citizens, so the communities with the most migrants are the "
            "ones the map knows least about. None of those people are drawn. "
            "**Orthodoxy was 94.8% of Cyprus in 2001 and is 74.5% now.** Almost none of "
            "that is Cypriots leaving the church. The population grew by a third in twenty "
            "years and the arrivals brought Roman Catholicism from Poland and the "
            "Philippines, Islam from Syria and Bangladesh, Buddhism from Vietnam and Sri "
            "Lanka, and Sikhism and Hinduism from India and Nepal. Every one of those is "
            "counted in its own row here, and the 2001 census, which is the only one Cyprus "
            "has ever published religion by district for, is too old to draw. "
            "**Two churches of a few thousand members are counted separately, which almost "
            "no census does.** The Armenian church at 2,025 and the Maronite church at "
            "4,486 are named because the 1960 constitution makes Armenians, Maronites and "
            "Latins religious groups electing their own representatives to the House. "
            "Against that, the census has no Jewish category at all, though the same census "
            "counts a Jewish community in its ethnic-group table and 885 Hebrew speakers in "
            "its language table. "
            "**The north is not here.** Cyprus's six districts are numbered 1 to 6 and the "
            "census's district list runs 1, 3, 4, 5, 6: Keryneia has no code, because none "
            "of its 47 communities was enumerated. 5,846 square kilometres are drawn of the "
            "island's 9,249. The 2011 census of northern Cyprus, the only one held there, "
            "publishes eleven tables and none of them asks about religion, so there is "
            "nothing to fill the blank with."),
        how="census, 2021, voluntary question; religion published nationally only",
        grain="municipalities and communities, 2,300 people on average",
        fill="from the national religion counts, split by each community's citizenship groups",
        gap="northern Cyprus, and the 17.3% who declined the question",
        gap_share=0.1731,
        counts=_cy_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "cy" / "cy_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_cy_place_weight,
        note="RELIGION IS NATIONAL AND LANGUAGE IS BY DISTRICT, WHICH IS THE WHOLE PROBLEM. "
             "CYSTAT-DB's 2021 census carries six tables under Language, Religion, Ethnic / "
             "Religious Group: three cut language by district and three cut religion by "
             "citizenship group, country-of-birth group and sex. The 2011 branch has no "
             "religion at all. So every community figure is CYSTAT's own arithmetic, "
             "P(religion | citizenship group) from table 1891632E applied to the four "
             "citizenship groups of each community from 1891213E, and it reconciles to the "
             "national counts within 0.001 of a person. The twelve national counts match "
             "UNSD table 28 to the person. "
             "THE JOIN IS AN INTEGER: PxWeb's community value codes ARE Cyprus's LAU codes, "
             "so the 396 units join GISCO LAU 2021 on the code, and all 396 Latin names then "
             "agree as a free check (sources/cy_geo.py). "
             "THE 2001 CENSUS IS THE ONLY MEASURED RELIGION GEOGRAPHY CYPRUS HAS, Volume 1 "
             "Table 29, five districts; it is used as a check and not drawn (sources/cy.md "
             "§5). Below district there is only the 1960 census, village by village and "
             "island-wide, as a scan with no text layer (§6).",
    ),
}
