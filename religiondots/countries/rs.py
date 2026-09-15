# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _rs_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read."""
    return _kontur_place_weight(place, "rs_grid_400m.gpkg", "sources/rs_geo.py")


def _rs_counts():
    """RZS Popis 2022 at municipality: 11 drawn categories on 168 units.

    ONE level, no allocation, nothing modelled. RZS publishes this table at municipality
    and nothing finer, and the categories are a true partition of each unit's population
    with no suppression and no rounding anywhere in it — so every row is `measured` and
    may ring. Croatia's shape (§9e), one country over and one census later.

    THE DRAWN TIER IS 168 UNITS AND THE SHEET CONTAINS SIX LEVELS. rs.csv also carries the
    republic, two halves, four regions, 25 oblasti and — the one that is easy to miss —
    four `city` rows, which are `Grad Niš`, `Grad Požarevac`, `Grad Užice` and
    `Grad Vranje` sitting in the municipality tier as parents of their own city
    municipalities. Filtering to `geo_level == "municipality"` drops all six aggregates,
    which is why sources/rs.py re-levels those four rather than leaving them looking like
    ordinary municipalities.

    BELGRADE AND NIŠ ARRIVE PRE-SPLIT, which removes §12's capital-in-one-polygon problem
    for free: Belgrade is 25.3% of Serbia and is drawn as its 17 city municipalities,
    Niš as its 5. Hungary got the same gift from KSH.

    THE DRAWN POPULATION IS 92.07% OF THE COUNTRY. Two categories resolve to nothing —
    169,486 who declined a question the constitution makes voluntary, and 355,484 whose
    religion RZS records as unknown. taxonomy/rs2022.py argues why neither is irreligion,
    and why excluding the second is not neutral: it is concentrated in the least religious
    municipalities in the country.
    """
    from rs2022 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "rs.csv",
                     dtype={"geo_id": str}, low_memory=False)
    df = df[df["geo_level"] == "municipality"].copy()

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df = df.rename(columns={"geo_id": "unit"})
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations"]]


ENTRY = {
    "rs": dict(
        name="Serbia",
        source="Popis stanovništva 2022 (Republički zavod za statistiku)",
        basis="self-identification, voluntary question",
        view=[18.6, 42.1, 23.2, 46.3],
        gap=("7.9%: 5.4% the census could not establish, and 2.6% who declined a voluntary "
             "question"),
        gap_share=0.07898,
        note_public=(
            "**Serbia is 81% Orthodox and the whole of the interesting map is in the "
            "remaining fifth**, which sits in two places and nowhere else. "
            "**Vojvodina in the north is where the Habsburg border used to be**, and it "
            "still reads that way. Kanjiža is 85% Catholic, Senta 74%, Ada 72%, Subotica "
            "48% — Hungarian towns along the Tisza — while **Bački Petrovac is 57% "
            "Protestant and Kovačica 41%**, which are the Slovak Lutheran colonies "
            "planted there in the 1740s and still legible as two dark spots in an "
            "otherwise Orthodox province. The census offers one Protestant cell, so "
            "nothing on the map says Lutheran; the geography says it instead. "
            "**The Muslim map is two separate places 300 km apart.** The Sandžak in the "
            "south-west is Bosniak — Tutin 94%, Novi Pazar 83%, Sjenica 78%, Prijepolje "
            "47% — and the Preševo valley on the Macedonian border is Albanian: Preševo "
            "94%, Bujanovac 69%. Tutin and Preševo are the two least Orthodox "
            "municipalities in Serbia, at 2.0% and 4.5%. "
            "**Irreligion is 1.25% and is almost entirely four Belgrade municipalities.** "
            "Stari grad reports 7.7% atheist or agnostic, Vračar 6.2%, Savski venac 5.4%, "
            "against 0.32% in Lazarevac an hour down the road. On this map that is a very "
            "small number: Czechia is 20 points higher and Estonia higher again. "
            "**Two answers are not drawn and together they are 7.9% of the country.** "
            "169,486 people declined the question, which the constitution makes voluntary, "
            "and they are concentrated in Vojvodina's mixed towns — Dimitrovgrad 10.2%, "
            "Subotica 10.1% — where declaring anything carries the most weight. A further "
            "355,484 are recorded as unknown, and those are a different pattern "
            "altogether: central Belgrade, 17.3% in Savski venac and 14.4% in Stari grad "
            "against under 1% in Preševo. That second group tracks the declared-atheist "
            "share, so **leaving it out makes every share on this map slightly more "
            "religious than the country is** — the correction runs one way and this map "
            "does not make it. "
            "**And 602 Jews in the whole country**, of whom 78 are in Stari grad and 66 "
            "in Novi Sad. Before 1941 Belgrade and Novi Sad each held thousands."),
        how="census, 2022, voluntary question",
        grain="municipalities, 36,000 people on average",
        counts=_rs_counts,
        # Counts are on the 168 municipalities; the Kontur hexes carry no municipality
        # code of their own, so sources/rs_geo.py assigns and clips every hex and writes
        # the `unit` column this reads. Russia's and Kenya's wiring exactly.
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "rs" / "rs_grid_400m.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_rs_place_weight,
        note="RZS is self_id on a question the constitution makes voluntary; the 2.55% who "
             "declined and the 5.35% RZS records as unknown are both excluded rather than "
             "drawn (spec §3.5), so this is 92.07% of Serbia. The table is exact — no "
             "rounding, no suppression, every municipality's categories summing to its own "
             "total — and it is published at municipality and nowhere finer, so no "
             "category is allocated and every row is measured. "
             "Boundaries are GISCO LAU 2021, which carries Serbia at no download because "
             "its set is not the EU27; GISCO splits Novi Sad into Novi Sad and "
             "Petrovaradin where the census does not, so Petrovaradin is dissolved back "
             "in (sources/rs_geo.py). Belgrade arrives as its 17 city municipalities and "
             "Niš as its 5, which removes §12's capital-in-one-polygon problem for free — "
             "and creates the only name collision in the file, since both cities have a "
             "Palilula. Placement is Kontur's 400 m H3 population grid, 59,823 hexes, a "
             "median of 312 per municipality; Kontur under-models the Albanian-majority "
             "Preševo valley by a factor of five, so Bujanovac's and Preševo's dots sit "
             "on the weakest surface in the country. Kosovo is in the source as an empty "
             "row and is not drawn from it.",
    ),
}
