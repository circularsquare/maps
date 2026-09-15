# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _lt_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read."""
    return _kontur_place_weight(place, "lt_grid_400m.gpkg", "sources/lt_geo.py")


def _lt_counts():
    """Statistics Lithuania, 2021 census, at municipality: 15 nodes on 60 units.

    ONE level, no allocation, nothing modelled. The cube publishes 16 religions at
    municipality and there is nothing finer, so every drawn row is `measured`. lt.csv also
    carries the country, two NUTS2 regions and ten counties, which are the same people
    three more times.

    THE TABLE IS DEEPER THAN ITS SIZE SUGGESTS — Lithuania separates Roman from Greek
    Catholics, Orthodox from Old Believers, and names the Karaims, which no other census on
    this map does. 2.8 million people carry fifteen distinct nodes; Serbia's 6.6 million
    carry eight.

    AND IT IS THE PROJECT'S SHARPEST CASE OF §3.8. 298 of the 1,020 municipality cells are
    withheld as confidential — disclosure control on small religions — so **60.4% of
    Lithuania's Karaims, 29.3% of its Greek Catholics and 28.2% of its Adventists have no
    municipality to be drawn in**, against 0.06% of the population overall. Those people
    are not filled in (§3.5); sources/lt.py reports the shortfall per category. The map
    therefore understates exactly the categories it is most interesting for, and it
    understates them by more the smaller they are.

    THE DRAWN POPULATION IS 86.3% OF THE COUNTRY. `Nenurodyta` — not stated, 384,094
    people, 13.67% — is excluded as a refusal, and taxonomy/lt2021.py notes that it has
    trebled since 2001 while the explicit no-religion answer has not moved.
    """
    from lt2021 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "lt.csv",
                     dtype={"geo_id": str}, low_memory=False)
    df = df[df["geo_level"] == "municipality"].copy()
    df["geo_id"] = df["geo_id"].str.zfill(2)

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df = df.rename(columns={"geo_id": "unit"})
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations"]]


ENTRY = {
    "lt": dict(
        name="Lithuania",
        source="Gyventojų surašymas 2021 (Statistics Lithuania)",
        basis="self-identification, voluntary question",
        view=[20.8, 53.8, 26.9, 56.5],
        gap="13.7% who did not answer a voluntary question, against 5.4% in 2001",
        gap_share=0.1373,
        note_public=(
            "**The most detailed religion question in Europe on a country this size, and "
            "every one of its answers is a historical boundary somewhere.** Lithuania is "
            "74% Roman Catholic and separates things no other census on this map "
            "separates: Roman from Greek Catholics, Orthodox from Old Believers, and the "
            "Karaims from the Jews. "
            "**Biržai is the Reformed municipality.** 8.9% of it is Evangelical Reformed "
            "against 0.49% in the next-highest place in the country — this is the Radvila "
            "family's Calvinist estate, granted in the 1560s, and four and a half centuries "
            "later it is still a single bright spot with nothing around it. "
            "**The Lutherans are the old Prussian border.** Tauragė 9.2%, Pagėgiai 5.4%, "
            "Šilutė 4.6%, Jurbarkas 3.4% — a band along the Nemunas that was Lithuania "
            "Minor under Prussia, Lutheran since the Reformation and never Catholic. The "
            "line between it and Catholic Samogitia (Šilalė is 91.5% Catholic) is a "
            "sixteenth-century state border still visible in a 2021 census. "
            "**Visaginas is 49% Orthodox** — a town built in the 1970s for the Ignalina "
            "nuclear plant and populated from across the Soviet Union, and the only "
            "municipality in Lithuania that is not majority Catholic. **The Old Believers "
            "are the north-east**: Zarasai 12.1%, Švenčionys 5.0%, descendants of refugees "
            "from the Russian church reforms of the 1650s, and 18,196 of them — the largest "
            "count of Old Believers any source on this map makes directly. "
            "**Two very small communities, and a warning about both.** 2,165 Sunni Muslims "
            "are the Lipka Tatars, settled around Vilnius and Alytus since the fourteenth "
            "century. 255 people answered Karaim — the Turkic-speaking Karaite community "
            "brought from Crimea in 1397, whose historic home is Trakai. **Only 101 of them "
            "are drawn, and not in Trakai.** Statistics Lithuania withholds any cell small "
            "enough to identify people, so 60% of the Karaims, 29% of the Greek Catholics "
            "and 28% of the Adventists have no municipality on this map at all. The "
            "suppression is 0.06% of Lithuania and a majority of its smallest religion, "
            "which is what disclosure control does: it hides the rare things, and this map "
            "does not fill them back in. "
            "**And 13.7% did not answer.** That is a refusal, not irreligion — 6.1% "
            "separately said they belong to no religion, and that share has not moved since "
            "2001 while non-response has trebled from 5.4%. Almost all of the fall in "
            "Catholic identification over twenty years has gone into the blank rather than "
            "into 'none'. The people who did say 'no religion' are not in Vilnius: Joniškis "
            "(12.5%), Akmenė (12.3%) and Klaipėda (11.8%) lead it, which is the north and "
            "the coast rather than the capital."),
        how="census, 2021, voluntary question",
        grain="municipalities, 40,000 people on average",
        counts=_lt_counts,
        # Counts are on the 60 savivaldybės; the Kontur hexes carry no municipality code,
        # so sources/lt_geo.py assigns and clips every hex and writes the `unit` column
        # this reads. Serbia's, Kenya's and Russia's wiring exactly.
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "lt" / "lt_grid_400m.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_lt_place_weight,
        note="Statistics Lithuania is self_id on a voluntary question; the 13.67% not "
             "stated are excluded rather than drawn (spec §3.5), so this is 86.3% of the "
             "country. The table is published at municipality and nowhere finer, so nothing "
             "is allocated and every drawn row is measured. "
             "**298 of the 1,020 municipality cells are withheld as confidential** and are "
             "NOT filled in — 1,683 people, 0.06% of Lithuania, but 60.4% of its Karaims, "
             "29.3% of its Greek Catholics and 28.2% of its Adventists (spec §3.8; "
             "sources/lt.py reports the shortfall per category). Small religions are "
             "understated here in proportion to how small they are. "
             "The cube is SDMX from `osp-rs.stat.gov.lt`, which is open — the Cloudflare "
             "wall recorded in sources.md §11 is on `osp.stat.gov.lt`, the web UI, and the "
             "two are different machines. "
             "Boundaries are GISCO LAU 2021 and the join is by CODE: GISCO's `LAU_ID` is "
             "the savivaldybė code the census keys on, so there is no name matching at all. "
             "Placement is Kontur's 400 m H3 grid, 63,766 hexes, a median of 1,206 per "
             "municipality. Kontur displaces city population outward — Šiauliai city reads "
             "0.48x and the rajono ring around it 1.92x — but every city/ring pair closes "
             "between 0.89x and 1.05x, which is what says the join is right and the surface "
             "is merely blurred (sources/lt_geo.py).",
    ),
}
