# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _at_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read.

    Austria is half Alps and its Gemeinden do not know it: Sölden is 466 km2, Neustift im
    Stubaital 250, and the Hohe Tauern fringe the same shape, all of them polygons whose
    people live along one valley floor and whose remaining nine tenths are rock, glacier and
    Nationalpark. An equal share per polygon draws a large part of Tirol, Salzburg and
    Kärnten onto ice. It also removes the lakes, which here sit INSIDE the Gemeinden rather
    than between them (the Neusiedler See, the Attersee, the Wörthersee).
    """
    return _kontur_place_weight(place, "at_hexes.gpkg", "sources/at_grid.py")


def _at_counts():
    """Volkszählung 2001 Tabelle 4: 9 drawn categories on 2,380 units.

    ONE level, no allocation, nothing modelled — every row is `measured` and may ring.

    THE UNITS ARE NOT ALL THE SAME KIND OF THING, on purpose. Outside Vienna they are the
    2,358 Gemeinden of 2001; Vienna, which is a single Gemeinde of 1,550,123 people, is
    drawn as its 23 Gemeindebezirke instead, because the Wien volume publishes Tabelle 4 at
    Zählbezirk and the Bezirk tier above it. Drawing the city as one polygon would make it
    the coarsest unit on this map by a factor of twenty, in the one part of Austria whose
    composition is least like the rest.

    TWO CATEGORIES RESOLVE TO NOTHING: the universe row, and `Unbekannt`. The latter is
    160,662 real people (2.00%) who did not state a religion and who are marked rather than
    filled (spec §3.5); taxonomy/at2001.py measures which way that hole leans.
    """
    from at2001 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "at.csv",
                     dtype={"geo_id": str}, low_memory=False)
    df = df[df["geo_level"].isin(["gemeinde", "gemeindebezirk"])].copy()

    lut = pd.read_csv(HERE / "data" / "geo" / "at" / "at_lookup.csv", dtype=str)
    known = set(lut["unit"])
    df["unit"] = df["geo_id"].where(df["geo_id"].isin(known))
    # Stallehr (AT80125) has no polygon in GISCO's 2001 commune layer and is the ONLY unit
    # that may go missing; anything else means the lookup is stale (sources/at_geo.py).
    missing = sorted(df.loc[df["unit"].isna(), "geo_id"].unique())
    if missing != ["AT80125"]:
        raise SystemExit(f"at.csv units with no polygon: {missing[:10]} -- expected only "
                         "AT80125 (Stallehr); re-run sources/at_geo.py")
    df = df[df["unit"].notna()]
    if df["unit"].nunique() != 2380:
        raise SystemExit(f"{df['unit'].nunique()} Austrian units, expected 2380")

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations"]]


ENTRY = {
    "at": dict(
        name="Austria",
        source="Volkszählung 2001, Tabelle 4 (Statistik Austria)",
        basis="self-identification, whole resident population",
        view=[9.3, 46.3, 17.3, 49.1],
        note_public=(
            "**This is the 2001 census, and Austria has not asked since.** The country "
            "moved to a register-based census in 2011, and religion was not carried over: "
            "Statistik Austria's own wording is that since the change *\"dieses Merkmal im "
            "Rahmen des Zensus nicht mehr erhoben\"* wird, the characteristic is no longer "
            "collected. There is no municipal religion figure for Austria after 2001 from "
            "any source, and the office's current numbers come from extra questions on the "
            "Mikrozensus labour force survey, which stop at the nine Bundesländer. "
            "**So read every share here as a quarter of a century old, because the country "
            "has moved a long way since.** Against the census's 73.6% Roman Catholic the "
            "2021 survey reads **55.2%**; against 12.0% with no religious affiliation it "
            "reads **22.4%**; Islam goes from 4.2% to **8.3%** and Orthodoxy from 2.2% to "
            "**4.9%**. That Catholic fall is the largest religious change in Austria in a "
            "century and none of it is on this map. What the map can still show is where "
            "each group was, at a resolution nothing since has matched: 2,358 Gemeinden "
            "plus Vienna's 23 districts, about 3,400 people each. "
            "**The pattern it shows is a city against a countryside.** Vienna was 49.2% "
            "Catholic when Austria was 73.6%, and 25.7% of the city already reported no "
            "religion. Inside it the range is wider still, from 40.2% Catholic in "
            "Rudolfsheim-Fünfhaus to the outer western districts; that same district was "
            "**14.7%** Muslim and **11.4%** Orthodox, which makes it the most mixed place "
            "in the country. At the other end Zwettl in the Waldviertel was **94.7%** "
            "Catholic. "
            "**The largest Muslim share in Austria was not in Vienna.** Lustenau in "
            "Vorarlberg was **15.7%**, ahead of every Viennese district, and Vorarlberg as "
            "a whole was 8.4% against a national 4.2%; the Rhine valley textile mills "
            "recruited in Anatolia from the 1960s and the pattern has not moved since. "
            "Protestants are similarly concentrated rather than thin: 13.3% in Burgenland "
            "and 10.3% in Kärnten against 2.2% in Vorarlberg, which is the "
            "Counter-Reformation's map. "
            "**Two of the ten boxes are catch-alls, and what is inside them is published "
            "for the country and not for any place in it.** *Andere christliche "
            "Gemeinschaften* holds 69,227 people: Jehovah's Witnesses 23,206, Old Catholics "
            "14,621, then Free Christians, Adventists, the New Apostolic Church, Anglicans, "
            "Latter-day Saints, Baptists, Methodists and Mennonites. *Andere "
            "nichtchristliche Gemeinschaften* holds 19,750: Buddhists 10,402, Hindus 3,629, "
            "Sikhs 2,794, Bahá'í 760 and a few hundred others. Those totals are national "
            "only, so this map draws the two cells whole rather than pretending to know "
            "which Gemeinde the Buddhists were in. "
            "**160,662 people, 2.0%, did not state a religion and are not drawn.** They are "
            "not spread evenly: across the drawn units that non-response runs with the "
            "no-religion share (+0.43) and the Orthodox share (+0.41) and against the "
            "Catholic share (−0.39), reaching 4.2% in Vienna and 0.6% in Burgenland. So the "
            "shares on this map are, very slightly, more Catholic than Austria was. "
            "**One more thing the census counted that is worth looking at.** Leopoldstadt "
            "and the Innere Stadt read **3.1%** and **3.3%** Jewish, the highest anywhere "
            "in the country and the only units above 1%. Before 1938 Leopoldstadt was "
            "somewhere around two fifths Jewish."),
        how="census, 2001",
        grain="Gemeinden and Vienna districts, 3,400 people on average",
        gap_share=0.02,
        gap="the 2.0% who did not state a religion; Stallehr (272 people), which has no "
            "polygon in the 2001 boundary set",
        counts=_at_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "at" / "at_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_at_place_weight,
        note="THE COLUMN NUMBERS ARE PRINTED OUT OF ORDER AND THE NINE VOLUMES DISAGREE "
             "WITH EACH OTHER. In the eight Länder volumes Tabelle 4's header numbers read "
             "1 2 3 5 4 6 7 8 9 10 11 left to right, so Orthodox is printed fourth and "
             "numbered 5 while Evangelisch is printed fifth and numbered 4; in the Wien "
             "volume the same eleven columns are numbered in print order. Keying on the "
             "printed number therefore swaps Orthodoxy and Protestantism in eight volumes "
             "of nine, silently, because both are plausible sizes and every total still "
             "reconciles. sources/at.py identifies columns from the header LABELS by "
             "x-position and asserts the resulting order; Vorarlberg's separately published "
             ".xls of the same table confirms the anomaly is the source's and not the "
             "parser's. "
             "THE BOUNDARY VINTAGE IS THE OTHER TRAP AND GISCO SOLVES IT OUTRIGHT. Austria "
             "has merged Gemeinden hard since 2001, Styria alone going from 542 to 287 in "
             "the reform of 2015, so a current boundary file loses a third of one "
             "Bundesland and mis-seats the rest. Eurostat's GISCO publishes a Communes 2001 "
             "layer, which is the census's own Gebietsstand, so no crosswalk is needed at "
             "all and the join is on the Topographische Kennziffer with names used only to "
             "check it (2,344 of 2,357 agree after folding). "
             "THE 31 CATEGORIES IN UNSD'S TABLE ARE NATIONAL AND DO NOT EXIST AT GEMEINDE. "
             "Every subnational table in the 2001 publications carries ten. The 31 are used "
             "here as a check instead, and a strong one: six of them reproduce a drawn "
             "column exactly and the other 25 decompose the remaining four columns to the "
             "person, with no row used twice and no remainder.",
    ),
}
