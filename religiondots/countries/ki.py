# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _ki_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read.

    An atoll is a ring and its middle is a lagoon. Weighting by polygon area would put most
    of Butaritari's and Abaiang's dots on open water inside the reef and most of Kiritimati's
    on salt flats; the land is a strip a few hundred metres wide (sources/ki_grid.py).
    """
    return _kontur_place_weight(place, "ki_hexes.gpkg", "sources/ki_grid.py")


def _ki_counts():
    """Kiribati NSO 2015 census Table 6: 14 categories on 24 inhabited islands.

    ONE level, no allocation, nothing modelled -- every row is `measured` and may ring.
    NOTHING IS EXCLUDED: the 2015 table has no refusal cell and no not-stated cell.

    THE ISLANDS SUM TO THE PRINTED NATIONAL ROW EXACTLY on all fifteen columns, and the
    national row then matches UNSD Demographic Yearbook table 28 on all fourteen categories
    to the person -- an outside witness, from the return Kiribati forwarded rather than from
    this PDF.

    2015 AND NOT 2020, WHICH IS THE ONE REAL COMPROMISE HERE. The 2020 census publishes
    religion NATIONALLY ONLY; its Census Atlas maps religion by island but the map is a
    raster with no numbers behind it. The per-island workbooks on nso.gov.ki go down to
    VILLAGE but they are the 2005 census, twenty years old, with a column set that differs
    island to island. 2015 is the newest year published with a geography and has the longest
    list of the three.
    """
    from ki2015 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "ki.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    df = df[df["geo_level"] == "island"].copy()

    lut = pd.read_csv(HERE / "data" / "geo" / "ki" / "ki_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    missing = sorted(df.loc[df["unit"].isna(), "geo_id"].unique())
    if missing:
        raise SystemExit(f"ki.csv islands with no polygon: {missing} -- re-run "
                         "sources/ki_geo.py, the lookup is stale")
    if df["unit"].nunique() != 24:
        raise SystemExit(f"{df['unit'].nunique()} islands, expected 24")

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df = df.groupby(["unit", "node"], as_index=False)["count"].sum()
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations"]]


ENTRY = {
    "ki": dict(
        name="Kiribati",
        source="2015 Population Census, Report Volume 1, Table 6 (Kiribati National "
               "Statistics Office)",
        basis="self-identification, whole enumerated population",
        # The inhabited extent, in Fiji's beyond-180 convention (§9bd): Banaba at 169.5 E to
        # Kiritimati at 157.4 W = 202.6 E, and Arorae at 2.7 S to Teraina at 4.7 N. The dot
        # bbox tiles.py would compute instead spans the antimeridian and frames badly.
        view=[169.0, -3.5, 203.5, 5.5],
        note_public=(
            "**The missions divided the Gilbert chain between them in the 1800s and the "
            "census still draws the line.** Kiribati's sixteen main atolls run roughly "
            "north to south over 600 km, and the Catholic share falls almost steadily as "
            "you go down them while the Protestant share rises to meet it. Butaritari at "
            "the top is **82.5%** Roman Catholic and 13.2% Kiribati Protestant Church; "
            "Arorae at the bottom is **98.0%** Protestant and 1.4% Catholic. Makin, "
            "Marakei and Abaiang in the north are all above 75% Catholic; Tamana beside "
            "Arorae is 95.8% Protestant. The Sacred Heart mission worked the northern "
            "islands and the Congregational missions the southern ones, and nothing since "
            "has blurred it much. "
            "**The Protestant church here came from two mission societies, not one.** The "
            "American Board sent Hawaiian pastors from 1857 and the London Missionary "
            "Society sent Samoan and Tuvaluan ones from 1870, and what grew out of both is "
            "the Kiribati Protestant Church, **31.3%** of the country. It is the same "
            "family as the national churches of Samoa, Tuvalu, Niue and the Cook Islands, "
            "which the same missionaries founded. "
            "**One island is 12.0% Bahá'í.** South Tabiteuea, against 2.1% nationally, "
            "which is the sharpest minority concentration in the country. "
            "**The map is drawn from 2015 rather than 2020, because 2020 is national "
            "only.** The 2020 census counted 119,438 people and published religion for the "
            "country as a whole; its atlas maps religion by island as a picture without the "
            "numbers behind it. The 2015 figures are the newest that come with a geography. "
            "**One thing 2015 cannot show is a split that had just happened.** The "
            "Kiribati Protestant Church became the Kiribati Uniting Church in 2014, and "
            "about ten thousand members declined the union and kept the old church going. "
            "The 2015 census still counts them as one answer; the 2020 census counts the "
            "Uniting Church at 21% and a continuing Protestant Church at 8%. Here they are "
            "drawn together. "
            "Nobody was left out: the 2015 table has no refusal cell and no not-stated "
            "cell, so all 110,136 people are in a named category. Kiribati is also the most "
            "spread-out country on this map, reaching from Banaba to Kiritimati across "
            "4,000 km and over the date line."),
        how="census, 2015, whole enumerated population",
        grain="islands, 4,600 people on average",
        counts=_ki_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "ki" / "ki_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_ki_place_weight,
        note="RELIGION BY ISLAND WITH FOURTEEN CATEGORIES, AND THE UN HAS THE SAME NUMBERS. "
             "Table 6 of the 2015 report gives each of the 24 inhabited islands a row, 4,589 "
             "people each. The islands sum to the printed national row EXACTLY on all fifteen "
             "columns, and the national row then matches UNSD table 28 on all fourteen "
             "categories to the person, from the return Kiribati forwarded rather than from "
             "this PDF. No tolerance anywhere. "
             "2015 AND NOT 2020, AND NOT 2005. The 2020 census publishes religion NATIONALLY "
             "ONLY (report Table G-3); the Census Atlas 2022 has a `Religious affiliation by "
             "island` map but it is a raster and its page carries 66 characters of text. The "
             "per-island workbooks on nso.gov.ki are finer still, religion by VILLAGE, but "
             "they are the 2005 census and their column set differs island to island (Beru "
             "has an AOG column Abaiang does not). 2015 is the newest year with a geography "
             "and the longest list of the three. "
             "KIRIBATI STRADDLES THE ANTIMERIDIAN AND THE USUAL WIDTH CHECK IS WRONG HERE. "
             "The country's own bounding box spans 351 degrees of longitude, correctly, so "
             "the assertion every other country gets would fire on good data and would have "
             "to be switched off -- which is how a real tear gets missed later. sources/"
             "ki_geo.py checks PER POLYGON instead (no island may span more than 3 degrees; "
             "the widest is Kiritimati) and ki_grid.py checks per hexagon. Fiji (§9bd) needed "
             "the opposite treatment because its PROVINCES genuinely cross 180; none of "
             "Kiribati's islands does. "
             "THE JOIN IS 24/24 ON THE NAME, with five contractions the report uses for the "
             "compass-point pairs (NTarawa, STarawa, NTabiteuea, STabiteuea, Teeraina). The "
             "twelve COD polygons with no census row are all genuinely uninhabited: five in "
             "the Line Islands and seven in the Phoenix group, where Kanton and its 20 people "
             "are the only settlement. "
             "39% OF KONTUR'S PEOPLE FALL OUTSIDE EVERY ISLAND and are snapped, not dropped, "
             "on Vanuatu's rule (§9bg) -- the highest share of any country here, because an "
             "atoll is a strip of land a few hundred metres wide and almost every cell is a "
             "shoreline cell. 100% of the strays are within 700 m; 9 people are dropped. "
             "r=0.964 over 24 units against a best of 0.72 over 2,000 random pairings. "
             "THE ORACLE'S LABELS FOR THIS COUNTRY ARE NOT TRUSTWORTHY AND THE NUMBERS ARE. "
             "UNSD table 28 renders `KPC` as `Kempsville Presbyterian Church`, a false "
             "expansion of the initials, and its 1995 row is mangled outright (`African "
             "Methodist Episcopal Church`, `Arya Samajist`, `Bengali` for a country that has "
             "none of them). sources/ki.py therefore pairs the two sources on an explicit "
             "alias table rather than on the string, so a disagreement about a NUMBER cannot "
             "hide behind one about a NAME. Every one of the fourteen numbers agrees.",
    ),
}
