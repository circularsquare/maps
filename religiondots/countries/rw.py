# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _rw_place_weight(place):
    """countries.py hook. `place` is the hex layer scatter.py has read.

    Rwanda's grid is not the usual one. It is Kontur's 400 m hexes RE-LEVELLED SECTOR BY
    SECTOR onto the census's own count, which NISR publishes for all 416 sectors even
    though religion stops at the 30 districts (sources/rw_grid.py). So the weight inside a
    district is measured rather than modelled down to about 15 km² and 32,000 people, and
    only the shape inside a sector comes from Kontur.

    It is worth the trouble because Rwanda's density is uneven at a scale a district hides:
    537 people per km² nationally, sectors in Kigali past 20,000 and sectors in the Akagera
    and Nyungwe belts under 100. An equal share per district would put dots in the national
    parks.
    """
    return _kontur_place_weight(place, "rw_hexes.gpkg", "sources/rw_grid.py")


def _rw_counts():
    """NISR RPHC-5 2022 at district: 10 drawn categories on 30 districts.

    ONE level, no allocation, nothing modelled — every row is `measured` and may ring.

    30 UNITS AND NOT 5, WHICH IS THE WHOLE POINT OF THE COUNTRY. sources.md §11p priced
    Rwanda at five provinces off the RPHC-5 *Social-cultural characteristics* thematic
    report, which is where religion sits in the thematic series and which works at
    province. NISR's thirty DISTRICT PROFILES, released May 2025 and outside that series,
    each carry the same eleven categories as their Table 2.2. 441k per unit instead of
    2.6M. Botswana's shape exactly (§9bu).

    THE JOIN IS PROVED AND NOT ASSERTED. The polygons are NISR's own `Population_2002_2022`
    ArcGIS layer, which carries the census count on every one of its 416 sector polygons,
    and dissolved to district all 30 populations equal the booklets' own to the person on
    30 distinct values. See sources/rw_geo.py.

    `Not stated` IS THE ONLY THING LEFT OUT: 17,785 people, 0.13%, the smallest §3.5
    residual of any census here that prints one.
    """
    from rw2022 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "rw.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    df = df[df["geo_level"] == "district"].copy()

    lut = pd.read_csv(HERE / "data" / "geo" / "rw" / "rw_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    missing = sorted(df.loc[df["unit"].isna(), "geo_id"].unique())
    if missing:
        raise SystemExit(f"rw.csv districts with no polygon: {missing} -- re-run "
                         "sources/rw_geo.py, the lookup is stale")
    if df["unit"].nunique() != 30:
        raise SystemExit(f"{df['unit'].nunique()} districts, expected 30")

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations"]]


ENTRY = {
    "rw": dict(
        name="Rwanda",
        source="RPHC-5 2022 district profiles, Table 2.2 (NISR)",
        basis="self-identification, whole census population",
        note_public=(
            "**One in five Rwandans belongs to a single named church, and the census "
            "counts it under that name.** ADEPR, the Association des Églises de Pentecôte "
            "du Rwanda, is **21.3%** of the country, 2,820,813 people, and the "
            "second-largest religious answer after Catholicism at **39.9%**. Nothing else "
            "on this map has one denomination counted as a census category at that size; "
            "the UN's own copy of these figures prints the cell as plain *Pentecostal* and "
            "loses the church. It grew out of the Swedish Free Mission that reached Rwanda "
            "from the Belgian Congo in 1940, and eighty years on the map still shows where "
            "the mission landed. Rusizi, on the Congo border, is its strongest district at "
            "**33.9%**, Nyamasheke next door is **30.7%**, and it thins eastward across "
            "the Catholic plateau to **11.1%** in Nyamagabe. "
            "**Rwanda is drawn on 30 districts, and the thematic report that carries "
            "religion only goes down to five provinces.** NISR published a separate "
            "profile for every district in May 2025 and each one repeats the religion "
            "question in full. One column shows what that is worth more than any other: "
            "Adventists are **12.2%** of Rwanda and run from **33.9%** in Nyanza to "
            "**2.4%** in Gicumbi, a belt across the middle of the country that the "
            "province figures flatten to a range of 9.5% to 14.6%. "
            "**Catholicism is the largest answer in 27 of the 30 districts**, and where it "
            "is strongest is the White Fathers' mission field of a century ago: **65.0%** "
            "in Muhanga, 62.5% in Rulindo, 60.8% in Gakenke, against **21.1%** in Karongi "
            "on Lake Kivu. Protestant Rwanda is close to the mirror image of that, highest "
            "in Nyamagabe at 27.7% and lowest in Rulindo at 4.8%, which is 62.5% Catholic. "
            "**Islam is 2.0% of the country and most of it is in one district.** "
            "Nyarugenge, which contains Nyamirambo, is **11.3%** Muslim; the next highest "
            "is Kicukiro at 4.0%, and Nyamasheke is 0.2%. "
            "**Traditional religion is recorded as 2,112 people, 0.02%, and that is a "
            "floor rather than a count.** The box is exclusive of the Christian ones, and "
            "*kubandwa*, the Ryangombe possession cult, has for a century been something "
            "people do alongside church membership rather than instead of it. The cell is "
            "largest in Kigali, at 0.03% in Gasabo and Kicukiro, which is one more reason "
            "not to read it as a measure of practice."),
        how="census, 2022",
        grain="districts, 442,000 people on average",
        gap_share=0.001343,
        gap="the 17,785 people, 0.13% of the country, who did not state a religion",
        counts=_rw_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "rw" / "rw_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_rw_place_weight,
        note="THE QUEUE ROW PRICED THIS COUNTRY AT FIVE PROVINCES AND IT IS THIRTY "
             "DISTRICTS. sources.md §11p read the RPHC-5 thematic report on social and "
             "cultural characteristics, which is where religion sits in the thematic "
             "series and which works at province. NISR's thirty DISTRICT PROFILES, "
             "released May 2025 on /district-statistics/<province> and outside that "
             "series, each carry the same eleven categories as Table 2.2. 441,546 people "
             "per unit instead of 2.65M. This is Botswana's shape exactly (§9bu): the "
             "national report stops at the nation and the per-district booklet series, "
             "outside the report set the sweep tested, carries the fine geography. "
             "THE JOIN IS PROVED TO THE PERSON AND NOT ASSERTED. The polygons are NISR's "
             "own `Population_2002_2022` ArcGIS Online layer, which carries the census "
             "count on each of its 416 SECTOR polygons; dissolved to district, all 30 "
             "populations equal the booklets' own exactly, on 30 distinct values, so no "
             "other pairing of these names to these polygons reproduces them. The layer's "
             "`province` field is checked against the province each booklet prints before "
             "the counts are looked at. "
             "THE PLACEMENT GRID IS RE-LEVELLED ONTO THOSE SECTORS, which is unusual here "
             "and is possible because Rwanda counted finer than it published religion: "
             "Kontur's hexes are scaled per sector so each sector sums to its census "
             "count, so everything coarser than ~15 km2 comes from the census and only the "
             "shape inside a sector is modelled. The Kontur-versus-census check is run on "
             "the RAW ratios before that scaling, because afterwards every ratio is 1.000 "
             "by construction; r = 0.9113 on 416 sectors against a best of 0.1507 over "
             "2,000 random pairings. "
             "THE OUTSIDE WITNESS IS THE UNSD DEMOGRAPHIC YEARBOOK. Table 28's eleven "
             "Rwanda 2022 figures are NISR's return to the UN, a separate publication from "
             "these booklets, and all eleven reproduce to the person from the thirty "
             "district tables. It also shows what the booklets add: the UN prints "
             "`Pentecostal` where NISR prints `ADEPR`. "
             "TWO PARSE TRAPS, BOTH SILENT. Table 2.1 (nationality) shares the page in "
             "several booklets and one of ITS column headers is the word `Rwanda`, so a "
             "reader anchoring on that lands in the wrong table. And Gakenke prints the "
             "URBAN cell of `Traditional/Animist` BLANK rather than as a zero, so a plain "
             "line read gives that district 0.01 traditionalists and shifts its "
             "percentages one column left, on a row of 51 people that nothing would query. "
             "The columns are cut on position off the `Catholic` row instead. "
             "NYABIHU IS THE ONE FILE A PATTERN-GUESSING FETCH MISSES: the other 29 are "
             "/sites/default/files/2025-05/<District>.pdf and it is at "
             "/2025-10/Nyabihu_RPHC2022.pdf. The /2025-06/ files on the same pages are the "
             "RPHC-4 (2012) profiles and must not be read for this.",
    ),
}
