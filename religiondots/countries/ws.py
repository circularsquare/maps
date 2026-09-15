# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _ws_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read.

    Upolu and Savai'i are volcanic islands whose interiors are forest and lava field and
    whose people live in a ring of villages along the coast road. Savai'i is 1,700 km2 with
    an empty middle, so weighting a district's dots by its area would put most of Palauli's
    and Gagaifomauga's on the slopes of Mount Silisili (sources/ws_grid.py).
    """
    return _kontur_place_weight(place, "ws_hexes.gpkg", "sources/ws_grid.py")


def _ws_counts():
    """SBS 2021 census Table 2: 26 categories, read at village, drawn at 25 districts.

    ONE level, no allocation, nothing modelled -- every row is `measured` and may ring.
    NOTHING IS EXCLUDED: Samoa has no not-stated cell at all, so the country is 100% drawn.

    THE TABLE IS THE CLEANEST ON THIS MAP. Its 26 categories sum to the printed total at
    EVERY one of its 395 place rows, and its four tiers -- country, 4 regions, 51 districts,
    339 villages -- nest exactly on all 27 columns. No tolerance anywhere.

    THE DRAWN UNIT IS COARSER THAN THE DATA AND THE REASON IS GEOMETRY. ws.csv holds all 339
    villages at 606 people each. Samoa is the one Pacific country with no COD-AB, and the
    only polygon layer below its 11 political districts is a 43-unit file (geoBoundaries and
    GADM are the same Pacific Data Hub source) that is a DIFFERENT cut from the census's 51.
    Both fold onto the same 25 traditional districts by name, which is what sources/ws_geo.py
    builds -- exactly, set against set, with nothing geocoded. 8,222 people per unit.
    """
    from ws2021 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "ws.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    df = df[df["geo_level"] == "village"].copy()

    lut = pd.read_csv(HERE / "data" / "geo" / "ws" / "ws_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    missing = sorted(df.loc[df["unit"].isna(), "geo_id"].unique())
    if missing:
        raise SystemExit(f"ws.csv villages with no district: {missing} -- re-run "
                         "sources/ws_geo.py, the lookup is stale")
    if df["unit"].nunique() != 25:
        raise SystemExit(f"{df['unit'].nunique()} districts, expected 25")

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df = df.groupby(["unit", "node"], as_index=False)["count"].sum()
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations"]]


ENTRY = {
    "ws": dict(
        name="Samoa",
        source="2021 Population and Housing Census, Table 2 (Samoa Bureau of Statistics)",
        basis="self-identification, whole enumerated population",
        view=[-172.9, -14.15, -171.35, -13.4],
        note_public=(
            "**The Samoan village tends to belong to one church, and the census counts "
            "finely enough to show it.** Malua on Upolu is **424 people and 424 "
            "Congregationalists**, with nobody at all in any of the other twenty-five "
            "columns. Amaile is **99.6%** Roman Catholic, Mulivai Safata 99.5%, Tapueleele "
            "**99.4%** Latter Day Saints, Gataivai 94.4% Methodist. Under the fa'amatai "
            "system the village council decides matters for the village, and which church "
            "the village has is one of them, so a village row here reads more like one "
            "collective answer than like six hundred separate ones. "
            "**What the map draws is coarser than that, and the reason is maps and not "
            "counting.** Samoa is the only country in the Pacific with no standard boundary "
            "dataset, so the 339 villages and the census's own 51 districts have no "
            "polygons anywhere; the finest geography that exists is the 25 traditional "
            "districts, at 8,200 people each. The village figures above are read from the "
            "census and are not on screen. "
            "**Three churches divide the country between them geographically.** The "
            "Methodists are Savai'i: **20.5%** of that island against 8 to 10% of everywhere "
            "else, and **62.7% of Satupaitea**. The Catholics are Apia and the two far ends, "
            "25.3% of the Apia urban area and **35.4%** of both Falealupo at the western tip "
            "of Savai'i and Aleipata Itupa i Lalo at the eastern end of Upolu. The Latter "
            "Day Saints run opposite the Catholics, 21.4% on Savai'i against 13.6% in Apia, "
            "and reach **34.6% of Vaa o Fonoti**. Over all of it the Congregational "
            "Christian Church is the largest body in every region, at **27.0%** nationally. "
            "**Samoa is the second most Latter-day Saint country the UN has a figure for**, "
            "at **17.6%**, behind Tonga's 19.7%. "
            "**And one church here exists in no other census on earth.** Aso Fitu, the Samoa "
            "Independent Seventh Day Adventist Church, is 1,962 people; the name is Samoan "
            "for the seventh day. "
            "Nobody was left out. Samoa's 2016 census had a not-stated box and the 2021 one "
            "does not, so all 205,557 people are in a named category, and the categories sum "
            "to the printed total at every one of the table's 395 rows."),
        how="census, 2021, whole enumerated population",
        grain="traditional districts, 8,200 people on average",
        counts=_ws_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "ws" / "ws_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_ws_place_weight,
        note="THE CLEANEST TABLE ON THIS MAP, AND THE WORST GEOMETRY IN THE PACIFIC. Table 2 "
             "is 401 x 82: 26 named religions plus a total, on four tiers nested by "
             "INDENTATION in column A (0 = Samoa, 4 = 4 statistical regions, 8 = 51 "
             "districts, 12 = 339 villages). The 26 categories sum to the printed total at "
             "every one of its 395 place rows, the villages sum to their districts, the "
             "districts to their regions and the regions to the country, all on 27 columns, "
             "with no tolerance anywhere. There is no not-stated cell and the only residual "
             "is a named OTHER CHURCHES at 1.9%, so the country is 100% drawn. One GET from "
             "sbs.gov.ws, 2.7 MB, no plugin and no wall. "
             "SAMOA IS THE ONE PACIFIC COUNTRY WITH NO COD-AB, re-checked 2026-09-08. Tonga, "
             "Fiji, the Solomons, PNG, Vanuatu, Kiribati, FSM and the Marshall Islands all "
             "have one. What exists is geoBoundaries ADM2 at 43 polygons from the Pacific "
             "Data Hub, and GADM 4.1 level 2 is THE SAME 43 with the same names and the same "
             "(PART) artefacts, so it is one source and not two. OSM has 11 admin relations "
             "for the whole country. pacificdata.org is behind a Cloudflare challenge. SBS "
             "publishes no geography itself: 849 media files with no shapefile, and its own "
             "census dashboard is a Looker Studio embed. "
             "THE 43 ARE A DIFFERENT CUT FROM THE CENSUS'S 51, NOT A COARSER ONE. The census "
             "numbers its districts (Vaimauga 1 to Vaimauga 4); the polygons name them by "
             "compass point (Vaimauga East, Vaimauga West). Neither nests in the other, and "
             "the name stem alone settles only 18 of the 51 districts, 23.1% of the people. "
             "WHAT WORKS IS THAT BOTH ARE CUTS OF THE SAME 25 TRADITIONAL DISTRICTS, so both "
             "fold onto those 25 by name. sources/ws_geo.py does that fold and asserts the "
             "two stem sets are equal, set against set, which is the whole argument of the "
             "build: nothing is geocoded and nothing is assumed. "
             "THE ROUTE TO 43 UNITS WAS TRIED AND REJECTED, and it is the next improvement. "
             "OSM has 554 Samoan villages; matching the census's 339 against them by name, "
             "disambiguated by requiring the census district's stem to agree with the "
             "polygon's, placed 285 villages and 85.2% of the population. The rest fail "
             "because the census anglicises (Lalovaea East is OSM's Lalovaea Sasa'e, Samata "
             "Uta is Samata-i-Uta) and qualifies repeated names with a district (Vailoa "
             "Faleata), while Solosolo, Falefa, Faleseela, Falevao and Tuanimato are absent "
             "from OSM entirely. A WRONG VILLAGE WOULD MOVE PEOPLE BETWEEN DISTRICTS AND "
             "EVERY TOTAL WOULD STILL BALANCE, so 85% is not a basis to draw on. "
             "THE 2016 CENSUS IS THE OUTSIDE CHECK, NOT ON THE NUMBERS BUT ON THE "
             "INSTRUMENT. UNSD table 28 has Samoa for 2001 and 2016 and not 2021, so it "
             "cannot verify a 2021 figure; sources/ws.py uses it to assert that all 24 of "
             "2016's churches still have a 2021 column, so none was quietly merged away. "
             "2021 adds Aso Fitu (SISDAC) and Amazing Love and drops Not Stated. The "
             "Yearbook also names three cells the 2021 workbook renders in Samoan, which is "
             "what identifies them: Protestant is POROTESANO, Baptist is PABTISM (the "
             "workbook's own spelling), and Aoga Tusi Paia is BIBLE STUDY. "
             "16% OF KONTUR'S PEOPLE FALL OUTSIDE EVERY DISTRICT and are snapped, not "
             "dropped, on Vanuatu's rule (§9bg): settlement is a coastal ribbon and the loss "
             "would be entirely seaward. 100% of the strays are within 700 m; 4 people are "
             "dropped. The census-against-Kontur correlation is r=0.986 over 25 units with "
             "all of them inside a factor of 3, but 25 units is a weak test and it is "
             "corroboration rather than the check that carries the join.",
    ),
}
