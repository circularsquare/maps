# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


class _MwHexWeighter(_KeHexWeighter):
    """Split a district's dots across Kontur 400 m hexagons by hex POPULATION.

    Kenya's class unchanged; Malawi's reason for needing it is different and worth keeping
    beside it. Kenya's problem is empty desert inside huge counties. **Malawi's is water.**
    Lake Malawi is 29,600 km² and it is not a hole in the country — the district boundaries
    run out into the middle of it, so Karonga, Rumphi, Nkhata Bay, Likoma, Salima,
    Nkhotakota and Mangochi each own a slab of open lake. An equal share per polygon would
    put a fifth of Malawi's dots on water, in a band down the whole eastern side, and
    Likoma — an island of 14,527 people whose polygon is almost all lake — would be a wash
    over nothing. A population grid has no hexes on the lake, so the problem does not arise
    rather than being patched (§8.2c, and Ethiopia's finding at §9u).

    Same caveat as Kenya's: it is a POPULATION weight and not a religion one. Nothing
    measures where Malawi's Anglicans sit inside a district, so an Anglican dot and a Muslim
    dot are spread identically. Read it as "religion by district, drawn where Malawians
    live".
    """

    def summary(self):
        return (f"{self.n_pop:,} (unit, node) rows placed on Kontur hex population, "
                f"{self.n_uniform:,} on equal shares where a district's hexes sum to zero "
                f"(sources/mw_grid.py)")


def _mw_place_weight(place):
    """countries.py hook. `place` is the hex layer scatter.py has read."""
    if "pop" not in place.columns:
        print("  !! mw_hexes.gpkg has no `pop` column — run sources/mw_grid.py; "
              "placing on equal shares (§8.2)")
        return None
    return _MwHexWeighter(place)


def _mw_counts():
    """NSO 2018 PHC Table E5 at district: 10 drawn categories on 32 districts.

    ONE level, no allocation, nothing modelled — NSO publishes these categories at this
    geography and the map draws exactly that, so every row is `measured` and may ring.

    32 UNITS FOR 17.56M PEOPLE IS ~549,000 EACH, which is finer per head than Kenya's 47
    counties (1.0M) and than Ghana's regions, and it is NSO's ceiling: Table E5 is the only
    religion table below the national one anywhere in the 311-page report, and the district
    reports carry no religion at all. sources/mw.md §2.

    THE FOUR CITIES ARE SEPARATE UNITS AND NOT PARTS OF THEIR DISTRICTS. Mzuzu, Lilongwe,
    Zomba and Blantyre Cities are printed as peers of Mzimba, Lilongwe, Zomba and Blantyre —
    2,115,867 people, 12.0% of the country — and `sources/mw.py` proves it rather than
    assuming it, by checking that each region's districts sum to the region row on all
    eleven columns. Reading them as nested would double-count them; reading them as absent
    would lose an eighth of Malawi into the wrong polygons.

    ONE CATEGORY RESOLVES TO NOTHING and it is the universe row, so the drawn population is
    the whole table: **17,563,749, which is the entire census count.** NSO publishes no
    `not stated` cell for religion and no residual — the ten denominations sum to the total
    exactly on all 36 printed rows — so Malawi is one of the very few countries here with
    no §3.5 gap of any kind.
    """
    from mw2018 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "mw.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    df = df[df["geo_level"] == "district"].copy()

    lut = pd.read_csv(HERE / "data" / "geo" / "mw" / "mw_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    missing = sorted(df.loc[df["unit"].isna(), "geo_id"].unique())
    if missing:
        raise SystemExit(f"mw.csv districts with no polygon: {missing} -- re-run "
                         "sources/mw_geo.py, the lookup is stale")
    if df["unit"].nunique() != 32:
        raise SystemExit(f"{df['unit'].nunique()} districts, expected 32")

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations"]]


ENTRY = {
    "mw": dict(
        name="Malawi",
        source="2018 Malawi Population and Housing Census, Table E5 (NSO)",
        basis="self-identification, denomination question, whole census population",
        view=[32.5, -17.3, 36.2, -9.2],
        note_public=(
            "**Malawi asks which denomination you belong to, not which religion**, and it "
            "is the only source on this map that names a single Presbyterian body. Eight of "
            "the ten answers are Christian groupings; the other two are Islam and no "
            "religion. That buys detail no other African census here offers — Catholic, "
            "CCAP and Anglican each counted apart — and it costs the rest, because Buddhism, "
            "Hinduism, Judaism and the Bahá'ís have no cell at district and sit in one "
            "residual. "
            "**The mission map of the 1880s is still legible.** The Church of Central "
            "Africa Presbyterian is the 1924 union of three Scottish and Dutch Reformed "
            "missions and you can still see all three: Livingstonia in the north (Mzuzu "
            "City 28.0%, Mzimba 23.9%, Rumphi 22.7%) and Nkhoma in the centre (Lilongwe "
            "City 23.2%, Dowa 22.4%), against 8.8% across the whole south, where Blantyre "
            "synod is smallest and shares the ground. "
            "**Likoma is 74.6% Anglican** — the sharpest single-denomination figure of any "
            "district in Malawi, on an island of 18 km² in the middle of the lake where the "
            "Universities' Mission put its cathedral in 1903. Ntchisi (21.5%) and "
            "Nkhotakota (15.3%) are the lakeshore stations behind it, and then it stops: "
            "those three units hold a third of the country's Anglicans on 4% of its people. "
            "**The Muslim south is Yao country and it is one block, not a scatter.** "
            "Mangochi is 72.7% and Machinga 67.0%, with Balaka and Salima behind them, "
            "against 13.8% nationally and 0.08% in Chitipa at the Tanzanian border — a "
            "900-fold range. These are the communities converted along the 19th-century "
            "trade routes from Kilwa, and they sit on the southern lakeshore in one piece. "
            "**Traditional religion is Dedza.** 6.13% there against 1.06% nationally: "
            "one district holds more than a quarter of every traditionalist NSO counted. "
            "Read the national figure as a floor — the box is exclusive of the Christian "
            "and Muslim ones, and Chewa Nyau practice commonly accompanies church "
            "membership rather than replacing it. "
            "**And the largest cell in the country is a residual.** *Other Christian "
            "Denominations* is 26.6%, more than the Catholics, and it runs from 48.3% in "
            "Nkhata Bay to 5.5% on Likoma. Most of it is Malawi's very large independent "
            "and Zion church sector, which the census does not name."),
        how="census, 2018",
        grain="districts, 549,000 people on average",
        counts=_mw_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "mw" / "mw_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_mw_place_weight,
        note="THE WHOLE CENSUS IS DRAWN AND THERE IS NO GAP OF ANY KIND, which is rare "
             "here. NSO publishes no `not stated` cell for religion and no residual row: "
             "the ten denominations sum to 17,563,749 exactly on all 36 printed rows, and "
             "that figure is the full 2018 census count. Every category except the universe "
             "total resolves to a node, so 100% of the table is drawn. "
             "THE COUNTS ARE DISTRICT AND THE PLACEMENT IS FINE, and the two must not be "
             "confused. Table E5 is the only religion table below the national one anywhere "
             "in the 311-page report and the 32 district reports carry none, so district is "
             "NSO's ceiling rather than a choice made here. 32 units for 17.56M people is "
             "~549,000 each — finer per head than Kenya's counties. "
             "THE FOUR CITIES ARE PEERS OF THEIR DISTRICTS, NOT PARTS OF THEM. Mzuzu, "
             "Lilongwe, Zomba and Blantyre Cities are 2,115,867 people, 12.0% of the "
             "country, printed beside Mzimba, Lilongwe, Zomba and Blantyre rather than "
             "inside them; sources/mw.py proves it by checking that each region's districts "
             "sum to the region row on all eleven columns, and COD-AB's ADM2 carries the "
             "same 32 units in the same 7/10/15 split. "
             "The dots are spread across 75,802 Kontur 400m hexagons weighted by hex "
             "population (sources/mw_grid.py), and Malawi's reason for needing that is "
             "WATER rather than emptiness: Lake Malawi is 29,600 km² and the district "
             "boundaries run into the middle of it, so an equal share per polygon would "
             "draw a fifth of the country onto open lake. A population grid has no hexes "
             "there, so §8.2c's problem does not arise instead of being patched. "
             "ONE CATEGORY IS A MERGE OF THREE TRADITIONS. SDA/Baptist/Apostolic is "
             "1,644,829 people, 9.4%, and the tree keeps Adventists, Baptists and the "
             "African Apostolic churches in three different places. It is drawn as Other "
             "Christian, because the census does not say which of the three anyone belongs "
             "to and no split is inferred; taxonomy/mw2018.py has the argument. "
             "AND `Other Denomination` IS PARTLY KNOWN AND NOT SEPARABLE: Table 3.4 splits "
             "the same national figure into Buddhism 5,506, Hinduism 3,211 and other "
             "non-Christian 983,587, and no table in the report gives any of the three a "
             "geography. Those 8,717 Buddhists and Hindus are drawn inside `other.mw`.",
    ),
}
