# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _vu_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read.

    Vanuatu is 83 islands and its people are on the coasts: the interiors of Santo and
    Malekula are close to empty while the shore is not, and Torres alone is six islands.
    An equal share per polygon would put dots inland and offshore in whatever proportion
    the polygon's area suggests (sources/vu_grid.py).
    """
    return _kontur_place_weight(place, "vu_hexes.gpkg", "sources/vu_grid.py")


def _vu_counts():
    """VNSO 2020 census Table 3.5 at area council: 12 drawn categories on 66 units.

    ONE level, no allocation, nothing modelled -- every row is `measured` and may ring.

    THE TIER IS AREA COUNCILS AND IT COST NOTHING. Table 3.5's `Region` column is the whole
    census hierarchy, so the same fourteen categories are printed at Vanuatu, at urban and
    rural, at the six provinces AND at the 64 rural area councils below them. 66 drawable
    units -- the 64 councils plus Port Vila and Luganville -- averaging 4,500 people. Fiji
    had to choose between categories and geography (§9bd §3); Vanuatu publishes both in one
    table, so there is no trade-off to argue about.

    THE STRUCTURAL ROWS ARE NOT UNITS. VANUATU, URBAN, RURAL and the six provinces are
    totals of the rows beneath them; sources/vu.py writes only the 66 leaf units to
    vu.csv and asserts the hierarchy adds up before it does.

    THE PRINTED TOTAL IS NOT THE UNIVERSE HERE. VNSO rounds each cell on its own, so a
    row's fourteen categories can miss its printed `Total` by one or two people -- verified
    on the rendered page, so it is the source's arithmetic. The drawn categories are what
    this map counts, and `Total` is carried in vu.csv for reporting only.
    """
    from vu2020 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "vu.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    df = df[df["geo_level"] == "area_council"].copy()

    lut = pd.read_csv(HERE / "data" / "geo" / "vu" / "vu_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    missing = sorted(df.loc[df["unit"].isna(), "geo_id"].unique())
    if missing:
        raise SystemExit(f"vu.csv units with no polygon: {missing} -- re-run "
                         "sources/vu_geo.py, the lookup is stale")
    if df["unit"].nunique() != 66:
        raise SystemExit(f"{df['unit'].nunique()} area councils, expected 66")

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df = df.groupby(["unit", "node"], as_index=False)["count"].sum()
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations"]]


ENTRY = {
    "vu": dict(
        name="Vanuatu",
        source="2020 National Population and Housing Census, Basic Tables Vol 1, Table 3.5 "
               "(Vanuatu National Statistics Office)",
        basis="self-identification, population in private households",
        view=[166.0, -20.6, 170.8, -12.7],
        note_public=(
            "**No church in Vanuatu is anywhere near a majority, and the islands do not "
            "agree with each other at all.** The largest is the Presbyterian Church of "
            "Vanuatu at **27.2%**, and then four bodies between 12% and 15%: Seventh Day "
            "Adventist 14.8%, Catholic 12.1%, Anglican 12.0%. Nothing else in the Pacific "
            "on this map is that evenly split; Fiji's Methodists are 34.7%, and the "
            "Tuvaluan and Tongan national churches are higher again. "
            "**What the map shows is the mission spheres of the 1850s, still legible.** "
            "Anglican is **77.3% of Torba** and **44.9% of Penama** against **0.2% of "
            "Tafea**, because the Melanesian Mission worked the Banks and Torres islands "
            "and northern Pentecost and essentially nowhere else in the group. Presbyterian "
            "is "
            "43.9% of Malampa and 40.7% of Shefa, the centre and south, and 0.5% of Torba. "
            "The Churches of Christ are 16.1% of Penama and 0.9% of Malampa. Three missions "
            "divided the islands between them and the boundaries have barely moved since. "
            "**And Vanuatu counts customary belief as a religion, which almost nowhere "
            "does.** `Customary beliefs` is a printed census category with **9,080 people, "
            "3.1%**, counted in four consecutive censuses since 1989, and it is almost "
            "entirely one island. **17.3% of Tafea province, and on Tanna itself South West "
            "Tanna is 30.3%, Middle Bush Tanna 25.3% and North Tanna 19.1%.** That is where "
            "the John Frum movement is. It is the highest indigenous-religion share of any "
            "unit on this map outside India. "
            "**Two of these churches were founded in Vanuatu rather than sent to it**: Neil "
            "Thomas Ministries, 3.2% and larger at every census since 1999, and the "
            "Apostolic churches at 2.4%. "
            "**No religion is 1.4%**, and 428 people, 0.15%, refused the question or left "
            "it blank."),
        how="census, 2020, population in private households",
        grain="area councils, 4,500 people on average",
        counts=_vu_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "vu" / "vu_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_vu_place_weight,
        note="THE FINE TIER WAS FREE, WHICH IS WHY THIS IS 66 UNITS AND NOT 6. The queue had "
             "Vanuatu down as `6 provinces`; Table 3.5's `Region` column turns out to be the "
             "whole census hierarchy, printing the same fourteen categories at the nation, "
             "at urban/rural, at the six provinces AND at the 64 rural AREA COUNCILS "
             "underneath them. With Port Vila and Luganville that is 66 units averaging "
             "4,500 people. There is no trade-off of the kind Fiji had to make (§9bd §3): "
             "VNSO publishes the deep list and the fine geography in the same table. "
             "THE JOIN IS ON NAME AND MATCHES 66/66 OUTRIGHT, nothing spare either way and "
             "no aliases -- COD-AB's ADM2 is the census's own tier. Because a name join that "
             "matches everything is the one nobody checks, sources/vu_geo.py asserts what "
             "names cannot fake: OCHA's ADM1 for each polygon against the province Table 3.5 "
             "prints each council under. Two organisations, agreeing on all 64. "
             "sources/vu_grid.py then correlates census against Kontur at r=0.974 over 66 "
             "units, which none of 2,000 random pairings comes near (best 0.41). "
             "THE PUBLISHED TABLE ROUNDS EACH CELL SEPARATELY, so 41 of its 75 rows have "
             "their fourteen categories missing the printed Total by one or two people. "
             "Verified on the rendered page -- East Santo prints 5,788 against categories "
             "summing to 5,786 -- and against Volume 2, which reproduces every national "
             "figure exactly. The drawn categories are the universe; `Total` is carried for "
             "reporting only. "
             "`OTHER CHURCHES` IS 12% AND DOES NOT GO TO `christianity`. Volume 1 heads the "
             "column `Other churches`; Volume 2 heads the same column `Other` and says it "
             "`includes 88 different religions ranging from one member to more than 2,000 "
             "members`. Vanuatu's Baha'i, Muslim and Witness communities have nowhere else "
             "in this table to be, so the cell goes to `other.vu` rather than claiming "
             "35,270 people for Christianity on a header the other volume contradicts "
             "(§14.4). "
             "13% OF KONTUR'S PEOPLE FALL OUTSIDE EVERY COUNCIL AND ARE SNAPPED, NOT "
             "DROPPED: 99.9% are within 500 m of a boundary and the largest are 15-150 m "
             "out, coastal cells just seaward of a detailed coastline on a 400 m grid, "
             "clustered on Port Vila and Luganville. Dropping them would weight every "
             "shoreline light and pull the dots inland, which is a direction and not a "
             "rounding. 51 people, 0.015%, remain unplaced.",
        gap_share=0.0015,
        gap="Refuse to answer (394) and Not Stated (34) are §3.5 residuals and are not "
            "drawn: 428 people, 0.15%. Vanuatu is 99.85% drawn.",
    ),
}
