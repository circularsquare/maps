# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _zm_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read.

    156 constituencies over 752,000 km2, from Kabushi's 11 km2 of Ndola to Kasempa's 21,877.
    The big rural ones hold their people along the Zambezi, the Luapula and the line of rail,
    and drawn flat a constituency's colour would spread over the empty miombo between
    (sources/zm_grid.py).
    """
    return _kontur_place_weight(place, "zm_hexes.gpkg", "sources/zm_grid.py")


def _zm_counts():
    """ZamStats 2022 census, Series B religion volume: 156 constituencies, every row `derived`.

    WHAT WAS MEASURED AT THE CONSTITUENCY is two columns, Christianity and every non-Christian
    together, each split rural and urban (Table B.5). The religions (B.1) and the twenty-five
    Christian rows (B.6, B.9, B.10) are province tables, so sources/zm.py spreads them inside
    each constituency: its rural Christians take the province's rural denominational mix and
    its urban Christians the urban one, and its non-Christians take the province's B.1 mix.
    Every province x category re-aggregates to the printed table.

    `roll` names `Christianity` for every Christian row (taxonomy/zm2022.py COLUMNS), so
    `inferred dots: not shown` redraws the measured column. The non-Christian column has no
    node and no roll, deliberately; see the comment on COLUMNS. A derived row may not ring
    (§3.10).

    TWO B.1 COLUMNS ARE FILED BY THE OFFICE'S ANALYTICAL REPORT, NOT THEIR PRINTED HEADERS:
    `Judaism` is the traditional religion and `Other Religious Groups` is no religion. zm.py
    asserts the evidence on every build.
    """
    import zm2022
    from zm2022 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "zm.csv",
                     dtype={"geo_id": str}, low_memory=False)
    lut = pd.read_csv(HERE / "data" / "geo" / "zm" / "zm_lookup.csv", dtype=str)
    if set(df["geo_id"]) != set(lut["unit"]) or len(lut) != 156:
        raise SystemExit("zm.csv and zm_lookup.csv disagree on the 156 constituencies -- "
                         "re-run sources/zm_geo.py, then sources/zm.py")
    df["unit"] = df["geo_id"]
    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(df.loc[df["node"].isna(), "source_category"].unique())
    if unmapped:
        raise SystemExit(f"zm.csv categories with no node: {unmapped}")
    df = df[df["count"] > 0].copy()
    df = _add_roll(df, zm2022.COLUMNS)
    df["may_ring"] = df["tier"] == "measured"
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations", "tier", "roll", "may_ring"]]


ENTRY = {
    "zm": dict(
        name="Zambia",
        source="2022 Census of Population and Housing, Series B religion tables (Zambia "
               "Statistics Agency, 2026)",
        basis="self-identification, census population present on census night",
        note_public=(
            "**Zambia's 2022 census names 23 Christian churches and prints them only by "
            "province.** For each of the 156 constituencies it gives the number of Christians "
            "and of everyone else, rural and urban, and nothing finer. So a constituency's "
            "churches on this map are its province's mix, applied separately to its rural and "
            "its urban Christians, and its other religions are the province's mix too. The "
            "rural and urban mixes differ: Pentecostals are **24.7%** of urban Christians and "
            "11.7% of rural ones. "
            "**Most of the large churches are regional.** Seventh-day Adventists are **42.1%** "
            "of Southern Province and 3.3% of Eastern, the New Apostolic Church is **38.0%** "
            "of Western, and Catholics are 34.0% of Northern. The Reformed Church in Zambia is "
            "13.5% of Eastern, the Brethren of the CMML missions 14.9% of Luapula and the "
            "Evangelical Church in Zambia 13.9% of North-Western. "
            "**Two religion columns are drawn under different names from the ones printed.** "
            "The census table heads one `Judaism` (30,502 people, half of them in Eastern "
            "Province) and another `Other Religious Groups` (233,260). ZamStats' own national "
            "analytical report gives traditional religion as 0.2% of Zambia and no religion as "
            "1.3%, and only those two columns make those figures, so they are drawn as "
            "traditional religion and as no religion."),
        how="census, 2022",
        grain="constituencies, 118,000 people on average",
        fill="from the same census at province level, rural and urban separately",
        gap="1,353,080 people, 6.9% of the census count: the religion tables cover the "
            "18,340,343 present on census night, not the 19,693,423 usual residents",
        gap_share=0.0687,
        counts=_zm_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "zm" / "zm_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_zm_place_weight,
        note="ZAMSTATS' SERIES B RELIGION VOLUME (April 2026) is the first Zambian release to "
             "print denominations, and it prints them by province; its constituency tables "
             "split only Christianity from the rest, rural and urban. sources/zm.py spreads "
             "the province tables inside each constituency by residence and asserts every "
             "province cell re-aggregates. Table B.1's `Judaism` and `Other Religious Groups` "
             "columns are filed as traditional religion and no religion on the office's 2022 "
             "National Analytical Report, asserted on every build. Earlier negatives in "
             "sources.md 11p and 11w read the 2022 Analytical Report's chart and the 2010 "
             "oracle row. sources/zm.md has the rest.",
    ),
}
