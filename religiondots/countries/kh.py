# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _kh_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read.

    25 provinces for 15.55M people, and wildly uneven: Phnom Penh is 2.28M in 679 km2 while
    Mondul Kiri is 92,213 in 14,288 km2 of the Eastern Highlands. An equal share would wash
    the empty north-east and squash a seventh of the country into one speck — and the
    north-east is exactly where the only substantial non-Buddhist colour on Cambodia's map
    is, so the wash would put it across 25,000 km2 of forest. It also handles the Tonle
    Sap, which runs from ~2,700 to ~16,000 km2 and sits inside five provinces
    (sources/kh_grid.py).
    """
    return _kontur_place_weight(place, "kh_hexes.gpkg", "sources/kh_grid.py")


def _kh_counts():
    """NIS 2019 GPCC Table 2.5.1 at province: 4 drawn categories on 25 provinces.

    ONE level, no allocation, nothing modelled — every row is `measured` and may ring.

    THE COUNTS ARE DERIVED FROM TWO PUBLISHED TABLES AND NEITHER OF THEM HOLDS A COUNT.
    NIS publishes religion as PERCENTAGES to one decimal place (Table 2.5.1) and province
    populations as counts (Table 2.1.1); `sources/kh.py` multiplies them and apportions by
    largest remainder so a province's four categories sum to its published population
    exactly. That is arithmetic on two published figures rather than an estimate, so the
    rows stay `measured` — but every cell carries a rounding band of +/-0.0005 x the
    province population, which is +/-1,141 people in Phnom Penh. See sources/kh.md §3.

    FIFTEEN OF THE HUNDRED CELLS ARE ZERO, and all fifteen are `Other` in provinces where
    NIS prints `0.0`. They are dropped by the `count > 0` filter below and draw nothing,
    which is right: zero is the published figure. It is not evidence that nobody is there
    — a `0.0` is anything under 0.05% — and §3.5 says to mark that rather than fill it.
    """
    from kh2019 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "kh.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    df = df[df["geo_level"] == "province"].copy()

    lut = pd.read_csv(HERE / "data" / "geo" / "kh" / "kh_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    missing = sorted(df.loc[df["unit"].isna(), "geo_id"].unique())
    if missing:
        raise SystemExit(f"kh.csv provinces with no polygon: {missing} -- re-run "
                         "sources/kh_geo.py, the lookup is stale")
    if df["unit"].nunique() != 25:
        raise SystemExit(f"{df['unit'].nunique()} provinces, expected 25")

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations"]]


ENTRY = {
    "kh": dict(
        name="Cambodia",
        source="General Population Census 2019, Tables 2.5.1 and 2.1.1 (NIS)",
        basis="self-identification, whole census population",
        view=[102.2, 9.9, 107.8, 14.8],
        note_public=(
            "**Cambodia is 97.1% Buddhist and the entire map is in the remaining 3%**, "
            "which sits in two places that have nothing to do with each other. "
            "**The Cham Muslim belt follows the water.** Tbong Khmum is 11.8% Muslim — "
            "91,667 people, the largest share and the largest number in the country — and "
            "behind it come Kratie at 6.6%, Kampong Chhnang at 5.8%, Stung Treng at 4.7% "
            "and Koh Kong at 4.6%. That is the Mekong upstream of Phnom Penh and the shore "
            "of the Tonle Sap, and it is where the Cham have lived since they arrived from "
            "the Champa kingdom of central Vietnam from the fifteenth century onward. "
            "Against 0.1% in Svay Rieng and Kampong Speu, a hundred kilometres away. "
            "**Read the Muslim figure as a floor, for a specific reason**: the Cham were "
            "singled out for destruction under Democratic Kampuchea and are estimated to "
            "have lost between a third and a half of their people between 1975 and 1979. "
            "**The north-eastern highlands are the other half of the map, and the census "
            "will not name what is there.** Ratanak Kiri is 23.2% *other religion* and "
            "Mondul Kiri 21.2%, while fifteen of the twenty-five provinces are printed as "
            "0.0%. It is the sharpest such geography anywhere on this map. NIS says in the "
            "paragraph above its own table what the cell mostly is — *the local religious "
            "system of the highland tribal groups* — which is the animist tradition of the "
            "Bunong, Tampuan, Jarai, Kreung, Brao and Kavet: spirit forests, buffalo "
            "sacrifice, ancestor practice. It gets no box of its own on the form, so this "
            "map can show that a fifth of two provinces answers none of the three named "
            "religions and cannot show what they answer instead. "
            "**Christianity is 0.32% of Cambodia and its highest shares are in those same "
            "two provinces** — Mondul Kiri 4.0%, Ratanak Kiri 2.1%, twelve and six times "
            "the national rate — which is evangelical mission among the same highland "
            "peoples. Phnom Penh has more Christians in absolute terms and a much lower "
            "share. The two categories are working on the same population and the map "
            "shows both at once. "
            "**Four categories is all the census offers.** There is no cell for the "
            "Buddhist school, none for the branch of Islam, and none naming a single "
            "Christian body, so nothing on this map distinguishes the Mahanikay from the "
            "Thommayut, the mainstream Sunni majority from the Kan Imam San of Udong, or a "
            "Catholic from a Pentecostal."),
        how="census, 2019",
        grain="provinces, 622,000 people on average",
        counts=_kh_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "kh" / "kh_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_kh_place_weight,
        note="THE SOURCE PUBLISHES PERCENTAGES AND NOT COUNTS, WHICH IS WHY TWO TABLES ARE "
             "READ. Table 2.5.1 gives each province's religious composition to one decimal "
             "place and no absolute figure anywhere; Table 2.1.1, nine pages earlier, gives "
             "each province's population. The counts here are the product, apportioned by "
             "largest remainder so a province's four categories sum to its published "
             "population exactly. That is arithmetic on two published figures rather than "
             "an estimate, so the rows are `measured` — but every cell carries a rounding "
             "band of ±0.0005 × the province population, ±1,141 people in Phnom Penh and "
             "±21 in Kep. "
             "FIFTEEN OF THE HUNDRED CELLS ARE PRINTED AS `0.0` AND DRAW NOTHING. All "
             "fifteen are `Other`. Zero is the published figure and is drawn as zero; it is "
             "not evidence that nobody is there, since `0.0` is anything under 0.05% — "
             "§3.5, marked and not filled. "
             "25 PROVINCES FOR 15.55M IS NIS's CEILING RATHER THAN A CHOICE. Table 2.5.1 is "
             "the only religion table in the 304-page report, and the 2008 census's own "
             "priority-table list gives `A4 Population by Religion` an age and a sex "
             "dimension and no geography at all. Spec §3.9b is why it is drawn anyway — "
             "there is no minimum unit count. READ IT AS COMPOSITION, NEVER AS LOCATION: a "
             "cluster of dots says 'this province, drawn where Cambodians live' and nothing "
             "about which village. "
             "THE UNIVERSE EXCLUDES CAMBODIANS WORKING ABROAD, which both tables state in a "
             "footnote. That is a large population — several hundred thousand in Thailand "
             "alone — and it is why the census total is 15,552,211 rather than the ~16.5M a "
             "projection would give. "
             "THE FOUR CATEGORIES SUM TO 100% AND THERE IS NO `not stated` CELL, so the "
             "whole of that universe is drawn. "
             "THE JOIN IS CHECKED ON A KEY THE PAIRING DOES NOT USE. Provinces are paired "
             "to OCHA's polygons by a Khmer-romanisation fold (three of the twenty-five "
             "names differ — Otdar/Oddar Meanchey, Siem Reap/Siemreap, Tbong/Tboung Khmum), "
             "and then the census's PRINT POSITION is checked against COD's own "
             "`ADM1_PCODE`, which agrees on all 25. The two have different origins, so a "
             "transposed row would break it while every total still reconciled. "
             "The dots are spread across 77,453 Kontur 400m hexagons weighted by hex "
             "population (sources/kh_grid.py), which matters because the north-east is both "
             "the emptiest part of the country and the only part with a distinctive "
             "religious mix. It also removes the Tonle Sap, which runs from ~2,700 to "
             "~16,000 km² and lies inside five provinces. "
             "KONTUR MODELS 4.6× TOO MANY PEOPLE IN PAILIN — 374,607 against a census "
             "75,112, where every other province sits between 0.62× and 1.72×. The join was "
             "cleared four ways (the polygons tile with no overlap, COD's own areas match "
             "the geometry, Pailin's hexes are inside Pailin's bounding box, and the "
             "population is spread over 779 hexes rather than spiking), so this is the "
             "model's error and not the map's. It changes no count — the grid is a "
             "within-unit weight — but Pailin's dots sit on the least trustworthy placement "
             "surface in the country.",
    ),
}
