# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _zw_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read.

    Zimbabwe is Kenya's placement problem in its purest form: 10 provinces for 15.2M people
    is the coarsest counting geography on this map, and the provinces are wildly uneven.
    Matabeleland North and South are 129,197 km² between them — a third of the country,
    much of it Hwange, the Zambezi escarpment and dry ranching land — while Harare and
    Bulawayo are 872 km² and 479 km² holding 3.1M people. An equal share would wash the
    empty west and squash a fifth of the country into two specks. It also handles Lake
    Kariba, 5,580 km² of which sits inside the provinces (sources/zw_grid.py).
    """
    return _kontur_place_weight(place, "zw_hexes.gpkg", "sources/zw_grid.py")


def _zw_counts():
    """ZIMSTAT 2022 PHC Table 2.14(c) at province: 11 drawn categories on 10 provinces.

    ONE level, no allocation, nothing modelled — every row is `measured` and may ring.

    10 UNITS FOR 15.2M PEOPLE IS ~1.5M EACH, THE COARSEST COUNTING GEOGRAPHY ON THIS MAP,
    and it is ZIMSTAT's ceiling rather than a choice: Table 2.14 is the only religion table
    in the 259-page report, and religion got none of the five 2022 PHC thematic reports.
    Spec §3.9b is why it is drawn anyway — there is no minimum unit count; take the finest
    geography published, and say what the country therefore cannot show.

    ONE CATEGORY RESOLVES TO NOTHING and it is the universe row, so the drawn population is
    the whole table: **15,178,957, the entire census count.** The eleven categories sum to
    the province total on all ten rows exactly — no `not stated`, no residual, no §3.5 gap.

    THE NO-RELIGION CATEGORY IS THE LITERAL STRING `None` AND PANDAS WILL DELETE IT. §12's
    Philippine trap, third sighting after `ph` and `gy`: default `read_csv` parsing turns
    that cell into NaN, it then resolves to nothing in the taxonomy, and 1,255,578 people —
    8.3% of Zimbabwe, the category a religion map most needs to be honest about — vanish
    with no error and no count anywhere. `keep_default_na=False, na_values=[""]` is
    load-bearing on this country, not boilerplate.
    """
    from zw2022 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "zw.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    df = df[df["geo_level"] == "province"].copy()
    if "None" not in set(df["source_category"]):
        raise SystemExit("zw.csv has no `None` category -- it has been read as NaN, and "
                         "1.26M people are about to disappear (§12, the Philippine trap)")

    lut = pd.read_csv(HERE / "data" / "geo" / "zw" / "zw_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    missing = sorted(df.loc[df["unit"].isna(), "geo_id"].unique())
    if missing:
        raise SystemExit(f"zw.csv provinces with no polygon: {missing} -- re-run "
                         "sources/zw_geo.py, the lookup is stale")
    if df["unit"].nunique() != 10:
        raise SystemExit(f"{df['unit'].nunique()} provinces, expected 10")

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations"]]


ENTRY = {
    "zw": dict(
        name="Zimbabwe",
        source="2022 Population and Housing Census Report, Table 2.14 (ZIMSTAT)",
        basis="self-identification, whole census population",
        view=[24.8, -22.6, 33.4, -15.4],
        note_public=(
            "**Two of every five Zimbabweans belong to an Apostolic church, and no other "
            "source on this map counts them at all.** The Vapostori are the white-robed "
            "prophetic churches founded in the 1930s by Johane Masowe and Johane Marange — "
            "worshipping in the open air rather than in buildings, with prophecy and faith "
            "healing at the centre, and a deliberate break from both the mission churches "
            "and the ancestral religion. At 40.3% they are the largest single religious "
            "answer in the country, larger than every other Christian category put "
            "together, and they nearly double the size of this map's African Instituted "
            "Churches. ZIMSTAT gives them one cell, so the dozens of separate Apostolic "
            "bodies cannot be told apart here. "
            "**The country splits into two Christianities and the split is town against "
            "country.** The Apostolic churches are the Shona north and east — Mashonaland "
            "Central 53%, Manicaland 50% — and fall to 21% in Bulawayo and 27% in Harare. "
            "The Pentecostals are the exact inverse: 29% in Harare and 25% in Bulawayo "
            "against 11% in Mashonaland Central. Between them they are most of Zimbabwe. "
            "**Matabeleland is the different half of the country in almost every "
            "category.** It has the lowest Apostolic share, the highest *other Christian* "
            "at 13-16% — the Brethren in Christ and Adventist mission field — and the "
            "highest *no religion*, 13.5% in Matabeleland South against 4.5% in "
            "Manicaland. Note that no-religion here is NOT an urban figure: both cities are "
            "below the national rate. "
            "**African traditional religion is 5.0%, and read that as a floor.** The box is "
            "exclusive of the Christian ones, and Shona and Ndebele practice — the "
            "ancestral *midzimu*, the Mwari shrines of the Matobo hills — commonly "
            "accompanies church membership rather than replacing it. Zimbabwe is a sharper "
            "case than most, because the Apostolic churches themselves grew out of that "
            "overlap. "
            "**And the 6,845 people recorded as Jewish are probably not who you would "
            "assume.** Zimbabwe's historic Ashkenazi community has largely emigrated and "
            "numbers in the hundreds; this figure is ten times larger and peaks in rural "
            "Midlands, Masvingo and Manicaland rather than in the two cities. It most "
            "likely counts the Lemba, who claim Judaic descent and keep dietary and "
            "circumcision laws. The census does not say."),
        how="census, 2022",
        grain="provinces, 1.5 million people on average",
        counts=_zw_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "zw" / "zw_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_zw_place_weight,
        note="THE COARSEST COUNTING GEOGRAPHY ON THIS MAP — 10 provinces for 15,178,957 "
             "people, ~1.5M each, against Kenya's 47 counties at 1.0M. It is ZIMSTAT's "
             "ceiling and not a choice made here: Table 2.14 is the only religion table in "
             "the 259-page 2022 PHC report, ZIMSTAT published five thematic reports and "
             "religion got none of them, and the district and ward files carry population "
             "only. Spec §3.9b is why it is drawn anyway — there is no minimum unit count. "
             "READ IT AS COMPOSITION, NEVER AS LOCATION. A province is a sixth of a "
             "country here, so a cluster of dots says 'this province, drawn where "
             "Zimbabweans live' and nothing at all about which town. "
             "THE WHOLE CENSUS IS DRAWN AND THERE IS NO GAP OF ANY KIND. The eleven "
             "categories sum to the province total on all ten rows and to 15,178,957 "
             "nationally — no `not stated`, no residual, no §3.5 gap. The third source "
             "here of which that is true, after Malawi and Guyana. "
             "THE `Male + Female == Total` CHECK IS WHAT VERIFIES THE READ. Tables 2.14(a) "
             "and (b) are the sex breakdowns and only (c) is drawn; the other two are "
             "parsed anyway because every other identity in this table reconciles inside "
             "one table whichever way its columns were taken, and this one does not. All "
             "132 cells agree. "
             "THE NO-RELIGION CELL IS THE LITERAL STRING `None` and pandas deletes it "
             "without the `keep_default_na=False` flags — §12's Philippine trap, third "
             "sighting, and it would silently remove 1,255,578 people. `_zw_counts` asserts "
             "the category is present rather than trusting the flags. "
             "The dots are spread across 245,701 Kontur 400m hexagons weighted by hex "
             "population (sources/zw_grid.py), which matters more here than almost "
             "anywhere: Matabeleland is a third of the country, holds 1.59M people, and has "
             "the least typical religious mix in Zimbabwe, so an equal share per polygon "
             "would paint that mix across empty bush. It also removes Lake Kariba, 5,580 "
             "km² of which is inside the provinces. "
             "AND ZIMBABWE IS THE MIRROR OF BENIN ON HOW THE JOIN IS CHECKED: here the "
             "Kontur/census ratio band is tight and discriminating (0.83-1.11 across ten "
             "very uneven provinces; a shuffle fails a median 4 of 10) while the "
             "correlation is weak, because ten similar log-populations correlate by luck. "
             "Both are measured and sources/zw_grid.py says which one is carrying it.",
    ),
}
