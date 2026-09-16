# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _kr_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read."""
    return _kontur_place_weight(place, "kr_grid_400m.gpkg", "sources/kr_geo.py")


def _kr_counts():
    """South Korea 2015 census, at si/gun/gu: 10 nodes on 229 units.

    ONE level, no allocation, nothing modelled. kr.csv also carries the country and the 17
    provinces, which are the same people twice more, and a `gu` level of 35 rows that is
    deliberately NOT drawn — KOSIS publishes the general gu of twelve large cities and no
    boundary set carries them, so the drawn tier is the 229 those cities belong to. See
    sources/kr.py.

    THE LAST TIME KOREA WAS ASKED. The religion question was dropped after 2015, so unlike
    every other country here this is not a vintage waiting to be superseded.
    """
    from kr2015 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "kr.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    df = df[df["geo_level"] == "sigungu"].copy()
    df["count"] = df["count"].astype(int)

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df = df.rename(columns={"geo_id": "unit"})
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations"]]


ENTRY = {
    "kr": dict(
        name="South Korea",
        # English rather than 인구총조사, unlike mk's Cyrillic `Попис 2021`: the panel is read
        # text and does not reflow, so a long source line overlaps the granularity line
        # beneath it. The table id is what actually identifies this source anyway.
        source="Population Census 2015 (KOSIS table DT_1PM1502)",
        # Kept short deliberately — the panel does not reflow, and a longer basis line
        # overlaps the granularity line beneath it.
        basis="self-identification, 20% census sample",
        view=[125.8, 33.0, 129.7, 38.7],
        note_public=(
            "**A majority of South Koreans report no religion at all** — 56.1%, the "
            "largest single answer in the country and the highest share of any country "
            "drawn here. What is left divides three ways: Protestant 19.7%, Buddhist "
            "15.5%, Catholic 7.9%. "
            "**And the Buddhist and Protestant halves are geographic opposites.** Buddhism "
            "is the south-east — Ulsan 29.8%, South Gyeongsang 29.4%, Busan 28.5%, Daegu "
            "23.8% — and thins to 8.6% in North Jeolla and 8.8% in Incheon. Protestantism "
            "runs the other way: North Jeolla 26.9%, Seoul 24.2%, South Jeolla 23.2%, "
            "against 10.5% in South Gyeongsang. That is the Yeongnam/Honam line, the "
            "deepest regional division in Korean politics, drawn here in religion. "
            "**Catholicism is Gangnam.** Seoul's Gangnam-gu is 16.3% Catholic and Seocho-gu "
            "16.1% — the wealthy districts south of the river — against a national 7.9%. "
            "Nowhere else in the country reaches those numbers, and the pattern is class "
            "rather than region. "
            "**Won Buddhism has a homeland and the map finds it exactly.** Sotaesan founded "
            "it in 1916 at Yeonggwang and put its headquarters at Iksan; 2015 counts Iksan "
            "at 4.07% and Yeonggwang at 3.91% against a national 0.17%, more than twenty "
            "times over. A movement of 84,141 people is legible as two bright spots on the "
            "county it started in. **Daesun Jinrihoe does the same at Yeoju** (0.86% "
            "against 0.08%), where its temple complex is. "
            "**Confucianism is counted as a religion here and almost nowhere else** — "
            "75,703 people, concentrated in the old lineage country of the south-west: "
            "Haenam 1.53%, Jangheung 1.36%. Read it as the institutional core around the "
            "hyanggyo, not as Confucian practice, which is near-universal and is not what "
            "anyone is reporting. "
            "**This is the last Korean census that asked.** The question was dropped after "
            "2015, so nothing here will be updated. "
            "**Foreign residents are not in the religion figures.** The 2015 census counted "
            "51,069,375 people, 1,363,712 of them foreign residents. The religion table counts "
            "49,052,389, fewer than the census's Korean nationals alone, and by its own footnote "
            "it leaves out special enumeration districts. The foreigners and the other 653,274 "
            "people missing, 3.9% of the census together, are in the not drawn part of the bar."),
        how="census, 2015, 20% sample",
        grain="si/gun/gu, 214,000 people on average",
        gap=("3.9% of the 2015 census population: 1,363,712 foreign residents (2.7%), who are "
             "outside the religion table; and 653,274 Korean nationals (1.3%), the difference "
             "between them and the table, which leaves out special enumeration districts"),
        gap_share=0.0395,
        counts=_kr_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "kr" / "kr_grid_400m.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_kr_place_weight,
        note="THE FIGURES ARE A 20% SAMPLE GROSSED UP. 2015 was a register-based census "
             "and religion rode on the sample survey rather than the register, so every "
             "cell carries sampling error — which matters most for the small categories "
             "this country is worth drawing for. Daejonggyo is 3,101 people nationally, so "
             "its district cells are a few sampled households each. "
             "The universe is 49,052,389 against a census population of 51,069,375, a gap "
             "of 3.95%. It is below the census's 49,705,663 Koreans alone, so the 1,363,712 "
             "foreign residents are outside it, and the other 653,274 sit with the special "
             "enumeration districts the table's footnote excludes; both are in gap and gap_share "
             "since the ask 033 sweep, not filled (spec §3.5, sources/kr.md §5 and §7). "
             "**KOSIS bot-blocks its own data endpoints**, so this table cannot be fetched "
             "by script — the metadata endpoint is open and the download endpoints answer "
             "200 with an HTML alert. Anita downloaded it through a browser on 2026-09-05; "
             "sources.md §11g has the click path. "
             "The drawn tier is 229 si/gun/gu. KOSIS publishes 252 at its second level, the "
             "difference being the general gu of twelve large cities, and no boundary set "
             "carries those — they are in kr.csv at a `gu` level, undrawn, against a source "
             "that ever ships them. Seoul's 25 gu are unaffected: those are full local "
             "governments and are drawn. "
             "The join is by NAME inside a province and never across the country, because "
             "Jung-gu, Dong-gu, Nam-gu, Seo-gu and Buk-gu each name five or six different "
             "districts nationally. Provinces bridge on ISO 3166-2:KR; districts are "
             "romanised from Hangul and matched on a deliberately loose fold, 1:1 or "
             "reported. **geoBoundaries omits an entire county** — Yeonggwang-gun, 53,984 "
             "people — and it is rebuilt from the eleven ADM3 eup and myeon that lie "
             "outside every ADM2 polygon, 481 km² against a published 475. That matters "
             "more than a missing rural county usually would: Yeonggwang is where Won "
             "Buddhism was founded and the second most Won Buddhist place on earth. "
             "Placement is Kontur's 400 m grid, 71,478 hexes; every unit's Kontur/census "
             "ratio lands between 0.30 and 3.5 with a national 1.044, which is what "
             "confirms the name join.",
    ),
}
