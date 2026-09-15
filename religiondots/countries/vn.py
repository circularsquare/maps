# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _vn_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read."""
    return _kontur_place_weight(place, "vn_grid_400m.gpkg", "sources/vn_geo.py")


def _vn_counts():
    """Vietnam 2009 census at province: 13 nodes on 63 units.

    ONE level, no allocation, nothing modelled — every row is `measured`.

    **100% OF THE CENSUS POPULATION IS DRAWN, AND FOUR FIFTHS OF IT LANDS ON ONE NODE.**
    Biểu 7 counts only people who reported belonging to a state-recognised religious
    organisation — 15.65M — and has no row at any geography for the other 70.2M. sources/vn.py
    computes that row as each province's Biểu 1 population minus its Biểu 7 total, which is
    the complement of a published partition and reconciles to the person, and it resolves to
    **`unknown`**: a node whose entire content is that these people were counted and the
    source does not say what they practise. Anita's call 2026-09-06, and spec §14.7's decision
    for China taken first for Vietnam. Drawing them as `unaffiliated` would be the largest
    false claim available on this map; leaving them out left the country reading as empty,
    which §6.12 could label and not fix.

    vn.csv also carries the country rows for 2009 AND the 2019 national table, which has no
    geography at all; only `province` is read, so neither is drawn.
    """
    from vn2009 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "vn.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    df = df[df["geo_level"] == "province"].copy()
    df["count"] = df["count"].astype(int)

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df = df.rename(columns={"geo_id": "unit"})
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations"]]


ENTRY = {
    "vn": dict(
        name="Vietnam",
        source="Population and Housing Census 2009, Biểu 7 (General Statistics Office)",
        basis="self-identification",
        view=[101.5, 8.0, 110.5, 23.6],
        note_public=(
            "**Four fifths of this map is one colour, and that colour means we do not "
            "know.** Vietnam's census asks which of the state-recognised religious "
            "organisations a person belongs to, and **81.8% of the country belongs to "
            "none of them**. That is not irreligion, which is why those people are drawn "
            "as *Religion unknown* rather than as no religion. Ancestor veneration is "
            "close to universal, the village đình and the mother-goddess rites of đạo Mẫu "
            "are everywhere, and the great majority of people who would call themselves "
            "Buddhist in conversation are in that grey rather than in the Buddhist figure "
            "below — the census counts 6.8 million Buddhists in 2009 in a country usually "
            "described as around 45% Buddhist by practice. Read the coloured dots as "
            "**registered religion** and the grey as a question the census did not ask. "
            "**Somebody has since measured what is inside the grey.** Pew surveyed Vietnam "
            "directly in 2024 and found that **92%** of religiously unaffiliated Vietnamese "
            "adults had made an offering to their ancestors in the past year, that **95%** "
            "of all adults keep an altar at home, with Buddhists, Christians and the "
            "unaffiliated keeping them at much the same rate, and that **96%** had burned "
            "incense in a veneration ritual. Asked their religion in the ordinary way "
            "rather than asked which registered organisation they belong to, **48%** of "
            "Vietnamese say none and **38%** say Buddhist, against this census's 18.2% in "
            "any religion at all. So the grey is neither irreligion nor one hidden faith. "
            "It is a country where almost everyone practises, and where what counts as "
            "having a religion is not what the census form means by it. "
            "**What is drawn is intensely regional, far more so than in any other country "
            "here.** Six provinces are over 45% affiliated and six are under 2%: **An Giang "
            "is 94.5%** and Sơn La is 0.4%. Almost nothing about Vietnamese religion is "
            "evenly spread. "
            "**An Giang is the reason.** The Mekong Delta produced its own Buddhist "
            "movements in the nineteenth and twentieth centuries and the census counts four "
            "of them separately: Bửu Sơn Kỳ Hương (1849), Tứ Ân Hiếu Nghĩa (1867), Hòa Hảo "
            "(1939) and Hiếu Nghĩa Tà Lơn. **An Giang alone is 43.7% Hòa Hảo** — 936,974 "
            "people, 65% of all Hòa Hảo in Vietnam — plus 84% of the country's Tứ Ân Hiếu "
            "Nghĩa and 76% of its Bửu Sơn Kỳ Hương. These are lay movements with no clergy "
            "and no temples, and they exist essentially in one province and its neighbours: "
            "Cần Thơ is 19.1% Hòa Hảo and Đồng Tháp 11.8%. "
            "**Tây Ninh is the other one-province religion.** Caodaism was founded there in "
            "1926, its Holy See is there, and the province is **35.6% Caodaist** — 47% of "
            "all Caodaists in the country. "
            "**Catholicism has two homelands and they are 1,500 km apart.** Đồng Nai is "
            "32.1% Catholic and Nam Định in the Red River Delta holds 369,793 — the "
            "seventeenth-century Jesuit mission field, and the place most of the southern "
            "Catholics came from when nearly a million people moved south in 1954. "
            "**The Central Highlands are the missionary map.** Kon Tum is 31.2% Catholic, "
            "Lâm Đồng 25.6%, Đắk Nông 20.5% — and the Protestant share peaks in the same "
            "provinces (Đắk Nông 10.3%, Gia Lai and Đắk Lắk 8.6%) rather than anywhere Kinh "
            "Vietnamese live. Protestantism here is Montagnard, and in Điện Biên (7.5%) and "
            "Lai Châu (7.1%) it is Hmong. It is also the only category that grew between the "
            "two censuses, by 31%. "
            "**Trà Vinh is 49.7% Buddhist and its Buddhism is not the same religion as Ho "
            "Chi Minh City's.** The delta provinces of Trà Vinh and Sóc Trăng are Khmer "
            "Krom, and Khmer Buddhism is Theravada; the northern and urban Buddhism is "
            "Mahayana. The census offers one box marked Buddhist and the map cannot show "
            "the difference. "
            "**Ninh Thuận is the Cham province.** 7.2% Cham Balamon — the last living Hindu "
            "tradition descended from the Indianised kingdoms of Southeast Asia — and 4.5% "
            "Muslim, and those Muslims are mostly **Bani**, a thousand-year-old Cham "
            "localisation of Islam that the census does not distinguish from the Sunni Cham "
            "of An Giang and Saigon. Two halves of one Cham religious system, and only one "
            "of them has a colour here."),
        how="census, 2009, state-recognised organisations only",
        grain="provinces, 1.4 million people on average",
        counts=_vn_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "vn" / "vn_grid_400m.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_vn_place_weight,
        note="**THE 70.2 MILLION IN GREY ARE COMPUTED, NOT PRINTED.** Biểu 7's universe is "
             "people with a religion, so the census publishes no row anywhere for anyone "
             "else; each province's *Religion unknown* figure is its published population "
             "minus its published religious total, which is the complement of a partition "
             "rather than an estimate — the religions, the not-stated cell and the residual "
             "sum to the population in all 63 provinces to the person. The 2019 census "
             "publishes the same category directly, nationally, at 83,046,105. "
             "**THE 2019 CENSUS IS BETTER AND HAS NO GEOGRAPHY, SO 2009 IS DRAWN.** Both "
             "censuses ask about religion. The 2019 volume publishes the answer on **one "
             "page, nationally**, with no province table anywhere, while giving ethnicity a "
             "full province × 54-group tabulation over 167 pages; the 2009 volume publishes "
             "religion by province over 32 pages. So the newer census is the better "
             "measurement of how many and says nothing about where, and this map is the "
             "older one drawn as it stands, at its own year (spec §3.4, as India 2011 and "
             "Russia 2012 are). It is NOT rescaled to 2019 totals: that would mean one "
             "national factor per religion applied to all 63 provinces, and the factors are "
             "not credible as history — Buddhism −32.3%, Hòa Hảo −31.4% and Cao Đài −31.2% "
             "over the same decade in which the population grew 12%, three unrelated "
             "traditions moving together to within one percentage point. That is the "
             "instrument changing, and smearing it evenly over every province would assert "
             "something nobody measured. "
             "**Three bodies recognised after 2009 therefore have no geography at all** and "
             "are absent from the map though present in the data: the Seventh-day Adventists "
             "(11,830), the Latter-day Saints (4,281) and Hiếu Nghĩa Tà Lơn (401). "
             "**The table reconciles completely.** All 63 provinces sum to their own printed "
             "totals, the provinces sum to the national row in every one of the fourteen "
             "categories, and each of the six socio-economic regions sums to its own row — "
             "which is also what verifies the province-to-region composition, since the "
             "census prints regions and provinces as separate flat blocks with no marker of "
             "which belongs to which. Non-response is 30 people nationally, the smallest on "
             "this map by three orders of magnitude, and is reported rather than drawn. "
             "**Boundaries are geoBoundaries ADM1 pinned to a pre-2025 commit**, because "
             "Vietnam merged its 63 provinces into 34 on 1 July 2025 and a current file "
             "would be a different country from the one the census counted. The join is a "
             "hand-built bridge from GSO's administrative codes to ISO 3166-2:VN — two "
             "numeric-looking code spaces that agree on **no** province, so a naive join "
             "would have produced 63 silent wrong answers — and it is re-derived by name on "
             "every run, with Ho Chi Minh City forced by elimination as the only unmatched "
             "province against the only unclaimed code. The boundary file's 64th feature is "
             "Côn Đảo, an offshore district carrying its parent's ISO code, so dissolving on "
             "that code reassembles the province. "
             "Placement is Kontur's 400 m H3 grid, 231,746 hexes; every province's "
             "Kontur/census ratio lands between 0.73 and 1.88 with a national 1.14, which is "
             "the expected shape for a 2023 surface over a 2009 census in a country that "
             "grew 12% — and it is also what confirms the code bridge, since a scrambled one "
             "would pair Ho Chi Minh City's 7.2 million with Bắc Kạn's 294,000.",
    ),
}
