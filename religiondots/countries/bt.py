# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _bt_place_weight(place):
    """countries.py hook. `place` is the 400 m hex layer scatter.py has read.

    Gasa, Bumthang, Lhuentse and Wangdue Phodrang are a third of Bhutan and hold 11% of its people.
    Kontur's hexes are calibrated to the 2017 census's 20 dzongkhags, with the hexes across the
    border in India dropped (Jaigaon, beside Phuentsholing; sources/bt_grid.py).
    """
    return _kontur_place_weight(place, "bt_hexes.gpkg", "sources/bt_grid.py")


def _bt_counts():
    """Bhutanese at the Centre for Bhutan Studies' 2010 shares, Hindus placed by Nepali mother
    tongue; non-Bhutanese by nationality: 20 dzongkhags.

    EVERY ROW IS `modelled` (§7b). The census asked religion in 2005 and never published it; the
    Gross National Happiness surveys ask and publish the national figure only. Both halves are the
    2017 census's counts per dzongkhag, so they partition each unit. sources/bt.py, sources/bt.md,
    ask 046.
    """
    from bt2010 import resolve

    nat = pd.read_csv(HERE / "data" / "normalized" / "bt.csv", dtype={"geo_id": str},
                      keep_default_na=False, na_values=[""])
    nat["node"] = nat["source_category"].map(resolve)
    unmapped = sorted(nat.loc[nat["node"].isna(), "source_category"].unique())
    if unmapped:
        raise SystemExit(f"bt.csv categories with no node: {unmapped}")
    ext = pd.read_csv(HERE / "data" / "normalized" / "bt_foreign.csv", dtype={"geo_id": str})
    df = pd.concat([nat[["geo_id", "node", "count"]], ext[["geo_id", "node", "count"]]],
                   ignore_index=True)
    df["unit"] = df["geo_id"]
    if df["unit"].nunique() != 20:
        raise SystemExit(f"{df['unit'].nunique()} dzongkhags, expected 20")
    df = df[df["count"] > 0].copy()
    df["tier"] = "modelled"
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations", "tier"]]


ENTRY = {
    "bt": dict(
        name="Bhutan",
        source="The Centre for Bhutan & GNH Studies' Gross National Happiness surveys (the 2010 "
               "estimate, and mother tongue by dzongkhag from 2015); the 2017 census count of "
               "Bhutanese and non-Bhutanese in each dzongkhag (National Statistics Bureau, PHCB "
               "2017), with non-Bhutanese' nationality through the United Nations' International "
               "Migrant Stock 2020 and Pew Research Center's 2020 estimates",
        basis="self-identification, Bhutanese aged 15 and over; foreign residents by nationality",
        note_public=(
            "**Bhutan's census asked everyone their religion in 2005 and has not published the "
            "answers.** The 2017 census did not ask. The figures here come from the Gross National "
            "Happiness surveys of the Centre for Bhutan & GNH Studies, which ask Bhutanese aged 15 "
            "and over their religion and publish the result for the whole country only. The "
            "Centre's estimate from its 2010 survey is **81%** Buddhist, **18%** Hindu and "
            "**1.2%** Christian. Its 2015 and 2022 reports print only the raw sample, 12.2% Hindu "
            "in 2022; in 2010 the raw sample was 13.1% Hindu and the weighted estimate 18%. Those "
            "shares are applied to the **681,720** Bhutanese the 2017 census counted, dzongkhag "
            "by dzongkhag. Nobody counted these dots, so they disappear when inferred dots are "
            "turned off. "
            "**Where the Hindus are drawn comes from language, not from religion.** Bhutan's "
            "Hindus are mostly Lhotshampa, the Nepali-speaking people of the southern foothills, "
            "and the Centre's 2015 survey published the share of each dzongkhag whose mother "
            "tongue is Nepali: 56% in Samtse, 44% in Tsirang, 40% in Sarpang, 39% in Chhukha, 14% "
            "in Thimphu and under 1% in Trashigang. The Hindus are spread in proportion to that, "
            "so their total is the 2010 estimate and only their location comes from language. No "
            "source counts religion and language together, so this assumes that Nepali speakers "
            "are Hindu and nobody else is. Buddhist Tamang and Gurung families, and Hindus with "
            "another mother tongue, are drawn in the wrong place, and nothing can check it "
            "dzongkhag by dzongkhag: treat the Hindus' location as the weakest thing on the map. "
            "Christians, whom no source places, are drawn at 1.2% in every dzongkhag. "
            "**Non-Bhutanese are drawn by nationality.** The census counted **45,425** "
            "non-Bhutanese living in the country, most of them in Thimphu and at the hydropower "
            "projects in Wangdue Phodrang, Chhukha and Trongsa, but tabulated no nationality. Of "
            "the migrants whose origin the United Nations' 2020 count names, 95% are from India, "
            "and each origin is drawn at Pew Research Center's 2020 estimate for that country, "
            "which puts 34,778 of them on Hinduism and 6,689 on Islam. Tourists and the day "
            "workers who cross from India each morning are not on the map. All together the map "
            "draws 21.7% of the people living in Bhutan as Hindu; Pew's own estimate for the "
            "country is 22.5%."),
        how="survey, national shares; Hindus placed by mother tongue, foreign residents by nationality",
        grain="dzongkhags, 36,000 people on average",
        counts=_bt_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "bt" / "bt_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_bt_place_weight,
        note="REOPENED ON ANITA'S PRIORITY (ask/RULINGS.md 2026-09-15) AND THE MAURITANIA RULING "
             "(2026-09-16). sources/bt.md is the record; ask 046 (spec §14: Lhotshampa). NOBODY "
             "PUBLISHES IT: PHCB 2005 asked religion of everyone (IHSN 1374 DDI q4ca; Buddhism, "
             "Hinduism, other) and tabulated none; microdata on signed application only; PHCB 2017 "
             "did not ask. LEVEL: GNH 2010 weighted, 81/18/1.2 (Extensive Analysis of GNH Index, "
             "2012, p.153); Buddhist is the remainder, 80.8. PLACEMENT: Hindus by GNH 2015 Table "
             "A1.5 Nepali mother tongue (weighted, by dzongkhag), hindu_d = 0.18 N_d L_d / L; "
             "Christians flat; HINDU_PLACEMENT in sources/bt.py switches to one national share. No "
             "check by dzongkhag exists (spec §14.12 condition 3). FOREIGNERS: 45,425 by dzongkhag "
             "(PHCB 2017 Table 2.8), UN DESA 2020 named origins (India 94.9%) through Pew 2020, "
             "islam.* folded to islam. GEOGRAPHY: COD-AB v01 20 dzongkhags; Table A2.8 swaps "
             "Lhuentse's and Monggar's areas. PLACEMENT: Kontur BT calibrated to dzongkhag, 116 "
             "hexes in India dropped (Jaigaon). Drawn Hindu 21.66% of residents against Pew 2020's "
             "22.51%.",
    ),
}
