# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _af_place_weight(place):
    """countries.py hook. `place` is the 400 m hex layer scatter.py has read.

    Helmand, Herat, Kandahar, Farah, Nimroz and Ghor are 46% of Afghanistan and hold 19% of NSIA's
    settled people. Kontur's hexes are calibrated to NSIA's 34 province estimates, after 16 false
    blocks at Kontur's density cap are lowered; Kabul and Herat cities are left (sources/af_grid.py).
    """
    return _kontur_place_weight(place, "af_hexes.gpkg", "sources/af_grid.py")


def _af_counts():
    """NSIA's 1404 (2025-26) settled population, every person on Islam: 34 provinces.

    EVERY ROW IS `modelled` (§7b). Nothing asks Afghans their religion, so the whole settled estimate
    is drawn on Islam; the 1.5 million Kuchis have no province and are the gap. sources/af.py and
    sources/af.md.
    """
    from af2025 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "af.csv", dtype={"geo_id": str},
                     keep_default_na=False, na_values=[""])
    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(df.loc[df["node"].isna(), "source_category"].unique())
    if unmapped:
        raise SystemExit(f"af.csv categories with no node: {unmapped}")
    if df["geo_id"].nunique() != 34:
        raise SystemExit(f"{df['geo_id'].nunique()} provinces in af.csv, expected 34")
    df["unit"] = df["geo_id"]
    df = df[df["count"] > 0].copy()
    df["tier"] = "modelled"
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations", "tier"]]


ENTRY = {
    "af": dict(
        name="Afghanistan",
        source="No census or survey asks; the settled population of each province in the National "
               "Statistics and Information Authority's Estimated Population of Afghanistan 2025-26",
        basis="everyone drawn as Muslim, which nobody asked",
        note_public=(
            "**Nobody in Afghanistan is asked their religion.** There has been no census since 1979, "
            "and the Asia Foundation's Survey of the Afghan People, which reached all 34 provinces "
            "in 2019, did not ask it. So the **34,935,197** settled people in the National "
            "Statistics and Information Authority's estimate for 2025-26 are all drawn as Muslim, "
            "province by province, and inside each province they follow Kontur's population grid. "
            "That estimate is projected from a household listing made between 2002 and 2006; the "
            "US government's own figure for mid-2023 is 39.2 million. Nobody counted these dots, "
            "and they are drawn desaturated to say so. "
            "**Shia and Sunni Muslims are drawn as one.** The World Religion Database's 2022 "
            "estimate is 89% Sunni and 11% Shia, Columbia University's Gulf 2000 project puts the "
            "Shia as high as 29%, and in Pew Research Center's 2011 survey 90% of Afghan Muslims "
            "called themselves Sunni and 7% Shia. None of them gives a figure for any province, "
            "and Shia Hazaras have been attacked repeatedly, so no split is drawn. "
            "**Almost no Sikhs or Hindus remain.** The US State Department's 2023 report on "
            "religious freedom says six are left, where community representatives counted more "
            "than 1,000 when the Taliban took power in 2021. Six people are fewer than one dot, "
            "and they are not drawn. The same report says there is no reliable estimate of "
            "Christians or Baha'is and no known Jews, so none of them are drawn either. Pew "
            "Research Center's 2020 estimate is 99.86% Muslim; its figures for everyone else "
            "(7,571 Christians, 7,814 Buddhists, 35,179 of other religions) are not drawn, because "
            "nothing says who or where they are. "
            "**The 1.5 million nomadic Kuchis are not on the map.** The statistics authority holds "
            "them at that fixed figure for the whole country, with no province, so they are the "
            "not-drawn part of the bar."),
        how="no source asks; everyone drawn as Muslim",
        grain="provinces, 1,028,000 people on average",
        gap="the 1.5 million nomadic Kuchis, 4.1% of the population, whom the estimate holds at one "
            "national figure with no province; and Afghans who are not Muslim, whom no source has "
            "counted",
        gap_share=0.04117,
        counts=_af_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "af" / "af_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_af_place_weight,
        note="REOPENED ON ANITA'S PRIORITY (ask/RULINGS.md 2026-09-15) AND THE MAURITANIA RULING "
             "(2026-09-16: drawn on a compiler's national figure). sources/af.md is the record. "
             "NOBODY ASKED: no census since 1979 (NSIA 1404 introduction); Survey of the Afghan "
             "People has no religion or sect item (sources.md §11ao); no UNSD row. All 34,935,197 "
             "settled people (NSIA 1404 Table 3, checked against its 1402-1404 table) on islam; Pew "
             "2020's 53,928 non-Muslims not drawn (no source places them); IRF 2023: six Sikhs and "
             "Hindus, no estimate for Christians or Baha'is. SECT: not drawn; WRD 2022 11% Shia, "
             "Gulf 2000 up to 29%, Pew 2011 7% of Muslims, none by province (sources/af.md §3; §14). "
             "GEOGRAPHY: COD-AB v03 34 provinces, names pinned to pcodes, COD-PS 2026 rank witness "
             "+0.892. PLACEMENT: Kontur AF calibrated to NSIA provinces (1.211x nationally; Sar-e "
             "Pol 0.11, shape kept); 16 false cap blocks lowered, Kabul and Herat cities left and "
             "checked against NSIA's urban column (sources/af_grid.py BLOCKS). NOT DRAWN: 1,500,000 "
             "Kuchi (gap_share), non-Muslims (no source).",
    ),
}
