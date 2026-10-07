# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _sy_place_weight(place):
    """countries.py hook. `place` is the 400 m hex layer scatter.py has read.

    Homs, Deir-ez-Zor, Rural Damascus, Ar-Raqqa and Al-Hasakeh hold the Syrian desert. Drawn flat,
    their dots would sit on the Badia (sources/sy_grid.py).
    """
    return _kontur_place_weight(place, "sy_hexes.gpkg", "sources/sy_grid.py")


def _sy_counts():
    """Pew 2020's national mix (the World Religion Database) on the CBS end-2011 estimate: 14 governorates.

    EVERY ROW IS `modelled` (§7b). No census has asked religion since 1960 and no survey with the
    question has been released, so each governorate's people take one national mix.
    sources/sy.py and sources/sy.md.
    """
    from sy2020 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "sy.csv",
                     dtype={"geo_id": str}, keep_default_na=False, na_values=[""])
    lut = pd.read_csv(HERE / "data" / "geo" / "sy" / "sy_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    missing = sorted(df.loc[df["unit"].isna(), "geo_id"].unique())
    if missing:
        raise SystemExit(f"sy.csv governorates with no polygon: {missing}; re-run sources/sy_geo.py")
    if df["unit"].nunique() != 14:
        raise SystemExit(f"{df['unit'].nunique()} governorates, expected 14")
    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(df.loc[df["node"].isna(), "source_category"].unique())
    if unmapped:
        raise SystemExit(f"sy.csv categories with no node: {unmapped}")
    df = df[df["count"] > 0]
    df["tier"] = "modelled"
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations", "tier"]]


ENTRY = {
    "sy": dict(
        name="Syria",
        source="No census or survey asks; Pew Research Center's 2020 estimate for Syria (from the "
               "World Religion Database), against each governorate's people in the Central Bureau "
               "of Statistics' estimate for the end of 2011 (published by OCHA)",
        basis="a compiler's national estimate, the same in every governorate",
        note_public=(
            "**No Syrian census has asked about religion since 1960, and no survey of Syria with "
            "the question has been released.** The Arab Barometer interviewed 1,229 Syrians in "
            "late 2025, and its data are not yet out. This map uses Pew Research Center's estimate "
            "for 2020, which for Syria comes from the World Religion Database rather than from a "
            "survey: **94.2%** Muslim, **3.8%** Christian and 2.0% with no religion, drawn at the "
            "same shares in every governorate. Nobody counted these dots, so they disappear "
            "when inferred dots are turned off. "
            "**The map does not show where Syria's religious communities live.** The Druze are "
            "concentrated in Suwayda, the Alawites on the coast and in the Homs and Hama "
            "countryside, and Christians in Damascus, Aleppo, Homs, the Wadi al-Nasara and "
            "Hasakah. No count or survey gives any of them by governorate, so each governorate is "
            "drawn at the national mix, and Suwayda appears 94% Muslim like everywhere else. "
            "**The Druze, Alawites, Ismailis and Shia are all inside the Muslim share.** The World "
            "Religion Database counts all four as Muslims, so this map does too, although many "
            "Druze do not consider their faith part of Islam. The same estimate put Christians at "
            "8.6% of Syria in 2010 and **3.8%** in 2020, after a decade of war in which many left "
            "the country. "
            "**The dots stand where people lived at the end of 2011.** They follow the Central "
            "Bureau of Statistics' estimate of the **21,377,000** people living in each "
            "governorate on 31 December 2011, the last before the war. Since then millions have "
            "left: in August 2026 UNHCR counted about 4.7 million Syrians registered as refugees "
            "in Türkiye, Lebanon, Jordan, Iraq, Egypt and North Africa, and many more were "
            "displaced inside the country, many of them to Idlib and the north. The UN's current "
            "population figures by governorate are for humanitarian use only and not for "
            "research, so the pre-war estimate is the most recent this map can draw. "
            "**The Golan, which Israel has held since 1967, is drawn with Israel**, whose census "
            "counts the people there. Quneitra on this map is the part Syria held. "
            "**A few people are not drawn.** Pew's estimate has about 2,800 Hindus, Jews and "
            "members of other religions (0.01%), and nothing says where they live. Palestinian "
            "refugees in Syria are inside the national mix."),
        how="no source asks; a compiler's national estimate, one mix everywhere",
        grain="governorates, 1.5 million people on average",
        gap="Hindus, Jews and members of other religions, about 2,800 people or 0.01% in Pew's "
            "estimate, whom nothing places",
        gap_share=0.000132,
        counts=_sy_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "sy" / "sy_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_sy_place_weight,
        note="BUILT ON ANITA'S RULINGS OF 2026-09-15 (a §14 case files an ask and carries on) AND "
             "2026-09-16 (a compiler's figure where nothing asks); ASK 050 holds the placement of "
             "Druze, Alawites and Christians. sources/sy.md is the record. LEVEL: Pew 2020 Syria "
             "row, which Pew's Appendix A sources to the World Religion Database (UN WPP 2024 "
             "population): Muslim 94.166, Christian 3.841, unaffiliated 1.980; Druze inside "
             "Muslims (Other_religions 0.003% against Lebanon's 4.3%); no sect split (om/sa "
             "ruling). Pew's 2,782 Hindus, Jews and others not drawn (gap). POPULATION: CBS "
             "Statistical Abstract 2012 Table 3/2, people actually living in Syria on 31/12/2011, "
             "21,377 thousand, via OCHA HDX; the 2025 Population Task Force baseline is marked "
             "not for research; HNO/HNRP files print no population; Kontur tracks 2011 within "
             "0.86-1.24 per governorate, so it does not see displacement either. GEOGRAPHY: "
             "COD-AB v02 14 governorates, joined by row order with English and Arabic witnesses; "
             "Natural Earth's Israeli-administered Golan clipped from Quneitra (1,093 km2), "
             "UNDOF zone kept. PLACEMENT: Kontur SY 2023, ratio 1.089, rank witness +0.982; three "
             "Damascus cap blocks real (kontur_cap.csv). NOT DRAWN: placement of any community "
             "(ask 050); displacement since 2011 (pre-war positions drawn).",
    ),
}
