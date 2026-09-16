# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _dz_place_weight(place):
    """countries.py hook. `place` is the 400 m hex layer scatter.py has read.

    Nine in ten Algerians live on the northern tenth of the country, and the four Saharan
    wilayas under one person per km2 are 1.49 million km2. Drawn flat, a wilaya's dots would sit
    on the desert (sources/dz_grid.py).
    """
    return _kontur_place_weight(place, "dz_hexes.gpkg", "sources/dz_grid.py")


def _dz_counts():
    """Arab Barometer waves V to VII pooled, at wilaya: 4 categories, all 48 units.

    EVERY ROW IS `modelled` (§7b). No Algerian census has asked religion. The survey gives a
    non-Muslim share for Kabylie (Tizi Ouzou, Béjaïa, Bouira, as one unit on Anita's ruling of
    2026-09-16) and one for the other 45 wilayas, and a national mix of what the non-Muslims
    said; the RGPH 2008 count of each wilaya gives the people. sources/dz.py and sources/dz.md.
    """
    from dz2022 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "dz.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    lut = pd.read_csv(HERE / "data" / "geo" / "dz" / "dz_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    missing = sorted(df.loc[df["unit"].isna(), "geo_id"].unique())
    if missing:
        raise SystemExit(f"dz.csv wilayas with no polygon: {missing}; re-run sources/dz_geo.py")
    if df["unit"].nunique() != 48:
        raise SystemExit(f"{df['unit'].nunique()} wilayas, expected 48")
    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(df.loc[df["node"].isna(), "source_category"].unique())
    if unmapped:
        raise SystemExit(f"dz.csv categories with no node: {unmapped}")
    df = df[df["count"] > 0]
    df["tier"] = "modelled"
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations", "tier"]]


ENTRY = {
    "dz": dict(
        name="Algeria",
        source="Arab Barometer, five rounds 2018 to 2022 (Arab Barometer, Princeton University), "
               "against each wilaya's population in the 2008 census (Office National des "
               "Statistiques, RGPH 2008)",
        basis="self-identification, adults 18 and over",
        note_public=(
            "**No Algerian census has asked about religion.** Neither the 2008 nor the 2022 "
            "census form has the question, and the state publishes no count of Christians or of "
            "people with no religion. This map uses the Arab Barometer instead: **7,653 adults** "
            "who named a religion in five rounds of interviews between 2018 and 2022, with the "
            "share who were not Muslim applied to each wilaya's population in the 2008 census. "
            "That is the last count by wilaya the statistics office has published, "
            "**34,080,030** people; the 2022 census's wilaya results have not appeared. Nobody "
            "counted these dots, and they are drawn desaturated to say so. "
            "**Non-Muslims are 0.43% of the country as drawn, and nearly half of them are in "
            "Kabylie.** The survey found 16 non-Muslims among 613 people interviewed in Tizi "
            "Ouzou, Béjaïa and Bouira, and 20 among 7,040 in the other 45 wilayas: **2.61%** "
            "against 0.24%, weighted. The gap is there in the 2018 to 2019 round on its own and "
            "again in the later rounds on their own. The three Kabyle wilayas are drawn as one "
            "area with one share, since 613 interviews cannot say which of them has more. "
            "Outside Kabylie no single wilaya stands apart once the rounds are checked against "
            "each other, so the other 45 share one figure. Everywhere, the non-Muslims are drawn "
            "at the same mix, from 36 answers: **52.5%** no religion, 39.2% Christian and 8.3% "
            "something else. "
            "**Treat these as floors.** Pew Research Center's 2020 estimate for Algeria is 0.29% "
            "Christian and 1.27% with no religion, against 0.17% and 0.23% drawn here, and "
            "neither is an easy thing to say to an interviewer in Algeria. "
            "**Three earlier rounds are left out.** Their answer cards had no box for having no "
            "religion, and in the rounds that had one, that box is half of the non-Muslim "
            "answers, so pooling them would mix two different questions. Six answers from "
            "2018 to 2019 are also left out because the religion given contradicts the branch "
            "or church given to the next question. All three people recorded as Jewish are "
            "among them, so no Jewish answer remains. "
            "**The Ibadis of the M'zab are not drawn.** The survey's follow-up question on "
            "branch records 7 Ibadi or Mozabite answers in the whole pool and none of them in "
            "Ghardaïa, which cannot place anyone, so every Muslim is on the one Islam colour. "
            "**The survey interviewed adults, and the shares are applied to everyone.** The "
            "people are the 2008 count, and Kabylie grew more slowly than the country before "
            "that census (0.2% a year in Tizi Ouzou and 0.6% in Béjaïa between 1998 and 2008, "
            "against 1.6% nationally), so its share of Algeria today is probably smaller than "
            "drawn. "
            "**The Sahrawi refugee camps near Tindouf are not drawn.** The 2008 census counted "
            "49,149 people in Tindouf wilaya, and the camps are not among them. The UN agencies "
            "working there, and Algeria's government, put the camps at 173,600 people, from a "
            "study in 2018. They are in the not drawn part of the bar."),
        how="survey, five rounds 2018 to 2022 pooled",
        grain="wilayas, 710,000 people on average",
        gap="the Sahrawi refugee camps near Tindouf, which the 2008 census did not count; 173,600 "
            "people by the UN agencies' planning figure from a 2018 study, 0.5% of residents",
        gap_share=0.00507,
        counts=_dz_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "dz" / "dz_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_dz_place_weight,
        note="BUILT ON ANITA'S MAGHREB RULINGS (ask/RULINGS.md 2026-09-15 and 2026-09-16): "
             "draw the 99.x% Muslim countries with the best evidence for where non-Muslims are, "
             "and Kabylie as one unit for the non-Muslim share. sources/dz.md is the record. "
             "THE POOL IS WAVES V TO VII. II, III and IV are in the files and out of the pool "
             "because their cards have no no-religion box, which is half of the non-Muslim "
             "answers where offered; wave I has no subnational column. Atheist (V) and No "
             "religion (VI, VII) are merged as each card's code 4. Six wave V answers whose "
             "religion contradicts the denomination follow-up are dropped (all three Jewish). "
             "Wave VI's WT is un-normalised for Algeria only (means 0.78-0.85) and is rescaled "
             "in ab.load. KABYLIE is re-tested on every build: 16 of 613 against 20 of 7,040, "
             "exact within wave P=2.2e-09, replicated in V alone and in VI-VII alone, largest "
             "PSU 3 of 16. No other wilaya stands apart (Bonferroni over 3, both halves). "
             "POPULATION IS RGPH 2008 UNSCALED: ONS has published no wilaya table since, and no "
             "ONS national estimate was opened to scale it by. Kontur 2023 reads 1.33x "
             "nationally and puts Kabylie at 6.98% of Algeria against 8.03% in 2008. "
             "PLACEMENT: Kontur has no Béchar town (448 people within 5 km against GeoNames' "
             "165,241), so a 3 km disc carries the wilaya's shortfall; Tindouf keeps only hexes "
             "within 15 km of the town, dropping 69,782 Kontur people where the refugee camps "
             "are (sources/dz_grid.py). The camps are in gap and gap_share at the Sahrawi "
             "Refugees Response Plan 2024-2025's 173,600 (ask 033; sources/dz.md 10). "
             "NOT DRAWN: foreigners (no table by wilaya found, REOPEN in queue.md), Ibadis "
             "(no magnitude), presence rings (Anita 2026-09-16: no rings anywhere in the Maghreb).",
    ),
}
