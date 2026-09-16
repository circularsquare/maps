# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _sd_place_weight(place):
    """countries.py hook. `place` is the 400 m hex layer scatter.py has read.

    Northern, North Darfur and Red Sea states are 900,000 km2 of desert and hold 11% of the 2022
    projection. Drawn flat, their dots would sit on the sand (sources/sd_grid.py).
    """
    return _kontur_place_weight(place, "sd_hexes.gpkg", "sources/sd_grid.py")


def _sd_counts():
    """Afrobarometer R5-R9 and Arab Barometer V and VII pooled: one national share and mix on 18 states.

    EVERY ROW IS `modelled` (§7b). The 2008 census had its religion question removed. 86 non-Muslim
    answers in 10,657 place nothing below the nation, so each state's people in the CBS projection
    for 2022 take the national non-Muslim share and mix. sources/sd.py and sources/sd.md.
    """
    from sd2022 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "sd.csv",
                     dtype={"geo_id": str}, keep_default_na=False, na_values=[""])
    lut = pd.read_csv(HERE / "data" / "geo" / "sd" / "sd_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    missing = sorted(df.loc[df["unit"].isna(), "geo_id"].unique())
    if missing:
        raise SystemExit(f"sd.csv states with no polygon: {missing}; re-run sources/sd_geo.py")
    if df["unit"].nunique() != 18:
        raise SystemExit(f"{df['unit'].nunique()} states, expected 18")
    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(df.loc[df["node"].isna(), "source_category"].unique())
    if unmapped:
        raise SystemExit(f"sd.csv categories with no node: {unmapped}")
    df = df[df["count"] > 0]
    df["tier"] = "modelled"
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations", "tier"]]


ENTRY = {
    "sd": dict(
        name="Sudan",
        source="Afrobarometer, five rounds 2013 to 2022, and Arab Barometer, two rounds 2018 to "
               "2022 (Arab Barometer, Princeton University), against each state's people in the "
               "2022 population projection (Central Bureau of Statistics, Sudan, published by "
               "OCHA)",
        basis="self-identification, Sudanese adults 18 and over",
        note_public=(
            "**The Presidency removed the religion question from Sudan's 2008 census, and no "
            "census has been held since.** This map uses two surveys instead: **10,657 Sudanese "
            "adults** who named a religion in five rounds of the Afrobarometer (2013 to 2022) "
            "and two of the Arab Barometer (2018-19 and 2022), with the share who were not "
            "Muslim applied to each state's people in the Central Bureau of Statistics' "
            "projection for 2022, **46,934,433** people. Nobody counted these dots, and they are "
            "drawn desaturated to say so. "
            "**Non-Muslims are 0.69% of Sudanese as drawn, at the same share in every state.** "
            "86 of the 10,657 said they were not Muslim: 62 Christian and 24 with no religion, "
            "drawn at one mix, 73% Christian and 27% with no religion. Christians answered in "
            "every region. In the 2021 and 2022 interviews they were more common in Kordofan and "
            "Khartoum than elsewhere, but the earlier rounds do not show the same, so the map "
            "does not place them. The Christian share is 0.50%. Pew Research Center's figures for "
            "Sudan are not a separate check on it: Pew's 2010 estimate matches the Afrobarometer's "
            "2013 round, so its estimates rest largely on the same surveys. Sudanese who have "
            "become Christian or left Islam may not say so to an interviewer; leaving Islam "
            "carried the death penalty until 2020, and four of the seven rounds were asked before "
            "then. "
            "**The map shows Sudan before the war that began in April 2023.** Every interview "
            "and the population projection come from before it. The UN Population Fund put the "
            "number of people who had fled their homes since then at more than 12 million in "
            "2025, so many of these dots are no longer where people live. "
            "**Foreign residents and refugees are not on this map.** Both surveys interview "
            "Sudanese citizens. The UN Population Division estimates **1,379,147** people born "
            "abroad or registered as refugees living in Sudan in 2020, and UNHCR counted 796,831 "
            "refugees from South Sudan at the end of 2022; Pew puts South Sudan at 61% "
            "Christian. They are in the not drawn part of the bar. "
            "**Abyei is not drawn.** The projection has no figure for it, neither survey "
            "interviews there, and its status between Sudan and South Sudan is not settled. The "
            "Halaib triangle, which Egypt administers, is left out of Sudan. "
            "**Some answers are left out.** Two earlier Arab Barometer rounds (2010-11 and 2013) "
            "had no box for having no religion. In the Afrobarometer's 2021 round, 31 people in "
            "Darfur were recorded with no religion, where the other rounds and the Arab "
            "Barometer found 3 there in about 2,400 interviews; they are treated as a recording "
            "error. Eight Arab Barometer answers that named a Muslim branch on the next question "
            "are left out too."),
        how="surveys, two series 2013 to 2022 pooled, one national share",
        grain="states, 2.6 million people on average",
        gap="foreign residents and refugees, whom neither survey interviews: about 1.38 million "
            "in 2020 by the UN Population Division's estimate, 2.9% of residents; and Abyei, "
            "which the projection leaves out",
        gap_share=0.02855,
        counts=_sd_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "sd" / "sd_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_sd_place_weight,
        note="BUILT ON ANITA'S MAGHREB AND MAURITANIA RULINGS (ask/RULINGS.md 2026-09-15 and "
             "2026-09-16) AND ASK 033. sources/sd.md is the record. POOL: Afrobarometer R5-R9 "
             "(6,557 answers) and Arab Barometer V and VII (4,100); AB II and III out on the card "
             "(no no-religion box). DROPPED: refusals; 8 AB V answers contradicting the "
             "follow-up (6 atheists, 1 Christian, 1 Jewish, all naming a Muslim branch); R8's 31 "
             "`None` in Darfur (2 in the other rounds, 1 in AB); 3 single answers (Bahai, "
             "Jewish, traditional). 86 non-Muslim: 62 Christian, 24 no religion. TESTS: held-out "
             "r +0.999 (AB V, VII, Afro R9 at 18 states); quota p 1; standouts at six regions "
             "fail in the early half (Kordofan and Khartoum pass the late half only); urban P "
             "0.086 Afro, 0.81 AB. So one national share, 0.687%, and mix everywhere; Christian "
             "0.504% against Pew 2020 0.488%, NOT an independent check (Pew's 2010 row equals "
             "Afro R5's shares; sd.md 10). POPULATION: COD-PS 2022 (CBS projection from 2008, "
             "pre-war), 18 states; Abyei dropped (no projection); Halaib clipped north of 22 N "
             "from Red Sea state (de facto Egypt, spec 14.18). PLACEMENT: Kontur SD, 1.025 of the "
             "projection, rank witness +0.864, no seat lost; 26 cap blocks on small towns or on "
             "no town, all judged false, in kontur_cap.csv: 21 capped, 5 with no populated ring "
             "outside the blocks `isolated` (lowered to Red Sea state's median density outside "
             "the blocks, sd.md 6). "
             "NOT DRAWN: foreigners and refugees (UN DESA IMS 2024, mid-2020, B R, 1,379,147; "
             "gap_share on the projection plus it; UNHCR's pre-war refugees by state is the REOPEN "
             "route); the war since April 2023 (pre-war positions drawn).",
    ),
}
