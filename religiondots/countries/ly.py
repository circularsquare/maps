# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _ly_place_weight(place):
    """countries.py hook. `place` is the 400 m hex layer scatter.py has read.

    Kufra, Murzuq and Ajdabiya are half of Libya and hold 5% of its Libyans. Drawn flat, their
    dots would sit on the sand seas (sources/ly_grid.py).
    """
    return _kontur_place_weight(place, "ly_hexes.gpkg", "sources/ly_grid.py")


def _ly_counts():
    """Arab Barometer waves V to VII pooled: one national share and mix on 22 districts.

    EVERY ROW IS `modelled` (§7b). No Libyan census asks religion. Seven non-Muslim answers place
    nothing, so each district's Libyans in BSC's 2020 estimate take the national non-Muslim share
    and mix. Non-Libyans are not in the base. sources/ly.py and sources/ly.md.
    """
    from ly2020 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "ly.csv",
                     dtype={"geo_id": str}, keep_default_na=False, na_values=[""])
    lut = pd.read_csv(HERE / "data" / "geo" / "ly" / "ly_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    missing = sorted(df.loc[df["unit"].isna(), "geo_id"].unique())
    if missing:
        raise SystemExit(f"ly.csv districts with no polygon: {missing}; re-run sources/ly_geo.py")
    if df["unit"].nunique() != 22:
        raise SystemExit(f"{df['unit'].nunique()} districts, expected 22")
    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(df.loc[df["node"].isna(), "source_category"].unique())
    if unmapped:
        raise SystemExit(f"ly.csv categories with no node: {unmapped}")
    df = df[df["count"] > 0]
    df["tier"] = "modelled"
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations", "tier"]]


ENTRY = {
    "ly": dict(
        name="Libya",
        source="Arab Barometer, five rounds 2018 to 2022 (Arab Barometer, Princeton University), "
               "against each district's Libyans in the 2020 population estimate (Bureau of "
               "Statistics and Census, Libya)",
        basis="self-identification, Libyan adults 18 and over",
        note_public=(
            "**No Libyan census has asked about religion, and none has been held since 2006.** "
            "This map uses the Arab Barometer instead: **7,191 Libyan adults** who named a "
            "religion in five rounds of interviews between 2018 and 2022, with the share who were "
            "not Muslim applied to each district's Libyans in the Bureau of Statistics and "
            "Census's estimate for 2020, **6,872,674** people. Nobody counted these dots, and they "
            "are drawn desaturated to say so. "
            "**Non-Muslims are 0.10% of Libyans as drawn, at the same share in every district.** "
            "7 of the 7,191 said they were not Muslim: 6 Christian and 1 atheist. Seven answers "
            "cannot show where anyone lives, so every district is drawn at one share and one mix, "
            "83% Christian and 17% with no religion. Neither is an easy thing to say to an "
            "interviewer in Libya. "
            "**Foreign residents are not on this map.** The survey interviews Libyan citizens and "
            "the 2020 estimate counts Libyans only. The UN Population Division estimates "
            "**826,537** non-Libyans living in Libya in 2020, refugees included, which would make "
            "them about one resident in nine, and 897,751 in 2024. No count says where each "
            "nationality lives. They are in the not drawn part of the bar. Pew Research Center's "
            "2020 estimate, which covers everyone living in Libya, is 0.52% Christian, 0.26% "
            "Buddhist and 0.09% Hindu, against 0.08% Christian drawn here for Libyans alone. "
            "**Ibadis are not drawn.** Five answers to the follow-up question on branch name the "
            "Ibadis, in Jafara, Zuwara, Murqub and Tripoli, which is too few to draw, so every "
            "Muslim is on the one Islam colour. "
            "**One earlier round is left out.** Its answer card, in the 2012 to 2014 round, had no "
            "box for having no religion."),
        how="survey, five rounds 2018 to 2022 pooled, one national share",
        grain="districts, 312,000 Libyans on average",
        gap="non-Libyans, who are not in the 2020 estimate; about 827,000 in 2020 by the UN "
            "Population Division's migrant stock estimate, refugees included, 10.7% of residents",
        gap_share=0.10735,
        counts=_ly_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "ly" / "ly_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_ly_place_weight,
        note="BUILT ON ANITA'S MAGHREB RULINGS (ask/RULINGS.md 2026-09-15 and 2026-09-16). "
             "sources/ly.md is the record. POOL: Arab Barometer V to VII (7,191 answers, 7 "
             "non-Muslim: 6 Christian, 1 atheist); III out on the card (no no-religion box, no "
             "non-Muslim). The survey samples citizens (AB VII technical report). WAVE VII's "
             "district labels fail the allocation check (Spearman +0.853 against +0.99 in V and "
             "VI; Ajdabiya 3.7x, Benghazi 0.32x): used for the national level only. TESTS on V and "
             "VI: held-out r +0.999; no district stands apart; urban P 0.61 (no Afrobarometer). So "
             "one national share, 0.102%, and mix everywhere. POPULATION: BSC's 2020 estimate of "
             "Libyans by region (PDF, 6,872,674); USCB's copy has Al Marj 58,387 too high "
             "(pinned). PLACEMENT: Kontur LY, rank witness +0.999, every district 0.98-1.06, no "
             "holes (seat check gated on the district ratio), no cap blocks. NOT DRAWN: "
             "non-Libyans (not in the base; 2012 survey by district in continent groups only; no "
             "IOM), sized in gap_share by UN DESA International Migrant Stock 2024, mid-2020, "
             "826,537 (foreign citizens plus refugees), ask 033; Ibadis (5), presence rings (no "
             "rings).",
    ),
}
