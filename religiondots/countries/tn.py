# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _tn_place_weight(place):
    """countries.py hook. `place` is the 400 m hex layer scatter.py has read.

    Tataouine and Kébili are 37% of Tunisia and held 2.9% of its people in 2024. Drawn flat, their
    dots would sit on the erg (sources/tn_grid.py).
    """
    return _kontur_place_weight(place, "tn_hexes.gpkg", "sources/tn_grid.py")


def _tn_counts():
    """Arab Barometer waves V to VIII pooled: one national share and mix on 24 governorates.

    EVERY ROW IS `modelled` (§7b). No Tunisian census asks religion. No governorate, Greater Tunis
    or urban stratum stands apart in the survey, so each governorate's 2024 census count takes the
    national non-Muslim share and mix. sources/tn.py and sources/tn.md.
    """
    from tn2024 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "tn.csv",
                     dtype={"geo_id": str}, keep_default_na=False, na_values=[""])
    lut = pd.read_csv(HERE / "data" / "geo" / "tn" / "tn_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    missing = sorted(df.loc[df["unit"].isna(), "geo_id"].unique())
    if missing:
        raise SystemExit(f"tn.csv governorates with no polygon: {missing}; re-run sources/tn_geo.py")
    if df["unit"].nunique() != 24:
        raise SystemExit(f"{df['unit'].nunique()} governorates, expected 24")
    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(df.loc[df["node"].isna(), "source_category"].unique())
    if unmapped:
        raise SystemExit(f"tn.csv categories with no node: {unmapped}")
    df = df[df["count"] > 0]
    df["tier"] = "modelled"
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations", "tier"]]


ENTRY = {
    "tn": dict(
        name="Tunisia",
        source="Arab Barometer, six rounds 2018 to 2024 (Arab Barometer, Princeton University), "
               "against each governorate's population in the 2024 census (Institut National de "
               "la Statistique, RGPH 2024)",
        basis="self-identification, adults 18 and over",
        note_public=(
            "**No Tunisian census has asked about religion.** The 2014 census form has no such "
            "question. This map uses the Arab Barometer instead: **10,373 Tunisian adults** who "
            "named a religion in six rounds of interviews between 2018 and 2024, with the share "
            "who were not Muslim applied to each governorate's population in the 2024 census, "
            "**11,972,169** people. Nobody counted these dots, and they are drawn desaturated to "
            "say so. "
            "**Non-Muslims are 0.62% of the country as drawn, at the same share in every "
            "governorate.** 65 of the 10,373 said they were not Muslim: 28 with no religion, 25 "
            "something else and 12 Christian. No governorate stands apart once the rounds are "
            "checked against each other. Greater Tunis reads the same as the rest of the country, "
            "0.61% against 0.62%, and people in towns were not measurably more likely to give a "
            "non-Muslim answer, in this survey or in the Afrobarometer, a separate one. So every "
            "governorate is drawn at one share and one mix: **40.7%** no religion, 37.8% something "
            "else and 21.5% Christian. "
            "**The mix is the loosest figure on this map.** Ten of the 25 answers of something "
            "else come from the 2021 to 2022 round, in Le Kef, Siliana and Sousse, where the "
            "earlier rounds found one non-Muslim in 580 interviews; without them non-Muslims would "
            "be about 0.52%. The Afrobarometer found 26 non-Muslims among 5,959 Tunisians over "
            "five rounds. Pew Research Center's 2020 estimate, which covers everyone living in "
            "Tunisia, is 0.25% Christian and 0.44% with no religion, against 0.13% and 0.25% "
            "drawn here. Part of that difference is the 66,349 foreign residents, who are drawn "
            "at the same shares as Tunisians, and part is that neither is an easy thing to say to "
            "an interviewer. "
            "**Three earlier rounds are left out.** Their answer cards had no box for having no "
            "religion, and in the rounds that had one, that box is 28 of the 65 non-Muslim "
            "answers. "
            "**Jews and Ibadis are not drawn.** No Jewish answer is in the rounds used, and the "
            "cards after 2019 have no Jewish box. The follow-up question on branch records 7 "
            "Ibadi answers, 4 of them in Médenine, Djerba's governorate, which is too few to "
            "draw, so every Muslim is on the one Islam colour. "
            "**The survey interviewed adults, and the shares are applied to everyone,** including "
            "the 66,349 foreign residents the 2024 census counted (0.55%), who are not counted "
            "by governorate."),
        how="survey, six rounds 2018 to 2024 pooled, one national share",
        grain="governorates, 499,000 people on average",
        counts=_tn_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "tn" / "tn_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_tn_place_weight,
        note="BUILT ON ANITA'S MAGHREB RULINGS (ask/RULINGS.md 2026-09-15 and 2026-09-16). "
             "sources/tn.md is the record. POOL: Arab Barometer V to VIII (10,373 answers with a "
             "governorate, 65 non-Muslim); II-IV out on the card (no no-religion box; III and IV "
             "have no non-Muslim at all). WAVE VI labels codes 21009 and 21010 both Jendouba: "
             "decoded on the code, 21009 is Le Kef by sample size (asserted). TESTS, all failing "
             "and asserted to: no governorate stands apart (Bonferroni over 12, both halves); "
             "Greater Tunis 0.614% against 0.621% (P 0.35); urban P 0.41 in the Arab Barometer "
             "and 0.12 in the Afrobarometer (Morocco's bar). So one national share, 0.619%, and "
             "mix everywhere. Wave VII's Other: 10 of 14 in Le Kef, Siliana and Sousse in "
             "consecutive PSUs, kept (0.519% without). POPULATION: RGPH 2024 by governorate, "
             "Bilan Demographique p.15. PLACEMENT: Kontur TN, rank witness +0.988, no seat holes, "
             "no cap blocks. NOT DRAWN: foreigners (66,349, national only; 2019-24 arrivals by "
             "governorate are a flow), Jews, Ibadis, presence rings (no rings).",
    ),
}
