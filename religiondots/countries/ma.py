# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _ma_place_weight(place):
    """countries.py hook. `place` is the 400 m hex layer scatter.py has read.

    Morocco's southern provinces are tens of thousands of km2 with their people in one or two
    towns, and Kontur has lost three of those towns (Smara, Tan-Tan, Assa), which carry discs from
    HCP's commune counts (sources/ma_grid.py).
    """
    return _kontur_place_weight(place, "ma_hexes.gpkg", "sources/ma_grid.py")


def _ma_counts():
    """Arab Barometer waves V to VIII for Moroccans, nationality for foreigners: 73 units.

    EVERY ROW IS `modelled` (§7b). No Moroccan census has asked religion. Moroccans take the
    survey's national non-Muslim share split urban and rural, at a national mix; foreign residents
    take HCP's national nationality mix through Pew. Both halves are rows of HCP's 2024
    legal-population workbook, so they partition each unit. sources/ma.py and sources/ma.md.
    """
    from ma2024 import resolve

    cit = pd.read_csv(HERE / "data" / "normalized" / "ma.csv", dtype={"geo_id": str},
                      keep_default_na=False, na_values=[""])
    cit["node"] = cit["source_category"].map(resolve)
    unmapped = sorted(cit.loc[cit["node"].isna(), "source_category"].unique())
    if unmapped:
        raise SystemExit(f"ma.csv categories with no node: {unmapped}")
    ext = pd.read_csv(HERE / "data" / "normalized" / "ma_foreign.csv", dtype={"geo_id": str})
    df = pd.concat([cit[["geo_id", "node", "count"]], ext[["geo_id", "node", "count"]]],
                   ignore_index=True)
    lut = pd.read_csv(HERE / "data" / "geo" / "ma" / "ma_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    missing = sorted(df.loc[df["unit"].isna(), "geo_id"].unique())
    if missing:
        raise SystemExit(f"ma rows with no unit: {missing}; re-run sources/ma_geo.py")
    if df["unit"].nunique() != 73:
        raise SystemExit(f"{df['unit'].nunique()} units, expected 73")
    df = df[df["count"] > 0]
    df["tier"] = "modelled"
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations", "tier"]]


ENTRY = {
    "ma": dict(
        name="Morocco",
        source="Arab Barometer, six rounds 2018 to 2024 (Arab Barometer, Princeton University), "
               "against the 2024 census count of Moroccans and foreign residents in each "
               "province (Haut-Commissariat au Plan, RGPH 2024)",
        basis="self-identification, adults 18 and over; foreign residents by nationality",
        note_public=(
            "**No Moroccan census has asked about religion.** Neither the 2014 nor the 2024 "
            "census form has the question. This map uses the Arab Barometer instead: "
            "**10,382 Moroccan adults** who named a religion in six rounds of interviews between "
            "2018 and 2024, laid on the 2024 census count of Moroccans in each province, "
            "**36,680,178** people. Nobody counted these dots, and they are drawn desaturated to "
            "say so. "
            "**Non-Muslims are 0.33% of Moroccans in the survey, and more of them live in towns.** "
            "36 of the 10,382 said they were not Muslim: 19 Christian, 15 with no religion and 2 "
            "something else. No region stands apart once the rounds are checked against each "
            "other. Non-Muslims were more urban than the people interviewed, in the Arab Barometer "
            "and again in the Afrobarometer, a separate survey, so each province is drawn at "
            "0.42% non-Muslim for its town dwellers and 0.17% for its rural population, from the "
            "census's own urban and rural counts. Everywhere the non-Muslims are drawn at one mix: "
            "65.8% Christian, 30.9% no religion and 3.3% something else. "
            "**The level moves a lot from round to round.** Nobody in the 2018 to 2019 round said "
            "they were not Muslim, and 0.82% did in 2023 to 2024, the round that holds 12 of the "
            "19 Christians. The Afrobarometer found 18 non-Muslims among 5,981 Moroccans over five "
            "rounds, only 3 of them Christian, and Pew Research Center's 2020 estimate for the "
            "whole country is 0.085% Christian and 0.13% with no religion. Moroccans who have "
            "become Christian or left Islam may not say so to an interviewer. The split between "
            "Christians and people with no religion is the loosest figure on this map. "
            "**The 148,152 foreign residents are drawn where the census counted them.** Their "
            "nationality is published only for the country as a whole: 59.9% from sub-Saharan "
            "Africa, led by Senegal and Côte d'Ivoire, and 20.3% from Europe, two thirds of them "
            "French. Each province's foreigners are drawn at that mix, and each nationality at "
            "Pew's estimate for its home country, which cannot see anyone who converted or "
            "stopped practising. That puts **78,286** of them on Islam and about 49,000 on "
            "Christian churches. "
            "**Western Sahara is drawn as far as Morocco administers it.** Laâyoune-Sakia El "
            "Hamra, Dakhla-Oued Ed-Dahab and the Assa-Zag commune of Al Mahbass hold 690,132 of "
            "the people counted, placed only west of the berm. The census counts garrisons along "
            "the berm in communes named for places beyond it, and their dots go to the "
            "province's towns. "
            "**Jews are not drawn.** No Jewish answer is in the Arab Barometer pool, and "
            "one of the 5,981 Afrobarometer answers is Jewish."),
        how="survey, six rounds 2018 to 2024 pooled; foreign residents by nationality",
        grain="provinces, 504,000 people on average",
        counts=_ma_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "ma" / "ma_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_ma_place_weight,
        note="BUILT ON ANITA'S MAGHREB RULINGS (ask/RULINGS.md 2026-09-15 and 2026-09-16). "
             "sources/ma.md is the record. MOROCCANS: Arab Barometer V to VIII (10,382 answers, "
             "36 non-Muslim); III and IV out on the card (no no-religion box) and the pre-2015 "
             "regions. No region stands apart (Bonferroni over 8, both halves). The urban excess "
             "passes a pre-registered P < 0.05 in the Arab Barometer (0.016) and the Afrobarometer "
             "(0.014), so provinces take 0.421% urban and 0.168% rural from HCP's urban and rural "
             "Moroccans. FOREIGNERS: 148,152 per commune from the same workbook, HCP's national "
             "nationality mix (study of November 2025), Pew 2020 per nationality and Pew's "
             "regional totals for the unnamed; Muslim branches folded to islam. GEOGRAPHY: 69 "
             "COD-AB Morocco provinces plus 4 COD-AB Western Sahara units cut to Natural Earth's "
             "B19 (ask 031); Tarfaya's strip north of 27°40'N is in neither COD file and its hexes "
             "go to Laâyoune; Al Mahbass moved to Es-Semara; Ceuta and Melilla cut out. PLACEMENT: "
             "Kontur MA and EH; discs on Smara, Tan-Tan and Assa, which Kontur has lost "
             "(sources/ma_grid.py). NOT DRAWN: Jews (no answer), Ibadis, presence rings (Anita "
             "2026-09-16: no rings).",
    ),
}
