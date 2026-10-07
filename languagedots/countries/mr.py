# Mauritania. The census asked mother tongue in 2013 and never published it. Mauritanians from
# Afrobarometer R10's home language by wilaya; foreign residents by the 2023 census's nationality
# groups, refugees on northern Mali's census mix (sources/mr_afro.py). Placed on religiondots'
# Kontur hexes, calibrated to the moughataas (read-only). Record: sources/mr.md.
from _shared import *  # noqa: F401,F403

RGPH_2023 = 4_927_531


def _counts():
    import mr2024
    df = pd.read_csv(NORM / "mr.csv", dtype={"geo_id": str})
    if df["geo_id"].nunique() != 15 or int(df["count"].sum()) != RGPH_2023:
        raise SystemExit(f"mr.csv: {df['geo_id'].nunique()} wilayas, {df['count'].sum():,} "
                         "people -- run sources/mr_afro.py")
    df["node"] = df["source_category"].map(mr2024.resolve)
    df["unit"] = df["geo_id"]   # religiondots' mr_lookup.csv: geo_id == unit
    return df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()


ENTRY = dict(
    name="Mauritania",
    source=("Afrobarometer Round 10 (2024), language spoken in the household; Recensement "
            "Général de la Population et de l'Habitat 2023 (ANSADE), Thème 1 and Thème 16 "
            "(foreign residents by wilaya and nationality); Mali's 2022 census for the refugees"),
    how=("survey, 2024, household language, as shares per wilaya applied to the 2023 census; "
         "foreign residents by nationality"),
    parts=[
        dict(covers="Mauritanians",
             source="Afrobarometer 2024, household language, 1,200 adults, applied to the 2023 "
                    "census",
             people=4_801_598),
        dict(covers="Foreign residents",
             source="2023 census, nationality, drawn on that country's languages; Mbera "
                    "refugees on northern Mali's",
             rest=True),
    ],
    grain="15 wilayas, 328,000 people on average",
    gap="survey answers of French with no ethnic group given (7 of 1,200), left out before the shares",
    view=[-17.2, 14.6, -4.7, 27.4],
    counts=_counts,
    mappings=["mr2024"],
    place=RD_GEO / "mr" / "mr_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "Mauritania's 2013 census asked everyone their mother tongue, but the results were "
        "never published. Mauritanians are drawn from the 2024 Afrobarometer survey, which "
        "asked 1,200 adult citizens the language spoken in their household: Hassaniya Arabic "
        "86%, Pulaar 11%, Soninke 3% and Wolof 0.5%. Those who answered French are drawn on "
        "their ethnic group's language. Each wilaya's shares come from its own respondents, "
        "as few as 16 in Adrar and Inchiri. A 2015 UNICEF survey found Wolof at 1.7% of "
        "household heads, so Wolof is probably drawn too low. The 46,800 refugees in the "
        "Mbera camp are drawn on the languages of northern Mali, where they came from."),
)
