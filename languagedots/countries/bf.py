# Burkina Faso. RGPH 2006, main language spoken by province (sources/bf_rgph.py: Tableau A5.3's
# counts, its two foreign-language groups split by région from A5.2), on religiondots' Kontur
# hexes for the same 45 provinces.
from _shared import *  # noqa: F401,F403


def _counts():
    import bf2006
    df = pd.read_csv(NORM / "bf.csv")
    df = df[(df["geo_level"] == "province") & (df["count"] > 0)]
    df = df[~df["source_category"].isin(bf2006.EXCLUDED)].copy()
    df["node"] = df["source_category"].map(bf2006.resolve)
    unmapped = sorted(set(df.loc[df["node"].isna(), "source_category"]))
    if unmapped:
        raise SystemExit(f"bf: categories with no node: {unmapped}")
    df = df.rename(columns={"geo_id": "unit"})
    if df["unit"].nunique() != 45:
        raise SystemExit(f"bf: {df['unit'].nunique()} provinces, expected 45")
    return df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()


ENTRY = dict(
    name="Burkina Faso",
    source=("General Census of Population and Housing (RGPH) 2006, État et structure de la "
            "population, Tableaux A5.2 and A5.3 (INSD)"),
    how="census, 2006, main language spoken",
    parts=[dict(covers="Everyone aged 3 and over", source="2006 census, main language spoken",
                rest=True)],
    grain="45 provinces, 276,000 people on average",
    gap=("children under three, who are not in the tables (about 1.4 million); and 192,924 "
         "people whose language was not recorded"),
    view=[-5.6, 9.3, 2.5, 15.2],
    counts=_counts,
    mappings=["bf2006"],
    place=RD_GEO / "bf" / "bf_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "The census asked each person the language they mainly speak, which need not be the "
        "first language they learnt; Dioula, the trade language of the west, is the main "
        "language of 36% in Houet, the province of Bobo-Dioulasso. The figures are from 2006: "
        "the 2019 census gives languages for the whole country only. Fighting in the north "
        "and east since 2015 has displaced many people, and the map shows where they lived "
        "before it. 5.0% named a Burkinabè language the census does not print by name; they "
        "are drawn as other African languages. Foreign languages are printed by province only "
        "as African and non-African totals, and are split here in their region's proportions."),
)
