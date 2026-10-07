# Angola. RGPH 2024, Quadro 6.2 (mother tongue) of the 21 provincial volumes, by municipality
# (sources/ao_rgph.py), on religiondots' Kontur hexes for the 326 municipalities of Lei 14/24.
# The record is sources/ao.md.
from _shared import *  # noqa: F401,F403


def _counts():
    import ao2024
    df = pd.read_csv(NORM / "ao.csv", dtype={"geo_id": str})
    df = df[(df["geo_level"] == "municipality") & (df["tier"] != "universe")].copy()
    df["node"] = df["source_category"].map(ao2024.NAMES)
    unresolved = sorted(set(df.loc[df["node"].isna(), "source_category"]) - set(ao2024.EXCLUDED))
    if unresolved:
        raise SystemExit(f"ao.csv categories that resolve to nothing: {unresolved}")
    df = df[df["node"].notna() & (df["count"] > 0)].copy()
    # religiondots joins its municipality ids to its polygons with this lookup
    lut = pd.read_csv(RD_GEO / "ao" / "ao_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    if df["unit"].isna().any() or df["unit"].nunique() != 326:
        raise SystemExit(f"ao: {df.loc[df['unit'].isna(), 'geo_id'].nunique()} municipalities "
                         f"missing from religiondots' ao_lookup.csv, {df['unit'].nunique()} units")
    return df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()


ENTRY = dict(
    name="Angola",
    source="Recenseamento Geral da População e Habitação 2024, Resultados Definitivos, Quadro "
           "6.2 of the 21 provincial volumes and of the national volume (INE Angola)",
    how="census, 2024, mother tongue, ages 2 and over",
    parts=[dict(covers="Everyone aged 2 and over", source="2024 census, mother tongue",
                rest=True)],
    grain="326 municipalities, 106,000 people on average; Moxico Leste by province",
    gap="147,468 people who did not know (0.4%); and the 1.68 million children under 2, who "
        "were not asked",
    view=[11.6, -18.1, 24.1, -4.4],
    counts=_counts,
    mappings=["ao2024"],
    place=RD_GEO / "ao" / "ao_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "The census asked each person's mother tongue, the first language they learned. "
        "Portuguese is 45% of Angolans aged 2 and over, and most of those are in the towns. "
        "INE folded some smaller languages into bigger ones before publishing: Lunda is "
        "counted as Chokwe, Luchazi and Mbunda as Ngangela, Herero as Kwanyama, and Songo as "
        "Kimbundu. Foreign languages, mostly Lingala, are one column in the provincial reports "
        "and are shared out in each province's mix from the national report. The Moxico "
        "Leste report has no language table, so its nine municipalities all show the "
        "province's mix."),
)
