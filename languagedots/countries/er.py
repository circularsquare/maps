# Eritrea. No census, ever. EPHS 2010's national ethnic shares read as language, on religiondots'
# zoba populations (UN WPP 2020 divided in the survey's proportions), each group drawn in its home
# zobas (sources/er_ephs.py). Placed on religiondots' Meta 2020 cells (read-only). Record: sources/er.md.
from _shared import *  # noqa: F401,F403

TOTAL = 3_291_271


def _counts():
    import er2010
    df = pd.read_csv(NORM / "er.csv", dtype={"geo_id": str})
    if df["geo_id"].nunique() != 6 or int(df["count"].sum()) != TOTAL:
        raise SystemExit(f"er.csv: {df['geo_id'].nunique()} zobas, {df['count'].sum():,} people "
                         "-- run sources/er_ephs.py")
    df["node"] = df["source_category"].map(er2010.resolve)
    df["unit"] = df["geo_id"]   # religiondots' er_lookup.csv: geo_id == unit
    return df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()


ENTRY = dict(
    name="Eritrea",
    source=("Eritrea Population and Health Survey 2010 (National Statistics Office and Fafo), "
            "Table 3-1, ethnicity of women and men aged 15-49; UN World Population Prospects 2024 "
            "estimate for 2020, divided between the zobas in the survey's proportions"),
    how=("survey, 2010, ethnic group read as language, national shares; each group drawn in its "
         "home zobas"),
    parts=[dict(covers="Everyone", source="2010 Population and Health Survey, ethnic group, "
                "aged 15-49", rest=True)],
    grain="the whole country, drawn on 6 zobas",
    view=[36.4, 12.3, 43.2, 18.1],
    counts=_counts,
    mappings=["er2010"],
    place=RD_GEO / "er" / "er_cells.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "Eritrea has never held a census, and no survey asks about language. The 2010 "
        "Population and Health Survey asked about 34,000 women and men their ethnic group, and "
        "each group is drawn here as its own language (Hedareb as Beja, Rashaida as Arabic). "
        "No source says how many in each group speak another language at home. The shares are "
        "national only, so each group is drawn in the zobas it lives in, with Tigrinya filling "
        "the rest; how the groups divide inside a mixed zoba is a guess by population. The "
        "population is the UN's estimate for 2020, 3.3 million."),
)
