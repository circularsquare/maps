# Finland. Population register, language by municipality, 31 Dec 2025 (sources/fi_register.py),
# on Statistics Finland's own 1 km population grid keyed to the same 308 municipalities
# (sources/fi_geo.py). The record is sources/fi.md.
from _shared import *  # noqa: F401,F403


def _counts():
    import fi2025
    df = pd.read_csv(NORM / "fi.csv", dtype={"geo_id": str})
    df = df[df["geo_level"] == "municipality"]
    if df["geo_id"].nunique() != 308:
        raise SystemExit(f"fi.csv: {df['geo_id'].nunique()} municipalities, expected 308")
    df["node"] = df["source_category"].map(fi2025.resolve)
    unresolved = sorted(set(df.loc[df["node"].isna(), "source_category"]) - fi2025.NOT_STATED)
    if unresolved:
        raise SystemExit(f"fi.csv categories that resolve to nothing: {unresolved}")
    df = df[df["node"].notna() & (df["count"] > 0)]
    df["unit"] = df["geo_id"]
    # `measured`: printed cells; `derived`: cells under 10 that Statistics Finland suppresses,
    # estimated from the maakunta's unsuppressed figure (sources/fi_register.py)
    return df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()


ENTRY = dict(
    name="Finland",
    source="Population structure, table 11rm, language by municipality, 31 December 2025 "
           "(Statistics Finland, from the population register)",
    how="population register, 2025, mother tongue as registered, one per person",
    parts=[dict(covers="Everyone", source="population register, 2025, registered language",
                rest=True)],
    grain="308 municipalities, 18,000 people on average",
    gap="2,209 people, 0.04%, whose language the register does not know",
    view=[19.0, 59.6, 31.7, 70.2],
    counts=_counts,
    mappings=["fi2025"],
    place=GEO / "fi" / "fi_grid1km.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "Finland has no census question on language. Every resident has one language recorded "
        "in the population register, usually given by the parents when a birth is registered or "
        "by the person on arriving in the country, and it is seldom changed later. A child of a "
        "Finnish and Swedish speaking family is registered under one of the two, so the map shows "
        "one language per person and cannot show bilingual homes. Sami is one entry in the "
        "register, covering North, Inari and Skolt Sami. Figures under 10 people in a "
        "municipality are not published; those 21,700 people (0.4%) are estimated from their "
        "region's figure."),
)
