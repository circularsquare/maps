# Poland. NSP 2021, language used at home, by gmina (sources/pl_nsp.py), on Kontur hexes keyed to
# religiondots' 2,477 gminas (sources/pl_geo.py). The record is sources/pl.md.
from _shared import *  # noqa: F401,F403


def _counts():
    import pl2021
    df = pd.read_csv(NORM / "pl.csv", dtype={"geo_id": str})
    df = df[df["geo_level"] == "gmina"]
    if df["geo_id"].nunique() != 2477:
        raise SystemExit(f"pl.csv: {df['geo_id'].nunique()} gminas, expected 2,477")
    df["node"] = df["source_category"].map(pl2021.resolve)
    unresolved = sorted(set(df.loc[df["node"].isna(), "source_category"]) - pl2021.NOT_STATED)
    if unresolved:
        raise SystemExit(f"pl.csv categories that resolve to nothing: {unresolved}")
    df = df[df["node"].notna() & (df["count"] > 0)]
    # the placement layer is keyed on TERYT's first six digits (the seventh is the gmina type)
    df["unit"] = df["geo_id"].str[:6]
    # every row is a person shared across the languages they named (spec §3.6)
    return df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()


ENTRY = dict(
    name="Poland",
    source="Narodowy Spis Powszechny 2021, language used at home by gmina (Statistics Poland)",
    how="census, 2021, language used at home, several allowed; each person shared across "
        "the languages they named",
    parts=[dict(covers="Everyone",
                source="2021 census, language used at home, each person shared across the "
                       "languages they named",
                rest=True)],
    grain="2,477 gminas, 15,000 people on average",
    gap="32,381 people, 0.1%, whose home language was not established",
    view=[14.0, 48.9, 24.2, 55.0],
    counts=_counts,
    mappings=["pl2021"],
    place=GEO / "pl" / "pl_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "The census let each person name Polish and up to two other languages used at home. "
        "1.6 million people named more than one, almost all of them Polish and another, so "
        "each person is drawn shared between the languages they named: someone who named "
        "Silesian and Polish counts half to each. That is why Silesian, named by 467,000 "
        "people, is drawn as about 254,000. Statistics Poland does not print a gmina's figure "
        "for a language under 10 people; those are placed within their powiat by population."),
)
