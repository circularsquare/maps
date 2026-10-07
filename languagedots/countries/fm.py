# Federated States of Micronesia. 2010 census, Table B10A, language mainly spoken at home,
# persons 3+, by state (sources/fm_census.py). Placed on religiondots' Kontur hexes re-keyed to
# the four states (data/geo/fm/fm_hexes.gpkg). Record: sources/fm.md.
from _shared import *  # noqa: F401,F403


def _counts():
    import fm2010
    df = pd.read_csv(NORM / "fm.csv")
    if int(df["count"].sum()) != 95_544 or df["geo_id"].nunique() != 4:
        raise SystemExit("fm.csv: expected 95,544 people in 4 states -- run sources/fm_census.py")
    df["node"] = df["source_category"].map(fm2010.resolve)
    df["unit"] = df["geo_id"]
    out = by_unit(df)
    out["tier"] = "measured"
    return out


ENTRY = dict(
    name="Micronesia",
    source=("2010 FSM Census of Population and Housing, Basic Tables, Table B10A (FSM Statistics "
            "Division)"),
    how="census 2010, language mainly spoken at home, people aged 3 and over",
    parts=[dict(covers="Everyone aged 3 and over", source="2010 census, language mainly "
                "spoken at home", rest=True)],
    grain="4 states, 24,000 people on average",
    gap="children under 3, who were not asked",
    view=[137.3, 0.8, 163.4, 10.2],
    counts=_counts,
    mappings=["fm2010"],
    place=ROOT / "data" / "geo" / "fm" / "fm_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "The 2010 census asked which language each person mainly speaks at home. Each state "
        "has its own language: Chuukese, Pohnpeian, Kosraean and Yapese, with Chuukese the "
        "largest by far. The census counts Yap's outer-island languages as one answer, and "
        "Pohnpei's Polynesian outliers (Nukuoro and Kapingamarangi) as another. Within a state "
        "the dots follow population, so outer islands are not separated from the main island. "
        "The 2023 census published no language table."),
)
