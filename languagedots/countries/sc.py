# Seychelles. Census 2022, Table B3.1a, first main language spoken at home, aged 3+, by 25 districts
# and 18 islands (sources/sc_census.py), on Kontur hexes cut by overlap along COD-AB's districts and
# islands (sources/sc_geo.py). Record: sources/sc.md.
from _shared import *  # noqa: F401,F403


def _counts():
    import sc2022
    df = pd.read_csv(NORM / "sc.csv", dtype={"geo_id": str})
    if df["geo_id"].nunique() != 43 or df["count"].sum() != 98_952:
        raise SystemExit("sc.csv: expected 43 units summing to 98,952; re-run sources/sc_census.py")
    df["node"] = df["source_category"].map(sc2022.resolve)
    df = df[df["node"].notna()]
    df["unit"] = df["geo_id"]
    return by_unit(df)


ENTRY = dict(
    name="Seychelles",
    source="Seychelles Population and Housing Census 2022, Table B3.1a (National Bureau of "
           "Statistics); districts and islands from OCHA COD-AB",
    how="census, 2022, first main language spoken at home, aged 3 and over",
    parts=[dict(covers="Everyone aged 3 and over",
                source="2022 census, main language spoken at home", rest=True)],
    grain="25 districts and 18 islands, 2,300 people on average",
    gap=("11,545 people aged 3 and over, 11.7%, with no language recorded, almost all of them "
         "not Seychellois"),
    view=[55.1, -4.85, 56.05, -3.65],
    counts=_counts,
    mappings=["sc2022"],
    place=GEO / "sc" / "sc_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "The 2022 census asked everyone aged 3 and over which language they spoke most often at "
        "home. Gujarati, Hindi and Tamil, 4.6% together, have grown with workers coming from "
        "India, the report says. 11.7% have no language recorded and are not drawn; almost "
        "all of them were not Seychellois, so the languages of foreign workers are "
        "undercounted, most of all in Cascade. On the Outer Islands the "
        "table records no Creole speaker at all, and 707 people as Gujarati speakers, the same "
        "number, island by island, as it counts Hindus. The map draws them as printed."),
)
