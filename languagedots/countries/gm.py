# Gambia. 2013 census ethnic group by LGA (Spatial Distribution report, Annex G), read as
# language and moved by Afrobarometer R7 (2018) mother-tongue retention (sources/gm_census.py);
# every row `derived`. On religiondots' Kontur hexes for the 8 LGAs. The record is sources/gm.md.
from _shared import *  # noqa: F401,F403

LGAS = 8
CENSUS_2013 = 1_857_181


def _counts():
    import gm2013
    df = pd.read_csv(NORM / "gm.csv", dtype={"geo_id": str})
    if df["geo_id"].nunique() != LGAS:
        raise SystemExit(f"gm.csv: {df['geo_id'].nunique()} LGAs, expected {LGAS} -- "
                         "re-run sources/gm_census.py")
    if int(df["count"].sum()) != CENSUS_2013:
        raise SystemExit(f"gm.csv sums to {int(df['count'].sum()):,}, not {CENSUS_2013:,}")
    df["node"] = df["source_category"].map(gm2013.resolve)
    missing = sorted(set(df.loc[df["node"].isna(), "source_category"]))
    if missing:
        raise SystemExit(f"gm.csv answers with no node: {missing}")
    lut = pd.read_csv(RD_GEO / "gm" / "gm_lookup.csv", dtype=str)
    if set(df["geo_id"]) != set(lut["unit"]):
        raise SystemExit("gm.csv LGAs do not match religiondots' gm_lookup.csv")
    df["unit"] = df["geo_id"]
    df = df[df["count"] > 0]
    out = by_unit(df)
    out["tier"] = "derived"
    return out


ENTRY = dict(
    name="Gambia",
    source=("2013 Population and Housing Census, Spatial Distribution of Population and "
            "Socio-Cultural Characteristics, Annex G (ethnic group by Local Government Area; "
            "Gambia Bureau of Statistics); Afrobarometer round 7 (2018), ethnic group and "
            "mother tongue"),
    how=("census, 2013, ethnic group read as language; the share of each group naming another "
         "mother tongue in the Afrobarometer (2018, 1,200 adults) moved onto it"),
    parts=[dict(covers="Everyone",
                source="2013 census, ethnic group, adjusted by Afrobarometer 2018 mother tongue",
                rest=True)],
    grain="8 Local Government Areas, 230,000 people on average",
    gap="non-Gambians, 110,749 (6%), drawn on their LGA's Gambian mix",
    view=[-16.9, 13.0, -13.7, 13.9],
    counts=_counts,
    mappings=["gm2013"],
    place=RD_GEO / "gm" / "gm_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "The Gambia's 2013 census asked Gambians their ethnic group but not their language. "
        "This map reads each group in each of the eight Local Government Areas as its "
        "language, then moves the share of each group who named another mother tongue in the "
        "2018 Afrobarometer survey onto it (Serer keep 57%, the rest mostly Wolof). Wolof "
        "spoken at home is more common than the mother-tongue figures drawn here. "
        "Non-Gambians, 6% of the population, are drawn on their area's mix."),
)
