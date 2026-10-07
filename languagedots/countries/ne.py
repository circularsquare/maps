# Niger. RGP/H 2001 ethnic group by département (Tableau 30), read as language and moved by
# Afrobarometer R5-R6 retention, applied to each région's 2012 census population
# (sources/ne_census.py); every row `derived`. On religiondots' Kontur hexes for the 8 régions.
# The record is sources/ne.md.
from _shared import *  # noqa: F401,F403

REGIONS = 8
CENSUS_2012 = 17_138_707


def _counts():
    import ne2001
    df = pd.read_csv(NORM / "ne.csv", dtype={"geo_id": str})
    if df["geo_id"].nunique() != REGIONS:
        raise SystemExit(f"ne.csv: {df['geo_id'].nunique()} régions, expected {REGIONS} -- "
                         "re-run sources/ne_census.py")
    if int(df["count"].sum()) != CENSUS_2012:
        raise SystemExit(f"ne.csv sums to {int(df['count'].sum()):,}, not {CENSUS_2012:,}")
    df["node"] = df["source_category"].map(ne2001.resolve)
    missing = sorted(set(df.loc[df["node"].isna(), "source_category"]))
    if missing:
        raise SystemExit(f"ne.csv answers with no node: {missing}")
    lut = pd.read_csv(RD_GEO / "ne" / "ne_lookup.csv", dtype=str)
    if set(df["geo_id"]) != set(lut["unit"]):
        raise SystemExit("ne.csv régions do not match religiondots' ne_lookup.csv")
    df["unit"] = df["geo_id"]
    df = df[df["count"] > 0]
    out = by_unit(df)
    out["tier"] = "derived"
    return out


ENTRY = dict(
    name="Niger",
    source=("Recensement Général de la Population et de l'Habitat 2001, État et structure de la "
            "population, Tableau 30 (ethnic group by département), and the 2012 census's région "
            "populations (Institut National de la Statistique); Afrobarometer round 7 (2018), "
            "ethnic group and mother tongue"),
    how=("census, 2001, ethnic group read as language, corrected by the Afrobarometer's 2018 "
         "mother-tongue answers; shares applied to 2012 populations"),
    parts=[dict(covers="Everyone",
                source="2001 census, ethnic group read as language, corrected by Afrobarometer "
                       "2018 (about 1,200 adults), on 2012 populations",
                rest=True)],
    grain="8 régions, 2.1 million people on average",
    gap=("5,951 people of other or undeclared ethnic groups in 2001 are left out of the shares, "
         "and foreign residents are drawn on the Nigerien mix"),
    view=[0.1, 11.6, 16.0, 23.6],
    counts=_counts,
    mappings=["ne2001"],
    place=RD_GEO / "ne" / "ne_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "Niger's censuses do not ask about language, and the 2012 census did not ask about "
        "ethnic group either. This map takes each région's ethnic groups from the 2001 census, "
        "reads each group as its language (Tuareg as Tamajaq, Fulani as Fulfulde) and applies "
        "the shares to the 2012 population. In the Afrobarometer survey of 2018, 6% of Tuareg, "
        "10% of Kanuri and 2% of Fulani named another mother tongue, and those shares are moved "
        "onto it. Many more speak Hausa, the language most Nigeriens use with each other, but "
        "this map shows mother tongues."),
)
