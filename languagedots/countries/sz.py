# Eswatini. No census language question (2007 and 2017 ask literacy only): Afrobarometer R5-R9
# (2011-2022; 6,000 respondents) home-language shares per region on the 2017 census region counts
# (sources/sz_afro.py); every row `modelled`. Placed on religiondots' WorldPop cells (read-only).
# Record: sources/sz.md.
from _shared import *  # noqa: F401,F403

REGIONS = 4
CENSUS_2017 = 1_093_238


def _counts():
    import sz2022
    df = pd.read_csv(NORM / "sz.csv", dtype={"geo_id": str})
    if df["geo_id"].nunique() != REGIONS or int(df["count"].sum()) != CENSUS_2017:
        raise SystemExit("sz.csv is not 4 regions summing to the 2017 census -- run "
                         "sources/sz_afro.py")
    df["node"] = df["source_category"].map(sz2022.resolve)
    df["unit"] = df["geo_id"]   # religiondots' sz cells: unit SZ1..SZ4
    out = by_unit(df[df["count"] > 0])
    out["tier"] = "modelled"
    return out


ENTRY = dict(
    name="Eswatini",
    source=("Afrobarometer rounds 5 to 9 (2011 to 2022), home language, on the 2017 census "
            "region populations (Central Statistical Office)"),
    how=("survey, 2011 to 2022 pooled, home language, 6,000 respondents, on 2017 census region "
         "populations; English from round 7's question on mother tongue, nationally"),
    parts=[
        dict(covers="English",
             source="Afrobarometer round 7, mother tongue, national share",
             nodes=["indoeuropean.germanic.english"]),
        dict(covers="Other languages",
             source="Afrobarometer 2011-2022, home language, about 6,000 adults, on 2017 "
                    "census region populations",
             rest=True),
    ],
    grain="4 regions, 273,000 people on average",
    view=[30.75, -27.35, 32.15, -25.7],
    counts=_counts,
    mappings=["sz2022"],
    place=RD_GEO / "sz" / "sz_cells.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "Eswatini's census asks which languages people can read and write, not which they "
        "speak, so this map uses the Afrobarometer survey, which asked 6,000 people between "
        "2011 and 2022 which language they speak at home. 96% said siSwati. English was named "
        "by about 1% in the earlier rounds and 8% in 2022; this map shows first languages, so "
        "English is drawn from a separate question on mother tongue, asked once, where 0.6% "
        "named it. Inside each region the dots follow where people live, not where each "
        "language is spoken."),
)
