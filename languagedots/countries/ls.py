# Lesotho. No census language question (2016 has none): Afrobarometer R4-R9 (2008-2022; 7,197
# respondents) home-language shares per district on the 2016 census district counts
# (sources/ls_afro.py); every row `modelled`. Placed on religiondots' Kontur hexes (read-only).
# Record: sources/ls.md.
from _shared import *  # noqa: F401,F403

DISTRICTS = 10
CENSUS_2016 = 2_007_201


def _counts():
    import ls2022
    df = pd.read_csv(NORM / "ls.csv", dtype={"geo_id": str})
    if df["geo_id"].nunique() != DISTRICTS or int(df["count"].sum()) != CENSUS_2016:
        raise SystemExit("ls.csv is not 10 districts summing to the 2016 census -- run "
                         "sources/ls_afro.py")
    df["node"] = df["source_category"].map(ls2022.resolve)
    df["unit"] = df["geo_id"]   # religiondots' ls hexes: unit == district pcode
    out = by_unit(df[df["count"] > 0])
    out["tier"] = "modelled"
    return out


ENTRY = dict(
    name="Lesotho",
    source=("Afrobarometer rounds 4 to 9 (2008 to 2022), home language, on the 2016 census "
            "district populations (Bureau of Statistics)"),
    how="survey, 2008 to 2022 pooled, home language, on 2016 census district populations",
    parts=[
        dict(covers="Everyone but English speakers",
             source="Afrobarometer 2008-22, home language, 7,197 adults, shares per district",
             rest=True),
        dict(covers="English",
             source="Afrobarometer 2018, mother tongue, one national share",
             nodes=["indoeuropean.germanic.english"]),
    ],
    grain="10 districts, 201,000 people on average",
    view=[27.0, -30.7, 29.5, -28.55],
    counts=_counts,
    mappings=["ls2022"],
    place=RD_GEO / "ls" / "ls_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "Lesotho's census does not ask about language, so this map uses the Afrobarometer "
        "survey, which asked 7,197 people between 2008 and 2022 which language they speak at "
        "home. 98% said Sesotho. Xhosa (Sethepu) and Phuthi are spoken in the south, in "
        "Quthing and Qacha's Nek. The interviews were held in Sesotho or English, which may "
        "undercount both."),
)
