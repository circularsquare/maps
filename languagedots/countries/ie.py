# Ireland. Census of Population 2022 (CSO), by Small Area (sources/ie_census.py), drawn on
# religiondots' Small Area polygons, the units every count is on, so there is no placement layer and
# no population weight: one polygon per unit, as religiondots draws Ireland. The record is
# sources/ie.md.
from _shared import *  # noqa: F401,F403


def _counts():
    import ie2022
    df = pd.read_csv(NORM / "ie.csv")
    df["node"] = df["source_category"].map(ie2022.resolve)
    return df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()


ENTRY = dict(
    name="Ireland",
    source=("Census of Population 2022, Small Area Population Statistics themes 2 and 3 and "
            "PxStat F5029 (Central Statistics Office)"),
    how=("census, 2022, language other than English or Irish spoken at home; Irish drawn for "
         "those who speak it daily outside the education system; everyone else drawn as English"),
    parts=[
        dict(covers="People speaking another language at home",
             source="2022 census, language other than English or Irish spoken at home",
             people=751_507),
        dict(covers="Daily Irish speakers",
             source="2022 census, speaks Irish daily outside the education system",
             nodes=["indoeuropean.celtic.irish"]),
        dict(covers="Everyone else", source="drawn as English", rest=True),
    ],
    grain="18,919 Small Areas, 270 people on average",
    view=[-10.7, 51.4, -5.9, 55.45],
    counts=_counts,
    mappings=["ie2022"],
    place=RD_GEO / "ie" / "smallareas2022" / "SMALL_AREA_2022.shp",
    place_unit=lambda g: g["SA_GUID__1"].astype(str),
    place_weight=None,
    note_public=(
        "The census does not ask anyone's first language. It asks which language other than "
        "English or Irish a person speaks at home. Of the 1.87 million who say they can speak "
        "Irish, only the 72,000 who speak it daily outside school are drawn as Irish. Everyone "
        "else is drawn as English. By Small Area only Polish, French and Spanish are named; "
        "other home languages are placed by county."),
)
