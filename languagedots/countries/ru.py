# Russia. 2021 census native language by federal subject, urban and rural (sources/ru_census.py),
# on religiondots' 3km Kontur hexes split into urban and rural by density (sources/ru_geo.py), and
# for Crimea and Sevastopol on Kontur UA's 400 m hexes cut out of Ukraine's layer (Anita's ruling,
# 2026-10-05). sources/ru.md is the record.
from _shared import *  # noqa: F401,F403


def _counts():
    import ru2021
    df = pd.read_csv(NORM / "ru.csv")
    # the 85 subjects the census counted, Crimea (UA-43) and Sevastopol (UA-40) among them;
    # the `country` rows are the same people again
    df = df[(df["geo_level"] == "subject") & (df["area"].isin(["urban", "rural"]))].copy()
    df["node"] = df["source_category"].map(ru2021.resolve)
    df = df[df["node"].notna()]            # "native language not stated" is the gap
    df["unit"] = df["geo_id"] + "/" + df["area"].str[0]
    return df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()


ENTRY = dict(
    name="Russia",
    source=("All-Russian Population Census 2020 (held in 2021), Volume 5, Table 6, population by "
            "native language (Rosstat)"),
    how="census, 2021, native language",
    parts=[dict(covers="Everyone", source="2021 census, native language", rest=True)],
    grain=("85 federal subjects (Crimea and Sevastopol included), each split urban and rural: "
           "168 units, 880,000 people on average"),
    gap="16.6 million people (11.3%) with no native language recorded, most counted from records",
    # religiondots' frame: it stops at the antimeridian, leaving Chukotka's sliver beyond it to a pan
    view=[19.0, 41.0, 180.0, 78.0],
    counts=_counts,
    mappings=["ru2021"],
    place=GEO / "ru" / "ru_grid_3km.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "The 2021 census asked each person's native language, which in the countries of the "
        "former Soviet Union leans towards identity rather than everyday use: people often name "
        "the language of their nationality while speaking Russian at home. Rosstat publishes the "
        "answers only for each federal subject, split into urban and rural, so all the dots in a "
        "subject's towns are drawn from one mixture and all those in its countryside from "
        "another; the densest grid cells stand in for the towns. Crimea and Sevastopol are "
        "drawn from this census, which has counted them since 2014; there 206,000 people named "
        "Crimean Tatar and 59,000 Tatar. 275,000 people said only \"Mordvin\" and 318,000 only "
        "\"Mari\", and are drawn that way. 11.3% have no native language recorded, from 27% in "
        "Khanty-Mansi and 24% in Moscow to under 3% in Tatarstan and Chechnya; they are not "
        "drawn."),
)
