# India. Census 2011 C-16 (sources/in_c16.py), on religiondots' sub-district settlement layer.
from _shared import *  # noqa: F401,F403


def _counts():
    import in2011
    df = pd.read_csv(NORM / "in.csv", dtype={"geo_id": str, "source_code": str})
    df = df[df["geo_level"] == "subdistrict"]
    df["node"] = [in2011.resolve(c, g[:2]) for c, g in zip(df["source_code"], df["geo_id"])]
    return by_unit(df.rename(columns={"geo_id": "unit"}))


ENTRY = dict(
    name="India",
    source="Census of India 2011, table C-16 (Office of the Registrar General)",
    how="census, 2011, mother tongue",
    parts=[dict(covers="Everyone", source="2011 census, mother tongue", rest=True)],
    grain="5,988 sub-districts, 200,000 people on average",
    view=[67.5, 6.5, 97.8, 36.0],
    counts=_counts,
    mappings=["in2011"],
    place=RD_GEO / "in" / "in_places.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public="",
)
