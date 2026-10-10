# India. Census 2011 C-16 (sources/in_c16.py), on religiondots' sub-district settlement layer.
from _shared import *  # noqa: F401,F403


def _counts():
    import in2011
    df = pd.read_csv(NORM / "in.csv", dtype={"geo_id": str, "source_code": str})
    df = df[df["geo_level"] == "subdistrict"]
    df["node"] = [in2011.resolve(c, g[:2], g[:5]) for c, g in zip(df["source_code"], df["geo_id"])]
    # Bengali drawn as Sylheti by district (in2011.SYLHETI_DISTRICTS) is a place split: derived
    df["tier"] = df["node"].str.endswith(".sylheti").map({True: "derived", False: "measured"})
    return df.rename(columns={"geo_id": "unit"}).groupby(["unit", "node", "tier"], as_index=False)["count"].sum()


ENTRY = dict(
    name="India",
    source="Census of India 2011, table C-16 (Office of the Registrar General)",
    how="census, 2011, mother tongue; Bengali in the Barak Valley and north Tripura drawn as Sylheti",
    parts=[dict(covers="Sylheti", source="2011 census Bengali in Cachar, Karimganj, Hailakandi and "
                "North Tripura, drawn as Sylheti by place", nodes=["indoeuropean.indoaryan.eastern.sylheti"]),
           dict(covers="Everyone else", source="2011 census, mother tongue", rest=True)],
    grain="5,988 sub-districts, 200,000 people on average",
    view=[67.5, 6.5, 97.8, 36.0],
    counts=_counts,
    mappings=["in2011"],
    place=RD_GEO / "in" / "in_places.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public="The census counts Sylheti as Bengali. In the Barak Valley (Cachar, Karimganj, "
                "Hailakandi) and North Tripura, where Sylheti is the local speech, Bengali speakers "
                "are drawn as Sylheti, as across the border in Sylhet.",
)
