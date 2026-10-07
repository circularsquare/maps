# Marshall Islands. 2021 census, languages spoken (several allowed), national: Marshallese, and
# the 4% who do not speak it on `other` (sources/mh_census.py). Placed on religiondots' Kontur
# hexes for the whole country (read-only). Record: sources/mh.md.
from _shared import *  # noqa: F401,F403


def _counts():
    import mh2021
    df = pd.read_csv(NORM / "mh.csv")
    if int(df["count"].sum()) != 42_418:
        raise SystemExit("mh.csv: expected 42,418 people -- run sources/mh_census.py")
    df["node"] = df["source_category"].map(mh2021.resolve)
    df["unit"] = "MH"   # religiondots' mh_hexes.gpkg: one unit
    out = by_unit(df)
    out["tier"] = "derived"
    return out


ENTRY = dict(
    name="Marshall Islands",
    source="2021 Census of Population and Housing, Analytical Report, Table 3.3 (RMI EPPSO, SPC)",
    how=("census, 2021, languages spoken, aged 5 and over, several allowed; national share "
         "applied to all ages"),
    parts=[dict(covers="Everyone",
                source="2021 census, speaks Marshallese or not, national share",
                rest=True)],
    grain="the country as one unit, 42,418 people",
    view=[160.7, 4.4, 172.3, 14.8],
    counts=_counts,
    mappings=["mh2021"],
    place=RD_GEO / "mh" / "mh_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "The 2021 census asked everyone aged 5 and over which languages they speak. 96% speak "
        "Marshallese, and they are drawn as Marshallese speakers. About a quarter also speak "
        "another language, mostly English learned at school, which is not drawn. The 4% who "
        "do not speak Marshallese are drawn as language not named, since the census report "
        "does not say which languages they speak. The share is national and applied to all "
        "ages."),
)
