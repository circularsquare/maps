# Fiji. No language question: 2007 census Fijians / Indians / Others per province read as
# language (sources/fj_census.py). Placed on religiondots' Kontur hexes for the 15 provinces
# (read-only; fj_lookup.csv maps the census province numbers to their units). Record: sources/fj.md.
from _shared import *  # noqa: F401,F403


def _counts():
    import fj2007
    df = pd.read_csv(NORM / "fj.csv", dtype={"geo_id": str})
    if int(df["count"].sum()) != 837_271 or df["geo_id"].nunique() != 15:
        raise SystemExit("fj.csv: expected 837,271 people in 15 provinces -- run sources/fj_census.py")
    lut = pd.read_csv(RD_GEO / "fj" / "fj_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    if df["unit"].isna().any():
        raise SystemExit("fj.csv provinces missing from religiondots' fj_lookup.csv")
    df["node"] = df["source_category"].map(fj2007.resolve)
    out = by_unit(df)
    out["tier"] = "derived"
    return out


ENTRY = dict(
    name="Fiji",
    source=("2007 Census of Population and Housing, Analytical Report, Tables I-5a to I-5d "
            "(Fiji Bureau of Statistics)"),
    how=("no language question: census 2007 ethnicity read as language, iTaukei as Fijian and "
         "Indo-Fijians as Fiji Hindi"),
    parts=[dict(covers="Everyone", source="2007 census, ethnic group, drawn as language",
                rest=True)],
    grain="15 provinces, 56,000 people on average",
    view=[176.8, -19.3, 180.6, -15.6],
    counts=_counts,
    mappings=["fj2007"],
    place=RD_GEO / "fj" / "fj_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "Fiji's census has not asked about language since 1946, so ethnicity from the 2007 "
        "census is drawn as language: iTaukei Fijians as Fijian speakers and Indo-Fijians as "
        "Fiji Hindi speakers. Rotumans on Rotuma are drawn as Rotuman and the Banabans of Rabi "
        "Island as Gilbertese; other minorities are drawn as language not named. The 2017 "
        "census did not publish ethnicity by province, so the figures are from 2007, when "
        "Indo-Fijians were 37.5% of the population. Western Fijian is not separated from "
        "Fijian."),
)
