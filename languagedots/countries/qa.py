# Qatar. No census or survey asks language. 2020 census: Qataris (estimated from the 10+ count) on
# Gulf Arabic, non-Qataris by sex at UN DESA's 2020 origins read as their languages
# (sources/qa_build.py, sources/gulf_mix.py), 8 municipalities, on religiondots' Kontur pieces
# re-keyed to the 2020 municipalities and weighted to the 2020 zones (sources/qa_geo.py).
# Every row derived. Record: sources/qa.md, sources/gulf.md.
from _shared import *  # noqa: F401,F403

CENSUS_2020 = 2_846_118


def _gp():
    """sources/gulf_place.py: the Gulf citizen / foreign placement."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("gulf_place", ROOT / "sources" / "gulf_place.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rows():
    import qa2020
    df = pd.read_csv(NORM / "qa.csv")
    if df["geo_id"].nunique() != 8:
        raise SystemExit(f"qa.csv: {df['geo_id'].nunique()} municipalities, expected 8")
    if int(df["count"].sum()) != CENSUS_2020:
        raise SystemExit(f"qa.csv sums to {df['count'].sum():,}, expected {CENSUS_2020:,}")
    df["unit"] = df["geo_id"]
    df["node"] = df["source_category"].map(qa2020.resolve)
    return df


def _counts():
    out = by_unit(_rows())
    out["tier"] = "derived"
    return out


def _weight(place):
    """citizens and foreign residents on their own weights inside each unit (gulf_place.py)"""
    gp = _gp()
    return gp.GulfWeighter("qa", place, *gp.citizen_tables(_rows()))


ENTRY = dict(
    name="Qatar",
    source=("Planning and Statistics Authority, Census 2020: population by municipality, zone "
            "and sex, Qataris aged 10 and over by municipality; UN DESA, International Migrant "
            "Stock 2020, migrants by origin and sex; Indians by state from the Kerala Migration "
            "Survey 2023 and India's emigration clearances 2011 to 2017; Pakistanis by province "
            "from the Bureau of Emigration's registrations 2019 to 2021; home languages from the "
            "censuses of India (2011), Pakistan (2023) and other countries on this map"),
    how=("census, 2020, no language question; Qataris drawn as Gulf Arabic, foreign residents by "
         "country of origin, each on its country's main language or language mix"),
    parts=[
        dict(covers="Qataris",
             source="2020 census, Qataris aged 10 and over, younger children estimated; drawn as "
                    "Gulf Arabic",
             people=340_298),
        dict(covers="Foreign residents",
             source="UN DESA migrant stock 2020 by origin and sex, each drawn on its country's "
                    "languages; Indians by state and Pakistanis by province from emigration "
                    "records",
             rest=True),
    ],
    grain="8 municipalities, 356,000 people on average",
    gap=("Qataris whose first language is not Arabic, whom no source counts, and Qataris under "
         "10, whose number the census does not print"),
    view=[50.7, 24.4, 51.7, 26.2],
    counts=_counts,
    mappings=["qa2020"],
    place=GEO / "qa" / "qa_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=_weight,
    note_public=(
        "Nobody in Qatar is asked their language, so nothing here is a count of a language. "
        "The 2020 census counted 246,000 Qataris aged 10 and over; with an estimate for younger "
        "children the map draws about 340,000 Qataris, 12% of the 2.85 million people, all as "
        "Gulf Arabic. The census publishes no nationality, so everyone else is drawn by "
        "country of origin from the UN's estimate of Qatar's migrants, men and women "
        "separately, each on their country's languages. Indians are split by home state "
        "(Keralites as Malayalam) and Pakistanis by province. Each municipality's foreign men "
        "and women are drawn at the national mix for their sex. Inside each municipality, "
        "foreign residents are placed more heavily in dense districts and wholly in industrial "
        "areas and labour camps, and Qataris in the rest; that is an estimate, not a count."),
)
