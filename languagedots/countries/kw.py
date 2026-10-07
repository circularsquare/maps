# Kuwait. No census or survey asks language. 2021 census: Kuwaitis on Gulf Arabic, non-Kuwaitis
# by area, sex and nationality group, each group at its nationalities read as languages
# (sources/kw_build.py, sources/gulf_mix.py), 157 areas on religiondots' 143 units (OSM areas,
# Kontur footprint). Every row derived. Record: sources/kw.md, sources/gulf.md.
from _shared import *  # noqa: F401,F403

CENSUS_2021 = 4_381_139     # 4,385,717 less the 4,578 whose area is not stated


def _gp():
    """sources/gulf_place.py: the Gulf citizen / foreign placement."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("gulf_place", ROOT / "sources" / "gulf_place.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rows():
    import kw2021
    df = pd.read_csv(NORM / "kw.csv", dtype={"geo_id": str})
    if df["geo_id"].nunique() != 157:
        raise SystemExit(f"kw.csv: {df['geo_id'].nunique()} areas, expected 157")
    if int(df["count"].sum()) != CENSUS_2021:
        raise SystemExit(f"kw.csv sums to {df['count'].sum():,}, expected {CENSUS_2021:,}")
    lut = pd.read_csv(RD_GEO / "kw" / "kw_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    if df["unit"].isna().any():
        raise SystemExit(f"kw: areas missing from religiondots' kw_lookup.csv: "
                         f"{sorted(df.loc[df['unit'].isna(), 'geo_id'].unique())}")
    if df["unit"].nunique() != 143:
        raise SystemExit(f"kw: {df['unit'].nunique()} units, expected 143")
    df["node"] = df["source_category"].map(kw2021.resolve)
    return df


def _counts():
    out = by_unit(_rows())
    out["tier"] = "derived"
    return out


def _weight(place):
    """citizens and foreign residents on their own weights inside each unit (gulf_place.py)"""
    gp = _gp()
    return gp.GulfWeighter("kw", place, *gp.citizen_tables(_rows()))


ENTRY = dict(
    name="Kuwait",
    source=("Central Statistical Bureau, Census 2021 (from the civil register): Kuwaitis and "
            "non-Kuwaitis by area, nationality groups by governorate; the Public Authority for "
            "Civil Information's 2014 nationality groups by locality and 2018 nationalities "
            "(as mirrored by the Gulf Labour Markets, Migration and Population programme); UN "
            "DESA, International Migrant Stock 2024; Indians by state from the Kerala Migration "
            "Survey 2023 and India's emigration clearances 2011 to 2017; Pakistanis by province "
            "from the Bureau of Emigration's registrations 2019 to 2021; home languages from the "
            "censuses of India (2011), Pakistan (2023) and other countries on this map"),
    how=("census, 2021, no language question; Kuwaitis drawn as Gulf Arabic, foreign residents "
         "by nationality, each on its country's main language or language mix"),
    parts=[
        dict(covers="Foreign residents",
             source="2021 census by area and origin, nationalities from the 2018 civil register "
                    "and UN estimates, drawn on each country's languages",
             people=2_892_704),
        dict(covers="Kuwaiti citizens", source="2021 census, drawn as Gulf Arabic", rest=True),
    ],
    grain="157 areas, 28,000 people on average",
    gap=("Kuwaitis whose first language is not Arabic, whom no source counts; 4,578 people "
         "whose area is not stated"),
    view=[46.5, 28.5, 48.5, 30.1],
    counts=_counts,
    mappings=["kw2021"],
    place=RD_GEO / "kw" / "kw_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=_weight,
    note_public=(
        "Nobody in Kuwait is asked their language, so nothing here is a count of a language. "
        "The 2021 census counted 1.5 million Kuwaiti citizens and 2.9 million foreign residents "
        "per area. Kuwaitis are all drawn as Gulf Arabic, so citizens who speak Persian or "
        "another language at home are not shown. Foreign residents are drawn by nationality on "
        "their country's languages, Indians by the states Gulf migrants come from and "
        "Pakistanis by province. The Bidoon, stateless residents who speak Gulf Arabic, are not "
        "named in any table and are drawn with the other Arab nationalities. Inside each area, "
        "foreign residents are placed wholly in its industrial land and labour camps where it "
        "has any, and Kuwaitis in the rest."),
)
