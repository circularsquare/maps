# Bahrain. No census or survey asks language. 2020 census: Bahrainis split Baharna Arabic / Gulf
# Arabic by religiondots' modelled Shia share, non-Bahrainis by governorate, nationality group and
# sex at UN DESA 2020's origins read as languages (sources/bh_build.py, sources/gulf_mix.py), 4
# governorates on religiondots' Kontur 400 m hexes. Every row derived. Record: sources/bh.md.
from _shared import *  # noqa: F401,F403

CENSUS_2020 = 1_501_635


def _gp():
    """sources/gulf_place.py: the Gulf citizen / foreign placement."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("gulf_place", ROOT / "sources" / "gulf_place.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rows():
    import bh2020
    df = pd.read_csv(NORM / "bh.csv")
    if df["geo_id"].nunique() != 4:
        raise SystemExit(f"bh.csv: {df['geo_id'].nunique()} governorates, expected 4")
    if int(df["count"].sum()) != CENSUS_2020:
        raise SystemExit(f"bh.csv sums to {df['count'].sum():,}, expected {CENSUS_2020:,}")
    lut = pd.read_csv(RD_GEO / "bh" / "bh_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    if df["unit"].isna().any():
        raise SystemExit("bh: governorates missing from religiondots' bh_lookup.csv")
    df["node"] = df["source_category"].map(bh2020.resolve)
    return df


def _counts():
    out = by_unit(_rows())
    out["tier"] = "derived"
    return out


def _weight(place):
    """citizens and foreign residents on their own weights inside each unit (gulf_place.py)"""
    gp = _gp()
    return gp.GulfWeighter("bh", place, *gp.citizen_tables(_rows()))


ENTRY = dict(
    name="Bahrain",
    source=("Information and eGovernment Authority, Census 2020: population by governorate, "
            "nationality group and sex; Bahraini Shia by governorate as modelled for this "
            "project's religion map (Arab Barometer 2009 and the Ja'fari and Sunni endowments' "
            "mosque registers); UN DESA, International Migrant Stock 2020, migrants by origin "
            "and sex; Indians by state from the Kerala Migration Survey 2023 and India's "
            "emigration clearances 2011 to 2017; Pakistanis by province from the Bureau of "
            "Emigration's registrations 2019 to 2021; home languages from the censuses of India "
            "(2011), Pakistan (2023) and other countries on this map"),
    how=("census, 2020, no language question; Shia Bahrainis drawn as Baharna Arabic, other "
         "Bahrainis as Gulf Arabic, foreign residents by country of origin"),
    parts=[
        dict(covers="Shia Bahrainis",
             source="2020 census citizens, Shia share from Arab Barometer 2009 placed by mosque "
                    "registers, drawn as Baharna Arabic", people=406_452),
        dict(covers="Other Bahrainis", source="2020 census citizens, drawn as Gulf Arabic",
             people=305_910),
        dict(covers="Foreign residents",
             source="2020 census, region of origin; countries from UN DESA 2020, Indian states "
                    "and Pakistani provinces from emigration records, drawn on their home "
                    "languages", rest=True),
    ],
    grain="4 governorates, 375,000 people on average",
    gap=("Bahrainis whose first language is Persian (some of the Ajam), whom no source counts"),
    view=[50.3, 25.75, 50.8, 26.35],
    counts=_counts,
    mappings=["bh2020"],
    place=RD_GEO / "bh" / "bh_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=_weight,
    note_public=(
        "Nobody in Bahrain is asked their language, so nothing here is a count of a language. "
        "The 2020 census counted 712,000 Bahraini citizens and 789,000 foreign residents in "
        "each governorate, the foreign residents by region of origin and sex. Bahrain's two "
        "Arabic dialects follow sect: the Baharna, who are Shia, speak Baharna Arabic, and "
        "Sunni Bahrainis speak Gulf Arabic. Nobody counts sect either, so the split is an "
        "estimate: a 2009 survey's 58% Shia, placed by governorate in proportion to the Shia and "
        "Sunni mosques each holds. The Ajam, Shia of Persian descent, are counted with the "
        "Baharna; how many still speak Persian is not known. Foreign residents are drawn by "
        "country of origin from the UN's estimate of Bahrain's migrants, Indians by where in "
        "India Gulf migrants come from and Pakistanis by province. Inside each governorate, "
        "foreign residents are placed more heavily in dense districts and wholly in industrial "
        "areas and labour camps, and Bahrainis in the rest; that is an estimate, not a count."),
)
