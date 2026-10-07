# Oman. No census or survey asks language. NCSI register, end 2024: Omanis on Omani Arabic per
# wilaya, expatriates per governorate by nationality (male workers, female workers, dependants)
# read as their languages (sources/om_build.py, sources/gulf_mix.py), 63 register wilayat on
# religiondots' 61 units and Kontur 400 m hexes. Every row derived. Record: sources/om.md.
from _shared import *  # noqa: F401,F403

REGISTER_2024 = 5_268_072


def _gp():
    """sources/gulf_place.py: the Gulf citizen / foreign placement."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("gulf_place", ROOT / "sources" / "gulf_place.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rows():
    import om2024
    df = pd.read_csv(NORM / "om.csv")
    if df["geo_id"].nunique() != 63:
        raise SystemExit(f"om.csv: {df['geo_id'].nunique()} wilayat, expected 63")
    if int(df["count"].sum()) != REGISTER_2024:
        raise SystemExit(f"om.csv sums to {df['count'].sum():,}, expected {REGISTER_2024:,}")
    lut = pd.read_csv(RD_GEO / "om" / "om_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    if df["unit"].isna().any():
        raise SystemExit(f"om: wilayat missing from religiondots' om_lookup.csv: "
                         f"{sorted(df.loc[df['unit'].isna(), 'geo_id'].unique())}")
    if df["unit"].nunique() != 61:
        raise SystemExit(f"om: {df['unit'].nunique()} units, expected 61")
    df["node"] = df["source_category"].map(om2024.resolve)
    return df


def _counts():
    out = by_unit(_rows())
    out["tier"] = "derived"
    return out


def _weight(place):
    """citizens and foreign residents on their own weights inside each unit (gulf_place.py)"""
    gp = _gp()
    return gp.GulfWeighter("om", place, *gp.citizen_tables(_rows()))


ENTRY = dict(
    name="Oman",
    source=("National Centre for Statistics and Information (NCSI), Statistical Year Book 2025: "
            "Omanis and expatriates by wilaya, expatriate workers by governorate, nationality "
            "and sex (end 2024); NCSI's mid-2018 population by nationality (as mirrored by the "
            "Gulf Labour Markets, Migration and Population programme); Indians by state from "
            "the Kerala Migration Survey 2023 and India's emigration clearances 2011 to 2017; "
            "Pakistanis by province from the Bureau of Emigration's registrations 2019 to 2021; "
            "home languages from the censuses of India (2011), Pakistan (2023) and other "
            "countries on this map"),
    how=("population register, 2024, no language question; Omanis drawn as Omani Arabic, "
         "foreign residents by nationality on their country's languages"),
    parts=[
        dict(covers="Omanis", source="2024 population register, drawn as Omani Arabic",
             people=2_984_793),
        dict(covers="Foreign residents",
             source="2024 register and 2018 counts by nationality, drawn on each country's "
                    "languages; Indians by state, Pakistanis by province",
             rest=True),
    ],
    grain="61 wilayat, 86,000 people on average",
    gap=("Omanis whose first language is not Arabic (Baluchi, Jibbali, Mehri, Kumzari, Swahili), "
         "whom no source counts"),
    view=[51.8, 16.6, 60.0, 26.5],
    counts=_counts,
    mappings=["om2024"],
    place=RD_GEO / "om" / "om_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=_weight,
    note_public=(
        "Nobody in Oman is asked their language, so nothing here is a count of a language. "
        "The population register counted 3.0 million Omanis and 2.3 million foreign residents "
        "at the end of 2024. Omanis are all drawn as Omani Arabic, so the many who speak "
        "Baluchi at home, the Jibbali and Mehri speakers of Dhofar and Kumzari speakers in "
        "Musandam are not shown. Foreign residents are drawn by nationality, at each "
        "governorate's mix, on their home country's languages. Indians are split by home "
        "state and Pakistanis by province, which probably undercounts the Baluch among them. "
        "Inside each wilaya, foreign residents are placed more heavily in dense districts and "
        "wholly in industrial areas and labour camps, and Omanis in the rest; that is an "
        "estimate, not a count."),
)
