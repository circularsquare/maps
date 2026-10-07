# United Arab Emirates. No census or survey asks language. Emiratis on Gulf Arabic, non-Emiratis
# by UN DESA 2024 origin read as their languages (sources/ae_build.py, sources/gulf_mix.py), 7
# emirates with Abu Dhabi in its 3 regions, on religiondots' Kontur 400 m hexes re-keyed
# (sources/gulf_place.py), citizens and foreigners placed apart inside each unit. Every row
# derived. Record: sources/ae.md, sources/gulf_place.md.
from _shared import *  # noqa: F401,F403

TOTAL_2024 = 11_294_243


def _gp():
    """sources/gulf_place.py: the Gulf citizen / foreign placement and unit splits."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("gulf_place", ROOT / "sources" / "gulf_place.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rows():
    import ae2024
    df = pd.read_csv(NORM / "ae.csv", dtype={"geo_id": str})
    if df["geo_id"].nunique() != 7:
        raise SystemExit(f"ae.csv: {df['geo_id'].nunique()} emirates, expected 7")
    if int(df["count"].sum()) != TOTAL_2024:
        raise SystemExit(f"ae.csv sums to {df['count'].sum():,}, expected {TOTAL_2024:,}")
    lut = pd.read_csv(RD_GEO / "ae" / "ae_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    if df["unit"].isna().any():
        raise SystemExit(f"ae: emirates missing from religiondots' ae_lookup.csv: "
                         f"{sorted(df.loc[df['unit'].isna(), 'geo_id'].unique())}")
    df["node"] = df["source_category"].map(ae2024.resolve)
    # Abu Dhabi emirate split into its three regions (SCAD; sources/gulf_place.py SPLITS)
    return _gp().split_units(df, "ae")


def _counts():
    out = by_unit(_rows())
    out["tier"] = "derived"
    return out


def _weight(place):
    gp = _gp()
    return gp.GulfWeighter("ae", place, *gp.citizen_tables(_rows()))


ENTRY = dict(
    name="United Arab Emirates",
    source=("Each emirate's statistics office and the Federal Competitiveness and Statistics "
            "Centre (FCSC), population 2024 and Emiratis by emirate, as assembled for this "
            "project's religion map; Statistics Centre Abu Dhabi, census 2023 population by "
            "region and Emiratis by region in 2016; OpenStreetMap industrial land; UN DESA, International Migrant Stock 2024, migrants by "
            "origin; Indians by state from the Kerala Migration Survey 2023 and India's "
            "emigration clearances 2011 to 2017; Pakistanis by province from the Bureau of "
            "Emigration's registrations 2019 to 2021; home languages from the censuses of India "
            "(2011), Pakistan (2023) and other countries on this map"),
    how=("no language question; Emiratis drawn as Gulf Arabic, foreign residents by country of "
         "origin, each on its country's main language or language mix"),
    parts=[
        dict(covers="Emirati citizens",
             source="2024 emirate statistics, Emiratis, drawn as Gulf Arabic", people=1_519_227),
        dict(covers="Foreign residents",
             source="UN DESA 2024 migrants by origin, one mix for every emirate; Indian states "
                    "and Pakistani provinces from emigration records", rest=True),
    ],
    grain="7 emirates, Abu Dhabi split into its 3 regions, 1.3 million people on average",
    gap=("Emiratis whose first language is not Arabic (Persian, Baluchi and Swahili-speaking "
         "families among them), whom no source counts"),
    view=[51.5, 22.6, 56.4, 26.1],
    counts=_counts,
    mappings=["ae2024"],
    place=GEO / "ae" / "ae_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=_weight,
    note_public=(
        "Nobody in the United Arab Emirates is asked their language, and no census has been "
        "taken across the federation since 2005, so nothing here is a count of a language. "
        "About 1.5 million of the 11.3 million people living there in 2024 were Emirati "
        "citizens, all drawn as Gulf Arabic: Emiratis who speak Persian, Baluchi or Swahili at "
        "home are not shown. Everyone else (87%) is drawn by country of origin, from the UN's "
        "estimate of the UAE's migrants, with no breakdown by emirate, so every emirate's "
        "foreign residents have the same mix. Indians are split by where in India Gulf "
        "migrants come from (about a quarter from Kerala), Pakistanis by province. The UN's "
        "unnamed origins (3%) are drawn as other languages. Abu Dhabi is drawn by its three "
        "regions, with Emiratis per region from the emirate's statistics centre (a quarter of "
        "Al Ain region's people, an eighth of the others'). Inside each unit, foreign residents "
        "are placed more heavily in dense districts and wholly in industrial areas and labour "
        "camps, and Emiratis in the rest; that is an estimate of where people live, not a "
        "count."),
)
