# Somalia. No census language question: REACH's JMCNA 2021 phone survey (11,349 households,
# "main language your household speaks at home") shares per region on the COD-PS 2026 region
# populations (sources/so_jmcna.py); every row `modelled`. Placed on religiondots' Kontur hexes
# for the 18 regions (read-only). Record: sources/so.md.
from _shared import *  # noqa: F401,F403

REGIONS = 18
POP2026 = 19_442_160


def _counts():
    import so2021
    df = pd.read_csv(NORM / "so.csv", dtype={"geo_id": str})
    if df["geo_id"].nunique() != REGIONS or int(df["count"].sum()) != POP2026:
        raise SystemExit("so.csv is not 18 regions summing to COD-PS 2026 -- run "
                         "sources/so_jmcna.py")
    df["node"] = df["source_category"].map(so2021.resolve)
    df["unit"] = df["geo_id"]   # religiondots' so_lookup.csv: geo_id == unit
    out = by_unit(df[df["count"] > 0])
    out["tier"] = "modelled"
    return out


ENTRY = dict(
    name="Somalia",
    source=("REACH Joint Multi-Cluster Needs Assessment 2021 (for the Somalia IMAWG), household "
            "main language, on OCHA's COD-PS 2026 region populations"),
    how=("survey, 2021, main language spoken at home, 11,349 households by phone, on 2026 "
         "region population estimates"),
    parts=[dict(covers="Everyone",
                source="2021 REACH phone survey, household main language, shares applied to "
                       "each region's 2026 population",
                rest=True)],
    grain="18 regions, 1.08 million people on average",
    gap=("3.2% of households whose answer is not drawn (mostly Somali Sign Language, read as a "
         "mistaken answer); the rest of each region is scaled up over them"),
    view=[40.9, -1.8, 51.5, 12.1],
    counts=_counts,
    mappings=["so2021"],
    place=RD_GEO / "so" / "so_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "Somalia has had no census since 1975, so this map uses a 2021 humanitarian phone "
        "survey run by REACH, which asked 11,349 households which language they mainly speak "
        "at home. Maay, spoken by the Digil and Mirifle clans and treated by linguists as a "
        "separate language from Somali, is 96% of Bay and 88% of Bakool. Benaadir Somali is "
        "drawn apart because the survey offered it as its own answer. The survey reached "
        "households with a phone and network coverage, so remote and al-Shabaab-held areas "
        "are under-covered, and Bantu languages such as Mushungulu may be undercounted. "
        "Middle Juba had no respondents and is drawn with Lower Juba's shares. Somaliland is "
        "included."),
)
