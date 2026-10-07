# Uganda. NPHC 2024 tribe by subcounty from UBOS's 10% person sample, read as home language
# with Afrobarometer R4-R9's shares per tribe (sources/ug_census.py); every row `modelled`. On
# religiondots' Kontur hexes for its 2,200 drawn subcounty units (read-only). Record: sources/ug.md.
from _shared import *  # noqa: F401,F403

UNITS = 2200
HHPOP = 44_378_756


def _counts():
    import ug2024
    df = pd.read_csv(NORM / "ug.csv")
    if df["geo_id"].nunique() != UNITS:
        raise SystemExit(f"ug.csv: {df['geo_id'].nunique()} units, expected {UNITS} -- "
                         "re-run sources/ug_census.py")
    if abs(df["count"].sum() - HHPOP) > 100:
        raise SystemExit(f"ug.csv sums to {df['count'].sum():,.0f}, not {HHPOP:,}")
    df["node"] = df["source_category"].map(ug2024.resolve)
    missing = sorted(set(df.loc[df["node"].isna(), "source_category"]))
    if missing:
        raise SystemExit(f"ug.csv answers with no node: {missing}")
    df = df[df["count"] > 0].rename(columns={"geo_id": "unit"})
    out = by_unit(df)
    out["tier"] = "modelled"
    return out


ENTRY = dict(
    name="Uganda",
    source=("2024 National Population and Housing Census, tribe or nationality of every "
            "household member in the 10% sample (Uganda Bureau of Statistics); home language by "
            "tribe from Afrobarometer rounds 4 to 9 (2008-2022)"),
    how=("census, 2024, tribe read as language, shared across the home languages each tribe "
         "speaks in the Afrobarometer survey; non-Ugandans drawn by nationality"),
    parts=[
        dict(covers="Ugandans, and Rwandan, Burundian and Somali nationals",
             source="2024 census 10% sample, tribe or nationality, read as home language with "
                    "Afrobarometer 2008-2022 shares per tribe (about 12,000 adults)",
             rest=True),
        dict(covers="Other African nationals",
             source="2024 census 10% sample, nationality, drawn as other African languages "
                    "(mostly South Sudanese and Congolese refugees)",
             people=877_873),
    ],
    grain="2,200 subcounties, 20,000 people on average",
    gap="people not in households (1.5 million: institutions, the homeless, travellers), "
        "whose tribe the sample does not record",
    view=[29.55, -1.5, 35.05, 4.25],
    counts=_counts,
    mappings=["ug2024"],
    place=RD_GEO / "ug" / "ug2024_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "Uganda's census does not ask about language. It asks each person's tribe, and this "
        "map reads each tribe as its language: the census's 10% sample gives the tribes of "
        "every subcounty, and the Afrobarometer survey, which asked about 12,000 adults their "
        "tribe and home language, moves those who speak another language at home. Few do: "
        "most groups keep their own language, though some Banyarwanda, Banyole and Bagwere "
        "families speak Luganda or Lusoga. Refugees are drawn by nationality, so the large "
        "South Sudanese and Congolese settlements appear as other African languages."),
)
