# Senegal. RGPH-5 2023, the language each person speaks most often, by région (sources/sn_rgph.py:
# the fourteen regional reports' language tables, counts as printed), on religiondots' Kontur
# hexes re-keyed from its 1988 units to today's 14 régions (sources/sn_geo.py).
from _shared import *  # noqa: F401,F403


def _counts():
    import sn2023
    df = pd.read_csv(NORM / "sn.csv")
    df = df[(df["geo_level"] == "region") & (df["count"] > 0)].copy()
    df["node"] = df["source_category"].map(sn2023.resolve)
    unmapped = sorted(set(df.loc[df["node"].isna(), "source_category"]))
    if unmapped:
        raise SystemExit(f"sn: categories with no node: {unmapped}")
    df = df.rename(columns={"geo_id": "unit"})
    if df["unit"].nunique() != 14:
        raise SystemExit(f"sn: {df['unit'].nunique()} régions, expected 14")
    return df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()


ENTRY = dict(
    name="Senegal",
    source=("Fifth General Census of Population and Housing (RGPH-5) 2023, the fourteen regional "
            "reports' tables of the principal language spoken (ANSD)"),
    how="census, 2023, language spoken most often, aged 3 and over",
    parts=[dict(covers="Everyone aged 3 and over",
                source="2023 census, language spoken most often, regional tables", rest=True)],
    grain="14 regions, 1,139,000 people on average",
    gap=("children under 3, who were not asked (about 1.7 million); and 492,752 people aged 3 "
         "and over (3.0%) who are in the national language table but in none of the regional "
         "ones, a difference the reports do not explain"),
    view=[-17.6, 12.2, -11.3, 16.7],
    counts=_counts,
    mappings=["sn2023"],
    place=GEO / "sn" / "sn_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "The census asked which language each person speaks most often, not their mother "
        "tongue. Wolof, the language most Senegalese share, was named by 53.5% nationally and "
        "by 68.7% in Dakar, so many people whose first language is Pulaar, Serer or Joola are "
        "drawn as Wolof, especially in the cities. "
        "In Kédougou, 31.5% named an African language that the census does not print by name; "
        "they are drawn as other African languages. The regional tables add up to 3% fewer "
        "people than the national table, by about the same share for every language, and are "
        "drawn as printed."),
)
