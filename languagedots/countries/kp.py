# North Korea. No language question in any census: everyone drawn on Korean, per province, on
# the 2008 census province populations (sources/kp_pop.py). Placed on religiondots' Kontur hexes
# for the 11 provinces (read-only). Record: sources/kp.md.
from _shared import *  # noqa: F401,F403


def _counts():
    import kp2008
    df = pd.read_csv(NORM / "kp.csv", dtype={"geo_id": str})
    if df["geo_id"].nunique() != 11:
        raise SystemExit(f"kp.csv: {df['geo_id'].nunique()} provinces, expected 11 -- "
                         "run sources/kp_pop.py")
    df["node"] = df["source_category"].map(kp2008.resolve)
    df["unit"] = df["geo_id"]   # religiondots' kp_lookup.csv: geo_id == unit
    out = by_unit(df)
    out["tier"] = "derived"
    return out


ENTRY = dict(
    name="North Korea",
    source=("population: 2008 census by province (Central Bureau of Statistics with UNFPA); no "
            "language source"),
    how="no language question: everyone drawn as Korean, the first language of nearly everyone",
    parts=[
        dict(covers="Everyone", source="2008 census province populations, drawn as Korean",
             rest=True),
    ],
    grain="11 provinces, 2.1 million people on average",
    gap="702,372 people counted in no province, mostly soldiers in barracks",
    view=[124.1, 37.6, 130.8, 43.1],
    counts=_counts,
    mappings=["kp2008"],
    place=RD_GEO / "kp" / "kp_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "North Korea's censuses have not asked about language, so every person is drawn as a "
        "Korean speaker, which nearly everyone in the country is. The small ethnic Chinese "
        "community has no published count and is drawn as Korean too. The dots follow the 2008 "
        "census by province; the 0.7 million it counted in no province, mostly soldiers, are "
        "not drawn."),
)
