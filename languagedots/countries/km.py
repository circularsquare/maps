# Comoros. No language question: French at Afrobarometer R10's national 9.9%, everyone else on
# their island's Comorian language (Ngazidja, Ndzwani, Mwali), on the 2017 census island counts
# (sources/km_pop.py); rows `modelled`. Placed on
# religiondots' Kontur hexes, already scaled to each island's census count (read-only).
# Record: sources/km.md.
from _shared import *  # noqa: F401,F403


def _counts():
    import km2017
    df = pd.read_csv(NORM / "km.csv", dtype={"geo_id": str})
    if df["geo_id"].nunique() != 3:
        raise SystemExit(f"km.csv: {df['geo_id'].nunique()} islands, expected 3 -- "
                         "run sources/km_pop.py")
    df["node"] = df["source_category"].map(km2017.resolve)
    df["unit"] = df["geo_id"]   # religiondots' km_lookup.csv: geo_id == unit (island names)
    out = by_unit(df)
    out["tier"] = "modelled"
    # Anita, 2026-10-05: French at Afrobarometer R10's "language spoken most at home", 9.9%,
    # national only (no island split published, microdata not released), applied to each island.
    # Swahili and Malagasy (0.1% each) stay on Comorian.
    fr = out.copy()
    fr["count"] = (out["count"] * FRENCH).round().astype(int)
    fr["node"] = "indoeuropean.romance.french"
    out["count"] = out["count"] - fr["count"]
    return pd.concat([out, fr], ignore_index=True)


FRENCH = 0.099   # Afrobarometer R10 (2024) Comoros summary, p. 6, Q2: French 9.9%


ENTRY = dict(
    name="Comoros",
    source=("population: RGPH 2017 island counts (INSEED); language: Afrobarometer Round 10 "
            "(2024), Comoros summary of results, language spoken most at home"),
    how=("no language question: French at its 9.9% national survey share on every island, "
         "everyone else drawn as their island's Comorian language"),
    parts=[
        dict(covers="French speakers",
             source="Afrobarometer 2024, language spoken most at home, 9.9% nationally",
             nodes=["indoeuropean.romance.french"]),
        dict(covers="Everyone else",
             source="2017 census island populations, drawn as the island's Comorian language",
             rest=True),
    ],
    grain="3 islands, 253,000 people on average",
    view=[43.15, -12.45, 44.6, -11.3],
    counts=_counts,
    mappings=["km2017"],
    place=RD_GEO / "km" / "km_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "The Comoros census does not ask about language, so everyone is drawn as a speaker of "
        "their own island's Comorian: Shingazidja on Grande Comore, Shindzuani on Anjouan and "
        "Shimwali on Mohéli. French is drawn at 10% on every island, its share in the 2024 "
        "Afrobarometer survey, whose sample was better educated than the country, so French "
        "may be drawn too high. Mayotte is drawn with France."),
)
