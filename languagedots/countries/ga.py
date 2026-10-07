# Gabon. No census asks language. Gabonese from Afrobarometer R6-R9's home language by province
# (French at R7's mother-tongue share, ask 018), foreign residents by UN DESA origin on each origin's home mix,
# on the RGPL 2026 province totals (sources/ga_afro.py). Placed on religiondots' Kontur hexes
# (read-only). Record: sources/ga.md.
from _shared import *  # noqa: F401,F403

RGPL_2026 = 3_518_621


def _counts():
    import ga2026
    df = pd.read_csv(NORM / "ga.csv", dtype={"geo_id": str})
    if df["geo_id"].nunique() != 9 or int(df["count"].sum()) != RGPL_2026:
        raise SystemExit(f"ga.csv: {df['geo_id'].nunique()} provinces, {df['count'].sum():,} "
                         "people -- run sources/ga_afro.py")
    df["node"] = df["source_category"].map(ga2026.resolve)
    df["unit"] = df["geo_id"]   # religiondots' ga_lookup.csv: geo_id == unit
    return df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()


ENTRY = dict(
    name="Gabon",
    source=("Afrobarometer rounds 6-9 (2015-2022), home language; RGPL 2026 province totals and "
            "the Gabonese/foreign split (Direction Générale de la Statistique), with 2013's "
            "foreign shares by province (RGPL 2013, Tableau 25); UN DESA International Migrant "
            "Stock 2020; home languages of other countries from their sources on this map"),
    how=("no language question: Gabonese from a survey of home language, foreign residents on "
         "their country of origin's languages, both as shares per province of the 2026 census"),
    parts=[
        dict(covers="Gabonese citizens",
             source="Afrobarometer 2015-2022, home language of adults, by province; French at "
                    "the 2017 round's mother-tongue share",
             people=2_318_365),
        dict(covers="Foreign residents",
             source="UN migrant stock 2020, country of origin, drawn on that country's "
                    "languages", rest=True),
    ],
    grain="9 provinces, 391,000 people on average",
    view=[8.6, -4.1, 14.6, 2.4],
    counts=_counts,
    mappings=["ga2026"],
    place=RD_GEO / "ga" / "ga_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "Gabon's census does not ask about language. Gabonese citizens are drawn from four "
        "Afrobarometer rounds between 2015 and 2022, which asked about 4,800 adults the "
        "language they speak at home. From 2017 about 74% answered French, but only 1.4% named "
        "it as their mother tongue in the 2017 round, so French is drawn at that share and the "
        "other languages from all four rounds. The 2026 census counted 1.2 million foreign "
        "residents, a third of the people, but has not published their nationality. They are "
        "drawn by the United Nations' 2020 count of migrants by country of origin, led by "
        "Equatorial Guinea, Mali, Benin and Cameroon, with one mix in every province."),
)
