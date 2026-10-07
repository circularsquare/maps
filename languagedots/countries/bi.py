# Burundi. No census language question (2008 asks literacy only): Afrobarometer R5-R6 (2012,
# 2014; 2,400 respondents) home-language shares per province on current province populations,
# the 2024 census total spread across the 2008 provinces by COD-PS 2022 communes
# (sources/bi_afro.py); every row `modelled`. Placed on religiondots' Kontur hexes (read-only).
# Record: sources/bi.md.
from _shared import *  # noqa: F401,F403

PROVINCES = 17
RGPHAE2024 = 12_332_788


def _counts():
    import bi2014
    df = pd.read_csv(NORM / "bi.csv", dtype={"geo_id": str})
    if df["geo_id"].nunique() != PROVINCES or int(df["count"].sum()) != RGPHAE2024:
        raise SystemExit("bi.csv is not 17 provinces summing to the 2024 census total -- run "
                         "sources/bi_afro.py")
    df["node"] = df["source_category"].map(bi2014.resolve)
    df["unit"] = df["geo_id"]   # religiondots' bi_lookup.csv: geo_id == unit
    out = by_unit(df[df["count"] > 0])
    out["tier"] = "modelled"
    return out


ENTRY = dict(
    name="Burundi",
    source=("Afrobarometer rounds 5 and 6 (2012, 2014), home language, on the 2024 census "
            "total (INSBU, preliminary) spread across provinces by OCHA's COD-PS 2022 "
            "projections"),
    how=("survey, 2012 and 2014 pooled, home language, 2,400 respondents, on the 2024 census "
         "population spread by 2022 projections"),
    parts=[dict(covers="Everyone",
                source="Afrobarometer 2012 and 2014, home language, 2,400 adults", rest=True)],
    grain="17 provinces as of 2008, 725,000 people on average",
    view=[28.95, -4.5, 30.9, -2.3],
    counts=_counts,
    mappings=["bi2014"],
    place=RD_GEO / "bi" / "bi_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "Burundi's census asks which languages people can read and write, not which they speak, "
        "so this map uses the Afrobarometer survey, which asked 2,400 Burundians in 2012 and "
        "2014 which language they speak at home. 99.4% said Kirundi. Swahili was named mostly "
        "in Bujumbura, where 4% of respondents gave it. With about 150 respondents per province, "
        "a small language's share in one province can rest on one or two answers. The shares "
        "are drawn on the 2024 census population, divided between the 17 provinces of 2008 by "
        "UN projections."),
)
