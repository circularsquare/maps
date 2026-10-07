# Palestine. No census asks language. PCBS 2017 census persons counted per governorate, at Arab
# Barometer's first language (2016) and ethnic group (2021-22), every answer Arabic
# (sources/ps_surveys.py). Placed on religiondots' Kontur 400 m hexes (settlement population taken
# off there). Record: sources/ps.md.
from _shared import *  # noqa: F401,F403

COUNTED_2017 = 4_705_601


def _counts():
    import ps2017
    df = pd.read_csv(NORM / "ps.csv")
    if df["geo_id"].nunique() != 16:
        raise SystemExit(f"ps.csv: {df['geo_id'].nunique()} governorates, expected 16")
    if int(df["count"].sum()) != COUNTED_2017:
        raise SystemExit(f"ps.csv sums to {df['count'].sum():,}, expected {COUNTED_2017:,}")
    df["node"] = df["source_category"].map(ps2017.resolve)
    df["unit"] = df["geo_id"]
    out = by_unit(df)
    out["tier"] = "modelled"
    return out


ENTRY = dict(
    name="Palestine",
    source=("Arab Barometer waves IV (2016) and VII (2021-22); Palestinian Central Bureau of "
            "Statistics, Population, Housing and Establishments Census 2017, Preliminary "
            "Results, Table 2"),
    how="no language question; survey answers (all Arabic) applied to the 2017 census population",
    parts=[dict(covers="Everyone",
                source="2017 census population; Arab Barometer 2016 first language and 2021-22 "
                       "ethnic group, every answer Arabic",
                rest=True)],
    grain="16 governorates, 294,000 people on average",
    gap="Israeli settlers in the West Bank are outside the census and drawn as their own entry.",
    view=[34.2, 31.2, 35.6, 32.6],
    counts=_counts,
    mappings=["ps2017"],
    place=RD_GEO / "ps" / "ps_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "Palestine's 2017 census asked no language question. The Arab Barometer asked 1,200 "
        "adults their first language in 2016 and 1,800 their ethnic group in 2021-22, and "
        "every answer was Arabic or Arab, so all 4.7 million people the census counted are "
        "drawn as Levantine Arabic. Domari, still spoken by some Dom in Jerusalem and Gaza, is "
        "not drawn because no source counts it. Israeli settlers were not counted by this "
        "census and are drawn as their own entry. Since October 2023 most of Gaza's people "
        "have been displaced, so the dots there show where people lived in 2017."),
)
