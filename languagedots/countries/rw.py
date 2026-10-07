# Rwanda. No language question in any census (2012 and 2022 ask literacy by language): everyone
# drawn on Kinyarwanda, per district, on the RPHC-5 2022 counts (sources/rw_pop.py). Placed on
# religiondots' Kontur hexes, already re-levelled sector by sector onto the census (read-only).
# Record: sources/rw.md.
from _shared import *  # noqa: F401,F403


def _counts():
    import rw2022
    df = pd.read_csv(NORM / "rw.csv", dtype={"geo_id": str})
    if df["geo_id"].nunique() != 30:
        raise SystemExit(f"rw.csv: {df['geo_id'].nunique()} districts, expected 30 -- "
                         "run sources/rw_pop.py")
    df["node"] = df["source_category"].map(rw2022.resolve)
    df["unit"] = df["geo_id"]   # rw.csv carries religiondots' unit ids (RWD11...) directly
    out = by_unit(df)
    out["tier"] = "derived"
    return out


ENTRY = dict(
    name="Rwanda",
    source="population: RPHC-5 2022 district counts (NISR); no language source",
    how=("no language question: everyone drawn as Kinyarwanda, the first language of nearly all "
         "Rwandans"),
    parts=[dict(covers="Everyone", source="2022 census population, drawn as Kinyarwanda",
                rest=True)],
    grain="30 districts, 442,000 people on average",
    view=[28.8, -2.9, 30.95, -1.0],
    counts=_counts,
    mappings=["rw2022"],
    place=RD_GEO / "rw" / "rw_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "Rwanda's 2012 and 2022 censuses asked which languages people can read and write, not "
        "which they speak, so every Rwandan is drawn as a Kinyarwanda speaker, the language "
        "nearly everyone grows up speaking. English, French and Swahili are official but "
        "mostly learned; the few families who speak one at home have no count. Refugees from "
        "Congo and Burundi in the camps are drawn as Kinyarwanda speakers too."),
)
