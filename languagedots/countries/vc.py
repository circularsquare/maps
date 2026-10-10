# Saint Vincent and the Grenadines. No census language question: everyone on Vincentian Creole
# except white Vincentians on English, per enumeration district, from the 2012 census ethnicity
# table (sources/vc_census.py). Every row derived. Placed on religiondots' ED polygons
# (read-only). Record: sources/vc.md.
from _shared import *  # noqa: F401,F403


def _counts():
    import vc2012
    df = pd.read_csv(NORM / "vc.csv")
    if not 215 <= df["geo_id"].nunique() <= 221:
        raise SystemExit(f"vc.csv: {df['geo_id'].nunique()} EDs, expected 219 -- "
                         "run sources/vc_census.py")
    df["node"] = df["source_category"].map(vc2012.resolve)
    df["unit"] = df["geo_id"]   # religiondots' vc_eds.gpkg `unit` (USCB GEO_MATCH)
    out = by_unit(df)
    out["tier"] = "derived"
    return out


ENTRY = dict(
    name="Saint Vincent and the Grenadines",
    source=("Census 2012, ethnicity by enumeration district (Statistical Office of Saint Vincent "
            "and the Grenadines, as tabulated by the US Census Bureau)"),
    how=("no language question: everyone drawn as Vincentian Creole except people who gave "
         "their ethnicity as white, drawn as English; census 2012"),
    parts=[dict(covers="Everyone",
                source="2012 census, ethnicity: white drawn as English, everyone else as "
                       "Vincentian Creole",
                rest=True)],
    grain="enumeration districts, 500 people on average",
    view=[-61.60, 12.50, -61.05, 13.42],
    counts=_counts,
    mappings=["vc2012"],
    # vc_eds.gpkg cut by Kontur hexes, so a district's dots follow where its people live
    # (sources/kontur_cut.py; 2026-10-08)
    place=GEO / "vc" / "vc_konturcut.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "The census of Saint Vincent and the Grenadines does not ask about language. Most "
        "Vincentians grow up speaking Vincentian Creole, an English-based creole, and learn "
        "standard English at school; the 889 people who gave their ethnicity as white, many of "
        "them on Bequia and Mustique, are drawn as English speakers. The 3,280 people who "
        "identified as Indigenous are Garifuna and Kalinago descendants whose languages are no "
        "longer spoken on the island, so they are drawn as Creole speakers."),
)
