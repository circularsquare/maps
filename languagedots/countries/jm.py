# Jamaica. No census language question: Jamaican Creole and English at the national shares of the
# Jamaican Language Unit's 2006 Language Competence Survey, on the 2011 census parish populations
# (sources/jm_jlu.py). Every row modelled. Placed on religiondots' 400 m grid (read-only).
# Record: sources/jm.md.
from _shared import *  # noqa: F401,F403


def _counts():
    import jm2006
    df = pd.read_csv(NORM / "jm.csv", dtype={"geo_id": str})
    if df["geo_id"].nunique() != 14:
        raise SystemExit(f"jm.csv: {df['geo_id'].nunique()} parishes, expected 14 -- "
                         "run sources/jm_jlu.py")
    df["node"] = df["source_category"].map(jm2006.resolve)
    df["unit"] = df["geo_id"]   # religiondots' grid is keyed by the same JAM_GEO1_nn ids
    out = by_unit(df)
    out["tier"] = "modelled"
    return out


ENTRY = dict(
    name="Jamaica",
    source=("Jamaican Language Unit (University of the West Indies, Mona), Language Competence "
            "Survey of Jamaica 2006; Census 2011 parish populations (Statistical Institute of "
            "Jamaica)"),
    how=("no language question: a 2006 survey of 1,000 adults; those who spoke Jamaican, alone "
         "or with English, drawn as Jamaican Creole, those who spoke only English as English; "
         "one national share applied to each parish's 2011 census population"),
    parts=[dict(covers="Everyone", source="Language Competence Survey 2006, national shares",
                rest=True)],
    grain="14 parishes, 191,000 people on average",
    view=[-78.5, 17.6, -76.1, 18.6],
    counts=_counts,
    mappings=["jm2006"],
    place=RD_GEO / "jm" / "jm_grid_400m.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "Jamaica's census does not ask about language. In a 2006 survey by the Jamaican "
        "Language Unit, 1,000 adults were drawn into conversation in both languages: the "
        "83% who spoke Jamaican are drawn as Jamaican Creole speakers, the 17% who spoke only "
        "English as English. That share is likely a ceiling, since some people avoid speaking "
        "Jamaican to strangers. English-only speakers were more common in towns and in the "
        "east, but every parish gets the national share."),
)
