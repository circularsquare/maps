# Greenland. No language question: the population register's birthplace by locality, 1 January
# 2026 (sources/gl_census.py). The Greenland-born on Greenlandic by district, the born-outside by
# citizenship (Danish for Danish citizens). Every row derived. Placed on religiondots' locality
# discs, one unit per town or settlement (data/geo/gl/gl_places.gpkg). Record: sources/gl.md.
from _shared import *  # noqa: F401,F403


def _counts():
    import gl2026
    df = pd.read_csv(NORM / "gl.csv", dtype={"geo_id": str})
    if df["geo_id"].nunique() != 78:
        raise SystemExit(f"gl.csv: {df['geo_id'].nunique()} localities, expected 78 -- "
                         "run sources/gl_census.py")
    df["node"] = df["source_category"].map(gl2026.resolve)
    df["unit"] = df["geo_id"]
    out = by_unit(df)
    out["tier"] = "derived"
    return out


ENTRY = dict(
    name="Greenland",
    source=("Statistics Greenland, population register 1 January 2026: place of birth by "
            "locality (BEXSTD) and citizenship for Greenland and Nuuk (BEXST6, BEXST6NUK)"),
    how=("no language question: people born in Greenland drawn as Greenlandic speakers by "
         "district, people born outside by citizenship, Danish citizens as Danish; register 2026"),
    parts=[
        dict(covers="People born in Greenland",
             source="2026 register, place of birth, drawn as the local Greenlandic variety",
             nodes=["eskimoaleut.greenlandic", "eskimoaleut.tunumiisut",
                    "eskimoaleut.inuktun"]),
        dict(covers="People born outside Greenland",
             source="2026 register, citizenship, drawn on that country's languages; Danish "
                    "citizens as Danish",
             rest=True),
    ],
    grain="78 towns and settlements, 730 people on average",
    view=[-73.5, 59.5, -11.0, 78.5],
    counts=_counts,
    mappings=["gl2026"],
    place=GEO / "gl" / "gl_places.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "Greenland's population register does not record language, so this map uses "
        "birthplace. People born in Greenland are drawn as Greenlandic speakers in the variety "
        "of where they live: Tunumiisut (East Greenlandic) around Tasiilaq and "
        "Ittoqqortoormiit, Inuktun around Qaanaaq, Kalaallisut elsewhere. Some of them, many in "
        "Nuuk, grow up speaking Danish, but no count exists. People born outside Greenland are "
        "drawn by citizenship, with Danish citizens as Danish speakers."),
)
