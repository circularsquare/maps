# Nauru. No language question in 2021: census ethnicity (Table I-1) read as language, foreign
# ethnicities through origin_mix, national (sources/nr_census.py). Placed on religiondots' Kontur
# hexes for the whole island (read-only). Record: sources/nr.md.
from _shared import *  # noqa: F401,F403


def _counts():
    import nr2021
    df = pd.read_csv(NORM / "nr.csv")
    if int(df["count"].sum()) != 11_680:
        raise SystemExit("nr.csv: expected 11,680 people -- run sources/nr_census.py")
    df["node"] = df["source_category"].map(nr2021.resolve)
    df["unit"] = "NR"   # religiondots' nr_hexes.gpkg: one unit
    out = by_unit(df)
    out["tier"] = "derived"
    return out


ENTRY = dict(
    name="Nauru",
    source=("2021 Population and Housing Census, Table I-1 (Nauru Bureau of Statistics); 2011 "
            "census report for the share speaking Nauruan at home"),
    how=("no language question in 2021: ethnicity read as language, other ethnicities on "
         "their country's languages"),
    parts=[dict(covers="Everyone",
                source="2021 census, ethnicity read as language", rest=True)],
    grain="the country as one unit, 11,680 people",
    view=[166.89, -0.56, 166.97, -0.49],
    counts=_counts,
    mappings=["nr2021"],
    place=RD_GEO / "nr" / "nr_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "Nauru's 2021 census asked ethnicity but not language, so ethnicity is drawn as "
        "language. 95% of residents are Nauruan and are drawn as Nauruan speakers; in the "
        "2011 census, which did ask, 95% of people spoke Nauruan at home. The others, mostly "
        "I-Kiribati, Fijians and Tuvaluans, are drawn on their home countries' languages. The "
        "census counts people in 15 districts, but at this size the map shows the island as "
        "one."),
)
