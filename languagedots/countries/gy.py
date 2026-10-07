# Guyana. Census 2012 ethnic background by region (sources/gy_census.py, Compendium 2 Table
# 2.3), each group read as a language (taxonomy/gy2012.py). Placed on religiondots' 400 m grid
# for the same ten regions (read-only).
from _shared import *  # noqa: F401,F403


def _counts():
    import gy2012
    df = pd.read_csv(NORM / "gy.csv", dtype={"geo_id": str})
    rows = [(r.geo_id, node, r.count * x) for r in df.itertuples()
            for node, x in gy2012.shares(r.source_category).items()]
    out = pd.DataFrame(rows, columns=["unit", "node", "count"])
    out = out.groupby(["unit", "node"], as_index=False)["count"].sum()
    if out["unit"].nunique() != 10:
        raise SystemExit("gy: expected 10 regions")
    out["tier"] = "derived"
    return out


ENTRY = dict(
    name="Guyana",
    source="Population and Housing Census 2012 (Bureau of Statistics), Compendium 2, Table 2.3",
    how="census, 2012, ethnic background (no language question); each group drawn as its "
        "language: Guyanese Creole for most, English for white Guyanese, and an Amerindian "
        "language for a fifth of Amerindians",
    parts=[
        dict(covers="Amerindian languages",
             source="2012 census, Amerindians; 20% of them, from a 2013 IDB survey of 11 villages",
             nodes=["americas_other"]),
        dict(covers="English", source="2012 census, white Guyanese",
             nodes=["indoeuropean.germanic.english"]),
        dict(covers="Everyone else", source="2012 census, ethnic background, drawn as Guyanese "
             "Creole", rest=True),
    ],
    grain="10 regions, 75,000 people on average",
    view=[-61.5, 1.1, -56.4, 8.6],
    counts=_counts,
    mappings=["gy2012"],
    place=RD_GEO / "gy" / "gy_grid_400m.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "Guyana's census does not ask about language. It asks ethnic background, and this map "
        "draws Guyanese Creole for every group except white Guyanese (English) and "
        "Amerindians. The census has a single Amerindian category, so their languages "
        "(Wapishana, Makushi, Patamona, Akawaio, Arawak, Carib, Warao and others) are drawn "
        "together as one unnamed Amerindian language. A 2013 survey of 337 households in 11 "
        "Amerindian villages by the Inter-American Development Bank found 20% of households "
        "fluent in their own language, more of them further from Georgetown; 20% of "
        "Amerindians in every region are drawn on it, which understates the Rupununi and "
        "overstates the coast. Indo-Guyanese are drawn as Creole speakers: Guyanese Bhojpuri "
        "is remembered by some elderly people, but no count exists."),
)
