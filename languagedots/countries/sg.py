# Singapore. Census of Population 2020, language most frequently spoken at home, residents 5+
# (sources/sg_census.py): seven groups for 30 planning areas and an `Others` row, with "Chinese
# Dialects" shared out by the national Hokkien/Teochew/Cantonese/other mix. Placed on
# religiondots' 332 URA subzones weighted by census resident population (read-only).
from _shared import *  # noqa: F401,F403

PLACE = RD_GEO / "sg" / "sg_subzones.gpkg"
DIALECTS = ["Hokkien", "Teochew", "Cantonese", "Other Chinese Dialects"]


def _counts():
    import sg2020
    df = pd.read_csv(NORM / "sg.csv", dtype={"geo_id": str})
    pa = df[df["geo_level"] == "planning_area"].copy()
    nat = df[df["geo_level"] == "country"].set_index("source_category")["count"]

    units = set(pa["geo_id"])
    import geopandas as gpd
    layer = set(gpd.read_file(PLACE, columns=["unit"], ignore_geometry=True)["unit"].astype(str))
    if units != layer or len(units) != 31:
        raise SystemExit(f"sg: units not in the layer {sorted(units - layer)}, "
                         f"layer units not in the table {sorted(layer - units)}")

    # Seven groups as published; "Chinese Dialects" opened by the national mix (Table 41), the
    # only geography at which the census splits the dialects.
    kept = pa[pa["source_category"] != "Chinese Dialects"].assign(tier="measured")
    share = nat[DIALECTS] / nat[DIALECTS].sum()
    cd = pa[pa["source_category"] == "Chinese Dialects"]
    split = pd.concat([cd.assign(source_category=d, count=cd["count"] * share[d], tier="derived")
                       for d in DIALECTS], ignore_index=True)
    if abs(split["count"].sum() - cd["count"].sum()) > 1e-6:
        raise SystemExit("sg: the dialect split lost people")

    out = pd.concat([kept, split], ignore_index=True)
    out["node"] = out["source_category"].map(sg2020.resolve)
    out = out[out["count"] > 0].rename(columns={"geo_id": "unit"})
    return out.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()


ENTRY = dict(
    name="Singapore",
    source="Census of Population 2020, TableBuilder CT/17596 (planning area) and Statistical "
           "Release 1 Table 41 (Department of Statistics Singapore, via data.gov.sg)",
    how="census, 2020, language most frequently spoken at home, residents aged 5 and over",
    parts=[
        dict(covers="Chinese dialects",
             source="2020 census, dialect speakers per planning area, split into Hokkien, "
                    "Cantonese, Teochew and others by the national mix",
             nodes=["sinotibetan.sinitic.min_nan", "sinotibetan.sinitic.cantonese",
                    "sinotibetan.sinitic.teochew", "sinotibetan.sinitic"]),
        dict(covers="Everyone else", source="2020 census, language most often spoken at home",
             rest=True),
    ],
    grain="30 planning areas plus one row holding the other 25, 116,000 people on average",
    gap="non-residents, 1,641,590 (29% of the people in Singapore), whom the census does not "
        "ask; and 447,926 residents outside the question, among them children under 5 and "
        "people living alone",
    view=[103.60, 1.16, 104.09, 1.48],
    counts=_counts,
    mappings=["sg2020"],
    place=PLACE,
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "The census asks which language a person speaks most often at home to the other members "
        "of the household, so this is home language, not mother tongue. It asks citizens and "
        "permanent residents only: the 1.64 million non-residents, from the dormitories at "
        "Tuas to the domestic workers in flats across the island, are missing. Which Chinese "
        "dialect is published only for the whole country, so each planning area's dialect "
        "speakers are split in the national proportion; those dots are inferred, not counted. "
        "Other Indian languages (Malayalam, Hindi, Punjabi and the rest) are not broken down "
        "anywhere and are drawn as other languages. Twenty-five small planning areas share "
        "one row, Rochor with Little India and Kampong Glam among them."),
)
