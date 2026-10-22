# Singapore. Census of Population 2020, language most frequently spoken at home, residents 5+
# (sources/sg_census.py): seven groups for 30 planning areas and an `Others` row, with "Chinese
# Dialects" shared out by the national Hokkien/Teochew/Cantonese/other mix. Placed on
# religiondots' 332 URA subzones weighted by census resident population (read-only).
# Coverage gap (2026-10-19): the 1,641,590 non-residents the census does not ask are added,
# `modelled`, from MOM's pass-type stock (sources/sg_nonres.py): dormitory residents on dormitory
# sites and industrial land, everyone else on planning areas by resident population. The layer is
# religiondots' subzones (read-only) plus those sites (python sources/sg_nonres.py --build).
# Record: sources/sg.md.
from _shared import *  # noqa: F401,F403

PLACE = GEO / "sg" / "sg_places.gpkg"
DIALECTS = ["Hokkien", "Teochew", "Cantonese", "Other Chinese Dialects"]


def _counts():
    import sg2020
    df = pd.read_csv(NORM / "sg.csv", dtype={"geo_id": str})
    pa = df[df["geo_level"] == "planning_area"].copy()
    nat = df[df["geo_level"] == "country"].set_index("source_category")["count"]

    units = set(pa["geo_id"])
    import pyogrio
    lay = pyogrio.read_dataframe(PLACE, read_geometry=False)
    layer = set(lay.loc[lay["kind"] == "resident", "unit"].astype(str))
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
    res = out.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()

    import sys
    sys.path.insert(0, str(ROOT / "sources"))
    import sg_nonres
    return pd.concat([res, sg_nonres.counts()], ignore_index=True)


NONRES = 1_641_590


def _dialect_people():
    df = pd.read_csv(NORM / "sg.csv", dtype={"geo_id": str})
    return int(round(df[(df["geo_level"] == "planning_area")
                        & (df["source_category"] == "Chinese Dialects")]["count"].sum()))


ENTRY = dict(
    name="Singapore",
    source="Census of Population 2020, TableBuilder CT/17596 (planning area) and Statistical "
           "Release 1 Table 41 (Department of Statistics Singapore, via data.gov.sg); "
           "non-residents from the Ministry of Manpower's foreign workforce numbers (June 2020), "
           "its 2020 dormitory counts, URA Master Plan 2019 land use and OpenStreetMap",
    how="census, 2020, language most frequently spoken at home, residents aged 5 and over; "
        "the 1.64 million non-residents modelled from work pass numbers, each pass type's "
        "nationalities drawn on their home languages",
    parts=[
        dict(covers="Chinese dialects",
             source="2020 census, dialect speakers per planning area, split into Hokkien, "
                    "Cantonese, Teochew and others by the national mix",
             people=_dialect_people()),
        dict(covers="Non-residents",
             source="Ministry of Manpower pass numbers for June 2020, dormitory counts, "
                    "nationality mixes from surveys and press (modelled); dormitory residents on "
                    "dormitory sites and industrial land, the rest by resident population",
             people=NONRES),
        dict(covers="Everyone else", source="2020 census, language most often spoken at home",
             rest=True),
    ],
    grain="30 planning areas plus one row holding the other 25, 116,000 people on average; "
          "non-residents on 29 dormitory sites and 66 industrial areas besides",
    gap="448,000 residents outside the question, among them children under 5 and people "
        "living alone; non-resident nationalities are estimates, since no office publishes them",
    view=[103.60, 1.16, 104.09, 1.48],
    counts=_counts,
    mappings=["sg2020"],
    place=PLACE,
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "The census asks which language a person speaks most often at home to the other members "
        "of the household, so this is home language, not mother tongue. It asks citizens and "
        "permanent residents only. The 1.64 million non-residents (29% of the people in "
        "Singapore) are added as estimates: the Ministry of Manpower's June 2020 numbers of "
        "Employment Pass, S Pass, work permit and domestic worker holders, with dependants and "
        "students as the remainder. The ministry does not publish nationality, so each pass "
        "type is given a nationality mix from what is public (a 2022 survey of dormitory "
        "workers: 44% Bangladeshi, 40% Indian) and from judgement, and each nationality is drawn "
        "on its home languages. About 323,000 workers lived in dormitories: 198,000 in 43 "
        "purpose-built ones, placed on 29 that OpenStreetMap names (the rest on industrial "
        "land), and 95,000 in converted factories, placed on industrial land. Everyone else is "
        "spread by resident population. Which Chinese dialect is published only for the whole "
        "country, so each planning area's dialect speakers are split in the national "
        "proportion; those dots are inferred, not counted. Other Indian languages (Malayalam, "
        "Hindi, Punjabi and the rest) are not broken down anywhere and are drawn as other "
        "languages. Twenty-five small planning areas share one row, Rochor with Little India "
        "and Kampong Glam among them."),
)
