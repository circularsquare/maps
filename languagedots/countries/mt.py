# Malta. Census 2021, main language spoken from early childhood (sources/mt_census.py), on
# religiondots' Kontur hexes cut to the same 68 localities (read only). The record is sources/mt.md.
# Maltese citizens are counted by locality (Table 3.6); non-Maltese residents only by district
# (Table 3.3), so their answers are placed inside each district by where the district's
# non-Maltese residents live (Vol. 1 Table 2.1), a placement inside the unit counted (brief §4.4).
from _shared import *  # noqa: F401,F403


def _counts():
    import mt2021
    df = pd.read_csv(NORM / "mt.csv", dtype={"geo_id": str})
    df["node"] = df["source_category"].map(mt2021.resolve)

    loc = df[(df["geo_level"] == "locality") & df["node"].notna()].copy()
    if loc["geo_id"].nunique() != 68:
        raise SystemExit(f"mt.csv: {loc['geo_id'].nunique()} localities, expected 68")
    loc["unit"] = loc["geo_id"]

    # each district's non-Maltese answers, shared among its localities by the locality's share
    # of the district's non-Maltese residents of all ages (Vol. 1 Table 2.1)
    nmpop = df[df["geo_level"] == "locality_nmpop"][["geo_id", "district", "count"]]
    nmpop = nmpop.assign(share=nmpop["count"] / nmpop.groupby("district")["count"].transform("sum"))
    dist = df[(df["geo_level"] == "district_nm") & df["node"].notna()]
    nm = dist[["district", "node", "count"]].merge(
        nmpop[["geo_id", "district", "share"]], on="district", validate="many_to_many")
    nm["count"] = nm["count"] * nm["share"]
    nm["unit"] = nm["geo_id"]
    if abs(nm["count"].sum() - dist["count"].sum()) > 1e-6:
        raise SystemExit("mt: non-Maltese split does not keep the district totals")

    # religiondots' unit ids are the same names with ASCII hyphens; check the join both ways
    lut = pd.read_csv(RD_GEO / "mt" / "mt_lookup.csv", dtype=str)
    rd = set(lut["unit"])
    ours = set(loc["unit"])
    if ours != rd:
        raise SystemExit(f"mt: units differ from religiondots' mt_lookup.csv: "
                         f"{sorted(ours - rd)[:5]} / {sorted(rd - ours)[:5]}")

    out = pd.concat([loc[["unit", "node", "count"]], nm[["unit", "node", "count"]]])
    out = out[out["count"] > 0]
    return out.groupby(["unit", "node"], as_index=False)["count"].sum()


ENTRY = dict(
    name="Malta",
    source="Census of Population and Housing 2021, Volume 3, Tables 3.3 and 3.6 (NSO Malta); "
           "Volume 1, Table 2.1 for placement",
    how="census, 2021, main language spoken from early childhood, aged 5 and over",
    parts=[
        dict(covers="Maltese citizens",
             source="2021 census, main language from early childhood, by locality",
             people=386_091),
        dict(covers="Residents without Maltese citizenship",
             source="2021 census, same question, by district only",
             rest=True),
    ],
    grain=("68 localities, 7,300 people aged 5 and over on average; non-Maltese residents by 6 "
           "districts"),
    gap="children under 5, 22,297 people or 4.3%, who were not asked",
    view=[14.17, 35.78, 14.59, 36.09],
    counts=_counts,
    mappings=["mt2021"],
    place=RD_GEO / "mt" / "mt_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "The 2021 census asked everyone aged 5 and over which language they mainly spoke from "
        "early childhood, with seven answers: Maltese, English, Italian, German, French, "
        "Arabic and other. English is the first language of 7.8% of Maltese citizens, and of "
        "37.7% in Is-Swieqi. The 111,174 residents without Maltese citizenship are published "
        "only by district, and their dots are placed where the district's foreign residents "
        "live, so they do not show which language each locality's foreign residents speak. "
        "Half of them answered other, drawn in grey."),
)
