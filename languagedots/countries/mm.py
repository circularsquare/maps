# Myanmar. GAD 2018 Township Profiles, ethnic nationality per township (sources/mm_gad.py), read as
# languages through taxonomy/mm2018.py: a proxy under AGENT_BRIEF section 2's ethnicity rule, every
# row `derived`. No retention share exists for any group (sources/mm.md). Placed on Kontur hexes
# for 325 townships (sources/mm_geo.py). The record is sources/mm.md.
from _shared import *  # noqa: F401,F403


def _counts():
    import mm2018
    df = pd.read_csv(NORM / "mm.csv", dtype={"geo_id": str})
    if df["geo_id"].nunique() != 325:
        raise SystemExit("mm.csv: expected 325 townships; re-run sources/mm_gad.py")
    df["node"] = df["source_category"].map(mm2018.resolve)
    missing = sorted(set(df.loc[df["node"].isna(), "source_category"]))
    if missing:
        raise SystemExit(f"mm.csv categories that resolve to nothing: {missing}")
    if set(df["tier"]) != {"derived"}:
        raise SystemExit(f"mm.csv: tiers {sorted(set(df['tier']))}")
    df["unit"] = df["geo_id"]
    df = df[df["count"] > 0]
    out = df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()
    return pd.concat([out, _rohingya(set(out["unit"]))], ignore_index=True)


def _rohingya(units):
    """Anita, 2026-10-05: Rakhine's people missing from GAD's ethnicity table are the Rohingya.
    Per township, the religion table's Muslims missing from the ethnicity table:
    min(religion total - ethnicity total, Muslims), Rakhine State only. Elsewhere the shortfall
    is mixed and stays in `gap` (sources/mm.md section 3)."""
    u = pd.read_csv(NORM / "mm_unrecorded.csv")
    u = u[u["ADM1_NAME"] == "RAKHINE STATE"].copy()
    u["count"] = u[["unrec", "RLG_ISL"]].min(axis=1).clip(lower=0).round().astype(int)
    u = u[u["count"] > 0]
    stray = sorted(set(u["GEO_MATCH"]) - units)
    if stray:
        raise SystemExit(f"mm_unrecorded.csv: Rakhine townships not in mm.csv: {stray}")
    total = int(u["count"].sum())
    if not 570_000 < total < 590_000:
        raise SystemExit(f"Rohingya rows sum to {total:,}; expected about 580,000")
    return pd.DataFrame({"unit": u["GEO_MATCH"], "tier": "derived", "count": u["count"],
                         "node": "indoeuropean.indoaryan.eastern.rohingya"})


ENTRY = dict(
    name="Myanmar",
    source=("General Administration Department, 2018 Township Profiles, Table 14, ethnic "
            "nationalities living in the township (records as of 1 April 2017), as transcribed by "
            "the U.S. Census Bureau (HDX, CC BY)"),
    how=("administrative records, 2017, ethnic group, each group drawn as its language; "
         "Rohingya in Rakhine from the same offices' religion records"),
    parts=[
        dict(covers="Recognised ethnic groups",
             source="GAD township records, 2017, ethnic group drawn as its language",
             people=47_798_404),
        dict(covers="Rohingya in Rakhine",
             source="GAD township records, 2017, Muslims missing from the ethnic records",
             nodes=["indoeuropean.indoaryan.eastern.rohingya"]),
    ],
    grain="325 townships, 147,000 people on average",
    gap=("people recorded with no ethnic group outside Rakhine, 620,241 (271,848 in Yangon), and "
         "the Wa and Mongla townships, about 433,000 people in 2014, which have no records"),
    view=[92.0, 9.4, 101.4, 28.7],
    counts=_counts,
    mappings=["mm2018"],
    place=GEO / "mm" / "mm_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "Myanmar has never published a count of languages, and the 2014 census's ethnic "
        "results were never released. This map uses the General Administration Department's "
        "2017 township records of ethnic groups and draws each group as its language (Bamar as "
        "Burmese, Pa'o as Pa'o). These are office records, not answers people gave. Karen, "
        "Chin, Kachin, Naga and Kayah each cover several languages the records do not split, "
        "and are drawn as language not named within their group. No survey says how many of "
        "each group still speak its language; many Mon now speak only Burmese, so Burmese is "
        "undercounted. The government does not recognise the Rohingya, so they are not in the "
        "ethnic records. In Rakhine the Muslims in the religion records who are missing from "
        "the ethnic records are drawn as Rohingya, 577,182 people, counted before most fled to "
        "Bangladesh in 2017. Elsewhere 620,241 people are missing the same way, with nothing to "
        "say who they are, and are left empty, as are the Wa and Mongla areas, which have no "
        "records. The records count only 5,638 Chinese and 1,921 Indians, far below the real "
        "numbers. The war since 2021 has displaced millions of people, so this is a picture of "
        "2017."),
)
