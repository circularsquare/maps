# Sri Lanka. CPH 2024 ethnic group by GN division (sources/lk_census.py), read as languages through
# taxonomy/lk2024.py, with each group's retention from CPH 2012's ability-to-speak tables by
# district. Every row `derived` (Anita's 2026-10-05 ruling for ethnicity-only countries). Placed
# on religiondots' own GN polygons, keyed on the same 7-digit census codes (read-only).
# The record is sources/lk.md.
from _shared import *  # noqa: F401,F403


def _counts():
    import lk2024
    df = pd.read_csv(NORM / "lk.csv", dtype={"geo_id": str})
    if df["geo_id"].nunique() != 14003:
        raise SystemExit("lk.csv: expected 14,003 GN divisions; re-run sources/lk_census.py")
    df["node"] = df["source_category"].map(lk2024.resolve)
    missing = sorted(set(df.loc[df["node"].isna(), "source_category"]))
    if missing:
        raise SystemExit(f"lk.csv categories that resolve to nothing: {missing}")
    if set(df["tier"]) != {"derived"}:
        raise SystemExit(f"lk.csv: tiers {sorted(set(df['tier']))}")
    df["unit"] = df["geo_id"]
    df = df[df["count"] > 0]
    return df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()


ENTRY = dict(
    name="Sri Lanka",
    source=("Census of Population and Housing 2024, ethnic group by GN division, with language "
            "ability by ethnic group and district from the 2012 census (Department of Census and "
            "Statistics)"),
    how=("census, 2024, ethnic group, each group drawn as its language, corrected by 2012 "
         "ability to speak Sinhala, Tamil and English"),
    parts=[dict(covers="Everyone",
                source="2024 census, ethnic group, corrected by 2012 census ability to speak "
                       "Sinhala, Tamil and English (aged 10 and over, by district)",
                rest=True)],
    grain="14,003 Grama Niladhari divisions, 1,550 people on average",
    gap="No one is left out; the census asked no home language question.",
    view=[79.4, 5.7, 82.1, 10.0],
    counts=_counts,
    mappings=["lk2024"],
    place=RD_GEO / "lk" / "lk_gnd.gpkg",
    place_unit=lambda g: g["gnd"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "Sri Lanka's census asks ethnic group, not language. This map draws each group as its "
        "usual language: Sinhalese as Sinhala; Sri Lankan Tamils, Indian Tamils (Malaiyaga "
        "Thamilar) and Sri Lankan Moors as Tamil; Burghers as English; Malays as Sri Lanka "
        "Malay. People who in 2012 could not speak their group's language are moved to the one "
        "they could: about 60,000 Sri Lankan Tamils and 28,000 Moors are drawn as Sinhala. "
        "People who speak both stay on their group's language, so Moors in the south and west "
        "who speak Sinhala at home are drawn as Tamil. Sri Lanka Malay is drawn for every "
        "ethnic Malay, though younger Malays in Colombo often speak Sinhala or English. The "
        "census moves any group under ten people in a division into 'other'."),
)
