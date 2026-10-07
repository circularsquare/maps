# Türkiye. No census has asked mother tongue since 1965. KONDA's Biz Kimiz? survey (2006, 47,958
# adults) gives the national mother-tongue split and the Kurdish share of each İBBS-1 region; the
# 1965 census places people inside those; DGMM's Syrians under temporary protection (1 Oct 2026)
# are added as Arabic (sources/tr_konda.py). Kontur hexes keyed to the 81 provinces
# (sources/tr_geo.py). The record is sources/tr.md.
from _shared import *  # noqa: F401,F403

POP = 86_092_168 + 2_206_483


def _counts():
    import tr2006
    df = pd.read_csv(NORM / "tr.csv", dtype={"geo_id": str}, keep_default_na=False)
    df = df[df["geo_level"] == "province"].copy()
    if df["geo_id"].nunique() != 81:
        raise SystemExit(f"tr.csv: {df['geo_id'].nunique()} provinces, expected 81")
    if df["count"].sum() != POP:
        raise SystemExit(f"tr.csv sums to {df['count'].sum():,}, expected {POP:,}")
    # the hex layer's `unit` is the ASCII province name (sources/tr_geo.py)
    df["unit"] = df["geo_id"]
    df["node"] = df["source_category"].map(tr2006.resolve)
    df = df[df["count"] > 0]
    return df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()


ENTRY = dict(
    name="Türkiye",
    source=("KONDA, Biz Kimiz? Toplumsal Yapı Araştırması 2006 (mother tongue, national; Kurdish "
            "share by region), placed by the 1965 census's mother tongue by province; Syrians "
            "under temporary protection from the Presidency of Migration Management, 1 October "
            "2026"),
    how=("a survey, 2006, mother tongue, applied to the 2025 register population and placed "
         "within regions by the 1965 census; Syrians under temporary protection added as Arabic"),
    parts=[
        dict(covers="Turkish citizens and residents",
             source="KONDA 2006, mother tongue, 47,958 adults; national shares and the Kurdish "
                    "share of 12 regions, placed by the 1965 census",
             rest=True),
        dict(covers="Syrians under temporary protection",
             source="Presidency of Migration Management, October 2026, by province, drawn as "
                    "Arabic",
             people=2_206_483),
    ],
    grain="81 provinces, 1.06 million people on average, modelled from 12 regions",
    gap="none left out; nothing here was counted, every figure is modelled",
    view=[25.6, 35.8, 44.9, 42.2],
    counts=_counts,
    mappings=["tr2006"],
    place=GEO / "tr" / "tr_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "Türkiye's census last asked mother tongue in 1965, and its statistical office publishes "
        "no survey of it. These dots come from a 2006 survey by the polling firm KONDA, which "
        "asked 47,958 adults the language they learned from their mother. It gives the national "
        "split and the share of Kurds and Zazas in each of twelve statistical regions, applied "
        "to the 2025 population register. Within each region, and for the smaller languages, "
        "people are placed where the 1965 census found speakers of the same language; in the "
        "western regions, where most Kurdish speakers moved in after 1965, they follow "
        "population. Provinces are therefore estimates, not counts, and the split between "
        "Kurdish and Zazaki in places such as Tunceli is weak. The 2.2 million Syrians under "
        "temporary protection are drawn as Arabic speakers, though some are Kurds or "
        "Turkmens."),
)
