# Israel. No census asks language. CBS Social Survey 2021 native language (age 20+) by sub-district
# and population group, applied to each statistical area's 2022 census population and raked inside
# each sub-district on the area's origin profile (sources/il_social.py). Drawn on religiondots'
# statistical-area polygons, which stop at the Green Line and keep the Golan. Record: sources/il.md.
import json

from _shared import *  # noqa: F401,F403

DRAWN = 8_418_491


def _counts():
    import il2021
    df = pd.read_csv(NORM / "il.csv", dtype={"geo_id": str})
    dropped = set(json.loads((RD_GEO / "il" / "dropped_units.json").read_text()))
    inside = df["geo_id"].isin(dropped)
    if inside.any():
        raise SystemExit(f"il.csv carries {df.loc[inside, 'geo_id'].nunique()} units beyond the "
                         "Green Line; re-run sources/il_social.py")
    if abs(df["count"].sum() - DRAWN) > 5:
        raise SystemExit(f"il.csv sums to {df['count'].sum():,.0f}, expected {DRAWN:,}")
    df["node"] = df["source_category"].map(il2021.resolve)
    df["unit"] = df["geo_id"]
    out = by_unit(df)
    out["tier"] = "modelled"
    return out


ENTRY = dict(
    name="Israel",
    source=("Central Bureau of Statistics: Social Survey 2021, native language, by sub-district "
            "and population group (table generator); 2022 Census of Population and Housing, "
            "statistical areas"),
    how=("a survey, 2021, native language of adults, applied to each statistical area's 2022 "
         "census population"),
    parts=[dict(covers="Everyone", source="Social Survey 2021, native language of adults, "
                "by sub-district and population group", rest=True)],
    grain="15 sub-districts for the shares, drawn on 2,968 statistical areas",
    gap="West Bank settlements and East Jerusalem, beyond the 1949 armistice line",
    view=[34.2, 29.4, 35.95, 33.35],
    counts=_counts,
    mappings=["il2021"],
    # il_units.gpkg cut by Kontur hexes, so an area's dots follow where its people live
    # (sources/kontur_cut.py; 2026-10-08)
    place=GEO / "il" / "il_konturcut.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "Israel's census asks no language question. The Social Survey of 2021 asked adults "
        "their native language; its shares by sub-district, for Arabs and for Jews and others, "
        "are applied to the 2022 census population. Children are given the answers of adults "
        "aged 20 to 24. Inside a sub-district, languages are placed by the origin of each "
        "area's people. Languages beyond the nine the survey names are drawn as other. The map "
        "stops at the 1949 armistice line and includes the Golan Heights. Israelis in the West "
        "Bank settlements and East Jerusalem are their own entry, and East Jerusalem's "
        "Palestinians are on Palestine's entry."),
)
