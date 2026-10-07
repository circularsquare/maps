# Israeli settlers beyond the Green Line (West Bank settlements and East Jerusalem's Israeli
# neighbourhoods). Not a country: religiondots' `xs` entry, same units and same line, so nobody here
# is on Israel's or Palestine's entry. CBS Social Survey 2021 native language (Jews and others, Judea
# and Samaria or Jerusalem sub-district) applied to each unit's 2022 census population
# (sources/xs_social.py). Placed on religiondots' Kontur pieces of those units. Record: sources/xs.md.
import json

from _shared import *  # noqa: F401,F403

DRAWN = 723_899


def _counts():
    import xs2021
    df = pd.read_csv(NORM / "xs.csv", dtype={"geo_id": str})
    dropped = set(json.loads((RD_GEO / "il" / "dropped_units.json").read_text()))
    inside = ~df["geo_id"].isin(dropped)
    if inside.any():
        raise SystemExit(f"xs.csv carries {df.loc[inside, 'geo_id'].nunique()} units inside the "
                         "Green Line, which Israel's entry draws; re-run sources/xs_social.py")
    if abs(df["count"].sum() - DRAWN) > 5:
        raise SystemExit(f"xs.csv sums to {df['count'].sum():,.0f}, expected {DRAWN:,}")
    df["node"] = df["source_category"].map(xs2021.resolve)
    df["unit"] = df["geo_id"]
    out = by_unit(df)
    out["tier"] = "modelled"
    return out


ENTRY = dict(
    name="Israeli settlements in the West Bank",
    source=("Central Bureau of Statistics: Social Survey 2021, native language, Judea and Samaria "
            "and Jerusalem sub-districts (table generator); 2022 Census of Population and "
            "Housing, units beyond the 1949 armistice line"),
    how=("a survey, 2021, native language of adults, applied to each area's 2022 census "
         "population"),
    grain=("2 survey sub-districts for the language shares; drawn on 267 statistical areas and "
           "localities, about 2,700 people each"),
    gap=("Muslims and Christians in the same areas, nearly all in East Jerusalem, are on "
         "Palestine's entry"),
    view=[34.9, 31.3, 35.6, 32.6],
    counts=_counts,
    mappings=["xs2021"],
    place=RD_GEO / "xs" / "xs_places.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "These are the Israelis living beyond the 1949 armistice line, in the West Bank "
        "settlements and in East Jerusalem: 723,899 people the 2022 census recorded as Jews or "
        "with no religion, 237,240 of them in East Jerusalem. Israel's entry stops at the "
        "armistice line and Palestine's census does not count them, so they are drawn here, as "
        "part of neither country. No census asks language. The Central Bureau of Statistics' "
        "Social Survey of 2021 asked adults their native language; the settlements take the "
        "answers from its Judea and Samaria district, and East Jerusalem those of Jews and others "
        "in the Jerusalem sub-district, where the survey files it. Children are given the answers "
        "of adults aged 20 to 24 across Israel. Between areas, Yiddish is drawn more heavily where "
        "more households are ultra-Orthodox, Russian where more people are of European origin and "
        "English where more are of American origin; the totals stay the survey's. Muslims and "
        "Christians in the same areas are drawn on Palestine's entry. Within each area the dots "
        "follow Kontur's population grid."),
)
