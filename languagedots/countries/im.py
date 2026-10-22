# Isle of Man. 2021 census Table 2.11, language mainly spoken at home (sources/im_census.py),
# island-wide only. Placed on religiondots' Kontur hexes for the island as one unit (read-only).
# Record: sources/im.md.
from _shared import *  # noqa: F401,F403


def _counts():
    import im2021
    df = pd.read_csv(NORM / "im.csv")
    if df["geo_id"].nunique() != 1 or int(df["count"].sum()) != 84_028:
        raise SystemExit("im.csv: expected 84,028 people in one island unit -- run sources/im_census.py")
    df["node"] = df["source_category"].map(im2021.resolve)
    df["unit"] = "IM"   # religiondots' im_hexes.gpkg: one unit, IM
    out = by_unit(df)
    out["tier"] = "measured"
    return out


ENTRY = dict(
    name="Isle of Man",
    source="2021 Isle of Man Census, Report Part I, Table 2.11 (Economic Affairs Division)",
    how="census, 2021, language used most in the home",
    parts=[dict(covers="Residents who answered", source="2021 census, language mainly spoken at home",
                rest=True)],
    grain="the island as one unit, 84,000 people",
    gap="41 residents who gave no answer",
    view=[-4.85, 54.03, -4.30, 54.42],
    counts=_counts,
    mappings=["im2021"],
    place=RD_GEO / "im" / "im_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "The 2021 census asked each resident which language they use most in the home. English "
        "is 96.1%, Polish 0.6%, Filipino 0.5%, Bulgarian 0.4% and Chinese 0.4%. The table "
        "prints 11 languages and groups the rest as Other, 1.2%, which is drawn as unnamed. "
        "Manx was asked about separately: 2,223 residents said they can speak, read or write "
        "it, and 1,005 of them can only speak it. That is ability, not the language used most "
        "at home, so Manx has no dots of its own. The table is for the whole island, so the "
        "dots follow where people live."),
)
