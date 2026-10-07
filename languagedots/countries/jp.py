# Japan. No census or survey asks language. 2020 census: Japanese nationals on Japanese, foreign
# residents by nationality read as their languages, the Ryukyuan languages from Okinawa's
# shimakutuba surveys and Ethnologue (sources/jp_census.py), 1,896 municipalities and wards, on
# Kontur 400 m hexes (sources/jp_geo.py). Every row derived. Record: sources/jp.md.
from _shared import *  # noqa: F401,F403

CENSUS_2020 = 126_146_099


def _counts():
    import jp2020
    df = pd.read_csv(NORM / "jp.csv", dtype={"geo_id": str})
    if int(df["count"].sum()) != CENSUS_2020:
        raise SystemExit(f"jp.csv sums to {df['count'].sum():,}, expected {CENSUS_2020:,}")
    df["unit"] = df["geo_id"]
    df["node"] = df["source_category"].map(jp2020.resolve)
    out = by_unit(df)
    out["tier"] = "derived"
    return out


ENTRY = dict(
    name="Japan",
    source=("Statistics Bureau of Japan, 2020 Population Census, table 44-1 (population by "
            "nationality, municipalities); Immigration Services Agency, resident foreigners by "
            "prefecture and nationality, December 2020; Aichi Prefecture foreign residents survey "
            "2022 (language used with children); Okinawa Prefecture shimakutuba surveys 2023 "
            "and 2024; Ethnologue speaker figures for the Amami languages; home languages from "
            "the maps of China, Vietnam, the Philippines and sixteen other countries"),
    how=("census, 2020, no language question; Japanese nationals drawn as Japanese, Ryukyuan "
         "languages from Okinawa surveys and Ethnologue, foreign residents by nationality"),
    parts=[
        dict(covers="Ryukyuan languages",
             source="Okinawa shimakutuba surveys 2023 and 2024; Ethnologue (2004) for Amami",
             people=131_526),
        dict(covers="Foreign residents, languages other than Japanese",
             source="2020 census, nationality, drawn on that country's languages; part moved "
                    "to Japanese by Aichi's 2022 survey of language used with children",
             people=1_593_767),
        dict(covers="Everyone else", source="drawn as Japanese", rest=True),
    ],
    grain="1,896 municipalities and wards, 67,000 people on average",
    gap=("other languages among Japanese nationals: naturalised citizens, Ainu, children of "
         "mixed marriages"),
    view=[122.9, 24.0, 146.2, 45.6],
    counts=_counts,
    mappings=["jp2020"],
    place=GEO / "jp" / "jp_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "Japan's census asks nationality, not language, so nothing here is a count of a "
        "language. Japanese nationals are drawn as Japanese, except for Ryukyuan speakers: in "
        "Okinawa, adults who say they mainly speak their island's language, and half of those "
        "who speak it as much as Japanese, each town on its traditional language. The Amami "
        "Islands use Ethnologue's 2004 estimates, probably too high today. Ainu has too few "
        "speakers for a dot. Foreign residents are drawn on their country's languages, less "
        "the share who always speak Japanese with their children in Aichi Prefecture's 2022 "
        "survey (81% of Koreans, 30% of Chinese)."),
)
