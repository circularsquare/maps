# South Korea. No census or survey asks language. MOIS foreign residents 2020 (2020 census
# population, foreign nationals by nationality per si-gun-gu) read as languages
# (sources/kr_census.py), 229 si-gun-gu, on religiondots' Kontur 400 m hexes. Every row derived.
# Record: sources/kr.md.
from _shared import *  # noqa: F401,F403

CENSUS_2020 = 51_829_136


def _counts():
    import kr2020
    df = pd.read_csv(NORM / "kr.csv", dtype={"geo_id": str})
    if df["geo_id"].nunique() != 229:
        raise SystemExit(f"kr.csv: {df['geo_id'].nunique()} units, expected 229")
    if int(df["count"].sum()) != CENSUS_2020:
        raise SystemExit(f"kr.csv sums to {df['count'].sum():,}, expected {CENSUS_2020:,}")
    # geo_id is already religiondots' unit (sources/kr_census.py joins to kr_lookup.csv)
    df["unit"] = df["geo_id"]
    df["node"] = df["source_category"].map(kr2020.resolve)
    out = by_unit(df)
    out["tier"] = "derived"
    return out


ENTRY = dict(
    name="South Korea",
    source=("Ministry of the Interior and Safety, foreign residents by local government, 1 November "
            "2020 (2020 census population; foreign nationals by nationality and visa type); "
            "Jejueo speakers from the Jejueo Project, University of Hawaii; home languages from "
            "the censuses of eighteen origin countries on this map; immigrants' home-language "
            "shift from France's TeO2 survey (2019 to 2020)"),
    how=("census, 2020, no language question; Korean nationals drawn as Korean, foreign "
         "residents by nationality, each on its country's main language or language mix"),
    parts=[
        dict(covers="Foreign residents, languages other than Korean",
             source="MOIS foreign residents 2020 by nationality, drawn on that country's "
                    "languages; Joseonjok and some marriage migrants moved to Korean",
             people=1_102_063),
        dict(covers="Jejueo speakers on Jeju", source="Jejueo Project estimate, 7,500 drawn",
             nodes=["koreanic.jejueo"]),
        dict(covers="Everyone else", source="2020 census, drawn as Korean", rest=True),
    ],
    grain="229 si/gun/gu, 226,000 people on average",
    gap=("Korean citizens whose first language is not Korean, such as naturalised citizens, "
         "whom no source counts; foreign residents the census missed"),
    view=[125.8, 33.0, 129.7, 38.7],
    counts=_counts,
    mappings=["kr2020"],
    place=RD_GEO / "kr" / "kr_grid_400m.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "South Korea's census asks no language question, so nothing here is a count of a "
        "language. Korean citizens, naturalised ones included, are drawn as Korean. The 1.7 "
        "million foreign nationals are drawn by nationality on their country's languages, "
        "except the 541,000 Chinese citizens of Korean descent (Joseonjok), drawn as Korean. "
        "Marriage migrants are partly drawn as Korean, at a rate taken from a French survey "
        "since no Korean one publishes it. Jejueo has at most 5,000 to 10,000 speakers, nearly "
        "all elderly; 7,500 are drawn across Jeju."),
)
