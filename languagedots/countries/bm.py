# Bermuda. No census language question: the 2016 census's country of birth, national
# (sources/bm_census.py): the Bermuda-born on English, the foreign-born on their birth country's
# languages through origin_mix. Every row derived. Placed on religiondots' Kontur hexes for the
# country as one unit (read-only). Record: sources/bm.md.
from _shared import *  # noqa: F401,F403


def _counts():
    import bm2016
    df = pd.read_csv(NORM / "bm.csv")
    if df["geo_id"].nunique() != 1 or abs(df["count"].sum() - 63_743) > 1:
        raise SystemExit("bm.csv: expected 63,743 people in one national unit -- "
                         "run sources/bm_census.py")
    df["node"] = df["source_category"].map(bm2016.resolve)
    df["unit"] = "BM"   # religiondots' bm_hexes.gpkg: one unit, BM
    out = by_unit(df)
    out["tier"] = "derived"
    return out


ENTRY = dict(
    name="Bermuda",
    source=("2016 Population and Housing Census Report (Department of Statistics, Bermuda), "
            "Table 1 and Table 4.5, country of birth"),
    how=("no language question: people born in Bermuda drawn as English speakers, people born "
         "abroad on their birth country's languages; census 2016"),
    parts=[
        dict(covers="People born abroad",
             source="2016 census, country of birth, drawn on that country's languages",
             people=19_332),
        dict(covers="People born in Bermuda", source="2016 census, drawn as English",
             rest=True),
    ],
    grain="the country as one unit, 63,800 people",
    gap="36 people whose birthplace was not stated",
    view=[-64.90, 32.24, -64.63, 32.40],
    counts=_counts,
    mappings=["bm2016"],
    place=RD_GEO / "bm" / "bm_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "Bermuda's census does not ask about language. The 70% of residents born in Bermuda "
        "are drawn as English speakers; Bermudian English is a variety of English, not a "
        "creole. People born abroad are drawn on the languages of their birth country, such as "
        "Portuguese for those born in the Azores and Portugal. Bermudians of Portuguese descent "
        "born on the island are drawn as English speakers, since most grew up speaking "
        "English. Birthplace is published for the country as a whole, so every language's dots "
        "follow population alone."),
)
