# Sweden. No language statistics: Swedish, plus immigrant languages from SCB's population by
# country of birth and parents' country of birth (31 Dec 2025, per kommun), plus the national
# minority languages from cited estimates, by Parkvall's method (sources/se_build.py). Every row
# derived. Placed on SCB's own 1 km population grid keyed to the 290 kommuner
# (sources/se_geo.py). The record is sources/se.md.
from _shared import *  # noqa: F401,F403


def _counts():
    import se2025
    df = pd.read_csv(NORM / "se.csv", dtype={"geo_id": str})
    df = df[df["geo_level"] == "kommun"]
    if df["geo_id"].nunique() != 290:
        raise SystemExit(f"se.csv: {df['geo_id'].nunique()} kommuner, expected 290")
    df["node"] = df["source_category"].map(se2025.resolve)
    df["unit"] = df["geo_id"]
    return df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()


ENTRY = dict(
    name="Sweden",
    source="SCB Statistikdatabasen, population by region and country of birth, by Swedish or "
           "foreign background, and by parents' country of birth, 31 Dec 2025; Parkvall, "
           "Sveriges språk (Stockholm University 2009) and Sveriges språk i siffror "
           "(Språkrådet 2016); ISOF and minoritet.se on the national minority languages",
    how="no language statistics: immigrants and their children drawn on their birth or "
        "parents' country's languages, following Parkvall's estimates; national minority "
        "languages from published speaker estimates; everyone else Swedish",
    parts=[
        dict(covers="National minority languages",
             source="published speaker estimates for Meänkieli, Sami, Romani and Yiddish "
                    "(Parkvall, ISOF), placed on the kommuner where they are spoken",
             nodes=["uralic.meankieli", "uralic.saami_north", "uralic.saami_lule",
                    "uralic.saami_south", "indoeuropean.indoaryan.romani.romani",
                    "indoeuropean.germanic.continental.yiddish"]),
        dict(covers="Immigrants and their Sweden-born children",
             source="2025 population register, country of birth and parents' country of "
                    "birth, drawn on that country's languages; children at 78% (two parents "
                    "born abroad) or 14% (one)",
             rest=True),
        dict(covers="Everyone else", source="2025 population register, drawn as Swedish",
             nodes=["indoeuropean.germanic.north.swedish"]),
    ],
    grain="290 kommuner, 37,000 people on average; inside a kommun, placed by SCB's 1 km grid",
    gap="1,121 people, 0.01%, whose country of birth the register does not know",
    view=[10.9, 55.2, 24.2, 69.1],
    counts=_counts,
    mappings=["se2025"],
    place=GEO / "se" / "se_grid1km.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "Sweden collects no statistics on language, so this map is an estimate built the way "
        "the linguist Mikael Parkvall estimated Sweden's languages (Stockholm University 2009, "
        "Språkrådet 2016). People born abroad are drawn on the languages of their country of "
        "birth, and their Sweden-born children on the same language at 78% where both parents "
        "were born abroad and 14% where one was, the shares two Swedish studies found. A "
        "quarter of the Finland-born are drawn as Swedish speakers, and so are most people "
        "born in South Korea and Ethiopia, who came as adopted children. People born in Iraq, "
        "Syria, Turkey and Iran are split between their countries' languages in the "
        "proportions Parkvall's 2006 counts imply; Syrians who came later are drawn as Arabic "
        "speakers. Meänkieli is drawn at 30,000 speakers (the Institute for Language and "
        "Folklore gives 50,000 to 75,000 in a wider sense), Sami at 6,000, Romani at 11,000 "
        "and Yiddish at 1,000. Inside each municipality dots follow where people live, not "
        "where each language's speakers do."),
)
