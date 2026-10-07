# Iceland. No language question: Icelandic citizens on Icelandic, foreign citizens on their
# country's language or drawn mix less TeO2 retention, per municipality, from Hagstofa MAN04203
# (1 January 2026; sources/is_pop.py). Every row `derived`. Placed on religiondots' 400 m Kontur
# grid, keyed by its municipality column (read-only). Record: sources/is.md.
from _shared import *  # noqa: F401,F403

MUNIS = 62
TOTAL_2026 = 394_324


def _counts():
    import is2026
    df = pd.read_csv(NORM / "is.csv", dtype={"geo_id": str})
    if df["geo_id"].nunique() != MUNIS or int(df["count"].sum()) != TOTAL_2026:
        raise SystemExit("is.csv is not 62 municipalities summing to 1 Jan 2026 -- run "
                         "sources/is_pop.py")
    df["node"] = df["source_category"].map(is2026.resolve)
    df["unit"] = df["geo_id"].str.zfill(4)
    out = by_unit(df)
    out["tier"] = "derived"
    return out


ENTRY = dict(
    name="Iceland",
    source=("Statistics Iceland (Hagstofa), population by municipality and citizenship, "
            "1 January 2026 (MAN04203); immigrant languages from citizenship, less the share "
            "who speak only the host language at home in France's TeO2 survey (INED-INSEE)"),
    how=("no language question: Icelandic citizens drawn as Icelandic, foreign citizens by "
         "their country's language, 2026 register"),
    parts=[
        dict(covers="Foreign citizens, languages other than Icelandic",
             source="2026 register, citizenship, drawn on that country's languages; about a "
                    "third moved to Icelandic by France's TeO2 survey",
             people=46_854),
        dict(covers="Everyone else", source="drawn as Icelandic", rest=True),
    ],
    grain="62 municipalities, 6,400 people on average",
    gap="Icelandic citizens with another first language, naturalised immigrants among them",
    view=[-24.6, 63.2, -13.4, 66.6],
    counts=_counts,
    mappings=["is2026"],
    place=RD_GEO / "is" / "is_grid_400m.gpkg",
    place_unit=lambda g: g["muni"].astype(str).str.zfill(4),
    place_weight=pop_weight,
    note_public=(
        "Iceland's population register records citizenship, not language, so Icelandic "
        "citizens are drawn as Icelandic speakers and the 70,555 foreign citizens on the "
        "languages of their country. The share of immigrants who speak only the host language "
        "at home in France's TeO2 survey, about a third of Europeans, is drawn as Icelandic. "
        "Naturalised Icelanders who grew up with another language are drawn as Icelandic too."),
)
