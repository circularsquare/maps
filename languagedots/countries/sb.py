# Solomon Islands. 2019 census, first language learnt as a child, printed for the whole country
# only (sources/sb_census.py), placed on wards by a model built from the census's own province
# tables and the languages' Glottolog locations (sources/sb_model.py), on religiondots' Kontur
# hexes for the 183 wards (read-only). sources/sb.md is the record.
from _shared import *  # noqa: F401,F403

# True draws sources/sb_model.py's placement (every row `modelled`; run that script after
# sb_census.py). False draws the census's national figures as one unit spread over the country
# by population; `how`, `grain` and `note_public` then need the national wording (sources/sb.md).
MODEL = True


def _counts():
    import sb2019
    if MODEL:
        df = pd.read_csv(NORM / "sb_model.csv", dtype={"geo_id": str})
        if df["unit"].nunique() != 183:
            raise SystemExit(f"sb_model.csv: {df['unit'].nunique()} wards, expected 183")
    else:
        df = pd.read_csv(NORM / "sb.csv")
        df["unit"], df["tier"] = "SB", "measured"
    unresolved = sorted(set(df["source_category"]) - set(sb2019.NAMES))
    if unresolved:
        raise SystemExit(f"sb categories with no mapping: {unresolved}")
    df["node"] = df["source_category"].map(sb2019.resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    return df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()


def _place_unit(g):
    u = g["unit"].astype(str)
    return u if MODEL else u.map(lambda _: "SB")


ENTRY = dict(
    name="Solomon Islands",
    source=("2019 National Population and Housing Census, National Report Volume 1, Tables "
            "9.6.1-9.6.3 (Solomon Islands National Statistics Office)"),
    how=("model: the 2019 census's first language, aged 5 and over, printed for the whole "
         "country only and placed by where each language is spoken"),
    parts=[
        dict(covers="Pidgin", source="2019 census, first language, placed by the census's own "
                                     "figure for each province",
             nodes=["creole.english_based.pijin"]),
        dict(covers="Local and other languages",
             source="2019 census, first language, national figures placed on each language's "
                    "home (Glottolog) and spread the way people born there moved",
             rest=True),
    ],
    grain="183 wards, 3,400 people aged 5 and over on average; languages within each modelled",
    gap="children under 5",
    view=[155.3, -12.5, 170.3, -5.0],
    counts=_counts,
    mappings=["sb2019"],
    place=RD_GEO / "sb" / "sb_hexes.gpkg",
    place_unit=_place_unit,
    place_weight=pop_weight,
    note_public=(
        "The 2019 census asked everyone aged 5 and over which language they learnt first, but "
        "it prints the answers only for the whole country, so the places on this map are "
        "modelled, not counted. Each language is drawn where it is spoken, and its speakers "
        "are shared between that home and the rest of the country the way the census found "
        "people born in its home province had moved. Pidgin is placed by the census's own "
        "figure for each province; 47% of those who learnt it first live in Honiara. The "
        "report names Pidgin and 58 local languages. The other 90,859 people, 14%, gave "
        "English, a foreign language or a local language the report does not list, such as "
        "Tikopia or Natügu, and are drawn in grey as language not named. Some of the smallest "
        "names are, in the report's words, probably tribe or lineage names rather than "
        "languages."),
)
