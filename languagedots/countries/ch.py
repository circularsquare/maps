# Switzerland. Volkszählung 2000, main language by commune (sources/ch_vz2000.py), moved onto the
# 1.1.2020 communes through BFS's correspondence, on religiondots' Kontur hexes plus Verzasca
# (sources/ch_geo.py). The record is sources/ch.md.
from _shared import *  # noqa: F401,F403

UNITS = 2198


def _counts():
    import ch2000
    df = pd.read_csv(NORM / "ch.csv", dtype={"geo_id": str})
    df = df[df["geo_level"] == "commune"]
    if df["geo_id"].nunique() != UNITS:
        raise SystemExit(f"ch.csv: {df['geo_id'].nunique()} communes, expected {UNITS:,}")
    df["node"] = df["source_category"].map(ch2000.resolve)
    if df["node"].isna().any():
        raise SystemExit(f"ch.csv categories that resolve to nothing: "
                         f"{sorted(set(df.loc[df['node'].isna(), 'source_category']))}")
    df["unit"] = df["geo_id"]
    # "measured", or "modelled" on the German split into Swiss and Standard German
    df["tier"] = df["tier"].where(df["tier"].isin(["measured", "modelled"]), "measured")
    return df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()


ENTRY = dict(
    name="Switzerland",
    source="Eidgenössische Volkszählung 2000, table px-x-4003000000_123, main language by "
           "commune (Federal Statistical Office, BFS)",
    how="census, 2000, main language by commune; German split into Swiss and Standard German "
        "at the same census's family-language rates by language region and nationality, "
        "applied to each commune's Swiss and foreign German speakers",
    parts=[
        dict(covers="German speakers",
             source="2000 census, main language German, split into Swiss and Standard German "
                    "by the same census's family-language rates (Lüdi and Werlen 2005)",
             people=4_639_870),
        dict(covers="Everyone else", source="2000 census, main language", rest=True),
    ],
    grain="2,198 communes as of 2020, 3,300 people on average",
    view=[5.8, 45.75, 10.6, 47.85],
    counts=_counts,
    mappings=["ch2000"],
    place=GEO / "ch" / "ch_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "This is Switzerland in 2000, the last census that asked everyone. The question was "
        "the one language a person thinks in and knows best, and it had one box for German. The "
        "split into Swiss German and Standard German is modelled: the same census asked which "
        "languages people speak at home with separate boxes for the two, but published that "
        "only by language region and nationality, so each commune's Swiss and foreign German "
        "speakers are split at those rates. Standard German falls mostly on immigrants "
        "and on German speakers in French and Italian Switzerland. Since 2010 the Federal Statistical Office asks only a sample and below the "
        "national level publishes only the national languages, English and other. In its 2024 "
        "figures English is named by 6.5% of residents against 1.0% in 2000, and Albanian and "
        "Portuguese by 3.4% each against 1.3% and 1.2%. The census merged some languages into "
        "one answer (Spanish includes Catalan and Galician, for one), and Tamil is not printed "
        "by commune and is drawn as other."),
)
