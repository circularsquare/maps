# Benin. RGPH-4 2013 "primary language spoken in the household" (IPUMS 10% sample, commune
# shares via CLEAR Global), read as first language through the census's full-count ethnic
# clusters (Tableau 8), on each commune's census population (sources/bj_census.py); every row
# `modelled`. On religiondots' Kontur hexes for the 77 communes. The record is sources/bj.md.
from _shared import *  # noqa: F401,F403

COMMUNES = 77
CENSUS_2013 = 10_008_749


def _counts():
    import bj2013
    df = pd.read_csv(NORM / "bj.csv", dtype={"geo_id": str})
    if df["geo_id"].nunique() != COMMUNES:
        raise SystemExit(f"bj.csv: {df['geo_id'].nunique()} communes, expected {COMMUNES} -- "
                         "re-run sources/bj_census.py")
    if int(df["count"].sum()) != CENSUS_2013:
        raise SystemExit(f"bj.csv sums to {int(df['count'].sum()):,}, not {CENSUS_2013:,}")
    df["node"] = df["source_category"].map(bj2013.resolve)
    missing = sorted(set(df.loc[df["node"].isna(), "source_category"]))
    if missing:
        raise SystemExit(f"bj.csv answers with no node: {missing}")
    lut = pd.read_csv(RD_GEO / "bj" / "bj_lookup.csv", dtype=str)
    if set(df["geo_id"]) != set(lut["unit"]):
        raise SystemExit("bj.csv communes do not match religiondots' bj_lookup.csv")
    df["unit"] = df["geo_id"]
    df = df[df["count"] > 0]
    out = by_unit(df)
    out["tier"] = "modelled"
    return out


ENTRY = dict(
    name="Benin",
    source=("Recensement Général de la Population et de l'Habitation 2013 (RGPH-4, INStaD): "
            "question 17, primary language spoken in the household, from the IPUMS "
            "International 10% sample as tabulated by commune by CLEAR Global (HDX, CC BY-SA); "
            "Tableau 2 (population) and Tableau 8 (ethnic group) of the twelve departmental "
            "Principaux indicateurs"),
    how=("census, 2013, main language spoken in the household (people 3 and over, 10% "
         "sample), read as first language: each commune's ethnic groups at their full-count "
         "shares, split into their languages by the answers"),
    parts=[dict(covers="Everyone",
                source="2013 census, main household language (IPUMS 10% sample via CLEAR "
                       "Global), within each commune's counted ethnic groups", rest=True)],
    grain="77 communes, 130,000 people on average",
    gap=("answers recorded as unknown (0.4%), and the 0.9% of other Beninese ethnic groups, "
         "whose language the answers do not name, are drawn on their commune's mix; French "
         "and English as household languages (1%) are not drawn as first languages"),
    view=[0.6, 6.1, 4.0, 12.6],
    counts=_counts,
    mappings=["bj2013"],
    place=RD_GEO / "bj" / "bj_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "Benin's 2013 census asked everyone aged 3 and over which language they mainly speak "
        "at home. The office has not published a table of the answers, so this map uses the "
        "10% sample of census records held by IPUMS International, tabulated by commune by "
        "CLEAR Global. Some answers name a common language rather than a person's own: French "
        "was named by 6% of people in Cotonou, and Dendi, the market language of the north, by "
        "more people in Malanville, Kandi and Parakou than are Dendi by ethnic group. The same "
        "census counted every resident's ethnic group in nine broad groups, so each commune "
        "is drawn with those groups at their counted size, each split into languages by how "
        "that group's languages were named in the commune. Drawn straight from the answers, "
        "Dendi would be 3.3% of Benin instead of 2.4%, and French 0.9% instead of none."),
)
