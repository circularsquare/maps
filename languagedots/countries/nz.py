# New Zealand. Census 2023, languages spoken, by SA1 (sources/nz_census.py), each person shared
# across the languages they named using the official language indicator. Placed on religiondots'
# SA1 polygons, read-only, one polygon per unit. The record is sources/nz.md.
from _shared import *  # noqa: F401,F403


def _counts():
    import nz2023
    df = pd.read_csv(NORM / "nz.csv", dtype={"geo_id": str, "sa2": str})
    df["node"] = df["source_category"].map(nz2023.resolve)
    if df["node"].isna().any():
        raise SystemExit(f"nz.csv categories that resolve to nothing: "
                         f"{sorted(set(df.loc[df['node'].isna(), 'source_category']))}")
    df = df[df["count"] > 0]
    df["unit"] = df["geo_id"]
    # every row is a person shared across the languages they named (spec §3.6)
    return df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()


def _place_unit(g):
    # LANDWATER 21 is inland water: 71 SA1s holding six people, kept out of the counts too
    unit = g["SA12023_V1_00"].astype(str)
    return unit.where(g["LANDWATER"].astype(str) != "21")


def _weight(place):
    # the layer carries the 2023 census population as VAR_1_3; one polygon per SA1, so this
    # only matters where religiondots' clip split an SA1 into pieces
    return pop_weight(place.assign(pop=place["VAR_1_3"].clip(lower=0)))


ENTRY = dict(
    name="New Zealand",
    source="Census 2023, languages spoken and official language indicator by SA1, and languages "
           "spoken by SA2 (Aotearoa Data Explorer, CEN23_ECI_011) (Stats NZ)",
    how="census, 2023, languages spoken, several allowed; each person shared across the "
        "languages they named",
    parts=[
        dict(covers="English, Māori, Samoan, NZ Sign Language, other",
             source="2023 census, languages spoken, by SA1; each person shared across the "
                    "languages they named",
             rest=True),
        dict(covers="Eleven more languages, split out of other",
             source="2023 census, languages spoken, by SA2, applied to each SA2's small areas",
             nodes=["sinotibetan.sinitic", "indoeuropean.indoaryan.central.hindi",
                    "austronesian.philippine.tagalog",
                    "indoeuropean.indoaryan.northwestern.punjabi", "indoeuropean.romance.french",
                    "indoeuropean.germanic.continental.afrikaans", "indoeuropean.romance.spanish",
                    "austronesian.oceanic.tongan", "indoeuropean.germanic.continental.german"]),
    ],
    grain="32,700 statistical areas (SA1), 150 people on average",
    gap="105,400 people, 2.1%: 104,847 too young to talk, and 726 counted in SA1s of inlets "
        "and sea",
    view=[166.0, -47.5, 178.8, -34.2],
    counts=_counts,
    mappings=["nz2023"],
    place=RD_GEO / "nz" / "sa1_2023_clipped.geojson",
    place_unit=_place_unit,
    place_weight=_weight,
    note_public=(
        "The census asked which languages a person could hold an everyday conversation in, as "
        "many as they liked, so it counts what people can speak rather than a first language. "
        "Each person is drawn shared between the languages they named, so Māori, named by "
        "213,800 people, is drawn as about 108,000. The smallest areas' tables name only "
        "English, Māori, Samoan and New Zealand Sign Language; eleven more languages are "
        "placed by the mix in areas of about 2,100 people. Every other language, from Korean "
        "to Fijian, is drawn together as other: about 187,000 people."),
)
