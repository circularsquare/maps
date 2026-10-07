# Slovenia. Popis 2002 mother tongue by obcina (sources/si_popis2002.py), on religiondots' Kontur
# hexes for the 192 municipalities of 2002 (read only; keyed by the official obcina code). The
# record is sources/si.md.
from _shared import *  # noqa: F401,F403

UNITS = 192


def _counts():
    import si2002
    df = pd.read_csv(NORM / "si.csv", dtype={"geo_id": str})
    df = df[df["geo_level"] == "municipality"].copy()
    if df["geo_id"].nunique() != UNITS:
        raise SystemExit(f"si.csv: {df['geo_id'].nunique()} units, expected {UNITS}")
    df["node"] = df["source_category"].map(si2002.resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df["unit"] = df["geo_id"]
    return by_unit(df)


ENTRY = dict(
    name="Slovenia",
    source="Popis prebivalstva 2002, table 05W1007S, population by mother tongue, by "
           "municipalities (Statistical Office of the Republic of Slovenia)",
    how="census, 2002, mother tongue",
    parts=[dict(covers="Everyone", source="2002 census, mother tongue", rest=True)],
    grain="192 municipalities as they stood in 2002, 10,200 people on average",
    gap="52,316 people, 2.7%, whose mother tongue is unknown; and 1,726 in cells the office "
        "withheld to protect small numbers",
    view=[13.3, 45.4, 16.7, 46.95],
    counts=_counts,
    mappings=["si2002"],
    place=RD_GEO / "si" / "si_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "This is Slovenia in 2002, the last census that asked about language; the censuses "
        "of 2011 and 2021 were drawn from registers and carry none. Mother tongue was one "
        "answer per person. "
        "Hungarian and Italian are the two minority languages with legal standing, in the "
        "bilingual areas along the Hungarian border and on the coast. The languages of the "
        "former Yugoslavia are mostly those of people who moved to Slovenia's industrial "
        "towns. The census printed Serbo-Croatian (36,265 people) apart from Serbian and "
        "Croatian, and so does this map. Other languages are one figure per municipality."),
)
