# Brunei. BPP 2021 Table A3, race by district, split into languages with published speaker
# estimates (sources/bn_build.py): Malays into the puak languages and Brunei Malay, Chinese by
# one national mix, Others into Iban and foreign residents by nationality. Brunei Malay rows
# `derived` (ethnic group read as language), every other row `modelled`. On religiondots' Kontur
# hexes for the four districts (read-only). The record is sources/bn.md.
from _shared import *  # noqa: F401,F403

TOTAL = 440_715


def _counts():
    import bn2021
    df = pd.read_csv(NORM / "bn.csv")
    if df["geo_id"].nunique() != 4:
        raise SystemExit("bn.csv: expected 4 districts; re-run sources/bn_build.py")
    if int(df["count"].sum()) != TOTAL:
        raise SystemExit(f"bn.csv sums to {int(df['count'].sum()):,}, not {TOTAL:,}")
    if not set(df["tier"]) <= {"derived", "modelled"}:
        raise SystemExit(f"bn.csv: tiers {sorted(set(df['tier']))}")
    df["node"] = df["source_category"].map(bn2021.resolve)
    df["unit"] = df["geo_id"]
    out = df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()
    return out[["unit", "node", "count", "tier"]]


ENTRY = dict(
    name="Brunei",
    source=("Population and Housing Census 2021 (Department of Economic Planning and Statistics), "
            "Table A3, race by district; split with speaker estimates from Ethnologue, Dewan "
            "Bahasa dan Pustaka Brunei, Omniglot, Joshua Project and Asia Harvest, and foreign "
            "residents by nationality (work pass figures, Government of India, UN DESA 2020)"),
    how=("census, 2021, race; Malays, Chinese and Others split into languages with published "
         "speaker estimates placed on their home districts"),
    parts=[
        dict(covers="Tutong, Kedayan, Dusun, Belait and Murut",
             source="speaker estimates (Ethnologue, Dewan Bahasa dan Pustaka Brunei, Joshua "
                    "Project), placed on their home districts",
             people=57_800),
        dict(covers="Chinese", source="2021 census, race; one national mix of Chinese "
             "languages and English (Asia Harvest, Joshua Project)", people=42_132),
        dict(covers="Iban", source="Omniglot's speaker estimate, placed in Belait, Tutong and "
             "Temburong", people=15_800),
        dict(covers="Foreign residents",
             source="2021 census Others less the Iban, by nationality (work pass figures, "
                    "Government of India, UN DESA 2020)", people=85_767),
        dict(covers="Other Malays", source="2021 census, race, drawn as Brunei Malay",
             rest=True),
    ],
    grain="4 districts, 110,000 people on average",
    gap=("the census asked home language but never published it; every figure under the three "
         "races is an estimate"),
    view=[113.95, 4.0, 115.4, 5.1],
    counts=_counts,
    mappings=["bn2021"],
    place=RD_GEO / "bn" / "bn_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "Brunei's 2021 census asked which language people mainly speak at home, but the answers "
        "were never published. What is published is race by district: Malays, Chinese and "
        "Others. This map splits each into languages using published estimates, so none of the "
        "languages drawn here is a count. The census's Malays include the seven native groups; "
        "Tutong, Kedayan, Dusun (with Bisaya), Belait and Murut are drawn at published speaker "
        "estimates in the districts they come from, and the rest of the Malays as Brunei Malay. "
        "These smaller languages are all losing speakers to Brunei Malay, and Kedayan is so "
        "close to it that many speak something between the two. The Chinese are drawn with one "
        "mix in every district: 16% English, 30% Mandarin, and the rest Hokkien, Cantonese, "
        "Foochow and Hakka. Others are the Iban and foreign residents, most of them temporary "
        "workers, drawn by nationality."),
)
