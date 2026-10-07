# Djibouti. RGPH-3 2024, Tome 4 Tableau 7: languages spoken, several allowed, by region; each
# person shared across the local languages named, French, English and Arabic folded as learned
# languages; native Arabic from Joshua Project's Arab groups, in the capital (sources/dj_rgph.py).
# On religiondots' Kontur hexes for the six regions. The record is sources/dj.md.
from _shared import *  # noqa: F401,F403

TOTAL_2024 = 1_003_800
UNITS = 6


def _counts():
    import dj2024
    df = pd.read_csv(NORM / "dj.csv")
    if df["geo_id"].nunique() != UNITS:
        raise SystemExit(f"dj.csv: {df['geo_id'].nunique()} regions, expected {UNITS} -- "
                         "re-run sources/dj_rgph.py")
    if int(df["count"].sum()) != TOTAL_2024:
        raise SystemExit(f"dj.csv sums to {int(df['count'].sum()):,}, not {TOTAL_2024:,}")
    lut = pd.read_csv(RD_GEO / "dj" / "dj_lookup.csv", dtype=str)
    if set(df["geo_id"]) != set(lut["unit"]):
        raise SystemExit("dj.csv regions do not match religiondots' dj_lookup.csv units")
    df["unit"] = df["geo_id"]
    df["node"] = df["source_category"].map(dj2024.resolve)
    out = df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()
    return out[["unit", "node", "count", "tier"]]


ENTRY = dict(
    name="Djibouti",
    source=("Third General Census of Population and Housing (RGPH-3) 2024, Tome 4: "
            "Caracteristiques socioculturelles de la population (INSTAD), Tableau 7, languages "
            "spoken by people aged 5 and over in ordinary households, by region; native Arabic "
            "speakers from Joshua Project's Arab people groups in Djibouti"),
    how=("census, 2024, languages spoken, several allowed; each person shared across the local "
         "languages they named, leaving out French, English and Arabic; Arabic as a first "
         "language from an estimate of the Arab community"),
    parts=[
        dict(covers="Arab community",
             source="Joshua Project's Arab people groups, placed in Djibouti-Ville",
             nodes=["afroasiatic.yemeni_arabic", "afroasiatic.arabic"]),
        dict(covers="Everyone else",
             source="2024 census, languages spoken, aged 5 and over, shared across the local "
                    "languages named", rest=True),
    ],
    grain="6 regions, 167,000 people on average",
    gap=("the 63,009 people outside ordinary and nomadic households (homeless people and those in "
         "collective housing); children under 5, drawn at their region's shares"),
    view=[41.7, 10.9, 43.5, 12.8],
    counts=_counts,
    mappings=["dj2024"],
    place=RD_GEO / "dj" / "dj_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "Djibouti's 2024 census asked everyone aged 5 and over which languages they speak, one "
        "question per language, and many named several: 78% Somali, 26% Afar, 42% French, 27% "
        "Arabic and 17% English. French, English and Arabic are mostly learned at school, at "
        "work or for religion, so they are left out here and each person is drawn shared across "
        "the other languages they named (Somali, Afar, Amharic, Oromo, sign language and "
        "others). Amharic and Oromo are mostly "
        "spoken by Ethiopian residents, and some of their speakers learned them as a second "
        "language. The census cannot tell who speaks Arabic as a first language, so the Arab "
        "community of Yemeni and Omani origin (about 6% of the people, a Joshua Project "
        "estimate) is drawn in Djibouti-Ville, where most of it lives."),
)
