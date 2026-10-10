# Mali. RGPH5 2022, mother tongue by région (sources/ml_rgph.py: annex A06's shares x its
# populations, the foreign languages split by annex A03), on religiondots' Kontur hexes for the
# same 20 régions. Inside a région, each language's dots go to its old (2009) cercles by CLEAR
# Global's shares (sources/ml_place.py, sources/clear_place.py), a placement weight only.
from _shared import *  # noqa: F401,F403


def _load(name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, ROOT / "sources" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _weight(place):
    if "zone" not in place.columns:
        raise SystemExit("data/geo/ml/ml_hexes.gpkg has no `zone` column: run sources/ml_place.py")
    shares = pd.read_csv(NORM / "ml_clear.csv")
    return _load("clear_place").ClearWeighter(place, shares, _load("ml_place").node_codes(),
                                              label="CLEAR Global's cercle")


def _counts():
    import ml2022
    df = pd.read_csv(NORM / "ml.csv")
    df = df[(df["geo_level"] == "region") & (df["count"] > 0)].copy()
    df["node"] = df["source_category"].map(ml2022.resolve)
    unmapped = sorted(set(df.loc[df["node"].isna(), "source_category"]))
    if unmapped:
        raise SystemExit(f"ml: categories with no node: {unmapped}")
    df = df.rename(columns={"geo_id": "unit"})
    if df["unit"].nunique() != 20:
        raise SystemExit(f"ml: {df['unit'].nunique()} régions, expected 20")
    return df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()


ENTRY = dict(
    name="Mali",
    source=("Fifth General Census of Population and Housing (RGPH5) 2022, Caractéristiques "
            "culturelles de la population, annexes A03 and A06 (INSTAT); placement inside "
            "régions from CLEAR Global's cercle shares (IPUMS sample of the 2009 census)"),
    how="census, 2022, mother tongue, aged 3 and over",
    parts=[
        dict(covers="Foreign languages",
             source="2022 census, one figure per region, split in the national proportions",
             people=20_856),
        dict(covers="Everyone else", source="2022 census, mother tongue, aged 3 and over",
             rest=True),
    ],
    grain=("20 regions, 957,000 people on average; in twelve of them, placed by cercle "
           "inside"),
    gap=("about 2.2 million children under three (not asked), 941,335 people in areas not "
         "enumerated because of insecurity, and 106,567 in collective households or homeless"),
    view=[-12.3, 10.1, 4.3, 25.0],
    counts=_counts,
    mappings=["ml2022"],
    place=GEO / "ml" / "ml_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=_weight,
    note_public=(
        "The census asked each person's mother tongue, the first language they learnt as a "
        "child. Bambara is the mother tongue of half the people counted, and more speak it as "
        "a second language: 57.8% named it as the language they speak most, a separate "
        "question not drawn here. "
        "About 941,000 people lived in areas the census could not reach because of insecurity, "
        "most of them in Tombouctou, Ségou, Ménaka and the old Mopti region; their languages are "
        "not known and they are not drawn. The 0.4% who gave no answer are spread over the "
        "languages of their own region, as the census's percentages do. In Ménaka, 10% have a "
        "language of Mali that the census does not name; it is drawn as other African "
        "languages. Inside twelve regions, each language's dots are spread across the old "
        "cercles using CLEAR Global's figures from a sample of the 2009 census."),
)
