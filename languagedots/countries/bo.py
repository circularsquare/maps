# Bolivia. Censo 2024 mother tongue (question 34.1, IDIOMAT; INE's universe, aged 4+ and usually
# resident) per municipality from INE's REDATAM server (sources/bo_censo.py), on Kontur hexes
# keyed to COD-AB's 339 municipalities (sources/bo_geo.py). Four census areas without a polygon
# (three TIOC and San Pedro de Macha) are folded into the municipality each was carved from,
# through data/geo/bo/bo_lookup.csv.
from _shared import *  # noqa: F401,F403


GERMAN = "indoeuropean.germanic.continental.german"
PLAUTDIETSCH = "indoeuropean.germanic.continental.lowgerman.plautdietsch"
CITIES = {"010101", "020101", "020105", "030101", "040101", "050101", "060101", "070101",
          "080101", "090101"}       # Sucre, La Paz, El Alto, Cochabamba ... Trinidad, Cobija


def _counts():
    import bo2024
    df = pd.read_csv(NORM / "bo.csv", dtype={"geo_id": str})
    df["node"] = df["source_category"].map(bo2024.resolve)
    df = df[df["node"].notna()].copy()        # "Sin especificar": in `gap`
    # "Alemán" outside the ten largest cities is the Mennonite colonies' Plautdietsch (session
    # 5d7dac7e-br, 2026-10-06): 72,976 of the 75,852 live in rural municipalities of Santa
    # Cruz, the Chaco and Beni where the colonies are (Pailón 19,556, San José 12,966, Cabezas,
    # Charagua, Cuatro Cañadas...); Wikipedia's Mennonites in Bolivia reads the census figure
    # as the colonies' own. In the department capitals and El Alto it stays German.
    df.loc[(df["node"] == GERMAN) & ~df["geo_id"].isin(CITIES), "node"] = PLAUTDIETSCH
    lut = pd.read_csv(GEO / "bo" / "bo_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    if df["unit"].isna().any():
        raise SystemExit(f"bo: {df.loc[df['unit'].isna(), 'geo_id'].nunique()} census areas "
                         "missing from data/geo/bo/bo_lookup.csv -- re-run sources/bo_geo.py")
    if df["unit"].nunique() != 339:
        raise SystemExit(f"bo: {df['unit'].nunique()} units, expected 339")
    return by_unit(df)


ENTRY = dict(
    name="Bolivia",
    source="Censo de Población y Vivienda 2024 (INE Bolivia), question 34.1, the first language "
           "learned as a child, tabulated per municipality from INE's own REDATAM server",
    how="census, 2024, mother tongue, aged 4 and over",
    parts=[dict(covers="Everyone aged 4 and over", source="2024 census, mother tongue",
                rest=True)],
    grain="339 municipalities, 31,200 people aged 4 and over on average",
    gap="children under 4, 661,467 (5.8%), whom INE leaves out of its language tables; 98,649 "
        "(0.9%) who usually live abroad or did not say where they live; and 96,032 (0.8%) who "
        "do not speak or named no language",
    view=[-69.7, -22.95, -57.4, -9.65],
    counts=_counts,
    mappings=["bo2024"],
    place=GEO / "bo" / "bo_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "The census asked everyone which language they first learned to speak as a child, and "
        "INE publishes the answers for people aged 4 and over who live in Bolivia. Every "
        "language it names is drawn, 36 native languages among them, down to a single speaker "
        "of Guarasu'we. Guaraní here is Eastern Bolivian Guaraní, the language of the Ava, "
        "Isoseño and Simba Guaraní of the Chaco. Most of the people counted as German speakers "
        "live in the Mennonite colonies of Santa Cruz, whose home language is Plautdietsch, a "
        "Low German; they are drawn as Plautdietsch everywhere outside the department "
        "capitals. Inside a municipality every language's dots are spread by population "
        "alone, so in Charagua the Mennonite and Guaraní dots are mixed across the same "
        "towns."),
)
