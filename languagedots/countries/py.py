# Paraguay. Censo 2002 household language (HOGAR.idiohog, every member counted) per census
# district from INE's REDATAM server (sources/py_censo.py), on religiondots' Kontur hexes for
# the 2002 districts, read-only. Asuncion's six census districts share one polygon (0000), and
# the 26 districts created after 2002 are dissolved into their 2002 parent there
# (religiondots/sources/py_geo.py), so 229 census districts land on 224 units.
from _shared import *  # noqa: F401,F403


GERMAN = "indoeuropean.germanic.continental.german"
PLAUTDIETSCH = "indoeuropean.germanic.continental.lowgerman.plautdietsch"
COLONIES = {
    "1602": "Mcal. Estigarribia, all of Boquerón in 2002: Menno, Fernheim, Neuland",
    "1504": "Villa Hayes: the Menno colony's land east of Loma Plata (uncited)",
    "0512": "Dr. J. Eulogio Estigarribia (Campo 9): Sommerfeld, Bergthal",
    "0218": "Santa Rosa del Aguaray: Río Verde",
    "0207": "Nueva Germania: Nuevo México, Santa Clara (and the 1887 German colony)",
    "0210": "Tacuatí: Manitoba",
    "1403": "Curuguaty (Maracaná since): Nueva Durango",
    "0205": "Itacurubí del Rosario: Friesland",
    "0213": "Villa del Rosario: Volendam",
}


def _counts():
    import py2002
    df = pd.read_csv(NORM / "py.csv", dtype={"geo_id": str})
    df["node"] = df["source_category"].map(py2002.resolve)
    df = df[df["node"].notna()].copy()        # collective dwellings and no answer: in `gap`
    # "Alemán" in the Mennonite colonies' districts is Plautdietsch (session 5d7dac7e-br,
    # 2026-10-06; sources/py.md, "Plautdietsch"). Elsewhere (Itapúa's and Alto Paraná's German
    # and German-Brazilian settlers, Asunción) it stays German.
    df.loc[(df["node"] == GERMAN) & df["geo_id"].isin(COLONIES), "node"] = PLAUTDIETSCH
    lut = pd.read_csv(RD_GEO / "py" / "py_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    if df["unit"].isna().any():
        raise SystemExit(f"py: {df.loc[df['unit'].isna(), 'geo_id'].nunique()} census districts "
                         "missing from religiondots' py_lookup.csv")
    if df["unit"].nunique() != 224:
        raise SystemExit(f"py: {df['unit'].nunique()} units, expected 224")
    return by_unit(df)


ENTRY = dict(
    name="Paraguay",
    source="Censo Nacional de Población y Viviendas 2002 (DGEEC, now INE Paraguay), the "
           "language spoken in the household most of the time, counted in people, tabulated "
           "per district from INE's own REDATAM server",
    how="census, 2002, home language (one answer per household); German in the Mennonite "
        "colonies' districts drawn as Plautdietsch",
    parts=[
        dict(covers="Everyone",
             source="2002 census, language the household speaks most of the time",
             rest=True),
        dict(covers="Mennonite colonies",
             source="2002 census German, drawn as Plautdietsch in the colonies' districts",
             nodes=[PLAUTDIETSCH]),
    ],
    grain="224 districts, 22,900 people on average",
    gap="40,216 people (0.8%) in collective dwellings and 510 whose household named no language",
    view=[-62.7, -27.7, -54.2, -19.2],
    counts=_counts,
    mappings=["py2002"],
    place=RD_GEO / "py" / "py_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "The 2002 census asked each household which language it spoke most of the time, and "
        "everyone in the household is drawn in that language, so a Guaraní-speaking "
        "grandmother in a household that answered Spanish is drawn as Spanish. Most "
        "Paraguayans speak both Guaraní and Spanish (73% said so in 2022), so this map shows "
        "which one a home leans on. The 2022 census asked which languages each person speaks "
        "and did not name the indigenous languages, so 2002 is used. Four indigenous languages "
        "are drawn under a neighbour's name for the same speech: Pai Tavyterã as Kaiowá, "
        "Guaraní Ñandeva as Tapieté, Guaraní Occidental as Eastern Bolivian Guaraní and Manjui "
        "as Chorote. The census records the Mennonite colonies' Plautdietsch as German; in "
        "the colonies' districts it is drawn as Plautdietsch. Districts created after 2002 are "
        "drawn with the district each was carved from."),
)
