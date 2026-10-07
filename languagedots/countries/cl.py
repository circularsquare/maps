# Chile. Censo 2024, the indigenous language each person aged 5+ speaks or understands best, per
# comuna (sources/cl_censo.py); everyone else aged 5+ drawn as Spanish (spec §3.5). Placed on
# religiondots' Kontur hexes for the same 345 comunas, keyed by INE's CUT code (read-only).
from _shared import *  # noqa: F401,F403

SPANISH = "indoeuropean.romance.spanish"
ANTARCTICA = "12202"     # 60 people at the bases; no polygon in religiondots' layer


def _counts():
    import cl2024
    df = pd.read_csv(NORM / "cl.csv", dtype={"geo_id": str})
    df = df[df["geo_id"] != ANTARCTICA].copy()
    df["node"] = [cl2024.resolve(c, g) for c, g in zip(df["source_category"], df["geo_id"])]
    df = df[df["node"].notna() & (df["count"] > 0)].copy()     # not declared: in `gap`
    df["tier"] = df["node"].map(lambda n: "derived" if n == SPANISH else "measured")
    df = df.rename(columns={"geo_id": "unit"})
    lut = pd.read_csv(RD_GEO / "cl" / "cl_lookup.csv", dtype=str)
    drawn = set(lut.loc[lut["drawn"] == "1", "geo_id"])
    missing = sorted(set(df["unit"]) - drawn)
    if missing:
        raise SystemExit(f"cl: comunas without a polygon in religiondots' layer: {missing}")
    df = df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()
    return _immigrants(df)


# 2026-10-05 (session edd42a8c-lats, sources/cl.md "Immigrant languages"): the foreign-born aged
# 5+ by birthplace (sources/cl_immig.py) on their origin's languages (origin_mix), retained at
# France's TeO2 rate (sources/latam_immig.py), taken out of the Spanish remainder. The census
# asked everyone, immigrants too, about Quechua and Aymara, so those are already measured: only
# an estimate above the measured count is added.
DEDUPE = {"quechuan.quechua", "aymaran.aymara", "araucanian.mapuche"}


def _immigrants(df):
    sys.path.insert(0, str(ROOT / "sources"))
    import latam_immig
    if latam_immig.active():          # an origin's home mix for another country's build
        return df
    imm = pd.read_csv(NORM / "cl_immig.csv", dtype={"unit": str}, keep_default_na=False)
    imm = imm[imm["unit"] != ANTARCTICA]
    rows = latam_immig.immigrant_languages(imm, SPANISH, "cl")
    out, _ = latam_immig.fold_into(df, rows, SPANISH, dedupe=DEDUPE)
    return out


ENTRY = dict(
    name="Chile",
    source="Censo de Población y Vivienda 2024 (INE), table P3 Lenguas indígenas and country "
           "of birth, by comuna; France's TeO2 survey for how many immigrants keep their "
           "language",
    how="census, 2024, indigenous language spoken or understood best, aged 5 and over; people "
        "born abroad drawn by their birth country's languages, less the share France's TeO2 "
        "survey finds speaking only the host language at home; everyone else drawn as Spanish",
    parts=[
        dict(covers="Indigenous languages",
             source="2024 census, indigenous language spoken or understood best, aged 5 and "
                    "over", people=510_462),
        dict(covers="People born abroad, languages other than Spanish",
             source="2024 census, country of birth, drawn on that country's languages; about "
                    "a quarter moved to Spanish by France's TeO2 survey", people=124_760),
        dict(covers="Everyone else", source="2024 census, aged 5 and over, drawn as Spanish",
             rest=True),
    ],
    grain="345 comunas, 51,000 people aged 5 and over on average",
    gap="children under 5, 870,693 (4.7%), whom the census does not ask; 103,050 people who did "
        "not say whether they speak an indigenous language; and the 60 people of Antártica comuna",
    # continental Chile; Rapa Nui (Isla de Pascua) is drawn but not framed, as in religiondots
    view=[-76.5, -56.0, -66.0, -17.3],
    counts=_counts,
    mappings=["cl2024"],
    place=RD_GEO / "cl" / "cl_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "The 2024 census asked everyone aged 5 or over whether they speak or understand one of "
        "Chile's indigenous languages, and someone who knows more than one gave the one they "
        "speak best. It did not ask about Spanish, so everyone who said no is drawn as a Spanish "
        "speaker, except people born abroad (1.6 million, most of them from Spanish-speaking "
        "Venezuela, Peru and Colombia), who are drawn by the languages of their birth country: "
        "Haitians as Haitian Creole speakers, Brazilians as Portuguese speakers. Since no "
        "Chilean survey asks which language immigrants speak at home, about a quarter of those "
        "from a country with another language are drawn as Spanish speakers, the share of "
        "immigrants in France who speak only French with their children. Speaking or "
        "understanding is a lower bar than a first language. Kunza (Ckunza) and Yagán have no "
        "native speakers left, so their speakers are people who have learned or kept some of "
        "the language. In Alto Biobío, Pewenche country, the answer \"another indigenous "
        "language of Chile\" is drawn as an unnamed Araucanian language. Languages are "
        "published per comuna, and inside each one the dots follow where people live, not "
        "where indigenous communities are."),
)
