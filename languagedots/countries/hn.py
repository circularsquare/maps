# Honduras. Censo 2013, self-identified people per municipio (sources/hn_censo.py, INE's REDATAM
# server), each people read as its language (AGENT_BRIEF §2, ethnicity rule). Placed on Kontur
# hexes keyed to the 298 municipios (sources/hn_geo.py).
from _shared import *  # noqa: F401,F403


def _counts():
    import hn2013
    df = pd.read_csv(NORM / "hn.csv", dtype={"geo_id": str})
    df = df[df["geo_level"] == "municipio"].copy()
    df["node"] = [hn2013.resolve(c, m) for c, m in zip(df["code"], df["geo_id"])]
    if df["geo_id"].nunique() != 298:
        raise SystemExit(f"hn: {df['geo_id'].nunique()} municipios, expected 298")
    df["unit"] = df["geo_id"]
    df["tier"] = "derived"
    # 2026-10-05, session edd42a8c-latn (sources/hn.md, "Immigrant languages"): people born in
    # non-Spanish-speaking countries (sources/hn_imm.py) through sources/latam_immig.py, taken out
    # of Spanish; US-born under 18 stay Spanish
    import sys
    sys.path.insert(0, str(ROOT / "sources"))
    import latam_immig
    org = pd.read_csv(NORM / "hn_imm.csv", dtype={"geo_id": str}, keep_default_na=False)
    imm = pd.DataFrame(latam_immig.unit_rows(org.rename(columns={"geo_id": "unit"}), "hn"),
                       columns=["unit", "node", "tier", "count"])
    out = pd.concat([df[["unit", "node", "tier", "count"]], imm], ignore_index=True)
    out = out.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()
    if (out["count"] < -1e-6).any():
        raise SystemExit("hn: immigrant languages exceed a municipio's Spanish")
    return out[out["count"] > 0]


ENTRY = dict(
    name="Honduras",
    source="XVII Censo de Población y VI de Vivienda 2013 (INE), tabulated on INE's REDATAM "
           "server; France's TeO2 survey for how many immigrants keep their language",
    how="census, 2013, indigenous or Afro-Honduran people (no language question), each drawn "
        "as its language where it is still spoken; immigrants on their birth country's "
        "languages; everyone else as Spanish",
    parts=[
        dict(covers="Indigenous and Afro-Honduran peoples whose language is still spoken",
             source="2013 census, people (Miskito, Garifuna, Bay Islands English, Pech, "
                    "Tawahka, Tol), drawn as their language",
             nodes=["misumalpan.miskito", "arawakan.garifuna",
                    "creole.english_based.bay_islands", "chibchan.pech", "misumalpan.tawahka",
                    "jicaquean.tol"]),
        dict(covers="People born in non-Spanish-speaking countries",
             source="2013 census, country of birth, drawn on that country's languages; about a "
                    "quarter moved to Spanish by France's TeO2 survey",
             people=3_087),
        dict(covers="Everyone else", source="drawn as Spanish", rest=True),
    ],
    grain="298 municipios, 26,000 people on average",
    view=[-89.5, 12.9, -83.1, 16.6],
    counts=_counts,
    mappings=["hn2013"],
    place=GEO / "hn" / "hn_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "The 2013 census did not ask about language. It asked which people each person "
        "belongs to, and this map draws each people as its language. Miskito, Tawahka, Pech, "
        "Garifuna and Bay Islands English are drawn in full, as upper bounds, since no source "
        "says what share still speaks them. Tol is drawn only in Orica and Marale, where it is "
        "still spoken. The Lenca (442,000 people), Maya Chortí and Nahua are drawn as Spanish "
        "speakers, since their languages are gone or nearly so. Bay Islanders who said white "
        "or mestizo are drawn as Spanish speakers, though many speak English. The census base "
        "is the 7.66 million people enumerated, not INE's published 8.30 million."),
)
