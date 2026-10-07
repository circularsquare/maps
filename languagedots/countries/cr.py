# Costa Rica. Censo 2011, indigenous people who speak an indigenous language, per distrito
# (sources/cr_censo.py, INEC's REDATAM server); everyone else drawn as Spanish (spec §3.5).
# Placed on Kontur hexes per 2011 distrito (sources/cr_geo.py, rebuilt from COD-AB 2024).
from _shared import *  # noqa: F401,F403

SPANISH = "indoeuropean.romance.spanish"
LIMONESE = "creole.english_based.limonese"


def _counts():
    import cr2011
    df = pd.read_csv(NORM / "cr.csv", dtype={"geo_id": str})
    df = df[df["geo_level"] == "distrito"].copy()
    df["node"] = df["code"].map(cr2011.resolve)
    if df["node"].isna().any():
        raise SystemExit(f"cr: unmapped codes {sorted(df.loc[df['node'].isna(), 'code'].unique())}")
    df["unit"] = df["geo_id"]                 # INEC's DTA code; sources/cr_geo.py keys the hexes on it
    if df["unit"].nunique() != 472:
        raise SystemExit(f"cr: {df['unit'].nunique()} distritos, expected 472")
    df["tier"] = df["node"].map(lambda n: "derived" if n == SPANISH else "measured")
    df = pd.concat([df[["unit", "node", "tier", "count"]], _immigrant_rows()], ignore_index=True)
    out = df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()
    if (out["count"] < -1e-6).any():
        raise SystemExit("cr: immigrant and Creole languages exceed a distrito's Spanish")
    return out[out["count"] > 0]


def _immigrant_rows():
    """2026-10-05, session edd42a8c-latn (sources/cr.md, "Immigrant and Creole languages"):
    taken out of Spanish, the foreign-born of non-Spanish-speaking countries
    (sources/cr_imm.py, through sources/latam_immig.py) and Limon province's Black
    self-identifiers on Limonese Creole."""
    import sys
    sys.path.insert(0, str(ROOT / "sources"))
    import latam_immig
    org = pd.read_csv(NORM / "cr_imm.csv", dtype={"geo_id": str}, keep_default_na=False)
    rows = []
    for geo, g in org.groupby("geo_id"):
        d = dict(zip(g["origin"], g["count"]))
        afro = d.pop("LIMON_AFRO", 0)
        spread = latam_immig.spread(d, "cr")
        spread[LIMONESE] = afro
        for node, n in spread.items():
            if node != SPANISH and n:
                rows.append((geo, node, "derived", n))
                rows.append((geo, SPANISH, "derived", -n))
    return pd.DataFrame(rows, columns=["unit", "node", "tier", "count"])


ENTRY = dict(
    name="Costa Rica",
    source="X Censo Nacional de Población y VI de Vivienda 2011 (INEC), tabulated on INEC's "
           "REDATAM server; France's TeO2 survey for how many immigrants keep their language",
    how="census, 2011, indigenous people who speak an indigenous language, drawn on their "
        "people's language; the foreign-born on their birth country's languages; Black "
        "Limonenses as Limonese Creole; everyone else as Spanish",
    parts=[
        dict(covers="Indigenous languages",
             source="2011 census, indigenous people who speak their language, drawn on the "
                    "people they named", people=31_686),
        dict(covers="People born abroad, languages other than Spanish",
             source="2011 census, country of birth, drawn on that country's languages; about "
                    "a quarter moved to Spanish by France's TeO2 survey", people=21_325),
        dict(covers="Black and Afro-descendant people in Limón province",
             source="2011 census, ethnic identity, drawn as Limonese Creole",
             nodes=["creole.english_based.limonese"]),
        dict(covers="Everyone else", source="drawn as Spanish", rest=True),
    ],
    grain="472 distritos, 9,100 people on average",
    view=[-86.0, 8.0, -82.5, 11.3],
    counts=_counts,
    mappings=["cr2011"],
    place=GEO / "cr" / "cr_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "Costa Rica's 2011 census asked the people who consider themselves indigenous whether "
        "they speak an indigenous language, without asking which one. Each speaker is drawn on "
        "the language of the people they named. Brunca (Boruca) and Térraba speakers are drawn "
        "as Chibchan with the language not named, because their own languages have very few "
        "speakers left. Nobody else was asked about language. People born in a country where "
        "Spanish is not the main language are drawn by that country's main languages, with "
        "about a quarter moved to Spanish, the share of immigrants France's TeO2 survey finds "
        "speaking only the host language at home. Limonese Creole (Mekatelyu) is drawn for the "
        "people in Limón province who identify as Black or Afro-descendant; the census does "
        "not ask who speaks it, and some of them now speak only Spanish. Everyone else, "
        "Nicaraguans included, is drawn as a Spanish speaker. Inside each distrito the dots "
        "follow where people lived in 2023, not where each community is."),
)
