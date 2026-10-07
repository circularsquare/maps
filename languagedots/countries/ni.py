# Nicaragua. Censo 2005, people who speak the language of their own indigenous people or ethnic
# community, per municipio (sources/ni_censo.py); everyone else drawn as Spanish (spec §3.5).
# Placed on religiondots' Kontur hexes for the same 153 municipios (read-only).
from _shared import *  # noqa: F401,F403

SPANISH = "indoeuropean.romance.spanish"


def _counts():
    import ni2005
    df = pd.read_csv(NORM / "ni.csv", dtype={"geo_id": str})
    df = df[df["geo_level"] == "municipio"].copy()
    df["node"] = df["code"].map(ni2005.resolve)
    df = df[df["node"].notna()].copy()                # P08 no answer: in `gap`
    # religiondots joins INIDE's four-digit codes to COD-AB p-codes on NAME, not code
    # (religiondots/sources/ni_geo.py: ten municipios were renumbered, INIDE 9105 Waspam is
    # COD's NI9105 Mulukuku); its lookup carries that join, so we key on INIDE's code here
    lut = pd.read_csv(RD_GEO / "ni" / "ni_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    if df["unit"].isna().any():
        raise SystemExit(f"ni: {df.loc[df['unit'].isna(), 'geo_id'].nunique()} municipios "
                         "missing from religiondots' ni_lookup.csv")
    if df["unit"].nunique() != 153:
        raise SystemExit(f"ni: {df['unit'].nunique()} municipios, expected 153")
    df["tier"] = df["node"].map(lambda n: "derived" if n == SPANISH else "measured")
    # 2026-10-05, session edd42a8c-latn (sources/ni.md, "Immigrant languages"): people born in
    # non-Spanish-speaking countries (sources/ni_imm.py) through sources/latam_immig.py, taken
    # out of Spanish; US-born under 18 stay Spanish
    import sys
    sys.path.insert(0, str(ROOT / "sources"))
    import latam_immig
    org = pd.read_csv(NORM / "ni_imm.csv", dtype={"geo_id": str}, keep_default_na=False)
    org["unit"] = org["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    if org["unit"].isna().any():
        raise SystemExit("ni: ni_imm.csv municipios missing from ni_lookup.csv")
    imm = pd.DataFrame(latam_immig.unit_rows(org, "ni"), columns=["unit", "node", "tier", "count"])
    out = pd.concat([df[["unit", "node", "tier", "count"]], imm], ignore_index=True)
    out = out.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()
    if (out["count"] < -1e-6).any():
        raise SystemExit("ni: immigrant languages exceed a municipio's Spanish")
    return out[out["count"] > 0]


ENTRY = dict(
    name="Nicaragua",
    source="VIII Censo de Población y IV de Vivienda 2005 (INIDE), tabulated on INIDE's "
           "REDATAM server; France's TeO2 survey for how many immigrants keep their language",
    how="census, 2005, speaks the language of their own people (Caribbean coast peoples only); "
        "people born in non-Spanish-speaking countries drawn by their birth country's "
        "languages; everyone else drawn as Spanish",
    parts=[
        dict(covers="Caribbean coast peoples",
             source="2005 census, speaks the language of their own people",
             people=142_151),
        dict(covers="People born abroad, languages other than Spanish",
             source="2005 census, country of birth, drawn on that country's languages; about a "
                    "quarter moved to Spanish by France's TeO2 survey",
             people=2_472),
        dict(covers="Everyone else", source="2005 census, drawn as Spanish", rest=True),
    ],
    grain="153 municipios, 34,000 people on average",
    gap="11,297 people of the Caribbean coast peoples who did not say whether they speak "
        "their people's language",
    view=[-87.7, 10.7, -82.7, 15.1],
    counts=_counts,
    mappings=["ni2005"],
    place=RD_GEO / "ni" / "ni_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "Nicaragua's last census, in 2005, asked about language only of people who identified "
        "with one of the seven peoples of the Caribbean coast (Miskitu, Mayangna, Ulwa, Rama, "
        "Garifuna, Creole and the coast's mestizos), and only whether they speak their "
        "people's language. Each speaker is drawn on that language and everyone else as "
        "Spanish, including the coast's mestizos and the Pacific and central peoples, whose "
        "languages are no longer spoken. Miskito spoken by people of another people is not "
        "recorded, and Rama and Garifuna who speak Creole English instead of their people's "
        "language are drawn as Spanish. People born where Spanish is not the main language "
        "(5,462) are drawn on their birth country's languages, about a quarter of them moved "
        "to Spanish; US-born children are drawn as Spanish. Inside each municipio the dots "
        "follow where people lived in 2023, not where each community is."),
)
