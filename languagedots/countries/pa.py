# Panama. Censo 2023 indigenous and Afro-descendant groups per corregimiento
# (sources/pa_censo.py, INEC's REDATAM server), each group laid out over languages by MICS 2013's
# mother tongue for that group (sources/pa_mics.py, taxonomy/pa2023.py). Placed on Kontur hexes
# keyed to 74 units built from the census's 82 districts (sources/pa_geo.py).
from _shared import *  # noqa: F401,F403


def _counts():
    import pa2023
    df = pd.read_csv(NORM / "pa.csv", dtype={"geo_id": str})
    lut = pd.read_csv(GEO / "pa" / "pa_units.csv", dtype=str)
    unit = dict(zip(lut["dist"], lut["unit"]))
    rows = []
    for r in df.itertuples():
        u = unit.get(r.geo_id[:4])
        if u is None:
            raise SystemExit(f"pa: district {r.geo_id[:4]} has no unit -- re-run sources/pa_geo.py")
        neither = r.indigenous in ("Ninguno", "No declarado") and r.afro not in pa2023.AFRO
        for node, x in pa2023.shares(r.indigenous, r.afro, r.geo_id[:2]).items():
            # 2026-10-05, session edd42a8c-latn: MICS's unnamed "other" (mostly immigrant
            # languages) and the English of people of no group go to Spanish; country of
            # birth draws the immigrant languages instead (sources/pa.md, "Immigrant languages")
            if node == OTHER or (neither and node == ENGLISH):
                node = SPANISH
            rows.append((u, node, "modelled", r.count * x))
    rows += _immigrant_rows(unit)
    out = pd.DataFrame(rows, columns=["unit", "node", "tier", "count"])
    out = out.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()
    if abs(out["count"].sum() - 4_064_780) > 1:
        raise SystemExit(f"pa: {out['count'].sum():,.0f} people, expected 4,064,780")
    if (out["count"] < -1e-6).any():
        raise SystemExit("pa: immigrant languages exceed a unit's Spanish")
    return out[out["count"] > 0]


def _immigrant_rows(unit):
    """Foreign-born of non-Spanish-speaking countries (sources/pa_imm.py), through
    sources/latam_immig.py, taken out of Spanish."""
    import sys
    sys.path.insert(0, str(ROOT / "sources"))
    import latam_immig
    org = pd.read_csv(NORM / "pa_imm.csv", dtype={"geo_id": str}, keep_default_na=False)
    org["unit"] = org["geo_id"].str[:4].map(unit)
    if org["unit"].isna().any():
        raise SystemExit(f"pa: corregimientos with no unit {org.loc[org['unit'].isna(), 'geo_id'].unique()[:5]}")
    rows = []
    for u, g in org.groupby("unit"):
        by = g.groupby("origin")["count"].sum().to_dict()     # several corregimientos per unit
        for node, n in latam_immig.spread(by, "pa").items():
            if node != SPANISH and n:
                rows.append((u, node, "derived", n))
                rows.append((u, SPANISH, "modelled", -n))
    return rows


SPANISH = "indoeuropean.romance.spanish"
ENGLISH = "indoeuropean.germanic.english"
OTHER = "other"


ENTRY = dict(
    name="Panama",
    source="XII Censo de Población y VIII de Vivienda 2023 (INEC), tabulated on INEC's REDATAM "
           "server; mother tongue by group from MICS 2013 (UNICEF and the Ministerio de Salud); "
           "France's TeO2 survey for how many immigrants keep their language",
    how="census, 2023, indigenous and Afro-descendant group (no language question), laid over "
        "languages by the 2013 MICS survey's mother tongue; people born abroad by birth country",
    parts=[
        dict(covers="Indigenous languages and Afro-Antillean English",
             source="2023 census, indigenous or Afro-descendant group, laid over languages by "
                    "MICS 2013 mother tongue for that group",
             people=534_371),
        dict(covers="People born abroad, languages other than Spanish",
             source="2023 census, country of birth, drawn on that country's languages; about a "
                    "third moved to Spanish by France's TeO2 survey",
             people=28_822),
        dict(covers="Everyone else",
             source="2023 census, drawn as Spanish, with group members whose mother tongue MICS "
                    "finds is Spanish",
             rest=True),
    ],
    grain="74 districts or groups of districts, 55,000 people on average",
    view=[-83.1, 7.1, -77.1, 9.7],
    counts=_counts,
    mappings=["pa2023"],
    place=GEO / "pa" / "pa_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "Panama's 2023 census did not ask about language. It asked whether each person "
        "belongs to an indigenous people and to an Afro-descendant group, and each group is "
        "laid over languages by the mother tongues its members gave in the 2013 MICS household "
        "survey. In that survey 93% of Ngäbe in the Ngäbe-Buglé comarca named Ngäbere and 69% "
        "of those elsewhere did; only 13% of Buglé named Buglere, and Naso's 4% rests on 41 "
        "people. People born abroad are drawn by their birth country's languages, the "
        "China-born mostly as Hakka. Everyone else is drawn as Spanish. The newest corregimiento "
        "boundaries predate about a hundred of the census's corregimientos, so the map uses "
        "districts."),
)
