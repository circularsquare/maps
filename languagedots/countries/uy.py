# Uruguay. Censos 2011, country of birth per department and Montevideo barrio
# (sources/uy_census.py, INE's REDATAM base on CEPAL's server): the Uruguayan-born on Spanish,
# the foreign-born on their origin's languages (sources/origin_mix.py). No language question.
# Placed on religiondots' Kontur hexes for the same 18 departments + 62 barrios (read-only).
from _shared import *  # noqa: F401,F403


def _counts():
    import uy2011
    df = pd.read_csv(NORM / "uy.csv", keep_default_na=False)    # "NA" is Namibia
    lut = pd.read_csv(RD_GEO / "uy" / "uy_lookup.csv", dtype=str)
    dept = {int(d): u for d, u in zip(lut["ine_dpto"], lut["unit"]) if u != "UY10"}
    df["unit"] = [dept.get(u) if u < 100 else f"UY10-B{u - 100:02d}" for u in df["unit"]]
    if df["unit"].isna().any() or df["unit"].nunique() != 80:
        raise SystemExit(f"uy: {df['unit'].nunique()} units, expected 80")
    rows = []
    for unit, g in df.groupby("unit"):
        known = g[g["origin"] != "rest"]
        scale = g["count"].sum() / known["count"].sum()   # unknown birthplace spread in proportion
        for r in known.itertuples():
            for node, x in uy2011.mix(r.origin).items():
                rows.append((unit, node, r.count * scale * x))
    out = pd.DataFrame(rows, columns=["unit", "node", "count"])
    out = out.groupby(["unit", "node"], as_index=False)["count"].sum()
    out["tier"] = "derived"
    return out


ENTRY = dict(
    name="Uruguay",
    source="Censos 2011 (Instituto Nacional de Estadística), tabulated on INE's REDATAM base "
           "at CEPAL",
    how="census, 2011, country of birth (no language question); people born in Uruguay drawn "
        "as Spanish, people born abroad by their birth country's languages",
    parts=[
        dict(covers="People born abroad",
             source="2011 census, country of birth, drawn on that country's languages",
             people=79_585),
        dict(covers="People born in Uruguay", source="2011 census, drawn as Spanish",
             rest=True),
    ],
    grain="18 departments and Montevideo's 62 barrios, 41,000 people on average",
    gap="birthplace was not collected for 117,000 people; they are spread over their area's "
        "known birthplaces",
    view=[-58.5, -35.0, -53.1, -30.0],
    counts=_counts,
    mappings=["uy2011"],
    place=RD_GEO / "uy" / "uy_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "Uruguay's census does not ask about language. Everyone born in Uruguay is drawn as a "
        "Spanish speaker and everyone born abroad by the main languages of their birth country, "
        "so Brazilians are drawn as Portuguese speakers and Italians as Italian speakers. "
        "Along the Brazilian border, in Artigas, Rivera and Cerro Largo, many Uruguayans grow "
        "up speaking Uruguayan Portuguese (Portuñol) at home, but no survey counts who speaks "
        "it first, so they are drawn as Spanish speakers here."),
)
