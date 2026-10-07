# Trinidad and Tobago. Census 2011 ethnic group x place of birth and country of birth per
# municipality (sources/tt_census.py, CSO's REDATAM base at CEPAL), read as languages by
# taxonomy/tt2011.py; scaled to the census's printed non-institutional municipal totals
# (religiondots' normalized tt.csv, read-only). Placed on religiondots' Kontur hexes.
from _shared import *  # noqa: F401,F403
import re as _re


def _fold(s):
    return _re.sub(r"[^a-z]+", "", str(s).lower())


def _counts():
    import tt2011
    df = pd.read_csv(NORM / "tt.csv", dtype={"geo_id": str}, keep_default_na=False)
    rd = pd.read_csv(RD_GEO.parent / "normalized" / "tt.csv")
    rd = rd[rd["source_category"] == "Total"]
    unit_of = {_fold(n): u for n, u in zip(rd["geo_name"], rd["geo_id"])}
    total = dict(zip(rd["geo_id"], rd["count"]))
    rows = []
    for m, g in df.groupby("geo_id"):
        unit = unit_of.get(_fold(g["geo_name"].iloc[0]))
        if unit is None:
            raise SystemExit(f"tt: {g['geo_name'].iloc[0]!r} has no religiondots unit")
        xt = g[g["kind"] == "ethnic_x_birthplace"]
        cnt = g[g["kind"] == "country"]
        scale = total[unit] / xt["count"].sum()
        for r in xt[xt["place"] == "Trinidad and Tobago"].itertuples():
            for node, x in tt2011.native(r.ethnic, m).items():
                rows.append((unit, node, r.count * scale * x))
        foreign = xt.loc[xt["place"] == "Foreign", "count"].sum() * scale
        for r in cnt.itertuples():
            for node, x in tt2011.born_in(r.country).items():
                rows.append((unit, node, foreign * r.count / cnt["count"].sum() * x))
    out = pd.DataFrame(rows, columns=["unit", "node", "count"])
    out = out.groupby(["unit", "node"], as_index=False)["count"].sum()
    if out["unit"].nunique() != 15 or abs(out["count"].sum() - sum(total.values())) > 1:
        raise SystemExit("tt: units or total do not match the census's municipal totals")
    out["tier"] = "derived"
    return out


ENTRY = dict(
    name="Trinidad and Tobago",
    source="Population and Housing Census 2011 (Central Statistical Office), tabulated on the "
           "CSO's REDATAM base at CEPAL",
    how="census, 2011, ethnic group and country of birth (no language question); people born "
        "in the country drawn as Trinidadian or Tobagonian Creole, white Trinidadians as "
        "English, people born abroad by their birth country's language",
    parts=[
        dict(covers="People born abroad",
             source="2011 census, country of birth, drawn on that country's language",
             people=52_847),
        dict(covers="People born in the country",
             source="2011 census, ethnic group, drawn as Trinidadian or Tobagonian Creole "
                    "(white Trinidadians as English)",
             rest=True),
    ],
    grain="15 municipalities, 88,000 people on average",
    gap="the institutional population, which the census's municipal tables leave out",
    view=[-61.95, 10.0, -60.5, 11.4],
    counts=_counts,
    mappings=["tt2011"],
    place=RD_GEO / "tt" / "tt_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "Trinidad and Tobago's census does not ask about language. Everyone born in the "
        "country is drawn as a speaker of Trinidadian Creole, or Tobagonian Creole in Tobago, "
        "except white Trinidadians, drawn as English speakers. Most people move between "
        "Standard English and the creole, so the line between them is not one a census could "
        "draw. Trinidad Bhojpuri is still known to some elderly Indo-Trinidadians, but no "
        "count of its speakers exists. Most Venezuelans arrived after 2016 and are not on "
        "this map."),
)
