# Saint Lucia. Census 2022 birthplace per district (sources/lc_census.py, CSO's REDATAM base at
# CEPAL), read as languages by taxonomy/lc2022.py and scaled to CSO's published (weighted)
# district totals (religiondots' normalized lc.csv, read-only). Placed on religiondots' hexes.
from _shared import *  # noqa: F401,F403
import re as _re


def _fold(s):
    return _re.sub(r"[^a-z]+", "", str(s).lower())


def _counts():
    import lc2022
    df = pd.read_csv(NORM / "lc.csv", dtype={"geo_id": str}, keep_default_na=False)
    rd = pd.read_csv(RD_GEO.parent / "normalized" / "lc.csv")
    rd = rd[rd["source_category"] == "Total"]
    unit_of = {_fold(n): u for n, u in zip(rd["geo_name"], rd["geo_id"])}
    total = dict(zip(rd["geo_id"], rd["count"]))
    df["unit"] = [unit_of["castries"] if g in ("01", "02", "03") else unit_of.get(_fold(n))
                  for g, n in zip(df["geo_id"], df["geo_name"])]
    if df["unit"].isna().any() or df["unit"].nunique() != 10:
        raise SystemExit("lc: districts do not join religiondots' 10 units")
    rows = []
    for unit, g in df.groupby("unit"):
        xt = g[g["kind"] == "ethnic_x_birthplace"]
        native = xt.loc[xt["place"] == "Born St Lucia", "count"].sum()
        foreign = xt.loc[xt["place"] == "Born elsewhere", "count"].sum()
        scale = total[unit] / (native + foreign)
        rows.append((unit, lc2022.CODES["Born St Lucia"], native * scale))
        cnt = g[(g["kind"] == "country") & (g["country"] != "Not reported")]
        for r in cnt.itertuples():
            rows.append((unit, lc2022.CODES[r.country], foreign * scale * r.count / cnt["count"].sum()))
    out = pd.DataFrame(rows, columns=["unit", "node", "count"])
    out = out.groupby(["unit", "node"], as_index=False)["count"].sum()
    if abs(out["count"].sum() - sum(total.values())) > 1:
        raise SystemExit("lc: total does not match CSO's district totals")
    out["tier"] = "derived"
    return out


ENTRY = dict(
    name="Saint Lucia",
    source="2022 Population and Housing Census (Central Statistics Office), tabulated on the "
           "CSO's REDATAM base at CEPAL, against the published district totals",
    how="census, 2022, place of birth (no language question); people born in Saint Lucia "
        "drawn as Kweyol speakers",
    parts=[
        dict(covers="Born in Saint Lucia",
             source="2022 census, place of birth, drawn as Kweyol",
             nodes=["creole.french_based.antillean"]),
        dict(covers="Born abroad",
             source="2022 census, birthplace by world region; North America drawn as English",
             rest=True),
    ],
    grain="10 districts, 17,200 people on average",
    gap="The 1,114 people outside households, whom the published district totals leave out.",
    view=[-61.1, 13.7, -60.85, 14.12],
    counts=_counts,
    mappings=["lc2022"],
    place=RD_GEO / "lc" / "lc_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "Saint Lucia's census does not ask about language. Everyone born in Saint Lucia is "
        "drawn as a speaker of Kweyol, the French-based creole. Many Saint Lucians, above all "
        "younger people in Castries and Gros Islet, grow up speaking English first, but no "
        "survey counts how many, so this map overstates Kweyol. The census missed about a "
        "quarter of the population; the map uses the statistics office's weighted district "
        "totals."),
)
