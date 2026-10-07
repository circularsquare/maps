# Central African Republic. RGPH03 2003, language commonly spoken, by commune
# (sources/cf_uscb.py), on religiondots' Kontur hexes for the same 177 communes (USCB GEO_MATCH
# ids on both sides, the 2003 boundary set).
from _shared import *  # noqa: F401,F403

FUL = "Fulfuldé"


def _counts():
    import cf2003
    df = pd.read_csv(NORM / "cf.csv")
    df = df[df["geo_level"] == "commune"].copy()
    df["tier"] = "measured"
    # The published Fulfulde column also holds the children under three, who were not asked.
    # Replace it with the estimate of real speakers left after removing them (sources/cf_uscb.py
    # measures the floor and writes cf_under3.csv); the under-threes are not drawn.
    u3 = pd.read_csv(NORM / "cf_under3.csv").set_index("geo_id")
    ful = df["source_category"] == FUL
    pub = df.loc[ful].set_index("geo_id")["count"]
    if not (pub == u3.loc[pub.index, "fulfulde_published"]).all():
        raise SystemExit("cf: cf_under3.csv is out of step with cf.csv; re-run sources/cf_uscb.py")
    df.loc[ful, "count"] = df.loc[ful, "geo_id"].map(u3["fulfulde_est"]).to_numpy()
    df.loc[ful, "tier"] = "derived"
    df = df[df["count"] > 0]
    df["node"] = df["source_category"].map(cf2003.resolve)
    df = df.rename(columns={"geo_id": "unit"})
    if df["unit"].nunique() != 177:
        raise SystemExit(f"cf: {df['unit'].nunique()} communes, expected 177")
    return df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()


ENTRY = dict(
    name="Central African Republic",
    source=("Third General Census of Population and Housing (RGPH03), 2003, language commonly "
            "spoken (ICASEES), as tabulated by the U.S. Census Bureau"),
    how="census, 2003, language commonly spoken",
    parts=[dict(covers="Everyone aged 3 and over", source="2003 census, language commonly "
                "spoken", rest=True)],
    grain="177 communes, 22,000 people on average",
    gap=("children under three, who were not asked (about 382,000, 10% of the census), and "
         "4.3% of the census with no language recorded, up to four in five people in a few "
         "communes of Ouham and Ouham-Pendé"),
    view=[14.2, 2.0, 27.6, 11.2],
    counts=_counts,
    mappings=["cf2003"],
    place=RD_GEO / "cf" / "cf_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "The census asked for the language each person commonly speaks, one answer each, so "
        "Sango, the national language, is drawn wherever people gave it rather than their "
        "own: it was a third of answers nationally and half of Bangui's. In Bangui 27% named a "
        "language from outside the country, which the census does not break down; French is "
        "the likeliest, but it is drawn as unnamed. The published table counts children under "
        "three as Fulfulde speakers, so the Fulfulde drawn here is what is left after taking "
        "out an estimate of those children. "
        "This is the country in 2003, and there has been no census since. The war that "
        "began in 2013 drove much of the Muslim population of the west and centre, Peul, "
        "Arabic and Hausa speakers among them, from their homes, and many have not returned."),
)
