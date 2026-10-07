# Ethiopia. 2007 census mother tongue by zone (sources/et_uscb.py), placed on religiondots'
# Kontur hexes for 738 woredas, each woreda's hexes keyed to its zone.
from _shared import *  # noqa: F401,F403


def _zone(woreda):
    # USCB ids nest: woreda ETH_aa_bb_cc sits in zone ETH_aa_bb. sources/et_uscb.py asserts it
    # (every woreda's prefix names its own ADM2, and woredas sum to their zone exactly).
    return woreda.astype(str).str.rsplit("_", n=1).str[0]


def _counts():
    import et2007
    df = pd.read_csv(NORM / "et.csv")
    df = df[df["geo_level"] == "zone"]
    df["node"] = df["source_category"].map(et2007.resolve)
    return by_unit(df.rename(columns={"geo_id": "unit"}))


ENTRY = dict(
    name="Ethiopia",
    source=("2007 Population and Housing Census, Table 3.2 (Central Statistical Agency), "
            "as tabulated by the U.S. Census Bureau"),
    how="census, 2007, mother tongue",
    parts=[dict(covers="Everyone", source="2007 census, mother tongue", rest=True)],
    grain="93 zones, 793,000 people on average",
    gap="three woredas in Afar and one in Oromia, for which the census published no figures",
    view=[32.9, 3.3, 48.1, 15.1],
    counts=_counts,
    mappings=["et2007"],
    place=RD_GEO / "et" / "et_hexes.gpkg",
    place_unit=lambda g: _zone(g["unit"]),
    place_weight=pop_weight,
    note_public=("Ethiopia has not held a census since 2007, and its population has nearly "
                 "doubled since then. The census published mother tongue only by zone, so "
                 "inside a zone every language is spread over where people live, not where "
                 "its own speakers live; towns inside a zone are not told apart from the "
                 "countryside around them."),
)
