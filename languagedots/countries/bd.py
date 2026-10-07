# Bangladesh. 2011 census ethnic groups by upazila (sources/bd_census.py), read as languages
# through taxonomy/bd2011.py: a proxy Anita allowed on 2026-10-05, every row `derived`. Placed on
# religiondots' Kontur hexes for the same 544 upazilas, keyed on the same GEO_MATCH ids (read-only),
# with the Rohingya camps taken out of the placement weight (sources/bd_camps.py).
# The record is sources/bd.md.
from _shared import *  # noqa: F401,F403


def _counts():
    import bd2011
    df = pd.read_csv(NORM / "bd.csv")
    if df["geo_id"].nunique() != 544:
        raise SystemExit("bd.csv: expected 544 upazilas; re-run sources/bd_census.py")
    df["node"] = df["source_category"].map(bd2011.resolve)
    missing = sorted(set(df.loc[df["node"].isna(), "source_category"]))
    if missing:
        raise SystemExit(f"bd.csv categories that resolve to nothing: {missing}")
    if set(df["tier"]) != {"derived"}:
        raise SystemExit(f"bd.csv: tiers {sorted(set(df['tier']))}")
    df["unit"] = df["geo_id"]
    df = df[df["count"] > 0]
    return df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()


class _CampFreeWeighter(PopWeighter):
    """Kontur's population, with every hex whose centre is in a Rohingya camp set to zero.

    The camps (2017 on) are not in the 2011 census, but Kontur 2023 holds about 140,000 people in
    them, close to half of Ukhia's weight; without this Ukhia's census residents are drawn mostly
    inside the camps. The camps themselves are not drawn (no census counts them).
    """

    def __init__(self, place):
        super().__init__(place)
        import geopandas as gpd
        camps = gpd.read_file(GEO / "bd" / "bd_camps.gpkg").to_crs(place.crs)
        pts = gpd.GeoDataFrame(geometry=place.geometry.representative_point(), crs=place.crs)
        hit = gpd.sjoin(pts, camps, predicate="within", how="inner").index.unique()
        pos = place.index.get_indexer(hit)
        self.n_camp = len(pos)
        self.camp_pop = float(self.pop[pos].sum())
        self.pop[pos] = 0.0

    def summary(self):
        return (super().summary() + f"; {self.n_camp} hexes in the Rohingya camps "
                f"({self.camp_pop:,.0f} Kontur people) given no weight")


def _place_weight(place):
    if "pop" not in place.columns:
        print("  !! bd_hexes.gpkg has no `pop` column; equal shares")
        return None
    if not (GEO / "bd" / "bd_camps.gpkg").exists():
        raise SystemExit("bd: data/geo/bd/bd_camps.gpkg missing; run sources/bd_camps.py")
    return _CampFreeWeighter(place)


ENTRY = dict(
    name="Bangladesh",
    source=("Population and Housing Census 2011, ethnic population by upazila (Bangladesh Bureau "
            "of Statistics), as tabulated by the U.S. Census Bureau"),
    how="census, 2011, ethnic group, each group drawn as its language; everyone else as Bengali",
    parts=[
        dict(covers="Ethnic minorities", source="2011 census, ethnic group, each drawn as its "
             "language", people=1_586_183),
        dict(covers="Everyone else", source="2011 census, drawn as Bengali", rest=True),
    ],
    grain="544 upazilas, 265,000 people on average",
    gap=("the census asked no language question; the Rohingya refugee camps in Cox's Bazar, about "
         "a million people who arrived mostly after 2011, are in no census and are not drawn"),
    view=[88.0, 20.6, 92.7, 26.7],
    counts=_counts,
    mappings=["bd2011"],
    place=RD_GEO / "bd" / "bd_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=_place_weight,
    note_public=(
        "Bangladesh's census asks ethnic group, not language. This map draws each of the 27 "
        "ethnic groups the 2011 census named as the language of that group (Chakma as Chakma, "
        "Santal as Santali, Tripura as Kokborok), and everyone not in one of them as Bengali. "
        "That is an estimate, and it is wrong in known ways. Sylheti and Chittagonian, spoken by "
        "perhaps 11 and 13 million people, are counted as Bengali and drawn as Bengali here. So "
        "are the Urdu-speaking communities often called Biharis, a few hundred thousand people, "
        "because the census has no category for them. Many Garo, Santal, Oraon and Munda families "
        "now speak Bengali or Sadri at home, and they are still drawn on their heritage language. "
        "The census's other ethnic groups, mostly tea garden communities in Habiganj and "
        "Moulvibazar, are drawn as other languages of Bangladesh's ethnic minorities. The counts "
        "are from 2011; the 2022 census's district reports are only partly online. Ethnic "
        "leaders have said the census undercounts their peoples. The Rohingya refugee camps "
        "are in no census and are left empty."),
)
