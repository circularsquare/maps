# Bangladesh. 2011 census ethnic groups by upazila (sources/bd_census.py), read as languages
# through taxonomy/bd2011.py: a proxy Anita allowed on 2026-10-05, every row `derived`. The
# census's Bengali remainder is split by place into Sylheti, Chittagonian and Bengali
# (bd2011.split_remainder, 2026-10-07), and the Rohingya camps, in no census, are added from
# UNHCR's registered figures (sources/bd_unhcr.py, 2026-10-07; Anita asked for both).
# Placed on religiondots' Kontur hexes for the same 544 upazilas, keyed on the same GEO_MATCH ids
# (read-only): everyone but the Rohingya off the camps, the Rohingya only on them.
# The record is sources/bd.md.
import re

import numpy as np

from _shared import *  # noqa: F401,F403

CAMP_DATE = "2026-08-31"          # the UNHCR figures' date, checked against bd_rohingya.csv
BHASAN_CHAR_RADIUS_M = 1500       # the island camp's housing cluster, around UNHCR's point


def key(name):
    """Camp name key; the same rule as sources/bd_unhcr.py, which asserts the join."""
    s = name.strip().lower().replace("extension", "x")
    s = re.sub(r"camp\s*0*(\d+)\s*", r"c\1", s)
    return re.sub(r"\s+", "", s)


def _camps():
    """The camps with UNHCR's count each, EPSG:4326: the A1 outlines joined by name, Bhasan Char
    as a disc around UNHCR's point, and UNHCR's `Other Camp` remainder added to the outline that
    holds its point (or the nearest)."""
    import geopandas as gpd
    from shapely.geometry import Point
    reg = pd.read_csv(NORM / "bd_rohingya.csv")
    if set(reg["date"]) != {CAMP_DATE}:
        raise SystemExit(f"bd_rohingya.csv is dated {sorted(set(reg['date']))}, not {CAMP_DATE}; "
                         "update CAMP_DATE, `gap`, `parts` and note_public together")
    out = gpd.read_file(GEO / "bd" / "bd_camps.gpkg").to_crs(4326)
    out["k"] = out["CampName"].map(key)
    reg["k"] = reg["camp"].map(key)
    g = out.merge(reg[["k", "count"]], on="k", how="left")
    g["count"] = g["count"].fillna(0.0)
    other = reg[reg["camp"] == "Other Camp"].iloc[0]
    pt = Point(other["lon"], other["lat"])
    inside = g.index[g.contains(pt)]
    j = inside[0] if len(inside) else g.geometry.distance(pt).idxmin()
    g.loc[j, "count"] += other["count"]
    bc = reg[reg["camp"] == "Bhasan Char"].iloc[0]
    disc = (gpd.GeoSeries([Point(bc["lon"], bc["lat"])], crs=4326).to_crs(32646)
            .buffer(BHASAN_CHAR_RADIUS_M).to_crs(4326).iloc[0])
    g = pd.concat([g[["CampName", "count", "geometry"]],
                   gpd.GeoDataFrame({"CampName": ["Bhasan Char"], "count": [float(bc["count"])]},
                                    geometry=[disc], crs=4326)], ignore_index=True)
    g = gpd.GeoDataFrame(g, geometry="geometry", crs=4326)
    if abs(g["count"].sum() - reg["count"].sum()) > 0.5:
        raise SystemExit("bd: camp counts lost in the outline join")
    return g


def _camp_units(camps, hexes):
    """Each camp's upazila: the unit of the hex holding its representative point, else of the
    nearest hex, so the unit always has hexes at the camp (Bhasan Char is keyed to Sandwip in
    the placement layer, though it is administered from Hatiya)."""
    import geopandas as gpd
    pts = gpd.GeoDataFrame(camps[["CampName"]], geometry=camps.geometry.representative_point(),
                           crs=camps.crs).to_crs(hexes.crs)
    hit = gpd.sjoin_nearest(pts, hexes[["unit", "geometry"]], how="left")
    hit = hit[~hit.index.duplicated()]
    return hit["unit"].astype(str)


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

    # the Bengali remainder by place: Sylheti, Chittagonian, Bengali (spec §3.3)
    rem = df["source_category"] == "Not an ethnic minority"
    rows = [dict(unit=r.unit, node=n, tier="derived", count=r.count * s)
            for r in df[rem].itertuples(index=False) for n, s in bd2011.split_remainder(r.unit)]
    split = pd.DataFrame(rows)
    if abs(split["count"].sum() - df.loc[rem, "count"].sum()) > 1:
        raise SystemExit("bd: the remainder split lost people")

    # the camps, from UNHCR: people the census does not have
    import geopandas as gpd
    camps = _camps()
    hexes = gpd.read_file(RD_GEO / "bd" / "bd_hexes.gpkg", columns=["unit"])
    camps["unit"] = _camp_units(camps, hexes).to_numpy()
    roh = camps.groupby("unit", as_index=False)["count"].sum()
    roh["node"], roh["tier"] = bd2011.ROHINGYA, "derived"

    out = pd.concat([df.loc[~rem, ["unit", "node", "tier", "count"]], split,
                     roh[["unit", "node", "tier", "count"]]], ignore_index=True)
    return out.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()


class _CampWeighter(PopWeighter):
    """Kontur's population, with the camps handled both ways.

    Everyone the census counted: every hex whose centre is in a Cox's Bazar camp outline gets
    zero weight. Kontur 2023 puts about 140,000 people in the camps, close to half of Ukhia's
    weight; without this Ukhia's census residents would be drawn mostly inside them.

    The Rohingya (UNHCR's camp figures): only on the camps, each camp's count spread over the
    hexes of its upazila by the share of the camp's area each covers (Bhasan Char: a 1.5 km disc).
    """

    def __init__(self, place):
        super().__init__(place)
        import bd2011
        import geopandas as gpd
        from regroup import move
        self.roh_node = move(bd2011.ROHINGYA)
        camps = _camps()
        camps["unit"] = _camp_units(camps, place).to_numpy()
        cxb = camps[camps["CampName"] != "Bhasan Char"].to_crs(place.crs)
        pts = gpd.GeoDataFrame(geometry=place.geometry.representative_point(), crs=place.crs)
        hit = gpd.sjoin(pts, cxb[["geometry"]], predicate="within", how="inner").index.unique()
        pos = place.index.get_indexer(hit)
        self.n_camp = len(pos)
        self.camp_pop = float(self.pop[pos].sum())
        self.pop[pos] = 0.0

        self.roh = np.zeros(len(place))
        hx = gpd.GeoDataFrame({"pos": np.arange(len(place)), "unit": place["unit"].astype(str).to_numpy()},
                              geometry=place.geometry.to_numpy(), crs=place.crs).to_crs(32646)
        cm = camps.to_crs(32646)
        for c in cm.itertuples(index=False):
            if c.count <= 0:
                continue
            cand = hx[(hx["unit"] == c.unit) & hx.intersects(c.geometry)]
            area = cand.intersection(c.geometry).area.to_numpy()
            if area.sum() <= 0:   # no hex of its unit overlaps: the nearest one takes it
                same = hx[hx["unit"] == c.unit]
                cand, area = same.loc[[same.distance(c.geometry).idxmin()]], np.array([1.0])
            np.add.at(self.roh, cand["pos"].to_numpy(), c.count * area / area.sum())
        self.roh_total = float(self.roh.sum())
        self.n_roh = int((self.roh > 0).sum())

    def weights(self, node, idx, count, plain=False):
        if node == self.roh_node:
            w = self.roh[idx]
            if w.sum() <= 0:
                raise SystemExit("bd: Rohingya row in a unit with no camp hexes")
            return w
        return super().weights(node, idx, count, plain)

    def summary(self):
        return (super().summary() + f"; {self.n_camp} hexes in the Cox's Bazar camps "
                f"({self.camp_pop:,.0f} Kontur people) given no census weight; "
                f"{self.roh_total:,.0f} Rohingya on {self.n_roh} camp hexes")


def _place_weight(place):
    if "pop" not in place.columns:
        print("  !! bd_hexes.gpkg has no `pop` column; equal shares")
        return None
    if not (GEO / "bd" / "bd_camps.gpkg").exists():
        raise SystemExit("bd: data/geo/bd/bd_camps.gpkg missing; run sources/bd_camps.py")
    return _CampWeighter(place)


ENTRY = dict(
    name="Bangladesh",
    source=("Population and Housing Census 2011, ethnic population by upazila (Bangladesh Bureau "
            "of Statistics), as tabulated by the U.S. Census Bureau; Rohingya refugee camps from "
            "UNHCR's registration figures, 31 August 2026"),
    how=("census, 2011, ethnic group, each group drawn as its language; everyone else as Bengali, "
         "Sylheti or Chittagonian by where they live; Rohingya camps from UNHCR registered "
         "refugees, August 2026"),
    parts=[
        dict(covers="Ethnic minorities", source="2011 census, ethnic group, each drawn as its "
             "language", people=1_586_183),
        dict(covers="Sylheti and Chittagonian", source="2011 census, everyone else, drawn by "
             "region", nodes=["indoeuropean.indoaryan.eastern.sylheti",
                              "indoeuropean.indoaryan.eastern.chittagonian"]),
        dict(covers="Rohingya refugee camps", source="UNHCR registered refugees, 31 August 2026",
             nodes=["indoeuropean.indoaryan.eastern.rohingya"]),
        dict(covers="Everyone else", source="2011 census, drawn as Bengali", rest=True),
    ],
    grain="544 upazilas, 265,000 people on average; the camps one by one",
    gap=("the census asked no language question; Rohingya living outside the camps and not "
         "registered there are not drawn"),
    view=[88.0, 20.6, 92.7, 26.7],
    counts=_counts,
    mappings=["bd2011"],
    place=RD_GEO / "bd" / "bd_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=_place_weight,
    note_public=(
        "Bangladesh's census asks ethnic group, not language. This map draws each of the 27 "
        "ethnic groups the 2011 census named as the language of that group (Chakma as Chakma, "
        "Santal as Santali, Tripura as Kokborok). Everyone else is drawn by where they live. In "
        "Sylhet and Moulvibazar districts, eastern Sunamganj and north-eastern Habiganj they are "
        "drawn as Sylheti, and in Chittagong district (except Sandwip island) and Cox's Bazar as "
        "Chittagonian. Elsewhere they are drawn as Bengali. Sylheti and Chittagonian are separate "
        "languages in linguists' catalogues, but no census or survey counts their speakers, so "
        "the regions are an estimate drawn from descriptions of where each is spoken. In "
        "Chittagong city only 56% are drawn as Chittagonian, the share a 2019 World Bank survey "
        "found born in Chittagong; in-migrants elsewhere, such as in Sylhet city, are not "
        "separated. The Urdu-speaking communities often called Biharis, a few hundred thousand "
        "people, have no census category and are drawn with their neighbours. Many Garo, Santal, "
        "Oraon and Munda families now speak Bengali or Sadri at home, and they are still drawn "
        "on their heritage language. The census's other ethnic groups, mostly tea garden "
        "communities in Habiganj and Moulvibazar, are drawn as other languages of Bangladesh's "
        "ethnic minorities. The counts are from 2011; the 2022 census's district reports are "
        "only partly online. Ethnic leaders have said the census undercounts their peoples. The "
        "Rohingya refugee camps in Ukhia and Teknaf, and on Bhasan Char, are in no census. They "
        "are drawn as Rohingya from UNHCR's figures for refugees registered in each camp on 31 "
        "August 2026, about 1.16 million people, placed on the camp outlines."),
)
