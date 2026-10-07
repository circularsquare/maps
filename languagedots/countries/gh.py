# Ghana. 2021 PHC ethnicity by district (GSS StatsBank), read as home language with Afrobarometer
# R4-R9's shares per ethnic group (sources/gh_census.py); every row `modelled`. On Kontur hexes
# keyed to religiondots' 272 district units (sources/gh_geo.py). The record is sources/gh.md.
from _shared import *  # noqa: F401,F403
import numpy as np

UNITS = 272
GHANAIANS = 30_484_536


def _counts():
    import gh2021
    df = pd.read_csv(NORM / "gh.csv")
    if df["geo_id"].nunique() != UNITS:
        raise SystemExit(f"gh.csv: {df['geo_id'].nunique()} units, expected {UNITS} -- "
                         "re-run sources/gh_census.py")
    if abs(df["count"].sum() - GHANAIANS) > 100:
        raise SystemExit(f"gh.csv sums to {df['count'].sum():,.0f}, not {GHANAIANS:,}")
    df["node"] = df["source_category"].map(gh2021.resolve)
    missing = sorted(set(df.loc[df["node"].isna(), "source_category"]))
    if missing:
        raise SystemExit(f"gh.csv answers with no node: {missing}")
    df = df[df["count"] > 0].rename(columns={"geo_id": "unit"})
    out = by_unit(df)
    out["tier"] = "modelled"
    return out


class _GhHomeWeighter:
    """Inside a district, a language with a home area (sources/gh_census.py's HOME: Glottolog's
    point and a radius) puts its dots on each hex in proportion to Kontur population times the
    same distance kernel the counts used, so Kusaal leans to the Bawku side of a district and
    Farefare to the Bolgatanga side. Every other language goes on population. A placement
    weight only: the district's counts are unchanged."""

    def __init__(self, place):
        import gh2021
        sys.path.insert(0, str(ROOT / "sources"))
        from gh_census import HOME, FLOOR
        self.pop = place["pop"].to_numpy(dtype=float)
        c = place.geometry.to_crs(3857).centroid.to_crs(4326)
        lat, lon = c.y.to_numpy(), c.x.to_numpy()
        self.k = {}
        for ans, (la, lo, r) in HOME.items():
            dy = (lat - la) * 111.0
            dx = (lon - lo) * 111.0 * np.cos(np.radians((lat + la) / 2))
            self.k[gh2021.resolve(ans)] = FLOOR + np.exp(-0.5 * (dx * dx + dy * dy) / (r * r))
        self.n = {"home": 0, "pop": 0, "none": 0}

    def weights(self, node, idx, count, plain=False):
        p = self.pop[idx]
        if p.sum() <= 0:
            self.n["none"] += 1
            return None
        k = self.k.get(node)
        if k is not None:
            self.n["home"] += 1
            return p * k[idx]
        self.n["pop"] += 1
        return p

    def summary(self):
        return (f"{self.n['home']:,} (district, language) rows placed by home area x population, "
                f"{self.n['pop']:,} on population, {self.n['none']:,} on equal shares")


def _weight(place):
    if "pop" not in place.columns:
        raise SystemExit("gh_hexes.gpkg has no `pop` column: run sources/gh_geo.py")
    return _GhHomeWeighter(place)


ENTRY = dict(
    name="Ghana",
    source=("2021 Population and Housing Census, ethnic group by district (Ghana Statistical "
            "Service, StatsBank); home language by ethnic group from Afrobarometer rounds 4 "
            "to 9 (2008-2022)"),
    how=("census, 2021, ethnicity (no language question): each district's nine ethnic groups "
         "shared across the home languages each group speaks in Afrobarometer surveys"),
    parts=[dict(covers="Ghanaians",
                source="2021 census, ethnic group by district, shared by Afrobarometer 2008-2022 "
                       "home language per group",
                rest=True)],
    grain="272 districts and sub-metros, 112,000 people on average",
    gap="non-Ghanaians, about 350,000 (1.1%), whom the ethnicity table leaves out",
    view=[-3.35, 4.5, 1.3, 11.25],
    counts=_counts,
    mappings=["gh2021"],
    place=GEO / "gh" / "gh_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=_weight,
    note_public=(
        "Ghana's census does not ask about language. It asks ethnicity, published by district "
        "as nine broad groups (Mole-Dagbani alone covers Dagbani, Dagaare, Farefare, Kusaal "
        "and more). Each group is shared across languages by what its members told the "
        "Afrobarometer survey they speak at home. Akan, English and Hausa are drawn at the "
        "2017 round's mother-tongue answers instead, since many speak them at home as a second "
        "language. Small languages the survey rarely met, such as Bimoba and Efutu, are "
        "likely under drawn."),
)
