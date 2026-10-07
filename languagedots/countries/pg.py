# Papua New Guinea. No census publishes first languages by area: a language-area model. Each
# province's 2024 people shared among the languages whose Glottolog point is in it, by Joshua
# Project's (Ethnologue-based) speaker figures; Port Moresby (NCD) by the national mix
# (sources/pg_build.py). Religiondots' Kontur hexes for the 22 provinces. Record: sources/pg.md.
from _shared import *  # noqa: F401,F403
import numpy as np

PNG_2024 = 10_185_363
UNITS = 22
NCD = "PG04"
FALLBACK_KM = 20.0   # a language whose nearest-point cell holds nobody: pop x exp(-d / 20 km)


def _counts():
    import pg2024
    df = pd.read_csv(NORM / "pg.csv", dtype={"geo_id": str})
    if df["geo_id"].nunique() != UNITS:
        raise SystemExit(f"pg.csv: {df['geo_id'].nunique()} provinces, expected {UNITS} -- "
                         "re-run sources/pg_build.py")
    if int(df["count"].sum()) != PNG_2024:
        raise SystemExit(f"pg.csv sums to {int(df['count'].sum()):,}, not {PNG_2024:,}")
    lut = pd.read_csv(RD_GEO / "pg" / "pg_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    if df["unit"].isna().any():
        raise SystemExit("pg.csv provinces missing from religiondots' pg_lookup.csv")
    df["node"] = df["source_category"].map(pg2024.resolve)
    out = df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()
    return out[["unit", "node", "count", "tier"]]


class _PgWeighter:
    """Inside a province, a language's dots go to the hexes nearer its Glottolog point than to
    any other drawn language's point in that province, by Kontur population (a placement only;
    the counts are sources/pg_build.py's). NCD has no points: plain population."""

    def __init__(self, place):
        import geopandas as gpd
        self.pop = place["pop"].to_numpy(dtype=float)
        c = place.geometry.to_crs(32755).centroid
        self.x, self.y = c.x.to_numpy(), c.y.to_numpy()
        self.unit = place["unit"].astype(str).to_numpy()
        lang = pd.read_csv(NORM / "pg_languages.csv", dtype={"unit": str})
        p = gpd.GeoSeries(gpd.points_from_xy(lang["Longitude"], lang["Latitude"]),
                          crs=4326).to_crs(32755)
        lang["px"], lang["py"] = p.x.to_numpy(), p.y.to_numpy()
        self.lang = lang
        self.where = dict(zip(lang["node"], zip(lang["unit"], lang["px"], lang["py"])))
        self.cells = {}
        self.n = {"cell": 0, "near": 0, "pop": 0, "none": 0}

    def _cell(self, unit, idx):
        key = (unit, len(idx), int(idx[0]))
        if key not in self.cells:
            L = self.lang[self.lang["unit"] == unit]
            d2 = ((self.x[idx][:, None] - L["px"].to_numpy()[None, :]) ** 2
                  + (self.y[idx][:, None] - L["py"].to_numpy()[None, :]) ** 2)
            nearest = L["node"].to_numpy()[d2.argmin(axis=1)]
            self.cells[key] = nearest
        return self.cells[key]

    def weights(self, node, idx, count, plain=False):
        pop = self.pop[idx]
        if pop.sum() <= 0:
            self.n["none"] += 1
            return None
        unit = self.unit[idx[0]]
        home = self.where.get(node)
        if unit == NCD or home is None or home[0] != unit:
            self.n["pop"] += 1
            return pop
        w = pop * (self._cell(unit, idx) == node)
        if w.sum() > 0:
            self.n["cell"] += 1
            return w
        d = np.hypot(self.x[idx] - home[1], self.y[idx] - home[2]) / 1000
        self.n["near"] += 1
        return pop * np.exp(-d / FALLBACK_KM)

    def summary(self):
        return (f"{self.n['cell']} (province, language) rows on the language's nearest-point "
                f"cell, {self.n['near']} near its point (cell empty), {self.n['pop']:,} on "
                f"population (NCD), {self.n['none']} on equal shares")


def _weight(place):
    if "pop" not in place.columns:
        raise SystemExit("pg_hexes.gpkg has no `pop` column")
    return _PgWeighter(place)


ENTRY = dict(
    name="Papua New Guinea",
    source=("Glottolog 5 language locations and classification (CC BY); speaker figures from "
            "Joshua Project's people groups for Papua New Guinea, which follow Ethnologue; each "
            "province's people from the 2024 census Final Figures"),
    how=("no census counts first languages; a language-area model: each province's people "
         "shared among the languages spoken there, by speaker estimates"),
    parts=[
        dict(covers="Every province but Port Moresby",
             source="2024 census population, shared among the languages Glottolog locates there "
                    "by Joshua Project's speaker estimates",
             rest=True),
        dict(covers="Port Moresby (National Capital District)",
             source="2024 census population, shared among every language by the national mix",
             people=756_754),
    ],
    grain="22 provinces, 463,000 people on average; inside a province, each language's own area",
    gap=("Tok Pisin as a first language, which many town-born people speak, is not drawn: no "
         "published figure says how many"),
    view=[140.8, -11.7, 156.0, -0.8],
    counts=_counts,
    mappings=["pg2024"],
    place=RD_GEO / "pg" / "pg_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=_weight,
    note_public=(
        "Papua New Guinea has about 840 languages, and no census has ever published how many "
        "people speak each one or where; the 2011 and 2024 censuses asked only about literacy. "
        "So this map is a model of language areas, not a count. Each province's 2024 people are "
        "shared among the languages Glottolog locates there, in proportion to Joshua Project's "
        "speaker estimates (which follow Ethnologue), and each language's dots go to the "
        "villages nearest its location. A language spoken across a province border is drawn "
        "wholly in one province. Port Moresby is drawn as a mix of every language, since "
        "nothing says where its people come from. Tok Pisin, the first language of many "
        "town-born people, is not drawn: no published figure says how many speak it first."),
)
