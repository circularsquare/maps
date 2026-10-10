# Afghanistan. MICS6 2022-23 (UNICEF and NSIA), language of the household head, read as every
# member's: weighted shares per province (urban and rural apart in 13 provinces) applied to
# NSIA's 1404 settled population (sources/af_mics.py), placed on religiondots' 34-province Kontur
# hexes. Languages MICS does not list come from cited speaker estimates (ask 019 route), kept to
# their homeland (ZONES). The 2006-07 village-majority proxy this replaced is sources/af_mrrd.py,
# kept as a comparison (data/normalized/af_mrrd.csv). Record: sources/af.md.
from _shared import *  # noqa: F401,F403
import numpy as np

SETTLED = 34_935_197
KUCHI = 1_500_000
SOURCE_IDS = {"mics6_2022_hc1b", "af_speaker_estimates"}

# The languages drawn from speaker estimates (sources/af_mics.py MINORITIES, BRAHUI) are kept to
# their homeland inside the province (a placement only; the counts are af_mics.py's either way).
# Glottolog points where they sit in Afghanistan, else the district from the sources:
#   Shughni: Afghan Shighnan along the Panj, 37.2-38.0N, 71.2-71.8E (Glottolog's point is Khorog,
#     across the river); Wakhi: the Wakhan corridor 71.7-73.6E, east of Ishkashim town;
#   Kyrgyz: the Little Pamir (east of 73.6E) and Big Pamir (north of 37.2N, east of 73.2E);
#   Ishkashimi, Sanglechi, Munji: within 12, 15 and 25 km of their Glottolog points (Ishkashim,
#     the Sanglech valley of Zebak, the Munjan valley of Kuran wa Munjan);
#   Parachi: within 15 km of upper Nijrab, Kapisa (35.05N 69.65E; Glottolog's point falls in
#     Badakhshan, so not used); Gawar-Bati: within 25 km of Glottolog's point at Arandu, which
#     takes in the Afghan side of the Kunar valley opposite;
#   Brahui: the southern belt, "Chakhansoor to Shorawak": Nimroz south of 31.0N outside Zaranj,
#     Helmand south of 31.3N, Kandahar south of 31.0N west of 66.3E (not Spin Boldak).
IR = "indoeuropean.iranian"
ZONES = {
    ("AF17", f"{IR}.shughni"): ("box", 71.2, 37.2, 71.8, 38.0),
    ("AF17", f"{IR}.wakhi"): ("box", 71.7, 36.4, 73.6, 37.3),
    ("AF17", "turkic.kyrgyz"): ("pamirs",),
    ("AF17", f"{IR}.ishkashimi"): ("near", 36.71, 71.61, 12),
    ("AF17", f"{IR}.sanglechi"): ("near", 36.45, 71.30, 15),
    ("AF17", f"{IR}.munji"): ("near", 35.93, 70.94, 25),
    ("AF02", f"{IR}.parachi"): ("near", 35.05, 69.65, 15),
    ("AF15", "indoeuropean.indoaryan.dardic.gawarbati"): ("near", 35.198, 71.543, 25),
    ("AF34", "dravidian.northern.brahui"): ("south_nimroz",),
    ("AF30", "dravidian.northern.brahui"): ("box", 60.0, 28.0, 66.0, 31.3),
    ("AF27", "dravidian.northern.brahui"): ("box", 60.0, 28.0, 66.3, 31.0),
}


def _table():
    import af2022
    df = pd.read_csv(NORM / "af.csv", dtype={"geo_id": str})
    if df["geo_id"].nunique() != 34:
        raise SystemExit(f"af.csv: {df['geo_id'].nunique()} provinces, expected 34")
    if df["count"].sum() != SETTLED:
        raise SystemExit(f"af.csv sums to {df['count'].sum():,}, expected {SETTLED:,}")
    if not set(df["source_id"]) <= SOURCE_IDS:
        raise SystemExit(f"af.csv sources {sorted(set(df['source_id']))}: rerun sources/af_mics.py")
    df["node"] = df["source_category"].map(af2022.resolve)
    # religiondots' hex layer is keyed by the same AF01-AF34 ids (its af_lookup.csv: unit == geo_id)
    df["unit"] = df["geo_id"]
    return df


def _counts():
    df = _table()
    df["tier"] = "modelled"
    return df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()


class _AfWeighter:
    """Population weights, with two refinements inside a province (placement only; the counts
    are af.csv's either way):
    - in the 13 provinces where sources/af_mics.py tabulates MICS's urban and rural strata apart,
      each language's urban part goes on the province's urban hexes and the rest on the rural
      ones. Urban hexes are the densest, taken in order of population until they hold NSIA's
      urban share of the province (a stand-in for municipal boundaries, which the layer lacks);
    - the (province, language) pairs in ZONES are cut to their homeland."""

    def __init__(self, place):
        self.pop = place["pop"].to_numpy(dtype=float)
        c = place.geometry.to_crs(3857).centroid.to_crs(4326)
        self.x, self.y = c.x.to_numpy(), c.y.to_numpy()
        self.unit = place["unit"].astype(str).to_numpy()
        self.n = {"zone": 0, "strata": 0, "pop": 0, "none": 0}
        t = _table()
        st = t[t["stratum"].isin(["urban", "rural"])]
        g = st.groupby(["unit", "node", "stratum"])["count"].sum().unstack(fill_value=0)
        self.f_urban = (g["urban"] / (g["urban"] + g["rural"])).to_dict()
        ushare = (st[st["stratum"] == "urban"].groupby("unit")["count"].sum()
                  / st.groupby("unit")["count"].sum())
        self.urban = np.zeros(len(self.pop), dtype=bool)
        for u, s in ushare.items():
            i = np.where(self.unit == u)[0]
            if not len(i):
                raise SystemExit(f"af: province {u} has no hexes")
            o = i[np.argsort(-self.pop[i], kind="stable")]
            cum = np.cumsum(self.pop[o])
            k = int(np.searchsorted(cum, s * cum[-1])) + 1
            self.urban[o[:k]] = True

    def _mask(self, z, idx):
        x, y = self.x[idx], self.y[idx]
        if z[0] == "box":
            w, s, e, n = z[1:]
            return (x >= w) & (x <= e) & (y >= s) & (y <= n)
        if z[0] == "near":
            lat, lon, km = z[1:]
            d = np.hypot((y - lat) * 111.0, (x - lon) * 111.0 * np.cos(np.radians(lat)))
            return d <= km
        if z[0] == "pamirs":
            return (x >= 73.6) | ((x >= 73.2) & (y >= 37.2))
        if z[0] == "south_nimroz":
            d = np.hypot((y - 30.96) * 111.0, (x - 61.86) * 111.0 * np.cos(np.radians(31)))
            return (y < 31.0) & (d > 15)
        raise ValueError(z)

    def weights(self, node, idx, count, plain=False):
        pop = self.pop[idx]
        if pop.sum() <= 0:
            self.n["none"] += 1
            return None
        z = ZONES.get((self.unit[idx[0]], node))
        if z:
            w = pop * self._mask(z, idx)
            if w.sum() <= 0:
                raise SystemExit(f"af: zone {z} holds no population for {node}")
            self.n["zone"] += 1
            return w
        f = self.f_urban.get((self.unit[idx[0]], node))
        if f is not None:
            u = self.urban[idx]
            up, rp = pop[u].sum(), pop[~u].sum()
            if (f == 0 or up > 0) and (f == 1 or rp > 0):
                self.n["strata"] += 1
                return np.where(u, pop * (f / up if up else 0), pop * ((1 - f) / rp if rp else 0))
        self.n["pop"] += 1
        return pop

    def summary(self):
        return (f"{self.n['zone']} (province, language) rows kept to their homeland, "
                f"{self.n['strata']} placed by MICS's urban and rural strata, "
                f"{self.n['pop']:,} on population, {self.n['none']} on equal shares")


def _weight(place):
    if "pop" not in place.columns:
        raise SystemExit("af_hexes.gpkg has no `pop` column")
    return _AfWeighter(place)


ENTRY = dict(
    name="Afghanistan",
    source=("Afghanistan Multiple Indicator Cluster Survey 2022-23 (MICS6; UNICEF and the "
            "National Statistics and Information Authority), microdata; the National Statistics "
            "and Information Authority's settled population for 2025-26; speaker estimates for "
            "minority languages from Ethnologue (via Wikipedia and Bashir 2003), the Endangered "
            "Language Alliance and Callahan (2007)"),
    how=("a household survey, 2022-23, language of the household head, read as every member's; "
         "weighted shares per province (towns and countryside apart in 13 provinces) applied to "
         "each province's 2025-26 population; languages the survey does not list from speaker "
         "estimates, placed in their home valleys"),
    parts=[
        dict(covers="Pamiri languages, Kyrgyz, Parachi, Gawar-Bati and Brahui",
             source="published speaker estimates (Ethnologue and others), placed in their "
                    "home districts",
             nodes=[f"{IR}.shughni", f"{IR}.wakhi", f"{IR}.munji", f"{IR}.sanglechi",
                    f"{IR}.ishkashimi", f"{IR}.parachi", "turkic.kyrgyz",
                    "indoeuropean.indoaryan.dardic.gawarbati", "dravidian.northern.brahui"]),
        dict(covers="Everyone else",
             source="UNICEF MICS 2022-23, about 23,000 households, language of the household "
                    "head, on the 2025-26 population",
             rest=True),
    ],
    grain="34 provinces, 1,028,000 people on average",
    gap="1.5 million nomadic Kuchis, who have no province in the population figures (4.1%)",
    view=[60.5, 29.3, 75.0, 38.5],
    counts=_counts,
    mappings=["af2022"],
    place=RD_GEO / "af" / "af_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=_weight,
    note_public=(
        "Afghanistan has had no census since 1979. These shares come from UNICEF's household "
        "survey of 2022-23, about 23,000 households in all 34 provinces, interviewed under the "
        "Taliban administration. Each household is drawn on the language of its head, and the "
        "shares are applied to the statistics authority's population for 2025-26. In 13 "
        "provinces the survey counted towns and countryside apart, and there each language's "
        "town share is placed in the province's most densely settled places; elsewhere dots "
        "follow population. The survey does not list Hazaragi, so Hazaras are drawn as Dari. "
        "Nor does it list the Pamiri languages of Badakhshan, the Kyrgyz of the Wakhan, "
        "Parachi, Gawar-Bati or Brahui. These are drawn from published speaker estimates, "
        "mostly Ethnologue's, placed in their home valleys and taken out of the language their "
        "speakers would most likely have given (Brahui out of Balochi). They are estimates, not "
        "counts. The survey visited 28 places in most provinces, so a minority living in a few "
        "valleys can be missed. The 1.5 million nomadic Kuchis are not on the map."),
)
