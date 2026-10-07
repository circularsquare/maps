# Afghanistan. The village-majority language of each province, from the Ministry of Rural
# Rehabilitation and Development's provincial profiles (c. 2006-07, reprinted in CALL Handbook
# 11-16 Annex A), on NSIA's 1404 settled population (sources/af_mrrd.py), placed on religiondots'
# 34-province Kontur hexes. A proxy Anita allowed on 2026-10-05 (ask/017-af.md). Minority
# languages the profiles never name come from cited speaker estimates (ask 019 route), kept to
# their homeland (ZONES). Record: sources/af.md.
from _shared import *  # noqa: F401,F403
import numpy as np

SETTLED = 34_935_197
KUCHI = 1_500_000

# The languages drawn from speaker estimates (sources/af_mrrd.py MINORITIES, BRAHUI) are kept to
# their homeland inside the province (a placement only; the counts are af_mrrd.py's either way).
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


def _counts():
    import af2007
    df = pd.read_csv(NORM / "af.csv", dtype={"geo_id": str})
    if df["geo_id"].nunique() != 34:
        raise SystemExit(f"af.csv: {df['geo_id'].nunique()} provinces, expected 34")
    if df["count"].sum() != SETTLED:
        raise SystemExit(f"af.csv sums to {df['count'].sum():,}, expected {SETTLED:,}")
    df["node"] = df["source_category"].map(af2007.resolve)
    df = df[df["node"].notna()]
    # religiondots' hex layer is keyed by the same AF01-AF34 ids (its af_lookup.csv: unit == geo_id)
    df["unit"] = df["geo_id"]
    df["tier"] = np.where(df["source_category"].isin(af2007.MODELLED), "modelled", "derived")
    return df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()


class _AfWeighter:
    """Population weights, cut to a homeland zone for the (province, language) pairs in ZONES."""

    def __init__(self, place):
        self.pop = place["pop"].to_numpy(dtype=float)
        c = place.geometry.to_crs(3857).centroid.to_crs(4326)
        self.x, self.y = c.x.to_numpy(), c.y.to_numpy()
        self.unit = place["unit"].astype(str).to_numpy()
        self.n = {"zone": 0, "pop": 0, "none": 0}

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
        self.n["pop"] += 1
        return pop

    def summary(self):
        return (f"{self.n['zone']} (province, language) rows kept to their homeland, "
                f"{self.n['pop']:,} on population, {self.n['none']} on equal shares")


def _weight(place):
    if "pop" not in place.columns:
        raise SystemExit("af_hexes.gpkg has no `pop` column")
    return _AfWeighter(place)


ENTRY = dict(
    name="Afghanistan",
    source=("Ministry of Rural Rehabilitation and Development provincial profiles, c. 2006-07, as "
            "reprinted in the US Army's CALL Handbook 11-16 (2011), Annex A; on the National "
            "Statistics and Information Authority's settled population for 2025-26; the Asia "
            "Foundation, Afghanistan in 2006: A Survey of the Afghan People, Q-45; speaker "
            "estimates for minority languages from Ethnologue (via Wikipedia and Bashir 2003), "
            "the Endangered Language Alliance and Callahan (2007)"),
    how=("no census or open survey asks; each village's majority language as a share of each "
         "province; Kabul city and Herat split to the Asia Foundation's 2006 national shares; "
         "minority languages from speaker estimates, placed in their home valleys"),
    parts=[
        dict(covers="Pamiri languages, Kyrgyz, Parachi, Gawar-Bati and Brahui",
             source="published speaker estimates (Ethnologue and others), placed in their "
                    "home districts",
             nodes=[f"{IR}.shughni", f"{IR}.wakhi", f"{IR}.munji", f"{IR}.sanglechi",
                    f"{IR}.ishkashimi", f"{IR}.parachi", "turkic.kyrgyz",
                    "indoeuropean.indoaryan.dardic.gawarbati", "dravidian.northern.brahui"]),
        dict(covers="Kabul city and Herat, Dari and Pashto",
             source="no figure of their own; drawn 92% Dari and 8% Pashto to match the Asia "
                    "Foundation's 2006 national first-language shares",
             people=7_696_871),
        dict(covers="Everyone else",
             source="Ministry of Rural Rehabilitation and Development profiles, c. 2006-07, "
                    "village majority language, on the 2025-26 population",
             rest=True),
    ],
    grain="34 provinces, 1,028,000 people on average",
    gap=("Takhar and Kunduz (2.5 million), whose profiles give no usable figure; 0.7 million the "
         "profiles leave out elsewhere; and 1.5 million nomadic Kuchis, who have no province. "
         "4.7 million in all, 13%"),
    view=[60.5, 29.3, 75.0, 38.5],
    counts=_counts,
    mappings=["af2007"],
    place=RD_GEO / "af" / "af_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=_weight,
    note_public=(
        "Afghanistan has had no census since 1979, and the one survey that asked the language "
        "spoken at home, the Asia Foundation's Survey of the Afghan People, withdrew its data in "
        "2021. This map uses the Ministry of Rural Rehabilitation and Development's provincial "
        "profiles from about 2006-07, which give the share of each province's people living in "
        "villages where most people speak each language. Everyone in a village is counted under "
        "that language, so minorities inside mixed villages vanish, and Hazaragi is drawn as "
        "Dari because the profiles do not name it. The shares are applied to the statistics "
        "authority's population for 2025-26. Kabul's figure describes its villages, so Kabul "
        "city has none, and Herat's gives one figure for Dari and Pashto together. Both are "
        "drawn 92% Dari and 8% Pashto, the split that brings the whole map to the Asia "
        "Foundation's 2006 national shares of first languages. It is a national figure, not a "
        "measure of either place. The profiles never name the Pamiri languages of Badakhshan, "
        "the Kyrgyz of the Wakhan, Parachi, Gawar-Bati or Brahui. These are drawn from "
        "published speaker estimates, mostly Ethnologue's, placed in their home valleys and "
        "taken out of the people each profile leaves undescribed (Brahui out of Balochi and "
        "Pashto). They are estimates of speakers, not counts. Takhar and Kunduz are left empty because their profiles give "
        "no usable figure, and the 1.5 million nomadic Kuchis are not on the map."),
)
