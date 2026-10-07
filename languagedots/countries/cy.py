# Cyprus. Census 2021, native language by district (CYSTAT-DB 1891616E, sources/cy_census.py),
# on religiondots' Kontur hexes for 396 municipalities and communities. Counts are the five
# districts'; inside a district each language is placed by the citizenship groups of each
# community (1891613E x 1891213E, data/normalized/cy_place.csv). The north is the 2011
# northern census by village, all drawn as Turkish, tier derived (ask 015; sources/cy_north.py).
# The record is sources/cy.md.
from _shared import *  # noqa: F401,F403
import numpy as np


def _counts():
    import cy2021
    df = pd.read_csv(NORM / "cy.csv", dtype={"geo_id": str})
    df = df[df["geo_level"] == "district"]
    if sorted(df["geo_id"].unique()) != ["1", "3", "4", "5", "6"]:
        raise SystemExit(f"cy.csv: districts {sorted(df['geo_id'].unique())}")
    df["node"] = df["source_category"].map(cy2021.resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df["unit"] = df["geo_id"]
    df["tier"] = "measured"
    south = df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()
    # The north: the 2011 northern census by village, no language question, everyone drawn
    # as Turkish (Anita, ask 015, 2026-10-05). sources/cy_north.py.
    n = pd.read_csv(NORM / "cy_north.csv", dtype={"geo_id": str})
    if not n["geo_id"].str.startswith("N").all() or len(n) != 192:
        raise SystemExit(f"cy_north.csv: {len(n)} units, expected 192 `N...` ids")
    north = pd.DataFrame({"unit": n["geo_id"], "node": cy2021.resolve("Turkish"),
                          "tier": "derived", "count": n["count"]})
    return pd.concat([south, north], ignore_index=True)


class _CyCitizenWeighter:
    """Inside a district, a language's dots go to each community in proportion to the speakers
    it would hold if each of its citizenship groups (Cypriot, other EU, non-EU, not stated)
    spoke as that group does across the district; inside a community, by the hexes' Kontur
    population. A placement weight only: every district's counts are the census's either way.
    """

    def __init__(self, place):
        import cy2021
        self.pop = place["pop"].to_numpy(dtype=float)
        if "comm" not in place.columns:
            raise SystemExit("cy: placement layer lacks `comm` (set by _place_unit)")
        self.comm = place["comm"].astype(str).to_numpy()
        # hex share of its community's Kontur population (equal shares where that is zero)
        cpop = pd.Series(self.pop).groupby(self.comm).transform("sum").to_numpy()
        ncount = pd.Series(1.0, index=range(len(self.comm))).groupby(self.comm) \
                   .transform("sum").to_numpy()
        self.share = np.where(cpop > 0, self.pop / np.where(cpop > 0, cpop, 1), 1 / ncount)
        w = pd.read_csv(NORM / "cy_place.csv", dtype={"unit": str, "district": str})
        w["node"] = w["source_category"].map(cy2021.resolve)
        w = w[w["node"].notna()]
        self.E = {n: g.groupby("unit")["weight"].sum().to_dict()
                  for n, g in w.groupby("node")}
        missing = set(w["unit"]) - set(self.comm)
        if missing:
            raise SystemExit(f"cy_place.csv has {len(missing)} communities not in the hexes")
        self.n = {"citizens": 0, "pop": 0, "none": 0}

    def weights(self, node, idx, count, plain=False):
        e = self.E.get(node, {})
        v = np.array([e.get(c, 0.0) for c in self.comm[idx]]) * self.share[idx]
        if v.sum() > 0:
            self.n["citizens"] += 1
            return v
        p = self.pop[idx]
        self.n["pop" if p.sum() > 0 else "none"] += 1
        return p if p.sum() > 0 else None

    def summary(self):
        return (f"{self.n['citizens']:,} (district, language) rows placed by community "
                f"citizenship groups, {self.n['pop']:,} on population, {self.n['none']:,} on "
                f"equal shares")


def _place_unit(g):
    """South: district = the LAU code's first digit. North: the village unit itself (`N...`).
    scatter.py overwrites `unit` with this, so the community code is kept in `comm` for the
    weighter first. The north has no citizenship weights, so the weighter falls back to Kontur
    population there."""
    g["comm"] = g["unit"].astype(str)
    return g["comm"].where(g["comm"].str.startswith("N"), g["comm"].str[0])


def _weight(place):
    return _CyCitizenWeighter(place)


ENTRY = dict(
    name="Cyprus",
    source="Census of Population and Housing 2021, CYSTAT-DB table 1891616E (language by "
           "district), placed with 1891613E and 1891213E (Statistical Service of Cyprus); the "
           "north from the 2011 census held there, Tablo 3 (population by mahalle)",
    how="census, 2021, native language; inside each district, placed by the citizenship "
        "groups of each municipality and community. The north: census, 2011, no language "
        "question; everyone drawn as Turkish",
    parts=[
        dict(covers="The north", source="northern census 2011, drawn as Turkish",
             people=285_778),
        dict(covers="The south", source="2021 census, native language", rest=True),
    ],
    grain="south: 5 districts, 185,000 people on average, placed by citizenship group per "
          "municipality and community; north: 192 villages and towns, 1,500 people on average",
    gap="11,111 people in the south (1.2%) whose language was not stated, and the 479 people "
        "the northern census counted at Pyla, in the buffer zone",
    # the whole island
    view=[32.15, 34.50, 34.70, 35.78],
    counts=_counts,
    mappings=["cy2021"],
    # religiondots' south hexes minus those north of the line, plus the north's (sources/cy_north.py)
    place=ROOT / "data" / "geo" / "cy" / "cy_hexes.gpkg",
    place_unit=_place_unit,
    place_weight=_weight,
    note_public=(
        "The 2021 census asked everyone's native language and publishes the answers by "
        "district, five of them, with 32 languages named. Inside each district the dots are "
        "placed using the census's count of Cypriot, other EU and non-EU citizens in every "
        "municipality and community, so a language spoken mostly by foreign residents is "
        "drawn where foreign residents live. That census covers only the area under the "
        "control of the Republic of Cyprus. The north is drawn from the 2011 census held "
        "there, which counted 286,257 usual residents by village and town quarter and asked "
        "no language question, so everyone in the north is drawn as Turkish. That is right "
        "for most people there but not for everyone: the census counted 15,213 people who "
        "held neither northern Cypriot nor Turkish citizenship, and the Maronites of Koruçam "
        "(Kormakitis) and the Greek Cypriots of the Karpas villages are drawn as Turkish too. "
        "The two censuses are ten years apart. In the south, Indian and Sri Lankan are printed "
        "as languages but name a country; they are drawn as other languages."),
)
