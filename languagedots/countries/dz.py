# Algeria. Arab Barometer VI-VII (2020-2022) ethnic group read as language, by wilaya, on the RGPH
# 2008 wilaya populations (sources/dz_survey.py); every row `modelled`. On religiondots' Kontur
# hexes for the 48 wilayas. Inside a wilaya, Berber dots lean to where the 1966 census found
# Berber speakers, Arabic dots away from there. The record is sources/dz.md.
from _shared import *  # noqa: F401,F403
import numpy as np

UNITS = 48
RGPH_2008 = 34_080_030
BERBER = "afroasiatic.berber"
ARABIC = "afroasiatic.algerian_arabic"


def _counts():
    import dz2022
    df = pd.read_csv(NORM / "dz.csv", dtype={"geo_id": str})
    if df["geo_id"].nunique() != UNITS:
        raise SystemExit(f"dz.csv: {df['geo_id'].nunique()} wilayas, expected {UNITS} -- "
                         "re-run sources/dz_survey.py")
    if int(df["count"].sum()) != RGPH_2008:
        raise SystemExit(f"dz.csv sums to {int(df['count'].sum()):,}, not {RGPH_2008:,}")
    df["node"] = df["source_category"].map(dz2022.resolve)
    missing = sorted(set(df.loc[df["node"].isna(), "source_category"]))
    if missing:
        raise SystemExit(f"dz.csv answers with no node: {missing}")
    lut = pd.read_csv(RD_GEO / "dz" / "dz_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    if df["unit"].isna().any():
        raise SystemExit("dz.csv wilayas missing from religiondots' dz_lookup.csv")
    df = df[df["count"] > 0]
    out = by_unit(df)
    out["tier"] = "modelled"
    return out


class _Dz1966Weighter:
    """Inside a wilaya, Berber dots go to each hex in proportion to Kontur population times p,
    and Algerian Arabic dots in proportion to population times (1 - p), where p is the 1966
    census's Berber-speaking share at that spot (sources/dz_survey.py's surface over the daira
    figures Nesson 1994 quotes) scaled so the wilaya's population times p adds up to the
    wilaya's drawn Berber count, and capped at 1. So in Bouira the Berber dots go north to the
    Kabyle side and the Arabic dots south to Sour El Ghozlane; in Batna, Berber to the Aures
    and Arabic to Barika. French goes on population. A placement weight only: every wilaya's
    counts are the survey's shares times RGPH 2008 either way."""

    def __init__(self, place):
        sys.path.insert(0, str(ROOT / "sources"))
        from dz_survey import surface
        self.pop = place["pop"].to_numpy(dtype=float)
        c = place.geometry.to_crs(3857).centroid.to_crs(4326)
        self.s = surface(c.y.to_numpy(), c.x.to_numpy()) / 100
        cnt = _counts()
        ber = cnt[cnt["node"].str.startswith(BERBER)].groupby("unit")["count"].sum()
        tot = cnt.groupby("unit")["count"].sum()
        self.share = (ber / tot).reindex(tot.index).fillna(0).to_dict()
        self.unit = place["unit"].astype(str).to_numpy()
        self.p = {}
        self.n = {"berber": 0, "arabic": 0, "pop": 0, "none": 0}

    def _profile(self, idx):
        u = self.unit[idx[0]]
        if u not in self.p:
            pop, s, b = self.pop[idx], self.s[idx], self.share.get(u, 0.0)
            target = b * pop.sum()
            lo, hi = 0.0, 1e4
            for _ in range(80):
                mid = (lo + hi) / 2
                if (pop * np.minimum(1.0, mid * s)).sum() < target:
                    lo = mid
                else:
                    hi = mid
            self.p[u] = np.minimum(1.0, hi * s)
        return self.p[u]

    def weights(self, node, idx, count, plain=False):
        pop = self.pop[idx]
        if pop.sum() <= 0:
            self.n["none"] += 1
            return None
        if node.startswith(BERBER):
            w = pop * self._profile(idx)
            key = "berber"
        elif node == ARABIC:
            w = pop * (1 - self._profile(idx))
            key = "arabic"
        else:
            self.n["pop"] += 1
            return pop
        if w.sum() <= 0:
            self.n["pop"] += 1
            return pop
        self.n[key] += 1
        return w

    def summary(self):
        return (f"{self.n['berber']:,} Berber and {self.n['arabic']:,} Algerian Arabic "
                f"(wilaya, language) rows placed by the 1966 surface x population, "
                f"{self.n['pop']:,} on population, {self.n['none']:,} on equal shares")


def _weight(place):
    if "pop" not in place.columns:
        raise SystemExit("dz_hexes.gpkg has no `pop` column")
    return _Dz1966Weighter(place)


ENTRY = dict(
    name="Algeria",
    source=("Arab Barometer waves VI and VII (2020-2022, Arab Barometer, Princeton University), "
            "ethnic group by wilaya; French from Arab Barometer II-IV and Afrobarometer R6 "
            "(2011-2016); each wilaya's population in the 2008 census (Office National des "
            "Statistiques, RGPH 2008)"),
    how=("survey, 2020-2022, ethnic group read as language: Arab as Algerian Arabic, Amazigh "
         "as the wilaya's Berber language; 3,366 adults"),
    parts=[
        dict(covers="French", source="Arab Barometer II-IV and Afrobarometer 2011-16, first "
             "language", nodes=["indoeuropean.romance.french"]),
        dict(covers="Everyone else",
             source="Arab Barometer 2020-22, Arab or Amazigh, 3,366 adults, on the 2008 census",
             rest=True),
    ],
    grain="48 wilayas, 710,000 people on average",
    gap=("the Sahrawi refugee camps near Tindouf, which the 2008 census did not count (173,600 "
         "people by the UN agencies' 2018 figure); foreigners"),
    view=[-8.7, 18.9, 12.0, 37.2],
    counts=_counts,
    mappings=["dz2022"],
    place=RD_GEO / "dz" / "dz_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=_weight,
    note_public=(
        "No Algerian census since 1966 has asked about language. The 1966 census found 19% of "
        "Algerians had a Berber mother tongue. This map uses the Arab Barometer survey "
        "instead, which asked 3,366 adults in 2020 to 2022 whether they are Arab or Amazigh "
        "(Berber), and draws each answer as a language: Arab as Algerian Arabic, Amazigh as "
        "the Berber language of the wilaya where the person lives (Kabyle, Chaoui, Mozabite "
        "or Tamahaq), or as Berber with no language named where several are spoken. That "
        "gives 23% Berber. Some Algerians who call themselves Amazigh speak Arabic at home, so "
        "Berber is probably overdrawn outside its heartlands. Surveys that asked for a first "
        "language found far fewer Berber speakers (6 to 11%), but every one of their "
        "interviews was held in Arabic. French is drawn for the 1% who named it as their first "
        "language. Inside each wilaya, Berber speakers are placed where the 1966 census found "
        "them. The people are the 2008 census count, the last by wilaya."),
)
