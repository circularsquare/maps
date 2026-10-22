# Algeria. Arab Barometer VI-VII (2020-2022) ethnic group read as language, by wilaya, on the RGPH
# 2008 wilaya populations (sources/dz_survey.py); every row `modelled`. Amazigh answers are split
# over ten Berber varieties by place (dz_survey.SPLIT, 2026-10-19), with a retention cut outside
# the Berber heartlands (2026-10-21, dz_survey.HEARTLAND). The Sahrawi camps near Tindouf,
# in no census, are added as Hassaniya from the UN agencies' 2018 figure (sources/dz_camps.py
# places them). On religiondots' Kontur hexes for the 48 wilayas plus the camps' patches. Inside a
# wilaya, Berber dots lean to where the 1966 census found Berber speakers, Arabic dots away from
# there. The record is sources/dz.md.
from _shared import *  # noqa: F401,F403
import numpy as np

UNITS = 48
RGPH_2008 = 34_080_030
BERBER = "afroasiatic.berber"
ARABIC = "afroasiatic.algerian_arabic"
HASSANIYA = "afroasiatic.hassaniya"
CAMPS_PEOPLE = 173_600          # the five Sahrawi camps, UN agencies' figure for 31 Dec 2017
CAMPS_UNIT = "DZ37"             # Tindouf wilaya
PLACE = GEO / "dz" / "dz_hexes_camps.gpkg"


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
    camps = pd.DataFrame([dict(unit=CAMPS_UNIT, node=HASSANIYA, count=CAMPS_PEOPLE)])
    out = pd.concat([out, camps], ignore_index=True)
    out["tier"] = "modelled"
    return out


class _Dz1966Weighter:
    """Inside a wilaya, Berber dots go to each hex in proportion to Kontur population times p,
    and Algerian Arabic dots in proportion to population times (1 - p), where p is the 1966
    census's Berber-speaking share at that spot (sources/dz_survey.py's surface over the daira
    figures Nesson 1994 quotes) scaled so the wilaya's population times p adds up to the
    wilaya's drawn Berber count, and capped at 1. So in Bouira the Berber dots go north to the
    Kabyle side and the Arabic dots south to Sour El Ghozlane; in Batna, Berber to the Aures
    and Arabic to Barika. French goes on population.

    Berber varieties (2026-10-19): where a wilaya draws one variety, as before. Where it draws
    two in set shares (Oran: Kabyle and Chenoua) both are spread the same way, by population x p.
    Where two varieties each hold part of the wilaya (Adrar, Tamanrasset, Ouargla; dz_survey.SPLIT
    "hex") each is kept to the hexes nearer its own Glottolog core.

    The Sahrawi camps near Tindouf (sources/dz_camps.py): Hassaniya goes only on the camps'
    hexagons, equally; every other language is kept off the hexes within 3.5 km of a camp centre
    (Kontur puts the Laayoune camp's people there, and they are not the census's).

    A placement weight only: every wilaya's counts are the survey's shares times RGPH 2008."""

    def __init__(self, place):
        sys.path.insert(0, str(ROOT / "sources"))
        from dz_survey import surface, SPLIT, hex_variety, VARIETIES
        self.pop = place["pop"].to_numpy(dtype=float)
        c = place.geometry.to_crs(3857).centroid.to_crs(4326)
        lat, lon = c.y.to_numpy(), c.x.to_numpy()
        self.s = surface(lat, lon) / 100
        self.camp = place["camp"].to_numpy(dtype=float)
        self.zone = place["camp_zone"].to_numpy(dtype=float)
        cnt = _counts()
        ber = cnt[cnt["node"].str.startswith(BERBER)].groupby("unit")["count"].sum()
        tot = cnt.groupby("unit")["count"].sum()
        self.share = (ber / tot).reindex(tot.index).fillna(0).to_dict()
        self.unit = place["unit"].astype(str).to_numpy()
        self.key_of = {v[0]: k for k, v in VARIETIES.items()}
        # the hex wilayas: which variety's core each hex is nearer
        self.var = np.full(len(place), None, dtype=object)
        for g, v in SPLIT.items():
            if isinstance(v, tuple):
                m = self.unit == g
                self.var[m] = hex_variety(lat[m], lon[m], v[1])
        self.p = {}
        self.n = {"berber": 0, "arabic": 0, "pop": 0, "none": 0, "camp": 0}

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
        keep = 1.0 - self.zone[idx]
        if node == HASSANIYA:
            w = self.camp[idx]
            if w.sum() > 0:
                self.n["camp"] += 1
                return w
            self.n["none"] += 1
            return None
        if (pop * keep).sum() <= 0:
            self.n["none"] += 1
            return None
        if node.startswith(BERBER):
            w = pop * self._profile(idx) * keep
            key = "berber"
            if self.var[idx[0]] is not None:
                mask = (self.var[idx] == self.key_of[node]).astype(float)
                w = w * mask
                if w.sum() <= 0:
                    w = pop * keep * mask
        elif node == ARABIC:
            w = pop * (1 - self._profile(idx)) * keep
            key = "arabic"
        else:
            self.n["pop"] += 1
            return pop * keep
        if w.sum() <= 0:
            self.n["pop"] += 1
            return pop * keep
        self.n[key] += 1
        return w

    def summary(self):
        return (f"{self.n['berber']:,} Berber and {self.n['arabic']:,} Algerian Arabic "
                f"(wilaya, language) rows placed by the 1966 surface x population, "
                f"{self.n['pop']:,} on population, {self.n['none']:,} on equal shares, "
                f"{self.n['camp']:,} Hassaniya rows on the Tindouf camps")


def _weight(place):
    if "pop" not in place.columns or "camp" not in place.columns:
        raise SystemExit("dz_hexes_camps.gpkg has no `pop` or `camp` column; run sources/dz_camps.py")
    return _Dz1966Weighter(place)


ENTRY = dict(
    name="Algeria",
    source=("Arab Barometer waves VI and VII (2020-2022, Arab Barometer, Princeton University), "
            "ethnic group by wilaya; French from Arab Barometer II-IV and Afrobarometer R6 "
            "(2011-2016); each wilaya's population in the 2008 census (Office National des "
            "Statistiques, RGPH 2008); the Sahrawi camps near Tindouf from the UN agencies' 2018 "
            "figure (UNHCR and WFP, Sahrawi Refugee Response Plan)"),
    how=("survey, 2020-2022, ethnic group read as language: Arab as Algerian Arabic, Amazigh "
         "as a Berber variety chosen by place (the survey does not ask which); 3,366 adults. "
         "Outside the Berber heartlands the Amazigh share is cut to halfway towards the "
         "first-language surveys, and to no more than the 1966 census found. "
         "Sahrawi camps from a UN estimate, drawn as Hassaniya"),
    parts=[
        dict(covers="French", source="Arab Barometer II-IV and Afrobarometer 2011-16, first "
             "language", nodes=["indoeuropean.romance.french"]),
        dict(covers="Berber languages",
             source="Arab Barometer 2020-22, Amazigh answers, split over ten varieties by "
             "wilaya (Glottolog places and judgement)",
             nodes=["afroasiatic.berber.kabyle", "afroasiatic.berber.chaouia",
                    "afroasiatic.berber.chenoua", "afroasiatic.berber.tumzabt",
                    "afroasiatic.berber.ouargli", "afroasiatic.berber.oued_righ",
                    "afroasiatic.berber.taznatit", "afroasiatic.berber.tidikelt",
                    "afroasiatic.berber.tamahaq", "afroasiatic.berber.tachelhit"]),
        dict(covers="Sahrawi refugee camps near Tindouf",
             source="UN agencies' 2018 figure for the five camps, drawn as Hassaniya, an equal "
             "share on each", people=CAMPS_PEOPLE),
        dict(covers="Everyone else",
             source="Arab Barometer 2020-22, Arab or Other, 3,366 adults, on the 2008 census",
             rest=True),
    ],
    grain="48 wilayas, 710,000 people on average",
    gap=("foreigners; the camps' 173,600 is the UN agencies' planning figure, which UNHCR "
         "itself holds at 90,000 pending a registration"),
    view=[-8.7, 18.9, 12.0, 37.2],
    counts=_counts,
    mappings=["dz2022"],
    place=PLACE,
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=_weight,
    note_public=(
        "No Algerian census since 1966 has asked about language. The 1966 census found 19% of "
        "Algerians had a Berber mother tongue. This map uses the Arab Barometer survey "
        "instead, which asked 3,366 adults in 2020 to 2022 whether they are Arab or Amazigh "
        "(Berber), and draws each answer as a language: Arab as Algerian Arabic, Amazigh as a "
        "Berber language. The survey did not ask which one, so it is chosen by place: Kabyle "
        "in Kabylie and among Kabyle migrants in Algiers and the other cities, Chaoui in the "
        "Aures and the east, Mozabite in Ghardaia, Tamahaq in the Tuareg south, Chenoua and "
        "its neighbours in the Dahra and the west, Taznatit in Gourara and the Sud Oranais, "
        "Ouargli and Oued Righ around Ouargla and Touggourt. Many Algerians who call themselves "
        "Amazigh speak Arabic at home, so outside Kabylie, the Aures, the Mzab and the "
        "Tuareg south the Amazigh share is cut to halfway between the survey's and the "
        "first-language surveys' (which found 6 to 11% Berber nationally, though every "
        "one of their interviews was held in Arabic), and to no more than the 1966 census "
        "found. That gives 17% Berber nationally and 22% in Algiers, from 24% and 40% "
        "before the cut. French is drawn for the 1% who named it as their "
        "first language. Inside each wilaya, Berber speakers are placed where the 1966 census "
        "found them. The people are the 2008 census count, the last by wilaya. The 173,600 "
        "Sahrawi refugees in the camps near Tindouf, whom no census counted, are added from "
        "the UN agencies' 2018 figure and drawn as Hassaniya on the five camps."),
)
