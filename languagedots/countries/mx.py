# Mexico. Censo 2020, indigenous languages per municipio from INEGI's cube (sources/mx_censo.py);
# everyone else aged 3+ drawn as Spanish (spec §3.5). Placed on Kontur hexes per municipio, the
# indigenous-language dots on ITER's speakers per locality (sources/mx_geo.py).
from _shared import *  # noqa: F401,F403
import numpy as np

if str(ROOT / "sources") not in sys.path:
    sys.path.insert(0, str(ROOT / "sources"))

SPANISH = "indoeuropean.romance.spanish"
# 2026-10-05, session edd42a8c-latn (sources/mx.md, "Immigrant and settler languages"): people
# who speak no indigenous language are Spanish except the foreign-born (sources/mx_origin.py,
# by country of birth, through sources/latam_immig.py) and the Mennonite colonies and Chipilo
# (mx_settlers.csv). In the colony municipios the Canadian-, Belizean-, Bolivian- and
# Paraguayan-born are taken to be colony Mennonites, already inside the colony count.
COLONY_ORIGINS = {"CA", "BZ", "BO", "PY"}


def _immigrant_rows():
    import latam_immig
    from mx_origin import COLONY_MUN
    org = pd.read_csv(NORM / "mx_origin.csv", dtype={"geo_id": str}, keep_default_na=False)
    org = org[~org["origin"].isin(["US_U18", "XX"])]    # US-born children, unstated: Spanish
    org = org[~(org["geo_id"].isin(COLONY_MUN) & org["origin"].isin(COLONY_ORIGINS))]
    rows = []
    for geo, g in org.groupby("geo_id"):
        for node, n in latam_immig.spread(dict(zip(g["origin"], g["count"])), "mx").items():
            if node != SPANISH:
                rows.append((geo, node, n))
    st = pd.read_csv(NORM / "mx_settlers.csv", dtype={"geo_id": str, "loc": str})
    rows += list(st.groupby(["geo_id", "node"])["count"].sum().reset_index().itertuples(
        index=False, name=None))
    return pd.DataFrame(rows, columns=["geo_id", "node", "count"]).assign(tier="derived")


def _counts():
    import mx2020
    df = pd.read_csv(NORM / "mx.csv", dtype={"geo_id": str})
    df["node"] = df["source_category"].map(mx2020.resolve)
    df["tier"] = "measured"
    st = pd.read_csv(NORM / "mx_status.csv", dtype={"geo_id": str})
    # Spanish: everyone aged 3+ who said they do not speak an indigenous language, less the
    # immigrant and settler languages. The 114,571 who did not say, and the 6.0 million
    # children under 3 (not asked), are not drawn.
    imm = _immigrant_rows()
    less = imm.groupby("geo_id")["count"].sum()
    es_n = st.set_index("geo_id")["no_hli"].astype(float).sub(less, fill_value=0)
    if (es_n < -1e-6).any():
        raise SystemExit(f"mx: immigrant languages exceed Spanish in {list(es_n[es_n < 0].index[:5])}")
    es = pd.DataFrame({"geo_id": es_n.index, "node": SPANISH, "count": es_n.clip(lower=0).values,
                       "tier": "derived"})
    out = pd.concat([df[["geo_id", "node", "count", "tier"]], es, imm], ignore_index=True)
    out = out[out["count"] > 0].rename(columns={"geo_id": "unit"})
    return out.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()


class _MxWeighter:
    """Inside a municipio, indigenous-language dots go to the hexes holding its speakers (`spk`,
    from ITER's localities, sources/mx_geo.py) and Spanish dots to the rest: Kontur's people,
    scaled to the municipio's census population aged 3+, less those speakers. Every language in
    one municipio shares the speakers' weight, since nothing published says which village speaks
    which. A placement weight only: every municipio's totals are the census's either way."""

    def __init__(self, place):
        import mx2020
        self.pop = place["pop"].to_numpy(dtype=float)
        self.spk = place["spk"].to_numpy(dtype=float)
        unit = place["unit"].astype(str).to_numpy()
        st = pd.read_csv(NORM / "mx_status.csv", dtype={"geo_id": str}).set_index("geo_id")
        k_sum = pd.Series(self.pop).groupby(unit).transform("sum").to_numpy()
        p3 = pd.Series(unit).map(st["p3"]).fillna(0).to_numpy(dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            scaled = np.where(k_sum > 0, self.pop * p3 / k_sum, 0)
        self.rest = (scaled - self.spk).clip(min=0)
        # indigenous languages go on the speakers' hexes; immigrant languages on the rest, like
        # Spanish; the settler languages on the hex nearest each of their villages in its
        # municipio (mx_settlers.csv: ITER's coordinates)
        labels = pd.read_csv(NORM / "mx.csv", usecols=["source_category"])["source_category"]
        self.indigenous = {mx2020.resolve(x) for x in labels.unique()}
        self.settled = self._settled(place, unit)
        self.n = {"speakers": 0, "rest": 0, "settlers": 0, "fallback": 0}

    @staticmethod
    def _settled(place, unit):
        import geopandas as gpd
        s = pd.read_csv(NORM / "mx_settlers.csv", dtype={"geo_id": str, "loc": str})
        pts = gpd.GeoSeries(gpd.points_from_xy(s["lon"], s["lat"]), crs="EPSG:4326")
        pts = pts.to_crs(place.crs) if place.crs is not None else pts
        import warnings
        with warnings.catch_warnings():    # 400 m hexes: a lon/lat centroid is fine
            warnings.simplefilter("ignore", UserWarning)
            cen = place.geometry.centroid
        cx, cy = cen.x.to_numpy(), cen.y.to_numpy()
        out = {}
        for i, r in enumerate(s.itertuples()):
            idx = np.flatnonzero(unit == r.geo_id)
            if not len(idx):
                raise SystemExit(f"mx: settler village {r.name} ({r.geo_id}) has no hexes")
            d = (cx[idx] - pts.iloc[i].x) ** 2 + (cy[idx] - pts.iloc[i].y) ** 2
            w = out.setdefault(r.node, np.zeros(len(place)))
            w[idx[np.argmin(d)]] += r.count
        return out

    def weights(self, node, idx, count, plain=False):
        if node in self.settled and self.settled[node][idx].sum() > 0:
            self.n["settlers"] += 1
            return self.settled[node][idx]
        es = node not in self.indigenous
        w = (self.rest if es else self.spk)[idx]
        if w.sum() > 0:
            self.n["rest" if es else "speakers"] += 1
            return w
        self.n["fallback"] += 1
        p = self.pop[idx]
        return p if p.sum() > 0 else None

    def summary(self):
        return (f"{self.n['speakers']:,} indigenous-language rows placed on ITER's speakers, "
                f"{self.n['rest']:,} Spanish and immigrant rows on the rest, "
                f"{self.n['settlers']:,} settler rows on their villages, {self.n['fallback']:,} "
                f"on plain population")


def _weight(place):
    return _MxWeighter(place)


ENTRY = dict(
    name="Mexico",
    source="Censo de Población y Vivienda 2020 (INEGI): the Población de 3 años y más cube for "
           "languages and country of birth per municipio, ITER for speakers per locality and "
           "for the Mennonite colony villages; Die Mennonitische Post's 2022 colony census, "
           "Reuters (Campeche) and El Siglo de Durango for the colonies; Ethnologue for Chipilo; "
           "France's TeO2 survey for how many immigrants keep their language",
    how="census, 2020, indigenous language spoken, aged 3 and over; people born abroad drawn by "
        "their birth country's languages (US-born children as Spanish), less the share France's "
        "TeO2 survey finds speaking only the host language; Plautdietsch in the Mennonite "
        "colonies and Venetian in Chipilo from published counts; everyone else drawn as Spanish",
    parts=[
        dict(covers="Indigenous languages",
             source="2020 census, indigenous language spoken, aged 3 and over",
             people=7_364_645),
        dict(covers="People born abroad, languages other than Spanish",
             source="2020 census, country of birth, drawn on that country's languages; about a "
                    "quarter moved to Spanish by France's TeO2 survey",
             people=245_061),
        dict(covers="Mennonite colonies and Chipilo",
             source="colony village populations (2020 census, Die Mennonitische Post 2022), "
                    "Ethnologue for Chipilo",
             people=56_614),
        dict(covers="Everyone else", source="2020 census, aged 3 and over, drawn as Spanish",
             rest=True),
    ],
    grain="2,469 municipios, 49,000 people aged 3 and over on average; indigenous languages placed "
          "by the census's count of speakers in each locality",
    gap="children under 3, 6.0 million (4.8%), whom the census does not ask, and 114,571 people "
        "who did not say whether they speak an indigenous language",
    view=[-117.2, 14.4, -86.6, 32.8],
    counts=_counts,
    mappings=["mx2020"],
    place=GEO / "mx" / "mx_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=_weight,
    note_public=(
        "The 2020 census asked everyone aged 3 or over whether they speak an indigenous "
        "language and which one, but not about Spanish. People who said no are drawn as "
        "Spanish speakers, except two groups. People born abroad are drawn by the main "
        "languages of their birth country, with about a quarter moved to Spanish because "
        "France's TeO2 survey finds that share of immigrants speaking only the host language "
        "at home; children aged 3 to 17 born in the United States are drawn as Spanish "
        "speakers, since most are children of Mexican families who came back. The Mennonite "
        "colonies are drawn as Plautdietsch (Low German) speakers, about 54,000 people aged 3 "
        "and over: the census does not record them, so the count is the colony villages' "
        "population (in Campeche and Durango, published estimates of the colonies' size). "
        "Chipilo, in Puebla, is drawn with 2,500 speakers of Venetian, Ethnologue's figure. "
        "Most speakers of an indigenous language also speak Spanish (87% of them). The census "
        "names languages by INALI's 68 groupings, so Zapoteco, "
        "Mixteco, Chinanteco, Mixe, Otomí and Náhuatl are each drawn as one language although "
        "each is a cluster of varieties that are not all mutually intelligible. Languages are "
        "published per municipio; inside one, indigenous-language dots are placed where the "
        "census counted speakers, locality by locality, but which village speaks which language "
        "is not published, so in a municipio with two languages their dots mix."),
)
