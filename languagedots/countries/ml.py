# Mali. RGPH5 2022, mother tongue by région (sources/ml_rgph.py: annex A06's shares x its
# populations, the foreign languages split by annex A03), on religiondots' Kontur hexes for the
# same 20 régions. Inside a région, each language's dots go to its old (2009) cercles by CLEAR
# Global's shares (sources/ml_place.py, sources/clear_place.py), a placement weight only.
# Coverage gap (2026-10-19): the 941,335 people in areas the census could not enumerate (INSTAT's
# GRID3 estimate, Tableau 1.2) are added as `modelled` rows at their région's mother-tongue mix,
# aged 3 and over, on the région's rural hexes (sources/ml_uncounted.py).
from _shared import *  # noqa: F401,F403
import numpy as np


def _load(name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, ROOT / "sources" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _uncounted():
    """The people in areas not enumerated, by (unit, node): sources/ml_uncounted.py's csv."""
    import ml2022
    df = pd.read_csv(NORM / "ml_uncounted.csv")
    df["node"] = df["label"].map(ml2022.resolve)
    unmapped = sorted(set(df.loc[df["node"].isna(), "label"]))
    if unmapped:
        raise SystemExit(f"ml: uncounted labels with no node: {unmapped}")
    return df.groupby(["unit", "node"], as_index=False)["count"].sum()


class _MlWeighter:
    """CLEAR's cercle weights, with the not-enumerated rows kept to rural hexes: those left once
    the région's densest hexes holding its urban population (Tableau 2.5) are taken out. A
    (région, language) row of not-enumerated people is recognised by its count, as its tier
    (modelled) is nobody else's in Mali except the not-enumerated rows."""

    def __init__(self, inner, place):
        self.inner = inner
        unit = place["unit"].astype(str).to_numpy()
        pop = place["pop"].to_numpy(dtype=float)
        mod = _load("ml_uncounted")
        self.rural = np.ones(len(pop), dtype=bool)
        for u, (urb, tot) in mod.URBAN.items():
            i = np.where(unit == u)[0]
            if not len(i):
                raise SystemExit(f"ml: no hexes for {u}")
            if urb <= 0:
                continue
            o = i[np.argsort(-pop[i], kind="stable")]
            cum = np.cumsum(pop[o])
            k = int(np.searchsorted(cum, urb / tot * cum[-1])) + 1
            self.rural[o[:k]] = False
        self.unit = unit
        u = _uncounted()
        self.unc = {(a, b): float(c) for a, b, c in zip(u["unit"], u["node"], u["count"])}
        self.n_rural = 0

    def weights(self, node, idx, count, plain=False):
        w = self.inner.weights(node, idx, count, plain)
        c = self.unc.get((self.unit[idx[0]], node))
        if w is not None and c is not None and abs(count - c) < 0.5:
            r = np.asarray(w, dtype=float) * self.rural[idx]
            if r.sum() > 0:
                self.n_rural += 1
                return r
        return w

    def summary(self):
        return self.inner.summary() + f"; {self.n_rural} not-enumerated rows on rural hexes"

    def __getattr__(self, name):
        return getattr(self.inner, name)


def _weight(place):
    if "zone" not in place.columns:
        raise SystemExit("data/geo/ml/ml_hexes.gpkg has no `zone` column: run sources/ml_place.py")
    shares = pd.read_csv(NORM / "ml_clear.csv")
    inner = _load("clear_place").ClearWeighter(place, shares, _load("ml_place").node_codes(),
                                               label="CLEAR Global's cercle")
    return _MlWeighter(inner, place)


def _counts():
    import ml2022
    df = pd.read_csv(NORM / "ml.csv")
    df = df[(df["geo_level"] == "region") & (df["count"] > 0)].copy()
    df["node"] = df["source_category"].map(ml2022.resolve)
    unmapped = sorted(set(df.loc[df["node"].isna(), "source_category"]))
    if unmapped:
        raise SystemExit(f"ml: categories with no node: {unmapped}")
    df = df.rename(columns={"geo_id": "unit"})
    if df["unit"].nunique() != 20:
        raise SystemExit(f"ml: {df['unit'].nunique()} régions, expected 20")
    df = df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()
    u = _uncounted().assign(tier="modelled")
    return pd.concat([df, u], ignore_index=True)


UNCOUNTED_3PLUS = int(round(pd.read_csv(NORM / "ml_uncounted.csv")["count"].sum()))

ENTRY = dict(
    name="Mali",
    source=("Fifth General Census of Population and Housing (RGPH5) 2022, Caractéristiques "
            "culturelles de la population, annexes A03 and A06 (INSTAT); the population of areas "
            "not enumerated, Tableau 1.2 of the state and structure volume (INSTAT with GRID3); "
            "placement inside régions from CLEAR Global's cercle shares (IPUMS sample of the "
            "2009 census)"),
    how="census, 2022, mother tongue, aged 3 and over; the people in areas the census could "
        "not reach modelled at their region's mix",
    parts=[
        dict(covers="Foreign languages",
             source="2022 census, one figure per region, split in the national proportions",
             people=20_856),
        dict(covers="Areas not enumerated",
             source="INSTAT's estimate of 941,335 people in 12 regions (GRID3, from building "
                    "footprints), aged 3 and over, at each region's mother-tongue mix",
             people=UNCOUNTED_3PLUS),
        dict(covers="Everyone else", source="2022 census, mother tongue, aged 3 and over",
             rest=True),
    ],
    grain=("20 regions, 957,000 people on average; in twelve of them, placed by cercle "
           "inside"),
    gap=("about 2.2 million children under three (not asked), 88,000 under-threes in areas "
         "not enumerated, and 106,567 in collective households or homeless"),
    view=[-12.3, 10.1, 4.3, 25.0],
    counts=_counts,
    mappings=["ml2022"],
    place=GEO / "ml" / "ml_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=_weight,
    note_public=(
        "The census asked each person's mother tongue, the first language they learnt as a "
        "child. Bambara is the mother tongue of half the people counted, and more speak it as "
        "a second language: 57.8% named it as the language they speak most, a separate "
        "question not drawn here. "
        "About 941,000 people lived in areas the census could not reach because of insecurity, "
        "most of them in Tombouctou, Ségou, Douentza and Ménaka. INSTAT estimated their number "
        "from satellite counts of buildings; nobody asked their language. Those aged 3 and over "
        "(about 853,000) are drawn at the mother-tongue mix of the rest of their region, on its "
        "rural land, so the dots are an estimate and the true mix there may differ. The 0.4% "
        "who gave no answer are spread over the "
        "languages of their own region, as the census's percentages do. In Ménaka, 10% have a "
        "language of Mali that the census does not name; it is drawn as other African "
        "languages. Inside twelve regions, each language's dots are spread across the old "
        "cercles using CLEAR Global's figures from a sample of the 2009 census."),
)
