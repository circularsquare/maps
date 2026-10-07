# Montenegro. Popis 2023 mother tongue by municipality (sources/me_census.py), on Kontur hexes keyed
# to the 25 municipalities (OSM polygons, so Tuzi and Zeta are their own) and, inside them, to the
# census's 1,462 settlements, whose own mother-tongue counts place the dots (sources/me_geo.py).
# The record is sources/me.md.
import numpy as np

from _shared import *  # noqa: F401,F403

PLACE = GEO / "me" / "me_hexes.gpkg"
SETTLEMENTS = GEO / "me" / "me_settlements.csv"
N_UNITS = 25


def _counts():
    import me2023
    df = pd.read_csv(NORM / "me.csv")
    df = df[df["geo_level"] == "municipality"].copy()
    if df["geo_id"].nunique() != N_UNITS:
        raise SystemExit("me: expected 25 municipalities in me.csv; re-run sources/me_census.py")
    df["unit"] = df["geo_id"]
    df["node"] = df["source_category"].map(me2023.resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    return by_unit(df)


class _MeWeighter:
    """Inside a municipality, each language follows the settlements where the census counted it.

    MONSTAT publishes mother tongue by settlement as well as by municipality, with cells under 10
    (and some others) printed as `z`. A hex's weight for a language is its Kontur population times
    its settlement's share of that language. Inside a settlement whose total is published, the
    people hidden by `z` are split evenly over its `z` cells, for placement only; a settlement
    whose total is itself `z` takes its municipality's shares. The counts drawn are the municipal
    table's either way. A language with no weight anywhere in the unit falls back to population."""

    def __init__(self, place):
        import me2023
        self.pop = place["pop"].to_numpy(dtype=float)
        st = pd.read_csv(SETTLEMENTS, dtype={"code": str}).set_index("code")
        cats = [c for c in me2023.NAMES if c in st.columns]
        cells = st[cats].astype(float)
        hidden = (st["Ukupno"] - cells.fillna(0).sum(axis=1)).clip(lower=0)
        nz = cells.isna().sum(axis=1).replace(0, 1)
        filled = cells.apply(lambda col: col.fillna(hidden / nz))
        share = filled.div(st["Ukupno"].where(st["Ukupno"] > 0), axis=0)

        norm = pd.read_csv(NORM / "me.csv")
        m = norm[norm["geo_level"] == "municipality"].pivot_table(
            index="geo_id", columns="source_category", values="count", aggfunc="sum").fillna(0)
        mshare = m[[c for c in cats if c in m.columns]].div(m["Ukupno"], axis=0)
        fallback = mshare.reindex(st["unit"]).reindex(columns=cats).fillna(0).to_numpy()
        sh = share.to_numpy()
        sh = np.where(np.isnan(sh), fallback, sh)
        share = pd.DataFrame(sh, index=st.index, columns=cats)

        code = place["settlement"].astype(str)
        self.w = {}
        for node in set(me2023.NAMES[c] for c in cats):
            cols = [c for c in cats if me2023.NAMES[c] == node]
            s = code.map(share[cols].sum(axis=1)).fillna(0).to_numpy(dtype=float)
            self.w[node] = self.pop * s
        self.n = {"settlement": 0, "pop": 0, "none": 0}

    def weights(self, node, idx, count, plain=False):
        if node in self.w:
            w = self.w[node][idx]
            if w.sum() > 0:
                self.n["settlement"] += 1
                return w
        p = self.pop[idx]
        if p.sum() > 0:
            self.n["pop"] += 1
            return p
        self.n["none"] += 1
        return None

    def summary(self):
        return (f"{self.n['settlement']} (municipality, language) rows placed by the settlements' "
                f"own mother-tongue counts, {self.n['pop']} on Kontur population, "
                f"{self.n['none']} on equal shares")


ENTRY = dict(
    name="Montenegro",
    source="Census of Population, Households and Dwellings 2023, Tabela 3, mother tongue by "
           "municipality (open data portal), with mother tongue by settlement for placement "
           "(MONSTAT)",
    how="census, 2023, mother tongue",
    parts=[dict(covers="Everyone", source="2023 census, mother tongue", rest=True)],
    grain="25 municipalities, 25,000 people on average; placed by 1,462 settlements",
    gap="11,401 people (1.8%) who did not declare a mother tongue or sit in cells withheld as "
        "too small",
    view=[18.4, 41.8, 20.4, 43.6],
    counts=_counts,
    mappings=["me2023"],
    place=PLACE,
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=lambda place: _MeWeighter(place),
    note_public=(
        "Serbian, Montenegrin, Bosnian and Croatian are one language with several standard "
        "names, and in Montenegro which name people give mostly follows how they declare their "
        "nationality. The census printed each name it was given, compound answers such as "
        "Montenegrin-Serbian included; 1,408 people who answered only \"mother tongue\" are "
        "drawn as South Slavic with no language named. The totals are per municipality, and "
        "inside each one the dots follow the census's own counts by settlement. Tuzi, split "
        "from Podgorica in 2018 and 60% Albanian, is drawn as its own municipality."),
)
