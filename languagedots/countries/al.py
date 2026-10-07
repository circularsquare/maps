# Albania. Census 2011 mother tongue by qark (sources/al_census.py), on religiondots' Kontur hexes
# tagged with the 373 administrative units of 2011 (sources/al_geo.py).
import numpy as np

from _shared import *  # noqa: F401,F403

PLACE = GEO / "al" / "al_hexes.gpkg"


def _norm():
    return pd.read_csv(NORM / "al.csv", dtype={"geo_id": str}, keep_default_na=False)


def _counts():
    import al2011
    df = _norm()
    df = df[df["geo_level"] == "qark"].copy()
    if df["geo_id"].nunique() != 12:
        raise SystemExit("al: expected 12 qarqe in al.csv; re-run sources/al_census.py")
    df["node"] = df["source_category"].map(al2011.resolve)
    df = df[df["node"].notna() & (df["count"].astype(int) > 0)]
    df["count"] = df["count"].astype(int)
    df = df.rename(columns={"geo_id": "unit"})
    return by_unit(df)


class _AlWeighter:
    """Albanian, Greek and Macedonian go where the census counted them, unit by unit.

    The census table is by qark; INSTAT also publishes those three languages for each of the 373
    administrative units of 2011 (sources/al_census.py rebuilds the exact counts and checks they
    sum to the qark figures). So inside a qark, each unit's hexes get that unit's count of the
    language, spread over them by Kontur population. Greek lands in Dropull, Finiq and Himare and
    not across all of Gjirokaster; Macedonian in Liqenas (Pustec) and not across all of Korce.
    The counts drawn are the qark table's either way. Every other language is placed by
    population within its qark: nothing finer is published for them."""

    def __init__(self, place):
        import al2011
        self.pop = place["pop"].to_numpy(dtype=float)
        au = place["au"].astype(str).to_numpy()
        df = _norm()
        df = df[df["geo_level"] == "au"]
        node_of = {lab: al2011.resolve(lab) for lab in df["source_category"].unique()}
        au_pop = pd.Series(self.pop).groupby(au).sum()
        au_n = pd.Series(1.0, index=range(len(au))).groupby(au).sum()
        self.w = {}
        for lab, g in df.groupby("source_category"):
            cnt = g.set_index("geo_id")["count"].astype(float)
            c = cnt.reindex(au).to_numpy()
            c = np.nan_to_num(c)
            p = au_pop.reindex(au).to_numpy()
            n = au_n.reindex(au).to_numpy()
            # a unit whose hexes hold no Kontur population spreads its count evenly over them
            self.w[node_of[lab]] = np.where(p > 0, c * self.pop / np.where(p > 0, p, 1), c / n)
        self.n = {"unit": 0, "pop": 0, "none": 0}

    def weights(self, node, idx, count, plain=False):
        if node in self.w:
            w = self.w[node][idx]
            if w.sum() > 0:
                self.n["unit"] += 1
                return w
        p = self.pop[idx]
        if p.sum() > 0:
            self.n["pop"] += 1
            return p
        self.n["none"] += 1
        return None

    def summary(self):
        return (f"{self.n['unit']} (qark, language) rows placed by the 2011 administrative-unit "
                f"counts, {self.n['pop']} on Kontur population, {self.n['none']} on equal shares")


ENTRY = dict(
    name="Albania",
    source="Census of Population and Housing 2011, table 1.1.14 by prefecture, with INSTAT's "
           "administrative-unit layers for Albanian, Greek and Macedonian (INSTAT)",
    how="census, 2011, mother tongue",
    parts=[dict(covers="Everyone", source="2011 census, mother tongue", rest=True)],
    grain="12 qarqe, 233,000 people on average; Albanian, Greek and Macedonian placed by 373 "
          "administrative units",
    gap="3,843 people, 0.14%, whose answer was invalid or undetermined",
    view=[19.2, 39.6, 21.1, 42.7],
    counts=_counts,
    mappings=["al2011"],
    place=PLACE,
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=lambda place: _AlWeighter(place),
    note_public=(
        "These are 2011 figures. The 2023 census asked which language people usually speak at "
        "home but published only Albanian, other and mixed, with 21% of answers missing, so it "
        "names no minority language. The 2011 census was disputed: the main Greek minority "
        "organisation called for a boycott, and the Council of Europe judged its minority "
        "figures unreliable, so the minority languages are probably undercounted. Greek and "
        "Macedonian are placed by the census's counts for the 373 communes of 2011; the other "
        "languages are spread across their prefecture."),
)
