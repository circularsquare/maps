# Austria. Volkszaehlung 2001, Umgangssprache (sources/at_vz2001.py): by Gemeinde in Burgenland
# and Kaernten, by Gemeindebezirk in Wien, by Politischer Bezirk elsewhere, the finest grain
# Tabelle 5 is printed at. On religiondots' Kontur hexes for the 2001 Gemeinden (read only);
# inside a Bezirk each language's dots follow Tabelle 2's citizenship counts per Gemeinde.
# The record is sources/at.md.
from _shared import *  # noqa: F401,F403
import importlib.util

import numpy as np

UNITS = 405          # 303 Gemeinden, 79 Bezirke, 23 Wien districts
GEM_LANDS = ("1", "2")


def _src():
    spec = importlib.util.spec_from_file_location("at_vz2001", ROOT / "sources" / "at_vz2001.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _counts():
    import at2001
    df = pd.read_csv(NORM / "at.csv", dtype={"geo_id": str})
    df = df[df["geo_level"] != "land"]
    if df["geo_id"].nunique() != UNITS:
        raise SystemExit(f"at.csv: {df['geo_id'].nunique()} units, expected {UNITS}")
    df["node"] = df["source_category"].map(at2001.resolve)
    df["unit"] = df["geo_id"]
    # measured: Tabelle 5's own columns. derived: Tabelle 5's "Sonstige" split into Tabelle 14's
    # languages inside each Land (IPF on citizenship; every Land total is the census's).
    return df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()


def _place_unit(g):
    """Hexes carry religiondots' 2001 Gemeinde id (AT + 5 digits; Wien AT + 3). The counted
    unit is that Gemeinde in Burgenland and Kaernten, the Wien district, and elsewhere the
    Politischer Bezirk, which is the Gemeinde code's first three digits."""
    gem = g["unit"].astype(str)
    g["gem"] = gem                      # kept for the weighter
    land = gem.str[2]
    keep = land.isin(GEM_LANDS) | (land == "9")
    return gem.where(keep, gem.str[:5])


class _AtWeighter:
    """Inside a counted unit, dots go to the unit's Gemeinden by the Tabelle 2 count of the
    citizenship the language follows (sources/at_vz2001.py PROXY and NAMED_PROXY), 90%, and
    by population, 10%; inside a Gemeinde by the hexes' population. Where the unit is one
    Gemeinde (Burgenland, Kaernten, Wien's districts) this is plain population. A node fed by
    several labels (`other`) blends their weights by the labels' shares of it in that unit."""

    def __init__(self, place):
        import at2001
        src = _src()
        self.floor = src.FLOOR
        self.pop = place["pop"].to_numpy(dtype=float)
        self.gem = place["gem"].astype(str).to_numpy()
        prox = pd.read_csv(NORM / "at_place.csv", dtype={"geo_id": str, "unit": str})
        self.prox = prox.set_index("geo_id")
        df = pd.read_csv(NORM / "at.csv", dtype={"geo_id": str})
        df = df[(df["geo_level"] != "land") & (df["count"] > 0)]
        df["node"] = df["source_category"].map(at2001.resolve)
        self.mix = {}
        for (u, n), g in df.groupby(["geo_id", "node"]):
            by = g.groupby("source_category")["count"].sum()
            cols = []
            for lab, v in by.items():
                if lab in src.NAMED_PROXY:
                    pc = src.NAMED_PROXY[lab]
                else:
                    pc = src.proxy_for(lab)
                cols.append((pc, v / by.sum()))
            self.mix[(u, n)] = cols
        self.n = {"citizens": 0, "pop": 0, "none": 0}

    def _proxy(self, gems, pc):
        out = np.zeros(len(gems))
        for i, gid in enumerate(gems):
            if gid not in self.prox.index:
                continue
            r = self.prox.loc[gid]
            out[i] = sum(r["eu"] if p == "eu" else r[f"c{p}"] for p in pc)
        return out

    def weights(self, node, idx, count, plain=False):
        p = self.pop[idx]
        gems = self.gem[idx]
        ug = np.unique(gems)
        if len(ug) == 1:                # one Gemeinde: population
            self.n["pop" if p.sum() > 0 else "none"] += 1
            return p if p.sum() > 0 else None
        # the counted unit of these hexes
        unit = _place_unit(pd.DataFrame({"unit": [ug[0]]})).iloc[0]
        gpop = pd.Series(p).groupby(gems).sum()
        w = np.zeros(len(idx))
        used = False
        for pc, share in self.mix.get((unit, node), []):
            if pc is None:
                continue
            gv = pd.Series(self._proxy(ug, pc), index=ug)
            if gv.sum() <= 0:
                continue
            gshare = gv / gv.sum()
            # a Gemeinde's share spread over its hexes by population (equal if it has none)
            gp = gpop.reindex(gems).to_numpy()
            cnt = pd.Series(gems).map(pd.Series(gems).value_counts()).to_numpy()
            inner = np.where(gp > 0, p / np.where(gp > 0, gp, 1), 1.0 / cnt)
            w += share * gshare.reindex(gems).to_numpy() * inner
            used = True
        if p.sum() > 0:
            base = p / p.sum()
        else:
            base = np.full(len(idx), 1.0 / len(idx))
        if used and w.sum() > 0:
            self.n["citizens"] += 1
            return (1 - self.floor) * w / w.sum() + self.floor * base
        self.n["pop" if p.sum() > 0 else "none"] += 1
        return p if p.sum() > 0 else None

    def summary(self):
        return (f"{self.n['citizens']:,} (unit, language) rows placed by Tabelle 2 citizenship per "
                f"Gemeinde, {self.n['pop']:,} by population, {self.n['none']:,} on equal shares")


def _weight(place):
    if "pop" not in place.columns or "gem" not in place.columns:
        raise SystemExit("at: placement layer lacks `pop` or the Gemeinde id")
    return _AtWeighter(place)


ENTRY = dict(
    name="Austria",
    source="Volkszählung 2001, Hauptergebnisse I, Tabellen 5 and 14 (Statistik Austria)",
    how="census, 2001, everyday language (Umgangssprache), German alone or another language "
        "with or without German; languages outside the census's local columns split by "
        "citizenship inside each state",
    parts=[dict(covers="Everyone", source="2001 census, everyday language", rest=True)],
    grain="municipalities in Burgenland and Carinthia, Vienna's 23 districts, 79 political "
          "districts elsewhere; 19,800 people on average",
    view=[9.3, 46.3, 17.3, 49.1],
    counts=_counts,
    mappings=["at2001"],
    place=RD_GEO / "at" / "at_hexes.gpkg",
    place_unit=_place_unit,
    place_weight=_weight,
    note_public=(
        "This is Austria in 2001, the last census that asked about language; the censuses "
        "of 2011 and 2021 are drawn from registers and carry none. The question was the "
        "language or languages usually spoken in private life, with family and friends. "
        "Someone who named German and another language is counted under the other language, "
        "so German here means German alone: 8.6% of Austrians gave German with another "
        "language and 2.8% gave only another. Language by municipality was published only in "
        "Burgenland and Carinthia, where the Croatian, Hungarian, Romani and Slovene "
        "minorities live. Below the state, the census names only German and a few minority "
        "and neighbouring languages; every other language is one total per area, divided here "
        "by each state's published figures and placed by where citizens of the countries "
        "speaking it lived. So each state's totals are the census's, but the mix inside a "
        "district is an estimate. Russian includes Ukrainian and Belarusian, which the census "
        "counted together."),
)
