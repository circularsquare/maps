# Guinea. RGPH 2014 Tableau 5.08, main national language by région (sources/gn_rgph.py), on
# religiondots' Kontur hexes with each hex's prefecture added (sources/gn_place.py). Inside a
# région, each language's dots go to its prefectures by CLEAR Global's prefecture shares of the
# same census (IPUMS 10% sample). The record is sources/gn.md.
from _shared import *  # noqa: F401,F403
import numpy as np

REGIONS = 8


def _counts():
    import gn2014
    df = pd.read_csv(NORM / "gn.csv")
    if df["geo_id"].nunique() != REGIONS:
        raise SystemExit(f"gn.csv: {df['geo_id'].nunique()} régions, expected {REGIONS} -- "
                         "re-run sources/gn_rgph.py")
    df["node"] = df["source_category"].map(gn2014.resolve)
    missing = sorted(set(df.loc[df["node"].isna(), "source_category"]))
    if missing:
        raise SystemExit(f"gn.csv categories with no node: {missing}")
    df = df[df["count"] > 0].rename(columns={"geo_id": "unit"})
    return df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()


class _GnPrefWeighter:
    """Inside a région, a language's dots go to each hex in proportion to its Kontur population
    times CLEAR Global's share of that language in the hex's prefecture (main language of the
    household, IPUMS 10% sample of the same census; data/normalized/gn_clear.csv). Manya
    ("Tomamania"), which CLEAR leaves in `Unknown` with children under 3, is weighted by each
    prefecture's Unknown above the national median of prefectures (Macenta stands out at 22.8%
    against 8.5-12%). "Aucune" and "Autre langue nationale" go on population. A placement weight
    only: every région's counts are INS's either way."""

    def __init__(self, place):
        import gn2014
        from importlib import util
        spec = util.spec_from_file_location("gn_rgph", ROOT / "sources" / "gn_rgph.py")
        mod = util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        code_of = mod.CLEAR_CODE
        clear = pd.read_csv(NORM / "gn_clear.csv")
        share = clear.pivot_table(index="pref", columns="clear_code", values="share",
                                  aggfunc="sum").fillna(0.0)
        unk = share["Unknown"]
        share["Unknown"] = (unk - unk.median()).clip(lower=0.0)
        self.pop = place["pop"].to_numpy(dtype=float)
        pref = place["pref"].astype(str)
        self.node_w = {}
        for label, code in code_of.items():
            if code is None:
                continue
            node = gn2014.resolve(label)
            self.node_w[node] = pref.map(share[code]).fillna(0.0).to_numpy() * self.pop
        self.n = {"clear": 0, "pop": 0, "none": 0}

    def weights(self, node, idx, count, plain=False):
        w = self.node_w.get(node)
        if w is not None and w[idx].sum() > 0:
            self.n["clear"] += 1
            return w[idx]
        p = self.pop[idx]
        if p.sum() > 0:
            self.n["pop"] += 1
            return p
        self.n["none"] += 1
        return None

    def summary(self):
        return (f"{self.n['clear']:,} (région, language) rows placed by CLEAR's prefecture "
                f"shares, {self.n['pop']:,} on population, {self.n['none']:,} on equal shares")


def _weight(place):
    if "pref" not in place.columns:
        raise SystemExit("gn_hexes.gpkg has no `pref` column: run sources/gn_place.py")
    return _GnPrefWeighter(place)


ENTRY = dict(
    name="Guinea",
    source="RGPH 2014, État et structure de la population (INS Guinée), Tableau 5.08; "
           "placement inside régions from CLEAR Global's prefecture shares (IPUMS sample)",
    how="census, 2014, main language (the national language usually spoken); inside each "
        "région, placed by prefecture using a 10% sample of the same census",
    parts=[dict(covers="People aged 3 and over",
                source="2014 census, national language usually spoken", rest=True)],
    grain="8 régions, 1.2 million people aged 3 and over on average; placed by prefecture "
          "inside",
    gap="children under 3, about 1.1 million (10%), whom the language question did not cover",
    view=[-15.2, 7.1, -7.6, 12.8],
    counts=_counts,
    mappings=["gn2014"],
    place=GEO / "gn" / "gn_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=_weight,
    note_public=(
        "Guinea's 2014 census asked which national language each person usually speaks, even "
        "if they speak others, so French does not appear as an answer. INS publishes the "
        "answers for the eight régions. Inside each région, the dots are spread across its "
        "prefectures using CLEAR Global's figures from a 10% sample of the same census."),
)
