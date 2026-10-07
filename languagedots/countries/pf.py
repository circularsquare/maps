# French Polynesia. RP 2012 (ISPF / INSEE), language most often spoken in the family, 15+, by
# subdivision (sources/pf_rp2012.py), on Kontur hexes calibrated to the 2017 census per commune
# associée (sources/pf_geo.py). Inside a subdivision, French and Polynesian dots are weighted by
# the 2007 census share speaking each per commune associée (ISPF's 2007 atlas). Record:
# sources/pf.md.
from _shared import *  # noqa: F401,F403
import numpy as np

FRENCH = "indoeuropean.romance.french"
POLYNESIAN = "austronesian.oceanic"


def _counts():
    import pf2012
    df = pd.read_csv(NORM / "pf.csv", dtype={"geo_id": str})
    df = df[df["geo_level"] == "subdivision"]
    if df["geo_id"].nunique() != 5:
        raise SystemExit(f"pf.csv: {df['geo_id'].nunique()} subdivisions, expected 5")
    df["node"] = df["source_category"].map(pf2012.resolve)
    df = df[df["count"] > 0]
    df["unit"] = df["geo_id"]
    return by_unit(df)


class _PfWeighter:
    """Inside a subdivision, French dots go on each hex's people times the 2007 share of its
    commune associée speaking French at home, Polynesian dots likewise with the Polynesian share
    (data/normalized/pf_place2007.csv), every other group on people. A placement weight only:
    each subdivision's counts are the 2012 census's either way."""

    def __init__(self, place):
        sh = pd.read_csv(NORM / "pf_place2007.csv", dtype={"comas": str}).set_index("comas")
        self.pop = place["pop"].to_numpy(dtype=float)
        comas = place["comas"].astype(str)
        if not set(comas) <= set(sh.index):
            raise SystemExit(f"pf: hexes on communes associées without a 2007 share: "
                             f"{sorted(set(comas) - set(sh.index))[:5]}")
        self.share = {FRENCH: comas.map(sh["share_fr"]).to_numpy(dtype=float),
                      POLYNESIAN: comas.map(sh["share_pol"]).to_numpy(dtype=float)}
        self.n = {"share": 0, "pop": 0, "none": 0}

    def weights(self, node, idx, count, plain=False):
        p = self.pop[idx]
        if node in self.share:
            w = p * self.share[node][idx]
            if w.sum() > 0:
                self.n["share"] += 1
                return w
        self.n["pop" if p.sum() > 0 else "none"] += 1
        return p if p.sum() > 0 else None

    def summary(self):
        return (f"{self.n['share']:,} (subdivision, language) rows placed on people x the 2007 "
                f"share per commune associée, {self.n['pop']:,} on people, {self.n['none']:,} on "
                f"equal shares")


def _weight(place):
    if "comas" not in place.columns:
        raise SystemExit("pf_hexes.gpkg has no `comas` column: run sources/pf_geo.py")
    return _PfWeighter(place)


ENTRY = dict(
    name="French Polynesia",
    source="Recensement de la population 2012, ISPF and INSEE: standard tables on languages "
           "(LAN1b, by subdivision); 2007 census atlas indicators for placement",
    how="census, 2012, language most often spoken in the family, aged 15 and over, in five "
        "groups",
    parts=[dict(covers="Everyone aged 15 and over",
                source="2012 census, language most often spoken in the family; placed by the "
                       "2007 census's shares",
                rest=True)],
    grain="5 subdivisions, 40,600 people aged 15 and over on average",
    gap="children under 15, about 65,000 people in 2012, who were not asked",
    view=[-154.9, -28.0, -134.2, -7.7],
    counts=_counts,
    mappings=["pf2012"],
    place=GEO / "pf" / "pf_hexes.gpkg",
    place_unit=lambda g: g["sub"].astype(str),
    place_weight=_weight,
    note_public=(
        "The census asks everyone aged 15 and over which language they speak most often in the "
        "family. ISPF has published the answers below the territory only for the five "
        "subdivisions and in five groups, most recently for 2012, so the Polynesian languages "
        "are drawn as one group (nationally 82% Tahitian). Inside each subdivision, French and "
        "Polynesian dots are placed by each one's share per commune associée at the 2007 "
        "census. French has kept gaining: in 2022, 76.2% spoke mainly French in the family "
        "and 22.8% a Polynesian language, against 70.0% and 28.2% here."),
)
