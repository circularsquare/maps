# North Macedonia. Popis 2021 mother tongue by municipality (sources/mk_census.py), on Kontur hexes
# keyed to the 80 municipalities and, inside them, to the census's 1,781 settlements
# (sources/mk_geo.py). The record is sources/mk.md.
import numpy as np

from _shared import *  # noqa: F401,F403

PLACE = GEO / "mk" / "mk_hexes.gpkg"
SETTLEMENTS = GEO / "mk" / "mk_settlements.csv"

# language node -> the ethnicity column of T1503P21 whose settlements it follows inside a unit
ETHNICITY = {
    "indoeuropean.slavic.south.macedonian": "macedonians",
    "indoeuropean.albanian.albanian": "albanians",
    "turkic.turkish": "turks",
    "indoeuropean.indoaryan.romani.romani": "roma",
    "indoeuropean.romance.aromanian": "vlachs",
    "indoeuropean.slavic.south.serbian": "serbs",
    "indoeuropean.slavic.south.bosnian": "bosniaks",
}


def _counts():
    import mk2021
    df = pd.read_csv(NORM / "mk.csv", dtype={"geo_id": str})
    df = df[df["geo_level"] == "municipality"].copy()
    if df["geo_id"].nunique() != 80:
        raise SystemExit("mk: expected 80 municipalities in mk.csv; re-run sources/mk_census.py")
    # religiondots' lookup from SSO's PxWeb codes to the GISCO LAU codes the polygons carry
    lut = pd.read_csv(RD_GEO / "mk" / "mk_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["kod"])))
    if df["unit"].isna().any():
        raise SystemExit("mk: municipalities missing from religiondots' mk_lookup.csv")
    df["node"] = df["source_category"].map(mk2021.resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    return by_unit(df)


class _MkWeighter:
    """Inside a municipality, each language follows the settlements of the matching ethnicity.

    The mother-tongue table is by municipality; the same census gives ethnicity by settlement
    (T1503P21), and every hex carries the settlement nearest it (sources/mk_geo.py). A hex's
    weight for Albanian is its Kontur population times the share of Albanians in its settlement,
    so in Kumanovo, Struga or Gostivar the Albanian dots go to the Albanian villages and quarters
    and the Macedonian dots to the Macedonian ones. The counts drawn are the municipal table's
    either way; only where inside the municipality they go is borrowed. Sign language and other
    follow plain population, and so does any language whose ethnicity has no one in the unit's
    placed settlements."""

    def __init__(self, place):
        self.pop = place["pop"].to_numpy(dtype=float)
        st = pd.read_csv(SETTLEMENTS, dtype={"code": str})
        st = st.set_index("code")
        code = place["settlement"].astype(str)
        tot = code.map(st["total"]).fillna(0).to_numpy(dtype=float)
        self.w = {}
        for node, col in ETHNICITY.items():
            e = code.map(st[col]).fillna(0).to_numpy(dtype=float)
            self.w[node] = self.pop * np.where(tot > 0, e / np.where(tot > 0, tot, 1), 0.0)
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
                f"ethnic mix, {self.n['pop']} on Kontur population, {self.n['none']} on equal shares")


ENTRY = dict(
    name="North Macedonia",
    source="Census of Population, Households and Dwellings 2021, table T1015P21, mother tongue "
           "by municipality, with T1503P21, ethnicity by settlement, for placement (State "
           "Statistical Office)",
    how="census, 2021, mother tongue; placed inside each municipality by settlement ethnicity",
    parts=[dict(covers="Everyone who answered", source="2021 census, mother tongue", rest=True)],
    grain="80 municipalities, 23,000 people on average; placed by 1,781 settlements",
    gap="132,662 people (7.2%), almost all counted from administrative registers and never asked",
    view=[20.4, 40.8, 23.1, 42.4],
    counts=_counts,
    mappings=["mk2021"],
    place=PLACE,
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=lambda place: _MkWeighter(place),
    note_public=(
        "The census counted mother tongue by municipality. Inside each municipality the dots "
        "follow the census's ethnicity of each village and town. Mother tongue and ethnicity "
        "mostly agree here; where they do not, as with Macedonian-speaking Muslims (Torbeshi) "
        "who declared themselves Turks, speakers may be drawn in the wrong villages of the "
        "right municipality. 7.2% of residents were counted from administrative registers, "
        "have no answer and are not drawn. Vlach is Aromanian, with a few Megleno-Romanian "
        "villages near Gevgelija."),
)
