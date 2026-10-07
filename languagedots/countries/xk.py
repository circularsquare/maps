# Kosovo. Census 2024 mother tongue by municipality (sources/xk_census.py), on religiondots' Kontur
# hexes keyed to the 38 municipalities and, inside them, to the 2011 census settlements
# (sources/xk_geo.py). The record is sources/xk.md.
import numpy as np

from _shared import *  # noqa: F401,F403

PLACE = GEO / "xk" / "xk_hexes.gpkg"
SETTLEMENTS = GEO / "xk" / "xk_settlements.csv"

# language node -> the 2011 settlement ethnicity columns whose settlements it follows inside a unit.
# Ashkali and Egyptians speak Albanian (nationally Albanian mother tongue 1,485,170 against
# 1,481,787 Albanians, Ashkali and Egyptians together). Gorani answered Serbian, Bosnian or other
# (Dragash: 7,828 Gorani; 1,676 Serbian, 3,674 Bosnian and 5,980 other mother tongue against 17
# Serbs and 2,900 Bosniaks), so those three also follow the Gorani villages.
ETHNICITY = {
    "indoeuropean.albanian.albanian": ["albanians", "ashkali", "egyptians"],
    "indoeuropean.slavic.south.serbian": ["serbs", "gorani"],
    "indoeuropean.slavic.south.bosnian": ["bosniaks", "gorani"],
    "turkic.turkish": ["turks"],
    "indoeuropean.indoaryan.romani.romani": ["roma"],
    "other": ["other", "gorani"],
}

# ASK's estimate for the north, by ethnicity (xk_north.csv) -> the language its additions sit in
NORTH_LANGUAGE = {
    "Serb": "indoeuropean.slavic.south.serbian",
    "Bosniak": "indoeuropean.slavic.south.bosnian",
    "Albanian": "indoeuropean.albanian.albanian",
    "Turk": "turkic.turkish",
    "Others": "other",
}


def _counts():
    """The published table, with the part of the four northern municipalities that is ASK's own
    estimate rather than anyone's answer marked tier="derived".

    census2024_22 already contains ASK's estimate for the north (its totals there are the
    estimated ethnicity table's, 16,949 more than were enumerated). Mother tongue is not published
    enumerated-only, so the estimated part is taken from the ethnicity pair: in each northern
    municipality, the people ASK added in an ethnicity are moved to `derived` in the matching
    language, up to that language's count. In Leposaviq, Zubin Potok and Zveqan the Serbian count
    equals the estimated Serb count exactly. The 49 added who "prefer not to answer" ethnicity
    have no language to come off and stay measured.
    """
    import xk2024
    df = pd.read_csv(NORM / "xk.csv", dtype={"geo_id": str})
    df = df[df["geo_level"] == "municipality"].copy()
    if df["geo_id"].nunique() != 38:
        raise SystemExit("xk: expected 38 municipalities in xk.csv; re-run sources/xk_census.py")
    df["unit"] = df["geo_id"]
    df["node"] = df["source_category"].map(xk2024.resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    out = by_unit(df)
    out["tier"] = "measured"

    nn = pd.read_csv(NORM / "xk_north.csv", dtype={"geo_id": str})
    nn = nn[nn["ethnicity"].isin(NORTH_LANGUAGE) & (nn["added"] > 0)]
    if nn.empty:
        raise SystemExit("xk_north.csv has no additions; re-run sources/xk_census.py")
    add = nn.assign(node=nn["ethnicity"].map(NORTH_LANGUAGE)) \
        .groupby(["geo_id", "node"])["added"].sum()
    derived = []
    for (unit, node), n in add.items():
        hit = (out["unit"] == unit) & (out["node"] == node) & (out["tier"] == "measured")
        if not hit.any():
            continue
        i = out.index[hit][0]
        d = int(min(n, out.at[i, "count"]))
        out.at[i, "count"] -= d
        derived.append({"unit": unit, "node": node, "count": d, "tier": "derived"})
    der = pd.DataFrame(derived)
    print(f"  xk: {der['count'].sum():,} people in the four northern municipalities marked derived "
          f"(ASK's estimate), {der.loc[der['node'].str.endswith('serbian'), 'count'].sum():,} "
          "of them Serbian")
    out = pd.concat([out[out["count"] > 0], der[der["count"] > 0]], ignore_index=True)
    return out


class _XkWeighter:
    """Inside a municipality, each language follows the 2011 settlements of the matching ethnicity.

    A hex's weight for Albanian is its Kontur population times the share of Albanians, Ashkali
    and Egyptians in its settlement in 2011 (sources/xk_geo.py gives every hex its nearest
    settlement of the same municipality). The counts drawn are the 2024 municipal table's; only
    where inside the municipality they go is borrowed. The four northern municipalities were not
    enumerated in 2011 and follow plain population, as does any language whose ethnicity has no
    one in a unit's placed settlements."""

    def __init__(self, place):
        self.pop = place["pop"].to_numpy(dtype=float)
        st = pd.read_csv(SETTLEMENTS, dtype={"code": str, "unit": str}).set_index("code")
        code = place["settlement"].fillna("").astype(str)
        tot = code.map(st["total"]).fillna(0).to_numpy(dtype=float)
        self.w = {}
        for node, cols in ETHNICITY.items():
            e = sum(code.map(st[c]).fillna(0).to_numpy(dtype=float) for c in cols)
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
        return (f"{self.n['settlement']} (municipality, language) rows placed by the 2011 "
                f"settlements' ethnic mix, {self.n['pop']} on Kontur population, "
                f"{self.n['none']} on equal shares")


ENTRY = dict(
    name="Kosovo",
    source="Census of Population, Households and Housing 2024, table census2024_22, mother "
           "tongue by municipality, with the 2011 census's ethnicity by settlement for placement "
           "(Kosovo Agency of Statistics)",
    how="census, 2024, mother tongue; placed inside each municipality by the 2011 census's "
        "ethnicity of each settlement",
    parts=[
        dict(covers="Everyone counted", source="2024 census, mother tongue", people=1_585_709),
        dict(covers="Non-participants in the four northern municipalities",
             source="the statistics agency's own estimate in the 2024 language table, nearly "
                    "all Serbian",
             rest=True),
    ],
    grain="38 municipalities, 42,000 people on average; placed by 1,300 settlements",
    view=[20.0, 41.85, 21.8, 43.25],
    counts=_counts,
    mappings=["xk2024"],
    place=PLACE,
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=lambda place: _XkWeighter(place),
    note_public=(
        "The census counted mother tongue by municipality. Inside each municipality the dots "
        "follow each village's ethnic make-up in the 2011 census, the latest that gives it; the "
        "totals are the 2024 language table's. Ashkali and Egyptians are drawn among Albanian "
        "speakers, which is the language most of them give. "
        "Most Serbs in the four northern municipalities did not take part in the census. The "
        "statistics agency's language table includes its own estimate for them, about 17,000 "
        "people, nearly all Serbian speakers; those dots are an estimate, not answers. "
        "Other is mostly the Gorani of Dragash, whose own Slavic speech the census does not "
        "name; some Gorani gave Serbian or Bosnian instead."),
)
