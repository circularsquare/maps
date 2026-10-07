# Cameroon. Afrobarometer R5-R9 (2013-2022, about 5,980 respondents), home language, shares per
# unit applied to COD-PS 2025 unit totals (sources/cm_afro.py); every row `modelled`. On
# religiondots' Kontur hexes for its 12 units (ten regions, Yaoundé and Douala apart), with each
# hex's COD-AB department added (sources/cm_place.py). Inside a unit, each language's dots lean
# towards the departments where the survey's own respondents named it. The record is
# sources/cm.md.
from _shared import *  # noqa: F401,F403
import numpy as np

UNITS = 12
POP_2025 = 29_442_318


def _counts():
    import cm2022
    df = pd.read_csv(NORM / "cm.csv", dtype={"geo_id": str})
    if df["geo_id"].nunique() != UNITS:
        raise SystemExit(f"cm.csv: {df['geo_id'].nunique()} units, expected {UNITS} -- "
                         "re-run sources/cm_afro.py")
    if int(df["count"].sum()) != POP_2025:
        raise SystemExit(f"cm.csv sums to {int(df['count'].sum()):,}, not {POP_2025:,}")
    df["node"] = df["source_category"].map(cm2022.resolve)
    missing = sorted(set(df.loc[df["node"].isna(), "source_category"]))
    if missing:
        raise SystemExit(f"cm.csv answers with no node: {missing}")
    # religiondots' cm_lookup.csv: geo_id is the hex layer's `unit`
    lut = pd.read_csv(RD_GEO / "cm" / "cm_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    if df["unit"].isna().any():
        raise SystemExit(f"cm.csv units missing from religiondots' cm_lookup.csv: "
                         f"{sorted(set(df.loc[df['unit'].isna(), 'geo_id']))}")
    df = df[df["count"] > 0]
    out = by_unit(df)
    out["tier"] = "modelled"
    return out


class _CmDepartmentWeighter:
    """Inside a unit, a language's dots go to each hex in proportion to its Kontur population
    times that language's share in the hex's department (data/normalized/cm_department.csv: the
    Afrobarometer's own respondents of rounds 6, 7 and 9 per COD-AB department, shrunk towards
    the unit's share with one enumeration area's weight; an unsampled department borrows from its
    nearest sampled ones). A placement weight only: every unit's counts are the same either way."""

    def __init__(self, place):
        import cm2022
        self.dep = place["department"].astype(str).to_numpy()
        self.unit = place["unit"].astype(str).to_numpy()
        self.pop = place["pop"].to_numpy(dtype=float)
        t = pd.read_csv(NORM / "cm_department.csv", dtype={"geo_id": str, "adm2_pcode": str})
        t["node"] = t["source_category"].map(cm2022.resolve)
        if t["node"].isna().any():
            raise SystemExit(f"cm_department.csv answers with no node: "
                             f"{sorted(set(t.loc[t['node'].isna(), 'source_category']))}")
        s = t.groupby(["geo_id", "node", "adm2_pcode"])["share"].sum()
        self.share = {}
        for (u, n), x in s.groupby(level=[0, 1]):
            self.share[(u, n)] = x.droplevel([0, 1]).to_dict()
        self.n = {"department": 0, "pop": 0, "none": 0}
        self.pid = _pidgin_share(place, self.unit, self.pop)

    def weights(self, node, idx, count, plain=False):
        if node == PIDGIN:
            w = self.pop[idx] * self.pid[idx]
            if w.sum() > 0:
                self.n["department"] += 1
                return w
        u = self.unit[idx[0]]
        s = self.share.get((u, node))
        if s is not None:
            dep = pd.Series(self.dep[idx])
            # the department's share among the languages other than Pidgin, whose own place is
            # set by the city rule below
            sp = self.share.get((u, PIDGIN), {})
            rest = 1.0 - dep.map(sp).fillna(0.0).to_numpy()
            w = dep.map(s).fillna(0.0).to_numpy() / np.maximum(rest, 0.05) * self.pop[idx]
            w = w * (1.0 - self.pid[idx])
            if w.sum() > 0:
                self.n["department"] += 1
                return w
        p = self.pop[idx] * (1.0 - self.pid[idx])
        if p.sum() > 0:
            self.n["pop"] += 1
            return p
        self.n["none"] += 1
        return None

    def summary(self):
        return (f"{self.n['department']:,} (unit, language) rows placed by the survey's department "
                f"shares, {self.n['pop']:,} on population, {self.n['none']:,} on equal shares")


PIDGIN = "creole.english_based.cameroonian_pidgin"
DENSE = 5000.0     # people per km²: a hex this dense counts as city for Pidgin's placement
CITY_MAX = 0.5     # Pidgin's share of a city hex at most; the rest goes on the unit's other hexes


def _pidgin_share(place, unit, pop):
    """Pidgin's share of each hex (Anita, 2026-10-06; sources/cm.md §2b). First-language Pidgin
    is a town language (Bamenda, Buea, Limbe, Kumba, Tiko...), so inside a unit its count goes
    on the hexes of DENSE people per km² or more, as a flat share of their people, at most
    CITY_MAX; anything over that is spread flat over the unit's other hexes. Every other
    language's placement weight is multiplied by (1 - this share)."""
    df = pd.read_csv(NORM / "cm.csv", dtype={"geo_id": str})
    lut = pd.read_csv(RD_GEO / "cm" / "cm_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    pid = df[df["source_category"] == "Cameroonian Pidgin"].set_index("unit")["count"]
    area = place.geometry.to_crs(6933).area.to_numpy() / 1e6
    dense = (pop / np.maximum(area, 1e-9)) >= DENSE
    out = np.zeros(len(pop))
    for u, P in pid.items():
        m = unit == str(u)
        U = pop[m & dense].sum()
        T = pop[m].sum()
        f = min(P / U, CITY_MAX) if U > 0 else 0.0
        g = (P - f * U) / max(T - U, 1.0)
        out[m & dense] = f
        out[m & ~dense] = g
        if g >= f and U > 0:
            raise SystemExit(f"cm: Pidgin in {u} is no denser in its towns ({f:.2f} vs {g:.2f})")
    return out


def _weight(place):
    if "department" not in place.columns:
        raise SystemExit("cm_hexes.gpkg has no `department` column: run sources/cm_place.py")
    return _CmDepartmentWeighter(place)


ENTRY = dict(
    name="Cameroon",
    source=("Afrobarometer rounds 5 to 9 (2013-2022), Cameroon, home language; unit populations "
            "from COD-PS 2025 (BUCREP projection)"),
    how=("a survey, 2013-2022, home language, five rounds pooled, about 5,980 adults, regional "
         "shares on the projected 2025 population; French, English and Fulfulde at their "
         "mother-tongue share; Pidgin at a published native-speaker share"),
    parts=[
        dict(covers="French, English and Fulfulde",
             source="Afrobarometer 2018, mother tongue",
             nodes=["indoeuropean.romance.french", "indoeuropean.germanic.english",
                    "nigercongo.atlantic.fulah"]),
        dict(covers="Cameroonian Pidgin",
             source="5% of the population, the native-speaker share in Neba, Chibaka and "
                    "Atindogbé (2006), placed in the towns where the survey heard it",
             nodes=["creole.english_based.cameroonian_pidgin"]),
        dict(covers="Everyone else",
             source="Afrobarometer 2013-2022, home language, about 5,980 adults", rest=True),
    ],
    grain=("10 regions with Yaoundé and Douala apart, 2.5 million people on average; inside a "
           "unit, placed by department (58) from the same survey's respondents"),
    gap=("no census count: Cameroon's censuses have not printed language or ethnicity, so every "
         "figure here is a survey share. Languages named by a single respondent are drawn as "
         "other Cameroonian languages"),
    view=[8.4, 1.6, 16.3, 13.1],
    counts=_counts,
    mappings=["cm2022"],
    place=GEO / "cm" / "cm_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=_weight,
    note_public=(
        "Cameroon's census does not ask about language, and the country has about 250 of them, "
        "so this map is built from a survey. The Afrobarometer asked about 5,980 adults in five "
        "rounds from 2013 to 2022 which language they speak at home, and each region's shares "
        "are applied to its projected 2025 population. A region's mix rests on 200 to 1,000 "
        "interviews, so most of the country's languages are missing and small ones can be over "
        "or under drawn: Bakundu, for one, is drawn far larger than any estimate of its "
        "speakers. Since 2016 the question has asked for the language spoken in the home, and "
        "French then rises from 16% of answers to 49%. So French, English and Fulfulde are "
        "drawn at their share of the 2018 round's mother-tongue answers instead: French about "
        "2%, English about 1%, Fulfulde 11%. Pidgin is drawn at 5% of Cameroonians, a "
        "published share of native speakers, mostly in the Nord-Ouest, Sud-Ouest and the "
        "towns of the Littoral. Within each region, the other languages' dots follow the "
        "departments where the survey's respondents named each language."),
)
