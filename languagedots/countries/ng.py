# Nigeria. Since 2026-10-09: MICS6 2021 (39,632 households), language of the household head, for
# the ten languages MICS names, per state; MICS's "other language" split by the pooled
# Afrobarometer R4-R9 (2008-2022) home-language answers; English and Pidgin as before
# (sources/ng_mics.py). Shares applied to COD-PS 2022 state populations; every row `modelled`.
# On religiondots' Kontur hexes with each hex's LGA added (sources/ng_place.py). Inside a state,
# each language's dots lean towards the LGAs where the Afrobarometer's respondents named it. The
# record is sources/ng.md.
from _shared import *  # noqa: F401,F403
import numpy as np
import ng2022 as _map

STATES = 37
CODPS_2022 = 216_798_930
K = 8.0          # prior weight, in respondents: one enumeration area of the survey
NEAREST = 3      # sampled LGAs an unsampled LGA borrows its shares from


def _counts():
    import ng2022
    df = pd.read_csv(NORM / "ng.csv", dtype={"geo_id": str})
    if df["geo_id"].nunique() != STATES:
        raise SystemExit(f"ng.csv: {df['geo_id'].nunique()} states, expected {STATES} -- "
                         "re-run sources/ng_mics.py")
    if int(df["count"].sum()) != CODPS_2022:
        raise SystemExit(f"ng.csv sums to {int(df['count'].sum()):,}, not {CODPS_2022:,}")
    df["node"] = df["source_category"].map(ng2022.resolve)
    missing = sorted(set(df.loc[df["node"].isna(), "source_category"]))
    if missing:
        raise SystemExit(f"ng.csv answers with no node: {missing}")
    df = df[df["count"] > 0].rename(columns={"geo_id": "unit"})
    out = by_unit(df)
    out["tier"] = "modelled"
    return out


class _NgLgaWeighter:
    """Inside a state, a language's dots go to each hex in proportion to its Kontur population
    times that language's share in the hex's LGA, from the Afrobarometer's own respondents
    (rounds 4, 6 and 9, the ones whose LGA labels hold; data/normalized/ng_lga.csv).

    An LGA the survey sampled gets (respondents naming the language + K x the state's share)
    / (respondents + K), so one enumeration area of eight moves its LGA most of the way but not
    all of it. An LGA it did not sample borrows the mean of the NEAREST sampled LGAs of its own
    state, weighted by inverse squared distance between LGA centres. A language no sampled
    respondent named in the state goes on population. A placement weight only: every state's
    counts are the survey's shares times COD-PS either way."""

    def __init__(self, place):
        import ng2022
        lga = place["lga"].astype(str).to_numpy()
        self.pop = place["pop"].to_numpy(dtype=float)
        t = pd.read_csv(NORM / "ng_lga.csv", dtype={"geo_id": str, "lga_pcode": str})
        t["node"] = t["source_category"].map(ng2022.resolve)
        if t["node"].isna().any():
            raise SystemExit(f"ng_lga.csv answers with no node: "
                             f"{sorted(set(t.loc[t['node'].isna(), 'source_category']))}")
        # LGA centres: population-weighted hex centroids, in metres
        c = place.geometry.to_crs(3857).centroid
        xy = pd.DataFrame({"lga": lga, "unit": place["unit"].astype(str).to_numpy(),
                           "x": c.x.to_numpy() * self.pop, "y": c.y.to_numpy() * self.pop,
                           "p": self.pop + 1e-9})
        g = xy.groupby("lga").agg(unit=("unit", "first"), x=("x", "sum"), y=("y", "sum"),
                                  p=("p", "sum"))
        g["x"] /= g["p"]
        g["y"] /= g["p"]
        self.share = {}            # (state, node) -> {lga: share}
        for st, ts in t.groupby("geo_id"):
            cnt = ts.pivot_table(index="lga_pcode", columns="node", values="w",
                                 aggfunc="sum", fill_value=0.0)
            tot = cnt.sum(axis=1)
            p = cnt.sum() / tot.sum()
            f = (cnt + K * p) .div(tot + K, axis=0)
            lgas = g[g["unit"] == st]
            samp = [x for x in f.index if x in lgas.index]
            f = f.loc[samp]
            sx, sy = lgas.loc[samp, "x"].to_numpy(), lgas.loc[samp, "y"].to_numpy()
            rows = {}
            for lg, r in lgas.iterrows():
                if lg in f.index:
                    rows[lg] = f.loc[lg].to_numpy()
                    continue
                d2 = (sx - r["x"]) ** 2 + (sy - r["y"]) ** 2
                near = np.argsort(d2)[:NEAREST]
                w = 1.0 / np.maximum(d2[near], 1e6)
                rows[lg] = (f.iloc[near].to_numpy() * w[:, None]).sum(axis=0) / w.sum()
            full = pd.DataFrame(rows, index=f.columns).T
            for node in full.columns:
                self.share[(st, node)] = full[node].to_dict()
        self.lga = lga
        self.unit = place["unit"].astype(str).to_numpy()
        self.n = {"lga": 0, "pop": 0, "none": 0}
        self.pid = _pidgin_share(place, self.unit, self.pop)

    def weights(self, node, idx, count, plain=False):
        if node == PIDGIN:
            w = self.pop[idx] * self.pid[idx]
            if w.sum() > 0:
                self.n["lga"] += 1
                return w
        st = self.unit[idx[0]]
        s = self.share.get((st, node))
        if s is not None:
            w = pd.Series(self.lga[idx]).map(s).fillna(0.0).to_numpy() * self.pop[idx]
            w = w * (1.0 - self.pid[idx])
            if w.sum() > 0:
                self.n["lga"] += 1
                return w
        p = self.pop[idx] * (1.0 - self.pid[idx])
        if p.sum() > 0:
            self.n["pop"] += 1
            return p
        self.n["none"] += 1
        return None

    def summary(self):
        return (f"{self.n['lga']:,} (state, language) rows placed by the survey's LGA shares, "
                f"{self.n['pop']:,} on population, {self.n['none']:,} on equal shares")


PIDGIN = "creole.english_based.nigerian_pidgin"
DENSE = 5000.0     # people per km²: a hex this dense counts as city for Pidgin's placement
CITY_MAX = 0.5     # Pidgin's share of a city hex at most; the rest goes on the state's other hexes


def _pidgin_share(place, unit, pop):
    """Pidgin's share of each hex (Anita, 2026-10-06; sources/ng.md §2b). First-language Pidgin
    is a city language (Warri, Sapele, Port Harcourt, Benin City, Aba, Calabar...), so inside a
    state its count goes on the hexes of DENSE people per km² or more, as a flat share of their
    people, at most CITY_MAX; anything over that is spread flat over the state's other hexes.
    Every other language's placement weight is multiplied by (1 - this share), so a hex's dots
    still add up to its people."""
    df = pd.read_csv(NORM / "ng.csv", dtype={"geo_id": str})
    pid = df[df["source_category"] == "Nigerian Pidgin"].set_index("geo_id")["count"]
    area = place.geometry.to_crs(6933).area.to_numpy() / 1e6
    dense = (pop / np.maximum(area, 1e-9)) >= DENSE
    out = np.zeros(len(pop))
    for st, P in pid.items():
        m = unit == st
        U = pop[m & dense].sum()
        T = pop[m].sum()
        f = min(P / U, CITY_MAX) if U > 0 else 0.0
        g = (P - f * U) / max(T - U, 1.0)
        out[m & dense] = f
        out[m & ~dense] = g
        if g >= f and U > 0:
            raise SystemExit(f"ng: Pidgin in {st} is no denser in its cities ({f:.2f} vs {g:.2f})")
    return out


def _weight(place):
    if "lga" not in place.columns:
        raise SystemExit("ng_hexes.gpkg has no `lga` column: run sources/ng_place.py")
    return _NgLgaWeighter(place)


ENTRY = dict(
    name="Nigeria",
    source=("MICS6 2021 (NBS/UNICEF), language of the household head; Afrobarometer rounds 4 "
            "to 9 (2008-2022), home language; state populations from COD-PS 2022 (NPC/UNFPA "
            "projection)"),
    how=("two surveys, shares per state applied to projected populations: MICS 2021, language "
         "of the household head, for the nine languages it names; its other languages split by "
         "the Afrobarometer's 2008-2022 home-language answers; English at its 2017 "
         "mother-tongue share; Pidgin at Ethnologue's estimate of first-language speakers"),
    parts=[
        dict(covers="Hausa, Yoruba, Igbo, Fulfulde, Kanuri, Tiv, Ibibio, Ijaw, Edo",
             source="MICS 2021, language of the household head, about 39,600 households",
             nodes=[_map.NAMES[k] for k in ("Hausa", "Yoruba", "Igbo", "Fula", "Kanuri",
                                            "Tiv", "Ibibio", "Ijaw", "Edo")]),
        dict(covers="English",
             source="Afrobarometer 2017, mother tongue, about 1,600 adults",
             nodes=[_map.NAMES["English"]]),
        dict(covers="Nigerian Pidgin",
             source="Ethnologue (2023), 4.7 million first-language speakers, shared by the "
                    "Afrobarometer's Pidgin-at-home answers",
             nodes=[_map.NAMES["Nigerian Pidgin"]]),
        dict(covers="Other languages",
             source="MICS 2021's other languages, split by Afrobarometer 2008-2022 home "
                    "language, about 11,900 adults",
             rest=True),
    ],
    grain=("37 states, 5.9 million people on average, placed by LGA from the Afrobarometer's "
           "respondents"),
    gap=("every figure is a survey share, as Nigeria has asked no language question since 1963; "
         "languages outside MICS's nine rest on the Afrobarometer's few hundred interviews a "
         "state, and where it met few speakers of them, part of MICS's other-language share is "
         "drawn as other Nigerian languages"),
    view=[2.6, 4.2, 14.7, 13.95],
    counts=_counts,
    mappings=["ng2022"],
    place=GEO / "ng" / "ng_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=_weight,
    note_public=(
        "Nigeria's census has not asked about language since 1963, so this map is built from "
        "two surveys. The 2021 Multiple Indicator Cluster Survey (MICS) asked about 40,000 "
        "households, around 1,000 in each state, the language of the household head, and its "
        "figures are used for the nine languages it lists: Hausa, Yoruba, Igbo, Fulfulde, "
        "Kanuri, Tiv, Ibibio, Ijaw and Edo. Everyone in a household is drawn on the head's "
        "language. The quarter of Nigerians whose head speaks another language are divided "
        "among those languages using the Afrobarometer, which asked about 11,900 adults from "
        "2008 to 2022 which language they speak at home; where it met few such speakers in a "
        "state, part of that share is drawn as other Nigerian languages. The MICS figure for "
        "Fulfulde may run high: about one Fulfulde-speaking head in five gave Hausa as their "
        "native language in another question of the same interview. English is drawn at its "
        "share of the Afrobarometer's 2017 mother-tongue answers, and Pidgin, the first "
        "language of many people in the cities of the south, at Ethnologue's estimate of 4.7 "
        "million speakers, in each state's densest neighbourhoods. Shares are applied to each "
        "state's 2022 projected population. Within each state, languages follow the local "
        "government areas where Afrobarometer respondents named them."),
)
