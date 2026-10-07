# South Sudan. High Frequency South Sudan Survey waves 1 (2015) and 2 (2016, Warrap), tribe of the
# household head read as language (sources/ss_hfs.py); shares per former state on the 2025
# county-based estimates; every row `modelled`. Seven of ten states: Jonglei, Unity and Upper Nile
# were never sampled and are not drawn. On religiondots' Kontur hexes (county-calibrated, with a
# `county` column). Inside a state, each language's dots lean towards the counties where that
# state's heads of the group were born. The record is sources/ss.md.
from _shared import *  # noqa: F401,F403

DRAWN = {"SS01", "SS02", "SS04", "SS05", "SS08", "SS09", "SS10"}
K = 12.0     # prior weight, in heads: one enumeration area of the survey


def _read():
    import ss2015
    df = pd.read_csv(NORM / "ss.csv", dtype={"geo_id": str}, keep_default_na=False, na_values=[""])
    if set(df["geo_id"]) != DRAWN:
        raise SystemExit(f"ss.csv states {sorted(set(df['geo_id']))}, expected {sorted(DRAWN)}; "
                         "re-run sources/ss_hfs.py")
    df["node"] = df["source_category"].map(ss2015.resolve)
    missing = sorted(set(df.loc[df["node"].isna(), "source_category"]) - set(ss2015.EXCLUDED))
    if missing:
        raise SystemExit(f"ss.csv answers with no node: {missing}")
    return df[df["node"].notna() & (df["count"] > 0)].rename(columns={"geo_id": "unit"})


def _counts():
    out = by_unit(_read())
    out["tier"] = "modelled"
    return out


class _SsBirthWeighter:
    """Inside a state, a language's dots go to each hex in proportion to its population times the
    language's share among heads born in the hex's county (data/normalized/ss_birth.csv: heads
    living in the state they were born in, 89% of them).

    A county's share is (heads of the language born there + K x the state's share) / (heads born
    there + K), the state's share being the drawn one (all heads, migrants included), so a
    migrant group such as Central Equatoria's Dinka still follows population. A county no head
    was born in gets the state's share. A placement weight only: the counts are the state's."""

    def __init__(self, place):
        import ss2015
        self.pop = place["pop"].to_numpy(dtype=float)
        self.county = place["county"].astype(str).to_numpy()
        self.unit = place["unit"].astype(str).to_numpy()
        df = _read()
        state = df.groupby(["unit", "node"])["count"].sum()
        state = state / state.groupby(level=0).transform("sum")
        b = pd.read_csv(NORM / "ss_birth.csv", dtype={"geo_id": str, "county": str},
                        keep_default_na=False, na_values=[""])
        b["node"] = b["source_category"].map(ss2015.resolve)
        b = b[b["node"].notna()]
        heads = b.groupby(["geo_id", "county", "node"])["heads"].sum()
        n_c = b.groupby(["geo_id", "county"])["heads"].sum()
        self.share = {}
        for (st, node), p in state.items():
            f = {}
            for (s2, c), n in n_c.items():
                if s2 != st:
                    continue
                f[c] = (float(heads.get((st, c, node), 0.0)) + K * p) / (n + K)
            self.share[(st, node)] = (f, p)
        self.n = {"county": 0, "pop": 0, "none": 0}

    def weights(self, node, idx, count, plain=False):
        st = self.unit[idx[0]]
        s = self.share.get((st, node))
        if s is not None:
            f, p = s
            w = pd.Series(self.county[idx]).map(f).fillna(p).to_numpy() * self.pop[idx]
            if w.sum() > 0:
                self.n["county"] += 1
                return w
        w = self.pop[idx]
        if w.sum() > 0:
            self.n["pop"] += 1
            return w
        self.n["none"] += 1
        return None

    def summary(self):
        return (f"{self.n['county']:,} (state, language) rows placed by birth-county shares, "
                f"{self.n['pop']:,} on population, {self.n['none']:,} on equal shares")


def _weight(place):
    if "county" not in place.columns or "pop" not in place.columns:
        raise SystemExit("ss_hexes.gpkg lacks `county` or `pop`: religiondots' sources/ss_geo.py")
    return _SsBirthWeighter(place)


ENTRY = dict(
    name="South Sudan",
    source=("High Frequency South Sudan Survey, waves 1 (2015) and 2 (2016), World Bank and "
            "National Bureau of Statistics, tribe of the household head; 2025 county-based "
            "population estimates (OCHA and NBS)"),
    how=("a survey, 2015 (Warrap 2016), ethnic group of the household head, read as language; "
         "shares per state applied to each state's 2025 estimate"),
    parts=[
        dict(covers="Six states",
             source="2015 survey (wave 1), tribe of the household head, about 3,500 households",
             people=6_885_072),
        dict(covers="Warrap",
             source="2016 survey (wave 2), tribe of the household head, Warrap's towns only",
             rest=True),
    ],
    grain=("7 former states, 1.2 million people on average; inside a state, placed by where "
           "heads of each group were born"),
    gap=("Jonglei, Unity and Upper Nile, 5.0 million people (38%), which the survey never "
         "visited; Warrap from a sample of its towns only; refugees in South Sudan"),
    view=[23.4, 3.4, 36.0, 12.3],
    counts=_counts,
    mappings=["ss2015"],
    place=RD_GEO / "ss" / "ss_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=_weight,
    note_public=(
        "South Sudan has never asked about language in a census. This map is drawn from a "
        "household survey by the World Bank and the National Bureau of Statistics, which in "
        "2015 asked about 3,500 household heads in six states which tribe they belong to; a "
        "second round in 2016 adds the towns of Warrap. Each group is drawn as its own "
        "language, and everyone in a household is drawn on the head's group, so children who "
        "have grown up speaking Juba Arabic or English at home are not shown. The survey "
        "never went to Jonglei, Unity or Upper Nile, where the war that began in 2013 was "
        "fought; they hold 38% of the country's people, most of the Nuer, Shilluk and Murle "
        "and many of the Dinka, and are left empty rather than filled from other states."),
)
