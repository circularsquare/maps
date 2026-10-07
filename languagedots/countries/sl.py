# Sierra Leone. 2015 PHC main language (sources/sl_census.py): Table 3.22's national counts and
# Table 3.23's printed district cells, the rest of each district from CLEAR Global's shares of
# the same census (IPUMS 10% sample). On religiondots' Kontur hexes for the 14 districts of 2015,
# each hex carrying its 2017 district (sources/sl_place.py). The record is sources/sl.md.
from _shared import *  # noqa: F401,F403

DISTRICTS = 14


def _counts():
    import sl2015
    df = pd.read_csv(NORM / "sl.csv")
    if df["geo_id"].nunique() != DISTRICTS:
        raise SystemExit(f"sl.csv: {df['geo_id'].nunique()} districts, expected {DISTRICTS} -- "
                         "re-run sources/sl_census.py")
    df["node"] = df["source_category"].map(sl2015.resolve)
    missing = sorted(set(df.loc[df["node"].isna(), "source_category"]))
    if missing:
        raise SystemExit(f"sl.csv categories with no node: {missing}")
    df = df[df["count"] > 0].rename(columns={"geo_id": "unit"})
    return df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()


class _SlDistrict17Weighter:
    """Inside a 2015 district, a language's dots go to each hex in proportion to its Kontur
    population times CLEAR Global's share of that language in the hex's 2017 district. Only
    Bombali and Port Loko (Karene was cut from both in 2017) and Koinadugu (Falaba) hold more
    than one, so elsewhere this is plain population. "Other" goes on population."""

    def __init__(self, place):
        import sl2015
        from importlib import util
        spec = util.spec_from_file_location("sl_census", ROOT / "sources" / "sl_census.py")
        mod = util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        codes = dict(mod.CODE, **mod.FOREIGN)
        clear = pd.read_csv(NORM / "sl_clear.csv")
        share = clear.pivot_table(index="pcode17", columns="clear_code", values="share",
                                  aggfunc="sum").fillna(0.0)
        self.pop = place["pop"].to_numpy(dtype=float)
        pc = place["pcode17"].astype(str)
        self.node_w = {sl2015.resolve(lab): pc.map(share[code]).fillna(0.0).to_numpy() * self.pop
                       for lab, code in codes.items()}
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
        return (f"{self.n['clear']:,} (district, language) rows placed by CLEAR's 2017-district "
                f"shares, {self.n['pop']:,} on population, {self.n['none']:,} on equal shares")


def _weight(place):
    if "pcode17" not in place.columns:
        raise SystemExit("sl_hexes.gpkg has no `pcode17` column: run sources/sl_place.py")
    return _SlDistrict17Weighter(place)


ENTRY = dict(
    name="Sierra Leone",
    source="2015 Population and Housing Census, National Analytical Report (Statistics Sierra "
           "Leone), Tables 3.22 and 3.23; the rest of each district from CLEAR Global's "
           "district shares (IPUMS sample)",
    how="census, 2015, main language; national totals and each district's three largest "
        "languages as published, the rest of each district from a 10% sample of the same census",
    parts=[
        dict(covers="Each district's three largest languages",
             source="2015 census, main language, as published (Table 3.23)",
             people=6_169_025),
        dict(covers="Other languages",
             source="CLEAR Global's district shares from the census's 10% sample (IPUMS), "
                    "fitted to the national totals",
             rest=True),
    ],
    grain="14 districts, 497,000 people on average",
    gap="121,417 household members with no language recorded (1.7%), and the 15,994 people "
        "in institutions",
    view=[-13.4, 6.8, -10.2, 10.1],
    counts=_counts,
    mappings=["sl2015"],
    place=GEO / "sl" / "sl_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=_weight,
    note_public=(
        "Sierra Leone's 2015 census asked the main language of everyone living in a household. "
        "Statistics Sierra Leone published the national count for each language, and the three "
        "largest languages in each of the 14 districts. The rest of each district comes from "
        "CLEAR Global's figures from a 10% sample of the same census, fitted to those published "
        "numbers. Krio is the main language of 18% of people, though only 1.3% gave Krio as "
        "their ethnic group."),
)
