# Tanzania. Afrobarometer R4 and R6-R9 (2008-2022, about 10,800 respondents), home language,
# shares per region applied to the 2022 census region totals (sources/tz_afro.py); every row
# `modelled`. On religiondots' Kontur hexes for its 30 units (Mbeya and Songwe are one), with
# each hex's COD-AB district added (sources/tz_place.py). Inside a unit, each language's dots
# lean towards the districts where the survey's own respondents named it. The record is
# sources/tz.md.
from _shared import *  # noqa: F401,F403

UNITS = 30
POP_2022 = 61_741_120


def _counts():
    import tz2022
    df = pd.read_csv(NORM / "tz.csv", dtype={"geo_id": str})
    if df["geo_id"].nunique() != UNITS:
        raise SystemExit(f"tz.csv: {df['geo_id'].nunique()} units, expected {UNITS} -- "
                         "re-run sources/tz_afro.py")
    if int(df["count"].sum()) != POP_2022:
        raise SystemExit(f"tz.csv sums to {int(df['count'].sum()):,}, not {POP_2022:,}")
    df["node"] = df["source_category"].map(tz2022.resolve)
    missing = sorted(set(df.loc[df["node"].isna(), "source_category"]))
    if missing:
        raise SystemExit(f"tz.csv answers with no node: {missing}")
    # religiondots' tz_lookup.csv: geo_id (TZ01..TZ55) -> its hex `unit`
    lut = pd.read_csv(RD_GEO / "tz" / "tz_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    if df["unit"].isna().any():
        raise SystemExit(f"tz.csv units missing from religiondots' tz_lookup.csv: "
                         f"{sorted(set(df.loc[df['unit'].isna(), 'geo_id']))}")
    df = df[df["count"] > 0]
    out = by_unit(df)
    out["tier"] = "modelled"
    return out


class _TzDistrictWeighter:
    """Inside a unit, a language's dots go to each hex in proportion to its Kontur population
    times that language's share in the hex's district (data/normalized/tz_district.csv: the
    Afrobarometer's own respondents per COD-AB district, shrunk towards the unit's share with
    one enumeration area's weight; an unsampled district borrows from its nearest sampled
    ones). A placement weight only: every unit's counts are the same either way."""

    def __init__(self, place):
        import tz2022
        self.dist = place["district"].astype(str).to_numpy()
        self.unit = place["unit"].astype(str).to_numpy()
        self.pop = place["pop"].to_numpy(dtype=float)
        t = pd.read_csv(NORM / "tz_district.csv", dtype={"geo_id": str, "adm2_pcode": str})
        t["node"] = t["source_category"].map(tz2022.resolve)
        if t["node"].isna().any():
            raise SystemExit(f"tz_district.csv answers with no node: "
                             f"{sorted(set(t.loc[t['node'].isna(), 'source_category']))}")
        lut = pd.read_csv(RD_GEO / "tz" / "tz_lookup.csv", dtype=str)
        t["unit"] = t["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
        s = t.groupby(["unit", "node", "adm2_pcode"])["share"].sum()
        self.share = {}
        for (u, n), x in s.groupby(level=[0, 1]):
            self.share[(u, n)] = x.droplevel([0, 1]).to_dict()
        self.n = {"district": 0, "pop": 0, "none": 0}

    def weights(self, node, idx, count, plain=False):
        s = self.share.get((self.unit[idx[0]], node))
        if s is not None:
            w = pd.Series(self.dist[idx]).map(s).fillna(0.0).to_numpy() * self.pop[idx]
            if w.sum() > 0:
                self.n["district"] += 1
                return w
        p = self.pop[idx]
        if p.sum() > 0:
            self.n["pop"] += 1
            return p
        self.n["none"] += 1
        return None

    def summary(self):
        return (f"{self.n['district']:,} (unit, language) rows placed by the survey's district "
                f"shares, {self.n['pop']:,} on population, {self.n['none']:,} on equal shares")


def _weight(place):
    if "district" not in place.columns:
        raise SystemExit("tz_hexes.gpkg has no `district` column: run sources/tz_place.py")
    return _TzDistrictWeighter(place)


ENTRY = dict(
    name="Tanzania",
    source=("Afrobarometer rounds 4 and 6 to 9 (2008-2022), Tanzania, home language; region "
            "populations from the 2022 Population and Housing Census (National Bureau of "
            "Statistics)"),
    how=("a survey, 2008-2022, five rounds pooled, home language (Swahili from the 2016 "
         "round's mother tongue question); shares per region applied to the 2022 census "
         "population"),
    parts=[
        dict(covers="Swahili",
             source="Afrobarometer 2016, mother tongue, region shares",
             nodes=["nigercongo.bantu.swahili"]),
        dict(covers="Other languages",
             source="Afrobarometer 2008-2022, five rounds, home language, on 2022 census "
                    "region populations",
             rest=True),
    ],
    grain=("30 regions (Mbeya and Songwe together), 2.1 million people on average; inside a "
           "region, placed by district from the survey's respondents"),
    gap="no census count: every figure here is a survey share",
    view=[29.3, -11.8, 40.5, -0.9],
    counts=_counts,
    mappings=["tz2022"],
    place=GEO / "tz" / "tz_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=_weight,
    note_public=(
        "Tanzania's census asks neither language nor ethnicity, so this map is built from a "
        "survey. The Afrobarometer asked about 10,800 adults in five rounds from 2008 to 2022 "
        "which language they speak at home, and each region's shares are applied to its 2022 "
        "census population. A region's mix rests on 100 to 1,200 interviews, so small "
        "languages can be over or under drawn. Since 2016 about two thirds of Tanzanians "
        "answer Swahili as the language of the home, many of whom grew up with another. This "
        "map shows first languages, so Swahili is drawn from the 2016 round's question on "
        "mother tongue, about 4% of Tanzanians, most of them in Zanzibar and Dar es Salaam. "
        "That is below the 7% of mainlanders the Languages of Tanzania project counted in "
        "2009. Mbeya and Songwe are drawn as one region, because the survey filed Songwe "
        "under Mbeya until 2021."),
)
