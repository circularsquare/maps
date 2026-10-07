# Netherlands (European part). No language question in the census: Dutch, plus the regional
# languages from CBS's 2019 survey of the language most spoken at home (by province, drawn as
# their Glottolog language), plus immigrant languages from CBS's population by country of
# origin, 1 Jan 2026, less the share speaking Dutch at home (SCP surveys, CBS 2019). Every row
# derived (sources/nl_cbs.py, sources/nl_build.py). Placed on Kontur hexes keyed to gemeente and
# postcode-4 area (sources/nl_geo.py). The Caribbean Netherlands are separate rows (bq, aw, cw,
# sx). The record is sources/nl.md.
from _shared import *  # noqa: F401,F403

NL_WEIGHTS = GEO / "nl" / "nl_weights.csv"
REGIONAL = {"Frisian", "Gronings", "Westphalian", "Limburgish", "Zeeuws"}
GROUPS = ["BEL", "DEU", "POL", "EUO", "IDN", "MAR", "NCAR", "SUR", "TUR", "AFR", "AMO", "ASI"]


def _counts():
    import nl2026
    df = pd.read_csv(NORM / "nl.csv", dtype={"geo_id": str})
    df = df[df["geo_level"] == "gemeente"]
    if df["geo_id"].nunique() != 342:
        raise SystemExit(f"nl.csv: {df['geo_id'].nunique()} gemeenten, expected 342")
    df["node"] = df["source_category"].map(nl2026.resolve)
    df["unit"] = df["geo_id"]
    return df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()


class _NlWeighter:
    """Inside a gemeente, by hex (each hex carries its postcode-4 area's CBS counts, spread by
    Kontur population; sources/nl_geo.py):
      * Dutch on people born in the Netherlands;
      * the regional languages on people born in the Netherlands of Dutch origin;
      * an immigrant language on the people born abroad of the origin groups it was built
        from, mixed in the proportions nl_weights.csv gives for that gemeente (Arabic in a
        gemeente is part Syrian, part Iraqi, part Egyptian...).
    A placement weight only: the counts are nl.csv's either way."""

    def __init__(self, place):
        import numpy as np
        import nl2026
        self.np = np
        self.pop = place["pop"].to_numpy(dtype=float)
        self.unit = place["unit"].astype(str).to_numpy()
        self.w = {c: np.clip(place[f"w_{c}"].to_numpy(dtype=float), 0, None)
                  for c in GROUPS + ["native", "nlborn"]}
        self.dutch = nl2026.resolve("Dutch")
        self.regional = {nl2026.resolve(l) for l in REGIONAL}
        mix = pd.read_csv(NL_WEIGHTS)
        mix["node"] = mix["label"].map(nl2026.resolve)
        self.mix = {k: g.groupby("group")["count"].sum()
                    for k, g in mix.groupby(["unit", "node"])}
        self.n = {"origin": 0, "native": 0, "nlborn": 0, "pop": 0, "none": 0}

    def weights(self, node, idx, count, plain=False):
        np = self.np
        unit = self.unit[idx[0]]
        w, key = None, None
        if node == self.dutch:
            w, key = self.w["nlborn"][idx], "nlborn"
        elif node in self.regional:
            w, key = self.w["native"][idx], "native"
        elif (unit, node) in self.mix:
            w = np.zeros(len(idx))
            for grp, c in self.mix[(unit, node)].items():
                g = self.w[grp][idx]
                if g.sum() > 0:
                    w += c * g / g.sum()
            key = "origin"
        if w is not None and w.sum() > 0:
            self.n[key] += 1
            return w
        p = self.pop[idx]
        self.n["pop" if p.sum() > 0 else "none"] += 1
        return p if p.sum() > 0 else None

    def summary(self):
        n = self.n
        return (f"{n['origin']:,} (gemeente, language) rows placed on their origin groups' "
                f"foreign-born, {n['nlborn']:,} on Dutch-born, {n['native']:,} on Dutch-born "
                f"of Dutch origin, {n['pop']:,} on population, {n['none']:,} on equal shares")


def _weight(place):
    need = {"pop", "unit", "w_native", "w_nlborn"} | {f"w_{g}" for g in GROUPS}
    if not need <= set(place.columns) or not NL_WEIGHTS.exists():
        raise SystemExit("nl: placement columns or nl_weights.csv missing: run "
                         "sources/nl_geo.py and sources/nl_cbs.py")
    return _NlWeighter(place)


ENTRY = dict(
    name="Netherlands",
    source="CBS StatLine, population by country of origin and birthplace, 1 Jan 2026 "
           "(85458NED, 85384NED, 85640NED); CBS Sociale samenhang en welzijn 2019 (language "
           "most spoken at home, by province; Statistische Trends 2021); SCP integration "
           "surveys (SIM 2015, SING 2009, NSN 2019); NIDI Demos 2023; De Fryske Taalatlas "
           "2020; Veldeke 2021; Driessen 2012",
    how="no language question: regional languages from CBS's 2019 home-language survey by "
        "province; people of foreign origin (2026 register) on their origin country's "
        "languages, less the share who speak Dutch at home (SCP surveys); everyone else Dutch",
    parts=[
        dict(covers="Regional languages",
             source="CBS 2019 survey, language most spoken at home, by province",
             nodes=["indoeuropean.germanic.continental.lowgerman.westphalian",
                    "indoeuropean.germanic.continental.limburgish",
                    "indoeuropean.germanic.frisian",
                    "indoeuropean.germanic.continental.lowgerman.gronings",
                    "indoeuropean.germanic.continental.zeeuws"]),
        dict(covers="People of foreign origin, languages other than Dutch",
             source="2026 population register, country of origin, less the share speaking "
                    "Dutch at home (SCP surveys)",
             people=1_706_738),
        dict(covers="Everyone else", source="2026 population register, drawn as Dutch",
             rest=True),
    ],
    grain="342 gemeenten, 53,000 people on average; inside a gemeente, placed by postcode area",
    gap="none: the register counts everyone; every figure is an estimate",
    view=[3.3, 50.7, 7.3, 53.6],
    counts=_counts,
    mappings=["nl2026"],
    place=GEO / "nl" / "nl_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=_weight,
    note_public=(
        "The Dutch census has no language question, so this map is put together from other "
        "sources. The regional languages come from Statistics Netherlands' 2019 survey of the "
        "language or dialect spoken most at home, by province, for people aged 15 and over: "
        "Frisian in Friesland (40%), Low Saxon in the north and east (drawn as Gronings and "
        "Westphalian), Limburgish in Limburg (48%) and Zeeuws in Zeeland (30%). Brabant's "
        "dialect answers are drawn as Dutch, because Glottolog classes Brabants as Dutch. "
        "Most gemeenten in a province get the same share; Frisian and Limburgish follow finer "
        "surveys. People of foreign origin are counted by the population register (2026) and "
        "drawn on their origin country's languages, less the share who speak Dutch at home in "
        "SCP's surveys: 63% of Turkish-born and 46% of Moroccan-born immigrants keep the "
        "origin language, and 5% of the Surinamese. Other origins take a common rate that "
        "matches Statistics Netherlands' finding that 44% of the first generation and 16% of "
        "the second mostly speak another language at home. People born in Indonesia, mostly "
        "Dutch who left the former colony, are drawn as Dutch."),
)
