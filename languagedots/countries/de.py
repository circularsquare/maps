# Germany. Mikrozensus 2023, language spoken mainly at home, by Land (sources/de_mz.py), placed
# on religiondots' 1km Zensus 2022 grid (each cell's `ars` starts with its Land code), re-keyed
# by sources/de_place.py with one weight column per language: the Zensus 2022 count of the
# citizens of the countries it is spoken in. The record is sources/de.md.
from _shared import *  # noqa: F401,F403
import numpy as np


def _rows():
    """de.csv's Land rows with their nodes, the three continental remainders split into
    languages by sources/de_rest.py (Zensus 2022 citizens' unnamed languages, per Land; the
    split keeps each Land's remainder total, and stays derived). Columns geo_id,
    source_category, node, tier, count."""
    import de2023
    df = pd.read_csv(NORM / "de.csv", dtype={"geo_id": str})
    df = df[df["geo_level"] == "land"]
    if df["geo_id"].nunique() != 16:
        raise SystemExit(f"de.csv: {df['geo_id'].nunique()} Laender, expected 16")
    df["node"] = df["source_category"].map(de2023.resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    rest = pd.read_csv(NORM / "de_rest.csv", dtype={"geo_id": str})
    split = df[df["source_category"].isin(set(rest["remainder"]))]
    if split.empty:
        raise SystemExit("de.csv has none of de_rest.csv's remainder labels")
    m = split.drop(columns="node").merge(
        rest[["geo_id", "remainder", "node", "share"]],
        left_on=["geo_id", "source_category"], right_on=["geo_id", "remainder"], how="left")
    if m["node"].isna().any():
        raise SystemExit("a Land's remainder has no split in de_rest.csv: run sources/de_rest.py")
    m["count"] = m["count"] * m["share"]
    if abs(m["count"].sum() - split["count"].sum()) > 1:
        raise SystemExit("the remainder split does not keep its total")
    keep = df[~df.index.isin(split.index)]
    cols = ["geo_id", "source_category", "node", "tier", "count"]
    out = pd.concat([keep[cols], m[cols]], ignore_index=True)
    return _regional(out)


def _regional(df):
    """Regional and minority languages (sources/de_regional.py: Low German from the 2016 IDS/INS
    survey, Sorbian, Frisian and Danish from published estimates), `modelled`, each taken out of
    the same Land's German without a migration history. Their label is "regional: <area>", which
    the weighter places on that area's Gemeinden."""
    reg = pd.read_csv(NORM / "de_regional.csv", dtype={"geo_id": str})
    reg = reg.assign(source_category="regional: " + reg["area"], tier="modelled")
    ger = (df["source_category"] == "Deutsch") & (df["tier"] == "derived")
    for land, n in reg.groupby("geo_id")["count"].sum().items():
        i = df.index[ger & (df["geo_id"] == land)]
        if len(i) != 1 or df.at[i[0], "count"] <= n:
            raise SystemExit(f"de: cannot take {n:,.0f} regional speakers out of {land}'s German")
        df.at[i[0], "count"] -= n
    return pd.concat([df, reg[df.columns]], ignore_index=True)


def _counts():
    df = _rows()
    df["unit"] = df["geo_id"]
    # measured: the five languages the Laender's J2 table names, for people with
    # Migrationsgeschichte. derived: J2's three mixed groups split by national shares, four
    # suppressed cells, everyone without Migrationsgeschichte at the national mix, and the
    # Mikrozensus' "another language of Europe / Asia / Africa" split by citizenship.
    return df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()


def _langs():
    """sources/de_place.py's LANGS: Mikrozensus label -> (column slug, citizenships)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("de_place", ROOT / "sources" / "de_place.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.LANGS


class _DeCtzWeighter:
    """Inside a Land, a language's dots go where Zensus 2022 counted the citizens of the
    countries it is spoken in (sources/de_place.py: Gemeinde counts per citizenship, spread
    inside each Gemeinde by the 1km grid). German goes on German citizens. A node fed by
    several Mikrozensus labels (`other`: another European, another Asian, any other language)
    blends their weights by the labels' shares of that node in that Land. A placement weight
    only: every Land's counts are the Mikrozensus' either way."""

    def __init__(self, place):
        langs = _langs()
        self.col = {c: place[c].to_numpy(dtype=float) for c in place.columns
                    if str(c).startswith("w_")}
        self.pop = place["pop"].to_numpy(dtype=float)
        self.unit = place["unit"].astype(str).to_numpy()
        # a language split out of a continental remainder keeps the remainder's label, so it is
        # placed by that remainder's citizens (all of the continent's citizenships not named)
        # a regional language goes on its area's Gemeinden, cells by population x the Gemeinde's
        # factor (sources/de_regional.py: 2 under 20,000 people for the rural-weighted areas)
        areas = pd.read_csv(NORM / "de_regional_areas.csv", dtype={"ars": str})
        ars = place["ars"].astype(str)
        for a, g in areas.groupby("area"):
            self.col[f"r_{a}"] = place["pop"].to_numpy(dtype=float) * ars.map(
                g.set_index("ars")["factor"]).fillna(0).to_numpy(dtype=float)
        label_col = lambda lab: (f"r_{lab[len('regional: '):]}"  # noqa: E731
                                 if lab.startswith("regional: ") else f"w_{langs[lab][0]}")
        df = _rows()
        self.mix = {}
        for (u, n), g in df.groupby(["geo_id", "node"]):
            by = g.groupby("source_category")["count"].sum()
            cols = [(label_col(lab), v / by.sum()) for lab, v in by.items()]
            missing = [c for c, _ in cols if c not in self.col]
            if missing:
                raise SystemExit(f"de_grid_1km.gpkg lacks {missing}: run sources/de_place.py")
            self.mix[(u, n)] = cols
        self.n = {"citizens": 0, "pop": 0, "none": 0}

    def weights(self, node, idx, count, plain=False):
        w = np.zeros(len(idx))
        for col, share in self.mix.get((self.unit[idx[0]], node), []):
            v = self.col[col][idx]
            if v.sum() > 0:
                w += share * v / v.sum()
        if w.sum() > 0:
            self.n["citizens"] += 1
            return w
        p = self.pop[idx]
        self.n["pop" if p.sum() > 0 else "none"] += 1
        return p if p.sum() > 0 else None

    def summary(self):
        return (f"{self.n['citizens']:,} (Land, language) rows placed on Zensus 2022 citizens "
                f"(sources/de_place.py) or a regional language's area (de_regional.py), "
                f"{self.n['pop']:,} on population, {self.n['none']:,} "
                f"on equal shares")


def _weight(place):
    if not any(str(c).startswith("w_") for c in place.columns):
        raise SystemExit("de_grid_1km.gpkg has no weight columns: run sources/de_place.py")
    return _DeCtzWeighter(place)


ENTRY = dict(
    name="Germany",
    source="Mikrozensus 2023, language spoken mainly at home: the Laender's Integrationsmonitoring "
           "(indicators J2 and A1a) by Land, Destatis table 12211-40 nationally; Zensus 2022, "
           "foreign citizens by citizenship and Land (table 1000A-1023), for the languages "
           "the survey does not name; IDS/INS survey 'Status und Gebrauch des Niederdeutschen "
           "2016' for Low German; published speaker estimates for Sorbian, Frisian and Danish",
    how="household survey (Mikrozensus, 1% sample), 2023, language spoken mainly at home, by "
        "state; languages it does not name estimated from census foreign citizens; Low German, "
        "Sorbian, Frisian and Danish from other surveys and estimates",
    parts=[
        dict(covers="Low German, Sorbian, Frisian and Danish",
             source="IDS/INS survey 2016, people who speak Low German very well; published "
                    "speaker estimates for the others; taken out of German",
             people=1_127_923),
        dict(covers="Everyone else",
             source="Mikrozensus 2023, language spoken mainly at home; languages it files "
                    "only by continent estimated from 2022 census foreign citizens",
             rest=True),
    ],
    grain="16 states, 5.2 million people on average; inside a state, placed by citizenship per "
          "municipality and 1 km square",
    gap="about 0.8 million people, 1%, living in communal housing such as care homes and "
        "shelters, whom the survey's figures leave out",
    view=[5.4, 47.0, 15.6, 55.3],
    counts=_counts,
    mappings=["de2023"],
    place=GEO / "de" / "de_grid_1km.gpkg",
    place_unit=lambda g: g["ars"].astype(str).str[:2],
    place_weight=_weight,
    note_public=(
        "Germany's 2022 census asked nothing about language. These figures come from the "
        "Mikrozensus, a survey of 1% of households, which asks which language each person "
        "mainly speaks at home, and publishes the answers by state. Destatis names 32 "
        "languages and files the rest by continent; about 1 million people in those groups "
        "are divided here among the languages of the foreign citizens the 2022 census counted "
        "in each state from that part of the world. That is an estimate of which languages "
        "these are, not a count. Inside a state, each language's dots are placed where the "
        "2022 census counted citizens of the countries it is spoken in, down to the square "
        "kilometre. Citizenship only locates a language: most Russian speakers in Germany are "
        "German citizens, and many Turkish speakers too, so these dots assume they live where "
        "the foreign citizens of the same origin do. The survey does not name Low German, "
        "Sorbian or Frisian, so they are estimated and taken out of German: Low German from a "
        "2016 survey of the north (people who speak it very well), the others from published "
        "speaker estimates placed in their home areas. Germany's Sinti, who are German "
        "citizens, are not on the map as Romani speakers."),
)
