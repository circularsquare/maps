# Timor-Leste. Census 2015 Volume 2 table 12, mother tongue by municipality
# (sources/tl_census.py), on religiondots' Kontur hexes, which are keyed by COD-AB p-code and
# split Atauro (TL0604) out of Dili (TL06); in 2015 Atauro was part of Dili, so the two are one
# unit here and the Atauro languages are placed on the island inside it. Read-only. The record is
# sources/tl.md.
from _shared import *  # noqa: F401,F403
import numpy as np

DILI = "TL06"
# English in these two municipalities is withheld, not drawn (sources/tl.md §4): rural, with the
# age profile of the whole municipality, absent from the 2010 census (Ermera 47, Manufahi 23)
# and from both municipalities' 2022 charts. Asserted, so a changed table stops the build.
ENGLISH_WITHHELD = {"TL07": 4774, "TL10": 1954}
# Census 2015 Volume 2 table 3: Atauro administrative post and the whole of Dili
ATAURO_POP_2015, DILI_POP_2015 = 9274, 277279
ATAURO_LAT = -8.40     # Atauro spans -8.31 to -8.13; mainland Dili ends at -8.485


def _counts():
    import tl2015
    df = pd.read_csv(NORM / "tl.csv", dtype={"geo_id": str})
    df = df[df["geo_level"] == "municipality"].copy()
    if df["geo_id"].nunique() != 13:
        raise SystemExit(f"tl.csv: {df['geo_id'].nunique()} municipalities, expected 13")
    for unit, n in ENGLISH_WITHHELD.items():
        m = (df["geo_id"] == unit) & (df["source_category"] == "English")
        if int(df.loc[m, "count"].sum()) != n:
            raise SystemExit(f"tl.csv: English in {unit} is {int(df.loc[m, 'count'].sum())}, "
                             f"expected {n}; re-read sources/tl.md §4 before changing this")
        df = df[~m]
    df["node"] = df["source_category"].map(tl2015.resolve)
    df = df[df["count"] > 0].copy()
    df["unit"] = df["geo_id"]
    return by_unit(df)


class _TlWeighter:
    """Population weights, except inside Dili, which in 2015 still held Atauro island.

    The census counts the Atauro languages (Rahesuk, Raklungu, Resuk, Adabe, Atauran, Dadu'a;
    7,359 people) in Dili's column without saying where in Dili. They are placed on the island's
    hexes only. Everyone else in Dili is placed on the mainland, plus the island's remainder:
    Atauro had 9,274 people in 2015 (Volume 2 table 3), about 9,220 of them in this table's
    universe, so about 1,860 of Dili's other-language speakers are put on the island and the rest
    on the mainland, each by Kontur's population. Dili's counts are the census's either way."""

    def __init__(self, place):
        import tl2015
        self.pop = place["pop"].to_numpy(dtype=float)
        self.unit = place["unit"].astype(str).to_numpy()
        y = place.geometry.representative_point().y.to_numpy()
        self.island = (self.unit == DILI) & (y > ATAURO_LAT)
        if not 100 <= self.island.sum() <= 130:
            raise SystemExit(f"tl: {self.island.sum()} Atauro hexes, expected about 111")
        self.atauro_nodes = set(tl2015.ATAURO)
        c = _counts()
        dili = c[c["unit"] == DILI]
        dili_total = dili["count"].sum()
        langs = dili.loc[dili["node"].isin(self.atauro_nodes), "count"].sum()
        resident = ATAURO_POP_2015 * dili_total / DILI_POP_2015
        rest = resident - langs
        if not 0 < rest < 3000:
            raise SystemExit(f"tl: Atauro remainder {rest:,.0f} people is implausible")
        share = rest / (dili_total - langs)
        a = self.pop[self.island].sum()
        m = self.pop[(self.unit == DILI) & ~self.island].sum()
        self.f = share * m / (a * (1 - share))      # island weight factor for other languages
        self.n = {"island": 0, "dili": 0, "pop": 0, "none": 0}
        self.note = (f"Atauro: {langs:,} island-language speakers on the island, "
                     f"{rest:,.0f} other residents ({share:.2%} of Dili's other languages)")

    def weights(self, node, idx, count, plain=False):
        p = self.pop[idx]
        if self.unit[idx[0]] == DILI:
            isl = self.island[idx]
            if node in self.atauro_nodes:
                self.n["island"] += 1
                return np.where(isl, p, 0.0)
            self.n["dili"] += 1
            return np.where(isl, p * self.f, p)
        self.n["pop" if p.sum() > 0 else "none"] += 1
        return p if p.sum() > 0 else None

    def summary(self):
        return (f"{self.n['pop']:,} rows on Kontur population; in Dili, {self.n['island']} "
                f"Atauro-language rows on the island only and {self.n['dili']} other rows on the "
                f"mainland plus the island's remainder; {self.n['none']} on equal shares. "
                + self.note)


def _weight(place):
    return _TlWeighter(place)


ENTRY = dict(
    name="Timor-Leste",
    source="Population and Housing Census 2015, Volume 2, table 12 (Direcção-Geral de "
           "Estatística, now INETL)",
    how="census, 2015, mother tongue",
    parts=[dict(covers="Everyone", source="2015 census, mother tongue", rest=True)],
    grain="13 municipalities, 91,000 people on average",
    gap="3,989 people outside private households, who are not in the table, and 6,728 answers "
        "recorded as English in rural Ermera and Manufahi that cannot be what they say; 0.9% of "
        "the country between them",
    view=[124.0, -9.55, 127.4, -8.1],
    counts=_counts,
    mappings=["tl2015"],
    place=RD_GEO / "tl" / "tl_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str).replace({"TL0604": DILI}),
    place_weight=_weight,
    note_public=(
        "These are the 2015 census figures, one answer per person to the question of mother "
        "tongue. The 2022 census asked which languages each person learned as a child and "
        "allowed two, and it has published the answers only as charts, so its figures count "
        "answers rather than people and are not drawn. "
        "Tetun Prasa, the Tetun of Dili and the country's common language, is the mother tongue "
        "of 31% of people and of 82% in Dili. In 2015 Atauro was part of Dili; its own "
        "languages are drawn on the island and Dili's others on the mainland. The census lists "
        "two of Geoffrey Hull's group names, Idalaka and Kawaimina, beside the languages they "
        "group, and the few hundred people who gave them are drawn as answered. About 6,700 "
        "people in rural Ermera and Manufahi are recorded with English as their mother "
        "tongue, which cannot be right, so they are left off the map."),
)
