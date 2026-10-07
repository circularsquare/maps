# Greece. No language question since the 1951 census: Greek, plus the minority languages from
# the most recent published figures (placed across regional units by the 1951 census's mother
# tongue by nomos), plus immigrant languages from the 2021 census's foreign citizens by regional
# unit (sources/gr_build.py). Placed on religiondots' LAUs (read-only). The record is
# sources/gr.md.
from _shared import *  # noqa: F401,F403

GR_WEIGHTS = GEO / "gr" / "gr_weights.csv"


def _counts():
    import gr2021
    df = pd.read_csv(NORM / "gr.csv", dtype={"geo_id": str})
    df = df[df["geo_level"] == "nuts3"]
    if df["geo_id"].nunique() != 53:
        raise SystemExit(f"gr.csv: {df['geo_id'].nunique()} regional units, expected 53")
    df["node"] = df["source_category"].map(gr2021.resolve)
    df["unit"] = df["geo_id"]
    return df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()


class _GrWeighter:
    """Inside a regional unit (NUTS 3), by LAU (local community):
      * Pomak on the Pomak villages (Myki in Xanthi, Kechros and Organi in Rodopi, Mikro Derio
        in Evros), up to 85% of their people, the rest over the unit;
      * Turkish in Thrace on every LAU but those villages; in the Dodecanese on Rhodes town
        (2,500) and Kos town (2,000) (gr_weights.csv);
      * everything else by LAU population (GISCO LAU 2021).
    A placement weight only: the counts are gr.csv's either way."""

    def __init__(self, place):
        import numpy as np
        import gr2021
        self.pop = place["pop"].to_numpy(dtype=float)
        lau = place["lau"].astype(str).to_numpy()
        pos = {c: i for i, c in enumerate(lau)}
        w = pd.read_csv(GR_WEIGHTS, dtype={"lau": str})
        self.by_node = {}
        for key, g in w.groupby("key"):
            arr = self.by_node.setdefault(gr2021.resolve(key), np.zeros(len(lau)))
            for c, v in zip(g["lau"], g["weight"]):
                if c in pos:
                    arr[pos[c]] += v
        self.n = {"assigned": 0, "pop": 0, "none": 0}

    def weights(self, node, idx, count, plain=False):
        arr = self.by_node.get(node)
        if arr is not None and arr[idx].sum() > 0:
            self.n["assigned"] += 1
            return arr[idx]
        p = self.pop[idx]
        self.n["pop" if p.sum() > 0 else "none"] += 1
        return p if p.sum() > 0 else None

    def summary(self):
        n = self.n
        return (f"{n['assigned']:,} (unit, language) rows placed on their own LAUs, "
                f"{n['pop']:,} on population, {n['none']:,} on equal shares")


def _weight(place):
    if not {"pop", "lau", "nuts3"} <= set(place.columns) or not GR_WEIGHTS.exists():
        raise SystemExit("gr: placement columns or gr_weights.csv missing: run "
                         "sources/gr_build.py")
    return _GrWeighter(place)


ENTRY = dict(
    name="Greece",
    source="ELSTAT census 2021, residents by citizenship and regional unit (Eurostat "
           "cens_21ctz_r3); census 1951, mother tongue by nomos; Roma settlement mapping 2017 "
           "and 2021; published minority estimates",
    how="no language question since 1951: minority languages from published estimates, "
        "foreign residents on the languages of their citizenship, everyone else as Greek",
    parts=[
        dict(covers="Foreign residents",
             source="2021 census, citizenship, drawn on that country's languages; 38.5% moved "
                    "to Greek by Italy's 2024 immigrant survey",
             people=466_863),
        dict(covers="Roma", source="government Roma settlement mapping 2021, all drawn as "
                                   "Romani", people=117_495),
        dict(covers="Muslims of Thrace, Turks of Rhodes and Kos",
             source="published estimates (1991, 2001): Turkish and Pomak", people=100_500),
        dict(covers="Aromanian, Arvanitika, Slavic speakers of Macedonia",
             source="published speaker estimates (1991-2018), spread by the 1951 census's "
                    "mother tongue", people=150_000),
        dict(covers="Everyone else", source="drawn as Greek", rest=True),
    ],
    grain="52 regional units and Mount Athos, 200,000 people on average; placed by local "
          "community inside",
    gap="naturalised citizens of foreign origin, many Albanian, drawn as Greek",
    view=[19.2, 34.7, 28.4, 41.8],
    counts=_counts,
    mappings=["gr2021"],
    place=RD_GEO / "gr" / "gr_lau.gpkg",
    place_unit=lambda g: g["nuts3"].astype(str),
    place_weight=_weight,
    note_public=(
        "Greece has not asked about language in a census since 1951, so this map is put "
        "together from other sources and every figure on it is an estimate. Minority languages "
        "come from the most recent published figures: Turkish and Pomak for the Muslims of "
        "Thrace, Romani for every Roma person in the 2021 settlement mapping, and Aromanian, "
        "Arvanitika and the Slavic speakers of Macedonia at 50,000 each (the low end for the "
        "Slavic speakers), spread by where the 1951 census found them. These are speaker "
        "estimates, and many speakers are older people who also use Greek at home, so the "
        "minority languages are probably drawn too large. Foreign residents are drawn on their "
        "country's languages, with 38.5% moved to Greek since no Greek survey gives a share. "
        "Albanians who became Greek citizens are drawn as Greek."),
)
