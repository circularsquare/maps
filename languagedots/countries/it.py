# Italy. No language question in the census: Italian, plus the regional languages from ISTAT's
# 2024 survey of language at home (its "dialetto" drawn as the language each place speaks),
# South Tyrol's 2024 and Trentino's 2021 language-group declarations, and immigrant languages
# from ISTAT's residents by citizenship, 1 Jan 2025 (sources/it_istat.py, sources/it_regional.py).
# Placed on religiondots' comuni (read-only). The record is sources/it.md.
from _shared import *  # noqa: F401,F403

IT_WEIGHTS = GEO / "it" / "it_weights.csv"


def _counts():
    import it2025
    df = pd.read_csv(NORM / "it.csv", dtype={"geo_id": str})
    df = df[df["geo_level"] == "nuts3"]
    if df["geo_id"].nunique() != 107:
        raise SystemExit(f"it.csv: {df['geo_id'].nunique()} provinces, expected 107")
    df["node"] = df["source_category"].map(it2025.resolve)
    df["unit"] = df["geo_id"]
    return df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()


class _ItWeighter:
    """Inside a province (NUTS 3), by comune:
      * a local or minority language on the comuni the build assigned it (it_weights.csv:
        the comuni whose "dialetto" it is, weighted by Italian citizens; South Tyrol's German,
        Italian and Ladin by each comune's 2024 declarations; Trentino's Ladin, Mocheno and
        Cimbrian by each comune's 2021 declarations; Slovenian and Italiot Greek on their
        villages);
      * immigrant languages on foreign citizens (religiondots' layer, ISTAT 2021);
      * Italian on Italian citizens.
    A placement weight only: the counts are it.csv's either way."""

    def __init__(self, place):
        import numpy as np
        import it2025
        self.pop = place["pop"].to_numpy(dtype=float)
        self.ital = place["ital"].to_numpy(dtype=float)
        self.foreign = place["foreign"].to_numpy(dtype=float)
        self.unit = place["unit"].astype(str).to_numpy()
        lau = place["lau"].astype(str).str.zfill(6).to_numpy()
        pos = {c: i for i, c in enumerate(lau)}
        w = pd.read_csv(IT_WEIGHTS, dtype={"lau6": str})
        self.by_node = {}
        for key, g in w.groupby("key"):
            label = key.split(":", 1)[1] if ":" in key else key
            node = it2025.resolve(label)
            arr = self.by_node.setdefault(node, np.zeros(len(lau)))
            for c, v in zip(g["lau6"], g["weight"]):
                if c in pos:
                    arr[pos[c]] += v
        self.italian = it2025.resolve("Italian")
        self.n = {"assigned": 0, "foreign": 0, "italian": 0, "pop": 0, "none": 0}

    def weights(self, node, idx, count, plain=False):
        arr = self.by_node.get(node)
        if arr is not None and arr[idx].sum() > 0:
            self.n["assigned"] += 1
            return arr[idx]
        if node == self.italian:
            w, key = self.ital[idx], "italian"
        else:
            w, key = self.foreign[idx], "foreign"
        if w.sum() > 0:
            self.n[key] += 1
            return w
        p = self.pop[idx]
        self.n["pop" if p.sum() > 0 else "none"] += 1
        return p if p.sum() > 0 else None

    def summary(self):
        n = self.n
        return (f"{n['assigned']:,} (province, language) rows placed on their own comuni, "
                f"{n['foreign']:,} on foreign citizens, {n['italian']:,} on Italian citizens, "
                f"{n['pop']:,} on population, {n['none']:,} on equal shares")


def _weight(place):
    need = {"pop", "ital", "foreign", "lau", "unit"}
    if not need <= set(place.columns) or not IT_WEIGHTS.exists():
        raise SystemExit("it: placement columns or it_weights.csv missing: run "
                         "sources/it_istat.py")
    return _ItWeighter(place)


ENTRY = dict(
    name="Italy",
    source="ISTAT, residents by citizenship and comune, 1 Jan 2025; ISTAT survey I cittadini e "
           "il tempo libero 2024 (language spoken in the family, by region); ASTAT language "
           "groups 2024 (South Tyrol); ISPAT minority declarations 2021 (Trentino)",
    how="no language question: dialect spoken at home from a 2024 survey, drawn as each "
        "place's language; South Tyrol and Trentino from language-group declarations; foreign "
        "residents on their citizenship's languages; everyone else drawn as Italian",
    parts=[
        dict(covers="Regional languages",
             source="ISTAT survey 2024, language spoken in the family, by region; 'both "
                    "Italian and dialect' counted half",
             people=12_378_732),
        dict(covers="South Tyrol, and Trentino's Ladin, Mocheno and Cimbrian",
             source="language-group declarations, South Tyrol 2024 and Trentino 2021",
             people=500_073),
        dict(covers="Foreign residents, languages other than Italian",
             source="ISTAT residents by citizenship 2025, drawn on that country's languages, "
                    "scaled to the survey's share speaking another language at home",
             people=3_988_999),
        dict(covers="Everyone else", source="drawn as Italian", rest=True),
    ],
    grain="107 provinces, 550,000 people on average; inside a province, placed by comune",
    gap="naturalised citizens, drawn on the languages of today's foreign residents",
    view=[6.2, 35.3, 18.8, 47.3],
    counts=_counts,
    mappings=["it2025"],
    place=RD_GEO / "it" / "it_lau.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=_weight,
    note_public=(
        "Italy's census has never asked about language. ISTAT's 2024 survey asked households "
        "which language they usually speak in the family, and 'dialect' is drawn as the "
        "language it is where the person lives (Neapolitan in Campania, Venetian in the "
        "Veneto). In Tuscany, Umbria, most of Lazio and central Marche the dialects are "
        "Italian. Survey shares are by region, so every province of a region gets the same "
        "share. South Tyrol's and Trentino's declarations record the group people belong to, "
        "not always the language they speak at home. Arbereshe, Occitan, Franco-Provencal, "
        "Molise Croatian, Gallurese, Sassarese and Alghero Catalan are drawn in their "
        "villages at the region's dialect rate, an estimate with no survey behind it."),
)
