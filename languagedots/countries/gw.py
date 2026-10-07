# Guinea-Bissau. RGPH 2009 Anexo Quadro 4, principal ethnic language ("dialecto") by etnia,
# national (sources/gw_rgph.py), on religiondots' Kontur hexes re-keyed to one national unit
# with the região kept (sources/gw_place.py). Each language's dots go to the regiões through the
# etnias that name it (Anexo Quadro 2; Bissau at the urban rates), then by population inside a
# região. The record is sources/gw.md.
from _shared import *  # noqa: F401,F403

UNIT = "Guinea-Bissau"


def _counts():
    import gw2009
    df = pd.read_csv(NORM / "gw.csv")
    if set(df["geo_id"]) != {UNIT}:
        raise SystemExit("gw.csv: expected one national unit -- re-run sources/gw_rgph.py")
    df["node"] = df["source_category"].map(gw2009.resolve)
    missing = sorted(set(df.loc[df["node"].isna(), "source_category"]))
    if missing:
        raise SystemExit(f"gw.csv categories with no node: {missing}")
    df = df[df["count"] > 0].rename(columns={"geo_id": "unit"})
    return df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()


class _GwRegionWeighter:
    """A language's dots go to each hex in proportion to its Kontur population times the
    language's estimated speakers in the hex's região over the região's Kontur population
    (data/normalized/gw_place.csv: Quadro 4's etnia x language spread by Anexo Quadro 2's
    região x etnia). A placement weight only: the national count per language is INE's."""

    def __init__(self, place):
        import gw2009
        est = pd.read_csv(NORM / "gw_place.csv")
        est["node"] = est["source_category"].map(gw2009.resolve)
        per = est.groupby(["node", "region"])["people"].sum().unstack(fill_value=0.0)
        self.pop = place["pop"].to_numpy(dtype=float)
        reg = place["reg"].astype(str)
        kpop = place.groupby(reg)["pop"].sum()
        unknown = set(reg) - set(per.columns)
        if unknown:
            raise SystemExit(f"gw placement: hex regiões not in gw_place.csv: {sorted(unknown)}")
        self.node_w = {}
        for node, row in per.iterrows():
            dens = (row / kpop.reindex(row.index)).fillna(0.0)
            self.node_w[node] = reg.map(dens).fillna(0.0).to_numpy() * self.pop
        self.n = {"region": 0, "pop": 0, "none": 0}

    def weights(self, node, idx, count, plain=False):
        w = self.node_w.get(node)
        if w is not None and w[idx].sum() > 0:
            self.n["region"] += 1
            return w[idx]
        p = self.pop[idx]
        if p.sum() > 0:
            self.n["pop"] += 1
            return p
        self.n["none"] += 1
        return None

    def summary(self):
        return (f"{self.n['region']:,} languages placed by their região estimates, "
                f"{self.n['pop']:,} on population, {self.n['none']:,} on equal shares")


def _weight(place):
    if "reg" not in place.columns:
        raise SystemExit("gw_hexes.gpkg has no `reg` column: run sources/gw_place.py")
    return _GwRegionWeighter(place)


ENTRY = dict(
    name="Guinea-Bissau",
    source="RGPH 2009, Características socioculturais (INE Guiné-Bissau), Anexo Quadro 4; "
           "placement across regiões from Anexo Quadro 2 (região by etnia)",
    how="census, 2009, main ethnic language (Guinean nationals); placed across regions "
        "through where each ethnic group lives",
    parts=[dict(covers="Guinean nationals",
                source="2009 census, main ethnic language, national table", rest=True)],
    grain="one national table, placed across 9 regions (160,000 people on average)",
    # NA, 131,640, and the 10,699 people outside the table (1,933 foreign nationals, 5,070 with
    # no nationality recorded, 3,696 in collective households), both as shares of the
    # 1,452,926 enumerated (religiondots/sources/gw.md §3).
    gap="9.8% of the people counted: 9.1% with no answer, mostly infants, and 0.7% outside "
        "the table (foreigners, collective households)",
    view=[-16.8, 10.8, -13.6, 12.75],
    counts=_counts,
    mappings=["gw2009"],
    place=GEO / "gw" / "gw_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=_weight,
    note_public=(
        "Guinea-Bissau's 2009 census asked each Guinean national for their main ethnic "
        "language. Kriol, which 90% of people speak, was asked separately as a language "
        "spoken, together with Portuguese and foreign languages, so it is not among these "
        "answers. The 6% who named no ethnic language as their main one are drawn in grey; "
        "most live in Bissau and nearly all speak Kriol. The answers are national only, so "
        "each language's split between regions is an estimate from where its ethnic groups "
        "live."),
)
