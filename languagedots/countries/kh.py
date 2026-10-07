# Cambodia. Census 2019, mother tongue, published for the whole country only
# (sources/kh_gpcc.py). One unit, Cambodia; the national counts are placed inside it by
# province-level census figures (AGENT_BRIEF §4.4), on religiondots' Kontur hexes for the 25
# provinces, read-only. The record is sources/kh.md.
from _shared import *  # noqa: F401,F403
import csv
import numpy as np

UNIT = "KH"
KERNEL_KM = 25.0   # how far a language's Glottolog point pulls its speakers between provinces
LAMBDA = 0.95      # share of a language's seed that follows that pull; the rest is even

KHMER = "austroasiatic.khmer"
CHAM = "austronesian.chamic.cham"
FOREIGN = {"austroasiatic.vietnamese", "sinotibetan.sinitic", "kradai.lao", "kradai.thai",
           "other"}
# Glottolog points for the minority languages (data/raw/glottolog/languages.csv); the rest of
# the minority languages have none that matches (taxonomy/kh2019.py) and get an even seed
GLOTTO = {
    "austronesian.chamic.jarai": "jara1266",
    "austronesian.chamic.rade": "rade1240",
    "austroasiatic.bahnaric.tampuan": "tamp1251",
    "austroasiatic.bahnaric.brao": "lave1249",
    "austroasiatic.bahnaric.kreung": "krun1240",
    "austroasiatic.bahnaric.kavet": "kave1238",
    "austroasiatic.bahnaric.lun": "lunb1239",
    "austroasiatic.bahnaric.bunong": "cent1992",
    "austroasiatic.bahnaric.stieng": "stie1250",
    "austroasiatic.bahnaric.kraol": "krao1238",
    "austroasiatic.bahnaric.mel": "melk1242",
    "austroasiatic.bahnaric.khaonh": "khao1245",
    "austroasiatic.katuic.kuy": "kuyy1240",
    "austroasiatic.pearic.pear": "pear1247",
    "austroasiatic.pearic.suoy": "suoy1242",
    "austroasiatic.pearic.saoch": "saoc1239",
}


def _national():
    import kh2019
    df = pd.read_csv(NORM / "kh.csv", dtype={"geo_id": str})
    df = df[(df["geo_level"] == "country") & ~df["source_category"].isin(kh2019.NOT_DRAWN)].copy()
    df["node"] = df["source_category"].map(kh2019.resolve)
    if df["node"].isna().any():
        raise SystemExit(f"kh: unmapped {sorted(df.loc[df['node'].isna(), 'source_category'])}")
    return df


def _counts():
    df = _national()
    if df["count"].sum() != 15_552_211:
        raise SystemExit(f"kh: drawn total {df['count'].sum():,} is not the census's 15,552,211")
    df["unit"] = UNIT
    return df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()


def _place_unit(g):
    # the layer's own `unit` is the province (KH01..KH25); keep it as `prov` for the weighter
    # before scatter.py overwrites `unit` with what this returns
    g["prov"] = g["unit"].astype(str)
    return pd.Series(UNIT, index=g.index)


def _ipf(seed, rows, cols, iters=300):
    """Rake seed (provinces x languages) to row sums `rows` and column sums `cols` (scaled to
    sum(rows)); ends on the rows, so each row is met exactly."""
    x = seed.copy()
    cols = cols * rows.sum() / cols.sum()
    for _ in range(iters):
        rs = x.sum(1)
        x *= np.divide(rows, rs, out=np.zeros_like(rows), where=rs > 0)[:, None]
        cs = x.sum(0)
        x *= np.divide(cols, cs, out=np.zeros_like(cols), where=cs > 0)[None, :]
    rs = x.sum(1)
    return x * np.divide(rows, rs, out=np.zeros_like(rows), where=rs > 0)[:, None]


class _KhWeighter:
    """Where inside Cambodia each language's dots go. A placement weight only: every count is
    the census's national figure either way.

    Per province, from the census: Khmer on the province's population less its minority-
    language speakers (Tables 2.1.1 and 2.2); Vietnamese, Chinese, Lao, Thai and other foreign
    languages on its whole population (nothing more specific is published); the minority
    languages on Table 2.2's minority speakers per province, shared among the languages by a
    rake (IPF) to Table 2.3's national totals from a seed: Cham by the province's Muslims (Table
    2.5.1 as religiondots reads it; Cambodia's Muslims are mostly Cham), the others by nearness
    to their Glottolog point, and an even seed for those with no point. Inside a province, every
    language follows Kontur's population."""

    def __init__(self, place):
        import kh2019
        self.pop = place["pop"].to_numpy(dtype=float)
        prov = place["prov"].astype(str).to_numpy()
        provs = sorted(set(prov))
        if len(provs) != 25:
            raise SystemExit(f"kh: {len(provs)} provinces on the placement layer, expected 25")
        pi = {p: i for i, p in enumerate(provs)}
        self.pidx = np.array([pi[p] for p in prov])
        kpop = np.bincount(self.pidx, weights=self.pop, minlength=25)

        nz = pd.read_csv(NORM / "kh.csv", dtype={"geo_id": str})
        pv = nz[nz["geo_level"] == "province"].copy()
        pv["prov"] = pv["geo_id"].str.replace("-", "", regex=False)
        P = pv[pv["source_category"] == "Total"].set_index("prov")["count"].reindex(provs)
        M = pv[pv["source_category"] == "Minority languages, all"].set_index("prov")["count"].reindex(provs)
        rd = pd.read_csv(RD / "data" / "normalized" / "kh.csv", dtype={"geo_id": str})
        rd = rd[(rd["geo_level"] == "province") & (rd["source_category"] == "Muslims")]
        mus = rd.assign(prov=rd["geo_id"].str.replace("-", "", regex=False)).set_index("prov")["count"].reindex(provs)
        if P.isna().any() or M.isna().any() or mus.isna().any():
            raise SystemExit("kh: a province lacks its population, minority or Muslim figure")
        P, M, mus = (s.to_numpy(dtype=float) for s in (P, M, mus))

        # per-province surface for each node, then spread inside the province by Kontur
        self.hex_share = np.divide(self.pop, kpop[self.pidx], out=np.zeros_like(self.pop),
                                   where=kpop[self.pidx] > 0)
        self.surface = {KHMER: P - M}
        for n in FOREIGN:
            self.surface[n] = P

        # Cham: a province's minority speakers up to its Muslims. Cambodia's Muslims are mostly
        # Cham, but not all: some speak Khmer (Battambang has 13,960 Muslims and 5,705 minority
        # speakers in all), and Tbong Khmum has more Muslims than minority speakers
        cham = np.minimum(mus, M)
        self.surface[CHAM] = cham
        R = M - cham     # the other minority languages' speakers per province

        nat = _national()
        minority = nat[(nat["source_id"] == "kh_gpcc_2019_em_t23") & (nat["node"] != CHAM)]
        langs = list(minority["node"])
        N = minority["count"].to_numpy(dtype=float)
        g = pd.read_csv(ROOT / "data" / "raw" / "glottolog" / "languages.csv",
                        usecols=["ID", "Latitude", "Longitude"]).set_index("ID")
        cent = place.geometry.representative_point()
        hx, hy = cent.x.to_numpy(), cent.y.to_numpy()
        seed = np.ones((25, len(langs)))
        self.how = {CHAM: "Muslims per province, up to its minority speakers"}
        for j, n in enumerate(langs):
            if n in GLOTTO:
                lat, lon = g.loc[GLOTTO[n], ["Latitude", "Longitude"]].astype(float)
                d = np.hypot((hx - lon) * 111.32 * np.cos(np.radians(lat)), (hy - lat) * 110.57)
                k = np.bincount(self.pidx, weights=self.pop * np.exp(-d / KERNEL_KM),
                                minlength=25) / np.where(kpop > 0, kpop, 1)
                seed[:, j] = (1 - LAMBDA) / 25 + LAMBDA * k / k.sum()
                self.how[n] = f"Glottolog {GLOTTO[n]}"
            else:
                seed[:, j] = 1.0 / 25
                self.how[n] = "even"
        X = _ipf(seed, R, N)
        if np.abs(X.sum(1) - R).max() > 1 or (X < 0).any():
            raise SystemExit("kh: the rake does not meet the provinces' remaining minority "
                             "speakers")
        for j, n in enumerate(langs):
            self.surface[n] = X[:, j]
        self.provs, self.langs = provs, [CHAM] + langs
        self.X = np.column_stack([cham, X])
        missing = set(kh2019.NAMES.values()) - set(self.surface)
        if missing:
            raise SystemExit(f"kh: no placement surface for {sorted(missing)}")
        self.n = 0

    def weights(self, node, idx, count, plain=False):
        self.n += 1
        w = self.surface[node][self.pidx[idx]] * self.hex_share[idx]
        return w if w.sum() > 0 else None

    def table(self):
        """The raked province x language table, for sources/kh.md and a look by eye."""
        return pd.DataFrame(self.X, index=self.provs, columns=self.langs)

    def summary(self):
        return (f"{self.n} languages placed: Khmer on population less minority speakers, "
                f"foreign languages on population, {len(self.langs)} minority languages on "
                f"Table 2.2's speakers per province raked to Table 2.3 "
                f"({sum(v.startswith('Glottolog') for v in self.how.values())} seeded by "
                f"Glottolog point, Cham by Muslims, {sum(v == 'even' for v in self.how.values())} "
                f"even)")


ENTRY = dict(
    name="Cambodia",
    source="General Population Census of Cambodia 2019: final report Table 2.7.1, and Ethnic "
           "Minorities in Cambodia Tables 2.2 and 2.3 (National Institute of Statistics)",
    how="census, 2019, mother tongue",
    parts=[
        dict(covers="Cham and the highland minority languages",
             source="2019 census, mother tongue, Ethnic Minorities in Cambodia Table 2.3",
             people=455_610),
        dict(covers="Everyone else", source="2019 census, mother tongue, Table 2.7.1",
             rest=True),
    ],
    grain="the whole country, 15.6 million people, placed by province from census figures",
    gap="Cambodians working abroad, whom the census leaves out (several hundred thousand)",
    view=[102.2, 9.9, 107.8, 14.8],
    counts=_counts,
    mappings=["kh2019"],
    place=RD_GEO / "kh" / "kh_hexes.gpkg",
    place_unit=_place_unit,
    place_weight=lambda place: _KhWeighter(place),
    note_public=(
        "The census published mother tongue for the whole country only, so where each language "
        "is drawn is an estimate. Per province it counts only how many people speak any "
        "minority language. Cham is placed by each province's Muslims and the highland "
        "languages towards where Glottolog puts them. Vietnamese and Chinese follow the "
        "population, though both live mostly in towns and along the rivers."),
)
