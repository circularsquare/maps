# Indonesia. Sensus Penduduk 2010, language used daily at home, persons aged 5+, by the 33
# provinces of 2010 (sources/id_sp2010.py, sources/id_shareout.py). Placed on religiondots' Kontur
# hexes, whose unit ids (kecamatan, regency, and one 2-digit residual) all begin with their BPS
# province code. Read-only. Inside a province, each language is placed on the kecamatan by a rake
# seeded by its Glottolog point(s) (AGENT_BRIEF §4.4: placement only, the counts stay the
# province's). The record is sources/id.md.
from _shared import *  # noqa: F401,F403
import numpy as np

PROVINCES_2010 = {"11", "12", "13", "14", "15", "16", "17", "18", "19", "21", "31", "32", "33",
                  "34", "35", "36", "51", "52", "53", "61", "62", "63", "64", "71", "72", "73",
                  "74", "75", "76", "81", "82", "91", "94"}

KERNEL_KM = 30.0    # how far a language's Glottolog point pulls its speakers inside a province
LAMBDA = 0.998      # share of the seed that follows that pull; the rest is even (at 0.9 the
                    # even part over a whole province outweighed the homeland: Toraja drew 27% of
                    # Tana Toraja, now 39%)
FAR_KM = 150.0      # a province whose people all live further than this from every point of a
                    # language (migrants) places that language evenly
MAJORITY = 0.4      # a language with this share of its province or more (Acehnese in Aceh,
                    # Balinese in Bali) is placed evenly: one point cannot stand for a homeland
                    # that is most of the province, and the minorities' pulls place it


def _counts():
    import id2010
    df = pd.read_csv(NORM / "id.csv", dtype={"geo_id": str})
    if set(df["geo_id"]) != PROVINCES_2010:
        raise SystemExit(f"id.csv: provinces {sorted(set(df['geo_id']) ^ PROVINCES_2010)}")
    df["node"] = df["source_category"].map(id2010.resolve)
    unresolved = sorted(set(df.loc[df["node"].isna(), "source_category"]) - id2010.EXCLUDED)
    if unresolved:
        raise SystemExit(f"id.csv categories that resolve to nothing: {unresolved}")
    df = df[df["node"].notna()].copy()
    df["unit"] = df["geo_id"]
    # the share-out of "other regional languages" (ask 009; sources/id_shareout.py) is
    # `modelled`; the source script asserts it sums to each province's measured remainder
    if set(df["tier"]) != {"measured", "modelled"}:
        raise SystemExit(f"id.csv: tiers {sorted(set(df['tier']))}")
    return df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()


def _province(g):
    # religiondots' units: 7-digit kecamatan and 4-digit regency codes, which begin with their
    # province's code, and "65", the Kalimantan Utara residual (its five regencies were carved
    # out of Kalimantan Timur in 2012, so in 2010 they are Kalimantan Timur, 64). The layer's
    # own unit is kept as `kec` for the weighter before scatter.py overwrites `unit`.
    u = g["unit"].astype(str)
    if not u.str.fullmatch(r"\d{2}|\d{4}|\d{7}").all():
        raise SystemExit("id_hexes.gpkg: a unit id is not a 2-, 4- or 7-digit BPS code")
    g["kec"] = u
    p = u.str[:2].replace({"65": "64"})
    if set(p) != PROVINCES_2010:
        raise SystemExit(f"id_hexes.gpkg: provinces {sorted(set(p) ^ PROVINCES_2010)}")
    return p


def _ipf(seed, rows, cols, iters=500):
    x = seed.copy()
    cols = cols * rows.sum() / cols.sum()
    for _ in range(iters):
        rs = x.sum(1)
        x *= np.divide(rows, rs, out=np.zeros_like(rows), where=rs > 0)[:, None]
        cs = x.sum(0)
        x *= np.divide(cols, cs, out=np.zeros_like(cols), where=cs > 0)[None, :]
    rs = x.sum(1)
    return x * np.divide(rows, rs, out=np.zeros_like(rows), where=rs > 0)[:, None]


# Measured languages with a Glottolog point that can stand for their homeland where they are a
# minority of the province: Minangkabau (Padang) in Riau, Jambi, Bengkulu and North Sumatra;
# Buginese (Bone) in South, West and Southeast Sulawesi. Javanese's and Sundanese's points sit in
# the middle of their own provinces and pulled them the wrong way in their neighbours; Malay and
# Banjar have none that stands for a homeland (Banjar's is in East Kalimantan's interior).
MEASURED_POINTS = {"austronesian.malayic.minangkabau": ["mina1268"],
                   "austronesian.buginese": ["bugi1244"]}
# Madurese's point is on Madura, but a kernel from it filled Surabaya (59%) before the far end of
# the island. Its seed is instead the island itself, the regencies of Bangkalan, Sampang,
# Pamekasan and Sumenep, against a low even seed elsewhere in East Java, so the half of East
# Java's Madurese speakers who do not fit on Madura spread with the population.
REGION_SEED = {"austronesian.madurese": ({"3526", "3527", "3528", "3529"}, 0.02)}


def node_points():
    """node -> Glottolog codes, from the share-out's languages (sources/id_shareout.py LANG)
    through taxonomy/id2010.py, and MEASURED_POINTS. Indonesian, Malay, Banjar, `other`, sign
    language and the unnamed remainder have none and follow population."""
    sys.path.insert(0, str(ROOT / "sources"))
    import id2010
    import id_shareout
    g = id_shareout.glottolog()
    out = dict(MEASURED_POINTS)
    for key, (label, _) in id_shareout.LANG.items():
        if id2010.SHARE + label not in id2010.NAMES:    # Papua's old clusters: stage 0 only
            continue
        cs = id_shareout.codes(key, g)
        if cs:
            out[id2010.resolve(id2010.SHARE + label)] = cs
    return out, g


# Indonesian New Guinea (2026-10-06; sources/id_papua.py). In the two provinces of 2010 each
# Papuan language is seeded by regency on id_papua's table (its share of the regency's people,
# from Ananta et al. 2016's Papuan share and largest group per regency), and inside the regency
# by REG_LOCAL of its seed following its Glottolog point. Javanese follows the same paper's
# Javanese share per regency, the other migrant languages the non-Papuan share; Indonesian,
# Papuan Malay, sign language and the unnamed remainder stay even.
PAPUA = {"91", "94"}
REG_LOCAL = 0.75
EVEN_IN_PAPUA = {"austronesian.malayic.indonesian", "austronesian.malayic.trade_malay.papuan",
                 "signlanguage", "indonesia_other"}


def _papua_seeds():
    """node -> {regency: share of its people}; regency -> (Papuan share, Javanese share)."""
    sys.path.insert(0, str(ROOT / "sources"))
    import id_papua
    t = pd.read_csv(NORM / "id_papua_regency.csv", dtype={"regency": str})
    by = {n: d.set_index("regency")["share"].to_dict() for n, d in t.groupby("node")}
    # Biak Numfor's Javanese share is misprinted in Table 4: the non-Papuan share x the
    # Javanese part of the non-Papuan people across both provinces' regencies that print one
    rs = [r for r, v in id_papua.REGENCY.items() if v[4] is not None]
    ratio = (sum(id_papua.REGENCY[r][4] for r in rs)
             / sum(100 - id_papua.REGENCY[r][1] for r in rs))
    reg = {r: (v[1] / 100, (v[4] if v[4] is not None else (100 - v[1]) * ratio) / 100)
           for r, v in id_papua.REGENCY.items()}
    return by, reg


# Indonesian by regency (2026-10-09; sources/id_lf2020.py, sources/id.md §11). The 2010 counts
# are by province only, and Indonesian had an even seed, so Jakarta's suburbs in West Java and
# Banten drew their province's rural mix and the map had an edge at the DKI border. The 2020
# long form's regency share of people who use no regional language in the family (`tidak`:
# Indonesian or foreign) now seeds Indonesian and foreign languages by t, and every regional
# language by 1 - t, so a province's split moves by one log-odds shift to meet the 2010 counts.
# Not in Papua and Papua Barat, which have their own regency seeds.
LF2020_T = (0.005, 0.995)   # t clipped so no regency closes to either side
TIDAK_NODES = {"austronesian.malayic.indonesian", "other"}
EVEN_NODES = {"signlanguage"}
# regencies created after 2010, folded into the 2010 regency they came out of (the hex layer's
# units are 2010's); Kalimantan Utara's five are the layer's residual unit "65"
LF2020_PARENT = {"1612": "1603", "1613": "1605", "1813": "1801", "3218": "3207", "5321": "5306",
                 "6411": "6402", "7211": "7201", "7212": "7203", "7411": "7404", "7412": "7403",
                 "7413": "7402", "7414": "7401", "7415": "7401", "7606": "7604", "8208": "8203",
                 "9111": "9105", "9112": "9105", "6501": "65", "6502": "65", "6503": "65",
                 "6504": "65", "6571": "65"}


def _lf2020_tidak():
    """2010 regency code (and "65") -> 2020 share of people 5+ using no regional language at
    home."""
    t = pd.read_csv(NORM / "id_lf2020_regency.csv", dtype={"regency": str})
    t["reg"] = t["regency"].map(lambda r: LF2020_PARENT.get(r, r))
    g = t.groupby("reg")[["tidak", "total"]].sum()
    return (g["tidak"] / g["total"]).to_dict()


class _IdWeighter:
    """Where inside its province each language's dots go. A placement weight only: every count
    is the province's.

    Per province, a kecamatan x language table is raked (IPF) to the kecamatan's Kontur
    population (scaled to the province's drawn total) and the province's count of each
    language, from a seed: for a language with Glottolog points, (1 - LAMBDA) + LAMBDA x
    exp(-distance to its nearest point / KERNEL_KM), normalised over the province; for the
    measured eight, Indonesian, foreign languages, sign language, the unnamed remainder, and any
    language whose points lie more than FAR_KM from all of a province's people, an even seed.
    So a minority language gathers towards its homeland, and the languages with no point fill in
    what is left. Inside a kecamatan, every language follows Kontur's population."""

    def __init__(self, place):
        self.pop = place["pop"].to_numpy(dtype=float)
        kec = place["kec"].astype(str).to_numpy()
        prov = place["unit"].astype(str).to_numpy()
        uk = sorted(set(kec))
        ui = {u: i for i, u in enumerate(uk)}
        self.uidx = np.array([ui[u] for u in kec])
        nU = len(uk)
        upop = np.bincount(self.uidx, weights=self.pop, minlength=nU)
        uprov = pd.Series(prov).groupby(self.uidx).first().reindex(range(nU)).to_numpy()
        ureg = np.array([u[:4] for u in uk])
        cen = pd.read_csv(GEO / "id" / "id_units.csv", dtype={"unit": str}).set_index("unit")
        if set(cen.index) != set(uk):
            raise SystemExit("data/geo/id/id_units.csv is not this layer's units; run "
                             "sources/id_units.py")
        lon = cen.loc[uk, "lon"].to_numpy(dtype=float)
        lat = cen.loc[uk, "lat"].to_numpy(dtype=float)

        pts, g = node_points()
        dmin = {}
        for n, cs in pts.items():
            best = np.full(nU, np.inf)
            for la, lo in g.loc[cs, ["Latitude", "Longitude"]].astype(float).to_numpy():
                d = np.hypot((lon - lo) * 111.32 * np.cos(np.radians(la)), (lat - la) * 110.57)
                best = np.minimum(best, d)
            dmin[n] = best

        counts = _counts().groupby(["unit", "node"])["count"].sum()
        pap_by, pap_reg = _papua_seeds()
        tidak = _lf2020_tidak()
        missing = sorted({r for r, p in zip(ureg, uprov) if p not in PAPUA} - set(tidak))
        if missing:
            raise SystemExit(f"id: regencies with no 2020 home-language share: {missing}")
        self.n_papua = 0
        self.share = {}          # node -> per-unit people of that language per Kontur person
        self.n_point = self.n_even = 0
        for p in sorted(PROVINCES_2010):
            us = np.where(uprov == p)[0]
            c = counts.loc[p]
            nodes = list(c.index)
            cols = c.to_numpy(dtype=float)
            rows = upop[us] * cols.sum() / upop[us].sum()
            seed = np.ones((len(us), len(nodes)))
            for j, n in enumerate(nodes):
                if p in PAPUA:
                    seed[:, j] = self._papua_seed(n, ureg[us], dmin.get(n, None),
                                                  us, pap_by, pap_reg)
                    self.n_papua += 1
                    continue
                if n in REGION_SEED and np.isin(ureg[us], list(REGION_SEED[n][0])).any():
                    seed[:, j] = np.where(np.isin(ureg[us], list(REGION_SEED[n][0])), 1.0,
                                          REGION_SEED[n][1])
                    self.n_point += 1
                elif (n in dmin and dmin[n][us].min() <= FAR_KM
                        and cols[j] < MAJORITY * cols.sum()):
                    k = np.exp(-dmin[n][us] / KERNEL_KM)
                    seed[:, j] = (1 - LAMBDA) + LAMBDA * k / k.max()
                    self.n_point += 1
                else:
                    self.n_even += 1
            if p not in PAPUA:
                t = np.clip([tidak[r] for r in ureg[us]], *LF2020_T)
                for j, n in enumerate(nodes):
                    if n in TIDAK_NODES:
                        seed[:, j] *= t
                    elif n not in EVEN_NODES:
                        seed[:, j] *= 1 - t
            X = _ipf(seed, rows, cols)
            if np.abs(X.sum(0) - cols).max() > max(1.0, 1e-6 * cols.sum()):
                raise SystemExit(f"id: the placement rake does not meet province {p}'s counts")
            per = np.divide(X, upop[us][:, None], out=np.zeros_like(X),
                            where=upop[us][:, None] > 0)
            for j, n in enumerate(nodes):
                v = self.share.setdefault(n, np.zeros(nU))
                v[us] = per[:, j]

    @staticmethod
    def _papua_seed(n, regs, dmin, us, pap_by, pap_reg):
        if n in pap_by:
            share = np.array([pap_by[n].get(r, 0.0) for r in regs])
            if dmin is not None:
                k = np.exp(-dmin[us] / KERNEL_KM)
                local = np.zeros_like(k)
                for r in set(regs):
                    m = regs == r
                    local[m] = k[m] / k[m].max() if k[m].max() > 0 else 1.0
                share = share * ((1 - REG_LOCAL) + REG_LOCAL * local)
            return share + 1e-9
        if n in EVEN_IN_PAPUA:
            return np.ones(len(regs))
        if n == "austronesian.javanese":
            return np.array([pap_reg[r][1] for r in regs]) + 1e-6
        # every other language here is a migrant's: Bugis, Makassarese, Butonese, Torajan ...
        return np.array([1 - pap_reg[r][0] for r in regs]) + 1e-6

    def weights(self, node, idx, count, plain=False):
        w = self.share[node][self.uidx[idx]] * self.pop[idx]
        return w if w.sum() > 0 else None

    def summary(self):
        return (f"placed inside provinces by kecamatan rakes: {self.n_point} (province, "
                f"language) pairs seeded by Glottolog points, {self.n_even} even, "
                f"{self.n_papua} in Papua and Papua Barat by regency; elsewhere Indonesian and "
                f"regional languages by the 2020 regency share using no regional language")


ENTRY = dict(
    name="Indonesia",
    source=("Sensus Penduduk 2010, Kewarganegaraan, Suku Bangsa, Agama, dan Bahasa Sehari-hari "
            "Penduduk Indonesia, Tables L4.1, L4.2, L4.5 and L2.6 (Badan Pusat Statistik, "
            "2011); Ananta et al., Demography of Indonesia's Ethnicity (2015), and IUSSP 2013 "
            "paper; Ananta, Utami and Handayani, Statistics on Ethnic Diversity in the Land of "
            "Papua, Indonesia (2016); Joshua Project; Glottolog; Sensus Penduduk 2020 Long Form, "
            "table 201, regional language used in the family, by regency (placement only)"),
    how=("census, 2010, language used daily at home, eight languages by province; the other "
         "regional languages, 20%, estimated from the census's ethnic groups and national "
         "language table; Papuan languages in Papua and Papua Barat estimated by regency"),
    parts=[
        dict(covers="Indonesian, seven large regional languages, foreign languages",
             source="2010 census, language used daily at home, by province",
             people=170_935_033),
        dict(covers="Other regional languages",
             source="the census's national language groups, shared by province through its "
                    "ethnic groups, Ananta et al.'s counts (2015) and own-language retention "
                    "(IUSSP 2013)",
             people=40_916_879),
        dict(covers="Papuan languages",
             source="Ananta, Utami and Handayani (2016) by regency, Joshua Project speaker "
                    "estimates, Glottolog",
             rest=True),
    ],
    grain=("33 provinces, 6.5 million people on average; inside a province each language is "
           "placed by where Glottolog puts it, and Indonesian by regency from the 2020 census; "
           "in Papua and Papua Barat, by regency"),
    gap=("22.7 million children under five, who were not asked; 905,695 people counted on the "
         "census's shorter forms, which did not ask it; and 561,711 who gave no answer"),
    view=[95.0, -11.0, 141.0, 6.1],
    counts=_counts,
    mappings=["id2010"],
    place=RD_GEO / "id" / "id_hexes.gpkg",
    place_unit=_province,
    place_weight=lambda place: _IdWeighter(place),
    note_public=(
        "Badan Pusat Statistik published this question by province for only the eight "
        "languages most used at home. The other regional languages, a fifth of the country, "
        "are an estimate. The census counted them nationally in 24 groups, and each province's "
        "total is shared among those groups using the census's ethnic groups per province and "
        "the share of each group that speaks its own language at home. A group is then split "
        "into languages by the national size of its peoples, as Ananta and colleagues counted "
        "them from the same census. The trade Malay of each eastern province is named for that "
        "province. About 2 million people, of peoples nobody has counted one by one, are drawn "
        "as other regional languages of Indonesia. Inside a province, each regional language "
        "is drawn towards where Glottolog places it, and the large languages fill the rest by "
        "population, so a district's mix is an estimate. Indonesian is drawn more thickly in "
        "the regencies where the 2020 census found more families using no regional language at "
        "home, such as Jakarta's suburbs, and the province's totals stay those of 2010. In Papua and Papua Barat (the 2010 "
        "provinces), the people who spoke a Papuan language at home are shared among about 230 "
        "languages by regency, using Ananta and colleagues' 2016 count of the largest groups "
        "and Joshua Project's speaker estimates for the rest. A household that used Indonesian "
        "at home was recorded as Indonesian, whatever else it spoke."),
)
