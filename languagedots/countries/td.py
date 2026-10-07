# Chad. RGPH2 2009, first national language spoken, published for the whole country only
# (sources/td_rgph.py, Tableau 5.10). One unit, Chad; the national counts are placed inside it by
# région (AGENT_BRIEF §4.4), on religiondots' Kontur hexes for the 22 régions of 2009,
# read-only. The record is sources/td.md.
from _shared import *  # noqa: F401,F403
import numpy as np

UNIT = "TD"
TOTAL = 8_088_816          # aged 6+ naming a first national language (Tableau 5.10)

# The one-line switch (ask 018's trap). "first_language": Chadian Arabic drawn at the Arab
# ethnic share of Tableau 5.02, the excess moved back to the languages of the groups whose
# own-language counts fall short of their ethnic counts (sources/td.md §3). "as_printed":
# Tableau 5.10 as the census prints it (Arabic 21.8%).
ARABIC = "first_language"

KERNEL_KM = 50.0    # how far a language's Glottolog points pull its speakers between régions
LAMBDA = 0.95       # share of a language's seed that follows that pull; the rest is even
LEAK = 0.15         # a language's seed among the other religious half of a région

AR = "Arabe local"
FULA = "Peul/Foulfouldé/Bodoré"
BET = {"Borkou", "Ennedi", "Tibesti"}
AUTRES = "Autres 1ere langues nationales parlées"

# Tableau 5.02's ethnic groups -> the Tableau 5.10 rows that name their languages, and whether
# some of the group's languages are filed in AUTRES (Annexes 2-3). Arabe and the groups of
# foreign origin are left out of the move.
GROUPS = {
    "Gorane": (["Gorane"], True),                     # Téda: Toubou is in Autres
    "Baguirmi/Barma et autres": (["Barma/Baguirmi", "Toumak/Ndom"], True),
    "Kanembou/Bornou/Boudouma": (["Kanembou"], True),  # Kanouri, Boudouma in Autres
    "Boulala/Médégo/Kouka": (["Boulala"], True),
    "Ouaddaï/Maba/Massalit/Mimi": (["Maba/Ouaddaï", "Massalit", "Mimi"], False),
    "Zaghawa (Bideyat/Kobé)": (["Zaghawa/Béri/Bideyat"], False),
    "Dadjo/Kibet/Mouro et autres": (["Dadjo"], True),
    "Bidio/Migami/Kinga/dangléat et autres": (["Moubi"], True),
    "Moundang": (["Moundang"], False),
    "Massa/Mousseye/Mousgoume": (["Massa", "Mousseye"], True),   # Mousgoum not printed
    "Toupouri/Kéra": (["Toupouri", "Kéra"], False),
    "Sara (Ngambaye/Sara Madjingaye/Mbaye et autres)": (["Sara", "Sara Kaba", "Daye", "Mboum"], False),
    "Peul/Foulbé/Bodoré": (["Peul/Foulfouldé/Bodoré"], False),
    "Tama/Assongori/Mararit": (["Tama"], True),
    "Gabri/Kabalaye/Nangtchéré/Soumraye et autres": (["Gabri", "Kabalaye", "Nangtchéré"], True),
    "Marba/Lélé/Mesmé": (["Marba", "Lélé", "Mesmé"], False),
    "Mesmedjé/Massalat/Kadjaksé": ([], True),
    "Karo/Zimé/Pévé": (["Karo/Kado", "Lamé/Pévé"], False),
    "Autres ethnies tchadiennes (Achit/Banda/Kim et autres)": (["Rounga", "Kim"], True),
}

# Which languages follow a région's Muslims (M) and which its non-Muslims (N) in the seed
MUSLIM = {AR, "Gorane", "Kanembou", "Maba/Ouaddaï", "Boulala", "Zaghawa/Béri/Bideyat",
          "Peul/Foulfouldé/Bodoré", "Barma/Baguirmi", "Massalit", "Mimi", "Rounga", "Tama",
          "Dadjo", "Moubi"}

# Glottolog points per language (data/raw/glottolog/languages.csv); none for Arabic, which has
# its own seed below, nor for Fula: herders across the Sahel and the south, so an even seed,
# a fifth of it in the BET desert régions (RGPH 1993 Tableau 30: under 2.6% there). Autres: the
# languages Annexe 3 lists in it that Glottolog places in Chad (Teda, Kanuri, Kotoko, Buduma,
# the Hadjaraï languages, Massalat, Kajakse, Masmaje, Niellim, Bua, Tunia, Ndam, Assangori,
# Mararit, Kibet, Toram, Somrai, Besme, Gidar), one point each.
GLOTTO = {
    AUTRES: ["teda1241", "cent2050", "lagw1237", "budu1265", "bidi1241", "dang1274",
             "miga1249", "keng1240", "soko1263", "mogu1251", "jonk1238", "muku1242",
             "saba1276", "mass1262", "kaja1254", "masm1239", "niel1243", "buaa1245",
             "tuni1251", "ndam1251", "assa1269", "mara1396", "kibe1241", "tora1267",
             "somr1248", "besm1235", "gida1247"],
    "Sara": ["ngam1268", "sarr1246", "mbay1241", "gula1268", "gorr1238", "laka1254",
             "mang1398", "bedj1245", "ngam1269", "kaba1281"],
    "Gorane": ["daza1242"],
    "Kanembou": ["kane1243"],
    "Maba/Ouaddaï": ["maba1277"],
    "Moundang": ["mund1325"],
    "Mousseye": ["muse1242"],
    "Boulala": ["naba1253"],
    "Zaghawa/Béri/Bideyat": ["zagh1240"],
    "Marba": ["marb1239"],
    "Massa": ["masa1322"],
    "Barma/Baguirmi": ["bagi1246"],
    "Massalit": ["nucl1440"],
    "Mimi": ["mimi1241", "mimi1240"],
    "Rounga": ["rung1258"],
    "Tama": ["tama1331"],
    "Dadjo": ["dard1243", "dars1235"],
    "Moubi": ["mubi1246"],
    "Mesmé": ["mesm1239"],
    "Gabri": ["gabr1253"],
    "Kabalaye": ["kaba1292"],
    "Kéra": ["kera1255"],
    "Kim": ["kimm1246"],
    "Lamé/Pévé": ["peve1243"],
    "Lélé": ["lele1276"],
    "Nangtchéré": ["nanc1253"],
    "Toupouri": ["tupu1244"],
    "Karo/Kado": ["herd1236", "nget1241"],
    "Daye": ["dayy1236"],
    "Mboum": ["kara1478", "nzak1246"],
    "Sara Kaba": ["sara1321", "sara1322"],
    "Toumak/Ndom": ["tuma1260"],
}

# Arabic's seed: the Arab share of each 1993 préfecture where RGPH 1993's Tableau 30 (État de
# la population, p117) prints it among the three largest groups, or the bound it sets when
# Arabs are not among them; the 2009 régions inherit their préfecture's figure. Préfectures
# on the missing pp118-119 of the only scan get the mean of the printed northern figures
# times the région's Muslim share, except Salamat, which the text calls Arab-predominant and
# gets Batha's 33.6, the lowest printed figure of the préfectures it names so.
AR1993 = {
    "Batha": 33.6, "Borkou": 2.6, "Ennedi": 2.6, "Tibesti": 2.6, "Wadi Fira": 9.6,
    "Chari Baguirmi": 37.7, "Hadjer Lamis": 37.7, "Guéra": 21.1, "Kanem": 5.0,
    "Barh El Gazal": 5.0, "Lac": 1.9, "Logone Occidental": 1.2, "Logone Oriental": 0.9,
    "Mayo Kebbi Est": 6.0, "Mayo Kebbi Ouest": 6.0, "Salamat": 33.6,
}
AR1993_NORTH_MEAN = (33.6 + 2.6 + 9.6 + 37.7 + 21.1 + 5.0 + 1.9) / 7


def _table():
    df = pd.read_csv(NORM / "td.csv")
    lang = df[df["geo_level"] == "country"].set_index("source_category")["count"]
    eth = df[df["geo_level"] == "country_ethnic"].set_index("source_category")["count"]
    reg = df[df["geo_level"] == "region"].set_index("geo_id")["count"]
    if lang.get("Total") != TOTAL or len(reg) != 22:
        raise SystemExit("td.csv is not the expected table -- re-run sources/td_rgph.py")
    return lang.drop("Total").astype(float), eth.astype(float), reg.astype(float)


def _move(lang, eth):
    """Tableau 5.10 counts -> (measured, derived) per label under ARABIC."""
    measured = lang.copy()
    derived = pd.Series(0.0, index=lang.index)
    info = {}
    if ARABIC == "as_printed":
        return measured, derived, info
    if ARABIC != "first_language":
        raise SystemExit(f"td: ARABIC is {ARABIC!r}")
    etot = 10_666_833
    X = lang[AR] - eth["Arabe"] / etot * TOTAL
    D = {g: max(0.0, eth[g] / etot * TOTAL - lang[names].sum()) for g, (names, _) in GROUPS.items()}
    clean = sum(D[g] for g, (_, m) in GROUPS.items() if not m)
    mixed = sum(D[g] for g, (_, m) in GROUPS.items() if m)
    if clean >= X:
        sc, rho = X / clean, 0.0
    else:
        sc, rho = 1.0, min(1.0, (X - clean) / mixed)
    for g, (names, m) in GROUPS.items():
        R = D[g] * (rho if m else sc)
        if R <= 0:
            continue
        L = lang[names].sum() if names else 0.0
        to_named = R * (L / (L + (1 - rho) * D[g]) if m else 1.0) if names else 0.0
        for n in names:
            derived[n] += to_named * lang[n] / L
        derived[AUTRES] += R - to_named
    measured[AR] -= derived.sum()
    info = dict(X=X, clean=clean, mixed=mixed, rho=rho, moved=derived.sum())
    return measured, derived, info


def _counts():
    import td2009
    lang, eth, _ = _table()
    measured, derived, _ = _move(lang, eth)
    rows = []
    for lab in lang.index:
        for tier, s in (("measured", measured), ("derived", derived)):
            if round(s[lab]) > 0:
                rows.append(dict(unit=UNIT, node=td2009.resolve(lab), count=int(round(s[lab])),
                                 tier=tier))
    df = pd.DataFrame(rows)
    drift = TOTAL - df["count"].sum()
    if abs(drift) > 40:
        raise SystemExit(f"td: drawn total off {TOTAL:,} by {drift}")
    # put the rounding drift on Arabic's measured row so the total is the census's exactly
    i = df.index[(df["node"] == td2009.resolve(AR)) & (df["tier"] == "measured")][0]
    df.loc[i, "count"] += drift
    return df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()


def _place_unit(g):
    # the layer's own `unit` is the région; keep it as `reg` for the weighter before
    # scatter.py overwrites `unit` with what this returns
    g["reg"] = g["unit"].astype(str)
    return pd.Series(UNIT, index=g.index)


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


class _TdWeighter:
    """Where inside Chad each language's dots go. A placement weight only: every count is the
    census's national figure either way.

    Each région is split into its Muslims and everyone else (Tableau 5.07 through religiondots'
    td.csv), and that 44-row x language table is raked (IPF) to those populations (scaled to the
    6+ universe) and to the languages' national counts, from a seed: the northern and central
    languages sit with Muslims and the southern ones with everyone else, each leaking LEAK into
    the other half (the census's southern-language total exceeds its non-Muslims, so some must);
    times nearness to the language's Glottolog points (Autres: the points of the languages
    Annexe 3 files there); Arabic's nearness is replaced by the 1993 census's Arab share by
    préfecture, and N'Djaména's by Tableau 5.10's urban column. Inside a région, every language
    follows Kontur's population."""

    def __init__(self, place):
        import td2009
        lang, eth, reg = _table()
        measured, derived, self.info = _move(lang, eth)
        tot = measured + derived
        self.pop = place["pop"].to_numpy(dtype=float)
        rg = place["reg"].astype(str).to_numpy()
        regs = sorted(set(rg))
        if set(regs) != set(reg.index):
            raise SystemExit(f"td: régions on the layer {sorted(set(regs) ^ set(reg.index))} "
                             "differ from td.csv's")
        ri = {r: i for i, r in enumerate(regs)}
        self.ridx = np.array([ri[r] for r in rg])
        nR = len(regs)
        kpop = np.bincount(self.ridx, weights=self.pop, minlength=nR)
        self.hex_share = np.divide(self.pop, kpop[self.ridx], out=np.zeros_like(self.pop),
                                   where=kpop[self.ridx] > 0)

        rd = pd.read_csv(RD / "data" / "normalized" / "td.csv")
        mus = rd[rd["source_category"] == "Musulmane"].set_index("geo_id")["count"]
        allp = rd.groupby("geo_id")["count"].sum()
        mshare = (mus / allp).reindex(regs).to_numpy(dtype=float)
        if np.isnan(mshare).any():
            raise SystemExit("td: a région lacks its Muslim share in religiondots' td.csv")

        g = pd.read_csv(ROOT / "data" / "raw" / "glottolog" / "languages.csv",
                        usecols=["ID", "Latitude", "Longitude"]).set_index("ID")
        cent = place.geometry.representative_point()
        hx, hy = cent.x.to_numpy(), cent.y.to_numpy()

        def near(codes):
            # each point's pull is normalised on its own, so a point in thinly peopled country
            # (Tedaga's, in Tibesti) counts as much as one in a crowded région
            k = np.zeros(nR)
            for gc in codes:
                lat, lon = g.loc[gc, ["Latitude", "Longitude"]].astype(float)
                d = np.hypot((hx - lon) * 111.32 * np.cos(np.radians(lat)), (hy - lat) * 110.57)
                kk = np.bincount(self.ridx, weights=self.pop * np.exp(-d / KERNEL_KM),
                                 minlength=nR) / np.where(kpop > 0, kpop, 1)
                k += kk / kk.sum()
            return (1 - LAMBDA) / nR + LAMBDA * k / k.sum()

        labs = list(lang.index)
        cols = tot[labs].to_numpy(dtype=float)
        scale = TOTAL / reg.sum()
        popr = reg.reindex(regs).to_numpy(dtype=float) * scale
        nd = ri["N'Djaména"]

        # N'Djaména is set first, to Tableau 5.10's urban column (it holds 40% of urban Chad),
        # with the Arabic move applied in the national proportions
        urban = self._urban(labs)
        printed = lang[labs].to_numpy(dtype=float)
        dv = derived[labs].to_numpy(dtype=float)
        v = urban.copy()
        ja = labs.index(AR)
        if dv.sum() > 0:
            cut = v[ja] * dv.sum() / printed[ja]
            v[ja] -= cut
            v += cut * dv / dv.sum()
        nd_row = popr[nd] * v / v.sum()
        cols = cols - nd_row
        if (cols < 0).any():
            raise SystemExit(f"td: N'Djaména's urban mix exceeds a national count: "
                             f"{[labs[j] for j in np.where(cols < 0)[0]]}")

        # the other 21 régions: rows 0..nR-1 a région's Muslims, nR..2nR-1 everyone else
        seed = np.zeros((2 * nR, len(labs)))
        for j, lab in enumerate(labs):
            if lab == AR:
                k = np.array([AR1993.get(r, AR1993_NORTH_MEAN * mshare[i]) / 100
                              for i, r in enumerate(regs)])
            elif lab == FULA:
                k = np.array([0.2 if r in BET else 1.0 for r in regs])
            else:
                k = near(GLOTTO[lab]) if lab in GLOTTO else np.ones(nR)
            k = k / k.sum()
            if lab == AUTRES:
                pm, pn = 1.0, 1.0
            elif lab in MUSLIM:
                pm, pn = 1.0, LEAK
            else:
                pm, pn = LEAK, 1.0
            seed[:nR, j] = k * pm
            seed[nR:, j] = k * pn
        seed[[nd, nR + nd], :] = 0
        rows = np.concatenate([popr * mshare, popr * (1 - mshare)])
        rows[[nd, nR + nd]] = 0
        X2 = _ipf(seed, rows, cols)
        if np.abs(X2.sum(1) - rows).max() > 1 or np.abs(X2.sum(0) - cols).max() > 50:
            raise SystemExit("td: the rake does not meet the régions and the national counts")
        X = X2[:nR] + X2[nR:]
        X[nd] = nd_row
        self.regs, self.labs, self.X, self.X2 = regs, labs, X, X2
        self.surface = {}
        for j, lab in enumerate(labs):
            n = td2009.resolve(lab)
            self.surface[n] = self.surface.get(n, 0) + X[:, j]
        self.n = 0

    @staticmethod
    def _urban(labs):
        import importlib.util
        spec = importlib.util.spec_from_file_location("td_rgph", ROOT / "sources" / "td_rgph.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return np.array([m.T510[lab][0] / 100 for lab in labs])

    def weights(self, node, idx, count, plain=False):
        self.n += 1
        w = self.surface[node][self.ridx[idx]] * self.hex_share[idx]
        return w if w.sum() > 0 else None

    def table(self):
        """The raked région x language table, for sources/td.md and a look by eye."""
        return pd.DataFrame(self.X, index=self.regs, columns=self.labs)

    def summary(self):
        return (f"{self.n} languages placed over 22 régions by a rake to Tableau 5.07 and the "
                f"national counts (Arabic {ARABIC}; seeds: Glottolog points x religious pool, "
                f"Arabic by RGPH 1993 Tableau 30, N'Djaména by Tableau 5.10's urban column)")


ENTRY = dict(
    name="Chad",
    source="RGPH2 2009, État et structures de la population (INSEED), Tableaux 5.10 and 5.02",
    how="census, 2009, first national language named, aged 6 and over; Arabic named first by "
        "non-Arabs moved back to their group's language",
    parts=[
        dict(covers="Everyone aged 6 and over",
             source="2009 census, first national language named, national figures",
             people=7_373_106),
        dict(covers="Non-Arabs who named Arabic first",
             source="2009 census ethnic table: Arabic above the Arab share returned to those "
                    "groups' own languages",
             rest=True),
    ],
    grain="the whole country, 8.1 million people aged 6 and over, placed across the 22 régions "
          "by religion, the 1993 Arab share and where each language is spoken",
    gap="2.4% of people aged 6 and over who named no national language (195,000), and the "
        "98,191 people in parts of Sila and Tibesti that enumerators could not reach",
    view=[13.4, 7.4, 24.0, 23.5],
    counts=_counts,
    mappings=["td2009"],
    place=RD_GEO / "td" / "td_hexes.gpkg",
    place_unit=_place_unit,
    place_weight=lambda place: _TdWeighter(place),
    note_public=(
        "Chad's 2009 census asked everyone aged 6 and over which national languages they "
        "speak, and published the first one named for the whole country only, so where each "
        "language is drawn is an estimate. The census counts 21.8% naming Chadian Arabic first "
        "but only 12.9% Arabs by ethnic group; the gap is people of other groups, mostly in the "
        "centre and east, who named the lingua franca first. Here Arabic is drawn at the Arab "
        "share and the rest returned to those groups' own languages, using the census's ethnic "
        "table. Each language is then shared out among the régions to match the census's "
        "Muslims and non-Muslims per région and where the language is spoken."),
)
