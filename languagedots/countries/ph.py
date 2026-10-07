# Philippines. 2020 CPH ethnicity by province and HUC (sources/ph_census.py), read as languages
# through taxonomy/ph2020.py after each group's non-speakers are moved to the local lingua franca
# by the census's national household-language table. Placed on religiondots' 42,042 barangays
# for the same 117 units (read-only); inside a unit small languages are raked towards their
# Glottolog points and the big regional ones, weakly, towards a neighbouring homeland
# (_PhWeighter, 2026-10-06, weakened the same day). The record is sources/ph.md.
from _shared import *  # noqa: F401,F403
import numpy as np


def _counts():
    import ph2020
    df = pd.read_csv(NORM / "ph.csv", dtype={"geo_id": str})
    if df["geo_id"].nunique() != 117:
        raise SystemExit("ph.csv: expected 117 provinces and cities; re-run sources/ph_census.py")
    df["node"] = df["source_category"].map(ph2020.resolve)
    missing = sorted(set(df.loc[df["node"].isna(), "source_category"]))
    if missing:
        raise SystemExit(f"ph.csv categories that resolve to nothing: {missing}")
    if set(df["tier"]) != {"derived", "modelled"}:
        raise SystemExit(f"ph.csv: tiers {sorted(set(df['tier']))}")
    df["unit"] = df["geo_id"]
    df = df[df["count"] > 0]
    return df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()


# ---- placement inside a province (AGENT_BRIEF §4.4; Anita 2026-10-06: province splits such as
# Negros Occidental all Hiligaynon beside Negros Oriental all Cebuano were the map's most visible
# false edge). Counts stay the province's; only where its barangays hold each language changes.
HOME = 0.4          # a language with this share of a unit makes that unit part of its homeland
MAJORITY = 0.4      # ... and is placed evenly there (by population) rather than pulled
# The neighbour pull, weakened 2026-10-06 (Anita: "the softening is kinda sus ... slightly weaken
# it ... especially for small languages in super mountainous areas we dont wanna dilute much"):
PULL_MIN = 1_000_000  # only the big regional languages (drawn by 1M+ people nationally: Tagalog,
                    # Cebuano, Boholano, Hiligaynon, Ilocano, Bikol, Waray, Kapampangan,
                    # Pangasinan, Maguindanao, Maranao, Tausug) are pulled towards a neighbouring
                    # homeland; every smaller language uses its Glottolog points or stays even
NEAR_KM = 15.0      # ... and only when the unit touches that homeland (was 25 km, which also
                    # reached across straits)
HOME_KERNEL_KM = 12.0
HOME_LAMBDA = 0.8   # seed floor 0.2: a barangay far from the line keeps at least a fifth of the
                    # seed of one on it (was 0.002, i.e. almost nothing)
BAND_KM = 15.0      # the cap: of a pulled language's speakers in a unit, the share in the
BAND_CAP = 2.0      # barangays within BAND_KM of the homeland is at most BAND_CAP times the
                    # share of the unit's people living there (lambda is halved until it holds)
UP_LO, UP_HI = 300.0, 700.0  # the pull fades out between these elevations (barangay, GEBCO;
                    # sources/ph_elev.py), so a lowland language is not pulled up into the
                    # Cordillera, the Mindanao highlands, or Mindoro's and Palawan's interiors
UPLAND_HOMES = {"austronesian.philippine.maranao"}  # homeland is itself upland (the Lanao
                    # plateau, ~700 m): no elevation fade
# And no neighbour pull at all inside a unit whose largest language is under PULL_MIN (Benguet,
# Mountain Province, Ifugao, Aklan, Tawi-Tawi... 12 units): there the big languages follow population.
# The Glottolog-point pull, for languages under PULL_MIN; unchanged, so small languages stay
# gathered where Glottolog places them:
KERNEL_KM = 12.0    # how quickly the pull fades with distance from the point
LAMBDA = 0.998      # share of the seed that follows the pull; the rest is even
FAR_KM = 150.0      # Glottolog points further than this from all of a unit's people: no pull

# Glottolog points for languages under PULL_MIN, looked up by name in
# data/raw/glottolog/languages.csv (2026-10-06), never from memory. A group node takes all its
# members' points. A child node with no entry takes its parent's. Not used: Glottolog's only
# Iranun point (phil1247) and Tausug point (taus1251) are in Sabah; it has no Ivatan point on
# Batanes at language level.
POINTS = {
    # smaller languages that hold 40%+ of some unit, which until the weakening were pulled
    # towards that unit instead
    "austronesian.philippine.kankanaey": ["kank1243", "nort2877"],
    "austronesian.philippine.kinaraya": ["kina1250"],
    "austronesian.philippine.aklanon": ["akla1240", "mala1491"],
    "austronesian.philippine.mansakan.mandaya": ["kara1489", "cata1284", "sang1338"],
    "austronesian.philippine.masbatenyo": ["masb1238"],
    "austronesian.philippine.romblomanon": ["romb1245"],
    "austronesian.philippine.surigaonon": ["suri1273"],
    "austronesian.sama_bajaw.sama": ["pang1291", "sout2918", "cent2092"],
    "austronesian.sama_bajaw.yakan": ["yaka1277"],
    "creole.spanish_based.chavacano": ["chav1241"],
    "austronesian.philippine.manobo": ["cina1236", "sara1327", "cota1241", "obom1235", "west2555",
                                       "ilia1236", "mati1250", "atam1240", "agus1235", "raja1254",
                                       "diba1242"],
    "austronesian.philippine.blaan": ["koro1310", "sara1326"],
    "austronesian.philippine.subanen": ["cent2089", "east2694", "nort2885", "west2557"],
    "austronesian.philippine.kolibugan": ["koli1253"],
    "austronesian.philippine.northern_luzon.itawis": ["itaw1240"],
    "austronesian.philippine.northern_luzon.ibanag": ["iban1267"],
    "austronesian.philippine.sambalic.sambal": ["boto1242", "tina1248"],
    "austronesian.philippine.tiruray": ["tiru1241"],
    "austronesian.sama_bajaw.bangingi": ["bala1311"],
    "austronesian.philippine.mansakan.tagakaulo": ["taga1268"],
    "austronesian.philippine.mansakan.kagan": ["kaga1255"],
    "austronesian.philippine.northern_luzon.ifugao": ["amga1235", "tuwa1243", "mayo1262", "bata1298"],
    "austronesian.philippine.northern_luzon.ifugao.tuwali": ["tuwa1243"],
    "austronesian.philippine.northern_luzon.kalanguya": ["ahin1234", "kaya1320", "tino1235"],
    "austronesian.philippine.palawano": ["cent2091", "broo1239", "sout2916"],
    "austronesian.philippine.asi": ["bant1288"],
    "austronesian.philippine.northern_luzon.kalinga": ["uppe1424", "limo1248", "lubu1243", "maba1279",
                                                       "butb1235", "lowe1412", "madu1248"],
    "austronesian.philippine.northern_luzon.kalinga.lubuagan": ["lubu1243"],
    "austronesian.philippine.northern_luzon.kalinga.butbut": ["butb1235"],
    "austronesian.philippine.northern_luzon.bontok": ["fina1242"],
    "austronesian.philippine.northern_luzon.itneg": ["inla1260", "bino1237", "masa1307", "moya1235",
                                                     "maen1235", "bana1288"],
    "austronesian.philippine.northern_luzon.itneg.inlaud": ["inla1260"],
    "austronesian.philippine.manobo.binukid": ["binu1244"],
    "austronesian.philippine.manobo.talaandig": ["binu1244"],
    "austronesian.philippine.manobo.higaonon": ["higa1237"],
    "austronesian.philippine.northern_luzon.ibaloi": ["ibal1244"],
    "austronesian.philippine.northern_luzon.isnag": ["isna1241"],
    "austronesian.philippine.northern_luzon.yogad": ["yoga1237"],
    "austronesian.philippine.northern_luzon.gaddang": ["gadd1244"],
    "austronesian.philippine.northern_luzon.isinai": ["isin1239"],
    "austronesian.philippine.tboli": ["tbol1240"],
    "austronesian.philippine.cuyonon": ["cuyo1237"],
    "austronesian.philippine.capiznon": ["capi1239"],
    "austronesian.philippine.mangyan.hanunoo": ["hanu1241"],
    "austronesian.philippine.mangyan.iraya": ["iray1237"],
    "austronesian.philippine.mangyan.alangan": ["alan1249"],
    "austronesian.philippine.molbog": ["molb1237"],
    "austronesian.philippine.tagbanwa": ["tagb1258"],
    "austronesian.philippine.ata": ["ataa1240"],
    "austronesian.philippine.agta.paranan": ["para1306"],
    "austronesian.philippine.sangil": ["sang1337"],
    "austronesian.philippine.mansakan.davawenyo": ["dava1245"],
    "austronesian.philippine.agta": ["vill1242", "umir1236", "dupa1235", "dica1235", "cama1250",
                                     "alab1246", "casi1235", "mtir1236", "mtir1235", "isar1235",
                                     "agta1234", "cent2084"],
}


def _ipf(seed, rows, cols, iters=300):
    x = seed.copy()
    for _ in range(iters):
        rs = x.sum(1)
        x *= np.divide(rows, rs, out=np.zeros_like(rows), where=rs > 0)[:, None]
        cs = x.sum(0)
        x *= np.divide(cols, cs, out=np.zeros_like(cols), where=cs > 0)[None, :]
    return x


class _PhWeighter:
    """Where inside its province or city each language's dots go. A placement weight only.

    Per unit, a barangay x language table is raked (IPF) to the barangays' population (scaled to
    the unit's drawn total) and the unit's count of each language, from a seed. A language with
    under MAJORITY of the unit gets a pulled seed:
    - a big regional language (PULL_MIN or more nationally) towards the nearest barangay of a
      neighbouring unit where it holds HOME or more, when one lies within NEAR_KM:
      (1 - HOME_LAMBDA) + HOME_LAMBDA x exp(-d / HOME_KERNEL_KM) x lowland(elevation), with
      HOME_LAMBDA halved until the BAND_CAP cap holds;
    - any smaller language towards its Glottolog points (when within FAR_KM):
      (1 - LAMBDA) + LAMBDA x exp(-d / KERNEL_KM).
    Everything else, the unit's main languages included, has an even seed and fills what the
    pulled languages leave. So Negros Occidental's Cebuano speakers lean towards Negros Oriental,
    and the unit's per-language counts are still the census's."""

    def __init__(self, place):
        from scipy.spatial import cKDTree
        self.pop = place["pop"].to_numpy(dtype=float)
        unit = place["unit"].astype(str).to_numpy()
        pt = place.geometry.representative_point()
        lat0 = 12.0
        kx, ky = 111.32 * np.cos(np.radians(lat0)), 110.57
        xy = np.column_stack([pt.x.to_numpy() * kx, pt.y.to_numpy() * ky])
        ef = GEO / "ph" / "ph_barangay_elev.csv"
        if not ef.exists():
            raise SystemExit(f"ph: {ef} missing; run python sources/ph_elev.py")
        elev = pd.read_csv(ef, dtype={"bgy": str}).set_index("bgy")["elev_m"]
        elev = place["bgy"].astype(str).map(elev)
        if elev.isna().any():
            raise SystemExit(f"ph: {elev.isna().sum()} barangays without an elevation; re-run sources/ph_elev.py")
        lowland = np.clip((UP_HI - elev.to_numpy(dtype=float)) / (UP_HI - UP_LO), 0.0, 1.0)

        counts = _counts().groupby(["unit", "node"])["count"].sum()
        share = counts / counts.groupby(level=0).transform("sum")
        national = counts.groupby(level=1).sum()
        self.pull_nodes = set(national[national >= PULL_MIN].index)
        home_units = share[share >= HOME].reset_index().groupby("node")["unit"].apply(set)
        home_units = home_units[home_units.index.isin(self.pull_nodes)]

        g = pd.read_csv(ROOT / "data" / "raw" / "glottolog" / "languages.csv").set_index("Glottocode")
        missing = sorted({c for cs in POINTS.values() for c in cs} - set(g.index))
        if missing:
            raise SystemExit(f"ph: glottocodes not in Glottolog: {missing}")
        trees = {}

        def tree_for(node):
            if node in trees:
                return trees[node]
            t = None
            hu = home_units.get(node)
            if hu:
                t = ("home", cKDTree(xy[np.isin(unit, list(hu))]))
            elif node not in self.pull_nodes:
                n = node
                while n and n not in POINTS:
                    n = n.rpartition(".")[0]
                if n:
                    ll = g.loc[POINTS[n], ["Longitude", "Latitude"]].astype(float).to_numpy()
                    t = ("point", cKDTree(np.column_stack([ll[:, 0] * kx, ll[:, 1] * ky])))
            trees[node] = t
            return t

        self.X = {}
        self.pos = np.zeros(len(place), dtype=np.int64)
        self.n_home = self.n_point = self.n_even = self.n_capped = self.n_local_led = 0
        by = pd.Series(np.arange(len(place))).groupby(unit).apply(lambda s: s.to_numpy())
        for u, c in counts.groupby(level=0):
            rows_i = by[u]
            c = c.droplevel(0)
            nodes = list(c.index)
            cols = c.to_numpy(dtype=float)
            p = self.pop[rows_i]
            if p.sum() <= 0:
                p = np.ones(len(rows_i))
            rows = p * cols.sum() / p.sum()
            seed = np.ones((len(rows_i), len(nodes)))
            home = {}   # column -> (boost, band mask, lambda)
            local_led = nodes[int(np.argmax(cols))] not in self.pull_nodes
            self.n_local_led += local_led
            for j, n in enumerate(nodes):
                if cols[j] >= MAJORITY * cols.sum():
                    self.n_even += 1
                    continue
                t = tree_for(n)
                if t is None or (t[0] == "home" and local_led):
                    self.n_even += 1
                    continue
                d, _ = t[1].query(xy[rows_i])
                if d.min() > (NEAR_KM if t[0] == "home" else FAR_KM):
                    self.n_even += 1
                    continue
                if t[0] == "home":
                    boost = np.exp(-(d - d.min()) / HOME_KERNEL_KM)
                    if n not in UPLAND_HOMES:
                        boost = boost * lowland[rows_i]
                    if boost.max() <= 0:
                        self.n_even += 1
                        continue
                    home[j] = (boost, d <= d.min() + BAND_KM, HOME_LAMBDA)
                    self.n_home += 1
                else:
                    k = np.exp(-(d - d.min()) / KERNEL_KM)
                    seed[:, j] = (1 - LAMBDA) + LAMBDA * k
                    self.n_point += 1
            capped = set()
            for _ in range(12):
                for j, (boost, band, lam) in home.items():
                    seed[:, j] = (1 - lam) + lam * boost
                X = _ipf(seed, rows, cols)
                over = []
                for j, (boost, band, lam) in home.items():
                    limit = BAND_CAP * rows[band].sum() / rows.sum()
                    if X[band, j].sum() / cols[j] > limit + 1e-9:
                        over.append(j)
                if not over:
                    break
                for j in over:
                    b, m, lam = home[j]
                    home[j] = (b, m, lam / 2)
                    capped.add(j)
            else:
                raise SystemExit(f"ph: unit {u}: the border cap does not hold after 12 halvings")
            self.n_capped += len(capped)
            if np.abs(X.sum(0) - cols).max() > max(1.0, 1e-6 * cols.sum()):
                raise SystemExit(f"ph: the placement rake does not meet unit {u}'s counts")
            self.X[u] = (rows_i, {n: X[:, j] for j, n in enumerate(nodes)})

            self.pos[rows_i] = np.arange(len(rows_i))
        self.unit = unit

    def weights(self, node, idx, count, plain=False):
        _, cols = self.X[self.unit[idx[0]]]
        if node not in cols:
            return None
        w = cols[node][self.pos[idx]]
        return w if w.sum() > 0 else None

    def summary(self):
        return (f"placed inside units by barangay rakes: {self.n_home} (unit, language) pairs "
                f"pulled towards a neighbouring homeland ({self.n_capped} of them held back by "
                f"the border cap), {self.n_point} towards Glottolog points, {self.n_even} even; "
                f"neighbour pull for {len(self.pull_nodes)} languages of {PULL_MIN:,}+, none in "
                f"the {self.n_local_led} units led by a smaller language")


ENTRY = dict(
    name="Philippines",
    source=("2020 Census of Population and Housing, household population by ethnicity, and "
            "households by language generally spoken at home (Philippine Statistics Authority)"),
    how=("census, 2020, ethnicity, each group drawn as its language; the people the census's "
         "national home-language table does not account for, 13%, drawn on the province's "
         "main language"),
    parts=[
        dict(covers="Ethnic groups, each drawn as its language",
             source="2020 census, ethnicity, by province and highly urbanised city",
             people=94_324_635),
        dict(covers="People who no longer speak their group's language",
             source="2020 census, national table of household language at home; drawn on "
                    "their province's main language",
             rest=True),
    ],
    grain=("117 provinces and cities, 930,000 people on average; inside each, placement is "
           "estimated"),
    gap=("18,590 people whose ethnicity was not reported, and the 368,300 people outside "
         "households (0.3%), whom the table does not count"),
    view=[116.5, 4.3, 127.0, 21.4],
    counts=_counts,
    mappings=["ph2020"],
    place=RD_GEO / "ph" / "ph_barangays.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=lambda place: _PhWeighter(place),
    note_public=(
        "The 2020 census asked every person's ethnicity and every household's language at home, "
        "but the Philippine Statistics Authority published the language answers only for the "
        "whole country. This map draws each ethnic group, by province and highly urbanised city, "
        "as its language. Many people no longer speak their group's language at home: "
        "nationally 1.03 million households speak Bikol, where the Bikol ethnic group would fill "
        "about 1.66 million. For each group, the people the national language table does not "
        "account for are drawn on the main language of the province they live in, taken first "
        "from provinces outside the group's home. That moves 14.3 million people, 13% of the "
        "country, mostly onto Tagalog around Manila and Cebuano in Mindanao. Bisaya or "
        "Binisaya is drawn as Cebuano. Most people in Aklan and many in Zambales answered "
        "\"other local ethnicity\", and there they are drawn as Aklanon and Sambal. The census "
        "gives nothing inside a province, so placement there is an estimate. Smaller languages "
        "are drawn towards where Glottolog places them. A large regional language that is a "
        "minority in a province leans a little towards the neighbouring province where it is "
        "the main language (Negros Occidental's Cebuano speakers towards Negros Oriental), in "
        "lowland barangays only."),
)
