"""Colombia — religion by departamento, from the LAPOP AmericasBarometer, waves 2010-2023.

Reads Colombia's rows out of data/raw/lapop/Grand_Merge_2004-2023_*.dta and writes
data/normalized/co.csv. `sources/lapop.py` holds the shared construction; `sources/co.md` is
this country's record; `sources.md` §11ap is the scouting and §9dk the write-up.

**NO COLOMBIAN CENSUS ASKS ABOUT RELIGION** (§11ac, §11ae). DANE's political-culture survey
does ask, at five regions only (§11ap). Every row this file writes is `modelled` in §7's sense.

## THE LABELS ARE RIGHT IN EVERY WAVE, AND THIS FILE PROVES IT THREE WAYS

Honduras's merge prints one wave's department labels on every wave (§11ap), so a label is not
evidence. Colombia's `prov` is 800 plus the DANE department code, and the file carries two
other columns that name places without going through that label:

  * **`municipio` (2012-2023)** is 800,000 plus the DANE municipality code. Its first three
    digits must be the respondent's `prov`, and its labels must be COD's municipality names
    at that code. **One disagreement, and it is a coding error in the file:** in 2012, 24
    interviews in Florida (`876275`, Valle del Cauca) carry `prov` 852, Nariño. Their `upm`
    (7627500) also says Florida, and the same municipality is under 876 in 2014, 2018 and
    2023. They are moved to Valle del Cauca, and the exact set of moved rows is asserted.
  * **`upm` (2010)**, the wave with no `municipio`, holds DANE municipality codes (5001
    Medellín, 76001 Cali, 97001006 Mitú). Their department prefix equals `prov - 800` for
    every 2010 respondent outside Bogotá, whose `upm` is 1-4 and whose `tamano` is the
    national capital on all 231 rows.
  * **Sample against population, wave by wave**, `print_wave_shares()`.

## WHICH ANSWERS CARRY THEIR OWN GEOGRAPHY

§9cy's construction (median Spearman over every distinct halving of the waves, against a
per-wave permutation null of the unit labels) plus Sweden's spatial chi-square as a veto, at
the 22 departments sampled in every wave, then again at LAPOP's six design regions
(`estratopri`), which nest the departments exactly. Five waves give ten 2-against-3 halvings.

**`cab.stability` would have tested four of the ten.** It keeps a halving only if it contains
wave 0, which removes each halving's mirror image when the wave count is even and, with an
odd count, throws away six distinct halvings. So this file enumerates its own.

Placed at the department: Católico, Evangélica y Pentecostal, Ninguna (creyente). Refused:
Testigos de Jehová passes the rank test at p=0.048 and fails the chi-square (p=0.29), which is
Sweden's case exactly. Nothing passes at the region that fails at the department, so the
mixed-level construction places no answer at the coarse level (Uzbekistan's result).
`Protestante Tradicional` fails at both levels, and the two Protestant boxes do not trade
places by department (Spearman +0.08 between their pooled shares), so they are tested and
drawn as the two answers the card offers.

## FOUR DEPARTMENTS ARE ASSUMED AND SEVEN ARE LEFT BLANK

Ecuador's line (§9bn, Anita 2026-09-08 on Carchi and Galápagos): whether anything measured
the place at all.

  * **Measured in one round only: La Guajira (2023, n=23), Quindío, Casanare and Vaupés
    (2010).** One wave measured each, so there is a reading to anchor an assumption on and no
    second one to test it against. Which assumption is spec §12's rule of 2026-09-14,
    `region_fallback()`: a department takes its LAPOP design region's shares for the three
    placed answers when that region, leaving each of its every-round departments out in
    turn, predicts them better than the country does (lower mean error, and closer for more
    than half), and the national rate otherwise. **La Guajira (Atlántica, 6 of 6) and
    Casanare (Oriental, 4 of 5) take their region; Quindío (Central, 2 of 5, means 0.06 pp
    apart) and Vaupés (the old national territories, 1 of 2) stay national.** The first
    build drew all four national, on a leave-one-out averaged over every region and four
    answers, which hid that the Caribbean is where the region wins (sources/co.md §6, §10).
  * **Not drawn: Chocó, Arauca, Vichada, Guaviare, Amazonas, San Andrés and Guainía.** LAPOP
    has no code for any of them, 1,303,929 people, 2.45% of Colombia, in `gap=`.

Usage:
    python sources/co.py --fetch    read Colombia's rows out of the 1.1 GB LAPOP .dta (~2 min)
    python sources/co.py            rebuild data/normalized/co.csv from that extract
"""

import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "6")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import lapop

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "co")
EXTRACT = os.path.join(RAW, "lapop_co.feather")
LABELS = os.path.join(RAW, "lapop_co_labels.csv")
ADM2 = os.path.join(RAW, "col_admpop_adm2_2025.csv")
POP = os.path.join(ROOT, "data", "geo", "co", "co_pop_2025.csv")
LOOKUP = os.path.join(ROOT, "data", "geo", "co", "co_lookup.csv")
OUT = os.path.join(ROOT, "data", "normalized", "co.csv")

PAIS = 8                       # LAPOP's country code for Colombia
WAVES = [2010, 2012, 2014, 2018, 2023]
SOURCE_ID = "co_lapop_2010_2023"
N_UNITS = 33
N_SAMPLED = 26
N_RESPONDENTS = 7_532
EXTRA = ["municipio", "upm", "cluster", "estratopri", "tamano"]

# What the tests select, asserted so a change is a failure here rather than a silent redraw.
#   1 Católico, 4 Ninguna (creyente), 5 Evangélica y Pentecostal
CARRIES = [1, 4, 5]
REFUSED = [12]                 # rank test passes, chi-square does not
COARSE = []                    # pass at the six regions and not at the 22 departments

# (wave, prov as printed, municipio) -> (prov it is, rows). See the module docstring.
RECODE = {(2012, 852, 876275): (876, 24)}

ONE_ROUND = {
    "CO44": "La Guajira: sampled in 2023 only, one municipality (Manaure)",
    "CO63": "Quindío: sampled in 2010 only",
    "CO85": "Casanare: sampled in 2010 only",
    "CO97": "Vaupés: sampled in 2010 only",
}
# The one-round departments `region_fallback` puts on their design region's shares; the rest
# of ONE_ROUND is at the national rate. Asserted, so a change is a failure and not a redraw.
ON_REGION = ["CO44", "CO85"]   # La Guajira (Atlántica), Casanare (Oriental)
NOT_DRAWN = {
    "CO27": "Chocó", "CO81": "Arauca", "CO88": "San Andrés, Providencia y Santa Catalina",
    "CO91": "Amazonas", "CO94": "Guainía", "CO95": "Guaviare", "CO99": "Vichada",
}

# LAPOP municipality labels that are not COD's spelling of the same DANE code. All three are
# the short everyday name of the city COD gives in full, at the same code, so they are aliases
# and not disagreements.
MUNI_ALIAS = {
    "bogota": "bogotadc",                         # CO11001 Bogotá, D.C.
    "cartagena": "cartagenadeindias",             # CO13001 Cartagena de Indias
    "tumaco": "sanandresdetumaco",                # CO52835 San Andrés de Tumaco
}

STAB_PERM, STAB_ALPHA, STAB_SEED = 2000, 0.05, 0


def fetch():
    """Colombia's rows, with the five place columns `lapop.USECOLS` does not carry."""
    import pyreadstat
    if not os.path.exists(lapop.DTA):
        raise SystemExit(f"{lapop.DTA} missing — see sources/gt.md for the download")
    cols = lapop.USECOLS + EXTRA
    print(f"reading {len(cols)} columns of the LAPOP grand merge…")
    df, meta = pyreadstat.read_dta(lapop.DTA, usecols=cols, apply_value_formats=False)
    df = df[df["pais"] == PAIS].reset_index(drop=True)
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(lambda x: None if x is None or (isinstance(x, float) and np.isnan(x))
                              else str(x))
    os.makedirs(RAW, exist_ok=True)
    df.to_feather(EXTRACT)
    rows = []
    for var in ["prov", "municipio", "estratopri", "tamano"]:
        for k, v in meta.value_labels.get(meta.variable_to_label.get(var), {}).items():
            if isinstance(k, (int, float)):
                rows.append((var, int(k), v))
    pd.DataFrame(rows, columns=["var", "code", "label"]).to_csv(LABELS, index=False,
                                                                 encoding="utf-8")
    print(f"wrote {EXTRACT} ({len(df):,} rows) and {LABELS}")


def load():
    """`lapop.load`'s filter on the extract, checked against `lapop.load` itself."""
    base = lapop.load(PAIS, WAVES)
    df = pd.read_feather(EXTRACT)
    both = (lapop.valid(df["q3c"]) & lapop.valid(df["q3cn"])).sum()
    if both:
        raise SystemExit(f"{both} respondents answer both q3c and q3cn")
    df["rel"] = df["q3c"].where(lapop.valid(df["q3c"]), df["q3cn"].where(lapop.valid(df["q3cn"])))
    df = df[lapop.valid(df["rel"]) & lapop.valid(df["prov"])].copy()
    df["code"] = df["rel"].astype(float).astype(int)
    df["prov_code"] = df["prov"].astype(float).astype(int)
    df["w"] = pd.to_numeric(df["weight1500"], errors="coerce").fillna(1.0)
    df["wave"] = df["wave"].astype(int)
    for c in EXTRA:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    a = df.groupby(["wave", "code"]).size()
    b = base.assign(wave=base["wave"].astype(int)).groupby(["wave", "code"]).size()
    if not a.equals(b):
        raise SystemExit("the extract and lapop.load disagree on Colombia's respondents — "
                         "re-run with --fetch")
    if len(df) != N_RESPONDENTS:
        raise SystemExit(f"{len(df):,} respondents, expected {N_RESPONDENTS:,}")
    return df


def fold(s):
    import unicodedata
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return "".join(ch for ch in s.lower() if ch.isalnum())


def check_labels(df, labels):
    """The three witnesses in the module docstring. Returns df with the recode applied."""
    print("\n  the department labels, checked without trusting them:")
    muni = df["municipio"]
    if muni[df["wave"] == 2010].notna().any() or muni[df["wave"] > 2010].isna().any():
        raise SystemExit("`municipio` coverage by wave has changed (expected none in 2010, "
                         "every row after)")

    # ---- witness 1: municipio's department prefix, and its names against COD ----
    has = muni.notna()
    pref = (muni // 1000)
    dis = df[has & (pref != df["prov_code"])]
    got = {(int(w), int(p), int(m)): n
           for (w, p, m), n in dis.groupby(["wave", "prov_code", "municipio"]).size().items()}
    want = {k: v[1] for k, v in RECODE.items()}
    if got != want:
        raise SystemExit(f"respondents whose municipio is in another department: {got}, "
                         f"expected {want}")
    for (w, p, m), (to, n) in RECODE.items():
        sel = (df["wave"] == w) & (df["prov_code"] == p) & (df["municipio"] == m)
        upm_dep = set((df.loc[sel, "upm"] // 100 // 1000).astype(int))
        if upm_dep != {to - 800}:
            raise SystemExit(f"the {n} rows being moved to {to} have upm departments {upm_dep}")
        df.loc[sel, "prov_code"] = to
    print(f"    1. municipio (2012-2023): every respondent's municipality is inside its prov, "
          f"except {sum(want.values())} Florida (Valle del Cauca) interviews printed under "
          "Nariño in 2012, whose upm also says Florida; moved")

    adm2 = pd.read_csv(ADM2, encoding="utf-8-sig", dtype={"ADM2_PCODE": str})
    cod = dict(zip(adm2["ADM2_PCODE"].str.strip(), adm2["ADM2_ES"]))
    lab = labels[labels["var"] == "municipio"].set_index("code")["label"]
    bad = []
    codes = sorted(int(m) for m in df["municipio"].dropna().unique())
    for m in codes:
        pc = f"CO{m - 800000:05d}"
        lname = lab.get(m)
        cname = cod.get(pc)
        ok = (cname is not None and lname is not None
              and (fold(lname) == fold(cname) or MUNI_ALIAS.get(fold(lname)) == fold(cname)))
        if not ok:
            bad.append((m, lname, pc, cname))
    for b in bad:
        print(f"       municipio {b[0]} LAPOP {b[1]!r} vs COD {b[2]} {b[3]!r}")
    if bad:
        raise SystemExit(f"{len(bad)} municipality labels do not match COD's name at their "
                         "DANE code — add a reasoned MUNI_ALIAS or stop")
    print(f"       all {len(codes)} sampled municipalities' labels match COD's name at the "
          "DANE code")

    # ---- witness 2: 2010's upm is a DANE municipality code ----
    d10 = df[df["wave"] == 2010]
    u = d10["upm"]
    bog = u < 1000
    if not ((d10.loc[bog, "prov_code"] == 811).all() and (d10.loc[bog, "tamano"] == 1).all()
            and (d10.loc[d10["prov_code"] == 811, "upm"] < 1000).all()):
        raise SystemExit("2010: the short upm codes are not exactly Bogotá's national-capital rows")
    dep = np.where(u >= 1_000_000, u // 1_000_000, u // 1000)
    off = d10[~bog & (dep + 800 != d10["prov_code"])]
    if len(off):
        raise SystemExit(f"2010: {len(off)} respondents whose upm municipality is in another "
                         "department")
    print(f"    2. upm (2010): {int((~bog).sum()):,} respondents in {int(u[~bog].nunique())} "
          f"DANE municipality codes, every one inside its prov; Bogotá's {int(bog.sum())} are "
          "all `Capital Nacional`")

    # ---- the design regions nest the departments ----
    nest = df.groupby("prov_code")["estratopri"].nunique()
    if (nest != 1).any():
        raise SystemExit(f"departments in two design regions: {sorted(nest[nest != 1].index)}")
    return df


def print_wave_shares(df, pop):
    """Witness 3: each wave's weighted sample share against COD-PS, with random pairings."""
    rng = np.random.default_rng(0)
    print("    3. sample share against COD-PS 2025, wave by wave:")
    for w in WAVES:
        s = df[df["wave"] == w]
        sh = s.groupby("geo_id")["w"].sum() / s["w"].sum()
        p = pop.loc[sh.index, "pop"] / pop.loc[sh.index, "pop"].sum()
        r = np.corrcoef(sh, p)[0, 1]
        perm = np.array([np.corrcoef(sh, rng.permutation(p.to_numpy()))[0, 1]
                         for _ in range(20000)])
        print(f"       {w}: {len(sh)} departments, r={r:+.3f}, "
              f"{int((perm >= r).sum())} of 20,000 random pairings reach it")


def stability(df, units, col, cats, label):
    """§9cy's median-over-halvings permutation test plus the spatial chi-square (§12, Sweden).

    Unweighted respondent counts. Also printed, never deciding: the single chronological
    halving on weighted shares (2010-2012 against 2014-2023, §11ap's figures), and the share of
    each answer's respondents in its largest (wave, sampling cluster) cell (§12, Uzbekistan).
    Returns (passed, refused).
    """
    from scipy.stats import spearmanr

    import stability as shared      # this function is named `stability` too
    units = list(units)
    ui = {u: i for i, u in enumerate(units)}
    wi = {w: i for i, w in enumerate(WAVES)}
    ci = {c: i for i, c in enumerate(cats)}
    cube = np.zeros((len(WAVES), len(units), len(cats)))
    for (w, u, k), n in df.groupby(["wave", col, "code"]).size().items():
        cube[wi[w], ui[u], ci[k]] += n
    tot = cube.sum(axis=2)
    if (tot == 0).any():
        raise SystemExit(f"{label}: a (wave, unit) cell has no respondent")
    splits = shared.halvings(len(WAVES))
    obs = shared.median_rho(cube, splits)
    null = shared.wave_null(cube, splits, STAB_PERM, STAB_SEED)
    early, late = df[df["wave"] <= 2012], df[df["wave"] >= 2014]

    print(f"\n  split-half at the {label}: {len(WAVES)} waves, median of {len(splits)} "
          f"halvings, {STAB_PERM:,}-draw per-wave permutation null, plus the chi-square:")
    print(f"    {'answer':<40}{'n':>6}{'share':>8}{'median':>8}{'null95':>8}{'p':>8}"
          f"{'chi2 p':>10}{'chrono':>8}{'cell':>6}  verdict")
    nat = df.groupby("code")["w"].sum() / df["w"].sum()
    passed, refused = [], []
    for j, c in enumerate(cats):
        n = int(cube[:, :, j].sum())
        chi = shared.chi2_p(cube[:, :, j].sum(0), tot.sum(0))

        def sh(d):
            g = d.groupby(col)
            return (g.apply(lambda x: x.loc[x["code"] == c, "w"].sum() / x["w"].sum(),
                            include_groups=False).reindex(units))
        with np.errstate(invalid="ignore"), __import__("warnings").catch_warnings():
            __import__("warnings").simplefilter("ignore")
            chrono = spearmanr(sh(early), sh(late)).statistic
        dk = df[df["code"] == c]
        cell = dk.groupby(["wave", "cluster"]).size().max() / len(dk)
        p, q95 = shared.permutation_p(obs[j], null[:, j], STAB_ALPHA)
        if not np.isfinite(p):
            verdict = "no test possible"
        elif p < STAB_ALPHA and chi < STAB_ALPHA:
            passed.append(c)
            verdict = "own geography"
        elif p < STAB_ALPHA:
            refused.append(c)
            verdict = "REFUSED: rank test passes, the units do not differ"
        else:
            verdict = "not distinguishable from chance"
        print(f"    {lapop.CATEGORY[c][:38]:<40}{n:>6,}{nat[c] * 100:7.2f}%{obs[j]:+8.3f}"
              f"{q95:+8.3f}{p:8.4f}{chi:10.2e}{chrono:+8.2f}{cell:6.0%}  {verdict}")
    return passed, refused


def region_fallback(df, fine_units, one_round, cats, unit_col="geo_id", region_col="estratopri",
                    names=None, region_names=None, n_perm=2000, seed=0):
    """Spec §12, 2026-09-14: which one-round units take their design region's shares.

    For each region, leave each of its every-round units out in turn and predict that unit's
    pooled weighted shares of the placed answers (`cats`) two ways: from the rest of the
    region's every-round units, and from every other every-round unit in the country. The
    error is the summed absolute difference over `cats`. **The region is used for its
    one-round units when its mean error is lower AND it is closer for more than half of its
    every-round units**; otherwise, and for a region with fewer than two every-round units,
    the national rate. The majority half stops one unit's miss or hit deciding a region.

    Only every-round units enter either pool, because a one-round unit's respondents are one
    wave's level and the shares being predicted are five-wave averages. It is also the pool
    the drawn region share uses, so the test measures the estimator that gets drawn.

    Printed and never deciding: how often a random set of as many every-round units (from
    regions with at least two) beats the country by as much, which says whether the design
    region is a real cluster or whether any group of that size would do.
    Returns {unit: region} for the one-round units that take their region.
    """
    names, region_names = names or {}, region_names or {}
    parent = df.groupby(unit_col)[region_col].first()
    units = list(fine_units)
    fine = df[df[unit_col].isin(units)]
    C = (fine[fine["code"].isin(cats)].groupby([unit_col, "code"])["w"].sum()
         .unstack(fill_value=0.0).reindex(index=units, columns=cats, fill_value=0.0).to_numpy())
    W = fine.groupby(unit_col)["w"].sum().reindex(units).to_numpy()
    own = C / W[:, None]
    e_country = np.abs(own - (C.sum(0) - C) / (W.sum() - W)[:, None]).sum(1) * 100
    idx = {u: i for i, u in enumerate(units)}

    def errors(members):
        m = np.array([idx[u] for u in members])
        reg = (C[m].sum(0) - C[m]) / (W[m].sum() - W[m])[:, None]
        return np.abs(own[m] - reg).sum(1) * 100, e_country[m]

    testable = [u for u in units if sum(parent[v] == parent[u] for v in units) >= 2]
    rng = np.random.default_rng(seed)
    print("\n  one-round departments: region or country? Leave-one-out over each region's "
          "every-round departments, summed absolute error over "
          + ", ".join(lapop.CATEGORY[c].split(" ")[0] for c in cats) + ", in points:")
    use = {}
    for r in sorted(parent[units].unique()):
        members = [u for u in units if parent[u] == r]
        ones = sorted(u for u in one_round if parent[u] == r)
        label = region_names.get(r, r)
        tag = ", ".join(names.get(u, u) for u in ones) or "no one-round department"
        if len(members) < 2:
            print(f"    {label} ({tag}): {len(members)} every-round department, cannot be "
                  "tested -> national rate")
            continue
        er, en = errors(members)
        for u, a, b in zip(members, er, en):
            print(f"      {names.get(u, u)[:22]:<24} region {a:5.1f}   country {b:5.1f}")
        closer, k = int((er < en).sum()), len(members)
        delta = en.mean() - er.mean()
        null = np.empty(n_perm)
        for i in range(n_perm):
            a, b = errors(list(rng.choice(testable, k, replace=False)))
            null[i] = b.mean() - a.mean()
        p = (1 + int((null >= delta).sum())) / (1 + n_perm)
        ok = er.mean() < en.mean() and closer * 2 > k
        print(f"    {label} ({tag}): mean region {er.mean():.2f}, country {en.mean():.2f}; "
              f"region closer in {closer} of {k}; a random {k} beat the country by as much in "
              f"p={p:.3f} -> {'REGION' if ok else 'national rate'}")
        if ok:
            use.update({u: r for u in ones})
    return use


def main():
    if "--fetch" in sys.argv:
        fetch()
    if not os.path.exists(EXTRACT):
        raise SystemExit(f"{EXTRACT} missing — run with --fetch")

    labels = pd.read_csv(LABELS)
    prov_labels = labels[(labels["var"] == "prov") & labels["code"].between(800, 899)]
    if len(prov_labels) != N_SAMPLED:
        raise SystemExit(f"{len(prov_labels)} Colombian prov labels, expected {N_SAMPLED}")

    df = load()
    df = check_labels(df, labels)

    lut = pd.read_csv(LOOKUP, dtype={"geo_id": str, "unit": str})
    if len(lut) != N_UNITS:
        raise SystemExit(f"{len(lut)} departments in the lookup, expected {N_UNITS}")
    sampled = lut.dropna(subset=["lapop_prov"])
    if len(sampled) != N_SAMPLED:
        raise SystemExit(f"{len(sampled)} lookup rows carry a prov code, expected {N_SAMPLED}")
    prov_to_unit = dict(zip(sampled["lapop_prov"].astype(int), sampled["unit"]))
    bad = sorted(set(df["prov_code"]) - set(prov_to_unit))
    if bad:
        raise SystemExit(f"prov codes with no department: {bad} — re-run sources/co_geo.py")
    df["geo_id"] = df["prov_code"].map(prov_to_unit)
    names = dict(zip(lut["geo_id"], lut["name"]))

    pop = pd.read_csv(POP, encoding="utf-8-sig", dtype={"geo_id": str}).set_index("geo_id")
    if sorted(pop.index) != sorted(lut["geo_id"]):
        raise SystemExit("the population file and the lookup cover different departments")

    print(f"\nColombia: {len(df):,} respondents, waves {WAVES[0]}-{WAVES[-1]}, "
          f"{df['geo_id'].nunique()} of {N_UNITS} departments")
    print_wave_shares(df, pop)
    lapop.held_out(df, pop.loc[sorted(sampled["geo_id"])], "Colombia", pop_source="COD-PS 2025")

    # ---- which departments can be ranked, assumed, or not drawn ----
    fine_units = sorted(u for u, s in df.groupby("geo_id") if s["wave"].nunique() == len(WAVES))
    at_national = sorted(set(sampled["geo_id"]) - set(fine_units))
    if set(at_national) != set(ONE_ROUND):
        raise SystemExit(f"departments outside the split-half are now {at_national}, not "
                         f"{sorted(ONE_ROUND)} — LAPOP's sample has changed")
    blank = sorted(set(lut["geo_id"]) - set(sampled["geo_id"]))
    if set(blank) != set(NOT_DRAWN) or set(df["geo_id"]) & set(NOT_DRAWN):
        raise SystemExit(f"the undrawn departments are now {blank}, not {sorted(NOT_DRAWN)}")
    gap = int(pop.loc[blank, "pop"].sum())
    print(f"\n  {len(fine_units)} departments in every wave; {len(at_national)} in one only "
          f"({int(pop.loc[at_national, 'pop'].sum()):,} people, "
          f"{pop.loc[at_national, 'pop'].sum() / pop['pop'].sum():.2%}):")
    for gid in at_national:
        print(f"    {gid} n={int((df['geo_id'] == gid).sum()):<4}{ONE_ROUND[gid]}")
    print(f"  {len(blank)} NOT DRAWN, no LAPOP code: {gap:,} people "
          f"({gap / pop['pop'].sum():.2%}): {', '.join(NOT_DRAWN[g] for g in blank)}")

    # ---- the tests ----
    nat = lapop.national(df)
    cats = sorted(nat.index, key=lambda k: -nat[k])
    fine, refused = stability(df[df["geo_id"].isin(fine_units)], fine_units, "geo_id", cats,
                              f"{len(fine_units)} departments")
    regions = sorted(df["estratopri"].unique())
    reg_pass, _ = stability(df, regions, "estratopri", cats, f"{len(regions)} design regions")
    coarse = [c for c in reg_pass if c not in fine]
    if sorted(fine) != sorted(CARRIES) or sorted(refused) != sorted(REFUSED) \
            or sorted(coarse) != sorted(COARSE):
        raise SystemExit(f"the tests now select {sorted(fine)} (refused {sorted(refused)}, "
                         f"region only {sorted(coarse)}), not {CARRIES} / {REFUSED} / {COARSE}. "
                         "That is a change in what this country claims to know — read the tables "
                         "above, then update the constants and the docstring deliberately.")
    large = [c for c in cats if c in fine]
    small = [c for c in cats if c not in fine]
    region_names = labels[labels["var"] == "estratopri"].set_index("code")["label"].to_dict()
    use_region = region_fallback(df, fine_units, at_national, large, names=names,
                                 region_names=region_names)
    if sorted(use_region) != sorted(ON_REGION):
        raise SystemExit(f"region_fallback now puts {sorted(use_region)} on their region, not "
                         f"{ON_REGION}. Read the table above, then update ON_REGION, the "
                         "docstring and sources/co.md §6 deliberately.")
    print(f"\n  -> {len(large)} answers on their own department shares, {len(small)} at the "
          "national rate inside each department's residual")

    out = lapop.build(df[df["geo_id"].isin(fine_units)], nat, large, small, pop["pop"],
                      fine_units, unit_noun="department")
    parts, rows = [out], []
    for gid in at_national:
        why = ONE_ROUND[gid].split(": ", 1)[1]
        if gid in use_region:
            # The same construction as an every-round department, with the region's every-round
            # departments standing in for the department's own respondents.
            r = use_region[gid]
            rname = region_names[int(r)]
            pool = df[(df["estratopri"] == r) & df["geo_id"].isin(fine_units)]
            part = lapop.build(pool.assign(geo_id=gid), nat, large, small, pop["pop"], [gid],
                               unit_noun="region")
            part["basis_note"] = part["basis_note"].map({
                "region share": f"share in the {rname} region's every-round departments",
                "national share within the region's residual":
                    f"national share within the residual of the {rname} region's shares",
            }) + "; " + why
            parts.append(part)
            continue
        p = int(pop.loc[gid, "pop"])
        for c in nat.index:
            rows.append((gid, lapop.CATEGORY[c], nat[c] * p, "national share; " + why))
    rest = pd.DataFrame(rows, columns=["geo_id", "source_category", "count", "basis_note"])
    rest["count"] = rest["count"].round().astype("int64")
    for gid in rest["geo_id"].unique():
        m = rest["geo_id"] == gid
        rest.loc[rest.loc[m, "count"].idxmax(), "count"] += (int(pop.loc[gid, "pop"])
                                                             - int(rest.loc[m, "count"].sum()))
    out = pd.concat(parts + [rest], ignore_index=True)
    if out["basis_note"].isna().any():
        raise SystemExit("a region row's basis note did not map — lapop.build's wording changed")

    out["geo_level"] = "departamento"
    out["geo_name"] = out["geo_id"].map(names)
    out["basis"] = "self_id"
    out["year"] = "2010-2023"
    out["source_id"] = SOURCE_ID
    n_by = df.groupby("geo_id").size()
    out["n_dept"] = out["geo_id"].map(n_by).astype(int)
    out["note"] = out.apply(
        lambda r: (f"LAPOP AmericasBarometer waves 2010-2023 pooled, n={r.n_dept} in this "
                   f"department; {r.basis_note} applied to the COD-PS 2025 population"), axis=1)

    total = int(out["count"].sum())
    if total != int(pop["pop"].sum()) - gap:
        raise SystemExit(f"drawn {total:,} against {int(pop['pop'].sum()) - gap:,}")
    if out["geo_id"].nunique() != N_UNITS - len(NOT_DRAWN):
        raise SystemExit(f"{out['geo_id'].nunique()} departments drawn")

    cols = ["geo_id", "geo_level", "geo_name", "source_category", "count",
            "basis", "year", "source_id", "note"]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out[cols].to_csv(OUT, index=False, encoding="utf-8")
    print(f"\nwrote {OUT} ({len(out)} rows, {total:,} people, "
          f"{out['source_category'].nunique()} categories, {out['geo_id'].nunique()} units)")

    print("\n  national, as drawn:")
    drawn = (out.groupby("source_category")["count"].sum() / total).sort_values(ascending=False)
    for cat, s in drawn.items():
        print(f"    {s * 100:6.2f}%  {int(out.loc[out['source_category'] == cat, 'count'].sum()):>12,}"
              f"  {cat}")

    print("\n  by wave (weighted):")
    wv = df.groupby(["wave", "code"])["w"].sum().unstack(fill_value=0.0)
    wv = wv.div(wv.sum(axis=1), axis=0) * 100
    print("    " + wv[[1, 5, 2, 4, 11, 12, 77]].round(2).to_string().replace("\n", "\n    "))

    print(f"\n  the {len(large)} answers on department shares, with the sample behind each:")
    show = out[out["geo_id"].isin(fine_units)].pivot_table(
        index="geo_id", columns="source_category", values="count", aggfunc="sum")
    show = show.div(show.sum(axis=1), axis=0)
    cs = [lapop.CATEGORY[c] for c in large]
    print(f"    {'department':<22}{'n':>6}" + "".join(f"{c.split(' ')[0][:12]:>13}" for c in cs))
    for gid in show[cs[0]].sort_values().index:
        print(f"    {names[gid][:20]:<22}{int(n_by[gid]):>6}"
              + "".join(f"{show.loc[gid, c] * 100:12.1f}%" for c in cs))
    t7 = df[df["code"] == 7].groupby("geo_id").size().sort_values(ascending=False)
    print(f"\n  `Religiones Tradicionales` by department ({int(t7.sum())} respondents): "
          + ", ".join(f"{names[g]} {n}" for g, n in t7.items()))


if __name__ == "__main__":
    main()
