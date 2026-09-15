"""Bolivia — religion by departamento, from LAPOP's AmericasBarometer single-country files, 2010-2023.

Reads data/raw/bo/lapop_bo_<year>.dta and writes data/normalized/bo.csv. `sources/bo.md` is this
country's record; `sources.md` §11ap is the scouting and §9dm the write-up. `sources/lapop.py`
holds the answer card; everything below is Bolivia's own.

**BOLIVIA IS IN THE FREE GRAND MERGE ONLY BEFORE 2010** (§11ad, §11ae). LAPOP publishes every
Bolivian wave as its own file, Free Tier, behind the same site usage agreement as the merge
(§11ap), and those files are what this reads. No census since 1992 has asked religion; the 2024
census did not. Every row this file writes is `modelled` in §7's sense.

## THE WAVES

Pooled: 2010, 2012, 2014, 2016/17, 2018/19 and 2023, the six on LAPOP's standard card
(`q3c`, `q3cn` from 2018). 2021 was the telephone round and asked neither religion nor place.
**2008 also asks religion, on an older card** (`q3`, seven answers: Mormons, Witnesses,
Spiritualists and Adventists share one box and there is no agnostic or atheist box), so it is not
pooled and is used as an out-of-pool check on the department ordering.

2008-2014 are stratified by all nine departments at about 300 interviews each and carry one
design weight per department. From 2016/17 the strata merge to six (Beni-Pando, Potosí-Oruro,
Chuquisaca-Tarija) and the file's `wt` is 1 for everyone, though Beni is 8.4% of the sample
against 4.2% of the people. **So every wave is post-stratified here**: each respondent keeps the
file's weight relative to the rest of their department, and each (wave, department) is scaled
to the department's 2024 census share of 1,500, LAPOP's `weight1500` convention.

## THE DEPARTMENT LABELS, CHECKED WITHOUT TRUSTING THEM

Honduras's merge printed one wave's labels on every wave (§11ap). These are single-country
files with their own labels, and they are still checked, three ways, on every run:

  * **`prov` is LAPOP's own department order in 2008-2018, not INE's**: 1001 La Paz, 1002 Santa
    Cruz, 1003 Cochabamba, 1004 Oruro, 1005 Chuquisaca, 1006 Potosí, 1007 Pando, 1008 Tarija,
    1009 Beni. In 2023 it is a PROVINCE code, 10 plus INE's department and province numbers.
  * **`municipio` names (2010-2023)**, against COD-AB's 339 municipality names: for every prov
    code in every wave, the departments its municipality names can belong to intersect in
    exactly one department, and it is the one the label names. In 2023 `municipio` minus
    1,000,000 is INE's municipality code and COD's pcode, and its department digits are asserted
    too. `MUNI_ALIAS` holds the official long forms of short names.
  * **Sample against population, wave by wave** (`sources/lits.py::held_out`, exhaustive over
    9! orderings): on the design weights in 2008-2014; on region-level weights in 2016-2023, so
    the within-stratum split of each merged pair is what is tested. And from 2016 `estratopri`
    must be the stratum of the department `prov` names, for every respondent.

## WHICH ANSWERS CARRY THEIR OWN GEOGRAPHY

§9cy's median over all ten halvings of the six waves against a per-wave permutation null, with
Sweden's chi-square veto, Uzbekistan's largest-cell refusal and Honduras's standout test, at the
nine departments and again at the six design strata, which nest them. Placed on department shares:
Católico, Evangélica y Pentecostal, Protestante tradicional, Ninguna (creyente) and Agnóstico o
ateo. Nothing passes at the strata that fails at the department, nothing is refused and nothing
stands apart. The two Protestant boxes swap between rounds but not by place (Spearman +0.03
between their department shares), and each passes alone, so they are kept apart.

## THE 1992 CENSUS IS A WITNESS, NOT A SOURCE

`sources/bo_checks.py`: 1992 orders the departments like this pool for Catholics (+0.75) and for
other Christians (+0.92), and its province pattern inside departments does not replicate, so it is
not used to split departments into provinces.

## THE LEVEL IS THE POOL'S

Catholic identification falls from 80.3% in 2010 to 64.8% in 2023. §12's Norway rule would take
the level from recent rounds; not done, because the Protestant and evangelical boxes swap from
round to round and a recent level per answer is unstable (`sources/bo.md` §8).

Usage:
    python sources/bo.py --fetch    download the seven Stata files from LAPOP (~7 MB)
    python sources/bo.py            rebuild data/normalized/bo.csv
"""

import os
import sys
import unicodedata
import urllib.request
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "6")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import lapop

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "bo")
GEO = os.path.join(ROOT, "data", "geo", "bo")
POP = os.path.join(GEO, "bo_pop_2024.csv")
LOOKUP = os.path.join(GEO, "bo_lookup.csv")
MUNIS = os.path.join(GEO, "bo_municipios.csv")
OUT = os.path.join(ROOT, "data", "normalized", "bo.csv")

# LAPOP's data directory, `?lp_download=<id>` for the Stata file of each wave (§11ap).
URL = "https://www.vanderbilt.edu/center-for-global-democracy/data/directory/?lp_download={}"
FILES = {2008: 1595, 2010: 1475, 2012: 1860, 2014: 2041, 2016: 2177, 2018: 2215, 2023: 2430}
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) religiondots/1.0"}

WAVES = [2010, 2012, 2014, 2016, 2018, 2023]
CHECK_WAVE = 2008
EARLY = [2010, 2012, 2014]          # nine strata; the chronological halving is these against the rest
SOURCE_ID = "bo_lapop_2010_2023"
N_UNITS = 9
N_RESPONDENTS = 13_882
WEIGHT_TOTAL = 1500.0

# prov 2008-2018, LAPOP's order -> COD pcode (INE's order)
LAPOP_ORDER = {1001: "BO02", 1002: "BO07", 1003: "BO03", 1004: "BO04", 1005: "BO01",
               1006: "BO05", 1007: "BO09", 1008: "BO06", 1009: "BO08"}

# The six design strata from 2016/17. They nest the departments, and are the coarse level.
REGION = {"BO02": "La Paz", "BO07": "Santa Cruz", "BO03": "Cochabamba",
          "BO08": "Beni-Pando", "BO09": "Beni-Pando", "BO04": "Potosí-Oruro",
          "BO05": "Potosí-Oruro", "BO01": "Chuquisaca-Tarija", "BO06": "Chuquisaca-Tarija"}
ESTRATO_REGION = {1001: "La Paz", 1002: "Santa Cruz", 1003: "Cochabamba", 1010: "Beni-Pando",
                  1011: "Potosí-Oruro", 1012: "Chuquisaca-Tarija"}

# LAPOP municipality labels (folded) -> COD-AB's name for the same place (folded). Each is the
# everyday short form of the official name, and each target is looked up by NAME, so where the
# official name is shared (San José, Santa Rosa, Santa Ana, San Ignacio) the witness gets every
# department that has one and has to be settled by the other municipalities under the same code.
MUNI_ALIAS = {
    "lapaz": "nuestrasenoradelapaz",
    "santacruz": "santacruzdelasierra",
    "carabuco": "puertomayordecarabuco",
    "guaqui": "puertomayordeguaqui",
    "rurrenabaque": "puertomenorderurrenabaque",
    "gonzalomoreno": "puertogonzalomoreno",
    "salinas": "salinasdegarcimendoza",
    "huari": "santiagodehuari",
    "icla": "villaricardomugiaicla",
    "caiza": "caizad",
    "tinquipaya": "tinguipaya",
    "sanjosedechiquitos": "sanjose",
    "santarosadelsara": "santarosa",
    "santarosadelabuna": "santarosa",
    "santaanadelyacuma": "santaana",
    "sanignaciodemoxos": "sanignacio",
}

# What the tests select, asserted so a change is a failure here rather than a silent redraw.
#   1 Católico, 5 Evangélica y Pentecostal, 2 Protestante tradicional, 4 Ninguna (creyente),
#   11 Agnóstico o ateo
CARRIES = [1, 5, 2, 4, 11]
REFUSED = []                        # rank test passes, chi-square or one sampling cell does not
COARSE = []                         # pass at the six strata and not at the nine departments
STANDOUTS = {}                      # fail, but one department tops both halves nearly always

STAB_PERM, STAB_ALPHA, STAB_SEED = 2000, 0.05, 0
STANDOUT_AGREE = 0.95
CLUSTER_CAP = 0.5
# Waves whose `wt` is 1 for every respondent although the design is disproportionate (Beni at
# twice its population share). `poststratify` rebuilds their weights; `read_wave` stops on a
# constant `wt` in any other wave, and on a wave listed here whose `wt` varies (2026-09-14).
WT_CONSTANT = {2016, 2018, 2023}
UNION = 25                          # traditional Protestant + evangelical, tested, never drawn

LABEL = dict(lapop.CATEGORY)
LABEL[UNION] = "Protestante tradicional + Evangélica (tested together)"


def fold(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return "".join(ch for ch in s.lower() if ch.isalnum())


def fetch():
    os.makedirs(RAW, exist_ok=True)
    for y, i in FILES.items():
        dst = os.path.join(RAW, f"lapop_bo_{y}.dta")
        if os.path.exists(dst) and os.path.getsize(dst) > 500_000:
            print(f"  have lapop_bo_{y}.dta ({os.path.getsize(dst):,} bytes)")
            continue
        req = urllib.request.Request(URL.format(i), headers=UA)
        with urllib.request.urlopen(req, timeout=600) as r, open(dst + ".part", "wb") as f:
            f.write(r.read())
        with open(dst + ".part", "rb") as fh:
            head = fh.read(11)
        if not (head.startswith(b"<stata_dta>") or head[:1] in (b"\x72", b"\x73")):
            raise SystemExit(f"{y}: the download is not a Stata file (starts {head!r}); the "
                             "site agreement may have become a real gate")
        os.replace(dst + ".part", dst)
        print(f"  got  lapop_bo_{y}.dta ({os.path.getsize(dst):,} bytes)")


def read_wave(y):
    import pyreadstat
    df, meta = pyreadstat.read_dta(os.path.join(RAW, f"lapop_bo_{y}.dta"),
                                   apply_value_formats=False)
    cols = {c.lower(): c for c in df.columns}
    rels = [c for c in ("q3", "q3c", "q3cn") if c in cols]
    if len(rels) != 1:
        raise SystemExit(f"{y}: religion variables {rels}, expected exactly one")

    def labels(var):
        if var not in cols:
            return {}
        ls = meta.variable_to_label.get(cols[var])
        return meta.value_labels.get(ls, {}) if ls else {}

    num = lambda v: (pd.to_numeric(df[cols[v]], errors="coerce") if v in cols
                     else pd.Series(np.nan, index=df.index))
    out = pd.DataFrame({
        "wave": y,
        "prov": num("prov"),
        "muni": num("municipio"),
        "estratopri": num("estratopri"),
        "cluster": df[cols["cluster"]].astype(str) if "cluster" in cols else "",
        "wt": num("wt"),
        "rel": num(rels[0]),
    })
    if out["prov"].isna().any():
        raise SystemExit(f"{y}: respondents with no prov")
    constant = out["wt"].nunique(dropna=False) <= 1
    if constant and y not in WT_CONSTANT:
        raise SystemExit(f"{y}: `wt` is the same for every respondent. In a disproportionate design "
                         "that draws each department at its sample share; post-stratify the wave "
                         "(poststratify) and add it to WT_CONSTANT.")
    if not constant and y in WT_CONSTANT:
        raise SystemExit(f"{y} is in WT_CONSTANT but its `wt` varies; the file has changed, read "
                         "it before post-stratifying over its weights")
    lab = {"prov": labels("prov"), "muni": labels("municipio"),
           "estratopri": labels("estratopri"), "rel": labels(rels[0])}
    return out, lab, rels[0]


def decode_wave(y, d, lab, munis, names):
    """The three label witnesses in the module docstring. Returns d with geo_id and region."""
    adm3, adm2 = {}, {}
    for n3, n2, p in zip(munis["adm3_name"], munis["adm2_name"], munis["adm1_pcode"]):
        adm3.setdefault(fold(n3), set()).add(p)
        adm2.setdefault(fold(n2), set()).add(p)

    def depts_of(name):
        f = fold(name)
        f = MUNI_ALIAS.get(f, f)
        # a province name standing for its capital municipality (2010's `cercado`) falls back
        # to every department with a province of that name
        return adm3.get(f) or adm2.get(f)

    if y < 2023:
        # Stata's letter missing codes (.a, .b) arrive as string keys among the labels
        pl = {int(k): v for k, v in lab["prov"].items() if isinstance(k, (int, float))}
        if set(d["prov"].astype(int)) != set(LAPOP_ORDER):
            raise SystemExit(f"{y}: prov codes {sorted(set(d['prov'].astype(int)))}")
        bad = [(p, pl.get(p), names[g]) for p, g in LAPOP_ORDER.items()
               if fold(pl.get(p, "")) != fold(names[g])]
        if bad:
            raise SystemExit(f"{y}: prov labels no longer LAPOP's order: {bad}")
        d["geo_id"] = d["prov"].astype(int).map(LAPOP_ORDER)
        if not (d["estratopri"] == d["prov"]).all() and y <= 2014:
            raise SystemExit(f"{y}: estratopri is not the department")
    else:
        codes = d["prov"].astype(int)
        if not codes.between(10101, 10999).all():
            raise SystemExit(f"{y}: prov is not 10DDPP")
        d["geo_id"] = codes.map(lambda p: f"BO{p // 100 % 100:02d}")
    d["region"] = d["geo_id"].map(REGION)
    if y >= 2016:
        er = d["estratopri"].astype(int).map(ESTRATO_REGION)
        if er.isna().any() or (er != d["region"]).any():
            raise SystemExit(f"{y}: {int((er != d['region']).sum())} respondents whose "
                             "estratopri is not the stratum of their prov")
        el = {int(k): v for k, v in lab["estratopri"].items() if isinstance(k, (int, float))}
        if any(fold(el[k]) != fold(v) for k, v in ESTRATO_REGION.items()):
            raise SystemExit(f"{y}: estratopri labels {el}")

    if d["muni"].isna().all():
        print(f"    {y}: no municipio; prov labels are LAPOP's order, estratopri equals prov")
        return d
    ml = lab["muni"]
    unresolved, derived, amb = [], {}, 0
    for p, g in d.groupby("prov"):
        sets = []
        for m in sorted(g["muni"].dropna().unique()):
            nm = ml.get(m)
            s = depts_of(nm) if nm is not None else None
            if not s:
                unresolved.append((int(p), int(m), nm))
                continue
            sets.append(s)
            amb += len(s) > 1
        derived[int(p)] = set.intersection(*sets) if sets else set()
    if unresolved:
        raise SystemExit(f"{y}: municipio labels COD-AB has no municipality or province for: "
                         f"{unresolved} — add a reasoned MUNI_ALIAS or stop")
    want = {p: ({LAPOP_ORDER[p]} if y < 2023 else {f"BO{p // 100 % 100:02d}"}) for p in derived}
    # Names alone must settle a prov code on exactly one department, except in 2023, where a code
    # whose only municipality has a name two departments share (10606 Burnet O'Connor) is
    # settled by the municipality's own INE code instead, checked below.
    loose = {p for p, s in derived.items() if len(s) > 1 and want[p] <= s}
    wrong = {p: sorted(s) for p, s in derived.items()
             if not want[p] <= s or (p in loose and y != 2023)}
    if wrong:
        raise SystemExit(f"{y}: the municipio names do not place these prov codes in the "
                         f"department the label names: {wrong}")
    n_m = int(d["muni"].nunique())
    extra = ""
    if y == 2023:
        pc = (d["muni"].astype(int) - 1_000_000).map(lambda m: f"BO{m:06d}")
        if (pc.str[:4] != d["geo_id"]).any():
            raise SystemExit("2023: municipio codes in another department than prov")
        byp = dict(zip(munis["adm3_pcode"], munis["adm3_name"]))

        def agrees(c):
            nm = fold(ml[int(c[2:]) + 1_000_000])
            return c in byp and (fold(byp[c]) == nm or MUNI_ALIAS.get(nm) == fold(byp[c]))
        for p in loose:
            unsettled = [c for c in pc[d["prov"].astype(int) == p].unique() if not agrees(c)]
            if unsettled:
                raise SystemExit(f"2023: prov {p}'s municipality name is shared between "
                                 f"departments and its code does not settle it: {unsettled}")
        known = pc[pc.isin(byp)].unique()
        same = sum(agrees(c) for c in known)
        extra = (f"; municipio - 1,000,000 is COD's pcode for {len(known)} of {n_m}, the name "
                 f"at that pcode agrees for {same}, and settles {len(loose)} prov code(s) whose "
                 "only municipality name two departments share")
    print(f"    {y}: {n_m} municipalities under {len(derived)} prov codes each resolve to "
          f"exactly the labelled department ({amb} names shared between departments, settled "
          f"by their neighbours){extra}")
    return d


def held_out_waves(frames, pop):
    """Each wave's sample share against the 2024 census, exhaustive over 9! orderings. REPORTED.

    Not asserted, because the only population here is 2024's and the waves' weights are not.
    2008's design weights put La Paz at 28.4% and Santa Cruz at 24.5%, against 26.7% and 27.5%
    in 2024: Santa Cruz overtook La Paz in between, and Oruro, Tarija, Beni and Chuquisaca sit
    within about a point of each other, so a handful of orderings that swap them beat the
    observed one with no label wrong (14 of 362,879 in 2008). The witness that pins the decode
    is `municipio`, in `decode_wave`, and that one is asserted for every pooled wave.
    """
    import lits
    share = pop / pop.sum()
    reg_share = share.groupby(pd.Series(REGION)).sum()
    for y in [CHECK_WAVE] + WAVES:
        d = frames[y]
        if y <= 2014:
            w = d["wt"]
            how = "design weights"
        else:
            w = d["region"].map(reg_share) / d.groupby("region")["region"].transform("size")
            how = "region-level weights, so the split inside each merged stratum is tested"
        print(f"\n  {y}, {how}:", end="")
        try:
            lits.held_out(d.assign(w=w.to_numpy()), pop, f"Bolivia {y}",
                          pop_source="the 2024 census")
        except SystemExit as e:
            print(f"      REPORTED, NOT ASSERTED (see the docstring): {e}")


def poststratify(df, pop):
    share = pop / pop.sum()
    df["w"] = 0.0
    for (y, g), idx in df.groupby(["wave", "geo_id"]).groups.items():
        wt = df.loc[idx, "wt"].fillna(1.0)
        df.loc[idx, "w"] = wt / wt.sum() * share[g] * WEIGHT_TOTAL
    return df


def stability(df, units, col, cats, label):
    """§9cy's median-over-halvings permutation test, Sweden's chi-square veto, Uzbekistan's
    largest-cell share and Honduras's which-unit-tops-both-halves, on unweighted counts."""
    from scipy.stats import spearmanr

    import stability as shared      # this function is named `stability` too
    units = list(units)
    ui = {u: i for i, u in enumerate(units)}
    wi = {w: i for i, w in enumerate(WAVES)}
    ci = {c: i for i, c in enumerate(cats)}
    cube = np.zeros((len(WAVES), len(units), len(cats)))
    tot = np.zeros((len(WAVES), len(units)))
    for (w, u, k), n in df.groupby(["wave", col, "code"]).size().items():
        tot[wi[w], ui[u]] += n
        if k in ci:
            cube[wi[w], ui[u], ci[k]] += n
    if (tot == 0).any():
        raise SystemExit(f"{label}: a (wave, unit) cell has no respondent")
    splits = shared.halvings(len(WAVES))
    obs = shared.median_rho(cube, splits, tot=tot)
    null = shared.wave_null(cube, splits, STAB_PERM, STAB_SEED, tot=tot)
    sa, sb = zip(*(shared.halves(cube, a, b, tot) for a, b in splits))
    top_unit, top_share = shared.top_both_halves(np.array(sa), np.array(sb))
    early, late = df[df["wave"].isin(EARLY)], df[~df["wave"].isin(EARLY)]
    nat = df.groupby("code")["w"].sum() / df["w"].sum()

    print(f"\n  split-half at the {label}: {len(WAVES)} waves, median of {len(splits)} "
          f"halvings, {STAB_PERM:,}-draw per-wave permutation null, chi-square beside it:")
    print(f"    {'answer':<40}{'n':>6}{'share':>8}{'median':>8}{'null95':>8}{'p':>8}"
          f"{'chi2 p':>10}{'chrono':>8}{'cell':>6}  {'tops both halves':<26}verdict")
    res = {}
    for j, c in enumerate(cats):
        n = int(cube[:, :, j].sum())
        chi = shared.chi2_p(cube[:, :, j].sum(0), tot.sum(0))
        codes = [c]                 # the union run passes a frame already recoded to UNION

        def sh(d):
            g = d.assign(hit=d["code"].isin(codes) * d["w"]).groupby(col)
            return (g["hit"].sum() / g["w"].sum()).reindex(units)
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            chrono = spearmanr(sh(early), sh(late)).statistic
        dk = df[df["code"].isin(codes)]
        cell = dk.groupby(["wave", "cluster"]).size().max() / len(dk) if len(dk) else 1.0
        if top_unit[j] >= 0:
            top_u, top_f = units[top_unit[j]], top_share[j]
        else:
            top_u, top_f = None, 0.0
        r = dict(passed=False, refused=False, chi=chi, cell=cell, top_unit=top_u, top_frac=top_f,
                 median=float(obs[j]))
        p, q95 = shared.permutation_p(obs[j], null[:, j], STAB_ALPHA)
        if not np.isfinite(p):
            verdict = "no test possible"
        elif p < STAB_ALPHA and chi < STAB_ALPHA and cell < CLUSTER_CAP:
            r["passed"] = True
            verdict = "own geography"
        elif p < STAB_ALPHA and chi >= STAB_ALPHA:
            r["refused"] = True
            verdict = "REFUSED: rank test passes, the units do not differ"
        elif p < STAB_ALPHA:
            r["refused"] = True
            verdict = "REFUSED: rank test passes, one sampling cell is most of it"
        else:
            verdict = "not distinguishable from chance"
        r["p"] = p
        share = sum(nat.get(k, 0.0) for k in codes)
        print(f"    {LABEL[c][:38]:<40}{n:>6,}{share * 100:7.2f}%{obs[j]:+8.3f}"
              f"{q95:+8.3f}{p:8.4f}{chi:10.2e}{chrono:+8.2f}{cell:6.0%}  "
              f"{str(top_u)[:14]:<14}{top_f:6.0%}      {verdict}")
        res[c] = r
    return res


def check_2008(d08, df, names):
    """The out-of-pool wave, on its own card, against the pooled department shares. Printed."""
    import spearman_null
    from scipy.stats import spearmanr
    d = d08[d08["rel"].notna()].copy()
    d["code"] = d["rel"].astype(int)
    units = sorted(names)
    print(f"\n  2008, not pooled (older card, n={len(d):,}), department ordering against the "
          "2010-2023 pool:")
    for label, c08, cpool in [("Catholic", [1], [1]), ("evangelical", [5], [5]),
                              ("traditional Protestant", [2], [2]),
                              ("no religion (2008 has no agnostic box)", [4], [4, 11])]:
        a = (d.assign(h=d["code"].isin(c08) * d["wt"]).groupby("geo_id")["h"].sum()
             / d.groupby("geo_id")["wt"].sum()).reindex(units)
        b = (df.assign(h=df["code"].isin(cpool) * df["w"]).groupby("geo_id")["h"].sum()
             / df.groupby("geo_id")["w"].sum()).reindex(units)
        rho = spearmanr(a, b).statistic
        print(f"    {label:<42} Spearman {rho:+.3f}, exact p {spearman_null.exact_p(rho, 9):.4f};"
              f" 2008 {(a * 100).round(1).to_dict()}")


def compose(df, nat, fine, coarse, standouts, pop, units):
    """Placed shares per department, then the national mix inside each department's residual."""
    cats = list(nat.index)
    bu = df.groupby(["geo_id", "code"])["w"].sum().unstack(fill_value=0.0).reindex(
        columns=cats, fill_value=0.0)
    ushare = bu.div(bu.sum(axis=1), axis=0)
    br = df.groupby(["region", "code"])["w"].sum().unstack(fill_value=0.0).reindex(
        columns=cats, fill_value=0.0)
    rshare = br.div(br.sum(axis=1), axis=0)
    placed = pd.DataFrame(index=units, columns=[], dtype=float)
    basis = {}
    for c in fine:
        placed[c] = ushare.loc[units, c]
        basis[c] = {u: "department share" for u in units}
    for c in coarse:
        placed[c] = [rshare.loc[REGION[u], c] for u in units]
        basis[c] = {u: f"share in the {REGION[u]} design stratum" for u in units}
    for c, u0 in standouts.items():
        rest = df[df["geo_id"] != u0]
        s = rest.loc[rest["code"] == c, "w"].sum() / rest["w"].sum()
        placed[c] = [ushare.loc[u0, c] if u == u0 else s for u in units]
        basis[c] = {u: ("department share, standing apart" if u == u0
                        else "share across the other eight departments") for u in units}
    small = [c for c in cats if c not in placed.columns]
    residual = 1.0 - placed.sum(axis=1)
    if (residual <= 0).any():
        raise SystemExit(f"departments with no room for the tail: "
                         f"{sorted(residual[residual <= 0].index)}")
    small_total = float(nat[small].sum())
    print(f"    the tail is {residual.min():.1%} of {residual.idxmin()} and {residual.max():.1%} "
          f"of {residual.idxmax()}, against {small_total:.1%} nationally")
    rows = []
    for u in units:
        p = int(pop[u])
        cells = []
        for c in placed.columns:
            cells.append([u, LABEL[c], placed.loc[u, c] * p, basis[c][u]])
        for c in small:
            cells.append([u, LABEL[c], residual[u] * nat[c] / small_total * p,
                          "national share within the department's residual"])
        cnt = np.array([x[2] for x in cells])
        r = np.round(cnt).astype(np.int64)
        r[int(np.argmax(cnt))] += p - int(r.sum())
        for x, v in zip(cells, r):
            rows.append((x[0], x[1], int(v), x[3]))
    out = pd.DataFrame(rows, columns=["geo_id", "source_category", "count", "basis_note"])

    print("\n  every category at the national rate, as drawn beside the survey's own department "
          "share (§12, Latvia: look for a reversal):")
    show = out.pivot_table(index="geo_id", columns="source_category", values="count",
                           aggfunc="sum")
    show = show.div(show.sum(axis=1), axis=0)
    from scipy.stats import spearmanr
    for c in small:
        n = int((df["code"] == c).sum())
        if n < 20:
            continue
        own, drawn = ushare.loc[units, c], show.loc[units, LABEL[c]]
        rho = spearmanr(own, drawn).statistic
        print(f"    {LABEL[c][:40]:<42} n={n:<5} Spearman(drawn, own) {rho:+.2f}   "
              + "  ".join(f"{u[2:]} {drawn[u] * 100:.1f}/{own[u] * 100:.1f}" for u in units))
    return out, small


def main():
    if "--fetch" in sys.argv:
        fetch()
    for y in FILES:
        if not os.path.exists(os.path.join(RAW, f"lapop_bo_{y}.dta")):
            raise SystemExit(f"lapop_bo_{y}.dta missing — run with --fetch")
    for p in (POP, LOOKUP, MUNIS):
        if not os.path.exists(p):
            raise SystemExit(f"{p} missing — run sources/bo_geo.py")

    lut = pd.read_csv(LOOKUP, dtype={"geo_id": str})
    names = dict(zip(lut["geo_id"], lut["name"]))
    pop = pd.read_csv(POP, encoding="utf-8-sig", dtype={"geo_id": str}).set_index("geo_id")["pop"]
    munis = pd.read_csv(MUNIS, dtype=str)
    if len(names) != N_UNITS or sorted(pop.index) != sorted(names):
        raise SystemExit("the lookup and the population file disagree")
    units = sorted(names)

    print("Bolivia, LAPOP single-country files. The department labels, checked:")
    frames, cards = {}, {}
    for y in [CHECK_WAVE] + WAVES:
        d, lab, relvar = read_wave(y)
        frames[y] = decode_wave(y, d, lab, munis, names)
        cards[y] = (relvar, {int(k): v for k, v in lab["rel"].items()
                             if isinstance(k, (int, float))})
    held_out_waves(frames, pop)

    df = pd.concat([frames[y] for y in WAVES], ignore_index=True)
    df = df[df["rel"].notna()].copy()
    df["code"] = df["rel"].astype(int)
    unknown = sorted(set(df["code"]) - set(lapop.CATEGORY))
    if unknown:
        raise SystemExit(f"religion codes with no label on the standard card: {unknown}")
    for y in WAVES:
        got = set(frames[y]["rel"].dropna().astype(int))
        miss = [c for c in got if c not in cards[y][1]]
        if miss:
            raise SystemExit(f"{y}: codes with no value label in the file: {miss}")
    if len(df) != N_RESPONDENTS:
        raise SystemExit(f"{len(df):,} respondents, expected {N_RESPONDENTS:,}")
    df = poststratify(df, pop)
    nat = df.groupby("code")["w"].sum() / df["w"].sum()
    print(f"\nBolivia: {len(df):,} respondents with a religion answer, waves "
          f"{WAVES[0]}-{WAVES[-1]}, all {df['geo_id'].nunique()} departments in every wave")
    print("  respondents by wave and department:")
    print("    " + df.groupby(["geo_id", "wave"]).size().unstack().to_string().replace("\n", "\n    "))

    print("\n  by wave, weighted % (post-stratified):")
    wv = df.groupby(["wave", "code"])["w"].sum().unstack(fill_value=0.0)
    wv = wv.div(wv.sum(axis=1), axis=0) * 100
    print("    " + wv.round(2).to_string().replace("\n", "\n    "))

    check_2008(frames[CHECK_WAVE], df, names)

    cats = sorted(nat.index, key=lambda k: -nat[k])
    fres = stability(df, units, "geo_id", cats, "9 departments")
    regions = sorted(set(REGION.values()))
    rres = stability(df, regions, "region", cats, "6 design strata")
    du = df.copy()
    du.loc[du["code"].isin([2, 5]), "code"] = UNION
    stability(du, units, "geo_id", [UNION], "9 departments, the two Protestant boxes as one")
    b2 = (df.assign(h=(df["code"] == 2) * df["w"]).groupby("geo_id")["h"].sum()
          / df.groupby("geo_id")["w"].sum())
    b5 = (df.assign(h=(df["code"] == 5) * df["w"]).groupby("geo_id")["h"].sum()
          / df.groupby("geo_id")["w"].sum())
    from scipy.stats import spearmanr
    print(f"    the two boxes' pooled department shares against each other: Spearman "
          f"{spearmanr(b2, b5).statistic:+.2f} (negative would mean respondents swap boxes by place)")

    fine = [c for c in cats if fres[c]["passed"]]
    refused = [c for c in cats if fres[c]["refused"]]
    coarse = [c for c in cats if rres[c]["passed"] and c not in fine]
    standouts = {c: fres[c]["top_unit"] for c in cats
                 if c not in fine and c not in coarse and fres[c]["top_frac"] >= STANDOUT_AGREE
                 and fres[c]["chi"] < STAB_ALPHA and fres[c]["cell"] < CLUSTER_CAP}
    print(f"\n  selected: department {fine}, refused {refused}, design stratum only {coarse}, "
          f"standouts {standouts}")
    if (CARRIES, REFUSED, COARSE, STANDOUTS) != (fine, refused, coarse, standouts):
        raise SystemExit(f"the tests now select {fine} / {refused} / {coarse} / {standouts}, not "
                         f"{CARRIES} / {REFUSED} / {COARSE} / {STANDOUTS}. That is a change in "
                         "what this country claims to know: read the tables above, then update "
                         "the constants and the docstring deliberately.")

    print(f"\n  -> {len(fine)} answers on department shares, {len(coarse)} on design-stratum "
          f"shares, {len(standouts)} standing apart in one department, the rest at the national "
          "rate inside each department's residual")
    out, small = compose(df, nat, fine, coarse, standouts, pop, units)

    out["geo_level"] = "departamento"
    out["geo_name"] = out["geo_id"].map(names)
    out["basis"] = "self_id"
    out["year"] = "2010-2023"
    out["source_id"] = SOURCE_ID
    n_by = df.groupby("geo_id").size()
    out["note"] = [f"LAPOP AmericasBarometer waves 2010-2023 pooled, n={int(n_by[g])} in this "
                   f"department; {b} applied to the 2024 census population"
                   for g, b in zip(out["geo_id"], out["basis_note"])]
    total = int(out["count"].sum())
    if total != int(pop.sum()):
        raise SystemExit(f"drawn {total:,} against {int(pop.sum()):,}")
    for g in units:
        if int(out.loc[out["geo_id"] == g, "count"].sum()) != int(pop[g]):
            raise SystemExit(f"{g} does not close on its census population")
    cols = ["geo_id", "geo_level", "geo_name", "source_category", "count",
            "basis", "year", "source_id", "note"]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out[cols].to_csv(OUT, index=False, encoding="utf-8")
    print(f"\nwrote {OUT} ({len(out)} rows, {total:,} people, "
          f"{out['source_category'].nunique()} categories, {out['geo_id'].nunique()} units)")

    print("\n  national, as drawn:")
    drawn = out.groupby("source_category")["count"].sum().sort_values(ascending=False)
    for cat, v in drawn.items():
        print(f"    {v / total * 100:6.2f}%  {int(v):>12,}  {cat}")

    placed = fine + coarse + list(standouts)
    show = out.pivot_table(index="geo_id", columns="source_category", values="count",
                           aggfunc="sum")
    show = show.div(show.sum(axis=1), axis=0)
    print("\n  the placed answers by department, with the sample behind each:")
    cs = [LABEL[c] for c in placed]
    print(f"    {'department':<14}{'n':>6}" + "".join(f"{c.split(' ')[0][:11]:>12}" for c in cs))
    for g in show[cs[0]].sort_values().index if cs else units:
        print(f"    {names[g][:12]:<14}{int(n_by[g]):>6}"
              + "".join(f"{show.loc[g, c] * 100:11.1f}%" for c in cs))
    for c in (7, 3, 77):
        t = df[df["code"] == c]
        print(f"\n  `{LABEL[c]}` ({len(t)} respondents) by department: "
              + ", ".join(f"{names[g]} {n}" for g, n in t.groupby("geo_id").size()
                          .sort_values(ascending=False).items())
              + "; by wave: " + ", ".join(f"{w} {n}" for w, n in t.groupby("wave").size().items()))


if __name__ == "__main__":
    main()
