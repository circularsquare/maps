"""Venezuela — religion by federal entity, from LAPOP's AmericasBarometer single-country files, 2010-2016/17.

Reads data/raw/ve/lapop_ve_<year>.dta and writes data/normalized/ve.csv. `sources/ve.md` is this
country's record; `sources.md` §11ap is the scouting. `sources/lapop.py` holds the answer card and
`sources/bo.py` is the pattern this copies; everything below is Venezuela's own.

**VENEZUELA IS NOT IN THE FREE GRAND MERGE AFTER 2008** (§11ad, §11ae). LAPOP publishes each
Venezuelan wave through 2016/17 as its own file, Free Tier, behind the site usage agreement the
merge is used under (§11ap). There is no later wave. No Venezuelan census asks religion (§11ac:
the 2011 REDATAM base has 67 person variables and none is religion). Every row is `modelled`.

## THE WAVES AND THE WEIGHTS

2010, 2012, 2014 and 2016/17, all on `q3c` and LAPOP's standard card (2016/17 adds `Otro` and
still offers Witnesses and Mormons, Bolivia's pattern). 5,894 respondents answered. **No wave has a
design weight**: 2010 carries no `wt` at all and 2012-2016's is 1 for everyone, while the design
allocates interviews by region and leaves states out (Falcón is 0.32x its population share in
2012, Cojedes 1.77x). Every (wave, state) is post-stratified to the state's 2011 census share of
1,500, `bo.py::poststratify`.

`lapop.wave_flags` raises two flags, both 2016/17: Eastern religions (19 respondents against 4
expected) and traditional religions (26 against 6). Judged in WAVE_FLAGS below and kept.

## THE STATE LABELS, CHECKED WITHOUT TRUSTING THEM

`prov` is alphabetical over 21 states in 2010 and a different order over 17 in 2012-2016 (1601 is
Anzoátegui in 2010 and the Distrito Capital after). Each wave's labels are joined to COD-AB by
name, and then:

  * **`municipio` names**, against COD-AB's 336 municipalities: the states each prov code's
    municipality names can belong to intersect in the labelled state, in every wave. Seven labels
    needed a reasoned alias (renamings and misspellings, MUNI_ALIAS). A state sampled in one
    municipality with a shared name (Libertador, Miranda, Ezequiel Zamora) is settled by
    elimination, and the Distrito Capital, left beside unsampled Monagas in 2012-2016, by `tamano`
    (every respondent in the national capital's metropolitan area).
  * **2012, 2014 and 2016 share one municipality numbering**: each of its 60 codes sits under one
    state in all three. 2016's unlabelled 1600028 is 2012's Sifontes, in Bolívar.
  * **Sample against the 2011 census**, reported: r=+0.99 in 2010 and +0.91 to +0.92 after, none of
    20,000 random pairings reaching any of them.

## WHICH ANSWERS CARRY THEIR OWN GEOGRAPHY

`bo.py`'s test on 20,000 draws (see STAB_PERM), at the 17 states sampled in every wave and at
2010's six design regions, which nest them. Placed on state shares: evangelical, traditional
religions. **Catholic fails at the state (p=0.052) and passes at the region (p=0.044)**, and so do
Jehovah's Witnesses (0.44 and 0.021), so both are drawn on region shares. Eastern religions pass
the rank test and fail the chi-square (refused). The rest take each state's residual at national
proportions; the 2x rule does not fire (worst 1.58x). **The residual reverses two answers against
the survey's own state shares** (believers without a church Spearman +0.13, Falcón drawn 11.7%
against 1.3% measured on 95 interviews; traditional Protestant -0.28, Bolívar 2.5% against 9.8%),
because Catholic at the region share leaves a state whose own Catholic share is above its region's
a large remainder. Spec §12 (Latvia): a reversal asks for a witness and licenses no override, and
Venezuela has none (sources/ve.md §8).

## FOUR STATES FROM ONE ROUND, FOUR BLANK

Apure, Barinas, Monagas and La Guaira (Vargas) were sampled in 2010 only. `co.py::region_fallback`
on 2010's regions puts Apure and Barinas on the llanos region's every-round states and La Guaira on
the capital region's; Monagas stays at the national rate for the state-level answers. Amazonas,
Delta Amacuro, Nueva Esparta and the Dependencias Federales were never sampled: 805,770 people,
2.96% of the 2011 census, not drawn.

Usage:
    python sources/ve.py --fetch    download the four Stata files from LAPOP (~3 MB)
    python sources/ve.py            rebuild data/normalized/ve.csv
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
RAW = os.path.join(ROOT, "data", "raw", "ve")
GEO = os.path.join(ROOT, "data", "geo", "ve")
POP = os.path.join(GEO, "ve_pop_2011.csv")
LOOKUP = os.path.join(GEO, "ve_lookup.csv")
MUNIS = os.path.join(GEO, "ve_municipios.csv")
OUT = os.path.join(ROOT, "data", "normalized", "ve.csv")

# LAPOP's data directory, `?lp_download=<id>` for the Stata file of each wave (§11ap).
URL = "https://www.vanderbilt.edu/center-for-global-democracy/data/directory/?lp_download={}"
FILES = {2010: 1582, 2012: 1970, 2014: 2020, 2016: 2195}
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) religiondots/1.0"}

WAVES = [2010, 2012, 2014, 2016]
EARLY = [2010, 2012]                # the chronological halving, printed only
SOURCE_ID = "ve_lapop_2010_2016"
N_UNITS = 25
N_RESPONDENTS = 5_894
WEIGHT_TOTAL = 1500.0

# LAPOP state labels (folded) that are not COD-AB's name for the same entity.
STATE_ALIAS = {"vargas": "laguaira"}          # renamed La Guaira in 2019

# 2010's six design regions (`estratopri`), read from the 2010 file and asserted there. They are
# the only scheme that holds the four states sampled in 2010 alone; 2012-2016 use eight regions
# and move Falcón, Cojedes and Bolívar between them, so neither scheme nests every wave.
REGION = {
    "VE01": "capital", "VE15": "capital", "VE24": "capital",
    "VE05": "centro-occidental", "VE08": "centro-occidental", "VE09": "centro-occidental",
    "VE13": "centro-occidental", "VE22": "centro-occidental",
    "VE04": "los llanos", "VE06": "los llanos", "VE12": "los llanos", "VE18": "los llanos",
    "VE14": "occidental", "VE20": "occidental", "VE21": "occidental",
    "VE03": "oriental", "VE07": "oriental", "VE16": "oriental", "VE19": "oriental",
    "VE11": "zuliana", "VE23": "zuliana",
}

ONE_ROUND = {
    "VE04": "Apure: sampled in 2010 only",
    "VE06": "Barinas: sampled in 2010 only",
    "VE16": "Monagas: sampled in 2010 only",
    "VE24": "La Guaira (Vargas): sampled in 2010 only",
}
NOT_DRAWN = {"VE02": "Amazonas", "VE10": "Delta Amacuro", "VE17": "Nueva Esparta",
             "VE25": "Dependencias Federales"}

# LAPOP municipality labels (folded) -> every COD-AB ADM2 name (folded) the label can stand for.
# The witness takes the union of the states holding any of them, so an alias never settles a
# state by itself; the other municipalities under the same prov code have to.
MUNI_ALIAS = {
    "heres": ("angosturadelorinoco",),           # Ciudad Bolívar's municipality, renamed 2021
    "guaicaipuro": ("bolivarianoguaicaipuro",),  # Los Teques, renamed Bolivariano Guaicaipuro
    "sancarlos": ("ezequielzamora",),            # San Carlos is the seat of Ezequiel Zamora (Cojedes)
    "zamora": ("zamora", "ezequielzamora"),      # Aragua's is Ezequiel Zamora (Villa de Cura);
                                                 #   Miranda's and Falcón's are Zamora
    "landertomaslander": ("lander",),            # 2012's "LANDER/ TOMAS LANDER"
    "fernadezfeo": ("fernandezfeo",),            # 2012's misspelling
    "turisticodiegobautistaurbaneja": ("diegobautistaurbaneja",),  # 2012's long form
}

# What the tests select, asserted so a change is a failure here rather than a silent redraw.
#   5 Evangélica y Pentecostal, 7 Religiones Tradicionales at the state;
#   1 Católico, 12 Testigos de Jehová at 2010's design region
CARRIES = [5, 7]
REFUSED = [3]                       # rank test passes, chi-square does not
COARSE = [1, 12]                    # pass at the six regions and not at the 17 states
STANDOUTS = {}
ON_REGION = ["VE04", "VE06", "VE24"]   # Apure, Barinas (los llanos); La Guaira (capital)

# lapop.wave_flags on Venezuela's four waves, with the judgement beside them.
WAVE_FLAGS = {("high", 3, 2016), ("high", 7, 2016)}
WAVE_FLAGS_JUDGED = (
    "not Honduras 2016's shifted codes: there evangelicals fell where code 3 rose; here the "
    "Christian answers keep their 2010-2014 course through 2016/17 (Catholic 78.1, 79.9, 73.7, "
    "67.5; evangelical 6.1, 9.0, 10.3, 13.0), code 7 was already rising (0.06, 0.46, 0.79, 1.79), "
    "and the 19 and 26 respondents sit in 18 and 24 sampling cells across 9 and 7 states. Both "
    "answers are small; the pooled shares of both are inflated by the wave, and say so in the "
    "mapping (sources/ve.md §5)")

# 20,000 draws, not bo.py's and co.py's 2,000. On 2,000, Catholic at the 17 states came out
# p=0.0500 on seed 0 and 0.045-0.0575 on seeds 1-5 (sources/ve.md §7): a verdict set by the
# seed, since the Monte Carlo error at p=0.05 on 2,000 draws is about 0.005. The test and the
# bar are unchanged; only the estimate of p is ten times finer.
STAB_PERM, STAB_ALPHA, STAB_SEED = 20_000, 0.05, 0
STANDOUT_AGREE = 0.95
CLUSTER_CAP = 0.5
UNION = 25                          # traditional Protestant + evangelical, tested, never drawn

LABEL = dict(lapop.CATEGORY)
LABEL[UNION] = "Protestante tradicional + Evangélica (tested together)"


def fold(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return "".join(ch for ch in s.lower() if ch.isalnum())


def fetch():
    os.makedirs(RAW, exist_ok=True)
    for y, i in FILES.items():
        dst = os.path.join(RAW, f"lapop_ve_{y}.dta")
        if os.path.exists(dst) and os.path.getsize(dst) > 400_000:
            print(f"  have lapop_ve_{y}.dta ({os.path.getsize(dst):,} bytes)")
            continue
        req = urllib.request.Request(URL.format(i), headers=UA)
        with urllib.request.urlopen(req, timeout=600) as r, open(dst + ".part", "wb") as f:
            f.write(r.read())
        with open(dst + ".part", "rb") as fh:
            head = fh.read(11)
        if not (head.startswith(b"<stata_dta>") or head[:1] in (b"\x71", b"\x72", b"\x73")):
            raise SystemExit(f"{y}: the download is not a Stata file (starts {head!r}); the "
                             "site agreement may have become a real gate")
        os.replace(dst + ".part", dst)
        print(f"  got  lapop_ve_{y}.dta ({os.path.getsize(dst):,} bytes)")


def read_wave(y):
    import pyreadstat
    df, meta = pyreadstat.read_dta(os.path.join(RAW, f"lapop_ve_{y}.dta"),
                                   apply_value_formats=False)
    cols = {c.lower(): c for c in df.columns}
    rels = [c for c in ("q3", "q3c", "q3cn") if c in cols]
    if rels != ["q3c"]:
        raise SystemExit(f"{y}: religion variables {rels}, expected q3c alone")

    def labels(var):
        if var not in cols:
            return {}
        ls = meta.variable_to_label.get(cols[var])
        return meta.value_labels.get(ls, {}) if ls else {}

    num = lambda v: (pd.to_numeric(df[cols[v]], errors="coerce") if v in cols
                     else pd.Series(np.nan, index=df.index))
    # 2010's `cluster` runs 1-10 inside each `upm`, so a sampling cell is the pair
    cell = df[cols["upm"]].astype(str) + "-" + df[cols["cluster"]].astype(str)
    out = pd.DataFrame({
        "wave": y,
        "prov": num("prov"),
        "muni": num("municipio"),
        "estratopri": num("estratopri"),
        "tamano": num("tamano"),
        "cluster": cell,
        "wt": num("wt"),
        "rel": num("q3c"),
    })
    if out["prov"].isna().any() or out["muni"].isna().any():
        raise SystemExit(f"{y}: respondents with no prov or no municipio")
    # No wave carries a design weight: 2010 has no `wt` at all, and 2012-2016's is 1 for everyone.
    # The design allocates by region and leaves states out, so every wave is post-stratified.
    if "wt" in cols and not (out["wt"] == 1).all():
        raise SystemExit(f"{y}: `wt` varies; read the design before post-stratifying over it")
    if "wt" not in cols and y != 2010:
        raise SystemExit(f"{y}: no `wt` column")
    lab = {"prov": labels("prov"), "muni": labels("municipio"),
           "estratopri": labels("estratopri"), "rel": labels("q3c")}
    return out, lab


def decode_wave(y, d, lab, munis, names):
    """State labels to COD pcodes by name, then two witnesses that do not use the label."""
    by_name = {fold(n): g for g, n in names.items()}
    pl = {int(k): v for k, v in lab["prov"].items() if isinstance(k, (int, float))}
    codes = sorted(set(d["prov"].astype(int)))
    pmap = {}
    for p in codes:
        f = fold(pl.get(p, ""))
        f = STATE_ALIAS.get(f, f)
        if f not in by_name:
            raise SystemExit(f"{y}: prov {p} label {pl.get(p)!r} is no COD entity name")
        pmap[p] = by_name[f]
    if len(set(pmap.values())) != len(pmap):
        raise SystemExit(f"{y}: two prov codes name one entity: {pmap}")
    d["geo_id"] = d["prov"].astype(int).map(pmap)

    # every state inside one design region
    nest = d.groupby("geo_id")["estratopri"].nunique()
    if (nest > 1).any():
        raise SystemExit(f"{y}: states in two design regions: {sorted(nest[nest > 1].index)}")
    if y == 2010:
        el = {int(k): v for k, v in lab["estratopri"].items() if isinstance(k, (int, float))}
        got = {g: fold(el[int(r)]).replace("region", "") for g, r in
               d.groupby("geo_id")["estratopri"].first().items()}
        want = {g: fold(r) for g, r in REGION.items()}
        if got != want:
            raise SystemExit(f"2010: design regions are {got}, not REGION")
    d["region"] = d["geo_id"].map(REGION)

    # municipio names against COD-AB's 336 municipalities
    adm2 = {}
    for n2, p in zip(munis["adm2_name"], munis["adm1_pcode"]):
        adm2.setdefault(fold(n2), set()).add(p)
    ml = {int(k): v for k, v in lab["muni"].items() if isinstance(k, (int, float))}
    unresolved, derived, amb, loose = [], {}, 0, {}
    for p, g in d.groupby("prov"):
        sets = []
        for m in sorted(g["muni"].astype(int).unique()):
            nm = ml.get(m)
            if nm is None:
                continue
            f = fold(nm)
            s = set().union(*(adm2.get(a, set()) for a in MUNI_ALIAS.get(f, (f,))))
            if not s:
                unresolved.append((int(p), m, nm))
                continue
            sets.append(s)
            amb += len(s) > 1
        derived[int(p)] = set.intersection(*sets) if sets else set()
    if unresolved:
        raise SystemExit(f"{y}: municipio labels COD-AB has no municipality for: {unresolved} — "
                         "add a reasoned MUNI_ALIAS or stop")
    wrong = {p: sorted(s) for p, s in derived.items() if pmap[p] not in s}
    if wrong:
        raise SystemExit(f"{y}: the municipio names do not place these prov codes in the state "
                         f"the label names: {[(p, pl[p], s) for p, s in wrong.items()]}")
    # A state sampled in one municipality whose name other states share (Libertador, Miranda,
    # Ezequiel Zamora) is settled by elimination: the decode is one-to-one, so every state another
    # prov code is settled on is out. Whatever is left must be the labelled state alone, except
    # the Distrito Capital, which can be left beside an unsampled state that also has a
    # Libertador and is settled by `tamano` instead: every respondent under it is in the national
    # capital's metropolitan area, which no other state with a Libertador is.
    settled = {pmap[p] for p, s in derived.items() if len(s) == 1}
    loose, by_tamano = {}, []
    for p, s in derived.items():
        if len(s) == 1:
            continue
        left = s - (settled - {pmap[p]})
        loose[pmap[p]] = sorted(s)
        if left == {pmap[p]}:
            continue
        cap = d.loc[d["prov"] == p, "tamano"]
        if pmap[p] == "VE01" and (cap == 1).all():
            by_tamano.append(f"{sorted(left)} settled on VE01 by tamano=1 for all {len(cap)}")
            continue
        raise SystemExit(f"{y}: prov {p} ({pl[p]}) is left on {sorted(left)} after elimination")
    n_m = int(d["muni"].nunique())
    unl = sorted(set(d["muni"].astype(int)) - set(ml))
    print(f"    {y}: {len(codes)} states; {n_m} municipalities, each state's names intersect in "
          f"the labelled state ({amb} names shared between states); one-municipality states "
          f"settled by elimination: {sorted(loose) or 'none'}{'; ' if by_tamano else ''}"
          f"{'; '.join(by_tamano)}; municipio codes with no label: {unl or 'none'}")
    return d, loose, {m: ml.get(m) for m in set(d["muni"].astype(int))}


def check_code_continuity(frames, mlabels):
    """2012, 2014 and 2016 share one municipality numbering (1601 -> 1600001) and one prov order.
    Each code must sit under the same state in every one of them, with the same folded name
    wherever both waves label it."""
    seen = {}
    for y in (2012, 2014, 2016):
        d = frames[y]
        for m, g in d.groupby("muni")["geo_id"]:
            m = int(m)
            key = m - 1600000 + 1600 if m > 1_000_000 else m
            states = set(g)
            if len(states) != 1:
                raise SystemExit(f"{y}: municipio {m} spans states {states}")
            st, nm = states.pop(), mlabels[y].get(m)
            if key in seen:
                st0, nm0, y0 = seen[key]
                if st != st0:
                    raise SystemExit(f"municipio {key} is in {st0} in {y0} and {st} in {y}")
                if nm and nm0 and fold(nm) != fold(nm0) and fold(nm0) not in fold(nm) \
                        and fold(nm) not in fold(nm0):
                    print(f"      municipio {key}: {nm0!r} in {y0}, {nm!r} in {y}")
            else:
                seen[key] = (st, nm, y)
            if nm is None:
                print(f"      {y}: municipio {m} has no label; the same code is "
                      f"{seen[key][1]!r} ({seen[key][2]}) in the same state")
    print(f"    2012-2016: {len(seen)} municipality codes, each under one state in every wave")


def held_out_waves(frames, pop):
    """Each wave's sample share against the 2011 census over the states it sampled. REPORTED:
    the design allocates by region and leaves states out, so a sampled state stands for its
    unsampled neighbours and the shares are not the census's."""
    import lits
    for y in WAVES:
        d = frames[y]
        units = sorted(d["geo_id"].unique())
        print(f"\n  {y}, unweighted, {len(units)} states:", end="")
        try:
            lits.held_out(d.assign(w=1.0), pop.loc[units], f"Venezuela {y}",
                          pop_source="the 2011 census")
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
    """`bo.py::stability`, copied because it reads its module's globals: §9cy's median over every
    halving against a per-wave permutation null, Sweden's chi-square veto, Uzbekistan's
    largest-cell refusal and Honduras's which-unit-tops-both-halves, on unweighted counts."""
    from scipy.stats import spearmanr

    import stability as shared
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
        codes = [c]

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


def region_fallback(df, fine_units, one_round, cats, unit_col="geo_id", region_col="region",
                    names=None, n_perm=2000, seed=0):
    """`co.py::region_fallback`, copied because co.py cannot be imported for it (spec §12,
    2026-09-14): a one-round unit takes its design region's shares when the region, leaving each
    of its every-round units out in turn, predicts them better than the country (lower mean error
    and closer for more than half), and the national rate otherwise."""
    names = names or {}
    parent = pd.Series(REGION)
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
    print("\n  one-round states: region or country? Leave-one-out over each 2010 design region's "
          "every-round states, summed absolute error over "
          + ", ".join(lapop.CATEGORY[c].split(" ")[0] for c in cats) + ", in points:")
    use = {}
    for r in sorted(parent[units].unique()):
        members = [u for u in units if parent[u] == r]
        ones = sorted(u for u in one_round if parent[u] == r)
        tag = ", ".join(names.get(u, u) for u in ones) or "no one-round state"
        if len(members) < 2:
            print(f"    {r} ({tag}): {len(members)} every-round state, cannot be tested -> "
                  "national rate")
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
        print(f"    {r} ({tag}): mean region {er.mean():.2f}, country {en.mean():.2f}; region "
              f"closer in {closer} of {k}; a random {k} beat the country by as much in "
              f"p={p:.3f} -> {'REGION' if ok else 'national rate'}")
        if ok:
            use.update({u: r for u in ones})
    return use


def compose(pools, nat, fine, coarse, standouts, pop, units, rshare, basis_of):
    """Placed shares per state from its pool (its own respondents, or its region's every-round
    states), then the national mix inside each state's residual. `bo.py::compose`'s construction."""
    cats = list(nat.index)
    small = [c for c in cats if c not in fine and c not in coarse and c not in standouts]
    rows, resid = [], {}
    for u in units:
        d = pools[u]
        b = d.groupby("code")["w"].sum().reindex(cats, fill_value=0.0)
        ushare = b / b.sum()
        placed = {}
        for c in fine:
            placed[c] = (ushare[c], basis_of[u])
        for c in coarse:
            placed[c] = (rshare.loc[REGION[u], c], f"share in the {REGION[u]} design region")
        for c, u0 in standouts.items():
            raise SystemExit("standouts are not wired for Venezuela; read bo.py::compose")
        residual = 1.0 - sum(v for v, _ in placed.values())
        if residual <= 0:
            raise SystemExit(f"{u}: no room for the tail")
        resid[u] = residual
        p = int(pop[u])
        small_total = float(nat[small].sum())
        cells = [[u, LABEL[c], v * p, why] for c, (v, why) in placed.items()]
        cells += [[u, LABEL[c], residual * nat[c] / small_total * p,
                   "national share within the state's residual"] for c in small]
        cnt = np.array([x[2] for x in cells])
        r = np.round(cnt).astype(np.int64)
        r[int(np.argmax(cnt))] += p - int(r.sum())
        rows += [(x[0], x[1], int(v), x[3]) for x, v in zip(cells, r)]
    out = pd.DataFrame(rows, columns=["geo_id", "source_category", "count", "basis_note"])
    return out, small, pd.Series(resid)


def main():
    if "--fetch" in sys.argv:
        fetch()
    for y in FILES:
        if not os.path.exists(os.path.join(RAW, f"lapop_ve_{y}.dta")):
            raise SystemExit(f"lapop_ve_{y}.dta missing — run with --fetch")
    for p in (POP, LOOKUP, MUNIS):
        if not os.path.exists(p):
            raise SystemExit(f"{p} missing — run sources/ve_geo.py")

    lut = pd.read_csv(LOOKUP, dtype=str)
    names = dict(zip(lut["geo_id"], lut["name"]))
    pop = pd.read_csv(POP, encoding="utf-8-sig", dtype={"geo_id": str}).set_index("geo_id")["pop"]
    munis = pd.read_csv(MUNIS, dtype=str)
    if len(names) != N_UNITS or sorted(pop.index) != sorted(names):
        raise SystemExit("the lookup and the population file disagree")

    print("Venezuela, LAPOP single-country files. The state labels, checked:")
    frames, mlabels, cards = {}, {}, {}
    for y in WAVES:
        d, lab = read_wave(y)
        frames[y], _loose, mlabels[y] = decode_wave(y, d, lab, munis, names)
        cards[y] = {int(k): v for k, v in lab["rel"].items() if isinstance(k, (int, float))}
    check_code_continuity(frames, mlabels)
    held_out_waves(frames, pop)

    df = pd.concat([frames[y] for y in WAVES], ignore_index=True)
    df = df[df["rel"].notna()].copy()
    df["code"] = df["rel"].astype(int)
    unknown = sorted(set(df["code"]) - set(lapop.CATEGORY))
    if unknown:
        raise SystemExit(f"religion codes with no label on the standard card: {unknown}")
    for y in WAVES:
        miss = [c for c in set(frames[y]["rel"].dropna().astype(int)) if c not in cards[y]]
        if miss:
            raise SystemExit(f"{y}: codes with no value label in the file: {miss}")
    off = sorted({(c, w) for c, w in zip(df["code"], df["wave"])
                  if w in lapop.CARD_ABSENT.get(c, ())})
    if off:
        raise SystemExit(f"respondents on codes lapop.CARD_ABSENT says were off the card: {off}")
    flags = lapop.wave_flags(df)
    print("\n  per-wave codes (lapop.wave_flags):")
    for k, c, w, o, e in flags:
        print(f"    {k:<5} code {c:>4} ({LABEL[c][:40]}) in {w}: {o} respondents against "
              f"{e:.1f} expected")
    if WAVE_FLAGS is not None and {(k, c, w) for k, c, w, _o, _e in flags} != WAVE_FLAGS:
        raise SystemExit("the per-wave flags changed; read them and re-judge WAVE_FLAGS")
    if N_RESPONDENTS is not None and len(df) != N_RESPONDENTS:
        raise SystemExit(f"{len(df):,} respondents, expected {N_RESPONDENTS:,}")

    df = poststratify(df, pop)
    nat = df.groupby("code")["w"].sum() / df["w"].sum()
    print(f"\nVenezuela: {len(df):,} respondents with a religion answer, waves "
          f"{WAVES[0]}-{WAVES[-1]}, {df['geo_id'].nunique()} of {N_UNITS} entities")
    print("  respondents by wave and state:")
    print("    " + df.groupby(["geo_id", "wave"]).size().unstack(fill_value=0)
          .rename(index=names).to_string().replace("\n", "\n    "))

    print("\n  by wave, weighted % (post-stratified):")
    wv = df.groupby(["wave", "code"])["w"].sum().unstack(fill_value=0.0)
    wv = wv.div(wv.sum(axis=1), axis=0) * 100
    print("    " + wv.round(2).to_string().replace("\n", "\n    "))

    fine_units = sorted(u for u, s in df.groupby("geo_id") if s["wave"].nunique() == len(WAVES))
    at_one = sorted(set(df["geo_id"]) - set(fine_units))
    blank = sorted(set(names) - set(df["geo_id"]))
    if set(at_one) != set(ONE_ROUND) or set(blank) != set(NOT_DRAWN):
        raise SystemExit(f"one-round {at_one}, never sampled {blank}: the sample has changed")
    gap = int(pop.loc[blank].sum())
    print(f"\n  {len(fine_units)} states in every wave; {len(at_one)} in one "
          f"({int(pop.loc[at_one].sum()):,} people, {pop.loc[at_one].sum() / pop.sum():.2%}); "
          f"{len(blank)} never sampled, NOT DRAWN: {gap:,} people ({gap / pop.sum():.2%}), "
          + ", ".join(NOT_DRAWN[g] for g in blank))

    cats = sorted(nat.index, key=lambda k: -nat[k])
    dfine = df[df["geo_id"].isin(fine_units)]
    fres = stability(dfine, fine_units, "geo_id", cats, f"{len(fine_units)} every-round states")
    regions = sorted({REGION[u] for u in fine_units})
    rres = stability(dfine, regions, "region", cats, f"{len(regions)} design regions (2010's)")
    du = dfine.copy()
    du.loc[du["code"].isin([2, 5]), "code"] = UNION
    stability(du, fine_units, "geo_id", [UNION], "every-round states, the two Protestant boxes as one")
    from scipy.stats import spearmanr
    b2 = (dfine.assign(h=(dfine["code"] == 2) * dfine["w"]).groupby("geo_id")["h"].sum()
          / dfine.groupby("geo_id")["w"].sum())
    b5 = (dfine.assign(h=(dfine["code"] == 5) * dfine["w"]).groupby("geo_id")["h"].sum()
          / dfine.groupby("geo_id")["w"].sum())
    print(f"    the two boxes' pooled state shares against each other: Spearman "
          f"{spearmanr(b2, b5).statistic:+.2f} (negative would mean respondents swap boxes by place)")

    fine = [c for c in cats if fres[c]["passed"]]
    refused = [c for c in cats if fres[c]["refused"]]
    coarse = [c for c in cats if rres[c]["passed"] and c not in fine]
    standouts = {c: fres[c]["top_unit"] for c in cats
                 if c not in fine and c not in coarse and fres[c]["top_frac"] >= STANDOUT_AGREE
                 and fres[c]["chi"] < STAB_ALPHA and fres[c]["cell"] < CLUSTER_CAP}
    print(f"\n  selected: state {fine}, refused {refused}, design region only {coarse}, "
          f"standouts {standouts}")
    if CARRIES is not None and (CARRIES, REFUSED, COARSE, STANDOUTS) != (fine, refused, coarse,
                                                                        standouts):
        raise SystemExit(f"the tests now select {fine} / {refused} / {coarse} / {standouts}, not "
                         f"{CARRIES} / {REFUSED} / {COARSE} / {STANDOUTS}. Read the tables above, "
                         "then update the constants and the docstring deliberately.")

    use_region = region_fallback(df, fine_units, at_one, fine, names=names)
    if ON_REGION is not None and sorted(use_region) != sorted(ON_REGION):
        raise SystemExit(f"region_fallback now puts {sorted(use_region)} on their region, not "
                         f"{ON_REGION}")

    rb = dfine.groupby(["region", "code"])["w"].sum().unstack(fill_value=0.0).reindex(
        columns=cats, fill_value=0.0)
    rshare = rb.div(rb.sum(axis=1), axis=0)
    pools, basis_of = {}, {}
    drawn_units = fine_units + at_one
    for u in fine_units:
        pools[u], basis_of[u] = df[df["geo_id"] == u], "state share"
    for u in at_one:
        why = ONE_ROUND[u].split(": ", 1)[1]
        if u in use_region:
            r = use_region[u]
            pools[u] = dfine[dfine["region"] == r]
            basis_of[u] = f"share in the {r} region's every-round states; {why}"
        else:
            # the state-level answers at the national share; an answer placed at the design
            # region is still drawn at the region's share, since the region is known
            pools[u] = df
            basis_of[u] = f"national share; {why}"
    out, small, resid = compose(pools, nat, fine, coarse, {}, pop, sorted(drawn_units), rshare,
                                basis_of)

    import stability as shared
    small_total = float(nat[small].sum())
    mult = (resid.loc[fine_units] / small_total)
    none = pd.DataFrame({c: [((df["geo_id"] == u) & (df["code"] == c)).sum() == 0
                             for u in fine_units] for c in small}, index=fine_units)
    rows_2x, worst = shared.residual_multiples(mult, none, small)
    print(f"\n  the tail is {resid.loc[fine_units].min():.1%} to {resid.loc[fine_units].max():.1%} "
          f"of a state against {small_total:.1%} nationally; the 2x rule's worst case: {worst}")
    if worst is not None and worst[2] >= shared.SMALL_CATEGORY_MULTIPLE:
        raise SystemExit("the residual draws a small answer at 2x its national share where the "
                         "survey found none; switch the tail to flat (spec §12)")

    print("\n  every category at the national rate, as drawn beside the survey's own state share "
          "(§12, Latvia: look for a reversal):")
    show = out.pivot_table(index="geo_id", columns="source_category", values="count",
                           aggfunc="sum")
    show = show.div(show.sum(axis=1), axis=0)
    ub = dfine.groupby(["geo_id", "code"])["w"].sum().unstack(fill_value=0.0)
    ushare = ub.div(ub.sum(axis=1), axis=0)
    for c in small:
        n = int((dfine["code"] == c).sum())
        if n < 20:
            continue
        own, drawn = ushare.reindex(fine_units)[c].fillna(0.0), show.loc[fine_units, LABEL[c]]
        rho = spearmanr(own, drawn).statistic
        print(f"    {LABEL[c][:40]:<42} n={n:<5} Spearman(drawn, own) {rho:+.2f}   "
              + "  ".join(f"{names[u][:6]} {drawn[u] * 100:.1f}/{own[u] * 100:.1f}"
                          for u in fine_units))

    out["geo_level"] = "estado"
    out["geo_name"] = out["geo_id"].map(names)
    out["basis"] = "self_id"
    out["year"] = "2010-2016/17"
    out["source_id"] = SOURCE_ID
    n_by = df.groupby("geo_id").size()
    out["note"] = [f"LAPOP AmericasBarometer waves 2010-2016/17 pooled, n={int(n_by[g])} in this "
                   f"state; {b} applied to the 2011 census population"
                   for g, b in zip(out["geo_id"], out["basis_note"])]
    total = int(out["count"].sum())
    if total != int(pop.sum()) - gap:
        raise SystemExit(f"drawn {total:,} against {int(pop.sum()) - gap:,}")
    for g in drawn_units:
        if int(out.loc[out["geo_id"] == g, "count"].sum()) != int(pop[g]):
            raise SystemExit(f"{g} does not close on its census population")
    if set(out["geo_id"]) & set(NOT_DRAWN):
        raise SystemExit("an unsampled entity was drawn")
    cols = ["geo_id", "geo_level", "geo_name", "source_category", "count",
            "basis", "year", "source_id", "note"]
    out = out.sort_values(["geo_id", "count"], ascending=[True, False])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out[cols].to_csv(OUT, index=False, encoding="utf-8")
    print(f"\nwrote {OUT} ({len(out)} rows, {total:,} people, "
          f"{out['source_category'].nunique()} categories, {out['geo_id'].nunique()} units)")

    print("\n  national, as drawn:")
    drawn = out.groupby("source_category")["count"].sum().sort_values(ascending=False)
    for cat, v in drawn.items():
        print(f"    {v / total * 100:6.2f}%  {int(v):>12,}  {cat}")

    placed = fine + coarse
    print("\n  the placed answers by state, with the sample behind each:")
    cs = [LABEL[c] for c in placed]
    print(f"    {'state':<20}{'n':>6}" + "".join(f"{c.split(' ')[0][:11]:>12}" for c in cs))
    for g in (show[cs[0]].sort_values().index if cs else drawn_units):
        print(f"    {names[g][:18]:<20}{int(n_by[g]):>6}"
              + "".join(f"{show.loc[g, c] * 100:11.1f}%" for c in cs))
    for c in (7, 3, 77, 12):
        t = df[df["code"] == c]
        print(f"\n  `{LABEL[c]}` ({len(t)} respondents) by state: "
              + ", ".join(f"{names[g]} {n}" for g, n in t.groupby("geo_id").size()
                          .sort_values(ascending=False).items())
              + "; by wave: " + ", ".join(f"{w} {n}" for w, n in t.groupby("wave").size().items()))


if __name__ == "__main__":
    main()
