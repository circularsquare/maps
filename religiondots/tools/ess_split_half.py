"""ESS split-half stability for the five ESS countries drawn before §9cy. REPORT ONLY.

    python tools/ess_split_half.py --fetch            # unweighted ESS passes for gr, fr, it
    python tools/ess_split_half.py gr fi fr de it     # print each country's table

Anita ruled on ask 012 on 2026-09-14: run the split-half Belgium and Sweden draw with on
Greece, Finland, France, Germany and Italy, with Sweden's spatial chi-square beside it, print
the tables into each `sources/<cc>.md`, and move no dots. This file is that and nothing else.
It writes nothing under data/normalized/, and no country module or build step reads it.

THE STATISTIC IS IMPORTED, NOT COPIED. The halvings, the median, the null, the p-value and the
chi-square are `sources/stability.py`'s, and alpha, draw count and seed are `sources/be.py`'s
`STAB_*` constants, so this report cannot quietly run a different test from the one Belgium
draws with. The splits are every combination of floor(R/2) rounds against its complement, and
for an even R only those containing the first round, so each halving is counted once (R=7 gives
Belgium's 35, R=4 gives Sweden's 3). The null permutes the unit labels PER ROUND, §9cy's bug.
The chi-square is the category against the rest of the tested base, over the units, on
unweighted counts. A category carries its own geography only with p < alpha on both.

THE POOL AND THE GEOGRAPHY ARE EACH COUNTRY'S OWN, read through that country's module:

    gr  rounds 5, 10, 11   13 NUTS 2 regions (gr.GR_TO_EL)     citizens, rlgdnm
    fi  rounds 5-11        19 maakunnat (fi.RECODE)            citizens, rlgblg x rlgdnafi
    fr  rounds 5-11        21 anciennes regions (fr.FR10_TO_16) citizens, rlgdnm
    de  rounds 6-9, 11     the Lander over de_ess.N_FLOOR      everyone, rlgdnade, as shares of
                                                               the register's residual
    it  rounds 9-11        5 ripartizioni                      citizens, the minority categories
        rounds 6, 8        the regioni over it.N_FLOOR         the Catholic : unaffiliated ratio

Counts are UNWEIGHTED, as in be.py and se.py. Greece, France and Italy were only ever fetched
weighted, so `--fetch` pulls the unweighted pass beside the weighted file with the same break
variables, as `data/raw/<cc>/ess_r<N>_n.json` (Finland's naming). The `national` column is
the WEIGHTED share of the tested base, which is what the build draws.
"""

import argparse
import importlib
import json
import os
import sys

# [[feedback_cap_cpu]]: several of these run side by side and the statistic is pure Python.
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "taxonomy"))
sys.path.insert(0, os.path.join(ROOT, "sources"))

import be  # noqa: E402  the constants
import stability  # noqa: E402  the statistic itself

ALPHA, PERM, SEED = be.STAB_ALPHA, be.STAB_PERM, be.STAB_SEED
REST = "__rest_of_base__"

# cc -> (ESS country code, break variables of the country's own weighted fetch)
FETCH = {
    "gr": ("GR", ["region", "ctzcntr", "rlgdnm"]),
    "fr": ("FR", ["region", "ctzcntr", "rlgdnm"]),
    "it": ("IT", ["region", "regunit", "ctzcntr", "rlgdnm"]),
}


# =======================================================================================
# fetch
# =======================================================================================

def fetch():
    for cc, (ess_cc, bv) in FETCH.items():
        mod = importlib.import_module(cc)
        for rnd, (fid, ver) in sorted(mod.ESS_ROUNDS.items()):
            dest = os.path.join(mod.RAW, f"ess_r{rnd}_n.json")
            if os.path.exists(dest):
                print(f"  {cc} round {rnd}: have {os.path.basename(dest)}")
                continue
            d = be._ess(be.ESS_TAB_N, {"id": fid, "v": ver, "bv": bv})
            hit = [x for x in d["analysis"]["frequencyTabulationByVariables"]["responses"]
                   if x["by"][0]["value"] == ess_cc]
            if not hit:
                sys.exit(f"!! {cc} round {rnd}: no {ess_cc} response")
            tmp = dest + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(hit[0]["response"], fh, ensure_ascii=False)
            os.replace(tmp, dest)
            n = sum(c["count"] for c in hit[0]["response"]["table"])
            print(f"  {cc} round {rnd}: {n:,.0f} unweighted -> {os.path.basename(dest)}")


# =======================================================================================
# reading
# =======================================================================================

def tidy(path):
    """One saved ESS response as a frame: region as its CODE, everything else as a LABEL.

    `path` indexes into codeList, as in gr.py, fr.py, it.py, fi.py and be.py. Missing flags
    ride along, but NOTHING HERE FILTERS ON THEM: ESS flags `Not applicable` (everyone who
    belongs to no religion) as missing, so refusals are dropped by each country's own
    EXCLUDED labels, the way its build drops them.
    """
    d = json.load(open(path, encoding="utf-8"))
    d = d.get("response", d)
    order = [v["name"] for v in d["variableValues"]]
    codes = {v["name"]: v["codeList"] for v in d["variableValues"]}
    rows = []
    for cell in d["table"]:
        rec = {"count": float(cell["count"])}
        for i, n in enumerate(order):
            c = codes[n][cell["path"][i]]
            rec[n] = c["value"] if n == "region" else c["label"]
            rec[n + "_miss"] = bool(c["isMissing"])
        rows.append(rec)
    return pd.DataFrame(rows)


def _frame(rows):
    return pd.DataFrame(rows, columns=["round", "unit", "cat", "count"])


def _pool(frames):
    df = pd.concat(frames, ignore_index=True)
    return df[df["count"] > 0][["round", "unit", "cat", "count"]]


# =======================================================================================
# the test
# =======================================================================================

def run(title, rounds, units, raw, wtd, tested, notes=()):
    """Print one table. `raw` / `wtd` are [round, unit, cat, count] over the tested BASE."""
    stray = sorted(set(raw["unit"]) - set(units))
    if stray:
        sys.exit(f"!! {title}: units outside the tested geography: {stray}")
    cols = list(tested) + [REST]
    ci = {c: i for i, c in enumerate(cols)}
    ri = {r: i for i, r in enumerate(rounds)}
    ui = {u: i for i, u in enumerate(units)}
    cube = np.zeros((len(rounds), len(units), len(cols)))
    for (rnd, u, c), v in raw.groupby(["round", "unit", "cat"])["count"].sum().items():
        cube[ri[rnd], ui[u], ci.get(c, ci[REST])] += v

    splits = stability.halvings(len(rounds))
    obs = stability.median_rho(cube, splits)[:len(tested)]            # the REST column is base only
    null = stability.wave_null(cube, splits, PERM, SEED)[:, :len(tested)]

    pooled = cube.sum(axis=0)
    per_unit = pooled.sum(axis=1)
    per_round = cube.sum(axis=(1, 2))
    wsum = wtd.groupby("cat")["count"].sum()
    wtot = float(wtd["count"].sum())
    k = len(splits[0][0])

    print(f"\n== {title}")
    print(f"   rounds {rounds}, {len(units)} units, {len(splits)} splits of {k} against "
          f"{len(rounds) - k}, {PERM}-draw per-round unit-label permutation null, seed {SEED}")
    print(f"   base: {per_unit.sum():,.0f} unweighted ({wtot:,.0f} weighted); per round "
          + ", ".join(f"r{r} {n:,.0f}" for r, n in zip(rounds, per_round)))
    print(f"   per unit: min {per_unit.min():,.0f} ({units[int(per_unit.argmin())]}), "
          f"median {np.median(per_unit):,.0f}, max {per_unit.max():,.0f} "
          f"({units[int(per_unit.argmax())]})")
    for n in notes:
        print(f"   {n}")
    print()
    print("| category | n | national | median rho | null 95th | p | chi² p | verdict |")
    print("|---|---:|---:|---:|---:|---:|---:|---|")
    moved, refused, kept = [], [], []
    for j, c in enumerate(tested):
        n = int(round(pooled[:, j].sum()))
        nat = 100 * float(wsum.get(c, 0.0)) / wtot
        a = pooled[:, j]
        chi = stability.chi2_p(a, per_unit)
        p, q95 = stability.permutation_p(obs[j], null[:, j], ALPHA)
        empty = int((a == 0).sum())
        if not np.isfinite(p):
            print(f"| {c} | {n:,} | {nat:.2f}% | | | | {chi:.1e} | no test possible |")
            moved.append((c, nat, "no test"))
            continue
        ok = p < ALPHA and np.isfinite(chi) and chi < ALPHA
        if ok:
            verdict, name = "**own geography**", f"**{c}**"
            kept.append((c, nat))
        elif p < ALPHA:
            verdict, name = "**refused on the chi-square**", c
            refused.append((c, nat))
        else:
            verdict, name = "national rate", c
            moved.append((c, nat, "rank"))
        print(f"| {name} | {n:,} | {nat:.2f}% | {obs[j]:+.3f} | "
              f"{q95:+.3f} | {p:.4f} | {chi:.1e} | {verdict} |"
              + (f"  <- {empty} empty units" if empty else ""))
    print()
    print(f"   own geography: {len(kept)} of {len(tested)}, "
          f"{sum(x[1] for x in kept):.2f}% of the base: "
          + ", ".join(x[0] for x in kept))
    gone = [(c, nat) for c, nat, _ in moved] + refused
    print(f"   would move to the national rate: {len(gone)}, "
          f"{sum(x[1] for x in gone):.2f}% of the base: "
          + ", ".join(x[0] for x in gone))
    return kept, gone


# =======================================================================================
# the countries, each through its own module
# =======================================================================================

def _order(wtd, cats):
    s = wtd.groupby("cat")["count"].sum()
    return sorted(cats, key=lambda c: -float(s.get(c, 0.0)))


def _rlgdnm(cc, rounds, recode, tax, suffix):
    """gr / fr / it: citizens, rlgdnm labels, EXCLUDED dropped, region recoded."""
    mod = importlib.import_module(cc)
    frames, dropped = [], 0.0
    for rnd in rounds:
        df = tidy(os.path.join(mod.RAW, f"ess_r{rnd}{suffix}.json"))
        df = df[df["ctzcntr"] == "Yes"]
        dropped += float(df.loc[df["region_miss"], "count"].sum())
        df = df[~df["region_miss"]]
        df = df[~df["rlgdnm"].isin(tax.EXCLUDED)]
        frames.append(df.assign(round=rnd, unit=df["region"].map(lambda r: recode.get(r, r)),
                                cat=df["rlgdnm"]))
    pool = _pool(frames)
    unknown = sorted(set(pool["cat"]) - set(tax.MAP))
    if unknown:
        sys.exit(f"!! {cc}: unmapped categories {unknown}")
    return pool, dropped


def country_gr():
    import gr
    import gr2024
    rounds = sorted(gr.ESS_ROUNDS)
    units = sorted(u for u in gr.NUTS2 if u != gr.ATHOS_UNIT)
    raw, d1 = _rlgdnm("gr", rounds, gr.GR_TO_EL, gr2024, "_n")
    wtd, _ = _rlgdnm("gr", rounds, gr.GR_TO_EL, gr2024, "")
    tested = _order(wtd, sorted(set(raw["cat"])))
    run("gr: Greek citizens, 13 NUTS 2 regions", rounds, units, raw, wtd, tested,
        [f"region-missing citizens dropped: {d1:,.0f}",
         "Mount Athos and the Thracian minority are authored and outside the test"])


def country_fi():
    import fi
    import fi2024
    rounds = sorted(fi.ESS_ROUNDS)
    units = sorted(fi.NUTS3)

    def load(tag):
        frames, dropped = [], 0.0
        for rnd in rounds:
            df = tidy(os.path.join(fi.RAW, f"ess_r{rnd}_{tag}.json"))
            df = df[df["ctzcntr"] == "Yes"].copy()
            dropped += float(df.loc[df["region_miss"], "count"].sum())
            df = df[~df["region_miss"]]
            df["cat"] = df.apply(fi._category, axis=1)
            df = df[df["cat"] != fi.REFUSAL]
            frames.append(df.assign(round=rnd,
                                    unit=df["region"].map(lambda r: fi.RECODE.get(r, r))))
        pool = _pool(frames)
        unknown = sorted(set(pool["cat"]) - set(fi2024.MAP))
        if unknown:
            sys.exit(f"!! fi: unmapped categories {unknown}")
        return pool, dropped

    raw, d1 = load("n")
    wtd, _ = load("w")
    tested = _order(wtd, sorted(set(raw["cat"])))
    run("fi: Finnish citizens, 19 maakunnat", rounds, units, raw, wtd, tested,
        [f"region-missing citizens dropped: {d1:,.0f}"])


def country_fr():
    import fr
    import fr2024
    rounds = sorted(fr.ESS_ROUNDS)
    units = sorted(fr.NUTS2)
    raw, d1 = _rlgdnm("fr", rounds, fr.FR10_TO_16, fr2024, "_n")
    wtd, _ = _rlgdnm("fr", rounds, fr.FR10_TO_16, fr2024, "")
    outside = float(raw.loc[~raw["unit"].isin(units), "count"].sum())
    raw = raw[raw["unit"].isin(units)]
    wtd = wtd[wtd["unit"].isin(units)]
    tested = _order(wtd, sorted(set(raw["cat"])))
    run("fr: French citizens, 21 anciennes regions", rounds, units, raw, wtd, tested,
        [f"region-missing citizens dropped: {d1:,.0f}; outside the 21: {outside:,.0f}",
         "the five overseas regions are Pew and outside the test"])


def country_de():
    import de_ess
    rounds = sorted(de_ess.ESS_ROUNDS)

    def load(tag):
        rows = []
        for rnd in rounds:
            path = os.path.join(de_ess.RAW, f"ess_de_r{rnd}_{tag}.json")
            rows += [(rnd, reg, den, c) for reg, den, c in de_ess._table(path)]
        df = _frame(rows)
        bad = sorted({x for x in set(df["unit"]) | set(df["cat"]) if x == "?"})
        if bad or not set(df["unit"]) <= set(de_ess.LAND):
            sys.exit("!! de: unreadable region or denomination labels")
        return df[df["count"] > 0]

    raw_all, wtd_all = load("n"), load("w")
    n_land = raw_all.groupby("unit")["count"].sum()
    thin = sorted(u for u in de_ess.LAND if float(n_land.get(u, 0.0)) < de_ess.N_FLOOR)
    units = sorted(u for u in de_ess.LAND if u not in thin)
    note_thin = ("below N_FLOOR and already drawn with the national composition, so not "
                 "tested: " + ", ".join(f"{u} n={n_land.get(u, 0):.0f}" for u in thin))

    def resid(df):
        return df[~df["cat"].isin(de_ess.REGISTER_ANSWERS)
                  & ~df["cat"].isin(de_ess.COUNTED_ELSEWHERE)
                  & df["unit"].isin(units)]

    raw, wtd = resid(raw_all), resid(wtd_all)
    tested = _order(wtd, sorted(set(raw["cat"]) - set(de_ess.NOT_DRAWN)))
    run("de: all respondents, the religions split out of the register's residual",
        rounds, units, raw, wtd, tested,
        [note_thin,
         "base = everyone but Catholic, EKD and Jewish; `national` is a share of that base",
         "Not applicable / Refusal / No answer / Don't know stay in `unrecorded` and are "
         "in the base, not tested"])

    ref_raw = raw_all[raw_all["unit"].isin(units)]
    ref_wtd = wtd_all[wtd_all["unit"].isin(units)]
    run("de REFERENCE: the two answers the register counts, as shares of everyone",
        rounds, units, ref_raw, ref_wtd, list(de_ess.REGISTER_ANSWERS),
        ["not drawn from ESS; printed because the register shows these geographies are real, "
         "so they say whether the test has the power to see one"])


def country_it():
    import it
    import it2024

    r1 = sorted(it.ESS_NUTS1_ROUNDS)
    raw1, d1 = _rlgdnm("it", r1, {}, it2024, "_n")
    wtd1, _ = _rlgdnm("it", r1, {}, it2024, "")
    tested1 = _order(wtd1, sorted(set(raw1["cat"]) - set(it.BIG_CATS)))
    run("it NUTS 1: Italian citizens, 5 ripartizioni, the minority categories",
        r1, sorted(it.NUTS1), raw1, wtd1, tested1,
        [f"region-missing citizens dropped: {d1:,.0f}",
         "base = every answered citizen, as in it.py's s1; Catholic and unaffiliated are in "
         "the base and are drawn from NUTS 2, not tested here"])
    run("it NUTS 1 REFERENCE: Catholic and unaffiliated at the same 5 units and rounds",
        r1, sorted(it.NUTS1), raw1, wtd1, [c for c in it.BIG_CATS],
        ["not drawn at this level; printed because Italy's north-south religiosity gradient is "
         "real, so it says whether five units give the test any power at all"])

    r2 = sorted(it.ESS_NUTS2_ROUNDS)
    raw2, d2 = _rlgdnm("it", r2, {}, it2024, "_n")
    wtd2, _ = _rlgdnm("it", r2, {}, it2024, "")
    size = wtd2.groupby("unit")["count"].sum()           # it.py's n2_size: weighted
    own = sorted(u for u in it.NUTS2 if float(size.get(u, 0.0)) >= it.N_FLOOR)
    present = [set(raw2.loc[raw2["round"] == r, "unit"]) for r in r2]
    absent = sorted(u for u in own if any(u not in s for s in present))
    units2 = [u for u in own if u not in absent]
    big = lambda df: df[df["cat"].isin(it.BIG_CATS) & df["unit"].isin(units2)]
    run("it NUTS 2: Italian citizens, the regioni over N_FLOOR, Catholic : unaffiliated",
        r2, units2, big(raw2), big(wtd2), ["Roman Catholic"],
        [f"{len(own)} of {len(it.NUTS2)} regioni carry their own ratio (weighted n >= "
         f"{it.N_FLOOR}); absent from a round and so untestable: {absent or 'none'}",
         "base = Catholic + Not applicable only; the unaffiliated share of that base is 1 "
         "minus the Catholic one, so its rho is identical and it is one test, not two",
         "TWO ROUNDS GIVE ONE SPLIT: the median is that one halving"])


COUNTRIES = {"gr": country_gr, "fi": country_fi, "fr": country_fr, "de": country_de,
             "it": country_it}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("cc", nargs="*")
    a = ap.parse_args()
    if a.fetch:
        fetch()
    for cc in a.cc:
        COUNTRIES[cc]()
