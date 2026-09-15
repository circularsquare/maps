"""Puerto Rico — religion by WVS region, from the World Values Survey wave 7, 2018.

Reads data/raw/pr/F00013157-WVS_Wave_7_Puerto_Rico_Csv_v5.1.zip and writes
data/normalized/pr.csv. `sources/pr.md` is this country's record; `sources.md` §11ap scouted it.

## THE SOURCE

WVS-7 Puerto Rico, fielded March to October 2018 by the Universidad del Sagrado Corazón with the
Instituto de Estadísticas de Puerto Rico, face to face, adults 18 and over. **1,127
respondents**, one round. The survey team's report (*Encuesta Mundial de Valores para Puerto Rico
2018*, 17 June 2019) describes the design on p.16: a sample allocated in proportion to population
across six regions, three municipios drawn at random in each, two block groups per socioeconomic
level in each municipio, Kish selection in the household. The file carries no weights
(`W_WEIGHT` is 1 for everyone), no PSU and no interviewer, so the design is self-weighting and
the municipio is the finest sampling unit available.

The question, `Q289`, on the Puerto Rican card (report p.121):

    0 No pertenece · 1 Católico · 2 Protestante · 3 Ortodoxo · 4 Judío · 5 Musulmanes ·
    6 Hindú · 7 Budista · 8 Otros (escribir)

**Code 8 is a write-in, not "Other Christian".** The archive harmonised all 227 write-ins to
`Q289CS9` 80000000, *Other Christian; nfd*, and the text is not in the file. `taxonomy/pr2018.py`
maps it to the `christianity` root and says why (it was `christianity.other` until the
2026-09-14 review, `sources/pr.md` §10-§11).

## THE FILE ENDS EVERY ROW WITH A SEMICOLON, AND THAT SHIFTS EVERY COLUMN BY ONE

The header has 404 names and every data row 405 fields, the last one empty. pandas' default then
takes the first field as the index and pairs every remaining value with the header one place to
its left: `A_YEAR` reads 630, `N_REGION_ISO` reads the six regions and `Q289` reads the 8-digit
detailed codes, and every column still looks like a plausible variable. `load()` asserts the
trailing delimiter and reads with `index_col=False`, and asserts three columns whose values can
only be themselves (spec §12).

## GEOGRAPHY: SIX REGIONS, EACH OF THEM THREE MUNICIPIOS

`N_REGION_WVS` is the region, `N_REGION_ISO` the municipio (`sources/pr_geo.py` has both code
lists and the report's map of which municipios make up each region). The region is taken from
the municipio, and three witnesses tie them:

  * the report's p.16 prints the interview count for every municipio and region, asserted here;
    the six region totals are all different, so a relabelling of regions (Denmark's round 9)
    cannot pass it, and the eighteen municipio totals tie only once (Moca and Cayey, 52 each);
  * `N_REGION_WVS`, a second column, agrees with the municipio's region in 1,126 of 1,127.

**Sample share against population is printed and does not pin anything.** The report says the
sample was allocated in proportion to population by region; five regions come within 10% of
their share of 2020 adults and **Centro has 1.54 times its share** (187 interviews, which the
report prints too). With four regions of nearly equal size, 37 of the 720 orderings of the six
reach the observed correlation. The shares drawn are within-region, so the over-sampling moves
no region's composition; nationally the build weights by population.

One respondent's `N_REGION_WVS` disagrees with the municipio (a Guayama interview coded Centro),
and the report's own counts side with the municipio. Asserted as exactly that one.

## WHICH CATEGORIES CARRY THEIR OWN GEOGRAPHY

§14.16's split-half, on the sampling unit: each region's three municipios are split one against
two, over every distinct halving of the country (23,328), and the statistic is the median
Spearman over the six regions (spec §12, Sweden: one halving is a draw). **The null deals the
18 municipios into random groups of three** and recomputes the same median: municipios nest in
regions rather than crossing them the way ESS rounds cross län, so this is the permutation that
destroys region structure while keeping every municipio's answers intact (spec §12, Belgium).
The §14.16 formula bar on six units, +0.877, is printed beside it and decides nothing.

Four more requirements, each of which can only make a category fail:

  1. Sweden's spatial chi-square at 0.05 over the six regions;
  2. no single municipio holds half of the category's respondents (spec §12, Uzbekistan);
  3. no single interview day in one municipio holds half of them, which is the nearest the file
     comes to an interviewer;
  4. for a failing category, Honduras's standout test: a region highest in both halves of 95%
     of halvings keeps its measured share.

Usage:
    python sources/pr.py      rebuild data/normalized/pr.csv (the zip is Anita's download)
"""

import io
import itertools
import os
import sys
import warnings
import zipfile

os.environ.setdefault("OMP_NUM_THREADS", "6")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "pr")
ZIP = os.path.join(RAW, "F00013157-WVS_Wave_7_Puerto_Rico_Csv_v5.1.zip")
CSV_NAME = "WVS_Wave_7_Puerto_Rico_Csv_v5.1.csv"
LOOKUP = os.path.join(ROOT, "data", "geo", "pr", "pr_lookup.csv")
MUNIS = os.path.join(ROOT, "data", "geo", "pr", "pr_municipios.csv")
OUT = os.path.join(ROOT, "data", "normalized", "pr.csv")

SOURCE_ID = "pr_wvs7_2018"
N_RESP = 1127
N_HEADER = 404
DOI = "doi.org/10.14281/18241.20"

# Q289 on the Puerto Rican card, verbatim from the report's questionnaire (p.121). -2 is the
# WVS codebook's own label. What reaches `source_category`.
CATEGORY = {
    1: "Católico", 2: "Protestante", 8: "Otros", 0: "No pertenece", 7: "Budista", 6: "Hindú",
    -2: "No answer/refused",
}
NONRESPONSE = -2
RELIGIONS = [1, 2, 8, 0, 7, 6]
# Q289 against Q289CS9, which the codebook annex labels 10100000 Roman Catholic; Latin Church,
# 20000000 Protestant; nfd, 60000000 Hindu, 70000000 Buddhist, 80000000 Other Christian; nfd,
# 100000020 Non-religious. Asserted one to one.
CS9 = {1: 10100000, 2: 20000000, 8: 80000000, 0: 100000020, 7: 70000000, 6: 60000000, -2: -2}
# Report Tabla 80 (p.78): the national frequencies, total 1,117 answering.
TABLA80 = {1: 554, 0: 227, 8: 227, 2: 103, 7: 5, 6: 1}

# Report p.16: interviews per municipio and per region.
REPORT_MUNI = {
    "Barceloneta": 45, "Toa Baja": 65, "Vega Baja": 62, "Cataño": 68, "San Juan": 151,
    "Trujillo Alto": 49, "Cayey": 52, "Naranjito": 55, "Corozal": 80, "Hormigueros": 73,
    "Moca": 52, "San Germán": 32, "Canóvanas": 59, "Juncos": 61, "Río Grande": 74,
    "Guayama": 74, "Peñuelas": 31, "Yauco": 44,
}
REPORT_REGION = {"630001": 172, "630002": 149, "630003": 157, "630004": 194, "630006": 187,
                 "630007": 268}
REGION_SLIPS = {("Guayama", "630006"): 1}
CENTRO = "630006"

STABILITY_BAR = 1.96 / np.sqrt(6 - 1)
STAB_PERM = 400
STAB_SEED = 20260914
ALPHA = 0.05
CELL_CAP = 0.5
STANDOUT_AGREE = 0.95

# Asserted against stability(), so a change in the data stops the build rather than redrawing
# the country. Edit deliberately, after reading the table it prints.
#   Católico     median +0.714 against a null 95th of +0.543, p 0.017, chi-square 1e-8
#   Otros        median +0.771, p 0.005, chi-square 8e-7
#   Protestante  +0.029, p 0.46; No pertenece +0.143, p 0.36 and chi-square 0.26
#   Budista      passes (p 0.007) and is REFUSED: 3 of its 5 are in San Juan
CARRIES = [1, 8]
STANDOUTS = {}
# §9bi's residual, not Honduras's flat shares, and both are printed on every run. The residual
# keeps the two categories with evidence exactly as measured; the flat construction scales them
# to make room and moves Centro's Catholics from 64.0% to 59.0% and Este's Otros from 33.3% to
# 36.4%. What the residual costs here is small: Budista and Hindú come out at up to 1.2x their
# national share in the regions with the largest remainders (Budista 0.53% of Este against 0.45%
# nationally, where the survey found none), against Honduras's 3x (Latter-day Saints at 3.85% of
# the Bay Islands against 1.26% measured). Protestante and No pertenece vary as drawn only
# because the remainder does: Oeste's Protestants are drawn at 9.6% against 5.9% measured,
# because Oeste's remainder is large and No pertenece-heavy. Neither ordering replicated across
# municipios, so neither drawn ordering is a claim; reader text says "at the island-wide ratio".
CONSTRUCTION = "residual"


# =======================================================================================
# load
# =======================================================================================

def load():
    if not os.path.exists(ZIP):
        raise SystemExit(f"{ZIP} missing; it is the WVS download Anita made")
    with zipfile.ZipFile(ZIP) as z:
        raw = z.read(CSV_NAME)
    if raw[:3] != b"\xef\xbb\xbf":
        raise SystemExit("the CSV has lost its BOM; check the encoding before trusting names")
    lines = raw.decode("utf-8-sig").splitlines()
    head, rows = lines[0], [ln for ln in lines[1:] if ln]
    if head.endswith(";") or head.count(";") + 1 != N_HEADER:
        raise SystemExit(f"header has {head.count(';') + 1} names, expected {N_HEADER}")
    if len(rows) != N_RESP or not all(r.endswith(";") and r.count(";") == N_HEADER for r in rows):
        raise SystemExit("the rows no longer all end in one trailing semicolon; re-check the "
                         "column alignment before removing index_col=False")
    df = pd.read_csv(io.BytesIO(raw), sep=";", encoding="utf-8-sig", low_memory=False,
                     index_col=False)
    # three columns that can only hold themselves if the alignment is right
    if not ((df["doi"] == DOI).all() and (df["A_YEAR"] == 2018).all()
            and (df["B_COUNTRY"] == 630).all()):
        raise SystemExit("doi / A_YEAR / B_COUNTRY do not hold their own values; columns shifted")
    if not (df["W_WEIGHT"] == 1).all():
        raise SystemExit("W_WEIGHT is no longer 1 for everyone; the file now carries weights")
    for c in ("N_REGION_WVS", "N_REGION_ISO", "Q289", "Q289CS9", "D_INTERVIEW", "J_INTDATE"):
        df[c] = df[c].astype("int64")
    got = {int(k): int(v) for k, v in df.groupby("Q289")["Q289CS9"].agg(
        lambda s: s.iloc[0] if s.nunique() == 1 else -999).items()}
    if got != CS9:
        raise SystemExit(f"Q289 against Q289CS9 is not the expected one-to-one: {got}")
    freq = df["Q289"].value_counts().to_dict()
    if {k: v for k, v in freq.items() if k != NONRESPONSE} != TABLA80:
        raise SystemExit(f"Q289 frequencies {freq} are not the report's Tabla 80 {TABLA80}")
    print(f"Puerto Rico WVS-7: {len(df):,} respondents, rows end in a trailing ';' (read with "
          f"index_col=False, alignment asserted); Q289 = Q289CS9 one to one; frequencies equal "
          f"the report's Tabla 80; no weights")
    return df


def attach_geography(df):
    m = pd.read_csv(MUNIS, dtype={"geoid": str, "region": str})
    m = m[m["wvs_code"].notna()].copy()
    m["wvs_code"] = m["wvs_code"].astype(int)
    if len(m) != 18:
        raise SystemExit(f"{len(m)} sampled municipios in {MUNIS}, expected 18")
    name_of = dict(zip(m["wvs_code"], m["name"]))
    region_of = dict(zip(m["wvs_code"], m["region"]))
    if set(df["N_REGION_ISO"]) != set(name_of):
        raise SystemExit("N_REGION_ISO codes are not the 18 in pr_municipios.csv")
    df["muni"] = df["N_REGION_ISO"].map(name_of)
    df["region"] = df["N_REGION_ISO"].map(region_of)

    # D_INTERVIEW is country 630, wave 07, then a serial 1-1228. It does NOT carry the
    # municipio: serial ranges and interview dates overlap across municipios, so neither is a
    # witness to N_REGION_ISO. Asserted only as one more column that must hold itself.
    if not (df["D_INTERVIEW"] // 10000 == 63007).all():
        raise SystemExit("D_INTERVIEW is no longer 63007 plus a serial; columns shifted?")
    # witness: the report's counts, per municipio and per region (region from the municipio)
    per_m = df["muni"].value_counts().to_dict()
    if per_m != REPORT_MUNI:
        raise SystemExit(f"interviews per municipio {per_m} are not the report's {REPORT_MUNI}")
    per_r = df["region"].value_counts().to_dict()
    if per_r != REPORT_REGION:
        raise SystemExit(f"interviews per region {per_r} are not the report's {REPORT_REGION}")
    # the file's own region column, against the municipio's
    slips = (df.loc[df["N_REGION_WVS"].astype(str) != df["region"]]
             .groupby(["muni", "N_REGION_WVS"]).size())
    slips = {(mu, str(r)): int(n) for (mu, r), n in slips.items()}
    if slips != REGION_SLIPS:
        raise SystemExit(f"N_REGION_WVS disagrees with the municipio's region at {slips}, "
                         f"expected {REGION_SLIPS}")
    print("  geography: interviews per municipio and per region equal the report's p.16; "
          "N_REGION_WVS agrees with the municipio's region "
          "except one Guayama interview coded Centro, which the report counts in Sur")
    return df


# =======================================================================================
# checks
# =======================================================================================

def held_out(df, lut):
    """Sample share by region against 2020 adults. Nothing here touches religion."""
    from scipy.stats import chisquare

    s = df["region"].value_counts().reindex(lut["geo_id"]).astype(float)
    a = lut.set_index("geo_id")["adults_2020"].astype(float)
    ss, aa = (s / s.sum()).to_numpy(), (a / a.sum()).to_numpy()
    r = np.corrcoef(ss, aa)[0, 1]
    perms = list(itertools.permutations(range(6)))
    reach = sum(np.corrcoef(ss, aa[list(p)])[0, 1] >= r - 1e-12 for p in perms)
    p_gof = chisquare(s.to_numpy(), aa * s.sum())[1]
    print(f"\n  held-out: sample share by region against 2020 adults, r = {r:+.4f}; {reach} of the "
          f"{len(perms)} orderings of the six regions reach it (the true one included); "
          f"goodness of fit to proportional allocation p = {p_gof:.3f}")
    ratio = {}
    for gid, g, x, y in zip(lut["geo_id"], lut["name"], ss, aa):
        ratio[gid] = x / y
        print(f"    {g:<14} sample {x:6.1%}  adults {y:6.1%}  ratio {x / y:.2f}")
    # IT DOES NOT PIN THE JOIN, AND THE REASON IS IN THE REPORT, NOT THE FILE. The team says the
    # sample was allocated in proportion, and five regions sit within 10% of that; Centro has
    # 1.54 times its share, and the report's own p.16 prints the same 187 interviews for it, so
    # the over-sampling is the survey's and not a mislabel. With four of six regions of nearly
    # equal size the permutation cannot tell them apart anyway. The join rests on the report's
    # six distinct region totals and eighteen municipio counts (asserted in attach_geography).
    # What is asserted here is the pattern, so a changed file is noticed.
    others = [v for k, v in ratio.items() if k != CENTRO]
    if not (ratio[CENTRO] > 1.3 and all(0.85 < v < 1.15 for v in others)):
        raise SystemExit(f"sample-to-adult ratios {ratio} are no longer 'Centro over-sampled, "
                         "the rest within 15%'; re-read the allocation before building")
    print(f"    the check does not pin the join ({reach} orderings reach it): Centro is "
          f"over-sampled {ratio[CENTRO]:.2f}x, which the report's own count confirms; the join "
          "rests on the report's region and municipio totals")


def stability(df, lut):
    """§14.16 on municipios, with the grouping null and the vetoes. See the module docstring."""
    from scipy.stats import rankdata

    import stability as shared      # this function is named `stability` too
    rel = df[df["Q289"] != NONRESPONSE]
    regions = list(lut["geo_id"])
    rname = dict(zip(lut["geo_id"], lut["name"]))
    munis = (rel.groupby("muni").agg(region=("region", "first"), code=("N_REGION_ISO", "first"))
             .reset_index().sort_values(["region", "code"]))
    if not (munis.groupby("region").size() == 3).all():
        raise SystemExit("a region does not have exactly three sampled municipios")
    mi = {m: i for i, m in enumerate(munis["muni"])}
    K = len(RELIGIONS)
    M = np.zeros((18, K))
    for (mu, code), n in rel.groupby(["muni", "Q289"]).size().items():
        M[mi[mu], RELIGIONS.index(code)] = n

    # Every distinct halving: per region, which municipio stands alone (3) and on which side (2);
    # the whole-country A/B swap gives each halving twice, so region 0 keeps its single on side A.
    opts = []
    for o in range(6):
        v = np.zeros(3)
        single, side = o % 3, o // 3
        v[single] = 1.0
        if side:
            v = 1.0 - v
        opts.append(v)
    combos = [c for c in itertools.product(range(6), repeat=6) if c[0] < 3]
    P = np.array([np.concatenate([opts[o] for o in c]) for c in combos])      # (S, 18)
    R = np.repeat(np.eye(6), 3, axis=1)                                          # (6, 18)

    def halves(Mo):
        A = np.einsum("sm,rm,mk->srk", P, R, Mo, optimize=True)
        B = np.einsum("sm,rm,mk->srk", 1.0 - P, R, Mo, optimize=True)
        return (A / A.sum(axis=2, keepdims=True), B / B.sum(axis=2, keepdims=True))

    def median_rho(Mo):
        sa, sb = halves(Mo)
        ra, rb = rankdata(sa, axis=1), rankdata(sb, axis=1)
        ra -= ra.mean(axis=1, keepdims=True)
        rb -= rb.mean(axis=1, keepdims=True)
        den = np.sqrt((ra ** 2).sum(axis=1) * (rb ** 2).sum(axis=1))
        with np.errstate(invalid="ignore", divide="ignore"):
            rho = np.where(den > 0, (ra * rb).sum(axis=1) / den, np.nan)       # (S, K)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)     # Hindú: one respondent, all NaN
            med = np.nanmedian(rho, axis=0)
        return med, np.mean(np.isnan(rho), axis=0), sa, sb

    obs, nan_share, sa, sb = median_rho(M)
    rng = np.random.default_rng(STAB_SEED)
    null = shared.cluster_null(lambda perm: median_rho(M[perm])[0], 18, STAB_PERM, rng)

    n_reg = rel.groupby("region").size().reindex(regions)
    print(f"\n  split-half on municipios (§14.16): median Spearman over all {len(combos):,} "
          f"one-against-two halvings, against {STAB_PERM} random groupings of the 18 municipios "
          f"into threes; formula bar +{STABILITY_BAR:.3f} printed only. Vetoes: chi-square over "
          f"regions, largest municipio, largest municipio-day.")
    print(f"    {'category':<14}{'n':>5}{'national':>9}{'median':>8}{'null95':>8}{'p':>7}"
          f"{'chi2 p':>10}{'top muni':>9}{'top day':>8}  verdict")
    nat = rel["Q289"].value_counts(normalize=True)
    carries, eligible = [], {}
    for j, code in enumerate(RELIGIONS):
        hit = rel[rel["Q289"] == code]
        n = len(hit)
        by_reg = hit.groupby("region").size().reindex(regions).fillna(0).to_numpy()
        chi = shared.chi2_p(by_reg, n_reg.to_numpy())
        top_m = hit.groupby("muni").size().max() / n
        top_d = hit.groupby(["muni", "J_INTDATE"]).size().max() / n
        nc = null[:, j][np.isfinite(null[:, j])]
        if not np.isfinite(obs[j]) or len(nc) < 20 or nan_share[j] > 0.5:
            p = np.nan
            verdict = "no test possible (the category is absent from most halves)"
            ok_rank = False
        else:
            p = shared.permutation_p(obs[j], null[:, j], ALPHA)[0]
            ok_rank = p < ALPHA
            verdict = ""
        ok_chi = np.isfinite(chi) and chi < ALPHA
        ok_cell = top_m < CELL_CAP and top_d < CELL_CAP
        eligible[code] = ok_chi and ok_cell
        if not verdict:
            if ok_rank and ok_chi and ok_cell:
                verdict = "own geography"
                carries.append(code)
            elif ok_rank and not ok_chi:
                verdict = "REFUSED: passes the rank test, the regions do not differ"
            elif ok_rank:
                verdict = "REFUSED: passes, but one municipio or one day is most of the answer"
            else:
                verdict = "fails; see the standout test"
        q95 = np.quantile(nc, 0.95) if len(nc) else np.nan
        print(f"    {CATEGORY[code]:<14}{n:>5}{nat[code] * 100:8.2f}%{obs[j]:+8.3f}{q95:+8.3f}"
              f"{p:7.3f}{chi:10.1e}{top_m:9.2f}{top_d:8.2f}  {verdict}")

    # a diagnostic only: the two non-Catholic Christian answers as one column
    both = M[:, RELIGIONS.index(2)] + M[:, RELIGIONS.index(8)]
    Mx = np.column_stack([both, M.sum(axis=1) - both])
    Kx = RELIGIONS
    ox = median_rho(Mx)[0][0]
    nx = shared.cluster_null(lambda perm: median_rho(Mx[perm])[0], 18, STAB_PERM, rng)[:, 0]
    print(f"    (diagnostic, not drawn: Protestante + Otros as one column, median {ox:+.3f}, "
          f"p {(1 + (nx >= ox).sum()) / (1 + len(nx)):.3f})")

    if carries != CARRIES:
        raise SystemExit(f"the split-half now carries {[CATEGORY[c] for c in carries]}, against "
                         f"CARRIES={[CATEGORY[c] for c in CARRIES]}. Read the table above and edit "
                         "CARRIES deliberately. Do not move the bar.")

    standouts = {}
    top_unit, top_share = shared.top_both_halves(sa, sb)
    print(f"\n  failing categories: how often one region is highest in both halves "
          f"(needs {STANDOUT_AGREE:.0%} of {len(combos):,}), and lowest, printed only:")
    for j, code in enumerate(RELIGIONS):
        if code in carries:
            continue
        la, lb = sa[:, :, j].argmin(axis=1), sb[:, :, j].argmin(axis=1)
        if top_unit[j] >= 0:
            reg, share = regions[top_unit[j]], top_share[j]
        else:
            reg, share = None, 0.0
        lagree = np.where(la == lb, la, -1)
        lv, lc = np.unique(lagree[lagree >= 0], return_counts=True)
        low = (rname[regions[lv[int(np.argmax(lc))]]], lc.max() / len(combos)) if len(lv) else ("-", 0)
        ok = share >= STANDOUT_AGREE and eligible[code]
        if ok:
            standouts[code] = reg
        print(f"    {CATEGORY[code]:<14} top {rname.get(reg, '-'):<14} {share:6.1%}   "
              f"bottom {low[0]:<14} {low[1]:6.1%}   {'kept in that region' if ok else 'not kept'}")
    if standouts != STANDOUTS:
        raise SystemExit(f"standouts are now {standouts}, against STANDOUTS={STANDOUTS}; read the "
                         "table above and edit STANDOUTS deliberately")
    return carries, standouts


# =======================================================================================
# build
# =======================================================================================

def compose(rel, lut, carries, standouts):
    regions = list(lut["geo_id"])
    gname = dict(zip(lut["geo_id"], lut["name"]))
    counts = rel.groupby(["region", "Q289"]).size().unstack(fill_value=0).reindex(regions)
    counts = counts.reindex(columns=RELIGIONS, fill_value=0)
    unit_share = counts.div(counts.sum(axis=1), axis=0)
    nat = counts.sum() / counts.to_numpy().sum()

    small = [c for c in RELIGIONS if c not in carries]
    fixed = pd.DataFrame(0.0, index=regions, columns=RELIGIONS)
    for c, g in standouts.items():
        rest = counts.drop(index=g)
        fixed[c] = float(rest[c].sum() / rest.to_numpy().sum())
        fixed.loc[g, c] = unit_share.loc[g, c]

    def residual():
        comp = pd.DataFrame(0.0, index=regions, columns=RELIGIONS)
        for c in carries:
            comp[c] = unit_share[c]
        for c in standouts:
            comp[c] = fixed[c]
        rest = [c for c in small if c not in standouts]
        left = 1.0 - comp.sum(axis=1)
        if (left < -1e-12).any():
            raise SystemExit("carried and standout shares exceed 1 somewhere")
        w = nat[rest] / nat[rest].sum()
        for c in rest:
            comp[c] = left * w[c]
        return comp

    def flat():
        comp = pd.DataFrame(0.0, index=regions, columns=RELIGIONS)
        for c in small:
            comp[c] = fixed[c] if c in standouts else float(nat[c])
        if carries:
            carried = unit_share[carries].sum(axis=1)
            for c in carries:
                comp[c] = unit_share[c] / carried * (1.0 - comp[small].sum(axis=1))
        else:
            comp = comp.div(comp.sum(axis=1), axis=0)
        return comp

    out = {}
    for label, fn in (("residual", residual), ("flat", flat)):
        comp = fn()
        if (comp.sum(axis=1) - 1).abs().max() > 1e-9:
            raise SystemExit(f"{label} composition does not sum to 1")
        out[label] = comp
        print(f"\n  construction '{label}', spec §12's reversal check (drawn against measured, per "
              "region):")
        for c in RELIGIONS:
            d = comp[c] - unit_share[c]
            invented = [gname[g] for g in regions if unit_share.loc[g, c] == 0
                        and comp.loc[g, c] > nat[c]]
            print(f"    {CATEGORY[c]:<14} drawn {comp[c].min():6.2%}..{comp[c].max():6.2%}  "
                  f"measured {unit_share[c].min():6.2%}..{unit_share[c].max():6.2%}  largest gap "
                  f"{gname[d.abs().idxmax()]} {d[d.abs().idxmax()] * 100:+.2f} pt"
                  + (f"  ABOVE NATIONAL WHERE NONE FOUND: {invented}" if invented else ""))
    return out[CONSTRUCTION], unit_share


def main():
    df = attach_geography(load())
    lut = pd.read_csv(LOOKUP, dtype={"geo_id": str, "unit": str}).sort_values("geo_id")
    if len(lut) != 6:
        raise SystemExit("pr_lookup.csv does not have six regions; run sources/pr_geo.py")
    gname = dict(zip(lut["geo_id"], lut["name"]))
    pop = lut.set_index("geo_id")["pop_2024"].astype("int64")

    held_out(df, lut)
    carries, standouts = stability(df, lut)

    rel = df[df["Q289"] != NONRESPONSE]
    comp, unit_share = compose(rel, lut, carries, standouts)
    nr = (df["Q289"] == NONRESPONSE).groupby(df["region"]).mean().reindex(lut["geo_id"])
    print(f"\n  construction used: '{CONSTRUCTION}'. No answer: {int((df['Q289'] == NONRESPONSE).sum())} "
          f"respondents, {nr.mul(pop).sum() / pop.sum():.3%} of people as drawn; not drawn")

    rows = []
    for g in lut["geo_id"]:
        for c in RELIGIONS:
            rows.append((g, CATEGORY[c], comp.loc[g, c] * (1 - nr[g]) * pop[g]))
        rows.append((g, CATEGORY[NONRESPONSE], nr[g] * pop[g]))
    out = pd.DataFrame(rows, columns=["geo_id", "source_category", "count"])
    out["count"] = out["count"].round().astype("int64")
    target = int(pop.sum())
    drift = target - int(out["count"].sum())
    if abs(drift) > len(out):
        raise SystemExit(f"rounding drift {drift} is larger than one person per row")
    if drift:
        out.loc[out["count"].idxmax(), "count"] += drift

    n_resp = df.groupby("region").size()
    code_of = {v: k for k, v in CATEGORY.items()}

    def how_drawn(g, cat):
        c = code_of[cat]
        if c == NONRESPONSE:
            return "not answered, not drawn"
        if c in carries:
            return "region share"
        if c in standouts:
            return ("region share, this region standing apart" if standouts[c] == g
                    else "its share across the other regions")
        return ("national proportions inside the region's residual" if CONSTRUCTION == "residual"
                else "national share")

    out["geo_level"] = "region"
    out["geo_name"] = out["geo_id"].map(gname)
    out["basis"] = "self_id"
    out["year"] = "2018"
    out["source_id"] = SOURCE_ID
    out["note"] = [
        (f"WVS-7 Puerto Rico 2018, n={int(n_resp[g])} respondents in three municipios; "
         + how_drawn(g, cat) + "; applied to the Census Bureau's Vintage 2024 municipio estimates")
        for g, cat in zip(out["geo_id"], out["source_category"])]
    if int(out["count"].sum()) != target:
        raise SystemExit("drawn total is not the population")
    cols = ["geo_id", "geo_level", "geo_name", "source_category", "count", "basis", "year",
            "source_id", "note"]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out[cols].to_csv(OUT, index=False, encoding="utf-8")
    print(f"\nwrote {OUT} ({len(out)} rows, {target:,} people, 6 regions; rounding drift {drift:+d})")

    drawn = out[out["source_category"] != CATEGORY[NONRESPONSE]]
    show = drawn.pivot_table(index="geo_id", columns="source_category", values="count",
                             aggfunc="sum")
    show = show.div(show.sum(axis=1), axis=0) * 100
    nat_d = drawn.groupby("source_category")["count"].sum() / drawn["count"].sum() * 100
    print("\n  as drawn, % of people drawn (measured in brackets):")
    print(f"    {'region':<14}{'n':>5}" + "".join(f"{CATEGORY[c][:10]:>18}" for c in RELIGIONS))
    for g in lut["geo_id"]:
        print(f"    {gname[g]:<14}{int(n_resp[g]):>5}" + "".join(
            f"{show.loc[g, CATEGORY[c]]:9.1f} ({unit_share.loc[g, c] * 100:5.1f})"
            for c in RELIGIONS))
    print(f"    {'Puerto Rico':<14}{len(df):>5}" + "".join(
        f"{nat_d[CATEGORY[c]]:9.2f}        " for c in RELIGIONS))


if __name__ == "__main__":
    main()
