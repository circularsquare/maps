"""Switzerland — spec §3.4 implemented: current canton magnitudes, 2000 commune structure.

Reads data/normalized/ch.csv and writes data/normalized/ch_commune_rescaled.csv.

`sources/ch.py` is a normaliser and deliberately stops at the 2000 census; this is the
interpolation, kept separate for `br_rescale.py`'s reason — a normaliser and a model are
different things and folding them together hides which is which.

## Why it is needed

Switzerland last asked everybody about religion in **2000**. Since 2010 the question lives in
the **Strukturerhebung**, a ~200,000-person annual sample, which publishes **eight categories
at canton and nothing below**. So the country offers a choice between the right categories on
the right geography twenty-six years stale, and the right vintage on twenty-six units.

Neither alone is worth drawing, and the two combine: the survey supplies the *magnitude*, the
census supplies the *shape*. That is exactly Brazil (§9-br), where 2022 totals are carried on
2010's 56 denominations because IBGE withheld the 2022 breakdown.

## Where this is a STRONGER claim than Brazil's, and it must not be glossed

Brazil's rescale changes the **categories** and keeps the geography: a 2022 município total is
a measured number for that município. **Switzerland's changes the geography too** — the
measured number is a *canton* total and it is being spread over that canton's communes. So
where Brazil has 103.6M people passing through untouched and `measured`, Switzerland has
**none: every drawn row here is `derived`**, and §3.10 forbids all of it from becoming a
presence ring.

## What is preserved, and it is two margins rather than one

A plain Brazil-style rescale would hold each commune's *share of its canton's* Reformed
population fixed since 2000. That is wrong in a specific and visible way: Switzerland grew
from 7.29M to about 9M between 2000 and 2021, very unevenly — the Zurich and Geneva belts and
the Mittelland suburbs gained, alpine communes lost — so a 2000 share puts dots in emptying
valleys and starves the places people actually moved to. On a *dot map*, whose whole subject
is where people are, that is not a rounding error.

So this is an **iterative proportional fit** on two margins instead of one scale factor:

    rows     each commune's current population, from GISCO LAU's `POP_2021`, scaled within
             its canton so the canton's rows sum to the Strukturerhebung's 15+ total
    columns  each canton's eight Strukturerhebung category totals
    seed     the 2000 census counts, aggregated to the same eight groups

Both margins come out exact. The 2000 census supplies **only the association** between
commune and religion — which commune within a canton is the Catholic one — and contributes no
magnitude at all. Each fitted (commune, group) cell is then split back into the census's 19
leaves by that commune's own 2000 within-group proportions, which is the step that recovers
Pentecostals and Old Catholics from a survey that only knows "other Christian".

## Three things this cannot do

1. **It cannot see a commune that changed religion.** If a village secularised faster than its
   canton, the fit does not know. The canton is the finest unit at which anything has been
   measured since 2000 and that is the resolution of the truth here.
2. **The 19 leaves are 2000's proportions, always.** Within "other Christian" the split
   between Pentecostals and Orthodox is a 2000 fact carried forward 24 years, and Switzerland's
   Orthodox population has grown far faster than its Pentecostal one. The group totals are
   current; the composition inside a group is not.
3. **The universe changes and is NOT scaled back up** (§14.4). The Strukturerhebung asks
   people **aged 15 and over living in private households**, so diplomats, international civil
   servants and people in collective households are outside it too. A Swiss dot is therefore
   an adult, as a Brazilian dot is someone aged 10+ and a Chilean 15+. countries.py declares it.

## The category map, 19 -> 8

Each 2000 leaf belongs to exactly one Strukturerhebung group, asserted below. The one worth
looking at twice is **Christ Catholic** (`Christkatholische Kirche`, the Old Catholic church):
it goes to *other Christian* and not to *Roman Catholic*, because it has not been in communion
with Rome since 1871 and BFS counts it separately from the Roman church in both instruments.

Usage:
    python ch_rescale.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).parent
NORM = HERE / "data" / "normalized"
RAW = HERE / "data" / "raw" / "ch"
SRC = NORM / "ch.csv"
OUT = NORM / "ch_commune_rescaled.csv"
LAU = HERE / "data" / "geo" / "lau2021" / "shp4326" / "LAU_RG_01M_2021_4326.shp"
LEVELS = RAW / "levels.csv"
SE = RAW / "je-01.08.02.02-canton.xlsx"

SOURCE_ID = "ch_se_2024_on_vz2000"
YEAR = 2024                      # the year of the MAGNITUDES, which is what the map shows
STRUCTURE_YEAR = 2000
SE_SHEET = "2024"

TOTAL_CAT = "Religionen - Total"

# Strukturerhebung group -> the 2000 census categories it contains. Written out rather than
# matched on names, and asserted to cover every 2000 category exactly once.
GROUPS = {
    "Evangélique réformé (protestant)": ["Evangelisch-reformierte Kirche"],
    "Catholique romain": ["Römisch-katholische Kirche"],
    "Autres communautés chrétiennes": [
        "Evangelisch-methodistische Kirche",
        "Neupietistisch-evangelikale Gemeinden",
        "Pfingstgemeinden",
        "Neuapostolische Kirchen",
        "Zeugen Jehovas",
        "Übrige protestantische Kirchen und Gemeinschaften",
        "Christkatholische Kirche",
        "Christlich-orthodoxe Kirchen",
        "Andere christliche Gemeinschaften",
    ],
    "Communautés juives": ["Jüdische Glaubensgemeinschaft"],
    "Communautés musulmanes*": ["Islamische Gemeinschaften"],
    "Autres communautés religieuses": [
        "Buddhistische Vereinigungen",
        "Hinduistische Vereinigungen",
        "Übrige Kirchen und Religionsgemeinschaften",
    ],
    "Sans appartenance religieuse": ["Keine Zugehörigkeit"],
    "Appartenance religieuse inconnue": ["Ohne Angabe"],
}

# levels.csv writes the canton multilingually; the Strukturerhebung workbook is the French
# edition and abbreviates two of them. Everything else folds to the same key.
CANTON_ALIAS = {
    "appenzellarh": "appenzellausserrhoden",
    "appenzellirh": "appenzellinnerrhoden",
}


def _key(name):
    import re
    import unicodedata
    s = str(name).split("/")[0].strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^A-Za-z]+", "", s).lower()
    return CANTON_ALIAS.get(s, s)


def _se():
    """{canton key: {group: people}} for SE_SHEET, plus how many cells were suppressed."""
    df = pd.read_excel(SE, sheet_name=SE_SHEET, header=None)
    head = df.iloc[2].tolist()
    # col 1 is Total; each group is a count column followed by a confidence interval.
    cols = {}
    for i, h in enumerate(head):
        if isinstance(h, str) and h.strip() in GROUPS:
            cols[h.strip()] = i
    missing = set(GROUPS) - set(cols)
    if missing:
        raise SystemExit(f"the Strukturerhebung sheet is missing {sorted(missing)} — BFS has "
                         "renamed or regrouped its categories and GROUPS needs re-deriving")

    out, suppressed = {}, 0
    for _, row in df.iloc[4:].iterrows():
        label = row.iloc[0]
        if not isinstance(label, str) or not label.strip():
            continue
        vals = {}
        for g, i in cols.items():
            v = row.iloc[i]
            # `X` is BFS's own marker for an extrapolation from four observations or fewer,
            # withheld for disclosure. Read as zero and counted, not silently coerced.
            if isinstance(v, str) or pd.isna(v):
                vals[g] = 0.0
                suppressed += 1
            else:
                vals[g] = float(v)
        out[_key(label)] = vals
    return out, suppressed


def _communes():
    """[commune, canton, POP_2021] from BFS's levels table and the GISCO boundary file."""
    import geopandas as gpd

    lv = pd.read_csv(LEVELS)
    canton = {int(b): _key(c) for b, c in zip(lv["BfsCode"], lv["Canton"])}

    g = gpd.read_file(LAU, columns=["CNTR_CODE", "LAU_ID", "POP_2021"])
    g = g[g["CNTR_CODE"] == "CH"].copy()
    g["bfs"] = g["LAU_ID"].str.replace("CH", "", regex=False).astype(int)
    g["canton"] = g["bfs"].map(canton)

    # GISCO'S SWISS LAU LAYER IS NOT ALL COMMUNES. 45 of its 2,242 features are absent from
    # BFS's own commune register because they are not communes: the lake surfaces, which BFS
    # numbers in the 9xxx block and which are apportioned to no municipality, and the Ticino
    # and Graubünden *comunanze* — common land held jointly by several communes. 44 are
    # empty; `5399` holds 802 people. Dropping them is what makes the polygon count agree
    # with the census's 2,197, and it is done by the register rather than by a code range so
    # that a renumbering fails loudly here instead of quietly deleting a real commune.
    orphan = g[g["canton"].isna()]
    if len(orphan):
        print(f"  {len(orphan)} LAU features are not communes in BFS's register (lakes and "
              f"comunanze), holding {orphan['POP_2021'].sum():,.0f} people — dropped")
        g = g[g["canton"].notna()].copy()
    return g[["bfs", "canton", "POP_2021"]]


def _ipf(seed, row_margin, col_margin, tol=1e-9, iters=200):
    """Classic two-margin iterative proportional fit. Returns a matrix hitting both."""
    m = seed.astype(float).copy()
    # A row or column with a zero seed can never receive mass; that is correct behaviour
    # (a commune with no Jews in 2000 gets none now) but it means the margins can only be
    # matched on the support, so the checks below test the achieved margins rather than
    # assuming convergence.
    for _ in range(iters):
        rs = m.sum(axis=1)
        m *= np.where(rs > 0, row_margin / np.where(rs > 0, rs, 1), 0)[:, None]
        cs = m.sum(axis=0)
        m *= np.where(cs > 0, col_margin / np.where(cs > 0, cs, 1), 0)[None, :]
        if np.abs(m.sum(axis=1) - row_margin).max() < tol:
            break
    return m


def build():
    src = pd.read_csv(SRC, dtype={"geo_id": str}, low_memory=False)
    src = src[src["geo_level"] == "commune"]
    leaves = [c for c in src["source_category"].unique() if c != TOTAL_CAT]

    covered = [c for cs in GROUPS.values() for c in cs]
    if sorted(covered) != sorted(leaves):
        raise SystemExit("GROUPS does not partition the census categories exactly.\n"
                         f"  in GROUPS not in data: {sorted(set(covered) - set(leaves))}\n"
                         f"  in data not in GROUPS: {sorted(set(leaves) - set(covered))}")
    print(f"  GROUPS partitions all {len(leaves)} census categories exactly once")

    wide = src.pivot_table(index="geo_id", columns="source_category", values="count",
                           aggfunc="sum").fillna(0.0)
    wide.index = wide.index.astype(int)

    com = _communes().set_index("bfs")
    se, suppressed = _se()

    missing = sorted(set(com["canton"]) - set(se))
    if missing:
        raise SystemExit(f"cantons in the boundary file with no Strukturerhebung row: "
                         f"{missing}. CANTON_ALIAS needs an entry.")
    print(f"  all {com['canton'].nunique()} cantons join to the Strukturerhebung "
          f"({suppressed} suppressed `X` cells read as zero)")

    idx = [b for b in wide.index if b in com.index]
    lost = wide.drop(index=idx)[TOTAL_CAT].sum() if len(idx) < len(wide) else 0.0
    if lost:
        print(f"  {len(wide) - len(idx)} communes have census data but no polygon "
              f"({lost:,.0f} people of 2000) — dropped")
    wide = wide.loc[idx]
    com = com.loc[idx]

    groups = list(GROUPS)
    seed = np.column_stack([wide[GROUPS[g]].sum(axis=1).to_numpy() for g in groups])

    out_rows = []
    report = []
    for canton, sub in com.groupby("canton"):
        pos = [idx.index(b) for b in sub.index]
        col = np.array([se[canton][g] for g in groups], dtype=float)
        total = col.sum()
        pop = sub["POP_2021"].to_numpy(dtype=float)
        if pop.sum() <= 0:
            raise SystemExit(f"{canton}: POP_2021 sums to zero")
        # The row margin is the commune's CURRENT population share of its canton, scaled to
        # the survey's 15+ total. That scale factor is the canton's 15+ share and is the one
        # place the two universes are reconciled.
        row = pop / pop.sum() * total
        fit = _ipf(seed[pos], row, col)

        report.append((canton, total, np.abs(fit.sum(axis=0) - col).max(),
                       np.abs(fit.sum(axis=1) - row).max(), len(pos),
                       total / pop.sum()))

        for r, b in enumerate(sub.index):
            w = wide.loc[b]
            for c, g in enumerate(groups):
                mass = fit[r, c]
                if mass <= 0:
                    continue
                inner = w[GROUPS[g]].to_numpy(dtype=float)
                s = inner.sum()
                share = inner / s if s > 0 else np.full(len(inner), 1.0 / len(inner))
                for leaf, f in zip(GROUPS[g], share):
                    if f <= 0:
                        continue
                    out_rows.append({"geo_id": f"{b:04d}", "geo_level": "commune",
                                     "canton": canton, "node_category": leaf,
                                     "count": mass * f, "tier": "derived",
                                     "year": YEAR, "source_id": SOURCE_ID})
    return pd.DataFrame(out_rows), report, se, groups


def check(df, report, se, groups):
    ok = True
    print(f"\n  {len(df):,} rows, {df['geo_id'].nunique():,} communes, "
          f"{df['count'].sum():,.0f} people (15+)")

    worst_c = max(r[2] for r in report)
    worst_r = max(r[3] for r in report)
    good = worst_c < 1e-6 and worst_r < 1e-6
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} the fit hits BOTH margins in every canton — worst "
          f"column error {worst_c:.2e}, worst row error {worst_r:.2e}")

    nat = df.groupby("node_category")["count"].sum().sort_values(ascending=False)
    tot = nat.sum()
    se_nat = {g: sum(se[c][g] for c in {r[0] for r in report}) for g in groups}
    print(f"\n  national, against the Strukturerhebung's own canton sums:")
    for g in groups:
        got = sum(nat.get(leaf, 0.0) for leaf in
                  __import__("ch_rescale").GROUPS[g])
        gap = got - se_nat[g]
        if abs(gap) > 1.0:
            ok = False
        print(f"    {'OK ' if abs(gap) <= 1.0 else 'BAD'} {g[:44]:<46} "
              f"{got:>10,.0f}  {100.0 * got / tot:6.2f}%  gap {gap:+.2f}")

    print(f"\n  the 19 leaves the census recovers inside those groups:")
    for leaf, n in nat.items():
        print(f"    {n:>10,.0f}  {100.0 * n / tot:6.2f}%  {leaf}")

    print(f"\n  what the rescale moved, 2000 -> {YEAR} (both as a share of their own "
          "universe):")
    src = pd.read_csv(SRC, dtype={"geo_id": str}, low_memory=False)
    old = src[src["geo_level"] == "country"].set_index("source_category")["count"]
    old_tot = old[TOTAL_CAT]
    for leaf, n in nat.head(8).items():
        a, b = 100.0 * old[leaf] / old_tot, 100.0 * n / tot
        print(f"    {leaf[:44]:<46} {a:6.2f}% -> {b:6.2f}%   {b - a:+6.2f}")

    # This factor is NOT purely an age share and saying so matters: it is the survey's
    # 2024 15+ private-household population over the boundary file's 2021 TOTAL population,
    # so it carries three years of growth as well as the adult share and the collective-
    # household exclusion. Switzerland's 15+ share is about 0.84; the band is set wider
    # than that on purpose, and its job is to catch a canton whose join went wrong, which
    # would land nowhere near it.
    ratios = sorted((r[5], r[0]) for r in report)
    print(f"\n  the factor each canton's 2021 population was scaled by to reach its {YEAR} "
          f"15+ total\n  (an adult share of ~0.84, plus three years of growth): median "
          f"{ratios[len(ratios) // 2][0]:.3f}, "
          f"min {ratios[0][0]:.3f} ({ratios[0][1]}), "
          f"max {ratios[-1][0]:.3f} ({ratios[-1][1]})")
    bad = [(v, c) for v, c in ratios if not 0.70 <= v <= 1.00]
    if bad:
        ok = False
        print(f"    BAD {len(bad)} canton(s) outside 0.70-1.00 — that is not an age "
              f"structure and the canton join is suspect: {bad[:6]}")
    else:
        print("    OK  every canton inside 0.70-1.00; a mis-joined canton would be nowhere "
              "near it")

    if not ok:
        raise SystemExit("the rescale FAILED its checks")


def main():
    df, report, se, groups = build()
    check(df, report, se, groups)
    df.to_csv(OUT, index=False, encoding="utf-8")
    print(f"\nwrote {OUT} ({len(df):,} rows)")


if __name__ == "__main__":
    main()
