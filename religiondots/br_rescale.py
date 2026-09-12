"""Brazil — spec §3.4 implemented: 2022 municipal totals, 2010 municipal structure.

Reads data/normalized/br.csv and writes data/normalized/br_municipio_rescaled.csv.

    est[m, leaf] = total_2022[m, group(leaf)]
                   x count_2010[m, leaf] / sum(count_2010[m, l] for l in group(leaf))

§3.4 has described this since the project began and `br.py` has deliberately not done it —
it is a normaliser, and this is an interpolation. `sources/br.md` §1 sets out why it is
needed: the 2022 census publishes **nine** categories at município and lumps 47.4M
evangelicals into one of them, while the 2010 census publishes **56** and is fifteen years
stale. IBGE said on release (6 June 2025) that the 2022 evangelical denominational
breakdown is withheld over data quality and may never appear, so this is not a stopgap
pending a better table.

WHAT THIS CHANGES ABOUT BRAZIL, and both halves matter.

- **The totals become 2022.** Brazil stops being a 2015-stale country. The moves are large:
  Catholics 123.3M -> 100.2M, evangelicals 42.3M -> 47.4M, no religion 15.3M -> 16.4M,
  Umbanda e Candomblé 588,810 -> 1,849,835.
- **The universe becomes people aged 10 or over.** The 2022 religion question was asked
  only of them, so the drawn population falls from 190.8M (2010, everybody) to 176.3M. That
  is a change in what a Brazilian dot MEANS and countries.py declares it. It is the same
  shape as Chile's 15+ and it is NOT scaled back up, for the same reason (§14.4).

THE CATEGORY MAP IS BY HAND AND ONE ENTRY IS A TRAP. `Outras religiosidades` exists in both
years and means opposite things: 11,307 people in 2010, a genuine leftover; **7,079,124 in
2022, holding Judaism, Islam, Buddhism, the Witnesses, the Latter-day Saints, Hinduism,
Orthodoxy and the esoteric traditions** — every one of which is its own row in 2010. Joining
the two years on the category name produces confident nonsense at 626x scale, which is why
`GROUPS` below is written out and asserted to cover every 2010 root exactly once.

TIER, and it is not uniform. Where a 2022 category maps to a SINGLE 2010 leaf the 2022 count
passes through untouched and is `measured` — Católica Apostólica Romana, Espírita and
Tradições indígenas, 103.6M people between them. Everything else is `derived`: a 2022
magnitude wearing a 2010 shape, which §3.10 forbids from ever becoming a presence ring.

Usage:
    python br_rescale.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).parent
NORM = HERE / "data" / "normalized"
SRC = NORM / "br.csv"
OUT = NORM / "br_municipio_rescaled.csv"

SOURCE_ID = "br_censo_2022_on_2010"
YEAR = 2022                      # the year of the TOTALS, which is what the map shows
STRUCTURE_YEAR = 2010

# 2022 category -> the 2010 ROOT categories it contains.
#
# Every 2010 root appears exactly once across this dict plus EXCLUDED_2010, and that is
# asserted at run time rather than trusted. `Outras religiosidades` is the trap (see the
# module docstring): its 2022 meaning is fifteen 2010 roots, not the identically-named one.
GROUPS = {
    "Católica Apostólica Romana": ["Católica Apostólica Romana"],
    "Evangélicas": ["Evangélicas"],
    "Sem religião": ["Sem religião"],
    "Espírita": ["Espírita"],
    "Umbanda e Candomblé": ["Umbanda e Candomblé"],
    "Tradições indígenas": ["Tradições indígenas"],
    "Outras religiosidades": [
        "Outras religiosidades cristãs",
        "Testemunhas de Jeová",
        # An answer rather than a non-response — 2010 keeps `Não sabe` and `Sem declaração`
        # separate and much smaller — so it belongs inside a religion category rather than
        # off the tree. Which 2022 cell absorbed it is not stated by IBGE; this is the only
        # one it can be. Arguable, and the largest single judgement in this file at 643,624.
        "Não determinada e multiplo pertencimento",
        "Católica Apostólica Brasileira",
        "Budismo",
        "Igreja de Jesus Cristo dos Santos dos Últimos Dias",
        "Novas religiões orientais",
        "Católica Ortodoxa",
        "Judaísmo",
        "Tradições esotéricas",
        "Espiritualista",
        "Islamismo",
        "Outras religiosidades",
        "Outras religiões orientais",
        "Hinduísmo",
    ],
}

# Off the tree in both years: universe totals and non-response (§3.5).
EXCLUDED_2022 = {"Total", "Sem declaração", "Não sabe"}
EXCLUDED_2010 = {"Sem declaração", "Não sabe"}


def load():
    df = pd.read_csv(SRC, dtype={"geo_id": str}, low_memory=False)
    d10 = df[(df["year"] == 2010) & (df["geo_level"] == "municipio")].copy()
    d22 = df[(df["year"] == 2022) & (df["geo_level"] == "municipio")].copy()

    d10["code"] = d10["note"].str.extract(r"code=(\d+)")[0]
    d10["parent"] = d10["note"].str.extract(r"parent=(\d*)")[0]
    d10 = d10[d10["code"].notna()]          # drops the per-município `Total` row

    cat = d10.drop_duplicates("source_category")[["source_category", "code", "parent"]]
    by_code = cat.set_index("code")
    internal = set(cat["parent"].dropna()) - {""}

    def root(c):
        for _ in range(12):
            p = by_code.loc[c, "parent"]
            if not isinstance(p, str) or p == "" or p not in by_code.index:
                return c
            c = p
        raise SystemExit(f"cycle in the 2010 category tree at {c}")

    name = dict(zip(cat["code"], cat["source_category"]))
    cat["rootname"] = [name[root(c)] for c in cat["code"]]
    # A code that is nobody's parent is a leaf. Derived, not hand-listed, so a category
    # IBGE adds later cannot silently be double counted (the rule _br_counts already used).
    cat["leaf"] = ~cat["code"].isin(internal)
    return d10, d22, cat


def check_map(cat):
    roots = set(cat["rootname"])
    mapped = [r for v in GROUPS.values() for r in v]
    dupes = {r for r in mapped if mapped.count(r) > 1}
    missing = roots - set(mapped) - EXCLUDED_2010
    phantom = set(mapped) - roots
    if dupes or missing or phantom:
        raise SystemExit(
            f"GROUPS does not partition the 2010 roots.\n"
            f"  in two groups : {sorted(dupes)}\n"
            f"  unmapped      : {sorted(missing)}\n"
            f"  do not exist  : {sorted(phantom)}")
    print(f"  OK  GROUPS partitions all {len(roots)} of 2010's root categories "
          f"({len(EXCLUDED_2010)} deliberately excluded)")


def main():
    if not SRC.exists():
        raise SystemExit(f"missing {SRC} -- run sources/br.py first")
    d10, d22, cat = load()
    print(f"2010: {d10['geo_id'].nunique():,} municípios, {len(cat)} categories, "
          f"{int(cat['leaf'].sum())} leaves")
    print(f"2022: {d22['geo_id'].nunique():,} municípios, "
          f"{d22['source_category'].nunique()} categories")
    check_map(cat)

    leaf_root = dict(zip(cat[cat["leaf"]]["source_category"],
                         cat[cat["leaf"]]["rootname"]))
    leaves = [l for l in leaf_root if leaf_root[l] not in EXCLUDED_2010]

    # 2010 leaf counts and 2022 totals, both as município x category matrices.
    s10 = (d10[d10["source_category"].isin(leaves)]
           .pivot_table(index="geo_id", columns="source_category", values="count",
                        aggfunc="sum", fill_value=0))
    t22 = (d22[~d22["source_category"].isin(EXCLUDED_2022)]
           .pivot_table(index="geo_id", columns="source_category", values="count",
                        aggfunc="sum", fill_value=0))
    names22 = d22.drop_duplicates("geo_id").set_index("geo_id")["geo_name"].to_dict()

    units = list(t22.index)
    state = pd.Series([u[:2] for u in units], index=units)
    print(f"\n  {len(units):,} municípios to write; {len(set(units) - set(s10.index))} of "
          "them have no 2010 row at all and fall straight to their state's shares")

    frames, stats = [], []
    for c22, roots in GROUPS.items():
        ls = [l for l in leaves if leaf_root[l] in roots]
        if not ls:
            raise SystemExit(f"{c22}: no 2010 leaves")
        total = t22[c22].reindex(units).fillna(0.0)

        if len(ls) == 1:
            # Nothing to split. The 2022 count passes through and stays `measured`.
            est = pd.DataFrame({ls[0]: total.to_numpy()}, index=units)
            tier, src = "measured", "passthrough"
            stats.append((c22, len(ls), int(total.sum()), 0, 0))
        else:
            sub = s10[ls].reindex(units).fillna(0.0)
            den = sub.sum(axis=1)

            # Fallback chain, most local first: the município's own 2010 shape, then its
            # state's, then the nation's. A município with no 2010 people in this group at
            # all cannot say anything about its own composition, and the state is the
            # nearest thing that can.
            st_num = sub.groupby(state).sum()
            st_den = st_num.sum(axis=1)
            nat_num = sub.sum(axis=0)
            nat_share = nat_num / nat_num.sum()

            share = sub.div(den.replace(0, np.nan), axis=0)
            used_state = den == 0
            st_share = st_num.div(st_den.replace(0, np.nan), axis=0)
            fill = st_share.reindex(state.to_numpy())
            fill.index = units
            share = share.where(~used_state, fill)
            used_nat = share.isna().all(axis=1)
            if used_nat.any():
                share.loc[used_nat] = nat_share.to_numpy()
            share = share.fillna(0.0)

            est = share.mul(total, axis=0)
            tier, src = "derived", "rescaled"
            stats.append((c22, len(ls), int(total.sum()),
                          int(used_state.sum()), int(used_nat.sum())))

        long = est.stack().reset_index()
        long.columns = ["geo_id", "source_category", "count"]
        long = long[long["count"] > 0]
        long["tier"] = tier
        long["structure_share"] = (
            long["count"] / long["geo_id"].map(total.to_dict()).replace(0, np.nan))
        long["derivation"] = src
        long["group22"] = c22
        frames.append(long)

    out = pd.concat(frames, ignore_index=True)
    out["geo_level"] = "municipio"
    out["geo_name"] = out["geo_id"].map(names22)
    out["basis"] = "self_id"
    out["year"] = YEAR
    out["source_id"] = SOURCE_ID
    out["note"] = (
        "structure_year=" + str(STRUCTURE_YEAR) + "; total_year=" + str(YEAR)
        + "; derivation=" + out["derivation"]
        + "; group22=" + out["group22"]
        + "; structure_share=" + out["structure_share"].round(6).astype(str))
    out = out[["geo_id", "geo_level", "geo_name", "source_category", "count",
               "basis", "year", "source_id", "tier", "note"]]

    # ---- checks ----
    print("\n  per 2022 category: leaves, people, and how often the local shape was missing")
    for c22, n, tot, st, nt in stats:
        print(f"    {c22[:32]:32s} {n:>2} leaves  {tot:>12,}  "
              f"state-fallback {st:>4}  national-fallback {nt:>3}")

    ok = True
    for c22 in GROUPS:
        got = out[out["note"].str.contains(f"group22={c22};|group22={c22}$", regex=True)]
        s, want = got["count"].sum(), t22[c22].sum()
        good = abs(s - want) < 1.0
        ok &= good
        print(f"  {'OK ' if good else 'BAD'} {c22[:34]:34s} rescaled {s:>15,.1f}  "
              f"2022 total {want:>13,}")

    per_unit = out.groupby("geo_id")["count"].sum()
    want_unit = t22.sum(axis=1)
    worst = (per_unit - want_unit.reindex(per_unit.index)).abs().max()
    good = worst < 1.0
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} every município's rescaled rows sum to its own "
          f"2022 drawn total (worst gap {worst:.4f})")

    drawn = out["count"].sum()
    print(f"\n  {len(out):,} rows, {out['geo_id'].nunique():,} municípios, "
          f"{out['source_category'].nunique()} categories")
    print(f"  drawn {drawn:,.0f} people — 2022's 10+ universe less non-response")
    meas = out[out["tier"] == "measured"]["count"].sum()
    print(f"  measured {meas:,.0f} ({100.0 * meas / drawn:.1f}%) — the three single-leaf "
          f"groups\n  derived  {drawn - meas:,.0f} ({100.0 * (drawn - meas) / drawn:.1f}%) "
          "— a 2022 magnitude on a 2010 shape, and may never ring (§3.10)")

    if not ok:
        raise SystemExit("rescale FAILED")
    out.to_csv(OUT, index=False, encoding="utf-8")
    print(f"\nwrote {OUT} ({len(out):,} rows)")


if __name__ == "__main__":
    main()
