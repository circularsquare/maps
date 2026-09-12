"""Bulgaria — split the 2021 municipal `Християнско` and `Мюсюлманско` columns.

Writes data/normalized/bg_split.csv: the 2021 census at obshtina, with its two undivided
columns replaced by denominations, every replacement row tagged `derived`.

WHY THIS EXISTS
---------------
Anita, 2026-09-08, looking at the first build: *"bulgaria looks kinda out of place as it's
the only one in the area where we dont have christianity breakdown to orthodox and not. so it
displays as generic light yellow. is there any way we could get the orthodox proportion?"*
and then *"yes 2011 oblast composition is ideal."*

Bulgaria is 97% Eastern Orthodox among Christians and the 2021 MUNICIPAL table cannot say so:
NSI publishes one `Християнско` column for all 265 obshtini. Drawn from that alone the country
is a flat Christianity colour wedged between Romania, Serbia, North Macedonia and Greece, all
of which draw on `christianity.orthodox.canonical`. The change of colour at those borders was
an artefact of what two statistical offices chose to publish.

WHAT IS MEASURED, AND WHERE
---------------------------
    2021, per obshtina (265)   `Християнско` and `Мюсюлманско`, undivided        <- the LEVEL
    2021, country only         Orthodox / Catholic / Protestant / Armenian /
                               other Christian                                   <- the TOTALS
    2011, per oblast (28)      the same four Christian bodies, and Sunni /
                               Shia / unspecified Muslim                         <- the SHAPE

No Bulgarian census publishes the breakdown below oblast, in either year; the 2011 report's
own dropdown offers 29 options and stops there (`sources/bg_2011.py`).

THE METHOD, AND WHY IT IS §14.10 RATHER THAN §14.4 RULE 1
--------------------------------------------------------
Rule 1 forbids estimating a magnitude a source does not publish. §14.9 and §14.10 amend it:
the map may run the model itself provided **the magnitude is the host state's**, the
coefficients are documented, and the output is checked against something independent. All
three hold here, and every magnitude in the output is NSI's:

  1. Build a 28 x 5 seed from the **2011 oblast** composition within Christians.
  2. Rake it (IPF) to two sets of **2021** margins: each oblast's row must sum to that
     oblast's own 2021 Christian total, and each denomination's column to NSI's published
     2021 national total. Both margins are measured, in the census being drawn.
  3. Push each oblast's denominations down to its obshtini in proportion to each obshtina's
     measured 2021 Christian count. This is the step that adds no information, and it is
     why every output row is `derived` rather than `measured`: the composition is asserted
     to be uniform inside an oblast, which is the honest reading of "no finer than the
     source publishes".

Muslims get steps 1 and 3 only. **There is no published 2021 Sunni/Shia national total to
rake to**, so their shape is 2011's alone and their level is 2021's. That makes the Muslim
split the weaker of the two and `note_public` says so.

`Друго християнско` has no 2011 counterpart at all, so it is seeded at its national share in
every oblast (0.330%) and raked with the rest. It asserts no geography, which is correct: a
uniform seed is what "we know the total and nothing about where" looks like.

THE CHECKS, INCLUDING ONE THE MODEL COULD HAVE FAILED
-----------------------------------------------------
* **The composition barely moved in ten years**, which is what makes the 2011 shape usable at
  all: Eastern Orthodox is 97.44% of Bulgarian Christians in 2011 and 97.30% in 2021. Had
  this drifted, the whole approach would be unsound and the check is run every build.
* Every obshtina's five Christian rows sum to its measured `Християнско` **to the person**,
  and its three Muslim rows to its measured `Мюсюлманско`. Nothing is created or lost.
* Every denomination's national total equals NSI's published figure to the person.
* **The Shia total is corroborated by a third publication that was not used to build it.**
  UNSD table 28 reports Bulgaria 2021 `Muslim` as 611,129 where NSI's municipal column says
  638,708, a difference of **27,579** which UNSD carries inside its `Other`. That is a 2021
  Muslim sub-category UNSD's classification could not code, and 2011's Shia count was 27,407.
  If this model is right the derived Shia total should land near 27,579; `check()` prints the
  comparison. It is corroboration and not a constraint, and nothing here is fitted to it.

WHAT A READER GETS BACK
-----------------------
Every derived row carries `parent_column=` naming the 2021 column it came out of, so spec
§7a-i-1's roll-up returns `christianity` and `islam` at the obshtina, and the viewer's
`inferred dots: not shown` control redraws exactly the measured table this started from.

Usage:
    python bg_split.py             build data/normalized/bg_split.csv
    python bg_split.py --dry-run   report the numbers, write nothing
"""

import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "6")          # [[feedback_cap_cpu]]
os.environ.setdefault("OPENBLAS_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_2021 = os.path.join(HERE, "data", "normalized", "bg.csv")
SRC_2011 = os.path.join(HERE, "data", "normalized", "bg_2011.csv")
OUT = os.path.join(HERE, "data", "normalized", "bg_split.csv")

CHR_COL = "Християнско"
MUS_COL = "Мюсюлманско"

# 2011 category -> the label this file writes. The Sunni row's source spelling is
# `Мюсюлмаснко-сунитско`, transposed at source; it is keyed verbatim per §2.4.
CHR_2011 = {
    "Източноправославно": "Източноправославно",
    "Католическо": "Католическо",
    "Протестантско": "Протестантско",
    "Арменско апостолическо православно": "Арменско апостолическо",
}
MUS_2011 = {
    "Мюсюлмаснко-сунитско": "Мюсюлманско сунитско",
    "Мюсюлманско-шиитско": "Мюсюлманско шиитско",
    "Мюсюлманско": "Мюсюлманско неуточнено",
}
OTHER_CHR = "Друго християнско"

# NSI's published 2021 NATIONAL breakdown of `Християнско` (press release
# `Census2021-ethnos.pdf` p.8; UNSD table 28 agrees to the person on all four named bodies).
NATIONAL_2021_CHR = {
    "Източноправославно": 4_091_780,
    "Протестантско": 69_852,
    "Католическо": 38_709,
    "Арменско апостолическо": 5_002,
    OTHER_CHR: 13_927,
}
NATIONAL_CHR_TOTAL = 4_219_270
NATIONAL_MUS_TOTAL = 638_708
# Not a constraint. See the docstring: UNSD's 2021 `Muslim` is 27,579 short of NSI's column.
UNSD_MUSLIM_RESIDUAL = 27_579

MAX_DRIFT = 0.5          # percentage points, 2011 vs 2021 Orthodox share of Christians


def _load():
    d21 = pd.read_csv(SRC_2021, dtype={"geo_id": str}, low_memory=False)
    d11 = pd.read_csv(SRC_2011, dtype={"geo_id": str}, low_memory=False)
    ob = d21[d21["geo_level"] == "obshtina"].copy()
    if ob["geo_id"].nunique() != 265:
        raise SystemExit(f"{ob['geo_id'].nunique()} obshtini in bg.csv, expected 265")
    ob["oblast"] = ob["geo_id"].str[:3]
    seed = d11[d11["geo_level"] == "oblast"].copy()
    if seed["geo_id"].nunique() != 28:
        raise SystemExit(f"{seed['geo_id'].nunique()} oblasti in bg_2011.csv, expected 28")
    missing = sorted(set(ob["oblast"]) - set(seed["geo_id"]))
    if missing:
        raise SystemExit(f"obshtina codes whose oblast is absent from the 2011 file: {missing}")
    return ob, seed


def _drift_check(seed):
    """The premise of the whole method, asserted rather than assumed."""
    nat = seed.groupby("source_category")["count"].sum()
    chr11 = sum(nat[c] for c in CHR_2011)
    share11 = 100.0 * nat["Източноправославно"] / chr11
    share21 = 100.0 * NATIONAL_2021_CHR["Източноправославно"] / (
        NATIONAL_CHR_TOTAL - NATIONAL_2021_CHR[OTHER_CHR])
    print(f"  Eastern Orthodox as a share of Christians: 2011 {share11:.2f}%, "
          f"2021 {share21:.2f}%, drift {abs(share11 - share21):.2f} points")
    if abs(share11 - share21) > MAX_DRIFT:
        raise SystemExit(f"the 2011 composition has drifted {abs(share11-share21):.2f} points "
                         f"from 2021, more than the {MAX_DRIFT} this method tolerates")


def _rake(seed, row_margin, col_margin, iters=200, tol=1e-9):
    """IPF a (oblast x denomination) frame onto measured row and column margins."""
    m = seed.copy().astype(float)
    m[m < 0] = 0.0
    # A zero row would be unfixable, and a zero column would drop a denomination entirely.
    m += 1e-12
    for _ in range(iters):
        r = m.sum(axis=1)
        m = m.mul((row_margin / r.replace(0, pd.NA)).fillna(0.0), axis=0)
        c = m.sum(axis=0)
        m = m.mul((col_margin / c.replace(0, pd.NA)).fillna(0.0), axis=1)
        if (m.sum(axis=1) - row_margin).abs().max() < tol:
            break
    return m


def _largest_remainder(shares, total):
    """Integer split that sums to `total` exactly, biggest fractional part first."""
    if total <= 0 or shares.sum() <= 0:
        return pd.Series(0, index=shares.index, dtype="int64")
    exact = shares / shares.sum() * total
    base = exact.astype("int64")
    short = int(total - base.sum())
    if short > 0:
        order = (exact - base).sort_values(ascending=False, kind="mergesort").index[:short]
        base.loc[order] += 1
    return base


def _repair(alloc, target, max_moves=10_000):
    """Move single people between denominations until the national totals are exact.

    Per-obshtina largest-remainder guarantees each row sums to its measured column and gives
    up the national per-denomination totals in exchange, by a handful of people. This buys
    them back WITHOUT touching any row sum: every move takes one person from a denomination
    that is over its published national total and gives them to one that is under, inside a
    single obshtina, so the row total is unchanged by construction.

    The obshtina chosen is the one holding the most of the surplus denomination, so a move
    lands where it is proportionally smallest. The whole correction is single digits against
    4.2M people; it exists so that `Католическо` reads 38,709 and not 38,713.
    """
    for _ in range(max_moves):
        diff = alloc.sum(axis=0) - target
        if (diff == 0).all():
            return
        over = diff.idxmax()
        under = diff.idxmin()
        if diff[over] <= 0 or diff[under] >= 0:
            raise SystemExit(f"cannot repair national totals, drift is one-sided: "
                             f"{diff[diff != 0].to_dict()}")
        donors = alloc[over]
        donors = donors[donors > 0]
        if donors.empty:
            raise SystemExit(f"no obshtina holds any {over} to move")
        who = donors.idxmax()
        alloc.loc[who, over] -= 1
        alloc.loc[who, under] += 1
    raise SystemExit("national-total repair did not converge")


def build():
    ob, seed = _load()
    _drift_check(seed)

    wide21 = ob.pivot_table(index=["geo_id", "oblast", "geo_name"],
                            columns="source_category", values="count",
                            aggfunc="sum").reset_index()
    wide11 = seed.pivot_table(index="geo_id", columns="source_category",
                              values="count", aggfunc="sum")

    rows = []
    for col, mapping, national, extra in (
            (CHR_COL, CHR_2011, NATIONAL_2021_CHR, OTHER_CHR),
            (MUS_COL, MUS_2011, None, None)):
        labels = list(mapping.values()) + ([extra] if extra else [])
        s = wide11[list(mapping)].rename(columns=mapping)
        if extra:
            # No 2011 counterpart: seed it at its national share, asserting no geography.
            s[extra] = s.sum(axis=1) * (national[extra] / (NATIONAL_CHR_TOTAL - national[extra]))
        s = s[labels]

        row_margin = wide21.groupby("oblast")[col].sum().reindex(s.index)
        if national is not None:
            col_margin = pd.Series({k: float(national[k]) for k in labels})
            if abs(col_margin.sum() - row_margin.sum()) > 0.5:
                raise SystemExit(f"{col}: the national breakdown sums to {col_margin.sum():,.0f} "
                                 f"and the municipal column to {row_margin.sum():,.0f}")
            m = _rake(s, row_margin, col_margin)
        else:
            # 2011 shape, 2021 level, no national target to rake to.
            m = s.div(s.sum(axis=1), axis=0).mul(row_margin, axis=0)

        # ---- ALLOCATE PER OBSHTINA, NOT PER LABEL ---------------------------------------
        # The hard constraint is that an obshtina's denomination rows sum to ITS OWN measured
        # column: those are the dots, and a rounding difference there would create or destroy
        # people in a specific place. Splitting each obshtina by largest remainder makes that
        # exact by construction. What it costs is the national per-denomination totals, which
        # then drift by single digits, and `_repair` puts those back.
        alloc = {}
        for _, unit in wide21.iterrows():
            shares = m.loc[unit["oblast"]]
            alloc[unit["geo_id"]] = _largest_remainder(shares, int(unit[col]))
        alloc = pd.DataFrame(alloc).T[labels]
        if national is not None:
            _repair(alloc, pd.Series({k: int(national[k]) for k in labels}))

        note = ("2011 oblast composition raked to the 2021 national breakdown (bg_split.py)"
                if national is not None else
                "2011 oblast composition at the 2021 municipal level (bg_split.py)")
        names = wide21.set_index("geo_id")["geo_name"]
        oblasts = wide21.set_index("geo_id")["oblast"]
        for geo_id, row in alloc.iterrows():
            for label in labels:
                count = int(row[label])
                if count <= 0:
                    continue
                rows.append(dict(
                    geo_id=geo_id, geo_level="obshtina", geo_name=names[geo_id],
                    source_category=label, count=count, basis="self_id",
                    year=2021, source_id="bg_census_2021_split", tier="derived",
                    note=f"{note}; oblast={oblasts[geo_id]}; parent_column={col}"))

    out = pd.DataFrame(rows)
    _check(out, wide21)
    return out


def _check(out, wide21):
    ok = True
    for col, mapping, extra, total in (
            (CHR_COL, CHR_2011, OTHER_CHR, NATIONAL_CHR_TOTAL),
            (MUS_COL, MUS_2011, None, NATIONAL_MUS_TOTAL)):
        labels = list(mapping.values()) + ([extra] if extra else [])
        part = out[out["source_category"].isin(labels)]
        per_unit = part.groupby("geo_id")["count"].sum()
        measured = wide21.set_index("geo_id")[col]
        diff = (per_unit.reindex(measured.index).fillna(0) - measured).abs()
        good = diff.max() == 0
        ok &= good
        print(f"  {'OK ' if good else 'BAD'} every obshtina's {col} rows sum to its measured "
              f"column (worst difference {diff.max():,.0f})")
        got = int(part["count"].sum())
        good = got == total
        ok &= good
        print(f"  {'OK ' if good else 'BAD'} {col} national total {got:,} against the "
              f"published {total:,}")

    nat = out.groupby("source_category")["count"].sum().sort_values(ascending=False)
    print("  derived national totals:")
    for cat, n in nat.items():
        target = NATIONAL_2021_CHR.get(cat)
        flag = "" if target is None else f"   (published {target:,})"
        print(f"      {n:>10,}  {cat}{flag}")

    shia = int(nat.get("Мюсюлманско шиитско", 0))
    print(f"  Shia comes out at {shia:,}; UNSD's uncoded 2021 Muslim residual is "
          f"{UNSD_MUSLIM_RESIDUAL:,}, a {100.0*abs(shia-UNSD_MUSLIM_RESIDUAL)/UNSD_MUSLIM_RESIDUAL:.1f}% "
          f"difference. Corroboration only, not a constraint.")
    if not ok:
        raise SystemExit("bg_split checks FAILED")


def main():
    out = build()
    if "--dry-run" in sys.argv:
        print("  --dry-run: nothing written")
        return
    tmp = OUT + ".part"
    out.to_csv(tmp, index=False, encoding="utf-8")
    os.replace(tmp, OUT)
    print(f"  wrote {OUT} ({len(out):,} rows)")


if __name__ == "__main__":
    main()
