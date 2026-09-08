"""El Salvador — religion by departamento, from the LAPOP AmericasBarometer.

Reads data/raw/lapop/Grand_Merge_2004-2023_*.dta and writes data/normalized/sv.csv.
`sources/lapop.py` holds the construction and the split-half; `sources/sv.md` is this
country's record; `sources.md` §11ad assesses the source across the nine countries it could
serve, and `sources/gt.md` is the first one built on it.

**EL SALVADOR'S CENSUS DOES NOT ASK ABOUT RELIGION.** sources.md §11x opened ONEC/BCR's
WordPress library, swept **777 media items**, found one census file and zero religion tables,
and closed the country. The UNSD oracle has no Salvadoran row at all. This does not reopen
that; it is a survey standing where a census would be, and every row is `modelled` in §7.

**THE BEST-SAMPLED OF THE NINE.** 9,063 respondents over **14 departments** is 647 apiece,
against Guatemala's 405 and Ecuador's 321, and it is six waves rather than three or four. The
split-half therefore runs on about 4,500 a side, which is the most power any country in this
set gives it — and the bar is correspondingly higher, because a Spearman over 14 units needs
**+0.54** to clear zero at 95% where Guatemala's 22 units needed +0.43.

## THE JOIN IS ON NAME, AND THE CODE JOIN HERE IS A TRAP

Guatemala joins on the code, because LAPOP's `prov` is 200 plus the official department
number and COD's pcode is `GT` plus the same number. **El Salvador looks identical and is
not**: LAPOP's `prov` is 300 plus the official west-to-east number, while COD's `SV` pcodes
are **alphabetical**. Two of fourteen coincide, twelve are wrong, and a permutation preserves
every total — San Salvador's 1.7 million would have been drawn in La Paz. `sources/sv_geo.py`
does the join on the name and asserts that the code join still mispairs, so nobody restores it.

## What the country looks like

    46.65%  Católico
    29.05%  Evangélica y Pentecostal
    12.44%  Ninguna (cree en un Ser Superior pero no pertenece a ninguna religión)
     7.97%  Protestante / Protestante Tradicional
     1.42%  Religiones Orientales no Cristianas
     0.93%  Otro
     0.74%  Testigos de Jehová
     0.41%  Mormones
     0.35%  Agnóstico o ateo
     0.03%  Religiones Tradicionales
     0.01%  Judío

**The Catholic share is the lowest in Central America** and the `Ninguna (creyente)` cell at
12.44% is two and a half times Guatemala's.

## THREE CATEGORIES CARRY THEIR OWN GEOGRAPHY, WHICH IS ONE MORE THAN GUATEMALA MANAGED

The split-half, on a bar of +0.54 (14 units, against Guatemala's +0.43 on 22):

    Católico                       46.65%   +0.88   own geography
    Evangélica y Pentecostal       29.05%   +0.82   own geography
    Ninguna (creyente)             12.44%   +0.74   own geography
    ------------------------------------------------------------------
    Protestante Tradicional         7.97%   +0.52   national rate — MISSES BY 0.02
    Religiones Orientales           1.42%   +0.45   national rate
    the six under 1%                                national rate (§11ad)

**Every one of these is higher than Guatemala's equivalent**, which is what 647 respondents a
department buys over 405: Guatemala's Catholic cell managed +0.57 where this is +0.88, and its
`Ninguna` cell failed at +0.21 where this passes at +0.74. **So El Salvador is the first
country in this set that can draw its grey ramp where the survey found it**, and that matters
because the cell is 790,000 people here.

**`Protestante Tradicional` misses the bar by 0.02 and is NOT drawn on its own shares.** That
is uncomfortable and it is left alone deliberately: the bar is 1.96/sqrt(n-1), which is what it
takes to be distinguishable from zero at 95%, and moving it because a value landed just under
is fitting the test to the answer. It is recorded here so the next reader knows the call was
close rather than clear, and `note_public` says so.

Usage:
    python sources/sv.py --fetch    rebuild the slim extract from the 1.1 GB LAPOP .dta
    python sources/sv.py            rebuild data/normalized/sv.csv from the slim extract
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import lapop

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
POP = os.path.join(ROOT, "data", "raw", "sv", "slv_admpop_adm1_2024.csv")
LOOKUP = os.path.join(ROOT, "data", "geo", "sv", "sv_lookup.csv")
OUT = os.path.join(ROOT, "data", "normalized", "sv.csv")

PAIS = 3                       # LAPOP's country code for El Salvador
WAVES = [2010, 2012, 2014, 2016, 2018, 2023]
SOURCE_ID = "sv_lapop_2010_2023"
N_UNITS = 14

# What the split-half returns, asserted so a change in the data is a failure here rather
# than a silent re-drawing of the country.
#   1 = Católico (+0.88), 5 = Evangélica y Pentecostal (+0.82), 4 = Ninguna, creyente (+0.74)
CARRIES = [1, 4, 5]


def main():
    if "--fetch" in sys.argv:
        lapop.fetch()

    df = lapop.load(PAIS, WAVES)

    # THE DECODE, AND IT IS NOT GUATEMALA'S. sv_geo.py wrote `lapop_prov` beside each pcode
    # after joining on the NAME and proving the code join mispairs twelve of fourteen. This
    # reads that table rather than doing arithmetic on `prov`, which is the whole point.
    lut = pd.read_csv(LOOKUP, dtype={"geo_id": str, "unit": str})
    prov_to_unit = dict(zip(lut["lapop_prov"].astype(int), lut["unit"]))
    if len(prov_to_unit) != N_UNITS:
        raise SystemExit(f"{len(prov_to_unit)} prov codes in the lookup, expected {N_UNITS}")
    bad = sorted(set(df["prov_code"]) - set(prov_to_unit))
    if bad:
        raise SystemExit(f"prov codes with no department: {bad} — re-run sources/sv_geo.py")
    df["geo_id"] = df["prov_code"].map(prov_to_unit)

    print(f"El Salvador: {len(df):,} respondents with a religion answer and a department, "
          f"waves {WAVES[0]}-{WAVES[-1]}")

    pop = pd.read_csv(POP, encoding="utf-8-sig")
    pop["geo_id"] = pop["ADM1_PCODE"].astype(str).str.strip()
    pop = pop.set_index("geo_id").rename(columns={"T_TL": "pop"})
    missing = sorted(set(lut["geo_id"]) - set(pop.index))
    if missing:
        raise SystemExit(f"no COD-PS population for {missing}")
    if sorted(df["geo_id"].unique()) != sorted(lut["geo_id"]):
        raise SystemExit("LAPOP's departments and the lookup's do not agree: "
                         f"{sorted(set(lut['geo_id']) ^ set(df['geo_id'].unique()))}")

    lapop.held_out(df, pop, "El Salvador")

    nat = lapop.national(df)
    large = lapop.stability(df, nat, N_UNITS)
    small = [c for c in nat.index if c not in large]
    if sorted(large) != sorted(CARRIES):
        raise SystemExit(
            f"the split-half now selects {sorted(large)}, not {sorted(CARRIES)}. That is a "
            "change in what this country claims to know, not a bug — read the numbers above, "
            "then update CARRIES and the docstring deliberately.")
    print(f"    -> {len(large)} categories drawn on their own department shares, "
          f"{len(small)} spread at the national rate")

    units = sorted(lut["geo_id"])
    out = lapop.build(df, nat, large, small, pop["pop"], units, unit_noun="department")

    out["geo_level"] = "departamento"
    out["geo_name"] = out["geo_id"].map(dict(zip(lut["geo_id"], lut["name"])))
    out["basis"] = "self_id"
    out["year"] = "2010-2023"
    out["source_id"] = SOURCE_ID
    out["n_dept"] = out["geo_id"].map(df.groupby("geo_id").size())
    out["note"] = out.apply(
        lambda r: (f"LAPOP AmericasBarometer waves 2010-2023 pooled, n={r.n_dept} in this "
                   f"department; {r.basis_note} applied to the COD-PS 2024 population"),
        axis=1)

    total = int(out["count"].sum())
    if total != int(pop["pop"].sum()):
        raise SystemExit(f"drawn {total:,} against COD-PS {int(pop['pop'].sum()):,}")

    cols = ["geo_id", "geo_level", "geo_name", "source_category", "count",
            "basis", "year", "source_id", "note"]
    out[cols].to_csv(OUT, index=False, encoding="utf-8")
    print(f"\nwrote {OUT} ({len(out)} rows, {total:,} people, "
          f"{out['source_category'].nunique()} categories, {out['geo_id'].nunique()} units)")

    print("\n  national, as drawn:")
    drawn = (out.groupby("source_category")["count"].sum() / total).sort_values(ascending=False)
    for cat, sh in drawn.items():
        print(f"    {sh * 100:6.2f}%  {cat}")

    print(f"\n  the {len(large)} categories drawn on their own department shares, with the "
          "sample behind each:")
    n_by = df.groupby("geo_id").size()
    show = out.pivot_table(index="geo_id", columns="source_category", values="count",
                           aggfunc="sum")
    show = show.div(show.sum(axis=1), axis=0)
    names = dict(zip(lut["geo_id"], lut["name"]))
    cols_show = [lapop.CATEGORY[c] for c in large]
    head = "".join(f"{c.split(',')[0][:14]:>15}" for c in cols_show)
    print(f"    {'dept':<16}{'n':>6}{head}")
    for geo_id in show[cols_show[0]].sort_values().index:
        n = int(n_by[geo_id])
        vals = "".join(f"{show.loc[geo_id, c] * 100:14.1f}%" for c in cols_show)
        print(f"    {names[geo_id]:<16}{n:>6}{vals}")
    thin = n_by.idxmin()
    print(f"\n    the thinnest department is {names[thin]} at n={int(n_by[thin])}, "
          f"the fullest {names[n_by.idxmax()]} at n={int(n_by.max())}")


if __name__ == "__main__":
    main()
