"""Israeli settlements in the West Bank: the CBS 2022 census rows beyond the Green Line.

Writes:
    data/normalized/xs.csv   every il.csv row for the 267 units `il_geo.py` dropped as beyond
                             the Green Line whose group is Jews (with the observance rows) or
                             the register's Others

Usage:
    python sources/xs.py [--fetch]    nothing is downloaded; --fetch is accepted for the
                                      COMMANDS.txt checklist. Needs the Israel build on disk:
                                      data/normalized/il.csv and data/geo/il/dropped_units.json

## WHY THIS IS ITS OWN ENTRY

Anita, 2026-09-15 (ask 028): Israel's entry stops at the Green Line (`sources/il.md` §7) and
Palestine's 2017 census does not count the Israelis living beyond it, so they are drawn as an
entry of their own, part of neither country, with no territory: Auto never picks it by position
(countries.py `territory=False`).

## WHO IS IN IT, AND WHO IS NOT

The same CBS units hold 1,097,156 people: 723,899 Jews and Others, 359,545 Muslims and 13,712
Christians. All but 67 of the Muslims and 61 of the Christians are in East Jerusalem (locality
3000), whose Palestinians PCBS counts and Palestine's entry draws, so they are left out here;
drawing them would draw East Jerusalem's Palestinians twice. The register does not say which Christian
is Arab, so the few non-Arab Christians living in these units are on neither entry. Outside
Jerusalem the settlements hold 67 Muslims and 61 Christians.

The religion mapping is Israel's (`taxonomy/il2022.py`), restricted to the two groups
(`taxonomy/xs2022.py`). `basis` is the population register, as for Israel.
"""

import json
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
IL_CSV = os.path.join(ROOT, "data", "normalized", "il.csv")
IL_DROPPED = os.path.join(ROOT, "data", "geo", "il", "dropped_units.json")
OUT = os.path.join(ROOT, "data", "normalized", "xs.csv")

DRAWN_GROUPS = ("Jews", "Others")
KNOWN_GROUPS = {"Jews", "Muslims", "Christians", "Druze", "Others"}
JERUSALEM = "3000"                  # CBS locality code; statistical areas are `3000_<n>`

# Pinned against il.csv as built on 2026-09-07 (sources/il.md). A rebuild of the Israel entry that
# moves them has to be read before this entry follows it.
UNITS = 267
PINNED = {"Jews": 695_648, "Others": 28_251, "Muslims": 359_545, "Christians": 13_712}
DRAWN_TOTAL = 723_899
SLACK = 2.0                         # il.csv carries shares times populations, so fractions


def main():
    import pandas as pd

    for p in (IL_CSV, IL_DROPPED):
        if not os.path.exists(p):
            raise SystemExit(f"missing {p} -- run sources/il.py and sources/il_geo.py first")
    dropped = set(json.load(open(IL_DROPPED, encoding="utf-8")))
    if len(dropped) != UNITS:
        raise SystemExit(f"dropped_units.json lists {len(dropped)} units, expected {UNITS}")

    df = pd.read_csv(IL_CSV, dtype={"geo_id": str}, keep_default_na=False, na_values=[""],
                     low_memory=False)
    d = df[df["geo_id"].isin(dropped)].copy()
    if d["geo_id"].nunique() != UNITS:
        raise SystemExit(f"il.csv has rows for {d['geo_id'].nunique()} of {UNITS} dropped units")
    if not set(d["geo_level"]) <= {"statarea", "locality"}:
        raise SystemExit(f"unexpected geo_level beyond the line: {sorted(set(d['geo_level']))}")
    d["grp"] = d["source_category"].str.split(" [", regex=False).str[0]
    if not set(d["grp"]) <= KNOWN_GROUPS:
        raise SystemExit(f"unexpected groups beyond the line: {sorted(set(d['grp']) - KNOWN_GROUPS)}")
    lump = d[d["source_category"].str.contains("Other religions", regex=False)]
    if len(lump):
        raise SystemExit(f"{len(lump)} unresolved 'Other religions' rows -- sources/il.py did "
                         "not run its allocation")

    by = d.groupby("grp")["count"].sum()
    print(f"CBS 2022 units beyond the Green Line: {UNITS}, {by.sum():,.0f} people")
    for g, want in PINNED.items():
        got = float(by.get(g, 0.0))
        flag = "ok" if abs(got - want) <= SLACK else "MOVED"
        print(f"  {g:<11} {got:>11,.0f}   pinned {want:>9,}  {flag}")
        if flag != "ok":
            raise SystemExit(f"{g} beyond the line is {got:,.0f}, pinned {want:,}; the Israel "
                             "build has changed -- read it before following it")
    if by.get("Druze", 0.0) > 0:
        raise SystemExit(f"{by['Druze']:,.0f} Druze beyond the line; none were when this was written")

    d["jlm"] = d["geo_id"].str.split("_").str[0] == JERUSALEM
    split = d.pivot_table(index="jlm", columns="grp", values="count", aggfunc="sum", fill_value=0)
    print("\n  East Jerusalem (locality 3000) against the rest:")
    for jlm, r in split.iterrows():
        print(f"    {'Jerusalem' if jlm else 'elsewhere':<10} "
              + "  ".join(f"{g} {r.get(g, 0):,.0f}" for g in ("Jews", "Others", "Muslims", "Christians")))

    out = d[d["grp"].isin(DRAWN_GROUPS)].drop(columns=["grp", "jlm"])
    total = out["count"].sum()
    if abs(total - DRAWN_TOTAL) > SLACK:
        raise SystemExit(f"Jews and Others sum to {total:,.0f}, expected {DRAWN_TOTAL:,}")
    if not set(out["geo_id"]) <= dropped:
        raise SystemExit("a row outside dropped_units.json reached xs.csv; Israel would draw it too")
    units = out.groupby("geo_id")["count"].sum()
    print(f"\n  drawn: Jews and Others, {total:,.0f} people on {len(units)} units "
          f"(mean {units.mean():,.0f}, median {units.median():,.0f})")
    print(f"  left to Palestine's entry: {by.get('Muslims', 0) + by.get('Christians', 0):,.0f} "
          "Muslims and Christians")

    tmp = OUT + ".part"
    out.to_csv(tmp, index=False, encoding="utf-8")
    os.replace(tmp, OUT)
    print(f"\nwrote {OUT} ({len(out):,} rows)")


if __name__ == "__main__":
    main()
