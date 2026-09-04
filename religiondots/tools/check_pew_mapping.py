"""Check taxonomy/us_pew2024.py — the §3.5a cut across Pew's tree.

    python tools/check_pew_mapping.py            the four checks and the national table
    python tools/check_pew_mapping.py --states   add the per-state self-ID against the roll

check_mapping.py cannot do this one. It tests a flat list of source categories against a
normalised <cc>.csv; Pew's source is a TREE, and the thing that can go wrong is not an unmapped
category but a mis-placed cut — take a node and its child and you count those people twice, take
neither and you drop them. So the checks here are about the cut being a cut:

  1. every one of the 51 states partitions exactly: the rows on the cut, plus the excluded
     non-response, sum to the state's published adult total;
  2. no Pew category is stranded — neither on the cut, below it, above it, nor excluded;
  3. every path the cut points at exists in branches.py;
  4. every root ASARB actually counts adherents in is reached by some line of the cut, since a
     root with a roll and no survey line is §3.5a's cap-and-record case arriving by accident.

Then it prints what the cut produces, which is the table §3.5a is made of.
"""

import argparse
import csv
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "taxonomy"))

PEW = os.path.join(ROOT, "data", "normalized", "us_pew.csv")


def load():
    by_state = {}
    with open(PEW, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["adults"] = float(r["adults"])
            r["sample_size"] = int(r["sample_size"])
            r["adult_total"] = float(r["adult_total"])
            r["depth"] = int(r["depth"])
            by_state.setdefault(r["state"], []).append(r)
    return by_state


def asarb_by_root():
    """ASARB 2020 adherents per state and per root, the national total, and state population."""
    import pandas as pd

    paths = pd.read_csv(os.path.join(ROOT, "taxonomy", "usrc_groups.csv"),
                        dtype={"Group Code": str})
    paths["code"] = paths["Group Code"].str.strip().str.zfill(3)
    grp = pd.read_excel(os.path.join(ROOT, "data", "raw", "2020_USRC_Group_Detail.xlsx"),
                        sheet_name="2020 Group by State", dtype={"Group Code": str})
    grp["code"] = grp["Group Code"].str.strip().str.zfill(3)
    grp = grp.merge(paths[["code", "path"]], on="code", how="left")
    grp = grp[grp.path.notna() & (grp.path != "UNMAPPED")].copy()
    grp["root"] = grp.path.str.split(".").str[0]
    per_state = grp.groupby(["State Name", "root"])["Adherents"].sum()

    # the sheet ends with a blank row and a "Totals" row, so the states cannot just be summed
    st = pd.read_excel(os.path.join(ROOT, "data", "raw", "2020_USRC_Summaries.xlsx"),
                       sheet_name="2020 State Summary")
    st = st[st["State Name"].notna()]
    pop = dict(zip(st["State Name"], st["2020 Population"]))
    return per_state, grp.groupby("root")["Adherents"].sum(), pop


def main():
    import us_pew2024 as M
    from branches import BRANCHES

    ap = argparse.ArgumentParser()
    ap.add_argument("--states", action="store_true",
                    help="print the per-state self-ID against the ASARB roll")
    args = ap.parse_args()

    by_state = load()
    print(f"us_pew2024.py: {len(M.CUT)} categories on the cut, {len(M.OPENED)} opened up, "
          f"{len(M.EXCLUDED)} excluded, {len(M.REVIEW)} recorded as arguable")
    print(f"data: {len(by_state)} states, "
          f"{sum(len(v) for v in by_state.values()):,} rows\n")

    problems = []

    # ---- 1 and 2. the cut partitions every state, and strands nothing
    covered = excluded_total = grand = 0.0
    worst = (0.0, None)
    for state, rows in sorted(by_state.items()):
        total = rows[0]["adult_total"]
        grand += total
        try:
            cut = M.apply(rows)
        except ValueError as e:
            problems.append(f"{state}: {e}")
            continue
        got = sum(a for _, a, _ in cut)
        exc = sum(r["adults"] for r in rows if r["name"] in M.EXCLUDED)
        covered += got
        excluded_total += exc
        gap = total - got - exc
        if abs(gap) > abs(worst[0]):
            worst = (gap, state)
        if abs(gap) > 1.0:
            problems.append(f"{state}: cut {got:,.2f} + excluded {exc:,.2f} != "
                            f"adult total {total:,.2f}, off by {gap:,.2f}")

    ok = "OK " if not problems else "BAD"
    print(f"  {ok} the cut partitions all {len(by_state)} states")
    print(f"      on the cut  {covered:>14,.0f}   {covered / grand:6.2%} of adults")
    print(f"      excluded    {excluded_total:>14,.0f}   {excluded_total / grand:6.2%}  "
          f"(no answer)")
    print(f"      worst per-state gap {worst[0]:+,.4f} ({worst[1]})")

    # ---- 3. every target exists
    branch_ids = {b[0] for b in BRANCHES}
    targets = {p for paths in M.CUT.values() for p in paths}
    unknown = sorted(t for t in targets if t not in branch_ids)
    print(f"\n  {'OK ' if not unknown else 'BAD'} {len(targets)} distinct target paths, "
          f"{len(unknown)} not declared in branches.py")
    for t in unknown:
        problems.append(f"target {t} is not in branches.py")
        print(f"      {t}")

    # ---- 4. every root ASARB counts people in has a survey line
    per_state_roll, national_roll, state_pop = asarb_by_root()
    reached = {p.split(".")[0] for p in targets}
    missed = sorted(r for r, v in national_roll.items() if v > 0 and r not in reached)
    print(f"  {'OK ' if not missed else 'BAD'} "
          f"{len([r for r, v in national_roll.items() if v > 0])} roots have an ASARB roll, "
          f"{len(missed)} of them have no line on the cut")
    for r in missed:
        problems.append(f"root {r} has a roll of {national_roll[r]:,.0f} and no survey line")
        print(f"      {r:<24}{national_roll[r]:>14,.0f}")

    # ---- what the cut produces
    print("\n" + "=" * 86)
    print("NATIONAL — what the cut gives each root, against ASARB's roll")
    print("=" * 86)
    print(f"  {'root(s)':<44}{'self-ID adults':>16}{'ASARB roll':>16}{'diff':>16}")
    tot = {}
    for rows in by_state.values():
        for paths, adults, _ in M.apply(rows):
            tot[paths] = tot.get(paths, 0.0) + adults
    for paths, adults in sorted(tot.items(), key=lambda kv: -kv[1]):
        roll = sum(float(national_roll.get(p.split(".")[0], 0.0)) for p in paths)
        label = " + ".join(paths)
        if len(label) > 43:
            label = label[:40] + "..."
        print(f"  {label:<44}{adults:>16,.0f}{roll:>16,.0f}{adults - roll:>16,.0f}")
    print(f"  {'':<44}{sum(tot.values()):>16,.0f}{float(national_roll.sum()):>16,.0f}")
    print(f"\n  Pew's adults are {grand:,.0f}; ASARB's rolls cover the whole population, so the "
          f"third\n  column is not comparable until §3.5a's 1.28x child conversion is applied. "
          f"This table\n  is the mapping's output, not the re-basing's.")

    # ---- the excluded share, which is not uniform
    print("\n  non-response, the excluded row, by state:")
    shares = sorted(((sum(r["adults"] for r in rows if r["name"] in M.EXCLUDED)
                      / rows[0]["adult_total"], state)
                     for state, rows in by_state.items()), reverse=True)
    for share, state in shares[:3]:
        print(f"    {state:<24}{share:6.2%}")
    print(f"    {'...':<24}")
    for share, state in shares[-2:]:
        print(f"    {state:<24}{share:6.2%}")

    # ---- where §3.5a's cap-and-record case actually falls
    #
    # §3.5a found it in Utah, on Christianity, from eight hand-checked states. Run over all
    # 51 and all five roots the survey and the roll can both see, with the adult shares
    # converted to the whole population as §3.5a specifies, the answer is different in kind:
    # Christianity behaves, and the small religions do not, because a survey of 36,908 people
    # cut 51 ways returns a TRUE ZERO for a small religion in a small state while ASARB has
    # congregations there. The overflow is what §3.5a charges to that state's unaffiliated.
    # ASARB titlecases every word, "District Of Columbia" included, in both workbooks.
    names = {s: s.replace("-", " ").title() for s in by_state}
    missing = sorted(n for n in names.values() if n not in state_pop)
    if missing:
        raise SystemExit(f"state names do not join to ASARB's: {missing}")
    ROOTS = ("christianity", "judaism", "islam", "buddhism", "hinduism")

    print("\n" + "=" * 86)
    print("CAP AND RECORD (§3.5a) — where the roll still exceeds the survey, all 51 states")
    print("=" * 86)
    print("  adult shares converted to the whole population, which is §3.5a's child "
          "assumption\n")
    neg, zero_cell = [], 0
    for state, rows in sorted(by_state.items()):
        per_root, total = {}, rows[0]["adult_total"]
        for paths, adults, _ in M.apply(rows):
            if len(paths) == 1:
                per_root[paths[0]] = per_root.get(paths[0], 0.0) + adults
        pop = float(state_pop[names[state]])
        for key in ROOTS:
            sid = per_root.get(key, 0.0) / total * pop
            roll = float(per_state_roll.get((names[state], key), 0.0))
            if sid < roll:
                neg.append((state, key, sid, roll, roll - sid))
                if per_root.get(key, 0.0) == 0.0:
                    zero_cell += 1

    print(f"  {'state':<24}{'root':<16}{'self-ID':>14}{'roll':>14}{'overflow':>14}")
    for state, key, sid, roll, over in sorted(neg, key=lambda t: -t[4])[:15]:
        print(f"  {state:<24}{key:<16}{sid:>14,.0f}{roll:>14,.0f}{over:>14,.0f}")
    if len(neg) > 15:
        print(f"  ... and {len(neg) - 15} more")
    by_root = {}
    for _, key, _, _, over in neg:
        by_root[key] = (by_root.get(key, (0, 0.0))[0] + 1, by_root.get(key, (0, 0.0))[1] + over)
    print(f"\n  {len(neg)} of {len(by_state) * len(ROOTS)} (state, root) pairs are negative, "
          f"{zero_cell} of them because the\n  survey returned a true zero for that state. "
          f"By root:")
    for key, (n, over) in sorted(by_root.items(), key=lambda kv: -kv[1][1]):
        print(f"    {key:<16}{n:>4} states, {over:>12,.0f} people of overflow")
    print(f"    {'TOTAL':<16}{len(neg):>4} pairs,  {sum(o for *_, o in neg):>12,.0f} people, "
          f"charged to those states' unaffiliated")

    if args.states:
        print("\n" + "=" * 86)
        print("BY STATE — every root, self-ID converted to the whole population")
        print("=" * 86)
        print(f"  {'state':<24}{'root':<16}{'self-ID':>14}{'roll':>14}{'residual':>14}")
        for state, rows in sorted(by_state.items()):
            per_root, total = {}, rows[0]["adult_total"]
            for paths, adults, _ in M.apply(rows):
                if len(paths) == 1:
                    per_root[paths[0]] = per_root.get(paths[0], 0.0) + adults
            pop = float(state_pop[names[state]])
            for key in ROOTS:
                sid = per_root.get(key, 0.0) / total * pop
                roll = float(per_state_roll.get((names[state], key), 0.0))
                print(f"  {state:<24}{key:<16}{sid:>14,.0f}{roll:>14,.0f}"
                      f"{sid - roll:>14,.0f}{'  NEGATIVE' if sid < roll else ''}")

    if problems:
        print(f"\n{len(problems)} problems:")
        for p in problems:
            print(f"  {p}")
        raise SystemExit("pew mapping check FAILED")
    print("\nall four checks pass.")


if __name__ == "__main__":
    main()
