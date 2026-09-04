"""spec.md §3.5a — re-base the United States on self-identification.

WHAT IT DOES, in one sentence: Pew supplies each state's total per religion, ASARB's rolls stay
exactly as they are inside it, and the difference is spread over the people in each county who
are on nobody's roll.

WHY. ASARB's 161.2M adherents are 48.6% of the country, so **171 million Americans are drawn as
nothing at all** — not as unaffiliated, not as unknown, absent. And because the residual of a
roll means "on no roll", the map cannot draw the American non-religious at all while Canada
draws 34.6% of itself that way. The 49th parallel is not a step in the data, it is a step
between two questions. §3.5a is the decision to stop asking the American one.

THE ARITHMETIC, per state and per root:

    self_id   Pew's ADULT share of the state, applied to the state's whole population
    residual  self_id - the ASARB rolls under that root, floored at zero
    overflow  where a roll exceeds the survey, what it exceeds it by
    spread    the residual over the state's counties, in proportion to
              `county population - county adherents`

Four of those five lines are §3.5a's own words. The fifth is the floor, and the overflow is
charged to that state's `unaffiliated`, because the likeliest reading of a name on a roll the
surveys cannot find is a person who now says "nothing in particular".

WHY THE POOL IS `population - adherents` AND NOT POPULATION. It is the same pool the
unaffiliated come out of, it is real per-county data rather than a smoothing, and it puts the
unspecified Christians of New Hampshire where New Hampshire's unchurched actually are. §3.6's
thirty counties that report MORE adherents than residents make it negative in thirty places out
of 3,143; those clip to zero, which is the same clamp §3.6 already applies for display.

WHAT IT DOES NOT DO. It never asks a denomination to clear a survey line of its own. §3.5a
settled that at L1 after nine of nineteen state-level subtractions came out negative, every one
a body that keeps a baptismal register rather than a membership list — a Catholic diocese
reports the baptised living in a parish's territory and an LDS ward everyone baptised who has
not resigned, and both hold people who would tell a surveyor they are something else now. So
ASARB's 372 bodies are structure inside the root and are drawn exactly as they always were.
Nothing in this file changes a single ASARB number.

THE CHILD ASSUMPTION IS LOAD-BEARING. Pew surveys adults, ASARB counts everyone. Applying adult
shares to the whole population scales the survey by about 1.28x, and that assumes children hold
their household's religion. Catholic then clears its roll by 1.1M, under 2%; applied to adults
only it is negative by 12.8M. What this map says about American Catholics rests on it.

CONFIDENCE. The rows this file emits are `derived` (§7). NOTHING ON SCREEN SHOWS THAT — the
desaturation that did was removed 2026-09-04, because every colour on the map has to be a colour
in the legend — so the tier is recorded and carried and the country note is the only place a
reader learns that half the American map is a survey residual. §7 has what a replacement would
have to look like. The tier is still `derived` rather than `modelled` because §7's
`derived` is "a national figure distributed by a proxy" and this is a state figure distributed
by a proxy, while `modelled` is for a country estimate where there is no subnational data at
all. It is the weak end of derived and it is worth saying why — the coarse total here is a
survey of 36,908 people cut 51 ways, with a state margin of error of 3 to 8 points, converted
by the child assumption above, where Ireland's equivalent coarse total is a census count. The
ASARB rows keep `measured`, which is what a register read in the county is.

Run: python us_rebase.py --report     prints the state table and the overflow record, writes
                                      nothing. countries.py calls residual_counts() directly.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "taxonomy"))

PEW = HERE / "data" / "normalized" / "us_pew.csv"
SUMMARIES = HERE / "data" / "raw" / "2020_USRC_Summaries.xlsx"
# The roll itself is not read here: it comes from countries.py's `_us_counts`, so that the
# residual is subtracted from exactly the frame that gets scattered. See `_asarb`.

# Where a residual lands when its Pew line covers more than one root: Sikhs, Daoists, Bahá'ís
# and Zoroastrians are published as one line and cannot be separated at n=36,908, so what is
# left after their rolls come out is "some other world religion, unspecified" (§3.11).
LUMP_NODE = "other.us"

# The overflow goes here. §3.5a: a name on a roll that the surveys cannot find is most likely
# a person who now says "nothing in particular".
OVERFLOW_NODE = "unaffiliated"


def _key(s):
    """A state name reduced to something three files can be joined on.

    ASARB'S TWO WORKBOOKS DISAGREE WITH EACH OTHER. `2020 County Summary` writes `District of
    Columbia`; `2020 State Summary` and `2020 Group by State` write `District Of Columbia`.
    Joining on the printed name silently loses DC from one side or the other — one state, no
    error — so nothing here joins on a spelling. Pew's slugs come in as `district-of-columbia`
    and reduce to the same key.
    """
    return " ".join(str(s).replace("-", " ").split()).casefold()


def _asarb(roll=None):
    """(the roll as drawn, keyed by state; the county table).

    THE ROLL HERE IS ASARB'S COUNTY SHEET, NOT ITS STATE SHEET, and the difference is not
    nothing: the state sheet totals 160,786,973 mapped adherents and the county sheet
    160,572,400, so 214,573 people are reported for a state and attributable to no county in
    it. Subtracting the state figure would compute a residual against a roll the map does not
    draw, and those 214,573 would simply vanish from the country's total. The residual has to
    be measured against what is actually on the map, which is the county sheet — the same
    frame `countries.py` scatters.

    The summaries workbook ends with a blank row and a Totals row whose FIPS is the STRING
    `Totals`, so `notna()` is not enough — that mistake doubles the US population, and it is
    where §3.6's "3,144 counties" came from.
    """
    if roll is None:
        from countries import _us_counts
        roll = _us_counts()

    cty = pd.read_excel(SUMMARIES, sheet_name="2020 County Summary", dtype={"FIPS": str})
    cty = cty[pd.to_numeric(cty["FIPS"], errors="coerce").notna()].copy()
    cty["FIPS"] = cty["FIPS"].str.strip().str.zfill(5)
    cty["key"] = cty["State Name"].map(_key)
    # §3.6: thirty counties report more adherents than residents, because a roll is attributed
    # to the congregation's county and people do not always worship where they live. The pool
    # goes negative there; clipping is the same clamp §3.6 already applies for display.
    cty["pool"] = (cty["2020 Population"] - cty["Adherents"]).clip(lower=0)

    # county FIPS -> state key, so the roll can be grouped the way the survey is
    state_of = dict(zip(cty["FIPS"].str[:2], cty["key"]))
    grp = roll.copy()
    grp["key"] = grp["unit"].str[:2].map(state_of)
    if grp["key"].isna().any():
        bad = sorted(grp.loc[grp["key"].isna(), "unit"].str[:2].unique())
        raise SystemExit(f"county FIPS prefixes with no state in the summaries: {bad}")
    grp = grp.rename(columns={"count": "Adherents"})
    return grp, cty


def _cut_by_target():
    """The §3.5a cut, folded to one entry per node the residual can land on.

    Several Pew lines share a target — atheist, agnostic and humanist all reach `secular` —
    so the self-ID adds up while the ROLL must not: it is the roll of the union of the paths,
    counted once. Getting that wrong would subtract Judaism's roll twice if two lines ever
    reached it.
    """
    import us_pew2024 as M

    by_target = {}
    for name, paths in M.CUT.items():
        target = paths[0] if len(paths) == 1 else LUMP_NODE
        entry = by_target.setdefault(target, {"names": [], "paths": set()})
        entry["names"].append(name)
        entry["paths"].update(paths)
    return by_target


def _roll_under(grp_state, paths):
    """ASARB adherents on any node at or under any of `paths`."""
    m = False
    for p in paths:
        m = m | (grp_state["node"].eq(p) | grp_state["node"].str.startswith(p + "."))
    return float(grp_state.loc[m, "Adherents"].sum()) if m is not False else 0.0


def compute(roll=None):
    """The whole §3.5a calculation. `roll` is countries.py's ASARB frame if it already has one.

    Returns (county_rows, states, overflows):
      county_rows  [(fips, node, count)] — the residual, spread
      states       one dict per state, the arithmetic at state level
      overflows    one dict per (state, node) where the roll beat the survey (§3.5a asks for
                   every one of these to be recorded at build time)
    """
    import us_pew2024 as M

    pew = pd.read_csv(PEW)
    grp, cty = _asarb(roll)
    by_target = _cut_by_target()
    state_pop = cty.groupby("key")["2020 Population"].sum()
    if set(state_pop.index) != set(grp["key"]):
        raise SystemExit("ASARB's county and state workbooks disagree about the state list: "
                         f"{sorted(set(state_pop.index) ^ set(grp['key']))}")

    county_rows, states, overflows = [], [], []

    for slug, sub in pew.groupby("state"):
        key = _key(slug)
        if key not in state_pop.index:
            raise SystemExit(f"{slug}: no ASARB state joins to {key!r}")
        pop = float(state_pop[key])
        adults = float(sub["adult_total"].iloc[0])
        grp_state = grp[grp["key"] == key]

        # Pew's cut, then the child conversion: an adult share applied to everyone.
        self_id = {}
        for paths, a, _ in M.apply(sub.to_dict("records")):
            target = paths[0] if len(paths) == 1 else LUMP_NODE
            self_id[target] = self_id.get(target, 0.0) + a

        overflow_total = 0.0
        per_target = {}
        for target, entry in by_target.items():
            sid = self_id.get(target, 0.0) / adults * pop
            roll = _roll_under(grp_state, entry["paths"])
            per_target[target] = dict(self_id=sid, roll=roll, residual=max(0.0, sid - roll))
            over = max(0.0, roll - sid)
            if over > 0:
                overflow_total += over
                overflows.append(dict(state=slug, node=target, self_id=sid, roll=roll,
                                      overflow=over,
                                      zero_cell=self_id.get(target, 0.0) == 0.0))

        # §3.5a: the overflow comes out of this state's unaffiliated residual.
        u = per_target[OVERFLOW_NODE]
        unabsorbed = max(0.0, overflow_total - u["residual"])
        u["residual"] = max(0.0, u["residual"] - overflow_total)

        # Spread each residual over the state's counties on the one pool §3.5a names. The
        # SAME weights for every node: the pool is "people on no roll", and which religion
        # they turn out to be is exactly what no county-level source can say.
        cs = cty[cty["key"] == key]
        w = cs["pool"].to_numpy(dtype=float)
        if w.sum() <= 0:
            raise SystemExit(f"{slug}: the unchurched pool is empty in every county")
        share = w / w.sum()
        fips = cs["FIPS"].to_numpy()
        for target, t in per_target.items():
            if t["residual"] <= 0:
                continue
            for f, s in zip(fips, share):
                if s > 0:
                    county_rows.append((f, target, t["residual"] * s))

        states.append(dict(
            state=slug, population=pop, adults=adults,
            self_id=sum(t["self_id"] for t in per_target.values()),
            roll=sum(t["roll"] for t in per_target.values()),
            residual=sum(t["residual"] for t in per_target.values()),
            overflow=overflow_total, unabsorbed=unabsorbed,
            drawn=sum(t["roll"] + t["residual"] for t in per_target.values())))

    return county_rows, states, overflows


def residual_counts(roll=None):
    """countries.py hook: the §3.5a residual as scatter.py's count rows.

    `roll` avoids re-reading the ASARB workbook countries.py has already read.
    """
    county_rows, _, _ = compute(roll)
    df = pd.DataFrame(county_rows, columns=["unit", "node", "count"])
    df = df.groupby(["unit", "node"], as_index=False)["count"].sum()
    df["congregations"] = 0
    df["may_ring"] = False          # §3.10: a spread total cannot establish presence
    df["tier"] = "derived"          # §7; see the module docstring for why not `modelled`
    return df[["unit", "node", "count", "congregations", "may_ring", "tier"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true",
                    help="print the state table and the overflow record; write nothing")
    args = ap.parse_args()
    if not args.report:
        ap.error("nothing to do but --report; countries.py calls residual_counts() itself")

    county_rows, states, overflows = compute()
    df = pd.DataFrame(county_rows, columns=["unit", "node", "count"])
    st = pd.DataFrame(states)

    print("=" * 92)
    print("§3.5a — the United States re-based on self-identification")
    print("=" * 92)
    print(f"  {'':<24}{'population':>14}{'ASARB roll':>14}{'residual':>14}{'drawn':>14}"
          f"{'of pop':>9}")
    for r in sorted(states, key=lambda r: -r["population"])[:8]:
        print(f"  {r['state']:<24}{r['population']:>14,.0f}{r['roll']:>14,.0f}"
              f"{r['residual']:>14,.0f}{r['drawn']:>14,.0f}"
              f"{r['drawn'] / r['population']:>9.1%}")
    print(f"  {'... 43 more':<24}")
    tot_pop, tot_roll = st["population"].sum(), st["roll"].sum()
    tot_res, tot_drawn = st["residual"].sum(), st["drawn"].sum()
    print(f"  {'UNITED STATES':<24}{tot_pop:>14,.0f}{tot_roll:>14,.0f}{tot_res:>14,.0f}"
          f"{tot_drawn:>14,.0f}{tot_drawn / tot_pop:>9.1%}")
    print(f"\n  before: {tot_roll / tot_pop:.1%} of Americans drawn, the rest as nothing.")
    print(f"  after:  {tot_drawn / tot_pop:.1%}. The 1.4% still missing is the "
          f"non-response §3.5a's mapping excludes,")
    print(f"          as Czechia's 30% and Ireland's 6.7% are.")

    print("\n" + "=" * 92)
    print("THE RESIDUAL, by node")
    print("=" * 92)
    by_node = df.groupby("node")["count"].sum().sort_values(ascending=False)
    for node, v in by_node.items():
        print(f"  {node:<34}{v:>14,.0f}{v / by_node.sum():>9.1%}")
    print(f"  {'TOTAL':<34}{by_node.sum():>14,.0f}")

    print("\n" + "=" * 92)
    print("OVERFLOW RECORD — where a roll beat the survey (§3.5a asks for all of it)")
    print("=" * 92)
    ov = pd.DataFrame(overflows)
    if len(ov):
        g = ov.groupby("node").agg(states=("state", "size"), people=("overflow", "sum"),
                                   zero_cells=("zero_cell", "sum"))
        for node, r in g.sort_values("people", ascending=False).iterrows():
            print(f"  {node:<24}{int(r['states']):>4} states  {r['people']:>13,.0f} people  "
                  f"({int(r['zero_cells'])} from a survey zero)")
        print(f"  {'TOTAL':<24}{len(ov):>4} pairs   {ov['overflow'].sum():>13,.0f} people")
        print("\n  the ten largest:")
        for _, r in ov.nlargest(10, "overflow").iterrows():
            print(f"    {r['state']:<22}{r['node']:<18}self-ID {r['self_id']:>12,.0f}  "
                  f"roll {r['roll']:>12,.0f}  over {r['overflow']:>11,.0f}")
    unab = st[st["unabsorbed"] > 0]
    print(f"\n  states where the unaffiliated residual could not absorb the overflow: "
          f"{len(unab)}")
    for _, r in unab.iterrows():
        print(f"    {r['state']:<22}{r['unabsorbed']:>12,.0f} people unplaced")

    print("\n" + "=" * 92)
    print(f"{len(df):,} residual rows over {df['unit'].nunique():,} counties and "
          f"{df['node'].nunique()} nodes")


if __name__ == "__main__":
    main()
