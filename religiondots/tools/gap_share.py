"""Compute `gap_share` from the data instead of typing it — spec §10.4, the hatched segment.

    python tools/gap_share.py                 # every country, as a table
    python tools/gap_share.py at -v           # one country, showing how it got there
    python tools/gap_share.py --check         # exit 1 where an authored share is too small
    python tools/gap_share.py --write         # fill in the countries that have no figure yet

WHY THIS IS ONLY HALF THE JOB. A country's undrawn people come in two kinds and only one of
them is in the archive (spec §10.4):

  1. **A column the census printed and this project chose not to draw** — non-response, "not
     stated", an explicit refusal. Those people are inside the census total, so the share is
     `1 - drawn / the unit's own population total` and BOTH numbers are in
     `data/normalized/<cc>.csv`. That is what this tool computes, exactly, with no outside
     figure and no vintage to get wrong. Austria comes out 2.00%, Eswatini 2.19%.
  2. **People who were never in any table** — Peru's under-twelves, the Galápagos, Abkhazia.
     Nothing here can see them. Those figures stay authored, from the source's own age or
     population table, and this tool must never be read as saying a country has no hole
     because it found none.

So a country where the authored figure is LARGER than the computed one is the normal, healthy
state: it means somebody added a kind-2 hole to a kind-1 one. The failure this tool exists to
catch is the other direction — an authored share SMALLER than a residual the data can prove,
which means the bar is understating a hole on screen.

TWO ROUTES TO THE SAME NUMBER, AND IT PRINTS BOTH. The residual can be reached from either
end, and which ends are available depends on what the adapter kept:

  A. `universe - drawn`, where the universe is an EXCLUDED row big enough to contain the drawn
     population. About a third of the files carry one ("Insgesamt", "Total", "Ogółem"). This
     needs no judgement at all — a row that is at least as large as everything drawn is a
     universe and cannot be anything else.
  B. the sum of the EXCLUDED rows that are SMALLER than the drawn population, which are the
     refusals and the not-stateds. Available almost everywhere, and the one that needs a human
     to look, because a row can be small and still not be a residual: a duplicate, a subtotal,
     a category another script replaces later.

Where both exist they are printed side by side and **they agreeing is the evidence** — B is
confirmed by an independent A, and neither could quietly be the wrong rows. Where only B
exists the tool prints the rows it summed and calls the country UNCONFIRMED: that is a
candidate for a human to accept, not an answer, and `--write` will not touch it.

`--write` only ever fills a country that has NO authored figure and whose two routes agree. It
never overwrites, because an authored figure larger than the computed one is usually a kind-2
hole somebody researched, and this tool cannot see those.

WHAT GOES WRONG, and both failures are loud rather than quiet: the wrong `geo_level`, and a
source whose normalised file is not the last word. Bulgaria is the worked example of the
second — `bg_split.py` replaces two of its categories after normalisation, so the file reads
95% undrawn against a real 20.7%, and it is in SKIP with that written down.

ONE THING THIS TOOL CANNOT DO: add a kind-1 residual to a kind-2 one. Both are shares, but of
different universes — Peru's non-response would be a share of the 12-and-overs, not of Peru —
so a country that has both needs the two combined by hand. Angola, Barbados and the Cayman
Islands are the three, and each says so on its row.
"""

import argparse
import importlib
import io
import os
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "taxonomy"))
sys.path.insert(0, HERE)

# A residual bigger than this is read as a broken denominator, not as a huge gap. No census on
# this map leaves half its own universe undrawn, and everything that has ever come out over
# this threshold has been a geo_level mistake.
MAX_RESIDUAL = 0.5

# An excluded row this close to the whole drawn population is a nested subtotal rather than a
# residual. See the note by `nested` for why it is not tighter.
NESTED_MIN = 0.9

# Anything at or under this is arithmetic dust rather than a hole: a rounding difference
# between a printed total and the columns under it. Malta's 895 people, 0.11%, is real and
# well over it; a 12-person mismatch in a 3-million-person table is not.
MIN_SHARE = 0.0005

# The share and the authored figure are the same claim, so they have to agree to something.
# A tenth of a point is the resolution the tooltip prints at above 10%.
TOLERANCE = 0.001

# Countries whose normalised file is NOT the last word on what gets drawn, so the residual
# computed from it is not the residual on the map. Each needs a reason, and each is a promise
# that the authored figure was arrived at some other way.
REPLACED = {
    # An EXCLUDED row is usually people leaving the map. These are the other kind: a row taken
    # off the tree because something downstream draws it in a different shape, so its people
    # ARE on the map and must be counted as drawn rather than as a hole. There is no size or
    # wording that separates the two, so they are listed, and the tell that one is missing from
    # this list is the `vs dots` column going positive: the map drawing more people than the
    # level counts is only possible when a row like this is being read as a residual.
    "in": {"Other religions and persuasions":
           "the parent bucket, replaced by its 83 Appendix children and by `other.in` for the "
           "remainder (spec §3.10). Counting it as undrawn puts India at 0.89% against a real "
           "0.24%, and the 8.4 million difference is exactly the dot surplus."},
    "th": {"อื่น ๆ ไม่มีศาสนา และไม่ทราบ":
           "the province residual that allocate.py splits into seven small categories. Never "
           "drawn as itself, but every person in it is drawn as one of the seven."},
}

SKIP = {
    "bg": "bg_split.py REPLACES `Християнско` and `Мюсюлманско` after normalisation, so both "
          "resolve to nothing here and the file reads 95% undrawn against a real 20.7%",
}


def _levels_for(cc, df):
    """The geo_level(s) the country is drawn at.

    Borrowed wholesale from check_mapping.py rather than duplicated: that table is the one
    place this project records a country whose drawn tier is not simply its finest, and two
    copies of it would drift the first time a country was added.
    """
    from check_mapping import DEFAULT_LEVELS
    if cc in DEFAULT_LEVELS:
        return DEFAULT_LEVELS[cc]
    return [df.groupby("geo_level")["geo_id"].nunique().idxmax()]


def level_check(cc, drawn, dots):
    """Is the geo_level this ran on the one the map is actually drawn from?

    THE ROUNDING IS ONE-SIDED, which is what makes this a test rather than a comparison. A dot
    is 1,000 people and a group under one dot leaves the map for a ring (§4.3), so the dots can
    come out BELOW the counts and on a small country must: Montserrat's whole population is
    under four dots. What they cannot do is come out above, because nothing invents people. So
    dots over counts means the counts are only part of the country, and Ireland is the case
    that needs it — its 345,165 not-stated over 755,455 counted is a perfectly plausible 31%
    read off one sixth of the country, and the dots are 535% of it.
    """
    if not dots:
        return None
    if dots > drawn * 1.05:
        return (f"the map draws {dots:,.0f} people and this level only counts {drawn:,.0f}, "
                f"{dots / drawn - 1:+.0%}; nothing invents people, so this is the wrong tier")
    if drawn > 2_000_000 and dots < drawn * 0.97:
        return (f"the map draws {dots:,.0f} against {drawn:,.0f} counted, "
                f"{dots / drawn - 1:+.0%}; too far to be sub-dot groups leaving, so this "
                f"level is probably two tiers at once")
    return None


def measure(cc, modname, dots=None, verbose=False):
    """-> dict with `a`/`b`, or with `refused` saying why not."""
    import pandas as pd

    if cc in SKIP:
        return {"refused": SKIP[cc]}
    path = os.path.join(ROOT, "data", "normalized", f"{cc}.csv")
    if not os.path.exists(path):
        return {"refused": f"no data/normalized/{cc}.csv"}
    try:
        mod = importlib.import_module(modname)
    except Exception as e:                                        # noqa: BLE001
        return {"refused": f"{modname} will not import: {e}"}
    resolve = getattr(mod, "resolve", None)
    if resolve is None:
        return {"refused": f"{modname} exposes no resolve()"}
    excluded = getattr(mod, "EXCLUDED", {})
    if not excluded:
        return {"refused": "the mapping excludes nothing"}
    # A mapping may normalise labels before lookup (Poland strips GUS's trailing "w tym:"), so
    # EXCLUDED has to be keyed the same way the module resolves. check_mapping.py's rule.
    key = getattr(mod, "_key", lambda c: " ".join(str(c).split()))

    # keep_default_na=False for check_mapping.py's reason: the Philippines has a category
    # literally called "None", and default parsing turns 43,931 irreligious people into NaN.
    df = pd.read_csv(path, dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    levels = _levels_for(cc, df)
    sub = df[df["geo_level"].isin(levels)].copy()
    if not len(sub):
        return {"refused": f"no rows at level {'+'.join(levels)}"}
    # Two vintages in one file (spec §3.4) means the universe row and the drawn rows may be
    # different years, and the difference between them is population growth rather than a
    # hole. Brazil reads 18% undrawn that way, against a real fraction of one per cent.
    years = sorted({str(y) for y in sub["year"].dropna().unique()})
    if len(years) > 1:
        return {"refused": f"the file carries {len(years)} vintages at this level ("
                           f"{', '.join(years)}), so a universe row and the drawn rows are "
                           f"not the same year (spec §3.4)"}
    sub["node"] = sub["source_category"].map(resolve)
    drawn = float(sub.loc[sub["node"].notna(), "count"].sum())
    # a row that is redrawn elsewhere is drawn, whatever the mapping says about it
    redrawn = REPLACED.get(cc, {})
    if redrawn:
        drawn += float(sub.loc[sub["source_category"].isin(redrawn), "count"].sum())
    if drawn <= 0:
        return {"refused": f"nothing resolves at level {'+'.join(levels)}"}
    wrong = level_check(cc, drawn, dots)
    if wrong:
        return {"refused": f"level {'+'.join(levels)}: {wrong}"}

    rows = []
    for cat in sub["source_category"].unique():
        if key(cat) not in excluded or cat in redrawn:
            continue
        n = float(sub.loc[sub["source_category"] == cat, "count"].sum())
        rows.append({"cat": str(cat), "n": n, "why": str(excluded[key(cat)])})
    rows.sort(key=lambda r: -r["n"])
    # A row at least as large as everything drawn CONTAINS everything drawn, so it is a
    # universe and cannot be a residual. Everything smaller is a candidate residual.
    #
    # THE 0.9999 IS NOT SLOP, it is Hong Kong: its `Total` row is three people under the drawn
    # sum on 7.4 million, because the sub-tables round independently. Tested exactly, that
    # universe becomes a "residual" of 7.4 million and the country reads 50% undrawn.
    #
    # AND A ROW THAT IS NEARLY ALL OF THE DRAWN POPULATION IS NOT A RESIDUAL EITHER. Below the
    # universe line there is a second kind of excluded row — a nested subtotal, a duplicate, a
    # category another script replaces later — and route B cannot tell one from a refusal by
    # name. It can by size, but only at the top: `NESTED_MIN` takes Poland's `należący do
    # wyznania w tym:` (92% of the drawn), Serbia's `Christian - All` (94%) and Myanmar's
    # `Total` (98%) out of B, and Poland and Serbia then land on A exactly.
    #
    # IT CANNOT BE TIGHTENED PAST HUNGARY, which is why the threshold is 0.9 and not something
    # comfortable. Hungary's 3.85 million no-answers are 67% of what it draws — 40% of the
    # country, the largest real gap on this map — and its `Catholic` subtotal is 50%. There is
    # no size that separates those two, so Hungary is a country a person has to look at, and
    # the tool leaves it as `rows only` with both rows printed.
    universes = [r for r in rows if r["n"] >= drawn * 0.9999]
    nested = [r for r in rows if drawn * NESTED_MIN < r["n"] < drawn * 0.9999]
    resid = [r for r in rows if r["n"] <= drawn * NESTED_MIN]
    if verbose:
        print(f"  level {'+'.join(levels)}, {sub['geo_id'].nunique():,} units, "
              f"{drawn:,.0f} drawn")
        kind = {id(r): k for k, rs in (("universe", universes), ("nested", nested),
                                       ("residual", resid)) for r in rs}
        for r in rows:
            print(f"    {kind[id(r)]:>8}  {r['n']:>14,.0f}  {r['cat'][:44]}")
            print(f"                              {r['why'][:96]}")

    out = {"drawn": drawn, "levels": levels, "units": int(sub["geo_id"].nunique()),
           "resid_rows": resid, "nested_rows": nested}
    # A: the widest universe, because for a nested set (Poland prints three) the widest is the
    # population the map is ABOUT and the narrower ones are already inside it.
    if universes:
        u = universes[0]
        if u["n"] - drawn <= MAX_RESIDUAL * u["n"]:
            out["a"] = (u["n"] - drawn) / u["n"]
            out["universe"] = u["n"]
            out["universe_cat"] = u["cat"]
        else:
            out["broken"] = (f"the widest universe {u['cat'][:30]!r} is {u['n']:,.0f} against "
                             f"{drawn:,.0f} drawn, which is not a residual, it is a wrong "
                             f"geo_level or a file that is not the last word")
    # B: the small rows, summed
    b = sum(r["n"] for r in resid)
    if b:
        out["b"] = b / (drawn + b)
        out["b_people"] = b
    if "a" not in out and "b" not in out:
        out["refused"] = out.get("broken") or "the mapping excludes nothing measurable here"
    elif "broken" in out and "a" not in out:
        out["refused"] = out["broken"]
    return out


def verdict(authored, got):
    """-> (verdict, confidence). Confidence is what gates `--write`.

    ROUTE A DECIDES WHEREVER IT EXISTS. A row at least as big as everything drawn is a
    universe and the subtraction cannot mean anything else; B is a check on it and not a rival,
    because B is the route that needs judgement — an excluded row can be small and still not be
    a residual. Where they disagree it is nearly always B having summed a nested subtotal
    (Estonia's "Feels an affiliation to a religion", Poland's "należący do wyznania"), so a
    disagreement downgrades B's confirmation rather than throwing A away.
    """
    if "refused" in got:
        return "refused", "none"
    a, b = got.get("a"), got.get("b")
    if a is not None:
        best = a
        conf = ("confirmed" if b is not None and abs(a - b) <= TOLERANCE
                else "universe only")
    else:
        best, conf = b, "rows only"
    # the agreement test comes FIRST, before the dust threshold: the Solomon Islands' 133
    # people are 0.018%, which is dust by any sane reading and is also exactly the authored
    # figure, and reporting that as a disagreement is how a check gets muted.
    if authored is not None and abs(authored - best) <= TOLERANCE:
        return "agrees", conf
    if best <= MIN_SHARE:
        return ("authored is larger" if authored else "nothing excluded"), conf
    if authored is None:
        return "NEW", conf
    if authored > best:
        return "authored is larger", conf
    return "AUTHORED IS SMALLER", conf


def write_new(rows):
    """Insert `gap_share=` above the `gap=` line for the NEW rows only.

    Matched on the country's own `"<cc>": dict(` header and then on the first `gap=` under it,
    because line positions in this file move under other sessions' edits (CLAUDE.md's second
    warning) and a positional patch would land in the wrong country.
    """
    path = os.path.join(ROOT, "countries.py")
    lines = io.open(path, encoding="utf-8").read().split("\n")
    head = re.compile(r'^    "([a-z]{2})": dict\($')
    gapline = re.compile(r"^(\s*)gap=")
    # never a "rows only" country: B alone is a candidate for a human, not an answer
    want = {cc: got["a"] for cc, _a, got, v, conf in rows
            if v == "NEW" and conf in ("confirmed", "universe only")}
    if not want:
        print("\nnothing to write: every country with a confirmed residual already has one")
        return
    at, cur = {}, None
    for i, ln in enumerate(lines):
        m = head.match(ln)
        if m:
            cur = m.group(1)
        if cur in want and cur not in at and gapline.match(ln):
            at[cur] = (i, gapline.match(ln).group(1))
    missing = sorted(set(want) - set(at))
    if missing:
        raise SystemExit(f"no gap= line found for {missing}; nothing written")
    for cc, (i, indent) in sorted(at.items(), key=lambda kv: -kv[1][0]):
        # four significant figures, which is finer than the tooltip prints and coarse enough
        # that the file does not carry a float nobody can read
        lines.insert(i, f"{indent}gap_share={float(f'{want[cc]:.4g}')!r},")
    io.open(path, "w", encoding="utf-8", newline="").write("\n".join(lines))
    print(f"\nwrote gap_share for {len(at)}: {', '.join(sorted(at))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cc", nargs="?", help="one country (default: all of them)")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="show every candidate universe and why it was taken or not")
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if any authored gap_share is smaller than the computed one")
    ap.add_argument("--write", action="store_true",
                    help="fill gap_share in countries.py where there is none yet")
    args = ap.parse_args()

    import countries as C
    from registry import discover
    reg = discover(drawn_only=False)

    # THE DOT TOTALS ARE THE LEVEL CHECK, and they are the only one available. Everything else
    # on a row is computed off the same geo_level, so a wrong level is self-consistent and
    # looks fine: Ireland's 345,165 not-stated over 755,455 drawn is a perfectly plausible 31%
    # and the country is five million people. The archive knows what the map actually draws, so
    # `drawn` against `dots × dot_value` catches it. They will not match exactly — the dots are
    # the universe rounded to the nearest 1,000 per unit, and on a small country they cannot:
    # Montserrat's 3,838 people are three dots.
    import json
    arch = json.load(open(os.path.join(ROOT, "data", "processed", "counts.json"),
                          encoding="utf-8"))
    dv = arch["dot_value"]
    dots = {cc: sum(e.get("dots", {}).values()) * dv
            for cc, e in arch.get("countries", {}).items()}

    ccs = [args.cc] if args.cc else sorted(C.COUNTRIES)
    rows = []
    for cc in ccs:
        if cc not in C.COUNTRIES:
            raise SystemExit(f"{cc!r} is not in countries.py")
        if cc not in reg:
            rows.append((cc, C.COUNTRIES[cc].get("gap_share"),
                         {"refused": "no taxonomy module (see registry.SPECIAL)"},
                         "refused", "none"))
            continue
        if args.verbose:
            print(f"{cc}  {C.COUNTRIES[cc].get('name', cc)}")
        got = measure(cc, reg[cc], dots=dots.get(cc), verbose=args.verbose)
        authored = C.COUNTRIES[cc].get("gap_share")
        rows.append((cc, authored, got) + verdict(authored, got))

    pc = lambda x: "-" if x is None else f"{x * 100:.2f}%"                    # noqa: E731
    interesting = [r for r in rows
                   if r[3] not in ("nothing excluded", "refused") or args.cc]
    print(f"\n{'cc':<4}{'authored':>10}{'A universe':>12}{'B rows':>10}"
          f"{'drawn':>14}{'vs dots':>9}  verdict")
    for cc, authored, got, v, conf in interesting:
        if "refused" in got:
            print(f"{cc:<4}{pc(authored):>10}{'-':>12}{'-':>10}{'-':>14}{'-':>9}  "
                  f"refused: {got['refused']}")
            continue
        d = dots.get(cc)
        off = ("-" if not d else f"{(d - got['drawn']) / got['drawn'] * 100:+.1f}%")
        print(f"{cc:<4}{pc(authored):>10}{pc(got.get('a')):>12}{pc(got.get('b')):>10}"
              f"{got['drawn']:>14,.0f}{off:>9}  {v} ({conf})")
        # the rows B summed, whenever a human still has to look at them
        if args.verbose or conf != "confirmed":
            for r in got["resid_rows"]:
                print(f"{'':<16}{r['n']:>12,.0f}  {r['cat'][:52]}")

    silent = [r for r in rows if r[3] == "nothing excluded"]
    refused = [r for r in rows if r[3] == "refused"]
    if not args.cc:
        print(f"\n{len(silent)} countries exclude nothing measurable and have no authored "
              f"figure; {len(refused)} refused")
        if refused:
            for cc, _a, got, _v, _c in refused:
                print(f"  {cc}: {got['refused'][:110]}")

    if args.write:
        write_new(rows)

    if args.check:
        # only where a universe confirms it. B alone disagreeing with an authored figure is as
        # likely to be B having summed a nested row, and a check that cries wolf gets muted.
        bad = [r for r in rows
               if r[3] == "AUTHORED IS SMALLER" and r[4] in ("confirmed", "universe only")]
        for cc, authored, got, _v, _c in bad:
            print(f"\nFAIL {cc}: gap_share={authored} but {got['universe'] - got['drawn']:,.0f} "
                  f"people, {got['a'] * 100:.2f}%, are excluded in the data against "
                  f"{got['universe_cat'][:30]!r}. The bar is understating the hole.")
        if bad:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
