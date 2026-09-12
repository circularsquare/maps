"""Collect every mapping module's `REVIEW` dict in one place, for reading in bulk.

    python tools/review_dump.py                  # one line per entry, all countries
    python tools/review_dump.py sb ao            # just these
    python tools/review_dump.py --full sb        # the whole reason, not the first line
    python tools/review_dump.py --since 3        # only mappings touched in the last 3 days

`REVIEW` is the cheap, uncapped tier: *"calls that are defensible but arguable, with the
reason"*, and 95 mapping modules carry one. Nothing has ever read them all together, so a
pattern across countries — the same family of body filed three different ways — has been
invisible. This is that read.

It is deliberately read-only and has no notion of resolving an entry. An entry stops being a
review when somebody edits the mapping and says why in `sources.md`.

Not to be confused with `tools/ask.py`, which is the short capped list of things that are
Anita's call. If an entry here turns out to need her, file an ask that points at it.
"""

import argparse
import importlib
import os
import re
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TAX = os.path.join(ROOT, "taxonomy")
sys.path.insert(0, ROOT)


def modules(only):
    """`<cc><YYYY>.py` mapping modules, oldest edit first. branches/build_tree are not ones."""
    out = []
    for f in sorted(os.listdir(TAX)):
        m = re.fullmatch(r"([a-z]{2})(\d{4})\.py", f)
        if not m:
            continue
        if only and m.group(1) not in only:
            continue
        out.append((m.group(1), m.group(2), f[:-3], os.path.join(TAX, f)))
    return out


def first_line(s):
    """The reason's first sentence-ish, which is where the -> node and the count live."""
    s = " ".join(str(s).split())
    cut = s.find(". ")
    return s if cut < 0 else s[: cut + 1]


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("cc", nargs="*", help="country codes; default every one")
    p.add_argument("--full", action="store_true", help="print the whole reason")
    p.add_argument("--since", type=float, metavar="DAYS",
                   help="only mappings edited in the last N days")
    args = p.parse_args()

    only = {c.lower() for c in args.cc}
    total = countries = 0
    for cc, year, mod, path in modules(only):
        if args.since and (time.time() - os.path.getmtime(path)) / 86400.0 > args.since:
            continue
        try:
            m = importlib.import_module(f"taxonomy.{mod}")
        except Exception as e:                       # a half-written mapping is normal here
            print(f"\n{cc} {year}  -- WILL NOT IMPORT: {type(e).__name__}: {e}")
            continue
        rev = getattr(m, "REVIEW", None)
        if not rev:
            continue
        countries += 1
        print(f"\n{cc} {year}  ({len(rev)})")
        for k, v in rev.items():
            total += 1
            if args.full:
                print(f"  * {k}\n      {' '.join(str(v).split())}\n")
            else:
                print(f"  * {k[:38]:38s}  {first_line(v)[:96]}")

    print(f"\n{total} review entries across {countries} countries"
          f"{' (filtered)' if only or args.since else ''}.")
    if not args.full:
        print("--full for the whole reason on any of them.")


if __name__ == "__main__":
    main()
