"""Print the --countries argument for tiles.py and buffers.py.

WHY THIS EXISTS. `tiles.py --countries` REPLACES the archive and `buffers.py --countries`
REPLACES the manifest, so a country left off either list is invisible on the map even though
its tiles, its counts.json entry and its legend are all perfect. COMMANDS.txt calls these two
"the silent ones" and Ghana shipped broken this way once already. Typing the list by hand is
the failure mode; this derives it.

The list is every country that has BOTH editions of its dots on disk, intersected with
countries.py's COUNTRIES. A country in COUNTRIES with no dots is reported rather than
included -- that is a country someone registered and has not scattered yet, and silently
tiling without it is exactly the bug this file is about.

Run:  python tools/built_countries.py
      python tiles.py --countries "$(python tools/built_countries.py)" --coarse --no-atomic
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PROC = os.path.join(ROOT, "data", "processed")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def built():
    fine, coarse = set(), set()
    for fn in os.listdir(PROC):
        m = re.match(r"^dots_([a-z]{2})(_10k)?\.geojson$", fn)
        if not m:
            continue
        (coarse if m.group(2) else fine).add(m.group(1))
    return fine, coarse


def main():
    sys.path.insert(0, ROOT)
    from countries import COUNTRIES

    fine, coarse = built()
    both = sorted(fine & coarse & set(COUNTRIES))

    problems = []
    for cc in sorted(set(COUNTRIES) - set(both)):
        why = []
        if cc not in fine:
            why.append("no 1:1,000 dots")
        if cc not in coarse:
            why.append("no 1:10,000 dots (scatter.py --dot-value 10000)")
        problems.append(f"{cc}: {', '.join(why)}")
    for cc in sorted((fine | coarse) - set(COUNTRIES)):
        problems.append(f"{cc}: has dots but is not in countries.py COUNTRIES")

    if "--check" in sys.argv:
        print(f"{len(both)} countries with both editions:")
        print("  " + ",".join(both))
        if problems:
            print(f"\n{len(problems)} not included:")
            for p in problems:
                print(f"  ! {p}")
        else:
            print("\nOK every registered country has both editions")
        return 0 if not problems else 1

    # bare output is the argument itself, so it can be substituted straight into a command
    print(",".join(both))
    return 0


if __name__ == "__main__":
    sys.exit(main())
