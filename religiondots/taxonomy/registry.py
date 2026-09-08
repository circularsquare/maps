"""Which taxonomy mapping module belongs to which country — DISCOVERED, not listed.

This file exists because the same cc -> module mapping was maintained by hand in two places
(`coverage.py`'s SIMPLE and `tools/check_mapping.py`'s MODULES) and both had to be edited
every time a country was added. On 2026-09-06 that was forgotten three times in one day —
Indonesia, South Korea and Bangladesh each shipped with dots on the map and no coverage
entry, which makes the country go dark for every religion it demonstrably contains. The
check caught it every time and a human still had to fix it every time, which is the wrong
division of labour. `todo.txt` asked for this: *"we should make this registration automatic
if possible"*.

THE CONVENTION IS THE REGISTRY. A country's mapping lives in `taxonomy/<cc><YYYY>.py`, where
`<cc>` is the two-letter code `countries.py` uses and `<YYYY>` is the source's vintage —
`pk2017.py`, `et2007.py`, `bd2011.py`. Nothing else in this directory matches that shape:
`branches.py` and `build_tree.py` have no digits, and `usrc2020.py` and `us_pew2024.py` are
four and three letters before theirs, so the pattern excludes them without a special case.

TWO THINGS IT REFUSES TO GUESS, because guessing either would be worse than the manual list:

  * **A country with TWO vintages on disk.** `pk2017.py` beside a future `pk2023.py` is a real
    prospect (sources/pk.md §7a), and only `countries.py` knows which one is drawn. Raises,
    and says to add an `OVERRIDE` entry.
  * **The two countries whose mapping is not one module exposing `MAP`.** They are named in
    `SPECIAL` below with the reason, and each consumer handles them itself. Note the UK is
    NOT one of them: `uk2021.py` is an ordinary module and the three-census split happens
    later, in `coverage.py`'s `regions()`.
"""

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# taxonomy/<cc><YYYY>.py — the whole convention, in one regex.
FILENAME = re.compile(r"^([a-z]{2})(\d{4})\.py$")

# Countries whose mapping is NOT one <cc><YYYY>.py module exposing MAP. Each is a different
# shape, so each consumer deals with it explicitly; discovery must not hand them over as if
# they were ordinary. The reason is kept here so the next reader does not have to infer it.
SPECIAL = {
    "us": "two instruments (spec §3.5a) — usrc2020 (ASARB, 372 bodies) plus us_pew2024, "
          "whose CUT maps one category to a TUPLE of paths. No us<YYYY>.py exists, so "
          "discovery never produces it; listed here for the reader.",
    "ca": "ca2021.py resolves up StatCan's own parent chain and exposes NODE + LEAF rather "
          "than MAP — AND its resolve() takes a second argument, `parent_of`, so it does "
          "not satisfy the ordinary contract for either consumer. Discovery WOULD find it, "
          "so it is excluded here.",
}

# A country whose module cannot be discovered — two vintages on disk, or a name off the
# convention. Empty is the healthy state; an entry here is a promise to keep it current.
OVERRIDE = {}


def discover(drawn_only=True):
    """Return {cc: module_name} for every country whose mapping can be found.

    `drawn_only` intersects with `countries.py`'s COUNTRIES, which is what keeps
    `coverage.py`'s `regions()` honest: a taxonomy module written before its country is
    registered must not produce a lit region with no dots behind it — that reads as
    "asked, and nobody is there" for a country the map does not draw at all (§6.12).

    SPECIAL countries are always dropped. That was briefly a parameter, on the theory that
    a consumer needing only `resolve()` could take Canada — it cannot: `ca2021.resolve()`
    has a different signature. SPECIAL means special for everyone.
    """
    found = {}
    for fn in sorted(os.listdir(HERE)):
        m = FILENAME.match(fn)
        if not m:
            continue
        cc, year = m.group(1), m.group(2)
        found.setdefault(cc, []).append((year, fn[:-3]))

    out = {}
    for cc, hits in sorted(found.items()):
        if cc in SPECIAL:
            continue
        if cc in OVERRIDE:
            out[cc] = OVERRIDE[cc]
            continue
        if len(hits) > 1:
            names = ", ".join(n for _, n in sorted(hits))
            raise SystemExit(
                f"taxonomy/registry.py: {cc!r} has {len(hits)} mapping modules on disk "
                f"({names}) and only countries.py knows which vintage is drawn. Add "
                f"`OVERRIDE[{cc!r}] = '<module>'` to taxonomy/registry.py.")
        out[cc] = hits[0][1]

    out.update({cc: mod for cc, mod in OVERRIDE.items() if cc not in SPECIAL})

    if not drawn_only:
        return out

    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from countries import COUNTRIES
    return {cc: mod for cc, mod in out.items() if cc in COUNTRIES}


def drawn_without_mapping():
    """Countries countries.py draws that discovery cannot find a mapping for.

    The inverse of the bug this file removes, and it is not hypothetical: it is what a
    typo in a filename, or a module named off the convention, would look like.
    """
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from countries import COUNTRIES
    known = set(discover(drawn_only=False)) | set(SPECIAL)
    return sorted(set(COUNTRIES) - known)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    reg = discover()
    print(f"{len(reg)} discovered mappings for drawn countries:")
    for cc, mod in sorted(reg.items()):
        print(f"   {cc}  ->  {mod}")
    print(f"\n{len(SPECIAL)} handled specially by each consumer: "
          f"{', '.join(sorted(SPECIAL))}")
    if OVERRIDE:
        print(f"{len(OVERRIDE)} pinned by OVERRIDE: {OVERRIDE}")
    missing = drawn_without_mapping()
    print(f"\n{'OK ' if not missing else 'BAD'} drawn countries with no mapping module: "
          f"{missing if missing else 'none'}")
    everything = discover(drawn_only=False)
    undrawn = sorted(set(everything) - set(reg))
    if undrawn:
        print(f"note: {len(undrawn)} mapping module(s) for countries countries.py does not "
              f"draw, ignored: {', '.join(undrawn)}")
