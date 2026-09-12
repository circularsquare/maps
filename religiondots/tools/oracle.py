"""UNSD Demographic Yearbook table 28 — the religion oracle, WITH ITS COUNTS.

    python tools/oracle.py --fetch          pull and cache the table (222 KB)
    python tools/oracle.py Palau            print one country's rows
    python tools/oracle.py --list           every country and its latest year

    from oracle import oracle
    oracle("Palau")            -> {year: {area: {religion: count}}}
    oracle("Palau", 2005)      -> {area: {religion: count}}

**sources.md §11r DOCUMENTED A COLUMN SET THAT RETURNS NO NUMBERS**, which is why the oracle
was only ever asked *"has this office ever tabulated religion at all"*. Its

    &c=0,1,2,3,4,5,6      -> Table Code | Country | Year | Reference Date | Area Code | Area

is an index of which tabulations exist with the values stripped out. The set that carries
them, found 2026-09-08 while wiring Bulgaria (§9be), is

    &c=0,2,3,6,8,10,15,16 -> Country | Year | Area | Sex | Religion | Source Year | Value

**THREE TRAPS, EACH OF WHICH LOOKS LIKE SOMETHING ELSE.**

* Asking for **17 columns or more returns a zero-byte body with HTTP 200**. That reads as a
  network fault or a dead host, not as a rejected request.
* Narrowing with `DataFilter=tableCode:28;countryCode:100` **also returns zero bytes**, so
  the filter is effectively table-code only: pull the whole file and filter locally.
* The country name is column **2**, not column 0. Filtering on `row[0]` matches nothing and
  is indistinguishable from *"this country is absent from the oracle"* — which is the single
  conclusion §11r warns is expensive to get wrong. It happened on the first pass at Bulgaria,
  which is present with 2001, 2011 and 2021.

**WHAT IT IS GOOD FOR, AND WHAT IT IS NOT.** Every row is national or urban/rural, so it can
never substitute for a subnational source in a country big enough to need one. Two uses:

  1. **A SOURCE for the microstate tier.** At 1 dot = 1,000 people a country of 20,000 draws
     twenty dots and their placement carries no claim, so spec §3.9b/§3.9c and Anita's call
     of 2026-09-08 make a national-only table a complete source. The rows are exact
     partitions: Palau's nine categories sum to 19,907 and the Cook Islands' nine to 14,974,
     to the person.
  2. **A CHECK everywhere else.** An independent publisher of the same census, agreeing or
     not to the person. sources/sk.py had to reconstruct Slovakia's category tail by hand
     from a table this prints.

**AND ABSENCE STILL PROVES ONLY WHAT §11r SAID IT DID.** Reporting is voluntary and
census-only; Spain, China, Kosovo, Bosnia, Georgia and Russia are all drawn here and all
absent. It proves "no census religion tabulation was forwarded", never "nothing to draw".
"""

import csv
import io
import os
import ssl
import sys
import urllib.request
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CACHE = os.path.join(ROOT, "data", "raw", "unsd", "dyb_table28_values.zip")

URL = ("http://data.un.org/Handlers/DownloadHandler.ashx?DataFilter=tableCode:28"
       "&DataMartId=POP&Format=csv&c=0,2,3,6,8,10,15,16")

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"}

# The columns, in the order the handler returns them for the set above.
COUNTRY, YEAR, AREA, SEX, RELIGION, SRC_YEAR, VALUE = 1, 2, 3, 4, 5, 6, 7
TOTAL = "Total"


def fetch(force=False):
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    if os.path.exists(CACHE) and os.path.getsize(CACHE) > 100_000 and not force:
        print(f"  have {CACHE} ({os.path.getsize(CACHE):,} bytes)")
        return
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    body = urllib.request.urlopen(urllib.request.Request(URL, headers=UA),
                                  timeout=240, context=ctx).read()
    if len(body) < 100_000:
        raise SystemExit(f"the handler returned {len(body)} bytes — see the docstring's "
                         "three traps; a short body is a rejected request, not a bad network")
    tmp = CACHE + ".part"
    with open(tmp, "wb") as fh:
        fh.write(body)
    os.replace(tmp, CACHE)
    print(f"  got {len(body):,} bytes -> {CACHE}")


def _rows():
    if not os.path.exists(CACHE):
        raise SystemExit(f"missing {CACHE} — run `python tools/oracle.py --fetch`")
    z = zipfile.ZipFile(CACHE)
    with z.open(z.namelist()[0]) as fh:
        yield from csv.reader(io.TextIOWrapper(fh, encoding="utf-8-sig"))


def table(sex="Both Sexes"):
    """-> {country: {year: {area: {religion: int}}}}, the whole oracle."""
    out = {}
    for r in _rows():
        if len(r) <= VALUE or r[SEX] != sex or r[COUNTRY] == "Country or Area":
            continue
        v = r[VALUE].strip()
        if not v.replace(".", "").isdigit():
            continue
        (out.setdefault(r[COUNTRY], {}).setdefault(r[YEAR], {})
            .setdefault(r[AREA], {})[r[RELIGION]]) = int(float(v))
    return out


def oracle(country, year=None, sex="Both Sexes"):
    """One country's rows. `year` picks a census; omitted returns every year it reported."""
    got = table(sex).get(country)
    if got is None:
        return None
    if year is None:
        return got
    return got.get(str(year))


def latest(country, sex="Both Sexes"):
    """-> (year, {religion: count}) for the country's most recent census, Total area."""
    got = table(sex).get(country)
    if not got:
        return None, None
    y = max(got, key=int)
    return y, got[y].get(TOTAL, {})


def partition(counts):
    """-> (categories_without_total, stated_total, exact?) — the check that makes a row usable."""
    cats = {k: v for k, v in counts.items() if k != TOTAL}
    total = counts.get(TOTAL)
    return cats, total, (total is not None and sum(cats.values()) == total)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--fetch" in sys.argv:
        fetch(force="--force" in sys.argv)
    if "--list" in sys.argv:
        t = table()
        print(f"{len(t)} countries\n")
        for name in sorted(t):
            y = max(t[name], key=int)
            cats, total, exact = partition(t[name][y].get(TOTAL, {}))
            mark = "exact" if exact else "NOT a partition"
            print(f"  {name[:36]:<38}{y:>6}{len(cats):>5} cats  "
                  f"{total if total is not None else '?':>12}  {mark}")
        return
    for name in args:
        got = oracle(name)
        if got is None:
            # A MISS IS NOT AN ABSENCE UNTIL THE NAME IS RIGHT. This lookup is an exact match
            # on UNSD's own country string, so a country CODE misses every time whatever the
            # oracle holds, and the old unconditional "proves nothing was forwarded" line was
            # read as a negative and shipped in Micronesia's note (spec §12, review 2026-09-08).
            near = [k for k in table() if name.lower() in k.lower()]
            if len(name) == 2 and name.isalpha():
                print(f"{name!r} looks like a country CODE, and this tool matches UNSD's own "
                      f"country NAME. A cc misses here whatever the oracle holds, so this is "
                      f"NOT evidence of absence. Run `oracle.py --list` and pass the name it "
                      f"prints, e.g. \"Micronesia (Federated States of)\".")
            elif near:
                print(f"{name}: no exact match, but the oracle has "
                      + "; ".join(repr(k) for k in sorted(near)[:5]))
            else:
                print(f"{name}: ABSENT from the oracle "
                      f"(proves no census tabulation was forwarded, nothing more)")
            continue
        for y in sorted(got, key=int, reverse=True):
            for area in ("Total", "Urban", "Rural"):
                if area not in got[y]:
                    continue
                cats, total, exact = partition(got[y][area])
                print(f"\n{name} {y} [{area}] — {len(cats)} categories, "
                      f"total {total:,}" if total is not None else
                      f"\n{name} {y} [{area}] — {len(cats)} categories")
                print(f"   partition: {'EXACT' if exact else 'does NOT sum to the total'}")
                for k, v in sorted(cats.items(), key=lambda kv: -kv[1]):
                    print(f"     {v:>10,}  {k}")


if __name__ == "__main__":
    main()
