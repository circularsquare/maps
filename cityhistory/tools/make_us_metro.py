"""make_us_metro.py -- build tools/us_metro.csv, annual Metropolitan Statistical Area
populations 2000-2024, keyed by build.py's entry keys.

WHY, ON TOP OF THE FUA LAYER. prep_fua.py fixes the *shape* of the American modern era -- a
functional urban area is a metro rather than a dense core, so Chicago stops being 3.68M. But
eFUA's US figures are a model of an MSA, not the MSA, and where the model is off it is off by
a lot: eFUA puts Boston at 2.79M against a real MSA of 4.9M, because its commuting zone missed
most of the Massachusetts suburbs. The Census Bureau publishes the actual thing, annually, for
every one of the 387 MSAs, so for the United States there is no reason to use an approximation.

This layer therefore REPLACES the FUA/WUP series for the US cities it matches, rather than
adjusting it -- the same one-definition-per-city rule the FUA layer follows. A city runs on
MSA figures for its whole modern era or it does not use them at all.

WHAT IT DOES NOT FIX. The seam itself. populstat's American tail is CITY PROPER -- its 2000
figures are the census place counts almost exactly (Atlanta 416,000 against a census 416,474,
Indianapolis 792,000 against 791,926) -- so handing over to any metropolitan figure is a real
change of unit and steps by whatever the suburbs are worth. Atlanta's is 11x. That step is not
an error to be tuned away; it is the difference between the city of Atlanta and metropolitan
Atlanta, and the viewer already draws the handover in its own style for exactly this reason.
What this file buys is that the number on the far side is the right one.

MATCHING. CBSA titles name their principal cities and states ("Atlanta-Sandy Springs-Alpharetta,
GA"), and build.py keys entries by the raw stadester key, "<City>-<State>" / "<City>-United
States". Rather than guess which of those exist, candidate keys are checked against the keys of
stadester_cities.json itself, so this file can never invent an entry.

Candidate names come from the title's place segment, tried longest-first: the whole segment,
then progressively shorter hyphen prefixes, then each "--"/"/" separated part. Longest-first is
what picks "Nashville-Davidson" out of "Nashville-Davidson--Murfreesboro--Franklin" while still
reducing "Atlanta-Sandy Springs-Alpharetta" to plain "Atlanta". "St." is also tried as "Saint"
(stadester writes Saint Louis) and a leading "Urban " is stripped (Urban Honolulu).

ONLY THE FIRST STATE in the title is used, which is the principal city's. Using all of them
double-counts wherever the same name exists on both sides of a state line: "Kansas City, MO-KS"
would otherwise hand the identical 2.25M to Kansas City-Missouri AND Kansas City-Kansas, two
real and separate entries five kilometres apart.

The bare "<City>-United States" key is emitted only for the LARGEST MSA with that principal
name, because it is the ambiguous one: Columbus GA and Columbus OH both want it and only the
Ohio one should have it. It is also the only route for a city whose state-qualified key does
not exist -- there is no "Washington-District of Columbia" entry, just "Washington-United
States".

Sources (US Census Bureau, Population Estimates Program, public domain):
  2000-2009  cbsa-est2009-alldata.csv   POPESTIMATE2000..2009
  2010-2019  cbsa-est2019-alldata.csv   POPESTIMATE2010..2019
  2020-2024  cbsa-est2024-alldata.csv   POPESTIMATE2020..2024
Run from the cityhistory directory:  python tools/make_us_metro.py
"""
import csv, json, os, sys
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

FILES = [("data/cbsa2009.csv", range(2000, 2010)),
         ("data/cbsa2019.csv", range(2010, 2020)),
         ("data/cbsa2024.csv", range(2020, 2025))]
VOCAB = "data/stadester/stadester_cities.json"
OUT   = "tools/us_metro.csv"
YEARS = list(range(2000, 2025))

STATES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "DC": "District of Columbia",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois",
    "IN": "Indiana", "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana",
    "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan",
    "MN": "Minnesota", "MS": "Mississippi", "MO": "Missouri", "MT": "Montana",
    "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey",
    "NM": "New Mexico", "NY": "New York", "NC": "North Carolina", "ND": "North Dakota",
    "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania",
    "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee",
    "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington",
    "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming", "PR": "Puerto Rico",
}


def read_cbsa():
    """{cbsa_code: (title, {year: pop})}, metropolitan statistical areas only.

    The CBSA-level row is the one with no metropolitan DIVISION and no county: the same file
    repeats each metro broken down both ways, and summing any of it would double-count."""
    pops, titles = defaultdict(dict), {}
    for path, years in FILES:
        if not os.path.exists(path):
            sys.exit(f"missing {path}")
        with open(path, encoding="latin-1", newline="") as f:
            for row in csv.DictReader(f):
                if (row.get("LSAD") or "").strip() != "Metropolitan Statistical Area":
                    continue
                if (row.get("MDIV") or "").strip() or (row.get("STCOU") or "").strip():
                    continue
                code = (row.get("CBSA") or "").strip()
                if not code:
                    continue
                titles[code] = (row.get("NAME") or "").strip()
                for y in years:
                    v = (row.get(f"POPESTIMATE{y}") or "").strip().replace(",", "")
                    if v:
                        try:
                            pops[code][y] = int(float(v))
                        except ValueError:
                            pass
    return {c: (titles[c], pops[c]) for c in pops if pops[c]}


def candidate_names(place):
    """Names to try for a CBSA place segment, longest/most-specific first.

    'Nashville-Davidson--Murfreesboro--Franklin' has to yield 'Nashville-Davidson' (a real
    entry) before it yields 'Nashville' (not one), while 'Atlanta-Sandy Springs-Alpharetta'
    must fall all the way through to 'Atlanta'. Trying hyphen prefixes longest-first does both,
    and requiring the result to exist as a stadester key is what stops the walk early."""
    out = []

    def add(s):
        s = s.strip()
        if not s or s in out:
            return
        out.append(s)
        if s.startswith("Urban "):              # 'Urban Honolulu'
            add(s[len("Urban "):])
        if "St. " in s:                         # stadester writes 'Saint Louis'
            add(s.replace("St. ", "Saint "))

    parts = [p for chunk in place.split("--") for p in chunk.split("/")]
    for part in parts:
        toks = [t for t in part.split("-") if t.strip()]
        for n in range(len(toks), 0, -1):
            add("-".join(toks[:n]))
    # last resort, after every prefix has failed: the secondary cities in the title. Only a
    # handful of metros are named after a place stadester has never heard of ("North Port-
    # Bradenton-Sarasota"), and it must stay last or Atlanta would resolve to Sandy Springs.
    for part in parts:
        for tok in part.split("-")[1:]:
            add(tok)
    return out


def parse_title(title):
    """'Atlanta-Sandy Springs-Alpharetta, GA' -> (place segment, ['GA', ...])."""
    if "," not in title:
        return None, []
    place, states = title.rsplit(",", 1)
    return place.strip(), [s.strip() for s in states.strip().split("-") if s.strip()]


def load_vocab():
    """The entry keys build.py will look up -- the raw stadester keys themselves."""
    with open(VOCAB, encoding="utf-8") as f:
        return set(json.load(f))


def main():
    cbsa = read_cbsa()
    vocab = load_vocab()
    print(f"MSAs read: {len(cbsa):,}   key vocabulary: {len(vocab):,}")

    # largest first, so the ambiguous bare "-United States" key goes to the biggest claimant
    order = sorted(cbsa.items(), key=lambda kv: -max(kv[1][1].values()))
    used_bare, unmatched = {}, []
    series, codes, titles = defaultdict(dict), defaultdict(list), {}
    for code, (title, pop) in order:
        place, states = parse_title(title)
        if not place:
            continue
        state = STATES.get(states[0], states[0]) if states else None
        keys = []
        for nm in candidate_names(place):
            k = f"{nm}-{state}" if state else None
            bare = f"{nm}-United States"
            hit_state = k in vocab if k else False
            hit_bare = bare in vocab
            if not (hit_state or hit_bare):
                continue
            if hit_state:
                keys.append(k)
            # The bare key is ambiguous -- Columbus OH and Columbus GA both want it -- so it
            # goes to the first (largest) claimant and is then OWNED by that (name, state).
            # Ownership rather than a one-shot flag, because a metro redrawn between vintages
            # comes back round as a second code with the same name and state, and a one-shot
            # set locked the later vintage out: Los Angeles 31080 claimed the bare key and
            # 31100 was refused it, leaving "Los Angeles-United States" with 15 of 25 years
            # while "Los Angeles-California" had all of them.
            if hit_bare and used_bare.setdefault(bare, (nm, state)) == (nm, state):
                keys.append(bare)
            break                      # first name that resolves wins; see candidate_names()
        if not keys:
            unmatched.append((max(pop.values()), title))
            continue
        # MERGE ACROSS CBSA VINTAGES. The three files use the delineations current when they
        # were published, and a metro that was redrawn gets a new code: Los Angeles is 31100
        # "Los Angeles-Long Beach-Santa Ana" for 2000-2009 and 31080 "...-Anaheim" from 2010,
        # Cleveland 17460 then 17410. Keyed on the code alone those look like two metros with
        # half a series each, and whichever was read last silently truncated the other. They
        # resolve to the same entry key, which is exactly the signal that they are one place,
        # so the series are combined -- the year ranges are disjoint by construction.
        for k in dict.fromkeys(keys):
            series[k].update(pop)
            codes[k].append(code)
            titles[k] = title            # the largest/most recent claimant names it

    rows = [(k, "|".join(codes[k]), titles[k], series[k]) for k in series]

    with open(OUT, "w", encoding="utf-8", newline="") as f:
        f.write("# US Metropolitan Statistical Area population, annual 2000-2024.\n")
        f.write("# Generated by tools/make_us_metro.py -- see that file for sourcing and\n")
        f.write("# matching rules. Loaded by build.py as load_us_metro().\n")
        f.write("# Source: US Census Bureau, Population Estimates Program (public domain),\n")
        f.write("#   cbsa-est2009-alldata / cbsa-est2019-alldata / cbsa-est2024-alldata.\n")
        f.write("# key,cbsa,title," + ",".join(f"y{y}" for y in YEARS) + "\n")
        for k, code, title, pop in sorted(rows):
            vals = ",".join(str(pop.get(y, "")) for y in YEARS)
            f.write(f'{k},{code},"{title}",{vals}\n')

    print(f"wrote {OUT}: {len(rows):,} keys over {len({r[1] for r in rows}):,} MSAs")
    partial = [(max(s.values()), k) for k, s in series.items()
               if len([y for y in YEARS if y in s]) < len(YEARS)]
    if partial:
        print(f"  {len(partial):,} keys with an incomplete 2000-2024 series")
        for p, k in sorted(partial, reverse=True)[:6]:
            print(f"    {k:32}{p:>12,.0f}  {len(series[k])} of {len(YEARS)} years")
    print(f"  MSAs with no matching key: {len(unmatched):,}")
    for p, t in sorted(unmatched, reverse=True)[:12]:
        print(f"    {p:>10,}  {t}")


if __name__ == "__main__":
    main()
