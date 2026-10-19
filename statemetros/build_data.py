"""build_data.py -- data/statemetros.json: the biggest metro area in each US state, 1900-2020.

Method (see NOTES.md): today's CBSA definitions (July 2023 delineation) are applied backwards
to every census since 1900 by adding up county populations. A metro that crosses a state line
counts as one candidate per state, using only the counties inside that state. Micropolitan areas
are candidates too, so small states have somebody to lead them in 1900.

Inputs, all in raw/ (see NOTES.md for where each came from):
  county_pop.csv      county populations 1900-2020 (Van Leuven's update of Forstall)
  list1_2023.xlsx     CBSA -> county membership
  list2_2023.xlsx     principal cities, used for labels
  2023_Gaz_counties_national.txt   county centroids, used to place bubbles
Run from this directory:  python build_data.py
"""
import csv, json, math, os, sys
from collections import defaultdict
import openpyxl

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")
OUT = os.path.join(HERE, "data", "statemetros.json")

YEARS = list(range(1900, 2021, 10))

# Connecticut's 2023 delineation uses planning regions, but the population file has the old
# counties. Mapped by hand; Waterbury (Naugatuck Valley) does not follow a county line and is
# dropped. Fairfield is listed twice on purpose (two regions, one county).
CT_REGIONS = {
    "09110": ["09003", "09013"],  # Capitol: Hartford, Tolland
    "09130": ["09007"],           # Lower Connecticut River Valley: Middlesex
    "09120": ["09001"],           # Greater Bridgeport: Fairfield
    "09190": ["09001"],           # Western Connecticut: Fairfield
    "09170": ["09009"],           # South Central: New Haven
    "09180": ["09011"],           # Southeastern: New London
    "09150": ["09015"],           # Northeastern: Windham
    "09160": ["09005"],           # Northwest Hills: Litchfield
    "09140": [],                  # Naugatuck Valley: dropped
}

# Counties with no row in the population file, so they add nothing. Broomfield CO (2001) is
# about 1% of Denver; the surrounding counties in the file may or may not still include it.
SKIP_STATES = {"72", "11"}  # Puerto Rico, District of Columbia


def sheet_rows(path, ncols):
    ws = openpyxl.load_workbook(path, read_only=True).worksheets[0]
    for r in ws.iter_rows(min_row=4, values_only=True):
        if r[0] and r[ncols - 1] is not None and str(r[0]).isdigit():
            yield r


# ---- counties
pop = {}
with open(os.path.join(RAW, "county_pop.csv"), encoding="utf-8") as f:
    for r in csv.DictReader(f):
        pop[r["cty_fips"]] = [int(r["pop_%d" % y]) for y in YEARS]

centroid = {}
state_abbr = {}
with open(os.path.join(RAW, "2023_Gaz_counties_national.txt"), encoding="utf-8") as f:
    for r in csv.reader(f, delimiter="\t"):
        if r[1] == "GEOID":
            continue
        centroid[r[1]] = (float(r[9]), float(r[8].strip()))  # lon, lat
        state_abbr[r[1][:2]] = r[0]

# The 2023 gazetteer has Connecticut's planning regions, not its old counties. Approximate
# county centres, good enough for placing a bubble.
for k, ll in {"09001": (-73.37, 41.23), "09003": (-72.73, 41.80), "09005": (-73.24, 41.79),
              "09007": (-72.52, 41.47), "09009": (-72.93, 41.35), "09011": (-72.10, 41.49),
              "09013": (-72.34, 41.86), "09015": (-71.99, 41.83)}.items():
    centroid[k] = ll

# ---- CBSA membership
cbsa = {}  # code -> dict(title, kind, members: {(statefips): set(county fips)})
for r in sheet_rows(os.path.join(RAW, "list1_2023.xlsx"), 12):
    code, title, kind, sfips, cfips = r[0], r[3], r[4], r[9], r[10]
    if not sfips or sfips in SKIP_STATES:
        continue
    c = cbsa.setdefault(code, dict(title=title, kind="metro" if kind.startswith("Metro") else "micro",
                                    members=defaultdict(set)))
    fips = sfips + cfips
    if sfips == "09":
        for f in CT_REGIONS.get(fips, []):
            c["members"]["09"].add(f)
    elif fips in pop:
        c["members"][sfips].add(fips)
    else:
        print("no population row for", title, fips)

# ---- principal cities, for labels
pcities = defaultdict(list)  # (code, statefips) -> [names]
for r in sheet_rows(os.path.join(RAW, "list2_2023.xlsx"), 6):
    pcities[(r[0], r[4])].append(r[3])


SUFFIXES = (" metropolitan government", " metro government", " consolidated government",
            " unified government", " urban county", " municipality", " borough", " town", " city")


SHORT = {"Nashville-Davidson": "Nashville", "Butte-Silver Bow": "Butte"}


def clean_city(n):
    # "Indianapolis city (balance)" -> "Indianapolis", but "Kansas City" stays whole
    if "(" in n or "/" in n:
        n = n.split("/")[0].split("(")[0].strip()
        for s in SUFFIXES:
            if n.lower().endswith(s):
                n = n[:-len(s)].strip()
    n = n[len("Urban "):] if n.startswith("Urban ") else n
    return n


def display(n):
    return SHORT.get(n, n)


def label(code, sfips, title):
    """Name of the state's part of a CBSA: its principal city that comes first in the CBSA title.
    Returns None when the CBSA has no principal city in this state (a spillover, not a metro
    of that state)."""
    names = pcities.get((code, sfips))
    if not names:
        return None
    place = title.rsplit(",", 1)[0]
    clean = [clean_city(n) for n in names]
    pos = [place.find(n) if place.find(n) >= 0 else 999 for n in clean]
    return display(clean[pos.index(min(pos))])


# ---- one candidate per (cbsa, state)
cands = []
for code, c in cbsa.items():
    for sfips, counties in c["members"].items():
        if not counties:
            continue
        series = [sum(pop[k][i] for k in counties) for i in range(len(YEARS))]
        wsum = sum(pop[k][-1] for k in counties) or 1
        lon = sum(centroid[k][0] * pop[k][-1] for k in counties) / wsum
        lat = sum(centroid[k][1] * pop[k][-1] for k in counties) / wsum
        name = label(code, sfips, c["title"])
        if name is None:
            continue
        cands.append(dict(id="%s-%s" % (code, sfips), state=state_abbr[sfips], sfips=sfips,
                          name=name, title=c["title"], kind=c["kind"],
                          pop=series, lon=round(lon, 3), lat=round(lat, 3),
                          counties=sorted(counties)))


def at(series, t):
    """Log-linear interpolation between census years."""
    i = min(int((t - YEARS[0]) // 10), len(YEARS) - 2)
    f = (t - YEARS[i]) / 10.0
    a, b = max(series[i], 1), max(series[i + 1], 1)
    return a * (b / a) ** f


by_state = defaultdict(list)
for c in cands:
    by_state[c["sfips"]].append(c)

keep = {}
timeline = {}
for sfips, cs in sorted(by_state.items()):
    ab = state_abbr[sfips]
    last = None
    tl = []
    for t in range(YEARS[0], YEARS[-1] + 1):
        lead = max(cs, key=lambda c: at(c["pop"], t))
        if lead["id"] != last:
            tl.append((t, lead["name"], lead["kind"]))
            keep[lead["id"]] = lead
            last = lead["id"]
    timeline[ab] = tl

# non-CBSA counties that would beat their state's leader in some census year
cbsa_counties = {k for c in cbsa.values() for s in c["members"].values() for k in s}
for k, p in pop.items():
    if k in cbsa_counties or k[:2] in SKIP_STATES:
        continue
    for i, y in enumerate(YEARS):
        best = max(c["pop"][i] for c in by_state[k[:2]])
        if p[i] > best:
            print("NOTE: non-CBSA county %s (%s) beats the leader in %d: %d > %d" % (k, state_abbr[k[:2]], y, p[i], best))

print()
for ab, tl in sorted(timeline.items()):
    print(ab, " -> ".join("%s %d" % (n, t) for t, n, _ in tl))

print()
for c in sorted(keep.values(), key=lambda c: (c["state"], c["id"])):
    print("%s %-18s %-45s %s" % (c["state"], c["name"], c["title"], " ".join("%d" % (p / 1000) for p in c["pop"])))

out = dict(years=YEARS, areas=[
    dict(id=c["id"], state=c["state"], name=c["name"], title=c["title"], kind=c["kind"],
         lon=c["lon"], lat=c["lat"], pop=c["pop"])
    for c in sorted(keep.values(), key=lambda c: (c["state"], c["id"]))])
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(out, f, separators=(",", ":"))
print("\nwrote", OUT, len(out["areas"]), "areas,", os.path.getsize(OUT), "bytes")
