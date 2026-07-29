"""Characterize Stadester coverage: density by century, geolocation, test cities."""
import json, sys, math
from collections import Counter

path = sys.argv[1] if len(sys.argv) > 1 else "data/stadester/stadester_cities.json"
print(f"Loading {path} ...")
with open(path, encoding="utf-8") as f:
    d = json.load(f)

print(f"Total city entries: {len(d):,}")

# --- provenance / type breakdown ---
types = Counter()
countries = Counter()
have_coords = 0
coord_zero = 0
year_min_global, year_max_global = 9999, -9999
pop_years_per_city = []

# For each benchmark year, how many cities have a population value?
benchmark_years = [-3000, -2000, -1000, -500, 1, 500, 1000, 1300, 1500, 1600,
                   1700, 1750, 1800, 1850, 1900, 1950, 2000, 2020]
# store as strings since keys are strings; negative years may be "-3000"
def year_pop(city, y):
    p = city.get("population", {})
    return p.get(str(y))

count_at_year = {y: 0 for y in benchmark_years}
count_at_year_big = {y: 0 for y in benchmark_years}  # >= 10k

for key, c in d.items():
    types[c.get("type", "?")] += 1
    countries[c.get("country", "?")] += 1
    co = c.get("coords")
    if co and len(co) == 2 and (co[0] or co[1]):
        have_coords += 1
        if co[0] == 0 and co[1] == 0:
            coord_zero += 1
    pop = c.get("population", {})
    if pop:
        yrs = [int(y) for y in pop.keys()]
        year_min_global = min(year_min_global, min(yrs))
        year_max_global = max(year_max_global, max(yrs))
        pop_years_per_city.append(len(yrs))
    for y in benchmark_years:
        v = year_pop(c, y)
        if v is not None and v > 0:
            count_at_year[y] += 1
            if v >= 10000:
                count_at_year_big[y] += 1

print(f"\nGlobal year range in data: {year_min_global} to {year_max_global}")
print(f"Cities with usable coords: {have_coords:,} ({100*have_coords/len(d):.1f}%)  |  at (0,0): {coord_zero}")

print("\n--- Provenance (type) ---")
for t, n in types.most_common():
    print(f"  {t:14s} {n:6,}")

print("\n--- Coverage by benchmark year (cities with pop > 0) ---")
print(f"{'year':>6} | {'#cities':>8} | {'#>=10k':>8}")
for y in benchmark_years:
    print(f"{y:>6} | {count_at_year[y]:>8,} | {count_at_year_big[y]:>8,}")

# --- test cities cross-check ---
print("\n--- Test cities (search by name substring) ---")
import unicodedata
def norm(s):
    return "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch)).lower()
targets = ["London", "Roma", "Baghdad", "Bagdad", "Tenochtitlan", "Mexico",
           "Beijing", "Peking", "Tokyo", "Edo", "Constantin", "Athens", "Alexandria"]
for t in targets:
    tn = norm(t)
    matches = [(k, c) for k, c in d.items() if tn in norm(c.get("name",""))
               or any(tn in norm(o) for o in c.get("other_names",[]))]
    for k, c in matches[:2]:
        pop = c.get("population", {})
        yrs = sorted(int(y) for y in pop.keys())
        span = f"{yrs[0]}..{yrs[-1]}" if yrs else "none"
        # sample a few values
        samples = []
        for sy in [-500, 1, 1000, 1500, 1800, 1900, 2000]:
            v = pop.get(str(sy))
            if v: samples.append(f"{sy}:{int(v/1000)}k")
        print(f"  {c.get('name'):16s} [{c.get('country','?'):18s}] type={c.get('type','?'):9s} "
              f"yrs={span:14s} n={len(yrs):4d}  {' '.join(samples)}")
