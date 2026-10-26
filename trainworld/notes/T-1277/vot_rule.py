"""T-1277 round 2: minutes per fare dollar by country, m = K (y / Y_REF)^-E, y GDP per head at PPP
(World Bank NY.GDP.PCAP.PP.KD, the world field's countries)."""
import json
import sys

d = json.load(open("C:/Users/anita/projects/maps/trainworld/data/work/world_field.countries.json", encoding="utf-8"))
gdp = {c["iso2"]: c["gdp"] for c in d["countries"] if c.get("gdp")}
Y_REF = 60000.0
SHOW = ["US", "CH", "NO", "DE", "CA", "FR", "KR", "GB", "IT", "JP", "ES", "PL", "TR", "CN", "MX", "BR", "TH", "EG", "ID", "VN", "ZA", "IN", "PH", "BD", "NG", "PK", "KE", "ET"]


def m(y, k, e):
    return k * (y / Y_REF) ** -e


for k, e in ((5.0, 0.7), (5.0, 0.5), (5.0, 1.0)):
    print(f"k {k} at ${Y_REF:.0f}, e {e}: " + ", ".join(f"{c} {m(gdp[c], k, e):.1f}" for c in SHOW if c in gdp))
# weights for the harness (relative to today's 3 minutes)
for c in ("US", "GB", "IN", "BR", "NG"):
    print(c, gdp[c], " ".join(f"e{e}: {m(gdp[c], 5.0, e) / 3:.3f}" for e in (0.5, 0.7, 1.0)))
# people-weighted share of the world's population by weight band, e 0.7
pops = {c["iso2"]: c.get("wb_pop") or 0 for c in d["countries"]}
bands = [(0, 5), (5, 7), (7, 10), (10, 15), (15, 25), (25, 1e9)]
tot = sum(pops[c] for c in gdp if pops.get(c))
for lo, hi in bands:
    s = sum(pops[c] for c in gdp if pops.get(c) and lo <= m(gdp[c], 5.0, 0.7) < hi)
    print(f"  e 0.7, {lo}-{hi} minutes: {100 * s / tot:.0f}% of people")
