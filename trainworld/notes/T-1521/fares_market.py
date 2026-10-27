"""T-1521 round 2: the fare table at market rates, scales from GDP per head at market rates."""
import json, math, sys
sys.argv = [sys.argv[0]]
src = open(__file__.replace("fares_market.py", "fares.py"), encoding="utf-8").read()
exec(src.split('print("scales:"')[0])
wb = json.load(open(r"C:\Users\anita\projects\maps\artifacts\trainworld\T-1521\wb_gdp_usd.json", encoding="utf-8"))[1]
I3 = {"USA": "US", "JPN": "JP", "CHN": "CN", "KOR": "KR", "FRA": "FR", "DEU": "DE", "GBR": "GB", "ESP": "ES", "IND": "IN", "NGA": "NG"}
G = {I3[r["countryiso3code"]]: r["value"] for r in wb if r["countryiso3code"] in I3}

def smk(c, e):
    return round((G[c] / G["IN"]) ** e * 10) / 10

def stats(nodes, e, show=False, basis="mkt"):
    rat = {}
    for c, band, trip, km, f, src_ in ROWS:
        ppp, mkt = FX[c]
        if basis == "mkt":
            g = smk(c, e) * curve(nodes, km); real = f / mkt
        else:
            g = scale(c) * curve(nodes, km); real = f / ppp
        rat.setdefault(band, []).append(real / g)
        if show:
            print(f"{c:3} {band:6} {trip:42} {km:5.0f} real {real:8.2f} game {g:8.2f} ratio {real / g:5.2f}")
    out = {}
    for b, v in rat.items():
        gm = math.exp(sum(map(math.log, v)) / len(v))
        sp = (sum(math.log(x / gm) ** 2 for x in v) / len(v)) ** .5
        out[b] = (gm, sp)
    return out

print("market GDP per head:", {c: round(G[c]) for c in G})
for e in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
    print(f"e {e}: scales", {c: smk(c, e) for c in ("US", "JP", "CN", "IN", "NG")})
    s = stats(OLD, e)
    print("   old-curve shape:", {b: f"mean {m:.2f} spread {sp:.2f}" for b, (m, sp) in s.items()})
print("PPP basis for comparison:", {b: f"mean {m:.2f} spread {sp:.2f}" for b, (m, sp) in stats(OLD, .5, basis='ppp').items()})
if len(sys.argv) > 1:
    pass

us = smk("US", 0.7)
for lab, per in (("old line 10c", 0.10), ("25c past 20", 0.25), ("22c past 20", 0.22), ("28c past 20", 0.28)):
    nodes = [[0, 1.5 / us], [20, 3.5 / us], [100, (3.5 + 80 * per) / us]]
    print(lab, {b: f"mean {m:.2f} spread {sp:.2f}" for b, (m, sp) in stats(nodes, 0.7).items()})
print()
stats([[0, 1.5 / us], [20, 3.5 / us], [100, 23.5 / us]], 0.7, show=True)
print("all scales e 0.7:", {c: smk(c, 0.7) for c in G})
