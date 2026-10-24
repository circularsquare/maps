"""Sitelinks as a worldwide fame proxy: coverage by country of attraction kinds with >= 10 and >= 30
sitelinks, and how sitelinks track Wikidata's visitor counts where both exist. Read-only, QLever."""
import json
import math
import os
import urllib.parse
import urllib.request

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "artifacts", "trainworld")  # outputs go to artifacts
UA = "trainworld-poi-coverage/0.1 (research script)"
EP = "https://qlever.dev/api/wikidata"
PFX = """PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX wikibase: <http://wikiba.se/ontology#>
"""
KINDS = "wd:Q33506 wd:Q207694 wd:Q43501 wd:Q2281788 wd:Q194195 wd:Q2416723 wd:Q5393308 wd:Q845945 wd:Q2977 wd:Q570116 wd:Q11303 wd:Q12518 wd:Q16970 wd:Q44539 wd:Q32815"


def run(q):
    url = EP + "?" + urllib.parse.urlencode({"query": PFX + q})
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/sparql-results+json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        data = json.load(r)
    return [{k: v["value"] for k, v in b.items()} for b in data["results"]["bindings"]]


cov = run(f"""
SELECT ?cc (COUNT(DISTINCT ?s) AS ?n10) WHERE {{
  VALUES ?cls {{ {KINDS} }}
  ?s wdt:P31 ?cls ; wdt:P625 ?c ; wdt:P17 ?k ; wikibase:sitelinks ?sl . ?k wdt:P297 ?cc . FILTER(?sl >= 10)
}} GROUP BY ?cc""")
cov30 = run(f"""
SELECT ?cc (COUNT(DISTINCT ?s) AS ?n30) WHERE {{
  VALUES ?cls {{ {KINDS} }}
  ?s wdt:P31 ?cls ; wdt:P625 ?c ; wdt:P17 ?k ; wikibase:sitelinks ?sl . ?k wdt:P297 ?cc . FILTER(?sl >= 30)
}} GROUP BY ?cc""")
pairs = run(f"""
SELECT ?s ?v ?sl WHERE {{
  VALUES ?cls {{ {KINDS} }}
  ?s wdt:P31 ?cls ; wdt:P1174 ?v ; wikibase:sitelinks ?sl . FILTER(?v >= 10000)
}}""")
json.dump({"cov10": cov, "cov30": cov30, "pairs": pairs}, open(os.path.join(HERE, "ql_sitelinks.json"), "w"))

a = {r["cc"]: int(r["n10"]) for r in cov}
b = {r["cc"]: int(r["n30"]) for r in cov30}
for c in ["US", "CN", "JP", "GB", "FR", "DE", "IT", "ES", "IN", "ID", "NG", "BR", "MX", "KR", "TR", "VN", "CA", "RU", "EG", "PK", "BD", "TH", "AU"]:
    print(c, a.get(c, 0), b.get(c, 0))

# log-log fit of visitors on sitelinks, one value per item (largest visitor figure)
best = {}
for r in pairs:
    v, sl = float(r["v"]), int(r["sl"])
    if sl < 1:
        continue
    if r["s"] not in best or v > best[r["s"]][0]:
        best[r["s"]] = (v, sl)
xs = [math.log(sl) for v, sl in best.values()]
ys = [math.log(v) for v, sl in best.values()]
n = len(xs)
mx, my = sum(xs) / n, sum(ys) / n
sxx = sum((x - mx) ** 2 for x in xs)
sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
syy = sum((y - my) ** 2 for y in ys)
slope = sxy / sxx
r2 = sxy * sxy / (sxx * syy)
resid = sorted(abs(y - (my + slope * (x - mx))) for x, y in zip(xs, ys))
print(f"items {n}; visitors ~ sitelinks^{slope:.2f}; r2 {r2:.2f}; median |log err| {resid[n // 2]:.2f} (x{math.exp(resid[n // 2]):.1f})")
for lo, hi in ((1, 5), (5, 15), (15, 40), (40, 1000)):
    vs = sorted(v for v, sl in best.values() if lo <= sl < hi)
    if vs:
        print(f"  sitelinks {lo}-{hi}: n {len(vs)}, median visitors {vs[len(vs) // 2]:,.0f}")
