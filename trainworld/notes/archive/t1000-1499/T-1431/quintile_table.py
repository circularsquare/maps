"""T-1431: add each country's income shares by quintile to app/src/game/countryIncome.json (`quint`,
per ISO 3166 alpha-2 the five shares in percent, lowest first): the World Bank's SI.DST.FRST.20 to
SI.DST.05TH.20, each country's latest survey; a country with none takes the median of its subregion's
(the world field's subregions), else the world's.

    python trainworld/notes/T-1431/quintile_table.py <wb_quintiles_all.json> [trainworld folder]

The input is the API's answer to
https://api.worldbank.org/v2/country/all/indicator/SI.DST.FRST.20;SI.DST.02ND.20;SI.DST.03RD.20;SI.DST.04TH.20;SI.DST.05TH.20?source=2&format=json&mrnev=1&per_page=5000
"""
import json, statistics, sys
from pathlib import Path

ROOT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parents[2]
ORDER = ["SI.DST.FRST.20", "SI.DST.02ND.20", "SI.DST.03RD.20", "SI.DST.04TH.20", "SI.DST.05TH.20"]
wb = json.load(open(sys.argv[1], encoding="utf-8"))[1]
countries = json.load(open(ROOT / "data/work/world_field.countries.json", encoding="utf-8"))["countries"]
iso2_of = {c["iso3"]: c["iso2"] for c in countries if c.get("iso2") and c.get("iso3")}
got, year = {}, {}
for r in wb:
    c = iso2_of.get(r["countryiso3code"])
    if c and r["value"] is not None:
        got.setdefault(c, {})[r["indicator"]["id"]] = r["value"]
        year[c] = max(year.get(c, ""), r["date"])
own = {c: [v[k] for k in ORDER] for c, v in got.items() if all(k in v for k in ORDER)}
norm = lambda q: [round(100 * x / sum(q), 2) for x in q]
own = {c: norm(q) for c, q in own.items()}
by_sub = {}
for c in countries:
    if c.get("iso2") in own:
        by_sub.setdefault(c["subregion"], []).append(own[c["iso2"]])
med = lambda qs: norm([statistics.median(q[i] for q in qs) for i in range(5)])
world = med(list(own.values()))
quint, filled = {}, 0
for c in countries:
    k = c.get("iso2")
    if not k:
        continue
    if k in own:
        quint[k] = own[k]
    else:
        quint[k] = med(by_sub[c["subregion"]]) if c["subregion"] in by_sub else world
        filled += 1
p = ROOT / "app/src/game/countryIncome.json"
out = json.load(open(p, encoding="utf-8"))
out["quint_note"] = "income share by quintile, %, lowest first (T-1431): World Bank SI.DST.FRST.20 to SI.DST.05TH.20, latest survey; with none, the subregion's median"
out["quint"] = dict(sorted(quint.items()))
p.write_text(json.dumps(out, separators=(",", ":"), ensure_ascii=False) + "\n", encoding="utf-8")
print(f"{len(own)} countries with their own survey (years {min(year.values())}-{max(year.values())}), {filled} from a median -> {p}")
