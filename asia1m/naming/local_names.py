"""Picked names with their local script -> local_names.csv (batch, n, key, name, zh, native, lang).

    python asia1m/naming/local_names.py

`zh` comes from the picked candidate's `local` field (admin suffixes like 县/市/区 dropped, since
the map name drops them too). `native` and `lang` come from local_native.csv, a hand-kept file
for names in a language other than Mandarin (Tibetan, Uyghur, Kazakh, Mongolian, Yi), and from
local_fill.csv for picks whose characters aren't in any candidate. Rows with an empty zh are the
ones still to fill.
"""
import os, re, csv, json, glob
HERE = os.path.dirname(os.path.abspath(__file__))
STEM = re.compile(r"(自治县|自治旗|林区|特区|新区|县|区|市|旗|地区|盟|自治州|街道)$")
norm = lambda s: re.sub(r"[^a-z]", "", s.lower().replace("ü", "u"))

cands = {}
for f in glob.glob(f"{HERE}/batches/*/candidates.json"):
    c = json.load(open(f, encoding="utf-8"))
    b = c.get("batch") or os.path.basename(os.path.dirname(f))
    for r in c["regions"]:
        for k in r["candidates"]:
            cands[(b, int(r["n"]), norm(k["name"]))] = k.get("local") or ""
def load(fn):
    p = os.path.join(HERE, fn)
    return {r["key"]: r for r in csv.DictReader(open(p, encoding="utf-8"))} if os.path.exists(p) else {}
fill, native = load("local_fill.csv"), load("local_native.csv")

rows = []
for p in csv.DictReader(open(f"{HERE}/picks.csv", encoding="utf-8")):
    zh = cands.get((p["batch"], int(p["n"]), norm(p["name"])), "")
    if p["key"] in fill and fill[p["key"]].get("zh"): zh = fill[p["key"]]["zh"]
    if zh and len(STEM.sub("", zh)) >= 2: zh = STEM.sub("", zh)
    nv = native.get(p["key"], {})
    rows.append(dict(batch=p["batch"], n=p["n"], key=p["key"], name=p["name"], zh=zh,
                     native=nv.get("native", ""), lang=nv.get("lang", "")))
rows.sort(key=lambda r: (r["batch"], int(r["n"])))
with open(f"{HERE}/local_names.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
missing = [r for r in rows if not r["zh"]]
print(f"{len(rows)} names, {len(missing)} without characters")
for r in missing: print(f"  {r['batch']} {r['n']} {r['key']} {r['name']}")
