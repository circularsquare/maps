"""Assemble data/coord_fixes.json from three sources, all resolving to REAL WUP centroids
(never a hand-typed coordinate):
  1. automatic name matches (difflib sim>=0.88 within the entry's own country, moved>=25km)
  2. the agent-resolved Romanization/rename cases -- given as a WUP centre NAME, looked up here
  3. two single bad geocodes found via the graft check rather than coord stacking
Run from the cityhistory/ directory, after tools/propose_coords.py has written the
automatic-match file:
    python tools/propose_coords.py tools/coordfix_auto.json
    python tools/make_coordfix.py  tools/coordfix_auto.json
"""
import json, sys, io, unicodedata
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
AUTO_FILE = sys.argv[1] if len(sys.argv) > 1 else "tools/coordfix_auto.json"

# stadester key -> exact WUP centre name, for cases string matching cannot bridge:
# old Romanizations (Wade-Giles, McCune-Reischauer), renamed cities (Chardzhou ->
# Turkmenabat), and mojibake'd diacritics (To`skent -> Toshkent).
AGENT = json.load(open("tools/agent_picks.json", encoding="utf-8"))
# districts and since-merged towns with no standalone location: "key|parent city" per line
SUBDISTRICTS = "tools/subdistricts.txt"
# single bad geocodes (not on a stacked point, so the stacking scan missed them):
#   Minsk sat at 53.90,37.57 -- longitude 37.57 instead of 27.56, i.e. near Tula, Russia
#   Rabat-Sale sat at 34.17,-6.24, ~60km NE of Rabat, so it grafted to Kenitra
MANUAL = {"Minsk-Belarus": "Minsk", "Rabat-Salé-Morocco": "Rabat"}


def norm(s):
    return "".join(ch for ch in unicodedata.normalize("NFKD", s or "")
                   if not unicodedata.combining(ch)).lower().strip()


wup = json.load(open("data/stadester/wup2025.json", encoding="utf-8"))
byname = defaultdict(list)
centres = []
for v in wup.values():
    co, pop = v.get("coords"), v.get("population") or {}
    if co and pop:
        rec = (co[0], co[1], max(pop.values()), v.get("name", ""), v.get("iso3", ""))
        centres.append(rec)
        byname[norm(rec[3])].append(rec)

raw = json.load(open("data/stadester/stadester_cities.json", encoding="utf-8"))
out, skipped = {}, []

# --- 1. automatic matches ---
auto = json.load(open(AUTO_FILE, encoding="utf-8"))
n_auto = 0
for r in auto:
    if r["sim"] >= 0.88 and (r["moved_km"] or 0) >= 25 and r["to"]:
        out[r["key"]] = {"lat": r["to"][0], "lon": r["to"][1],
                         "via": f"auto name match -> {r['match']}", "peak": r["peak"]}
        n_auto += 1

# --- 2 + 3. name -> WUP centroid ---
n_agent = 0
for key, wname in list(AGENT.items()) + list(MANUAL.items()):
    if key not in raw:
        skipped.append((key, "key not in source")); continue
    cands = byname.get(norm(wname))
    if not cands:
        # tolerate the parenthetical forms WUP uses, e.g. "Toshkent (Tashkent)"
        cands = [c for c in centres if norm(c[3]).startswith(norm(wname))]
    if not cands:
        skipped.append((key, f"no WUP centre named {wname!r}")); continue
    best = max(cands, key=lambda c: c[2])          # largest if the name is ambiguous
    peak = 0
    for v in (raw[key].get("population") or {}).values():
        try:
            peak = max(peak, float(v))
        except (ValueError, TypeError):
            pass
    out[key] = {"lat": round(best[0], 4), "lon": round(best[1], 4),
                "via": f"resolved -> {best[3]}", "peak": int(peak)}
    n_agent += 1

# --- 4. district / merged-town drops ---
n_drop = 0
for line in io.open(SUBDISTRICTS, encoding="utf-8"):
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    key, _, parent = line.partition("|")
    if key not in raw:
        skipped.append((key, "key not in source")); continue
    peak = 0
    for v in (raw[key].get("population") or {}).values():
        try:
            peak = max(peak, float(v))
        except (ValueError, TypeError):
            pass
    out[key] = {"drop": True, "via": f"district/merged town inside {parent}", "peak": int(peak)}
    n_drop += 1
print(f"  district drops: {n_drop}")

json.dump(dict(sorted(out.items())), open("data/coord_fixes.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print(f"wrote data/coord_fixes.json: {len(out)} entries ({n_auto} auto, {n_agent} resolved)")
print(f"  >=100k: {sum(1 for v in out.values() if v['peak'] >= 100000)}")
for k, why in skipped:
    print(f"  SKIPPED {k}: {why}")
print("\nlargest repairs:")
for k, v in sorted(out.items(), key=lambda kv: -kv[1]["peak"])[:12]:
    print(f"  {v['peak']:>9,}  {k:<32} {v['via']}")
