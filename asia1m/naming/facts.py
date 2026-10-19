"""One fact sheet per drawn region -> regions.json, the input for naming batches.

Reads the region and admin rasters that ../regionnaming builds (region_id.npy, adm_all.npy,
adm_twn.npy) plus pop.npy from pop_raster.py and the exported text layer, and records per region:

  units      admin units by share of the region's people, with how much of each unit's own
             people the region takes (cover). Parent units (prefecture, province) the same way.
  towns      China only: townships by people (the county layer lumps many city districts).
  cities     worldcities.csv places of 100k+ inside the region.
  neighbours regions across a border line, by length of shared border.
  text_px    pixels of the text layer inside the region; 0 usually means no label yet, but
             labels sometimes sit outside their region, so it is a hint only.

    python asia1m/naming/facts.py <text.png exported from the 'text' layer>
"""
import os, sys, csv, json, time, re
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__))
RN = os.path.join(HERE, "..", "regionnaming")
sys.path.insert(0, RN)
import numpy as np
from PIL import Image
from geo import lonlat_to_px
Image.MAX_IMAGE_PIXELS = None

t0 = time.time()
def log(m): print(f"[{time.time()-t0:.0f}s] {m}", flush=True)

import scipy.ndimage as ndi
rid = np.load(f"{RN}/region_id.npy")
H, W = rid.shape
# border lines are drawn on the region layer; give each line pixel (and the people on it) to
# the nearest region within 3px, or dense city cores lose a real share of their people
dist, (iy, ix) = ndi.distance_transform_edt(rid == 0, return_indices=True)
rid = np.where((rid == 0) & (dist <= 3), rid[iy, ix], rid)
del dist, iy, ix
regs = {int(r["rid"]): r for r in csv.DictReader(open(f"{RN}/regions_prelim.csv"))}
N = int(rid.max())
pop = np.load(f"{HERE}/pop.npy")
adm = np.load(f"{RN}/adm_all.npy")
names = json.load(open(f"{RN}/adm_all_names.json", encoding="utf-8"))
twn = np.load(f"{RN}/adm_twn.npy")
tnames = json.load(open(f"{RN}/adm_twn_names.json", encoding="utf-8"))
text_a = np.array(Image.open(sys.argv[1]).convert("RGBA"))[:, :, 3] > 0
log("loaded")

flat_r = rid.ravel()
sel = np.flatnonzero(flat_r > 0)
r_sel = flat_r[sel].astype(np.int64)
p_sel = pop.ravel()[sel].astype(np.float64)
reg_pop = np.bincount(r_sel, weights=p_sel, minlength=N + 1)
reg_area = np.bincount(r_sel, minlength=N + 1)
reg_text = np.bincount(r_sel, weights=text_a.ravel()[sel], minlength=N + 1)

def crosstab(lab):
    """(region, unit) -> [pixels, people] over region pixels."""
    a_sel = lab.ravel()[sel].astype(np.int64)
    M = int(lab.max()) + 1
    key = r_sel * M + a_sel
    u, inv = np.unique(key, return_inverse=True)
    px = np.bincount(inv)
    pp = np.bincount(inv, weights=p_sel)
    out = defaultdict(list)
    for k, n, p in zip(u, px, pp):
        out[int(k // M)].append((int(k % M), int(n), float(p)))
    unit_area = np.bincount(lab.ravel(), minlength=M)
    unit_pop = np.bincount(lab.ravel(), weights=pop.ravel(), minlength=M)
    return out, unit_area, unit_pop

ct, u_area, u_pop = crosstab(adm)
log("admin crosstab")
tct, t_area, t_pop = crosstab(twn)
log("township crosstab")

# parent totals (prefecture/district = 'dist', province = 'prov') over the whole map
par_area = defaultdict(float); par_pop = defaultdict(float)
def parents(rec):
    out = []
    if rec.get("prov"):
        out.append(("province", rec["country"], rec["prov"], None))
    if rec.get("dist"):
        out.append(("prefecture", rec["country"], rec["prov"], rec["dist"]))
    return out
for gid in range(1, len(names)):
    rec = names[gid]
    if not rec: continue
    for p in parents(rec):
        par_area[p] += u_area[gid]; par_pop[p] += u_pop[gid]

# China: current counties from the township layer (the GADM county layer is ~2012 and some names
# are stale or wrong: Chengcheng shows as "Chencang"). Key = (province, prefecture, county).
from pypinyin import lazy_pinyin
cn_area = defaultdict(float); cn_pop = defaultdict(float)
for g in range(1, len(tnames)):
    t = tnames[g]
    if not t: continue
    k = (t["prov"], t["city"], t["county"])
    cn_area[k] += t_area[g]; cn_pop[k] += t_pop[g]
STEM = re.compile(r"(自治县|自治旗|林区|特区|新区|县|区|市|旗|地区|盟|自治州)$")
# pypinyin misreads some polyphones (长垣 -> Zhangyuan), so prefer the county layer's own spelling
# when its Chinese name matches; it is old but hand-romanised
GADM_PY = {}
for rec in names[1:]:
    if rec and rec.get("country") == "China" and rec.get("local") and rec.get("name"):
        GADM_PY.setdefault(rec["local"], re.sub(r"\s+(Shì|Qū|Xiàn|Shi|Qu|Xian)$", "", rec["name"]))
def py(zh):
    if not zh: return None
    if zh in GADM_PY: return GADM_PY[zh]
    s = STEM.sub("", zh) or zh
    out = ""
    for i, syl in enumerate(lazy_pinyin(s)):
        out += ("'" if i and syl[:1] in "aeo" else "") + syl
    return out.capitalize()

def cover(pp, px, tp, ta):
    return round(pp / tp, 3) if tp > 0 else round(px / max(ta, 1), 3)

# neighbours: region pairs facing each other across a 1-2px line
pairs = defaultdict(int)
for dy, dx in [(0, 1), (1, 0), (0, 2), (2, 0), (1, 1), (1, -1), (2, 2), (2, -2), (0, 3), (3, 0)]:
    a = rid[: H - dy, max(0, -dx): W - max(0, dx)]
    b = rid[dy:, max(0, dx): W - max(0, -dx)]
    m = (a > 0) & (b > 0) & (a != b)
    if not m.any(): continue
    lo = np.minimum(a[m], b[m]).astype(np.int64); hi = np.maximum(a[m], b[m]).astype(np.int64)
    u, n = np.unique(lo * (N + 1) + hi, return_counts=True)
    for k, c in zip(u, n):
        pairs[(int(k // (N + 1)), int(k % (N + 1)))] += int(c)
nbr = defaultdict(list)
for (x, y), c in pairs.items():
    if c >= 3:
        nbr[x].append((y, c)); nbr[y].append((x, c))
log(f"neighbours: {len(pairs)} pairs")

# cities of 100k+
cities = defaultdict(list)
for c in csv.DictReader(open(os.path.join(HERE, "..", "..", "data", "worldcities.csv"), encoding="utf-8")):
    try: p = float(c["population"] or 0)
    except ValueError: continue
    if p < 100_000: continue
    x, y = lonlat_to_px(float(c["lng"]), float(c["lat"]))
    x, y = int(round(x)), int(round(y))
    if 0 <= x < W and 0 <= y < H and rid[y, x] > 0:
        cities[int(rid[y, x])].append(dict(name=c["city"], admin=c["admin_name"], pop=int(p)))

# keys: the colour, plus position where a colour is used for more than one region
by_color = defaultdict(list)
for r in regs.values(): by_color[r["color"]].append(r)
def key_of(r):
    if len(by_color[r["color"]]) == 1: return r["color"]
    return f'{r["color"]}@{float(r["lon"]):.1f},{float(r["lat"]):.1f}'
keys = {i: key_of(r) for i, r in regs.items()}

out = []
for i, r in regs.items():
    rp = reg_pop[i]; ra = reg_area[i]
    parts = [(g, n, p) for g, n, p in ct.get(i, []) if g != 0]
    land = sum(n for _, n, _ in parts)
    def share(p, n): return round(p / rp, 3) if rp > 0 else round(n / max(ra, 1), 3)
    units = []
    for g, n, p in sorted(parts, key=lambda t: -(t[2] if rp > 0 else t[1])):
        s = share(p, n)
        if s < 0.02 and len(units) >= 3: continue
        rec = names[g]
        units.append(dict(name=rec.get("name"), local=rec.get("local"), prefecture=rec.get("dist"),
                          prefecture_local=rec.get("dist_local"), province=rec.get("prov"),
                          country=rec.get("country"), people_share=s,
                          area_share=round(n / max(land, 1), 3),
                          cover=cover(p, n, u_pop[g], u_area[g])))
    tparts = [(g, n, p) for g, n, p in tct.get(i, []) if g != 0]
    if units and units[0]["country"] == "China" and tparts:
        agg = defaultdict(lambda: [0, 0.0])
        for g, n, p in tparts:
            t = tnames[g]; k = (t["prov"], t["city"], t["county"])
            agg[k][0] += n; agg[k][1] += p
        tland = sum(n for n, _ in agg.values())
        units = []
        for k, (n, p) in sorted(agg.items(), key=lambda t: -(t[1][1] if rp > 0 else t[1][0])):
            s = share(p, n)
            if s < 0.02 and len(units) >= 3: continue
            prov, city, county = k
            units.append(dict(name=py(county), local=county, prefecture=py(city), prefecture_local=city,
                              province=prov, country="China", people_share=s,
                              area_share=round(n / max(tland, 1), 3), cover=cover(p, n, cn_pop[k], cn_area[k])))
    pagg = defaultdict(lambda: [0, 0.0])
    for g, n, p in parts:
        for pk in parents(names[g]):
            pagg[pk][0] += n; pagg[pk][1] += p
    pars = []
    for pk, (n, p) in sorted(pagg.items(), key=lambda t: -t[1][1] if rp > 0 else -t[1][0]):
        s = share(p, n)
        if s < 0.05: continue
        level, country, prov, dist = pk
        local = None
        if level == "prefecture":
            local = next((names[g].get("dist_local") for g, _, _ in parts
                          if names[g].get("dist") == dist and names[g].get("prov") == prov), None)
        pars.append(dict(level=level, name=dist or prov, local=local, province=prov if dist else None,
                         country=country, people_share=s, cover=cover(p, n, par_pop[pk], par_area[pk])))
    countries = defaultdict(float)
    for u in units: countries[u["country"]] += u["people_share"]
    towns = []
    if units and units[0]["country"] == "China":
        # the county data is ~2012 and lumps many city districts together; townships split them
        for g, n, p in sorted(tct.get(i, []), key=lambda t: -(t[2] if rp > 0 else t[1])):
            if g == 0: continue
            s = share(p, n)
            if s < 0.03 and len(towns) >= 4: break
            t = tnames[g]
            towns.append(dict(town=t["town"], county=t["county"], city=t["city"], people_share=s,
                              cover=cover(p, n, t_pop[g], t_area[g])))
            if len(towns) >= 12: break
    out.append(dict(
        key=keys[i], color=r["color"], lon=float(r["lon"]), lat=float(r["lat"]),
        bbox=[int(r["bx0"]), int(r["by0"]), int(r["bx1"]), int(r["by1"])],
        area_px=int(ra), people=int(round(rp)), land_share=round(land / max(ra, 1), 3),
        country=max(countries, key=countries.get) if countries else None,
        units=units[:15], parents=pars[:6], towns=towns,
        cities=sorted(cities.get(i, []), key=lambda c: -c["pop"])[:8],
        neighbours=[dict(key=keys[j], border_px=c) for j, c in sorted(nbr.get(i, []), key=lambda t: -t[1])],
        text_px=int(reg_text[i])))
out.sort(key=lambda d: (d["country"] or "~", -d["lat"]))
json.dump(out, open(f"{HERE}/regions.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
log(f"wrote regions.json: {len(out)} regions, {sum(d['people'] for d in out)/1e9:.2f}bn people")
