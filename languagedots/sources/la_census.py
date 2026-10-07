"""Laos: Lao Statistics Bureau, Population and Housing Census 2015, ethnicity at VILLAGE level.

Reads (or fetches) data/raw/la/ and writes data/normalized/la.csv. The record is sources/la.md.

THE COUNTS. LSB published the 2015 census at village level through the K4D atlas platform
(`gis.cde.unibe.ch/.../Decide`, the same channel religiondots' Laos uses; licence field "With
proper citation of the data sources, the data can be used freely"). For ethnicity it gives ten
ethno-linguistic CATEGORIES as a percentage of each village's population (Lao, Tai-Thai,
Khmuic, Palaungic, Katuic, Bahnaric-Khmer, Vietic, Hmong, Mien, Tibeto-Burman), not the 49
official groups. The 49 groups exist only nationally (report Table P2.7, p.121-122). Three
categories are one group each (Lao, Hmong, Mien = Iu Mien); the other seven hold 2-14 groups.

THE SPLIT OF A CATEGORY INTO ITS NAMED GROUPS (the modelled step). The 2011 agricultural census
on the same server gives each village's main, second and third most numerous ethnic group as a
code 1-49 in Table P2.7's order, and the main group's share of farm households. Within each
village a category's people go to whichever of the village's three 2011 groups belong to that
category (main weighted by its household share, second and third sharing the rest, 2:1); where
none does, or the village has no 2011 record, they go to the category's groups in proportion to
the district's already-assigned mix, else the province's, else the nation's. Then each group's
village counts are raked (iterative proportional fitting, villages x groups within a category)
so that every group's national total is Table P2.7's and every village's category total is the
census's. Village category counts are therefore exact, national group totals are exact, and
only which villages of a category hold which of its groups is modelled. Rows are `derived`.

CHECKS. The ten percentages sum to ~100 in each village; the category percentages times the
village population land on integers (LSB's eight significant digits, as for religion); the
four category COUNT services (Tai-Thai, Mon-Khmer, Mien, Sino-Tibetan) agree with the
percentages; the national category totals reproduce Table P2.7 summed by the group->category
map in GROUP_CATEGORY (that check is what fixes the map).

Usage:
    python sources/la_census.py --fetch
    python sources/la_census.py
"""

import csv
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "la")
OUT = os.path.join(ROOT, "data", "normalized", "la.csv")
SERVER = "https://gis.cde.unibe.ch/gis/rest/services/Decide"

CATS = ["lao", "tai_thai", "khmuic", "palaungic", "katuic", "bahnaric_khmer", "vietic",
        "hmong", "mien", "tibeto_burman"]
LAYERS = {f"pct_{c}": f"laos_2015_pcnt_population_ethno_linguistic_category_{c}" for c in CATS}
LAYERS.update({
    "pop": "laos_2015_total_population",
    "cnt_tai_thai": "laos_2015_distribution_of_ethno_linguistic_category_tai_thai",
    "cnt_mon_khmer": "laos_2015_distribution_of_ethno_linguistic_category_mon_khmer",
    "cnt_mien": "laos_2015_distribution_of_ethno_linguistic_category_mien",
    "cnt_sino_tibetan": "laos_2015_distribution_of_ethno_linguistic_category_sino_tibetan",
    "e11_main": "laos_2011_main_ethnicity_in_the_village_code",
    "e11_second": "laos_2011_second_most_numerous_ethnicity_in_the_village_code",
    "e11_third": "laos_2011_third_most_numerous_ethnicity_in_the_village_code",
    "e11_mainpct": "laos_2011_pcnt_of_agricultural_hhs_of_the_main_ethnicity",
})
GEO = {"OBJECTID", "FID", "Shape", "PCODE", "PNAME", "L_PNAME", "DCODE", "DNAME", "L_DNAME",
       "VName", "L_VName", "VCODE", "Shape_Length", "Shape_Area", "Shape_Leng",
       "Shape_Le_1", "Shape_Area_1"}

# Table P2.7 (2015 PHC report, Appendix 1, pdf p.121-122): Lao citizens by ethnicity. Code,
# report spelling, total. "Other and Not Stated" 77,297. Foreigners (45,538) are in P2.8.
P27 = [
    (1, "Lao", 3_427_665), (2, "Tai", 201_576), (3, "Phouthay", 218_108), (4, "Lue", 126_229),
    (5, "Nhoaun", 27_779), (6, "Yang", 5_843), (7, "Xaek", 3_841), (8, "Thaineau", 14_148),
    (9, "Khmou", 708_412), (10, "Pray", 28_732), (11, "Xingmoun", 9_874), (12, "Phong", 30_696),
    (13, "Thaen", 828), (14, "Oedou", 602), (15, "Bid", 2_372), (16, "Lamed", 22_383),
    (17, "Samtao", 3_417), (18, "Katang", 144_255), (19, "Makong", 163_285), (20, "Tri", 37_446),
    (21, "Yrou", 56_411), (22, "Trieng", 38_407), (23, "Ta-oy", 45_991), (24, "Yae", 11_452),
    (25, "Brao", 26_010), (26, "Katu", 28_378), (27, "Harak", 25_430), (28, "Oy", 23_513),
    (29, "Kriang", 16_807), (30, "Cheng", 8_688), (31, "Sadang", 898), (32, "Xuay", 46_592),
    (33, "Nhaheun", 8_976), (34, "Lavy", 1_215), (35, "Pacoh", 22_640), (36, "Khmer", 7_141),
    (37, "Toum", 3_632), (38, "Ngouan", 886), (39, "Moy", 789), (40, "Kree", 1_067),
    (41, "Hmong", 595_028), (42, "Ewmien", 32_400), (43, "Akha", 112_979),
    (44, "Pounoy", 39_192), (45, "Lahou", 19_187), (46, "Syla", 3_151), (47, "Hayi", 741),
    (48, "Lolo", 2_203), (49, "Hor", 12_098),
]
P27_TOTAL = 6_446_690
P27_OTHER = 77_297
NAME = {c: n for c, n, _ in P27}
NATIONAL = {c: v for c, _, v in P27}

# Which K4D category each group belongs to. Set from the 2008 Socio-Economic Atlas's
# classification and confirmed (or corrected) by the national reconciliation in check().
GROUP_CATEGORY = {
    1: "lao",
    2: "tai_thai", 3: "tai_thai", 4: "tai_thai", 5: "tai_thai", 6: "tai_thai", 7: "tai_thai",
    8: "tai_thai",
    9: "khmuic", 10: "khmuic", 11: "khmuic", 12: "khmuic", 13: "khmuic", 14: "khmuic",
    15: "palaungic",
    16: "palaungic", 17: "palaungic",
    18: "katuic", 19: "katuic", 20: "katuic", 21: "bahnaric_khmer", 22: "bahnaric_khmer",
    23: "katuic", 24: "bahnaric_khmer", 25: "bahnaric_khmer", 26: "katuic", 27: "bahnaric_khmer",
    28: "bahnaric_khmer", 29: "katuic", 30: "bahnaric_khmer", 31: "bahnaric_khmer",
    32: "katuic", 33: "bahnaric_khmer", 34: "bahnaric_khmer", 35: "katuic",
    36: "bahnaric_khmer",
    37: "vietic", 38: "vietic", 39: "vietic", 40: "vietic",
    41: "hmong", 42: "mien",
    43: "tibeto_burman", 44: "tibeto_burman", 45: "tibeto_burman", 46: "tibeto_burman",
    47: "tibeto_burman", 48: "tibeto_burman", 49: "tibeto_burman",
}

PAD = 8_000

# Groups the K4D categories put in a second category besides GROUP_CATEGORY's. The national
# reconciliation is the evidence (sources/la.md): K4D's Lao category is 592,207 short of
# P2.7's Lao and its Tai-Thai 588,127 over, so about 590,000 people the census calls Lao
# (Phuan, Yoy, Kaleung and other Tai-speaking subgroups the 49-group list folds into Lao) sit
# in Tai-Thai; Khmuic is 8,656 short and Vietic 8,886 over, and P2.7's Phong (30,696) covers
# both Glottolog's Khmuic Phong-Kniang (phon1246) and the Vietic Phong dialect of Hung
# (phon1243), so Phong is the group that crosses. Bid (Bit, bitt1240) is Palaungic in
# Glottolog and Palaungic is over by 2,375 = Bid's 2,372, so Bid's primary is Palaungic.
ALSO_IN = {1: ["tai_thai"], 12: ["vietic"]}
LABEL_BY_CATEGORY = {12: {"khmuic": "Phong (Khmuic category)",
                          "vietic": "Phong (Vietic category)"}}
OTHER_LABEL = "Other, not stated and foreigners"
SPLIT_STATS = {}


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:  # noqa: BLE001
            if attempt == 3:
                raise
            print(f"  retry after {e}")
            time.sleep(3)


def fetch():
    os.makedirs(RAW, exist_ok=True)
    for key, svc in LAYERS.items():
        dest = os.path.join(RAW, f"{key}.json")
        if os.path.exists(dest) and os.path.getsize(dest) > 100_000:
            continue
        meta = _get(f"{SERVER}/{svc}/MapServer/0?f=json")
        fields = [f["name"] for f in meta.get("fields", []) if f["name"] not in GEO]
        if len(fields) != 1:
            raise SystemExit(f"{svc}: expected one data field, got {fields}")
        field = fields[0]
        # Some of these services refuse resultOffset, so page by province instead.
        rows = []
        for p in range(1, 19):
            q = urllib.parse.urlencode({"where": f"PCODE={p}", "returnGeometry": "false",
                                        "outFields": "VCODE,PCODE,DCODE,VName," + field,
                                        "f": "json"})
            d = _get(f"{SERVER}/{svc}/MapServer/0/query?{q}")
            if "error" in d:
                raise SystemExit(f"{svc}: {d['error']}")
            if d.get("exceededTransferLimit"):
                raise SystemExit(f"{svc} province {p}: transfer limit exceeded")
            rows += [f["attributes"] for f in d["features"]]
            time.sleep(0.1)
        if len(rows) < PAD:
            raise SystemExit(f"{svc}: {len(rows)} rows")
        with open(dest, "w", encoding="utf-8") as fh:
            json.dump({"service": svc, "field": field, "rows": rows}, fh)
        print(f"  {key}: {len(rows):,} villages, field {field}")


def _load(key):
    with open(os.path.join(RAW, f"{key}.json"), encoding="utf-8") as fh:
        d = json.load(fh)
    if d["service"] != LAYERS[key]:
        raise SystemExit(f"{key}.json holds {d['service']}")
    out = {}
    for r in d["rows"]:
        if r["VCODE"] in out:
            raise SystemExit(f"{key}: VCODE {r['VCODE']} twice")
        out[r["VCODE"]] = (r, r[d["field"]])
    return out


def _round_to(target, weights):
    """Integer split of `target` in proportion to `weights` (largest remainder)."""
    tot = sum(weights.values())
    if target == 0 or tot <= 0:
        return {k: 0 for k in weights}
    raw = {k: target * w / tot for k, w in weights.items()}
    base = {k: int(v) for k, v in raw.items()}
    left = target - sum(base.values())
    for k in sorted(raw, key=lambda k: raw[k] - base[k], reverse=True)[:left]:
        base[k] += 1
    return base


def build():
    data = {k: _load(k) for k in LAYERS}
    pop = data["pop"]
    villages = sorted(pop)
    base = set(villages)
    for k in LAYERS:
        if k.startswith("pct_") or k.startswith("cnt_"):
            if set(data[k]) != base:
                raise SystemExit(f"{k}: roster differs from population by "
                                 f"{len(base ^ set(data[k]))} villages")
    print(f"villages {len(villages):,}, population {sum(int(pop[v][1] or 0) for v in villages):,}")

    # --- village category counts from the percentages, with the integer-recovery check ---
    cat = {}
    worst, misses, sum_bad = 0.0, [], []
    for v in villages:
        p = int(pop[v][1] or 0)
        rec = {}
        s = 0.0
        for c in CATS:
            pct = float(data[f"pct_{c}"][v][1] or 0)
            s += pct
            x = pct / 100.0 * p
            err = abs(x - round(x))
            if err > 1e-3:
                misses.append((v, c, pct, p))
            worst = max(worst, err if err <= 1e-3 else 0)
            rec[c] = int(round(x))
        if p and abs(s - 100) > 0.5:
            sum_bad.append((v, round(s, 3), p))
        rec["other"] = p - sum(rec.values())
        cat[v] = rec
    print(f"integer recovery: {len(misses)} misses of {len(villages) * len(CATS):,} cells, "
          f"worst clean error {worst:.2e}")
    for m in misses[:10]:
        print("   miss", m)
    print(f"villages whose ten shares do not sum to 100 +- 0.5: {len(sum_bad)}")
    for m in sum_bad[:10]:
        print("   ", m)
    neg = [v for v in villages if cat[v]["other"] < 0]
    if neg:
        raise SystemExit(f"{len(neg)} villages with categories above population: {neg[:5]}")

    # --- the count services agree with the percentages ---
    fam = {"cnt_tai_thai": ["tai_thai"], "cnt_mien": ["mien"],
           "cnt_mon_khmer": ["khmuic", "palaungic", "katuic", "bahnaric_khmer", "vietic"],
           "cnt_sino_tibetan": ["tibeto_burman"]}
    for k, cs in fam.items():
        bad = [v for v in villages if int(data[k][v][1] or 0) != sum(cat[v][c] for c in cs)]
        tot_c = sum(int(data[k][v][1] or 0) for v in villages)
        print(f"{k}: {tot_c:,}; villages disagreeing with the percentages: {len(bad)}")
        for v in bad[:5]:
            print("   ", v, data[k][v][1], {c: cat[v][c] for c in cs})

    # --- national reconciliation with Table P2.7 ---
    nat = defaultdict(int)
    for v in villages:
        for c, n in cat[v].items():
            nat[c] += n
    want = defaultdict(int)
    for g, c in GROUP_CATEGORY.items():
        want[c] += NATIONAL[g]
    print(f"{'category':<16}{'villages':>12}{'P2.7':>12}{'ratio':>8}")
    for c in CATS + ["other"]:
        w = want.get(c, P27_OTHER if c == "other" else 0)
        print(f"{c:<16}{nat[c]:>12,}{w:>12,}{nat[c] / w if w else 0:>8.4f}")

    return data, villages, cat


def split(data, villages, cat):
    """Each village's category counts split into named groups; see the module docstring.

    Returns {(village, category, group): people}. A group can sit in more than one category
    (ALSO_IN): census `Lao` in K4D's Tai-Thai category, census `Phong` in Vietic.
    """
    allowed = defaultdict(list)                     # category -> groups it can hold
    for g, c in GROUP_CATEGORY.items():
        allowed[c].append(g)
        for c2 in ALSO_IN.get(g, ()):
            allowed[c2].append(g)

    def code(key, v):
        r = data[key].get(v)
        if not r or r[1] in (None, "", 0):
            return None
        return int(r[1])

    # Stage 1: seed weights per (village, category) from the village's 2011 groups.
    seed = {}
    n11 = sum(1 for v in villages if code("e11_main", v))
    print(f"villages with a 2011 main group: {n11:,} of {len(villages):,}")
    SPLIT_STATS["villages_2011"] = n11
    for v in villages:
        m, s2, s3 = code("e11_main", v), code("e11_second", v), code("e11_third", v)
        mp = data["e11_mainpct"].get(v)
        mp = float(mp[1]) / 100 if mp and mp[1] not in (None, "") else 1.0
        rest = max(1.0 - mp, 0.0)
        w = defaultdict(float)
        if m:
            w[m] += max(mp, 0.01)
        if s2:
            w[s2] += max(rest * 2 / 3 if s3 else rest, 0.005)
        if s3:
            w[s3] += max(rest / 3, 0.0025)
        for c in CATS:
            if cat[v][c] <= 0:
                continue
            ww = {g: x for g, x in w.items() if g in allowed[c]}
            seed[(v, c)] = ww or None

    prov = {v: data["pop"][v][0]["PCODE"] for v in villages}
    dist = {v: data["pop"][v][0]["DCODE"] for v in villages}

    # Stage 2: cells with no 2011 group of their category borrow the district's seeded mix
    # for that category, else the province's, else the national totals.
    def mix(keyf):
        acc = defaultdict(lambda: defaultdict(float))
        for (v, c), ww in seed.items():
            if ww:
                t = sum(ww.values())
                for g, x in ww.items():
                    acc[(keyf(v), c)][g] += cat[v][c] * x / t
        return acc
    dmix, pmix = mix(lambda v: dist[v]), mix(lambda v: prov[v])
    filled = defaultdict(int)
    seeded = sum(cat[v][c] for (v, c), ww in seed.items() if ww)
    for (v, c), ww in list(seed.items()):
        if ww:
            continue
        for lvl, src in (("district", dmix.get((dist[v], c))),
                         ("province", pmix.get((prov[v], c)))):
            if src:
                seed[(v, c)] = dict(src)
                filled[lvl] += cat[v][c]
                break
        else:
            seed[(v, c)] = {g: NATIONAL[g] for g in allowed[c]}
            filled["nation"] += cat[v][c]
    print(f"people in a category one of their village's 2011 groups belongs to: {seeded:,}")
    print("people whose split came from a wider mix:", dict(filled))
    SPLIT_STATS.update(seeded=seeded, **filled)

    # Stage 3: rake (iterative proportional fitting) so each group's national total is
    # Table P2.7's, scaled to the village file's total for the pool of categories the group
    # can sit in, while every (village, category) keeps the census's count.
    pools = [["lao", "tai_thai"], ["khmuic", "vietic"]]
    pools += [[c] for c in CATS if not any(c in p for p in pools)]
    out = {}
    for pool in pools:
        gs = sorted({g for c in pool for g in allowed[c]})
        cells = {}
        for (v, c), ww in seed.items():
            if c not in pool:
                continue
            t = sum(ww.values())
            for g in allowed[c]:
                # a small floor lets the rake reach a group no seed named
                cells[(v, c, g)] = cat[v][c] * ((ww.get(g, 0) / t if t else 0) + 1e-6)
        rowtot = {(v, c): cat[v][c] for (v, c) in seed if c in pool}
        scale = sum(rowtot.values()) / sum(NATIONAL[g] for g in gs)
        coltot = {g: NATIONAL[g] * scale for g in gs}
        err = 1.0
        for it in range(1000):
            cs = defaultdict(float)
            for k, x in cells.items():
                cs[k[2]] += x
            for k in cells:
                cells[k] *= coltot[k[2]] / cs[k[2]]
            rs = defaultdict(float)
            for k, x in cells.items():
                rs[k[:2]] += x
            for k in cells:
                cells[k] *= rowtot[k[:2]] / rs[k[:2]]
            cs = defaultdict(float)
            for k, x in cells.items():
                cs[k[2]] += x
            err = max(abs(cs[g] - coltot[g]) / coltot[g] for g in gs)
            if err < 1e-5:
                break
        if err > 1e-3:
            raise SystemExit(f"rake of {pool} did not converge: {err:.2e}")
        per = defaultdict(dict)
        for (v, c, g), x in cells.items():
            per[(v, c)][g] = x
        for (v, c), ww in per.items():
            for g, n in _round_to(cat[v][c], ww).items():
                if n:
                    out[(v, c, g)] = n
        print(f"  raked {'+'.join(pool)}: {len(gs)} groups, village/P2.7 {scale:.4f}, "
              f"{it + 1} iterations, worst column error {err:.1e}")
    return out


def label(c, g):
    return LABEL_BY_CATEGORY.get(g, {}).get(c, NAME[g])


def write(data, villages, cat, out):
    tot_vc = defaultdict(int)
    for (v, c, g), n in out.items():
        tot_vc[(v, c)] += n
    bad = [(v, c) for v in villages for c in CATS if tot_vc.get((v, c), 0) != cat[v][c]]
    if bad:
        raise SystemExit(f"{len(bad)} village categories do not add back, e.g. {bad[:3]}")
    by_v = defaultdict(lambda: defaultdict(int))
    for (v, c, g), n in out.items():
        by_v[v][label(c, g)] += n
    rows = []
    for v in villages:
        a = data["pop"][v][0]
        name = " ".join(str(a.get("VName") or "").split())
        note = f"province={a['PCODE']}; district={a['DCODE']}"
        for lab, n in sorted(by_v[v].items()):
            rows.append([f"LA-{v}", "village", name, lab, n, "derived", note])
        if cat[v]["other"]:
            rows.append([f"LA-{v}", "village", name, OTHER_LABEL, cat[v]["other"],
                         "derived", note + "; village population less the ten categories"])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["geo_id", "geo_level", "geo_name", "source_category", "count", "tier",
                    "note"])
        w.writerows(rows)
    tot = defaultdict(int)
    for r in rows:
        tot[r[3]] += r[4]
    allpop = sum(int(data["pop"][v][1] or 0) for v in villages)
    if sum(tot.values()) != allpop:
        raise SystemExit(f"la.csv sums to {sum(tot.values()):,}, villages {allpop:,}")
    print(f"wrote {OUT}: {len(rows):,} rows, {sum(tot.values()):,} people (= the villages)")
    for lab in sorted(tot, key=lambda k: -tot[k]):
        g = next((g for g in NAME if NAME[g] == lab), None)
        ref = f"{NATIONAL[g]:>10,}{tot[lab] / NATIONAL[g]:>8.4f}" if g else ""
        print(f"   {lab:<34}{tot[lab]:>10,}{ref}")


if __name__ == "__main__":
    if "--fetch" in sys.argv:
        fetch()
    d, vs, c = build()
    if "--explore" in sys.argv:
        sys.exit(0)
    o = split(d, vs, c)
    write(d, vs, c, o)
