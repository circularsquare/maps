"""Estimate the provinces with no open 2020 county table: the 2000 county pattern, scaled to
2020 totals. Anita's call, 2026-09-14; NOTES.md §7 is the record.

For every province parse.py has no county table for:

  * the pattern is the 2000 census county table (A0106 of the 2000 Population Census Data
    Assembly, religiondots' copy from Harvard's `chinacensus` dataverse), matched to today's
    county polygons by religiondots' resolver (religiondots/sources/cn.md §5);
  * the totals are the province's own 2020 table 1-4 where it is published by prefecture
    (Hebei, Liaoning, Hunan, Sichuan), and otherwise the national yearbook's row for the
    province;
  * every group is scaled on its own inside each prefecture (or province):
        county_2020[g] = county_2000[g] x total_2020[g] / total_2000[g]
    so each prefecture's 2020 total is exact for every group, and only the split between
    the counties inside it is 2000's.

A group with nobody in a unit in 2000 and people there in 2020 has no pattern to scale, so it
is spread over the unit's counties by their 2020 population (ASPECT). The report lists each.

Counties are not also fitted to modern county populations. religiondots considered that and
rejected it (cn.md §6): the city growth since 2000 in Xinjiang and Tibet was mostly Han, so
fitting to it inflates minority counts in exactly the cities where they are most contested.
The report prints how far each county's drawn total is from ASPECT instead.

Districts carved out since 2000 have no 2000 row. CARVED lists the counties each came from;
those rows are spread over the new polygon as well, with its population divided between its
parents in proportion to theirs.

Writes data/work/leaves_fallback.csv (join.py's leaves_2020.csv columns; counts are not whole
numbers) and data/work/fallback_report.txt.

Usage:
    python fallback.py
"""
import importlib.util
import os
import sys
from collections import defaultdict

import geopandas as gpd
import numpy as np
import pandas as pd

import parse
from common import GEO, KEYS, NCAT, PROVINCES, RELIGIONDOTS, WORK
from fetch import PREFECTURE_SOURCES
from join import SURPLUS_TOLERANCE

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MEASURED = os.path.join(WORK, "table_2020.csv")
OUT = os.path.join(WORK, "leaves_fallback.csv")
REPORT = os.path.join(WORK, "fallback_report.txt")
COUNTIES = os.path.join(GEO, "counties.gpkg")
CELLS = os.path.join(GEO, "cells.npz")
RD_CN = os.path.join(RELIGIONDOTS, "sources", "cn.py")
RD_INDEX = os.path.join(RELIGIONDOTS, "data", "raw", "cn", "datav", "county_index.json")

# a prefecture row whose printed name is not DataV's
PREFECTURE_ALIASES = {("hunan", "湘西州"): "湘西土家族苗族自治州"}

# A zone printed as its own prefecture-level row. Its people are added to the prefectures it
# sits in, in proportion to each one's ASPECT surplus over its own row (join.py's method).
ZONE_HOSTS = {("liaoning", "辽宁省沈抚新区管委会"): ["沈阳市", "抚顺市"]}

# People a ① row leaves out beyond the rows printed under it, and the counties they live in.
# Hebei's Baoding ① also excludes Xiong'an New Area, which has no row: Baoding minus ① minus
# Dingzhou is 1,205,440 people, no column negative, and ASPECT holds 1,191,135 in these three.
REMAINDERS = {("hebei", "保定市①"): ("雄安新区", ["雄县", "容城县", "安新县"])}

# County polygons with no 2000 row, because they were carved out of older counties later:
# child adcode -> the counties it came from. Each parent must border its child (checked).
CARVED = {
    "341504": ["341522"],                      # Yeji district, 2015, from Huoqiu county
    "340506": ["340521"],                      # Bowang district, 2012, from Dangtu county
    "360113": ["360112"],                      # Honggutan district, 2019, from Xinjian
    "360482": ["360426", "360425", "360483"],  # Gongqingcheng, 2010, from De'an, Yongxiu, Xingzi
    "440309": ["440306"],                      # Longhua district, 2016, from Bao'an
    "440311": ["440306"],                      # Guangming district, 2018, from Bao'an
    "440310": ["440307"],                      # Pingshan district, 2016, from Longgang
    "440115": ["440113"],                      # Nansha district, 2005, from Panyu
    "440514": ["440513"],                      # Chaonan district, 2003, from Chaoyang
    "440404": ["440403"],                      # Jinwan district, 2001, from Doumen
    "510904": ["510903"],                      # Anju district, 2003, from Chuanshan
    "511903": ["511902"],                      # Enyang district, 2013, from Bazhou
    "511603": ["511602"],                      # Qianfeng district, 2013, from Guang'an
    "520303": ["520302"],                      # Huichuan district, 2003, from Honghuagang
    "520115": ["520181", "520113", "520103"],  # Guanshanhu, 2012: the districts it borders most
    "540630": ["540629", "540625"],            # Shuanghu county, 2012, from Nyima and Xainza
    "652702": ["652701", "652722"],            # Alashankou, 2012: Bole and Jinghe, which it borders
    # Bingtuan cities, whose regiments were counted in the counties around them in 2000;
    # parents are the counties each borders most
    "659002": ["652901", "652928", "652924"],  # Aral
    "659003": ["653130"],                      # Tumxuk, inside Bachu
    "659004": ["652301", "650109"],            # Wujiaqu
    "654004": ["654023"],                      # Horgos, 2014, from Huocheng county
    "659005": ["654323", "654301"],            # Beitun
    "659008": ["654022", "654023"],            # Kokdala
    "659006": ["652801"],                      # Tiemenguan, inside Korla
    "659007": ["652701", "652722"],            # Shuanghe
    "659009": ["653223", "653222", "653225"],  # Kunyu
    "659010": ["654202", "654003", "654223"],  # Huyanghe
}

FACTOR_REPORT = (0.5, 2.0)


def load_religiondots_cn():
    spec = importlib.util.spec_from_file_location("religiondots_cn", RD_CN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def prefecture_units(pkey, fn, nat_row, u, aspect, where, report):
    """A 2020 table 1-4 that stops at prefectures, as disjoint scaling units covering the
    province: [[printed name, set of adcodes, 59-column float vector]]. The checks are
    parse.py's: total = male + female in every row and column, the rows add up to the
    province, and the province equals the national table."""
    lines, _ = parse.lines_from_sheet(parse.sheet_rows("xls", fn), where)
    for l in lines:
        if not np.array_equal(l["v"], l["m"] + l["f"]):
            raise SystemExit(f"{where} {l['name']}: total != male + female")
    vmax = max(l["v"][0] for l in lines)
    first = next(i for i, l in enumerate(lines) if l["v"][0] == vmax)
    total = lines[first]["v"]
    rest = lines[first + 1:]
    while rest and np.array_equal(rest[0]["v"], total):
        rest = rest[1:]
    if not np.array_equal(total, nat_row):
        raise SystemExit(f"{where}: province total does not match the national table")

    # Hebei prints Shijiazhuang and Baoding twice: whole, and marked ① without the
    # county-level city printed under it (Xinji, Dingzhou). The parts are used, once they are
    # shown to add up to the whole in every column. What a ① row leaves out beyond the rows
    # under it must be named in REMAINDERS.
    whole, extra = set(), []
    for i, l in enumerate(rest):
        if not l["name"].endswith("①"):
            continue
        j = i + 1
        while j < len(rest) and rest[j]["indent"] > l["indent"]:
            j += 1
        w = [k for k, m in enumerate(rest) if m["name"] == l["name"][:-1]]
        if len(w) != 1:
            raise SystemExit(f"{where} {l['name']}: no single row named {l['name'][:-1]}")
        left = rest[w[0]]["v"] - sum((m["v"] for m in rest[i:j]), np.zeros(NCAT, dtype=np.int64))
        if left.any():
            if (pkey, l["name"]) not in REMAINDERS or (left < 0).any():
                raise SystemExit(f"{where} {l['name']}: it and the rows under it are "
                                 f"{left[0]:,} short of {l['name'][:-1]}")
            name, county_names = REMAINDERS[(pkey, l["name"])]
            extra.append(dict(name=name, v=left, counties=county_names))
        whole.add(w[0])
    rows = [l for k, l in enumerate(rest) if k not in whole] + extra
    s = sum((l["v"] for l in rows), np.zeros(NCAT, dtype=np.int64))
    if not np.array_equal(s, total):
        raise SystemExit(f"{where}: rows add to {s[0]:,}, province total {total[0]:,}")

    cities = {n: set(g["adcode"]) for n, g in u.groupby("city")}
    units, county_units, zones = [], set(), []
    for l in rows:
        if (pkey, l["name"]) in ZONE_HOSTS:
            zones.append(l)
            continue
        nm = l["name"].rstrip("①")
        nm = PREFECTURE_ALIASES.get((pkey, nm), nm)
        if nm in cities and "counties" not in l:
            units.append([l["name"], set(cities[nm]), l["v"].astype(np.float64)])
            continue
        codes = set()
        for cname in l.get("counties", [nm]):
            hit = u[u["name"] == cname]
            if len(hit) != 1:
                raise SystemExit(f"{where} {l['name']}: no prefecture or county named {cname}")
            codes.add(hit["adcode"].iloc[0])
        units.append([l["name"], codes, l["v"].astype(np.float64)])
        county_units.add(len(units) - 1)
        if "counties" in l:
            held = sum(aspect.get(a, 0.0) for a in codes)
            report.append(f"  {l['name']} (left out of its ① row, {l['v'][0]:,}): ASPECT holds "
                          f"{held:,.0f} in {', '.join(l['counties'])}")
    for k, unit in enumerate(units):
        if k not in county_units:
            for c in county_units:
                unit[1] -= units[c][1]

    for z in zones:
        hosts = [x for x in units if x[0] in ZONE_HOSTS[(pkey, z["name"])]]
        if len(hosts) != len(ZONE_HOSTS[(pkey, z["name"])]):
            raise SystemExit(f"{where} {z['name']}: host prefectures not found")
        held = np.array([sum(aspect.get(a, 0.0) for a in h[1]) for h in hosts])
        surplus = np.maximum(0.0, held - SURPLUS_TOLERANCE * np.array([h[2][0] for h in hosts]))
        w = surplus / surplus.sum() if surplus.sum() > 0 else held / held.sum()
        for h, wi in zip(hosts, w):
            h[2] = h[2] + z["v"] * wi
        report.append(f"  zone {z['name']} ({z['v'][0]:,}): ASPECT surplus "
                      f"{surplus.sum():,.0f} ({surplus.sum() / z['v'][0]:.2f}x) -> "
                      + ", ".join(f"{h[0]} {wi:.0%}" for h, wi in zip(hosts, w)))

    claimed = defaultdict(list)
    for unit in units:
        for a in unit[1]:
            claimed[a].append(unit[0])
    doubled = {a: n for a, n in claimed.items() if len(n) > 1}
    missing = set(u["adcode"]) - set(claimed)
    if doubled or missing:
        raise SystemExit(f"{where}: prefecture rows do not partition the polygons "
                         f"(twice {sorted(doubled)[:5]}, none {sorted(missing)[:5]})")
    return units


def main():
    table = pd.read_csv(MEASURED, dtype={"prov": str})
    measured = set(table["prov"])
    todo = [c for c in sorted(PROVINCES) if c not in measured]
    nat = parse.national_table()

    cn = load_religiondots_cn()
    by_prov = cn.read_2000()
    resolve, _ = cn.build_resolver(RD_INDEX)

    units = gpd.read_file(COUNTIES)
    units["prov2"] = units["prov_code"].str[:2]
    geom = dict(zip(units["adcode"], units.geometry))
    c = np.load(CELLS)
    sums = np.bincount(c["county"].astype(np.int64), weights=c["pop"].astype(np.float64),
                       minlength=len(c["adcodes"]))
    aspect = dict(zip(c["adcodes"].astype(str), sums))

    for child, parents in CARVED.items():
        for a in [child] + parents:
            if a not in geom:
                raise SystemExit(f"CARVED: {a} is not a county polygon")
        for p in parents:
            if not geom[child].buffer(0.01).intersects(geom[p]):
                raise SystemExit(f"CARVED: {p} does not border {child}")
    children_of = defaultdict(list)
    for child, parents in CARVED.items():
        for p in parents:
            children_of[p].append(child)
    parent_pop = {ch: sum(aspect.get(p, 0.0) for p in ps) for ch, ps in CARVED.items()}

    out, report = [], []
    problems = 0
    report.append(f"measured provinces (table_2020.csv): {len(measured)}; estimated here: "
                  f"{len(todo)}")
    for code in todo:
        pkey, pname, pcn = PROVINCES[code]
        u = units[units["prov2"] == code]
        rows = by_prov[int(code) * 10000]
        names = [n for n, _ in rows]
        res = resolve(int(code) * 10000, names)
        unresolved = [names[i] for i, (a, _) in enumerate(res) if a is None]
        if unresolved:
            raise SystemExit(f"{pname}: 2000 rows that do not resolve: {unresolved[:8]}")
        own = [a for a, _ in res]
        X = np.array([v for _, v in rows], dtype=np.float64)[:, 1:]

        # each row's placement: its own polygon, plus its part of any polygon carved from it
        n_rows_on = defaultdict(int)
        for a in own:
            n_rows_on[a] += 1
        specs, pop20 = [], np.zeros(len(rows))
        for i, a in enumerate(own):
            parts = {a: aspect.get(a, 0.0)}
            for ch in children_of.get(a, []):
                parts[ch] = (aspect.get(ch, 0.0) * aspect.get(a, 0.0) / parent_pop[ch]
                             if parent_pop[ch] > 0 else 0.0)
            s = sum(parts.values())
            specs.append(";".join(f"{k}:{(v / s if s > 0 else 1 / len(parts)):.6f}"
                                  for k, v in parts.items()))
            pop20[i] = s / n_rows_on[a]
        covered = set(own) | {ch for a in own for ch in children_of.get(a, [])}
        missing = sorted(set(u["adcode"]) - covered)
        if missing:
            raise SystemExit(f"{pname}: polygons no 2000 row reaches (add them to CARVED): "
                             + ", ".join(f"{a} {u.loc[u['adcode'] == a, 'name'].iloc[0]}"
                                         for a in missing))

        report.append(f"\n== {pname}")
        if pkey in PREFECTURE_SOURCES:
            how = "scaled-prefecture"
            sunits = prefecture_units(pkey, PREFECTURE_SOURCES[pkey][1], nat[code], u, aspect,
                                      pname, report)
        else:
            how = "scaled-province"
            sunits = [[pcn, set(u["adcode"]), nat[code].astype(np.float64)]]
        unit_of = np.full(len(rows), -1)
        for k, (_, codes, _) in enumerate(sunits):
            for i, a in enumerate(own):
                if a in codes:
                    unit_of[i] = k
        for i, a in enumerate(own):
            for ch in children_of.get(a, []):
                if ch not in sunits[unit_of[i]][1]:
                    raise SystemExit(f"{pname}: {ch} is carved from {a} but sits in another "
                                     f"prefecture row")

        est = np.zeros_like(X)
        zero_base, odd = [], []
        for k, (uname, codes, T) in enumerate(sunits):
            R = np.nonzero(unit_of == k)[0]
            if not len(R):
                raise SystemExit(f"{pname} {uname}: no 2000 row falls inside it")
            base = X[R].sum(axis=0)
            for g in range(len(KEYS)):
                want = T[1 + g]
                if want == 0:
                    continue
                if base[g] > 0:
                    est[R, g] = X[R, g] * want / base[g]
                    f = want / base[g]
                    if want > 5000 and not FACTOR_REPORT[0] <= f <= FACTOR_REPORT[1]:
                        odd.append((want, uname, KEYS[g], base[g], f))
                else:
                    est[R, g] = want * pop20[R] / pop20[R].sum()
                    zero_base.append((want, uname, KEYS[g]))
            got = est[R].sum(axis=0)
            if not np.allclose(got, T[1:], rtol=1e-9, atol=1e-6):
                report.append(f"  !! {uname}: scaled rows do not add up to its 2020 totals")
                problems += 1
        if not np.allclose(est.sum(axis=0), nat[code][1:], rtol=1e-9, atol=1e-3):
            report.append(f"  !! {pname}: does not add up to the national table")
            problems += 1

        total = est.sum(axis=1)
        report.append(f"  {len(rows)} 2000 rows ({X.sum():,.0f} people) scaled to "
                      f"{len(sunits)} {'prefecture rows' if how == 'scaled-prefecture' else 'province total'} "
                      f"({total.sum():,.0f}); {how}")
        carved_here = sorted({ch for a in own for ch in children_of.get(a, [])})
        if carved_here:
            report.append(f"  carved polygons placed from their parents: "
                          + ", ".join(u.loc[u["adcode"] == a, "name"].iloc[0]
                                      for a in carved_here))
        if zero_base:
            report.append(f"  no 2000 pattern, spread by 2020 population: "
                          f"{len(zero_base)} (unit, group) pairs, "
                          f"{sum(z[0] for z in zero_base):,.0f} people; largest: "
                          + ", ".join(f"{g} in {n} {w:,.0f}"
                                      for w, n, g in sorted(zero_base, reverse=True)[:5]))
        for want, uname, g, base, f in sorted(odd, reverse=True)[:8]:
            report.append(f"  factor {f:5.2f}  {g:12s} {uname}: 2000 {base:,.0f} -> 2020 "
                          f"{want:,.0f}")

        # drawn county totals against ASPECT, which the scaling does not fit to
        drawn = defaultdict(float)
        for spec, t in zip(specs, total):
            for part in spec.split(";"):
                a, w = part.split(":")
                drawn[a] += t * float(w)
        gap = sorted(((drawn[a] - aspect.get(a, 0.0), a) for a in u["adcode"]),
                     key=lambda x: -abs(x[0]))
        ratio = np.array([drawn[a] / max(aspect.get(a, 0.0), 1.0) for a in u["adcode"]])
        name_of = dict(zip(u["adcode"], u["city"] + "/" + u["name"]))
        report.append(f"  drawn county totals / ASPECT: median {np.median(ratio):.2f}, "
                      f"{(np.abs(np.log(ratio)) > np.log(1.5)).sum()} of {len(ratio)} "
                      f"outside 1.5x; largest gaps: "
                      + ", ".join(f"{name_of[a]} {d:+,.0f}" for d, a in gap[:4]))

        for i in range(len(rows)):
            out.append([code, names[i], sunits[unit_of[i]][0], how, specs[i],
                        round(float(total[i]), 3)] + [round(float(v), 3) for v in est[i]])

    df = pd.DataFrame(out, columns=["prov", "name", "parent", "how", "adcodes", "total"] + KEYS)
    os.makedirs(WORK, exist_ok=True)
    df.to_csv(OUT, index=False, encoding="utf-8")
    with open(REPORT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(report) + "\n")
    print("\n".join(report))
    print(f"\nwrote {OUT} ({len(df):,} rows, {df['total'].sum():,.0f} people) and {REPORT}")
    if problems:
        raise SystemExit(f"{problems} problems listed above")


if __name__ == "__main__":
    main()
