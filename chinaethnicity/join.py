"""Match every county row of the 2020 tables to the DataV county polygons it covers.

The tables carry Chinese names and no codes; DataV carries names and GB/T 2260 adcodes. So
this is a name join, done inside the row's own prefecture first, in four tiers:

    1. exact name, within the prefecture the table prints the row under
    2. the name's stem (administrative suffix and ethnic designation removed), same scope
    3. exact name anywhere in the province
    4. stem anywhere in the province

A tier only counts if it finds exactly one unit. Tiers 3-4 are for county-level units the
table prints at top level (Hubei's Xiantao, Jilin's Meihekou), and a match that lands in a
different prefecture from the one the table printed the row under is REFUSED rather than
taken: a same-named county in the wrong prefecture passes every total there is.

Rows that are real units but do not match by name are in OVERRIDES: renames and mergers
between the census (November 2020) and DataV's boundaries (2025), and abbreviations.

## Development zones, and why they go where ASPECT has people the table does not

Development zones, new areas, scenic areas and the like are census rows but not
administrative units. Their land belongs to one or more districts or counties, DataV has no
polygon for them, and the table does not say which counties host them. What does say is the
ASPECT grid (aspect.py), which is built from the same census's township counts and so
includes the zones' people on their real ground. A county hosting a zone therefore holds
more people in ASPECT than its own census row: Kaifeng's Longting district is 156,245 in the
table and 624,408 in ASPECT, and the zone folded out of that prefecture is 418,307.

So within each prefecture, every county's surplus is ASPECT minus 1.05x its own census row
(the 5% absorbs ordinary disagreement between the two), and the prefecture's zone rows are
spread over the counties with a surplus, in proportion to it. The report prints each
prefecture's zone total against its surplus, which is the check that this is finding the
zones and not noise. Without the grid, zones fall back to the prefecture's urban districts.

Then a check that shares nothing with the names: every named row's census population
against ASPECT summed over its polygons. A wrong-twin match shows up as a ratio far from 1.

Writes data/work/leaves_2020.csv (prov, name, parent, how, adcodes, total, <groups>) and
data/work/join_report.txt. `adcodes` is one code, "a;b" (spread over both by population), or
"a:0.62;b:0.38" (explicit shares, from a fold).

Usage:
    python join.py
"""
import os
import re
import sys
from collections import defaultdict

import geopandas as gpd
import numpy as np
import pandas as pd

from common import GEO, GROUPS, KEYS, PROVINCES, WORK

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

IN = os.path.join(WORK, "table_2020.csv")
OUT = os.path.join(WORK, "leaves_2020.csv")
REPORT = os.path.join(WORK, "join_report.txt")
COUNTIES = os.path.join(GEO, "counties.gpkg")
CELLS = os.path.join(GEO, "cells.npz")

SUFFIX = re.compile(r"(自治县|自治旗|自治州|林区|特区|矿区|新区|地区|县|市|区|旗|盟)$")
ETHNIC = sorted([g[2] for g in GROUPS[:56]] + ["各族"], key=len, reverse=True)
SURPLUS_TOLERANCE = 1.05
MIN_SHARE = 0.005

# Hangzhou reorganised its districts in 2021: Xiacheng merged into Gongshu, Jianggan split
# between Shangcheng and the new Qiantang, Linping was carved out of Yuhang, and Qiantang also
# took part of Xiaoshan. The six old rows cannot be divided among the six new polygons, so
# all six spread over the whole affected area by population. The area's total is exact; only
# the composition is averaged across its old districts.
HANGZHOU = ["杭州市/上城区", "杭州市/拱墅区", "杭州市/钱塘区", "杭州市/临平区",
            "杭州市/余杭区", "杭州市/萧山区"]

# (province key, row name as printed, whitespace removed) -> the county polygons its people
# live on, each "name" or "city/name", resolved against DataV within the province, so a typo
# fails instead of pointing somewhere else.
OVERRIDES = {
    # Changbai Mountain protection and development area: three management zones carved out
    # of the counties around the mountain, printed under their own committee.
    ("jilin", "长白山保护开发区池北区"): ["延边朝鲜族自治州/安图县"],
    ("jilin", "长白山保护开发区池西区"): ["白山市/抚松县"],
    ("jilin", "长白山保护开发区池南区"): ["白山市/长白朝鲜族自治县"],
    # the table abbreviates Dorbod Mongol Autonomous County
    ("heilongjiang", "杜蒙自治县"): ["杜尔伯特蒙古族自治县"],
    # Daxing'anling's forest districts have no GB code and no polygon. ASPECT puts Huma
    # county at 73,054 against its own row of 36,362, which Huzhong (16,359) and Xinlin
    # (20,362) account for to within 1%.
    ("heilongjiang", "呼中区"): ["呼玛县"],
    ("heilongjiang", "新林区"): ["呼玛县"],
    ("heilongjiang", "松岭区"): ["加格达奇区"],
    # merged into Chongchuan in 2020, after the census date's boundaries were fixed
    ("jiangsu", "港闸区"): ["崇川区"],
    ("zhejiang", "上城区"): HANGZHOU,
    ("zhejiang", "下城区"): HANGZHOU,
    ("zhejiang", "江干区"): HANGZHOU,
    ("zhejiang", "拱墅区"): HANGZHOU,
    ("zhejiang", "余杭区"): HANGZHOU,
    ("zhejiang", "萧山区"): HANGZHOU,
    # Sanming, 2021: Meilie merged into Sanyuan; Sha county became Shaxian district
    ("fujian", "梅列区"): ["三元区"],
    ("fujian", "沙县"): ["沙县区"],
    # Luoyang, 2021: Jili merged into the new Mengjin district
    ("henan", "吉利区"): ["孟津区"],
    # Heng county became Hengzhou city in 2021
    ("guangxi", "横县"): ["横州市"],
    # Longganhu farm management area lies beside Huangmei county on the lake; ASPECT finds no
    # surplus anywhere in Huanggang to put it, so the surplus fold would default it to the
    # city's one district, 150 km away
    ("hubei", "龙感湖管理区"): ["黄梅县"],
    # Dachaidan administrative committee: DataV's polygon for it is the prefecture's
    # directly administered area
    ("qinghai", "大柴旦行政委员会"): ["海西蒙古族藏族自治州直辖"],
    # Sansha's residents are counted on the Paracels, where the city is seated. Its Nansha
    # district polygon covers the Spratlys, where several states administer features, and
    # is deliberately given nobody. Anita's call, 2026-09-14.
    ("hainan", "三沙市"): ["三沙市/西沙区"],
}

# Polygons expected to carry no census row, and why.
KNOWN_EMPTY = {
    "350527": "Kinmen, administered by Taiwan; outside the census",
    "460302": "Nansha district (Spratly Islands); Sansha's people are placed in Xisha",
}


def stem(name):
    s = SUFFIX.sub("", name) or name
    changed = True
    while changed:
        changed = False
        for e in ETHNIC:
            if s.endswith(e) and len(s) > len(e):
                s = s[:-len(e)]
                changed = True
                break
    return s


def resolve_target(units, target, where):
    if "/" in target:
        city, name = target.split("/", 1)
        hit = units[(units["city"] == city) & (units["name"] == name)]
    else:
        hit = units[units["name"] == target]
    if len(hit) != 1:
        raise SystemExit(f"OVERRIDES {where}: {target!r} matches {len(hit)} units")
    return hit["adcode"].iloc[0]


def codes_of(spec):
    return [p.split(":")[0] for p in spec.split(";")]


def main():
    table = pd.read_csv(IN, dtype={"prov": str})
    units = gpd.read_file(COUNTIES, ignore_geometry=True)
    units["prov2"] = units["prov_code"].str[:2]
    units["stem"] = units["name"].map(stem)

    aspect = {}
    if os.path.exists(CELLS):
        c = np.load(CELLS)
        sums = np.bincount(c["county"].astype(np.int64), weights=c["pop"].astype(np.float64),
                           minlength=len(c["adcodes"]))
        aspect = dict(zip(c["adcodes"].astype(str), sums))
    else:
        print("!! no cells.npz: development zones fall back to urban districts")

    out, report = [], []
    used_overrides = set()
    problems = 0
    for prov in sorted(table["prov"].unique()):
        pkey, pname = PROVINCES[prov][0], PROVINCES[prov][1]
        rows = table[table["prov"] == prov]
        u = units[units["prov2"] == prov]
        cities = {}
        for cc, g in u.groupby("city_code"):
            cities.setdefault(g["city"].iloc[0], g)
        city_by_stem = defaultdict(list)
        for name in cities:
            city_by_stem[stem(name)].append(name)

        report.append(f"\n== {pname}")
        claimed = defaultdict(list)
        pending = []
        start = len(out)
        for r in rows[rows["kind"] == "leaf"].itertuples(index=False):
            name, parent = r.name, (r.parent if isinstance(r.parent, str) else "")
            key = (pkey, name)
            scope = None
            if parent:
                if parent in cities:
                    scope = cities[parent]
                elif len(city_by_stem.get(stem(parent), [])) == 1:
                    scope = cities[city_by_stem[stem(parent)][0]]

            codes, how = None, None
            if key in OVERRIDES:
                codes = [resolve_target(u, t, key) for t in OVERRIDES[key]]
                how = "override"
                used_overrides.add(key)
            else:
                tiers = []
                if scope is not None:
                    tiers += [("prefecture", scope, "name", name),
                              ("prefecture-stem", scope, "stem", stem(name))]
                tiers += [("province", u, "name", name), ("province-stem", u, "stem", stem(name))]
                for label, pool, col, want in tiers:
                    hit = pool[pool[col] == want]
                    if len(hit) == 1:
                        if (scope is not None and label.startswith("province")
                                and hit["city_code"].iloc[0] != scope["city_code"].iloc[0]):
                            report.append(f"  !! {parent}/{name}: {label} match "
                                          f"{hit['city'].iloc[0]}/{hit['name'].iloc[0]} is in "
                                          f"another prefecture; refused")
                            continue
                        codes, how = [hit["adcode"].iloc[0]], label
                        break
            values = [int(r.total)] + [int(getattr(r, k)) for k in KEYS]
            if codes is None and scope is not None:
                pending.append((parent, name, scope, values))
                continue
            if codes is None:
                report.append(f"  !! UNMATCHED {parent}/{name} ({int(r.total):,})")
                problems += 1
                continue
            if how != "override":
                claimed[codes[0]].append(name)
            out.append([prov, name, parent, how, ";".join(codes)] + values)

        # ---- development zones: spread over the prefecture's ASPECT surplus
        census_in = defaultdict(float)
        for row in out[start:]:
            cs = codes_of(row[4])
            pops = np.array([aspect.get(c, 0.0) for c in cs])
            shares = pops / pops.sum() if pops.sum() > 0 else np.full(len(cs), 1 / len(cs))
            for c, s in zip(cs, shares):
                census_in[c] += row[5] * s
        by_city = defaultdict(list)
        for item in pending:
            by_city[item[2]["city_code"].iloc[0]].append(item)
        for city_code, items in by_city.items():
            scope = items[0][2]
            zone_total = sum(it[3][0] for it in items)
            surplus = {a: max(0.0, aspect.get(a, 0.0) - SURPLUS_TOLERANCE * census_in.get(a, 0.0))
                       for a in scope["adcode"]}
            total_surplus = sum(surplus.values())
            if total_surplus > 0:
                weights = {a: v / total_surplus for a, v in surplus.items()}
                how = "fold-surplus"
            else:
                pool = scope[scope["name"].str.endswith("区")]
                pool = pool if len(pool) else scope
                pops = {a: aspect.get(a, 1.0) for a in pool["adcode"]}
                weights = {a: v / sum(pops.values()) for a, v in pops.items()}
                how = "fold-districts"
            weights = {a: w for a, w in weights.items() if w >= MIN_SHARE}
            norm_ = sum(weights.values())
            weights = {a: w / norm_ for a, w in sorted(weights.items(), key=lambda kv: -kv[1])}
            spec = ";".join(f"{a}:{w:.4f}" for a, w in weights.items())
            name_of = dict(zip(scope["adcode"], scope["name"]))
            top = ", ".join(f"{name_of[a]} {w:.0%}" for a, w in list(weights.items())[:4])
            report.append(f"  zones {scope['city'].iloc[0]}: {len(items)} rows, "
                          f"{zone_total:,} people; ASPECT surplus {total_surplus:,.0f} "
                          f"({total_surplus / max(zone_total, 1):.2f}x) -> {top}")
            for parent, name, _, values in items:
                report.append(f"        {name} ({values[0]:,})")
                out.append([prov, name, parent, how, spec] + values)

        for code, names in claimed.items():
            if len(names) > 1:
                report.append(f"  !! {len(names)} rows matched one polygon {code}: {names}")
                problems += 1
        covered = {c for row in out[start:] for c in codes_of(row[4])}
        for e in u[~u["adcode"].isin(covered)].itertuples(index=False):
            if e.adcode in KNOWN_EMPTY:
                report.append(f"  empty by design: {e.city}/{e.name}: {KNOWN_EMPTY[e.adcode]}")
                continue
            ap = f", ASPECT {aspect.get(e.adcode, 0):,.0f}" if aspect else ""
            report.append(f"  !! polygon with no census row: {e.city}/{e.name} {e.adcode}{ap}")
            problems += 1

    for k in sorted(set(OVERRIDES) - used_overrides):
        report.append(f"  unused override {k}")

    cols = ["prov", "name", "parent", "how", "adcodes", "total"] + KEYS
    df = pd.DataFrame(out, columns=cols)

    if aspect:
        df["aspect"] = df["adcodes"].map(lambda s: sum(aspect.get(c, 0.0) for c in codes_of(s)))
        named = df[~df["how"].str.startswith("fold")].copy()
        named["ratio"] = named["total"] / named["aspect"].clip(lower=1)
        odd = named[(named["ratio"] > 1.5) | (named["ratio"] < 1 / 1.5)]
        report.append(f"\n== census rows against ASPECT, named matches only: median ratio "
                      f"{named['ratio'].median():.3f}; {len(odd)} outside 1.5x "
                      f"(a county hosting a development zone reads low, by design)")
        for r in odd.sort_values("ratio").itertuples(index=False):
            report.append(f"  {PROVINCES[r.prov][1]:14s} {r.parent}/{r.name} ({r.how}) census "
                          f"{r.total:,} ASPECT {r.aspect:,.0f} ratio {r.ratio:.2f}")
        df = df.drop(columns="aspect")

    os.makedirs(WORK, exist_ok=True)
    df.to_csv(OUT, index=False, encoding="utf-8")
    with open(REPORT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(report) + "\n")

    print(f"{len(df):,} county rows: {df['how'].value_counts().to_dict()}")
    print(f"people: {df['total'].sum():,} "
          f"(tables: {table.loc[table['kind'] != 'agg', 'total'].sum():,})")
    print("\n".join(line for line in report if "!!" in line or "unused" in line))
    print(f"\nwrote {OUT} and {REPORT}")
    if problems:
        print(f"\n{problems} problems listed above")


if __name__ == "__main__":
    main()
