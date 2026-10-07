"""Kazakhstan population by region (level 1) and rayon / city (level 2).

Sources (all Bureau of National Statistics, stat.gov.kz, open):
  * 2026, 2025: the annual bulletin "Численность населения по полу и по типу
    местности" on 1 January, table 2 "...в разрезе областей, городов и
    районов". Register-based current estimates rolled forward from the 2021
    census. Downloaded to data/kazakhstan/raw/ if missing.
  * 2021: National Population Census 2021, settlement-level table 3.1 of
    "Численность населения Республики Казахстан по этносам, населенным пунктам
    и возрасту" (published 2025), read from religiondots' copy (read-only).
  * KATO 18.09.2026, the administrative-territorial classifier, which gives
    every current unit its code and lists every settlement under it.

Everything is put on the 1 Jan 2026 geography (228 KATO units, 224 shipped
once Shymkent's five districts are merged, common.MERGES):
  * 2021: each census settlement is carried to the current unit that lists it
    in KATO 2026 (by code where the code still sits under a successor, else by
    name among the successors). Only rayons that were split since 2021 need
    this; the rest pass whole. Settlements that no longer exist in KATO and
    were absorbed by a city go to that city (ABSORBED).
  * 2025: 227 units, identical to 2026 except two changes made during 2025.
    Taraz took villages from Zhambyl and Baizak districts: the moved share of
    each district's 2025 figure is the share of its 2021 census population
    living in the settlements Taraz absorbed. Astana's Almaty district was
    split in two: its 2025 figure is divided in the 2026 proportions.

Writes data/kazakhstan/population.csv (code, level, year, pop) and
data/kazakhstan/xwalk_2021.csv (census settlement -> current unit).
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import csv
import sys
from collections import defaultdict

import openpyxl
import pandas as pd
import requests

from common import (DATA, KATO_URL, KATO_XLSX, MERGES, RAW, REGIONS, REPO,
                    BULLETIN_REGIONS, load_kato, norm, norm_settlement, shipped_code)

UA = "Mozilla/5.0"
BULLETINS = {
    2026: (RAW / "kz_pop_sex_locality_2026.xlsx",
           "https://stat.gov.kz/api/iblock/element/341322/file/ru/"),
    2025: (RAW / "kz_pop_sex_locality_2025.xls",
           "https://stat.gov.kz/api/iblock/element/330829/file/ru/"),
}
CENSUS_XLSX = REPO / "religiondots" / "data" / "raw" / "kz" / "kz2021_ethnos_settlement.xlsx"
OUT = DATA / "population.csv"
XWALK_OUT = DATA / "xwalk_2021.csv"

# Published national totals, checked on every run.
NATIONAL = {2021: 19_186_015, 2025: 20_283_399, 2026: 20_499_822}

# Bulletin and census spellings that differ from KATO 2026 (after norm()).
BULLETIN_ALIASES = {
    "саркандскии": "сарканскии",          # 2025 bulletin: Саркандский район
    "чиилиискии": "шиелиискии",           # 2025 bulletin: Чиилийский (now Шиелийский)
}

# Census 2021 rayon -> current units carved out of it since. The first
# successor (found by code or name) is implicit; these are the extra ones.
SPLITS = {
    "632800000": ["104100000"],  # Semey city admin. -> + Zhanasemey district (2022)
    "636400000": ["104500000"],  # Urdzhar -> + Makanshy (2022)
    "635800000": ["103400000"],  # Tarbagatay -> + Aksuat (2022)
    "635000000": ["635600000"],  # Kokpekty -> + Samar (2022)
    "635200000": ["635500000"],  # Kurchum -> + Markakol (2022)
    "635400000": ["636300000"],  # Katon-Karagay -> + Ulken Naryn (2022)
    "196800000": ["191800000"],  # Ili -> + Alatau city (2024)
    "314000000": ["311000000"],  # Zhambyl district -> part to Taraz (2025)
    "313600000": ["311000000"],  # Baizak -> part to Taraz (2025)
}
# Census settlements that vanished from KATO because a city swallowed them.
ABSORBED = {"196800000": "191800000", "314000000": "311000000", "313600000": "311000000"}
# Census rayons whose first successor the name/code match cannot find.
FIRST_SUCCESSOR = {
    "191600000": "191000000",  # Kapchagay city admin. -> renamed Konaev (2022)
    "435200000": "435200000",  # Chiili -> Shieli (spelling)
    "613800000": "613800000",  # Zhetisai (spelling)
}
# Region recodes: census region -> current regions its rayons can be in.
REGION_SPLITS = {"19": ["19", "33"], "35": ["35", "62"], "63": ["63", "10"]}
# Shipped units whose 2021 census figure is on a different boundary and has no
# settlement table to rebuild it (city districts): 2021 is left out for them.
NO_2021 = {"711110000", "711210000", "711510000"}  # Astana: Almaty(+Saraishyk), Esil, Nura
# 2025 units that lost territory during 2025: (loser, gainer).
TRANSFERS_2025 = [("314000000", "311000000"), ("313600000", "311000000")]
# Astana's Almaty district was split on 29 Jan 2025, after the 1 Jan 2025
# count: Almaty district (north) + Saraishyk (south).
SPLIT_2025_BY_2026 = {"711110000": ["711110000", "711610000"]}


def download(path, url):
    if path.exists():
        return
    print(f"downloading {url}")
    r = requests.get(url, headers={"User-Agent": UA}, timeout=120)
    r.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(r.content)


def read_bulletin(path, kato):
    """Table 2 -> {kato_l2: pop}, every row matched to a KATO unit 1:1."""
    df = pd.read_excel(path, sheet_name="2", header=None)
    l2 = kato[kato.level == 2]
    out, region = {}, None
    for name, tot in zip(df[0], df[1]):
        if pd.isna(name) or not isinstance(tot, (int, float)) or pd.isna(tot):
            region = None if pd.isna(name) or not isinstance(tot, (int, float)) else region
            continue
        key = norm(name)
        key = BULLETIN_ALIASES.get(key, key)
        if key.startswith("республика"):
            continue
        if region is None:
            region = BULLETIN_REGIONS[key]
            continue
        cand = l2[(l2.ab == region) & (l2.rus_name.map(norm) == key)]
        if len(cand) != 1:
            sys.exit(f"{path.name}: {name!r} in region {region} matched {len(cand)} KATO units")
        code = cand.te.iloc[0]
        if code in out:
            sys.exit(f"{path.name}: {code} matched twice")
        out[code] = int(tot)
    return out


def read_region_totals(path):
    """Table 1 -> {region code: pop}, the bulletin's own region totals."""
    df = pd.read_excel(path, sheet_name="1", header=None)
    out = {}
    for name, tot in zip(df[0], df[1]):
        if pd.isna(name) or not isinstance(tot, (int, float)) or pd.isna(tot):
            continue
        key = norm(name)
        if key in BULLETIN_REGIONS:
            out[BULLETIN_REGIONS[key]] = int(tot)
    return out


CENSUS_REGIONS = {}  # census region code -> 2021 total, filled by read_census()


def read_census():
    """Census table 3.1 -> (rayon rows, settlement rows)."""
    wb = openpyxl.load_workbook(CENSUS_XLSX, read_only=True)
    rayons, settles, okrugs, cur = [], [], {}, None
    for row in wb["3.1"].iter_rows(min_row=5, values_only=True):
        lv, code, name, pop = row[:4]
        if lv is None or code is None:
            continue
        code = str(code).strip().zfill(9)
        if lv == 1:
            CENSUS_REGIONS[code[:2]] = int(pop)
        elif lv == 2:
            cur = code
            rayons.append((code, str(name).strip(), int(pop)))
        elif lv == 3:
            okrugs[code] = str(name).strip()
        elif lv == 4:
            settles.append((cur, code, str(name).strip(), int(pop)))
    return rayons, settles, okrugs


def census_to_current(kato):
    """2021 census population on 2026 KATO level-2 units, plus the crosswalk."""
    rayons, settles, okrug_name = read_census()
    l2 = kato[kato.level == 2]
    below = kato[kato.level == 3]
    code_l2 = dict(zip(below.te, below.l2))
    kato_name = dict(zip(kato.te, kato.rus_name))
    first = {}
    for code, name, _ in rayons:
        if code in FIRST_SUCCESSOR:
            first[code] = FIRST_SUCCESSOR[code]
            continue
        regions = REGION_SPLITS.get(code[:2], [code[:2]])
        if code[:2] not in REGION_SPLITS and code in set(l2.te):
            first[code] = code
            continue
        key = BULLETIN_ALIASES.get(norm(name), norm(name))
        cand = l2[l2.ab.isin(regions) & (l2.rus_name.map(norm) == key)]
        if len(cand) != 1:
            sys.exit(f"census rayon {code} {name}: {len(cand)} current matches")
        first[code] = cand.te.iloc[0]
    if len(set(first.values())) != len(first):
        sys.exit("two census rayons share a first successor")

    by_rayon = defaultdict(list)
    for p, code, name, pop in settles:
        by_rayon[p].append((code, name, pop))
    result = defaultdict(int)
    xwalk = []
    for code, name, pop in rayons:
        succ = [first[code]] + SPLITS.get(code, [])
        if len(succ) == 1:
            result[succ[0]] += pop
            xwalk.append((code, name, "", "(whole rayon)", pop, succ[0], "whole"))
            continue
        pool = below[below.l2.isin(succ)]
        by_name = defaultdict(set)
        for n, t in zip(pool.rus_name.map(norm_settlement), pool.te):
            by_name[n].add(t)
        got = 0
        for scode, sname, spop in by_rayon[code]:
            hits = by_name.get(norm_settlement(sname), set())
            if len(hits) > 1:
                # Twins: prefer the one whose rural okrug has the same name as
                # the census settlement's okrug, then the same code tail.
                okrug = norm(okrug_name.get(scode[:6] + "000", ""))
                same_okrug = {t for t in hits if norm(kato_name.get(t[:6] + "000", "")) == okrug}
                same_tail = {t for t in hits if t[4:] == scode[4:]}
                hits = same_okrug if len(same_okrug) == 1 else same_tail if len(same_tail) == 1 else hits
            # Name first: codes get reassigned inside a rayon when part of it is
            # split off (Katon-Karagay's 635430100 was Ulken Naryn village in
            # 2021 and is Katon-Karagay village now). The code only decides
            # for a settlement renamed since, whose old name is nowhere.
            if len(hits) == 1:
                to, how = code_l2[next(iter(hits))], "name"
            elif not hits and code_l2.get(scode) in succ:
                to, how = code_l2[scode], "code"
            elif hits:
                to, how = first[code], "name-ambiguous"
            elif code in ABSORBED:
                to, how = ABSORBED[code], "absorbed"
            else:
                to, how = first[code], "gone"
            result[to] += spop
            got += spop
            xwalk.append((code, name, scode, sname, spop, to, how))
        if got != pop:
            sys.exit(f"census rayon {code}: settlements sum {got} != {pop}")
    return result, xwalk, rayons


def main():
    download(KATO_XLSX, KATO_URL)
    for path, url in BULLETINS.values():
        download(path, url)
    kato = load_kato()
    l2_codes = set(kato[kato.level == 2].te)

    b26 = read_bulletin(BULLETINS[2026][0], kato)
    b25 = read_bulletin(BULLETINS[2025][0], kato)
    print(f"bulletin 2026: {len(b26)} units, {sum(b26.values()):,}")
    print(f"bulletin 2025: {len(b25)} units, {sum(b25.values()):,}")
    if set(b26) != l2_codes:
        sys.exit(f"2026 units != KATO level 2: {sorted(set(b26) ^ l2_codes)}")
    print(f"  2025 lacks: {sorted(l2_codes - set(b25))}")

    c21, xwalk, rayons = census_to_current(kato)
    print(f"census 2021: {len(rayons)} rayons, {sum(c21.values()):,} on {len(c21)} current units")
    with XWALK_OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["census_rayon", "census_rayon_name", "settlement", "settlement_name",
                    "pop2021", "to_kato", "how"])
        w.writerows(xwalk)

    # 2025 onto 2026 boundaries.
    b25 = dict(b25)
    census_rayon_pop = {c: p for c, _, p in rayons}
    for loser, gainer in TRANSFERS_2025:
        moved21 = sum(r[4] for r in xwalk if r[0] == loser and r[5] == gainer)
        frac = moved21 / census_rayon_pop[loser]
        moved = round(b25[loser] * frac)
        print(f"  2025 transfer {loser} -> {gainer}: {frac:.3f} of {b25[loser]:,} = {moved:,}")
        b25[loser] -= moved
        b25[gainer] += moved
    # Units that did not exist on 1 Jan 2025: their parent's 2025 figure is
    # split in the two units' 1 Jan 2026 proportions.
    for parent, parts in SPLIT_2025_BY_2026.items():
        whole = b25.pop(parent)
        tot26 = sum(b26[p] for p in parts)
        shares = [round(whole * b26[p] / tot26) for p in parts[:-1]]
        shares.append(whole - sum(shares))
        for p, v in zip(parts, shares):
            b25[p] = v
        print(f"  2025 {parent} ({whole:,}) split by 2026 shares: "
              + ", ".join(f"{p} {v:,}" for p, v in zip(parts, shares)))

    for y in (2025, 2026):
        official = read_region_totals(BULLETINS[y][0])
        ours = defaultdict(int)
        for code, pop in (b25 if y == 2025 else b26).items():
            ours[code[:2]] += pop
        bad = {ab: (ours[ab], official.get(ab)) for ab in REGIONS if ours[ab] != official.get(ab)}
        print(f"  {y}: {len(official)} region totals in table 1, "
              f"{20 - len(bad)} equal to the sum of their rayons")
        if bad:
            sys.exit(f"region mismatch {y}: {bad}")

    # 2021: the 20 current regions folded back to the census's 17.
    back = {"10": "63", "33": "19", "62": "35"}
    ours = defaultdict(int)
    for code, pop in c21.items():
        ours[back.get(code[:2], code[:2])] += pop
    bad = {ab: (ours[ab], v) for ab, v in CENSUS_REGIONS.items() if ours[ab] != v}
    print(f"  2021: {len(CENSUS_REGIONS)} census regions, {len(CENSUS_REGIONS) - len(bad)} "
          f"equal to the sum of their current rayons")
    if bad:
        sys.exit(f"census region mismatch: {bad}")

    years = {2021: c21, 2025: b25, 2026: b26}
    for y, d in years.items():
        tot = sum(d.values())
        flag = "ok" if tot == NATIONAL[y] else f"!= published {NATIONAL[y]:,}"
        print(f"  national {y}: {tot:,} {flag}")
        if tot != NATIONAL[y]:
            sys.exit("national total mismatch")

    rows = []
    for y, d in years.items():
        ship = defaultdict(int)
        region = defaultdict(int)
        for code, pop in d.items():
            ship[shipped_code(code)] += pop
            region[code[:2]] += pop
        for code, pop in ship.items():
            if y == 2021 and code in NO_2021:
                continue
            rows.append((code, 2, y, pop))
        for ab, pop in region.items():
            rows.append((f"{ab}0000000", 1, y, pop))
    rows.sort(key=lambda r: (r[1], r[0], r[2]))
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["code", "level", "year", "pop"])
        w.writerows(rows)
    n2 = len({r[0] for r in rows if r[1] == 2})
    print(f"wrote {len(rows)} rows, {n2} level-2 units, 20 regions -> {OUT}")


if __name__ == "__main__":
    main()
