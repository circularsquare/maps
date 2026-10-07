"""Nepal populations for helper1m: province, district, palika (local level).

Writes helper1m/data/nepal/population.csv  (columns: code, level, year, pop)

  level 3 = palika (753 local levels), code = COD-AB adm3_pcode
  level 2 = district (77),            code = COD-AB adm2_pcode
  level 1 = province (7),             code = COD-AB adm1_pcode

Years:
  2021  National Population and Housing Census 2021, final count (29,164,578),
        from NSO's Religion_NPHC_2021.xlsx, sheet "Prov_District_local level".
        The district INSTITUTIONAL rows (239,098 people) are spread over each
        district's palikas pro rata, so every district equals its census total.
  2026, 2031  NSO's official projection from that census, medium scenario
        ("Population Projections for Nepal 2021-2051", Annex 5), which NSO
        publishes down to the palika and the ward. Read from the API behind
        censusresults.nsonepal.gov.np/population-projection.

Districts and provinces are the exact sums of their palikas in every year. For
the projection years that is also checked against NSO's own district,
province and national projection rows.

Neither NSO source carries a geographic code. The palika list behind the
projection page comes in the census workbook's own order with the census's own
labels, so those two are paired by position and the labels asserted equal. The
census is paired to COD-AB by name inside each district (the religiondots rule,
sources/np_geo.py): strip the unit-type word, strip a leading district name,
then a unique one-character fallback (it fires once: Melanchi / Melamchi).
COD's 22 park polygons (unit-type digit 5) are dropped before the join.

Usage (from anywhere):
    C:\\Python39\\python.exe helper1m\\scripts\\nepal\\fetch.py
Downloads are cached under helper1m/data/nepal/raw/; a rerun is offline.
"""
from __future__ import annotations

import csv
import json
import re
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HELPER = Path(__file__).resolve().parents[2]
REPO = HELPER.parent
RAW = HELPER / "data" / "nepal" / "raw"
PROJ_DIR = RAW / "projection"
OUT = HELPER / "data" / "nepal" / "population.csv"
ADM3_SHP = REPO / "data" / "asia1m" / "nepal" / "npl_admin3.shp"

UA = {"User-Agent": "Mozilla/5.0"}
SITE = "https://censusresults.nsonepal.gov.np"
CENSUS_URL = SITE + "/files/caste/Religion_NPHC_2021.xlsx"
CENSUS_XLSX = RAW / "Religion_NPHC_2021.xlsx"
API = "https://censusapi.cbs.gov.np/api/v1/population-projection/population-table"
LISTS_JSON = RAW / "nso_unit_lists.json"

CENSUS_YEAR = 2021
PROJ_YEARS = [2026, 2031]
NATIONAL_2021 = 29_164_578
NATIONAL_PROJ = {2021: 29_356_136, 2026: 30_034_040, 2031: 30_603_859}  # Annex 5
EXPECTED = {"province": 7, "district": 77, "local": 753, "institutional": 77}
EXPECTED_TYPES = {"1": 6, "2": 11, "3": 276, "4": 460}  # metro/sub-metro/municipality/rural
PARK_TYPE = "5"

SHEET = "Prov_District_local level"
C_NEPAL, C_PROV, C_DIST, C_LOCAL, C_SEX, C_TOTAL = 0, 1, 2, 3, 4, 6

SUFFIX = re.compile(
    r"[\s\-]*(sub[\s\-]*metropolit[a-z]*[\s\-]*city|metropolit[a-z]*[\s\-]*city|"
    r"upa[\s\-]*mahanagarpalika|mahanagarpalika|nagarpalika|gaunpalika|gaupalika|"
    r"rural[\s\-]*municipality|municipality)\s*$", re.I)


# --------------------------------------------------------------------------- download

_SESSION = None


def get(url, **kw):
    global _SESSION
    if _SESSION is None:
        import requests
        _SESSION = requests.Session()
        _SESSION.headers.update(UA)
    for attempt in range(4):
        try:
            r = _SESSION.get(url, timeout=120, **kw)
            r.raise_for_status()
            return r
        except Exception as e:  # noqa: BLE001 - retry anything, then give up loudly
            if attempt == 3:
                raise
            print(f"    retry {url} ({e})")
            time.sleep(3 * (attempt + 1))


def fetch_census():
    if CENSUS_XLSX.exists() and CENSUS_XLSX.stat().st_size > 100_000:
        return
    print("GET", CENSUS_URL)
    r = get(CENSUS_URL)
    if r.content[:2] != b"PK":
        raise SystemExit(f"{CENSUS_URL} did not return an xlsx")
    CENSUS_XLSX.write_bytes(r.content)


def fetch_lists():
    """The palika/district code lists the projection page's dropdowns use.

    They are not served by the API; they are a literal inside one of the
    Next.js chunks of /population-projection. Find that chunk from the build
    manifest rather than hard-coding its hashed name, which changes per deploy.
    """
    if LISTS_JSON.exists():
        return json.loads(LISTS_JSON.read_text(encoding="utf-8"))
    html = get(SITE + "/population-projection").text
    build = re.search(r'"buildId":"([^"]+)"', html).group(1)
    manifest = get(f"{SITE}/_next/static/{build}/_buildManifest.js").text
    chunks = sorted(set(re.findall(r'static/chunks/[\w\-/\[\]]+\.js', manifest)))
    dists = munis = None
    for c in chunks:
        js = get(f"{SITE}/_next/{c}").text
        if "no_of_wards" not in js:
            continue
        d = re.findall(r'\{label:"([^"]+)",value:"(\d+)",province:"(\d+)"\}', js)
        m = re.findall(
            r'\{district:"(\d+)",value:"(\d+)",label:"([^"]*)",no_of_wards:(\d+)\}', js)
        if len(d) == 77 and len(m) == 753:
            dists, munis = d, m
            print("  code lists found in", c)
            break
    if dists is None:
        raise SystemExit("no chunk carries the 77-district / 753-palika lists any more")
    lists = {
        "districts": [{"key": k, "district": int(v), "province": int(p)}
                      for k, v, p in dists],
        "palikas": [{"district": int(d), "palika": int(v), "label": lab, "wards": int(w)}
                    for d, v, lab, w in munis],
    }
    if len(lists["districts"]) != 77 or len(lists["palikas"]) != 753:
        raise SystemExit(f"code lists: {len(lists['districts'])} districts, "
                         f"{len(lists['palikas'])} palikas (want 77, 753)")
    LISTS_JSON.write_text(json.dumps(lists, ensure_ascii=False, indent=0), encoding="utf-8")
    return lists


def projection(params, name):
    """{year: total} for one unit, cached as raw/projection/<name>.json."""
    path = PROJ_DIR / f"{name}.json"
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):       # missing, or cut short by an interrupted run
        r = get(API, params=params)
        body = r.json()
        if not body.get("success"):
            raise SystemExit(f"API refused {params}: {body}")
        rows = body["data"]
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(rows), encoding="utf-8")
        tmp.replace(path)
        time.sleep(0.1)
    if len(rows) != 31:
        raise SystemExit(f"{name}: {len(rows)} years, expected 2021-2051")
    return {int(row["year"]): int(row["total"]) for row in rows}


# --------------------------------------------------------------------------- census

def _label(v):
    return str(v).strip() if v not in (None, "") else None


def read_census():
    """Return (units, districts, provinces) from the census workbook.

    units: list of dicts (prov_n, dist_n, local_n, district, name, pop) in sheet
    order; districts: {(prov_n, dist_n): {name, pop, inst}}; provinces:
    {prov_n: {name, pop}}. Parent rows are kept only to check the nesting.
    """
    import openpyxl
    wb = openpyxl.load_workbook(CENSUS_XLSX, read_only=True, data_only=True)
    grid = list(wb[SHEET].iter_rows(values_only=True))
    wb.close()
    if _label(grid[2][C_TOTAL]) != "Total Population":
        raise SystemExit(f"total column header moved: {grid[2][C_TOTAL]!r}")

    units, districts, provinces = [], {}, {}
    national = None
    prov_n = dist_n = local_n = 0
    cur = None
    for r in grid[4:]:
        if _label(r[C_NEPAL]) == "NEPAL":
            cur = ("nepal",)
            continue
        if _label(r[C_PROV]):
            prov_n += 1
            dist_n = 0
            provinces[prov_n] = {"name": _label(r[C_PROV])}
            cur = ("prov", prov_n)
            continue
        if _label(r[C_DIST]):
            dist_n += 1
            local_n = 0
            districts[(prov_n, dist_n)] = {"name": _label(r[C_DIST]), "inst": 0}
            cur = ("dist", prov_n, dist_n)
            continue
        if _label(r[C_LOCAL]):
            lab = _label(r[C_LOCAL])
            if lab.upper() == "INSTITUTIONAL":
                cur = ("inst", prov_n, dist_n)
            else:
                local_n += 1
                units.append({"prov_n": prov_n, "dist_n": dist_n, "local_n": local_n,
                              "district": districts[(prov_n, dist_n)]["name"],
                              "name": lab, "pop": None})
                cur = ("local", len(units) - 1)
            continue
        if _label(r[C_SEX]) != "Total":
            continue
        v = r[C_TOTAL]
        if not isinstance(v, (int, float)) or v != int(v) or v < 0:
            raise SystemExit(f"{cur}: {v!r} is not a count")
        v = int(v)
        kind = cur[0]
        if kind == "nepal":
            national = v
        elif kind == "prov":
            provinces[cur[1]]["pop"] = v
        elif kind == "dist":
            districts[(cur[1], cur[2])]["pop"] = v
        elif kind == "inst":
            districts[(cur[1], cur[2])]["inst"] = v
        else:
            units[cur[1]]["pop"] = v

    # nesting checks: palikas + institutional = district, districts = province
    ok = (len(provinces), len(districts), len(units)) == (7, 77, 753) and national == NATIONAL_2021
    for key, d in districts.items():
        s = sum(u["pop"] for u in units if (u["prov_n"], u["dist_n"]) == key) + d["inst"]
        if s != d["pop"]:
            ok = False
            print(f"  BAD district {d['name']}: palikas+inst {s:,} vs {d['pop']:,}")
    for p, pv in provinces.items():
        s = sum(d["pop"] for k, d in districts.items() if k[0] == p)
        if s != pv["pop"]:
            ok = False
            print(f"  BAD province {pv['name']}: districts {s:,} vs {pv['pop']:,}")
    if not ok:
        raise SystemExit("census workbook failed its nesting checks")
    inst = sum(d["inst"] for d in districts.values())
    print(f"census 2021: {len(units)} palikas, {len(districts)} districts, "
          f"{len(provinces)} provinces, national {national:,}; nesting OK")
    print(f"  institutional rows: {inst:,} ({inst / national:.2%}), spread pro rata "
          "within each district")
    return units, districts, provinces


def largest_remainder(weights, total):
    """Split integer `total` in proportion to `weights`, summing exactly."""
    base = sum(weights)
    shares = [total * w / base for w in weights]
    add = [int(s) for s in shares]
    order = sorted(range(len(weights)), key=lambda i: shares[i] - add[i], reverse=True)
    for i in order[:total - sum(add)]:
        add[i] += 1
    return add


def spread_institutional(units, districts):
    """Add each district's institutional row to its palikas by population share,
    so the district total is exact."""
    by_d = defaultdict(list)
    for u in units:
        by_d[(u["prov_n"], u["dist_n"])].append(u)
    for key, us in by_d.items():
        add = largest_remainder([u["pop"] for u in us], districts[key]["inst"])
        for u, a in zip(us, add):
            u["pop21"] = u["pop"] + a
        assert sum(u["pop21"] for u in us) == districts[key]["pop"]


# --------------------------------------------------------------------------- COD join

def _strip_types(s):
    prev = None
    while prev != s:
        prev = s
        s = SUFFIX.sub("", s)
    return s


def fold(name, district=None):
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = _strip_types(s)
    if district:
        d = _strip_types(unicodedata.normalize("NFKD", str(district))).strip()
        if d and re.match(rf"^{re.escape(d)}\s+\S", s, re.I):
            s = s[len(d):]
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def _edit1(a, b):
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) == 1
    short, long_ = (a, b) if len(a) < len(b) else (b, a)
    return any(long_[:i] + long_[i + 1:] == short for i in range(len(long_)))


def join_cod(units):
    """Attach adm3/adm2/adm1 p-codes to each census palika."""
    import pyogrio
    g = pyogrio.read_dataframe(ADM3_SHP, read_geometry=False)
    g["unit_type"] = g["adm3_pcode"].str[6]
    got = g["unit_type"].value_counts().to_dict()
    if {t: got.get(t, 0) for t in EXPECTED_TYPES} != EXPECTED_TYPES:
        raise SystemExit(f"COD unit-type digits {got} no longer give 6/11/276/460")
    parks = g[g["unit_type"] == PARK_TYPE]
    g = g[g["unit_type"] != PARK_TYPE]
    print(f"COD adm3: dropped {len(parks)} park polygons, {len(g)} palikas left")

    cod = defaultdict(dict)
    for nm, dn, p3, p2, p1 in zip(g["adm3_name"], g["adm2_name"], g["adm3_pcode"],
                                  g["adm2_pcode"], g["adm1_pcode"]):
        k = fold(nm, dn)
        if k in cod[fold(dn)]:
            raise SystemExit(f"COD fold collision inside {dn}: {nm}")
        cod[fold(dn)][k] = (nm, p3, p2, p1)

    used = defaultdict(set)
    near, missing = [], []
    for u in units:
        d, k = fold(u["district"]), fold(u["name"], u["district"])
        if d not in cod:
            raise SystemExit(f"district {u['district']!r} has no COD polygons")
        hit = cod[d].get(k)
        if hit and k not in used[d]:
            used[d].add(k)
            u["cod"] = hit
        else:
            missing.append(u)
    for u in missing:
        d, k = fold(u["district"]), fold(u["name"], u["district"])
        cands = [kk for kk in cod[d] if kk not in used[d] and _edit1(kk, k)]
        if len(cands) != 1:
            raise SystemExit(f"cannot place {u['name']!r} in {u['district']!r}: {cands}")
        used[d].add(cands[0])
        u["cod"] = cod[d][cands[0]]
        near.append(f"{u['name']} -> {u['cod'][0]} ({u['district']})")
    spare = [(d, k) for d in cod for k in cod[d] if k not in used[d]]
    if spare:
        raise SystemExit(f"COD palikas with no census row: {spare}")
    # independent check: a district's palikas all land in one COD district
    for key in {(u["prov_n"], u["dist_n"]) for u in units}:
        p2s = {u["cod"][2] for u in units if (u["prov_n"], u["dist_n"]) == key}
        if len(p2s) != 1:
            raise SystemExit(f"census district {key} spans COD districts {p2s}")
    print(f"  joined {len(units)}/753 census palikas to COD, 0 unmatched either way; "
          f"one-character fallback used for: {'; '.join(near) or 'none'}")


# --------------------------------------------------------------------------- main

def main():
    RAW.mkdir(parents=True, exist_ok=True)
    PROJ_DIR.mkdir(parents=True, exist_ok=True)
    fetch_census()
    units, districts, provinces = read_census()
    spread_institutional(units, districts)
    join_cod(units)

    lists = fetch_lists()
    # NSO's projection lists come in the census workbook's order with its labels:
    # pair by position, then assert the labels agree (they do, all 753).
    dist_seq = sorted(districts)               # (prov_n, dist_n) in sheet order
    if [d["district"] for d in lists["districts"]] != list(range(1, 78)):
        raise SystemExit("NSO district list is not numbered 1..77 in order")
    for i, d in enumerate(lists["districts"]):
        cen = districts[dist_seq[i]]["name"]
        if re.sub(r"[^a-z]", "", d["key"]) != re.sub(r"[^a-z]", "", cen.lower()):
            raise SystemExit(f"district {i + 1}: NSO list {d['key']!r} vs census {cen!r}")
        if d["province"] != dist_seq[i][0]:
            raise SystemExit(f"district {cen}: province {d['province']} vs {dist_seq[i][0]}")
    for u, p in zip(units, lists["palikas"]):
        dn = dist_seq.index((u["prov_n"], u["dist_n"])) + 1
        if (p["district"], p["palika"], p["label"].strip()) != (dn, u["local_n"], u["name"]):
            raise SystemExit(f"palika list out of step: {p} vs census {u}")
        u["nso_d"], u["nso_p"] = dn, u["prov_n"]
    print("NSO projection code lists: 77 districts and 753 palikas, labels identical "
          "to the census, in the same order")

    # projections: every palika, plus every district, province and Nepal as checks
    print("projection API (cached in raw/projection/) ...")
    for n, u in enumerate(units, 1):
        u["proj"] = projection({"province": u["nso_p"], "district": u["nso_d"],
                                "municipality": u["local_n"]},
                               f"m_{u['nso_d']:02d}_{u['local_n']:02d}")
        if n % 100 == 0:
            print(f"  {n}/753")
    dproj = {}
    for i, key in enumerate(dist_seq, 1):
        dproj[key] = projection({"province": key[0], "district": i}, f"d_{i:02d}")
    pproj = {p: projection({"province": p}, f"p_{p}") for p in provinces}
    nproj = projection({}, "nepal")

    # The projection keeps the institutional population the way the census does:
    # in the district rows but in no palika. Each district row minus its palikas
    # is that district's projected institutional population (2021: 238,818 against
    # the census's 239,098). Spread it the same way as the census year, so
    # palikas sum to NSO's district rows exactly.
    years_all = sorted(nproj)
    bad = 0
    inst_proj = defaultdict(int)
    for key in dist_seq:
        us = [u for u in units if (u["prov_n"], u["dist_n"]) == key]
        for u in us:
            u["proj_full"] = {}
        for y in years_all:
            pal = [u["proj"][y] for u in us]
            gap = dproj[key][y] - sum(pal)
            inst21 = districts[key]["inst"]
            if gap < 0 or (y == 2021 and abs(gap - inst21) > 0.05 * inst21 + 30):
                bad += 1
                print(f"  BAD {districts[key]['name']} {y}: district row "
                      f"{dproj[key][y]:,} minus palikas {sum(pal):,} = {gap:,}, "
                      f"census institutional {inst21:,}")
                continue
            inst_proj[y] += gap
            for u, v in zip(us, largest_remainder(pal, gap)):
                u["proj_full"][y] = u["proj"][y] + v
    for p in provinces:
        for y in years_all:
            s = sum(dproj[k][y] for k in dist_seq if k[0] == p)
            if s != pproj[p][y]:
                bad += 1
                print(f"  BAD province {p} {y}: districts {s:,} vs {pproj[p][y]:,}")
    for y in years_all:
        s = sum(pproj[p][y] for p in provinces)
        if s != nproj[y]:
            bad += 1
            print(f"  BAD {y}: provinces {s:,} vs national {nproj[y]:,}")
    for y, want in NATIONAL_PROJ.items():
        if nproj[y] != want:
            bad += 1
            print(f"  BAD national {y}: API {nproj[y]:,} vs report Annex 5 {want:,}")
    if bad:
        raise SystemExit(f"{bad} projection hierarchy mismatches")
    print(f"  districts sum to the 7 province rows and the national row in every year "
          f"{years_all[0]}-{years_all[-1]}; national matches the report's Annex 5")
    print("  district rows minus their palikas (projected institutional population): "
          + ", ".join(f"{y} {inst_proj[y]:,}" for y in (2021, 2026, 2031))
          + "; spread pro rata like the census year")

    # ---- write
    rows = []
    agg = defaultdict(int)
    for u in units:
        _, p3, p2, p1 = u["cod"]
        vals = {CENSUS_YEAR: u["pop21"], **{y: u["proj_full"][y] for y in PROJ_YEARS}}
        for y, v in vals.items():
            rows.append((p3, 3, y, v))
            agg[(p2, 2, y)] += v
            agg[(p1, 1, y)] += v
    rows += [(c, lv, y, v) for (c, lv, y), v in agg.items()]
    rows.sort(key=lambda r: (r[1], r[0], r[2]))

    # the 2021 district/province sums must be the census's own rows
    p2_of = {(u["prov_n"], u["dist_n"]): u["cod"][2] for u in units}
    p1_of = {u["prov_n"]: u["cod"][3] for u in units}
    for key, d in districts.items():
        assert agg[(p2_of[key], 2, CENSUS_YEAR)] == d["pop"], d["name"]
    for p, pv in provinces.items():
        assert agg[(p1_of[p], 1, CENSUS_YEAR)] == pv["pop"], pv["name"]
    # and the projection-year sums must be NSO's own district/province rows
    for y in PROJ_YEARS:
        for key in dist_seq:
            assert agg[(p2_of[key], 2, y)] == dproj[key][y], (key, y)
        for p in provinces:
            assert agg[(p1_of[p], 1, y)] == pproj[p][y], (p, y)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["code", "level", "year", "pop"])
        w.writerows(rows)
    for y in [CENSUS_YEAR] + PROJ_YEARS:
        tot = sum(v for (c, lv, yy), v in agg.items() if lv == 1 and yy == y)
        print(f"  {y}: national {tot:,}")
    print(f"wrote {OUT} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
