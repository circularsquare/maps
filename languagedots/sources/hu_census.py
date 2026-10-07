"""Hungary: KSH, Népszámlálás 2022, mother tongue, by settlement (and Budapest's 23 districts).

    python sources/hu_census.py --fetch    download the structure and the two data cubes
    python sources/hu_census.py            normalise data/raw/hu/ -> data/normalized/hu.csv

THE SOURCE. KSH's census database (https://nepszamlalas2022.ksh.hu/adatbazis/) is an SDMX
backend behind a JavaScript client; religiondots/sources/hu.md §2 found its routes in app.js:

    GET /api/version                         -> {"version": "V67"}
    GET /api/structure/{flow}/{version}      -> dimensions and codelists
    GET /api/dataflows/{flow}/{version}/d/DIM:code+code,DIM,...   -> the data (a bare DIM = all)

religiondots could not pull WBS003 whole (all 149 variables, >60 MB, truncated). Naming the
mother-tongue codes with `/d/` makes it a ~1 MB request that comes back in a second, so here
the whole of Hungary is fetched by script; nothing is hand-exported.

  WBS003  "Population data by settlement": TEL_SZ_ADAT's MT block, every TERUL_GEO5 code
          (settlement, járás, vármegye, region, country). MT is the total population.
  WBS009  "Population by ethnic attributes, county and type of settlement": the same MT block
          per vármegye, with TOTAL. The second table this census offers; its counties must
          equal WBS003's settlements summed by vármegye.

THE QUESTION. Anyanyelv, mother tongue, optional, up to two answers. The MT_<lang> cells count
everyone who named the language, alone or as one of two, so they add to more than the people
who answered. MT_GI (Romani or Boyash) and MT_D (any of the 13 recognised minority languages)
are PERSON counts, not sums: MT_GI < MT_GIR + MT_GIB by the people who named both, and MT_D
< the 14 minority mentions by the people who named two minority languages. How the two-answer
people are shared out is in `split()` and sources/hu.md §2.
"""
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "hu"
OUT = ROOT / "data" / "normalized" / "hu.csv"

API = "https://nepszamlalas2022.ksh.hu"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"}

# the 14 minority-language labels (Romani and Boyash apart; MT_GI is their person-union)
MINORITY = ["MT_BU", "MT_GIR", "MT_GIB", "MT_GR", "MT_CR", "MT_PO", "MT_GE", "MT_AR", "MT_RO",
            "MT_RU", "MT_SE", "MT_SK", "MT_SL", "MT_UK"]
MT_CODES = ["MT", "MT_HU", "MT_D", "MT_GI"] + MINORITY + ["MT_O", "MT_NA"]
DRAWN = ["MT_HU"] + MINORITY + ["MT_O"]

FILES = {"WBS003": "hu_WBS003_mt.json", "WBS009": "hu_WBS009_mt.json"}
STRUCT = {"WBS003": "hu_structure_WBS003.json", "WBS009": "hu_structure_WBS009.json"}

NATIONAL = 9_603_634          # KSH, resident population, census 2022 (religiondots' hu.py too)


def _get(url, timeout=600):
    import requests
    r = requests.get(url, headers=UA, timeout=timeout)
    r.raise_for_status()
    return r


def fetch():
    RAW.mkdir(parents=True, exist_ok=True)
    ver = _get(f"{API}/api/version", 60).json()["version"]
    print(f"census database version {ver}")
    for flow, name in STRUCT.items():
        doc = _get(f"{API}/api/structure/{flow}/{ver}").json()
        if not doc.get("data", {}).get("codelists"):
            raise SystemExit(f"{flow}: structure message carries no codelists")
        (RAW / name).write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
        print(f"  {name}: {(RAW / name).stat().st_size:,} bytes")
    mt = "+".join(MT_CODES)
    urls = {
        "WBS003": f"{API}/api/dataflows/WBS003/{ver}/d/TIME_PERIOD:2022,TERUL_GEO5,TEL_SZ_ADAT:{mt}",
        "WBS009": (f"{API}/api/dataflows/WBS009/{ver}/d/TIME_PERIOD:2022,TERUL_GEO3,"
                   f"TERUL_TELTIP2:HU,NEMZ:TOTAL+{mt},TARSJELL2:NEME_SEX"),
    }
    for flow, url in urls.items():
        r = _get(url)
        rows = r.json()
        if not isinstance(rows, list) or not rows or "OBS_VALUE" not in rows[0]:
            raise SystemExit(f"{flow}: not a data response: {r.text[:300]}")
        (RAW / FILES[flow]).write_text(json.dumps({"url": url, "version": ver, "rows": rows},
                                                  ensure_ascii=False), encoding="utf-8")
        print(f"  {FILES[flow]}: {len(rows):,} cells")


# --------------------------------------------------------------------------- read
def _load(flow):
    p = RAW / FILES[flow]
    if not p.exists():
        raise SystemExit(f"missing {p} -- run with --fetch")
    return json.loads(p.read_text(encoding="utf-8"))["rows"]


def _codelists(flow):
    p = RAW / STRUCT[flow]
    if not p.exists():
        raise SystemExit(f"missing {p} -- run with --fetch")
    return {cl["id"]: {c["id"]: (c.get("names", {}).get("hu", ""), c.get("names", {}).get("en", ""),
                                 c.get("parent")) for c in cl.get("codes", [])}
            for cl in json.loads(p.read_text(encoding="utf-8"))["data"]["codelists"]}


def _value(r, where):
    """(count or None). OBS_STATUS Q is a suppressed cell; every other status is unexpected."""
    if r["OBS_STATUS"] == "Q":
        if r["OBS_VALUE"] is not None:
            raise SystemExit(f"{where}: suppressed cell carries a value")
        return None
    if r["OBS_STATUS"] not in (None, ""):
        raise SystemExit(f"{where}: unknown OBS_STATUS {r['OBS_STATUS']!r}")
    s = str(r["OBS_VALUE"]).strip()
    if not s.isdigit():
        raise SystemExit(f"{where}: non-numeric {r['OBS_VALUE']!r}")
    return int(s)


def read():
    cl = _codelists("WBS003")
    labels = cl["CL_TEL_SZ_ADAT"]
    geo = cl["CL_TERUL_GEO5"]
    for c in MT_CODES:
        if c not in labels:
            raise SystemExit(f"WBS003: code {c} not in CL_TEL_SZ_ADAT -- KSH renamed it")
    val = {}
    for r in _load("WBS003"):
        if r["TIME_PERIOD"] != "2022":
            raise SystemExit(f"unexpected period {r['TIME_PERIOD']}")
        g, c = r["TERUL_GEO5"], r["TEL_SZ_ADAT"]
        val[(g, c)] = _value(r, f"WBS003 {g}/{c}")
    geos = {g for g, _ in val}
    settle = sorted(g for g in geos if g.isdigit() and len(g) == 5)
    jaras = sorted(g for g in geos if g.isdigit() and len(g) == 3)
    county = sorted(g for g in geos if len(g) == 5 and g.startswith("HU"))
    for g in settle:
        for c in MT_CODES:
            if (g, c) not in val:
                raise SystemExit(f"WBS003: settlement {g} has no {c} cell")
    w9 = {}
    for r in _load("WBS009"):
        if r["TERUL_TELTIP2"] != "HU" or r["TARSJELL2"] != "NEME_SEX":
            raise SystemExit("WBS009: a cell outside the totals that were asked for")
        w9[(r["TERUL_GEO3"], r["NEMZ"])] = _value(r, f"WBS009 {r['TERUL_GEO3']}/{r['NEMZ']}")
    return val, labels, geo, settle, jaras, county, w9


# --------------------------------------------------------------------------- suppression
def fill(val, geo, settle, jaras, county):
    """Estimate KSH's suppressed settlement cells (OBS_STATUS Q) from the járás and vármegye.

    Every printed settlement cell is 0 or at least 3, so a suppressed cell is 1 or 2. A
    settlement's járás (Budapest's districts are their own) is printed for most cells: its
    figure less the printed settlements is shared evenly among the suppressed ones. Where the
    járás cell is suppressed too, the vármegye's remainder is shared the same way. Each estimate
    is clipped to [1, 2]. Returns ({(settlement, code): estimate}, stats)."""
    kids = defaultdict(list)
    for s in settle:
        kids[geo[s][2]].append(s)
    jkids = defaultdict(list)
    for j in jaras:
        jkids[geo[j][2]].append(j)
    est, clipped = {}, 0
    for c in MT_CODES:
        for cty in county:
            pending = []
            for j in jkids[cty]:
                sup = [s for s in kids[j] if val[(s, c)] is None]
                if not sup:
                    continue
                jt = val.get((j, c))
                if jt is None:
                    pending += sup
                    continue
                each = (jt - sum(val[(s, c)] or 0 for s in kids[j])) / len(sup)
                for s in sup:
                    est[(s, c)] = each
            if pending:
                known = sum((val[(s, c)] if val[(s, c)] is not None else est.get((s, c), 0))
                            for j in jkids[cty] for s in kids[j])
                each = (val[(cty, c)] - known) / len(pending)
                for s in pending:
                    est[(s, c)] = each
    for k, v in est.items():
        if not 1 - 1e-6 <= v <= 2 + 1e-6:
            clipped += 1
        est[k] = min(2.0, max(1.0, v))
    return est, clipped


# --------------------------------------------------------------------------- the split
def raw_dpair(v):
    dm = sum(v[c] for c in MINORITY)
    return min(max(dm - v["MT_D"], 0.0), dm / 2)


def split(v, dscale=1.0):
    """One settlement's cells (suppression filled) -> {code: (measured, derived)}.

    The census counts a two-answer person under both languages and publishes no combinations,
    so who is paired with whom is inferred:
      * E = all mentions - people who answered = the people who named two (at most two allowed).
      * Dpair = minority mentions - MT_D (people naming at least one minority language) = the
        people who named two MINORITY languages (Romani and Boyash, German and Croatian...).
      * HX = E - Dpair: every other two-answer person is taken to have named Hungarian and one
        other language. ASSUMPTION, the one in this file: a pair of "other" with a minority
        language, or of two "other" languages, is counted as Hungarian plus one.
    Then spec §3.6's half a person to each language named: Hungarian loses HX/2; each other
    language loses its proportional part of HX/2, and each minority language its part of Dpair.
    `measured` is the people who named the language alone; `derived` is the half shares.

    Dpair from suppression ESTIMATES runs high (a settlement with German, Slovak and MT_D all
    suppressed shows a spurious pair), so `dscale` rescales it to the vármegye's exact figure.
    """
    A = v["MT"] - v["MT_NA"]
    dm = sum(v[c] for c in MINORITY)
    nonhu = dm + v["MT_O"]
    E = v["MT_HU"] + nonhu - A
    dpair = min(raw_dpair(v) * dscale, dm / 2)
    hx = min(max(E - dpair, 0.0), v["MT_HU"], max(nonhu - 2 * dpair, 0.0))
    out = {"MT_HU": (v["MT_HU"] - hx, hx / 2)}
    for c in MINORITY + ["MT_O"]:
        m = v[c]
        if m <= 0:
            continue
        half = hx / 2 * m / nonhu + (dpair * m / dm if c in MINORITY else 0.0)
        out[c] = (m - 2 * half, half)
    return out, A, E, dpair, hx


# --------------------------------------------------------------------------- normalise
def build():
    val, labels, geo, settle, jaras, county, w9 = read()
    ok = True

    def say(good, msg):
        nonlocal ok
        ok &= bool(good)
        print(f"  {'OK ' if good else 'BAD'} {msg}")

    print("units:")
    say(len(settle) == 3177, f"{len(settle):,} settlements (3,177 incl. Budapest's 23 districts)")
    say(len(county) == 20, f"{len(county)} vármegye (19 + Budapest)")
    say(not any(g == "13578" for g, _ in val),
        "Budapest's 'not divisible by district' residual (13578) has no cells")
    say(val[("HU", "MT")] == NATIONAL, f"MT (total) = {val[('HU', 'MT')]:,} = the population")

    # the two cubes agree at vármegye level, every code
    bad = [(g, c) for g in county + ["HU"] for c in MT_CODES if val.get((g, c)) != w9.get((g, c))]
    say(not bad, f"WBS009 = WBS003 at vármegye and country, {len(county) + 1} units x "
                 f"{len(MT_CODES)} codes" + (f"; differ: {bad[:5]}" if bad else ""))
    say(w9.get(("HU", "TOTAL")) == NATIONAL, "WBS009 TOTAL = the population")

    est, clipped = fill(val, geo, settle, jaras, county)
    say(clipped < 50, f"{len(est):,} suppressed settlement cells estimated, {clipped} clipped to [1, 2]")

    def cell(s, c):
        return val[(s, c)] if val[(s, c)] is not None else est[(s, c)]

    # settlements, filled, sum to the vármegye cells
    worst = 0.0
    for cty in county:
        ss = [s for s in settle if geo[geo[s][2]][2] == cty]
        for c in MT_CODES:
            worst = max(worst, abs(sum(cell(s, c) for s in ss) - val[(cty, c)]))
    say(worst < 3, f"settlements summed per vármegye = the vármegye cell, every code "
                   f"(worst gap {worst:.2f}, from clipping)")
    for c in ("MT", "MT_HU"):
        say(not any(val[(s, c)] is None for s in settle)
            and sum(val[(s, c)] for s in settle) == val[("HU", c)],
            f"{c} printed for every settlement and summing to the country")

    # national arithmetic of the two-answer people
    n = {c: val[("HU", c)] for c in MT_CODES}
    gi_pairs = n["MT_GIR"] + n["MT_GIB"] - n["MT_GI"]
    dm = sum(n[c] for c in MINORITY)
    A = n["MT"] - n["MT_NA"]
    E = n["MT_HU"] + dm + n["MT_O"] - A
    print(f"\nnational: {A:,} answered, {n['MT_NA']:,} did not ({n['MT_NA'] / n['MT']:.1%});")
    print(f"  {E:,} named two; {dm - n['MT_D']:,} of them two minority languages "
          f"({gi_pairs:,} Romani and Boyash); the other {E - dm + n['MT_D']:,} taken as Hungarian + one")

    cty_of = {s: geo[geo[s][2]][2] for s in settle}
    vals = {s: {c: cell(s, c) for c in MT_CODES} for s in settle}
    raw = defaultdict(float)
    for s in settle:
        raw[cty_of[s]] += raw_dpair(vals[s])
    dscale = {}
    for cty in county:
        true = sum(val[(cty, c)] for c in MINORITY) - val[(cty, "MT_D")]
        dscale[cty] = true / raw[cty] if raw[cty] > 0 else 1.0
    print(f"  minority pairs from settlement estimates {sum(raw.values()):,.0f}, rescaled per "
          f"vármegye to its exact figure")

    rows, stats = [], defaultdict(float)
    tot_drawn = 0.0
    for s in settle:
        v = vals[s]
        parts, a, e, dpair, hx = split(v, dscale[cty_of[s]])
        name = geo[s][0]
        stats["dpair"] += dpair
        stats["hx"] += hx
        for c, (meas, half) in parts.items():
            sup = val[(s, c)] is None
            lab = labels[c][0]
            if meas > 1e-9:
                rows.append((s, name, lab, meas, "derived" if sup else "measured",
                             "suppressed, estimated" if sup else "one tongue"))
            if half > 1e-9:
                rows.append((s, name, lab, half, "derived",
                             "suppressed, estimated" if sup else "half of two"))
            tot_drawn += meas + half
        rows.append((s, name, labels["MT_NA"][0], v["MT_NA"], "measured", "not stated"))
    say(abs(tot_drawn - A) < 100, f"drawn {tot_drawn:,.0f} = answered {A:,} "
                                  f"(gap {tot_drawn - A:,.1f}, from suppression estimates and clips)")
    print(f"  shared out: {stats['dpair']:,.0f} minority pairs, {stats['hx']:,.0f} Hungarian pairs")

    by = defaultdict(float)
    for r in rows:
        if r[5] != "not stated":
            by[r[2]] += r[3]
    print("\ndrawn, national:")
    for lab, x in sorted(by.items(), key=lambda kv: -kv[1]):
        print(f"  {x:>12,.0f}  {100 * x / A:6.3f}%  {lab}")
    if not ok:
        raise SystemExit("checks FAILED")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["geo_id", "geo_level", "geo_name", "source_category", "count", "tier", "part",
                     "year", "source_id"])
        for s, name, lab, x, tier, part in rows:
            w.writerow([s, "settlement", name, lab, round(x, 4), tier, part, 2022,
                        "hu_nepszamlalas_2022_wbs003"])
    print(f"\nwrote {OUT} ({len(rows):,} rows)")


def main():
    if "--fetch" in sys.argv:
        fetch()
    build()


if __name__ == "__main__":
    main()
