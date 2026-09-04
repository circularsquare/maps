"""Hungary — KSH, Népszámlálás 2022, religion, down to settlement.

Reads data/raw/hu/ and writes data/normalized/hu.csv.

Two tables of one census, and they are a textbook §3.9 split:

  WBS003  11 categories x 3,177 settlements   (Budapest arrives as its 23 kerület)
  WBS008  29 categories x 20 vármegye         (six Orthodox jurisdictions, nine
                                               other Christian bodies, Muslim,
                                               Buddhist, Hindu kept apart)

Unlike India's, the two reconcile EXACTLY -- every WBS008 category sums into its WBS003
parent to the person -- so `taxonomy/hierarchy/hu.csv` is a deduction rather than a
judgement, and `allocate.py --within 5` pushes the 29 down onto the settlements.

**THE CATEGORY CODES ARE NOT SELF-EXPLANATORY AND MUST NOT BE GUESSED.** The database
exports carry codes only (`RE_C`, `RE_CA`, `RE_OU`), and the obvious readings are wrong:
`RE_C` is Catholic but `RE_CA` is *Calvinist*; `RE_CO` is "Other Christian" and not
anything Coptic; `RE_OU` is *Ukrainian* Orthodox, a jurisdiction that does not appear in
KSH's own prose list of the five Orthodox churches in Hungary. The labels here are read
from KSH's SDMX codelists at run time (see `fetch`), never transcribed by hand, and
`check()` re-derives every one of them against the published national figures so a
renamed or re-ordered code fails the run instead of quietly relabelling the map.

Basis is self_id: the 2022 census asked the religion question and answering was voluntary.
**40.1% did not answer** -- the largest non-response of any source in this project -- which
is a property of the question, not of the country, and belongs in note_public.

Usage:
    python sources/hu.py --fetch    download the SDMX structure messages (labels + geography)
    python sources/hu.py            normalise from data/raw/hu/
"""

import csv
import json
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "hu")
OUT = os.path.join(ROOT, "data", "normalized", "hu.csv")

SOURCE_ID = "hu_nepszamlalas_2022"
YEAR = 2022
BASIS = "self_id"

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

# The two exports, taken from the census database UI (see sources/hu.md §2 -- the app is a
# JavaScript client and its full WBS003 cube cannot be pulled whole).
SETTLEMENT_CSV = "WBS003_settlement.csv"
COUNTY_CSV = "WBS008_county.csv"

# The SDMX structure messages, which carry the codelists. These ARE fetchable.
API = "https://nepszamlalas2022.ksh.hu"
STRUCTURE = {"WBS003": "hu_structure_WBS003.json", "WBS008": "hu_structure_WBS008.json"}

NATIONAL = 9_603_634          # KSH, resident population, census 2022

# nsz2022-1.1.7-eng.xlsx, "Total" block, 2022 column. Every code's national figure is
# checked against this, which is what pins a code to a label (sources/hu.md §3).
PUBLISHED = {
    "RE_C": 2_886_619, "RE_RC": 2_643_855, "RE_GC": 165_135, "RE_OC": 15_578,
    "RE_CA": 943_982, "RE_LU": 176_503, "RE_J": 7_635, "RE_CD": 141_197,
    "RE_OCD": 29_977, "RE_NOT": 1_549_610, "RE_NA": 3_852_533,
}

# WBS003's nine-way partition of the population. RE_RC and RE_GC are named as subsets of
# RE_C by their own labels ("Roman Catholic among Catholics") and must never be added in.
PARTITION = ("RE_C", "RE_OC", "RE_CA", "RE_LU", "RE_J", "RE_CD", "RE_OCD",
             "RE_NOT", "RE_NA")
RE_C, RE_RC, RE_GC = "RE_C", "RE_RC", "RE_GC"
CATHOLIC_CHILDREN = (RE_RC, RE_GC)

# The derived remainder of RE_C after its two named rites. Emitted as its own category so
# that the DRAWN set is a partition with no nesting: taxonomy/hu2022.py excludes RE_C
# itself and maps these three. Its English name is written to look like KSH's own, but it
# is ours -- `derived=` in the note says so on every row.
CATHOLIC_RESIDUAL = "Catholic, rite not stated"

# WBS008 categories that are NOT part of its own partition: the universe total, and the
# two Catholic subsets it repeats from WBS003.
COUNTY_UNIVERSE = ("TOTAL",)

SUPPRESSED = "Q"              # OBS_STATUS marking a withheld cell; OBS_VALUE is "null"

DIM = {"WBS003": ("TEL_SZ_ADAT", "TERUL_GEO5"), "WBS008": ("VALLAS_V2", "TERUL_GEO3")}


# --------------------------------------------------------------------------- fetch
def fetch():
    """Download the SDMX structure messages. Small, reliable, and the only remote step.

    The DATA cannot be fetched the same way: `/api/dataflows/WBS003/<v>/s/...` selects all
    149 of CL_TEL_SZ_ADAT's variables, not just the eleven religion ones, and the ~60MB
    response is truncated by the server on every attempt. sources/hu.md §2 records what a
    working subset request would need.
    """
    import requests

    os.makedirs(RAW, exist_ok=True)
    ua = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"}
    ver = requests.get(f"{API}/api/version", headers=ua, timeout=60).json()["version"]
    print(f"census database version {ver}")
    for flow, name in STRUCTURE.items():
        dest = os.path.join(RAW, name)
        r = requests.get(f"{API}/api/structure/{flow}/{ver}", headers=ua, timeout=180)
        r.raise_for_status()
        doc = r.json()
        # §5a: HTTP 200 is not a download. Assert the thing we came for is present.
        if not doc.get("data", {}).get("codelists"):
            raise SystemExit(f"{flow}: structure message carries no codelists")
        with open(dest, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, ensure_ascii=False)
        print(f"  {name}: {os.path.getsize(dest):,} bytes")

    for name in (SETTLEMENT_CSV, COUNTY_CSV):
        p = os.path.join(RAW, name)
        if not os.path.exists(p):
            raise SystemExit(
                f"missing {p}. The two data exports come from the census database UI and "
                "cannot be fetched -- see sources/hu.md §2 for how they were obtained.")


# --------------------------------------------------------------------------- codelists
def _structure(flow):
    path = os.path.join(RAW, STRUCTURE[flow])
    if not os.path.exists(path):
        raise SystemExit(f"missing {path} -- run with --fetch first")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _codelist(flow, want_ids):
    """The codelist covering `want_ids`, as {code: (english, hungarian)}."""
    for cl in _structure(flow)["data"].get("codelists", []):
        ids = {c["id"] for c in cl.get("codes", [])}
        if want_ids <= ids:
            return {c["id"]: (c.get("names", {}).get("en", ""),
                              c.get("names", {}).get("hu", ""))
                    for c in cl["codes"]}
    raise SystemExit(f"{flow}: no codelist contains all of {sorted(want_ids)} -- KSH "
                     "reorganised its codelists, do not guess the labels")


def _geography(flow):
    """{code: (name, parent)} for the geography dimension, from the source's own codelist.

    KSH's CL_TERUL_GEO5 carries the full parent chain -- settlement -> jaras -> vármegye
    -> region -> country -- so a settlement's county is read off the source rather than
    derived from a boundary file or guessed from the code.
    """
    for cl in _structure(flow)["data"].get("codelists", []):
        if cl["id"].startswith("CL_TERUL_GEO"):
            return {c["id"]: (c.get("names", {}).get("hu", ""), c.get("parent"))
                    for c in cl["codes"]}
    raise SystemExit(f"{flow}: no CL_TERUL_GEO* codelist")


def _nuts3_of(code, geo):
    """Climb the parent chain to the vármegye (NUTS3, five characters, 'HU110')."""
    seen = 0
    while code is not None and seen < 10:
        if len(code) == 5 and code.startswith("HU"):
            return code
        code = geo.get(code, ("", None))[1]
        seen += 1
    return None


# --------------------------------------------------------------------------- read
def _rows(name, catdim, geodim):
    path = os.path.join(RAW, name)
    if not os.path.exists(path):
        raise SystemExit(f"missing {path} -- see sources/hu.md §2")
    with open(path, encoding="utf-8-sig", newline="") as fh:
        out = list(csv.DictReader(fh, delimiter=";"))
    if not out:
        raise SystemExit(f"{path} is empty")
    for need in ("OBS_STATUS", "OBS_VALUE", catdim, geodim, "TIME_PERIOD"):
        if need not in out[0]:
            raise SystemExit(f"{path}: no {need!r} column, got {list(out[0])}")
    periods = {r["TIME_PERIOD"] for r in out}
    if periods != {"2022"}:
        raise SystemExit(f"{path}: expected only 2022, got {sorted(periods)}")
    return out


def _value(row, where):
    """Return (count, kind). Suppression is in-band and must never become NaN (§12)."""
    status, raw = row["OBS_STATUS"], row["OBS_VALUE"]
    if status == SUPPRESSED:
        return None, "suppressed"
    if status not in ("null", "", None):
        raise SystemExit(f"{where}: unknown OBS_STATUS {status!r} -- KSH added a sentinel")
    s = str(raw).strip()
    if s.lstrip("-").isdigit():
        return int(s), "n"
    raise SystemExit(f"{where}: non-numeric OBS_VALUE {raw!r} on an unsuppressed row")


def read():
    rows, stats = [], {"n": 0, "suppressed": 0}

    def emit(gid, level, gname, cat, n, note):
        rows.append({"geo_id": gid, "geo_level": level, "geo_name": gname,
                     "source_category": cat, "count": n, "basis": BASIS,
                     "year": YEAR, "source_id": SOURCE_ID, "note": note})

    # ---- WBS003: settlements, plus the national row for reconciliation ----
    catdim, geodim = DIM["WBS003"]
    src = _rows(SETTLEMENT_CSV, catdim, geodim)
    codes = {r[catdim] for r in src}
    labels = _codelist("WBS003", codes)
    geo = _geography("WBS003")

    missing = codes - set(labels)
    if missing:
        raise SystemExit(f"WBS003: no label for {sorted(missing)}")

    # Gather per unit first, because the Catholic residual below is a fact about a unit
    # rather than about a row.
    units = {}
    for r in src:
        code, gcode = r[catdim], r[geodim]
        # Settlements are the five-digit KSH codes; everything else in this file is an
        # aggregate level (jaras, vármegye, region, country) and is a UNIVERSE, not a unit.
        if gcode.isdigit() and len(gcode) == 5:
            level = "settlement"
        elif gcode == "HU":
            level = "country"
        else:
            continue
        n, kind = _value(r, f"WBS003 {code}/{gcode}")
        stats[kind] = stats.get(kind, 0) + 1
        units.setdefault((level, gcode), {})[code] = n

    n_residual, n_blurred, residual_total = 0, 0, 0
    for (level, gcode), vals in units.items():
        name, _ = geo.get(gcode, ("", None))
        if level == "settlement":
            nuts3 = _nuts3_of(gcode, geo)
            if nuts3 is None:
                raise SystemExit(f"settlement {gcode} has no vármegye in the parent chain")
            gid = f"{nuts3}_{gcode}"
            stem = f"level=settlement; ksh={gcode}; nuts3={nuts3}"
        else:
            gid, stem = "HU", "level=country"

        for code, n in vals.items():
            if n is not None:
                emit(gid, level, name or gcode, labels[code][0], n,
                     f"{stem}; code={code}")

        # ---- the Catholic residual -------------------------------------------------
        # RE_RC and RE_GC are named by KSH as subsets ("Roman Catholic among Catholics"),
        # so RE_C is an aggregate. Drawing the two children alone drops the 77,629 people
        # who answered Catholic without naming a rite; drawing the parent as well would
        # count 2.8M of them twice. §12: a source with a remainder needs it emitted as a
        # category. It is exact arithmetic, not an estimate — except where a child is
        # suppressed, which moves those few people into the residual and is counted below.
        c = vals.get(RE_C)
        if c is None:
            continue
        rc, gc = vals.get(RE_RC), vals.get(RE_GC)
        if rc is None or gc is None:
            n_blurred += 1
        resid = c - (rc or 0) - (gc or 0)
        if resid < 0:
            raise SystemExit(f"{gid}: Roman+Greek Catholic exceed Catholic by {-resid} -- "
                             "RE_RC/RE_GC are not subsets of RE_C after all")
        if resid:
            n_residual += 1
            if level == "settlement":
                residual_total += resid
            emit(gid, level, name or gcode, CATHOLIC_RESIDUAL, resid,
                 f"{stem}; code=RE_CX; derived=RE_C-RE_RC-RE_GC")

    print(f"  Catholic rite-not-stated derived in {n_residual:,} units "
          f"({residual_total:,} people at settlement level); "
          f"{n_blurred:,} units had a suppressed rite absorbed into it")

    # ---- WBS008: the 29 categories at vármegye ----
    catdim, geodim = DIM["WBS008"]
    src = _rows(COUNTY_CSV, catdim, geodim)
    codes8 = {r[catdim] for r in src}
    labels8 = _codelist("WBS008", codes8)
    geo8 = _geography("WBS008")
    missing = codes8 - set(labels8)
    if missing:
        raise SystemExit(f"WBS008: no label for {sorted(missing)}")

    counties = {}
    for r in src:
        code, gcode = r[catdim], r[geodim]
        if not (len(gcode) == 5 and gcode.startswith("HU")):
            raise SystemExit(f"WBS008: unexpected geography {gcode!r}, expected NUTS3")
        n, kind = _value(r, f"WBS008 {code}/{gcode}")
        stats[kind] = stats.get(kind, 0) + 1
        counties.setdefault(gcode, {})[code] = n

    for gcode, vals in counties.items():
        name, _ = geo8.get(gcode, ("", None))
        for code, n in vals.items():
            if n is not None:
                emit(gcode, "county", name or gcode, labels8[code][0], n,
                     f"level=county; code={code}; nuts3={gcode}")
        # The same residual as at settlement level, and for the same reason — but here it
        # is exact, because WBS008 suppresses nothing. It also has to exist at BOTH levels
        # or allocate.py drops the category: the allocation carries a fine column forward
        # only when some coarse category lands on it.
        c, rc, gc = vals.get(RE_C), vals.get(RE_RC), vals.get(RE_GC)
        if c is None:
            continue
        resid = c - (rc or 0) - (gc or 0)
        if resid < 0:
            raise SystemExit(f"{gcode}: Roman+Greek Catholic exceed Catholic")
        if resid:
            emit(gcode, "county", name or gcode, CATHOLIC_RESIDUAL, resid,
                 f"level=county; code=RE_CX; nuts3={gcode}; derived=RE_C-RE_RC-RE_GC")

    print(f"  cells: {stats.get('n', 0):,} numeric, "
          f"{stats.get('suppressed', 0):,} suppressed")
    return rows, labels, labels8


# --------------------------------------------------------------------------- check
def check(rows, labels, labels8):
    ok = True
    by_level = {}
    for r in rows:
        by_level.setdefault(r["geo_level"], set()).add(r["geo_id"])

    # KSH publishes a 'Budapest kerületre nem bontható adatai' residual, code 13578, for
    # figures it cannot split by district. It is EMPTY for religion, and that is the proof
    # that Budapest's 23 kerület account for the whole city rather than most of it. If it
    # ever carries people the settlement count moves off 3,177 and the check below fails.
    resid = [r for r in rows if r["note"].find("ksh=13578") >= 0]
    good = not resid
    ok &= good
    print(f"\n    {'OK ' if good else 'BAD'} the Budapest 'not divisible by district' "
          f"residual (13578) is empty: {len(resid)} rows")

    expected_units = {"settlement": 3_177, "county": 20, "country": 1}
    print("\n  units per level:")
    for lv, want in expected_units.items():
        got = len(by_level.get(lv, ()))
        good = got == want
        ok &= good
        print(f"    {'OK ' if good else 'BAD'} {lv:<11} {got:>6,} units "
              f"(expected {want:,})")

    # ---- the codes pin to labels by ARITHMETIC, not by reading the abbreviation ----
    nat = {}
    for r in rows:
        if r["geo_level"] == "country":
            nat[r["source_category"]] = r["count"]
    print("\n  national figures vs nsz2022-1.1.7-eng.xlsx (this is what pins each code):")
    for code, want in PUBLISHED.items():
        label = labels[code][0]
        got = nat.get(label)
        good = got == want
        ok &= good
        print(f"    {'OK ' if good else 'BAD'} {code:<7} {want:>11,}  {label}")
        if not good:
            print(f"        got {got!r} -- the code no longer means what hu.py assumes")

    # ---- WBS003 partitions the population ----
    part = sum(nat.get(labels[c][0], 0) for c in PARTITION)
    good = part == NATIONAL
    ok &= good
    print(f"\n    {'OK ' if good else 'BAD'} the nine-way partition sums to "
          f"{part:,} (population {NATIONAL:,})")
    kids = sum(nat.get(labels[c][0], 0) for c in CATHOLIC_CHILDREN)
    cath = nat.get(labels["RE_C"][0], 0)
    good = kids < cath
    ok &= good
    print(f"    {'OK ' if good else 'BAD'} Roman + Greek Catholic {kids:,} sit INSIDE "
          f"Catholic {cath:,} (residual {cath - kids:,}, rite not stated)")

    # The DRAWN set: RE_C replaced by its two rites plus the derived remainder, so that
    # nothing nests and nothing is lost. This is the partition countries.py maps.
    drawn = [labels[c][0] for c in PARTITION if c != RE_C]
    drawn += [labels[RE_RC][0], labels[RE_GC][0], CATHOLIC_RESIDUAL]
    tot = sum(nat.get(c, 0) for c in drawn)
    good = tot == NATIONAL
    ok &= good
    print(f"    {'OK ' if good else 'BAD'} the DRAWN partition ({len(drawn)} categories, "
          f"no nesting) sums to {tot:,}")
    good = nat.get(CATHOLIC_RESIDUAL, 0) == cath - kids
    ok &= good
    print(f"    {'OK ' if good else 'BAD'} derived {CATHOLIC_RESIDUAL!r} = "
          f"{nat.get(CATHOLIC_RESIDUAL, 0):,}")

    # ---- settlements sum to the national figure, less what is suppressed ----
    print("\n  settlement sums vs national — the gap is suppression (§3.5), not loss:")
    tot_gap = 0
    for code in PARTITION + CATHOLIC_CHILDREN:
        label = labels[code][0]
        s = sum(r["count"] for r in rows
                if r["geo_level"] == "settlement" and r["source_category"] == label)
        gap = nat.get(label, 0) - s
        tot_gap += gap
        bad = gap < 0 or gap > 2_000
        ok &= not bad
        print(f"    {'BAD' if bad else 'OK '} {label:<40} {s:>11,}  gap {gap:>6,}")
    frac = tot_gap / NATIONAL
    good = frac < 0.001
    ok &= good
    print(f"    {'OK ' if good else 'BAD'} total withheld {tot_gap:,} = {frac:.4%} "
          "of the population")

    # ---- WBS008 reconciles into WBS003, exactly. This is what makes the split usable ----
    cty = {}
    for r in rows:
        if r["geo_level"] == "county":
            cty[r["source_category"]] = cty.get(r["source_category"], 0) + r["count"]
    good = cty.get(labels8["TOTAL"][0], 0) == NATIONAL
    ok &= good
    print(f"\n    {'OK ' if good else 'BAD'} WBS008 counties sum to the population")

    GROUPS = {
        "RE_OC": ("RE_OG", "RE_ORU", "RE_OS", "RE_OB", "RE_ORO", "RE_OU", "RE_OO"),
        "RE_CD": ("RE_UN", "RE_BA", "RE_ME", "RE_AD", "RE_PE", "RE_AN", "RE_JW",
                  "RE_FC", "RE_CO"),
        "RE_OCD": ("RE_M", "RE_B", "RE_H", "RE_OCD2"),
    }
    print("\n  WBS008 -> WBS003, the join taxonomy/hierarchy/hu.csv encodes:")
    for parent, children in GROUPS.items():
        want = nat.get(labels[parent][0], 0)
        got = sum(cty.get(labels8[c][0], 0) for c in children)
        good = got == want
        ok &= good
        print(f"    {'OK ' if good else 'BAD'} {labels[parent][0]:<38} "
              f"{want:>9,} = {got:>9,}  ({len(children)} categories)")

    by_cat = sorted(((v, k) for k, v in nat.items()), reverse=True)
    print(f"\n  {len(rows):,} rows. WBS003's categories, national:")
    for n, name in by_cat:
        print(f"    {n:>10,}  {100.0 * n / NATIONAL:5.2f}%  {name}")

    if not ok:
        raise SystemExit("reconciliation FAILED")


def main():
    if "--fetch" in sys.argv:
        fetch()
    rows, labels, labels8 = read()
    check(rows, labels, labels8)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print("\nwrote", OUT, f"({len(rows):,} rows)")


if __name__ == "__main__":
    main()
