"""Slovenia — SURS, Popis 2002, religion by občina.

Reads (or fetches) data/raw/si/ and writes data/normalized/si.csv.

**THE CLOSURE THAT HID THIS WAS ABOUT A DIFFERENT CENSUS.** sources.md §11c and §11k both
struck Slovenia off, and both were right about what they checked: the 2021 census is
register-based and does not ask religion, so SURS has no recent figure at any geography.
§11k went further and recorded that *"SURS PxWeb is live and its catalogue is 908 KB"* while
still concluding there was nothing to draw. **The catalogue it downloaded contains
`05W1006S.px`, `Prebivalstvo po veroizpovedi, občine, Slovenija, popis 2002`** — 192
municipalities, ten religion answers, the last conventional census Slovenia ran. Finland's
2026-09-08 review was the same shape: a correct finding about the register tier, read as a
finding about the country.

**One API, no key, no login.** `pxweb.stat.si/SiStatData/api/v1/sl/Data/05W1006S.px` is a
plain PxWeb endpoint; GET returns the variables, POST returns 19 KB of json-stat2. The same
table is also on the 2002 census microsite, which is still up, as a frameset over
`rezultati_html/OBC-T-06SLO.htm` and as `rezultati/OBC-T-06si.xls`; WebFetch cannot see
either because it drops frames. The API is used here because it carries the status flags and
the HTML does not distinguish them from the letter `z`.

**THE MUNICIPALITY CODES IN THIS TABLE ARE NOT SLOVENIA'S MUNICIPALITY CODES.** The `OBČINA`
dimension runs `001`-`193` where `001` is SLOVENIJA and the rest are an alphabetical
sequence, so Ajdovščina is `002` here and `001` everywhere else, including in GISCO. Joining
on it would shift all 192 units by one place and every total would still reconcile
([[reference_name_join_wrong_neighbour]]). The real codes come from a second table of the
same census, `05W0405S.px`, whose `NASELJA` labels are of the form `001  AJDOVŠČINA` — the
office's own code, in its own dimension, for the same enumeration. The join is by name and
**is checked on population**: the two tables must agree to the person on all 192
municipalities, which they do, and 192 distinct populations cannot all survive a shifted or
crossed join.

**THE `z` CELLS ARE NOT ZEROS AND ARE NOT RECONSTRUCTED.** json-stat2's `status` map
separates `z` (zaupno, withheld) from `-` (no such phenomenon, a true zero); reading either
as the other is spec §3.8's in-band trap, and the HTML table prints both as ordinary text.
249 of the 960 detail cells are withheld, holding **1,974 people, 0.10% of Slovenia**, and
they fall entirely on the four small answers: Catholic, the totals and `Neznano` are never
suppressed. In seven municipalities only one detail cell is withheld, which makes it exactly
recoverable by subtraction from the declared subtotal, and this file **does not recover it**.
Undoing an office's disclosure control to gain a few dozen people is not a trade worth
making, and the shortfall is reported per category instead (spec §3.5, §3.8).

Usage:
    python sources/si.py --fetch    two POSTs, 19 KB and 25 KB
    python sources/si.py            normalise from data/raw/si/
"""

import csv
import json
import os
import re
import sys
import urllib.request

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "si")
OUT = os.path.join(ROOT, "data", "normalized", "si.csv")

SOURCE_ID = "si_popis_2002"
YEAR = 2002
BASIS = "self_id"

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

API = "https://pxweb.stat.si/SiStatData/api/v1/sl/Data/"
REL_TABLE = "05W1006S.px"          # religion x občina, popis 2002
POP_TABLE = "05W0405S.px"          # population x naselje, popis 2002 -- carries the codes
DET_TABLE = "05W1606S.px"          # religion x settlement type, popisa 1991 and 2002
REL_JSON = os.path.join(RAW, "05W1006S.json")
POP_JSON = os.path.join(RAW, "05W0405S.json")
DET_JSON = os.path.join(RAW, "05W1606S.json")

NATIONAL = 1_964_036               # SURS's published 2002 enumerated population
EXPECTED_MUNICIPALITIES = 192
COUNTRY_LABEL = "SLOVENIJA"

# Status flags, resolved on their TEXT rather than their position (spec §3.8).
STATUS_WITHHELD = "z"              # zaupno
STATUS_TRUE_ZERO = "-"             # ni pojava

# The universe row and the declared-a-religion subtotal.  Both are people counted twice if
# drawn, and taxonomy/si2002.py excludes them by label; they are kept in the CSV because
# check() reconciles against them.
TOTAL_LABEL = "Veroizpoved - SKUPAJ"
DECLARED_LABEL = "Opredeljeni po veroizpovedi - skupaj"
DETAIL_LABELS = ["katoliška", "evangeličanska in druge protestantske", "pravoslavna",
                 "islamska", "druge veroizpovedi"]

# Two municipalities are spelt differently in the two 2002 tables.  Both are the office's
# own older forms, both resolve to exactly one remaining code, and both are confirmed by the
# population check below rather than trusted.
#   018  Destrnik  -- printed DESTERNIK in the settlement table
#   116  Sveti Jurij ob Ščavnici -- printed SVETI JURIJ, its name before the 2001 rename
ALIASES = {"DESTRNIK": "DESTERNIK", "SVETI JURIJ OB ŠČAVNICI": "SVETI JURIJ"}


def _post(table, query):
    body = json.dumps(query).encode("utf-8")
    req = urllib.request.Request(API + table, data=body, headers={
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    })
    return urllib.request.urlopen(req, timeout=300).read()


def _get(table):
    req = urllib.request.Request(API + table, headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    return json.loads(urllib.request.urlopen(req, timeout=300).read())


def fetch():
    os.makedirs(RAW, exist_ok=True)

    if not os.path.exists(REL_JSON) or os.path.getsize(REL_JSON) < 5_000:
        print("POST", API + REL_TABLE)
        raw = _post(REL_TABLE, {"query": [
            {"code": "OBČINA", "selection": {"filter": "all", "values": ["*"]}},
            {"code": "VEROIZPOVED", "selection": {"filter": "all", "values": ["*"]}},
        ], "response": {"format": "json-stat2"}})
        doc = json.loads(raw)
        # §5a: HTTP 200 is not a download.  Assert it is the cube and not an error envelope.
        if doc.get("size") != [193, 11]:
            raise SystemExit(f"unexpected cube shape {doc.get('size')} -- SURS has changed "
                             f"{REL_TABLE}; re-read its metadata before trusting anything")
        with open(REL_JSON, "wb") as fh:
            fh.write(raw)
        print(f"  {len(raw):,} bytes")
    else:
        print("already have", REL_JSON)

    if not os.path.exists(POP_JSON) or os.path.getsize(POP_JSON) < 5_000:
        # Only the 192 municipality rows of the settlement table are wanted; asking for all
        # 6,152 naselja would be 20x the bytes for a code lookup and a population check.
        meta = _get(POP_TABLE)
        var = next(v for v in meta["variables"] if v["code"] == "NASELJA")
        muni = [v for v, t in zip(var["values"], var["valueTexts"])
                if re.match(r"^\d{3}\s\s", t)]
        if len(muni) != EXPECTED_MUNICIPALITIES:
            raise SystemExit(f"{len(muni)} municipality rows in {POP_TABLE}, expected "
                             f"{EXPECTED_MUNICIPALITIES}")
        print("POST", API + POP_TABLE, f"({len(muni)} municipality rows)")
        raw = _post(POP_TABLE, {"query": [
            {"code": "NASELJA", "selection": {"filter": "item", "values": muni}},
            {"code": "SPOL", "selection": {"filter": "item", "values": ["1"]}},
        ], "response": {"format": "json-stat2"}})
        with open(POP_JSON, "wb") as fh:
            fh.write(raw)
        print(f"  {len(raw):,} bytes")
    else:
        print("already have", POP_JSON)

    if not os.path.exists(DET_JSON) or os.path.getsize(DET_JSON) < 1_000:
        print("POST", API + DET_TABLE)
        raw = _post(DET_TABLE, {"query": [
            {"code": "MERITVE", "selection": {"filter": "item", "values": ["1"]}},
            {"code": "VEROIZPOVED", "selection": {"filter": "all", "values": ["*"]}},
            {"code": "TIP NASELJA", "selection": {"filter": "item", "values": ["1"]}},
            {"code": "LETO", "selection": {"filter": "all", "values": ["*"]}},
        ], "response": {"format": "json-stat2"}})
        doc = json.loads(raw)
        if doc.get("size") != [1, 15, 1, 2]:
            raise SystemExit(f"unexpected cube shape {doc.get('size')} for {DET_TABLE}")
        with open(DET_JSON, "wb") as fh:
            fh.write(raw)
        print(f"  {len(raw):,} bytes")
    else:
        print("already have", DET_JSON)


def _load(path):
    if not os.path.exists(path):
        raise SystemExit(f"missing {path} -- run with --fetch first")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _key(name):
    """Fold the two tables' municipality names to one form: SURS punctuates the hyphenated
    names with spaces in one table and without in the other."""
    s = " ".join(str(name).split())
    s = re.sub(r"\s*-\s*", "-", s)
    return s.upper()


def codes():
    """Official občina code -> (name, 2002 population), from the census's own settlement
    table.  This is the only place the real codes exist for this vintage."""
    doc = _load(POP_JSON)
    cat = doc["dimension"]["NASELJA"]["category"]
    out = {}
    for value, pos in cat["index"].items():
        label = cat["label"][value]
        m = re.match(r"^(\d{3})\s\s(.+)$", label)
        if not m:
            raise SystemExit(f"settlement-table label {label!r} is not a municipality row")
        out[m.group(1)] = (m.group(2).strip(), doc["value"][pos])
    if len(out) != EXPECTED_MUNICIPALITIES:
        raise SystemExit(f"{len(out)} codes, expected {EXPECTED_MUNICIPALITIES}")
    return out


def detail_rows():
    """The 1991-and-2002 national table, `05W1606S.px`, at fourteen categories.

    THIS IS THE TABLE THE UNSD ORACLE COUNTS, and it exists only for the country: Popis 2002
    published fourteen answers nationally and five per municipality, which is spec §3.9's
    trade in its usual direction. It is written into the CSV as a check level, never drawn,
    and it earns its place twice over. It DECOMPOSES both of the drawn table's catch-alls to
    the person, which is what pins their mapping (16,135 Protestants are 14,736 Evangelical
    Lutherans and 1,399 others; 3,831 `druge veroizpovedi` are 1,877 other Christians, 1,026
    Oriental religions, 558 other, 271 agnostics and 99 Jews). And it carries 1991 beside
    2002 on identical categories, which is the only measurement of how fast any of this was
    moving when Slovenia stopped asking.
    """
    doc = _load(DET_JSON)
    dims, sizes = doc["id"], doc["size"]
    cats = {k: doc["dimension"][k]["category"] for k in dims}
    vdim = "VEROIZPOVED"
    rows = []
    for year in cats["LETO"]["index"]:
        for v, vpos in sorted(cats[vdim]["index"].items(), key=lambda kv: kv[1]):
            idx = 0
            sel = {dims[0]: "1", vdim: v, "TIP NASELJA": "1", "LETO": year}
            for name, size in zip(dims, sizes):
                idx = idx * size + cats[name]["index"][sel[name]]
            n = doc["value"][idx]
            if n is None:
                continue                     # 1991 has no agnostic cell
            rows.append({"geo_id": f"SI-{year}", "geo_level": "country_detail",
                         "geo_name": COUNTRY_LABEL,
                         "source_category": cats[vdim]["label"][v], "count": int(n),
                         "basis": BASIS, "year": int(year), "source_id": SOURCE_ID,
                         "note": f"level=country_detail; code={v}; national only, fourteen "
                                 "categories, not drawn"})
    return rows


def check_detail(detail, nat):
    """The two published tables must agree to the person on every shared cell, and the
    fourteen-category one must decompose the drawn one's two catch-alls exactly."""
    d = {r["source_category"]: r["count"] for r in detail if r["year"] == YEAR}
    same = {"katoliška": "Katoliška", "pravoslavna": "Pravoslavna", "islamska": "Islamska",
            "Ni vernik, ateist": "Ni vernik, ateist",
            "Ni želel odgovoriti": "Ni želel odgovoriti", "Neznano": "Neznano",
            "Je vernik, ne pripada nobeni veroizpovedi":
                "Je vernik, vendar ne pripada nobeni veroizpovedi",
            TOTAL_LABEL: TOTAL_LABEL}
    ok = True
    for drawn_label, det_label in same.items():
        good = nat[drawn_label] == d[det_label]
        ok &= good
        print(f"    {'OK ' if good else 'BAD'} {drawn_label[:44]:<46} "
              f"{nat[drawn_label]:>9,} = {d[det_label]:,}")
    for drawn_label, parts in (
            ("evangeličanska in druge protestantske",
             ["Evangeličanska", "Druge protestantske"]),
            ("druge veroizpovedi",
             ["Druge krščanske", "Judovska", "Orientalske", "Druge veroizpovedi",
              "Agnostiki"])):
        s = sum(d[p] for p in parts)
        good = nat[drawn_label] == s
        ok &= good
        print(f"    {'OK ' if good else 'BAD'} {drawn_label[:44]:<46} {nat[drawn_label]:>9,} "
              f"= {' + '.join(f'{d[p]:,}' for p in parts)}")
    if not ok:
        raise SystemExit("the two published tables disagree -- one of them is misparsed")


def read():
    doc = _load(REL_JSON)
    ocat = doc["dimension"]["OBČINA"]["category"]
    vcat = doc["dimension"]["VEROIZPOVED"]["category"]
    vcodes = sorted(vcat["index"], key=lambda k: vcat["index"][k])
    nv = len(vcodes)
    value, status = doc["value"], doc.get("status", {})

    by_code = codes()
    lookup = {_key(n): (c, n, p) for c, (n, p) in by_code.items()}
    for census_name, older in ALIASES.items():
        hit = lookup.pop(_key(older), None)
        if hit is None:
            raise SystemExit(f"alias {older!r} is not in the settlement table any more; "
                             "SURS has re-spelt something and the join needs re-checking")
        lookup[_key(census_name)] = hit

    rows, withheld, zeros = [], {}, 0
    seen_codes = {}
    for oval, opos in sorted(ocat["index"].items(), key=lambda kv: kv[1]):
        name = ocat["label"][oval]
        if name == COUNTRY_LABEL:
            geo_id, level, pop = "SI", "country", NATIONAL
        else:
            hit = lookup.get(_key(name))
            if hit is None:
                raise SystemExit(f"no official code for {name!r} -- add it to ALIASES only "
                                 "after checking the population agrees")
            geo_id, level, pop = hit[0], "municipality", hit[2]
            if geo_id in seen_codes:
                raise SystemExit(f"code {geo_id} claimed by {seen_codes[geo_id]!r} and "
                                 f"{name!r} -- the name join has crossed two municipalities")
            seen_codes[geo_id] = name

        for vval in vcodes:
            label = vcat["label"][vval]
            idx = opos * nv + vcodes.index(vval)
            n, flag = value[idx], status.get(str(idx))
            if n is None:
                if flag == STATUS_WITHHELD:
                    withheld.setdefault(label, []).append(geo_id)
                    continue                      # withheld: not zero, and not drawn (§3.5)
                if flag == STATUS_TRUE_ZERO:
                    zeros += 1
                    n = 0
                else:
                    raise SystemExit(f"{geo_id}/{label}: null with status {flag!r}, which "
                                     "is neither `z` nor `-`")
            note = f"level={level}; code={vval}"
            if label == TOTAL_LABEL:
                note += "; universe total, not a religion category"
            elif label == DECLARED_LABEL:
                note += "; subtotal of the five religion answers, not a category"
            if n == 0 and flag == STATUS_TRUE_ZERO:
                note += "; published as 'ni pojava', i.e. a true zero"
            rows.append({"geo_id": geo_id, "geo_level": level, "geo_name": name,
                         "source_category": label, "count": int(n), "basis": BASIS,
                         "year": YEAR, "source_id": SOURCE_ID, "note": note})

        # The population check that makes the name join evidence rather than a guess.
        tot = value[opos * nv + vcodes.index("01")]
        if tot != pop:
            raise SystemExit(f"{name} ({geo_id}): religion table says {tot:,}, the "
                             f"settlement table says {pop:,} -- the join is wrong")
    return rows, withheld, zeros


def check(rows, withheld, zeros):
    ok = True
    muni = [r for r in rows if r["geo_level"] == "municipality"]
    nat = {r["source_category"]: r["count"] for r in rows if r["geo_level"] == "country"}

    n_units = len({r["geo_id"] for r in muni})
    good = n_units == EXPECTED_MUNICIPALITIES
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} {n_units} municipalities "
          f"(expected {EXPECTED_MUNICIPALITIES}); every one joined to its official code by "
          "name and confirmed on population")
    print(f"  {zeros:,} cells published as a true zero; "
          f"**{sum(len(v) for v in withheld.values()):,} withheld as confidential**")

    good = nat.get(TOTAL_LABEL) == NATIONAL
    ok &= good
    print(f"\n  {'OK ' if good else 'BAD'} national total {nat.get(TOTAL_LABEL):,} "
          f"(published {NATIONAL:,})")

    blocks = [DECLARED_LABEL, "Je vernik, ne pripada nobeni veroizpovedi",
              "Ni vernik, ateist", "Ni želel odgovoriti", "Neznano"]
    s = sum(nat[b] for b in blocks)
    good = s == NATIONAL
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} the five answer blocks partition the country "
          f"({s:,})")

    s = sum(nat[d] for d in DETAIL_LABELS)
    good = s == nat[DECLARED_LABEL]
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} the five religions partition the declared "
          f"({s:,} of {nat[DECLARED_LABEL]:,})")

    s = sum(r["count"] for r in muni if r["source_category"] == TOTAL_LABEL)
    good = s == NATIONAL
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} municipality totals sum to the country ({s:,})")

    print("\n  the fourteen-category national table against the drawn one, cell by cell:")
    check_detail([r for r in rows if r["geo_level"] == "country_detail"], nat)

    # ---- what disclosure control removes, per category (spec §3.5, §3.8) ----
    print("\n  municipality sums vs national -- the gap is disclosure control, not loss:")
    total_gap = 0
    for label in blocks[1:] + DETAIL_LABELS:
        s = sum(r["count"] for r in muni if r["source_category"] == label)
        gap = nat[label] - s
        total_gap += gap
        held = len(withheld.get(label, ()))
        bad = gap < 0
        ok &= not bad
        pct = 100.0 * gap / nat[label] if nat[label] else 0.0
        print(f"    {'BAD' if bad else 'OK '} {label[:44]:<46} {s:>9,}  "
              f"withheld {gap:>5,} ({pct:5.2f}%) in {held:>3} of {EXPECTED_MUNICIPALITIES}")
    frac = total_gap / NATIONAL
    good = frac < 0.002
    ok &= good
    print(f"    {'OK ' if good else 'BAD'} total withheld {total_gap:,} = {frac:.4%} of the "
          "population")

    print(f"\n  {len(rows):,} rows. Categories, national:")
    for label, n in sorted(nat.items(), key=lambda kv: -kv[1]):
        mark = ""
        if label == TOTAL_LABEL:
            mark = "  <- universe"
        elif label == DECLARED_LABEL:
            mark = "  <- subtotal"
        print(f"    {n:>10,}  {100.0 * n / NATIONAL:6.2f}%  {label}{mark}")

    if not ok:
        raise SystemExit("reconciliation FAILED")


def main():
    if "--fetch" in sys.argv:
        fetch()
    rows, withheld, zeros = read()
    rows += detail_rows()
    check(rows, withheld, zeros)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print("\nwrote", OUT, f"({len(rows):,} rows)")


if __name__ == "__main__":
    main()
