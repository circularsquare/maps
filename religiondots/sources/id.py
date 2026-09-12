"""Indonesia — BPS, Sensus Penduduk 2010, religion by kabupaten/kota.

Reads (or fetches) data/raw/id/ and writes data/normalized/id.csv.

237.1M people on 492 regencies with 9 religion categories. The second largest country on this
map after India, and it needed no account, no key and no application — which is the whole
finding, because sources.md §11d had written it up as blocked on a free BPS developer key
one hour before this was found. See sources.md §11e.

**THE UI HOST IS WALLED AND THE DATA HOST IS NOT.** `www.bps.go.id` returns Cloudflare's
403 to every scripted client, `webapi.bps.go.id` demands an API key, and
**`sensus.bps.go.id` asks for nothing at all**. That is §9q's Lithuania pattern for the
third time, and it is why this route was missed twice: both walls are real, and neither is
in front of the data.

**THE TRAILING PATH SEGMENT IS A FORMAT, NOT A GEOGRAPHY.** `/topik/tabular/sp2010/12/{wid}/{n}`
serves the same table as HTML for n=0,1,5, **as a PDF for n=2**, and **as JSON for n=3**.
Nothing on the page says so, and the sizes actively mislead: the national table is 133 KB at
n=2 and 5.4 MB at n=3, so n=3 reads as a finer geography. It is the same numbers, serialised
differently. See spec §12.

**AND THE URL's `wid` IS NOT THE PAYLOAD's `id_wilayah`.** They are different id spaces that
overlap, so the mistake returns HTTP 200 and real data for the wrong unit. The response for
the country gives Aceh `id_wilayah: "1675"`; requesting `wid=1675` returns **Kabupaten
Merangin, in Jambi** — and 1674 and 1676 return it too, so the space is not even injective
there. Nothing downstream can catch this, because every number is genuine and only the unit
is wrong. `wid` is therefore enumerated POSITIONALLY here and the identity of what came back
is read out of `kode_wilayah`, which is BPS's real 2/4/7-digit geographic code.

The wid space is contiguous and ordered:

    0, 1      the 34 provinces
    2 .. 35   one province each, returning ITS REGENCIES (level_wilayah 2)   <- what we want
    36 ..     one regency each, returning its kecamatan (level_wilayah 3)

`wid=25` is legitimately EMPTY and is asserted to be: it is Kalimantan Utara, split off from
Kalimantan Timur in 2012 and therefore absent from a 2010 census. A source that publishes a
unit as empty is saying something (§9p/Serbia); here the emptiness is a date check on the
whole enumeration, and if it ever fills the wid map has shifted under us.

**A PROVINCE'S CHILD LISTING IS INCOMPLETE, AND ONLY ARITHMETIC SAYS SO.** Ten of the 33
province responses omit between one and five of their own regencies — 16 units and 2,674,311
people — with no gap, no marker and no error. Sumatera Utara returns 31 consecutive-looking
rows and is missing `1273` Pematangsiantar and `1277` Padangsidimpuan; every row present is
correct, and the province's own row is correct, so nothing but the parent/child sum can see
it. The province rows sum to SP2010's published 237,641,326 **exactly**, which is what proves
the omission is in the listing rather than in the data.

The missing units ARE in the wid space, at their own wid, **carrying their kecamatan but not
their own summary row** — wid 86 holds Pematangsiantar's eight kecamatan and no `1273` row.
So each is recovered by summing its children, and the recovered figure is checked against the
province shortfall to the person: 234,698 + 191,531 = 426,229 = exactly Sumatera Utara's gap.
`--fetch-gaps` does this, scanning ONLY the blocks of provinces that fail the sum.

**AND A KECAMATAN LISTING HAS THE SAME DISEASE, so it is not trusted where anything better
exists.** `6104` sums to 193,661 against a true 234,021 and `6302` to 281,162 against
290,142 — both UNDERSTATE, which is worse than missing, because an understated unit still
draws. Where a province is missing exactly one regency the PROVINCE RESIDUAL is used
instead and is exact; kecamatan sums are kept only where several units are missing at once,
and are checked against the residual. See `_merge_gaps`.

**CATEGORIES ARE SHALLOW AND THAT IS THE STATE'S DOING, NOT THE CENSUS'S.** Indonesia
recognises six religions administratively and the census asks which one, so `Kristen`
(Protestant) vs `Katolik` is the ONLY Christian split and there is no denominational depth
to be had at any geography. `Khong Hu Chu` (Confucianism) is worth having: it was struck off
the recognised list under the New Order and restored in 2000, and 2010 is the first census
that counts it. `Lainnya` ("other") therefore carries every unrecognised tradition —
including Aliran Kepercayaan and the many local religions — compressed into one cell, and
that compression is a fact about Indonesian law rather than about Indonesian religion. It is
said in note_public.

**TWO NON-RESPONSE CATEGORIES, AND THEY MEAN OPPOSITE THINGS** (§3.5, and Serbia's pair in
§9p). `Tidak Terjawab` is "not answered" — asked and declined. `Tidak Ditanyakan` is "not
asked" — the question never reached them. They must not be merged, and neither is a religion.

Usage:
    python sources/id.py --fetch        34 GETs, ~2s apart, cached; about two minutes
    python sources/id.py --fetch-gaps   scans the failing provinces' wid blocks; ~140 GETs
    python sources/id.py --fetch-all    walks the whole regency space for the kecamatan
                                        layer; ~500 slots, most of them already cached
    python sources/id.py                normalise both tiers from data/raw/id/
    python sources/id.py --no-kecamatan regency tier only
"""

import csv
import json
import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "id")
OUT = os.path.join(ROOT, "data", "normalized", "id.csv")

SOURCE_ID = "id_sp2010"
YEAR = 2010
BASIS = "self_id"

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

BASE = "https://sensus.bps.go.id/topik/tabular/sp2010"
TABLE = 12          # Penduduk Menurut Wilayah dan Agama yang Dianut
FMT_JSON = 3        # the format segment; 2 is a PDF of the same numbers

# One request per province. Sequential, with a real pause: this is a national statistical
# office serving a public table and there is no reason to lean on it. 34 requests is a
# rounding error to them at this rate and the whole pull takes under two minutes.
WID_NATIONAL = 0
WID_PROVINCES = range(2, 36)
WID_EMPTY = 25                  # Kalimantan Utara, created 2012 -- see the docstring
REQUEST_DELAY = 2.0
MAX_RETRIES = 4

LEVEL_PROVINCE = 1
LEVEL_REGENCY = 2
LEVEL_KECAMATAN = 3

# SP2010's own category list, in the order the cube returns it. Pinned so that a reordered
# or renamed category fails the run instead of silently relabelling the map (§12/Hungary).
CATEGORIES = ["Islam", "Kristen", "Katolik", "Hindu", "Budha", "Khong Hu Chu",
              "Lainnya", "Tidak Terjawab", "Tidak Ditanyakan"]
TOTAL_CAT = "Total"

# The two crossed classifications we do NOT want split. Taking every combination would
# multiply every unit by nine and count the country nine times over.
URBAN_TOTAL = "Total"
SEX_TOTAL = "Total"

EXPECTED_PROVINCES = 33         # 34 wids, one of them (Kalimantan Utara) not yet existing
NATIONAL_POPULATION = 237_641_326    # SP2010 headline count

# A PROVINCE'S RESIDUAL IS ITSELF A MEASURED UNIT WHERE ITS MISSING CHILDREN ARE CONTIGUOUS.
#
# Kalimantan Timur's row is the 2010 province and INCLUDES the five regencies that became
# Kalimantan Utara in 2012; its listing has only the nine that remain, and the five are
# served nowhere on this host (wid=25 is empty, and the regency slots where their codes sort
# — 397 and 401..405 — are holes). That was written off as an unrecoverable 0.22%, and it
# left a conspicuous hole in the north of Borneo.
#
# It is recoverable, and no other source is needed. The residual — province minus its nine
# listed regencies, PER CATEGORY — is 524,656 people that sum to the published Total
# exactly, are non-negative in every category, and match Kalimantan Utara's documented 2010
# population to the person. Those five regencies were carved wholly out of Kalimantan Timur
# and form one contiguous block, so the residual is not a scattering of unrelated places: it
# is exactly the territory of the modern province, and drawing it as ONE unit asserts
# nothing that was not measured. This is `_merge_gaps`'s residual rule (see there) applied
# one level up, where the missing children are a province's worth rather than a single
# regency's.
#
# What it costs: this unit is coarser than the rest of the map — one polygon for 524,656
# people where Indonesia is otherwise drawn at kecamatan — so it is the third geo_level in
# the drawn tier and is marked as such. What it buys: 100% of SP2010 rather than 99.78%,
# and no hole.
PROVINCE_RESIDUAL = {
    "64": ("65", "KALIMANTAN UTARA",
           "the five regencies that became Kalimantan Utara in 2012, which BPS serves under "
           "neither province; recovered as Kalimantan Timur's per-category residual and "
           "drawn as one unit over their combined territory"),
}
RESIDUAL_MEMBERS = {"65": ["6501", "6502", "6503", "6504", "6571"]}

# THE ONE THING THIS ROUTE CANNOT DELIVER, and it is a vintage collision rather than a bug.
# Kalimantan Timur's province row is the 2010 province, 3,553,143 — which INCLUDES the five
# regencies that became Kalimantan Utara in 2012 (Malinau, Bulungan, Tana Tidung, Nunukan
# and Kota Tarakan). Its regency listing has only the nine that remain in the province
# today. The five are not under Kalimantan Utara either: wid=25, the province slot, is
# empty, and the regency slots where their codes would sort — wids 397 and 401..405 — are
# holes. BPS is serving a 2010 census through a post-2012 geography and these units fell
# between the two, so they exist in no response on this host.
#
# 524,656 people, 0.22% of Indonesia. Documented, subtracted from the target, and asserted
# to be EXACTLY this: if the figure ever moves, the assumption behind it has changed.
UNRECOVERABLE = {}      # emptied 2026-09-06: see PROVINCE_RESIDUAL above
UNRECOVERABLE_NOTE = ("the five regencies that became Kalimantan Utara in 2012; BPS serves "
                      "SP2010 through a post-2012 geography and they are in neither province")
EXPECTED_REGENCIES = 493        # 492 kabupaten/kota, plus the Kaltara residual unit


def _ua():
    return {"User-Agent": ("religiondots/1.0 (map research; contact via "
                           "github.com/anitagarden) python-requests")}


def _get_json(session, wid):
    """One polite GET. Retries with backoff; raises on anything that is not usable JSON."""
    url = f"{BASE}/{TABLE}/{wid}/{FMT_JSON}"
    delay = 3.0
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.get(url, timeout=180, headers=_ua())
        except Exception as exc:                       # noqa: BLE001 - report and retry
            if attempt == MAX_RETRIES:
                raise SystemExit(f"wid={wid}: {type(exc).__name__} after "
                                 f"{MAX_RETRIES} attempts -- {exc}")
            print(f"    wid={wid} {type(exc).__name__}, retrying in {delay:.0f}s")
            time.sleep(delay)
            delay *= 2
            continue
        if r.status_code != 200:
            if attempt == MAX_RETRIES:
                raise SystemExit(f"wid={wid}: HTTP {r.status_code} after "
                                 f"{MAX_RETRIES} attempts")
            print(f"    wid={wid} HTTP {r.status_code}, retrying in {delay:.0f}s")
            time.sleep(delay)
            delay *= 2
            continue
        # §5a: HTTP 200 is not a download. This host serves a PDF from the neighbouring
        # path and HTML from three others, so assert we got the JSON we asked for.
        text = r.text.lstrip()
        if not text.startswith("{"):
            raise SystemExit(f"wid={wid}: expected JSON, got {text[:60]!r}")
        doc = json.loads(text)
        if doc.get("status") != 200 or "data" not in doc:
            raise SystemExit(f"wid={wid}: unexpected payload keys {list(doc)}")
        return doc
    raise SystemExit(f"wid={wid}: unreachable")


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    session = requests.Session()

    wids = [WID_NATIONAL] + list(WID_PROVINCES)
    for i, wid in enumerate(wids):
        dest = os.path.join(RAW, f"wid_{wid:02d}.json")
        if os.path.exists(dest) and os.path.getsize(dest) > 200:
            print(f"  [{i+1}/{len(wids)}] wid={wid:<3} cached")
            continue
        if i:
            time.sleep(REQUEST_DELAY)
        doc = _get_json(session, wid)
        n = doc.get("data_count", len(doc.get("data", [])))
        _save_json(dest, doc)
        print(f"  [{i+1}/{len(wids)}] wid={wid:<3} {n:>6,} rows  "
              f"{os.path.getsize(dest):>10,} bytes")


def _load(wid):
    p = os.path.join(RAW, f"wid_{wid:02d}.json")
    if not os.path.exists(p):
        raise SystemExit(f"missing {p} -- run with --fetch first")
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


# --- the gap stage -------------------------------------------------------------------
#
# The regency wid space starts at WID_REGENCY_BASE and runs in ascending kode_wilayah
# order across the whole country. It is NOT densely packed: three shapes occur.
#
#   normal   a level-2 row for the regency plus its kecamatan at level 3
#   gap      kecamatan at level 3 and NO level-2 row -- the units the province listing
#            omits, which is what this stage exists to recover
#   hole     an empty response, no rows at all. wid=173 sits between Lampung's 1812 and
#            its 1871 and returns nothing.
#
# The holes are why block starts cannot be accumulated from the province listings: a
# province that reconciles perfectly may still contain a hole, so its block is longer than
# its row count and every later block start computed from those counts drifts. An earlier
# version of this stage did exactly that and walked into the wrong province.
#
# The space IS ordered by code, but only up to its end -- somewhere past Papua. **Beyond
# that, wids do not 404: they return other provinces' data.** wid 609, 705, 753 and 800 all
# answer with Aceh codes, so a binary search whose upper bound lands in that region reads a
# small code, concludes the target is further right, and walks off the end. That is how an
# earlier version of this stage "found" province 73 at wid=800 holding Aceh's 1117.
#
# So the block start is found by walking FORWARD from the nearest anchor already in the
# cache instead. Every slot ever fetched is cached with its code, which makes the cache an
# index: the walk starts at the highest cached wid whose code still sorts below the target
# province and steps forward from there, so re-running costs nothing and the first run
# never probes a wid it cannot interpret.

WID_REGENCY_BASE = 36
WID_WALK_CAP = 900           # the ordered space is ~500 long; a runaway walk stops here
MAX_TRAILING_HOLES = 12      # a run this long ends the space rather than sitting inside it
GAP_PREFIX = "gap_wid_"


def _gap_path(wid):
    return os.path.join(RAW, f"{GAP_PREFIX}{wid:05d}.json")


def _save_json(path, doc):
    """Write through a temp file and rename.

    These responses run to several MB and the scan is long enough that it gets interrupted;
    a plain `open(...,"w")` killed mid-write leaves a truncated file that looks cached and
    then fails the NEXT run's parse, several hundred requests later. `os.replace` is atomic
    on Windows and POSIX alike, so a cached file is either complete or absent.
    """
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False)
    os.replace(tmp, path)


def _units_in(doc, level):
    out = {}
    for r in doc.get("data", []):
        if r.get("level_wilayah") == level:
            out.setdefault(r["kode_wilayah"], r["nama_wilayah"])
    return out


def _slot(session, wid):
    """(doc, regency code, has own level-2 row) for one wid. Cached; None code is a hole."""
    path = _gap_path(wid)
    if os.path.exists(path) and os.path.getsize(path) > 20:
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
    else:
        time.sleep(REQUEST_DELAY)
        doc = _get_json(session, wid)
        _save_json(path, doc)
    lvl2 = _units_in(doc, LEVEL_REGENCY)
    lvl3 = _units_in(doc, LEVEL_KECAMATAN)
    codes = set(lvl2) | {k[:4] for k in lvl3}
    if len(codes) > 1:
        raise SystemExit(f"wid={wid} spans regencies {sorted(codes)}")
    return doc, (codes.pop() if codes else None), bool(lvl2)


def fetch_all():
    """Walk the whole regency wid space, caching every slot — the kecamatan pull.

    The space is ordered by code, so **its end is found by monotonicity, not by a count**:
    past the last regency the wids do not 404, they serve other provinces' data (wid 609
    answers with Aceh). The first code that sorts BELOW the highest one seen is therefore
    the boundary, and everything from there on is the junk region. Holes inside the space
    are tolerated; a long run of them is also an ending.

    Every slot is cached, so this resumes for free and the ~300 slots the gap scan already
    fetched cost nothing.
    """
    import requests

    os.makedirs(RAW, exist_ok=True)
    session = requests.Session()
    wid, high, holes, n_new = WID_REGENCY_BASE, "", 0, 0
    seen = {}

    while wid < WID_REGENCY_BASE + WID_WALK_CAP:
        cached = os.path.exists(_gap_path(wid)) and os.path.getsize(_gap_path(wid)) > 20
        doc, code, _ = _slot(session, wid)
        if not cached:
            n_new += 1
        if code is None:
            holes += 1
            if holes >= MAX_TRAILING_HOLES:
                print(f"  stopped at wid={wid}: {holes} holes in a row")
                break
            wid += 1
            continue
        if code < high:
            print(f"  stopped at wid={wid}: code {code} sorts below {high}, so this is "
                  "past the end of the ordered space")
            break
        holes = 0
        high = max(high, code)
        seen[code] = wid
        if len(seen) % 50 == 0 and not cached:
            print(f"  ... {len(seen)} regencies, at {code} (wid {wid})")
        wid += 1
    else:
        raise SystemExit(f"walked {WID_WALK_CAP} wids without finding the end of the space")

    print(f"walked wid {WID_REGENCY_BASE}..{wid-1}: {len(seen)} regencies, "
          f"{n_new} newly fetched")
    provinces = {c[:2] for c in seen}
    print(f"  {len(provinces)} provinces: {' '.join(sorted(provinces))}")
    return seen


def fetch_gaps():
    import requests

    prov_rows, reg_rows, _ = read(gaps=False, strict=False)
    prov, reg = _tidy(prov_rows), _tidy(reg_rows)

    # Group by the REGENCY data, not by the province rows: wid=2 (Aceh) returns regencies
    # and no province row at all, so a province-keyed walk skips its block entirely and
    # every later block start is 23 too low.
    per_prov = {}
    for code, e in reg.items():
        per_prov.setdefault(e["prov"], []).append(code)
    for kids in per_prov.values():
        kids.sort()

    # Which provinces do not reconcile, and by how many people. A province with no row of
    # its own cannot be checked and is assumed complete -- it is not scanned, so a wrong
    # assumption here shows up as a block misalignment in the NEXT province and stops it.
    failing = {}
    for pcode, kids in per_prov.items():
        pe = prov.get(pcode)
        if pe is None:
            continue
        short = pe["cats"][TOTAL_CAT] - sum(reg[k]["cats"][TOTAL_CAT] for k in kids)
        if short:
            failing[pcode] = short
    order = sorted(per_prov)
    print(f"{len(failing)} provinces short by {sum(failing.values()):,} people: "
          f"{', '.join(sorted(failing))}")

    session = requests.Session()
    os.makedirs(RAW, exist_ok=True)
    recovered = 0

    def slot(wid):
        return _slot(session, wid)

    # Index the cache ONCE. These responses run to several MB each and there are hundreds
    # of them, so re-reading the directory per province is minutes of pure JSON parsing.
    def _index():
        import glob
        import re as _re
        idx = {}
        for path in sorted(glob.glob(os.path.join(RAW, f"{GAP_PREFIX}*.json"))):
            wid = int(_re.search(rf"{GAP_PREFIX}(\d+)", path).group(1))
            with open(path, encoding="utf-8") as fh:
                doc = json.load(fh)
            lvl2 = _units_in(doc, LEVEL_REGENCY)
            lvl3 = _units_in(doc, LEVEL_KECAMATAN)
            codes = set(lvl2) | {k[:4] for k in lvl3}
            idx[wid] = codes.pop() if len(codes) == 1 else None
        return idx

    print(f"  indexing {RAW} ...")
    index = _index()

    # ONLY THE ORDERED PREFIX OF THE CACHE MAY BE USED AS AN ANCHOR. Past the end of the
    # space the wids return other provinces' data, so the cache contains low codes at high
    # wids -- wid 839 answers with Aceh's 1201. "Highest cached wid whose code sorts below
    # the target" then picks a junk slot and the walk starts 400 wids too far right. So the
    # index is truncated at the first place the codes stop being non-decreasing, and that
    # boundary is where the real space ends.
    ordered, last = {}, ""
    for wid in sorted(index):
        code = index[wid]
        if code is None:
            ordered[wid] = None          # a hole inside the space is fine
            continue
        if code < last:
            break                        # monotonicity broke: this is past the end
        ordered[wid] = code
        last = code
    print(f"  {len(index)} slots cached, {len(ordered)} of them in the ordered space "
          f"(wid <= {max(ordered) if ordered else WID_REGENCY_BASE})")

    def cached_anchor(pcode):
        """Highest ORDERED cached wid whose code sorts below this province, else the base."""
        best = WID_REGENCY_BASE
        for wid, code in ordered.items():
            if code is not None and code < pcode + "00" and wid > best:
                best = wid
        return best

    for pcode in order:
        if pcode not in failing:
            continue

        # Walk forward from the nearest cached anchor to the province's first slot.
        start = cached_anchor(pcode)
        steps = 0
        while True:
            _, code, _ = slot(start)
            if code is not None and code[:2] == pcode:
                break
            if code is not None and code[:2] > pcode:
                raise SystemExit(f"province {pcode}: walked past it to {code} at "
                                 f"wid={start} without ever entering the province")
            start += 1
            steps += 1
            if steps > 400:
                raise SystemExit(f"province {pcode}: no block found within 400 wids of "
                                 f"the anchor -- the wid space has been repacked")
        first = code
        print(f"  province {pcode}: block starts at wid={start} ({first})")

        want = set(per_prov[pcode])
        seen, found, holes, wid = set(), {}, 0, start
        while True:
            _, code, has_parent = slot(wid)
            if code is None:                          # a hole; keep going, but not forever
                holes += 1
                if holes > 8:
                    break
                wid += 1
                continue
            if code[:2] != pcode:
                break                                 # walked into the next province
            seen.add(code)
            if not has_parent:                        # the gap shape
                found[code] = wid
            wid += 1
            if wid - start > 200:
                raise SystemExit(f"province {pcode}: block did not end within 200 wids")

        # The block must contain everything the province listing already had. If it does
        # not, the search landed in the wrong place and the recovered units would be
        # attached to the wrong parents -- which no downstream check could see.
        if not want <= seen:
            raise SystemExit(
                f"province {pcode}: block from wid={start} yielded {len(seen)} codes but "
                f"the province listing has {sorted(want - seen)} that the block does not")
        missing = seen - want
        print(f"    wid {start}..{wid-1}; recovered {sorted(missing)} "
              f"at wid {[found.get(c) for c in sorted(missing)]}")
        if missing != set(found):
            raise SystemExit(f"province {pcode}: codes missing from the province listing "
                             f"{sorted(missing)} but gap wids found for {sorted(found)}")
        recovered += len(missing)

    print(f"recovered {recovered} regencies into {RAW}")


def read(gaps=True, strict=True):
    """One row per (regency, category). Provinces are kept for the reconciliation only.

    `gaps` folds in the regencies recovered from `--fetch-gaps` — the ones their province's
    listing omits, rebuilt by summing their kecamatan. `strict` is what `--fetch-gaps`
    itself turns off, because it has to read the incomplete state in order to repair it.
    """
    prov_rows, reg_rows = {}, {}
    seen_wid = {}

    for wid in WID_PROVINCES:
        doc = _load(wid)
        data = doc.get("data", [])
        if wid == WID_EMPTY:
            if data:
                raise SystemExit(
                    f"wid={WID_EMPTY} returned {len(data)} rows but is expected to be "
                    "empty (Kalimantan Utara, created 2012). The wid map has shifted.")
            continue
        if not data:
            raise SystemExit(f"wid={wid} returned no rows; only wid={WID_EMPTY} may be empty")

        # Identity comes out of the PAYLOAD, never out of the wid -- see the docstring.
        kodes = sorted({r["kode_wilayah"] for r in data
                        if r.get("level_wilayah") == LEVEL_REGENCY})
        if not kodes:
            raise SystemExit(f"wid={wid} returned no level-{LEVEL_REGENCY} rows")
        prefixes = {k[:2] for k in kodes}
        if len(prefixes) != 1:
            raise SystemExit(f"wid={wid} mixes provinces {sorted(prefixes)} -- the wid "
                             "space is not one province per request as assumed")
        prov_code = prefixes.pop()
        if prov_code in seen_wid:
            raise SystemExit(f"province {prov_code} returned by both wid={seen_wid[prov_code]} "
                             f"and wid={wid} -- the wid space is not injective")
        seen_wid[prov_code] = wid

        for r in data:
            lvl = r.get("level_wilayah")
            if lvl not in (LEVEL_PROVINCE, LEVEL_REGENCY):
                continue
            if r.get("nama_item__kategori_2") != URBAN_TOTAL:
                continue
            if r.get("nama_item__kategori_3") != SEX_TOTAL:
                continue
            cat = r.get("nama_item__kategori_1")
            code = r["kode_wilayah"]
            name = r["nama_wilayah"]
            val = r.get("nilai")
            if val is None:
                continue
            bucket = prov_rows if lvl == LEVEL_PROVINCE else reg_rows
            key = (code, cat)
            if key in bucket and bucket[key][2] != int(val):
                raise SystemExit(f"{code}/{cat} appears twice with different values "
                                 f"{bucket[key][2]:,} and {int(val):,}")
            bucket[key] = (name, prov_code, int(val))

    if gaps:
        _merge_gaps(prov_rows, reg_rows)
        _merge_province_residuals(prov_rows, reg_rows)

    if strict and not reg_rows:
        raise SystemExit("no regency rows at all")
    return prov_rows, reg_rows, seen_wid


def _merge_gaps(prov_rows, reg_rows):
    """Fold the omitted regencies in, preferring the province residual to a kecamatan sum.

    Two ways to rebuild a unit the province listing drops, and they are not equally good:

      * **the province residual** — the province's own row minus its listed regencies. This
        is EXACT, because the province row is exact (they sum to SP2010's national figure to
        the person). It only works where a province is missing exactly ONE regency, since
        otherwise the residual is a lump covering several.
      * **summing the unit's kecamatan** — works for any number of missing units, but is
        only as complete as the kecamatan listing, and that listing has the same disease as
        the province one. `6104` Pontianak sums to 193,661 against a true 234,021, and
        `6302` Kotabaru to 281,162 against 290,142: both understate, which is worse than
        missing, because an understated unit still draws and draws wrong.

    So: residual where a single unit is missing, kecamatan sums where several are, and the
    kecamatan sums are then CHECKED against the residual so a shortfall is caught rather
    than drawn. Categories absent from a kecamatan sum are zero-filled — a category with no
    adherents in any kecamatan simply has no row, and its absence is not missing data.
    """
    gaps = {}
    for code, cat, prov_code, n in _read_gaps():
        gaps.setdefault(code, {"prov": prov_code, "cats": {}})["cats"][cat] = n
    if not gaps:
        return

    all_cats = CATEGORIES + [TOTAL_CAT]
    by_prov = {}
    for code, e in gaps.items():
        by_prov.setdefault(e["prov"], []).append(code)

    for pcode, codes in sorted(by_prov.items()):
        residual = {}
        for cat in all_cats:
            p = prov_rows.get((pcode, cat))
            if p is None:
                raise SystemExit(f"province {pcode} has no {cat!r} row to take a residual "
                                 "from")
            listed = sum(n for (c, k), (_, pr, n) in reg_rows.items()
                         if k == cat and pr == pcode)
            residual[cat] = p[2] - listed

        if len(codes) == 1:
            code = codes[0]
            summed = gaps[code]["cats"].get(TOTAL_CAT, 0)
            if summed != residual[TOTAL_CAT]:
                print(f"  note {code}: kecamatan sum {summed:,} but the province residual "
                      f"is {residual[TOTAL_CAT]:,}; using the residual, which is exact")
            cats = residual
        else:
            cats = {}
            for cat in all_cats:
                cats[cat] = sum(gaps[c]["cats"].get(cat, 0) for c in codes)
            for cat in all_cats:
                if cats[cat] != residual[cat]:
                    raise SystemExit(
                        f"province {pcode}: {len(codes)} recovered regencies sum to "
                        f"{cats[cat]:,} for {cat!r} but the residual is {residual[cat]:,} "
                        "-- the kecamatan listings are incomplete and cannot be split")
            cats = None                     # per-unit values are already right

        for code in codes:
            src = residual if (cats is not None) else gaps[code]["cats"]
            for cat in all_cats:
                key = (code, cat)
                if key in reg_rows:
                    raise SystemExit(f"{code} was recovered but is also in its province's "
                                     "listing -- the gap scan is stale")
                reg_rows[key] = (f"[{code}]", pcode, int(src.get(cat, 0)))


def _ordered_slots():
    """{wid: regency code} for the ORDERED prefix of the cached wid space.

    Past the end of the space the wids serve other provinces' data, so a cached directory
    listing is not the same thing as the space. The prefix is cut at the first code that
    sorts below the highest one seen — the same boundary `fetch_all` walks to.
    """
    import glob
    import re as _re

    slots = {}
    for path in glob.glob(os.path.join(RAW, f"{GAP_PREFIX}*.json")):
        wid = int(_re.search(rf"{GAP_PREFIX}(\d+)", path).group(1))
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        lvl2 = _units_in(doc, LEVEL_REGENCY)
        lvl3 = _units_in(doc, LEVEL_KECAMATAN)
        codes = set(lvl2) | {k[:4] for k in lvl3}
        slots[wid] = codes.pop() if len(codes) == 1 else None

    out, high = {}, ""
    for wid in sorted(slots):
        code = slots[wid]
        if code is None:
            out[wid] = None
            continue
        if code < high:
            break
        out[wid] = code
        high = max(high, code)
    return out


def read_kecamatan():
    """One row per (kecamatan, category), from every slot in the ordered space.

    The kecamatan are the level-3 rows the regency slots already carry, so this costs no
    request beyond `--fetch-all`. Identity comes from `kode_wilayah` as everywhere else:
    seven digits, of which the first four are the 2010 regency — which is exactly what makes
    this layer able to rebuild the 2010 regency geography from a 2020 boundary file.
    """
    slots = _ordered_slots()
    rows, per_regency = {}, {}

    for wid in sorted(slots):
        if slots[wid] is None:
            continue
        path = _gap_path(wid)
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        for r in doc.get("data", []):
            if r.get("level_wilayah") != LEVEL_KECAMATAN:
                continue
            if r.get("nama_item__kategori_2") != URBAN_TOTAL:
                continue
            if r.get("nama_item__kategori_3") != SEX_TOTAL:
                continue
            val = r.get("nilai")
            if val is None:
                continue
            code, cat, name = r["kode_wilayah"], r["nama_item__kategori_1"], r["nama_wilayah"]
            if len(code) != 7:
                raise SystemExit(f"kecamatan code {code!r} is not 7 digits")
            key = (code, cat)
            prev = rows.get(key)
            if prev is not None and prev[2] != int(val):
                raise SystemExit(f"{code}/{cat}: {prev[2]:,} vs {int(val):,}")
            rows[key] = (name, code[:4], int(val))
            per_regency.setdefault(code[:4], set()).add(code)

    return rows, per_regency


def check_kecamatan(kec, per_regency, reg):
    """Every kecamatan sums to its regency, or the shortfall is named.

    The regency figures are the authoritative ones — for sixteen units they came from a
    province residual rather than from any child listing — so this is a real check and not a
    tautology. Where a regency's kecamatan listing is short, the kecamatan LAYER is
    understated for that unit and the difference is reported rather than absorbed.
    """
    ok = True
    print(f"  {len(per_regency)} regencies carry {sum(len(v) for v in per_regency.values()):,}"
          f" kecamatan")

    tot = {}
    for (code, cat), (_, parent, n) in kec.items():
        if cat == TOTAL_CAT:
            tot[parent] = tot.get(parent, 0) + n

    short, missing = [], []
    for code, e in sorted(reg.items()):
        want = e["cats"][TOTAL_CAT]
        got = tot.get(code)
        if got is None:
            missing.append((code, want))
        elif got != want:
            short.append((code, want, got))

    if missing:
        print(f"\n  REGENCIES WITH NO KECAMATAN AT ALL ({len(missing)}), "
              f"{sum(w for _, w in missing):,} people:")
        for code, want in missing:
            print(f"    {code}  {want:>10,}")
    if short:
        print(f"\n  REGENCIES WHOSE KECAMATAN DO NOT SUM TO THEM ({len(short)}):")
        for code, want, got in short:
            print(f"    {code}  regency {want:>10,}  kecamatan {got:>10,}  "
                  f"{got - want:+,}")

    drawn = sum(tot.values())
    reg_total = sum(e["cats"][TOTAL_CAT] for e in reg.values())
    print(f"\n  kecamatan total {drawn:,} against the regency layer's {reg_total:,} "
          f"({drawn - reg_total:+,})")
    print(f"  that is {100 * drawn / NATIONAL_POPULATION:.2f}% of SP2010's "
          f"{NATIONAL_POPULATION:,}")
    return ok


def _merge_province_residuals(prov_rows, reg_rows):
    """Emit a province's leftover as one drawn unit — see PROVINCE_RESIDUAL.

    Only where the missing children are CONTIGUOUS and known, which is the whole condition:
    a residual scattered over unrelated places would be a lump with no shape and could not
    be drawn at all. Kalimantan Utara's five regencies were carved wholly out of Kalimantan
    Timur, so the residual is exactly one province's territory.

    The result is checked the way every other reconstruction here is: categories must sum to
    the residual Total, and no category may be negative — a negative residual would mean the
    listing exceeded its own parent, which is the shape a double-counted child makes.
    """
    all_cats = CATEGORIES + [TOTAL_CAT]
    for pcode, (code, name, _why) in sorted(PROVINCE_RESIDUAL.items()):
        resid = {}
        for cat in all_cats:
            parent = prov_rows.get((pcode, cat))
            if parent is None:
                raise SystemExit(f"province {pcode} has no {cat!r} row")
            listed = sum(n for (c, k), (_, pr, n) in reg_rows.items()
                         if k == cat and pr == pcode)
            resid[cat] = parent[2] - listed

        neg = {c: v for c, v in resid.items() if v < 0}
        if neg:
            raise SystemExit(f"province {pcode} residual is negative for {neg} -- its "
                             "listing exceeds its own parent")
        s = sum(resid[c] for c in CATEGORIES)
        if s != resid[TOTAL_CAT]:
            raise SystemExit(f"province {pcode} residual categories sum to {s:,} but its "
                             f"residual Total is {resid[TOTAL_CAT]:,}")
        if resid[TOTAL_CAT] == 0:
            continue

        for cat in all_cats:
            key = (code, cat)
            if key in reg_rows:
                raise SystemExit(f"{code} already exists; the residual would double it")
            reg_rows[key] = (name, pcode, int(resid[cat]))
        print(f"  province {pcode}: residual {resid[TOTAL_CAT]:,} people drawn as "
              f"unit {code} ({name})")


def _read_gaps():
    """Rebuild each omitted regency by summing its kecamatan.

    The gap responses carry level-3 rows and no level-2 row, so the regency's own figure
    does not exist in the source and is derived. Its NAME does not exist either — the
    kecamatan carry their own names — so it is left to `id_geo.py` to supply from the
    boundary file, and a placeholder is written here rather than a guess.
    """
    import glob

    # The cache holds every slot the scan touched, in all three shapes. Only the gap shape
    # -- kecamatan present, no level-2 row -- is a unit to rebuild; a normal slot is
    # already in its province's listing and a hole has nothing in it.
    for path in sorted(glob.glob(os.path.join(RAW, f"{GAP_PREFIX}*.json"))):
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        if _units_in(doc, LEVEL_REGENCY):
            continue                                  # normal slot
        if not _units_in(doc, LEVEL_KECAMATAN):
            continue                                  # hole

        # kecamatan -> the regency they belong to, taken from their own codes
        totals, kec = {}, {}
        for r in doc.get("data", []):
            if r.get("level_wilayah") != LEVEL_KECAMATAN:
                continue
            if r.get("nama_item__kategori_2") != URBAN_TOTAL:
                continue
            if r.get("nama_item__kategori_3") != SEX_TOTAL:
                continue
            val = r.get("nilai")
            if val is None:
                continue
            cat = r["nama_item__kategori_1"]
            code = r["kode_wilayah"]
            # rows are duplicated in some responses; identical values are fine, differing
            # ones are not
            prev = kec.get((code, cat))
            if prev is not None and prev != int(val):
                raise SystemExit(f"{code}/{cat}: {prev:,} vs {int(val):,} in one response")
            kec[(code, cat)] = int(val)

        for (code, cat), n in kec.items():
            totals[cat] = totals.get(cat, 0) + n

        parents = {c[:4] for c, _ in kec}
        if len(parents) != 1:
            raise SystemExit(f"{os.path.basename(path)} spans regencies {sorted(parents)}")
        parent = parents.pop()

        # The derived Total must equal the derived category sum, or the reconstruction is
        # not internally consistent and nothing downstream should trust it.
        s = sum(totals[c] for c in CATEGORIES if c in totals)
        if TOTAL_CAT in totals and s != totals[TOTAL_CAT]:
            raise SystemExit(f"{parent}: kecamatan categories sum to {s:,} but their "
                             f"Total sums to {totals[TOTAL_CAT]:,}")

        for cat, n in totals.items():
            yield parent, cat, parent[:2], n


def _tidy(rows):
    """{(code, cat): (name, prov, n)} -> {code: {"name":…, "prov":…, "cats":{cat:n}}}."""
    out = {}
    for (code, cat), (name, prov, n) in rows.items():
        e = out.setdefault(code, {"name": name, "prov": prov, "cats": {}})
        e["cats"][cat] = n
    return out


def check(prov, reg, seen_wid):
    ok = True
    provinces = {e["prov"] for e in reg.values()}

    if len(provinces) != EXPECTED_PROVINCES:
        print(f"  FAIL {len(provinces)} provinces, expected {EXPECTED_PROVINCES}")
        ok = False
    if len(reg) != EXPECTED_REGENCIES:
        print(f"  FAIL {len(reg)} regencies, expected {EXPECTED_REGENCIES}")
        ok = False

    # Every regency carries the full category list plus the published Total, and its
    # categories sum to that Total exactly. This is the parse check and the suppression
    # check at once -- SP2010 publishes no sentinels, so any shortfall is a read error.
    want = set(CATEGORIES) | {TOTAL_CAT}
    for code, e in sorted(reg.items()):
        got = set(e["cats"])
        if got != want:
            print(f"  FAIL {code} {e['name']}: categories {sorted(got ^ want)} differ")
            ok = False
            continue
        s = sum(e["cats"][c] for c in CATEGORIES)
        if s != e["cats"][TOTAL_CAT]:
            print(f"  FAIL {code} {e['name']}: categories sum to {s:,}, "
                  f"published total {e['cats'][TOTAL_CAT]:,}")
            ok = False

    # A province's regencies partition it. The province row comes from the SAME response,
    # so this is weaker than an independent check -- but it is free, and it is what
    # notices a dropped or duplicated regency row.
    for pcode, pe in sorted(prov.items()):
        kids = [e for e in reg.values() if e["prov"] == pcode]
        if not kids:
            continue
        s = sum(e["cats"][TOTAL_CAT] for e in kids)
        # A province with a residual unit needs no allowance: the residual is emitted with
        # that province as its parent, so it is already inside this sum and the province
        # closes exactly. That is the point of keeping it under `64` rather than under the
        # province it geographically became -- the arithmetic stays checkable.
        allowed = UNRECOVERABLE.get(pcode, 0)
        gap = pe["cats"].get(TOTAL_CAT) - s
        if gap != allowed:
            print(f"  FAIL province {pcode} {pe['name']}: {len(kids)} regencies sum to "
                  f"{s:,}, province row is {pe['cats'].get(TOTAL_CAT):,} "
                  f"(gap {gap:,}, allowed {allowed:,})")
            ok = False
        elif allowed:
            print(f"  province {pcode} {pe['name']} is short by exactly {allowed:,} -- "
                  f"{UNRECOVERABLE_NOTE}")

    # The province rows sum to SP2010's published figure to the person, so the regencies
    # must too. This is an EXACT check rather than a tolerance: the whole point of the gap
    # stage is that a shortfall here is a missing unit, and a tolerance would hide exactly
    # the failure the stage exists to catch.
    national = sum(e["cats"][TOTAL_CAT] for e in reg.values())
    target = NATIONAL_POPULATION - sum(UNRECOVERABLE.values())
    print(f"  regencies {len(reg):,} in {len(provinces)} provinces, {national:,} people")
    if national != target:
        print(f"  FAIL national total {national:,} != {target:,} "
              f"({national - target:+,})")
        ok = False
    else:
        pct = 100 * national / NATIONAL_POPULATION
        print(f"  national total matches SP2010's {NATIONAL_POPULATION:,} less the "
              f"{sum(UNRECOVERABLE.values()):,} documented above, exactly -- {pct:.2f}% drawn")

    for c in CATEGORIES:
        n = sum(e["cats"][c] for e in reg.values())
        print(f"    {c:<18} {n:>13,}  {100*n/national:6.3f}%")
    return ok


NOTES = {
    "Tidak Terjawab": "non-response: asked and not answered; not a religion",
    "Tidak Ditanyakan": "non-response: the question was not asked; not a religion",
    "Lainnya": ("every tradition outside the six the state recognises, including Aliran "
                "Kepercayaan and the local religions, compressed into one cell by law"),
    "Khong Hu Chu": ("Confucianism; de-recognised under the New Order and restored in "
                     "2000, so 2010 is the first census that counts it"),
}


def write(reg, kec=None):
    """Both tiers into one file, with the DRAWN tier picked per parent (spec §12).

    Ghana's shape (§9n), decided per unit rather than by rule: a regency's kecamatan
    **replace** it where they sum to it exactly, and are set aside where they do not. So the
    four `geo_level` values mean different things and only two of them are drawn:

        kecamatan          the finest measured unit -- DRAWN
        regency            a regency whose kecamatan are incomplete -- DRAWN in their place
        regency_covered    a regency its kecamatan already cover -- record only
        kecamatan_partial  kecamatan of an incomplete regency -- record only, and NOT drawn
                           because their siblings are missing, so drawing them would put a
                           unit's whole population into part of it

    The drawn tier is therefore `kecamatan` + `regency`, which is disjoint, covers the
    country once, and is 99.78% of SP2010 with **every row still measured**. The two
    record-only levels are kept because they are real readings and a later source may fill
    the gaps; nothing downstream should draw them.
    """
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    # Sum the kecamatan per parent PER CATEGORY, not just the Total.
    #
    # RECONCILING ON THE TOTAL IS NOT RECONCILING. Nduga (9429) in Papua publishes eight
    # kecamatan that carry a `Total` row and NO religion categories at all: the Totals sum
    # to the regency exactly, so a Total-only test calls it complete, and 79,053 Kristen
    # would then be drawn as nothing at all — a unit present on the map with no religion in
    # it. Every category has to match, and that is what promotes Nduga to the regency tier
    # where its Kristen figure is real.
    kec_sum = {}
    for (code, cat), (_, parent, n) in (kec or {}).items():
        kec_sum.setdefault(parent, {})[cat] = kec_sum.get(parent, {}).get(cat, 0) + n

    all_cats = CATEGORIES + [TOTAL_CAT]
    complete = {code for code, e in reg.items()
                if code in kec_sum
                and all(kec_sum[code].get(c, 0) == e["cats"][c] for c in all_cats)}

    n, drawn_units, drawn_people = 0, 0, 0
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()

        for code, e in sorted(reg.items()):
            covered = code in complete
            level = ("province_residual" if code in RESIDUAL_MEMBERS
                     else "regency_covered" if covered else "regency")
            if not covered:
                drawn_units += 1
                drawn_people += e["cats"][TOTAL_CAT]
            for cat in CATEGORIES + [TOTAL_CAT]:
                note = f"province={e['prov']}"
                if cat == TOTAL_CAT:
                    note += "; universe total, not a religion category"
                elif cat in NOTES:
                    note += "; " + NOTES[cat]
                if covered:
                    note += ("; NOT DRAWN -- this unit's kecamatan sum to it exactly, in "
                             "every category, and are drawn in its place")
                elif code in kec_sum:
                    gap = e["cats"][TOTAL_CAT] - kec_sum[code].get(TOTAL_CAT, 0)
                    why = (f"is {gap:,} short" if gap
                           else "reconciles on the total but not on every category")
                    note += f"; DRAWN at this level -- its kecamatan listing {why}"
                else:
                    note += "; DRAWN at this level -- no kecamatan listing at all"
                w.writerow({"geo_id": code, "geo_level": level,
                            "geo_name": e["name"], "source_category": cat,
                            "count": e["cats"][cat], "basis": BASIS, "year": YEAR,
                            "source_id": SOURCE_ID, "note": note})
                n += 1

        for (code, cat), (name, parent, count) in sorted((kec or {}).items()):
            covered = parent in complete
            level = "kecamatan" if covered else "kecamatan_partial"
            if covered and cat == TOTAL_CAT:
                drawn_units += 1
                drawn_people += count
            note = f"regency={parent}; province={parent[:2]}"
            if cat == TOTAL_CAT:
                note += "; universe total, not a religion category"
            elif cat in NOTES:
                note += "; " + NOTES[cat]
            if not covered:
                note += ("; NOT DRAWN -- this regency's kecamatan listing is incomplete, "
                         "so its siblings are missing and the regency is drawn instead")
            w.writerow({"geo_id": code, "geo_level": level, "geo_name": name,
                        "source_category": cat, "count": count, "basis": BASIS,
                        "year": YEAR, "source_id": SOURCE_ID, "note": note})
            n += 1

    print(f"wrote {OUT} -- {n:,} rows")
    print(f"  DRAWN TIER: {drawn_units:,} units, {drawn_people:,} people "
          f"({100 * drawn_people / NATIONAL_POPULATION:.2f}% of SP2010)")
    print(f"    {len(complete):,} regencies replaced by their kecamatan, "
          f"{len(reg) - len(complete):,} drawn as regencies")


def main():
    if "--fetch" in sys.argv:
        fetch()
        return
    if "--fetch-gaps" in sys.argv:
        fetch_gaps()
        return
    if "--fetch-all" in sys.argv:
        fetch_all()
        return
    prov_rows, reg_rows, seen_wid = read()
    prov, reg = _tidy(prov_rows), _tidy(reg_rows)
    if not check(prov, reg, seen_wid):
        raise SystemExit("checks failed -- not writing")

    kec = None
    if "--no-kecamatan" not in sys.argv:
        kec, per_regency = read_kecamatan()
        if kec:
            print()
            if not check_kecamatan(kec, per_regency, reg):
                raise SystemExit("kecamatan checks failed -- not writing")
    write(reg, kec)


if __name__ == "__main__":
    main()
