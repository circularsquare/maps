"""Israel — CBS, 2022 Census of Population and Housing, religion by statistical area.

Reads (or fetches) data/raw/il/ and writes data/normalized/il.csv.

**3,236 drawn units at about 3,000 people each** — 1,043 localities drawn whole plus 2,193
statistical areas inside the 142 localities CBS splits. That is the finest geography on this
map after the United States, and it exists because CBS publishes a per-area dashboard for
every unit of the 2022 census.

**THE BASIS IS A STATE REGISTER, NOT A QUESTION** (spec §3.1, §3.9a). Israel's religion
variable comes from the population register, where religion is ascribed at registration from
parentage or a state-recognised conversion. Nobody was asked. So `basis` is `roll` and not
`self_id`, for Germany's reason: these numbers are not comparable with a census that asks.
Two consequences the map has to carry rather than hide:

  * **There is no irreligion category at all.** A secular Israeli Jew is registered as a Jew,
    so Israel draws as ~100% religious. That is a fact about the register. The observance
    axis below is the only corrective, and it is why it is worth the trouble.
  * **"Others" is `unrecorded`, not `none`.** 441,700 people are "not classified by religion"
    in the register — overwhelmingly immigrants under the Law of Return who are not Jewish by
    halakha, largely from the former Soviet Union. Germany's `unrecorded` case exactly
    (§6.3a-i): the register never had a box for them, which is not a statement of belief.

**THE DATA IS BEHIND A PER-AREA DASHBOARD AND THE IDs ARE OPAQUE.** census.cbs.gov.il is an
Astro shell over Looker. `/api/<anything>` returns the same 2,804-byte SPA shell, so a 200
proves nothing and the API reads as absent — but `/en/api/get-csv?dashboardId=..&ID=..`
returns a **zip of the underlying CSVs**, unauthenticated. The `ID` is a 7-hex-char hash, not
a CBS code: 3000 (Jerusalem's real code) 404s, and the space is sparse, so it cannot be
enumerated or guessed. The mapping comes from the site's own htmx autocomplete,
**`/he/partials/search/area?search=<term>`**, which returns `<button data-id data-search
data-type>` for every geographic unit matching a term. That endpoint is the whole key to the
country and nothing links to it. IDs are language-independent.

**A WALL IS A FACT ABOUT A HOST.** data.gov.il serves the census files from three hostnames.
`aws-e.data.gov.il` — the one CKAN advertises — redirects to a **Google OAuth login**;
`e.data.gov.il` returns HTML; plain **`data.gov.il` returns the file**. Same path, same
resource id, three answers.

**THE `religion` COLUMN IN THE BULK CSV IS A MODAL LABEL, NOT A COUNT** (§12 failure 5). The
open-data file `selected-data-by-localities-and-statistical-areas` has one row per unit with
a single `religion` value — the unit's *dominant* group. Abu Sinan is labelled Muslim and its
statistical area 1 is labelled Druze; using it as a composition would make every unit
homogeneous and erase every minority in the country. It is used here only for the unit list,
the population, and as a cross-check on the dashboard shares.

**AND THE DASHBOARD COLLAPSES MINOR CATEGORIES INTO "Other religions", WHICH IS NOT
"Others".** Two different labels with unrelated meanings, and telling them apart is
load-bearing:

  * `Others` — the register's "not classified by religion". Appears only where the full
    five-category breakdown is published (nationwide, districts, large mixed cities).
  * `Other religions` — **everything except the dominant group, lumped**. Nazareth returns
    `Muslims 73.1% / Other religions 26.9%` and that 26.9% is essentially all Christian;
    Shefar'am's 37.1% is Christians and Druze together. Read naively it would erase Israel's
    Christians, which is precisely §14.2's second risk.

  So a unit whose breakdown is `Other religions` is emitted as such and resolved by
  allocation against its sub-district's published Christian / Druze / Muslim / not-classified
  totals (§3.10). The lump is never silently assigned to one religion.

**THE OBSERVANCE AXIS IS NOT A SPLIT OF JUDAISM AND MUST NOT BE MAPPED AS ONE.** The same
dashboard publishes "population by main lifestyle in the household" —
Ultra-religious / Religious / Traditional / Secular / Mixed / Other. It is asked of the whole
population, not of Jews: Umm al-Fahm is 99.8% Muslim and returns Traditional 47.2%,
Religious 42.2%. Crossing it with religion per unit would be a model. What is done instead is
narrow and stated: **where a unit is at least OBSERVANCE_MIN_JEWISH Jewish, the Jewish count
is split by that unit's observance distribution**; everywhere else Judaism is drawn
undifferentiated and the unit's Jews carry the plain category. Israel's statistical areas are
segregated enough that this covers most Israeli Jews while never applying the split to a unit
where the observance figures describe somebody else. Those rows are `derived` in §7's
sense -- counted as Jews, inferred only as to branch; see the comment at the split itself.

**AND `Masorti` IS A FALSE FRIEND.** In Israel *masorti* means traditional-but-not-strictly-
observant and has nothing to do with Conservative Judaism, which is called Masorti everywhere
else. Mapping it to `judaism.conservative` would be flatly wrong — the same trap as
`animismus` in §12's taxonomy notes. See taxonomy/il2022.py.

Territory: this file emits **every unit CBS publishes**, including the Judea and Samaria Area
and East Jerusalem, because a normalised file reproduces its source. What is drawn is decided
in `sources/il_geo.py`, which builds the units layer on the Green Line (Anita, 2026-09-07 —
the Golan is kept, see il_geo.py's docstring for why the two are not the same question).

Usage:
    python sources/il.py --fetch     ~4,500 small requests, ~3.5 h, resumable and cached
                                     DO NOT parallelise it; see the rate-limit note above
    python sources/il.py             normalise from data/raw/il/
"""

import csv
import html as html_mod
import io
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "il")
OUT = os.path.join(ROOT, "data", "normalized", "il.csv")

SOURCE_ID = "il_census_2022"
YEAR = 2022
BASIS = "roll"                      # a state register, not a question -- see docstring

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "tier", "note"]

# The five register categories, in the dashboard's English. `Others` is the register's
# "not classified by religion" and is a real category; `Other religions` is the LUMP and is
# not one -- see the module docstring, they are not the same thing.
RELIGIONS = ("Jews", "Muslims", "Christians", "Druze", "Others")
LUMP = "Other religions"

# CBS district codes, as `bycode2023.xlsx` column 4 carries them, against the row labels in
# ST02-11x. 7 is the Judea and Samaria Area: it is in the table and in il.csv, and it is not
# drawn (sources/il_geo.py cuts it), which is why the allocation still has to handle it.
DISTRICTS = {
    1: "Jerusalem District",
    2: "Northern District",
    3: "Haifa District",
    4: "Central District",
    5: "Tel Aviv District",
    6: "Southern District",
    7: "Judea and Samaria Area",
}
# Section banners in ST02-11x / ST02-11y, and the category each one's rows belong to.
ST_SECTIONS = {
    "JEWS": "Jews",
    "MUSLIMS": "Muslims",
    "CHRISTIANS": "Christians",
    "DRUZE": "Druze",
    "NOT CLASSIFIED BY RELIGION": "Others",
}

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")

# ---- the three bulk files ------------------------------------------------------------
# data.gov.il, PLAIN host: aws-e.* wants Google OAuth and e.* returns HTML (docstring).
SA_CSV_URL = ("https://data.gov.il/dataset/3bd97fde-6cc3-456d-ab63-1caad16b2b6a/resource/"
              "9a9e085f-3bc8-41df-b15f-be0daaf99e30/download/"
              "selected-data-by-localities-and-statistical-areas-2022-census.csv")
BYCODE_URL = ("https://data.gov.il/dataset/d9b1e04c-426f-4e32-ba40-a1ad9d8748a7/resource/"
              "d47a54ff-87f0-44b3-b33a-f284c0c38e5a/download/bycode2023.xlsx")
# CBS Statistical Abstract 2025 table ST02-11x, the published register totals we reconcile
# against. Note the `x` suffix: the geographic tables of chapter 2 are st02_NNx.xlsx while
# the rest are st02_NN.xlsx, and the plain name returns a 2,056-byte SharePoint fake 200.
ST0211_URL = ("https://www.cbs.gov.il/he/publications/doclib/2025/"
              "2.shnatonpopulation/st02_11x.xlsx")

SA_CSV = os.path.join(RAW, "sa2022.csv")
BYCODE = os.path.join(RAW, "bycode2023.xlsx")
ST0211 = os.path.join(RAW, "st02_11x.xlsx")
IDS_JSON = os.path.join(RAW, "area_ids.json")
DASH_JSON = os.path.join(RAW, "dashboards.json")

# Search in HEBREW and fetch data in ENGLISH. The opaque area IDs are language-independent,
# so the two halves can use different editions -- and they want different ones, which is
# §12's North Macedonia rule: the Hebrew labels are what the census file's own locality
# names join against, and the English category labels are what the taxonomy keys on. In
# English the two lumps are distinguishable at a glance ("Others" vs "Other religions");
# nothing about the Hebrew makes that difference visible.
SEARCH = "https://census.cbs.gov.il/he/partials/search/area?search="
GETCSV = "https://census.cbs.gov.il/en/api/get-csv?dashboardId={dash}&ID={aid}"
DASH_RELIGION = "gIRhDCmNLsR3sI0G53n89n"

T_LOCALITY = "יישוב"
T_STATAREA = "אזור סטטיסטי"
SA_WORD = "אזור סטטיסטי"

# The nationwide unit, cached beside the 3,236 drawn ones under this key. It is not drawn --
# it is the only published figure that describes the CENSUS universe by religion, and
# st0211_districts() re-bases the register's marginals onto it.
NATIONAL_TERM = "כלל ארצי"
NATIONAL_KEY = "_national"

SHELL_BYTES = (2803, 2804)          # the SPA shell / error page; never a real answer
# 0.15s was enough to get every connection refused after ~30 minutes. This is a small
# statistical office, the run is thousands of requests, and there is no hurry.
POLITE = 0.6

# Where the Jewish observance split is applied at all -- see docstring. A unit below this is
# drawn as undifferentiated Judaism rather than being given somebody else's observance.
#
# 0.85 — Anita, 2026-09-07, after seeing the country drawn and the coverage measured. It
# started at 0.95, which sounds safer and buys less than it looks:
#
#     threshold   Israeli Jews given an observance colour   non-Jews in those units
#       0.95                    41.7%                              2.9%
#       0.90                    69.7%                              4.9%
#       0.85                    83.3%                              6.3%
#
# At 0.95 most Israeli Jews drew as one undifferentiated blue, and that blue is the most
# misreadable thing on the map: `unspecified` is not a fifth observance category, it is
# "we declined to split this unit", and it reads as a spatial pattern of its own. Trading a
# 2.9% contamination for a 6.3% one to more than double what the map can actually say is the
# better bargain — and the contamination is not evenly damaging, because CBS's
# `Ultra-religious` is in practice a Jewish-only answer, so what leaks in is Arab households
# answering Traditional or Religious rather than anything landing in Haredi.
OBSERVANCE_MIN_JEWISH = 0.85

RELIGION_FILE = "Population, by religion"
LIFESTYLE_FILE = "Population, by main lifestyle in the household"

# Published register totals, 31.12.2024, CBS ST02-11x sheets x and y (thousands).
# Used as a magnitude sanity band only: the census shares are 2022 and these are 2024.
PUBLISHED_2022 = {          # the 2.4.2022 census column, thousands
    "Jews": 6962.0, "Muslims": 1713.1, "Christians": 178.1,
    "Druze": 148.7, "Others": 387.2,
}
NATIONAL_2022 = 9389.1

# TWO NATIONAL TOTALS, AND THEY COUNT DIFFERENT PEOPLE. ST02-11x is "POPULATION OF ISRAELIS"
# and totals 9,389,100 at census day; the census unit file's own nationwide row is
# 9,601,720. The 212,620-person gap is Israel's foreign residents -- labour migrants and
# asylum seekers -- who are in the census and not in the religion register. It matters twice:
# the drawn population must be checked against the CENSUS figure (the units are census
# units), and ST02-11x may only ever be used for the RATIOS between religions, never as an
# absolute marginal. See resolve_lumps.
CENSUS_NATIONAL = 9_601_720


# =====================================================================================
# fetch
# =====================================================================================

_SESSION = None
_STATE = {"delay": POLITE, "errors": 0, "requests": 0}


def _session():
    """One keep-alive session. ~6,800 requests over one TCP connection rather than 6,800
    handshakes, which is both faster and markedly gentler on the host."""
    global _SESSION
    if _SESSION is None:
        import requests
        s = requests.Session()
        s.headers.update({"User-Agent": UA})
        _SESSION = s
    return _SESSION


def _get(url, headers=None, timeout=90, tries=6):
    """GET with backoff.

    census.cbs.gov.il throttles: a sustained run plus one concurrent probe was enough to
    turn every connection into a SYN that never completed. §12's KOSIS entry is the rule --
    a reactive wall SPREADS while being pushed, so the answer is to slow down and retry,
    never to push harder. The politeness delay ratchets up on every error and never back
    down within a run.
    """
    s = _session()
    delay = 2.0
    for attempt in range(tries):
        try:
            time.sleep(_STATE["delay"])
            r = s.get(url, headers=headers, timeout=timeout)
            _STATE["requests"] += 1
            if r.status_code == 200:
                # §12 failure 4: HTTP 200 is not a download. This host answers every
                # unknown path -- and, intermittently, a perfectly good one -- with the
                # 2,804-byte Astro shell or its 2,803-byte error page. Treat that as a
                # RETRYABLE server hiccup rather than a verdict: it cleared on retry every
                # time it was seen. Raising something fatal here is what killed a two-hour
                # run at unit 1,699 (SystemExit is not an Exception, so the per-unit
                # handler could not catch it either).
                if len(r.content) in SHELL_BYTES:
                    raise IOError(f"shell/error page ({len(r.content)} bytes)")
                # Decay back towards the base delay on sustained success. Backing off and
                # never recovering sounds like the safe choice and is not: this host emits
                # a transient shell page on roughly 1.5% of requests, so a one-way ratchet
                # saturates at the cap within minutes and stays there, turning a two-hour
                # run into a six-hour one. Recovery is deliberately far slower than
                # backoff -- 1.5x up per error against 0.98x down per success, so ~35 clean
                # requests to undo one error.
                _STATE["delay"] = max(POLITE, _STATE["delay"] * 0.98)
                return r.content
            if r.status_code in (429, 500, 502, 503, 504):
                raise IOError(f"HTTP {r.status_code}")
            r.raise_for_status()
            return r.content
        except Exception:                               # noqa: BLE001
            _STATE["errors"] += 1
            _STATE["delay"] = min(_STATE["delay"] * 1.5, 3.0)
            if attempt == tries - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 60.0)
    raise IOError("unreachable")


def _download(url, path, min_bytes, magic=None):
    if os.path.exists(path) and os.path.getsize(path) >= min_bytes:
        print(f"  have {os.path.basename(path)} ({os.path.getsize(path):,} b)")
        return
    print(f"  GET {url[:100]}")
    blob = _get(url)
    # §12 failure 4: assert size and type, never the absence of an exception.
    if len(blob) < min_bytes:
        raise SystemExit(f"{path}: {len(blob)} bytes, expected >= {min_bytes}")
    if magic and not blob.startswith(magic):
        raise SystemExit(f"{path}: magic {blob[:8]!r} is not {magic!r} -- "
                         "a login page or a SharePoint fake 200?")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(blob)
    print(f"    {len(blob):,} bytes")


def fetch_bulk():
    print("bulk files:")
    _download(SA_CSV_URL, SA_CSV, 500_000)
    _download(BYCODE_URL, BYCODE, 200_000, b"PK\x03\x04")
    _download(ST0211_URL, ST0211, 20_000, b"PK\x03\x04")


def _search(term):
    """One autocomplete query -> [(id, label, type)]. Up to 100 results, unpaged.

    The shell/error-page check lives in `_get`, where the retry machinery can act on it.
    """
    blob = _get(SEARCH + urllib.parse.quote(term), {"HX-Request": "true"})
    doc = blob.decode("utf-8", "replace")
    hits = re.findall(
        r'data-id="([^"]*)"\s+data-search="([^"]*)"\s+data-type="([^"]*)"', doc)
    # UNESCAPE THE ATTRIBUTE VALUES. Hebrew abbreviations are written with the gershayim,
    # which is a literal `"` -- בני עי"ש, כפר ביל"ו, גבעת ח"ן -- and inside an HTML attribute
    # that has to be `&quot;`. Matched raw, the captured label is `גבעת ח&quot;ן`, which
    # never equals the census file's own name, so **35 real localities silently failed to
    # resolve and nothing about the output looked wrong**: they are 1.1% of the country, well
    # inside any tolerance, and the run reports them only because it lists what it missed.
    return [(i, html_mod.unescape(lab), html_mod.unescape(t)) for i, lab, t in hits]


def _save(path, obj):
    """Write via a temp file and os.replace -- never truncate the cache in place
    ([[reference_wb_truncates]]); an interrupted run must leave the previous cache intact."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False)
    os.replace(tmp, path)


def fetch_ids(units):
    """Resolve every drawn unit to its opaque dashboard ID. Cached; resumable.

    Resumable is not a nicety here: the run is thousands of requests against a host that
    throttles, so it WILL be interrupted, and a cache written only at the end is a cache
    that never gets written.
    """
    cache = {}
    if os.path.exists(IDS_JSON):
        with open(IDS_JSON, encoding="utf-8") as fh:
            cache = json.load(fh)
    want = dict(units)
    todo = [k for k in want if k not in cache]
    print(f"area IDs: {len(cache):,} cached, {len(todo):,} to resolve", flush=True)

    # One query per locality name returns that locality and (usually) all its statistical
    # areas, so resolve in name batches rather than one query per unit.
    by_name = {}
    for key in todo:
        by_name.setdefault(want[key].split(f" {SA_WORD} ")[0], []).append(key)

    done, since_save = 0, 0
    try:
        for name, keys in sorted(by_name.items()):
            try:
                hits = _search(name)
            except Exception as e:                      # noqa: BLE001
                # DO NOT `continue` HERE. A bare locality name can fail PERSISTENTLY on this
                # endpoint while the longer, more specific queries below succeed:
                # `מבשרת ציון` returns the error page every time and
                # `מבשרת ציון אזור סטטיסטי 1` returns all seven of its statistical areas.
                # Skipping the name abandoned a town of 25,000 that was fully reachable.
                print(f"    ! {name}: {e} — falling through to the narrower queries",
                      flush=True)
                hits = []
            found = {label: aid for aid, label, _t in hits}
            for key in keys:
                if want[key] in found:
                    cache[key] = found[want[key]]

            # The endpoint caps at 100 results and Jerusalem has 194 statistical areas, so
            # a name batch cannot cover the big cities. Ask by NUMBER PREFIX rather than one
            # query per area: "<name> אזור סטטיסטי 1" brings back every area whose number
            # starts with 1. Ten queries where Jerusalem would otherwise need 194, which is
            # the difference between a polite run and another rate-limit wall.
            missing = [k for k in keys if k not in cache]
            if missing:
                prefixes = sorted({want[k].rsplit(" ", 1)[-1][:1] for k in missing})
                for p in prefixes:
                    if all(k in cache for k in keys):
                        break
                    try:
                        hits2 = _search(f"{name} {SA_WORD} {p}")
                    except Exception as e:              # noqa: BLE001
                        print(f"    ! {name} prefix {p}: {e}", flush=True)
                        continue
                    found2 = {label: aid for aid, label, _t in hits2}
                    for key in keys:
                        if key not in cache and want[key] in found2:
                            cache[key] = found2[want[key]]

            # Whatever is still missing gets its own exact query.
            for key in [k for k in keys if k not in cache]:
                try:
                    hits3 = _search(want[key])
                except Exception as e:                  # noqa: BLE001
                    print(f"    ! {want[key]}: {e}", flush=True)
                    continue
                for aid, label, _t in hits3:
                    if label == want[key]:
                        cache[key] = aid
                        break
            done += len(keys)
            since_save += len(keys)
            if since_save >= 50:
                since_save = 0
                _save(IDS_JSON, cache)
                print(f"    {done:,}/{len(todo):,} units, {len(cache):,} IDs, "
                      f"{_STATE['requests']:,} requests, {_STATE['errors']} errors, "
                      f"delay {_STATE['delay']:.2f}s", flush=True)
    finally:
        _save(IDS_JSON, cache)

    missing = [want[k] for k in want if k not in cache]
    print(f"  {len(cache):,} IDs; {len(missing):,} unresolved", flush=True)
    for m in missing[:15]:
        print(f"    unresolved: {m}")
    return cache


def _dashboard(aid):
    """-> {csv name: [(column, value), ...]} for one area, or None."""
    blob = _get(GETCSV.format(dash=DASH_RELIGION, aid=aid))
    if not zipfile.is_zipfile(io.BytesIO(blob)):
        return None
    z = zipfile.ZipFile(io.BytesIO(blob))
    out = {}
    for info in z.infolist():
        rows = z.read(info.filename).decode("utf-8-sig").strip().splitlines()
        if len(rows) < 3:
            continue
        out[info.filename.strip().rstrip(".csv").strip()] = list(
            zip([c.strip() for c in rows[0].split(",")],
                [c.strip() for c in rows[-1].split(",")]))
    return out


def fetch_dashboards(ids):
    cache = {}
    if os.path.exists(DASH_JSON):
        with open(DASH_JSON, encoding="utf-8") as fh:
            cache = json.load(fh)
    todo = [k for k in ids if k not in cache]
    print(f"dashboards: {len(cache):,} cached, {len(todo):,} to fetch", flush=True)
    failed = []
    try:
        for i, key in enumerate(todo, 1):
            try:
                d = _dashboard(ids[key])
            except Exception as e:                      # noqa: BLE001
                failed.append(key)
                print(f"    ! {key}: {e}", flush=True)
                continue
            if d is not None:
                cache[key] = d
            if i % 50 == 0:
                _save(DASH_JSON, cache)
                print(f"    {i:,}/{len(todo):,} ({len(cache):,} held, "
                      f"{_STATE['errors']} errors, delay {_STATE['delay']:.2f}s)",
                      flush=True)
    finally:
        _save(DASH_JSON, cache)
    print(f"  {len(cache):,} dashboards; {len(failed):,} failed", flush=True)
    return cache


# =====================================================================================
# the unit list
# =====================================================================================

def canon_cmb(cmb):
    """`2312+2311+2312` -> `2312+2311`, order preserved.

    CBS repeats a code inside one Jerusalem `StatAreaCmb` string. It is a typo in the
    source, and it is not cosmetic: the published unit is named `2312+2311`, so the
    uncanonicalised key matches nothing in the search index and 20,450 people go unresolved.
    `sources/il_geo.py` does the same thing to the same column -- if one is changed the
    other must be, or the counts and the polygons stop sharing a key.
    """
    seen, out = set(), []
    for part in str(cmb).split("+"):
        part = part.strip()
        if part and part not in seen:
            seen.add(part)
            out.append(part)
    return "+".join(out)


def load_units():
    """-> ({key: hebrew label}, {key: row}) for the 3,237 drawn units.

    A locality that CBS splits is NOT drawn itself -- its statistical areas are, and
    drawing both would double the country (§12's Serbia trap).
    """
    if not os.path.exists(SA_CSV):
        raise SystemExit(f"missing {SA_CSV} -- run with --fetch first")
    # §9m: `None` is a category name in this project's sources; never let pandas or the
    # csv module invent NaN. Reading with the csv module keeps every cell a string.
    with open(SA_CSV, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))

    split_parents = {r["LocalityCode"] for r in rows if r["StatArea"].strip()}
    labels, recs = {}, {}
    for r in rows:
        loc, sa = r["LocalityCode"].strip(), canon_cmb(r["StatAreaCmb"])
        name = r["LocNameHeb"].strip()
        if not loc:
            continue                                    # the nationwide row
        if sa:
            key = f"{loc}_{sa}"
            labels[key] = f"{name} {SA_WORD} {sa}"
            level = "statarea"
        else:
            if loc in split_parents:
                continue                                # drawn via its statistical areas
            key = loc
            labels[key] = name
            level = "locality"
        recs[key] = dict(r, _level=level, _loc=loc, _sa=sa, _name=name)
    return labels, recs


# =====================================================================================
# the lump, and what it is resolved against
# =====================================================================================

def national_shares(dash):
    """The nationwide dashboard's own religion shares — the ONE figure available for the
    census universe rather than the register's.

    Needed because ST02-11x counts Israelis and the units count everybody, and the gap is
    not spread evenly: converted onto the census population the two agree within 3% for
    Jews, Muslims, Christians and Druze, and disagree by **51%** for `Others`. Israel's
    ~200,000 foreign residents are almost all carried as not-classified-by-religion, and
    almost none of them as Christians. Using the register's ratios as-is to allocate the
    lump therefore starves `Others` and inflates everything else.
    """
    d = dash.get(NATIONAL_KEY)
    if not d:
        return None
    rel = dict(d.get(RELIGION_FILE, []))
    if not rel:
        return None
    return {k: _pct(v) / 100.0 for k, v in rel.items()}


def st0211_districts(dash=None):
    """-> {religion: {district_code: people}} from ST02-11x/y, the 2.4.2022 column.

    **District is the finest level this table is COMPLETE at, and that is what decides the
    allocation's constraint.** CBS publishes sub-districts too, but only some of them per
    religion — the rows are prefixed `Thereof:` and Christians get four sub-districts of
    fifteen. Constraining on a level the table only partly covers would silently leave the
    uncovered remainder unconstrained, so the marginal here is the district.

    Druze appear only under Northern and Haifa; every other district is a genuine ~0 and is
    recorded as such rather than left missing, so the allocation cannot put Druze in Tel Aviv.
    """
    import openpyxl

    if not os.path.exists(ST0211):
        raise SystemExit(f"missing {ST0211} -- run with --fetch first")
    wb = openpyxl.load_workbook(ST0211, data_only=True)

    want = {v: k for k, v in DISTRICTS.items()}
    out = {r: {c: 0.0 for c in DISTRICTS} for r in RELIGIONS}
    seen = set()

    for sheet in wb.sheetnames:
        ws = wb[sheet]
        section = None
        for r in range(1, ws.max_row + 1):
            label = ws.cell(r, 1).value
            banner = ws.cell(r, 2).value
            # A BANNER ROW IS A STRING IN COLUMN 2 WITH NO LABEL IN COLUMN 1. Testing
            # truthiness alone does not work and fails silently: column 2 holds the 2024
            # PERCENTAGE on every data row, so `if banner:` fires on the number 100 and
            # resets the section immediately after finding it -- which read the whole table
            # as zeros while raising nothing.
            if label is None and isinstance(banner, str) and banner.strip():
                b = re.sub(r"\(\d+\)", "", banner).strip().upper()
                section = ST_SECTIONS.get(b)
                if section:
                    seen.add(section)
                continue
            if section is None:
                continue
            if not label:
                continue
            # strip the `Thereof:` prefix, the footnote markers and the indentation
            lab = re.sub(r"\(\d+\)", "", str(label))
            lab = lab.replace("Thereof:", " ").strip()
            lab = " ".join(lab.split())
            if lab not in want:
                continue
            v = ws.cell(r, 11).value        # the 2.4.2022 column, in thousands
            if isinstance(v, (int, float)):
                out[section][want[lab]] = float(v) * 1000.0

    missing = sorted(set(ST_SECTIONS.values()) - seen)
    if missing:
        raise SystemExit(f"ST02-11x: never found section(s) {missing} -- CBS has "
                         "restructured the table and the marginals cannot be trusted")

    # RE-BASE ONTO THE CENSUS UNIVERSE. Each religion's district figures are scaled so its
    # national sum matches the nationwide dashboard's own share of the census population.
    # The DISTRIBUTION between districts is ST02-11x's and is kept; only the LEVEL moves,
    # which is the half the register gets wrong for this map's universe. §3.4 again:
    # structure from the detailed source, totals from the one that counts the right people.
    shares = national_shares(dash) if dash else None
    if shares:
        print("  re-basing ST02-11x onto the census universe:")
        for rel in RELIGIONS:
            have = sum(out[rel].values())
            want = shares.get(rel, 0.0) * CENSUS_NATIONAL
            if have <= 0 or want <= 0:
                continue
            f = want / have
            for d in out[rel]:
                out[rel][d] *= f
            note = "  <-- the foreign-resident gap" if f > 1.2 or f < 0.83 else ""
            print(f"    {rel:<12} {have:>10,.0f} -> {want:>10,.0f}  x{f:.3f}{note}")
    return out


def locality_district():
    """-> {locality_code: district_code} from bycode2023.xlsx (col 2 -> col 4)."""
    import openpyxl

    if not os.path.exists(BYCODE):
        raise SystemExit(f"missing {BYCODE} -- run with --fetch first")
    ws = openpyxl.load_workbook(BYCODE, data_only=True)[
        openpyxl.load_workbook(BYCODE, data_only=True).sheetnames[0]]
    out = {}
    for r in range(2, ws.max_row + 1):
        code, dist = ws.cell(r, 2).value, ws.cell(r, 4).value
        if code is None or dist is None:
            continue
        try:
            out[str(int(code))] = int(dist)
        except (TypeError, ValueError):
            continue
    return out


def resolve_lumps(rows, recs):
    """Replace every `Other religions` row with named religions, constrained per district.

    The lump is "everything except the group(s) this unit already names" (see the module
    docstring). Two things are known about it and both are used:

      * its SIZE, exactly, per unit;
      * the DISTRICT TOTAL for every religion, from ST02-11x.

    So per district: subtract what the named rows already account for, and share the
    remainder out over that district's lumps by iterative proportional fitting — the unit's
    own lump size is a hard row constraint, the district's residual per religion is the
    column target, and a religion the unit already names is forbidden to it. **A unit's
    dominant group can never be poured back into its own lump**, which is the constraint
    that stops Nazareth's 26.9% being read as more Muslims.

    Output rows are `derived` (§7) and may not ring (§3.10): allocation spreads a total, it
    cannot establish presence.
    """
    with open(DASH_JSON, encoding="utf-8") as fh:
        dash = json.load(fh)
    totals = st0211_districts(dash)
    loc_dist = locality_district()

    by_unit = {}
    for r in rows:
        by_unit.setdefault(r["geo_id"], []).append(r)

    def district_of(key):
        return loc_dist.get(recs[key]["_loc"])

    # ---- what the named rows already account for, per district
    known = {rel: {d: 0.0 for d in DISTRICTS} for rel in RELIGIONS}
    lumps = {d: [] for d in DISTRICTS}
    for key, rs in by_unit.items():
        d = district_of(key)
        if d is None:
            continue
        named = set()
        lump_size = 0.0
        for r in rs:
            cat = r["source_category"]
            if cat == LUMP:
                lump_size += r["count"]
                continue
            base = "Jews" if cat.startswith("Jews") else cat
            if base in RELIGIONS:
                known[base][d] += r["count"]
                named.add(base)
        if lump_size > 0:
            lumps[d].append((key, lump_size, named))

    out = [r for r in rows if r["source_category"] != LUMP]
    report = {"units": 0, "people": 0.0, "districts": {}, "shortfall": 0.0}

    for d in DISTRICTS:
        items = lumps[d]
        if not items:
            continue
        resid = {rel: max(0.0, totals[rel][d] - known[rel][d]) for rel in RELIGIONS}

        # SCALE THE RESIDUAL TO THE LUMPS: the marginals are a COMPOSITION, not a total.
        # ST02-11x is "POPULATION OF ISRAELIS" (9,389,100 at census day) and the census unit
        # file is everybody (9,601,720) -- a 212,620-person universe gap that is Israel's
        # foreign residents, and it is NOT spread evenly across religions: measured against
        # the published figures, Jews, Muslims and Druze come out within 0.7% while
        # Christians run +7.6% and not-classified +13.0%, which is exactly where labour
        # migrants and asylum seekers sit. Used as ABSOLUTE marginals these totals cannot
        # absorb the lumps -- Tel Aviv's lumps exceeded its residuals by 54%, so IPF
        # overshot every column by half again and the allocation was visibly wrong.
        # Normalising the residual to the district's own lump total keeps the ratios
        # between religions, which is the part ST02-11x is authoritative about, and takes
        # the magnitude from the census, which is the universe being drawn. §3.4's rule --
        # structure from one source, totals from the other -- applied within one country.
        s = sum(resid.values())
        lump_total = sum(s2 for _k, s2, _n in items)
        if s > 0 and lump_total > 0:
            resid = {rel: v * lump_total / s for rel, v in resid.items()}
        allowed = [{rel for rel in RELIGIONS if rel not in named and resid[rel] > 0}
                   for _k, _s, named in items]
        sizes = [s for _k, s, _n in items]

        # seed proportionally to the district residual, then IPF: rows are hard (a unit's
        # allocation must equal its lump), columns are targets (the district residual).
        x = []
        for (key, size, _named), allow in zip(items, allowed):
            tot = sum(resid[rel] for rel in allow) or 1.0
            x.append({rel: size * resid[rel] / tot for rel in allow})
        for _ in range(40):
            for rel in RELIGIONS:
                col = sum(xi.get(rel, 0.0) for xi in x)
                if col > 0 and resid[rel] > 0:
                    f = resid[rel] / col
                    for xi in x:
                        if rel in xi:
                            xi[rel] *= f
            for xi, size in zip(x, sizes):
                s = sum(xi.values())
                if s > 0:
                    f = size / s
                    for rel in xi:
                        xi[rel] *= f

        got = {rel: sum(xi.get(rel, 0.0) for xi in x) for rel in RELIGIONS}
        report["districts"][d] = {
            "lumps": len(items), "people": sum(sizes),
            "resid": resid, "allocated": got,
        }
        for (key, size, _named), xi in zip(items, x):
            proto = by_unit[key][0]
            for rel, n in sorted(xi.items()):
                if n <= 0.5:
                    continue
                out.append({
                    "geo_id": key, "geo_level": proto["geo_level"],
                    "geo_name": proto["geo_name"], "source_category": rel,
                    "count": round(n, 3), "basis": BASIS, "year": YEAR,
                    "source_id": SOURCE_ID, "tier": "derived",
                    "note": f"allocated from '{LUMP}' ({size:,.0f} people in this unit) "
                            f"against district {d}'s published total; derived"})
            report["units"] += 1
            report["people"] += size

    return out, report


# =====================================================================================
# read
# =====================================================================================

def _pct(v):
    v = v.strip().rstrip("%").strip()
    return float(v) if v else 0.0


def read():
    labels, recs = load_units()
    if not os.path.exists(DASH_JSON):
        raise SystemExit(f"missing {DASH_JSON} -- run with --fetch first")
    with open(DASH_JSON, encoding="utf-8") as fh:
        dash = json.load(fh)

    rows, stats = [], {"units": 0, "nodash": 0, "lumped": 0, "observance": 0,
                       "people": 0.0, "popmismatch": []}

    for key, rec in sorted(recs.items()):
        d = dash.get(key)
        if not d:
            stats["nodash"] += 1
            continue
        pop = float(rec["pop_approx"].replace(",", "") or 0)
        if pop <= 0:
            continue
        rel = dict(d.get(RELIGION_FILE, []))
        if not rel:
            stats["nodash"] += 1
            continue
        shares = {k: _pct(v) for k, v in rel.items()}
        total = sum(shares.values())
        if not (99.0 <= total <= 101.0):
            raise SystemExit(f"{key}: religion shares sum to {total} -- {rel}")

        life = {k: _pct(v) for k, v in dict(d.get(LIFESTYLE_FILE, [])).items()}
        jewish_share = shares.get("יהודים", shares.get("Jews", 0.0)) / 100.0
        use_obs = (life and jewish_share >= OBSERVANCE_MIN_JEWISH)

        stats["units"] += 1
        if "Other religions" in shares or "דת אחרת" in shares:
            stats["lumped"] += 1

        for cat, share in sorted(shares.items()):
            n = pop * share / 100.0
            if n <= 0:
                continue
            note = f"level={rec['_level']}; share={share}%; pop={pop:g}"
            is_jewish = cat in ("Jews", "יהודים")
            if is_jewish and use_obs:
                # Split this unit's Jews by its own observance distribution. Only here, and
                # only because the unit clears OBSERVANCE_MIN_JEWISH (0.85).
                #
                # TIER IS `derived`, NOT `modelled`, AND THE CHOICE IS DELIBERATE. §7's
                # ladder is counted / inferred / we-do-not-know. These people ARE counted,
                # and counted as Jews: what is inferred is only which observance branch they
                # sit in, from two tables CBS published per unit, under a rule stated in one
                # line. That is the same shape as allocate.py's share assumption, which is
                # `derived` everywhere else in this project. Calling it `modelled` would put
                # a THIRD of Israel behind §7a's "inferred dots: hidden" and make the
                # country read as far less measured than it is -- the religion of every one
                # of these dots is measured; only its sub-branch is not.
                stats["observance"] += 1
                lt = sum(life.values()) or 100.0
                for lcat, lshare in sorted(life.items()):
                    m = n * lshare / lt
                    if m <= 0:
                        continue
                    # Bracket rather than a slash: one of CBS's own observance labels is
                    # "Religious / Very religious", so a slash separator would produce
                    # "Jews / Religious / Very religious" and no parser could tell which
                    # slash was the join.
                    rows.append({
                        "geo_id": key, "geo_level": rec["_level"],
                        "geo_name": labels[key],
                        "source_category": f"{cat} [{lcat}]",
                        "count": round(m, 3), "basis": BASIS, "year": YEAR,
                        "source_id": SOURCE_ID, "tier": "derived",
                        "note": note + f"; observance={lshare}% of unit"})
                    stats["people"] += m
                continue
            # The lump is emitted as it comes and resolved afterwards, against district
            # marginals it takes the whole country to compute (see resolve_lumps).
            rows.append({"geo_id": key, "geo_level": rec["_level"],
                         "geo_name": labels[key], "source_category": cat,
                         "count": round(n, 3), "basis": BASIS, "year": YEAR,
                         "source_id": SOURCE_ID, "tier": "measured", "note": note})
            stats["people"] += n

    return rows, stats, recs


# =====================================================================================
# check
# =====================================================================================

def check(rows, stats, recs):
    ok = True
    print(f"  {stats['units']:,} units with a dashboard; {stats['nodash']:,} without")
    good = stats["nodash"] < 0.02 * (stats["units"] + stats["nodash"])
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} units without data are under 2% of the country")

    # `pop_approx` is rounded per unit (to the nearest 10, and to the nearest 1,000 for the
    # smallest), so a sum over 3,236 units cannot equal the published national figure and a
    # band is the honest test rather than an equality (§12, reconciliation discipline).
    # Against the CENSUS national row, which is the universe the units belong to -- not
    # against ST02-11x, which counts Israelis only and is 212,620 people smaller.
    people = stats["people"]
    ratio = people / CENSUS_NATIONAL
    good = 0.98 <= ratio <= 1.02
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} people placed {people:,.0f} vs the census's own "
          f"nationwide row {CENSUS_NATIONAL:,} (ratio {ratio:.4f}, band 0.98-1.02)")

    cats = {}
    for r in rows:
        cats[r["source_category"]] = cats.get(r["source_category"], 0.0) + r["count"]
    print(f"\n  {len(rows):,} rows over {len(cats)} categories:")
    for c, n in sorted(cats.items(), key=lambda kv: -kv[1]):
        print(f"    {n:>12,.0f}  {100 * n / people:6.2f}%  {c}")

    # ---- the lump must be gone, and the allocation must have hit its marginals ----
    left = sum(n for c, n in cats.items() if c == LUMP)
    good = left == 0
    ok &= good
    print(f"\n  {'OK ' if good else 'BAD'} no '{LUMP}' rows survive "
          f"({left:,.0f} people) -- the lump is not a category and must never be drawn")

    a = stats.get("alloc", {})
    print(f"  {stats['lumped']:,} units published the lump; {a.get('units', 0):,} were "
          f"allocated, {a.get('people', 0):,.0f} people "
          f"({100 * a.get('people', 0) / people:.2f}% of the country)")
    print("  A district marked OFF below is not a bug and cannot be fixed by the allocator: "
          "where every lumped unit in a district already NAMES a religion, that religion is "
          "forbidden in their lumps by construction, so the district's residual for it has "
          "nowhere to go and the row constraint pushes the difference into the religions "
          "that are allowed. Central and Judea/Samaria are Jewish-majority districts whose "
          "lumped units almost all name Jews. The residual there is mostly the universe gap "
          "(census counts everybody, ST02-11x counts Israelis) rather than unplaced people.")
    for d, info in sorted(a.get("districts", {}).items()):
        got, resid = info["allocated"], info["resid"]
        line = "  ".join(f"{r[:5]} {got[r]:>8,.0f}/{resid[r]:>8,.0f}"
                         for r in RELIGIONS if resid[r] > 0 or got[r] > 0)
        worst, worst_rel = 0.0, ""
        for r in RELIGIONS:
            if resid[r] > 500:
                dev = abs(got[r] - resid[r]) / resid[r]
                if dev > worst:
                    worst, worst_rel = dev, r
        mark = "" if worst < 0.2 else f"   <-- {worst_rel} off by {worst:.0%}"
        print(f"    district {d} {DISTRICTS[d][:22]:<22} {info['lumps']:>4} lumps "
              f"{info['people']:>9,.0f} ppl{mark}")
        print(f"        allocated/residual  {line}")

    # ---- against the published register figures, and what the gap IS ----
    # Roll the observance rows back up to Judaism so the five register categories can be
    # compared with the table they came from.
    fold = {}
    for c, n in cats.items():
        base = "Jews" if c.startswith("Jews") else c
        fold[base] = fold.get(base, 0.0) + n
    # THE TARGET IS THE CENSUS UNIVERSE, NOT THE REGISTER. Both are printed because the
    # difference between them is the single most confusing thing about this source, and a
    # reader who only sees one column will draw the wrong conclusion from it.
    tgt = stats.get("national_target") or {}
    print("\n  the five register categories, against BOTH published universes:")
    print(f"    {'':<12}{'drawn':>12}{'census univ.':>13}{'ratio':>7}"
          f"{'register':>12}{'ratio':>7}")
    worst = 0.0
    for rel in RELIGIONS:
        g = fold.get(rel, 0.0)
        c = tgt.get(rel, 0.0)
        p = PUBLISHED_2022[rel] * 1000
        cr = g / c if c else float("nan")
        worst = max(worst, abs(cr - 1)) if c else worst
        print(f"    {rel:<12}{g:>12,.0f}{c:>13,.0f}{cr:>7.3f}{p:>12,.0f}{g / p:>7.3f}")
    good = worst < 0.10
    ok &= good
    print(f"    {'OK ' if good else 'BAD'} worst deviation from the census universe is "
          f"{worst:.1%} (band 10%).")
    print("    The `register` column is ST02-11x, which counts ISRAELIS -- 9,389,100 against "
          "the census's 9,601,720. The ~212,000 difference is Israel's foreign residents, "
          "and the striking thing is WHERE they sit: converted onto the census population "
          "the two sources agree within 3% for Jews, Muslims, Christians AND Druze, and "
          "disagree by 51% for `Others`. Foreign residents are carried almost entirely as "
          "not-classified-by-religion, essentially none of them as Christians -- which is "
          "the opposite of the obvious guess and is why the marginals are re-based per "
          "religion rather than scaled by one national factor.")

    tiers = {}
    for r in rows:
        tiers[r["tier"]] = tiers.get(r["tier"], 0.0) + r["count"]
    print("\n  by confidence tier (§7):")
    for t, n in sorted(tiers.items(), key=lambda kv: -kv[1]):
        print(f"    {n:>12,.0f}  {100 * n / people:6.2f}%  {t}")
    print(f"  {stats['observance']:,} units got the Jewish observance split "
          f"(>= {OBSERVANCE_MIN_JEWISH:.0%} Jewish)")

    if not ok:
        raise SystemExit("reconciliation FAILED")


def main():
    if "--fetch" in sys.argv:
        os.makedirs(RAW, exist_ok=True)
        fetch_bulk()
        labels, _recs = load_units()
        print(f"\n{len(labels):,} drawn units")
        # The nationwide unit rides along with the drawn ones so it is fetched and cached by
        # the same resumable machinery; read() ignores it because it is not in `recs`.
        labels = dict(labels)
        labels[NATIONAL_KEY] = NATIONAL_TERM
        ids = fetch_ids(labels)
        fetch_dashboards(ids)
    rows, stats, recs = read()
    rows, alloc = resolve_lumps(rows, recs)
    stats["alloc"] = alloc
    with open(DASH_JSON, encoding="utf-8") as fh:
        _shares = national_shares(json.load(fh)) or {}
    stats["national_target"] = {r: _shares.get(r, 0.0) * CENSUS_NATIONAL
                                for r in RELIGIONS}
    check(rows, stats, recs)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print("\nwrote", OUT, f"({len(rows):,} rows)")


if __name__ == "__main__":
    main()
