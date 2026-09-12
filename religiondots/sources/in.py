"""India — Census 2011, table C-01, religion down to the sub-district.

Reads (or fetches) data/raw/in/ and writes data/normalized/in.csv.

1.21 BILLION PEOPLE, which is more than the twelve countries already drawn put together.
It is also the oldest source on the map by a decade: the 2021 census was postponed
indefinitely and has still not happened, so Census 2011 is not merely the best Indian
source, it is the only one, and §3.4's "rescale the structure to a recent total" is not
available because there is no recent total to rescale to.

WHAT THE CENSUS PUBLISHES, and the three tables are very different in quality:

  C-01           8 categories — Hindu, Muslim, Christian, Sikh, Buddhist, Jain, `Other
                 religions and persuasions`, `Religion not stated` — at India, state,
                 district, SUB-DISTRICT and town level.  This is the spine: 5,924
                 sub-districts, average 204,000 people, which is coarse per unit by
                 European standards and is the finest religion geography India has.

  C-01 Appendix  the break-up of `Other religions and persuasions`: 84 named religions
                 covering 7.94M people, at India and STATE level only.  This is where the
                 source earns its place — Sarna 4,957,467, Gond/Gondi 1,026,344, Sari
                 Dharma 506,369, Doni Polo 331,370, Sanamahi 222,422, Khasi 138,512,
                 Niamtre 84,276, Parsi/Zoroastrian 57,264, Atheist 33,304.  No other
                 census on earth names these.

  C-01 Annexure  ostensibly the sects clubbed under each main religion.  IN PRACTICE IT IS
                 WRITE-IN RESIDUE AND NOT A PARTITION, and this is the single most
                 important thing to know before planning any work on it: of 172.2M
                 Muslims it names 573 Shia and 267 Sunni; of 27.8M Christians, 8,399
                 Catholics and 603 Protestants.  These are the handful of people who wrote
                 a sect on the form where essentially everyone else wrote the religion.
                 Treating them as a denominational breakdown would be a serious error.
                 ONE entry is substantial and real — Lingayat / Veer Shaiva, 2,663,229 —
                 and a few more are in the tens of thousands (Bathou 245,954, Satnami
                 101,740, Sanatan Dharma 98,382, Ravidasi 88,650, Meitei 41,673, Nav
                 Buddhist 34,123, Bohra 33,460).  Everything under ~1,000 is noise.

So India is §3.9 in its purest form: fine geography with 8 categories, fine categories at
state level, and no table joining them.  allocate.py is the §3.10 repair, and India is the
first source where it must run WITHIN each coarse unit rather than pooling them — see
`--within` there and sources/in.md.  Doni Polo is 99.9% Arunachal Pradesh and Sanamahi is
99.9% Manipur; a pooled national share would put both in every sub-district in India.

FOUR THINGS IN THE DATA THAT WILL BITE:

  1. **TOWN ROWS ARE URBAN-ONLY SUBSETS OF THEIR SUB-DISTRICT.**  The geography nests
     state > district > sub-district > town, and a town row repeats people already counted
     in the sub-district above it.  Summing the file as delivered counts urban India twice.
     Town rows carry `Total/Rural/Urban == Urban` and never `Total`, so filtering to Total
     drops them — but that is a coincidence of how the table is laid out, not a rule, so
     this file ASSERTS it rather than relying on it.

  2. **Every level is in one column set, distinguished only by which code is non-zero.**
     `State - JHARKHAND`, `District - Garhwa`, `Sub-District - Kharaundhi` are all bare
     rows; the level is `sd == "00000" and d == "000"` -> state, and so on.  §12's
     "universe rows are not categories" in geographic form.

  3. **The 35 state files are 35 separate downloads with per-file resource ids** that are
     not derivable from the state code, so the catalogue page is scraped for each.  The
     India-level file is a 36th and is used only as a check.

  4. **censusindia.gov.in serves an incomplete certificate chain.**  curl, requests and
     certifi all fail identically with "unable to get local issuer certificate".  That is
     the server's fault, not a bad URL and not a proxy — the same defect as stat.gov.pl
     (sources.md §5a).  Verification is disabled FOR THIS ONE HOST and the bytes are
     validated structurally instead: OLE2/zip magic, expected sheet, expected header.

BASIS is self_id.  The question was "religion", answered by the head of household for the
household, which is a weaker self-identification than a personal answer and is why India's
`Religion not stated` is only 0.24% — far below Poland's 20.5% refusal or Romania's 14%
absent variable.  Nobody refuses on behalf of someone else.

Usage:
    python sources/in.py --fetch    download 38 files (~14MB) if missing
    python sources/in.py            normalise from data/raw/in/
"""

import argparse
import csv
import os
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "in")
OUT = os.path.join(ROOT, "data", "normalized", "in.csv")

SOURCE_ID = "in_census_2011"
YEAR = 2011
BASIS = "self_id"

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

NATIONAL = 1_210_854_977        # Census 2011, total population of India

# The catalogue block that holds C-01. 11361 is India, 11362..11396 are the 35 states and
# union territories in alphabetical order, 11397 the Annexure, 11398 the Appendix. The
# per-file resource id inside the download URL is NOT derivable from the state code, so the
# page is fetched and the link read off it (§12: derive, do not hard-code).
CAT_FIRST, CAT_LAST = 11361, 11396
CAT_ANNEXURE, CAT_APPENDIX = 11397, 11398

HOST = "censusindia.gov.in"
CAT_URL = "https://censusindia.gov.in/nada/index.php/catalog/{cid}"
DL_RE = re.compile(r'href="(https://censusindia\.gov\.in/nada/index\.php/catalog/'
                   r'\d+/download/[^"]+)"')
# The Appendix and Annexure are also `DDW00C-01 ...`, so the C-01 pattern has to anchor on
# what follows the table name or it matches all three and the file count check fails.
STATE_RE = re.compile(r"DDW(\d\d)C-01 MDDS\.XLS$", re.I)

MIN_BYTES = 20_000

# C-01's eight columns, in the order the sheet lays them out. The header is three merged
# rows deep so it is not parsed; these are matched by position and the position is asserted
# against the header text before anything is read.
CATEGORIES = [
    "Total",                            # universe row, not a religion
    "Hindu",
    "Muslim",
    "Christian",
    "Sikh",
    "Buddhist",
    "Jain",
    "Other religions and persuasions",
    "Religion not stated",
]
FIRST_COUNT_COL = 7          # column index of `Total` Persons
COL_STRIDE = 3               # Persons, Males, Females

TOTAL_LABEL = "Total"
TOTAL_NOTE = "universe total, not a religion category"

# `Religion not stated` is a real refusal/unknown and is emitted so the country's coverage
# can be stated honestly, but countries.py excludes it from the dots exactly as it does
# Poland's refusals and Romania's absent variable (§3.5).
NOT_STATED = "Religion not stated"

# The part of `Other religions and persuasions` that the Appendix does not name, emitted as
# its own category so the allocation's shares are exact rather than normalised over the
# named religions alone. See the block that writes it in normalise().
UNNAMED = "Other religions and persuasions, not separately named"


def resid_total(rows):
    return sum(r["count"] for r in rows
               if r["source_category"] == UNNAMED and r["geo_level"] == "nation")

LEVEL_NAMES = {"nation": "INDIA"}

# Appendix and Annexure share a layout: one row per (state, category, T/R/U).
APX_SHEET = "Sheet1"


# ---------------------------------------------------------------------------- fetching

def _session():
    """A requests session with verification off for censusindia.gov.in only.

    The host omits its intermediate certificate. Everything fetched through this session is
    validated structurally by _check_bytes() instead, which is the honest trade: we cannot
    authenticate the server, so we authenticate the payload.
    """
    import requests
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    s = requests.Session()
    s.verify = False
    s.headers["User-Agent"] = "religiondots/1.0 (map project; contact via repo)"
    return s


def _check_bytes(path, blob):
    """HTTP 200 is not a download (sources.md §5a)."""
    if len(blob) < MIN_BYTES:
        raise SystemExit(f"{path}: {len(blob)} bytes, expected >= {MIN_BYTES}")
    if not (blob[:8] == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" or blob[:2] == b"PK"):
        raise SystemExit(f"{path}: not an OLE2 or zip container, first bytes {blob[:8]!r}")


def fetch():
    os.makedirs(RAW, exist_ok=True)
    s = _session()
    wanted = list(range(CAT_FIRST, CAT_LAST + 1)) + [CAT_ANNEXURE, CAT_APPENDIX]
    got = {}
    for cid in wanted:
        page = s.get(CAT_URL.format(cid=cid), timeout=120)
        if page.status_code != 200:
            raise SystemExit(f"catalog {cid}: HTTP {page.status_code}")
        links = sorted(set(DL_RE.findall(page.text)))
        if len(links) != 1:
            raise SystemExit(f"catalog {cid}: expected 1 download link, found {len(links)}")
        url = links[0]
        name = url.rsplit("/", 1)[-1].replace("%20", " ")
        dest = os.path.join(RAW, name)
        got[cid] = name
        if os.path.exists(dest) and os.path.getsize(dest) >= MIN_BYTES:
            print(f"  have {name}")
            continue
        r = s.get(url, timeout=300)
        if r.status_code != 200:
            raise SystemExit(f"{name}: HTTP {r.status_code}")
        _check_bytes(name, r.content)
        with open(dest, "wb") as fh:
            fh.write(r.content)
        print(f"  got  {name}  ({len(r.content):,} bytes)")

    # The 36 C-01 files must cover state codes 00..35 exactly once. A reissue that renamed
    # or dropped one fails here rather than silently producing a short map.
    codes = sorted(STATE_RE.search(n).group(1) for cid, n in got.items()
                   if CAT_FIRST <= cid <= CAT_LAST)
    expected = [f"{i:02d}" for i in range(36)]
    if codes != expected:
        missing = sorted(set(expected) - set(codes))
        extra = sorted(set(codes) - set(expected))
        raise SystemExit(f"C-01 state codes wrong: missing {missing}, unexpected {extra}")
    print(f"\n  {len(codes)} C-01 files, state codes 00-35 complete")


# ---------------------------------------------------------------------------- C-01

def _assert_header(df, path):
    """Prove the count columns are where CATEGORIES says before reading any number.

    The header is three merged rows; the category name sits on row 2 of the block at the
    Persons column and is blank over Males/Females. Checking it by position is the only
    defence against a state file with a different column order, and a wrong offset here
    would silently swap Sikh and Buddhist for a whole state.
    """
    head = df.iloc[:6]
    row = None
    for i in range(len(head)):
        vals = [str(v).strip() for v in head.iloc[i].tolist()]
        if "Hindu" in vals:
            row = head.iloc[i]
            break
    if row is None:
        raise SystemExit(f"{path}: no header row containing 'Hindu'")
    for k, name in enumerate(CATEGORIES):
        col = FIRST_COUNT_COL + k * COL_STRIDE
        got = str(row.iloc[col]).strip()
        # Seven of the eight headers are identical in all 36 files. The `Other` column is
        # not: it is written `Other religions and persuasions (incl.Unclassified Sect.)`
        # and, in some states, with a trailing ` - 2011`. Compare on the leading text so
        # that variation passes while a genuine column swap — Sikh where Buddhist should
        # be — still fails.
        if not got.lower().startswith(name.lower()):
            raise SystemExit(f"{path}: column {col} is {got!r}, expected {name!r}")


def _code(v, width):
    """Zero-padded code, whatever Excel decided the cell was.

    The C-01 state files store `00`/`000`/`00000` as text and the Appendix stores the same
    codes as numbers, so `str(cell)` gives `"00"` in one file and `"0"` in the other. That
    difference is invisible and expensive: it made India's own row look like a 36th state
    and double-counted the whole appendix. Normalise once, here.
    """
    s = str(v).strip()
    if s.endswith(".0"):
        s = s[:-2]
    if not s.isdigit():
        raise SystemExit(f"non-numeric code {v!r}")
    if len(s) > width:
        raise SystemExit(f"code {s!r} wider than the expected {width}")
    return s.zfill(width)


def _level_of(s, d, sd, town):
    if town != "000000":
        return "town"
    if sd != "00000":
        return "subdistrict"
    if d != "000":
        return "district"
    if s != "00":
        return "state"
    return "nation"


def read_c01(path):
    """One state's C-01, as (level, geo_id, geo_name, category, count) tuples."""
    xl = pd.ExcelFile(path)
    if "C01" not in xl.sheet_names:
        raise SystemExit(f"{path}: no C01 sheet, have {xl.sheet_names}")
    raw = xl.parse("C01", header=None, dtype=object)
    _assert_header(raw, path)

    rows = []
    seen_tru = set()
    for _, r in raw.iterrows():
        if str(r.iloc[0]).strip() != "C0101":
            continue                                    # header/blank/footnote
        s, d, sd, town = (_code(r.iloc[i], w)
                          for i, w in ((1, 2), (2, 3), (3, 5), (4, 6)))
        name = str(r.iloc[5]).strip()
        tru = str(r.iloc[6]).strip()
        level = _level_of(s, d, sd, town)
        seen_tru.add((level, tru))
        if tru != "Total":
            continue                                    # Rural/Urban are splits of Total

        # geo_id is the full path so it is unique across levels and joinable: the SHRUG
        # boundary file keys on (pc11_s_id, pc11_d_id, pc11_sd_id) with these same widths.
        gid = {"nation": "IN", "state": s, "district": s + d,
               "subdistrict": s + d + sd, "town": s + d + sd + town}[level]
        clean = re.sub(r"^(State|District|Sub-District)\s*-\s*", "", name).strip()

        for k, cat in enumerate(CATEGORIES):
            col = FIRST_COUNT_COL + k * COL_STRIDE
            v = r.iloc[col]
            if pd.isna(v):
                raise SystemExit(f"{path}: blank count for {gid} {cat!r}")
            n = int(v)
            if n < 0:
                raise SystemExit(f"{path}: negative count {n} for {gid} {cat!r}")
            rows.append((level, gid, clean, cat, n))
    return rows, seen_tru


# ---------------------------------------------------------------------------- appendix

def read_appendix(path):
    """C-01 Appendix — the 84 named religions inside `Other religions and persuasions`.

    India + state level. `distt` is 000 throughout: the catalogue blurb claims district
    level and the published file does not have it, which is worth knowing before anyone
    plans work around district-level tail data that does not exist.
    """
    df = pd.read_excel(path, sheet_name=APX_SHEET, header=None, skiprows=4,
                       names=["table", "s", "d", "area", "relcode", "relname",
                              "tru", "persons", "males", "females"], dtype=object)
    df = df[df["table"].astype(str).str.strip() == "C01APX"]
    bad_d = sorted({_code(x, 3) for x in df["d"]} - {"000"})
    if bad_d:
        raise SystemExit(f"appendix: unexpected district codes {bad_d[:5]} — the file now "
                         f"has district detail and read_appendix should use it")
    df = df[df["tru"].astype(str).str.strip() == "Total"]

    rows = []
    for _, r in df.iterrows():
        s = _code(r["s"], 2)
        code = _code(r["relcode"], 6)
        name = str(r["relname"]).strip()
        level = "nation" if s == "00" else "state"
        gid = "IN" if level == "nation" else s
        rows.append((level, gid, str(r["area"]).strip(), name, int(r["persons"]), code))
    return rows


def read_annexure(path):
    """C-01 Annexure — sects written in under a main religion.

    Emitted for the record with `basis=self_id` and a note saying what it is, because the
    numbers are real; NOT used to split anything. See the module docstring: 573 Shia in a
    country of 172M Muslims is a count of people who wrote the word, not a sect breakdown.
    """
    df = pd.read_excel(path, sheet_name=APX_SHEET, header=None, skiprows=3,
                       names=["table", "s", "d", "relcode", "sectcode", "area",
                              "label", "tru", "persons", "males", "females"], dtype=object)
    df = df[df["table"].astype(str).str.strip() == "C01ANX"]
    df = df[df["tru"].astype(str).str.strip() == "Total"]

    rows = []
    for _, r in df.iterrows():
        label = str(r["label"]).strip()
        if not label.startswith("Sect:"):
            continue                                    # `Religion:X` and `All Religious
                                                        # Community` are the parent rows
        sect = label[len("Sect:"):].strip()
        s = _code(r["s"], 2)
        level = "nation" if s == "00" else "state"
        gid = "IN" if level == "nation" else s
        rel = _code(r["relcode"], 6)
        sc = _code(r["sectcode"], 6)
        rows.append((level, gid, str(r["area"]).strip(), sect, int(r["persons"]), rel, sc))
    return rows


# ---------------------------------------------------------------------------- main

def normalise():
    files = sorted(f for f in os.listdir(RAW) if STATE_RE.search(f))
    if len(files) != 36:
        raise SystemExit(f"expected 36 C-01 files in {RAW}, found {len(files)} — "
                         f"run with --fetch")

    out = []
    tru_seen = set()
    india_file = None
    per_state_total = {}

    for f in files:
        code = STATE_RE.search(f).group(1)
        path = os.path.join(RAW, f)
        if code == "00":
            india_file = path
            continue                                    # read separately, as a check
        rows, tru = read_c01(path)
        tru_seen |= tru
        got_levels = {r[0] for r in rows}
        if "subdistrict" not in got_levels:
            raise SystemExit(f"{f}: no sub-district rows")
        for level, gid, name, cat, n in rows:
            if level == "town":
                raise SystemExit(f"{f}: a town row survived the Total filter ({gid}) — "
                                 f"town rows are urban-only subsets and would double count")
            note = TOTAL_NOTE if cat == TOTAL_LABEL else ""
            out.append(dict(geo_id=gid, geo_level=level, geo_name=name,
                            source_category=cat, count=n, basis=BASIS, year=YEAR,
                            source_id=SOURCE_ID, note=note))
            if level == "state" and cat == TOTAL_LABEL:
                per_state_total[gid] = n
        print(f"  {f}: {len(rows):,} rows, "
              f"{len({r[1] for r in rows if r[0] == 'subdistrict'}):,} sub-districts")

    # ASSERTION 1: town rows are urban-only. If a town ever carries a Total row the filter
    # above stops protecting against double counting, so fail rather than hope.
    town_tru = {t for lv, t in tru_seen if lv == "town"}
    if town_tru - {"Urban"}:
        raise SystemExit(f"town rows carry {sorted(town_tru)}, not just Urban — the "
                         f"Total filter no longer excludes them and totals will double")
    print(f"\n  town rows carry {sorted(town_tru)} only, as assumed")

    # ASSERTION 2: the sum of the 35 state files reproduces the published national total,
    # exactly. India neither rounds nor suppresses, so there is no band to compute.
    got = sum(per_state_total.values())
    if got != NATIONAL:
        raise SystemExit(f"states sum to {got:,}, published national total is "
                         f"{NATIONAL:,} (diff {got - NATIONAL:+,})")
    print(f"  35 states sum to {got:,} = published national total, exactly")

    # ASSERTION 3: the India-level file agrees with each state file, per state and per
    # category. DDW00C-01 turns out to hold NO India row — it is 35 states x 3 (T/R/U) and
    # nothing else — so it cannot check the national total, which is what assertion 2 is
    # for. What it can do is better: it is an independently published copy of every state's
    # eight figures, so comparing it state by state catches a column-offset error in any
    # ONE state file. Assertion 2 cannot see that, because a swapped Sikh/Buddhist pair
    # leaves the Total column correct and the national sum still lands.
    india_rows, _ = read_c01(india_file)
    if any(lv == "nation" for lv, *_ in india_rows):
        raise SystemExit("DDW00C-01 now carries an India row; the nation level is "
                         "synthesised from the states and would be duplicated")
    summary = {(gid, cat): n for lv, gid, _, cat, n in india_rows if lv == "state"}
    per_state = {(r["geo_id"], r["source_category"]): r["count"]
                 for r in out if r["geo_level"] == "state"}
    if set(summary) != set(per_state):
        miss = sorted(set(summary) ^ set(per_state))[:6]
        raise SystemExit(f"India file and state files disagree on which (state, category) "
                         f"pairs exist: {miss}")
    bad = [(k, summary[k], per_state[k]) for k in summary if summary[k] != per_state[k]]
    if bad:
        raise SystemExit(f"{len(bad)} (state, category) figures disagree between the India "
                         f"file and the state files, e.g. {bad[:3]}")
    print(f"  all {len(summary):,} (state x category) figures agree between the India "
          f"file and the 35 state files")

    # The nation level is the sum of the states, which assertion 2 has just proved lands on
    # the published national total.
    nation = {}
    for r in out:
        if r["geo_level"] == "state":
            nation[r["source_category"]] = nation.get(r["source_category"], 0) + r["count"]
    india_nat = nation
    for cat in CATEGORIES:
        out.append(dict(geo_id="IN", geo_level="nation", geo_name="INDIA",
                        source_category=cat, count=nation[cat], basis=BASIS, year=YEAR,
                        source_id=SOURCE_ID,
                        note=TOTAL_NOTE if cat == TOTAL_LABEL else "summed from the 35 "
                                                                  "state files"))

    # ---- appendix: the tail, with its parent named so allocate.py can find it
    apx = os.path.join(RAW, "DDW00C-01 Appendix MDDS.xlsx")
    bucket = "Other religions and persuasions"
    apx_rows = read_appendix(apx)
    apx_state_sum = 0
    n_named = 0
    for level, gid, name, cat, n, code in apx_rows:
        # EACH STATE BLOCK REPEATS ITS OWN BUCKET TOTAL as relcode 700000, and the Appendix
        # capitalises it `Other Religions and Persuasions` where C-01 writes `Other
        # religions and persuasions`. Matching on the string as delivered therefore fails
        # to recognise the parent, and every state's total is added as though it were a
        # named religion — which doubles the tail to 15.7M against a 7.9M bucket. Match on
        # the CODE, which is unambiguous, and let the spelling be whatever it is.
        if code == "700000":
            continue
        out.append(dict(
            geo_id=gid, geo_level=level, geo_name=name, source_category=cat, count=n,
            basis=BASIS, year=YEAR, source_id=SOURCE_ID + "_appendix",
            note=f"level=leaf; cat={cat}; parent={bucket}; relcode={code}"))
        if level == "state":
            apx_state_sum += n
            n_named += 1
    named = len({c for *_, c, _, code in apx_rows if code != "700000"})
    print(f"  appendix: {named} named religions, {n_named:,} state rows, "
          f"{apx_state_sum:,} people at state level")

    # THE UNNAMED REMAINDER OF THE BUCKET, emitted as a category of its own.
    #
    # The Appendix lists a religion only if it has 100+ adherents NATIONALLY, so every unit
    # has some `Other religions and persuasions` left over that no name accounts for —
    # 149,668 people, 1.9%, across India. Without a row for them the allocation would
    # normalise the named religions' shares to sum to 1 and hand the remainder out
    # proportionally, quietly inflating every Adivasi religion by about 2%. With this row
    # the shares are exact and the remainder lands on `other.in`, which is what it is.
    apx_by_geo = {}
    for level, gid, name, cat, n, code in apx_rows:
        if code != "700000":
            apx_by_geo[(level, gid)] = apx_by_geo.get((level, gid), 0) + n
    bucket_by_geo = {(r["geo_level"], r["geo_id"]): r["count"] for r in out
                     if r["source_category"] == bucket}
    n_resid = 0
    for (level, gid), total in bucket_by_geo.items():
        if level not in ("nation", "state"):
            continue                                    # the Appendix goes no finer
        rest = total - apx_by_geo.get((level, gid), 0)
        if rest < 0:
            raise SystemExit(f"{level} {gid}: named Appendix religions sum to "
                             f"{apx_by_geo[(level, gid)]:,}, more than the bucket's "
                             f"{total:,}")
        name = next(r["geo_name"] for r in out
                    if r["geo_level"] == level and r["geo_id"] == gid)
        out.append(dict(
            geo_id=gid, geo_level=level, geo_name=name,
            source_category=UNNAMED, count=rest, basis=BASIS, year=YEAR,
            source_id=SOURCE_ID + "_appendix",
            note=(f"level=leaf; cat={UNNAMED}; parent={bucket}; "
                  f"the Appendix names a religion only at 100+ adherents nationally, so "
                  f"this is that floor's remainder")))
        n_resid += 1
    print(f"  unnamed remainder emitted for {n_resid} units "
          f"({resid_total(out):,} people nationally)")

    # ASSERTION 4: the appendix's named religions must not exceed the bucket they came out
    # of. They will fall short — the appendix only lists religions with >=100 people
    # nationally — and the shortfall is the honest measure of the unnamed remainder.
    bucket_total = india_nat[bucket]
    if apx_state_sum > bucket_total:
        raise SystemExit(f"appendix names {apx_state_sum:,} people but the bucket holds "
                         f"{bucket_total:,} — the parent is wrong or rows are duplicated")
    resid = bucket_total - apx_state_sum
    print(f"  appendix covers {apx_state_sum / bucket_total:.1%} of the bucket; "
          f"{resid:,} people ({resid / bucket_total:.1%}) are in unnamed religions")

    # ---- annexure: recorded, not used
    anx = os.path.join(RAW, "DDW00C-01 Annexure MDDS.xlsx")
    n_anx = 0
    for level, gid, name, cat, n, rel, sc in read_annexure(anx):
        out.append(dict(
            geo_id=gid, geo_level=level, geo_name=name, source_category=f"Sect: {cat}",
            count=n, basis=BASIS, year=YEAR, source_id=SOURCE_ID + "_annexure",
            note=(f"write-in sect under relcode={rel}; sectcode={sc}; "
                  f"NOT a partition of the parent religion — see sources/in.md")))
        n_anx += 1
    print(f"  annexure: {n_anx:,} write-in sect rows recorded, not used for splitting")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(out)

    nsub = len({r["geo_id"] for r in out if r["geo_level"] == "subdistrict"})
    print(f"\nwrote {OUT}: {len(out):,} rows, {nsub:,} sub-districts")
    ns = india_nat[NOT_STATED]
    print(f"  religion not stated: {ns:,} ({ns / NATIONAL:.2%}) — the lowest of any "
          f"country on the map")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fetch", action="store_true", help="download the source files first")
    args = ap.parse_args()
    if args.fetch:
        fetch()
    normalise()


if __name__ == "__main__":
    main()
