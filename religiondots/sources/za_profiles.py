"""South Africa — the nine CS 2016 provincial profiles, kept as the RECONCILIATION CHECK.

This file used to be `sources/za.py`. It was the country's source until 2026-09-09, when the
CS 2016 person microdata arrived and South Africa was redrawn at 213 local municipalities
(ask/answered/002-za). The nine published provincial profiles are no longer what the map is
built from — **they are what the build is checked against**, which is a correctness check
almost no country here gets: the same survey, published by the same office, tabulated by
somebody else, over the same 24 categories.

`sources/za.py` imports `read_all()` and refuses to write `data/normalized/za.csv` unless
every one of the 24 published province cells reproduces from the microdata.

Nothing below is new. The parser is the one that shipped on 2026-09-08 and the five traps in
its docstring are still live, because a re-fetched PDF is still a PDF:

  * **The table NUMBER is not stable across the nine reports.** Free State, KwaZulu-Natal,
    Mpumalanga, North West and Northern Cape use 2.10a/2.10b; Eastern Cape and Gauteng use
    2.9a/2.9b; Limpopo uses 2.7/2.8 with no letter at all; Western Cape uses 2.11a/2.11b.
    Gauteng writes its own caption as `Table 2.9 a:` with a space inside the number. So the
    anchor is the caption TEXT (`religious affiliation`, `Christian denomination`) and the
    number is never matched on.
  * **The TOTAL row is usually labelled with the PROVINCE NAME, not "Total".** Eastern Cape,
    KwaZulu-Natal, Limpopo, Mpumalanga and North West all do this. Read as a data row it
    doubles the table and every share is halved, with no error anywhere.
  * **Northern Cape prints Bahaism as a bare dash** in both columns rather than as 0. Skip
    it and the province quietly has ten categories where the other eight have eleven.
  * **Gauteng spells it `Buddism`** and KwaZulu-Natal and Limpopo write `Seventh-Day
    Adventist` against everyone else's `Seventh Day Adventist`. Folded, and the fold is
    asserted to leave exactly 11 and 14 distinct categories. `Buddism` is the CS 2016
    codebook's own spelling and the microdata carries it too.
  * **Both "Other" rows exist and mean different things** -- one is a non-Christian religion,
    one is a Christian denomination. Every category is therefore emitted prefixed,
    `Religion: ...` / `Christian: ...`.

**AND THE ONE THING THE MICRODATA SETTLED.** Report 03-01-11 (North West) table 2.10b prints
fourteen rows summing to 3,072,039 against its own printed total of 3,408,521, and its
`Other` cell reads `21 873` -- character for character the `Do not know` figure in that same
table's own footnote. The province build could not tell whether that was a mis-set row or a
different universe, so it parked the 336,482 unaccounted people on bare `christianity`. The
microdata puts North West's `Other` at **358,355**, which is the printed 21,873 plus exactly
336,482, so the mis-set-row story is right to the person and the other one is wrong. The
defect is still asserted here, because a reissue that fixed it would change what
`sources/za.py`'s reconciliation is comparing against.

Usage:
    python sources/za_profiles.py --fetch   nine PDFs, ~58 MB, from cs2016.statssa.gov.za
    python sources/za_profiles.py           re-parse and print the nine tables
"""

import os
import re
import ssl
import sys
import urllib.request

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "za")

# cs2016.statssa.gov.za serves the nine provincial profiles as static WordPress uploads.
#
# ACCESS. The Stats SA hosts sit behind Imperva and their HTML pages are unreachable from a
# script -- `?page_id=` and `?p=` listings come back as a ~1 KB `_Incapsula_Resource` stub
# for curl AND for WebFetch, and a browser User-Agent does not help. But these PDF uploads
# are NOT walled. What blocks a plain `curl` on this host is the TLS chain, not the bot wall:
# it presents a self-signed intermediate and curl exits 60 before it ever sends the request,
# which reads exactly like a dead host. Downloading them needs the certificate check relaxed
# and nothing else.
BASE = "https://cs2016.statssa.gov.za/wp-content/uploads/2018/07/"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

# THE REPORT NUMBERS ARE READ OUT OF THE PDFs AND ASSERTED, not trusted from here. They do
# not run in province-code order and they are not guessable: Western Cape is 03-01-07 and
# Mpumalanga is 03-01-13, which is the number an alphabetical or a code-order guess gives to
# Western Cape. Three of nine were guessed wrong on the first pass, so `read_province` checks
# each one against the running header of the file it just opened.
PROVINCES = [
    ("ZA01", "WesternCape",  "Western Cape",  "03-01-07"),
    ("ZA02", "EasternCape",  "Eastern Cape",  "03-01-08"),
    ("ZA03", "NorthernCape", "Northern Cape", "03-01-14"),
    ("ZA04", "FreeState",    "Free State",    "03-01-12"),
    ("ZA05", "KZN",          "KwaZulu-Natal", "03-01-10"),
    ("ZA06", "NorthWest",    "North West",    "03-01-11"),
    ("ZA07", "Gauteng",      "Gauteng",       "03-01-09"),
    ("ZA08", "Mpumalanga",   "Mpumalanga",    "03-01-13"),
    ("ZA09", "Limpopo",      "Limpopo",       "03-01-15"),
]

CAPTION = re.compile(r"^Table\s+\d+\.\d+\s*[ab]?\s*:\s*(.*)$")
COUNT = re.compile(r"^-?[\d][\d\s   ]*$")
PCT = re.compile(r"^\d{1,3},\d$")
DASH = {"-", "–", "—"}

# Header cells that would otherwise be glued onto the first row's label -- PyMuPDF emits a
# table's header as ordinary lines, so `Christian domination` / `Catholic` reads as one
# two-line label.
#
# THE SET HAS TO BE PER-TABLE. Northern Cape heads its denomination table with the bare word
# `Christianity`, which is ALSO a real row label in the religion table; a single shared set
# either loses Northern Cape's Catholic row or deletes every province's Christianity row, and
# each failure is silent. Western Cape's `Christian domination` is Report 03-01-07's own typo,
# kept verbatim so the match works.
_HEADER_COMMON = {"number", "n", "%", "percentage", "religious affiliation",
                  "religious affiliation/belief"}
HEADER = {
    "a": _HEADER_COMMON | {"religion"},
    "b": _HEADER_COMMON | {"christian denomination", "christian denominations",
                           "christian domination", "denomination", "christianity"},
}

# The canonical answer sets. Asserted after folding, per province, both tables. These are the
# labels `sources/za.py` emits and `taxonomy/za2016.py` keys on, so they are the project's
# canonical spellings and the microdata's codebook labels are folded onto them.
RELIGIONS = [
    "Christianity", "Islam", "Traditional African religion", "Hinduism", "Buddhism",
    "Bahaism", "Judaism", "Atheism", "Agnosticism", "No religious affiliation/belief",
    "Other",
]
DENOMINATIONS = [
    "Catholic", "Anglican/Episcopalian", "Baptist", "Lutheran", "Methodist",
    "Presbyterian", "Pentecostal/Evangelistic",
    "African Independent Church/African Initiated Church", "Jehovah's Witness",
    "Seventh Day Adventist", "Mormon", "Reformed church",
    "Just a Christian/non-denominational", "Other",
]

VARIANTS = {
    "buddism": "Buddhism",
    "seventh-day adventist": "Seventh Day Adventist",
    "jehovahs witness": "Jehovah's Witness",
    "reformed church": "Reformed church",
    "just a christian/non-denominational": "Just a Christian/non-denominational",
}

# North West, Report 03-01-11 table 2.10b. Both figures are printed in that report and the
# microdata reproduces the second exactly, so the defect is a fixed, checkable quantity
# rather than a judgement.
NW_PRINTED_OTHER = 21_873
NW_SHORTFALL = 336_482


def fold(label):
    """Strip the parenthesised exemplar list, the smart apostrophes and the case.

    The exemplar lists are long and differ between reports -- Free State prints
    `Pentecostal/Evangelistic (e.g. Assemblies of God; ...)` where North West prints the bare
    word -- so they cannot be part of the key. The microdata's own codebook labels carry the
    same lists, truncated, which is why `sources/za.py` folds with this same function.
    """
    s = str(label)
    s = s.replace("‘", "'").replace("’", "'").replace("�", "'")
    s = re.sub(r"\((?:e\.?g\.?|eg)[^)]*\)?", " ", s, flags=re.I)
    s = re.sub(r"\(.*?\)", " ", s)
    s = " ".join(s.split()).strip(" .:,")
    return s


def key(label):
    s = fold(label).lower().replace("'", "").replace("’", "")
    return re.sub(r"\s+", " ", s).strip()


CANON = {}
for _c in RELIGIONS + DENOMINATIONS:
    CANON[key(_c)] = _c
for _k, _v in VARIANTS.items():
    CANON[_k] = _v


def to_int(s):
    return int(re.sub(r"[\s   ]", "", s))


def fetch():
    os.makedirs(RAW, exist_ok=True)
    ctx = ssl.create_default_context()
    # See the note on BASE. The chain, not the content, is what a plain client trips on.
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    for _gid, stem, name, _rep in PROVINCES:
        dest = os.path.join(RAW, stem + ".pdf")
        if os.path.exists(dest) and os.path.getsize(dest) > 1_000_000:
            print(f"have {stem}.pdf ({os.path.getsize(dest):,} bytes)")
            continue
        url = BASE + stem + ".pdf"
        print("GET", url)
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        body = urllib.request.urlopen(req, context=ctx, timeout=600).read()
        # §5a: a 200 is not a download. The Imperva stub is a ~1 KB HTML body served with a
        # 200, so the magic and the trailer are both checked
        # ([[reference_pdf_truncated_at_source]]).
        if not body.startswith(b"%PDF"):
            raise SystemExit(f"{url} did not return a PDF -- {len(body):,} bytes "
                             f"starting {body[:40]!r}")
        if b"%%EOF" not in body[-4096:]:
            raise SystemExit(f"{url} has no %%EOF trailer -- truncated at source, "
                             f"{len(body):,} bytes")
        tmp = dest + ".part"
        with open(tmp, "wb") as fh:
            fh.write(body)
        os.replace(tmp, dest)
        print(f"  {name}: {len(body):,} bytes")


def find_table(doc, needle):
    """Return the lines following the caption whose TITLE contains `needle`.

    Skips the front matter, because the contents page carries the same caption text with a
    dot leader and a page number after it.
    """
    for i, page in enumerate(doc):
        if i < 10:
            continue
        lines = page.get_text().split("\n")
        for j, line in enumerate(lines):
            m = CAPTION.match(line.strip())
            if not m or needle not in m.group(1).lower() or "......" in line:
                continue
            return i, line.strip(), lines[j + 1:]
    return None, None, None


def parse(lines, province, tab):
    """Rows are label / count / percent, one per line, the label sometimes wrapping.

    Terminates on a row whose label is `Total` OR the province's own name -- see the module
    docstring; five of the nine reports use the latter.
    """
    ends = {"total", province.lower(), province.lower().replace("-", " ")}
    rows, label, total = [], [], None
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line in DASH and label:
            rows.append({"label": " ".join(" ".join(label).split()), "n": 0,
                         "pct": 0.0, "dash": True})
            label = []
            if i + 1 < len(lines) and lines[i + 1].strip() in DASH:
                i += 1
        elif COUNT.match(line) and label:
            name = " ".join(" ".join(label).split())
            n = to_int(line)
            pct = None
            if i + 1 < len(lines) and PCT.match(lines[i + 1].strip()):
                pct = float(lines[i + 1].strip().replace(",", "."))
                i += 1
            if name.lower().strip(" .:") in ends:
                total = n
                i += 1
                break
            rows.append({"label": name, "n": n, "pct": pct, "dash": False})
            label = []
        elif not COUNT.match(line):
            if line.lower().strip(" .:") in HEADER[tab]:
                i += 1
                continue
            label.append(line)
        i += 1
    return rows, total


def read_province(stem, name, report):
    import fitz

    path = os.path.join(RAW, stem + ".pdf")
    if not os.path.exists(path):
        raise SystemExit(f"missing {path} -- run sources/za_profiles.py --fetch first")
    doc = fitz.open(path)
    # PyMuPDF reports page_count=0 on a file damaged at source rather than raising
    # ([[reference_pdf_truncated_at_source]]).
    if doc.page_count == 0:
        raise SystemExit(f"{path}: PyMuPDF reports zero pages -- damaged download")

    # Every page carries the running header `Provincial profile: <name> [Community Survey
    # 2016], Report <nnn>`, so the province AND its report number are both checkable against
    # the file actually opened. Eastern Cape's bibliography cites 03-01-63 as well, so the
    # test is that the expected number is the one in the running header -- i.e. the most
    # frequent -- rather than merely present somewhere.
    seen = {}
    for i in range(min(40, doc.page_count)):
        for m in re.findall(r"0\d-\d\d-\d\d", doc[i].get_text()):
            seen[m] = seen.get(m, 0) + 1
    if not seen:
        raise SystemExit(f"{path}: no report number anywhere in the first 40 pages")
    top = max(seen, key=lambda k: seen[k])
    if top != report:
        raise SystemExit(f"{path}: running header says report {top}, PROVINCES says "
                         f"{report} -- the wrong province's PDF is at this filename, or "
                         "Stats SA has renumbered")
    head = "".join(doc[i].get_text() for i in range(min(5, doc.page_count)))
    if name.lower() not in head.lower():
        raise SystemExit(f"{path}: {name!r} does not appear in the first five pages")

    out = {}
    for tab, needle, expect in (("a", "religious affiliation", RELIGIONS),
                                ("b", "christian denomination", DENOMINATIONS)):
        pg, cap, lines = find_table(doc, needle)
        if lines is None:
            raise SystemExit(f"{name}: no table captioned {needle!r} -- the report's table "
                             "numbering or wording has changed")
        rows, total = parse(lines, name, tab)
        if total is None:
            raise SystemExit(f"{name} 2.10{tab}: no total row found (looked for 'Total' "
                             f"and {name!r})")
        got = {}
        for r in rows:
            k = key(r["label"])
            canon = CANON.get(k)
            if canon is None:
                raise SystemExit(f"{name} 2.10{tab}: unrecognised category "
                                 f"{r['label']!r} (folded to {k!r})")
            if canon in got:
                raise SystemExit(f"{name} 2.10{tab}: {canon!r} appears twice")
            got[canon] = r
        missing = [c for c in expect if c not in got]
        extra = [c for c in got if c not in expect]
        if missing or extra:
            raise SystemExit(f"{name} 2.10{tab}: missing={missing} extra={extra}")
        out[tab] = {"page": pg, "caption": cap, "rows": got, "total": total}
    doc.close()
    return out


def read_all(verbose=True):
    """province name -> {'a': {category: count}, 'b': {...}, 'report': '03-01-nn'}.

    The published tables, as printed. `sources/za.py` compares the microdata against this.
    """
    out = {}
    for _gid, stem, name, rep in PROVINCES:
        rec = read_province(stem, name, rep)
        a = {c: r["n"] for c, r in rec["a"]["rows"].items()}
        b = {c: r["n"] for c, r in rec["b"]["rows"].items()}
        sa = sum(a.values())
        if abs(sa - rec["a"]["total"]) > 2:
            raise SystemExit(f"{name} 2.10a: rows sum to {sa:,}, printed total "
                             f"{rec['a']['total']:,}")
        out[name] = {"a": a, "b": b, "report": rep,
                     "a_total": rec["a"]["total"], "b_total": rec["b"]["total"],
                     "exemplars": {c: r["label"] for c, r in rec["b"]["rows"].items()}}
        if verbose:
            print(f"  {name:15s} Report {rep}  2.10a {sa:>10,}  "
                  f"2.10b {sum(b.values()):>10,}")

    # ---- the one province whose own table does not add up ----
    nw = out["North West"]
    short = nw["b_total"] - sum(nw["b"].values())
    if short != NW_SHORTFALL or nw["b"]["Other"] != NW_PRINTED_OTHER:
        raise SystemExit(
            f"North West's table 2.10b has changed: shortfall {short:,} (expected "
            f"{NW_SHORTFALL:,}), Other cell {nw['b']['Other']:,} (expected "
            f"{NW_PRINTED_OTHER:,}). Stats SA has reissued Report 03-01-11, so re-read this "
            "module's docstring and sources/za.md §4.1 before trusting the reconciliation "
            "in sources/za.py, which expects exactly this defect.")
    if verbose:
        print(f"\n  KNOWN DEFECT, Report 03-01-11 (North West): table 2.10b's fourteen rows "
              f"fall\n    {short:,} short of its own printed total, and its `Other` cell "
              f"reads {nw['b']['Other']:,},\n    which is the `Do not know` figure from that "
              "table's own footnote. The microdata\n    puts that cell at "
              f"{NW_PRINTED_OTHER + NW_SHORTFALL:,}, so the row was mis-set.")
    return out


def main():
    if "--fetch" in sys.argv:
        fetch()
    print("South Africa - Community Survey 2016 provincial profiles, Stats SA")
    print("  (the RECONCILIATION CHECK; the map is built from the microdata by sources/za.py)\n")
    prov = read_all()
    nat_a, nat_b = {}, {}
    for rec in prov.values():
        for c, n in rec["a"].items():
            nat_a[c] = nat_a.get(c, 0) + n
        for c, n in rec["b"].items():
            nat_b[c] = nat_b.get(c, 0) + n
    print("\n  national, as published:")
    for c in RELIGIONS:
        print(f"    Religion:  {c:52s} {nat_a[c]:>11,}")
    for c in DENOMINATIONS:
        print(f"    Christian: {c:52s} {nat_b[c]:>11,}")


if __name__ == "__main__":
    main()
