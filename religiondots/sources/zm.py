"""Zambia — the 2022 census religion volume, at constituency, with the denominations spread from
the province tables inside each constituency's own rural and urban population.

Writes data/normalized/zm.csv. Reads two ZamStats PDFs into data/raw/zm/:

  * **2022 Census of Population and Housing, Series B: Religion Descriptive Tables** (ZamStats,
    April 2026), 39 pages. `wp-content/uploads/2026/04/Religion-Descriptive-Tables-Final.pdf`.
  * **2022 Census of Population and Housing, Summary Report Part 2** (ZamStats, Sept 2024),
    148 pages, for Table 5.2's de jure constituency totals. Used for a check and the gap
    figure only; nothing drawn comes from it.

## WHAT THE RELIGION VOLUME PRINTS, AND AT WHICH TIER

| table | geography | categories |
|---|---|---|
| B.1 | 10 provinces | 10 religions: Christianity, Islam, Judaism, Hinduism, Buddhism, Bahai Faith, Sikhism, African Traditional Religion, Non-Religious, Other Religious Groups |
| B.3 | province x rural/urban | Christianity / Other Religious Groups |
| B.4 | province, district, **constituency** x sex | Christianity / Other Religious Groups |
| B.5 | province, district, **constituency** x rural/urban | Christianity / Other Religious Groups |
| B.6 | 10 provinces | **23 named Christian denominations**, plus None and Other |
| B.9, B.10 | province, rural and urban separately | the same 25 |
| B.12 | national x sex x rural/urban | the same 25 |

So the constituency knows how many Christians and non-Christians it has, rural and urban, and
nothing finer; the province knows the rest. §3.10c's `--within` rule applies unchanged: each
constituency's rural Christians take their province's rural denominational mix and its urban
Christians the urban mix, so every denomination reproduces B.9 and B.10 exactly on
re-aggregation, and every non-Christian religion reproduces B.1. Nothing here is at a tier the
census did not print. The residence split is worth having: Pentecostals are 24.7% of urban
Christians and 11.7% of rural ones, the Adventists and the New Apostolic Church the reverse.

EVERY DRAWN ROW IS `derived` (§7). The two measured columns are the parents, and every row
carries `parent_column=` naming the one it came out of (§7a-i-1).

## TWO COLUMN LABELS IN B.1 ARE WRONG, AND THE OFFICE'S OWN ANALYTICAL REPORT SAYS WHICH

The *2022 Census National Analytical Report* (ZamStats, August 2025), §3.4 and Figure 3.13,
gives the national split as **Christianity 98.0, Islam 0.5, African Traditional 0.2, Other 0.1,
None 1.3**, and its key findings say *"1.3 percent reported no religious affiliation"*. Read
with B.1's printed headers that cannot be reproduced: `Non-Religious` is 9,238 (0.05%) and
`African Traditional Religion` is 463 (0.003%). It reproduces EXACTLY, all five figures, if
B.1's `Judaism` column (30,502, 0.17%) is the traditional religion and its `Other Religious
Groups` column (233,260, 1.27%) is the no-religion answer, with the six small columns between
them (21,401, 0.12%) the report's `Other`. The demographic evidence agrees and does not depend
on the arithmetic:

  * the report's Figure 3.14 puts its `None` at **1.8% of men and 0.8% of women**. Irreligion is
    the male-skewed answer here: the 2010 census's Lusaka `None` was 26,603 men to 8,676 women
    while its `Other` was 23,182 to 24,077 (2010 Lusaka Province tables, B5).
  * the report puts `African Traditional` at 0.3% rural and 0.1% urban, and B.1's `Judaism`
    column is half in Eastern Province (15,625 of 30,502), the Chewa and Ngoni country. A
    Zambian Jewish community of 30,000 concentrated in rural Eastern Province does not exist.

So the mapping (taxonomy/zm2022.py) files the two by the report's reading and says why, and
this module asserts on every build that the report's five figures come out of B.1 under that
reading and do NOT come out under the printed one. A corrected re-issue of B.1 breaks the build
rather than leaving the relabel standing by inertia. The source_category strings stay as B.1
printed them, per §2.4.

## A COLUMN NAME THAT COLLIDES WITH ITSELF (spec §12, Iraq)

`Other Religious Groups` is B.1's last column, 233,260 people, and ALSO B.2-B.5's name for every
non-Christian at once, 373,966. They are written here as `Other Religious Groups (B.1 column)`
and `Other Religious Groups (B.2-B.5, all non-Christian)` so neither can be read as the other.

## THE UNIVERSE IS DE FACTO, AND IT IS 6.9% SMALLER THAN THE CENSUS TOTAL

Every religion table is headed *population (de facto)* and sums to **18,340,343**. The census
total everyone quotes is the de jure count, **19,693,423** (Summary Report Part 2, 2.1), which
adds usual members absent on census night, the institutional population (47,941) and the
homeless (4,063) and drops visitors. The difference is 1,353,080 and no ZamStats publication
breaks it down. The per-constituency ratio is printed below and is the check that the join is
right: a constituency joined to the wrong polygon shows up as a ratio far from the rest.

Usage:
    python sources/zm.py --fetch     download the two PDFs (~9 MB) into data/raw/zm/
    python sources/zm.py             parse, check, allocate, write data/normalized/zm.csv
"""

import os
import re
import sys
import urllib.request

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "zm")
LOOKUP = os.path.join(ROOT, "data", "geo", "zm", "zm_lookup.csv")
OUT = os.path.join(ROOT, "data", "normalized", "zm.csv")

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) religiondots/1.0"}

PDFS = {
    "zm2022_religion_descriptive_tables.pdf":
        "https://www.zamstats.gov.zm/wp-content/uploads/2026/04/Religion-Descriptive-Tables-Final.pdf",
    "zm2022_summary_report_part2.pdf":
        "https://www.zamstats.gov.zm/wp-content/uploads/2024/09/"
        "2022-Census-of-Population-and-Housing-Summary-Report-Part-2.pdf",
}
RELIGION_PDF = os.path.join(RAW, "zm2022_religion_descriptive_tables.pdf")
SUMMARY_PDF = os.path.join(RAW, "zm2022_summary_report_part2.pdf")

SOURCE_ID = "zm_census2022_religion_series_b"
DE_FACTO_TOTAL = 18_340_343      # every religion table
DE_JURE_TOTAL = 19_693_423       # Summary Report Part 2, 2.1

PROVINCES = ["Central", "Copperbelt", "Eastern", "Luapula", "Lusaka", "Muchinga", "Northern",
             "North Western", "Southern", "Western"]
PROVINCE_ALIASES = {"Northwestern": "North Western"}

CHRISTIAN = "Christianity"
NONCHRISTIAN = "Other Religious Groups (B.2-B.5, all non-Christian)"

# B.1's columns after Total and Christianity, in printed order, as printed.
B1_OTHER = ["Islam", "Judaism", "Hinduism", "Buddhism", "Bahai Faith", "Sikhism",
            "African Traditional Religion", "Non-Religious", "Other Religious Groups (B.1 column)"]
B1_HEADER = ("Total Christianity Islam Judaism Hinduism Buddhism Bahai Faith Sikhism African "
             "Traditional Religion Non- Religious Other Religious Groups")

# The National Analytical Report's Figure 3.13, and which B.1 columns make each figure under
# the report's reading. Asserted both ways in check_relabel().
REPORT_FIG_3_13 = {"Christianity": 98.0, "Islam": 0.5, "African Traditional": 0.2,
                   "Other": 0.1, "None": 1.3}
REPORT_READING = {
    "Christianity": [CHRISTIAN],
    "Islam": ["Islam"],
    "African Traditional": ["Judaism"],
    "Other": ["Hinduism", "Buddhism", "Bahai Faith", "Sikhism",
              "African Traditional Religion", "Non-Religious"],
    "None": ["Other Religious Groups (B.1 column)"],
}
PRINTED_READING = {
    "Christianity": [CHRISTIAN],
    "Islam": ["Islam"],
    "African Traditional": ["African Traditional Religion"],
    "Other": ["Judaism", "Hinduism", "Buddhism", "Bahai Faith", "Sikhism",
              "Other Religious Groups (B.1 column)"],
    "None": ["Non-Religious"],
}

# B.6 / B.9 / B.10 row labels as printed, whitespace collapsed and the apostrophe straightened.
# `None` and `Other` are the rows of a table headed `Christianity Denomination`, so they are
# Christians who named no denomination and Christians of an unlisted one; prefixed here so they
# cannot be mistaken for B.1's religion-level answers.
DENOMS = ["Anglican", "Apostolic Faith Mission", "Baptist", "Brethren in Christ", "Catholic",
          "Christian Missions in Many Lands (CMML)", "Church of Christ", "Episcopal",
          "Evangelical Church in Zambia", "Jehovah's Witness (Watchtower)", "Latter-Day Saints",
          "Lutheran", "Methodist", "New Apostolic", "Orthodox", "Pentecostal", "Presbyterian",
          "Reformed Church In Zambia (RCZ/Dutch)", "Restoration", "Salvation Army",
          "Seventh Day Adventist (SDA)", "Wesleyan", "United Church Of Zambia (UCZ)",
          "None", "Other"]
DENOM_PREFIXED = {"None": "Christianity Denomination: None",
                  "Other": "Christianity Denomination: Other"}

NUM = re.compile(r"^(?:\d{1,3}(?:,\d{3})*|\d+|-|\*)$")


# One North-Western district is spelled three ways: B.4 `Ikelenge`, B.5 `Ikelengi`, COD-AB
# `Ikeleng'i`. Folded to one key; anything else that fails to match still fails.
FOLD_ALIASES = {"ikelenge": "ikelengi"}

# (folded district, folded census constituency) -> folded COD-AB constituency. Two are
# spellings; three are districts with one constituency that COD-AB names for the district and
# the Electoral Commission names for Solwezi, which the join asserts is safe (sole
# constituency of its district in both files).
CONSTITUENCY_ALIASES = {
    ("shibuyunji", "mwembeshi"): "mwembezhi",
    ("petauke", "petaukecentral"): "petauke",
    ("kalumbila", "solweziwest"): "kalumbila",
    ("mushindamo", "solwezieast"): "mushindamo",
    ("solwezi", "solwezicentral"): "solwezi",
}
SOLE_CONSTITUENCY = {"kalumbila", "mushindamo", "solwezi"}

# Summary Report Table 5.2 -> the religion volume's name, (folded district, folded name).
DEJURE_ALIASES = {("mpongwe", "mpongwecentral"): "mpongwe"}


def fold(s):
    k = "".join(ch for ch in str(s).lower() if ch.isalnum())
    return FOLD_ALIASES.get(k, k)


def num(tok):
    """A cell. `-` is ZamStats' "not applicable", a zero. `*` is a suppressed cell and comes back
    as None; only B.5 has any, and parse_b5() recovers every one or fails."""
    if tok == "-":
        return 0
    if tok == "*":
        return None
    return int(tok.replace(",", ""))


def clean(s):
    return " ".join(s.replace("’", "'").replace("�", "'").split())


def fetch():
    os.makedirs(RAW, exist_ok=True)
    for name, url in PDFS.items():
        dst = os.path.join(RAW, name)
        if os.path.exists(dst) and os.path.getsize(dst) > 1_000_000:
            print(f"  have {name} ({os.path.getsize(dst):,} bytes)")
            continue
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=600) as r:
            data = r.read()
        # §5a and [[reference_pdf_truncated_at_source]]: a 200 is not a download, and a
        # matching Content-Length is not a whole file either.
        if data[:4] != b"%PDF" or b"%%EOF" not in data[-2048:]:
            raise SystemExit(f"{name} is not a whole PDF ({len(data):,} bytes)")
        with open(dst + ".part", "wb") as f:
            f.write(data)
        os.replace(dst + ".part", dst)
        print(f"  got  {name} ({len(data):,} bytes)")


# =======================================================================================
# the religion volume
# =======================================================================================

def page_lines(doc, pno, title):
    """A page's text lines from its TABLE title onward, which drops the page number."""
    lines = doc[pno - 1].get_text().splitlines()
    for i, ln in enumerate(lines):
        if ln.strip().upper().startswith("TABLE") and title in clean(ln).upper().replace(" ", ""):
            return [clean(x) for x in lines[i:] if x.strip()]
    raise SystemExit(f"page {pno}: no title matching {title!r}")


def walk_labelled(lines, label_re, n, what):
    """Records of a label line followed by exactly n numbers."""
    recs, cur = [], None
    for s in lines:
        m = label_re.match(s)
        if m:
            if cur is not None and len(cur[1]) != n:
                raise SystemExit(f"{what}: {cur[0]} has {len(cur[1])} numbers, expected {n}")
            cur = (m.groups(), [])
            recs.append(cur)
        elif NUM.match(s):
            if cur is None:
                continue                      # the national row above the first label
            if len(cur[1]) >= n:
                raise SystemExit(f"{what}: extra number {s!r} after {cur[0]}")
            cur[1].append(num(s))
        elif cur is not None and 0 < len(cur[1]) < n:
            raise SystemExit(f"{what}: text {s!r} inside the numbers of {cur[0]}")
    if cur is not None and len(cur[1]) != n:
        raise SystemExit(f"{what}: last record {cur[0]} has {len(cur[1])} numbers")
    return recs


def walk_named(lines, n, what):
    """Records of one or more text lines (a name) followed by n numbers, from the `Total` row."""
    start = next((i for i in range(len(lines) - 1)
                  if lines[i] == "Total" and NUM.match(lines[i + 1])), None)
    if start is None:
        raise SystemExit(f"{what}: no Total row")
    header = " ".join(lines[:start])
    recs, name, cur = [], [], None
    for s in lines[start:]:
        if NUM.match(s):
            if cur is None or len(cur[1]) == n:
                if not name:
                    raise SystemExit(f"{what}: numbers with no name after {cur and cur[0]}")
                cur = (" ".join(name), [])
                recs.append(cur)
                name = []
            cur[1].append(num(s))
        else:
            if cur is not None and len(cur[1]) != n:
                raise SystemExit(f"{what}: text {s!r} inside the numbers of {cur[0]}")
            name.append(s)
    if cur is None or len(cur[1]) != n or name:
        raise SystemExit(f"{what}: table ends mid-record")
    return header, recs


def parse_b1(doc):
    lines = page_lines(doc, 6, "B.1:")
    cut = next(i for i, s in enumerate(lines) if s.upper().startswith("TABLE B.2"))
    lines = lines[:cut]
    head = " ".join(lines[: next(i for i, s in enumerate(lines) if s == "Zambia")])
    if B1_HEADER not in head:
        raise SystemExit(f"Table B.1's header has changed -- re-read the column order:\n{head}")
    names = ["Zambia"] + PROVINCES
    label = re.compile(r"^(" + "|".join(map(re.escape, names)) + r")$")
    # The national row IS a label here, so walk from the first `Zambia` after the header.
    recs = walk_labelled(lines, label, 11, "B.1")
    cols = ["Total", CHRISTIAN] + B1_OTHER
    b1 = pd.DataFrame([r[1] for r in recs], index=[r[0][0] for r in recs], columns=cols)
    if b1.isna().any().any():
        raise SystemExit("B.1 has a suppressed cell")
    if list(b1.index) != names:
        raise SystemExit(f"B.1 rows {list(b1.index)}")
    parts = b1[cols[1:]].sum(axis=1)
    if not (parts == b1["Total"]).all():
        raise SystemExit(f"B.1 columns do not sum to Total:\n{(parts - b1['Total'])}")
    if not (b1.loc[PROVINCES].sum() == b1.loc["Zambia"]).all():
        raise SystemExit("B.1 provinces do not sum to Zambia")
    if b1.loc["Zambia", "Total"] != DE_FACTO_TOTAL:
        raise SystemExit(f"B.1 total {b1.loc['Zambia', 'Total']:,}")
    return b1


def parse_b5(doc, table="B.5", pages=range(19, 31), cols=("Total", "Rural", "Urban")):
    lines = []
    for p in pages:
        lines += page_lines(doc, p, table.upper() + ":" if table == "B.4" else table.upper())
    label = re.compile(r"^(Province|District|Constituency) (.+)$")
    recs = walk_labelled(lines, label, 9, table)
    rows, prov, dist = [], None, None
    for (kind, name), v in recs:
        if kind == "Province":
            # B.4 and B.5 write `Northwestern`; B.1, B.3 and B.6-B.10 write `North Western`.
            name = PROVINCE_ALIASES.get(name, name)
            prov, dist = name, None
        elif kind == "District":
            dist = name
        rows.append(dict(kind=kind, province=prov, district=dist, name=name,
                         **{f"all_{c}": v[i] for i, c in enumerate(cols)},
                         **{f"chr_{c}": v[3 + i] for i, c in enumerate(cols)},
                         **{f"oth_{c}": v[6 + i] for i, c in enumerate(cols)}))
    df = pd.DataFrame(rows)
    # THE SUPPRESSED CELLS ARE RECOVERABLE, EXACTLY, TWO WAYS. Every `*` in B.5 is the ninth
    # value of its row, non-Christians in the urban part of a small-town constituency (Chilubi,
    # Zambezi West, Gwembe, Luampa, Mwandi, Nkeyema, Shangombo, Sikongo, and their districts).
    # The same row prints the urban total and the urban Christians, and also all non-Christians
    # and the rural ones, so the cell is `all_Urban - chr_Urban` and `oth_Total - oth_Rural`.
    # Both are asserted, and so is every other column being unsuppressed.
    last = f"oth_{cols[2]}"
    others = [c for c in df.columns if c.startswith(("all_", "chr_", "oth_")) and c != last]
    if df[others].isna().any().any():
        raise SystemExit(f"{table}: a suppressed cell outside the non-Christian {cols[2]} column")
    hole = df[last].isna()
    if hole.any():
        a = df.loc[hole, f"all_{cols[2]}"] - df.loc[hole, f"chr_{cols[2]}"]
        b = df.loc[hole, "oth_Total"] - df.loc[hole, f"oth_{cols[1]}"]
        if not (a == b).all() or (a < 0).any():
            raise SystemExit(f"{table}: suppressed cells do not recover consistently")
        df.loc[hole, last] = a
        print(f"  {table}: {int(hole.sum())} suppressed cells recovered by subtraction, two ways "
              f"agreeing ({int(a.sum()):,} people in total, largest {int(a.max()):,})")
    df[last] = df[last].astype("int64")
    for c in cols:
        if not (df[f"chr_{c}"] + df[f"oth_{c}"] == df[f"all_{c}"]).all():
            raise SystemExit(f"{table}: Christianity + Other != Total in column {c}")
    for g in ("all", "chr", "oth"):
        if not (df[f"{g}_{cols[1]}"] + df[f"{g}_{cols[2]}"] == df[f"{g}_{cols[0]}"]).all():
            raise SystemExit(f"{table}: {cols[1]} + {cols[2]} != {cols[0]} for {g}")
    # The nesting: constituencies sum to their district, districts to their province.
    con = df[df.kind == "Constituency"]
    dis = df[df.kind == "District"]
    pro = df[df.kind == "Province"]
    num_cols = [c for c in df.columns if c.startswith(("all_", "chr_", "oth_"))]
    a = con.groupby(["province", "district"])[num_cols].sum()
    b = dis.set_index(["province", "name"])[num_cols]
    b.index.names = ["province", "district"]
    if not a.sort_index().equals(b.sort_index()):
        raise SystemExit(f"{table}: constituencies do not sum to their districts")
    a = dis.groupby("province")[num_cols].sum()
    if not a.sort_index().equals(pro.set_index("name")[num_cols].sort_index()):
        raise SystemExit(f"{table}: districts do not sum to their provinces")
    if list(pro["name"]) != PROVINCES:
        raise SystemExit(f"{table}: provinces {list(pro['name'])}")
    if int(pro["all_Total"].sum()) != DE_FACTO_TOTAL:
        raise SystemExit(f"{table}: total {pro['all_Total'].sum():,}")
    print(f"  {table}: {len(pro)} provinces, {len(dis)} districts, {len(con)} constituencies, "
          "every level sums to the one above it")
    return df


def parse_denoms(doc, pno, title, n=11, ordered=True):
    """`ordered=False` for B.12, which sorts its rows by size instead of B.6's order."""
    lines = page_lines(doc, pno, title)
    header, recs = walk_named(lines, n, title)
    # B.6 wraps `(RCZ/` onto its own line and B.9 does not, so a joined label can carry a
    # space after the slash that the other table's does not.
    recs = [(re.sub(r"/\s+", "/", clean(r[0])), r[1]) for r in recs]
    got = [r[0] for r in recs]
    body = got[1:] if ordered else sorted(got[1:])
    want = DENOMS if ordered else sorted(DENOMS)
    if got[0] != "Total" or body != want:
        raise SystemExit(f"{title}: rows are {got}")
    return header, recs


def parse_province_denoms(doc, pno, title):
    header, recs = parse_denoms(doc, pno, title)
    want = "Total Central Copperbelt Eastern Luapula Lusaka Muchinga Northern North Western " \
           "Southern Western"
    if want not in header:
        raise SystemExit(f"{title}: province columns have changed:\n{header}")
    df = pd.DataFrame([r[1] for r in recs], index=["Total"] + DENOMS,
                      columns=["Zambia"] + PROVINCES)
    if df.isna().any().any():
        raise SystemExit(f"{title}: a suppressed cell")
    if not (df.loc[DENOMS].sum() == df.loc["Total"]).all():
        raise SystemExit(f"{title}: denominations do not sum to the Total row")
    if not (df[PROVINCES].sum(axis=1) == df["Zambia"]).all():
        raise SystemExit(f"{title}: provinces do not sum to the national column")
    return df


# =======================================================================================
# the summary report's de jure constituency totals (a check and the gap, nothing drawn)
# =======================================================================================

def parse_dejure(doc, census_keys):
    """Table 5.2's constituency rows. Indentation separates constituency from ward on most
    pages but not all, so a shallow row is a constituency only if B.5 lists a constituency of
    that name in that district; the rest are printed. The sums below then prove it."""
    rows, prov, dist, cur = [], None, None, None
    for p in range(36, 81):
        raw = doc[p - 1].get_text().splitlines()
        i = next((k for k, s in enumerate(raw) if s.strip().startswith("TABLE 5.2")), None)
        if i is None:
            raise SystemExit(f"summary report page {p}: no TABLE 5.2 title")
        for s in raw[i + 1:]:
            if not s.strip():
                continue
            toks = s.split()
            if all(NUM.match(t) for t in toks):
                if cur is None:
                    raise SystemExit(f"Table 5.2 page {p}: numbers before any label")
                cur["v"] += [num(t) for t in toks]
                if len(cur["v"]) > 9:
                    raise SystemExit(f"Table 5.2: {cur['name']} has {len(cur['v'])} numbers")
                continue
            if s.strip().startswith(("Dejure Population", "Province/", "Constituency/")):
                continue                     # the title and column header
            if s.strip() in ("Total", "Rural", "Urban", "Both", "Sexes", "Male", "Female"):
                continue
            if cur is not None and not cur["v"]:
                # A name wrapped onto a second line (`Harry Mwanga` / `Nkumbula`): the record
                # has no numbers yet, so this line is the rest of its name.
                cur["name"] = clean(cur["name"] + " " + s)
                continue
            indent = len(s) - len(s.lstrip(" "))
            name = clean(s)
            # Most pages write `CENTRAL PROVINCE` / `CHIBOMBO DISTRICT` / an indented name, but
            # a run of pages from Luapula on writes `PROVINCE LUAPULA` / `DISTRICT KAWAMBWA` /
            # `Constituency Liuwa`. Both forms are read.
            if name.endswith("PROVINCE"):
                name = name[: -len("PROVINCE")].strip()
                prov, kind = name, "province"
            elif name.startswith("PROVINCE "):
                name = name[len("PROVINCE "):].strip()
                prov, kind = name, "province"
            elif name.endswith("DISTRICT"):
                name = name[: -len("DISTRICT")].strip()
                dist, kind = name, "district"
            elif name.startswith("DISTRICT "):
                name = name[len("DISTRICT "):].strip()
                dist, kind = name, "district"
            elif name.lower().startswith("constituency "):
                name, kind = name[len("constituency "):].strip(), "constituency"
            elif indent <= 13:
                kind = "constituency"
            else:
                kind = "ward"
            cur = dict(kind=kind, province=prov, district=dist, name=name, v=[])
            rows.append(cur)
    bad = [r for r in rows if len(r["v"]) != 9]
    if bad:
        raise SystemExit(f"Table 5.2: {len(bad)} rows without 9 numbers, first {bad[0]}")
    df = pd.DataFrame([dict(kind=r["kind"], province=r["province"], district=r["district"],
                            name=r["name"], total=r["v"][0]) for r in rows])
    shallow = df.kind == "constituency"
    df["key"] = [(fold(d), DEJURE_ALIASES.get((fold(d), fold(n)), fold(n)))
                 for d, n in zip(df.district, df.name)]
    listed = [k in census_keys for k in df["key"]]
    stray = df[shallow & ~pd.Series(listed, index=df.index)]
    for _, r in stray.iterrows():
        print(f"    Table 5.2: {r['name']!r} in {r.district} is indented like a constituency "
              "and is not one in B.5; read as a ward")
    df.loc[stray.index, "kind"] = "ward"
    con = df[df.kind == "constituency"]
    if len(con) != 156:
        raise SystemExit(f"Table 5.2: {len(con)} constituencies, expected 156")
    if int(df.loc[df.kind == "province", "total"].sum()) != DE_JURE_TOTAL:
        raise SystemExit("Table 5.2 provinces do not sum to the de jure total")
    if int(con["total"].sum()) != DE_JURE_TOTAL:
        raise SystemExit("Table 5.2 constituencies do not sum to the de jure total")
    return con


# =======================================================================================
# checks that are about meaning, not arithmetic
# =======================================================================================

def check_relabel(b1):
    nat = b1.loc["Zambia"]
    tot = float(nat["Total"])

    def shares(reading):
        return {k: round(100.0 * sum(nat[c] for c in cols) / tot, 1)
                for k, cols in reading.items()}

    rep, prn = shares(REPORT_READING), shares(PRINTED_READING)
    print("\n  National Analytical Report, Figure 3.13, against B.1 read two ways:")
    print(f"    {'':<22}{'report':>8}{'B.1 relabelled':>16}{'B.1 as printed':>16}")
    for k, v in REPORT_FIG_3_13.items():
        print(f"    {k:<22}{v:>8.1f}{rep[k]:>16.1f}{prn[k]:>16.1f}")
    if rep != REPORT_FIG_3_13:
        raise SystemExit("B.1 under the analytical report's reading no longer reproduces "
                         "Figure 3.13 -- the table has been re-issued; re-read the columns and "
                         "revisit taxonomy/zm2022.py's REVIEW before building")
    if prn == REPORT_FIG_3_13:
        raise SystemExit("B.1 AS PRINTED now reproduces Figure 3.13 -- the headers have been "
                         "corrected; drop the relabel in taxonomy/zm2022.py")
    print("    -> the report's five figures come out of B.1 only with `Judaism` read as the "
          "traditional religion and `Other Religious Groups` as no religion")


def print_none_sex(doc):
    header, recs = parse_denoms(doc, 38, "B.12:", n=9, ordered=False)
    d = {clean(r[0]): r[1] for r in recs}
    for k in ("Total", "Catholic", "Pentecostal", "None"):
        t, m, f = d[k][0], d[k][1], d[k][2]
        print(f"    {k:<14} {t:>11,}  men {100.0 * m / t:5.1f}%")
    print("    -> Christians who named no denomination are 68% men, the no-religion profile")


# =======================================================================================

def main():
    if "--fetch" in sys.argv:
        fetch()
    for p in (RELIGION_PDF, SUMMARY_PDF):
        if not os.path.exists(p):
            raise SystemExit(f"missing {p} -- run with --fetch first")
    if not os.path.exists(LOOKUP):
        raise SystemExit(f"missing {LOOKUP} -- run sources/zm_geo.py first")
    import fitz

    doc = fitz.open(RELIGION_PDF)
    if doc.page_count != 39:
        raise SystemExit(f"religion volume has {doc.page_count} pages, expected 39")

    print("Series B, Religion Descriptive Tables:")
    b1 = parse_b1(doc)
    b5 = parse_b5(doc)
    b4 = parse_b5(doc, table="B.4", pages=range(8, 19), cols=("Total", "Male", "Female"))
    # B.4 and B.5 are two typings of the same unit list and do not spell every name the same
    # way, so they are matched on folded names and the raw differences are printed.
    for t in (b4, b5):
        t["key"] = [(k_, fold(p), fold(d), fold(n)) for k_, p, d, n in
                    zip(t["kind"], t["province"], t["district"], t["name"])]
    if set(b4["key"]) != set(b5["key"]) or b4["key"].duplicated().any():
        raise SystemExit(f"B.4 and B.5 do not list the same units even folded:\n"
                         f"  B.5 only {sorted(set(b5['key']) - set(b4['key']))[:10]}\n"
                         f"  B.4 only {sorted(set(b4['key']) - set(b5['key']))[:10]}")
    m = b5.merge(b4, on="key", suffixes=("", "_b4"))
    diff = m[(m["name"] != m["name_b4"]) | (m["district"].fillna("") != m["district_b4"].fillna(""))]
    for _, r in diff.iterrows():
        print(f"  B.4 writes {r['district_b4']!s} / {r['name_b4']!r} where B.5 writes "
              f"{r['district']!s} / {r['name']!r}")
    for g in ("all", "chr", "oth"):
        if not (m[f"{g}_Total"] == m[f"{g}_Total_b4"]).all():
            raise SystemExit(f"B.4 and B.5 disagree on {g} totals")
    print("  B.4 (by sex) and B.5 (by residence) agree on every unit's totals")

    pro5 = b5[b5.kind == "Province"].set_index("name")
    if not (pro5["chr_Total"] == b1.loc[PROVINCES, CHRISTIAN]).all():
        raise SystemExit("B.5 province Christians != B.1")
    if not (pro5["oth_Total"] == b1.loc[PROVINCES, B1_OTHER].sum(axis=1)).all():
        raise SystemExit("B.5 province non-Christians != the sum of B.1's nine columns")

    b6 = parse_province_denoms(doc, 31, "B.6:")
    b9 = parse_province_denoms(doc, 34, "B.9:")
    b10 = parse_province_denoms(doc, 35, "B.10:")
    if not (b9 + b10).equals(b6):
        raise SystemExit("B.9 rural + B.10 urban != B.6")
    if not (b6.loc["Total", PROVINCES] == b1.loc[PROVINCES, CHRISTIAN]).all():
        raise SystemExit("B.6 province Christians != B.1")
    if not (b9.loc["Total", PROVINCES] == pro5["chr_Rural"]).all():
        raise SystemExit("B.9 rural Christians != B.5's province rows")
    if not (b10.loc["Total", PROVINCES] == pro5["chr_Urban"]).all():
        raise SystemExit("B.10 urban Christians != B.5's province rows")
    print("  B.6 = B.9 + B.10 cell for cell, and all three agree with B.1 and B.5")

    check_relabel(b1)
    print("\n  B.12, share of men in a denomination row:")
    print_none_sex(doc)

    # ---- the join, asserted as a bijection
    #
    # ON DISTRICT AND CONSTITUENCY, NOT PROVINCE. District names are unique nationally in both
    # files (asserted), and the province is the one attribute the two disagree on: COD-AB
    # (boundaries created 2020-11) still files Chirundu district under Lusaka, while the census,
    # on the December 2021 boundaries, tabulates it under Southern. The polygon is the same
    # district. The allocation uses the CENSUS's province, because that is the table Chirundu's
    # people were counted into.
    lut = pd.read_csv(LOOKUP, dtype=str)
    con = b5[b5.kind == "Constituency"].copy()
    for what, frame in (("COD-AB", lut), ("census", con)):
        dp = frame.groupby(frame["district"].map(fold))["province"].nunique()
        if (dp > 1).any():
            raise SystemExit(f"{what}: a district name is used in two provinces: "
                             f"{sorted(dp.index[dp > 1])}")
    key_l = {(fold(r.district), fold(r["name"])): r.unit for _, r in lut.iterrows()}
    if len(key_l) != len(lut):
        raise SystemExit("two COD constituencies fold to the same key")
    n_in_district = lut.groupby(lut["district"].map(fold)).size()
    n_in_district_c = con.groupby(con["district"].map(fold)).size()
    con["cod_key"] = [(fold(d), CONSTITUENCY_ALIASES.get((fold(d), fold(n)), fold(n)))
                      for d, n in zip(con.district, con.name)]
    for (d, cn), cod in CONSTITUENCY_ALIASES.items():
        if (d, cod) not in key_l:
            raise SystemExit(f"alias target {(d, cod)} is not a COD constituency")
        if d in SOLE_CONSTITUENCY and not (n_in_district[d] == 1 and n_in_district_c[d] == 1):
            raise SystemExit(f"{d}: named for its district in COD but not the district's "
                             "only constituency in both files")
    con["unit"] = con["cod_key"].map(key_l)
    miss = con.loc[con.unit.isna(), ["province", "district", "name"]].values.tolist()
    if miss:
        cod = sorted(set(key_l) - set(con["cod_key"]))
        raise SystemExit(f"census constituencies with no COD polygon: {miss}\n"
                         f"COD keys left over: {cod}")
    if con.unit.duplicated().any() or len(con) != 156:
        raise SystemExit("the join is not a bijection")
    cod_prov = dict(zip(lut["unit"], lut["province"]))
    moved = [(n, d, p, cod_prov[u]) for n, d, p, u in zip(con.name, con.district, con.province,
                                                         con.unit) if fold(cod_prov[u]) != fold(p)]
    print(f"\n  join: all 156 census constituencies land on one COD-AB polygon, on district and "
          f"constituency name ({len(CONSTITUENCY_ALIASES)} by an explicit alias)")
    for n, d, p, cp in moved:
        print(f"    {n} ({d}) is in {p} in the census and {cp} in COD-AB; the census's is used")
    if sorted({d for _n, d, _p, _cp in moved}) != ["Chirundu"]:
        raise SystemExit(f"the province disagreement is no longer just Chirundu: {moved}")

    # ---- the de jure check
    sdoc = fitz.open(SUMMARY_PDF)
    dj = parse_dejure(sdoc, {(fold(d), fold(n)) for d, n in zip(con.district, con.name)})
    dj_key = dict(zip(dj["key"], dj["total"]))
    con["dejure"] = [dj_key.get((fold(d), fold(n))) for d, n in zip(con.district, con.name)]
    if con["dejure"].isna().any():
        raise SystemExit("constituencies missing from Table 5.2: "
                         f"{con.loc[con.dejure.isna(), ['district', 'name']].values.tolist()}")
    con["ratio"] = con["all_Total"] / con["dejure"]
    s = con.sort_values("ratio")
    print(f"\n  de facto (religion tables) / de jure (Table 5.2): national "
          f"{DE_FACTO_TOTAL / DE_JURE_TOTAL:.4f}, constituencies {s.ratio.min():.3f} to "
          f"{s.ratio.max():.3f}, median {s.ratio.median():.3f}")
    for _, r in pd.concat([s.head(4), s.tail(4)]).iterrows():
        print(f"      {r['name']:<20} {r.district:<16} {int(r.all_Total):>9,} / "
              f"{int(r.dejure):>9,}  {r.ratio:.3f}")
    if s.ratio.min() < 0.70 or s.ratio.max() > 1.10:
        raise SystemExit("a constituency's de facto/de jure ratio is out of band -- a join "
                         "or a parse has gone wrong")

    # ---- the allocation, within province and residence
    rows = []
    for _, c in con.iterrows():
        p = c.province
        base = dict(geo_id=c.unit, geo_level="constituency", geo_name=c["name"],
                    basis="self_id", year=2022, source_id=SOURCE_ID, tier="derived")
        for d in DENOMS:
            v = 0.0
            for res, tab in (("Rural", b9), ("Urban", b10)):
                if tab.loc["Total", p] > 0:
                    v += c[f"chr_{res}"] * tab.loc[d, p] / tab.loc["Total", p]
            rows.append(dict(base, source_category=DENOM_PREFIXED.get(d, d), count=round(v, 3),
                             note=("Table B.5 constituency Christians, rural and urban, split by "
                                   "Tables B.9 and B.10's province denominational mix for the "
                                   f"same residence; parent_column={CHRISTIAN}")))
        oth_p = float(b1.loc[p, B1_OTHER].sum())
        for k_ in B1_OTHER:
            v = c["oth_Total"] * b1.loc[p, k_] / oth_p if oth_p > 0 else 0.0
            rows.append(dict(base, source_category=k_, count=round(v, 3),
                             note=("Table B.5 constituency non-Christians split by Table B.1's "
                                   f"province mix; parent_column={NONCHRISTIAN}")))
    out = pd.DataFrame(rows)

    # Re-aggregation reproduces the province tables (to rounding).
    out["province"] = out["geo_id"].map(dict(zip(con.unit, con.province)))
    agg = out.groupby(["province", "source_category"])["count"].sum()
    worst = 0.0
    for p in PROVINCES:
        for d in DENOMS:
            worst = max(worst, abs(agg[(p, DENOM_PREFIXED.get(d, d))] - b6.loc[d, p]))
        for k_ in B1_OTHER:
            worst = max(worst, abs(agg[(p, k_)] - b1.loc[p, k_]))
    if worst > 1.0:
        raise SystemExit(f"re-aggregation misses a province table cell by {worst:.2f} people")
    print(f"\n  allocation: {len(out):,} rows; every province x category re-aggregates to "
          f"Tables B.6 and B.1 within {worst:.3f} people")
    if abs(out["count"].sum() - DE_FACTO_TOTAL) > 5:
        raise SystemExit(f"allocated total {out['count'].sum():,.1f}")

    nat = out.groupby("source_category")["count"].sum().sort_values(ascending=False)
    print("\n  national, as drawn (share of the de facto 18,340,343):")
    for cat, v in nat.items():
        print(f"    {cat:<48}{v:>13,.0f}  {100.0 * v / DE_FACTO_TOTAL:6.2f}%")

    out = out.drop(columns="province")
    out = out[["geo_id", "geo_level", "geo_name", "source_category", "count", "basis", "year",
               "source_id", "tier", "note"]]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_csv(OUT + ".part", index=False, encoding="utf-8")
    os.replace(OUT + ".part", OUT)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
