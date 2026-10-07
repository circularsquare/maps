"""Belize: SIB, 2022 Population and Housing Census, General Characteristics Table 7,
"Population Four Years and Older by Languages Spoken and District: 2022".

Reads (or fetches) data/raw/bz/ and writes data/normalized/bz.csv. Record: sources/bz.md.

The question (Individual Questionnaire 1.6, asked of everyone aged 4 and over): "Which
language(s) do you/does N speak well enough to conduct a conversation? [MULTIPLE RESPONSES
ALLOWED]", with eleven languages, Other (specify), Cannot speak and DK/NS. So Table 7 counts
MENTIONS, about 1.96 per person, and is drawn under spec §3.6 (countries/bz.py).

Table 7 prints no universe row. The population aged 4 and over per district is taken as all
ages (Table 9's Total row) less the under-4s of the census's age tables: Table 6's `Less than 1`
plus its `1-4` times the national share of ages 1-3 in 1-4 (Table 4, single ages, national
only). Nationally that is 368,924.4, and Table 3.9 of SIB's Key Findings Report prints 368,924.

Table 3.9 also prints a 4+ population per district and every language's share of it: its 84
shares agree with Table 7's counts over its own 4+ row to 0.051 pp, which is the second table
this file reconciles against. Its 4+ row is NOT the one drawn: four districts agree with the age
tables within 130 people, but Stann Creek's is 1,591 lower and Toledo's 1,571 higher, and
Stann Creek's would leave 5,191 under-4s in a district whose Table 6 has 4,597 under-5s, which
cannot be. Both rows are written to the normalised file.

THE FIGURES ARE FRACTIONAL AND THAT IS THE SOURCE. SIB publishes undercount-adjusted counts
(religiondots/sources/bz.py found the same in the same workbook): identities are asserted to a
1e-6 relative tolerance, not to zero.

SUPPRESSED CELLS. SIB prints "<10" for small cells. In the Total columns only one is
suppressed, Hindi in Toledo; it is recovered exactly as the national figure less the other five
districts (the districts sum to the national figure on every unsuppressed row, asserted), and
asserted to be under 10.

DISTRICT ORDER. SIB prints the districts north to south; COD-AB's pcodes are alphabetical
(BZ01 is Belize District). The pcode is looked up by name, as religiondots/sources/bz.py does,
whose placement layer this country reuses.

Usage:
    python sources/bz_census.py --fetch    one 69 KB xlsx and a 2.7 MB pdf, seconds
    python sources/bz_census.py            normalise from data/raw/bz/
"""
import csv
import os
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "bz")
OUT = os.path.join(ROOT, "data", "normalized", "bz.csv")

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/125.0 Safari/537.36")
# Plain links on https://sib.org.bz/census/2022-census/, no wall, no key. sib.org.bz answers
# 406 to a bare "Mozilla/5.0" and 200 to a full browser token (religiondots/sources/bz.py).
FILES = {
    "Census2022_GeneralCharacteristics.xlsx":
        "https://sib.org.bz/wp-content/uploads/Census2022_GeneralCharacteristics.xlsx",
    "CensusKeyFindingsReport_2022.pdf":
        "https://sib.org.bz/wp-content/uploads/CensusKeyFindingsReport_2022.pdf",
}
XLSX, PDF = list(FILES)

SHEET = "Languages_Spoken"
TITLE = "Table 7: Population Four Years and Older by Languages Spoken and District: 2022"
REL_SHEET = "Religion_by_District"      # only for each district's all-ages population

DISTRICTS = [                           # SIB's order, north to south, with COD-AB's pcode
    ("Corozal", "BZ03"),
    ("Orange Walk", "BZ04"),
    ("Belize", "BZ01"),
    ("Cayo", "BZ02"),
    ("Stann Creek", "BZ05"),
    ("Toledo", "BZ06"),
]
NAMES = [d for d, _ in DISTRICTS]
GROUPS = ["Total"] + NAMES
SEXES = ["Total", "Male", "Female"]

# Table 7's rows in SIB's order (it sorts by national count, with Cannot Speak in its place).
LABELS = [
    "Speaks English", "Speaks Spanish", "Speaks Creole", "Speaks Maya Ketchi",
    "Speaks Maya Mopan", "Speaks German", "Speaks Garifuna", "Speaks Other",
    "Speaks Maya Yucatec", "Speaks Chinese", "Cannot Speak", "Speaks Hindi",
]
# Table 3.9's row label for each, in the Key Findings Report.
KF_LABEL = {l: l.replace("Speaks ", "") for l in LABELS}

POP4 = "Population aged 4 and over"     # source_category of the universe rows
POP4_KF = "Population aged 4 and over, Key Findings"
POPALL = "Population, all ages"         # from Table 9's Total row, for the gap
TOL = 1e-6


def fetch():
    import requests
    os.makedirs(RAW, exist_ok=True)
    for name, url in FILES.items():
        dest = os.path.join(RAW, name)
        if os.path.exists(dest) and os.path.getsize(dest) > 20_000:
            print("already have", dest)
            continue
        print("GET", url)
        r = requests.get(url, timeout=300, headers={"User-Agent": UA})
        r.raise_for_status()
        magic = r.content[:4]
        if magic not in (b"PK\x03\x04", b"%PDF"):
            raise SystemExit(f"{url} is not an xlsx or pdf -- starts {magic!r}")
        with open(dest + ".part", "wb") as fh:
            fh.write(r.content)
        os.replace(dest + ".part", dest)
        print(f"  {os.path.getsize(dest):,} bytes")


def _norm(s):
    return " ".join(str(s or "").split())


def _close(a, b):
    return abs(a - b) <= TOL * max(1.0, abs(b))


def _grid(ws):
    return [list(r) for r in ws.iter_rows(values_only=True)]


def read_table7(wb):
    """{sex: {group: {label: float or None}}}, None where SIB printed '<10'."""
    ws = wb[SHEET]
    g = _grid(ws)
    title = next((_norm(c) for row in g[:3] for c in row if c and _norm(c).startswith("Table 7")), "")
    if title != TITLE:
        raise SystemExit(f"{SHEET} title is {title!r}, expected {TITLE!r}")
    hg = next(i for i, row in enumerate(g) if any(_norm(c) == "Language" for c in row))
    c0 = next(j for j, c in enumerate(g[hg]) if _norm(c) == "Language") + 1
    for k, want in enumerate(GROUPS):
        got = _norm(g[hg][c0 + 3 * k])
        if got != want:
            raise SystemExit(f"{SHEET}: column group {k} is {got!r}, expected {want!r}")
        for s, sw in enumerate(SEXES):
            got = _norm(g[hg + 1][c0 + 3 * k + s])
            if got != sw:
                raise SystemExit(f"{SHEET}: {want}/{s} is {got!r}, expected {sw!r}")
    out = {s: {grp: {} for grp in GROUPS} for s in SEXES}
    r = hg + 2
    for want in LABELS:
        while r < len(g) and not _norm(g[r][c0 - 1]):
            r += 1
        got = _norm(g[r][c0 - 1]) if r < len(g) else "<eof>"
        if got != want:
            raise SystemExit(f"{SHEET}: expected row {want!r}, read {got!r} -- SIB has changed "
                             "Table 7's language list; revisit taxonomy/bz2022.py")
        for k, grp in enumerate(GROUPS):
            for s, sex in enumerate(SEXES):
                v = g[r][c0 + 3 * k + s]
                if isinstance(v, (int, float)):
                    out[sex][grp][want] = float(v)
                elif _norm(v) == "<10":
                    out[sex][grp][want] = None
                else:
                    raise SystemExit(f"{SHEET}: {grp}/{sex}/{want} is {v!r}")
        r += 1
    rest = [_norm(row[c0 - 1]) for row in g[r:] if _norm(row[c0 - 1])]
    if any(not x.startswith("Source") for x in rest):
        raise SystemExit(f"{SHEET}: unexpected rows after the table: {rest}")
    return out


def read_popall(wb):
    """Each district's all-ages population: Table 9's Total row, Total sex column."""
    g = _grid(wb[REL_SHEET])
    hg = next(i for i, row in enumerate(g) if any(_norm(c) == "Religion" for c in row))
    c0 = next(j for j, c in enumerate(g[hg]) if _norm(c) == "Religion") + 1
    r = next(i for i in range(hg + 2, len(g)) if _norm(g[i][c0 - 1]) == "Total")
    out = {}
    for k, grp in enumerate(GROUPS):
        if _norm(g[hg][c0 + 3 * k]) != grp:
            raise SystemExit(f"{REL_SHEET}: group {k} is not {grp!r}")
        out[grp] = float(g[r][c0 + 3 * k])
    return out


def read_ages(wb):
    """Each group's population under 4, from the census's own age tables.

    Table 6 gives `Less than 1` and `1-4` per district; Table 4 gives single ages, national
    only. Under 4 = `Less than 1` + `1-4` x (national ages 1-3 / national ages 1-4), so the
    six districts sum exactly to the national single-age figure. Returns {group: under4}.
    """
    g = _grid(wb["Single_Age_Sex"])
    hr = next(i for i, row in enumerate(g) if _norm(row[1]) == "Age")
    single = {}
    for row in g[hr + 1:]:
        a = _norm(row[1])
        if a.isdigit() and int(a) <= 4:
            single[int(a)] = float(row[2])
    if sorted(single) != [0, 1, 2, 3, 4]:
        raise SystemExit(f"Single_Age_Sex: ages 0-4 not all found: {sorted(single)}")
    f13 = (single[1] + single[2] + single[3]) / (single[1] + single[2] + single[3] + single[4])

    g = _grid(wb["Age_Groups"])
    cur, d = None, {}
    for row in g:
        if _norm(row[1]):
            cur = _norm(row[1])
        if cur and _norm(row[2]) in ("Less than 1", "1-4"):
            d.setdefault(cur, {})[_norm(row[2])] = float(row[4])     # column: Census 2022
    hdr = next(row for row in g if _norm(row[2]) == "Age Group")
    if _norm(hdr[4]) != "Census 2022":
        raise SystemExit(f"Age_Groups: column 4 is {hdr[4]!r}, expected 'Census 2022'")
    if not _close(d["National"]["Less than 1"], single[0]):
        raise SystemExit("Age_Groups' national under-1 disagrees with Single_Age_Sex")
    out = {}
    for grp in GROUPS:
        key = "National" if grp == "Total" else grp
        out[grp] = d[key]["Less than 1"] + f13 * d[key]["1-4"]
    nat = single[0] + single[1] + single[2] + single[3]
    if not _close(out["Total"], nat) or not _close(sum(out[n] for n in NAMES), nat):
        raise SystemExit(f"under-4 estimate does not reconcile to the national {nat:,.3f}")
    return out


def read_kf():
    """Table 3.9 of the Key Findings Report: the 4+ population and every label's share, per
    district, as {label_or_POP4: [Belize, Corozal, ... Toledo]}."""
    import fitz
    doc = fitz.open(os.path.join(RAW, PDF))
    page = next(p for p in doc if "Table 3.9" in p.get_text()
                and "Population Aged 4+ years" in p.get_text())
    lines = [l.strip() for l in page.get_text().splitlines() if l.strip()]
    num = re.compile(r"^-?[\d,]+(\.\d+)?$")

    def after(label, start):
        i = lines.index(label, start)
        vals = lines[i + 1:i + 8]
        if len(vals) != 7 or not all(num.match(v) for v in vals):
            raise SystemExit(f"Key Findings Table 3.9: {label!r} is followed by {vals}")
        return i, [float(v.replace(",", "")) for v in vals]

    i, pop = after("Population Aged 4+ years", 0)
    out = {POP4: pop}
    for l in LABELS:
        _, out[l] = after(KF_LABEL[l], i)
    return out


def check_and_rows(t7, popall, under4, kf):
    ok = True
    tot = t7["Total"]

    # 1. Recover the one suppressed Total cell (Hindi, Toledo) by difference, after checking
    #    the districts sum to the national figure on every unsuppressed row.
    bad, recovered = [], []
    for l in LABELS:
        miss = [n for n in NAMES if tot[n][l] is None]
        if tot["Total"][l] is None:
            raise SystemExit(f"national {l} is suppressed")
        if not miss:
            s = sum(tot[n][l] for n in NAMES)
            if not _close(s, tot["Total"][l]):
                bad.append((l, s, tot["Total"][l]))
        elif len(miss) == 1:
            v = tot["Total"][l] - sum(tot[n][l] for n in NAMES if n != miss[0])
            if not (0 <= v < 10):
                raise SystemExit(f"{l}/{miss[0]} recovers as {v}, not in [0, 10)")
            tot[miss[0]][l] = v
            recovered.append((l, miss[0], v))
        else:
            raise SystemExit(f"{l}: {len(miss)} suppressed district cells, cannot recover")
    ok &= not bad
    print(f"  {'OK ' if not bad else 'BAD'} the 6 districts sum to the national column "
          f"({len(LABELS) - len(recovered) - len(bad)}/{len(LABELS) - len(recovered)} rows)")
    for l, n, v in recovered:
        print(f"      recovered {l} in {n} by difference: {v:.3f} (printed '<10')")

    # 2. Male + Female == Total wherever all three are printed: the check on the READ.
    n_cells, bad = 0, []
    for grp in GROUPS:
        for l in LABELS:
            m, f, t = t7["Male"][grp][l], t7["Female"][grp][l], t7["Total"][grp][l]
            if m is None or f is None:
                continue
            n_cells += 1
            if not _close(m + f, t):
                bad.append((grp, l, m + f, t))
    ok &= not bad
    print(f"  {'OK ' if not bad else 'BAD'} Male + Female == Total on {n_cells} printed cells "
          f"({len(bad)} failures) {bad[:3]}")

    # 3. The 4+ universe. Two figures: the Key Findings Report's printed row (Table 3.9), and
    #    all ages less the under-4s of the census's own age tables (read_ages). Nationally they
    #    agree to the person. Per district they do not in the south: the report's figure for
    #    Stann Creek implies more under-4s than Table 6 has under-5s, which cannot be, and
    #    Toledo's is ~1,600 too high. The age-table figure is the one drawn (countries/bz.py);
    #    sources/bz.md says why. Both are written out.
    pop4 = dict(zip(GROUPS, kf[POP4]))
    s = sum(pop4[n] for n in NAMES)
    good = abs(s - pop4["Total"]) <= 3
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} Key Findings 4+ population: districts sum {s:,.0f}, "
          f"Belize {pop4['Total']:,.0f}")
    good = abs(pop4["Total"] - (popall["Total"] - under4["Total"])) < 1
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} national 4+: Key Findings {pop4['Total']:,.0f}, all ages "
          f"less Table 4's ages 0-3 {popall['Total'] - under4['Total']:,.1f}")
    print("      district  all ages    under 4 (Tables 4+6)   4+ age tables   4+ Key Findings   diff")
    for n in NAMES:
        a4 = popall[n] - under4[n]
        m = sum(tot[n][l] for l in LABELS if l != "Cannot Speak")
        print(f"      {n:<12} {popall[n]:>9,.0f}  {under4[n]:>9,.0f}  {a4:>18,.0f}  "
              f"{pop4[n]:>16,.0f}  {pop4[n] - a4:>+7,.0f}   mentions/person "
              f"{m / a4:.3f} (age tables) {m / pop4[n]:.3f} (KF)")

    # 4. Table 3.9's 84 printed shares against Table 7 / the 4+ population. Printed to one
    #    decimal (Belize's own column too), so they must agree within 0.05 pp plus slack for
    #    the rounded denominator.
    worst, bad = 0.0, []
    for l in LABELS:
        for k, grp in enumerate(GROUPS):
            mine = 100.0 * tot[grp][l] / pop4[grp]
            d = abs(mine - kf[l][k])
            worst = max(worst, d)
            if d > 0.051:
                bad.append((l, grp, round(mine, 3), kf[l][k]))
    ok &= not bad
    print(f"  {'OK ' if not bad else 'BAD'} Key Findings Table 3.9: {len(LABELS) * len(GROUPS)} "
          f"shares agree with Table 7 / 4+ population, worst {worst:.3f} pp")
    for b in bad[:6]:
        print("      ", b)

    if not ok:
        raise SystemExit("reconciliation FAILED")

    nat = tot["Total"]
    m = sum(nat[l] for l in LABELS if l != "Cannot Speak")
    print(f"\n  mentions per person aged 4+ who named a language: "
          f"{m / (pop4['Total'] - nat['Cannot Speak']):.3f}")
    print("  national, mentions and share of the 4+ population:")
    for l in LABELS:
        print(f"    {nat[l]:>12,.1f}  {100 * nat[l] / pop4['Total']:5.1f}%  {l}")
    print(f"  under 4, not asked: {popall['Total'] - pop4['Total']:,.1f}")

    rows = []
    for name, pcode in DISTRICTS:
        def row(cat, v, note):
            rows.append({"geo_id": pcode, "geo_level": "district", "geo_name": name,
                         "source_category": cat, "count": v, "note": note})
        row(POPALL, popall[name], "Table 9 Total row; universe, not a language")
        row(POP4, popall[name] - under4[name],
            "all ages less under-4s (Table 6 <1 + 1-4 x national 1-3 share, Table 4); drawn")
        row(POP4_KF, pop4[name], "Key Findings Report Table 3.9; not drawn, see sources/bz.md")
        for l in LABELS:
            note = "mentions; several answers allowed"
            if any(l == a and name == b for a, b, _ in recovered):
                note += "; printed <10, recovered as national less the other districts"
            row(l, tot[name][l], note)
    return rows


def main():
    if "--fetch" in sys.argv:
        fetch()
    import openpyxl
    p = os.path.join(RAW, XLSX)
    if not os.path.exists(p):
        raise SystemExit(f"missing {p} -- run with --fetch")
    wb = openpyxl.load_workbook(p, data_only=True)
    t7 = read_table7(wb)
    popall = read_popall(wb)
    under4 = read_ages(wb)
    kf = read_kf()
    rows = check_and_rows(t7, popall, under4, kf)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT + ".part", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    os.replace(OUT + ".part", OUT)
    print("\nwrote", OUT, f"({len(rows)} rows)")


if __name__ == "__main__":
    main()
