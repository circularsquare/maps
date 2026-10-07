"""Burkina Faso RGPH 2006, main language spoken ("principale langue parlée") by province
-> data/normalized/bf.csv.

    python sources/bf_rgph.py [--fetch]

SOURCE. INSD, RGPH 2006, *Thème 2: État et structure de la population* (181 pages), the volume
religiondots draws Burkina Faso's religion from (religiondots/sources/bf.md). It is not on
insd.bf's current site; Wayback holds it on three retired paths (WAYBACK below). --fetch copies
religiondots' verified download (read-only) when it is there, and otherwise tries the captures;
the digest is pinned either way.

QUESTION (p37, p43). "La principale langue parlée par un individu, qui peut être une langue
locale ou étrangère", asked of every member of the household (1985 and 1996 asked one language
per household), one answer. The tables cover residents aged 3 and over. A main-language
question, so `how` says "main language spoken".

TABLES READ (1-based PDF pages):
  A5.3, pp159-162   province x language, COUNTS, in three blocks of 15 provinces (the first
                    with a "Burkina" column). 26 named national languages, "Langues
                    Africaines" and "Langues Non Africaines" (the foreign languages, grouped),
                    "Autres langues nationales", ND, Total. No thousands separator.
  A5.2, pp157-158   région x language, COUNTS, 13 régions + Burkina Faso. The same rows, but
                    the foreign languages are printed apart: Ashanti, Djerma, Haoussa, Ouolof,
                    "Autre langue africaine"; Français, Arabe, Anglais, Russe, "Autre langue non
                    africaine". Thousands separated by a space; the text layer gives each cell
                    its own line.
  A5.1, p156        national by milieu and sex; its Ensemble/Total column is a third copy of the
                    national counts.

THE BUILD. A5.3 as printed for the 26 named languages and "Autres langues nationales"
(`measured`). Each province's "Langues Africaines" and "Langues Non Africaines" are split among
the five languages of A5.2 in the proportions of the province's région (`derived`; Kadiogo is the
Centre région on its own, so its split is exact and `measured`). Because A5.3 summed by région
equals A5.2's group (check 4), this split reproduces A5.2's régional counts for every foreign
language up to rounding (check 7). ND (192,924, 1.5%) is carried per province and not drawn.

CHECKS (all must pass):
  1. the PDF is the pinned 181-page volume (size, sha256, %%EOF)
  2. A5.3: 31 rows x 45 provinces (+ Burkina); labels in the expected order; every province's
     rows sum to its Total; the Burkina column is the sum of the 45 provinces on every row
  3. A5.2: 41 rows x 14 columns; labels in order; every row's 13 régions sum to Burkina Faso;
     every column's rows sum to its Total
  4. A5.3 summed over each région's provinces = A5.2, on all 31 rows (the two foreign groups
     against the sum of their five A5.2 rows); this also proves the province -> région table
  5. A5.1's national total column = A5.2's Burkina Faso column, all 41 rows
  6. each province's A5.3 Total is 0.85-0.95 of its whole 2006 population (Tableau A 3.1 bis,
     from religiondots' bf_lookup.csv): the share aged 3 and over
  7. the split foreign languages, summed by région, are A5.2's counts within rounding
  8. the 45 provinces = religiondots' bf_hexes units, both ways
"""
import argparse
import hashlib
import re
import shutil
import sys
import unicodedata
import urllib.request
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from rdlink import RD_GEO  # noqa: E402

RAW = HERE / "data" / "raw" / "bf"
OUT = HERE / "data" / "normalized" / "bf.csv"
NAME = "Theme2-Etat_et_structure_de_la_population.pdf"
PDF = RAW / NAME
RD_PDF = HERE.parent / "religiondots" / "data" / "raw" / "bf" / NAME
WAYBACK = [
    "https://web.archive.org/web/20190203182046id_/http://www.insd.bf:80/documents/publications/"
    "insd/publications/resultats_enquetes/RGPH2006/" + NAME,
    "https://web.archive.org/web/20210824000000id_/http://www.insd.bf/contenu/"
    "enquetes_recensements/rgph-bf/themes_en_demographie/" + NAME,
    "https://web.archive.org/web/20101113000000id_/http://www.insd.bf/fr/IMG/pdf/" + NAME,
]
PDF_BYTES = 2_348_972
SHA256 = "722497af0da6a0e1a2a9826efbf42b3ffbb7b849a22edca0dcec9079a9d6419f"
PAGES = 181

# A5.3's three blocks: (pages 0-based, the page the block's last rows run onto, province order)
BLOCKS = [
    (158, 159, ["Kossi", "Mouhoun", "Sourou", "Balé", "Banwa", "Nayala", "Comoé", "Léraba",
                "Kadiogo", "Boulgou", "Kouritenga", "Koulpelogo", "Bam", "Namentenga",
                "Sanmatenga"]),
    (159, 160, ["Boulkiemdé", "Sanguié", "Sissili", "Ziro", "Bazèga", "Nahouri", "Zoundwéogo",
                "Gnagna", "Gourma", "Tapoa", "Komandjoari", "Kompienga", "Houet", "Kénédougou",
                "Tuy"]),
    (160, 161, ["Passoré", "Yatenga", "Loroum", "Zondoma", "Ganzourgou", "Oubritenga",
                "Kourwéogo", "Oudalan", "Séno", "Soum", "Yagha", "Bougouriba", "Poni", "Ioba",
                "Noumbiel"]),
]
NATIONAL = ["Bissa", "Bobo", "Bwamu", "Dafing", "Dagara", "Dioula", "Dogon", "Fulfuldé", "Gouin",
            "Goulmancema", "Kasséna", "Ko", "Koussassé", "Lyélé", "Lobiri", "Minianka", "Mooré",
            "Nuni", "San", "Sembla", "Sénoufo", "Siamou", "Sissaka", "Sonrhaï", "Tamachèque",
            "Gurunsi"]
LA, LNA, AUTRES = "Langues Africaines", "Langues Non Africaines", "Autres langues nationales"
A53_ROWS = ["Total", LA, LNA] + NATIONAL + [AUTRES, "ND"]

AFRICAN = ["Ashanti", "Djerma", "Haoussa", "Ouolof", "Autre langue africaine"]
NON_AFRICAN = ["Français", "Arabe", "Anglais", "Russe", "Autre langue non africaine"]
# A5.2's labels for the national rows where they differ from A5.3's
A52_ALIAS = {"Bwamu (ou Bwamou)": "Bwamu", "Dioula (ou Bambara)": "Dioula",
             "Dogon (ou Kaado)": "Dogon", "Fulfuldé (ou Peulh)": "Fulfuldé",
             "Nuni (Nounouma)": "Nuni", "San (ou Samogho ou Samo)": "San",
             "Tamachèque (ou Bella)": "Tamachèque", "Autres langues nationales": AUTRES}
A52_ROWS = AFRICAN + NON_AFRICAN + NATIONAL + [AUTRES, "ND", "Total"]
A52_COLS = ["Burkina Faso", "Boucle du Mouhoun", "Cascades", "Centre", "Centre-est",
            "Centre-nord", "Centre-ouest", "Centre-sud", "Est", "Hauts-bassins", "Nord",
            "Plateau central", "Sahel", "Sud-ouest"]
# A5.1 prints the national rows with its own spellings; only the order is used
A51_ROWS = AFRICAN + NON_AFRICAN + NATIONAL + [AUTRES, "ND", "Total"]

# The 13 régions of 2001-2025 (religiondots/sources/bf.py's REGION_OF, proven there against the
# religion tables and here again by check 4)
REGION_OF = {
    **dict.fromkeys(["Balé", "Banwa", "Kossi", "Mouhoun", "Nayala", "Sourou"], "Boucle du Mouhoun"),
    **dict.fromkeys(["Comoé", "Léraba"], "Cascades"),
    "Kadiogo": "Centre",
    **dict.fromkeys(["Boulgou", "Koulpelogo", "Kouritenga"], "Centre-est"),
    **dict.fromkeys(["Bam", "Namentenga", "Sanmatenga"], "Centre-nord"),
    **dict.fromkeys(["Boulkiemdé", "Sanguié", "Sissili", "Ziro"], "Centre-ouest"),
    **dict.fromkeys(["Bazèga", "Nahouri", "Zoundwéogo"], "Centre-sud"),
    **dict.fromkeys(["Gnagna", "Gourma", "Komandjoari", "Kompienga", "Tapoa"], "Est"),
    **dict.fromkeys(["Houet", "Kénédougou", "Tuy"], "Hauts-bassins"),
    **dict.fromkeys(["Loroum", "Passoré", "Yatenga", "Zondoma"], "Nord"),
    **dict.fromkeys(["Ganzourgou", "Kourwéogo", "Oubritenga"], "Plateau central"),
    **dict.fromkeys(["Oudalan", "Séno", "Soum", "Yagha"], "Sahel"),
    **dict.fromkeys(["Bougouriba", "Ioba", "Noumbiel", "Poni"], "Sud-ouest"),
}

INT = re.compile(r"^\d+$")
SPACED = re.compile(r"^\d{1,3}( \d{1,3})*$")   # a line of one or more space-grouped numbers


def norm(s):
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def say(ok, msg):
    print(("   ok  " if ok else "  FAIL ") + msg)
    if not ok:
        raise SystemExit(f"check failed: {msg}")


def fetch():
    RAW.mkdir(parents=True, exist_ok=True)
    if PDF.exists() and PDF.stat().st_size == PDF_BYTES:
        print("already have", PDF)
        return
    if RD_PDF.exists() and RD_PDF.stat().st_size == PDF_BYTES:
        shutil.copyfile(RD_PDF, PDF)
        print(f"copied religiondots' verified download -> {PDF}")
        return
    ua = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"}
    for url in WAYBACK:
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=ua), timeout=600) as r:
                body = r.read()
        except Exception as e:  # noqa: BLE001
            print(f"  {url}: {e}")
            continue
        if len(body) == PDF_BYTES and hashlib.sha256(body).hexdigest() == SHA256:
            PDF.write_bytes(body)
            print(f"wrote {PDF} ({len(body):,} bytes)")
            return
        print(f"  {url}: {len(body):,} bytes, not the pinned file")
    raise SystemExit("no capture gave the pinned PDF; fetch one of WAYBACK by hand into " + str(RAW))


# ---------------------------------------------------------------- A5.3 (no thousands separator)
def read_a53_part(text, start_label, ncol, nrows):
    """Rows from `start_label` on: label words, then `ncol` integers. Returns [(label, [ints])]."""
    toks = text.split()
    i = toks.index(start_label)
    rows, label, nums = [], [], []
    for t in toks[i:]:
        if len(rows) == nrows:
            break
        if INT.match(t):
            nums.append(int(t))
            if len(nums) == ncol:
                rows.append((" ".join(label), nums))
                label, nums = [], []
        else:
            if nums:
                raise SystemExit(f"A5.3: label word {t!r} inside a row of numbers")
            label.append(t)
    if len(rows) != nrows:
        raise SystemExit(f"A5.3: {len(rows)} rows from {start_label!r}, expected {nrows}")
    return rows


CAP2, CAP3 = "Tableau A 5.3 (suite)", "Table A5.3 (suite et fin)"
# each block runs onto the next page from this row; the page's captions bound the blocks
TAIL_FROM = [AUTRES, "Siamou", "Minianka"]


def read_a53(doc):
    """{row label: {province or 'Burkina': count}}"""
    table = {r: {} for r in A53_ROWS}
    for b, (p0, p1, provs) in enumerate(BLOCKS):
        cols = (["Burkina"] if b == 0 else []) + provs
        t0 = doc.load_page(p0).get_text()
        t1 = doc.load_page(p1).get_text()
        if b == 1:
            t0 = t0[t0.index(CAP2):]
        if b == 2:
            t0 = t0[t0.index(CAP3):]
        if b == 0:
            t1 = t1[:t1.index(CAP2)]
        if b == 1:
            t1 = t1[:t1.index(CAP3)]
        nhead = A53_ROWS.index(TAIL_FROM[b])
        rows = (read_a53_part(t0, "Total", len(cols), nhead)
                + read_a53_part(t1, TAIL_FROM[b].split()[0], len(cols), len(A53_ROWS) - nhead))
        for (label, nums), want in zip(rows, A53_ROWS):
            if norm(label) != norm(want) and not (norm(want) == "languesafricaines"
                                                  and norm(label) == "languesafricaine"):
                raise SystemExit(f"A5.3 block {b + 1}: row {label!r}, expected {want!r}")
            for c, v in zip(cols, nums):
                table[want][c] = v
    return table


# ---------------------------------------------------------------- A5.2, A5.1 (a cell per line)
def line_cells(line):
    """Every way to cut one text line of space-separated digit groups into numbers: a number is
    a 1-3 digit group followed by any run of exactly-3-digit groups."""
    toks = line.split()
    out = []

    def rec(i, acc):
        if i == len(toks):
            out.append(acc)
            return
        if not 1 <= len(toks[i]) <= 3:
            return
        j = i + 1
        while True:
            rec(j, acc + [int("".join(toks[i:j]))])
            if j < len(toks) and len(toks[j]) == 3:
                j += 1
            else:
                break
    rec(0, [])
    return out


def solve_row(lines, ncol, ok):
    """The one way to cut a row's number lines into `ncol` cells that satisfies `ok(cells)`.
    A line break always ends a cell; most lines are one cell, a few hold two or three."""
    ways = [line_cells(ln) for ln in lines]
    lo = [sum(min(len(c) for c in w) for w in ways[i:]) for i in range(len(ways))] + [0]
    hi = [sum(max(len(c) for c in w) for w in ways[i:]) for i in range(len(ways))] + [0]
    sols = []

    def rec(i, acc):
        if not lo[i] <= ncol - len(acc) <= hi[i]:
            return
        if i == len(ways):
            if ok(acc):
                sols.append(acc)
            return
        for c in ways[i]:
            rec(i + 1, acc + c)
    rec(0, [])
    if len(sols) != 1:
        raise SystemExit(f"{len(sols)} solutions for row {lines}")
    return sols[0]


def read_lines_table(doc, pages, first_labels, ncol, rows_expected, ok):
    """A run of non-number lines is a label; the number lines after it are its cells, cut by
    solve_row against the row's own totals (`ok`)."""
    out = []
    for pno, first in zip(pages, first_labels):
        lines = [ln.strip() for ln in doc.load_page(pno).get_text().splitlines()]
        lines = [ln for ln in lines if ln]
        k = next(i for i, ln in enumerate(lines) if ln == first)
        label, nums = [], []
        for ln in lines[k:] + ["<end>"]:
            if SPACED.match(ln):
                nums.append(ln)
                continue
            if nums:
                out.append((" ".join(label), solve_row(nums, ncol, ok)))
                label, nums = [], []
                if len(out) == len(rows_expected):
                    break
            label.append(ln)
    if len(out) != len(rows_expected):
        raise SystemExit(f"{len(out)} rows read, expected {len(rows_expected)}")
    return out


def read_a52(doc):
    rows = read_lines_table(doc, [156, 157], ["Ashanti", "Lyélé"], 14, A52_ROWS,
                            lambda c: sum(c[1:]) == c[0])
    table = {}
    for (label, nums), want in zip(rows, A52_ROWS):
        got = A52_ALIAS.get(label, label)
        if norm(got) != norm(want):
            raise SystemExit(f"A5.2: row {label!r}, expected {want!r}")
        table[want] = dict(zip(A52_COLS, nums))
    return table


def read_a51(doc):
    def ok(c):  # (M, F, T) for ensemble, urbain, rural
        return (all(c[k] + c[k + 1] == c[k + 2] for k in (0, 3, 6))
                and c[5] + c[8] == c[2])
    rows = read_lines_table(doc, [155], ["Ashanti"], 9, A51_ROWS, ok)
    return {want: nums[2] for (label, nums), want in zip(rows, A51_ROWS)}


# ---------------------------------------------------------------- split and checks
def split(total, weights):
    """Largest-remainder split of the integer `total` in proportion to `weights` (a dict)."""
    s = sum(weights.values())
    if total == 0 or s == 0:
        return {k: 0 for k in weights}
    raw = {k: total * w / s for k, w in weights.items()}
    out = {k: int(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: raw[k] - out[k], reverse=True)[: total - sum(out.values())]:
        out[k] += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    args = ap.parse_args()
    if args.fetch:
        fetch()
    import fitz

    body = PDF.read_bytes()
    doc = fitz.open(PDF)
    say(len(body) == PDF_BYTES and hashlib.sha256(body).hexdigest() == SHA256
        and body.rstrip().endswith(b"%%EOF") and doc.page_count == PAGES,
        f"1. file: {len(body):,} bytes, sha256 pinned, {doc.page_count} pages, %%EOF")

    a53 = read_a53(doc)
    provs = [p for _, _, ps in BLOCKS for p in ps]
    say(len(provs) == 45 and set(provs) == set(REGION_OF), "2. A5.3: 45 provinces, each in one région")
    bad = [p for p in provs if sum(a53[r][p] for r in A53_ROWS[1:]) != a53["Total"][p]]
    say(not bad, f"2. A5.3: every province's {len(A53_ROWS) - 1} rows sum to its Total {bad}")
    bad = [r for r in A53_ROWS if sum(a53[r][p] for p in provs) != a53[r]["Burkina"]]
    say(not bad, f"2. A5.3: Burkina column = sum of 45 provinces on all {len(A53_ROWS)} rows {bad}")

    a52 = read_a52(doc)
    bad = [r for r in A52_ROWS if sum(a52[r][c] for c in A52_COLS[1:]) != a52[r]["Burkina Faso"]]
    say(not bad, f"3. A5.2: {len(A52_ROWS)} rows, 13 régions sum to Burkina Faso on every row {bad}")
    bad = [c for c in A52_COLS if sum(a52[r][c] for r in A52_ROWS[:-1]) != a52["Total"][c]]
    say(not bad, f"3. A5.2: every column's rows sum to its Total {bad}")

    groups = {LA: AFRICAN, LNA: NON_AFRICAN}
    mism = []
    for r in A53_ROWS:
        for reg in A52_COLS[1:]:
            got = sum(a53[r][p] for p in provs if REGION_OF[p] == reg)
            want = sum(a52[x][reg] for x in groups[r]) if r in groups else a52[r][reg]
            if got != want:
                mism.append((r, reg, got, want))
    say(not mism, f"4. A5.3 summed by région = A5.2: {len(A53_ROWS)} rows x 13 régions {mism[:3]}")

    a51 = read_a51(doc)
    bad = [r for r in A51_ROWS if a51[r] != a52[r]["Burkina Faso"]]
    say(not bad, f"5. A5.1 national total = A5.2 Burkina Faso on all {len(A51_ROWS)} rows {bad}")

    lut = pd.read_csv(RD_GEO / "bf" / "bf_lookup.csv")
    pop06 = dict(zip(lut["unit"], lut["census_pop_2006"]))
    ratio = {p: a53["Total"][p] / pop06[p] for p in provs}
    lo, hi = min(ratio, key=ratio.get), max(ratio, key=ratio.get)
    say(all(0.85 <= v <= 0.95 for v in ratio.values()),
        f"6. aged 3+ / whole population per province: {ratio[lo]:.3f} ({lo}) to "
        f"{ratio[hi]:.3f} ({hi}); national {a53['Total']['Burkina'] / sum(pop06.values()):.3f}")

    # ---- emit
    rows = []
    for p in provs:
        reg = REGION_OF[p]
        for r in NATIONAL + [AUTRES, "ND"]:
            rows.append((p, r, a53[r][p], "measured"))
        tier = "measured" if reg == "Centre" else "derived"
        for g, members in groups.items():
            for c, n in split(a53[g][p], {c: a52[c][reg] for c in members}).items():
                rows.append((p, c, n, tier))
    df = pd.DataFrame(rows, columns=["geo_id", "source_category", "count", "tier"])
    df.insert(1, "geo_level", "province")
    df.insert(2, "geo_name", df["geo_id"])
    df.insert(3, "region", df["geo_id"].map(REGION_OF))

    worst = 0
    for c in AFRICAN + NON_AFRICAN:
        for reg in A52_COLS[1:]:
            got = df[(df["source_category"] == c) & (df["region"] == reg)]["count"].sum()
            n_prov = sum(1 for p in provs if REGION_OF[p] == reg)
            worst = max(worst, abs(got - a52[c][reg]) / n_prov)
    say(worst <= 1, f"7. split foreign languages by région = A5.2 within one person per province "
                    f"(worst {worst:.2f})")

    import pyogrio
    units = set(pyogrio.read_dataframe(RD_GEO / "bf" / "bf_hexes.gpkg", columns=["unit"],
                                       read_geometry=False)["unit"].astype(str))
    say(units == set(provs), f"8. join: the 45 provinces = religiondots' bf_hexes units "
                             f"(missing {sorted(set(provs) - units)}, extra {sorted(units - set(provs))})")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    drawn = df[df["source_category"] != "ND"]["count"].sum()
    print(f"wrote {OUT}: {len(df):,} rows, {drawn:,} people with an answer, "
          f"ND {a53['ND']['Burkina']:,}, universe {a53['Total']['Burkina']:,}")
    nat = df.groupby("source_category")["count"].sum().sort_values(ascending=False)
    print((nat / a53["Total"]["Burkina"] * 100).round(2).to_string())


if __name__ == "__main__":
    main()
