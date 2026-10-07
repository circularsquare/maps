"""Malta: NSO, Census of Population and Housing 2021, main language spoken from early childhood.

    python sources/mt_census.py --fetch    Volume 3 from the Wayback Machine into data/raw/mt/
    python sources/mt_census.py            normalise from data/raw/mt/

-> data/normalized/mt.csv, three levels:
   `locality`      68 localities x 7 answers + total, MALTESE CITIZENS aged 5 and over (Vol. 3,
                   Table 3.6, pp.58-60). Drawn.
   `district_nm`   6 districts x 7 answers + total, NON-MALTESE residents aged 5 and over (Vol. 3,
                   Table 3.3, p.55). Drawn; the locality table covers Maltese citizens only.
   `locality_nmpop` 68 localities, non-Maltese residents of ALL AGES (Vol. 1, Table 2.1,
                   pp.116-118). Not a language count: countries/mt.py uses it only to place each
                   district's non-Maltese answers among the district's localities.

SOURCE. Volume 3 ("Health, education, labour and languages"),
https://nso.gov.mt/wp-content/uploads/volume3-Census-of-Population-2021.pdf. nso.gov.mt answers
403 to curl even with a browser User-Agent (religiondots met the same wall for Volume 1), so the
file comes from the Wayback Machine's 2026-08-11 capture (raw `id_` form, %%EOF present).
Volume 1 is religiondots' copy, read in place (../religiondots/data/raw/mt/), never written.

THE QUESTION. "Main language spoken from early childhood", asked of everyone aged 5 and over;
seven answers printed: Maltese, English, Italian, German, French, Arabic, Other. No not-stated
column: every row's answers sum to its total.

CHECKS (all asserted): every row's answers sum to its total; in Table 3.6 the localities sum to
their district and the districts to the islands and the nation; Table 3.6's district rows equal
Table 3.3's Maltese rows; Table 3.3's Maltese + non-Maltese = its district total; its national row
equals Table 3.1's; in Table 2.1 Maltese + non-Maltese = total and localities sum to districts;
each locality's Maltese citizens aged 5+ (Table 3.6) are fewer than its Maltese citizens of all
ages (Table 2.1), and the national ratio is printed.
"""

import csv
import os
import sys
import urllib.request

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "mt")
OUT = os.path.join(ROOT, "data", "normalized", "mt.csv")
VOL3 = os.path.join(RAW, "volume3.pdf")
VOL1 = os.path.join(ROOT, "..", "religiondots", "data", "raw", "mt",
                    "Census-of-Population-2021-volume1-final.pdf")

URL = "https://nso.gov.mt/wp-content/uploads/volume3-Census-of-Population-2021.pdf"
WAYBACK = "http://web.archive.org/web/20260811082947id_/" + URL
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
SIZE3 = 6_930_701
PAGES3 = 117

# 0-based page indices (printed page = index + 1)
P_T31 = 52
P_T33 = 54
P_T36 = (57, 58, 59)
P_T21 = (115, 116, 117)

CATS = ["Maltese", "English", "Italian", "German", "French", "Arabic", "Other"]

# The six districts and their localities as religiondots/sources/mt.py lists them (its unit ids),
# ASCII hyphens; the census prints U+2010 and NBSPs, folded by norm().
DISTRICTS = {
    "Southern Harbour": [
        "Bormla", "Floriana", "Ħal Luqa", "Ħal Tarxien", "Ħaż-Żabbar", "Il-Birgu", "Il-Fgura",
        "Il-Kalkara", "Il-Marsa", "Ix-Xgħajra", "L-Isla", "Raħal Ġdid", "Santa Luċija",
        "Valletta"],
    "Northern Harbour": [
        "Birkirkara", "Ħal Qormi", "Il-Gżira", "Il-Ħamrun", "Is-Swieqi", "L-Imsida", "Pembroke",
        "San Ġiljan", "San Ġwann", "Santa Venera", "Ta' Xbiex", "Tal-Pieta'", "Tas-Sliema"],
    "South Eastern": [
        "Birżebbuġa", "Ħal Għaxaq", "Ħal Kirkop", "Ħal Safi", "Il-Gudja", "Il-Qrendi",
        "Iż-Żejtun", "Iż-Żurrieq", "L-Imqabba", "Marsaskala", "Marsaxlokk"],
    "Western": [
        "Ħad-Dingli", "Ħal Balzan", "Ħal Lija", "Ħ'Attard", "Ħaż-Żebbuġ", "Ir-Rabat",
        "Is-Siġġiewi", "L-Iklin", "L-Imdina", "L-Imtarfa"],
    "Northern": [
        "Ħal Għargħur", "Il-Mellieħa", "Il-Mosta", "In-Naxxar", "L-Imġarr",
        "San Pawl Il-Baħar"],
    "Gozo and Comino": [
        "Għajnsielem and Comino", "Il-Fontana", "Il-Munxar", "Il-Qala", "In-Nadur",
        "Ir-Rabat, Għawdex", "Ix-Xagħra", "Ix-Xewkija", "Iż-Żebbuġ", "L-Għarb", "L-Għasri",
        "San Lawrenz", "Ta' Kerċem", "Ta' Sannat"],
}
LOCALITIES = [(d, loc) for d, locs in DISTRICTS.items() for loc in locs]
assert len(LOCALITIES) == 68
MALTA_DISTRICTS = ["Southern Harbour", "Northern Harbour", "South Eastern", "Western", "Northern"]


def norm(s):
    return " ".join(s.replace("‐", "-").replace("\xa0", " ").split()).casefold()


def num(s):
    s = s.strip().replace("\xa0", "")
    if s in ("‐", "-"):
        return 0
    t = s.replace(",", "")
    return int(t) if t.isdigit() else None


def rows_of(pdf, pages, k):
    """Ordered [(label, [k ints])] from the text of `pages`: a non-numeric line followed by
    exactly k numeric lines. Header lines (column names, captions) fail that test."""
    import fitz
    doc = fitz.open(pdf)
    lines = [ln for p in pages for ln in doc[p].get_text().splitlines() if ln.strip()]
    out, i = [], 0
    while i < len(lines):
        if num(lines[i]) is None:
            vals = [num(x) for x in lines[i + 1:i + 1 + k]]
            if len(vals) == k and all(v is not None for v in vals):
                out.append((lines[i].strip(), vals))
                i += 1 + k
                continue
        i += 1
    return out


def fetch():
    os.makedirs(RAW, exist_ok=True)
    req = urllib.request.Request(WAYBACK, headers=UA)
    data = urllib.request.urlopen(req, timeout=300).read()
    assert data.rstrip().endswith(b"%%EOF"), "truncated PDF"
    tmp = VOL3 + ".part"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, VOL3)
    print(f"fetched {len(data):,} bytes -> {VOL3}")


def expect_sequence(rows, names, what):
    got = [norm(r[0]) for r in rows]
    want = [norm(n) for n in names]
    if got != want:
        for j, (g, w) in enumerate(zip(got, want)):
            if g != w:
                raise SystemExit(f"{what}: row {j} is {rows[j][0]!r}, expected {names[j]!r}")
        raise SystemExit(f"{what}: {len(got)} rows, expected {len(want)}")


def main():
    import fitz
    if "--fetch" in sys.argv:
        fetch()
    doc = fitz.open(VOL3)
    assert doc.page_count == PAGES3 and os.path.getsize(VOL3) == SIZE3, "unexpected Volume 3 file"
    assert "TABLE 3.6." in doc[P_T36[0]].get_text() and "TABLE 3.3." in doc[P_T33].get_text()

    k = len(CATS) + 1
    for lab, v in rows_of(VOL3, (P_T31, P_T33, *P_T36), k):
        assert sum(v[:-1]) == v[-1], f"row {lab!r} does not sum: {v}"

    # ---- Table 3.1 (national, by citizenship) ----
    t31 = {norm(lab): v for lab, v in rows_of(VOL3, (P_T31,), k)[::3]}  # Total, Maltese, Non-M
    nat = t31[norm("Total")]

    # ---- Table 3.3: Total, Malta, Gozo and Comino, then per district: total, Maltese, Non-M
    t33 = rows_of(VOL3, (P_T33,), k)
    seq33 = ["Total", "Malta", "Gozo and Comino"] + [
        x for d in DISTRICTS for x in (d, "Maltese", "Non-Maltese")]
    expect_sequence(t33, seq33, "Table 3.3")
    assert t33[0][1] == nat, "Table 3.3 national row differs from Table 3.1"
    t33d = {}
    for j, d in enumerate(DISTRICTS):
        tot, mal, nm = (t33[3 + 3 * j + m][1] for m in range(3))
        assert [a + b for a, b in zip(mal, nm)] == tot, f"Table 3.3 {d}: Maltese + non-Maltese"
        t33d[d] = dict(total=tot, maltese=mal, nm=nm)
    assert [sum(t33d[d]["total"][c] for d in DISTRICTS) for c in range(k)] == nat

    # ---- Table 3.6: Maltese citizens by locality
    t36 = rows_of(VOL3, P_T36, k)
    seq36 = ["Total", "Malta", "Gozo and Comino"] + [
        x for d, locs in DISTRICTS.items() for x in [d] + locs]
    expect_sequence(t36, seq36, "Table 3.6")
    by36 = {}
    pos = 3
    for d, locs in DISTRICTS.items():
        drow = t36[pos][1]
        assert drow == t33d[d]["maltese"], f"Table 3.6 {d} differs from Table 3.3's Maltese row"
        sub = [t36[pos + 1 + j][1] for j in range(len(locs))]
        assert [sum(s[c] for s in sub) for c in range(k)] == drow, f"Table 3.6 {d}: localities"
        for loc, v in zip(locs, sub):
            by36[(d, loc)] = v
        pos += 1 + len(locs)
    assert t36[0][1] == t31[norm("Maltese")], "Table 3.6 total differs from Table 3.1 Maltese"

    # ---- Volume 1 Table 2.1: population by citizenship (Maltese M,F,T; non-Maltese M,F,T;
    # total M,F,T), all ages
    d1 = fitz.open(VOL1)
    assert "TABLE 2.1." in d1[P_T21[0]].get_text()
    t21 = rows_of(VOL1, P_T21, 9)
    seq21 = ["MALTA", "Malta", "Gozo and Comino"] + [
        x for d, locs in DISTRICTS.items() for x in [d] + locs]
    expect_sequence(t21, seq21, "Vol. 1 Table 2.1")
    for lab, v in t21:
        assert v[0] + v[1] == v[2] and v[3] + v[4] == v[5] and v[6] + v[7] == v[8], lab
        assert v[2] + v[5] == v[8], lab
    nmpop, malpop = {}, {}
    pos = 3
    for d, locs in DISTRICTS.items():
        drow = t21[pos][1]
        sub = [t21[pos + 1 + j][1] for j in range(len(locs))]
        assert [sum(s[c] for s in sub) for c in range(9)] == drow, f"Table 2.1 {d}: localities"
        for loc, v in zip(locs, sub):
            nmpop[(d, loc)] = v[5]
            malpop[(d, loc)] = v[2]
        pos += 1 + len(locs)
    for key, v in by36.items():
        assert v[-1] <= malpop[key], f"{key}: more Maltese aged 5+ than Maltese of all ages"
    nat21 = t21[0][1]
    print(f"Maltese citizens aged 5+ / all ages: {t31[norm('Maltese')][-1]:,} / "
          f"{nat21[2]:,} = {t31[norm('Maltese')][-1] / nat21[2]:.3f}; non-Maltese "
          f"{t31[norm('Non-Maltese')][-1]:,} / {nat21[5]:,} = "
          f"{t31[norm('Non-Maltese')][-1] / nat21[5]:.3f}; under-5s {nat21[8] - nat[-1]:,} "
          f"({(nat21[8] - nat[-1]) / nat21[8]:.2%}) of {nat21[8]:,}")

    rows = []
    for (d, loc), v in by36.items():
        for c, x in zip(CATS + ["Total"], v):
            rows.append(dict(geo_id=loc, geo_level="locality", geo_name=loc, district=d,
                             source_category=c, count=x, tier="measured",
                             source_id="nso_census2021_vol3_t3_6",
                             note="Maltese citizens aged 5 and over"))
    for d, r in t33d.items():
        for c, x in zip(CATS + ["Total"], r["nm"]):
            rows.append(dict(geo_id=d, geo_level="district_nm", geo_name=d, district=d,
                             source_category=c, count=x, tier="measured",
                             source_id="nso_census2021_vol3_t3_3",
                             note="non-Maltese residents aged 5 and over"))
    for (d, loc), x in nmpop.items():
        rows.append(dict(geo_id=loc, geo_level="locality_nmpop", geo_name=loc, district=d,
                         source_category="Non-Maltese population, all ages", count=x,
                         tier="measured", source_id="nso_census2021_vol1_t2_1",
                         note="placement weight only, not a language count"))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tmp = OUT + ".part"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) + ["year"])
        w.writeheader()
        for r in rows:
            w.writerow({**r, "year": 2021})
    os.replace(tmp, OUT)
    print(f"wrote {len(rows):,} rows -> {OUT}")
    print("national (Table 3.1):", dict(zip(CATS + ["Total"], nat)))
    print("non-Maltese:", dict(zip(CATS + ["Total"], t31[norm("Non-Maltese")])))


if __name__ == "__main__":
    main()
