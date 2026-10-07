"""Cambodia: NIS, General Population Census of Cambodia 2019, mother tongue.

Reads (or fetches) data/raw/kh/ and writes data/normalized/kh.csv.

THE ONLY GEOGRAPHY NIS PUBLISHES FOR MOTHER TONGUE IS THE COUNTRY. Three tables are read:

  * Final report (National Report on Final Census Results), Table 2.7.1, p.25 (PDF p.53):
    "Distribution of population by mother tongue and sex". Seven rows on 15,552,211 people:
    Khmer, Vietnam, Chinese, Lao, Thai, Other, Minority Languages.
  * Thematic report "Ethnic Minorities in Cambodia" (Sept 2022), Table 2.3, p.7 (PDF p.31):
    "Total size of various ethnic minority populations by sex, 2008 and 2019". 23 named minority
    languages and an Other, 455,610 people. The census has no ethnicity question; the
    questionnaire's column 9 is "Native language" with codes 10-28 for the minority languages
    and 29 "Native language other" (report p.118 of the questionnaire, PDF p.142). The report's
    "ethnic minority" is that answer, so this table is mother tongue.
  * The same report's Table 2.2, p.6 (PDF p.30): the minority-language population (all
    languages together) by province, 2008 and 2019. Not drawn as counts: countries/kh.py uses
    it to place the national counts inside Cambodia (AGENT_BRIEF §4.4).

THE TWO NATIONAL TABLES SPLIT THE SAME 466,251 PEOPLE DIFFERENTLY. Table 2.7.1's Minority
Languages (448,282) and Other (17,969) sum to 466,251; Table 2.3's total (455,610) is 10,641
fewer. Table 2.3 has its own Other (7,413), the questionnaire's code 29, so its 455,610 is the
minority languages plus their unnamed remainder, and Table 2.7.1's Other holds that remainder
(less 85) together with the foreign languages the form has codes for (French, English, Korean,
Japanese). Drawn: Table 2.3's 24 rows, and `Other foreign language` = 466,251 - 455,610 = 10,641
as a derived row. Every person in Table 2.7.1's total is drawn once.

TABLE 2.7.1's SEX COLUMNS DO NOT ADD UP. Male + Female exceeds Total on every row (Khmer by
2,812, Minority Languages by 85) and the Male and Female columns each sum to more than their
column total, while the Total column sums to 15,552,211 exactly. The Total column is used; the
mismatch is printed, not asserted.

Checks: Table 2.7.1's Total column sums to the census population; Table 2.3's and Table 2.2's
Male + Female = Total on every row in both years; Table 2.3's rows sum to its total; Table
2.2's provinces, its four regions and urban + rural each sum to its total; Table 2.2's total
equals Table 2.3's in both years (the only check that crosses tables); every province's
minority population is under its census population (religiondots' Table 2.1.1 read,
read-only); and the provinces are joined to religiondots' KH-01..KH-25 by name AND by print
position, which must agree.

Usage:
    python sources/kh_gpcc.py --fetch    two PDFs, 26.6 MB and 10.3 MB
    python sources/kh_gpcc.py            normalise from data/raw/kh/
"""

import csv
import os
import re
import sys
import urllib.request

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "kh")
OUT = os.path.join(ROOT, "data", "normalized", "kh.csv")
RD_NORM = os.path.join(os.path.dirname(ROOT), "religiondots", "data", "normalized", "kh.csv")

YEAR = 2019
COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count", "tier", "year",
           "source_id", "note"]

FILES = {
    "final": ("kh_gpcc2019_final_en.pdf",
              "https://nis.gov.kh/wp-content/uploads/2025/09/"
              "Final-General-Population-Census-2019-English.pdf"),
    "ethnic": ("kh_gpcc2019_ethnic_minorities.pdf",
               "https://www.nis.gov.kh/nis/Census2019/Ethnic%20Minorities.pdf"),
}
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/126.0 Safari/537.36")

POPULATION = 15_552_211
T271_ROWS = ["Khmer", "Vietnam", "Chinese", "Lao", "Thai", "Other", "Minority Languages"]
T23_ROWS = ["Charai", "Cham", "Kavet", "Khloeng", "Kuoy", "Kroeng", "Lorn", "Punorng", "Prov",
            "Tumpuon", "Steang", "Ro-ong", "Kroul", "Rodae", "Thmoon", "Mael", "Khonh", "Por",
            "Suoy", "Sa-ouch", "Ka-chrook", "Morn", "Kanh-Chok", "Other"]
T22_REGIONS = ["Central plain", "Tonle Sap", "Coastal & sea", "Plateau & mountains"]
# Table 2.2's province rows in print order; the order is NIS's province code order, which is
# religiondots' KH-01..KH-25 (checked by name below as well)
T22_PROVINCES = ["Banteay Meanchey", "Battambang", "Kampong Cham", "Kampong Chhnang",
                 "Kampong Speu", "Kampong Thom", "Kampot", "Kandal", "Koh Kong", "Kratie",
                 "Mondulkiri", "Phnom Penh", "Preah Vihear", "Prey Veng", "Pursat",
                 "Ratanakkiri", "Siem Reap", "Preah Sihanouk", "Stoeng Treng", "Svay Rieng",
                 "Takeo", "Oddar Meanchey", "Kep", "Pailin", "Tbong Khmum"]

NUM = re.compile(r"^\d{1,3}(,\d{3})*$")


def fetch():
    os.makedirs(RAW, exist_ok=True)
    for name, url in FILES.values():
        path = os.path.join(RAW, name)
        if os.path.exists(path):
            print(f"  have {name}")
            continue
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        data = urllib.request.urlopen(req, timeout=600).read()
        with open(path + ".part", "wb") as f:
            f.write(data)
        os.replace(path + ".part", path)
        print(f"  fetched {name}: {len(data):,} bytes")


def page_lines(name, pno):
    import fitz
    fitz.TOOLS.mupdf_display_errors(False)
    doc = fitz.open(os.path.join(RAW, FILES[name][0]))
    return [ln.strip() for ln in doc[pno - 1].get_text().splitlines()]


def rows_after(lines, labels, ncols, start_pat):
    """Line read: a row label on its own line, then its figures one per line. Walks from the
    first line matching start_pat and takes each wanted label in order, asserting the figures."""
    i = next(k for k, ln in enumerate(lines) if re.search(start_pat, ln))
    out = {}
    for lab in labels:
        while True:
            while lines[i] != lab:
                i += 1
                if i >= len(lines):
                    raise SystemExit(f"row {lab!r} not found after {start_pat!r}")
            vals = []
            j = i + 1
            while len(vals) < ncols and j < len(lines):
                tok = lines[j]
                if tok:
                    vals.append(tok)
                j += 1
            # a header cell with the same word ("Total" is both a column and a row) is
            # followed by more header text, not figures: skip it
            if vals and NUM.match(vals[0]):
                break
            i += 1
        out[lab] = vals
        i = j
    return out


def ints(vals):
    for v in vals:
        if not NUM.match(v):
            raise SystemExit(f"not a count: {v!r} in {vals}")
    return [int(v.replace(",", "")) for v in vals]


def read_t271():
    lines = page_lines("final", 53)
    raw = rows_after(lines, ["Total"] + T271_ROWS, 6, r"^Table 2\.7\.1")
    t = {k: ints(v[:3]) for k, v in raw.items()}
    assert t["Total"][0] == POPULATION, t["Total"]
    assert sum(t[k][0] for k in T271_ROWS) == POPULATION, "2.7.1 rows do not sum to the total"
    assert t["Total"][1] + t["Total"][2] == POPULATION
    bad = {k: v[1] + v[2] - v[0] for k, v in t.items() if v[1] + v[2] != v[0]}
    print(f"  2.7.1: Total column sums to {POPULATION:,}; Male + Female - Total, by row "
          f"(NIS's own mismatch, not used): {bad}")
    return {k: v[0] for k, v in t.items()}


def read_t23():
    lines = page_lines("ethnic", 31)
    raw = rows_after(lines, ["Total"] + T23_ROWS, 6, r"^Table 2\.3")
    t = {k: ints(v) for k, v in raw.items()}
    for k, v in t.items():
        assert v[1] + v[2] == v[0] and v[4] + v[5] == v[3], (k, v)
    for c in (0, 3):
        assert sum(t[k][c] for k in T23_ROWS) == t["Total"][c], ("2.3 sum", c)
    assert t["Total"][3] == 455_610
    print(f"  2.3: 24 rows, M+F=T on all, rows sum to 2008 {t['Total'][0]:,} and "
          f"2019 {t['Total'][3]:,}")
    return t


def read_t22():
    lines = page_lines("ethnic", 30)
    labels = ["Total", "Urban", "Rural"] + T22_REGIONS + T22_PROVINCES
    raw = rows_after(lines, labels, 6, r"^Table 2\.2")
    t = {k: ints(v) for k, v in raw.items()}
    for k, v in t.items():
        assert v[1] + v[2] == v[0] and v[4] + v[5] == v[3], (k, v)
    for c in (0, 3):
        tot = t["Total"][c]
        assert t["Urban"][c] + t["Rural"][c] == tot, ("urban+rural", c)
        assert sum(t[p][c] for p in T22_PROVINCES) == tot, ("provinces", c)
    # the four regions miss the 2019 total by 5 (455,605): NIS's own, the provinces are exact
    reg = [sum(t[r][c] for r in T22_REGIONS) - t["Total"][c] for c in (0, 3)]
    assert reg[0] == 0 and abs(reg[1]) <= 5, reg
    print(f"  2.2: 25 provinces and urban + rural each sum to 2008 {t['Total'][0]:,} and 2019 "
          f"{t['Total'][3]:,}; the 4 regions miss by {reg[0]} and {reg[1]}; M+F=T on all "
          f"{len(t)} rows")
    return t


def fold(s):
    s = s.lower().replace(" ", "").replace("-", "")
    for a, b in (("kk", "k"), ("oe", "u"), ("dd", "d"), ("otdar", "odar"), ("tbaung", "tbong"),
                 ("siemreap", "siemreap")):
        s = s.replace(a, b)
    return s


def rd_provinces():
    """religiondots' province ids, names and Table 2.1.1 populations (read-only)."""
    out = []
    with open(RD_NORM, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["geo_level"] == "province" and r["source_category"] == "Total":
                out.append((r["geo_id"], r["geo_name"], int(r["count"])))
    out.sort()
    assert len(out) == 25 and sum(p for _, _, p in out) == POPULATION, "religiondots kh.csv"
    return out


def main():
    if "--fetch" in sys.argv:
        fetch()
    t271 = read_t271()
    t23 = read_t23()
    t22 = read_t22()
    assert t22["Total"][0] == t23["Total"][0] and t22["Total"][3] == t23["Total"][3], \
        "Tables 2.2 and 2.3 disagree on the minority total"
    print("  2.2 total == 2.3 total in 2008 and 2019 (the cross-table check)")

    named = sum(t23[k][3] for k in T23_ROWS if k != "Other")
    pool = t271["Minority Languages"] + t271["Other"]
    foreign_other = pool - t23["Total"][3]
    assert 0 < foreign_other < t271["Other"]
    print(f"  2.3 named minority languages {named:,} against 2.7.1's Minority Languages "
          f"{t271['Minority Languages']:,} (differ by {t271['Minority Languages'] - named:,}); "
          f"2.7.1 Minority + Other {pool:,} = 2.3 total {t23['Total'][3]:,} + other foreign "
          f"{foreign_other:,}")

    rd = rd_provinces()
    for k, ((gid, gname, pop), lab) in enumerate(zip(rd, T22_PROVINCES), start=1):
        assert gid == f"KH-{k:02d}", gid
        assert fold(gname) == fold(lab), (gid, gname, lab)
        assert t22[lab][3] < pop, (lab, t22[lab][3], pop)
    print("  provinces joined to religiondots' KH-01..KH-25 by print position, and the names "
          "agree under the romanisation fold on all 25; each under its census population")

    rows = []
    src271 = "kh_gpcc_2019_t271"
    src23 = "kh_gpcc_2019_em_t23"
    src22 = "kh_gpcc_2019_em_t22"
    rows.append(["KH", "country", "Cambodia", "Total", POPULATION, "measured", YEAR, src271,
                 "universe total (excludes Cambodians working abroad), not a language"])
    for k in T271_ROWS:
        note = "level=country; Table 2.7.1"
        if k in ("Other", "Minority Languages"):
            note += "; not drawn: split by Table 2.3 and the derived Other foreign language row"
        rows.append(["KH", "country", "Cambodia", k, t271[k], "measured", YEAR, src271, note])
    for k in T23_ROWS:
        cat = "Other minority language" if k == "Other" else k
        rows.append(["KH", "country", "Cambodia", cat, t23[k][3], "measured", YEAR, src23,
                     "level=country; Ethnic Minorities in Cambodia, Table 2.3, 2019 column"])
    rows.append(["KH", "country", "Cambodia", "Other foreign language", foreign_other,
                 "derived", YEAR, src271,
                 "level=country; Table 2.7.1 Minority Languages + Other less Table 2.3's total"])
    for (gid, gname, pop), lab in zip(rd, T22_PROVINCES):
        rows.append([gid, "province", gname, "Minority languages, all", t22[lab][3], "measured",
                     YEAR, src22, "level=province; placement only, not drawn as a count"])
        rows.append([gid, "province", gname, "Total", pop, "measured", YEAR,
                     "kh_gpcc_2019_t211",
                     "level=province; Table 2.1.1 as read by religiondots; placement only"])

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT + ".part", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        w.writerows(rows)
    os.replace(OUT + ".part", OUT)
    drawn = (sum(t271[k] for k in T271_ROWS[:5]) + t23["Total"][3] + foreign_other)
    assert drawn == POPULATION
    print(f"wrote {OUT}: {len(rows)} rows; drawn categories sum to {drawn:,}")


if __name__ == "__main__":
    main()
