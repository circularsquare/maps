"""Indonesia, Sensus Penduduk 2010: language used daily at home, by province
-> data/normalized/id.csv.

    python sources/id_sp2010.py [--fetch]

THE PUBLICATION. *Kewarganegaraan, Suku Bangsa, Agama, dan Bahasa Sehari-hari Penduduk
Indonesia: Hasil Sensus Penduduk 2010* (BPS, 2011; ISBN 978-979-064-417-5, catalogue 2102032),
64 PDF pages. The publication page on www.bps.go.id is behind Cloudflare, but the PDF itself is
served by web-api.bps.go.id/download.php with no key or session (the token below is the one the
publication page links; it was still valid on 2026-10-04).

THE QUESTION (Catatan Teknis, PDF p.14): "bahasa sehari-hari", the language usually used to talk
with the other members of the household at home, persons aged 5 and over, one answer each,
recorded as one of three boxes (Indonesian / a regional language, named / a foreign language,
named) and the named language then coded to BPS's 1,211-language list.

THE TABLES (Lampiran 2), all "only from document SP2010-C1", 214,056,929 persons aged 5+:
  L4.1 (PDF p.56)  national, 35 language groups (Jawa, Indonesia, Sunda, ... "Bahasa-bahasa asal
                   NTT", ..., foreign, sign, not answered). National only.
  L4.2 (PDF p.57)  33 provinces x the three boxes + not answered + total.
  L4.5 (PDF p.60-61) 33 provinces x the eight largest groups (Jawa, Indonesia, Sunda, Melayu,
                   Madura, Minangkabau, Banjar, Bugis) + not answered + Lainnya ("other").
Nothing finer is published: sensus.bps.go.id serves SP2010 tables to kecamatan, but for language
only "ability to speak Indonesian".

WHAT IS WRITTEN. Per province: the eight named groups, then L4.5's Lainnya split in two with
L4.2's boxes, which is arithmetic on the same census and not a model:
    "Bahasa daerah lainnya" = L4.2 Bahasa Daerah - the seven named regional languages of L4.5
    "Bahasa asing"          = L4.2 Bahasa Asing
and "Tidak terjawab". By construction Lainnya = the two (check 4 asserts it per province).
Then (ask 009, Anita 2026-10-05) "Bahasa daerah lainnya" is shared out across the ethnic groups
of Table L2.6 (PDF p.45-50, province x 31 groups) whose language is not already counted, in
proportion to their numbers in the province, one row "Bahasa daerah lainnya: <group>" each with
tier `modelled`; every other row is `measured`. See main() and sources/id.md §5.

CHECKS, all asserted:
  1. L4.2: every province's four columns sum to its Total, and the 33 provinces sum to the
     printed INDONESIA row, column by column;
  2. L4.5: every province's ten columns sum to the L4.2 Total of that province, and the 33 sum
     to the printed Total Nasional row, column by column;
  3. the two tables agree on the cells they share: Indonesian and not answered, per province;
  4. Lainnya = (Daerah - 7 named) + Asing, and both parts are non-negative, per province;
  5. L4.5's national totals equal L4.1's rows for the same groups, and L4.1 sums to its Grand
     Total 214,056,929;
  6. L4.3 (PDF p.58, row percentages of L4.2) recomputed from L4.2 to 0.01 point;
  7. L2.6's 33 provinces sum to P1.2's national figure for every one of the 31 groups;
  8. the share-out sums exactly to each province's remainder.
     L4.1's 35 printed rows sum to 312,062 LESS than its own Grand Total. The gap is exactly L4.2's
     Asing box (756,035) minus L4.1's foreign row (443,973), and L4.1's regional rows plus its
     sign-language row (40,373) are exactly L4.2's Daerah box. So L4.1 drops one foreign row
     from print, and BPS filed sign language in the regional-language box: the drawn "other
     regional language" remainder holds Indonesia's signers (asserted, not assumed).
"""
import re
import sys
import urllib.request
from pathlib import Path

import fitz  # PyMuPDF
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "id"
PDF = RAW / "sp2010_kewarganegaraan_suku_bangsa_agama_bahasa.pdf"
OUT = ROOT / "data" / "normalized" / "id.csv"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/124.0 Safari/537.36")
PAGE_URL = ("https://www.bps.go.id/en/publication/2012/05/23/55eca38b7fe0830834605b35/"
            "kewarganegaraan-suku-bangsa-agama-dan-bahasa-sehari-hari-penduduk-indonesia.html")
URL = ("https://web-api.bps.go.id/download.php?f=CHtwnQtYohb6+T7izXbdMTdFTU5FaDNleDN1UWJPNTZrTHF3"
       "cVBTbkppK0FUT2VzYjVpRnZkbjFqdFkrNTBHRGFXeHdLNWZLVTU5SnNTMjFQNUhoVy8rWWUvazF4bmpzZ2h4aUd4"
       "cDZtQzZPcUR2MWo3NXRTQmswd3lUenVVWGFJRzFTSkNOQWdTNnM2UDE3MkRCdFBqcWh4WFNXV0htVzNWMkZISFh5"
       "SVZvc0JuZ1M2cVpSRFd0RnFCN0hCeFU2QTFXbm1lbmxWZ3pYL1J4RlNkRWZKVUtLZW9nQkc5MHJndk9PdVVoYXVw"
       "U0o2YS9INlhUaHRpQlVUUXR1SXZ2Z1FjbjR2RWRhZ2EyZ3JYOXRQTlhIZ1JTK2F6MWIvL2o5NnFydXRLRFR1b3BR"
       "TzZOcXFmZkx6OTU1cFJ2VmZqS3BHUlBCUkVURmRIcDA0QUJ0")

# printed province name -> BPS 2010 province code (the codes religiondots' units begin with)
PROVINCES = {
    "Aceh": "11", "Sumatera Utara": "12", "Sumatera Barat": "13", "Riau": "14", "Jambi": "15",
    "Sumatera Selatan": "16", "Bengkulu": "17", "Lampung": "18", "Bangka Belitung": "19",
    "Kepulauan Riau": "21", "DKI Jakarta": "31", "Jawa Barat": "32", "Jawa Tengah": "33",
    "DI Yogyakarta": "34", "Jawa Timur": "35", "Banten": "36", "Bali": "51",
    "Nusa Tenggara Barat": "52", "Nusa Tenggara Timur": "53", "Kalimantan Barat": "61",
    "Kalimantan Tengah": "62", "Kalimantan Selatan": "63", "Kalimantan Timur": "64",
    "Sulawesi Utara": "71", "Sulawesi Tengah": "72", "Sulawesi Selatan": "73",
    "Sulawesi Tenggara": "74", "Gorontalo": "75", "Sulawesi Barat": "76", "Maluku": "81",
    "Maluku Utara": "82", "Papua Barat": "91", "Papua": "94",
}
# L2.6 prints three provinces in short form
L26_ALIAS = {"D I Yogyakarta": "DI Yogyakarta", "NTB": "Nusa Tenggara Barat",
             "NTT": "Nusa Tenggara Timur"}
# the 31 ethnic groups of Tables L2.6 and P1.2, in their printed column order
ETHNIC = ["Suku asal Aceh", "Batak", "Nias", "Melayu", "Minangkabau", "Suku asal Jambi",
          "Suku asal Sumatera Selatan", "Suku asal Lampung", "Suku asal Sumatera Lainnya",
          "Betawi", "Suku asal Banten", "Sunda", "Jawa", "Cirebon", "Madura", "Bali", "Sasak",
          "Suku Nusa Tenggara Barat lainnya", "Suku asal Nusa Tenggara Timur", "Dayak", "Banjar",
          "Suku Asal Kalimantan lainnya", "Makassar", "Bugis", "Minahasa", "Gorontalo",
          "Suku Asal Sulawesi lainnya", "Suku Asal Maluku", "Suku Asal Papua", "Cina",
          "Asing/Luar Negeri"]
# Groups whose language is already counted elsewhere in the table, so they take no share of the
# "other regional languages" remainder: the seven named regional languages of L4.5; Banten
# (Bantenese and Baduy speak Sundanese or Banten Javanese, and L4.1, whose rows exhaust the
# regional box, has no Banten language, so they were coded Sunda or Jawa); and Cina and Asing,
# whose languages are in the foreign box, drawn on its own.
MEASURED_ETHNIC = {"Melayu", "Minangkabau", "Suku asal Banten", "Sunda", "Jawa", "Madura",
                   "Banjar", "Bugis", "Cina", "Asing/Luar Negeri"}
REMAINDER = "Bahasa daerah lainnya"
L42_COLS = ["Indonesia", "Daerah", "Asing", "Tidak terjawab", "Total"]
L45_COLS = ["Jawa", "Indonesia", "Sunda", "Melayu", "Madura",
            "Minangkabau", "Banjar", "Bugis", "Tidak terjawab", "Lainnya"]
NAMED_REGIONAL = ["Jawa", "Sunda", "Melayu", "Madura", "Minangkabau", "Banjar", "Bugis"]

NUM = re.compile(r"^(\d{1,3}(?: \d{3})*|-|–|—|�)$")


def fetch():
    RAW.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(URL, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=180) as r:
        data = r.read()
    if not data.startswith(b"%PDF") or b"%%EOF" not in data[-2048:]:
        raise SystemExit("download is not a complete PDF (the token may have expired; take a "
                         f"fresh download link from {PAGE_URL})")
    PDF.write_bytes(data)
    print(f"wrote {PDF} ({len(data):,} bytes)")


def _num(tok):
    return 0 if tok in ("-", "–", "—", "�") else int(tok.replace(" ", ""))


def _lines(page):
    return [l.strip() for l in page.get_text().split("\n") if l.strip()]


def province_table(page, ncols, total_label):
    """A province-by-column table printed as: name line, then ncols number lines."""
    lines = _lines(page)
    start = lines.index(f"({ncols + 1})") + 1   # after the (1)..(n) column-number row
    rows, i = {}, start
    while i < len(lines):
        name = lines[i]
        if name.startswith("Keterangan"):
            break
        vals = lines[i + 1:i + 1 + ncols]
        if len(vals) != ncols or not all(NUM.match(v) for v in vals):
            raise SystemExit(f"p.{page.number + 1}: row {name!r} does not have {ncols} numbers: "
                             f"{vals}")
        rows[name] = [_num(v) for v in vals]
        i += 1 + ncols
    total = rows.pop(total_label)
    if set(rows) != set(PROVINCES):
        raise SystemExit(f"p.{page.number + 1}: provinces {sorted(set(rows) ^ set(PROVINCES))}")
    return rows, total


def national_l41(page):
    lines = _lines(page)
    start = lines.index("(4)") + 1
    out, i = {}, start
    while not lines[i].startswith("Grand Total"):
        rank, name, n, pct = lines[i:i + 4]
        assert rank.isdigit(), (rank, name)
        out[name] = _num(n)
        i += 4
    return out, _num(lines[i + 1])


def ethnic_table(pages):
    """L2.6 over its six pages: province x 31 ethnic groups, no total row. Each page is a header,
    the column-number row "(a)".."(b)", then a name line and one number line per column."""
    out = {p: [] for p in PROVINCES}
    for page in pages:
        lines = _lines(page)
        nums = [i for i, l in enumerate(lines) if re.fullmatch(r"\(\d+\)", l)]
        ncols = len(nums)
        i = nums[-1] + 1
        seen = set()
        while not lines[i].startswith("Keterangan"):
            name = L26_ALIAS.get(lines[i], lines[i])
            vals = lines[i + 1:i + 1 + ncols]
            if name not in PROVINCES or not all(NUM.match(v) for v in vals):
                raise SystemExit(f"L2.6 p.{page.number + 1}: row {lines[i]!r}: {vals}")
            out[name] += [_num(v) for v in vals]
            seen.add(name)
            i += 1 + ncols
        if seen != set(PROVINCES):
            raise SystemExit(f"L2.6 p.{page.number + 1}: provinces {sorted(set(PROVINCES) ^ seen)}")
    return pd.DataFrame.from_dict(out, orient="index", columns=ETHNIC)


def national_p12(page):
    """P1.2: national count per ethnic group (name, count, percent, rank), then Total."""
    lines = _lines(page)
    i = lines.index("(4)") + 1
    out = {}
    while lines[i] != "Total":
        out[lines[i]] = _num(lines[i + 1])
        i += 4
    return out, _num(lines[i + 1])


def largest_remainder(total, weights):
    """Integers proportional to `weights` that sum exactly to `total`."""
    w = weights / weights.sum()
    raw = w * total
    base = raw.astype(int)
    short = int(total - base.sum())
    order = (raw - base).sort_values(ascending=False, kind="mergesort").index
    base[order[:short]] += 1
    assert base.sum() == total and (base >= 0).all()
    return base


def percent_table(page):
    lines = _lines(page)
    start = lines.index("(6)") + 1
    rows, i = {}, start
    while i < len(lines) and not lines[i].startswith("Keterangan"):
        rows[lines[i]] = [0.0 if NUM.match(v) and _num(v) == 0 else float(v.replace(",", "."))
                          for v in lines[i + 1:i + 6]]
        i += 6
    return rows


def main():
    if "--fetch" in sys.argv or not PDF.exists():
        fetch()
    doc = fitz.open(PDF)
    page_of = {}
    for p in doc:
        for t in ("Tabel L4.1", "Tabel L4.2", "Tabel L4.3", "Tabel L4.5", "Lanjutan Tabel L4.5",
                  "Tabel P1.2"):
            if t in _lines(p)[:4]:
                page_of[t] = p
    l41, l41_total = national_l41(page_of["Tabel L4.1"])
    l42, l42_tot = province_table(page_of["Tabel L4.2"], 5, "INDONESIA")
    a, a_tot = province_table(page_of["Tabel L4.5"], 5, "Total Nasional")
    b, b_tot = province_table(page_of["Lanjutan Tabel L4.5"], 5, "Total Nasional")
    l45 = {p: a[p] + b[p] for p in PROVINCES}
    l45_tot = a_tot + b_tot
    L42 = pd.DataFrame.from_dict(l42, orient="index", columns=L42_COLS)
    L45 = pd.DataFrame.from_dict(l45, orient="index", columns=L45_COLS)

    # 1
    assert (L42[L42_COLS[:4]].sum(axis=1) == L42["Total"]).all(), "L4.2 rows"
    assert L42.sum().tolist() == l42_tot, ("L4.2 national", L42.sum().tolist(), l42_tot)
    # 2
    bad = L45.sum(axis=1) != L42["Total"]
    assert not bad.any(), ("L4.5 rows vs L4.2 totals", L45.sum(axis=1)[bad], L42["Total"][bad])
    assert L45.sum().tolist() == l45_tot, ("L4.5 national", L45.sum().tolist(), l45_tot)
    # 3
    assert (L45["Indonesia"] == L42["Indonesia"]).all(), "Indonesian differs between L4.2/L4.5"
    assert (L45["Tidak terjawab"] == L42["Tidak terjawab"]).all(), "not answered differs"
    # 4
    other_regional = L42["Daerah"] - L45[NAMED_REGIONAL].sum(axis=1)
    assert (other_regional >= 0).all(), other_regional[other_regional < 0]
    assert (other_regional + L42["Asing"] == L45["Lainnya"]).all(), "Lainnya split"
    # 5
    assert l41_total == L42["Total"].sum() == 214_056_929
    for col in L45_COLS[:-1]:
        key = {"Tidak terjawab": "Tidak Terjawab"}.get(col, col)
        assert l41[key] == L45[col].sum(), (col, l41[key], L45[col].sum())
    # L4.1's 35 rows fall short of its own Grand Total by exactly the part of L4.2's Asing box
    # that its foreign row does not hold, and its regional rows plus sign language are exactly
    # L4.2's Daerah box: the unprinted row is foreign, and sign language was filed as regional.
    unprinted = l41_total - sum(l41.values())
    foreign_row = l41["Bahasa-bahasa asal bahasa asing"]
    assert unprinted == L42["Asing"].sum() - foreign_row == 312_062, unprinted
    regional_rows = sum(v for k, v in l41.items() if k not in (
        "Indonesia", "Tidak Terjawab", "Bahasa-bahasa asal bahasa asing", "Bahasa Isyarat"))
    assert regional_rows + l41["Bahasa Isyarat"] == L42["Daerah"].sum(), "Daerah vs L4.1"
    # 6
    pct = percent_table(page_of["Tabel L4.3"])
    for name, vals in pct.items():
        if name not in PROVINCES:
            continue
        row = L42.loc[name]
        calc = [round(100 * row[c] / row["Total"], 2) for c in L42_COLS[:4]]
        assert all(abs(x - y) <= 0.011 for x, y in zip(calc, vals[:4])), (name, calc, vals)

    # 7. L2.6 (province x 31 ethnic groups, citizens of every age, forms C1 and C2-Apartemen):
    # each group's 33 provinces sum to its national figure in P1.2, in printed order, and the 31
    # sum to P1.2's Total. That also proves the columns were read in the right order.
    ep = [p for p in doc if any(t in _lines(p)[:4] for t in ("Tabel L2.6", "Lanjutan Tabel L2.6"))]
    assert len(ep) == 6, len(ep)
    E = ethnic_table(ep)
    p12, p12_total = national_p12(page_of["Tabel P1.2"])
    def _key(s):
        return re.sub(r"\b(suku|asal)\b", "", s.lower()).split()
    assert [_key(k) for k in p12] == [_key(k) for k in ETHNIC], list(p12)
    assert E.sum().tolist() == list(p12.values()), (E.sum().tolist(), list(p12.values()))
    assert int(E.values.sum()) == p12_total == 236_728_379

    print(f"checks ok: 33 provinces, {L42['Total'].sum():,} persons aged 5+; L4.1 leaves "
          f"{unprinted:,} foreign-box people out of its printed rows; L2.6 = P1.2 for all 31 "
          f"ethnic groups ({p12_total:,} citizens)")

    # THE SHARE-OUT (ask 009, Anita 2026-10-05; widened 2026-10-05, session edd42a8c-id2): each
    # province's "other regional languages" remainder is shared among named languages by
    # sources/id_shareout.py, raked to L4.1's national language groups. Every row `modelled`.
    import id_shareout
    shared, info = id_shareout.share_out(E, other_regional, L42["Total"], l41, PROVINCES)
    print(f"share-out: {len(shared)} province x language rows, L4.1 groups met to "
          f"{info['col_err']:.2g} people")

    rows = []
    for name, code in PROVINCES.items():
        vals = {c: (int(L45.loc[name, c]), "measured") for c in L45_COLS[:-1]}
        for r in shared[shared["province"] == name].itertuples():
            vals[f"{REMAINDER}: {r.label}"] = (int(r.count), "modelled")
        vals["Bahasa asing"] = (int(L42.loc[name, "Asing"]), "measured")
        assert sum(n for n, _ in vals.values()) == L42.loc[name, "Total"]
        for cat, (n, tier) in vals.items():
            rows.append(dict(geo_id=code, geo_level="province", geo_name=name,
                             source_category=cat, count=n, tier=tier))
    out = pd.DataFrame(rows)
    assert out.loc[out.tier == "modelled", "count"].sum() == other_regional.sum()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"wrote {OUT}: {len(out)} rows, {out['count'].sum():,} persons, "
          f"{out.loc[out.tier == 'modelled', 'count'].sum():,} of them modelled")
    nat = shared.groupby("label")["count"].sum().sort_values(ascending=False)
    print("modelled languages, national:")
    print(nat.to_string())


if __name__ == "__main__":
    main()
