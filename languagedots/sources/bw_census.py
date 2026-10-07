"""Botswana, Population and Housing Census 2011: language spoken at home by census district
-> data/normalized/bw.csv.

    python sources/bw_census.py [--fetch]

THE TABLE. *Population & Housing Census 2011: National Statistical Tables Report* (Statistics
Botswana, 2015), Category B, Table 2, "Distribution of Numbers of Persons Aged Two Years and Over
By District and Language Spoken; Botswana 2011" (PDF page 52, printed page 40). Persons aged two
and over, one answer each, the language spoken at home (the 2022 report calls the same question
"language spoken most often at home"). Sixteen languages and remainders plus Not Stated, for the
28 census districts and the six districts that group them (Southern, Kweneng, Central, North
West, Ghanzi, Kgalagadi). Table 4 on PDF page 54 is the same table printed a second time.

WHY 2011 AND NOT 2022. The 2022 census asked the same question and a new one, the language a
person spoke in early childhood, but its Analytical Report Volume 1 prints both by census district
only as Appendix 3, which lists the few districts where each language is largest (Setswana in
seven, Kalanga in six...), not a full district table. Appendix 4 has 2022's finer national list
(Sesarwa split into a dozen San languages, Setswana and Shekgalagadi into their dialects), national
only. The coverage sweep read "Appendices 3-4 tabulate both by census district"; they do not.

CHECKS, all asserted:
  1. every row's sixteen languages and Not Stated sum to its printed Total;
  2. the census districts of each grouping district sum to its row, column by column;
  3. the 28 census districts sum to the printed Total row, column by column;
  4. Table 2 and Table 4 agree cell by cell (a reprint, so this checks the parse, not the census);
  5. A SECOND PUBLICATION: the 2022 Analytical Report Volume 1, Appendix 3, reprints the 2011
     figure for 49 (language, census district) cells. Transcribed below from the rendered page;
     every one must equal Table 2's cell;
  6. the same report's Appendix 2 gives 2011's national total, 2,024,904, as 1,919,350 stated
     plus 105,554 "not applicable" (the under-twos with the 888 not stated folded in): Table 2's
     stated total must be 1,919,350 and its Not Stated 888.

RAW FILES. data/raw/bw/phc2011_national_statistical_tables.pdf and
data/raw/bw/phc2022_analytical_vol1.pdf, from statsbots.org.bw (URLs below).
"""
import re
import sys
import urllib.request
from pathlib import Path

import fitz  # PyMuPDF
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "bw"
OUT = ROOT / "data" / "normalized" / "bw.csv"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/124.0 Safari/537.36")
FILES = {
    "phc2011_national_statistical_tables.pdf":
        "https://www.statsbots.org.bw/sites/default/files/publications/national_statisticsreport.pdf",
    "phc2022_analytical_vol1.pdf":
        "https://www.statsbots.org.bw/sites/default/files/census_documents/"
        "Botswana%20Population%20%26%20Housing%20Census%202022%20Analytical%20Report%20Volume%201"
        "-Demographic%20and%20Social%20Characteristics%2CRegistration%2C%20Youth%20and%20Elderly"
        "%2CEducation.pdf",
}

# the table's columns, in print order (the header is hyphenated across lines, so it is pinned
# here and checked by sums rather than parsed)
COLS = ["Setswana", "English", "Sekalanga", "Shekgalagadi", "Sesubiya", "Sesarwa", "Seyeyi",
        "Sembukushu", "Afrikaans", "Ndebele", "Zezuru/Shona", "Seherero",
        "Other African languages", "Other European languages", "Other Asian languages",
        "Other (NEC)", "Not Stated"]

# printed row label -> (census district code = COD-AB 2011 ADM2_PCODE, or None for a grouping row)
# In print order; "Ghanzi" is printed twice, the district and then its census district.
ROWS = [
    ("Gaborone", "BW0101"), ("Francistown", "BW0201"), ("Lobatse", "BW0301"),
    ("Selebi Phikwe", "BW0401"), ("Orapa", "BW0501"), ("Jwaneng", "BW0601"),
    ("Sowa Town", "BW0701"),
    ("Southern", None), ("Ngwaketse", "BW0801"), ("Barolong", "BW0802"),
    ("Ngwaketse West", "BW0803"),
    ("South East", "BW0901"),
    ("Kweneng", None), ("Kweneng East", "BW1001"), ("Kweneng West", "BW1002"),
    ("Kgatleng", "BW1101"),
    ("Central", None), ("Central Serowe Palapye", "BW1201"), ("Central Mahalapye", "BW1202"),
    ("Central Bobonong", "BW1203"), ("Central Boteti", "BW1204"), ("Central Tutume", "BW1205"),
    ("North East", "BW1301"),
    ("North West", None), ("Ngamiland East", "BW1401"), ("Ngamiland West", "BW1402"),
    ("Chobe", "BW1501"), ("Okavango Delta", "BW1403"),
    ("Ghanzi", None), ("Ghanzi", "BW1601"), ("Central Kgalagadi Game Reserve (CKGR)", "BW1602"),
    ("Kgalagadi", None), ("Kgalagadi South", "BW1701"), ("Kgalagadi North", "BW1702"),
]
GROUPS = {"Southern": ["BW0801", "BW0802", "BW0803"],
          "Kweneng": ["BW1001", "BW1002"],
          "Central": ["BW1201", "BW1202", "BW1203", "BW1204", "BW1205"],
          "North West": ["BW1401", "BW1402", "BW1501", "BW1403"],
          "Ghanzi": ["BW1601", "BW1602"],
          "Kgalagadi": ["BW1701", "BW1702"]}

# 2022 Analytical Report Vol 1, Appendix 3 (PDF pages 79-80), the 2011 column, transcribed from
# the rendered page. Its "Ghanzi" is the census district (13,372 Sesarwa), not the district.
APPX3_2011 = {
    ("Setswana", "BW1001"): 218192, ("Setswana", "BW0101"): 166365,
    ("Setswana", "BW1201"): 160264, ("Setswana", "BW0801"): 118048,
    ("Setswana", "BW1202"): 107422, ("Setswana", "BW1101"): 81594,
    ("Setswana", "BW0901"): 71324,
    ("English", "BW0101"): 23934, ("English", "BW0201"): 4578, ("English", "BW1001"): 5001,
    ("English", "BW0901"): 3738,
    ("Sekalanga", "BW1205"): 57310, ("Sekalanga", "BW1301"): 26470,
    ("Sekalanga", "BW0201"): 20780, ("Sekalanga", "BW0101"): 11544,
    ("Sekalanga", "BW1204"): 6796, ("Sekalanga", "BW1001"): 5024,
    ("Shekgalagadi", "BW1002"): 20383, ("Shekgalagadi", "BW1702"): 13723,
    ("Shekgalagadi", "BW1601"): 12265, ("Shekgalagadi", "BW0803"): 8218,
    ("Shekgalagadi", "BW1701"): 1801,
    ("Sesubiya", "BW1501"): 5188,
    ("Sesarwa", "BW1601"): 13372, ("Sesarwa", "BW1204"): 4650, ("Sesarwa", "BW1402"): 3924,
    ("Sesarwa", "BW1205"): 2941, ("Sesarwa", "BW1702"): 1703,
    ("Seyeyi", "BW1402"): 1975,
    ("Sembukushu", "BW1402"): 25685, ("Sembukushu", "BW1401"): 3242,
    ("Afrikaans", "BW1701"): 4378,
    ("Ndebele", "BW1001"): 2489, ("Ndebele", "BW1301"): 3114, ("Ndebele", "BW0101"): 2894,
    ("Ndebele", "BW1205"): 2732,
    ("Zezuru/Shona", "BW1001"): 8764, ("Zezuru/Shona", "BW0101"): 8379,
    ("Zezuru/Shona", "BW1101"): 1975, ("Zezuru/Shona", "BW0201"): 3516,
    ("Zezuru/Shona", "BW1205"): 2575, ("Zezuru/Shona", "BW0901"): 2681,
    ("Zezuru/Shona", "BW1201"): 2133, ("Zezuru/Shona", "BW1301"): 1125,
    ("Seherero", "BW1401"): 7342, ("Seherero", "BW1601"): 3995, ("Seherero", "BW1402"): 1969,
    ("Seherero", "BW1204"): 1541, ("Seherero", "BW1202"): 1216,
}
NUM = re.compile(r"^(-|\d{1,3}(,\d{3})*)$")


def fetch():
    RAW.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        p = RAW / name
        if p.exists() and p.stat().st_size > 1_000_000:
            print(f"  have {p.name}")
            continue
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=120) as r:
            p.write_bytes(r.read())
        print(f"  fetched {p.name} ({p.stat().st_size:,} bytes)")


def parse(page_text, title):
    """Rows of (label, [17 values], total) in print order, and the Total row's values."""
    lines = [s.strip() for s in page_text.splitlines()]
    if not lines[0].startswith(title):
        raise SystemExit(f"page does not start with {title!r}: {lines[0]!r}")
    i = lines.index("Total") + 1          # the header's last cell
    rows, name, nums, total_row = [], [], [], None
    for s in lines[i:]:
        if NUM.match(s):
            nums.append(0 if s == "-" else int(s.replace(",", "")))
            if len(nums) == 18 and " ".join(name) != "Total":
                rows.append((" ".join(name), nums[:17], nums[17]))
                name, nums = [], []
            continue
        if " ".join(name) == "Total" and nums:   # the Total row ends at the page footer
            total_row = nums
            break
        if nums:
            raise SystemExit(f"{title}: {len(nums)} numbers before {s!r}")
        name.append(s)
    return rows, total_row


def main():
    if "--fetch" in sys.argv:
        fetch()
    doc = fitz.open(RAW / "phc2011_national_statistical_tables.pdf")
    t2, tot2 = parse(doc[51].get_text(), "Table 2. Distribution of Numbers of Persons Aged Two")
    t4, tot4 = parse(doc[53].get_text(), "Table 4: Distribution of Numbers of Persons Aged Two")

    # 4. the reprint agrees with the table, cell by cell
    assert [(r[1], r[2]) for r in t2] == [(r[1], r[2]) for r in t4], "Table 2 and Table 4 differ"
    assert tot2 == tot4[:17] and tot4[17] == 1_920_238, (tot2, tot4)

    if len(t2) != len(ROWS):
        raise SystemExit(f"{len(t2)} rows parsed, expected {len(ROWS)}: {[r[0] for r in t2]}")
    for (label, _, _), (want, _) in zip(t2, ROWS):
        norm = re.sub(r"\s+", " ", label.replace("-", " ")).strip()
        if norm != want:
            raise SystemExit(f"row {label!r} where {want!r} was expected")

    # 1. every row adds up
    for label, vals, total in t2:
        assert sum(vals) == total, (label, sum(vals), total)

    cd = {code: vals for (_, vals, _), (_, code) in zip(t2, ROWS) if code}
    grp = {want: vals for (_, vals, _), (want, code) in zip(t2, ROWS) if code is None}
    assert len(cd) == 28 and len(grp) == 6
    # 2. census districts sum to their district
    for g, codes in GROUPS.items():
        s = [sum(cd[c][k] for c in codes) for k in range(17)]
        assert s == grp[g], (g, s, grp[g])
    # 3. and the 28 to the Total row
    s = [sum(v[k] for v in cd.values()) for k in range(17)]
    assert s == tot2, (s, tot2)

    # 5. the 2022 report's reprint of 49 cells
    bad = [(k, v, cd[k[1]][COLS.index(k[0])]) for k, v in APPX3_2011.items()
           if cd[k[1]][COLS.index(k[0])] != v]
    if bad:
        raise SystemExit(f"2022 Appendix 3 disagrees with Table 2: {bad}")
    print(f"  2022 Analytical Report Appendix 3: all {len(APPX3_2011)} reprinted 2011 cells agree")

    # 6. the national figures the 2022 report gives for 2011
    stated = sum(tot2) - tot2[COLS.index("Not Stated")]
    assert stated == 1_919_350 and tot2[COLS.index("Not Stated")] == 888, stated
    assert stated + 105_554 == 2_024_904

    names = {code: re.sub(r"\s+", " ", lab).strip() for (lab, _, _), (_, code) in zip(t2, ROWS)
             if code}
    recs = []
    for code, vals in cd.items():
        for col, v in zip(COLS, vals):
            if v:
                recs.append(dict(geo_id=code, geo_level="census_district", geo_name=names[code],
                                 source_category=col, count=v))
    df = pd.DataFrame(recs)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT}: {len(df)} rows, 28 census districts, "
          f"{df['count'].sum():,} people aged 2+ ({stated:,} with a language stated)")


if __name__ == "__main__":
    main()
