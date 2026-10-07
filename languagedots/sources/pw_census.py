"""Palau, 2015 Census of Population, Housing and Agriculture, Table 16 "Language Spoken and
Language Used Most by Usual Residence, Palau: 2015" -> data/normalized/pw.csv.

    python sources/pw_census.py [--fetch]

SOURCE. Office of Planning and Statistics (Ministry of Finance), report "2015 Census of
Population, Housing and Agriculture Tables", PDF page 25 (index 24):
https://www.palaugov.pw/wp-content/uploads/2017/02/2015-Census-of-Population-Housing-Agriculture-.pdf
The tables are images, so the figures below are a transcription read off a 300 dpi render;
the checks below are what make that trustworthy (every row and every column adds up).

THE QUESTIONS (questionnaire, PDF pages 232-233, and the definitions in the 2020 report):
C22 "Does ... speak Palauan at home?" (Palauan and another / another / Palauan only),
C23 the other language spoken at home (one write-in), C24 whether it is spoken more often than
Palauan, equally, or less often. All persons, all ages. So each person is either "Palauan only"
or filed under ONE named other language; countries/pw.py then applies C24 (see sources/pw.md).

WHY 2015 AND NOT 2020: the 2020 report's matching table (Table 16, 2020) only names the other
language for people who speak Palauan at home (12,399 rows: 12,297 English); the 5,215 who do
not speak Palauan, mostly Filipino workers, are a single unnamed row. 2015 names it for everyone.
"""
import os
import sys
import urllib.request
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
RAW = HERE / "data" / "raw" / "pw"
PDF = RAW / "2015-Census-of-Population-Housing-Agriculture.pdf"
URL = ("https://www.palaugov.pw/wp-content/uploads/2017/02/"
       "2015-Census-of-Population-Housing-Agriculture-.pdf")
OUT = HERE / "data" / "normalized" / "pw.csv"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/124.0 Safari/537.36")

# column order as printed; ids are ISO 3166-2 (geoBoundaries shapeISO)
COLS = ["TOTAL", "PW-100", "PW-218", "PW-214", "PW-228", "PW-212", "PW-226", "PW-004",
        "PW-002", "PW-224", "PW-222", "PW-227", "PW-010", "PW-350", "PW-150", "PW-370",
        "PW-050", "OUTSIDE", "UNKNOWN"]
NAMES = {"PW-100": "Kayangel", "PW-218": "Ngarchelong", "PW-214": "Ngaraard",
         "PW-228": "Ngiwal", "PW-212": "Melekeok", "PW-226": "Ngchesar", "PW-004": "Airai",
         "PW-002": "Aimeliik", "PW-224": "Ngatpang", "PW-222": "Ngardmau",
         "PW-227": "Ngeremlengui", "PW-010": "Angaur", "PW-350": "Peleliu", "PW-150": "Koror",
         "PW-370": "Sonsorol", "PW-050": "Hatohobei", "OUTSIDE": "Outside of Palau",
         "UNKNOWN": "Unknown"}

# "-" printed = 0
T = {
    "Total": [17661, 54, 316, 413, 282, 277, 291, 2455, 334, 282, 185, 350, 119, 484, 11444, 40, 25, 284, 26],
    "Yes, Palauan and another language": [4219, 14, 59, 106, 20, 84, 163, 933, 66, 38, 16, 84, 13, 111, 2457, 5, 0, 47, 3],
    "No, another language": [4785, 2, 35, 43, 22, 45, 14, 519, 56, 44, 8, 41, 8, 54, 3662, 6, 1, 223, 2],
    "Yes, Palauan only": [8657, 38, 222, 264, 240, 148, 114, 1003, 212, 200, 161, 225, 98, 319, 5325, 29, 24, 14, 21],
    "Other language spoken": [9004, 16, 94, 149, 42, 129, 177, 1452, 122, 82, 24, 125, 21, 165, 6119, 11, 1, 270, 5],
    "English": [5640, 16, 72, 128, 27, 117, 170, 1114, 87, 59, 19, 112, 18, 141, 3359, 0, 1, 195, 5],
    "Carolinian": [12, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 3, 4, 0, 0, 0],
    "Other micronesian": [357, 0, 0, 5, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 329, 6, 0, 7, 0],
    "Philippine languages": [2069, 0, 15, 8, 11, 6, 3, 192, 24, 21, 4, 7, 0, 12, 1731, 0, 0, 35, 0],
    "Japanese": [161, 0, 0, 2, 0, 0, 0, 13, 2, 0, 0, 0, 0, 0, 138, 0, 0, 6, 0],
    "Korean": [64, 0, 0, 0, 0, 0, 0, 13, 0, 0, 0, 0, 0, 0, 49, 1, 0, 1, 0],
    "Chinese languages": [247, 0, 0, 0, 0, 0, 0, 28, 3, 0, 1, 3, 0, 0, 207, 0, 0, 5, 0],
    "Taiwanese": [52, 0, 0, 0, 0, 1, 0, 10, 0, 0, 0, 0, 0, 0, 40, 0, 0, 1, 0],
    "Other language": [402, 0, 7, 6, 4, 5, 4, 67, 6, 2, 0, 3, 3, 12, 263, 0, 0, 20, 0],
    "Yes, more often than Palauan": [1367, 0, 27, 24, 13, 11, 22, 215, 15, 30, 6, 13, 7, 41, 896, 6, 1, 40, 0],
    "Both equally": [1422, 0, 10, 12, 1, 26, 22, 217, 23, 8, 5, 53, 3, 22, 1009, 0, 0, 11, 0],
    "No, less frequently than Palauan": [1768, 14, 23, 74, 10, 50, 121, 530, 29, 12, 5, 27, 3, 52, 807, 0, 0, 8, 3],
    "Does not speak Palauan": [4447, 2, 34, 39, 18, 42, 12, 490, 55, 32, 8, 32, 8, 50, 3407, 5, 0, 211, 2],
}
LANGS = ["English", "Carolinian", "Other micronesian", "Philippine languages", "Japanese",
         "Korean", "Chinese languages", "Taiwanese", "Other language"]
USAGE = ["Yes, more often than Palauan", "Both equally", "No, less frequently than Palauan",
         "Does not speak Palauan"]


def fetch():
    RAW.mkdir(parents=True, exist_ok=True)
    if not PDF.exists():
        req = urllib.request.Request(URL, headers={"User-Agent": UA})
        PDF.write_bytes(urllib.request.urlopen(req, timeout=600).read())
    if not PDF.read_bytes()[-1024:].strip().endswith(b"%%EOF"):
        raise SystemExit(f"{PDF} looks truncated (no %%EOF)")


def check():
    n = len(COLS)
    bad = []
    for k, v in T.items():
        if len(v) != n:
            raise SystemExit(f"row {k!r} has {len(v)} cells, want {n}")
        # 1. the Total column is the sum of the places, in every row
        if v[0] != sum(v[1:]):
            bad.append(f"row {k!r}: Total {v[0]} != sum of places {sum(v[1:])}")
    for i, c in enumerate(COLS):
        def g(k):
            return T[k][i]
        # 2. C22's three answers add to the total
        if g("Yes, Palauan and another language") + g("No, another language") + g("Yes, Palauan only") != g("Total"):
            bad.append(f"{c}: C22 answers do not add to the total")
        # 3. the other-language group is exactly the people who speak another language at home
        if g("Yes, Palauan and another language") + g("No, another language") != g("Other language spoken"):
            bad.append(f"{c}: Palauan-and-another + another != other language spoken")
        # 4. the named languages add to the group, and so do C24's four answers
        if sum(g(k) for k in LANGS) != g("Other language spoken"):
            bad.append(f"{c}: languages {sum(g(k) for k in LANGS)} != {g('Other language spoken')}")
        if sum(g(k) for k in USAGE) != g("Other language spoken"):
            bad.append(f"{c}: usage {sum(g(k) for k in USAGE)} != {g('Other language spoken')}")
        # 5. Palauan-only + named languages = everyone (single answer)
        if g("Yes, Palauan only") + sum(g(k) for k in LANGS) != g("Total"):
            bad.append(f"{c}: Palauan only + languages != total")
    if bad:
        raise SystemExit("CHECKS FAILED:\n  " + "\n  ".join(bad))
    print(f"  checks ok: {len(T)} rows x {n} columns add up both ways; Total {T['Total'][0]:,}")
    # the C22 'No' answer and C24's 'does not speak Palauan' disagree slightly; reported, not fixed
    d = T["No, another language"][0] - T["Does not speak Palauan"][0]
    print(f"  C22 'No, another language' {T['No, another language'][0]:,} vs C24 'does not speak "
          f"Palauan' {T['Does not speak Palauan'][0]:,} (difference {d:,})")
    # where the C24 move (countries/pw.py) takes people off English, English must hold them
    for i, c in enumerate(COLS[1:17], start=1):
        less, eng = T["No, less frequently than Palauan"][i], T["English"][i]
        if less > eng:
            print(f"  !! {NAMES[c]}: {less} use Palauan more but only {eng} English")


def write():
    rows = []
    for i, c in enumerate(COLS):
        if c == "TOTAL":
            continue
        level = "state" if c.startswith("PW-") else "unplaced"
        for k in ["Yes, Palauan only"] + LANGS + ["No, less frequently than Palauan",
                                                  "Both equally", "Yes, more often than Palauan",
                                                  "Does not speak Palauan"]:
            rows.append(dict(geo_id=c, geo_level=level, geo_name=NAMES[c], source_category=k,
                             count=T[k][i], tier="measured",
                             source_id="pw_census2015_table16", year=2015))
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"  wrote {OUT} ({len(df)} rows)")


if __name__ == "__main__":
    if "--fetch" in sys.argv:
        fetch()
    check()
    write()
