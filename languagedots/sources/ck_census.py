"""Cook Islands: language spoken at home, 2021 census, by island -> data/normalized/ck.csv.

    python sources/ck_census.py [--fetch]

THE QUESTION (2021 census form, Q18): "What language(s) do you speak at home?  Cook Islands Maori?
yes / no.  Any other language(s)?  yes / no." Two yes/no answers per person aged 5 and over; the
other language is not named. Asked of everyone, in a full count.

THE TABLE. Cook Islands Statistics Office, 2021 Census of Population and Dwellings, report with
tables and questionnaire (stats.gov.ck/2021-census-of-population-and-dwellings/ ->
data/raw/ck/census2021.pdf), Table 3.07 "Resident Population 5 Years and Over, Reading, Writing and
Spoken language by Usual Residence", Both Sex (PDF p. 98), 12 columns per row: reading (4 levels),
writing (4 levels), speak Cook Islands Maori (yes, no), speak other language (yes, no).
Rows: Rarotonga and its 12 districts, five Southern Group islands, six Northern Group islands.

THE COMBINATIONS. The report's text (p. 24) says 29.9% spoke a language other than Cook Islands
Maori only, 11.4% Cook Islands Maori only and 58.7% both, which sum to 100%: nobody answered no
to both. So within any unit, with N people, a = CIM yes and b = other yes:
    both = a + b - N,   CIM only = N - b,   other only = N - a.
Checked nationally against the report's 11.4 / 29.9 / 58.7.

THE DRAWING (AGENT_BRIEF §2, multi-answer): each person shared across the languages they named, so
a person who speaks both is half Cook Islands Maori and half the other language; all rows `derived`.
"Other language(s)" is not named by the census: drawn as English (sources/ck.md).

CHECKS (asserted): the 12 Rarotonga districts sum to the Rarotonga row; the five Southern and six
Northern islands to their group rows; Rarotonga + Southern + Northern to the Cook Islands row;
Table 3.08 (by age) has the same Cook Islands row; N is 13,874 in both questions in every row;
combinations non-negative everywhere; national shares equal the report's.
"""
import argparse
import re
import sys
import urllib.request
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
PDF = HERE / "data" / "raw" / "ck" / "census2021.pdf"
URL = "https://stats.gov.ck/download/83/census-2021/1497/2021-census-report-with-tables-and-questionnaire.pdf"
OUT = HERE / "data" / "normalized" / "ck.csv"
DISTRICTS = ["Kiikii-Tupapa-Maraerenga", "Takuvaine", "Tutakimoa-Teotue", "Avatiu-Ruatonga-Atupa",
             "Nikao-Panama", "Ruaau-Arerenga", "Akaoa-Betela", "Murienua", "Titikaveka", "Ngatangiia",
             "Matavera"]
SOUTH = ["Aitutaki", "Mangaia", "Atiu", "Mauke", "Mitiaro"]
NORTH = ["Palmerston", "Pukapuka", "Nassau", "Manihiki", "Rakahanga", "Penrhyn"]
NUM = re.compile(r"^(-|[\d,]+)$")
TOTAL_5PLUS = 13_874


def _n(s):
    return 0 if s == "-" else int(s.replace(",", ""))


def parse(page):
    import fitz
    lines = [x.strip() for x in fitz.open(PDF)[page - 1].get_text().split("\n") if x.strip()]
    rows, i = {}, 0
    while i < len(lines):
        if not NUM.match(lines[i]) and i + 12 < len(lines) and all(NUM.match(v) for v in lines[i + 1:i + 13]):
            rows[lines[i]] = [_n(v) for v in lines[i + 1:i + 13]]
            i += 13
        else:
            i += 1
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    if ap.parse_args().fetch and not PDF.exists():
        PDF.parent.mkdir(parents=True, exist_ok=True)
        import ssl
        ctx = ssl.create_default_context()
        req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
        PDF.write_bytes(urllib.request.urlopen(req, context=ctx, timeout=300).read())

    r = parse(98)
    want = ["RAROTONGA", *DISTRICTS, "SOUTHERN ISLANDS", *SOUTH, "NORTHERN ISLANDS", *NORTH, "COOK ISLANDS"]
    assert list(r) == want, [x for x in r if x not in want]
    # columns: 0-3 reading, 4-7 writing, 8 CIM yes, 9 CIM no, 10 other yes, 11 other no
    for k, v in r.items():
        assert v[8] + v[9] == v[10] + v[11], (k, v)
    a = {k: (v[8], v[10], v[8] + v[9]) for k, v in r.items()}
    # sums
    def tot(names):
        return tuple(sum(a[n][j] for n in names) for j in range(3))
    assert tot(DISTRICTS + ["Kiikii-Tupapa-Maraerenga"][:0]) == a["RAROTONGA"], (tot(DISTRICTS), a["RAROTONGA"])
    assert tot(SOUTH) == a["SOUTHERN ISLANDS"], (tot(SOUTH), a["SOUTHERN ISLANDS"])
    assert tot(NORTH) == a["NORTHERN ISLANDS"], (tot(NORTH), a["NORTHERN ISLANDS"])
    assert tot(["RAROTONGA", "SOUTHERN ISLANDS", "NORTHERN ISLANDS"]) == a["COOK ISLANDS"]
    assert a["COOK ISLANDS"][2] == TOTAL_5PLUS
    r2 = parse(101)
    assert r2["COOK ISLANDS"][8:] == r["COOK ISLANDS"][8:], r2["COOK ISLANDS"]
    # combinations
    units = ["RAROTONGA"] + SOUTH + NORTH
    rows = []
    for u in units:
        ycim, yoth, N = a[u]
        both, cim_only, oth_only = ycim + yoth - N, N - yoth, N - ycim
        assert min(both, cim_only, oth_only) >= 0, (u, a[u])
        rows.append(dict(unit=u.title(), N=N, both=both, cim_only=cim_only, other_only=oth_only))
    c = pd.DataFrame(rows)
    nat = c[["both", "cim_only", "other_only"]].sum() / c["N"].sum()
    assert [round(100 * x, 1) for x in nat] == [58.7, 11.4, 29.9], nat.tolist()
    # shared across the languages named: both counts half each
    out = []
    for _, x in c.iterrows():
        cim = x.cim_only + x.both / 2
        oth = x.other_only + x.both / 2
        out.append(dict(geo_id=x.unit, geo_level="island", source_category="Cook Islands Maori", count=cim))
        out.append(dict(geo_id=x.unit, geo_level="island", source_category="Other language", count=oth))
    df = pd.DataFrame(out)
    df = df[df["count"] > 0]
    df["tier"] = "derived"
    df["year"] = 2021
    assert abs(df["count"].sum() - TOTAL_5PLUS) < 1e-6
    df.to_csv(OUT, index=False, encoding="utf-8")
    print(f"wrote {OUT}: {df['geo_id'].nunique()} islands, {df['count'].sum():,.0f} people aged 5+")
    c["cim_share_of_mentions"] = (c.cim_only + c.both / 2) / c.N
    print(c.round(3).to_string(index=False))
    print(f"national: both {nat.iloc[0]:.1%}, Cook Islands Maori only {nat.iloc[1]:.1%}, other only {nat.iloc[2]:.1%}")
    print("Rarotonga districts (CIM yes, other yes, N):")
    for d in DISTRICTS:
        print(f"    {d:26s} {a[d]}")


if __name__ == "__main__":
    main()
