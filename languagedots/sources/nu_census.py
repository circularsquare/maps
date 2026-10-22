"""Niue: household languages, 2022 census -> data/normalized/nu.csv.

    python sources/nu_census.py [--fetch]

THE QUESTION. Niue Census of Population and Dwellings 2022 asked each private occupied dwelling
which languages are spoken in the household ("Household languages spoken", report p. 79; several
may be named), and separately each person aged 5 and over how well they speak, read, write and
understand Vagahau Niue. The household question is the home-language one.

THE TABLE. Niue Statistics Office, 2022 Niue Census of Population and Housing Report
(niuestatistics.nu/download/35/census/2793/2022-niue-census-of-population-and-housing-report.pdf ->
data/raw/nu/census2022.pdf), the table "Main language spoken in Household" for private occupied
dwellings by village (PDF p. 70, below Table 19; it carries the next table's number, Table 20, in
the PDF): per village the households naming Vagahau Niue, English and Other languages, and the
number of households. Rows overlap (Hikutavake: 13 + 13 + 0 mentions in 13 households), so it is
a multi-answer table: 395 + 328 + 65 = 788 mentions in 513 households.

THE DRAWING (AGENT_BRIEF §2, multi-answer; spec §3.6). Each mention is a share of the island's
1,564 usual residents: count = mentions x 1,564 / 788, so Vagahau Niue 784, English 651, other 129.
All rows `derived`. The island is one unit: the table is by village but 1,564 people draw one
or two dots at 1:1000, so a village grain would draw nothing more. The village rows are kept in
sources/nu.md.

CHECKS (asserted): the 14 villages sum to the printed Total row, 513 households and each language;
513 is also Table 19's total (occupants by household size, same villages); 1,564 usual residents is
the figure in the report's text (pp. 10, 13).
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
PDF = HERE / "data" / "raw" / "nu" / "census2022.pdf"
URL = "https://niuestatistics.nu/download/35/census/2793/2022-niue-census-of-population-and-housing-report.pdf"
OUT = HERE / "data" / "normalized" / "nu.csv"
VILLAGES = ["Alofi South", "Alofi North", "Makefu", "Tuapa", "Namukulu", "Hikutavake", "Toi", "Mutalau",
            "Lakepa", "Liku", "Hakupu", "Vaiea", "Avatele", "Tamakautoga"]
RESIDENTS = 1_564


def parse(doc):
    lines = [x.strip() for x in doc[69].get_text().splitlines() if x.strip()]
    i = max(k for k, x in enumerate(lines) if x == "Main language spoken in Household")
    i = lines.index("Alofi South", i)
    rows = {}
    for v in VILLAGES + ["Total"]:
        assert lines[i] == v, (lines[i], v)
        rows[v] = [int(x) for x in lines[i + 1:i + 5]]
        i += 5
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    if ap.parse_args().fetch and not PDF.exists():
        PDF.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
        PDF.write_bytes(urllib.request.urlopen(req, timeout=300).read())
    import fitz
    doc = fitz.open(PDF)
    rows = parse(doc)
    tot = rows.pop("Total")
    s = [sum(r[k] for r in rows.values()) for k in range(4)]
    assert s == tot == [395, 328, 65, 513], (s, tot)
    t = re.sub(r"\s+", " ", doc[69].get_text())
    assert "Total 124 119 80 74 55 24 37 513" in t                       # Table 19's total, same villages
    assert "1,564 (93%) considered Niue" in re.sub(r"\s+", " ", doc[9].get_text())
    mentions = sum(tot[:3])
    labels = ["Vagahau Niue", "English", "Other languages"]
    df = pd.DataFrame([dict(geo_id="NU", geo_level="island", geo_name="Niue", source_category=lab,
                            count=n * RESIDENTS / mentions, tier="derived", year=2022)
                       for lab, n in zip(labels, tot[:3])])
    assert abs(df["count"].sum() - RESIDENTS) < 1e-9
    df.to_csv(OUT, index=False, encoding="utf-8")
    print(f"wrote {OUT}: {RESIDENTS:,} usual residents; {tot[3]} households, {mentions} mentions")
    print(df[["source_category", "count"]].round(0).to_string(index=False))
    print("villages (V, E, other, households):")
    for v, r in rows.items():
        print(f"    {v:14s} {r}")


if __name__ == "__main__":
    main()
