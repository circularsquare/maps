"""Isle of Man: language mainly spoken at home, 2021 census -> data/normalized/im.csv.

    python sources/im_census.py

THE TABLE. Isle of Man Census 2021, Report Part I (Economic Affairs Division, Isle of Man
Government), Table 2.11 "Language Mainly Spoken at Home" (PDF p. 28, printed p. 27), from Q6 of the
form: "What language do you use most in the home?", with tick boxes for English, Filipino,
Bengali, Polish, Bulgarian, Punjabi, Chinese, Urdu and a write-in. One answer per person. The
table is the island as a whole (no parish table of language is published) and prints 11 named
languages and "Other"; 41 people have no answer. The report's PDF is religiondots' copy,
read-only: religiondots/data/raw/im/2021-isle-of-man-census-report-part-i.pdf.

Q10 separately asks "Can you speak, read, or write Manx Gaelic?": 2,223 answered yes to some part,
1,005 speak only. That is ability, not the language mostly used at home, so it is not drawn
(sources/im.md); any Manx-at-home answers are inside "Other".

CHECKS (asserted): the table's rows sum to 84,069 less the 41 without an answer; every percentage
equals count / 84,069 to the printed digit; the 84,069 is the resident population stated on p. 7,
9, 16 and 19 of the report.
"""
import re
import sys
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
PDF = HERE.parent / "religiondots" / "data" / "raw" / "im" / "2021-isle-of-man-census-report-part-i.pdf"
OUT = HERE / "data" / "normalized" / "im.csv"
POP = 84_069
NO_ANSWER = 41
EXPECT = ["English", "Polish", "Filipino", "Bulgarian", "Chinese", "Afrikaans", "Hungarian",
          "Portuguese", "Romanian", "Spanish", "Italian", "Other"]


def main():
    import fitz
    doc = fitz.open(PDF)
    text = doc[27].get_text()
    assert "Table 2.11: Language Mainly Spoken at Home" in text
    lines = [x.strip() for x in text.split("\n") if x.strip()]
    i = lines.index("% of population") + 1
    rows = {}
    while lines[i].startswith("Table") is False:
        lab, n, pct = lines[i], lines[i + 1], lines[i + 2]
        if not re.fullmatch(r"[\d,]+", n):
            break
        rows[lab.rstrip("0123456789")] = (int(n.replace(",", "")), float(pct.rstrip("%")))
        i += 3
    assert list(rows) == EXPECT, list(rows)
    tot = sum(n for n, _ in rows.values())
    assert tot == POP - NO_ANSWER, tot
    for lab, (n, pct) in rows.items():
        assert abs(100 * n / POP - pct) < 0.051, (lab, n, pct)
    allt = " ".join(doc[p].get_text() for p in (6, 8, 15, 18))
    assert allt.count("84,069") >= 4
    df = pd.DataFrame([dict(geo_id="IM", geo_level="island", geo_name="Isle of Man", source_category=k,
                            count=n, tier="measured", year=2021) for k, (n, _) in rows.items()])
    df.to_csv(OUT, index=False, encoding="utf-8")
    print(f"wrote {OUT}: {len(df)} languages, {df['count'].sum():,} people ({NO_ANSWER} without an answer, "
          f"not drawn)")
    print(df[["source_category", "count"]].to_string(index=False))


if __name__ == "__main__":
    main()
