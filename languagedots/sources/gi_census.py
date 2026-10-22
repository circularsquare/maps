"""Gibraltar: main language, 2022 census, population aged 14 and over -> data/normalized/gi.csv.

    python sources/gi_census.py

THE QUESTION. Census of Gibraltar 2022, Q12: "What is your main language?", options "English" and
"Other (including British Sign Language)" with the language written in; asked of everyone aged 14
and over. Those who did not give English were asked how well they speak it.

THE TABLES. HM Government of Gibraltar, Census of Gibraltar 2022 report (religiondots'
copy, data/raw/gi/census_of_gibraltar_2022_report.pdf, read in place):
  Table 51 (PDF p. 191), Main Language by Nationality Group and Sex: English 27,984, Spanish 1,784,
    Arabic 806, Other 1,384 (including 4 BSL), total 31,958 aged 14+. Males and females add to
    every cell. Table 52 (p. 192) gives the same totals by age group and Table 53 by country of
    birth; the script checks the grand total and the four rows against Table 52's.
  Figure 11 (p. 55), a bar chart of the main languages other than English, as percentages of the
    population aged 14+, for ten languages. Only Spanish (5.6%) and Arabic (2.5%) are printed in the
    text; the other eight are bars of an embedded image. They are read off it here: the image is
    measured against its gridlines (0% to 6%), which reproduces the tabled Spanish (5.58%) and
    Arabic (2.52%) to 0.01 point; the eight small bars come out to about +-10 people each.
    The rest of "Other" (1,384 less those eight less BSL) is unnamed.

THE GRAIN. The island as one unit: the census prints Table 55 by seven residential areas but
defines them only as lists of enumeration areas, with no map (as sources/gi.md in religiondots).

THE UNIVERSE. People aged 14 and over: 31,958 of 37,936 usual residents. The 5,978 under 14 were
not asked and are the gap.

CHECKS (asserted): sex columns add to totals; Table 51's male and female totals equal Table 52's (by age); Spanish and
Arabic read off the chart equal the tabled counts within 0.02 point; the chart's rows are read
in the printed order (descending bar length).
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from rdlink import RD  # noqa: E402

PDF = RD / "data" / "raw" / "gi" / "census_of_gibraltar_2022_report.pdf"
OUT = HERE / "data" / "normalized" / "gi.csv"
NUM = re.compile(r"^[\d,]+$")
TOTAL_14 = 31_958
CHART = ["Spanish", "Arabic", "Portuguese", "Polish", "French", "German", "Hebrew", "Russian", "Chinese",
         "Filipino"]   # bottom to top in the figure


def table51(doc):
    """{language: (both-sexes row, male row, female row)}, each 7 cells ending in the Total column."""
    lines = [x.strip() for x in doc[190].get_text().splitlines() if x.strip()]
    i = lines.index("English")
    out = {}
    for lab in ("English", "Spanish", "Arabic", "Other", "Total"):
        assert lines[i] == lab, (lines[i], lab)
        i += 1
        rows = []
        for sex in (None, "M", "F"):
            if sex:
                assert lines[i] == sex, (lab, lines[i])
                i += 1
            vals = [int(x.replace(",", "")) for x in lines[i:i + 7]]
            rows.append(vals)
            i += 7
        out[lab] = rows
    return out


def chart(doc):
    p = doc[54]
    pix = p.get_pixmap(dpi=400, clip=__import__("fitz").Rect(240, 170, 515, 396))
    a = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)[:, :, :3].astype(int)
    bar = (abs(a[:, :, 0] - 91) < 25) & (abs(a[:, :, 1] - 155) < 25) & (abs(a[:, :, 2] - 213) < 25)
    rows = np.where(bar.sum(axis=1) > 0)[0]
    groups, s, prev = [], rows[0], rows[0]
    for r in rows[1:]:
        if r - prev > 2:
            groups.append((s, prev))
            s = r
        prev = r
    groups.append((s, prev))
    assert len(groups) == 10, groups
    gray = (abs(a[:, :, 0] - 217) < 12) & (abs(a[:, :, 1] - 217) < 12) & (abs(a[:, :, 2] - 217) < 12)
    cs = gray.sum(axis=0)
    cols = np.where(cs > 0.5 * cs.max())[0]
    cols = cols[cols < a.shape[1] - 40]            # the right-hand margin line is not a gridline
    lines, run = [], [cols[0]]
    for c in cols[1:]:
        if c - run[-1] > 2:
            lines.append(np.mean(run))
            run = []
        run.append(c)
    lines.append(np.mean(run))
    lines = [x for x in lines if x > 100]
    assert len(lines) == 7, lines                   # 0% .. 6%
    per = (lines[-1] - lines[0]) / 6.0
    out = {}
    for (y0, y1), lang in zip(reversed(groups), CHART):       # groups are top-to-bottom: Filipino first
        right = np.where(bar[y0:y1 + 1].sum(axis=0) > 0)[0].max() + 1
        out[lang] = (right - lines[0]) / per
    return out


def main():
    import fitz
    doc = fitz.open(PDF)
    t = table51(doc)
    for lab, (both, m, f) in t.items():
        assert [x + y for x, y in zip(m, f)] == both, lab
    for k in range(6):           # the nationality columns add to the total column
        assert all(sum(t[lab][0][:6]) == t[lab][0][6] for lab in t)
    tot = {lab: v[0][6] for lab, v in t.items()}
    assert tot == {"English": 27_984, "Spanish": 1_784, "Arabic": 806, "Other": 1_384, "Total": TOTAL_14}, tot
    assert sum(v for k, v in tot.items() if k != "Total") == TOTAL_14
    # Table 52 (by age group) has the same four rows' totals
    t52 = re.sub(r"\s+", " ", doc[191].get_text())
    mf = " ".join(f"{t[lab][1][6]:,} {t[lab][2][6]:,}" for lab in ("English", "Spanish", "Arabic", "Other"))
    assert f"Total {mf} {t['Total'][1][6]:,} {t['Total'][2][6]:,} {TOTAL_14:,}" in t52, mf
    pct = chart(doc)
    assert abs(pct["Spanish"] - 100 * 1_784 / TOTAL_14) < 0.02, pct["Spanish"]
    assert abs(pct["Arabic"] - 100 * 806 / TOTAL_14) < 0.02, pct["Arabic"]
    small = {k: int(round(pct[k] / 100 * TOTAL_14, -1)) for k in CHART[2:]}
    assert list(small.values()) == sorted(small.values(), reverse=True), small
    rest = tot["Other"] - sum(small.values()) - 4
    assert rest > 0
    rows = [("English", tot["English"]), ("Spanish", tot["Spanish"]), ("Arabic", tot["Arabic"]),
            *small.items(), ("Other", rest), ("British Sign Language", 4)]
    assert sum(n for _, n in rows) == TOTAL_14
    df = pd.DataFrame([dict(geo_id="GI", geo_level="territory", geo_name="Gibraltar", source_category=k,
                            count=n, tier="measured" if k in ("English", "Spanish", "Arabic", "British Sign Language")
                            else "derived", year=2022) for k, n in rows])
    df.to_csv(OUT, index=False, encoding="utf-8")
    print(f"wrote {OUT}: {len(df)} rows, {df['count'].sum():,} people aged 14+")
    print("chart read (percent of 14+): " + ", ".join(f"{k} {v:.2f}" for k, v in pct.items()))
    print(df[["source_category", "count", "tier"]].to_string(index=False))


if __name__ == "__main__":
    main()
