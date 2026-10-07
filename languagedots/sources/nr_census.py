"""Nauru: 2021 Population and Housing Census, Table I-1 (population by district and ethnicity),
read as language, national -> data/normalized/nr.csv.

    python sources/nr_census.py

SOURCE: Nauru Bureau of Statistics, population-housing-census-2021-tables-vol1.xlsx, sheet I-1,
religiondots' download (read-only). 11,680 people, 16 ethnicities, 15 districts.

NO LANGUAGE TABLE in 2021 (AGENT_BRIEF section 2, ethnicity read as language). RETENTION: the
2011 census report (National Report on Population and Housing, 2011, p. on language): 95% of
people 5+ spoke Nauruan at home, against 94.6% Nauruan ethnicity in 2021, so ethnic Nauruans are
drawn on Nauruan whole. Foreign ethnicities go through origin_mix (the country's home mix on
this map, else its main language); Kosraean on Kosraean (fm.txt), "Other ethnicity" on `other`.

National grain: religiondots' placement layer is one unit and the 15 districts hold 12 dots
between them; the district counts are summed and the sum asserted against the TOTAL row.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sources"))
sys.path.insert(0, str(ROOT / "taxonomy"))
from rdlink import RD  # noqa: E402

XLSX = RD / "data" / "raw" / "nr" / "population-housing-census-2021-tables-vol1.xlsx"
OUT = ROOT / "data" / "normalized" / "nr.csv"
ETH = {"Nauruan": "NR", "Kiribati": "KI", "Tuvaluan": "TV", "Australian": "AU",
       "New Zealander": "NZ", "Fijian": "FJ", "Solomon Islander": "SB", "Filipino": "PH",
       "Chinese": "CN", "Taiwanese": "TW", "Indian": "IN", "Tongan": "TO", "Samoan": "WS",
       "Vanuatu": "VU", "Kosraean": None, "Other ethnicity": None}
FIXED = {"Nauruan": "austronesian.oceanic.nauruan", "Kosraean": "austronesian.oceanic.kosraean",
         "Other ethnicity": "other"}


def main():
    import openpyxl
    from origin_mix import mix
    ws = openpyxl.load_workbook(XLSX, read_only=True)["I-1"]
    rows = list(ws.iter_rows(values_only=True))
    hdr = next(r for r in rows if r[0] == "Total" or (r[0] is None and r[1] == "Total") or "Nauruan" in r)
    cols = [c for c in hdr if c is not None]
    tot = next(r for r in rows if r[0] == "TOTAL")
    vals = [v for v in tot[1:] if v is not None]
    assert cols[0] == "Total" and len(cols) == len(vals) == 17, (cols, vals)
    total = dict(zip(cols, vals))
    dist = [r for r in rows if isinstance(r[0], str) and r[0].strip()[:2].rstrip("-").isdigit()]
    assert len(dist) == 15
    for j, c in enumerate(cols):
        s = sum(int([v for v in r[1:] if v is not None][j]) for r in dist)
        assert s == total[c], (c, s, total[c])
    assert sum(total[c] for c in cols[1:]) == total["Total"]
    assert set(cols[1:]) == set(ETH), set(cols[1:]) ^ set(ETH)
    print(f"I-1: {total['Total']:,} people, 15 districts sum to TOTAL in every column")

    acc = {}
    for c in cols[1:]:
        m = {FIXED[c]: 1.0} if c in FIXED else mix(ETH[c], "nr")
        for node, s in m.items():
            acc[node] = acc.get(node, 0) + total[c] * s
    s = pd.Series(acc)
    fl = s.apply(int)
    fl[(s - fl).sort_values(ascending=False).index[:int(total["Total"]) - int(fl.sum())]] += 1
    df = pd.DataFrame([dict(geo_id="NR", geo_level="country", geo_name="Nauru",
                            source_category=n, count=int(v), tier="derived", year=2021,
                            source_id="nr_census_2021_i1") for n, v in fl.items() if v > 0])
    assert df["count"].sum() == total["Total"]
    df.to_csv(OUT, index=False, encoding="utf-8")
    print(df[["source_category", "count"]].sort_values("count", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
