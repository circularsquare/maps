"""Fiji: 2007 census, Fijians / Indians / Others per province (Analytical Report, Tables I-5a..d,
pp. 31-37), read as language -> data/normalized/fj.csv.

    python sources/fj_census.py

SOURCE: Fiji Bureau of Statistics, 2007 Census of Population and Housing, Analytical Report
(2012), religiondots' download data/raw/fj/fj_2007_analytical_report.pdf (read-only). Each
province block prints Total, Fijian, Indian, Other for 1996 and 2007; the parser takes the 2007
column and keeps a block only where its Total equals the province's total in religiondots' 2007
religion table (fj.csv, Table P01-3) - all 15 match exactly once. The 2017 census published no
ethnicity by province, and no Fijian census since 1946 has tabulated language.

AGENT_BRIEF section 2, ethnicity read as language:
  Fijian (iTaukei)  -> Fijian. Western Fijian varieties (Nadroga, Ba highlands) not split.
  Indian            -> Fiji Hindi, the home language of Indo-Fijians whatever their ancestral
                       language (Tamil, Telugu and Gujarati descendants included).
  Other, Rotuma     -> Rotuman (the island's own people).
  Other, Cakaudrove -> Gilbertese: the province's large "Other" (5,437, 11%, against 1-3%
                       elsewhere off Rewa and Naitasiri) is Rabi Island's Banaban community,
                       resettled from Banaba in 1945, who speak Gilbertese; Kioa's few hundred
                       Tuvaluans are folded in.
  Other, elsewhere  -> `other` (Part-Europeans, Rotumans off Rotuma, Chinese, Europeans, other
                       Pacific Islanders: no province split exists).
"""
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rdlink import RD  # noqa: E402

PDF = RD / "data" / "raw" / "fj" / "fj_2007_analytical_report.pdf"
RD_FJ = RD / "data" / "normalized" / "fj.csv"
OUT = ROOT / "data" / "normalized" / "fj.csv"
NATIONAL = {"Total": 837_271, "Fijian": 475_739, "Indian": 313_801, "Other": 47_731}  # Table I-3


def blocks():
    import fitz
    doc = fitz.open(PDF)
    lines = []
    for p in range(54, 63):
        lines += [l.strip() for l in doc[p].get_text().splitlines() if l.strip()]
    out, cur = [], {}
    for i, l in enumerate(lines):
        if l in NATIONAL and i + 2 < len(lines):
            try:
                v = int(re.sub(r"[,.\s]", "", lines[i + 2]))   # 1996, then 2007
            except ValueError:
                continue
            if l == "Total":
                cur = {}
                out.append(cur)
            cur[l] = v
    return out


def main():
    rd = pd.read_csv(RD_FJ, dtype=str)
    rd = rd[(rd["geo_level"] == "province") & (rd["source_category"] == "Total")]
    prov = {r.geo_id: (r.geo_name, int(r.count)) for r in rd.itertuples()}
    assert len(prov) == 15
    bl = blocks()
    rows = []
    sums = dict.fromkeys(NATIONAL, 0)
    for gid, (name, tot) in prov.items():
        hits = [b for b in bl if b.get("Total") == tot]
        assert len(hits) == 1, (name, tot, hits)
        b = hits[0]
        assert b["Fijian"] + b["Indian"] + b["Other"] == tot, (name, b)
        for k in NATIONAL:
            sums[k] += b[k]
        other = {"Rotuma": "Other (Rotuma)", "Cakaudrove": "Other (Cakaudrove)"}.get(name, "Other")
        for lab, n in (("Fijian", b["Fijian"]), ("Indian", b["Indian"]), (other, b["Other"])):
            rows.append(dict(geo_id=gid, geo_level="province", geo_name=name, source_category=lab,
                             count=n, tier="derived", year=2007, source_id="fj_census_2007_i5"))
    assert sums == NATIONAL, (sums, NATIONAL)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False, encoding="utf-8")
    print(f"15 provinces, each matched once and summing; national {sums}")
    print(df.groupby("source_category")["count"].sum().to_string())


if __name__ == "__main__":
    main()
