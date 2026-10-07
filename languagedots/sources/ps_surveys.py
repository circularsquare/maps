"""Palestine: PCBS census 2017 persons counted per governorate, at Arab Barometer's first language
and ethnic group -> data/normalized/ps.csv.

    python sources/ps_surveys.py

NO CENSUS ASKS LANGUAGE (the 2017 form has no language item; queue.csv note). Population: PCBS
2017 Preliminary Results Table 2, everyone counted per governorate (4,705,601), from
religiondots' ps_lookup.csv (`counted_t2`), read-only. Its 16 units are COD-AB admin 2.

Surveys (religiondots' .sav files, unweighted, sources/ab_firstlang.py):
  Arab Barometer IV  (2016)    q1019a first language     16 governorates, 1,200: all Arabic
  Arab Barometer VII (2021-22) Q1012B ethnic group       16 governorates, 1,800: Arab 1,795,
                                                         Other 2 (Jerusalem), missing 3
  Arab Barometer II and III have Palestine rows but the language item is empty for them.
"Other" ethnic group names no language: on the country's language, as Algeria (sources/dz.md).
So every governorate is 100% Arabic, drawn as Levantine Arabic (South Levantine, sout3123, is a
dialect of Glottolog's nort3139). Domari (Jerusalem's and Gaza's Dom) is not counted by any
source found and is not drawn.

CHECKS: row counts per wave; every survey governorate mapped; the 16 units sum to 4,705,601.
"""
import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(HERE), str(HERE / "sources")]
from rdlink import RD_GEO  # noqa: E402
import ab_firstlang as ab  # noqa: E402

OUT = HERE / "data" / "normalized" / "ps.csv"
LOOKUP = RD_GEO / "ps" / "ps_lookup.csv"
TOTAL = 4_705_601
AB_N = {"ABIV": 1200, "ABVII": 1800}
REGION = {
    "Betlehem": "Bethlehem", "Bethlehem": "Bethlehem", "Deir al Balah": "Dier Al-Balah",
    "Gaza City": "Gaza", "Gaza": "Gaza", "Hebron": "Hebron", "Jabalia": "North Gaza",
    "Jenin": "Jenin", "Jericho": "Jericho & Al-Aghwar", "Jerico": "Jericho & Al-Aghwar",
    "Jerusalem": "Jerusalem", "Khan Younis": "Khan Yunis", "Khan Yunis": "Khan Yunis",
    "Nablus": "Nablus", "Nabulus": "Nablus", "Qalqilia": "Qalqiliya", "Qalqilya": "Qalqiliya",
    "Rafah": "Rafah", "Ramallah": "Ramallah & Al-Bireh", "Salfit": "Salfit",
    "Taulkarm": "Tulkarm", "Tulkarem": "Tulkarm", "Tobas": "Tubas and the Northern Valleys",
    "Tubas": "Tubas and the Northern Valleys",
}
ANSWER = {"Arabic": "Arabic", "Arab": "Arabic", "Other": "Arabic"}


def main():
    pop = {}
    with open(LOOKUP, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            pop[r["unit"]] = int(r["counted_t2"])
    if len(pop) != 16 or sum(pop.values()) != TOTAL:
        raise SystemExit(f"ps_lookup.csv: {len(pop)} units, {sum(pop.values()):,}")
    if set(REGION.values()) != set(pop):
        raise SystemExit(f"survey regions vs units: {set(REGION.values()) ^ set(pop)}")
    pooled = {u: {} for u in pop}
    for wave, n in AB_N.items():
        for u, v in ab.answers("Palestin", wave, REGION, n).items():
            for a, k in v.items():
                if a not in ANSWER:
                    raise SystemExit(f"{wave}: answer {a!r} not mapped")
                pooled[u][ANSWER[a]] = pooled[u].get(ANSWER[a], 0) + k
    rows = []
    for u in sorted(pop):
        v = pooled[u]
        if not v:
            raise SystemExit(f"{u}: no answers")
        for lab, c in ab.shares_to_counts(v, pop[u]).items():
            rows.append(dict(geo_id=u, geo_level="governorate", geo_name=u, source_category=lab,
                             count=c, tier="modelled", source_id="arabbarometer_iv_vii",
                             year=2017, note=f"{sum(v.values())} survey answers, all Arabic; "
                                             f"PCBS 2017 persons counted {pop[u]}"))
    ab.report(rows, TOTAL)
    ab.write_csv(OUT, rows)
    print(f"  wrote {OUT.relative_to(HERE)}")


if __name__ == "__main__":
    main()
