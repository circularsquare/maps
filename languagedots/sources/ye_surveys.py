"""Yemen: Arab Barometer III first language per governorate, applied to the Population Task
Force's 2025 governorate estimates; Socotra on Soqotri -> data/normalized/ye.csv.

    python sources/ye_surveys.py

NO CENSUS ASKS LANGUAGE (the 2004 census has no language item that anyone has found; queue.csv
note). Population: religiondots' ye_lookup.csv, the Population Task Force's 2025 estimate (CSO,
UNFPA, IOM, OCHA), which projects the 2004 census and corrects it for displacement; 22
governorates (COD-AB), Socotra included.

Survey (religiondots' .sav, read-only, unweighted, sources/ab_firstlang.py):
  Arab Barometer III (2013) q1019_1 first language, 20 governorates, 1,200: all Arabic,
  al-Mahrah's 20 included. "Sana'a" (170) covers both Sana'a City and Sana'a governorate.
  Arab Barometer II's Yemen rows have the item empty; IV, VII and VIII did not field Yemen with
  a language or ethnicity item; V (2018-19) has none.
Raymah and al-Jawf, Ma'rib and the rest are each reached; Socotra (a governorate since 2013) is
not, and is drawn on Soqotri: the island's people are its speakers, Ethnologue's 110,000 (2020)
is more than the 75,725 the Task Force counts there, and no source splits off the Arabic of
Hadibu's newcomers.

MEHRI is not drawn. Every figure found is for Yemen and Oman together (Rubin's "about 130,000",
Ethnologue's 260,000), none splits Yemen's from Oman's, and the survey's 20 answers in al-Mahrah
were all Arabic; Oman's build left its Mehri undrawn for the same reason (sources/om.md).

REFUGEES are not drawn: UNHCR's 63,000 in Yemen (2025: Somali 40,335, Ethiopian 4,371 plus
12,171 asylum seekers, Syrian 2,446, ...) are published nationally only (api.unhcr.org), and the
Task Force figures do not say whether they include them.

CHECKS: the 22 units sum to the Task Force total; every survey governorate mapped; every
governorate but Socotra reached.
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

OUT = HERE / "data" / "normalized" / "ye.csv"
LOOKUP = RD_GEO / "ye" / "ye_lookup.csv"
SOCOTRA = "YE32"
REGION = {
    "Ibb": "YE11", "Abyan": "YE12", "al-Bayda'": "YE14", "Ta'izz": "YE15", "al-Jawf": "YE16",
    "Hajjah": "YE17", "al-Hudaydah": "YE18", "Hadhramaut": "YE19", "Dhamar": "YE20",
    "Shabwah": "YE21", "Saada": "YE22", "Aden": "YE24", "Lahij": "YE25", "Ma'rib": "YE26",
    "al-Mahwit": "YE27", "al-Mahrah": "YE28", "Amran": "YE29", "Ad Dali": "YE30",
    "Raymah": "YE31",
    "Sana'a": ("YE13", "YE23"),        # the capital and the governorate, one sampling region
}
ANSWER = {"Arabic": "Arabic"}


def main():
    pop, names = {}, {}
    with open(LOOKUP, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            pop[r["unit"]] = int(r["pop"])
            names[r["unit"]] = r["name"]
    total = sum(pop.values())
    if len(pop) != 22:
        raise SystemExit(f"ye_lookup.csv: {len(pop)} units")
    region = {k: (v if isinstance(v, str) else "SANAA") for k, v in REGION.items()}
    got = ab.answers("Yemen", "ABIII", region, 1200)
    if "SANAA" in got:
        for u in REGION["Sana'a"]:
            got[u] = dict(got["SANAA"])
        del got["SANAA"]
    reached = set(got)
    if reached != set(pop) - {SOCOTRA}:
        raise SystemExit(f"survey reach: {sorted(set(pop) ^ reached)}")
    rows = []
    for u in sorted(pop):
        if u == SOCOTRA:
            rows.append(dict(geo_id=u, geo_level="governorate", geo_name=names[u],
                             source_category="Soqotri (Socotra)", count=pop[u], tier="derived",
                             source_id="task_force_2025", year=2025,
                             note="Socotra's whole population on Soqotri; no survey reached it"))
            continue
        v = {ANSWER[a]: n for a, n in got[u].items()}
        for lab, c in ab.shares_to_counts(v, pop[u]).items():
            rows.append(dict(geo_id=u, geo_level="governorate", geo_name=names[u],
                             source_category=lab, count=c, tier="modelled",
                             source_id="arabbarometer_iii", year=2013,
                             note=f"{sum(v.values())} survey answers; Task Force 2025 {pop[u]}"))
    ab.report(rows, total)
    ab.write_csv(OUT, rows)
    print(f"  wrote {OUT.relative_to(HERE)}")


if __name__ == "__main__":
    main()
