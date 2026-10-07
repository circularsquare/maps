"""Tunisia: first language by governorate from three Arab Barometer rounds, pooled, as shares of
the RGPH 2024 governorate populations, plus Tunisian Berber from a published estimate
-> data/normalized/tn.csv.

    python sources/tn_surveys.py

NO CENSUS ASKS. The RGPH 2024 form has no language item (queue.csv note), nor had 2014 beyond
literacy. Surveys (religiondots' .sav files, read-only, unweighted, sources/ab_firstlang.py):

  Arab Barometer II   (2011)  q10191 first language    24 governorates  1,196
  Arab Barometer III  (2013)  q1019_1 first language   24 governorates  1,199
  Arab Barometer IV   (2016)  q1019a first language    24 governorates  1,200
  Arab Barometer VII  (2021-22) Q1012B ethnicity: NOT USED; 829 "Other" and 778 "Don't know"
                      of 2,400, so the item did not work in Tunisia.

The pool of 3,595 answers is 99.6% Arabic; one respondent (Tunis, 2013) named Amazigh. That is
the Sudan trap in miniature: Arabic-language fieldwork does not find a minority of half a percent.

TUNISIAN BERBER from Gabsi (2011), "Attrition and maintenance of the Berber language in Tunisia",
International Journal of the Sociology of Language 211: 45,000-50,000 speakers (drawn 47,500),
in Djerba (Guellala, Sedouikech, Ouirsighen, Medenine governorate), Chenini and Douiret
(Tataouine) and the Matmata villages (Gabes). Nothing splits the figure between the three, so
it is shared over them in proportion to population (a placement inside the three-governorate
south-east, not a count per governorate), taken out of each governorate's surveyed Arabic.

CHECKS: Arab Barometer row counts per wave; every survey region mapped; units = religiondots'
24 governorates summing to RGPH 2024, 11,972,169 (tn_lookup.csv, asserted).
"""
import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "sources"))
from rdlink import RD_GEO  # noqa: E402
import ab_firstlang as ab  # noqa: E402

OUT = HERE / "data" / "normalized" / "tn.csv"
LOOKUP = RD_GEO / "tn" / "tn_lookup.csv"
SOURCE_ID = "arabbarometer_ii_iii_iv_gabsi2011"
TOTAL = 11_972_169          # RGPH 2024, religiondots' tn_lookup.csv
N = {"ABII": 1196, "ABIII": 1199, "ABIV": 1200}
BERBER = 47_500             # Gabsi 2011: 45,000-50,000
BERBER_UNITS = ("TN51", "TN52", "TN53")   # Gabes, Medenine, Tataouine

_G = {"Tunis": "TN11", "Ariana": "TN12", "Ben Arous": "TN13", "Manouba": "TN14",
      "Nabeul": "TN15", "Zaghouan": "TN16", "Zaghouane": "TN16", "Bizerte": "TN17",
      "Beja": "TN21", "Jendouba": "TN22", "Le Kef": "TN23", "Kef": "TN23", "Caf": "TN23",
      "Siliana": "TN24", "Seliana": "TN24", "Sousse": "TN31", "Monastir": "TN32",
      "Mahdia": "TN33", "Sfax": "TN34", "Kairouan": "TN41", "Kasserine": "TN42",
      "Sidi Bouzid": "TN43", "Gabes": "TN51", "Qabs": "TN51", "Medenine": "TN52",
      "Mednine": "TN52", "Tataouine": "TN53", "Tatouine": "TN53", "Gafsa": "TN61",
      "Tozeur": "TN62", "Kebili": "TN63", "Qbli": "TN63"}
REGION = dict(_G)
REGION.update({f"80{i:02d}. {k}": v for i, (k, v) in zip(range(7, 31), [
    ("Tunis", "TN11"), ("Ariana", "TN12"), ("Manouba", "TN14"), ("Ben Arous", "TN13"),
    ("Nabeul", "TN15"), ("Zaghouan", "TN16"), ("Bizerte", "TN17"), ("Beja", "TN21"),
    ("Jendouba", "TN22"), ("Caf", "TN23"), ("Siliana", "TN24"), ("Kairouan", "TN41"),
    ("Kasserine", "TN42"), ("Sidi Bouzid", "TN43"), ("Sousse", "TN31"), ("Monastir", "TN32"),
    ("Mahdia", "TN33"), ("Sfax", "TN34"), ("Gafsa", "TN61"), ("Tozeur", "TN62"),
    ("Qbli", "TN63"), ("Qabs", "TN51"), ("Medenine", "TN52"), ("Tataouine", "TN53")])})

LABEL = {"1. Arabic": "Arabic", "Arabic": "Arabic", "2. English": "English",
         "3. French": "French", "French": "French", "6. Italian": "Italian",
         "Amazigh": "Berber"}


def main():
    pop, names = {}, {}
    with open(LOOKUP, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            pop[r["unit"]] = int(r["pop"])
            names[r["unit"]] = r["name"]
    if len(pop) != 24 or sum(pop.values()) != TOTAL:
        raise SystemExit(f"tn_lookup.csv: {len(pop)} units, {sum(pop.values()):,}")

    pooled = {u: {} for u in pop}
    for wave, n in N.items():
        for u, v in ab.answers("Tunisia", wave, REGION, n).items():
            for a, k in v.items():
                if a not in LABEL:
                    raise SystemExit(f"{wave}: answer {a!r} has no LABEL")
                pooled[u][LABEL[a]] = pooled[u].get(LABEL[a], 0) + k
    if any(not v for v in pooled.values()):
        raise SystemExit("a governorate with no answers")

    # Berber: Gabsi's total over the three south-eastern governorates by population
    bpop = sum(pop[u] for u in BERBER_UNITS)
    bshare = {u: pop[u] / bpop for u in BERBER_UNITS}
    rows = []
    for u in sorted(pop):
        v = pooled[u]
        tot = sum(v.values())
        p = pop[u]
        extra = {}
        if u in BERBER_UNITS:
            b = round(BERBER * bshare[u]) if u != BERBER_UNITS[-1] else (
                BERBER - sum(round(BERBER * bshare[x]) for x in BERBER_UNITS[:-1]))
            extra["Berber"] = b
            p -= b
        cnt = ab.shares_to_counts(v, p)
        for k, b in extra.items():
            cnt[k] = cnt.get(k, 0) + b
        for lab in sorted(cnt, key=cnt.get, reverse=True):
            if not cnt[lab]:
                continue
            note = f"{v.get(lab, 0)} of {tot} pooled first-language answers (Arab Barometer II-IV)"
            if lab == "Berber" and u in BERBER_UNITS:
                note += f"; plus {extra['Berber']:,} of Gabsi 2011's 47,500 by population share"
            rows.append(dict(geo_id=u, geo_level="governorate", geo_name=names[u],
                             source_category=lab, count=cnt[lab], tier="modelled",
                             source_id=SOURCE_ID, year=2016,
                             note=note + f"; RGPH 2024 population {pop[u]}"))
        print(f"  {names[u]:<12} n {tot:>4}  " + ", ".join(f"{k} {n}" for k, n in v.items()))
    ab.report(rows, TOTAL)
    ab.write_csv(OUT, rows)
    print(f"  wrote {OUT.relative_to(HERE)}")


if __name__ == "__main__":
    main()
