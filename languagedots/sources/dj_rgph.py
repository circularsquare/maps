"""Djibouti: RGPH-3 2024, languages spoken, several allowed, by region (Tome 4, Tableau 7), plus
native Arabic from Joshua Project's Arab groups. Every count rests on the census's own regional
population in ordinary and nomadic households (religiondots' dj_lookup.csv, `table_2024`).

    python sources/dj_rgph.py      -> data/normalized/dj.csv (region x node, counts)

The source PDF is religiondots' download (read-only): data/raw/dj/rgph3/Tome 4_Caracteristiques
socioculturelles de la population.pdf, Tableau 7 (pp. 39-40), "Repartition de la population
residente de 5 ans et plus des menages ordinaires par region selon les langues et le nombre de
langues parlees". The question is "languages spoken" (SC08_*, one yes/no per language), so it is
AGENT_BRIEF section 2's multi-answer case, with the learned-languages rule:

  * Somali, Afar, Amharic, Oromo, sign language, other: each person shared across the ones they
    named (`count = mentions * base / sum(mentions)`), rows `derived`.
  * French, English: school and work languages; Djibouti's census says so itself (French 45% in
    the capital, 22% in Obock; English "limite a des usages specifiques"). Folded: their mentions
    are dropped before the scaling, so their speakers stay on the languages they also named.
  * Arabic: named by 26.6%, almost all as a learned language (school, religion; 24.5% of
    Djiboutian citizens). Folded the same way, except for the Arabic-speaking Yemeni and Omani
    communities, whose first language it is: Joshua Project's "Arab, Yemeni" (41,000) and
    "Arab, Omani" (28,000) as shares of its Djibouti total, applied to the census population
    and placed in Djibouti-Ville (ARAB_HOME). Rows `modelled` (ask 019's
    ruling: published estimates placed by the census's geography). The census's own Yemeni
    nationals (3,781, 69.9% Arabic-speaking) are a floor: most of the community is Djiboutian.
"""
import csv
import io
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from rdlink import RD_GEO  # noqa: E402

LOOKUP = RD_GEO / "dj" / "dj_lookup.csv"
JP = ROOT / "data" / "raw" / "pg" / "joshuaproject_pgic.csv"
OUT = ROOT / "data" / "normalized" / "dj.csv"
TOTAL_2024 = 1_003_800      # ordinary and nomadic households (Tome 4, Tableau 25 "Ensemble")
BASE_5PLUS = 882_594        # Tableau 7's universe: 5 and over, ordinary households

REGIONS = ["Djibouti-Ville", "Ali-Sabieh", "Dikhil", "Tadjourah", "Obock", "Arta"]
# Tableau 7, "Parle en ..." rows, in the order of REGIONS; the last value is "Ensemble".
T7 = {
    "Ensemble (5+)": [647_786, 62_922, 53_832, 48_836, 30_182, 39_036, 882_594],
    "Francais":      [292_404, 22_411, 17_267, 16_118, 6_760, 14_060, 369_020],
    "Arabe":         [189_031, 12_635, 9_908, 9_241, 7_555, 6_802, 235_172],
    "Anglais":       [121_454, 11_563, 6_311, 4_226, 1_992, 4_567, 150_113],
    "Somali":        [561_652, 59_736, 28_395, 4_709, 2_772, 34_513, 691_777],
    "Afar":          [115_886, 963, 30_893, 47_881, 27_533, 4_703, 227_859],
    "Amharique":     [30_502, 1_189, 1_848, 2_421, 1_384, 988, 38_332],
    "Oromo":         [24_401, 1_935, 953, 1_998, 849, 1_271, 31_407],
    "Langue des signes": [2_356, 147, 48, 162, 314, 28, 3_055],
    "Autres langues non listees": [4_970, 428, 194, 1_265, 346, 165, 7_368],
}
# "Ne parle pas ..." rows, for the check that each pair sums to the region's base
T7_NOT = {
    "Francais": [355_382, 40_511, 36_565, 32_718, 23_422, 24_976, 513_574],
    "Arabe": [458_755, 50_287, 43_924, 39_595, 22_627, 32_234, 647_422],
    "Somali": [86_134, 3_186, 25_437, 44_127, 27_410, 4_523, 190_817],
    "Afar": [531_900, 61_959, 22_939, 955, 2_649, 34_333, 654_735],
}
FOLDED = {"Francais", "Arabe", "Anglais"}
SHARED = ["Somali", "Afar", "Amharique", "Oromo", "Langue des signes",
          "Autres langues non listees"]
NODE = {
    "Somali": "afroasiatic.cushitic.lowland.somali",
    "Afar": "afroasiatic.cushitic.lowland.afar",
    "Amharique": "afroasiatic.ethiosemitic.amharic",
    "Oromo": "afroasiatic.cushitic.lowland.oromo",
    "Langue des signes": "signlanguage",
    "Autres langues non listees": "other",
}
# Joshua Project people groups in Djibouti whose primary language is a variety of Arabic
JP_ARAB = {"Arab, Yemeni": "afroasiatic.yemeni_arabic", "Arab, Omani": "afroasiatic.arabic"}
# Djibouti's Arab trading community lives in the capital (the Arab quarter of Djibouti-Ville);
# spreading it by the census's Arabic mentions would put it in Afar villages, where Arabic is
# the school and Quran language. The capital also has the highest share of Arabic speakers
# in Tableau 7 (29.2%).
ARAB_HOME = "Djibouti-Ville"


def say(ok, msg):
    print(("  ok  " if ok else "  FAIL") + "  " + msg)
    if not ok:
        raise SystemExit("check failed: " + msg)


def largest_remainder(vals, total):
    f = np.asarray(vals, dtype=float)
    base = np.floor(f)
    k = int(round(total - base.sum()))
    base[np.argsort(-(f - base))[:k]] += 1
    return base.astype(int)


def jp_arab_shares():
    txt = JP.read_text(encoding="utf-8-sig")
    rows = [r for r in csv.DictReader(io.StringIO(txt[txt.index("ROG3,"):])) if r["ROG3"] == "DJ"]
    tot = sum(int(r["Population"] or 0) for r in rows)
    out = {}
    for r in rows:
        if r["PeopNameInCountry"] in JP_ARAB:
            out[JP_ARAB[r["PeopNameInCountry"]]] = int(r["Population"]) / tot
            print(f"  Joshua Project: {r['PeopNameInCountry']} {int(r['Population']):,} of "
                  f"{tot:,} ({int(r['Population']) / tot:.2%}), primary language "
                  f"{r['PrimaryLanguageName']}")
    say(len(out) == len(JP_ARAB), "both Arab groups found in Joshua Project's Djibouti rows")
    return out


def main():
    lut = pd.read_csv(LOOKUP)
    lut = lut.set_index("unit").loc[REGIONS]
    pop = lut["table_2024"].astype(int)
    say(int(pop.sum()) == TOTAL_2024, f"dj_lookup.csv table_2024 sums to {int(pop.sum()):,}")
    for k, v in T7.items():
        say(sum(v[:6]) == v[6], f"Tableau 7 {k}: regions sum to the national {v[6]:,}")
    for k, v in T7_NOT.items():
        say(all(a + b == c for a, b, c in zip(v, T7[k], T7["Ensemble (5+)"])),
            f"Tableau 7 {k}: speaks + does not speak = the region's base, every region")
    say(T7["Ensemble (5+)"][6] == BASE_5PLUS, "Tableau 7 base is 882,594")

    arab = jp_arab_shares()
    rows = []
    for i, reg in enumerate(REGIONS):
        n = pop[reg]
        native = ({node: s * TOTAL_2024 for node, s in arab.items()} if reg == ARAB_HOME
                  else {})
        rest = n - sum(native.values())
        m = np.array([T7[k][i] for k in SHARED], dtype=float)
        for k, v in zip(SHARED, m):
            rows.append((reg, NODE[k], k, v * rest / m.sum(), "derived"))
        for node, v in native.items():
            rows.append((reg, node, "Arabic first language (Joshua Project's Arab groups)", v,
                         "modelled"))
        print(f"  {reg:15s} base {n:>8,}  mentions/person (shared languages) "
              f"{m.sum() / T7['Ensemble (5+)'][i]:.3f}; native Arabic {sum(native.values()):,.0f}")
    df = pd.DataFrame(rows, columns=["geo_id", "node", "label", "count", "tier"])
    out = []
    for g, d in df.groupby("geo_id"):
        d = d.copy()
        d["count"] = largest_remainder(d["count"], pop[g])
        out.append(d)
    df = pd.concat(out)
    say(int(df["count"].sum()) == TOTAL_2024, f"drawn total {int(df['count'].sum()):,}")
    nat = df.groupby("node")["count"].sum().sort_values(ascending=False)
    print("\n  national, as drawn:")
    for k, v in nat.items():
        print(f"    {k:42s} {v:>9,}  {v / TOTAL_2024:7.2%}")
    print("\n  by region, as drawn (share):")
    piv = df.pivot_table(index="geo_id", columns="node", values="count", aggfunc="sum").fillna(0)
    print((piv.div(piv.sum(axis=1), axis=0) * 100).round(1).loc[REGIONS].to_string())
    df = df[df["count"] > 0]
    res = pd.DataFrame({
        "geo_id": df["geo_id"], "geo_level": "region", "geo_name": df["geo_id"],
        "source_category": df["node"], "source_label": df["label"], "count": df["count"],
        "tier": df["tier"], "source_id": "dj_rgph3_2024_tome4_t7", "year": 2024,
    })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.sort_values(["geo_id", "count"], ascending=[True, False]).to_csv(OUT, index=False,
                                                                         encoding="utf-8")
    print(f"wrote {OUT} ({len(res)} rows)")


if __name__ == "__main__":
    main()
