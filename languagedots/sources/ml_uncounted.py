"""Mali: the 941,335 people in areas RGPH5 could not enumerate (INSTAT, state and structure volume,
Tableau 1.2), added to the map as `modelled` rows.

    python sources/ml_uncounted.py    check the transcription against the PDF and write
                                      data/normalized/ml_uncounted.csv

INSTAT estimated them with GRID3 from building footprints and roads: 1,087 enumeration sections,
935,960 rural and 5,375 urban, by région (Tableau 1.2, PDF p.45), all ages.

How they are drawn:
  * Language mix: the région's own mother-tongue mix from A06 (ml.csv), foreign languages
    included. Nothing finer exists; INSTAT's percentages do the same for the 0.4% who gave no
    answer. The sections are rural and insecure (Ségou's Macina and Niono, Tombouctou, Douentza,
    Ménaka...), so towns' mix is probably wrong for them: Bambara is more urban than Fula or Tamasheq.
  * Age: A06's universe is residents of ordinary households aged 3 and over (19,144,628 of
    21,347,587, 0.897). Children under three are drawn nowhere on this map, so the same share
    of each région is drawn, and the under-threes are left as the gap.
  * Placement (countries/ml.py): the région's rural hexes, those left once the densest hexes
    holding the région's urban population (Tableau 2.5, PDF p.58-59) are taken out, weighted as
    the région's enumerated people of that language are (CLEAR Global's cercle shares).
The report is religiondots' copy (read only): ../religiondots/data/raw/ml/
rapport-etat-structure-population-rgph5-rgph.pdf.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from rdlink import RD, RD_GEO  # noqa: E402

import pandas as pd  # noqa: E402

PDF = RD / "data" / "raw" / "ml" / "rapport-etat-structure-population-rgph5-rgph.pdf"
NORM = ROOT / "data" / "normalized"
OUT = NORM / "ml_uncounted.csv"

# Tableau 1.2, total (all of it but 5,375 urban is rural)
UNCOUNTED = {"Koulikoro": 1_786, "Nara": 28_873, "Bougouni": 1_078, "Koutiala": 12_508,
             "Ségou": 228_421, "Mopti": 90_120, "Douentza": 154_979, "Bandiagara": 986,
             "Tombouctou": 280_392, "Gao": 44_795, "Kidal": 3_744, "Ménaka": 93_653}
TOTAL = 941_335
# Tableau 2.5, urban population of the région (de droit, 2022) and its total
URBAN = {"Koulikoro": (268_939, 2_255_157), "Nara": (0, 307_777), "Bougouni": (113_088, 1_570_979),
         "Koutiala": (217_403, 1_169_882), "Ségou": (320_157, 2_455_260), "Mopti": (215_854, 945_654),
         "Douentza": (42_221, 150_037), "Bandiagara": (32_099, 878_995),
         "Tombouctou": (148_962, 975_139), "Gao": (144_225, 727_516), "Kidal": (27_226, 83_192),
         "Ménaka": (45_133, 318_876)}


def fmt(n):
    return f"{n:,}".replace(",", " ")


def check_pdf():
    import fitz
    d = fitz.open(PDF)
    p45, p58, p59 = (" ".join(d[i].get_text().split()) for i in (44, 57, 58))
    if "941 335" not in p45:
        raise SystemExit("ml_uncounted: 941 335 not on PDF p.45")
    for r, n in UNCOUNTED.items():
        if fmt(n) not in p45:
            raise SystemExit(f"ml_uncounted: {r} {fmt(n)} not on PDF p.45")
    for r, (u, t) in URBAN.items():
        if u and fmt(u) not in (p58 + p59) or fmt(t) not in (p58 + p59):
            raise SystemExit(f"ml_uncounted: {r} urban {fmt(u)} / {fmt(t)} not on PDF p.58-59")
    if sum(UNCOUNTED.values()) != TOTAL:
        raise SystemExit(f"ml_uncounted: régions sum to {sum(UNCOUNTED.values()):,}, not {TOTAL:,}")


def build():
    check_pdf()
    ml = pd.read_csv(NORM / "ml.csv")
    ml = ml[(ml["geo_level"] == "region") & (ml["count"] > 0)]
    look = pd.read_csv(RD_GEO / "ml" / "ml_lookup.csv").set_index("unit")
    rows = []
    for unit, n in UNCOUNTED.items():
        g = ml[ml["geo_id"] == unit]
        if g.empty:
            raise SystemExit(f"ml_uncounted: no région {unit} in ml.csv")
        f = g["count"].sum() / look.loc[unit, "census_pop_ordinary_2022"]
        for node_label, c in g.groupby("source_category")["count"].sum().items():
            rows.append((unit, node_label, n * f * c / g["count"].sum()))
    out = pd.DataFrame(rows, columns=["unit", "label", "count"])
    out.to_csv(OUT, index=False, encoding="utf-8")
    return out


if __name__ == "__main__":
    out = build()
    print(f"{OUT}: {len(out)} rows, {out['count'].sum():,.0f} people (of {TOTAL:,}, all ages)")
    print(out.groupby("unit")["count"].sum().round(0).to_string())
