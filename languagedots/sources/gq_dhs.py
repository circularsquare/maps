"""Equatorial Guinea: ethnic group from the 2011 DHS, read as language (AGENT_BRIEF §2, tier D),
on the 2015 census's provinces.

    python sources/gq_dhs.py          -> data/normalized/gq.csv

No census or survey asks language. The DHS (EDSGE-I 2011, report FR271, Cuadro 3.1, pdf p.60)
prints ethnicity of women and men aged 15-49, nationally only: Fang, Bubi, Ndowe, Bisio,
Annobones, Extranjero, Otro, Sin informacion. The national shares are the mean of the women's and
men's weighted percentages ("Sin informacion" dropped), applied to the 2015 census population
(religiondots' gq_lookup.csv, 7 provinces, read-only).

Placement across provinces (the DHS's unit is the whole country, so this moves no count; it
says where in the country each group is drawn; sources/gq.md section 3):
  Bubi              Bioko Norte and Bioko Sur, by population
  Annobonese        all of Annobon; the rest in Bioko Norte (Malabo) and Litoral (Bata), by population
  Ndowe, Bisio      Litoral (the mainland coast)
  foreign, other    every province but Annobon, by population
  Fang              what is left in each province
Rows `derived` (ethnic group read as language, no retention source).
"""
import os
import re
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
import sys  # noqa: E402
sys.path.insert(0, str(ROOT))
from rdlink import RD_GEO  # noqa: E402

PDF = ROOT / "data" / "raw" / "gq" / "dhs2011_FR271.pdf"
PDF_URL = "https://dhsprogram.com/pubs/pdf/FR271/FR271.pdf"
PAGE = 60
LOOKUP = RD_GEO / "gq" / "gq_lookup.csv"
OUT = ROOT / "data" / "normalized" / "gq.csv"
CENSUS_2015 = 1_225_377
GROUPS = ["Fang", "Bubi", "Ndowe", "Bisio", "Annobones", "Extranjero", "Otro", "Sin información"]
# pinned (women %, men %), the read is asserted against these
PINNED = {"Fang": (79.4, 74.9), "Bubi": (8.9, 10.4), "Ndowe": (2.8, 2.7), "Bisio": (1.0, 0.6),
          "Annobones": (2.7, 2.8), "Extranjero": (4.1, 8.2), "Otro": (1.0, 0.2),
          "Sin información": (0.1, 0.3)}

BIOKO = ["Bioko Norte", "Bioko Sur"]
ANNOBON = "Annobón"
LITORAL = "Litoral"


def fetch():
    import requests
    PDF.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(PDF_URL, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
                     timeout=120)
    r.raise_for_status()
    PDF.write_bytes(r.content)


def read_table():
    import fitz
    if not PDF.exists():
        fetch()
    lines = [ln.strip() for ln in fitz.open(PDF)[PAGE - 1].get_text().splitlines() if ln.strip()]
    if not any(ln.startswith("Cuadro 3.1") for ln in lines):
        raise SystemExit(f"Cuadro 3.1 is not on pdf p.{PAGE}")
    i = lines.index("Etnicidad")
    out = {}
    for g in GROUPS:
        j = lines.index(g, i)
        vals = [float(v.replace(".", "").replace(",", ".")) for v in lines[j + 1:j + 7]]
        out[g] = (vals[0], vals[3])     # women's and men's weighted percentage
        if out[g] != PINNED[g]:
            raise SystemExit(f"Cuadro 3.1 {g}: read {out[g]}, pinned {PINNED[g]}")
    for k in (0, 1):
        s = sum(v[k] for v in out.values())
        if abs(s - 100) > 0.2:
            raise SystemExit(f"Cuadro 3.1 ethnicity sums to {s}")
    print("DHS 2011 Cuadro 3.1 ethnicity read and matches the pinned values; women and men each "
          "sum to 100 +-0.2")
    return out


def main():
    tab = read_table()
    share = {g: (w + m) / 2 for g, (w, m) in tab.items() if g != "Sin información"}
    tot = sum(share.values())
    share = {g: v / tot for g, v in share.items()}
    print("national shares, mean of women and men: " +
          ", ".join(f"{g} {v:.2%}" for g, v in share.items()))

    lut = pd.read_csv(LOOKUP, dtype=str)
    lut["pop"] = lut["pop"].astype(int)
    pop = dict(zip(lut["name"], lut["pop"]))
    gid = dict(zip(lut["name"], lut["geo_id"]))
    N = sum(pop.values())
    if len(pop) != 7 or N != CENSUS_2015:
        raise SystemExit(f"{LOOKUP}: {len(pop)} provinces, {N:,} people")
    C = {g: share[g] * N for g in share}

    def by_pop(total, names):
        p = sum(pop[n] for n in names)
        return {n: total * pop[n] / p for n in names}

    alloc = {n: {} for n in pop}
    for n, v in by_pop(C["Bubi"], BIOKO).items():
        alloc[n]["Bubi"] = v
    alloc[ANNOBON]["Annobones"] = pop[ANNOBON]
    for n, v in by_pop(C["Annobones"] - pop[ANNOBON], ["Bioko Norte", LITORAL]).items():
        alloc[n]["Annobones"] = v
    alloc[LITORAL]["Ndowe"] = C["Ndowe"]
    alloc[LITORAL]["Bisio"] = C["Bisio"]
    for g in ("Extranjero", "Otro"):
        for n, v in by_pop(C[g], [n for n in pop if n != ANNOBON]).items():
            alloc[n][g] = v
    rows = []
    for n, d in alloc.items():
        d["Fang"] = pop[n] - sum(d.values())
        if d["Fang"] < 0:
            raise SystemExit(f"{n}: the placed groups exceed its population")
        s = pd.Series(d)
        fl = s.apply(int)
        short = pop[n] - int(fl.sum())
        fl[(s - fl).sort_values(ascending=False).index[:short]] += 1
        for g, c in fl.items():
            if c > 0:
                rows.append(dict(geo_id=gid[n], geo_level="province", geo_name=n,
                                 source_category=g, count=int(c)))
    out = pd.DataFrame(rows)
    nat = out.groupby("source_category")["count"].sum()
    for g in share:
        if abs(nat[g] - C[g]) > 1:
            raise SystemExit(f"{g}: placed {nat[g]:,} against {C[g]:,.0f}")
    if int(out["count"].sum()) != N:
        raise SystemExit("total moved")
    print(f"every province sums to its 2015 census count; national totals equal the DHS shares x "
          f"{N:,}")
    print(out.pivot_table(index="geo_name", columns="source_category", values="count",
                          aggfunc="sum", fill_value=0).to_string())
    out["tier"] = "derived"
    out["year"] = 2015
    out["source_id"] = "dhs2011_ethnicity_x_census2015"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}: {len(out)} rows")


if __name__ == "__main__":
    main()
