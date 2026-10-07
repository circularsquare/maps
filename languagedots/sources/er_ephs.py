"""Eritrea: ethnic group from the Eritrea Population and Health Survey 2010, read as language
(AGENT_BRIEF §2, tier D), on religiondots' zoba populations.

    python sources/er_ephs.py          -> data/normalized/er.csv

Eritrea has never held a census. EPHS 2010 (NSO and Fafo, final report, Table 3-1, pdf p.70;
religiondots' copy, read-only) prints ethnicity of women 15-49 ("Women ALL", 30,224 weighted) and
men 15-49 (4,299), nationally only; zoba is a separate block, never crossed. Share = mean of the
women's and men's percentages. Population: religiondots' er_lookup.csv (UN WPP 2024 for 2020,
3,291,271, divided between zobas in the survey's own proportions; sources/er.md there).

Placement across zobas (the survey's unit is the whole country, so no count moves; each group is
drawn in the zobas it is known to live in, sources/er.md section 3):
  Afar        Debubawi Keih Bahri up to CAP of it, the rest Semenawi Keih Bahri
  Bilen       Anseba
  Hedareb     Gash-Barka
  Kunama      Gash-Barka
  Nara        Gash-Barka
  Rashaida    Semenawi Keih Bahri
  Saho        Debub and Semenawi Keih Bahri, by population
  Tigre       Anseba, Semenawi Keih Bahri, Gash-Barka, by population
  Other       every zoba, by population
  Tigrigna    what is left in each zoba
Rows `derived`.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from rdlink import RD, RD_GEO  # noqa: E402

PDF = RD / "data" / "raw" / "er" / "ephs2010_final_report_v4.pdf"
PAGE = 70
LOOKUP = RD_GEO / "er" / "er_lookup.csv"
OUT = ROOT / "data" / "normalized" / "er.csv"
TOTAL = 3_291_271
GROUPS = ["Afar", "Bilen", "Hedarib", "Kunama", "Nara", "Rashaida", "Saho", "Tigre", "Tigrigna",
          "Other"]
# (women ALL %, men %), Table 3-1; the read is asserted against these
PINNED = {"Afar": (2.1, 1.8), "Bilen": (3.2, 2.8), "Hedarib": (1.0, 0.8), "Kunama": (1.3, 0.9),
          "Nara": (3.4, 2.4), "Rashaida": (0.1, 0.1), "Saho": (4.4, 4.7), "Tigre": (23.2, 19.6),
          "Tigrigna": (61.2, 66.8), "Other": (0.2, 0.2)}
DKB, SKB, ANSEBA, GB, DEBUB = ("Debubawi Keih Bahri", "Semenawi Keih Bahri", "Anseba",
                               "Gash-Barka", "Debub")
CAP = 0.90          # Afar's share of Debubawi Keih Bahri; Assab's townspeople are mixed
HOMES = {"Bilen": [ANSEBA], "Hedarib": [GB], "Kunama": [GB], "Nara": [GB], "Rashaida": [SKB],
         "Saho": [DEBUB, SKB], "Tigre": [ANSEBA, SKB, GB]}


def read_table():
    import fitz
    lines = [ln.strip() for ln in fitz.open(PDF)[PAGE - 1].get_text().splitlines() if ln.strip()]
    if not any(ln.startswith("Table 3-1") for ln in lines):
        raise SystemExit(f"Table 3-1 is not on pdf p.{PAGE}")
    i = lines.index("Ethnicity")
    out = {}
    for g in GROUPS:
        j = lines.index(g, i)
        v = [float(x.replace(",", "")) for x in lines[j + 1:j + 10]]
        out[g] = (v[3], v[6])        # women ALL %, men %
        if out[g] != PINNED[g]:
            raise SystemExit(f"Table 3-1 {g}: read {out[g]}, pinned {PINNED[g]}")
    for k in (0, 1):
        s = sum(v[k] for v in out.values())
        if abs(s - 100) > 0.2:
            raise SystemExit(f"Table 3-1 ethnicity sums to {s}")
    print("EPHS 2010 Table 3-1 ethnicity read and matches the pinned values")
    return out


def main():
    tab = read_table()
    share = {g: (w + m) / 2 for g, (w, m) in tab.items()}
    tot = sum(share.values())
    share = {g: v / tot for g, v in share.items()}
    lut = pd.read_csv(LOOKUP, dtype=str)
    lut["pop"] = lut["pop"].astype(int)
    pop = dict(zip(lut["name"], lut["pop"]))
    gid = dict(zip(lut["name"], lut["geo_id"]))
    if len(pop) != 6 or sum(pop.values()) != TOTAL:
        raise SystemExit(f"{LOOKUP} moved")
    C = {g: share[g] * TOTAL for g in share}
    print("national shares, mean of women and men: " +
          ", ".join(f"{g} {v:.2%}" for g, v in share.items()))

    alloc = {n: {} for n in pop}
    afar_dkb = min(C["Afar"], CAP * pop[DKB])
    alloc[DKB]["Afar"] = afar_dkb
    alloc[SKB]["Afar"] = C["Afar"] - afar_dkb
    for g, homes in HOMES.items():
        p = sum(pop[n] for n in homes)
        for n in homes:
            alloc[n][g] = C[g] * pop[n] / p
    for n in pop:
        alloc[n]["Other"] = C["Other"] * pop[n] / TOTAL
    rows = []
    for n, d in alloc.items():
        d["Tigrigna"] = pop[n] - sum(d.values())
        if d["Tigrigna"] < 0:
            raise SystemExit(f"{n}: the placed groups exceed its population")
        s = pd.Series(d)
        fl = s.apply(int)
        fl[(s - fl).sort_values(ascending=False).index[:pop[n] - int(fl.sum())]] += 1
        for g, c in fl.items():
            if c > 0:
                rows.append(dict(geo_id=gid[n], geo_level="zoba", geo_name=n, source_category=g,
                                 count=int(c)))
    out = pd.DataFrame(rows)
    nat = out.groupby("source_category")["count"].sum()
    for g in share:
        if abs(nat[g] - C[g]) > 2:
            raise SystemExit(f"{g}: placed {nat[g]:,} against {C[g]:,.0f}")
    if int(out["count"].sum()) != TOTAL:
        raise SystemExit("total moved")
    pv = out.pivot_table(index="geo_name", columns="source_category", values="count",
                         aggfunc="sum", fill_value=0)
    print((pv.div(pv.sum(axis=1), axis=0) * 100).round(1).to_string())
    out["tier"] = "derived"
    out["year"] = 2010
    out["source_id"] = "ephs2010_ethnicity_x_wpp2020"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}: {len(out)} rows, {TOTAL:,} people")


if __name__ == "__main__":
    main()
