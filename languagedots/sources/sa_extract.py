"""Saudi Arabia: the 2022 census tables religiondots already parsed and checked, copied out as
plain CSVs for sources/sa_census.py. Run as its own process (sa_census.py does that), because it
loads religiondots' sources/sa.py, which puts religiondots' taxonomy folder on sys.path, and that
folder's per-country modules share names with ours.

Reads religiondots/data/raw/sa/ (GLMM's mirror of the census portal, GASTAT's report, RCRC's
Riyadh table) through religiondots' own parsers and checks. READ-ONLY: nothing is written there.

Writes data/raw/sa/:
  * sa_regions.csv      13 regions: Saudis, non-Saudis, and non-Saudi men and women (the report's
                        Figure 12 sex ratios raked to the national sexes, as religiondots does)
  * sa_nationality.csv  non-Saudis by nationality and sex, national (GLMM's four tables), plus a
                        `rest` row for the non-Saudis in none of them (the Americas, mostly)
"""
import importlib.util
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RD_SA = ROOT.parent / "religiondots" / "sources" / "sa.py"
OUT = ROOT / "data" / "raw" / "sa"


def main():
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    spec = importlib.util.spec_from_file_location("rd_sa", RD_SA)
    sa = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sa)
    import pandas as pd

    regions = sa.region_table()
    tables = {k: sa.group_table(k) for k in sa.TABLES}
    rows = {k: v for t in tables.values() for k, v in t[0].items()}
    rest = (sa.NON_SAUDI_M - sum(m for m, _f in rows.values()),
            sa.NON_SAUDI_F - sum(f for _m, f in rows.values()))
    if min(rest) <= 0:
        raise SystemExit(f"the four tables hold more non-Saudis than the census: {rest}")
    sa.continent_check(tables, rest)

    # religiondots' sex split by region (sources/sa.py main(), same arithmetic)
    fem = {pc: n * 100.0 / (100 + sa.CENSUS[sa.REGIONS[pc][0]][2][2])
           for pc, (_s, n) in regions.items()}
    k = sa.NON_SAUDI_F / sum(fem.values())
    fem = {pc: v * k for pc, v in fem.items()}

    iso_of = {t: {lab: iso for _sub, members in groups for lab, iso in members.items()}
              for t, (groups, _final) in sa.TABLES.items()}
    nat = [dict(table=t, subtotal=sub, label=lab, iso=iso_of[t][lab] or "", men=m, women=f)
           for (t, sub, lab), (m, f) in rows.items()]
    nat.append(dict(table="rest", subtotal="", label="not in the four tables", iso="",
                    men=rest[0], women=rest[1]))
    nat = pd.DataFrame(nat)
    if (int(nat["men"].sum()), int(nat["women"].sum())) != (sa.NON_SAUDI_M, sa.NON_SAUDI_F):
        raise SystemExit("nationality rows do not sum to the census's non-Saudi men and women")

    reg = pd.DataFrame([dict(geo_id=pc, name=sa.REGIONS[pc][0], saudi=s, non_saudi=n,
                             non_saudi_women=fem[pc], non_saudi_men=n - fem[pc])
                        for pc, (s, n) in regions.items()])
    OUT.mkdir(parents=True, exist_ok=True)
    reg.to_csv(OUT / "sa_regions.csv", index=False)
    nat.to_csv(OUT / "sa_nationality.csv", index=False)
    print(f"wrote {OUT / 'sa_regions.csv'} (13 regions, {int(reg['saudi'].sum()):,} Saudis, "
          f"{int(reg['non_saudi'].sum()):,} non-Saudis) and {OUT / 'sa_nationality.csv'} "
          f"({len(nat)} rows)")


if __name__ == "__main__":
    main()
