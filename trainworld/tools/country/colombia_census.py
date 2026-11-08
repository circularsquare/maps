"""T-2712: Colombia's people by municipio for admin_totals "CO": DANE's projections for 2025 (2018 census base) as COD-PS COL republishes them on HDX, one file by municipio and one by department, both mirrored in
religiondots/data/raw/co/. Codes are COD-AB COL's ADM2_PCODE ("CO13810", DIVIPOLA after "CO"), which the 1,122
polygons of the same mirror carry; one totals file of municipios, departments and the national row, for admin_totals'
parent check.

    python tools/country/colombia_census.py

Writes data/raw/admin_totals/CO/co_mpio_2025.csv (code, name, pop).
"""
import os

import pandas as pd

TW = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(os.path.dirname(TW), "religiondots", "data", "raw", "co")
OUT = os.path.join(TW, "data", "raw", "admin_totals", "CO", "co_mpio_2025.csv")


def main():
    m = pd.read_csv(os.path.join(SRC, "col_admpop_adm2_2025.csv"), encoding="utf-8")
    d = pd.read_csv(os.path.join(SRC, "col_admpop_adm1_2025.csv"), encoding="utf-8")
    assert (m.Year == 2025).all() and (d.Year == 2025).all()
    rows = pd.concat([
        pd.DataFrame({"code": m.ADM2_PCODE, "name": m.ADM2_ES, "pop": m.T_TL}),
        pd.DataFrame({"code": d.ADM1_PCODE, "name": d.ADM1_ES, "pop": d.T_TL}),
        pd.DataFrame({"code": ["CO"], "name": ["Colombia"], "pop": [int(d.T_TL.sum())]}),
    ])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    rows.to_csv(OUT + ".part", index=False, encoding="utf-8")
    os.replace(OUT + ".part", OUT)
    print(f"{len(m):,} municipios, {len(d)} departments, {d.T_TL.sum():,} people -> {OUT}")


if __name__ == "__main__":
    main()
