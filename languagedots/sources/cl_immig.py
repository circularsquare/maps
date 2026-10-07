"""Chile, Censo 2024 (INE): people born abroad by country or continent of birth, per comuna ->
data/normalized/cl_immig.csv (unit = CUT code, iso, count). Read by countries/cl.py, which turns
them into languages with sources/latam_immig.py (record: sources/cl.md, "Immigrant languages").

    python sources/cl_immig.py [--fetch]

THE SOURCE is INE's second results release (30 April 2025), workbook D4_Inmigracion-
Internacional.xlsx, sheet 4: comuna x birthplace in 13 categories (Argentina, Bolivia, Colombia,
Haiti, Peru, Venezuela, then other South America, other Central America and the Caribbean,
Northern America, Europe, Asia, Africa, Oceania) and "not declared". The published microdata
carry the same 13 categories (diccionario_variables_censo2024.xlsx, p25_lug_nacimiento_esp:
the rest anonymised), so nothing finer exists for 2024.

SPLITTING THE GROUPS. Each continent group is split at the 2017 census's national shares of the
countries INE names in it (INE, "Caracteristicas de la inmigracion internacional en Chile, Censo
2017", 2018, Tabla 5; data/raw/cl/ine_inmigracion_censo2017.pdf p. 25). The groups follow UN M49
(INE's codes are M49: 5 South America, 13 Central America, 21 Northern America), so Mexico is in
13 and Ecuador, not named in 2024, in 5. Countries Tabla 5 folds into "Otro pais" cannot be
named: Asia is drawn all as China (the only Asian country it names), Northern America as the
United States, Oceania as Australia; Africa (1,580) has no named country and stays Spanish.

AGE. cl.csv counts people aged 5+; the foreign-born are all ages. Each comuna's foreign-born are
scaled by its own 5+ share (cl_units.csv), which slightly undercounts immigrants' children
under 5 being fewer; said in the record.

CHECKS: per comuna the categories sum to the printed total; comunas sum to the national rows;
the comunas are cl.csv's.
"""
import sys
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
RAW = HERE / "data" / "raw" / "cl"
NORM = HERE / "data" / "normalized"
URL = "https://censo2024.ine.gob.cl/wp-content/uploads/2025/04/D4_Inmigracion-Internacional.xlsx"

T5 = {   # 2017 census, Tabla 5, the named countries in each 2024 group
    "Otros países de América del Sur": {"EC": 27692, "BR": 14227, "UY": 5172, "PY": 4492},
    "Otros países de América Central y El Caribe": {"DO": 11926, "CU": 6718, "MX": 5806},
    "América del Norte": {"US": 12323},
    "Europa": {"ES": 16675, "DE": 5736, "FR": 5447, "IT": 4097},
    "Asia": {"CN": 9213},
    "Oceanía": {"AU": 1},
}
ONE = {"Argentina": "AR", "Bolivia (Estado Plurinacional de)": "BO", "Colombia": "CO",
       "Haití": "HT", "Perú": "PE", "Venezuela (República Bolivariana de)": "VE"}
SPANISH_LEFT = {"África"}            # no named country: left on the Spanish remainder
UNKNOWN = "País de nacimiento no declarado"
TOTAL = "Total nacidos fuera del país"


def fetch():
    import requests
    RAW.mkdir(parents=True, exist_ok=True)
    r = requests.get(URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=300)
    r.raise_for_status()
    (RAW / "D4_Inmigracion-Internacional.xlsx").write_bytes(r.content)


def main():
    if "--fetch" in sys.argv:
        fetch()
    d = pd.read_excel(RAW / "D4_Inmigracion-Internacional.xlsx", sheet_name="4", header=3)
    d.columns = ["reg", "regn", "prov", "provn", "cut", "com", "cat", "n"]
    d = d[d["cat"].notna() & d["n"].notna()].copy()
    d["cut"] = d["cut"].astype(int)
    nat = d[d["cut"] == 0].set_index("cat")["n"]
    com = d[(d["reg"] != 0) & (d["prov"] != 0) & (d["cut"] != 0)]
    com = com[com["cut"] > 1000]
    w = com.pivot_table(index="cut", columns="cat", values="n", aggfunc="sum")
    ok = True

    def check(c, m):
        nonlocal ok
        print(("ok    " if c else "FAIL  ") + m)
        ok &= bool(c)
    parts = [c for c in w.columns if c != TOTAL]
    check((w[parts].sum(axis=1) == w[TOTAL]).all(), f"{len(w)} comunas; categories sum to the "
                                                    f"comuna total")
    check(all(w[c].sum() == nat[c] for c in w.columns), "comunas sum to the national rows "
                                                        f"(foreign-born {nat[TOTAL]:,.0f})")
    w.index = [f"{c:05d}" for c in w.index]
    units = pd.read_csv(NORM / "cl_units.csv", dtype={"geo_id": str})
    age5 = dict(zip(units["geo_id"], units["asked"] / units["population"]))
    check(set(age5) == set(w.index), "cl_units.csv has every comuna's 5+ share")
    base = pd.read_csv(NORM / "cl.csv", dtype={"geo_id": str})
    ids = set(base["geo_id"])
    check(set(w.index) == ids, "the same comunas as cl.csv")
    known = set(ONE) | set(T5) | SPANISH_LEFT | {UNKNOWN}
    check(set(parts) <= known, f"every category handled ({set(parts) - known})")
    if not ok:
        raise SystemExit("checks failed")
    rows = []
    for cut, r in w.iterrows():
        k = r[[c for c in parts if c != UNKNOWN]].sum()
        f = (1 + r[UNKNOWN] / k if k else 1) * age5[cut]
        for c in parts:
            if c in ONE and r[c]:
                rows.append((cut, ONE[c], r[c] * f))
            elif c in T5 and r[c]:
                t = sum(T5[c].values())
                for iso, n in T5[c].items():
                    rows.append((cut, iso, r[c] * f * n / t))
    out = pd.DataFrame(rows, columns=["unit", "iso", "count"])
    out.to_csv(NORM / "cl_immig.csv", index=False)
    print(f"wrote cl_immig.csv ({len(out):,} rows, {out['count'].sum():,.0f} people aged 5+ by each comuna's share)")


if __name__ == "__main__":
    main()
