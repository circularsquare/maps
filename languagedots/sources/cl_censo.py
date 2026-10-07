"""Chile, Censo de Poblacion y Vivienda 2024 (INE): the indigenous language each person speaks or
understands, per comuna.

    python sources/cl_censo.py [--fetch]

Writes
  data/normalized/cl.csv          geo_id (INE's 5-digit CUT comuna code, as religiondots' cl
                                  layer), geo_level "comuna", geo_name, source_category (INE's
                                  column heading), count. Everyone aged 5 and over; the "does not
                                  speak or understand" and "not declared" columns are kept as rows.
  data/normalized/cl_units.csv    geo_id, geo_name, population (all ages), asked (aged 5+)

THE QUESTION. P30 "Habla o entiende una de las siguientes lenguas indigenas u originarias?", put to
everyone aged 5 and over, indigenous or not. ONE answer: "if a person speaks or understands more
than one indigenous language, select the one they speak best", and "do not count people who
understand only isolated words or greetings" (the English-language questionnaire,
Cuestionario-Ingles_CPV2024.pdf). Eight languages: Mapuzugun, Aymara, Quechua, Rapa Nui, Ckunza,
Kawesqar, Yagan, "another indigenous language of Chile", plus "does not speak or understand any
indigenous language of Chile". Spanish and foreign languages are not asked. The 2017 census asked
pueblo only.

THE SOURCE is INE's published workbook P3_Lenguas-indigenas.xlsx (fourth results release, 30 June
2025; sheet 2 re-issued 4 December 2025 for the renaming of Paihuano). Sheet 2 is comuna x
pertenencia (Total / indigenous / not indigenous) x language; only the Total rows are drawn. The
age workbook D1 (27 March 2025, sheet 4, comuna x five-year age group) gives each comuna's
population and its under-5s, which are not asked.

CHECKS (the script stops unless all hold):
  1. 346 comunas on sheet 2; on every Total row the ten answer columns sum to the 5+ population.
  2. per comuna and column, the pertenencia rows (indigenous, not indigenous, not declared where
     printed) sum to the Total row.
  3. summed over comunas, each column equals the national row and each region's sheet-1 row.
  4. per comuna, D1's population less its "0 a 4" row equals P3's 5+ population, exactly, and
     the national figures are 18,480,432 and 17,609,739.
"""
import sys
import zipfile
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
RAW = HERE / "data" / "raw" / "cl"
NORM = HERE / "data" / "normalized"

BASE = "https://censo2024.ine.gob.cl/wp-content/uploads/"
FILES = {
    "P3_Lenguas-indigenas.xlsx": BASE + "2025/06/P3_Lenguas-indigenas.xlsx",
    "D1_Poblacion-censada-por-sexo-y-edad.xlsx":
        BASE + "2025/03/D1_Poblacion-censada-por-sexo-y-edad-en-grupos-quinquenales.xlsx",
}

POPULATION = 18_480_432
ASKED = 17_609_739
N_COMUNAS = 346

GEO_COLS = ["Código región", "Región", "Código provincia", "Provincia", "Código comuna", "Comuna"]
PERT = "Pertenencia a un pueblo indígena u originario"
UNIVERSE = "Población de 5 años o más"
LANGS = [
    "Mapuzungun (lengua mapuche)",
    "Aymara",
    "Quechua",
    "Rapa Nui",
    "Ckunza",
    "Kawésqar",
    "Yagán",
    "Otra lengua indígena de Chile",
    "No habla ni entiende ninguna lengua indígena u originaria",
    "Manejo de alguna lengua indígena u originaria no declarado",
]
TOTAL = ["Total País", "Total Región", "Total Comuna"]   # INE labels the Total row per level


def fetch():
    import requests
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    RAW.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        dest = RAW / name
        if dest.exists() and dest.stat().st_size > 100_000:
            print("already have", dest)
            continue
        print("GET", url)
        r = requests.get(url, timeout=300, verify=False, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        dest.write_bytes(r.content)
        if not zipfile.is_zipfile(dest):
            raise SystemExit(f"{dest} is not an xlsx")
        print(f"  {dest.stat().st_size:,} bytes")


def _int(v, where):
    if isinstance(v, (int, float)) and v == v and float(v) == int(v):
        return int(v)
    raise SystemExit(f"{where}: unrecognised cell {v!r}")


def _sheet(path, sheet, cols):
    df = pd.read_excel(path, sheet_name=sheet, header=3, dtype=object)
    got = [str(c).strip() for c in df.columns]
    if got != cols:
        raise SystemExit(f"{path.name} sheet {sheet}: columns changed\n got  {got}\n want {cols}")
    df.columns = cols
    df = df[pd.to_numeric(df[cols[0]], errors="coerce").notna()].copy()   # drop the footnote
    return df


def main():
    if "--fetch" in sys.argv:
        fetch()
    p3 = RAW / "P3_Lenguas-indigenas.xlsx"
    d1 = RAW / "D1_Poblacion-censada-por-sexo-y-edad.xlsx"
    for p in (p3, d1):
        if not p.exists():
            raise SystemExit(f"missing {p}; run with --fetch")

    num = [UNIVERSE] + LANGS
    s2 = _sheet(p3, "2", GEO_COLS + [PERT] + num)
    for c in num:
        s2[c] = [_int(v, f"sheet 2 {c}") for v in s2[c]]
    s2["geo_id"] = [f"{_int(v, 'code'):05d}" for v in s2["Código comuna"]]
    nat = s2[s2["geo_id"] == "00000"]
    com = s2[s2["geo_id"] != "00000"]
    tot = com[com[PERT].isin(TOTAL)].set_index("geo_id")
    print("pertenencia labels:", sorted(com[PERT].unique()))

    # 1
    if len(tot) != N_COMUNAS or tot.index.duplicated().any():
        raise SystemExit(f"check 1: {len(tot)} comuna Total rows, want {N_COMUNAS}")
    bad = tot[tot[LANGS].sum(axis=1) != tot[UNIVERSE]]
    if len(bad):
        raise SystemExit(f"check 1: {len(bad)} comunas whose answers do not sum to 5+: {list(bad.index)[:5]}")
    print(f"check 1 ok: {N_COMUNAS} comunas, answers sum to the 5+ population in each")

    # 2
    parts = com[~com[PERT].isin(TOTAL)].groupby("geo_id")[num].sum()
    diff = (parts.reindex(tot.index).fillna(0) - tot[num]).abs().to_numpy().sum()
    if diff:
        raise SystemExit(f"check 2: pertenencia rows differ from Total rows by {diff}")
    print("check 2 ok: pertenencia rows sum to the Total row in every comuna and column")

    # 3
    natrow = nat[nat[PERT].isin(TOTAL)][num].iloc[0]
    if (tot[num].sum() != natrow).any():
        raise SystemExit(f"check 3: comunas do not sum to the national row\n{tot[num].sum() - natrow}")
    s1 = _sheet(p3, "1", ["Código región", "Región", PERT] + num)
    s1 = s1[(s1[PERT].isin(TOTAL)) &(s1["Código región"] != 0)]
    s1 = s1.set_index(s1["Código región"].astype(int))[num].astype("int64")
    byreg = tot.assign(r=tot["Código región"].astype(int)).groupby("r")[num].sum()
    if not byreg.reindex(s1.index).equals(s1):
        raise SystemExit("check 3: comunas do not sum to sheet 1's regions")
    print(f"check 3 ok: comunas sum to the national row and to all {len(s1)} regions")

    # 4
    age = _sheet(d1, "4", GEO_COLS + ["Grupos de edad", "Población censada", "Hombres", "Mujeres",
                                      "Razón hombre-mujer"])
    age["geo_id"] = [f"{_int(v, 'code'):05d}" for v in age["Código comuna"]]
    age["n"] = [_int(v, "D1") for v in age["Población censada"]]
    allage = age[age["Grupos de edad"].isin(TOTAL)].set_index("geo_id")["n"]
    u5 = age[age["Grupos de edad"] == "0 a 4"].set_index("geo_id")["n"]
    if allage["00000"] != POPULATION or allage["00000"] - u5["00000"] != ASKED:
        raise SystemExit("check 4: national population figures changed")
    allage, u5 = allage.drop("00000"), u5.drop("00000")
    if set(allage.index) != set(tot.index):
        raise SystemExit("check 4: D1 and P3 comuna codes differ")
    off = (allage - u5).reindex(tot.index) - tot[UNIVERSE]
    if off.abs().sum():
        raise SystemExit(f"check 4: 5+ population differs from D1 in {(off != 0).sum()} comunas")
    print(f"check 4 ok: D1 all ages less under-5s equals the 5+ population in every comuna "
          f"({POPULATION:,} people, {ASKED:,} aged 5+)")

    rows = []
    for gid, r in tot.iterrows():
        for c in LANGS:
            rows.append((gid, "comuna", r["Comuna"], c, r[c]))
    out = pd.DataFrame(rows, columns=["geo_id", "geo_level", "geo_name", "source_category", "count"])
    NORM.mkdir(parents=True, exist_ok=True)
    out.to_csv(NORM / "cl.csv", index=False, encoding="utf-8")
    units = pd.DataFrame({"geo_id": tot.index, "geo_name": tot["Comuna"].to_numpy(),
                          "population": allage.reindex(tot.index).to_numpy(),
                          "asked": tot[UNIVERSE].to_numpy()})
    units.to_csv(NORM / "cl_units.csv", index=False, encoding="utf-8")
    print(f"wrote {len(out):,} rows, {NORM / 'cl.csv'}")
    print(tot[LANGS].sum().to_string())
    ind = com[~com[PERT].isin(TOTAL)].groupby(PERT)[LANGS].sum().T
    print("\nby pertenencia (national):\n", ind.to_string())


if __name__ == "__main__":
    main()
