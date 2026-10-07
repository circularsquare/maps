"""Kyrgyzstan populations for helper1m.

Source: National Statistical Committee (NSC, stat.gov.kg), "Численность
постоянного населения областей, районов, городов, айылных аймаков и айылов
(сел)", download/operational/825. One workbook per year, overwritten in place
each spring, so the earlier years come from the Wayback Machine:
  2026  live file (start of 2026)
  2025  Wayback capture 2025-05-25 (start of 2025)
  2024  Wayback capture 2025-02-05 (start of 2024)
Rayon, city and oblast figures are NSC estimates; aiyl aimak and village
figures are the aiyl okmotu (local government) registers, which do not sum to
the rayon. Each rayon's aiyl aimaks and towns are scaled to the NSC rayon total.

The 2024 file is on the old units (452 aiyl aimaks, COD's 2018 codes); 2025
and 2026 are on the units of the 2024-25 reform (230 aiyl aimaks, Bishkek,
Osh, Jalal-Abad/Manas and others enlarged). crosswalk.py ties COD's 2018
polygons to the new units; see README.md.

Writes, under helper1m/data/kyrgyzstan/:
  population.csv   code,level,year,pop   (levels 1-3)
  units.csv        one row per output unit, names and parents
  crosswalk.csv    COD atom -> level-3 unit, and how it was decided
  crosswalk_log.txt

Run: C:\\Python39\\python.exe helper1m\\scripts\\kyrgyzstan\\fetch.py
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import re  # noqa: E402
import sys  # noqa: E402
import urllib.request  # noqa: E402
import warnings  # noqa: E402

import pandas as pd  # noqa: E402

warnings.filterwarnings("ignore", message=".*geographic CRS.*")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import atoms as atoms_mod  # noqa: E402
import crosswalk as cw  # noqa: E402
import nsc  # noqa: E402

HELPER = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HELPER, "data", "kyrgyzstan")
RAW = os.path.join(OUT, "raw")

PAGE = "https://stat.gov.kg/ru/statistics/download/operational/825/"
DOWNLOADS = {
    "op825_2026.xls": PAGE,
    "op825_wb20250525.xls": "https://web.archive.org/web/20250525172135if_/" + PAGE,
    "op825_wb20250205.xls": "https://web.archive.org/web/20250205234553if_/" + PAGE,
    "geonames_KG.zip": "https://download.geonames.org/export/dump/KG.zip",
}
YEAR_OF = {"op825_2026.xls": 2026, "op825_wb20250525.xls": 2025, "op825_wb20250205.xls": 2024}
OFFICIAL_NATIONAL = {2024: 7161910, 2025: 7281827, 2026: 7404329}
RECODE25 = {"41703215610020": "41703215600020", "41706207809000": "41706207600010",
            "41708213600020": "41708213610020"}   # same units, recoded in the 2026 file
# 2024->2025 change outside this band means the crosswalk moved people between
# the units, so the 2024 figure is left out at level 3 (it stays in the sums)
GROWTH_OK = (-0.06, 0.12)


def download():
    os.makedirs(RAW, exist_ok=True)
    for name, url in DOWNLOADS.items():
        path = os.path.join(RAW, name)
        if os.path.exists(path):
            continue
        print("downloading", url)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as r, open(path, "wb") as f:
            f.write(r.read())
    # the live file is replaced every spring; make sure it is still 2026
    for name, year in YEAR_OF.items():
        head = pd.read_excel(os.path.join(RAW, name), sheet_name=0, header=None, nrows=4)
        text = " ".join(str(v) for v in head.values.ravel())
        if f"начало {year}" not in text:
            raise SystemExit(f"{name} is not the start-of-{year} file: {text[:200]}")


TR = {"а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "yo", "ж": "zh",
      "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m", "н": "n", "ң": "ng",
      "о": "o", "ө": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ү": "u",
      "ф": "f", "х": "kh", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "shch", "ъ": "", "ы": "y",
      "ь": "", "э": "e", "ю": "yu", "я": "ya"}


def romanize(s):
    s = s.replace("Дж", "J").replace("дж", "j")
    out = []
    for ch in s:
        lo = ch.lower()
        if lo in TR:
            t = TR[lo]
            out.append(t.capitalize() if ch != lo else t)
        else:
            out.append(ch)
    return "".join(out)


def clean_ru(name):
    n = re.sub(r"(?i)айыль?ный аймак", "", name)
    n = re.sub(r"\(.*?\)", "", n)
    n = re.sub(r"^\s*г\.\s*", "", n.strip())
    n = n.replace("Cай", "Сай").replace("c.", "с.")      # Latin c typos in the file
    n = re.sub(r"\s*-\s*", "-", n)
    return re.sub(r"\s+", " ", n).strip(" ,")


def main():
    download()
    log_lines = []

    def log(*a):
        s = " ".join(str(x) for x in a)
        log_lines.append(s)

    d = cw.load_years()
    for y, t in d.items():
        nat = float(t[t.kind == "national"]["pop"].iloc[0])
        assert nat == OFFICIAL_NATIONAL[y], (y, nat)
    g1, g2, g3 = atoms_mod.load_cod()
    A = atoms_mod.build_atoms()
    A, o2n, m, raked, deficit, dsu = cw.build(d, A, g1, g2, log=log)

    t26 = d[2026]
    cap = [nsc.BISHKEK, nsc.OSH]
    new = t26[t26.kind.isin(["aa", "town", "city"]) | t26.code.isin(cap)].copy()
    new["rep"] = new.code.map(dsu.find)

    def adm2_of(u):
        k = new.set_index("code").kind.get(u)
        if u in cap or k == "city":
            return u
        return u[:8] + "000000"

    def adm1_of(u):
        return u if u in cap else u[:5] + "000000000"

    # ---- level 3 units: clusters of 2026 units, plus rayon land -------------
    A["adm3"] = A.unit26.map(lambda u: dsu.find(u) if isinstance(u, str) else None)
    land = A.adm3.isna()
    A.loc[land, "adm3"] = A.loc[land, "cod_adm2"].map(lambda c: "417" + c[2:7] + "000900")
    A.loc[land, "how"] = "land outside any aiyl aimak"
    units = []
    kinds = new.set_index("code").kind
    for rep, g in new.groupby("rep"):
        g = g.sort_values("pop", ascending=False)
        a2s = {adm2_of(u) for u in g.code}
        if len(a2s) > 1:
            log(f"WARNING cluster {rep} spans level-2 units {a2s}")
        ru = " + ".join(clean_ru(n) for n in g.name)
        en = " + ".join(romanize(clean_ru(n)) for n in g.name)
        k = kinds[rep]
        if rep in cap or k == "city":
            en += " (city)"
        elif k == "town":
            en += " (town)"
        units.append(dict(code=rep, level=3, name=en, name_ru=ru, kind="capital" if rep in cap else k,
                          adm2=adm2_of(rep), adm1=adm1_of(rep), members="|".join(g.code)))
    for code in sorted(A.loc[land, "adm3"].unique()):
        units.append(dict(code=code, level=3, name="Land outside any aiyl aimak",
                          name_ru="Земли вне айылных аймаков", kind="land",
                          adm2=code[:8] + "000000", adm1=code[:5] + "000000000", members=""))
    U3 = pd.DataFrame(units)

    # ---- level 2 and 1 names ------------------------------------------------------
    cod2 = g2.set_index(g2.adm2_pcode.map(atoms_mod.soate))
    cod1 = g1.set_index(g1.adm1_pcode.map(atoms_mod.soate))
    U2 = []
    for c in sorted(U3.adm2.unique()):
        if c in cod2.index and c[5] != "4" and c != nsc.NARYN_CITY:
            en, ru = cod2.adm2_name[c], cod2.adm2_name1[c]
        elif c in cap:
            en, ru = cod1.adm1_name[c], cod1.adm1_name1[c]
        else:
            ru = clean_ru(t26.set_index("code").name[c])
            en = romanize(ru) + " (city)"
        if c == "41703410000010":
            en = "Manas (Jalal-Abad city)"
        U2.append(dict(code=c, level=2, name=en, name_ru=ru, adm1=adm1_of(c)))
    U2 = pd.DataFrame(U2)
    U1 = pd.DataFrame([dict(code=c, level=1, name=cod1.adm1_name[c], name_ru=cod1.adm1_name1[c])
                       for c in sorted(U3.adm1.unique())])
    # the level-3 Manas carries the old name too
    U3.loc[U3.code == "41703410000010", "name"] = "Manas (Jalal-Abad city)"

    # ---- populations --------------------------------------------------------------
    r25 = {RECODE25.get(k, k): v for k, v in raked[2025].items()}
    r26 = raked[2026]
    r24 = dict(raked[2024])
    old24 = d[2024]
    for r in old24[old24.kind == "city_pgt"].itertuples():   # 2024 city rows include their pgt
        city = r.code[:8] + "000010"
        r24[city] -= r24.get(r.code, 0)
    rep_of = new.set_index("code").rep
    p3 = {2024: {}, 2025: {}, 2026: {}}
    for u in new.code:
        rp = rep_of[u]
        p3[2025][rp] = p3[2025].get(rp, 0) + r25.get(u, 0)
        p3[2026][rp] = p3[2026].get(rp, 0) + r26.get(u, 0)
    lost = 0
    for o, u in o2n.items():
        if o not in r24:
            continue
        rp = rep_of.get(dsu.find(u))
        if rp is None:
            lost += r24[o]
            continue
        p3[2024][rp] = p3[2024].get(rp, 0) + r24[o]
    carried = sum(p3[2024].values())
    log(f"2024 carried onto the new units: {carried:.0f} of {OFFICIAL_NATIONAL[2024]} "
        f"(lost {lost:.0f}, unplaced old units "
        f"{OFFICIAL_NATIONAL[2024] - carried - lost:.0f})")
    for code in U3[U3.kind == "land"].code:
        for y in p3:
            p3[y][code] = 0.0

    rows = []
    U3i = U3.set_index("code")
    dropped = []
    for code in U3.code:
        v24, v25, v26 = p3[2024].get(code), p3[2025].get(code, 0), p3[2026].get(code, 0)
        rows.append((code, 3, 2025, v25))
        rows.append((code, 3, 2026, v26))
        if v24 is not None:
            g = v25 / v24 - 1 if v24 else None
            if v24 == 0 and v25 == 0 or (g is not None and GROWTH_OK[0] <= g <= GROWTH_OK[1]):
                rows.append((code, 3, 2024, v24))
            else:
                dropped.append((code, U3i.name[code], v24, v25))
    for lvl, key, U in [(2, "adm2", U2), (1, "adm1", U1)]:
        for code in U.code:
            members = U3[U3[key] == code].code
            for y in (2024, 2025, 2026):
                rows.append((code, lvl, y, sum(p3[y].get(c, 0) for c in members)))
    pop = pd.DataFrame(rows, columns=["code", "level", "year", "pop"])
    pop["pop"] = pop["pop"].round().astype("int64")
    # rounding: make each level-3 set sum to its NSC rayon / city exactly
    for y in (2025, 2026):
        for code in U2.code:
            members = set(U3[U3.adm2 == code].code)
            sel = (pop.level == 3) & (pop.year == y) & pop.code.isin(members)
            target = pop[(pop.level == 2) & (pop.year == y) & (pop.code == code)]["pop"].iloc[0]
            diff = target - pop.loc[sel, "pop"].sum()
            if diff:
                i = pop.loc[sel, "pop"].idxmax()
                pop.loc[i, "pop"] += diff
    pop.to_csv(os.path.join(OUT, "population.csv"), index=False)

    units_all = pd.concat([U1, U2, U3], ignore_index=True)
    units_all.to_csv(os.path.join(OUT, "units.csv"), index=False, encoding="utf-8")
    A[["atom", "kind", "name", "name_ru", "cod_adm2", "old", "unit26", "adm3", "how"]].to_csv(
        os.path.join(OUT, "crosswalk.csv"), index=False, encoding="utf-8")
    log(f"level-3 units with 2024 left out ({len(dropped)}):")
    for x in dropped:
        log(f"   {x[0]} {x[1]}: 2024 {x[2]:.0f} -> 2025 {x[3]:.0f}")
    with open(os.path.join(OUT, "crosswalk_log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")

    print(f"level 1: {len(U1)}  level 2: {len(U2)}  level 3: {len(U3)} "
          f"({(U3.kind == 'land').sum()} land, {U3.members.str.count('[|]').gt(0).sum()} merged)")
    for y in (2024, 2025, 2026):
        tot = pop[(pop.level == 1) & (pop.year == y)]["pop"].sum()
        t3 = pop[(pop.level == 3) & (pop.year == y)]["pop"].sum()
        print(f"  {y}: level 1 sum {tot:,}  level 3 sum {t3:,}  official {OFFICIAL_NATIONAL[y]:,}")
    print(f"  2024 left out at level 3 for {len(dropped)} units")
    print("wrote", os.path.join(OUT, "population.csv"))


if __name__ == "__main__":
    main()
