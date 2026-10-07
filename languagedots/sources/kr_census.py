"""South Korea: Korean nationals, Jejueo, and foreign residents by nationality, read as languages.
-> data/normalized/kr.csv

    python sources/kr_census.py [--fetch]

No Korean census or survey asks a language (the 2020 census items are nationality and entry date).
Built under Anita's 2026-10-05 ruling for countries with no language question (AGENT_BRIEF.md
§2): the national language, a regional language from a cited estimate, immigrant languages
proxied by nationality. EVERY ROW IS `derived`.

Table: Ministry of the Interior and Safety (MOIS), "2020 지방자치단체 외국인주민 현황" (foreign
residents by local government, 2020-11-01), the statistical workbook posted on mois.go.kr
(board BBSMSTR_000000000014, article 88648). Open download, no login. Sheets used:
  1-2  총인구 (A) per si-gun-gu: the 2020 census population, foreigners included
       (51,829,136 nationally), and 한국국적을 가지지 않은 자 (C), foreign nationals;
  4-2  foreign nationals by nationality per si-gun-gu, 36 named nationalities or remainders,
       with Korean-Chinese (중국(한국계)) and Russian Koreans (러시아(한국계)) apart;
  5-4-2  the same for 외국국적동포 (ethnic Koreans of foreign nationality) alone, which gives the
       Koryo-saram among Central Asian nationals.
Cells under 5 are printed `*`; each unit's starred cells share its residual (합계 minus the
printed cells) equally.

  1. Korean nationals (A - C), naturalised citizens and foreign residents' children included,
     on Korean; 7,500 of them in Jeju on Jejueo (sources/kr.md §3).
  2. foreign nationals: each nationality on a language or a home mix (ORIGIN below); Korean-
     Chinese on Korean; Russian Koreans and Central Asian 동포 on Russian.
  3. retention: for marriage migrants only (sheet 5-2-2, nationals married to Koreans), France's
     TeO2 share speaking only the host language (fr_build.TEO_FRENCH, by region of origin) moves
     onto Korean. Workers, students, ethnic-Korean and other foreign nationals keep their
     origin language whole (sources/kr.md §4).
"""
import os
import sys
import urllib.request
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT / "taxonomy"), str(ROOT)]

import numpy as np  # noqa: E402
import openpyxl  # noqa: E402
import pandas as pd  # noqa: E402

RAW = ROOT / "data" / "raw" / "kr"
XLSX = RAW / "mois_foreign_residents_2020.xlsx"
URL = ("https://www.mois.go.kr/cmm/fms/FileDown.do?atchFileId=FILE_00105361PZHQd7S&fileSn=5")
OUT = ROOT / "data" / "normalized" / "kr.csv"
RD_LOOKUP = ROOT.parent / "religiondots" / "data" / "geo" / "kr" / "kr_lookup.csv"

TOTAL_2020 = 51_829_136          # 총인구, sheet 1-2 national row (= 2020 census)
FOREIGN_2020 = 1_695_643         # 한국국적을 가지지 않은 자

KOREAN = "koreanic.korean"
JEJUEO = "koreanic.jejueo"
# Jejueo Project, University of Hawai'i at Manoa (O'Grady, Yang): "Jejueo has at most 5000 to
# 10,000 speakers ... largely by elderly speakers". UNESCO, critically endangered (2010).
# Midpoint of the range.
JEJUEO_SPEAKERS = 7_500
JEJU = "제주특별자치도"

# sheet 4-2 / 5-4-2: column of each nationality's 계 (both sheets share the layout) ->
# (iso or None, TeO2 block for remainders, how it is drawn)
#   iso: the origin's language via ORIGIN; None: a remainder, drawn on `other`
COLS = {
    7: "CN", 10: "CN-KR", 13: "TW", 16: "JP", 19: "MN",
    25: "VN", 28: "PH", 31: "TH", 34: "ID", 37: "KH", 40: "MM", 43: "MY", 46: "LA", 49: "TL",
    52: "SEA-other",
    58: "LK", 61: "PK", 64: "BD", 67: "NP", 70: "SA-other",
    76: "UZ", 79: "KZ", 82: "KG", 85: "CA-other", 88: "ASIA-other",
    91: "US", 94: "CA",
    100: "RU", 103: "RU-KR", 106: "UK", 109: "EUR-other",
    112: "OCE", 115: "LATAM", 118: "AFR", 121: "other",
}
SUBTOTALS = {4: (7, 19), 22: (25, 52), 55: (58, 70), 73: (76, 85), 97: (100, 109)}
TOTAL_COL = 1
CENTRAL_ASIA_DONGPO = (76, 79, 82)   # Koryo-saram among these nationals (sheet 5-4-2)

# origins at their own drawn mix (multilingual, on this map); 1% cut as sources/sa_census.py
HOME_MIX = {"CN", "TW", "VN", "PH", "TH", "ID", "KH", "MM", "MY", "LA", "TL", "LK", "PK", "BD",
            "NP", "UZ", "KZ", "KG"}
SINGLE = {  # fr2023.NAMES labels
    "JP": "Japanese", "MN": "Mongolian", "US": "English", "CA": "English", "UK": "English",
    "RU": "Russian", "RU-KR": "Russian", "OCE": "English", "LATAM": "Spanish",
}
NODE = {"CN-KR": KOREAN, "AFR": "africa_other", "SEA-other": "other", "SA-other": "other",
        "CA-other": "other", "ASIA-other": "other", "EUR-other": "other", "other": "other"}
# TeO2 region (fr_build.TEO2) for each origin; ethnic-Korean rows take no retention (already Korean
# or, for Russian Koreans, a host-language switch TeO says nothing about)
TEO_BLOCK = {"CN": "China", "TW": "China", "JP": "Asia", "MN": "Asia",
             "VN": "Southeast Asia", "PH": "Southeast Asia", "TH": "Southeast Asia",
             "ID": "Southeast Asia", "KH": "Southeast Asia", "MM": "Southeast Asia",
             "MY": "Southeast Asia", "LA": "Southeast Asia", "TL": "Southeast Asia",
             "SEA-other": "Southeast Asia", "LK": "Asia", "PK": "Asia", "BD": "Asia",
             "NP": "Asia", "SA-other": "Asia", "UZ": "Asia", "KZ": "Asia", "KG": "Asia",
             "CA-other": "Asia", "ASIA-other": "Asia", "US": "Americas, Oceania",
             "CA": "Americas, Oceania", "OCE": "Americas, Oceania",
             "LATAM": "Americas, Oceania", "RU": "Europe", "UK": "Europe",
             "EUR-other": "Europe", "AFR": "Africa", "other": "Asia"}
PROVINCE_SUFFIX = ("특별시", "광역시", "특별자치시", "특별자치도", "도")
RENAMED = {("인천광역시", "미추홀구"): "남구"}


def fetch():
    RAW.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(URL, headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
        "Referer": "https://www.mois.go.kr/"})
    data = urllib.request.urlopen(req, timeout=120).read()
    if data[:2] != b"PK":
        raise SystemExit("MOIS download is not an xlsx")
    XLSX.write_bytes(data)
    print(f"fetched {XLSX} ({len(data):,} bytes)")


def num(v):
    if v is None or v == "-":
        return 0
    if v == "*":
        return None
    return int(v)


def sheet_rows(wb, name):
    ws = wb[name]
    out = []
    for r in list(ws.iter_rows(values_only=True))[7:]:
        if r[0] is None:
            continue
        out.append(r)
    return out


def with_units(rows):
    """[(province, name, row)] for si-gun-gu rows, general gu (수원시장안구) dropped: their city's
    own row covers them, and religiondots' units are the cities."""
    prov, out, names = None, [], set()
    for r in rows[1:]:          # rows[0] is 전국
        name = str(r[0]).strip()
        if name.endswith(PROVINCE_SUFFIX):
            prov, names = name, set()
            continue
        # Incheon's Nam-gu was renamed Michuhol-gu in 2018; religiondots' 2015 units keep 남구
        name = RENAMED.get((prov, name), name)
        if any(name.startswith(c) and name != c for c in names if c.endswith("시")):
            continue
        names.add(name)
        out.append((prov, name, r))
    return out


def nationality_table(rows):
    """unit rows -> DataFrame (province, name, one column per COLS key), starred cells filled."""
    recs = []
    for prov, name, r in rows:
        vals = {j: num(r[j]) for j in COLS}
        total = num(r[TOTAL_COL])
        star = [j for j, v in vals.items() if v is None]
        if star:
            # fill within each subtotal first when it is printed, else against the unit total
            for sub, (a, b) in SUBTOTALS.items():
                s = num(r[sub])
                grp = [j for j in star if a <= j <= b]
                if s is not None and grp:
                    resid = s - sum(vals[j] for j in COLS if a <= j <= b and vals[j] is not None)
                    for j in grp:
                        vals[j] = max(resid, 0) / len(grp)
            star = [j for j, v in vals.items() if v is None]
            if star:
                resid = total - sum(v for v in vals.values() if v is not None)
                for j in star:
                    vals[j] = max(resid, 0) / len(star)
        got = sum(vals.values())
        if abs(got - total) > 0.5 + 0.01 * total and total > 50:
            raise SystemExit(f"{prov} {name}: nationalities sum to {got:,.0f}, 합계 {total:,}")
        rec = {"province": prov, "name": name, "total": total}
        rec.update({COLS[j]: vals[j] for j in COLS})
        recs.append(rec)
    return pd.DataFrame(recs)


MIN_SHARE = 0.01


def home_mix(cc):
    from countries import load_one
    df = load_one(cc.lower())["counts"]()
    s = df.groupby("node")["count"].sum()
    s = s[s > 0] / s.sum()
    s = s[s >= MIN_SHARE]
    return (s / s.sum()).to_dict()


def origin_mix(key):
    import fr2023
    if key in NODE:
        return {NODE[key]: 1.0}
    if key in ("RU-KR", "OCE", "LATAM"):   # ethnic Koreans from Russia; continent remainders
        return {fr2023.NAMES[SINGLE[key]]: 1.0}
    # every country: the shared origin table (sources/origin_mix.py, 2026-10-05), which takes
    # HOME_MIX's home mixes and SINGLE's countries alike
    from origin_mix import mix
    return mix({"UK": "GB"}.get(key, key), "kr")


def main():
    if "--fetch" in sys.argv or not XLSX.exists():
        fetch()
    from fr_build import TEO_FRENCH

    wb = openpyxl.load_workbook(XLSX, read_only=True, data_only=True)
    pop_rows = sheet_rows(wb, "1-2. 유형 및 지역별(시.군.구)")
    nat_rows = sheet_rows(wb, "4-2. 국적별(시.군.구)")
    dp_rows = sheet_rows(wb, "5-4-2. 외국국적동포(시.군.구)")
    if num(pop_rows[0][1]) != TOTAL_2020 or num(pop_rows[0][6]) != FOREIGN_2020:
        raise SystemExit("sheet 1-2's national row is not the 2020 figures")
    if num(nat_rows[0][TOTAL_COL]) != FOREIGN_2020:
        raise SystemExit("sheet 4-2's national row does not sum to the foreign nationals")

    pop = {(p, n): (num(r[1]), num(r[6])) for p, n, r in with_units(pop_rows)}
    nat = nationality_table(with_units(nat_rows))
    dp = {(p, n): r for p, n, r in with_units(dp_rows)}
    # marriage migrants by nationality (sheet 5-2-2): the only foreign nationals given retention
    mm = nationality_table(with_units(sheet_rows(wb, "5-2-2. 결혼이민자(시.군.구)")))
    mm = {(p, n): row for p, n, row in zip(mm["province"], mm["name"], mm.to_dict("records"))}

    # ---- join to religiondots' 229 units, both ways ----
    lut = pd.read_csv(RD_LOOKUP, dtype=str)
    key = {(r.province, r.name): r.unit for r in lut.itertuples()}
    nat["unit"] = [key.get((p, n)) for p, n in zip(nat["province"], nat["name"])]
    if nat["unit"].isna().any():
        raise SystemExit(f"MOIS units not in kr_lookup: "
                         f"{nat.loc[nat['unit'].isna(), ['province', 'name']].values.tolist()}")
    if set(nat["unit"]) != set(lut["unit"]) or len(nat) != len(lut):
        raise SystemExit(f"join is not 1:1: {len(nat)} MOIS units, {len(lut)} religiondots units")
    if set(pop) != set(zip(nat["province"], nat["name"])):
        raise SystemExit("sheets 1-2 and 4-2 list different units")
    if sum(a for a, _ in pop.values()) != TOTAL_2020:
        raise SystemExit("si-gun-gu populations do not sum to the national total")
    for (p, n), (a, c) in pop.items():
        t = int(nat.loc[(nat["province"] == p) & (nat["name"] == n), "total"].iloc[0])
        if t != c:
            raise SystemExit(f"{p} {n}: sheet 1-2 says {c:,} foreign nationals, 4-2 {t:,}")

    import fr2023
    RUSSIAN = fr2023.NAMES["Russian"]
    mixes = {k: origin_mix(k) for k in set(COLS.values())}
    rows = []
    moved = {}
    for _, r in nat.iterrows():
        a, c = pop[(r["province"], r["name"])]
        koreans = a - c
        acc = {KOREAN: float(koreans)}
        # Koryo-saram among Central Asian nationals: 외국국적동포 of that nationality (sheet 5-4-2)
        d = dp.get((r["province"], r["name"]))
        for j in CENTRAL_ASIA_DONGPO:
            k = COLS[j]
            v = num(d[j]) if d is not None else 0
            v = 2.5 if v is None else v
            v = min(v, r[k])
            r[k] -= v
            acc[RUSSIAN] = acc.get(RUSSIAN, 0.0) + v
        for k in set(COLS.values()):
            n = float(r[k])
            if n <= 0:
                continue
            m_n = min(float(mm[(r["province"], r["name"])][k]), n)
            for node, s in mixes[k].items():
                x = n * s
                if node != KOREAN and k in TEO_BLOCK:
                    # TeO2's share speaking only the host language, for the marriage migrants
                    # among this unit's nationals of this origin only
                    f = TEO_FRENCH[TEO_BLOCK[k]] * m_n / n
                    acc[KOREAN] += x * f
                    moved[node] = moved.get(node, 0.0) + x * f
                    x *= 1 - f
                acc[node] = acc.get(node, 0.0) + x
        if r["province"] == JEJU:
            acc["_jeju_koreans"] = koreans
        rows.append((r["unit"], r["province"], r["name"], a, acc))

    # Jejueo: 7,500 speakers in Jeju's two units by Korean nationals
    jeju_k = sum(acc["_jeju_koreans"] for *_, acc in rows if "_jeju_koreans" in acc)
    out = []
    for unit, prov, name, a, acc in rows:
        if "_jeju_koreans" in acc:
            j = JEJUEO_SPEAKERS * acc.pop("_jeju_koreans") / jeju_k
            acc[KOREAN] -= j
            acc[JEJUEO] = j
        # round within the unit to its census total
        nodes = list(acc)
        v = np.array([acc[n] for n in nodes])
        base = np.floor(v).astype(int)
        short = a - int(base.sum())
        if short:
            base[np.argsort(-(v - base))[:short]] += 1
        for n, cnt in zip(nodes, base):
            if cnt > 0:
                out.append(dict(geo_id=unit, geo_level="sigungu", geo_name=f"{prov} {name}",
                                source_category=n, count=int(cnt)))
    df = pd.DataFrame(out)
    df["tier"] = "derived"
    df["year"] = 2020
    if int(df["count"].sum()) != TOTAL_2020:
        raise SystemExit("output does not sum to the census")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, encoding="utf-8")

    tot = df.groupby("source_category")["count"].sum().sort_values(ascending=False)
    print(f"wrote {OUT}: {len(df)} rows, {df['geo_id'].nunique()} units, "
          f"{tot.sum():,} people, {len(tot)} nodes")
    for n, cnt in tot.head(25).items():
        print(f"    {n:<50} {cnt:>11,}  {cnt / TOTAL_2020:7.3%}")
    print(f"  moved onto Korean by TeO2 retention: {sum(moved.values()):,.0f}")
    for n, v in sorted(moved.items(), key=lambda kv: -kv[1])[:8]:
        print(f"    {n:<50} {v:>11,.0f}")
    print("  starred-cell residual assumed for Central Asian 동포 where printed `*`: 2.5")


if __name__ == "__main__":
    main()
