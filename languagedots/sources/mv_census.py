"""Maldives: Dhivehi plus foreign residents by nationality, per atoll, from the 2022 census
-> data/normalized/mv.csv.

    python sources/mv_census.py            (downloads the two inputs from the Wayback Machine
                                            if they are missing; census.gov.mv answers 404)

NO LANGUAGE QUESTION (the 2022 census asks none; the 2014 form did not either). Under AGENT_BRIEF
§2 (no language question, proxy by citizenship), every row `derived`:
  1. Maldivians on Dhivehi.
  2. foreign residents by nationality: Table 6.1 of the census's migration report ("Population
     Movement & Migration Dynamics", 2024) gives India, Sri Lanka, Bangladesh, the Philippines,
     Indonesia, Nepal and Others separately for Male and for the atolls together. Each atoll's
     foreign residents (Table MG11) take the atolls' mix; Male takes Male's.
  3. each nationality at its country's own drawn mix on this map (sources/sa.md's home-mix
     method, languages of 1%+ kept and scaled to 100%); "Others" (5,123: resort staff from
     everywhere, nothing names them) on `other`.
No retention step: the foreign residents are temporary workers (the report: most stay 3-4
years) and Dhivehi is not their home language.

Counts are usual residents (MG11), 515,132, the same total as the enumerated population; Table
6.1's Male and atoll totals are usual residents too (52,799 and 79,694, asserted).
"""
import os
import sys
import urllib.request
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "taxonomy"))
from rdlink import RD_GEO  # noqa: E402

RAW = HERE / "data" / "raw" / "mv"
OUT = HERE / "data" / "normalized" / "mv.csv"
WAYBACK = {
    "MG11.xlsx": "https://web.archive.org/web/20240523200828id_/https://census.gov.mv/2022/"
                 "wp-content/uploads/2023/08/MG11.xlsx",
    "Migration-Report-Census-2022.pdf": "https://web.archive.org/web/20240527005830id_/"
                 "https://census.gov.mv/2022/wp-content/uploads/2024/02/"
                 "Migration-Report-Census-2022.pdf",
}
TOTAL = 515_132
DHIVEHI = "indoeuropean.indoaryan.dhivehi"
# Table 6.1 (migration report p.73), usual residents: (Male, atolls). Transcribed; the script
# asserts the columns add to the printed totals and finds the row text in the PDF.
T61 = {"India": (11_421, 21_550), "Sri Lanka": (4_048, 7_290), "Bangladesh": (33_380, 41_435),
       "Phillippines": (639, 1_084), "Indonesia": (320, 1_981), "Nepal": (1_580, 2_642),
       "Others": (1_411, 3_712)}
T61_TOTAL = (52_799, 79_694)
ORIGIN = {"India": "in", "Sri Lanka": "lk", "Bangladesh": "bd", "Phillippines": "ph",
          "Indonesia": "id", "Nepal": "np"}
MIN_SHARE = 0.01
ATOLL = {"Maale": "MV021", "HA": "MV001", "HDh": "MV002", "Sh": "MV003", "N": "MV004",
         "Lh": "MV005", "R": "MV006", "B": "MV007", "AA": "MV008", "K": "MV009", "V": "MV010",
         "F": "MV011", "Dh": "MV012", "M": "MV013", "Th": "MV014", "L": "MV015",
         "GA": "MV016", "GDh": "MV017", "Gn": "MV018", "S": "MV019", "ADh": "MV020"}
# the administrative letters against religiondots' COD names (its mv_lookup.csv)
NAME = {"MV001": "Haa Alifu", "MV002": "Haa Dhaalu", "MV003": "Shaviyani", "MV004": "Noonu",
        "MV005": "Lhaviyani", "MV006": "Raa", "MV007": "Baa", "MV008": "Alifu Alifu",
        "MV009": "Kaafu", "MV010": "Vaavu", "MV011": "Faafu", "MV012": "Dhaalu",
        "MV013": "Meemu", "MV014": "Thaa", "MV015": "Laamu", "MV016": "Gaafu Alifu",
        "MV017": "Gaafu Dhaalu", "MV018": "Gnaviyani", "MV019": "Seenu", "MV020": "Alifu Dhaalu",
        "MV021": "Male"}


def fetch():
    RAW.mkdir(parents=True, exist_ok=True)
    for f, url in WAYBACK.items():
        if not (RAW / f).exists():
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            (RAW / f).write_bytes(urllib.request.urlopen(req, timeout=300).read())


def mg11():
    import openpyxl
    ws = openpyxl.load_workbook(RAW / "MG11.xlsx", read_only=True, data_only=True).active
    out = {}
    for r in ws.iter_rows(values_only=True):
        if r and isinstance(r[0], str) and r[0].strip() in ATOLL | {"Republic": 0}:
            out[r[0].strip()] = (int(r[1]), int(r[2]), int(r[3]))   # total, Maldivian, foreign
    return out


def check_t61():
    import fitz
    t = " ".join(p.get_text() for p in fitz.open(RAW / "Migration-Report-Census-2022.pdf"))
    t = " ".join(t.split())
    for k, (a, b) in T61.items():
        s = f"{k} {a + b:,} {a:,} {b:,}"
        assert s in t, f"Table 6.1 row not found: {s!r}"
    assert tuple(sum(v[i] for v in T61.values()) for i in (0, 1)) == T61_TOTAL


def home_mix(cc):
    from countries import load_one
    df = load_one(cc)["counts"]()
    s = df.groupby("node")["count"].sum()
    s = s[s > 0] / s.sum()
    s = s[s >= MIN_SHARE]
    return (s / s.sum()).to_dict()


def main():
    fetch()
    check_t61()
    t = mg11()
    assert t["Republic"][0] == TOTAL, t["Republic"]
    units = [k for k in t if k != "Republic"]
    assert sorted(ATOLL[k] for k in units) == sorted(NAME), units
    assert sum(t[k][0] for k in units) == TOTAL
    for k in units:
        assert t[k][1] + t[k][2] == t[k][0], k
    assert t["Maale"][2] == T61_TOTAL[0]
    assert sum(t[k][2] for k in units if k != "Maale") == T61_TOTAL[1]
    lut = pd.read_csv(RD_GEO / "mv" / "mv_lookup.csv", dtype=str)
    assert dict(zip(lut["unit"], lut["name"])) == NAME

    mixes = {k: home_mix(cc) for k, cc in ORIGIN.items()}
    mixes["Others"] = {"other": 1.0}
    rows = []
    for k in units:
        u = ATOLL[k]
        col = 0 if k == "Maale" else 1
        nat = {n: v[col] / T61_TOTAL[col] for n, v in T61.items()}
        langs = {DHIVEHI: float(t[k][1])}
        for n, share in nat.items():
            for node, f in mixes[n].items():
                langs[node] = langs.get(node, 0.0) + t[k][2] * share * f
        s = pd.Series(langs)
        c = s.astype(int)
        c[(s - c).sort_values(ascending=False).index[:t[k][0] - int(c.sum())]] += 1
        assert int(c.sum()) == t[k][0]
        for node, v in c.items():
            if v > 0:
                rows.append(dict(geo_id=u, geo_level="atoll", geo_name=NAME[u],
                                 source_category=node, count=int(v), tier="derived",
                                 source_id="mv_census2022_mg11_t61", year=2022))
    df = pd.DataFrame(rows)
    assert int(df["count"].sum()) == TOTAL and df["geo_id"].nunique() == 21
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, encoding="utf-8")
    nat = df.groupby("source_category")["count"].sum().sort_values(ascending=False)
    print(f"wrote {OUT}: 21 units, {len(nat)} languages, {nat.sum():,} people")
    for k, v in nat.head(15).items():
        print(f"  {k:<44} {v:>8,}  {v / nat.sum():.2%}")


if __name__ == "__main__":
    main()
