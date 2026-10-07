"""Pakistan for helper1m: fetch, build boundaries, write population.csv.

    C:\\Python39\\python.exe helper1m\\scripts\\pakistan\\fetch.py            everything
    C:\\Python39\\python.exe helper1m\\scripts\\pakistan\\fetch.py --no-geo   skip prep_boundaries

Steps
  1. Download PBS 2023 census Table 1 (five PDFs), OSM admin 6/7 for Pakistan (Overpass), and
     the AJK / GB government booklets that republish the census for those two areas.
  2. Parse Table 1 -> data/pakistan/census_units.csv (591 tehsil-tier units, 2023 and 2017).
  3. prep_boundaries.py -> data/pakistan/boundaries/adm{1,2,3}.gpkg + crosswalk.csv.
  4. Sum census units onto the boundary units -> data/pakistan/population.csv
     (code, level, year, pop), years 2017 and 2023, every level an exact sum of the one below.
"""

import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ajk_gb
import osm
import table1

REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
DATA = os.path.join(REPO, "helper1m", "data", "pakistan")
RAW = os.path.join(DATA, "raw")

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"}
BOOKLETS = {
    # read for the record; the figures used are transcribed in ajk_gb.py
    "ajk_statistical_year_book_2025.pdf":
        "https://www.pndajk.gov.pk/uploadfiles/downloads/Statistical%20Year%20Book%202025.pdf",
    "ajk_at_a_glance_2025.pdf":
        "https://pndajk.gov.pk/uploadfiles/downloads/AJK%20At%20a%20Glance%202025.pdf",
    "gb_at_a_glance_2025.pdf":
        "https://pnd.gog.pk/storage/downloads/AiRIlDEcscWPC1s58oXIgpjlVAS7jd-metaR0IgQVQgR2xhbmNlIDIwMjUuMS5wZGY=-.pdf",
}

# Published headline figures, the checks' targets.
# 2023: PBS's announced provincial totals (incl. 1,041,342 counted by head only in restricted
# areas), which are Table 1's province rows. 2017: the 2017 census total, 207,684,626.
OFFICIAL_2023 = {"Punjab": 127_688_922, "Sindh": 55_696_147, "Khyber Pakhtunkhwa": 40_856_097,
                 "Balochistan": 14_894_402, "Islamabad": 2_363_863}
OFFICIAL_NATIONAL = {2023: 241_499_431, 2017: 207_684_626}


def download():
    table1.fetch(RAW)
    osm.fetch(RAW)
    for name, url in BOOKLETS.items():
        dest = os.path.join(RAW, name)
        if os.path.exists(dest) and os.path.getsize(dest) > 100_000:
            continue
        r = requests.get(url, headers=UA, timeout=600)
        r.raise_for_status()
        if r.content[:5] != b"%PDF-":
            raise SystemExit(f"{name}: not a PDF")
        open(dest, "wb").write(r.content)
        print(f"  fetched {name}")


def parse():
    units, errors = table1.read_all(RAW)
    for e in errors:
        print("  TABLE 1:", e)
    if errors:
        raise SystemExit(f"{len(errors)} Table 1 row checks failed")
    df = pd.DataFrame(units)
    # Islamabad's file is ISLAMABAD DISTRICT (the territory) and its one tehsil
    isl = df.province == "Islamabad"
    df.loc[isl & (df.level == "tehsil"), "district"] = "ISLAMABAD DISTRICT"
    row = df[isl & (df.level == "province")].iloc[0].to_dict()
    row.update(level="district", district="ISLAMABAD DISTRICT")
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    ok = True
    for prov, g in df.groupby("province"):
        p = g[g.level == "province"].iloc[0]
        d = g[g.level == "district"]
        t = g[g.level == "tehsil"]
        for col in ("pop23", "pop17"):
            for lab, part in (("districts", d), ("tehsils", t)):
                if part[col].sum() != p[col]:
                    ok = False
                    print(f"  BAD {prov}: {lab} {col} {part[col].sum():,} != {p[col]:,}")
        bad = [x for x in d.name if t[t.district == x][["pop23", "pop17"]].sum().tolist()
               != d[d.name == x][["pop23", "pop17"]].iloc[0].tolist()]
        if bad:
            ok = False
            print(f"  BAD {prov}: tehsils do not sum to districts {bad}")
        print(f"  {prov:20s} 2023 {p.pop23:>12,} (official {OFFICIAL_2023[prov]:,}) "
              f"2017 {p.pop17:>12,}; {len(d)} districts, {len(t)} tehsil-tier units, all sums exact")
        ok &= int(p.pop23) == OFFICIAL_2023[prov]
    nat23 = int(df[df.level == "province"].pop23.sum())
    nat17 = int(df[df.level == "province"].pop17.sum())
    print(f"  national 2023 {nat23:,} (official {OFFICIAL_NATIONAL[2023]:,}); "
          f"2017 {nat17:,} (official {OFFICIAL_NATIONAL[2017]:,})")
    ok &= nat23 == OFFICIAL_NATIONAL[2023] and nat17 == OFFICIAL_NATIONAL[2017]
    bad = ajk_gb.check()
    for b in bad:
        print("  BAD", b)
    ok &= not bad
    print(f"  Azad Kashmir {ajk_gb.AJK_TOTAL[2023]:,} / {ajk_gb.AJK_TOTAL[2017]:,}, "
          f"Gilgit-Baltistan {ajk_gb.GB_TOTAL[2023]:,} / {ajk_gb.GB_TOTAL[2017]:,} (2023 / 2017)")
    if not ok:
        raise SystemExit("census checks FAILED")
    df.to_csv(os.path.join(DATA, "census_units.csv"), index=False)
    return df


def population(df):
    cwdf = pd.read_csv(os.path.join(DATA, "crosswalk.csv"))
    t = df[df.level == "tehsil"][["province", "district", "name", "pop23", "pop17"]]
    a = cwdf[cwdf.source == "pbs_table1"].merge(t, left_on=["province", "district", "unit"],
                                                right_on=["province", "district", "name"], how="outer",
                                                indicator=True)
    if (a._merge != "both").any():
        print(a[a._merge != "both"][["province", "district", "unit", "name"]])
        raise SystemExit("crosswalk and census units disagree")
    extra = [{"adm3": r.adm3, "adm2": r.adm2, "adm1": r.adm1, "pop23": p23, "pop17": p17}
             for (name, d, cname, p23, p17), r in
             zip(ajk_gb.AJK_TEHSILS, cwdf[cwdf.source == "ajk_yearbook"].itertuples())]
    extra += [{"adm3": r.adm3, "adm2": r.adm2, "adm1": r.adm1, "pop23": p23, "pop17": p17}
              for (name, cods, p23, p17), r in
              zip(ajk_gb.GB_DISTRICTS, cwdf[cwdf.source == "gb_at_a_glance"].itertuples())]
    allu = pd.concat([a[["adm3", "adm2", "adm1", "pop23", "pop17"]], pd.DataFrame(extra)])
    rows = []
    for level, col in ((3, "adm3"), (2, "adm2"), (1, "adm1")):
        s = allu.groupby(col)[["pop23", "pop17"]].sum()
        for code, r in s.iterrows():
            rows.append({"code": code, "level": level, "year": 2017, "pop": int(r.pop17)})
            rows.append({"code": code, "level": level, "year": 2023, "pop": int(r.pop23)})
    out = pd.DataFrame(rows).sort_values(["level", "code", "year"])
    out.to_csv(os.path.join(DATA, "population.csv"), index=False)
    for level in (1, 2, 3):
        x = out[out.level == level]
        print(f"  level {level}: {x.code.nunique()} units, 2017 {x[x.year == 2017]['pop'].sum():,}, "
              f"2023 {x[x.year == 2023]['pop'].sum():,}")
    # district level against Table 1's own district rows
    d = df[df.level == "district"]
    cw_d = a.groupby(["province", "district"]).adm2.first()
    lv2 = out[(out.level == 2) & (out.year == 2023)].set_index("code")["pop"]
    mism = [(r.district, r.pop23, lv2.get(cw_d.get((r.province, r.name)))) for r in d.itertuples()
            if lv2.get(cw_d.get((r.province, r.name))) != r.pop23]
    print(f"  districts equal to Table 1's district rows: {len(d) - len(mism)} of {len(d)} {mism[:5]}")
    print(f"wrote {os.path.join(DATA, 'population.csv')} ({len(out)} rows)")


def main():
    os.makedirs(RAW, exist_ok=True)
    download()
    print("Table 1 and the AJK/GB figures:")
    df = parse()
    if "--no-geo" not in sys.argv:
        import prep_boundaries
        print("boundaries:")
        prep_boundaries.main()
    print("population.csv:")
    population(df)


if __name__ == "__main__":
    main()
