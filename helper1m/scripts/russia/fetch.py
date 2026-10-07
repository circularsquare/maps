"""Russia populations for helper1m.

Sources (all Rosstat, all fetched through the Wayback Machine because
rosstat.gov.ru presents a TLS chain no Western trust store verifies):

  Chisl_MO_01-01-2025.xlsx, Chisl_MO_01-01-2024.xlsx
      "Численность постоянного населения Российской Федерации по
      муниципальным образованиям на 1 января 2025 / 2024 года": every
      municipal district, municipal okrug and urban okrug (and every
      settlement under them), resident population on 1 January.
  Tom1_tab-5_VPN-2020.xlsx
      2021 census, volume 1 table 5: population by subject (and below, by
      2021 municipal names only, no codes; used here for subjects only).

Level 2 is the 1 January 2025 set of municipalities. Both years sit on that
set: 2024's units are carried onto it by crosswalk.py (renamed units by
name, merged ones summed, two boundary moves rebased). Inside Moscow and St
Petersburg level 2 is the 12 administrative okrugs and 18 districts.

Writes:
  helper1m/data/russia/units.csv        one row per 2025 municipality:
                                        oktmo, iso, name, pop2024, pop2025
  helper1m/data/russia/population.csv   code, level, year, pop  (needs
                                        boundaries/unit_map.csv from
                                        prep_boundaries.py; without it only
                                        units.csv is written)

INCLUDE_CRIMEA: Crimea and Sevastopol are left out by default, matching the
internationally recognised borders (and the geoBoundaries subject file).
Rosstat counts them; flip the switch to ship them. The four Ukrainian oblasts
Russia has claimed since 2022 are not in Rosstat's municipal tables at all.
"""
import csv
import re
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

import openpyxl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crosswalk  # noqa: E402
import rosstat_mo  # noqa: E402

INCLUDE_CRIMEA = False
CRIMEA_ISO = ("UA-43", "UA-40")

HELPER = Path(__file__).resolve().parents[2]
DATA = HELPER / "data" / "russia"
RAW = DATA / "raw"
UNIT_MAP = DATA / "boundaries" / "unit_map.csv"

WAYBACK = "https://web.archive.org/web/{ts}if_/https://rosstat.gov.ru/storage/mediabank/{name}"
FILES = {
    # local name: (wayback timestamp, name on rosstat.gov.ru; the leading С
    # of Сhisl is Cyrillic on the server)
    "Chisl_MO_01-01-2025.xlsx": ("20260906222243", "%D0%A1hisl_MO_01-01-2025.xlsx"),
    "Chisl_MO_01-01-2024.xlsx": ("20240428232801", "%D0%A1hisl_MO_01-01-2024.xlsx"),
    "Tom1_tab-5_VPN-2020.xlsx": ("20230725211953", "Tom1_tab-5_VPN-2020.xlsx"),
}
UA = "helper1m-research/1.0"


def download():
    RAW.mkdir(parents=True, exist_ok=True)
    for local, (ts, name) in FILES.items():
        path = RAW / local
        if path.exists():
            continue
        url = WAYBACK.format(ts=ts, name=name)
        print(f"downloading {local}", flush=True)
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=300) as r:
            body = r.read()
        if not body.startswith(b"PK"):
            raise SystemExit(f"{url} did not return an xlsx")
        path.write_bytes(body)


def norm_subject(name):
    n = name.lower().replace("ё", "е").replace("–", "-").replace("—", "-")
    n = re.sub(r"^г\.?\s+", "", n)
    n = re.sub(r"\s*-\s*(город федерального значения|городское население)\s*$", "", n)
    n = re.sub(r",?\s*включая.*$", "", n)
    n = re.sub(r"\s*\(кроме.*$", "", n)
    n = re.sub(r"\s+без\s+(ненецкого\s+)?автономн.*$", "", n)
    return re.sub(r"\s+", " ", n).strip()


def census_2021_subjects(names_by_iso):
    """{iso: 2021 census population} from Tom1 table 5. Subject rows are found
    by name, using the subject names of the 2025 municipal workbook; for
    Arkhangelsk and Tyumen the "без автономного округа" row is the one."""
    want = {norm_subject(n): iso for iso, n in names_by_iso.items()}
    wb = openpyxl.load_workbook(RAW / "Tom1_tab-5_VPN-2020.xlsx", read_only=True, data_only=True)
    out = {}
    for row in wb[wb.sheetnames[0]].iter_rows(values_only=True):
        if not row or not isinstance(row[0], str) or not isinstance(row[1], (int, float)):
            continue
        name = row[0].strip()
        iso = want.get(norm_subject(name))
        if iso is None:
            continue
        if iso in ("RU-ARK", "RU-TYU"):
            if re.search(r"\bбез\b", name):
                out[iso] = int(row[1])
        elif iso not in out:
            out[iso] = int(row[1])
    wb.close()
    missing = set(names_by_iso) - set(out)
    if missing:
        raise SystemExit(f"census table 5: no row for {sorted(missing)}")
    return out


def subject_names(path):
    """{iso: subject name} from the subject header rows of a workbook."""
    names = {}
    for digits, name, pop, raw in rosstat_mo.read_rows(path):
        if not digits:
            continue
        o = rosstat_mo.normalise(digits, None)
        if re.search(r"\bбез\b|\bкроме\b", name) and o[:2] in ("11", "71"):
            names[rosstat_mo.subject_of(o)] = name
        elif rosstat_mo.kind(o) == "subject":
            names.setdefault(rosstat_mo.subject_of(o), name)
    return names


def build_units():
    s25, u25, notes25 = rosstat_mo.parse(RAW / "Chisl_MO_01-01-2025.xlsx")
    s24, u24, notes24 = rosstat_mo.parse(RAW / "Chisl_MO_01-01-2024.xlsx")
    for tag, notes in (("2025", notes25), ("2024", notes24)):
        for n in notes:
            print(f"  {tag} code repair: {n}")
    for tag, s, u in (("2025", s25, u25), ("2024", s24, u24)):
        sums = defaultdict(int)
        for iso, name, pop in u.values():
            sums[iso] += pop
        bad = {iso: (s[iso], sums[iso]) for iso in s if s[iso] != sums[iso]}
        if bad:
            raise SystemExit(f"{tag}: units do not sum to their subject: {bad}")
        print(f"{tag}: {len(s)} subjects, {len(u)} level-2 units, "
              f"national {sum(s.values()):,} (with Crimea and Sevastopol)")

    links, report = crosswalk.link(u24, u25)
    old_values = {n: sum(u24[o][2] for o in links.get(n, [])) for n in u25}
    report += crosswalk.rebase(old_values, u25)
    merges = [r for r in report if not r.startswith("linked")]
    for r in merges:
        print("  crosswalk:", r)
    renamed = sum(1 for r in report if r.startswith("linked"))
    print(f"crosswalk 2024 -> 2025: {renamed} renamed or merged units linked; "
          f"{sum(1 for n in u25 if links.get(n) == [n])} unchanged")
    if any(r.startswith(("UNLINKED", "NO OLD")) for r in report):
        raise SystemExit("crosswalk left units without a partner; see above")

    names = subject_names(RAW / "Chisl_MO_01-01-2025.xlsx")
    census = census_2021_subjects(names)

    keep = lambda iso: INCLUDE_CRIMEA or iso not in CRIMEA_ISO  # noqa: E731
    rows = [(o, iso, name, old_values[o], pop) for o, (iso, name, pop) in sorted(u25.items())
            if keep(iso)]
    DATA.mkdir(parents=True, exist_ok=True)
    with (DATA / "units.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["oktmo", "iso", "name", "pop2024", "pop2025"])
        w.writerows(rows)
    # Old code -> new code, so prep_boundaries.py can use a 2024 OKTMO that
    # OSM or Wikidata still carries.
    with (DATA / "links_2024.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["old", "new"])
        w.writerows(sorted((o, n) for n, olds in links.items() for o in olds))
    subj = {iso: {2021: census[iso], 2024: s24[iso], 2025: s25[iso]} for iso in s25 if keep(iso)}
    with (DATA / "subjects.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["iso", "name", "census2021", "pop2024", "pop2025"])
        for iso in sorted(subj):
            w.writerow([iso, names[iso], subj[iso][2021], subj[iso][2024], subj[iso][2025]])
    for y in (2021, 2024, 2025):
        print(f"  {y}: {len(subj)} subjects, {sum(v[y] for v in subj.values()):,}")
    print(f"wrote {len(rows)} units -> {DATA / 'units.csv'}")
    return rows, subj


def write_population(rows=None, subj=None):
    """population.csv from units.csv + the unit map prep_boundaries.py wrote
    (Rosstat OKTMO -> boundary code; several units may share one polygon)."""
    if rows is None:
        with (DATA / "units.csv").open(encoding="utf-8") as f:
            rows = [(r["oktmo"], r["iso"], r["name"], int(r["pop2024"]), int(r["pop2025"]))
                    for r in csv.DictReader(f)]
        with (DATA / "subjects.csv").open(encoding="utf-8") as f:
            subj = {r["iso"]: {2021: int(r["census2021"]), 2024: int(r["pop2024"]),
                               2025: int(r["pop2025"])} for r in csv.DictReader(f)}
    if not UNIT_MAP.exists():
        print(f"no {UNIT_MAP.name} yet: run prep_boundaries.py, then this again")
        return
    with UNIT_MAP.open(encoding="utf-8") as f:
        umap = {r["oktmo"]: (r["code"], r["iso"]) for r in csv.DictReader(f)}
    l2 = defaultdict(lambda: defaultdict(int))
    l1 = defaultdict(lambda: defaultdict(int))
    unmapped = []
    for o, iso, name, p24, p25 in rows:
        if o not in umap:
            unmapped.append((o, iso, name, p25))
            continue
        code, giso = umap[o]
        for y, p in ((2024, p24), (2025, p25)):
            l2[code][y] += p
            l1[giso][y] += p
    if unmapped:
        print(f"WARNING: {len(unmapped)} Rosstat units ({sum(u[3] for u in unmapped):,} people "
              f"in 2025) have no polygon; first: {unmapped[:10]}")
    out = []
    for code, by in l2.items():
        out += [(code, 2, y, p) for y, p in by.items()]
    for iso, by in l1.items():
        out += [(iso, 1, y, p) for y, p in by.items()]
        out.append((iso, 1, 2021, subj[iso][2021]))
        for y in (2024, 2025):
            if by[y] != subj[iso][y]:
                print(f"  level 1 {iso} {y}: units {by[y]:,} vs Rosstat {subj[iso][y]:,}")
    out.sort(key=lambda r: (r[1], r[0], r[2]))
    with (DATA / "population.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["code", "level", "year", "pop"])
        w.writerows(out)
    print(f"wrote {len(out)} rows -> {DATA / 'population.csv'} "
          f"({len(l1)} subjects, {len(l2)} level-2 polygons)")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    download()
    rows, subj = build_units()
    write_population(rows, subj)


if __name__ == "__main__":
    main()
