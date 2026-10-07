"""Uzbekistan populations for helper1m.

Writes:
  data/uzbekistan/population.csv   code,level,year,pop
  data/uzbekistan/units.csv        code,level,name,name_uz,parent   (read by prep_boundaries.py)
  data/uzbekistan/events.csv       every boundary reconciliation move, for the record

Source: National Statistics Committee, SIAT indicator 2.01.02.0001 "Permanent
population - total", thousands, 1 January of each year, 2010-2026, country +
14 regions + 206 districts and cities, fetched from
  https://api.siat.stat.uz/media/uploads/sdmx/sdmx_data_246.json
(no key). Cached under data/uzbekistan/raw/.

Boundaries are OCHA COD-AB 2018b (199 districts). SIAT codes are SOATO, and the
COD pcode is the same number with "UZ" in place of the leading "17", so the join
is by code. Seven SIAT units post-date the COD file and a handful of later
transfers moved people between COD units; EVENTS below folds them back onto the
2018 polygons (see README.md).

CENSUS_LEVEL: when True, every SIAT year of a region's districts is multiplied by
that region's 2026-census / SIAT-1-Jan-2026 ratio, so the level is the census's
and the year-to-year trend stays SIAT's.
"""
import csv
import json
import statistics
import sys
import urllib.request
from pathlib import Path

CENSUS_LEVEL = True

HELPER = Path(__file__).resolve().parents[2]
OUT = HELPER / "data" / "uzbekistan"
RAW = OUT / "raw"
SIAT_URL = "https://api.siat.stat.uz/media/uploads/sdmx/sdmx_data_246.json"
SIAT_FILE = RAW / "sdmx_data_246.json"

# 2010 sits on a different basis: the country rises 1.12 million from 2010 to
# 2011 against 0.4-0.75 million in every other year, and every district jumps
# with it. Start at 2011.
YEARS = list(range(2011, 2027))

# 2026 census, preliminary results, census moment 15 January 2026.
# "Preliminary Results of the Population and Agriculture Census of the Republic
# of Uzbekistan, 2026", National Statistics Committee, English edition, printed
# p. 23 (PDF p. 25), "Distribution of the population by sex, by region".
# https://stat.uz/img/news/english_natija_merged-2_p42445.pdf
CENSUS_2026 = {
    "1735": 2149932,  # Republic of Karakalpakstan
    "1703": 3531777,  # Andijan
    "1706": 2104874,  # Bukhara
    "1708": 1524055,  # Jizzakh
    "1710": 3692323,  # Kashkadarya
    "1712": 1184591,  # Navoi
    "1714": 3149161,  # Namangan
    "1718": 4404575,  # Samarkand
    "1722": 2984084,  # Surkhandarya
    "1724": 954361,   # Syrdarya
    "1727": 3763093,  # Tashkent region
    "1730": 4238911,  # Fergana
    "1733": 2140746,  # Khorezm
    "1726": 3224838,  # Tashkent city
}
CENSUS_NATIONAL = 39047321

# Boundary reconciliation onto the COD 2018b polygons. Each event is the year
# label in which SIAT first shows the new arrangement and the units whose series
# step that year. For each unit the step is its change that year minus its usual
# change (median of the changes three years either side). Units that rose hand
# back their rise, units that fell get it back in proportion to their fall, and
# the moved amount follows the receiving unit's own later growth. A unit the COD
# file does not have ("new") is handed back whole in every year.
#   basis "receivers": the amount moved is what the receivers gained
#   basis "donors":    the amount moved is what the donors lost (used where the
#                      receivers also grew for other reasons)
#   basis "both":      the mean of the two, so a region total is conserved
EVENTS = [
    dict(year=2018, basis="receivers", note="Takhiatash district re-formed out of Khojeyli",
         units=["UZ35228", "UZ35236"]),
    dict(year=2020, basis="receivers", note="Bozatau district formed out of Kegeyli and Chimbay",
         units=["UZ35209", "UZ35212", "UZ35240"]),
    dict(year=2020, basis="receivers", note="Gazgan city formed out of Nurota; Navoi city takes part of Karmana",
         units=["UZ12412", "UZ12238", "UZ12401", "UZ12234"]),
    dict(year=2020, basis="receivers",
         note="Bandikhan district formed out of Kizirik, Kumkurgan and Baysun; Termez city takes part of Termez district",
         units=["UZ22203", "UZ22215", "UZ22214", "UZ22204", "UZ22401", "UZ22220"]),
    dict(year=2021, basis="receivers", note="Tuprakkala district formed out of Khazarasp",
         units=["UZ33221", "UZ33220"]),
    dict(year=2021, basis="receivers",
         note="Yangihayot district (Tashkent city) formed, mostly out of Sergeli, partly out of Tashkent region",
         units=["UZ26292", "UZ26283", "UZ26264", "UZ27237", "UZ27253"]),
    dict(year=2022, basis="donors",
         note="Tashkent city takes land from Zangiata, Kibray and Urtachirchik (Tashkent region)",
         units=["UZ27237", "UZ27248", "UZ27253", "UZ26264", "UZ26269", "UZ26290", "UZ26283"]),
    dict(year=2022, basis="both",
         note="Tashkent region internal: Yangiyul city out of Yangiyul district; Angren city to Akhangaran district and Parkent; Yukorichirchik",
         units=["UZ27424", "UZ27259", "UZ27407", "UZ27212", "UZ27249", "UZ27239"]),
    dict(year=2023, basis="receivers", note="Kukdala district formed out of Chirakchi",
         units=["UZ10240", "UZ10242"]),
    dict(year=2023, basis="both",
         note="Kokand and Fergana cities take land from Uzbekistan, Dangara, Fergana and Uchkuprik districts",
         units=["UZ30405", "UZ30401", "UZ30230", "UZ30236", "UZ30233", "UZ30221"]),
]

# Steps at or before the COD vintage that changed a unit's territory: the years
# before them are on another basis, so the district rows start here. The region
# rows keep every year (the region's territory did not change).
BASIS_START = {}
for c in ["UZ10405", "UZ10245",            # Shahrisabz city out of Shahrisabz district, 2018
          "UZ33406", "UZ33226",            # Khiva city out of Khiva district
          "UZ27401", "UZ27253",            # Nurafshon city out of Urtachirchik
          "UZ27415", "UZ27212",            # Akhangaran city out of Akhangaran district
          "UZ27424", "UZ27259",            # Yangiyul city out of Yangiyul district
          "UZ27265", "UZ27237",            # Tashkent district out of Zangiata
          "UZ35236",                       # Khojeyli (Takhiatash taken out in 2013, back in 2018)
          "UZ08218", "UZ08220"]:           # Zomin to Zarbdar
    BASIS_START[c] = 2018
for c in ["UZ14401", "UZ14212", "UZ14229"]:  # Namangan city takes from Namangan and Uychi districts
    BASIS_START[c] = 2017
for c in ["UZ18401", "UZ18233"]:             # Samarkand city takes from Samarkand district
    BASIS_START[c] = 2012

ALL_YEARS = list(range(2010, 2027))

NAME_FIX = {
    "UZ10207": "Guzar district",       # SIAT's English says "Gissar"; Uzbek is G'uzor
    "UZ03408": "Khanabad city",
    "UZ18406": "Kattakurgan city",
    "UZ03211": "Jalakuduk district",   # SIAT's English has a Cyrillic a in it
}


def load_siat():
    RAW.mkdir(parents=True, exist_ok=True)
    if not SIAT_FILE.exists() or "--refresh" in sys.argv:
        req = urllib.request.Request(SIAT_URL, headers={"User-Agent": "Mozilla/5.0"})
        SIAT_FILE.write_bytes(urllib.request.urlopen(req, timeout=60).read())
    blob = json.loads(SIAT_FILE.read_text(encoding="utf-8"))[0]
    meta = {m["name_en"]: m["value_en"] for m in blob["metadata"]}
    return blob["data"], meta


def trend(s, y):
    ch = [s[t] - s[t - 1] for t in range(y - 3, y + 4)
          if t != y and t - 1 in s and t in s and s[t - 1] > 0 and s[t] > 0]
    return statistics.median(ch)


def apply_event(work, ev, cod_codes, log):
    y0 = ev["year"]
    exc = {}
    for u in ev["units"]:
        s = work[u]
        if u not in cod_codes:
            assert s.get(y0 - 1, 0) == 0, (u, "new unit already populated before", y0)
            exc[u] = s[y0]
        else:
            exc[u] = s[y0] - s[y0 - 1] - trend(s, y0)
    recv = {u: e for u, e in exc.items() if e > 0}
    don = {u: -e for u, e in exc.items() if e < 0}
    new = [u for u in recv if u not in cod_codes]
    p, n = sum(recv.values()), sum(don.values())
    m = {"receivers": p, "donors": n, "both": (p + n) / 2}[ev["basis"]]
    # New units go whole; existing receivers share what is left of m.
    p_new = sum(recv[u] for u in new)
    p_old = p - p_new
    k = max(m - p_new, 0) / p_old if p_old else 0
    take0 = {u: (recv[u] if u in new else recv[u] * k) for u in recv}
    base = {u: work[u][y0] for u in recv}
    for y in ALL_YEARS:
        if y < y0:
            continue
        moved = 0.0
        for u in recv:
            t = work[u][y] if u in new else take0[u] * work[u][y] / base[u]
            work[u][y] -= t
            moved += t
        for u in don:
            work[u][y] += moved * don[u] / n
    for u in ev["units"]:
        log.append(dict(year=y0, unit=u, step_thousands=round(exc[u], 1),
                        role="new, dissolved" if u in new else ("gives back" if u in recv else "gets back"),
                        moved_at_event_thousands=round(take0.get(u, 0) if u in recv else m * don.get(u, 0) / n if n else 0, 1),
                        note=ev["note"]))
    for u in new:
        assert all(abs(work[u][y]) < 1e-9 for y in ALL_YEARS), u
        del work[u]


def cod_codes_from_shapefile():
    import pyogrio
    shp = HELPER.parent / "data" / "asia1m" / "uzbekistan" / "uzb_admbnda_adm2_2018b.shp"
    df = pyogrio.read_dataframe(shp, read_geometry=False)
    return set(df["ADM2_PCODE"]), dict(zip(df["ADM2_PCODE"], df["ADM1_PCODE"]))


def clean_en(code, s):
    if code in NAME_FIX:
        return NAME_FIX[code]
    return " ".join(s.split())


def build(census_level=CENSUS_LEVEL, write=True, verbose=True):
    """Returns {(level, code): {year: pop}}; writes the CSVs when write is set."""
    rows, meta = load_siat()
    if verbose:
        print(f"SIAT {meta['Indicator identification number (code)']}, last modified {meta['Last modified date']}, {len(rows)} rows")
    cod, cod_parent = cod_codes_from_shapefile()

    regions = {r["Code"]: r for r in rows if len(r["Code"]) == 4 and r["Code"] != "1700"}
    national = [r for r in rows if r["Code"] == "1700"][0]
    assert set(regions) == set(CENSUS_2026)
    assert sum(CENSUS_2026.values()) == CENSUS_NATIONAL

    ratio = {}
    for rc, r in regions.items():
        ratio[rc] = CENSUS_2026[rc] / (r["2026"] * 1000) if census_level else 1.0

    work, units = {}, []
    for r in rows:
        c = r["Code"]
        if len(c) != 7:
            continue
        u = "UZ" + c[2:]
        work[u] = {y: (r.get(str(y)) or 0.0) * ratio[c[:4]] for y in ALL_YEARS}
        units.append((u, clean_en(u, r["Klassifikator_en"]), " ".join(r["Klassifikator"].split())))

    # Every COD unit must be a SIAT unit.
    missing = cod - set(work)
    assert not missing, missing

    log = []
    for ev in EVENTS:
        apply_event(work, ev, cod, log)
    assert set(work) == cod, (set(work) ^ cod)

    # Names: the units folded into a COD polygon are named in brackets.
    folded = {"UZ35236": "Takhiatash", "UZ35212": "most of Bozatau", "UZ12238": "Gazgan city",
              "UZ22215": "most of Bandikhan", "UZ33220": "Tuprakkala", "UZ26283": "most of Yangihayot",
              "UZ10242": "Kukdala"}

    out_rows = []
    adm1 = {}
    for u in sorted(work):
        start = max(BASIS_START.get(u, YEARS[0]), YEARS[0])
        for y in YEARS:
            v = round(work[u][y] * 1000)
            adm1.setdefault(cod_parent[u], {}).setdefault(y, 0)
            adm1[cod_parent[u]][y] += v
            if y >= start:
                out_rows.append((u, 2, y, v))
    for a in sorted(adm1):
        for y in YEARS:
            out_rows.append((a, 1, y, adm1[a][y]))

    result = {(lvl, c): {} for c, lvl, _, _ in out_rows}
    for c, lvl, y, v in out_rows:
        result[(lvl, c)][y] = v
    if verbose:
        print(f"CENSUS_LEVEL = {census_level}")
        print("region  SIAT 1 Jan 2026   census 15 Jan 2026   ratio")
        for rc, r in sorted(regions.items(), key=lambda kv: kv[1]["Klassifikator_en"]):
            print(f"  {rc} {r['Klassifikator_en'][:26]:26} {r['2026'] * 1000:12,.0f} {CENSUS_2026[rc]:14,} {CENSUS_2026[rc] / (r['2026'] * 1000):8.3f}")
        print(f"  national {national['2026'] * 1000:,.0f} vs census {CENSUS_NATIONAL:,}  ratio {CENSUS_NATIONAL / (national['2026'] * 1000):.3f}")
        print("national total by year (ours / SIAT):")
        for y in YEARS:
            tot = sum(adm1[a][y] for a in adm1)
            print(f"  {y} {tot:12,} {national[str(y)] * 1000:12,.0f} {tot / (national[str(y)] * 1000):.4f}")
        n2 = sum(1 for r in out_rows if r[1] == 2)
        print(f"{len(out_rows)} rows ({n2} district rows, {len(cod)} districts, {len(adm1)} regions)")
    if not write:
        return result

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "population.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["code", "level", "year", "pop"])
        w.writerows(out_rows)

    with open(OUT / "units.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["code", "level", "name", "name_uz", "parent"])
        for rc, r in sorted(regions.items()):
            w.writerow(["UZ" + rc[2:], 1, " ".join(r["Klassifikator_en"].split()),
                        " ".join(r["Klassifikator"].split()), ""])
        for u, en, uz in sorted(units):
            if u not in cod:
                continue
            if u in folded:
                en = f"{en} (incl. {folded[u]})"
            w.writerow([u, 2, en, uz, cod_parent[u]])

    with open(OUT / "events.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(log[0]))
        w.writeheader()
        w.writerows(log)
    if verbose:
        print(f"wrote {OUT / 'population.csv'}")
    return result


if __name__ == "__main__":
    build()
