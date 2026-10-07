"""Cyprus, Census of Population and Housing 2021 (CYSTAT), language by district.

    python sources/cy_census.py --fetch    four small JSON cubes from cystatdb.cystat.gov.cy
    python sources/cy_census.py            normalise from data/raw/cy/

Writes
  data/normalized/cy.csv        language x district, 5 districts, 33 languages + Not stated
                                (table 1891616E). What counts() draws.
  data/normalized/cy_place.csv  per municipality/community and per 1891616E label, the number
                                of speakers that community would hold if, inside its district,
                                each citizenship group spoke as the district's group does
                                (1891613E x 1891213E). A placement weight only: it sums back to
                                the district counts exactly, and countries/cy.py uses it to place
                                a district's dots, never to change them.

CYSTAT-DB publishes the 2021 language question three ways, all by district and nothing finer:
1891616E (33 named languages), 1891610E (20 named, x urban/rural) and 1891613E (20 named, x
citizenship group). The table note says "the language that the respondent stated that s/he
speaks best"; the questionnaire's Q11(b) asks "What is ...'s native language?" (sources/cy.md).

Checks, all asserted:
  * every table's total is 923,381, the census population, and the districts sum to it
  * 1891610E and 1891613E agree with 1891616E on all 20 shared languages in every district
  * 1891613E's `Other languages` equals 1891616E's 12 extra languages plus its own `Other
    languages`, in every district (so the extra 12 are a split of that row and nothing else)
  * 1891613E's citizenship groups per district equal 1891213E's communities summed by district
  * cy_place.csv sums back to cy.csv for every (district, label)

API: CYSTAT-DB's PxWeb JSON API is at /api/v1/, not /pxweb/api/v1/ (religiondots' sources/cy.py
found that out). Generic browser User-Agent only.
"""
import argparse
import csv
import itertools
import json
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "cy")
OUT = os.path.join(ROOT, "data", "normalized", "cy.csv")
OUT_PLACE = os.path.join(ROOT, "data", "normalized", "cy_place.csv")

API = "https://cystatdb.cystat.gov.cy/api/v1/en/8.CYSTAT-DB"
CENSUS = "Population/Census of Population and Housing 2021/Population"
LRE = f"{CENSUS}/Population - Language, Religion, Ethnic Religious Group"
CCB = f"{CENSUS}/Population - Country of Citizenship, Country of Birth"
TABLES = {
    "lang_district.json": f"{LRE}/1891616E.px",     # language x district x sex, 33 languages
    "lang_urban.json":    f"{LRE}/1891610E.px",     # language x district x urban/rural x sex
    "lang_citizen.json":  f"{LRE}/1891613E.px",     # language x citizenship group x district
    "cit_comm.json":      f"{CCB}/1891213E.px",     # citizenship group x community x sex
}
NATIONAL_TOTAL = 923381
YEAR = 2021

# The language tables spell the districts by name (one misspelt `Lekfosia`); the community
# table codes them 1, 3, 4, 5, 6, which are the first digit of every LAU code. Keryneia (2) is
# not enumerated. Mapped explicitly so a relabelled edition fails here.
DISTRICT = {"Lefkosia": "1", "Lekfosia": "1", "Ammochostos": "3", "Larnaka": "4",
            "Lemesos": "5", "Pafos": "6"}
DISTRICT_NAME = {"1": "Lefkosia", "3": "Ammochostos", "4": "Larnaka", "5": "Lemesos",
                 "6": "Pafos"}
# 1891213E spells the groups differently from 1891613E
GROUPS = {"Cypriots *": "Cypriots (2)",
          "Other European Union citizens": "Other European Union citizens",
          "Non European Union citizens": "Non-European Union citizens",
          "Not stated": "Not stated"}


def fetch():
    import requests
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    os.makedirs(RAW, exist_ok=True)
    ua = {"User-Agent": "Mozilla/5.0"}
    for name, path in TABLES.items():
        dest = os.path.join(RAW, name)
        if os.path.exists(dest) and os.path.getsize(dest) > 2_000:
            print("already have", dest)
            continue
        url = f"{API}/{path}"
        meta = requests.get(url, timeout=120, verify=False, headers=ua)
        meta.raise_for_status()
        codes = [v["code"] for v in meta.json()["variables"]]
        query = {"query": [{"code": c, "selection": {"filter": "all", "values": ["*"]}}
                           for c in codes],
                 "response": {"format": "json-stat2"}}
        r = requests.post(url, json=query, timeout=300, verify=False, headers=ua)
        r.raise_for_status()
        tmp = dest + ".part"
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(r.text)
        os.replace(tmp, dest)
        print(f"  {name}: {os.path.getsize(dest):,} bytes")


def _load(name):
    p = os.path.join(RAW, name)
    if not os.path.exists(p):
        raise SystemExit(f"missing {p}: run with --fetch first")
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def _order(js, dim):
    cat = js["dimension"][dim]["category"]
    idx = cat["index"]
    keys = sorted(idx, key=lambda k: idx[k]) if isinstance(idx, dict) else list(idx)
    return [(k, cat["label"][k]) for k in keys]


def _cells(js):
    """json-stat2: one flat row-major value array over the dimensions in `id`."""
    ids = js["id"]
    dims = [_order(js, d) for d in ids]
    vals = js["value"]
    if len(vals) != len(list(itertools.product(*dims))):
        raise SystemExit("json-stat cube size does not match its dimensions")
    for i, combo in enumerate(itertools.product(*dims)):
        yield {d: lab for d, (_, lab) in zip(ids, combo)}, \
              {d: code for d, (code, _) in zip(ids, combo)}, vals[i] or 0


def build():
    # ---- 1891616E: the drawn table
    t16 = {}
    for lab, _, v in _cells(_load("lang_district.json")):
        if lab["SEX"] == "Total":
            t16[(lab["LANGUAGE (1)"], lab["DISTRICT"])] = v
    langs16 = [l for _, l in _order(_load("lang_district.json"), "LANGUAGE (1)") if l != "Total"]
    if t16[("Total", "Total")] != NATIONAL_TOTAL:
        raise SystemExit(f"1891616E total {t16[('Total', 'Total')]:,}")
    dists = [d for d in {d for _, d in t16} if d != "Total"]
    for d in dists:
        s = sum(t16[(l, d)] for l in langs16)
        assert s == t16[("Total", d)], f"1891616E {d}: languages sum {s} != {t16[('Total', d)]}"
    assert sum(t16[("Total", d)] for d in dists) == NATIONAL_TOTAL
    for l in langs16:
        assert sum(t16[(l, d)] for d in dists) == t16[(l, "Total")], l
    print(f"1891616E: {len(langs16)} categories x {len(dists)} districts, total "
          f"{NATIONAL_TOTAL:,}")

    # ---- 1891610E: same languages x urban/rural
    t10 = {}
    for lab, _, v in _cells(_load("lang_urban.json")):
        if lab["SEX"] == "Total" and lab["URBAN, RURAL"] == "Total":
            t10[(lab["LANGUAGE (1)"], lab["DISTRICT"])] = v
    # ---- 1891613E: same languages x citizenship group
    t13 = {}
    for lab, _, v in _cells(_load("lang_citizen.json")):
        if lab["SEX"] == "Total":
            t13[(lab["LANGUAGE (1)"], lab["CITIZENSHIP GROUP"], lab["DISTRICT"])] = v
    langs13 = [l for _, l in _order(_load("lang_citizen.json"), "LANGUAGE (1)") if l != "Total"]
    groups = [g for _, g in _order(_load("lang_citizen.json"), "CITIZENSHIP GROUP")
              if g != "Total"]
    assert sorted(groups) == sorted(GROUPS.values()), groups
    shared = [l for l in langs13 if l != "Other languages"]
    assert all(l in langs16 for l in shared), [l for l in shared if l not in langs16]
    extra = [l for l in langs16 if l not in langs13]
    print(f"1891610E/1891613E: {len(langs13)} categories; 1891616E adds {len(extra)}: "
          + ", ".join(extra))
    norm = lambda d: "Lefkosia" if d == "Lekfosia" else d  # noqa: E731
    for d in dists + ["Total"]:
        for l in shared + ["Total"]:
            a = t16[(l, d)]
            assert t10[(l, norm(d))] == a, f"1891610E {l} {d}: {t10[(l, norm(d))]} vs {a}"
            assert t13[(l, "Total", norm(d))] == a, f"1891613E {l} {d}"
        oth13 = t13[("Other languages", "Total", norm(d))]
        oth16 = sum(t16[(l, d)] for l in extra) + t16[("Other languages", d)]
        assert oth13 == oth16 and t10[("Other languages", norm(d))] == oth16, \
            f"{d}: 1891613E other {oth13} vs 1891616E extra+other {oth16}"
    print("  all 20 shared languages agree in all three tables in every district; "
          "1891613E's Other languages = 1891616E's 12 extra + Other languages, every district")

    # ---- 1891213E: citizenship group x community
    comm, names = {}, {}
    for lab, code, v in _cells(_load("cit_comm.json")):
        if lab["SEX"] != "Total" or lab["CITIZENSHIP GROUP"] == "Total":
            continue
        c = code["DISTRICT, MUNICIPALITY/COMMUNITY"]
        if c == "TOTAL" or len(c) == 1:
            continue
        names[c] = lab["DISTRICT, MUNICIPALITY/COMMUNITY"]
        comm.setdefault(c, {})[GROUPS[lab["CITIZENSHIP GROUP"]]] = v
    assert len(comm) == 396, len(comm)
    assert sum(sum(g.values()) for g in comm.values()) == NATIONAL_TOTAL
    for d in dists:
        dc = DISTRICT[d]
        for g in groups:
            s = sum(comm[c][g] for c in comm if c[0] == dc)
            assert s == t13[("Total", g, norm(d))], f"{d} {g}: communities {s} vs " \
                                                    f"1891613E {t13[('Total', g, norm(d))]}"
    print(f"1891213E: 396 communities; per district and citizenship group they sum to "
          f"1891613E exactly")

    # ---- write cy.csv
    rows = []
    for d in sorted(dists, key=lambda d: DISTRICT[d]):
        dc = DISTRICT[d]
        for l in langs16:
            rows.append(dict(geo_id=dc, geo_level="district", geo_name=DISTRICT_NAME[dc],
                             source_category=l, count=t16[(l, d)], tier="measured",
                             year=YEAR, source_id="cystat_census2021_1891616E"))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT} ({len(rows)} rows)")

    # ---- placement weights: E(label, c) = sum_g P(label | g, district) N(g, c)
    prow = []
    for d in dists:
        dc = DISTRICT[d]
        D = norm(d)
        cs = sorted(c for c in comm if c[0] == dc)
        for l in langs16:
            if t16[(l, d)] == 0:
                continue
            if l in shared:     # 20 named languages and Not stated: their own profile
                prof = {g: t13[(l, g, D)] / t13[("Total", g, D)] if t13[("Total", g, D)] else 0
                        for g in groups}
                scale = 1.0
            else:   # one of the 12 extra: the Other languages profile, scaled to its count
                prof = {g: t13[("Other languages", g, D)] / t13[("Total", g, D)]
                        if t13[("Total", g, D)] else 0 for g in groups}
                scale = t16[(l, d)] / t13[("Other languages", "Total", D)]
            got = 0.0
            for c in cs:
                e = scale * sum(prof[g] * comm[c][g] for g in groups)
                got += e
                if e > 0:
                    prow.append(dict(unit=c, district=dc, geo_name=names[c],
                                     source_category=l, weight=round(e, 6)))
            assert abs(got - t16[(l, d)]) < 1e-6 * max(1, t16[(l, d)]) + 1e-6, \
                f"{d} {l}: weights sum {got} vs {t16[(l, d)]}"
    with open(OUT_PLACE, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(prow[0]))
        w.writeheader()
        w.writerows(prow)
    print(f"wrote {OUT_PLACE} ({len(prow):,} rows); every (district, label) sums back to "
          "1891616E")

    print("\nnational:")
    for l in sorted(langs16, key=lambda l: -t16[(l, "Total")]):
        print(f"  {t16[(l, 'Total')]:>9,}  {l}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fetch", action="store_true")
    a = ap.parse_args()
    if a.fetch:
        fetch()
    build()


if __name__ == "__main__":
    main()
