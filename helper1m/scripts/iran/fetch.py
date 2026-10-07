"""helper1m Iran: population.csv for provinces (31) and counties (429), plus districts (bakhsh)
when scripts/iran/bakhsh.py has run.

Years
  2011  1390 census (Aban 1390), carried settlement by settlement onto the 1395 counties
  2016  1395 census (Aban 1395), SCI's settlement workbooks, the counties' own rows
  2024  SCI's provincial estimate for 1403, each county moved forward from 2016 by its province,
        with half its 2011-2016 lead or lag over its province carried on (see forward_cast)

Run download.py first. Reads helper1m/data/iran/raw/, writes
  helper1m/data/iran/population.csv     code,level,year,pop
  helper1m/data/iran/counties.csv       one row per county: SCI code, COD pcode, names, all years
  helper1m/data/iran/carry_1390.csv     every 1390 settlement unit and where its people went

Codes: level 1 = COD-AB `adm1_pcode` (IR001 Alborz ... IR031 Zanjan, alphabetical by English
name); level 2 = COD-AB `adm2_pcode` (IR001001 ...). See README "Codes".
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "6")

import csv  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
from collections import defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402
import requests  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent))
import census  # noqa: E402
from census import fold  # noqa: E402

HELPER = Path(__file__).resolve().parents[2]
REPO = HELPER.parent
DATA = HELPER / "data" / "iran"
RAW = DATA / "raw"
COD1 = REPO / "data" / "asia1m" / "iran" / "irn_admin1.shp"
COD2 = REPO / "data" / "asia1m" / "iran" / "irn_admin2.shp"
PS_NAME = "irn_admpop_adm2_2016_v2.csv"
PS_URL = ("https://data.humdata.org/dataset/07f4ec78-42c7-4606-ae62-4f1bff918c45/resource/"
          "81700d7b-fbea-49ab-9972-c867231cb3c0/download/irn_admpop_adm2_2016_v2.csv")
EST_HTML = RAW / "sci" / "iod-06124.html"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) helper1m-build/1.0"}

PROVS = [f"{i:02d}" for i in range(31)]
YEAR90, YEAR95, YEAR_EST = 2011, 2016, 2024
# Published national totals (SCI): 1390 75,149,669; 1395 79,926,270; 1403 estimate 85,961,000.
NATIONAL = {YEAR90: 75_149_669, YEAR95: 79_926_270}
FORWARD_DAMP = 0.5  # share of a county's 2011-16 lead over its province carried to 2024


# 1390 cities whose land holds more than one 1395 city in different counties. Their 1390 people
# are shared over those 1395 cities by the 1395 counts. Found by listing every 1395 city of over
# 5,000 with no 1390 city or village of its name (scripts/iran/README.md, "The 1390 carry").
CITY_SPLITS = {
    # Fardis (181,174 in 1395) was part of Karaj city until 2013 and is its own county since;
    # the 1390 file has no Fardis, and Karaj's 1390 zones add up to Karaj + Fardis of 1395.
    ("30", fold("کرج")): [fold("کرج"), fold("فرديس")],
}
CITY_POP95 = {}


def city_key(name):
    """A city's name with a zone number stripped: the 1390 file lists a zoned city only as its
    zones ('اراك 1', 'تبريز4-', 'مشهد10'), the 1395 file as the city plus its zones."""
    return re.sub(r"[\d\-]+$", "", fold(name))


# ----------------------------------------------------------------------------- 1395
def load95():
    rows = {nn: census.read95(nn) for nn in PROVS}
    counties, bakhshs, cities, villages = {}, {}, defaultdict(list), defaultdict(list)
    prov = {}
    for nn, rr in rows.items():
        for r in rr:
            k = r["kind"]
            if k == "ostan":
                prov[nn] = (r["name"], r["pop"])
            elif k == "county":
                counties[(nn, r["cty"])] = {"name": r["name"], "pop": r["pop"]}
            elif k == "bakhsh":
                bakhshs[(nn, r["cty"], r["bkh"])] = {"name": r["name"], "pop": r["pop"]}
            elif k == "city":
                cities[(nn, city_key(r["name"]))].append((nn, r["cty"], r["bkh"]))
                CITY_POP95[(nn, city_key(r["name"]))] = r["pop"]
            elif k == "abadi":
                villages[r["abadi"]].append(((nn, r["cty"], r["bkh"]), r["name"]))
    # nesting: province = sum of counties; county >= sum of its bakhshs (nomads at county level)
    for nn, (_, p) in prov.items():
        s = sum(v["pop"] for (o, _), v in counties.items() if o == nn)
        assert s == p, f"1395 os{nn}: counties sum {s} != province {p}"
    bsum = defaultdict(int)
    for (o, c, _), v in bakhshs.items():
        bsum[(o, c)] += v["pop"]
    for k, v in counties.items():
        v["settled"] = bsum[k]
        v["nomad"] = v["pop"] - bsum[k]
        assert v["nomad"] >= 0, (k, v)
    total = sum(p for _, p in prov.values())
    assert total == NATIONAL[YEAR95], total
    print(f"1395: {len(prov)} provinces, {len(counties)} counties, {len(bakhshs)} districts, "
          f"national {total:,} (SCI's published total); non-settled at county level only "
          f"{sum(v['nomad'] for v in counties.values()):,}")
    return prov, counties, bakhshs, cities, villages


# ----------------------------------------------------------------------------- 1390
def carry90(c95, v95):
    """Spread every 1390 settlement unit over the 1395 districts (ost, cty, bkh) its people now
    sit in. Returns {1395 district: people} (float), {(ost, cty) 1395: nomads}, a log, totals."""
    out = defaultdict(float)
    nomads = defaultdict(float)
    log = []
    prov_tot = {}
    for nn in PROVS:
        rr = census.read90(nn)
        prov_tot[nn] = next(r["pop"] for r in rr if r["kind"] == "ostan")
        # 1390 units: dehestans as they are; a zoned city's zones gathered into one city
        units = {}
        order = []
        alias = {}
        for r in rr:
            if r["kind"] == "dehestan":
                key = ("d", r["cty"], r["bkh"], r["unit"])
                units[key] = {"kind": "dehestan", "name": r["name"], "pop": r["pop"] or 0,
                              "cty": r["cty"], "bkh": r["bkh"], "w": defaultdict(float)}
                order.append(key)
            elif r["kind"] == "city":
                key = ("c", r["cty"], r["bkh"], city_key(r["name"]))
                if key not in units:
                    units[key] = {"kind": "city", "name": city_key(r["name"]), "pop": 0,
                                  "cty": r["cty"], "bkh": r["bkh"], "w": defaultdict(float)}
                    order.append(key)
                units[key]["pop"] += r["pop"] or 0
                alias[("d", r["cty"], r["bkh"], r["unit"])] = key
        # second pass: a dehestan's row can come after its villages in the file
        for r in rr:
            if r["kind"] == "abadi" and r["pop"]:
                # a few villages sit under a city's code (its outlying parts); they are inside
                # that city's count, so they steer the city
                key = ("d", r["cty"], r["bkh"], r["unit"])
                key = alias.get(key, key)
                cand = v95.get(r["abadi"], [])
                hits = [t for t, _ in cand if t[0] == nn]
                if not hits:  # the county changed province (Tabas: Yazd in 1390, S. Khorasan in 1395)
                    hits = [t for t, n in cand if fold(n) == fold(r["name"])]
                if len(hits) == 1:
                    units[key]["w"][hits[0]] += r["pop"]
        for key in order:
            u = units[key]
            if u["kind"] == "city":
                hits = c95.get((nn, u["name"]), [])
                if not hits:  # moved province: look everywhere
                    hits = [t for (o, n), ts in c95.items() if n == u["name"] for t in ts]
                if len(set(hits)) == 1:
                    u["w"] = defaultdict(float, {hits[0]: u["pop"]})
                split = CITY_SPLITS.get((nn, u["name"]))
                if split:
                    u["w"] = defaultdict(float)
                    for k95 in split:
                        u["w"][c95[(nn, k95)][0]] += CITY_POP95[(nn, k95)]
        # 1390 nomads (bakhsh 99) and county rows
        nomad90 = defaultdict(int)
        cty90 = {}
        for r in rr:
            if r["kind"] == "nomad":
                nomad90[r["cty"]] += r["pop"] or 0
            elif r["kind"] == "county":
                cty90[r["cty"]] = r["pop"]
        # settled 1390 people per 1390 county, from units
        for c, p in cty90.items():
            s = sum(u["pop"] for u in units.values() if u["cty"] == c)
            # suppressed '*' nomad rows: the county row still counts them
            nomad90[c] = p - s
        # unmatched units take the distribution of their 1390 bakhsh, then of their county
        def dist(pred):
            d = defaultdict(float)
            for u in units.values():
                if pred(u) and u["w"]:
                    tw = sum(u["w"].values())
                    for t, w in u["w"].items():
                        d[t] += u["pop"] * w / tw
            return d
        for key in order:
            u = units[key]
            how = "own"
            w = u["w"]
            if not w and u["pop"]:
                w = dist(lambda x: x["cty"] == u["cty"] and x["bkh"] == u["bkh"])
                how = "district"
                if not w:
                    w = dist(lambda x: x["cty"] == u["cty"])
                    how = "county"
                if not w:
                    raise SystemExit(f"os{nn} {u['kind']} {u['name']}: nothing to carry it by")
            tw = sum(w.values())
            for t, x in w.items():
                out[t] += u["pop"] * x / tw
            top = max(w, key=w.get) if w else None
            log.append([nn, u["cty"], u["bkh"], u["kind"], u["name"], u["pop"], how,
                        "|".join(f"{t[1]}-{t[2]}:{x / tw:.3f}" for t, x in
                                 sorted(w.items(), key=lambda kv: -kv[1])) if w else "",
                        f"{top[1]}-{top[2]}" if top else ""])
        # nomads follow the settled people of their 1390 county
        for c, p in nomad90.items():
            if not p:
                continue
            d = dist(lambda x: x["cty"] == c)
            tw = sum(d.values())
            cc = defaultdict(float)
            for t, x in d.items():
                cc[(t[0], t[1])] += x
            for k2, x in cc.items():
                nomads[k2] += p * x / tw
            log.append([nn, c, "99", "nomad", "", p, "county", "", ""])
    return out, nomads, log, prov_tot


def largest_remainder(values, total):
    """Round a {key: float} to integers summing to `total`."""
    fl = {k: int(v) for k, v in values.items()}
    short = total - sum(fl.values())
    rem = sorted(values, key=lambda k: -(values[k] - int(values[k])))
    for k in rem[:short]:
        fl[k] += 1
    assert sum(fl.values()) == total, (sum(fl.values()), total)
    return fl


# ----------------------------------------------------------------------------- COD join
def cod_join(prov95, counties95):
    import geopandas as gpd

    ps_path = RAW / PS_NAME
    if not ps_path.exists():
        r = requests.get(PS_URL, headers=UA, timeout=120)
        r.raise_for_status()
        ps_path.write_bytes(r.content)
    ps = pd.read_csv(ps_path, dtype=str, encoding="utf-8-sig")
    ps.columns = [c.strip() for c in ps.columns]
    ps["T_TL"] = ps["T_TL"].str.strip().astype(int)
    a1 = gpd.read_file(COD1, engine="fiona")
    a2 = gpd.read_file(COD2, engine="fiona")
    assert len(a1) == 31 and len(a2) == 429 and set(a2["adm2_pcode"]) == set(ps["ADM2_PCODE"])
    # province: Persian names, folded
    p_of = {}
    for nn, (name, pop) in prov95.items():
        hit = a1[a1["adm1_name1"].map(fold) == fold(name)]
        assert len(hit) == 1, (nn, name)
        p_of[nn] = hit.iloc[0]["adm1_pcode"]
    assert len(set(p_of.values())) == 31
    # county: inside the province, by 1395 population (COD-PS is the same census), names checked
    c_of = {}
    for (nn, c), v in counties95.items():
        cand = ps[(ps["ADM1_PCODE"] == p_of[nn]) & (ps["T_TL"] == v["pop"])]
        if len(cand) != 1:
            cand = ps[(ps["ADM1_PCODE"] == p_of[nn])
                      & (ps["ADM2_FA"].map(lambda s: fold(s).replace("شهرستان", ""))
                         == fold(v["name"]))]
        assert len(cand) == 1, (nn, c, v, len(cand))
        row = cand.iloc[0]
        c_of[(nn, c)] = row["ADM2_PCODE"]
        if row["T_TL"] != v["pop"]:
            raise SystemExit(f"{nn}-{c} {v['name']}: SCI {v['pop']} != COD-PS {row['T_TL']}")
    assert len(set(c_of.values())) == 429
    namediff = [(k, counties95[k]["name"], ps.set_index("ADM2_PCODE").loc[p, "ADM2_FA"])
                for k, p in c_of.items()
                if fold(counties95[k]["name"]) not in
                fold(ps.set_index("ADM2_PCODE").loc[p, "ADM2_FA"])]
    print(f"COD join: 31 provinces by Persian name; 429 counties by 1395 population inside the "
          f"province, all equal to COD-PS ADM2 2016; {len(namediff)} with a different spelling")
    for k, a, b in namediff:
        print(f"    {k} SCI {a} / COD {b}")
    return p_of, c_of, a1, a2


# ----------------------------------------------------------------------------- 1403 estimate
def read_estimate(a1):
    """SCI's 1403 provincial estimate, as reproduced by Iran Open Data (iod-06124; Wayback copy
    of 2025-06-14), in thousands. Returns {adm1_pcode: people}."""
    import html as H

    s = EST_HTML.read_text(encoding="utf-8")
    t = re.sub(r"<script.*?</script>|<style.*?</style>", "", s, flags=re.S)
    t = H.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", t)))
    seg = t[t.index("جمعیت - ۱۴۰۳"):t.index("untitled")]
    pairs = re.findall(r"\d+(?:\.0)? ([^\d,]+?) ([\d,]{5,})", seg)
    est = {name.replace("NaN", "").strip(): int(v.replace(",", "")) for name, v in pairs}
    nat = est.pop("کل کشور")
    assert len(est) == 31, est
    out = {}
    for name, v in est.items():
        hit = a1[a1["adm1_name1"].map(fold) == fold(name)]
        assert len(hit) == 1, name
        out[hit.iloc[0]["adm1_pcode"]] = v
    s31 = sum(out.values())
    print(f"1403 estimate: 31 provinces sum {s31:,} against the printed national {nat:,} "
          f"(rounded to thousands: {s31 - nat:+,})")
    assert abs(s31 - nat) <= 31_000
    return out


def forward_cast(c11, c16, prov_of, est):
    """County 2024 = county 2016 x its province's 1403/1395 ratio, then half of each county's
    own 2011-2016 annual lead or lag over its province is carried on for the eight years and
    the province is raked back to its estimate. So the level is SCI's, and inside a province
    the counties that were outgrowing it keep doing so, at half the pace.

    Why half (FORWARD_DAMP): six county figures from SCI's own 1400 county estimate, quoted in
    the press (Tehran, Mashhad, Isfahan, Sanandaj, Saqqez, Sarvabad), sit closest to a line
    through 2016 and 2024 at 0.5 (within 1.2% for five of six, Isfahan 5.8% high at every
    setting). 0 (flat province rate) and 1 (full trend) are each up to 6-10% off on one of them,
    and the full trend runs Taleghan down 10% a year for eight years."""
    out = {}
    byp = defaultdict(list)
    for k in c16:
        byp[prov_of[k]].append(k)
    for p, ks in byp.items():
        p11 = sum(c11[k] for k in ks)
        p16 = sum(c16[k] for k in ks)
        g_p = (p16 / p11) ** (1 / 5)
        raw = {}
        for k in ks:
            rel = ((c16[k] / c11[k]) ** (1 / 5) / g_p) if c11[k] > 0 else 1.0
            raw[k] = c16[k] * rel ** (8 * FORWARD_DAMP)
        f = est[p] / sum(raw.values())
        out.update(largest_remainder({k: raw[k] * f for k in ks}, est[p]))
    return out


def main():
    prov95, counties95, bakhshs95, c95, v95 = load95()
    p_of, c_of, a1, a2 = cod_join(prov95, counties95)

    carried, nomads90, log, prov90 = carry90(c95, v95)
    assert sum(prov90.values()) == NATIONAL[YEAR90], sum(prov90.values())
    # 1390 by 1395 county (settled + nomads), rounded inside each 1390 province total
    by_c = defaultdict(float)
    for t, x in carried.items():
        by_c[(t[0], t[1])] += x
    for k, x in nomads90.items():
        by_c[k] += x
    missing = set(counties95) - set(by_c)
    assert not missing, missing
    # round provinces first, then counties inside each, so untouched provinces stay exact
    pf = defaultdict(float)
    for k, v in by_c.items():
        pf[k[0]] += v
    pint = largest_remainder({k: round(v, 6) for k, v in pf.items()}, NATIONAL[YEAR90])
    c11 = {}
    for nn in PROVS:
        c11.update(largest_remainder({k: v for k, v in by_c.items() if k[0] == nn}, pint[nn]))
    print(f"1390 carried: national {sum(c11.values()):,} (SCI's published 1390 total)")
    for nn in PROVS:
        s = sum(v for k, v in c11.items() if k[0] == nn)
        if s != prov90[nn]:
            print(f"  province {nn} {prov95[nn][0]}: 1390 on 1395 lines {s:,}, "
                  f"1390 as published {prov90[nn]:,} ({s - prov90[nn]:+,})")
    how = defaultdict(int)
    for row in log:
        how[row[6]] += row[5] or 0
    print("  1390 people carried by:", {k: f"{v:,}" for k, v in how.items()})

    c16 = {k: v["pop"] for k, v in counties95.items()}
    est = read_estimate(a1)
    prov_of = {k: p_of[k[0]] for k in c16}
    c24 = forward_cast(c11, c16, prov_of, est)

    rows = []
    for k in sorted(c16):
        code = c_of[k]
        for y, d in ((YEAR90, c11), (YEAR95, c16), (YEAR_EST, c24)):
            rows.append((code, 2, y, d[k]))
    for nn in PROVS:
        pc = p_of[nn]
        ks = [k for k in c16 if k[0] == nn]
        for y, d in ((YEAR90, c11), (YEAR95, c16), (YEAR_EST, c24)):
            rows.append((pc, 1, y, sum(d[k] for k in ks)))
        assert sum(c16[k] for k in ks) == prov95[nn][1]
        assert sum(c24[k] for k in ks) == est[pc]

    # district level, if bakhsh.py has written its unit table
    lvl3 = DATA / "bakhsh_units.csv"
    if lvl3.exists():
        import bakhsh
        rows += bakhsh.population_rows(bakhshs95, counties95, carried, nomads90, c11, c16, c24)

    with open(DATA / "population.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["code", "level", "year", "pop"])
        w.writerows(rows)
    print(f"wrote {DATA / 'population.csv'} ({len(rows)} rows)")

    with open(DATA / "counties.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["sci_code", "adm2_pcode", "adm1_pcode", "name_fa", "name_en", "pop2011",
                    "pop2016", "pop2024", "nomad2016"])
        en = a2.set_index("adm2_pcode")["adm2_name"]
        for k in sorted(c16):
            w.writerow([k[0] + k[1], c_of[k], p_of[k[0]], counties95[k]["name"], en[c_of[k]],
                        c11[k], c16[k], c24[k], counties95[k]["nomad"]])
    with open(DATA / "carry_1390.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["ost", "cty90", "bkh90", "kind", "name", "pop1390", "carried_by",
                    "to_1395_cty-bkh_shares", "main_1395_cty-bkh"])
        w.writerows(log)
    print("wrote counties.csv, carry_1390.csv")


if __name__ == "__main__":
    main()
