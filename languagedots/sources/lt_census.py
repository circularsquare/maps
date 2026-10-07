"""Lithuania: Statistics Lithuania, Gyventojų surašymas 2021, mother tongue by municipality.

    python sources/lt_census.py --fetch    two SDMX GETs (29 KB and 9 KB) if missing
    python sources/lt_census.py            normalise from data/raw/lt/

-> data/normalized/lt.csv (levels `country`, `region`, `county`, `municipality`; four nested
   levels in one dimension, alternatives, never summed; only `municipality` is drawn)

THE TABLE. SDMX dataflow **`S3R778_GBS010509`**, "Population | Administrative territory (from
2000) | Mother tongue (2021)", on `osp-rs.stat.gov.lt`, the open SDMX host (the web UI on
`osp.stat.gov.lt` sits behind Cloudflare; religiondots/sources/lt.py found the API host). It was
found in the 9,521-dataflow catalogue `/rest_xml/dataflow/`, whose descriptions list each cube's
dimensions; three cubes carry a mother-tongue dimension (this one; GBS010302, territory x
ethnicity x mother tongue x sex for 2001-2011 only; GBS010302_1, national ethnicity x mother
tongue for 2001-2021). The question is `gimtoji kalba`, mother tongue, and a person could give
two; the table has eleven values: Lithuanian, Polish, Russian, Belarusian, Ukrainian, Latvian,
German, Romani, `Kitos` (other), `Dvi gimtosios kalbos` (two mother tongues, the pair not
published) and the total. There is no "not stated" column, and none is needed: the national
cube's `Nenurodyta` is 0 for 2021 (it was 2.1% in 2011), and in every unit with nothing
withheld the categories exhaust the total exactly. Mother tongue was evidently filled for
everyone in 2021's register-based census; the record (sources/lt.md) says what that implies.

NULLS. One OBS_STATUS value exists here, `konfidencialūs duomenys` (confidential): a withheld
cell. 168 municipality cells (864 people, 0.03%, but 56% of German and 18% of Latvian speakers)
and 4 county cells. **Estimated from the margins** (Anita, 2026-10-05: "estimate the suppressed
cells from the totals, as ro and fi did"), by `impute()`: county cells first from the national
row, then each county's municipality cells from its county row, by iterative proportional
fitting as sources/ro_census.py does. Every estimated row is tier `derived`; the national
total of every language is then reproduced by the municipalities (asserted).

CHECKS: the dimension ids and the level counts (1, 2, 10, 60); the national total against
2,810,761 (religiondots' religion cube for the same census); every level's totals sum to the
country; municipalities sum to their county in every category where no cell is withheld; the
national row against the second cube, GBS010302_1 (ethnicity x mother tongue, the 2021 total
over ethnicity), category by category, and the not-stated complement against that cube's own
`Nenurodyta`; each municipality's total against religiondots' lt.csv (same census, religion).
"""

import csv
import json
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "lt")
OUT = os.path.join(ROOT, "data", "normalized", "lt.csv")
RD_NORM = os.path.join(os.path.dirname(ROOT), "religiondots", "data", "normalized", "lt.csv")

SOURCE_ID = "lt_surasymas_2021_mt"
YEAR = 2021
DRAWN_YEAR = "2021"

API = "https://osp-rs.stat.gov.lt/rest_json/data/{}/"
FLOW = "S3R778_GBS010509"        # territory x mother tongue, 2021: the drawn table
FLOW_NAT = "S3R778_GBS010302_1"  # ethnicity x mother tongue, national, 2001-2021: the check
CUBE = os.path.join(RAW, f"{FLOW}.json")
CUBE_NAT = os.path.join(RAW, f"{FLOW_NAT}.json")

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count", "tier", "year",
           "source_id", "note"]

NATIONAL = 2_810_761
COUNTRY_CODE = "00"
TOTAL_CAT = "TOT"
NOT_STATED = "Nenurodyta"
EXPECTED = {"country": 1, "region": 2, "county": 10, "municipality": 60}
VILKAVISKIS_SHORT = 3   # see check()
CROSS_CUBE_TOL = 3      # two tabulations of one census, a few people apart in some cells
STATUS_CONFIDENTIAL = "konfidencial"
STATUS_NO_PHENOMENON = "nebuvo"   # the national cube's nulls: a true zero

# Municipality code -> county code. The cube nests them in one dimension without saying which
# municipality is in which county; this is the standard 2000- layout (ten apskritys), asserted
# below by every county's total equalling the sum of its municipalities' totals.
COUNTY_OF = {
    "01": ["11", "33", "15", "59", "38"],                                    # Alytus
    "02": ["12", "19", "52", "46", "49", "53", "69", "72"],                  # Kaunas
    "03": ["21", "55", "56", "23", "25", "88", "75"],                        # Klaipėda
    "04": ["58", "48", "18", "84", "39"],                                    # Marijampolė
    "05": ["36", "57", "27", "66", "67", "73"],                              # Panevėžys
    "06": ["32", "47", "54", "65", "71", "29", "91"],                        # Šiauliai
    "07": ["94", "63", "87", "77"],                                          # Tauragė
    "08": ["61", "68", "74", "78"],                                          # Telšiai
    "09": ["34", "45", "62", "82", "30", "43"],                              # Utena
    "10": ["42", "85", "89", "86", "79", "81", "13", "41"],                  # Vilnius
}


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    for flow, path in ((FLOW, CUBE), (FLOW_NAT, CUBE_NAT)):
        if os.path.exists(path) and os.path.getsize(path) > 5_000:
            print("already have", path)
            continue
        url = API.format(flow)
        print("GET", url)
        r = requests.get(url, timeout=300,
                         headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        r.raise_for_status()
        doc = r.json()
        for key in ("structure", "dataSets"):
            if key not in doc:
                raise SystemExit(f"{flow}: not an SDMX-JSON cube, keys {list(doc)}")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, ensure_ascii=False)
        print(f"  {os.path.getsize(path):,} bytes")


def _load(path):
    if not os.path.exists(path):
        raise SystemExit(f"missing {path} -- run with --fetch first")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _level(code):
    if code == COUNTRY_CODE:
        return "country"
    if not code.isdigit():
        return "region"           # LT01, LT02, the two NUTS2 regions
    return "county" if int(code) <= 10 else "municipality"


def _axis(dims, want):
    for n, d in enumerate(dims):
        if want == d["id"]:
            return n
    raise SystemExit(f"no dimension {want!r} among {[d['id'] for d in dims]}")


def _cells(doc, want_dims):
    """{(value ids...): count or None (withheld)} for DRAWN_YEAR, keyed in want_dims order."""
    dims = doc["structure"]["dimensions"]["observation"]
    idx = [_axis(dims, w) for w in want_dims]
    i_time = _axis(dims, "LAIKOTARPIS")
    obs_attrs = doc["structure"]["attributes"]["observation"]
    pos = next((n for n, a in enumerate(obs_attrs) if a["id"] == "OBS_STATUS"), None)
    status_vals = [(v.get("name") or "") for v in obs_attrs[pos]["values"]] if pos is not None else []
    out, names = {}, [{v["id"]: v["name"].strip() for v in dims[i]["values"]} for i in idx]
    for key, val in doc["dataSets"][0]["observations"].items():
        k = [int(x) for x in key.split(":")]
        if dims[i_time]["values"][k[i_time]]["id"] != DRAWN_YEAR:
            continue
        ids = tuple(dims[i]["values"][k[i]]["id"] for i in idx)
        n = val[0]
        if n is None:
            st = val[pos + 1] if pos is not None and len(val) > pos + 1 else None
            text = status_vals[st] if st is not None else ""
            if STATUS_NO_PHENOMENON in text.lower():
                n = 0                       # "there was no such phenomenon": a true zero
            elif STATUS_CONFIDENTIAL not in text.lower():
                raise SystemExit(f"{ids}: null with status {text!r}, not 'confidential' -- "
                                 "a null of unknown meaning cannot be read as zero or withheld")
        out[ids] = n
    return out, names


def read():
    doc = _load(CUBE)
    dims = doc["structure"]["dimensions"]["observation"]
    ids = [d["id"] for d in dims]
    if ids.count("SavivaldybesM1411") != 1 or ids.count("GBS_kalbos_09") != 1:
        raise SystemExit(f"unexpected dimensions {ids}")
    cells, (geo_names, cat_names) = _cells(doc, ["SavivaldybesM1411", "GBS_kalbos_09"])
    return cells, geo_names, cat_names


def build_rows(cells, geo_names, cat_names):
    rows, withheld, rest = [], {}, {}
    cats = [c for c in cat_names if c != TOTAL_CAT]
    for g, gname in geo_names.items():
        level = _level(g)
        tot = cells.get((g, TOTAL_CAT))
        if tot is None:
            raise SystemExit(f"{g} {gname}: total missing or withheld")
        rows.append(dict(geo_id=g, geo_level=level, geo_name=gname,
                         source_category=cat_names[TOTAL_CAT], count=tot, tier="measured",
                         year=YEAR, source_id=SOURCE_ID,
                         note=f"level={level}; universe total, not a language"))
        named, held = 0, []
        for c in cats:
            if (g, c) not in cells:
                raise SystemExit(f"{g}/{c}: no observation at all")
            n = cells[(g, c)]
            if n is None:
                held.append(c)
                withheld.setdefault(c, []).append(g)
                continue
            named += n
            rows.append(dict(geo_id=g, geo_level=level, geo_name=gname,
                             source_category=cat_names[c], count=n, tier="measured",
                             year=YEAR, source_id=SOURCE_ID, note=f"level={level}; code={c}"))
        if not held:
            rest[g] = tot - named
    return rows, withheld, rest


def _ipf(cells, row_margin, col_margin, rounds=2000):
    """Fill withheld cells so each row and each column sums to its margin (sources/ro_census.py's
    `_ipf`, copied). cells: [(row, col)]; a row margin is a unit's total less its published
    cells, a column margin the parent's cell less its published children. Starts every cell at 1
    and scales rows and columns in turn: the even spread that meets both totals. Returns ({cell: value}, summed row miss,
    largest column miss)."""
    v = {c: 1.0 for c in cells}
    by_r, by_c = {}, {}
    for c in cells:
        by_r.setdefault(c[0], []).append(c)
        by_c.setdefault(c[1], []).append(c)
    for _ in range(rounds):
        for grp, margin in ((by_r, row_margin), (by_c, col_margin)):
            for k, cs in grp.items():
                tot = sum(v[c] for c in cs)
                want = max(margin.get(k, 0.0), 0.0)
                for c in cs:
                    v[c] = v[c] * want / tot if tot > 0 else want / len(cs)
    err_r = sum(abs(sum(v[c] for c in cs) - max(row_margin.get(k, 0), 0))
                for k, cs in by_r.items())
    err_c = max((abs(sum(v[c] for c in cs) - max(col_margin.get(k, 0), 0))
                 for k, cs in by_c.items()), default=0.0)
    return v, err_r, err_c


def impute(cells, geo_names, cat_names):
    """Estimate every withheld county and municipality cell; rows tier `derived`.

    Both margins of each withheld cell are known exactly: its unit's total less the unit's
    published cells (nothing is not-stated, so the categories exhaust the total), and its
    parent's cell for that language less the parent's published children. County cells (4) are
    estimated from the national row first; those estimates then stand as the parent cells for
    their counties' municipalities. Within each county the margins agree exactly (checked
    2026-10-05, Vilkaviškis's 3-person slip included, since it is in both its total and Kitos)."""
    cats = [c for c in cat_names if c != TOTAL_CAT]
    note = "confidential cell, estimated from the margins (sources/lt_census.py impute)"

    def rest(g):
        return cells[(g, TOTAL_CAT)] - sum(cells[(g, c)] or 0 for c in cats)

    out, est = [], {}
    counties = list(COUNTY_OF)
    hc = [(g, c) for g in counties for c in cats if cells[(g, c)] is None]
    row_m = {g: rest(g) for g, _ in hc}
    col_m = {c: cells[(COUNTRY_CODE, c)] - sum(cells[(g, c)] or 0 for g in counties) for _, c in hc}
    v, er, ec = _ipf(hc, row_m, col_m)
    print(f"\n  county: {len(hc)} withheld cells, {sum(v.values()):,.1f} people; "
          f"miss {er:.3g} on rows, {ec:.3g} on a language")
    worst = max(er, ec)
    est.update(v)

    n_m, total_m = 0, 0.0
    for cty, ms in COUNTY_OF.items():
        mc = [(g, c) for g in ms for c in cats if cells[(g, c)] is None]
        if not mc:
            continue
        parent = {c: cells[(cty, c)] if cells[(cty, c)] is not None else est[(cty, c)] for _, c in mc}
        row_m = {g: rest(g) for g, _ in mc}
        col_m = {c: parent[c] - sum(cells[(g, c)] or 0 for g in ms) for _, c in mc}
        v, er, ec = _ipf(mc, row_m, col_m)
        worst = max(worst, er, ec)
        est.update(v)
        n_m += len(mc)
        total_m += sum(v.values())
    print(f"  municipality: {n_m} withheld cells, {total_m:,.1f} people; largest miss on any "
          f"margin {worst:.3g}")
    if worst > 0.5:
        raise SystemExit("withheld-cell estimate does not meet its margins")
    for (g, c), n in est.items():
        out.append(dict(geo_id=g, geo_level=_level(g), geo_name=geo_names[g],
                        source_category=cat_names[c], count=round(n, 3), tier="derived",
                        year=YEAR, source_id=SOURCE_ID, note=f"level={_level(g)}; code={c}; {note}"))
    return out


def check_imputed(rows, cat_names):
    """With the estimates, the municipalities reproduce the national row, language by language."""
    nat, muni = {}, {}
    for r in rows:
        if r["geo_level"] == "country":
            nat[r["source_category"]] = r["count"]
        elif r["geo_level"] == "municipality":
            muni[r["source_category"]] = muni.get(r["source_category"], 0) + r["count"]
    total_label = cat_names[TOTAL_CAT]
    worst = 0.0
    for label, n in nat.items():
        if label == total_label:
            continue
        miss = abs(muni.get(label, 0) - n)
        slip = VILKAVISKIS_SHORT if label == cat_names["OTH"] else 0
        worst = max(worst, abs(miss - slip))
    good = worst < 0.5
    print(f"  {'OK ' if good else 'BAD'} with the estimates the municipalities hold every "
          f"language's national total (largest miss {worst:.3g}, Vilkaviškis's 3 in Kitos aside)")
    if not good:
        raise SystemExit("estimates do not reproduce the national row")


def check(rows, withheld, rest, cat_names):
    ok = True
    by = {}
    for r in rows:
        by.setdefault(r["geo_level"], {}).setdefault(r["geo_id"], {})[r["source_category"]] = r["count"]
    total_label = cat_names[TOTAL_CAT]

    for lv, want in EXPECTED.items():
        got = len(by.get(lv, {}))
        good = got == want
        ok &= good
        print(f"  {'OK ' if good else 'BAD'} {lv:<13} {got:>3} units (expected {want})")

    nat = by["country"][COUNTRY_CODE]
    muni = by["municipality"]
    good = nat[total_label] == NATIONAL
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} national total {nat[total_label]:,} (published {NATIONAL:,})")
    for lv in ("region", "county"):
        s = sum(u[total_label] for u in by[lv].values())
        good = s == NATIONAL
        ok &= good
        print(f"  {'OK ' if good else 'BAD'} {lv:<13} totals sum to the country ({s:,})")
    # THE ONE SLIP IN THE CUBE: Vilkaviškis (39) is published 3 people short, in its total and
    # in `Kitos` alike (35,365 and 32), against its county (Marijampolė 138,292, Kitos 210) and
    # against the religion cube of the same census (35,368). Internally it still adds up, so it
    # is drawn as published; 3 people, 0.0001%.
    s = sum(u[total_label] for u in muni.values())
    good = s == NATIONAL - VILKAVISKIS_SHORT
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} municipality  totals sum to the country less the "
          f"{VILKAVISKIS_SHORT} Vilkaviškis is short ({s:,})")

    # national not-stated is zero; where a municipality publishes every cell, the categories
    # must therefore exhaust its total
    nz = {g: n for g, n in rest.items() if n != 0 and _level(g) == "municipality"}
    good = nat.get(cat_names.get("XXX", NOT_STATED), 0) == 0 and not nz and rest[COUNTRY_CODE] == 0
    ok &= good
    n_full = sum(1 for g in rest if _level(g) == "municipality")
    print(f"  {'OK ' if good else 'BAD'} no not-stated: the categories exhaust the total in the "
          f"country and in all {n_full} municipalities with nothing withheld {nz or ''}")

    # county = sum of its municipalities, every category where nothing is withheld
    listed = sorted(m for ms in COUNTY_OF.values() for m in ms)
    good = listed == sorted(muni)
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} the county lookup lists the 60 municipalities once each")
    bad, n_cmp = 0, 0
    for cty, ms in COUNTY_OF.items():
        for cat, n in by["county"][cty].items():
            if any(cat not in muni[m] for m in ms):
                continue
            n_cmp += 1
            s = sum(muni[m][cat] for m in ms)
            slip = cty == "04" and cat in (total_label, cat_names["OTH"]) and n - s == VILKAVISKIS_SHORT
            if s != n and not slip:
                bad += 1
                print(f"    BAD county {cty} {cat}: {n:,} vs municipalities {s:,}")
    ok &= bad == 0
    print(f"  {'OK ' if bad == 0 else 'BAD'} every county equals the sum of its municipalities "
          f"in all {n_cmp} fully published (county, category) cells, Vilkaviškis's slip aside")

    # second cube: national ethnicity x mother tongue, summed over ethnicity. The two cubes are
    # separate tabulations of the same census and disagree by a person or three in some cells.
    nd = _load(CUBE_NAT)
    nc, (lang_names, _eth) = _cells(nd, ["GBS_kalbos_302", "GBS_TAUTYBE2_0301"])
    second = {lang_names[l]: v for (l, e), v in nc.items() if e == "TOT"}
    second_other = second[cat_names["OTH"]]
    # it folds Latvian, German and Romani into `Kitos`
    here = dict(nat)
    here[cat_names["OTH"]] = sum(nat[cat_names[c]] for c in ("lv", "de", "ROM", "OTH"))
    print("\n  national row against GBS010302_1 (ethnicity x mother tongue, total over ethnicity;"
          " its Kitos holds Latvian, German and Romani):")
    for label, other in sorted(second.items(), key=lambda kv: -(kv[1] or 0)):
        n = here.get(label, 0)
        good = abs(other - n) <= CROSS_CUBE_TOL
        ok &= good
        print(f"    {'OK ' if good else 'BAD'} {label:<32} {n:>10,}  {other:>10,}  {other - n:+d}")

    # municipality totals against religiondots' religion cube, same census
    if os.path.exists(RD_NORM):
        rd = {}
        with open(RD_NORM, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r["geo_level"] == "municipality" and r["note"].endswith("not a religion category"):
                    rd[r["geo_id"].zfill(2)] = int(r["count"])
        diff = {m: rd.get(m, 0) - muni[m][total_label] for m in muni
                if rd.get(m) != muni[m][total_label]}
        good = len(rd) == 60 and diff == {"39": VILKAVISKIS_SHORT}
        ok &= good
        print(f"\n  {'OK ' if good else 'BAD'} all 60 municipality totals equal religiondots' "
              f"religion cube but Vilkaviškis's slip ({len(rd)} read, differences {diff})")

    print("\n  withheld (confidential) municipality cells, estimated by impute():")
    tot_held = 0
    for c, gs in withheld.items():
        muni_held = [g for g in gs if _level(g) == "municipality"]
        if not muni_held:
            continue
        label = cat_names[c]
        s = sum(muni[m].get(label, 0) for m in muni)
        gap = nat[label] - s
        tot_held += gap
        print(f"    {label:<28} {len(muni_held):>2} of 60 withheld; "
              f"national {nat[label]:,}, municipalities {s:,}, "
              f"withheld {gap:,} ({100 * gap / nat[label]:.1f}%)")
    print(f"    total about {tot_held:,} people, {100 * tot_held / NATIONAL:.3f}% of the country")

    print("\n  national categories:")
    for label, n in sorted(nat.items(), key=lambda kv: -kv[1]):
        print(f"    {n:>10,}  {100.0 * n / NATIONAL:6.2f}%  {label}")
    if not ok:
        raise SystemExit("reconciliation FAILED")


def main():
    if "--fetch" in sys.argv:
        fetch()
    cells, geo_names, cat_names = read()
    rows, withheld, rest = build_rows(cells, geo_names, cat_names)
    check(rows, withheld, rest, cat_names)
    rows += impute(cells, geo_names, cat_names)
    check_imputed(rows, cat_names)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print("\nwrote", OUT, f"({len(rows):,} rows)")


if __name__ == "__main__":
    main()
