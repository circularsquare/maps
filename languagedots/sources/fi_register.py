"""Finland: Statistics Finland, population register, language by municipality, 31 Dec 2025.

    python sources/fi_register.py --fetch    download the two tables and two boundary layers
    python sources/fi_register.py            normalise from data/raw/fi/

-> data/normalized/fi.csv (levels `country`, `maakunta`, `municipality`; alternatives, never
   summed; only `municipality` is drawn)

SOURCE. StatFin's PxWeb API (CC BY 4.0), database StatFin/vaerak (population structure):

  11rm  Language according to sex by municipality, 1990-2025. 308 municipalities of the
        1 Jan 2026 division (Aland's 16 included) x 169 language codes. The drawn table.
  11rl  Language according to age and sex by region, 1990-2025. The 19 maakunnat (regions),
        the same 169 codes. The second-table check: 11rm summed by maakunta must equal it.

The language is the one recorded in the Population Information System (vaestotietojarjestelma),
which the register calls mother tongue (aidinkieli): one code per person, given when a birth or
an immigrant is registered and seldom changed afterwards. Codes are ISO 639-1 plus three of
Statistics Finland's own: `98` other language, `X` unknown, and the subtotals `SSS` (total),
`01` (national languages: Finnish, Swedish, Sami) and `02` (foreign languages). The subtotals
are checked and dropped; every other code is written, labelled with its own code and Statistics
Finland's English text ("sv Swedish"), so taxonomy/fi2025.py keys on the code.

Both tables are fetched as json-stat2 with sex = total, age = total, year = 2025.

Boundaries for the maakunta check and for sources/fi_geo.py: Statistics Finland's WFS
(geo.stat.fi, `tilastointialueet:kunta1000k_2026` and `maakunta1000k_2026`, CC BY 4.0), the
same 1 Jan 2026 division the tables use. A municipality's maakunta is taken by a point inside
its polygon; the check against 11rl then tests that join and the data at once.

CHECKS: the three subtotals equal the sums of their codes in every unit; 308 municipalities,
each matched to one polygon of the 2026 layer both ways; the national row of 11rl equals the
municipalities summed, per code; each of the 19 maakunnat equals its municipalities summed,
per code.
"""
import csv
import json
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "fi")
OUT = os.path.join(ROOT, "data", "normalized", "fi.csv")

SOURCE_ID = "fi_statfin_vaerak_2025"
YEAR = 2025
API = "https://pxdata.stat.fi/PxWeb/api/v1/en/StatFin/vaerak/"
WFS = ("https://geo.stat.fi/geoserver/tilastointialueet/wfs?service=WFS&version=2.0.0"
       "&request=GetFeature&typeNames=tilastointialueet:{layer}&outputFormat=json"
       "&srsName=EPSG:3067")
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

MUNI_JSON = os.path.join(RAW, "11rm_2025.json")
REGION_JSON = os.path.join(RAW, "11rl_2025.json")
KUNTA = os.path.join(RAW, "kunta1000k_2026.geojson")
MAAKUNTA = os.path.join(RAW, "maakunta1000k_2026.geojson")

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count", "tier", "year",
           "source_id", "note"]
SUBTOTALS = {"SSS", "01", "02"}
NATIONAL_CODES = ("fi", "sv", "se")
N_MUNI, N_MAAKUNTA = 308, 19


def _query(area_code, extra):
    q = [{"code": area_code, "selection": {"filter": "all", "values": ["*"]}},
         {"code": "kieli_15_20180102", "selection": {"filter": "all", "values": ["*"]}},
         {"code": "sukupuoli_9_20180101", "selection": {"filter": "item", "values": ["SSS"]}},
         {"code": "timeperiod_y", "selection": {"filter": "item", "values": [str(YEAR)]}}]
    q += extra
    return {"query": q, "response": {"format": "json-stat2"}}


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    jobs = [
        (MUNI_JSON, "11rm.px", _query("alue_23_20260101", [])),
        (REGION_JSON, "11rl.px", _query("alue_23_20260101", [
            {"code": "ikaryhma_10_20180101", "selection": {"filter": "item", "values": ["SSS"]}}])),
    ]
    for dest, table, body in jobs:
        if os.path.exists(dest) and os.path.getsize(dest) > 10_000:
            print("already have", dest)
            continue
        print("POST", API + table)
        r = requests.post(API + table, json=body, headers=UA, timeout=300)
        r.raise_for_status()
        js = r.json()
        if js.get("class") != "dataset":
            raise SystemExit(f"{table}: not a json-stat2 dataset")
        with open(dest + ".part", "w", encoding="utf-8") as fh:
            json.dump(js, fh, ensure_ascii=False)
        os.replace(dest + ".part", dest)
        print(f"  {os.path.getsize(dest):,} bytes")
    for dest, layer in ((KUNTA, "kunta1000k_2026"), (MAAKUNTA, "maakunta1000k_2026")):
        if os.path.exists(dest) and os.path.getsize(dest) > 10_000:
            print("already have", dest)
            continue
        url = WFS.format(layer=layer)
        print("GET", url)
        r = requests.get(url, headers=UA, timeout=600)
        r.raise_for_status()
        if not r.content.lstrip().startswith(b"{"):
            raise SystemExit(f"{layer}: not GeoJSON ({r.content[:80]!r})")
        with open(dest + ".part", "wb") as fh:
            fh.write(r.content)
        os.replace(dest + ".part", dest)
        print(f"  {os.path.getsize(dest):,} bytes")


def _cube(path):
    """json-stat2 -> ({(area, language): count}, area labels, language labels)."""
    with open(path, encoding="utf-8") as fh:
        js = json.load(fh)
    ids, size = js["id"], js["size"]
    cats = {}
    for d in ids:
        idx = js["dimension"][d]["category"]["index"]
        order = sorted(idx, key=idx.get) if isinstance(idx, dict) else list(idx)
        cats[d] = (order, js["dimension"][d]["category"]["label"])
    for d, n in zip(ids, size):
        if d not in ("alue_23_20260101", "kieli_15_20180102") and n != 1:
            raise SystemExit(f"{path}: dimension {d} has {n} values, expected 1")
    area_d, lang_d = "alue_23_20260101", "kieli_15_20180102"
    vals = js["value"]
    if isinstance(vals, dict):
        raise SystemExit(f"{path}: sparse value object; this reader expects a full array")
    # row-major over ids; every other dimension has size 1
    strides, s = {}, 1
    for d, n in reversed(list(zip(ids, size))):
        strides[d] = s
        s *= n
    status = js.get("status") or {}
    out = {}
    for ai, a in enumerate(cats[area_d][0]):
        for li, lang in enumerate(cats[lang_d][0]):
            i = ai * strides[area_d] + li * strides[lang_d]
            v = vals[i]
            if v is None:
                # the only symbol either table uses is '...', "confidential": under 10 people
                st = status.get(str(i)) if isinstance(status, dict) else status[i]
                if st != "...":
                    raise SystemExit(f"{path}: empty cell at {a} / {lang} with status {st!r}")
                out[(a, lang)] = None
            else:
                out[(a, lang)] = int(v)
    return out, cats[area_d], cats[lang_d]


def fill_suppressed(mcube, rcube, munis, region_of, regions, codes):
    """Estimate 11rm's '...' cells (under 10 people, zero included) from both margins.

    Within each maakunta and each block of codes (the three national languages; everything
    else), the suppressed cells must add up to the region's published count per code (11rl, not
    suppressed) less the municipalities' printed cells, and per municipality to its printed
    block subtotal less its printed cells. Each cell is 0..9. Iterative proportional fitting
    from an even seed meets both margins, clipping at 9 after each pass. Returns
    {(muni, code): estimate} and a list of check lines.
    """
    leaves = [c for c in codes if c not in SUBTOTALS]
    blocks = {"01": [c for c in leaves if c in NATIONAL_CODES],
              "02": [c for c in leaves if c not in NATIONAL_CODES]}
    est, worst, lines = {}, 0.0, []
    for m in munis:
        if mcube[(m, "SSS")] is None:
            raise SystemExit(f"{m}: the municipality total itself is suppressed")
    for r in regions:
        mine = [m for m in munis if region_of[m] == r]
        for sub, cols in blocks.items():
            row_t = {}
            for m in mine:
                st = mcube[(m, sub)]
                if st is None:
                    other = mcube[(m, "02" if sub == "01" else "01")]
                    if other is None:
                        raise SystemExit(f"{m}: both block subtotals suppressed")
                    st = mcube[(m, "SSS")] - other
                row_t[m] = st - sum(mcube[(m, c)] or 0 for c in cols)
            col_t = {c: rcube[(r, c)] - sum(mcube[(m, c)] or 0 for m in mine) for c in cols}
            cells = [(m, c) for m in mine for c in cols if mcube[(m, c)] is None]
            n_row = {m: sum(1 for x in cells if x[0] == m) for m in mine}
            n_col = {c: sum(1 for x in cells if x[1] == c) for c in cols}
            for m, t in row_t.items():
                if t < 0 or t > 9 * n_row[m]:
                    raise SystemExit(f"{m} block {sub}: residual {t} outside 0..9 x {n_row[m]}")
            for c, t in col_t.items():
                if t < 0 or t > 9 * n_col[c]:
                    raise SystemExit(f"{r} / {c}: residual {t} outside 0..9 x {n_col[c]}")
            if sum(row_t.values()) != sum(col_t.values()):
                raise SystemExit(f"{r} block {sub}: residuals {sum(row_t.values())} by "
                                 f"municipality, {sum(col_t.values())} by language")
            import numpy as np

            rt = np.array([row_t[m] for m in mine], dtype=float)
            ct = np.array([col_t[c] for c in cols], dtype=float)
            mask = np.array([[mcube[(m, c)] is None for c in cols] for m in mine])
            x = mask.astype(float)
            x[rt == 0, :] = 0.0                   # a margin of zero pins its cells at zero
            x[:, ct == 0] = 0.0
            err = 0.0
            for _ in range(5000):
                s = x.sum(axis=1)
                x = np.minimum(9.0, x * np.divide(rt, s, out=np.zeros_like(rt), where=s > 0)[:, None])
                s = x.sum(axis=0)
                x = np.minimum(9.0, x * np.divide(ct, s, out=np.zeros_like(ct), where=s > 0)[None, :])
                err = max(np.abs(x.sum(axis=1) - rt).max(initial=0.0),
                          np.abs(x.sum(axis=0) - ct).max(initial=0.0))
                if err < 1e-6:
                    break
            worst = max(worst, float(err))
            for i, j in zip(*np.nonzero(mask)):
                est[(mine[i], cols[j])] = float(x[i, j])
    total = sum(est.values())
    lines.append(f"  {'OK ' if worst < 0.01 else 'BAD'} {len(est):,} suppressed cells estimated "
                 f"from both margins; worst margin error {worst:.2g} people")
    lines.append(f"      they hold {total:,.0f} people, "
                 f"{100 * total / mcube[('SSS', 'SSS')]:.2f}% of Finland")
    return est, worst < 0.01, lines


def _leaf_label(code, labels):
    return f"{code} {' '.join(labels[code].split())}"


def _check_subtotals(cube, areas, codes, where):
    leaves = [c for c in codes if c not in SUBTOTALS]
    foreign = [c for c in leaves if c not in NATIONAL_CODES]
    bad, n = [], 0
    for a in areas:
        for sub, cols in (("SSS", leaves), ("01", NATIONAL_CODES), ("02", foreign)):
            vals = [cube[(a, c)] for c in cols]
            if cube[(a, sub)] is None or None in vals:
                continue                          # a suppressed cell: fill_suppressed() checks
            n += 1
            if sum(vals) != cube[(a, sub)]:
                bad.append((a, sub))
    print(f"  {'OK ' if not bad else 'BAD'} {where}: {n:,} fully printed subtotals (total, "
          "national languages, foreign languages) equal their codes summed"
          + (f"; differ: {bad[:6]}" if bad else ""))
    return not bad, leaves


def read():
    for p in (MUNI_JSON, REGION_JSON, KUNTA, MAAKUNTA):
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}; run with --fetch first")
    ok = True
    mcube, (mareas, mlabels), (codes, llabels) = _cube(MUNI_JSON)
    rcube, (rareas, rlabels), (rcodes, _) = _cube(REGION_JSON)
    if rcodes != codes:
        raise SystemExit("11rl and 11rm have different language code lists")
    munis = [a for a in mareas if a != "SSS"]
    good = all(a.startswith("KU") and len(a) == 5 for a in munis) and len(munis) == N_MUNI
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} {len(munis)} municipalities in 11rm (expected {N_MUNI}), "
          "all coded KUnnn")
    g, leaves = _check_subtotals(mcube, mareas, codes, "11rm")
    ok &= g
    g, _ = _check_subtotals(rcube, rareas, codes, "11rl")
    ok &= g

    # national: 11rm's own whole-country row against 11rl's, wherever 11rm prints it
    bad = [c for c in codes if mcube[("SSS", c)] is not None
           and mcube[("SSS", c)] != rcube[("SSS", c)]]
    n_nat = sum(1 for c in codes if mcube[("SSS", c)] is not None)
    ok &= not bad
    print(f"  {'OK ' if not bad else 'BAD'} whole-country rows of 11rm and 11rl agree in all "
          f"{n_nat} codes 11rm prints there" + (f" (differ: {bad[:6]})" if bad else ""))
    for m in munis:
        for sub in ("SSS", "01"):
            if mcube[(m, sub)] is None:
                raise SystemExit(f"{m}: subtotal {sub} suppressed; fill_suppressed() needs it")

    # municipality -> maakunta, by a point inside each 2026 polygon
    import geopandas as gpd

    k = gpd.read_file(KUNTA)
    mk = gpd.read_file(MAAKUNTA)
    k["code"] = "KU" + k["kunta"].astype(str).str.zfill(3)
    a, b = set(k["code"]), set(munis)
    good = a == b and len(k) == N_MUNI and k["code"].is_unique
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} the 2026 boundary layer has the same {len(b)} "
          f"municipality codes as 11rm, both ways" + ("" if good else f" {sorted(a ^ b)[:8]}"))
    pts = k[["code", "geometry"]].copy()
    pts["geometry"] = pts.geometry.representative_point()
    j = gpd.sjoin(pts, mk[["maakunta", "geometry"]], how="left", predicate="within")
    if j["code"].duplicated().any() or j["maakunta"].isna().any():
        raise SystemExit("a municipality fell in no maakunta or in two")
    region_of = {c: "MK" + str(m).zfill(2) for c, m in zip(j["code"], j["maakunta"])}
    regions = [a for a in rareas if a.startswith("MK")]
    good = len(regions) == N_MAAKUNTA and set(region_of.values()) == set(regions)
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} {len(regions)} maakunnat in 11rl, every one holding at "
          "least one municipality")
    # Exact where 11rm prints every municipality of the maakunta; elsewhere the printed cells
    # must not exceed the region's figure (fill_suppressed() then checks the residual's range)
    exact = over = 0
    bad = []
    for r in regions:
        mine = [m for m in munis if region_of[m] == r]
        for c in codes:
            vals = [mcube[(m, c)] for m in mine]
            s = sum(v for v in vals if v is not None)
            if None not in vals:
                exact += 1
                if s != rcube[(r, c)]:
                    bad.append((r, c, s, rcube[(r, c)]))
            elif s > rcube[(r, c)]:
                over += 1
                bad.append((r, c, s, rcube[(r, c)]))
    ok &= not bad
    print(f"  {'OK ' if not bad else 'BAD'} maakunta x code, 11rl against 11rm's municipalities: "
          f"{exact:,} cells with nothing suppressed match exactly, and in the other "
          f"{len(regions) * len(codes) - exact:,} the printed cells stay under the region's figure")
    for r, c, s, v in bad[:6]:
        print(f"        {r} / {c}: {s:,} summed, {v:,} published")

    est, good, lines = fill_suppressed(mcube, rcube, munis, region_of, regions, codes)
    ok &= good
    print("\n".join(lines))

    rows = []

    def emit(geo_id, level, name, cube, area, note):
        for c in leaves:
            n = cube[(area, c)]
            tier, nn = "measured", note
            if n is None:
                n, tier, nn = round(est[(area, c)], 4), "derived", \
                    note + "; under 10, suppressed: estimated from the maakunta's figure"
            if n == 0:
                continue
            rows.append(dict(geo_id=geo_id, geo_level=level, geo_name=name,
                             source_category=_leaf_label(c, llabels), count=n, tier=tier,
                             year=YEAR, source_id=SOURCE_ID, note=nn))

    emit("FI", "country", "Whole country", rcube, "SSS", "level=country; 11rl")
    for r in regions:
        emit(r, "maakunta", rlabels[r], rcube, r, "level=maakunta; 11rl")
    for m in munis:
        emit(m[2:], "municipality", " ".join(mlabels[m].split()), mcube, m,
             f"level=municipality; maakunta={region_of[m]}; 11rm")

    print("\n  codes, national (31 Dec 2025, 11rl):")
    total = rcube[("SSS", "SSS")]
    nonzero = [c for c in leaves if rcube[("SSS", c)] > 0]
    for c in sorted(nonzero, key=lambda c: -rcube[("SSS", c)])[:45]:
        v = rcube[("SSS", c)]
        print(f"    {v:>10,}  {100.0 * v / total:6.3f}%  {_leaf_label(c, llabels)}")
    print(f"    ... {len(nonzero)} codes with anyone, {len(leaves) - len(nonzero)} empty; "
          f"total {total:,}")
    if not ok:
        raise SystemExit("reconciliation FAILED")
    return rows


def main():
    if "--fetch" in sys.argv:
        fetch()
    rows = read()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print("\nwrote", OUT, f"({len(rows):,} rows)")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    main()
