"""Mauritania: boundaries and 2023 census populations for the 15 wilayas and 63 moughataas.

Writes data/geo/mr/mr_wilayas.gpkg, data/geo/mr/mr_moughataas.gpkg and data/geo/mr/mr_lookup.csv.

Two sources:

  * **boundaries**: COD-AB `cod-ab-mrt` (source ANSADE, updated on HDX 2026-01-26),
    `mrt_admin1.geojson` (15 wilayas, Nouakchott in three) and `mrt_admin2.geojson` (63
    moughataas). Not the `_em` edge-matched layers beside them, whose ADM2 names Brakna's Maal
    `0`.
  * **populations**: ANSADE, RGPH 2023 (RGPH-5), *Thème 1 : État et répartition spatiale de la
    population* (45 pages, text layer), the population de droit, 4,927,532:
      - p.3, *Principaux indicateurs*: each wilaya's urban, rural, nomad and total;
      - Tableau 1.5 (p.19): each wilaya's population and area in km2;
      - Tableau A.1.4 (pp.33-44): population by moughataa and commune.
    `ansade.mr/wp-content/...` returns the site's HTML shell; the file is under `admin.ansade.mr`.

## THE KEY IS ANSADE'S WILAYA ORDER

Both p.3 and Tableau 1.5 print the wilayas in the official order, which is COD's `adm1_pcode`
order (MR01 Hodh Chargui ... MR15 Nouakchott-Sud); each row is read against `WILAYAS` with a
fragment of its name as the witness (Tableau 1.5's text layer spells `Guidirnakha` and
`lnchiri`). Moughataas are NOT in pcode order everywhere (Nouakchott-Nord prints Teyaret, Dar
Naim, Toujounine; COD numbers them the other way), so they join on pinned names, `MOUGHATAAS`.

## CHECKS

  1. p.3: urban + rural + nomad = total in every wilaya, and each column sums to its printed
     national figure;
  2. Tableau 1.5's population equals p.3's in every wilaya, and its areas sum to its printed
     1,030,700;
  3. Tableau A.1.4 is read as a sum: a moughataa's communes add up to it, which is how the parser
     tells a moughataa row from a commune row, and each wilaya's moughataas add up to its printed
     total. Those totals differ from p.3 by one person in three wilayas (`A14_DIFF`, pinned);
  4. COD has 15 and 63 features, the name under every pcode is the expected one, every moughataa's
     `adm1_pcode` is its wilaya's, and the census's 63 moughataas and COD's 63 pair one to one;
  5. **area witness**: COD's area per wilaya over Tableau 1.5's, divided by the national ratio,
     inside `AREA_BAND`. Measured before this file was written: 0.95 (Tiris Zemmour) to 1.06
     (Guidimakha). The rank witness for the moughataa join, Kontur people against the census, is
     in `sources/mr_grid.py`.

Usage:
    python sources/mr_geo.py --fetch    COD-AB geojson zip (2.0 MB) and Thème 1 (1.3 MB)
    python sources/mr_geo.py            rebuild from data/raw/mr/
"""

import io
import os
import re
import sys
import unicodedata
import urllib.request
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ.setdefault("OMP_NUM_THREADS", "6")

import geopandas as gpd
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "mr")
OUT_DIR = os.path.join(ROOT, "data", "geo", "mr")
OUT_WILAYAS = os.path.join(OUT_DIR, "mr_wilayas.gpkg")
OUT_MOUGHATAAS = os.path.join(OUT_DIR, "mr_moughataas.gpkg")
LOOKUP = os.path.join(OUT_DIR, "mr_lookup.csv")

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/128.0 Safari/537.36"}

COD_URL = ("https://data.humdata.org/dataset/8d49f50d-92a8-46d9-9462-f821a8058f6d/resource/"
           "916850dc-92e3-4c2b-a972-f6964498346e/download/mrt_admin_boundaries.geojson.zip")
COD_ZIP = os.path.join(RAW, "mrt_admin_boundaries.geojson.zip")
THEME1_URL = ("https://admin.ansade.mr/wp-content/uploads/2026/04/"
              "Theme-1-Etat-et-repartition-spatiale-de-la-population_RGPH2023.pdf")
THEME1 = os.path.join(RAW, "Theme-1-Etat-et-repartition-spatiale-de-la-population_RGPH2023.pdf")

TOTAL = 4_927_532
URBAN, RURAL, NOMAD = 2_641_552, 2_266_457, 19_523      # p.3's national row
AREA_TOTAL = 1_030_700                                   # Tableau 1.5's national row, km2
AREA_ROWS = 1_036_000                                    # its 15 wilaya rows summed; 5,300 more
A14_PAGES = range(33, 45)                                # 1-based; printed page = pdf page - 5
A14_DIFF = {"MR01": -1, "MR09": 1, "MR12": 1}           # A.1.4 wilaya total minus p.3; asserted
# p.3 prints its components rounded: a wilaya's total minus urban + rural + nomad, and each column
# of wilaya rows minus the national row. Pinned from the first read; the totals are what is drawn.
P3_COMPONENT_DIFF = {"MR01": 1, "MR05": 1, "MR07": 1, "MR10": -1}
P3_COLUMN_DIFF = [-2, 0, -1, -1]
COMMUNE_TOL = 3                                          # a moughataa closes when its communes are this close
AREA_BAND = (0.90, 1.10)

# pcode -> (the name this map prints, COD adm1_name, fragment every PDF row must contain)
WILAYAS = {
    "MR01": ("Hodh Chargui", "Hodh Chargui", "Chargui"),
    "MR02": ("Hodh El Gharbi", "Hodh El Gharbi", "Gharbi"),
    "MR03": ("Assaba", "Assaba", "Assaba"),
    "MR04": ("Gorgol", "Gorgol", "Gorgol"),
    "MR05": ("Brakna", "Brakna", "Brakna"),
    "MR06": ("Trarza", "Trarza", "Trarza"),
    "MR07": ("Adrar", "Adrar", "Adrar"),
    "MR08": ("Dakhlet Nouadhibou", "Dakhlet Nouadhibou", "Nouad"),
    "MR09": ("Tagant", "Tagant", "Tagant"),
    "MR10": ("Guidimakha", "Guidimagha", "Guid"),
    "MR11": ("Tiris Zemmour", "Tiris Zemmour", "Zemmour"),
    "MR12": ("Inchiri", "Inchiri", "nchiri"),
    "MR13": ("Nouakchott-Ouest", "Nouakchott Ouest", "Ouest"),
    "MR14": ("Nouakchott-Nord", "Nouakchott Nord", "Nord"),
    "MR15": ("Nouakchott-Sud", "Nouakchott Sud", "Sud"),
}

# COD adm2_pcode -> (Tableau A.1.4's spelling, COD adm2_name)
MOUGHATAAS = {
    "MR011": ("Amourj", "Amourj"), "MR012": ("Bassiknou", "Bassiknou"),
    "MR013": ("Djigueni", "Djiguenni"), "MR014": ("Nema", "Néma"),
    "MR015": ("Oualata", "Oualata"), "MR016": ("Timbédra", "Timbédra"),
    "MR017": ("Nbeiket Lahwache", "N'Beiket Lehwach"), "MR018": ("Adel Bagrou", "Adel Bagrou"),
    "MR021": ("Aïoun", "Aïoun"), "MR022": ("Koubenni", "Kobeni"),
    "MR023": ("Tamcheket", "Tamchekett"), "MR024": ("Tintane", "Tintane"),
    "MR025": ("Touil", "Touil"),
    "MR031": ("Barkewol", "Barkéol"), "MR032": ("Boumdeid", "Boumdeid"),
    "MR033": ("Guerou", "Guerou"), "MR034": ("Kankoussa", "Kankoussa"), "MR035": ("Kiffa", "Kiffa"),
    "MR041": ("Kaedi", "Kaedi"), "MR042": ("Maghama", "Maghama"), "MR043": ("Mbout", "M'Bout"),
    "MR044": ("Mounguel", "Mounguel"), "MR045": ("Lexeiba1", "Lexeibe 1"),
    "MR051": ("Aleg", "Aleg"), "MR052": ("Bababé", "Bababé"), "MR053": ("Boghé", "Boghé"),
    "MR054": ("Magta lahjar", "Magtalahjar"), "MR055": ("Mbagne", "M'Bagne"),
    "MR056": ("Maal", "Maal"),
    "MR061": ("Boutilimit", "Boutilimit"), "MR062": ("Keur Macene", "Keur Macen"),
    "MR063": ("Mederdra", "Mederdra"), "MR064": ("Ouad Naga", "Ouad Naga"),
    "MR065": ("Rkiz", "R'Kiz"), "MR066": ("Rosso", "Rosso"), "MR067": ("Tekane", "Tekane"),
    "MR071": ("Aoujeft", "Aoujeft"), "MR072": ("Atar", "Atar"),
    "MR073": ("Chinguitti", "Chinguitti"), "MR074": ("Ouadane", "Ouadane"),
    "MR081": ("Nouadhibou", "Nouadhibou"), "MR082": ("Chami", "Chami"),
    "MR091": ("Moudjeria", "Moudjeria"), "MR092": ("Tichit", "Tichit"),
    "MR093": ("Tidjikdja", "Tidjikja"),
    "MR101": ("Ould Yenge", "Ould Yengé"), "MR102": ("Sélibaby", "Sélibaby"),
    "MR103": ("Ghabou", "Ghabou"), "MR104": ("Wompou", "Wompou"),
    "MR111": ("Bir Mogrein", "Bir Moughrein"), "MR112": ("Fdeirik", "F'Deirick"),
    "MR113": ("Zoueratt", "Zoueirat"),
    "MR121": ("Akjoujt", "Akjoujt"), "MR122": ("Bennechab", "Bennechab"),
    "MR131": ("Ksar", "Ksar"), "MR132": ("Tevragh Zeina", "Tevragh Zeina"),
    "MR133": ("Sebkha", "Sebkha"),
    "MR141": ("Toujounine", "Toujounine"), "MR142": ("Dar Naim", "Dar Naïm"),
    "MR143": ("Teyaret", "Teyarett"),
    "MR151": ("Elmina", "El Mina"), "MR152": ("Arafat", "Arafat"), "MR153": ("Riyad", "Riad"),
}

HEADER = {"Population", "Moughataa", "Commune", "Masculin", "Féminin", "Total"}
EQUAL_AREA = "EPSG:6933"


def fold(s):
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower()
    return re.sub(r"[^a-z0-9]", "", s)


def num(s):
    t = s.replace(" ", "").replace("\xa0", "").replace(" ", "").strip()
    if t == "-":
        return 0
    if not t.isdigit():
        raise ValueError(s)
    return int(t)


def isnum(s):
    return bool(re.fullmatch(r"[\d  \xa0]+", s)) and any(ch.isdigit() for ch in s)


def _get(url, dst, magic, minsize):
    if os.path.exists(dst) and os.path.getsize(dst) > minsize:
        print(f"  have {os.path.basename(dst)} ({os.path.getsize(dst):,} bytes)")
        return
    print("  GET", url)
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=600) as r:
        data = r.read()
    if not data.startswith(magic):
        raise SystemExit(f"{url} did not return the expected file; starts {data[:24]!r}")
    with open(dst + ".part", "wb") as f:
        f.write(data)
    os.replace(dst + ".part", dst)
    print(f"  got  {os.path.basename(dst)} ({os.path.getsize(dst):,} bytes)")


def fetch():
    os.makedirs(RAW, exist_ok=True)
    _get(COD_URL, COD_ZIP, b"PK", 1_000_000)
    _get(THEME1_URL, THEME1, b"%PDF", 1_000_000)


def page_lines(doc, p):
    return [ln.strip() for ln in doc[p - 1].get_text().splitlines() if ln.strip()]


def rows_after(lines, anchor, width):
    """The 15 wilaya rows after the line `anchor`, each a name and `width` numbers."""
    i = lines.index(anchor) + 1
    out = {}
    for pc, (_name, _cod, frag) in WILAYAS.items():
        name = lines[i]
        if frag not in name or isnum(name):
            raise SystemExit(f"after {anchor!r}: row {name!r} should be {pc} (fragment {frag!r})")
        out[pc] = lines[i + 1:i + 1 + width]
        i += 1 + width
    return out, lines[i:]


def indicators(doc):
    """p.3: {pcode: (urban, rural, nomad, total)}."""
    lines = page_lines(doc, 3)
    k = lines.index("Population Totale")
    nat = [num(x) for x in lines[k + 1:k + 5]]
    if nat != [URBAN, RURAL, NOMAD, TOTAL]:
        raise SystemExit(f"p.3's national row is {nat}, pinned {[URBAN, RURAL, NOMAD, TOTAL]}")
    rows, _rest = rows_after(lines, "Population par Wilaya", 4)
    out = {pc: tuple(num(x) for x in v) for pc, v in rows.items()}
    diff = {pc: t - (u + r + n) for pc, (u, r, n, t) in out.items() if u + r + n != t}
    sums = [sum(v[j] for v in out.values()) for j in range(4)]
    col = [s - x for s, x in zip(sums, nat)]
    print(f"  p.3: total minus urban + rural + nomad, where not 0: {diff}; wilaya rows minus the "
          f"national row (urban, rural, nomad, total): {col}")
    if diff != P3_COMPONENT_DIFF or col != P3_COLUMN_DIFF:
        raise SystemExit(f"p.3's rounding is {diff} and {col}, pinned {P3_COMPONENT_DIFF} and "
                         f"{P3_COLUMN_DIFF}")
    return out


def densities(doc):
    """Tableau 1.5 (p.19): {pcode: (population, area km2)}."""
    lines = page_lines(doc, 19)
    rows, rest = rows_after(lines, "Densité (hbts/km2)", 4)
    out = {pc: (num(v[0]), num(v[1])) for pc, v in rows.items()}
    if rest[0] != "Mauritanie" or num(rest[1]) != TOTAL or num(rest[2]) != AREA_TOTAL:
        raise SystemExit(f"Tableau 1.5's national row is {rest[:3]}")
    rows_area = sum(a for _p, a in out.values())
    if rows_area != AREA_ROWS:
        raise SystemExit(f"Tableau 1.5's wilaya areas sum to {rows_area:,}, pinned {AREA_ROWS:,}")
    print(f"  Tableau 1.5: wilaya areas sum to {rows_area:,} km2 against its national row's "
          f"{AREA_TOTAL:,} (pinned); the witness below uses the rows")
    return out


def moughataa_table(doc):
    """Tableau A.1.4: ({pcode of wilaya: printed total}, [(wilaya pcode, moughataa name, total)])."""
    by_fold = {fold(n): pc for pc, (n, _c, _f) in WILAYAS.items()}
    tokens = []
    for p in A14_PAGES:
        lines = page_lines(doc, p)
        if lines and lines[0] == str(p - 5):
            lines = lines[1:]
        for ln in lines:
            if ln.startswith("Tableau A.1. 4"):
                continue
            if ln.startswith("Source"):
                break
            tokens.append(ln)

    wil_totals, mough = {}, []
    wil, open_m, national = None, None, None
    pending = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        if t in HEADER:
            pending = []
            i += 1
            continue
        w = by_fold.get(fold(re.sub(r"^Wilaya\s*:\s*", "", t)))
        if w and nxt == "Population":
            if wil is not None and wil not in wil_totals:
                raise SystemExit(f"{wil} ended without a total row")
            wil, open_m, pending = w, None, []
            i += 1
            continue
        if not isnum(t):
            pending.append(t)
            i += 1
            continue
        vals = tokens[i:i + 3]
        if len(vals) < 3 or not all(isnum(v) for v in vals):
            raise SystemExit(f"A.1.4: numbers after {pending!r} are {vals!r}")
        m, f, tot = (num(v) for v in vals)
        if m + f != tot and abs(m + f - tot) > 1:
            raise SystemExit(f"A.1.4 {pending!r}: {m} + {f} != {tot}")
        i += 3
        if not pending:
            raise SystemExit(f"A.1.4: three numbers with no name before them ({vals})")
        name = pending[-1]
        names = pending[-2:] if len(pending) >= 2 else pending
        pending = []
        if name == "Mauritanie":
            national = tot
            continue
        if name.startswith("Total") or by_fold.get(fold(name)) == wil:
            if open_m is not None and open_m[2] - open_m[3] > COMMUNE_TOL:
                raise SystemExit(f"{wil}: moughataa {open_m[1]} has communes for "
                                 f"{open_m[3]:,} of {open_m[2]:,}")
            wil_totals[wil] = tot
            open_m = None
            continue
        # A moughataa and its one commune of the same name printed as two name lines, one row.
        if len(names) == 2 and fold(names[0]) == fold(names[1]):
            open_m = [wil, names[0], tot, tot]
            mough.append((wil, names[0], tot))
            continue
        if open_m is None or open_m[2] - open_m[3] <= COMMUNE_TOL:
            if open_m is not None and open_m[2] != open_m[3]:
                print(f"    note: {open_m[1]}'s communes sum to {open_m[3]:,} against {open_m[2]:,}")
            open_m = [wil, name, tot, 0]
            mough.append((wil, name, tot))
        else:
            if open_m[3] + tot > open_m[2] + COMMUNE_TOL:
                raise SystemExit(f"{wil}: commune {name} ({tot:,}) overruns moughataa "
                                 f"{open_m[1]} ({open_m[3]:,} of {open_m[2]:,})")
            open_m[3] += tot
    if national != TOTAL:
        raise SystemExit(f"A.1.4's national row is {national}, expected {TOTAL:,}")
    if sorted(wil_totals) != sorted(WILAYAS):
        raise SystemExit(f"A.1.4 closed wilayas {sorted(wil_totals)}")
    for w in WILAYAS:
        s = sum(t for ww, _n, t in mough if ww == w)
        if abs(s - wil_totals[w]) > COMMUNE_TOL:
            raise SystemExit(f"A.1.4 {w}: moughataas sum to {s:,}, total row {wil_totals[w]:,}")
    return wil_totals, mough


def main():
    import fitz

    if "--fetch" in sys.argv:
        fetch()
    for p in (COD_ZIP, THEME1):
        if not os.path.exists(p):
            raise SystemExit(f"{p} missing; run with --fetch")
    with open(THEME1, "rb") as fh:
        if b"%%EOF" not in fh.read()[-2048:]:
            raise SystemExit(f"{THEME1} has no %%EOF trailer; the download is truncated")
    doc = fitz.open(THEME1)
    if len(doc) != 45:
        raise SystemExit(f"Thème 1 has {len(doc)} pages, expected 45")

    # ---- 1 and 2: the wilaya populations, two tables ----
    ind = indicators(doc)
    den = densities(doc)
    bad = {pc: (ind[pc][3], den[pc][0]) for pc in WILAYAS if ind[pc][3] != den[pc][0]}
    if bad:
        raise SystemExit(f"p.3 and Tableau 1.5 disagree: {bad}")
    print(f"Thème 1: 15 wilayas, {TOTAL:,} people ({URBAN:,} urban, {RURAL:,} rural, {NOMAD:,} "
          "nomad); p.3 and Tableau 1.5 agree in every wilaya")

    # ---- 3: the moughataa table ----
    wil_totals, mough = moughataa_table(doc)
    diff = {w: wil_totals[w] - ind[w][3] for w in WILAYAS if wil_totals[w] != ind[w][3]}
    if diff != A14_DIFF:
        raise SystemExit(f"A.1.4's wilaya totals minus p.3: {diff}, pinned {A14_DIFF}")
    print(f"  Tableau A.1.4: {len(mough)} moughataas, each wilaya's summing to its total row "
          f"(which is p.3's figure except {A14_DIFF}, pinned)")

    # ---- 4: COD ----
    with zipfile.ZipFile(COD_ZIP) as zf:
        a1 = gpd.read_file(io.BytesIO(zf.read("mrt_admin1.geojson")))
        a2 = gpd.read_file(io.BytesIO(zf.read("mrt_admin2.geojson")))
    if len(a1) != 15 or len(a2) != 63 or a1.crs.to_epsg() != 4326 or a2.crs.to_epsg() != 4326:
        raise SystemExit(f"COD-AB: {len(a1)} ADM1 and {len(a2)} ADM2 features; expected 15 and 63")
    got1 = dict(zip(a1["adm1_pcode"], a1["adm1_name"]))
    got2 = dict(zip(a2["adm2_pcode"], a2["adm2_name"]))
    bad = [pc for pc, (_n, c, _f) in WILAYAS.items() if got1.get(pc) != c]
    bad += [pc for pc, (_n, c) in MOUGHATAAS.items() if got2.get(pc) != c]
    bad += [pc for pc, par in zip(a2["adm2_pcode"], a2["adm1_pcode"]) if pc[:4] != par]
    if bad or len(got2) != len(MOUGHATAAS):
        raise SystemExit(f"COD names or parents under these pcodes are not the expected ones: {bad}")
    print("  COD-AB: 15 wilayas and 63 moughataas, every name and parent under its pcode as expected")

    census = {}
    for w, name, tot in mough:
        hit = [pc for pc, (n, _c) in MOUGHATAAS.items() if pc[:4] == w and fold(n) == fold(name)]
        if len(hit) != 1:
            raise SystemExit(f"A.1.4 moughataa {name!r} in {w} matches {hit}")
        if hit[0] in census:
            raise SystemExit(f"two A.1.4 rows join {hit[0]}")
        census[hit[0]] = tot
    if sorted(census) != sorted(MOUGHATAAS):
        raise SystemExit(f"COD moughataas with no census row: {sorted(set(MOUGHATAAS) - set(census))}")
    print("  the census's 63 moughataas and COD's 63 pair one to one on the pinned names")

    # ---- 5: areas ----
    a1 = a1.set_index("adm1_pcode")
    cod_area = a1.to_crs(EQUAL_AREA).geometry.area / 1e6
    nat_ratio = cod_area.sum() / AREA_ROWS
    rel = {pc: (cod_area[pc] / den[pc][1]) / nat_ratio for pc in WILAYAS}
    print(f"  COD area over Tableau 1.5's, national {nat_ratio:.4f}; per wilaya over that:")
    for pc in sorted(rel, key=rel.get):
        print(f"      {WILAYAS[pc][0]:<20} {cod_area[pc]:>9,.0f} / {den[pc][1]:>9,}  {rel[pc]:.3f}")
    out_band = {pc: round(r, 3) for pc, r in rel.items() if not AREA_BAND[0] <= r <= AREA_BAND[1]}
    if out_band:
        raise SystemExit(f"wilayas outside the area band {AREA_BAND}: {out_band}")
    a2u = a2.dissolve("adm1_pcode").to_crs(EQUAL_AREA).geometry.area / 1e6
    worst = max(abs(a2u[pc] / cod_area[pc] - 1) for pc in WILAYAS)
    print(f"  ADM2 dissolved to wilayas against ADM1: largest area difference {worst:.2%}")

    lut = pd.DataFrame([dict(geo_id=pc, unit=pc, name=n, pop=ind[pc][3], urban=ind[pc][0],
                             rural=ind[pc][1], nomad=ind[pc][2], area_census=den[pc][1],
                             cod_area=round(float(cod_area[pc]), 1))
                        for pc, (n, _c, _f) in WILAYAS.items()])
    os.makedirs(OUT_DIR, exist_ok=True)
    g1 = a1.reset_index()[["adm1_pcode", "geometry"]].rename(columns={"adm1_pcode": "unit"})
    g1 = g1.merge(lut[["unit", "name", "pop"]], on="unit", how="inner", validate="1:1")
    g1[["unit", "name", "pop", "geometry"]].to_file(OUT_WILAYAS, layer="wilayas", driver="GPKG")
    g2 = a2[["adm2_pcode", "adm2_name", "adm1_pcode", "geometry"]].rename(
        columns={"adm2_pcode": "moughataa", "adm2_name": "cod_name", "adm1_pcode": "unit"})
    g2["name"] = g2["moughataa"].map(lambda pc: MOUGHATAAS[pc][0])
    g2["pop"] = g2["moughataa"].map(census)
    g2[["moughataa", "unit", "name", "cod_name", "pop", "geometry"]].to_file(
        OUT_MOUGHATAAS, layer="moughataas", driver="GPKG")
    lut.to_csv(LOOKUP, index=False, encoding="utf-8")
    print(f"\nwrote {OUT_WILAYAS} (15), {OUT_MOUGHATAAS} (63, {int(g2['pop'].sum()):,} people) and "
          f"{LOOKUP} ({int(lut['pop'].sum()):,} people)")


if __name__ == "__main__":
    main()
