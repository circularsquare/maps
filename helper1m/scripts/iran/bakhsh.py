"""District (bakhsh) level for helper1m Iran: the 1,057 districts of the 1395 census, drawn with
OpenStreetMap's district polygons of October 2026 and regrouped onto the 1395 districts.

Why it is not a plain name join: since 1395 about 40 counties and many districts have been
created, so an OSM district's name often is not a 1395 district's name (a district promoted to
a county comes back as that county's "Central District"). The link goes through SCI's own
settlement file for the 1400 divisions (GEO1400.xlsx), which lists every village of every 1400
district with the same 6-digit village code the 1395 census uses:

  OSM district polygon --(county + district name)--> 1400 district
  1400 district --(its villages' codes, its cities' names)--> 1395 districts, with 1395 people

Each OSM polygon is cut on COD's 1395 county lines first, so the level nests in the counties.
A piece goes to the 1395 district that holds most of its 1400 district's people inside that
county. When a piece's people come from two 1395 districts (no line between them exists in
OSM), those districts are merged into one unit, the same rule Pakistan used (merge to the
common piece, never split). A 1395 district that receives no piece is merged into the unit
that holds most of its people in 1400 terms.

Run after fetch.py has run once (it needs the 1395 tables) and osm_admin.py:
    C:\\Python39\\python.exe helper1m\\scripts\\iran\\bakhsh.py
then fetch.py again, which adds level 3 to population.csv.

Writes helper1m/data/iran/boundaries/adm{1,2,3}.gpkg, data/iran/bakhsh_units.csv and
data/iran/bakhsh_pieces.csv (every OSM piece, its 1400 district and where it went).
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "6")

import csv  # noqa: E402
import difflib  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent))
import census  # noqa: E402
from census import fold  # noqa: E402

HELPER = Path(__file__).resolve().parents[2]
REPO = HELPER.parent
DATA = HELPER / "data" / "iran"
RAW = DATA / "raw"
GEO1400 = RAW / "sci" / "GEO1400.xlsx"
OSM = DATA / "osm_admin.gpkg"
OUTB = DATA / "boundaries"
WD_CACHE = RAW / "wikidata_labels.json"
UNITS_CSV = DATA / "bakhsh_units.csv"
PIECES_CSV = DATA / "bakhsh_pieces.csv"
EQ = "ESRI:54009"
MIN_PIECE_KM2 = 2.0      # OSM/COD line mismatch slivers below this follow their neighbour
MIN_PIECE_SHARE = 0.02   # ... or below this share of the OSM district
MERGE_SHARE = 0.15       # a second 1395 district this large in a piece forces a merge


_TR = {"ا": "a", "آ": "a", "ب": "b", "پ": "p", "ت": "t", "ث": "s", "ج": "j", "چ": "ch", "ح": "h",
       "خ": "kh", "د": "d", "ذ": "z", "ر": "r", "ز": "z", "ژ": "zh", "س": "s", "ش": "sh", "ص": "s",
       "ض": "z", "ط": "t", "ظ": "z", "ع": "", "غ": "gh", "ف": "f", "ق": "q", "ک": "k", "گ": "g",
       "ل": "l", "م": "m", "ن": "n", "و": "v", "ه": "h", "ی": "i", "ء": "", "أ": "a", "ئ": "i",
       " ": " ", "‌": "-"}


def translit(s):
    """A rough letter-for-letter romanisation for the few districts with no English name
    anywhere (Persian script leaves out short vowels, so 'Bazrgan' for Bazargan). Marked with
    a trailing * in the name."""
    s = str(s).translate(str.maketrans({"ي": "ی", "ك": "ک", "ة": "ه"}))
    out = "".join(_TR.get(ch, "") for ch in s)
    out = re.sub(r"(?<=[^aeiou ])v(?=[^aeiou ]|$)", "u", out)
    return " ".join(w.capitalize() for w in out.split())


def strip_prefix(s, words):
    s = fold(s)
    for w in words:
        w = fold(w)
        if s.startswith(w):
            s = s[len(w):]
    return s


def bname(s):
    """District name key. OSM writes a central district as 'بخش مرکزی', 'بخش مرکزی خمام' or
    'بخش مرکزی شهرستان جاسک'; SCI as 'مرکزي'. All become 'مرکزی'."""
    s = strip_prefix(s or "", ["بخش"])
    return fold("مرکزی") if s.startswith(fold("مرکزی")) else s


def cname(s):
    return strip_prefix(s or "", ["شهرستان"])


# ----------------------------------------------------------------------------- 1395 side
def load95_detail():
    """village code -> (T, pop, name) and (nn, city key) -> (T, pop) for 1395."""
    from fetch import city_key
    vill, city = {}, {}
    for i in range(31):
        nn = f"{i:02d}"
        for r in census.read95(nn):
            T = (nn, r["cty"], r["bkh"])
            if r["kind"] == "abadi":
                vill[(nn, r["abadi"])] = (T, r["pop"] or 0, r["name"])
            elif r["kind"] == "city":
                city[(nn, city_key(r["name"]))] = (T, r["pop"] or 0)
    return vill, city


def load1400(vill95, city95):
    """1400 districts with their names and their people in 1395 districts."""
    import openpyxl
    from fetch import city_key

    wb = openpyxl.load_workbook(GEO1400, read_only=True)
    ws = wb.worksheets[0]
    d = {}
    for i, r in enumerate(ws.iter_rows(values_only=True)):
        if i == 0:
            continue
        ost, oname, cty, cnm, bkh, bnm, unit, unm, abd, rec, name, diag = r
        rec = str(rec)
        if rec in ("1", "2"):
            continue
        key = (ost, cty, bkh)
        if key not in d:
            d[key] = {"county": cnm, "name": bnm, "comp": defaultdict(float), "n": 0}
        if rec in ("6", "8") and abd:
            hit = vill95.get((ost, abd))
            if hit:
                d[key]["comp"][hit[0]] += hit[1]
                d[key]["n"] += 1
        elif rec == "5":
            hit = city95.get((ost, city_key(name)))
            if hit:
                d[key]["comp"][hit[0]] += hit[1]
                d[key]["n"] += 1
    wb.close()
    return d


# ----------------------------------------------------------------------------- Wikidata
def wikidata_labels(qids):
    import requests

    cache = json.loads(WD_CACHE.read_text(encoding="utf-8")) if WD_CACHE.exists() else {}
    todo = sorted({q for q in qids if q and q not in cache})
    for i in range(0, len(todo), 50):
        batch = todo[i:i + 50]
        for attempt in range(8):
            r = requests.get("https://www.wikidata.org/w/api.php",
                             params={"action": "wbgetentities", "ids": "|".join(batch),
                                     "props": "labels", "languages": "en", "format": "json",
                                     "maxlag": 5},
                             headers={"User-Agent": "helper1m-build/1.0 (map research)"},
                             timeout=60)
            if r.status_code == 200 and "entities" in r.json():
                break
            wait = int(r.headers.get("Retry-After", 30 * (attempt + 1)))
            print(f"    wikidata {r.status_code}, waiting {wait}s")
            time.sleep(wait)
        else:
            raise SystemExit("Wikidata kept refusing")
        for q, e in r.json().get("entities", {}).items():
            cache[q] = e.get("labels", {}).get("en", {}).get("value")
        WD_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=0), encoding="utf-8")
        time.sleep(5)
    return cache


WD_SEARCH = RAW / "wikidata_search.json"


def wikidata_search_district(name_fa):
    """English label of the Wikidata item whose Persian label is 'بخش <name>', for a district
    no OSM polygon names. Only an English label ending in 'District' is taken."""
    import requests

    cache = json.loads(WD_SEARCH.read_text(encoding="utf-8")) if WD_SEARCH.exists() else {}
    q = "بخش " + str(name_fa).translate(str.maketrans({"ي": "ی", "ك": "ک", "ئ": "ی"}))
    if q not in cache:
        res = None
        for attempt in range(6):
            r = requests.get("https://www.wikidata.org/w/api.php",
                             params={"action": "wbsearchentities", "search": q, "language": "fa",
                                     "type": "item", "limit": 5, "format": "json",
                                     "uselang": "en"},
                             headers={"User-Agent": "helper1m-build/1.0 (map research)"},
                             timeout=60)
            if r.status_code == 200:
                for hit in r.json().get("search", []):
                    m = hit.get("match", {})
                    t = fold(m.get("text", ""))
                    if m.get("language") != "fa" or not (
                            t == fold(q) or t.startswith(fold(q + " شهرستان"))):
                        # only the exact Persian label, which Wikidata often writes with its
                        # county ('بخش آبدان شهرستان دیر'); no 'North Kamfiruz' for 'Kar'
                        continue
                    if hit.get("label", "").startswith("Central District"):
                        continue
                    lab = hit.get("display", {}).get("label", {})
                    if lab.get("language") == "en" and lab.get("value", "").endswith("District"):
                        res = lab["value"]
                        break
                    # the search returns the label in the asked UI language when it exists
                    if hit.get("label", "").endswith("District"):
                        res = hit["label"]
                        break
                break
            time.sleep(int(r.headers.get("Retry-After", 20 * (attempt + 1))))
        cache[q] = res
        WD_SEARCH.write_text(json.dumps(cache, ensure_ascii=False, indent=0), encoding="utf-8")
        time.sleep(2)
    return cache[q]


def located_composition(pcs, c_of, a2):
    """For each piece, {1395 district: 1395 people} of the 1395 cities and villages whose name
    matches exactly one OSM place node inside their own COD county."""
    import geopandas as gpd
    from fetch import city_key

    pl = gpd.read_file(DATA / "osm_places.gpkg").to_crs(4326)
    pl = gpd.sjoin(pl, a2[["adm2_pcode", "geometry"]].to_crs(4326), how="inner",
                   predicate="within").drop(columns="index_right")
    pl = pl.reset_index(drop=True)
    keys = defaultdict(list)
    for i, r in pl.iterrows():
        for nm in {r["name"], r["name_fa"]}:
            if isinstance(nm, str):
                keys[(r["adm2_pcode"], fold(nm))].append(i)
                keys[(r["adm2_pcode"], city_key(nm))].append(i)
    keys = {k: set(v) for k, v in keys.items()}
    pts = []
    for i in range(31):
        nn = f"{i:02d}"
        for r in census.read95(nn):
            if r["kind"] not in ("city", "abadi") or not r["pop"]:
                continue
            pc = c_of[(nn, r["cty"])]
            k = city_key(r["name"]) if r["kind"] == "city" else fold(r["name"])
            hit = keys.get((pc, k), set())
            if len(hit) == 1:
                pts.append(((nn, r["cty"], r["bkh"]), r["pop"], next(iter(hit))))
    print(f"located settlements: {len(pts):,} 1395 settlements with "
          f"{sum(p[1] for p in pts):,} people found as one OSM place node in their county")
    loc = gpd.GeoDataFrame({"T": [p[0] for p in pts], "pop": [p[1] for p in pts]},
                           geometry=pl.geometry.iloc[[p[2] for p in pts]].values, crs=4326)
    j = gpd.sjoin(loc, pcs[["geometry"]].to_crs(4326), how="inner", predicate="within")
    comp = defaultdict(lambda: defaultdict(float))
    for _, r in j.iterrows():
        comp[r["index_right"]][r["T"]] += r["pop"]
    return [dict(comp.get(i, {})) for i in pcs.index]


# ----------------------------------------------------------------------------- build
def build():
    import geopandas as gpd
    import pandas as pd
    import fetch

    prov95, counties95, bakhshs95, c95, v95 = fetch.load95()
    p_of, c_of, a1, a2 = fetch.cod_join(prov95, counties95)
    nn_of_p = {v: k for k, v in p_of.items()}
    vill95, city95 = load95_detail()
    d1400 = load1400(vill95, city95)
    print(f"GEO1400: {len(d1400)} districts of 1400; "
          f"{sum(1 for v in d1400.values() if v['comp'])} reach a 1395 district through "
          "village codes or city names")

    osm = gpd.read_file(OSM)
    b = osm[osm.admin_level == "6"].copy().to_crs(4326)
    c = osm[osm.admin_level == "5"].copy().to_crs(4326)
    b = b[b.geometry.notna() & ~b.geometry.is_empty].reset_index(drop=True)
    b["bid"] = range(len(b))
    rp = b.copy()
    rp["geometry"] = b.geometry.representative_point()
    # OSM county of each district (for its name), COD province (for the SCI province code)
    b["geometry"] = b.geometry.make_valid()
    j = gpd.sjoin_nearest(rp[["bid", "geometry"]].to_crs(EQ), c[["name", "geometry"]].rename(
        columns={"name": "osm_county"}).to_crs(EQ), how="left")
    j = j[~j.bid.duplicated()].set_index("bid")
    b["osm_county"] = b["bid"].map(j["osm_county"])
    # nearest, not within: a coastal district's point can fall just outside COD's coastline
    j = gpd.sjoin_nearest(rp[["bid", "geometry"]].to_crs(EQ),
                          a1[["adm1_pcode", "geometry"]].to_crs(EQ), how="left")
    j = j[~j.bid.duplicated()].set_index("bid")
    b["nn"] = b["bid"].map(j["adm1_pcode"]).map(nn_of_p)

    # ---- OSM district -> 1400 district, by county name + district name inside the province
    idx = defaultdict(list)
    for k, v in d1400.items():
        idx[k[0]].append((cname(v["county"]), bname(v["name"]), k))
    match, how = {}, {}
    for _, r in b.iterrows():
        if pd.isna(r["nn"]):
            continue
        cands = idx[r["nn"]]
        on, oc = bname(r["name"]), cname(r["osm_county"]) if isinstance(r["osm_county"], str) else ""
        counties14 = {cn for cn, _, _ in cands}
        if oc not in counties14:
            # county spelling differs between SCI and OSM ('چاه بهار' / 'چابهار')
            best = difflib.get_close_matches(oc, list(counties14), n=1, cutoff=0.8)
            if best:
                oc = best[0]
        hit = [k for cn, bn, k in cands if bn == on and cn == oc]
        h = "exact"
        if len(hit) != 1:
            # a district name is often unique in its province on its own
            hit2 = [k for cn, bn, k in cands if bn == on]
            if len(hit2) == 1:
                hit, h = hit2, "name in province"
        if len(hit) != 1:
            # spelling: closest name within the same county
            same_c = [(bn, k) for cn, bn, k in cands if cn == oc]
            best = difflib.get_close_matches(on, [x[0] for x in same_c], n=1, cutoff=0.85)
            if best:
                hit, h = [k for bn, k in same_c if bn == best[0]], "close name in county"
        if len(hit) == 1:
            match[r["bid"]] = hit[0]
            how[r["bid"]] = h
    print(f"OSM districts: {len(b)}; matched to a 1400 district {len(match)} "
          f"({dict(pd.Series(list(how.values())).value_counts())})")

    # ---- cut on COD counties
    a2c = a2[["adm2_pcode", "geometry"]].to_crs(4326)
    a2c["geometry"] = a2c.geometry.make_valid()
    pcs = gpd.overlay(b[["bid", "name", "osm_county", "nn", "wikidata", "geometry"]], a2c,
                      how="intersection", keep_geom_type=True)
    pcs["km2"] = pcs.to_crs(EQ).area / 1e6
    tot = pcs.groupby("bid")["km2"].transform("sum")
    pcs["share"] = pcs["km2"] / tot
    sci_of = {v: k for k, v in c_of.items()}  # adm2_pcode -> (nn, cty)

    # composition of each piece in 1395 districts of its own COD county
    rows = []
    for _, p in pcs.iterrows():
        nn, cty = sci_of[p["adm2_pcode"]]
        comp = {}
        k14 = match.get(p["bid"])
        if k14:
            comp = {T: x for T, x in d1400[k14]["comp"].items() if T[0] == nn and T[1] == cty}
        rows.append({"comp": comp, "k14": k14})
    pcs["comp"] = [r["comp"] for r in rows]
    pcs["k14"] = [r["k14"] for r in rows]

    # ---- a second composition that needs no district names: 1395 settlements located by
    # an OSM place node of the same name inside their COD county
    pcs["comp_sp"] = located_composition(pcs, c_of, a2)
    # where the name route found nothing, the located settlements decide
    filled = overruled = 0
    for i in pcs.index:
        sp = pcs.at[i, "comp_sp"]
        if not pcs.at[i, "comp"] and sp:
            pcs.at[i, "comp"] = sp
            filled += 1
        elif pcs.at[i, "comp"] and sp and sum(sp.values()) >= 1000:
            # the name route can pick the wrong 1400 district when OSM still uses an old name
            # (Jarquyeh Sofla, now its county's 'Central'); located settlements overrule it
            # when they clearly point elsewhere
            cm = pcs.at[i, "comp"]
            top_n = max(cm, key=cm.get)
            top_s = max(sp, key=sp.get)
            if top_s != top_n and sp[top_s] / sum(sp.values()) >= 0.6:
                pcs.at[i, "comp"] = sp
                overruled += 1
    print(f"located settlements overruled the name route on {overruled} pieces")
    agree_n = agree_d = 0
    for i, p in pcs.iterrows():
        if p["k14"] and p["comp_sp"] and p["comp"]:
            top = max(p["comp"], key=p["comp"].get)
            agree_n += p["comp_sp"].get(top, 0)
            agree_d += sum(p["comp_sp"].values())
    print(f"located settlements: {filled} pieces with no 1400 match took their composition from "
          f"them; on pieces matched by name, {agree_n / max(agree_d, 1):.1%} of the located "
          "1395 people sit in the 1395 district the name route chose")

    # ---- assign pieces to 1395 districts; collect forced merges
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    assign = {}
    sliver = (pcs["km2"] < MIN_PIECE_KM2) | (pcs["share"] < MIN_PIECE_SHARE)
    for i, p in pcs.iterrows():
        comp = p["comp"]
        if not comp or sum(comp.values()) == 0:
            continue
        tot_c = sum(comp.values())
        ranked = sorted(comp.items(), key=lambda kv: -kv[1])
        assign[i] = ranked[0][0]
        if not sliver[i]:
            for T, x in ranked[1:]:
                if x / tot_c >= MERGE_SHARE:
                    union(T, ranked[0][0])
    # pieces with no composition (unmatched OSM district, or slivers outside their 1400 county):
    # take a 1395 district by name in the COD county, else the neighbour they share most border
    # with (done after, once neighbours are assigned)
    nm95 = defaultdict(dict)
    for T, v in bakhshs95.items():
        nm95[(T[0], T[1])][bname(v["name"])] = T
    for i, p in pcs.iterrows():
        if i in assign:
            continue
        nn, cty = sci_of[p["adm2_pcode"]]
        if bname(p["name"]) == fold("مرکزی"):
            continue  # an unmatched central district is most likely a newer county's: neighbours
        T = nm95[(nn, cty)].get(bname(p["name"]))
        if T:
            assign[i] = T
    pend = [i for i in pcs.index if i not in assign]
    geom_eq = pcs.to_crs(EQ).geometry.make_valid()
    for _ in range(5):
        left = []
        for i in pend:
            same = pcs.index[(pcs["adm2_pcode"] == pcs.at[i, "adm2_pcode"])
                             & pcs.index.isin(list(assign))]
            best, bl = None, 0
            for k in same:
                try:  # shared border, measured as overlap of a 100 m halo (robust to noding)
                    ln = geom_eq[i].buffer(100).intersection(geom_eq[k]).area
                except Exception:  # noqa: BLE001
                    ln = 0
                if ln > bl:
                    best, bl = k, ln
            if best is not None:
                assign[i] = assign[best]
            else:
                left.append(i)
        pend = left
        if not pend:
            break
    for i in pend:  # alone in its county with nothing known: the county's biggest district
        nn, cty = sci_of[pcs.at[i, "adm2_pcode"]]
        assign[i] = max((T for T in bakhshs95 if T[:2] == (nn, cty)),
                        key=lambda T: bakhshs95[T]["pop"])
    pcs["T"] = [assign[i] for i in pcs.index]

    # 1395 districts with no piece: merge into the unit holding most of their people
    got = set(pcs["T"])
    orphan = [T for T in bakhshs95 if T not in got]
    for T in orphan:
        best, bx = None, 0
        for i, p in pcs.iterrows():
            x = p["comp"].get(T, 0)
            if x > bx:
                best, bx = p["T"], x
        if best is None:  # nothing at all: biggest district of its county
            best = max((U for U in got if U[:2] == T[:2]), key=lambda U: bakhshs95[U]["pop"])
        union(T, best)
    # every 1395 district belongs to the unit of its root
    groups = defaultdict(list)
    for T in bakhshs95:
        groups[find(T)].append(T)
    unit_of = {T: tuple(sorted(groups[find(T)])) for T in bakhshs95}
    pcs["unit"] = pcs["T"].map(lambda T: unit_of[T])
    nunits = len(set(unit_of.values()))
    print(f"pieces: {len(pcs)} ({int(sliver.sum())} slivers); 1395 districts {len(bakhshs95)} "
          f"-> {nunits} units ({sum(1 for g in set(unit_of.values()) if len(g) > 1)} merged "
          f"groups, {len(orphan)} districts with no polygon of their own)")
    return pcs, unit_of, bakhshs95, counties95, c_of, p_of, a1, a2, b


def write(pcs, unit_of, bakhshs95, counties95, c_of, p_of, a1, a2, b):
    import geopandas as gpd

    OUTB.mkdir(parents=True, exist_ok=True)
    qids = [q for q in b["wikidata"].dropna()]
    labels = wikidata_labels(qids)
    en_of_bid = {r["bid"]: labels.get(r["wikidata"]) if isinstance(r["wikidata"], str) else None
                 for _, r in b.iterrows()}
    osm_en = {r["bid"]: r["name_en"] for _, r in b.iterrows() if isinstance(r["name_en"], str)}
    osm_all = gpd.read_file(OSM, ignore_geometry=True)
    cty_en = dict(zip(osm_all.loc[osm_all.admin_level == "5", "name"],
                      osm_all.loc[osm_all.admin_level == "5", "name_en"]))

    def ucode(unit):
        nn, cty = unit[0][0], unit[0][1]
        return c_of[(nn, cty)] + "-" + "".join(T[2] for T in unit)

    pcs["code"] = pcs["unit"].map(ucode)
    g = pcs.dissolve(by="code", as_index=False, aggfunc="first")
    g["geometry"] = g.geometry.buffer(0)
    # names: 1395 Persian names, and an English one from the OSM polygon(s) that carry the
    # unit's main district name, through Wikidata
    names, names_en, parents, groups = [], [], [], []
    for _, r in g.iterrows():
        unit = r["unit"]
        fa = " + ".join(bakhshs95[T]["name"] for T in unit)
        en_parts = []
        for T in unit:
            key = bname(bakhshs95[T]["name"])
            if key == fold("مرکزی"):
                en_parts.append("Central District")
                continue
            cand = pcs[(pcs["unit"] == unit) & (pcs["name"].map(bname) == key)]
            en = None
            for bid in cand["bid"]:
                en = en_of_bid.get(bid) or osm_en.get(bid) or en
            if not en:
                # a 1395 district promoted to a county since: OSM names the county after it
                for oc in pcs.loc[pcs["unit"] == unit, "osm_county"].dropna().unique():
                    if cname(oc) == key and cty_en.get(oc):
                        en = re.sub(r"\s+County$", "", cty_en[oc]) + " District"
            if not en:
                en = wikidata_search_district(bakhshs95[T]["name"])
            if en:
                en = re.sub(r"\s*\(.*\)$", "", en)
            en_parts.append(en or (translit(bakhshs95[T]["name"]) + " District*"))
        en = " + ".join(en_parts)
        names.append(fa)
        names_en.append(en)
        parents.append(c_of[(unit[0][0], unit[0][1])])
        groups.append(p_of[unit[0][0]])
    g["name_fa"] = names
    g["name_en"] = names_en
    g["parent"] = parents
    g["group"] = groups
    g["name"] = g["name_en"]
    g["name_cn"] = g["name_fa"]  # the viewer's second-name slot (as Kazakhstan uses it)
    g["sci_districts"] = g["unit"].map(lambda u: "|".join("".join(T) for T in u))
    out = g[["code", "name", "name_cn", "parent", "group", "sci_districts", "geometry"]]
    out.to_file(OUTB / "adm3.gpkg", layer="adm3", driver="GPKG")

    # counties and provinces: COD as they are, with English names and the province group
    a2o = a2[["adm2_pcode", "adm2_name", "adm2_name1", "adm1_pcode", "adm1_name",
              "geometry"]].copy()
    a2o.columns = ["code", "name", "name_cn", "parent", "parent_name", "geometry"]
    a2o["name_cn"] = a2o["name_cn"].str.replace("شهرستان ", "", regex=False)
    a2o["group"] = a2o["parent"]
    a2o.to_file(OUTB / "adm2.gpkg", layer="adm2", driver="GPKG")
    a1o = a1[["adm1_pcode", "adm1_name", "adm1_name1", "geometry"]].copy()
    a1o.columns = ["code", "name", "name_cn", "geometry"]
    a1o["group"] = a1o["code"]
    a1o.to_file(OUTB / "adm1.gpkg", layer="adm1", driver="GPKG")

    with open(UNITS_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["code", "parent", "sci_districts", "name_fa", "name_en", "pop2016_settled"])
        for _, r in g.iterrows():
            w.writerow([r["code"], r["parent"], r["sci_districts"], r["name_fa"], r["name_en"],
                        sum(bakhshs95[T]["pop"] for T in r["unit"])])
    with open(PIECES_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["osm_name", "osm_county", "adm2_pcode", "km2", "share_of_osm_district",
                    "district_1400", "to_1395", "unit"])
        for _, p in pcs.iterrows():
            w.writerow([p["name"], p["osm_county"], p["adm2_pcode"], round(p["km2"], 2),
                        round(p["share"], 4), "".join(p["k14"]) if p["k14"] else "",
                        "".join(p["T"]), p["code"]])
    print(f"wrote {OUTB / 'adm3.gpkg'} ({len(out)} units), adm1/adm2 copies, {UNITS_CSV.name}, "
          f"{PIECES_CSV.name}")


# ----------------------------------------------------------------------------- population
def population_rows(bakhshs95, counties95, carried, nomads90, c11, c16, c24):
    """Rows for level 3, called by fetch.py. Each county's figure is shared over its units:
    1395 by the districts' own counts (the county's non-settled people pro rata), 1390 by the
    carried settled people (the county's 1390 total, so units sum to it), 2024 by the county's
    2024 figure with half the unit's 2011-16 lead over its county carried on, as for counties."""
    import pandas as pd
    from fetch import largest_remainder, FORWARD_DAMP

    u = pd.read_csv(UNITS_CSV, dtype=str)
    unit_of = {}
    for _, r in u.iterrows():
        for s in r["sci_districts"].split("|"):
            unit_of[(s[:2], s[2:4], s[4:6])] = r["code"]
    assert set(unit_of) == set(bakhshs95), set(bakhshs95) ^ set(unit_of)
    rows = []
    byc = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))
    for T, v in bakhshs95.items():
        byc[T[:2]][unit_of[T]][1] += v["pop"]
    for T, x in carried.items():
        if T in unit_of:
            byc[T[:2]][unit_of[T]][0] += x
        else:  # a 1390 settlement that landed in a 1395 county's non-district part: none expected
            raise SystemExit(f"carried to unknown district {T}")
    def share(d, total):
        s = sum(d.values())
        return largest_remainder({c: v * total / s for c, v in d.items()}, total)

    for k, units in byc.items():
        u11 = share({c: v[0] for c, v in units.items()}, c11[k])
        u16 = share({c: v[1] for c, v in units.items()}, c16[k])
        g_c = (c16[k] / c11[k]) ** 0.2
        raw = {}
        for cc in units:
            rel = ((u16[cc] / u11[cc]) ** 0.2 / g_c) if u11[cc] > 0 else 1.0
            raw[cc] = u16[cc] * rel ** (8 * FORWARD_DAMP)
        f = c24[k] / sum(raw.values())
        u24 = largest_remainder({cc: raw[cc] * f for cc in units}, c24[k])
        for cc in units:
            rows += [(cc, 3, 2011, u11[cc]), (cc, 3, 2016, u16[cc]), (cc, 3, 2024, u24[cc])]
    return rows


def main():
    res = build()
    write(*res)


if __name__ == "__main__":
    main()
