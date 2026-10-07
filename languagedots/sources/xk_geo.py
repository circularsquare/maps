"""Kosovo: the placement layer. religiondots' Kontur 400 m hexes keyed to the 38 municipalities,
here also keyed to the census settlements inside them.

    python sources/xk_geo.py --fetch   ASK's 2011 ethnicity-by-settlement tables (38 PxWeb POSTs,
                                       about 45 s each, four at a time) and the OSM settlement
                                       points (from Geofabrik's kosovo-latest.osm.pbf, deleted
                                       once the points are read)
    python sources/xk_geo.py           -> data/geo/xk/xk_hexes.gpkg, data/geo/xk/xk_settlements.csv

UNITS AND HEXES. religiondots' data/geo/xk/xk_municipalities.gpkg (geoBoundaries ADM2, 38 polygons,
`unit` = ASK's municipality code, joined there by name 38/38) and xk_grid_400m.gpkg (9,258 Kontur
hexes with `unit` and `pop`), read only. The ASK codes are the same ones the mother-tongue table
carries (checked here by name). Nothing about the hexes is redone; each gets a `settlement`.

SETTLEMENTS (AGENT_BRIEF section 4.4). Mother tongue is published by municipality only. ASK's
PxWeb has ethnicity by settlement for 2011 only (`6_Sipas vendbanimeve/<code> .../censusetn<code>
.px`, one table per municipality); 2024 settlement tables give population, sex and age, not
ethnicity. So inside a municipality each language follows the 2011 ethnic mix of each settlement
(countries/xk.py says which ethnicity goes with which language); the counts drawn are the 2024
municipal table's. The 2011 census did not enumerate the four northern municipalities, so their
tables are empty and they are placed on Kontur population alone.
  1. Settlement points are OSM place nodes from Geofabrik's kosovo-latest.osm.pbf, by `name:sq`,
     else `name`, plus `name:sr-Latn` and `name:sr` as second keys.
  2. A census settlement is matched to a node of the same folded name inside its municipality's
     polygon buffered by BUF_M; Albanian names have definite and indefinite forms (Prishtinë /
     Prishtina), so a second pass drops a final vowel on both sides, after spelling "upper" and
     "lower" one way (i Ulët / i Poshtëm) and dropping OSM's parenthetical municipality; a third
     drops j and y (Grejkoc / Greikoc). Two settlements known under another name are aliased. Of several candidates the
     one inside the polygon wins, then the higher place rank; a tie is left unmatched.
  3. An unmatched settlement holding more than half its municipality's 2011 population (a town
     OSM spells differently) sits at the municipality's Kontur-weighted centroid.
  4. Each hex goes to the nearest placed settlement OF ITS OWN MUNICIPALITY.

CHECKS: per municipality, the 2011 settlements sum to the 2011 municipal ethnicity table
(census2024_05, year 2011) in every ethnicity; the share of each municipality's 2011 population
whose settlement was placed.
"""
import json
import os
import sys
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
from rdlink import RD_GEO  # noqa: E402

UNITS = RD_GEO / "xk" / "xk_municipalities.gpkg"
GRID = RD_GEO / "xk" / "xk_grid_400m.gpkg"
RAW = ROOT / "data" / "raw" / "xk"
SETT_RAW = RAW / "settlements"
OUT = ROOT / "data" / "geo" / "xk" / "xk_hexes.gpkg"
SETTLEMENTS = ROOT / "data" / "geo" / "xk" / "xk_settlements.csv"
PBF = RAW / "kosovo-latest.osm.pbf"
PLACES = RAW / "osm_places.csv"
PBF_URL = "https://download.geofabrik.de/europe/kosovo-latest.osm.pbf"
SETT_BASE = ("https://askdata.rks-gov.net/api/v1/en/ASKdata/Census population/"
             "6_Sipas vendbanimeve/")
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

N_UNITS = 38
NORTH = {"12", "28", "29", "38"}       # Leposaviq, Zubin Potok, Zveqan, Mitrovicë e Veriut
BUF_M = 1500
CRS_M = 32634                          # UTM 34N
PLACE_TAGS = {"city", "town", "village", "hamlet", "suburb", "neighbourhood", "quarter",
              "isolated_dwelling", "locality"}
ETHN = {"Albanian": "albanians", "Serbian": "serbs", "Turkish": "turks", "Bosnian": "bosniaks",
        "Roma": "roma", "Ashkali": "ashkali", "Egyptian": "egyptians", "Goran": "gorani",
        "Other": "other", "Prefers not to answer": "undeclared", "Not available": "unknown",
        "Total": "total"}
# the 2011 municipal ethnicity table (census2024_05) names the same groups differently
ETHN_MUNI = {"Albanian": "albanians", "Serb": "serbs", "Turk": "turks", "Bosniak": "bosniaks",
             "Romani": "roma", "Ashkali": "ashkali", "Egyptian": "egyptians", "Gorani": "gorani",
             "Others": "other", "Prefers not to answer": "undeclared", "Total": "total"}


def _post(url):
    import requests
    for attempt in range(3):
        try:
            r = requests.post(url, headers=UA, timeout=400,
                              json={"query": [], "response": {"format": "json-stat2"}})
            r.raise_for_status()
            doc = r.json()
            if "value" in doc:
                return doc
        except Exception as e:  # noqa: BLE001
            print(f"    retry {url.rsplit('/', 1)[-1]}: {e}")
    raise SystemExit(f"{url}: no json-stat2 cube after 3 tries")


def fetch():
    import requests
    SETT_RAW.mkdir(parents=True, exist_ok=True)
    folders = requests.get(SETT_BASE, headers=UA, timeout=120).json()
    jobs = []
    for f in folders:
        code = f["id"].split(" ", 1)[0]
        dest = SETT_RAW / f"censusetn{code}.json"
        if dest.exists() and dest.stat().st_size > 500:
            continue
        jobs.append((dest, SETT_BASE + f["id"] + "/"))
    if len(folders) != N_UNITS:
        raise SystemExit(f"{len(folders)} settlement folders, expected {N_UNITS}")
    print(f"  {len(jobs)} settlement tables to fetch, four at a time")

    def one(job):
        dest, folder = job
        # the table id is not always censusetn<code>.px (Deçan's is PopEtnicititeti03.px), so
        # list the folder and take the one table titled with ethnicity. North Mitrovica, created
        # in 2013, has none (2011 did not enumerate it anyway).
        tabs = [t["id"] for t in requests.get(folder, headers=UA, timeout=300).json()
                if "ethnicity" in t["text"].lower()]
        if len(tabs) != 1:
            print(f"    !! {folder}: ethnicity tables {tabs}")
            return
        doc = _post(folder + tabs[0])
        dest.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
        print(f"    {dest.name} {dest.stat().st_size:,} bytes", flush=True)
    with ThreadPoolExecutor(4) as ex:
        list(ex.map(one, jobs))

    if PLACES.exists():
        print("already have", PLACES)
        return
    if not PBF.exists():
        r = requests.get(PBF_URL, timeout=1200, stream=True, headers=UA)
        r.raise_for_status()
        with open(PBF, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
    import csv
    import osmium

    class H(osmium.SimpleHandler):
        def __init__(self):
            super().__init__()
            self.rows = []

        def node(self, n):
            p = n.tags.get("place")
            if p in PLACE_TAGS:
                self.rows.append({"osm_id": n.id, "place": p, "name": n.tags.get("name", ""),
                                  "name_sq": n.tags.get("name:sq", ""),
                                  "name_sr_latn": n.tags.get("name:sr-Latn", ""),
                                  "name_sr": n.tags.get("name:sr", ""),
                                  "lat": n.location.lat, "lon": n.location.lon})

        def area(self, a):
            # many villages are mapped only as a place area or a cadastral boundary (admin
            # level 8 and finer); take the mean of the outer ring's vertices
            p = a.tags.get("place")
            adm = a.tags.get("boundary") == "administrative" and \
                a.tags.get("admin_level", "") in {"8", "9", "10"}
            if p not in PLACE_TAGS and not adm:
                return
            try:
                xs, ys = [], []
                for ring in a.outer_rings():
                    for nd in ring:
                        xs.append(nd.lon); ys.append(nd.lat)
            except Exception:  # noqa: BLE001  (a broken multipolygon)
                return
            if not xs:
                return
            self.rows.append({"osm_id": f"a{a.id}", "place": p if p in PLACE_TAGS else "boundary",
                              "name": a.tags.get("name", ""),
                              "name_sq": a.tags.get("name:sq", ""),
                              "name_sr_latn": a.tags.get("name:sr-Latn", ""),
                              "name_sr": a.tags.get("name:sr", ""),
                              "lat": sum(ys) / len(ys), "lon": sum(xs) / len(xs)})
    hd = H()
    hd.apply_file(str(PBF), locations=True)
    with open(PLACES, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(hd.rows[0]))
        w.writeheader()
        w.writerows(hd.rows)
    print(f"  {PLACES}: {len(hd.rows):,} place nodes")
    PBF.unlink()
    print(f"  deleted {PBF.name}")


CYR = dict(zip("абвгдђежзијклљмнњопрстћуфхцчџш",
               ["a", "b", "v", "g", "d", "dj", "e", "zh", "z", "i", "j", "k", "l", "lj", "m", "n",
                "nj", "o", "p", "r", "s", "t", "c", "u", "f", "h", "c", "c", "dz", "sh"]))


def fold(s):
    s = "".join(CYR.get(c, c) for c in str(s).lower())
    s = s.replace("ç", "c").replace("č", "c").replace("ć", "c").replace("š", "sh") \
         .replace("ž", "zh").replace("đ", "dj").replace("xh", "dz").replace("q", "c")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return "".join(c for c in s if c.isalnum())


# "Upper" and "Lower" have two spellings each in Albanian place names (i Ulët / i Poshtëm,
# e Epërme / i Epërm); ASK and OSM do not always pick the same one.
UPLOW = [(r"\b(i|e)\s+(ulet|ulte|poshtem|poshtme)\b", "lower"),
         (r"\b(i|e)\s+(eperm|eperme|siperm|siperme)\b", "upper")]


def _plain(s):
    import re
    s = re.sub(r"\(.*?\)", " ", str(s).lower())
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    for pat, rep in UPLOW:
        s = re.sub(pat, rep, s)
    return s


def stem(s):
    k = fold(_plain(s))
    return k[:-1] if len(k) > 3 and k[-1] in "ae" else k


def loose(s):
    """A last pass: j and y read as i, doubled i collapsed (Grejkoc / Greikoc, Sopijë / Sopië)."""
    import re
    k = re.sub("i+", "i", fold(_plain(s)).replace("j", "i").replace("y", "i"))
    return k[:-1] if len(k) > 3 and k[-1] in "ae" else k


# Settlements whose OSM node carries a different name altogether; checked by hand on the map
# (each node lies inside the named municipality).
ALIASES = {("3", "Gllogoc"): "Drenas",              # the town, renamed Drenas in 2000s usage
           ("19", "Nëntë Jugoviq"): "Bardhosh"}     # Devet Jugovića, now Bardhosh


def settlements():
    """The 38 tables -> one row per settlement: unit, name, 2011 ethnic counts."""
    import pandas as pd
    rows = []
    for p in sorted(SETT_RAW.glob("censusetn*.json")):   # saved as censusetn<code> whatever ASK calls it
        unit = p.stem.replace("censusetn", "")
        doc = json.loads(p.read_text(encoding="utf-8"))
        ids, size, dim, val = doc["id"], doc["size"], doc["dimension"], doc["value"]
        # labels are stripped of stray punctuation too: Gjilan's table prints "Albanian,"
        lab = {k: {c: str(v).strip().rstrip(",;.").strip()
                   for c, v in dim[k]["category"]["label"].items()} for k in ids}
        order = {}
        for k in ids:
            idx = dim[k]["category"]["index"]
            order[k] = sorted(idx, key=lambda c: idx[c]) if isinstance(idx, dict) else list(idx)
        k_eth = next(k for k in ids if "Ashkali" in lab[k].values())
        k_sex = next(k for k in ids if "Female" in lab[k].values())
        k_set = next(k for k in ids if k not in (k_eth, k_sex))
        if set(lab[k_eth].values()) != set(ETHN):
            raise SystemExit(f"{p.name}: ethnicity labels {sorted(lab[k_eth].values())}")
        si = next(i for i, c in enumerate(order[k_sex]) if lab[k_sex][c] == "Total")
        for gi, gc in enumerate(order[k_set]):
            r = {"unit": unit, "code": f"{unit}-{gc}", "name": lab[k_set][gc]}
            for ei, ec in enumerate(order[k_eth]):
                pos = {k_sex: si, k_set: gi, k_eth: ei}
                n = 0
                for k, s in zip(ids, size):
                    n = n * s + pos[k]
                v = val[n]
                r[ETHN[lab[k_eth][ec]]] = 0 if v is None else int(v)
            rows.append(r)
    st = pd.DataFrame(rows)
    # Each table ends with the municipality's own total, in capitals (DEÇAN, PRIZREN); the town
    # itself is a separate row in ordinary case. Drop exactly one row per table, the one equal to
    # the sum of all the others in every group, and say so.
    drop = []
    cols = list(ETHN.values())
    for u, g in st.groupby("unit"):
        if g["total"].sum() == 0:
            continue
        hit = [i for i in g.index
               if (g.loc[i, cols] == g.drop(index=i)[cols].sum()).all()]
        if len(hit) == 2 and len(g) == 2:
            # a one-settlement municipality (Mamushë): the town and the total are equal
            hit = [i for i in hit if g.loc[i, "name"].upper() == g.loc[i, "name"]][:1]
        if len(hit) != 1:
            raise SystemExit(f"censusetn{u}: {len(hit)} rows equal the sum of the rest")
        drop.append(hit[0])
    caps = st.loc[drop, "name"]
    if not (caps.str.upper() == caps).all():
        raise SystemExit(f"a total row not in capitals: {list(caps)}")
    # the four northern tables are empty; their capitals row is dropped by name
    empty_caps = st[(st.groupby("unit")["total"].transform("sum") == 0)
                    & (st["name"].str.upper() == st["name"])].index
    st = st.drop(index=drop + list(empty_caps)).reset_index(drop=True)
    print(f"  dropped {len(drop)} municipal-total rows (in capitals, each the sum of the rest)")
    if set(st["unit"]) | {"38"} != {str(c) for c in range(1, N_UNITS + 1)}:
        raise SystemExit(f"settlement tables for {st['unit'].nunique()} municipalities; expected "
                         "every one but North Mitrovica")
    return st


def muni2011():
    """{unit: {ethnicity: count}} for 2011 from census2024_05 (fetched by xk_census.py)."""
    doc = json.loads((RAW / "census2024_05.json").read_text(encoding="utf-8"))
    ids, size, dim, val = doc["id"], doc["size"], doc["dimension"], doc["value"]
    lab = {k: {c: str(v).strip() for c, v in dim[k]["category"]["label"].items()} for k in ids}
    order = {}
    for k in ids:
        idx = dim[k]["category"]["index"]
        order[k] = sorted(idx, key=lambda c: idx[c]) if isinstance(idx, dict) else list(idx)
    k_geo = next(k for k in ids if "KOSOVA" in lab[k].values())
    k_year = next(k for k in ids if "2011" in lab[k].values())
    k_eth = next(k for k in ids if "Ashkali" in lab[k].values())
    k_sex = next(k for k in ids if k not in (k_geo, k_year, k_eth))
    yi = next(i for i, c in enumerate(order[k_year]) if lab[k_year][c] == "2011")
    si = next(i for i, c in enumerate(order[k_sex]) if lab[k_sex][c] == "Total")
    out, names = {}, {}
    for gi, gc in enumerate(order[k_geo]):
        row = {}
        for ei, ec in enumerate(order[k_eth]):
            pos = {k_geo: gi, k_year: yi, k_sex: si, k_eth: ei}
            n = 0
            for k, s in zip(ids, size):
                n = n * s + pos[k]
            v = val[n]
            row[ETHN_MUNI[lab[k_eth][ec]]] = 0 if v is None else int(v)
        out[gc] = row
        names[gc] = lab[k_geo][gc]
    return out, names


def main():
    if "--fetch" in sys.argv:
        fetch()

    import geopandas as gpd
    import numpy as np
    import pandas as pd

    units = gpd.read_file(UNITS)[["unit", "geo_name", "geometry"]]
    units["unit"] = units["unit"].astype(str)
    if len(units) != N_UNITS or units["unit"].nunique() != N_UNITS:
        raise SystemExit(f"{UNITS}: {len(units)} polygons")
    norm = pd.read_csv(ROOT / "data" / "normalized" / "xk.csv", dtype={"geo_id": str})
    mt = norm[(norm["geo_level"] == "municipality") & (norm["source_category"] == "Total")]
    nm_mt = dict(zip(mt["geo_id"], mt["geo_name"]))
    nm_poly = dict(zip(units["unit"], units["geo_name"]))
    bad = [u for u in nm_poly if stem(nm_poly[u]) != stem(nm_mt.get(u, ""))]
    print(f"  {'OK ' if set(nm_poly) == set(nm_mt) else 'BAD'} the 38 polygons' ASK codes are the "
          f"mother-tongue table's; names differ on {len(bad)} "
          + ", ".join(f"{u}: {nm_poly[u]} / {nm_mt.get(u)}" for u in bad))
    if set(nm_poly) != set(nm_mt):
        raise SystemExit("polygon codes and table codes differ")

    # --- settlements, checked against the 2011 municipal ethnicity table
    st = settlements()
    m11, _ = muni2011()
    cols = [c for c in ETHN_MUNI.values()]
    off = []
    # the municipal table folds the settlement tables' "Not available" into "Prefers not to answer"
    for u, g in st.groupby("unit"):
        s = g[cols].sum()
        s["undeclared"] += g["unknown"].sum()
        if any(int(s[c]) != m11[u][c] for c in cols):
            off.append((u, {c: (int(s[c]), m11[u][c]) for c in cols if int(s[c]) != m11[u][c]}))
    print(f"  {'OK ' if not off else 'BAD'} {len(st):,} settlements sum to the 2011 municipal "
          f"ethnicity table in every group, in every municipality ({len(off)} off: {off[:6]})")
    if off:
        raise SystemExit("settlement sums disagree with the municipal table")
    eth_cols = [c for c in ETHN.values() if c != "total"]
    badp = int((st[eth_cols].sum(axis=1) != st["total"]).sum())
    print(f"  {'OK ' if not badp else 'BAD'} the 11 categories partition every settlement")
    if badp:
        raise SystemExit("ethnic categories do not partition the settlements")
    empty = sorted(st.groupby("unit")["total"].sum().loc[lambda s: s == 0].index)
    print(f"  municipalities with no 2011 count: {empty} "
          f"({', '.join(nm_poly[u] for u in empty)})")
    empty = sorted(set(empty) | {"38"})      # no table at all for North Mitrovica
    if set(empty) != NORTH:
        raise SystemExit(f"expected exactly the four northern municipalities empty, got {empty}")
    st = st[~st["unit"].isin(NORTH)].copy()

    # --- settlement points
    pl = pd.read_csv(PLACES, dtype=str, keep_default_na=False)
    pts = gpd.GeoDataFrame(pl, geometry=gpd.points_from_xy(pl["lon"].astype(float),
                                                            pl["lat"].astype(float)),
                           crs=4326).to_crs(CRS_M)
    # every name a node carries, as (node index, key)
    keys = []
    for i, r in pl.iterrows():
        names = {r["name_sq"], r["name"], r["name_sr_latn"], r["name_sr"]}
        for n in names:
            for part in str(n).replace(" – ", "/").replace(" - ", "/").split("/"):
                if part.strip():
                    keys.append((i, fold(_plain(part.strip())), stem(part.strip()),
                                 loose(part.strip())))
    kd = pd.DataFrame(keys, columns=["i", "key", "stem", "loose"]).drop_duplicates()
    um = units.to_crs(CRS_M).set_index("unit")
    rank = {"city": 0, "town": 1, "village": 2, "suburb": 3, "hamlet": 4, "quarter": 5,
            "neighbourhood": 6, "isolated_dwelling": 7, "locality": 8, "boundary": 9}
    pts["rank"] = pts["place"].map(rank)

    xs, ys, how = [], [], []
    for _, s in st.iterrows():
        poly = um.geometry[s["unit"]]
        found = None
        nm = ALIASES.get((s["unit"], s["name"]), s["name"])
        for col, k in (("key", fold(_plain(nm))), ("stem", stem(nm)), ("loose", loose(nm))):
            ii = kd.loc[kd[col] == k, "i"].unique()
            cand = pts.loc[ii]
            cand = cand[cand.geometry.within(poly.buffer(BUF_M))]
            if len(cand) > 1:
                inside = cand[cand.geometry.within(poly)]
                cand = inside if len(inside) else cand
            if len(cand) > 1 and cand["rank"].nunique() > 1:
                top = cand["rank"].min()
                cand = cand[cand["rank"] == top]
            if len(cand) == 1:
                found = cand.geometry.iloc[0]
                break
            if len(cand) > 1:
                found = "ambiguous"
                break
        if found is None or isinstance(found, str):
            xs.append(np.nan); ys.append(np.nan); how.append(found or "none")
        else:
            xs.append(found.x); ys.append(found.y); how.append("osm")
    st["x"], st["y"], st["how"] = xs, ys, how

    # --- hexes: religiondots' layer, read only
    hx = gpd.read_file(GRID)
    hx["unit"] = hx["unit"].astype(str)
    hp = hx.to_crs(CRS_M)
    hp["geometry"] = hp.geometry.centroid
    cx = hp.assign(wx=hp.geometry.x * hp["pop"], wy=hp.geometry.y * hp["pop"]) \
        .groupby("unit")[["wx", "wy", "pop"]].sum()
    mtot = st.groupby("unit")["total"].sum()
    big = (st["how"] != "osm") & (st["total"] > 0.5 * st["unit"].map(mtot))
    for i in st.index[big]:
        u = st.at[i, "unit"]
        st.at[i, "x"] = cx.at[u, "wx"] / cx.at[u, "pop"]
        st.at[i, "y"] = cx.at[u, "wy"] / cx.at[u, "pop"]
        st.at[i, "how"] = "centroid"
    placed = st["how"].isin(["osm", "centroid"])
    print(f"  settlements placed: {int((st['how'] == 'osm').sum()):,} on an OSM node, "
          f"{int((st['how'] == 'centroid').sum())} at their municipality's population centroid; "
          f"{int((~placed).sum())} not placed ({int((st['how'] == 'ambiguous').sum())} ambiguous), "
          f"holding {st.loc[~placed, 'total'].sum():,} of {st['total'].sum():,} people in 2011 "
          f"({100 * st.loc[~placed, 'total'].sum() / st['total'].sum():.2f}%)")
    miss = st[~placed].sort_values("total", ascending=False).head(12)
    print("    largest not placed: " + ", ".join(
        f"{a} ({nm_poly[u]}, {h}) {c:,}" for a, u, h, c in
        zip(miss["name"], miss["unit"], miss["how"], miss["total"])))
    share = st[placed].groupby("unit")["total"].sum() / mtot
    low = share.sort_values().head(6)
    print("    lowest share of a municipality placed: " + ", ".join(
        f"{nm_poly[u]} {v:.0%}" for u, v in low.items()))

    hp["settlement"] = ""
    sp = st[placed]
    for u, g in hp.groupby("unit"):
        if u in NORTH:
            continue
        s = sp[sp["unit"] == u]
        if s.empty:
            raise SystemExit(f"no placed settlement in {u}")
        sx, sy = s["x"].to_numpy(), s["y"].to_numpy()
        gx, gy = g.geometry.x.to_numpy()[:, None], g.geometry.y.to_numpy()[:, None]
        k = ((gx - sx) ** 2 + (gy - sy) ** 2).argmin(axis=1)
        hp.loc[g.index, "settlement"] = s["code"].to_numpy()[k]
    got = set(hp.loc[hp["pop"] > 0, "settlement"])
    nohex = sp[~sp["code"].isin(got) & (sp["total"] > 0)]
    print(f"  {len(nohex)} placed settlements own no populated hex ({nohex['total'].sum():,} "
          "people in 2011; their neighbours' hexes carry them)")

    out = gpd.GeoDataFrame(hx[["unit", "pop"]].assign(settlement=hp["settlement"].to_numpy()),
                           geometry=hx.geometry, crs=hx.crs)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_file(OUT, layer="hexes", driver="GPKG")
    print(f"  wrote {OUT} ({len(out):,} hexes, {out['pop'].sum():,.0f} people; "
          f"{int((out['settlement'] == '').sum())} hexes in the north carry no settlement)")
    keep = ["code", "unit", "name", "how", "x", "y"] + list(ETHN.values())
    st[keep].to_csv(SETTLEMENTS, index=False, encoding="utf-8")
    print(f"  wrote {SETTLEMENTS} ({len(st):,} settlements; x, y in EPSG:{CRS_M})")


if __name__ == "__main__":
    main()
