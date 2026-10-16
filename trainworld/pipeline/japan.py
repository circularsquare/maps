"""T-420: Japan's inputs for a tier 0 pack (Greater Tokyo first), through tier0's hooks. Written for
every Japanese pack: a pack names its prefectures (or Hokkaido's subprefectures, `SUBPREFECTURES`),
and everything below is national.

- **Outline and municipalities**: KSJ N03 administrative areas (MLIT, 2025-01-01, per prefecture),
  municipalities dissolved from their parts; designated cities by ward, the unit the census uses.
  Tokyo's Izu and Ogasawara islands (codes 13360 and up) are left out.
- **Population**: 2020 census 500 m mesh (e-Stat T001141, JGD2011, per first mesh). A suppressed
  mesh's people are counted in the mesh it was merged into (HTKSAKI), so the numbered meshes sum
  to the census. Each mesh is spread over the signal cells inside it by GHSL; a mesh cut by the
  outline on land keeps only its share of land inside (neighbouring prefectures' N03 polygons).
- **Jobs**: 2021 Economic Census for Business Activity, 500 m mesh, persons employed in all
  industries, government included (e-Stat T001162), spread inside each mesh by tier0's proxy, then
  scaled per municipality to the 2020 census's employed people 15+ by place of work (table 4,
  which also counts farmers, household services and those working at home).
- **Work at home**: per cell, the census's "working at home" share of workers by place of work in
  the cell's municipality (T-465), plus the pack's telework share of the rest on a given day
  (config `telework`).
- **Curve**: the same census table, municipality to municipality (notes/t420_curve.py).
- **Schools**: KSJ P29 (2023) elementary, junior high, compulsory and secondary schools as points;
  each mesh's children 0-14 x tier0.SCHOOL_ENROL go to the nearest such school.
- **Names**: `kana_romaji`, Hepburn romaji from a kana reading, and `kanji_romaji`, the kana
  reading of a kanji name from UniDic (fugashi with unidic-lite), for places.py.

All open: e-Stat under the Government Standard Terms 2.0 (CC BY 4.0 compatible); KSJ N03 and P29
(2023) CC BY 4.0. Cached in data/raw/tier0/japan/.

    python pipeline/japan.py fetch tyo        download every input of a pack
    python pipeline/japan.py outline tyo      draw the outline (artifacts/trainworld/T-420-<id>-outline.png)
"""
import io
import json
import math
import os
import sys
import zipfile

import numpy as np
import pandas as pd
import shapely
from shapely import STRtree, make_valid, union_all
from shapely.geometry import mapping

import build_city as b
import england

ESTAT_MESH = "https://www.e-stat.go.jp/gis/statmap-search/data?statsId={sid}&code={code}&downloadType=2"
POP_MESH, JOBS_MESH = "T001141", "T001162"   # census 2020 and economic census 2021, 500 m, JGD2011
N03_URL = "https://nlftp.mlit.go.jp/ksj/gml/data/N03/N03-2025/N03-20250101_{pref}_GML.zip"
P29_URL = "https://nlftp.mlit.go.jp/ksj/gml/data/P29/P29-23/P29-23_{pref}_GML.zip"
# 2020 census, table 4: workers and students by municipality of work or school, per prefecture
# of work (both sexes: e04-PP-01, statInfId 000032214305 + 3 x PP)
FLOWS_URL = "https://www.e-stat.go.jp/stat-search/file-download?statInfId={:012d}&fileKind=0"
FLOWS_ID0 = 32214305  # e04-00-01; each prefecture's three files follow in order, with gaps
FINEST = {"0", "2", "3"}
MIN_PART_DEG2 = 2e-6
ISLANDS = 13360         # Tokyo codes from here on are the Izu and Ogasawara islands
# Units smaller than a prefecture (T-432), usable wherever a prefecture code is: Hokkaido's
# subprefectures by their N03 name (N03_002), or one municipality as "<pref>-<code>"; "~<code>"
# after a unit leaves that municipality out (stage 3: Shiribeshi less Otaru, which is Sapporo's)
SUBPREFECTURES = {"01-ishikari": "石狩振興局", "01-shiribeshi": "後志総合振興局", "01-sorachi": "空知総合振興局",
                  "01-iburi": "胆振総合振興局", "01-rumoi": "留萌振興局", "01-kamikawa": "上川総合振興局",
                  "01-soya": "宗谷総合振興局", "01-okhotsk": "オホーツク総合振興局", "01-tokachi": "十勝総合振興局",
                  "01-kushiro": "釧路総合振興局", "01-nemuro": "根室振興局", "01-hidaka": "日高振興局",
                  "01-oshima": "渡島総合振興局", "01-hiyama": "檜山振興局"}
# N03 lists the six villages of the Russian-administered southern Kurils under Nemuro; the census
# counts no one there, so no zone draws them (T-420 stage 3)
NOT_ADMINISTERED = {"01695", "01696", "01697", "01698", "01699", "01700"}
# Stage 3 outlines leave out uninhabited far islets (Takeshima, the Senkakus): parts under
# FAR_ISLET_KM2 more than FAR_ISLET_KM from every larger part of the zone
FAR_ISLET_KM2, FAR_ISLET_KM = 5.0, 50.0
# Wards redrawn between the 2020 census and N03 2025 (Hamamatsu, 2024: seven wards became three,
# old Kita-ku split): both sides are read as the whole city
CITY_OF_WARD = {**{c: "22130" for c in ("22131", "22132", "22133", "22134", "22135", "22136", "22137",
                                        "22138", "22139", "22140")}}
CITY_NAME = {"22130": "浜松市"}


def _real(unit):
    """The prefecture code of a unit: a prefecture, a subprefecture or a single municipality."""
    return unit[:2]
# P29 school kinds (P29_003): elementary, junior high, compulsory, secondary
P29_SCHOOLS = {"16001", "16002", "16003", "16004"}


def raw(*p):
    return os.path.join(b.RAW, "tier0", "japan", *p)


def _get(url, path):
    """Download once through curl: KSJ's server can stall a Python request for good while curl
    gets the same file in seconds (T-432, T-441). A read slower than 1 kB/s for a minute aborts."""
    if not os.path.exists(path):
        import subprocess
        os.makedirs(os.path.dirname(path), exist_ok=True)
        for attempt in range(3):
            r = subprocess.run(["curl", "-sSfL", "--retry", "2", "--connect-timeout", "30", "--speed-limit", "1000",
                                "--speed-time", "60", "-A", b.UA["User-Agent"], "-o", path + ".part", url])
            if r.returncode == 0:
                break
        else:
            raise RuntimeError(f"curl failed ({r.returncode}) for {url}")
        os.replace(path + ".part", path)
    return path


# ---------------------------------------------------------------- mesh codes

def mesh_sw(code):
    """South-west corner (lat, lon) of a 500 m mesh (9 digits: first, second, third mesh, quarter)."""
    c = np.asarray(code, np.int64)
    p, u = c // 10_000_000, c // 100_000 % 100
    q, v = c // 10_000 % 10, c // 1_000 % 10
    r, w = c // 100 % 10, c // 10 % 10
    m = c % 10
    lat = p / 1.5 + q / 12 + r / 120 + (m >= 3) / 240
    lon = 100 + u + v / 8 + w / 80 + ((m == 2) | (m == 4)) / 160
    return lat, lon


def mesh_of(lat, lon):
    """The 500 m mesh code holding each point."""
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    a = np.floor(lat * 240 + 1e-9).astype(np.int64)  # 1/240 degree rows
    o = np.floor((lon - 100) * 160 + 1e-9).astype(np.int64)  # 1/160 degree columns
    p, a = a // 160, a % 160
    u, o = o // 160, o % 160
    q, a = a // 20, a % 20
    v, o = o // 20, o % 20
    r, a = a // 2, a % 2
    w, o = o // 2, o % 2
    m = 1 + o + 2 * a
    return (((((p * 100 + u) * 10 + q) * 10 + v) * 10 + r) * 10 + w) * 10 + m


def first_meshes(bounds):
    w, s, e, n = bounds
    return [f"{p}{u}" for p in range(int(s * 1.5), int(n * 1.5) + 1) for u in range(int(w) - 100, int(e) - 100 + 1)]


def mesh_table(sid, codes):
    """A 500 m mesh table over the first meshes: DataFrame indexed by the 9-digit code, numeric
    columns (suppressed '*' and '-' as 0), rows for meshes with figures only."""
    parts = []
    for code in codes:
        path = raw("mesh", f"{sid}_{code}.zip")
        if not os.path.exists(path) and not os.path.exists(path + ".none"):
            import requests
            r = requests.get(ESTAT_MESH.format(sid=sid, code=code), headers=b.UA, timeout=600)
            if r.status_code != 404:
                r.raise_for_status()
            if r.status_code == 404 or not r.content.startswith(b"PK"):  # no land with figures in this first mesh
                os.makedirs(os.path.dirname(path), exist_ok=True)
                open(path + ".none", "w").close()
                continue
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path + ".part", "wb") as f:
                f.write(r.content)
            os.replace(path + ".part", path)
        if not os.path.exists(path):
            continue
        with zipfile.ZipFile(path) as z:
            name = next(n for n in z.namelist() if n.lower().endswith((".txt", ".csv")))
            t = pd.read_csv(z.open(name), encoding="cp932", skiprows=[1], dtype=str)
        parts.append(t)
    t = pd.concat(parts, ignore_index=True)
    t = t[t.KEY_CODE.str.len() == 9].copy()
    key = t.KEY_CODE.astype(np.int64)
    num = t[[c for c in t.columns if c.startswith(sid)]].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    num.index = key.to_numpy()
    return num.groupby(level=0).sum()


# ---------------------------------------------------------------- municipalities and the outline

def _n03(pref):
    """N03 parts of one prefecture: DataFrame code, pref, name, geometry (lon/lat)."""
    import pyogrio
    path = _get(N03_URL.format(pref=pref), raw("n03", f"N03-20250101_{pref}_GML.zip"))
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
    shp = next((n for n in names if n.lower().endswith(".geojson")), None) or next(n for n in names if n.lower().endswith(".shp"))
    df = pyogrio.read_dataframe(f"/vsizip/{path}/{shp}", encoding="cp932" if shp.endswith(".shp") else None)
    code = df["N03_007"].astype(str).str.zfill(5)
    city = df["N03_004"].fillna("").astype(str)
    ward = df["N03_005"].fillna("").astype(str) if "N03_005" in df else ""
    sub = df["N03_002"].fillna("").astype(str) if "N03_002" in df else ""
    return pd.DataFrame({"code": code, "pref": pref, "name": (city + ward), "sub": sub, "geometry": df.geometry.to_numpy()})


def municipalities(prefs):
    """[(code, name, prefecture, polygon)], designated cities by ward, Tokyo's islands left out.
    Cached as one GeoJSON per prefecture set."""
    path = raw(f"municipalities_{'_'.join(prefs)}.geojson")
    if not os.path.exists(path) and len(prefs) > 1:  # one cache per unit, joined (stage 3: 38 zones share them)
        return [m for p in prefs for m in municipalities([p])]
    if not os.path.exists(path):
        feats = []
        for pref in prefs:
            base, *excl = pref.split("~")  # "01-shiribeshi~01203": the subprefecture less Otaru
            df = _n03(_real(base))
            df = df[df.code.str.match(r"^\d{5}$") & ~df.code.isin(excl)]
            if base in SUBPREFECTURES:
                df = df[df["sub"] == SUBPREFECTURES[base]]
            elif len(base) > 2:
                df = df[df.code == base[3:]]
            if df.empty:
                raise ValueError(f"N03 has no municipality for unit {pref}")
            if pref == "13":
                df = df[df.code.astype(int) < ISLANDS]
            df = df[~df.code.isin(NOT_ADMINISTERED)]
            merged = df.code.isin(CITY_OF_WARD)
            if merged.any():
                df = df.copy()
                df.loc[merged, "name"] = df.loc[merged, "code"].map(CITY_OF_WARD).map(CITY_NAME)
                df.loc[merged, "code"] = df.loc[merged, "code"].map(CITY_OF_WARD)
            named = df[~df.code.str.endswith("000")]
            near = union_all([make_valid(x) for x in named.geometry if x is not None]).buffer(0.02)
            for code, g in df.groupby("code"):
                geom = union_all([make_valid(x) for x in g.geometry if x is not None]).buffer(0)
                if code.endswith("000"):  # land of no municipality (Tokyo Bay's landfill, far islets): only beside the rest
                    geom = union_all([p for p in getattr(geom, "geoms", [geom]) if p.intersects(near)])
                    if geom.is_empty:
                        continue
                feats.append({"type": "Feature", "properties": {"code": code, "name": g.name.iloc[0], "pref": pref},
                              "geometry": mapping(geom)})
        with open(path + ".part", "w", encoding="utf-8") as f:
            json.dump({"type": "FeatureCollection", "features": feats}, f, ensure_ascii=False)
        os.replace(path + ".part", path)
    from shapely.geometry import shape
    with open(path, encoding="utf-8") as f:
        fc = json.load(f)
    return [(ft["properties"]["code"], ft["properties"]["name"], ft["properties"]["pref"], make_valid(shape(ft["geometry"])))
            for ft in fc["features"]]


def _drop_far_islets(parts):
    """Parts under FAR_ISLET_KM2 farther than FAR_ISLET_KM from every larger part go."""
    k = math.cos(math.radians(union_all(parts).centroid.y))
    km2 = [q.area * 111.32 ** 2 * k for q in parts]
    big = [q for q, a in zip(parts, km2) if a >= FAR_ISLET_KM2]
    tree = STRtree(big)
    keep = []
    for q, a in zip(parts, km2):
        if a < FAR_ISLET_KM2 and big:
            near = big[tree.nearest(q)]
            if near.distance(q) * 111.32 * k > FAR_ISLET_KM:  # degrees to km, conservatively (east-west scale)
                continue
        keep.append(q)
    return keep


def boundary_file(city, prefs, far_islets=True):
    """The prefectures, each dissolved from its municipalities, as one GeoJSON for tier0's
    `boundary_file`; its path. far_islets=False (stage 3 zones) leaves out uninhabited far islets."""
    out = raw(f"{city}_prefectures.geojson")
    if os.path.exists(out):
        return out
    ms = municipalities(prefs)
    feats = []
    for p in prefs:
        g = union_all([m[3] for m in ms if m[2] == p]).buffer(0)
        # N03 draws every pier, breakwater and rock apart: parts under about 0.02 km² go
        parts = [q for q in getattr(g, "geoms", [g]) if q.area >= MIN_PART_DEG2]
        if not far_islets:
            parts = _drop_far_islets(parts)
        g = union_all(parts)
        feats.append({"type": "Feature", "properties": {"pref": p}, "geometry": mapping(g)})
    with open(out + ".part", "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": feats}, f)
    os.replace(out + ".part", out)
    return out


_LAND = {}


def _land(prefs, neighbours):
    key = tuple(sorted(set(prefs) | set(neighbours)))
    if key not in _LAND:
        _LAND[key] = union_all([m[3] for m in municipalities(list(key))])
        shapely.prepare(_LAND[key])
    return _LAND[key]


def mesh_inside_share(codes, boundary, land, k=6):
    """Per mesh: the share of its land inside the boundary, from k x k points over the mesh (1 for
    a mesh wholly inside; a mesh with no point on N03 land, reclaimed or offshore, counts by its
    centre)."""
    lat, lon = mesh_sw(codes)
    boxes = shapely.box(lon, lat, lon + 1 / 160, lat + 1 / 240)
    shapely.prepare(boundary)
    share = np.ones(len(codes))
    cut = np.flatnonzero(~shapely.contains(boundary, boxes))
    if len(cut):
        f = (np.arange(k) + 0.5) / k
        px = (lon[cut, None, None] + f[None, None, :] / 160 + 0 * f[None, :, None]).ravel()
        py = (lat[cut, None, None] + f[None, :, None] / 240 + 0 * f[None, None, :]).ravel()
        inb = shapely.contains_xy(boundary, px, py).reshape(len(cut), -1).sum(1)
        onl = shapely.contains_xy(land, px, py).reshape(len(cut), -1).sum(1)
        centre = shapely.contains_xy(boundary, lon[cut] + 1 / 320, lat[cut] + 1 / 480)
        share[cut] = np.where(onl > 0, np.minimum(inb / np.maximum(onl, 1), 1.0), centre.astype(float))
    return share


def spread_mesh(sig, values, weight, boundary, land):
    """Mesh totals (Series by code) over the signal cells: inside each mesh by weight (evenly where
    the weights are all zero), times the mesh's land share inside; a mesh with no cell gives its
    share to the cell nearest its centre."""
    codes = values.index.to_numpy(np.int64)
    share = mesh_inside_share(codes, boundary, land)
    tot = values.to_numpy(float) * share
    cm = mesh_of(sig["lat"], sig["lon"])
    pos = {c: i for i, c in enumerate(codes)}
    idx = np.array([pos.get(int(c), -1) for c in cm])
    ok = idx >= 0
    w = np.asarray(weight, float)
    n = len(codes)
    ws = np.bincount(idx[ok], weights=w[ok], minlength=n)
    cnt = np.bincount(idx[ok], minlength=n)
    out = np.zeros(len(cm))
    s = np.where(ws[idx[ok]] > 0, w[ok] / np.maximum(ws[idx[ok]], 1e-30), 1.0 / np.maximum(cnt[idx[ok]], 1))
    out[ok] = tot[idx[ok]] * s
    empty = np.flatnonzero((cnt == 0) & (tot > 0))
    if len(empty):
        lat, lon = mesh_sw(codes[empty])
        near = STRtree(shapely.points(sig["lon"], sig["lat"])).nearest(shapely.points(lon + 1 / 320, lat + 1 / 480))
        np.add.at(out, near, tot[empty])
    return out, float(tot.sum()), float(tot[empty].sum())


def mesh_population(city, boundary, sig, prefs, neighbours):
    """People per signal cell from the 2020 census 500 m mesh; tier0's `population` hook."""
    t = mesh_table(POP_MESH, first_meshes(boundary.bounds))
    out, total, lone = spread_mesh(sig, t[POP_MESH + "001"], sig["pop_ghsl"], boundary, _land(prefs, neighbours))
    print(f"  census 2020 500 m mesh: {len(t):,} meshes in the box ({t[POP_MESH + '001'].sum():,.0f} people), "
          f"{out.sum():,.0f} inside ({lone:,.0f} in meshes with no signal cell, put on the nearest)", flush=True)
    return out, "2020 Population Census 500 m mesh (e-Stat T001141), spread inside each mesh by GHSL GHS-POP E2020"


def mesh_children(boundary, prefs, neighbours):
    """(mesh codes, centres lat, lon, children 0-14 inside the boundary) from the census mesh."""
    t = mesh_table(POP_MESH, first_meshes(boundary.bounds))
    codes = t.index.to_numpy(np.int64)
    kids = t[POP_MESH + "004"].to_numpy(float) * mesh_inside_share(codes, boundary, _land(prefs, neighbours))
    lat, lon = mesh_sw(codes)
    return codes, lat + 1 / 480, lon + 1 / 320, kids


# ---------------------------------------------------------------- the census commuting table

def _flows_id(pref):
    """The statInfId of table 4 for a prefecture of work (file e04-PP-01.xlsx): found by reading
    the file names upward from where the order puts it (ids skip), cached."""
    import requests
    path = raw("flows", "statinfids.json")
    ids = json.load(open(path)) if os.path.exists(path) else {}
    want = f"e04-{pref}-01.xlsx"
    if want not in ids:
        for i in range(FLOWS_ID0 + 3 * int(pref), FLOWS_ID0 + 3 * int(pref) + 400):
            with requests.get(FLOWS_URL.format(i), headers=b.UA, stream=True, timeout=60) as r:
                name = r.headers.get("content-disposition", "").rsplit("''", 1)[-1] if r.ok else ""
            if name:
                ids[name] = i
            if name == want:
                break
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(ids, f, indent=0, sort_keys=True)
    return ids[want]


def flows_table(pref):
    """Table 4 of the 2020 census for one prefecture of work, employed people 15+: DataFrame
    w (municipality of work), h (municipality of residence, or 'home' for working at home, 'all'
    for every worker there), n. Municipalities are the finest unit (wards of designated cities).
    A unit smaller than a prefecture keeps the rows of its own municipalities of work."""
    if len(pref) > 2:
        t = flows_table(_real(pref))
        return t[t.w.isin({m[0] for m in municipalities([pref])})].reset_index(drop=True)
    cache = raw("flows", f"census2020_t4_{pref}.csv")
    if not os.path.exists(cache):
        path = _get(FLOWS_URL.format(_flows_id(pref)), raw("flows", f"e04-{pref}-01.xlsx"))
        x = pd.read_excel(path, header=None, dtype=str)
        head = x.index[x.iloc[:, 0].astype(str).str.startswith("男女")][0]
        cols = x.iloc[head - 2].astype(str).tolist()
        col = next(i for i, c in enumerate(cols) if c.startswith("11_"))  # 15+ employed
        d = x.iloc[head + 1:]
        # area kinds: a prefecture or the country, 1 a designated city, 0 its wards, 2 and 3 other
        # municipalities; only the finest (0, 2, 3) at both ends
        d = d[d.iloc[:, 1].astype(str).isin(FINEST)]
        wcode = d.iloc[:, 3].astype(str).str[:5]
        level = d.iloc[:, 5].astype(str)
        hcode = d.iloc[:, 8].astype(str).str[:5]
        hkind = d.iloc[:, 6].astype(str)
        n = pd.to_numeric(d.iloc[:, col].str.replace(",", ""), errors="coerce").fillna(0.0)
        rows = []
        for wc, lv, hc, hk, v in zip(wcode, level, hcode, hkind, n):
            if lv.startswith("0_"):
                rows.append((wc, "all", v))
            elif lv.startswith("011_"):
                rows.append((wc, "home", v))
            elif lv.startswith("012_"):
                rows.append((wc, wc, v))  # elsewhere in the home municipality
            elif lv.startswith("02_") and hk in FINEST:  # another municipality
                rows.append((wc, hc, v))
        t = pd.DataFrame(rows, columns=["w", "h", "n"])
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        t.to_csv(cache + ".part", index=False)
        os.replace(cache + ".part", cache)
    t = pd.read_csv(cache, dtype={"w": str, "h": str})
    if t.w.isin(CITY_OF_WARD).any() or t.h.isin(CITY_OF_WARD).any():
        t["w"], t["h"] = t.w.replace(CITY_OF_WARD), t.h.replace(CITY_OF_WARD)
        t = t.groupby(["w", "h"], as_index=False, sort=False).n.sum()
    return t


def flows(prefs):
    t = pd.concat([flows_table(p) for p in prefs], ignore_index=True)
    return t


def workers_at_work(prefs):
    """{municipality of work: employed people 15+ working there}, and the share working at home."""
    t = flows(prefs)
    allw = t[t.h == "all"].groupby("w").n.sum()
    home = t[t.h == "home"].n.sum()
    return allw.to_dict(), float(home / allw.sum())


def at_home_share(lat, lon, prefs):
    """Per cell, the census's share of workers by place of work who work at home, by the cell's
    municipality (table 4's "home" rows, T-465: 0.5% in Chiyoda, 21% in Suginami); the units' mean
    for a cell in no municipality or one with no figure."""
    import england
    t = flows(prefs)
    allw = t[t.h == "all"].groupby("w").n.sum()
    home = t[t.h == "home"].groupby("w").n.sum().reindex(allw.index, fill_value=0.0)
    mean = float(home.sum() / allw.sum())
    share = (home / allw.where(allw > 0)).fillna(mean).to_dict()
    ms = municipalities(prefs)
    idx, _ = england.cell_units(np.asarray(lat), np.asarray(lon), [(m[0], m[3]) for m in ms])
    by_unit = np.array([share.get(m[0], mean) for m in ms] + [mean])
    return by_unit[np.where(idx >= 0, idx, len(ms))]


def census_jobs(city, boundary, sig, weight, prefs, neighbours):
    """Jobs per signal cell: the 2021 economic census 500 m mesh, spread inside each mesh by the
    proxy weight, scaled per municipality to the 2020 census's workers by place of work; tier0's
    `jobs` hook."""
    t = mesh_table(JOBS_MESH, first_meshes(boundary.bounds))
    col = JOBS_MESH + "023"  # persons employed, all industries (A-S); 001-022 count establishments
    raw_jobs, total, lone = spread_mesh(sig, t[col], weight, boundary, _land(prefs, neighbours))
    ms = municipalities(prefs)
    idx, near = england.cell_units(sig["lat"], sig["lon"], [(m[0], m[3]) for m in ms])
    want, home_share = workers_at_work(prefs)
    out = raw_jobs.copy()
    have = np.bincount(idx[idx >= 0], weights=raw_jobs[idx >= 0], minlength=len(ms))
    worst = []
    for i, m in enumerate(ms):
        sel = idx == i
        tgt = float(want.get(m[0], 0.0))
        if have[i] > 0:
            out[sel] *= tgt / have[i]
        elif tgt > 0:  # no economic census jobs on the municipality's cells: by the proxy
            wsum = weight[sel].sum()
            if wsum > 0:
                out[sel] = tgt * weight[sel] / wsum
            elif i in near:
                out[near[i]] += tgt
        worst.append((tgt / max(have[i], 1.0) - 1, m[1], have[i], tgt))
    missing = [m[0] for m in ms if m[0] not in want]
    worst.sort()
    print(f"  economic census 2021 500 m mesh: {t[col].sum():,.0f} persons employed in the box, {raw_jobs.sum():,.0f} inside; "
          f"scaled to census 2020 workers by place of work: {out.sum():,.0f} ({len(missing)} municipalities with no figure)", flush=True)
    for r in worst[:3] + worst[-3:]:
        print(f"    {r[1]}: economic census {r[2]:,.0f} -> census {r[3]:,.0f} ({r[0]:+.1%})", flush=True)
    return out, ("2021 Economic Census for Business Activity 500 m mesh (e-Stat T001162), persons employed, scaled per municipality to "
                 "the 2020 Population Census's employed people 15+ by place of work (table 4)")


# ---------------------------------------------------------------- everyday destinations (T-453)

# T001162: 001-021 establishments, 022-042 persons employed in the same order (notes/T-456.md 8)
MESH_ALL, MESH_I, MESH_M, MESH_N = "022", "032", "036", "037"
# economic census 2021, table 9-1B: persons employed by minor industry by municipality
EC_9_1B_URL = "https://www.e-stat.go.jp/stat-search/file-download?statInfId=000040067885&fileKind=0"
# JSIC divisions: I's retail (56-61), M's customer part (75-77), N's 78-79 and 80; head-office units
# (three-digit codes ending 0) left out
SPLITS = {"ret": [str(c) for c in range(56, 62)], "m2": ["75", "76", "77"], "p78": ["78", "79"], "p80": ["80"]}
# the source passed tier0.sector_check's concentration test on Tokyo (DECISIONS 2026-10-16, T-453);
# sparser packs fail it on roadside malls and resort hotels their OSM lacks, so they run the level test only
SECTOR_JUDGED = {"errand": "Tokyo, 2.5%", "leisure": "Tokyo, 2.6%"}


def sector_splits():
    """Per municipality code (and '00000', the country): persons employed in I, M, N and in the
    customer-facing parts (SPLITS) from table 9-1B, cached as a small CSV."""
    cache = raw("ec2021_9-1B_splits.csv")
    if not os.path.exists(cache):
        import openpyxl
        wb = openpyxl.load_workbook(_get(EC_9_1B_URL, raw("ec2021_9-1B.xlsx")), read_only=True)
        rows = list(wb.worksheets[0].iter_rows(values_only=True))
        head = [h.split("_")[0] if isinstance(h, str) else "" for h in rows[5]]
        cols = {k: [j for j, h in enumerate(head) if h == k] for k in ("I", "M", "N")}
        for k, codes in SPLITS.items():
            two = [j for j, h in enumerate(head) if h in codes]
            adm = [j for j, h in enumerate(head) if len(h) == 3 and h.endswith("0") and h[:2] in codes]
            cols[k], cols[k + "_adm"] = two, adm
        assert all(cols.values()), {k: v for k, v in cols.items() if not v}
        num = lambda v: float(v) if isinstance(v, (int, float)) else 0.0
        out = []
        for r in rows[9:]:
            if isinstance(r[1], str) and "_" in r[1]:
                out.append({"code": r[1].split("_")[0], **{k: sum(num(r[j]) for j in js) for k, js in cols.items()}})
        t = pd.DataFrame(out)
        for k in SPLITS:
            t[k] = (t[k] - t.pop(k + "_adm")).clip(lower=0.0)
        t.to_csv(cache + ".part", index=False)
        os.replace(cache + ".part", cache)
    t = pd.read_csv(cache, dtype={"code": str})
    t["code"] = t.code.replace(CITY_OF_WARD)
    return t.groupby("code").sum()


def mesh_sectors(city, lat, lon, jobs, prefs):
    """tier0's `everyday` hook: per cell, its jobs x its 500 m mesh's share of persons employed in
    I (wholesale and retail), M and N (T001162), I split to retail and N to 78-79 and 80 by the
    municipality's table 9-1B shares; errands = retail + 78-79, leisure = M's customer part + 80."""
    t = mesh_table(JOBS_MESH, first_meshes((lon.min(), lat.min(), lon.max(), lat.max())))
    col = lambda k: t[JOBS_MESH + k]
    AS = col(MESH_ALL)
    cm = mesh_of(lat, lon)
    share = {}
    for k, c in (("I", MESH_I), ("M", MESH_M), ("N", MESH_N)):
        s = (col(c) / AS.where(AS > 0)).dropna()
        share[k] = pd.Series(cm).map(s).fillna(col(c).sum() / AS.sum()).to_numpy()
    ms = municipalities(prefs)
    idx, _ = england.cell_units(lat, lon, [(m[0], m[3]) for m in ms])
    sp = sector_splits()
    ratio = sp[list(SPLITS)].div(sp[["I", "M", "N", "N"]].to_numpy()).fillna(0.0)
    ratio.columns = list(SPLITS)
    nat = ratio.loc["00000"]
    code = np.array([ms[i][0] if i >= 0 else "" for i in idx])
    r = ratio.reindex(code)
    miss = r.ret.isna().to_numpy()
    r = r.fillna(nat)
    f = lambda k: r[k].to_numpy()
    if miss.any():
        print(f"  9-1B: {miss.mean():.2%} of cells with no municipal row, national shares", flush=True)
    errand = jobs * (share["I"] * f("ret") + share["N"] * f("p78"))
    leisure = jobs * (share["M"] * f("m2") + share["N"] * f("p80"))
    src = ("2021 Economic Census for Business Activity, persons employed by 500 m mesh (e-Stat T001162; {}), split by "
           "municipality by table 9-1B")
    return {"errand": (errand, src.format("retail and personal services, JSIC 56-61 and 78-79"), SECTOR_JUDGED["errand"]),
            "leisure": (leisure, src.format("accommodation, eating and drinking, JSIC 75-77, and amusement, 80"), SECTOR_JUDGED["leisure"])}


# ---------------------------------------------------------------- schools

def p29_schools(boundary, prefs, neighbours):
    """KSJ P29 (2023) elementary, junior high, compulsory and secondary schools in the boundary, each
    given the children 0-14 x tier0.SCHOOL_ENROL of the census meshes nearest it: (DataFrame lat,
    lon, pupils; coverage; url); tier0's `schools` hook."""
    import pyogrio
    import tier0
    pts = []
    for pref in dict.fromkeys(_real(p) for p in prefs):
        path = _get(P29_URL.format(pref=pref), raw("p29", f"P29-23_{pref}_GML.zip"))
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
        f = next((n for n in names if n.lower().endswith(".geojson")), None) or next(n for n in names if n.lower().endswith(".shp"))
        df = pyogrio.read_dataframe(f"/vsizip/{path}/{f}", encoding="cp932" if f.endswith(".shp") else None)
        kind = df["P29_003"].astype(str)
        df = df[kind.isin(P29_SCHOOLS).to_numpy()]
        pts.append(pd.DataFrame({"lat": df.geometry.y.to_numpy(), "lon": df.geometry.x.to_numpy()}))
    s = pd.concat(pts, ignore_index=True)
    shapely.prepare(boundary)
    s = s[shapely.contains_xy(boundary, s.lon.to_numpy(), s.lat.to_numpy())].reset_index(drop=True)
    _, lat, lon, kids = mesh_children(boundary, prefs, neighbours)
    keep = kids > 0
    near = STRtree(shapely.points(s.lon.to_numpy(), s.lat.to_numpy())).nearest(shapely.points(lon[keep], lat[keep]))
    s["pupils"] = np.bincount(near, weights=kids[keep] * tier0.SCHOOL_ENROL, minlength=len(s))
    return s, (f"KSJ P29 schools 2023 (MLIT, CC BY 4.0): {len(s):,} elementary, junior high, compulsory and secondary schools; "
               f"pupils = 2020 census 500 m mesh children 0-14 x {tier0.SCHOOL_ENROL}, each mesh to its nearest school"), \
        "https://nlftp.mlit.go.jp/ksj/gml/datalist/KsjTmplt-P29-2023.html"


# ---------------------------------------------------------------- names

_KANA = {
    "きゃ": "kya", "きゅ": "kyu", "きょ": "kyo", "しゃ": "sha", "しゅ": "shu", "しょ": "sho", "ちゃ": "cha", "ちゅ": "chu",
    "ちょ": "cho", "にゃ": "nya", "にゅ": "nyu", "にょ": "nyo", "ひゃ": "hya", "ひゅ": "hyu", "ひょ": "hyo", "みゃ": "mya",
    "みゅ": "myu", "みょ": "myo", "りゃ": "rya", "りゅ": "ryu", "りょ": "ryo", "ぎゃ": "gya", "ぎゅ": "gyu", "ぎょ": "gyo",
    "じゃ": "ja", "じゅ": "ju", "じょ": "jo", "ぢゃ": "ja", "ぢゅ": "ju", "ぢょ": "jo", "びゃ": "bya", "びゅ": "byu",
    "びょ": "byo", "ぴゃ": "pya", "ぴゅ": "pyu", "ぴょ": "pyo", "しぇ": "she", "じぇ": "je", "ちぇ": "che", "てぃ": "ti",
    "でぃ": "di", "ふぁ": "fa", "ふぃ": "fi", "ふぇ": "fe", "ふぉ": "fo", "うぃ": "wi", "うぇ": "we", "ゔぁ": "va",
    "あ": "a", "い": "i", "う": "u", "え": "e", "お": "o", "か": "ka", "き": "ki", "く": "ku", "け": "ke", "こ": "ko",
    "さ": "sa", "し": "shi", "す": "su", "せ": "se", "そ": "so", "た": "ta", "ち": "chi", "つ": "tsu", "て": "te", "と": "to",
    "な": "na", "に": "ni", "ぬ": "nu", "ね": "ne", "の": "no", "は": "ha", "ひ": "hi", "ふ": "fu", "へ": "he", "ほ": "ho",
    "ま": "ma", "み": "mi", "む": "mu", "め": "me", "も": "mo", "や": "ya", "ゆ": "yu", "よ": "yo", "ら": "ra", "り": "ri",
    "る": "ru", "れ": "re", "ろ": "ro", "わ": "wa", "ゐ": "i", "ゑ": "e", "を": "o", "ん": "n", "が": "ga", "ぎ": "gi",
    "ぐ": "gu", "げ": "ge", "ご": "go", "ざ": "za", "じ": "ji", "ず": "zu", "ぜ": "ze", "ぞ": "zo", "だ": "da", "ぢ": "ji",
    "づ": "zu", "で": "de", "ど": "do", "ば": "ba", "び": "bi", "ぶ": "bu", "べ": "be", "ぼ": "bo", "ぱ": "pa", "ぴ": "pi",
    "ぷ": "pu", "ぺ": "pe", "ぽ": "po", "ゔ": "vu", "ぁ": "a", "ぃ": "i", "ぅ": "u", "ぇ": "e", "ぉ": "o", "ゃ": "ya",
    "ゅ": "yu", "ょ": "yo", "ゎ": "wa",
}


def kana_romaji(kana):
    """Hepburn romaji from a kana reading (hiragana or katakana), each word capitalised; None if
    the reading has anything else in it. Long vowels written short (Oji, Ome, Tokyo), as OSM's
    name:en and most signs do, so the basemap needs no macron glyphs; n before a vowel or y is n'."""
    s = "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in kana.strip())
    out, i, geminate = [], 0, False
    while i < len(s):
        c = s[i]
        if c == "っ":
            geminate = True
            i += 1
            continue
        if c == "ー":
            i += 1
            continue
        if c in " 　・":
            out.append(" ")
            i += 1
            continue
        syl = _KANA.get(s[i:i + 2]) or _KANA.get(c)
        if syl is None:
            return None
        i += 2 if s[i:i + 2] in _KANA else 1
        if geminate:
            syl = ("t" + syl) if syl.startswith("ch") else (syl[0] + syl)
            geminate = False
        prev = out[-1][-1] if out and out[-1] else ""
        if (prev == "o" and syl in ("u", "o")) or (prev == "u" and syl == "u"):
            continue  # a long vowel, written short as on most signs and OSM's name:en

        if out and out[-1] == "n" and syl[0] in "aiueoy":
            syl = "'" + syl
        out.append(syl)
    text = "".join(out)
    return " ".join(w[:1].upper() + w[1:] for w in text.split()) or None


# A name's first kanji read as in place names when the dictionary splits it off (上大岡 is Kamiooka,
# not Ueooka; 南流山 Minaminagareyama, not Nanryusan)
PLACE_PREFIX = {"上": "カミ", "下": "シモ", "東": "ヒガシ", "西": "ニシ", "南": "ミナミ", "北": "キタ", "中": "ナカ",
                "新": "シン", "元": "モト", "小": "コ"}
_TAGGER = []


# A town's 町 reads cho in OSM's name:en far more often than machi; a city's 市 is left off, as there
_SUFFIX = {"町": "cho", "市": ""}
_PN_WEIGHT = {"地名": 2, "人名": 1}
_PN_MIN = 2
_KANSUJI = dict(zip("一二三四五六七八九", range(1, 10)))


def _reading(text):
    """(kana reading, tokens) from UniDic (fugashi with unidic-lite), or None if no analysis knows
    every token. Of MeCab's ten best analyses, the one with most characters in proper nouns
    (place names twice a person's name) wins: 宮内 Miyauchi, not the common noun kunai."""
    if not _TAGGER:
        import fugashi
        _TAGGER.append(fugashi.Tagger())
    best = None
    for toks in _TAGGER[0].nbestToNodeList(text, 10):
        kana = [t.feature.kana for t in toks]
        if not toks or any(t.is_unk or not k or k == "*" for t, k in zip(toks, kana)):
            continue
        score = sum(len(t.surface) * _PN_WEIGHT.get(t.feature.pos3, 0)
                    for t in toks if t.feature.pos2 == "固有名詞" and (len(t.surface) >= _PN_MIN or len(toks) == 1))
        if best is None or score > best[0]:
            best = (score, "".join(kana), len(toks))
    return best and best[1:]


def _kansuji(s):
    if s.isdigit():
        return int(s)
    if s == "十":
        return 10
    if "十" in s:
        a, _, c = s.partition("十")
        return 10 * (_KANSUJI.get(a, 1) if a else 1) + (_KANSUJI.get(c, 0) if c else 0)
    return _KANSUJI.get(s)


def _word_romaji(part):
    if len(part) > 2 and part[-1] in _SUFFIX:
        stem = _word_romaji(part[:-1])
        return stem and stem + _SUFFIX[part[-1]]
    r = _reading(part)
    if r and r[1] > 1 and len(part) > 1 and part[0] in PLACE_PREFIX:
        rest = _reading(part[1:])
        if rest:
            r = (PLACE_PREFIX[part[0]] + rest[0], 1)
    return kana_romaji(r[0]) if r else None


def kanji_romaji(name):
    """Romaji for a kanji name from a reading dictionary (UniDic), through kana_romaji; None where a
    word is not in the dictionary. One word per part of the name, as Japanese place names are
    written (Higashiarai, Kamiooka); a block number as "Shintsuka 1-chome"; a leading 大字 (an
    administrative word) and anything in brackets left out. T-426."""
    import re
    name = re.sub(r"[（(][^）)]*[）)]", "", name).strip()
    name = re.sub(r"^大字", "", name)
    # 本田ケ谷 Hondagaya: a ケ between kanji is ga, and ケ谷 nearly always gaya
    name = re.sub(r"(?<=[一-鿿])[ケヶ]谷", "がや", name)
    name = re.sub(r"(?<=[一-鿿])[ケヶ](?=[一-鿿])", "が", name)
    suffix = ""
    m = re.match(r"^(.+?)([一二三四五六七八九十]+|\d+)丁目$", name)
    if m and _kansuji(m.group(2)):
        name, suffix = m.group(1), f" {_kansuji(m.group(2))}-chome"
    else:
        m = re.match(r"^(.+?)(\d+)$", name)
        if m:
            name, suffix = m.group(1), f" {int(m.group(2))}"
    parts = []
    for part in name.replace("　", " ").replace("・", " ").split():
        roma = _word_romaji(part)
        if not roma:
            return None
        parts.append(roma)
    return (" ".join(parts) + suffix) if parts else None


# ---------------------------------------------------------------- command line

def outline(city):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cfg = b.CITIES[city]
    ms = municipalities(cfg["prefectures"])
    fig, ax = plt.subplots(figsize=(11, 10))
    for code, _, pref, g in ms:
        for p in getattr(g, "geoms", [g]):
            x, y = p.exterior.xy
            ax.fill(x, y, color={"11": "#cde", "12": "#edc", "13": "#dec", "14": "#ecd"}.get(pref, "#ddd"), ec="#888", lw=0.2)
    from shapely.geometry import shape
    with open(boundary_file(city, cfg["prefectures"])) as f:
        fc = json.load(f)
    merged = union_all([make_valid(shape(ft["geometry"])) for ft in fc["features"]])
    for p in getattr(merged, "geoms", [merged]):
        ax.plot(*p.exterior.xy, color="k", lw=0.8)
        for r in p.interiors:
            ax.plot(*r.xy, color="r", lw=1.2)
    n_holes = sum(len(p.interiors) for p in getattr(merged, "geoms", [merged]))
    n_parts = len(getattr(merged, "geoms", [merged]))
    ax.set_title(f"{city}: {len(ms)} municipalities, {n_parts} parts, {n_holes} holes (red)")
    ax.set_aspect(1 / math.cos(math.radians(merged.centroid.y)))
    path = os.path.join(b.ROOT, "..", "artifacts", "trainworld", f"T-420-{city}-outline.png")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    areas = sorted(((p.area, p.representative_point().coords[0]) for p in getattr(merged, "geoms", [merged])), reverse=True)
    print(f"{city}: {len(ms)} municipalities, {n_parts} parts (largest first, deg2): "
          + ", ".join(f"{a:.4f} at {c[0]:.3f},{c[1]:.3f}" for a, c in areas[:8]) + f"; {n_holes} holes -> {path}")


def fetch(city):
    cfg = b.CITIES[city]
    prefs, nb = cfg["prefectures"], cfg["neighbours"]
    municipalities(prefs)
    municipalities(sorted(set(prefs) | set(nb)))
    from shapely.geometry import shape
    with open(boundary_file(city, prefs)) as f:
        bounds = union_all([shape(ft["geometry"]) for ft in json.load(f)["features"]]).bounds
    for sid in (POP_MESH, JOBS_MESH):
        t = mesh_table(sid, first_meshes(bounds))
        print(f"{sid}: {len(t):,} meshes; columns {list(t.columns[:3])}; first column total {t.iloc[:, 0].sum():,.0f}", flush=True)
    for p in prefs:
        t = flows_table(p)
        print(f"flows {p}: {len(t):,} rows, workers there {t[t.h == 'all'].n.sum():,.0f}, at home {t[t.h == 'home'].n.sum():,.0f}", flush=True)
        _get(P29_URL.format(pref=_real(p)), raw("p29", f"P29-23_{_real(p)}_GML.zip"))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    {"fetch": fetch, "outline": outline}[sys.argv[1]](sys.argv[2])
