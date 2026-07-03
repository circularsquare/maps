"""Build unified district-level admin raster across all drawn countries.
Each unit stores its parent chain (dist=adm2/prefecture, prov=adm1) for hierarchy roll-up."""
import sys, json, time, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np, geopandas as gpd
from rasterio.features import rasterize
from geo import TRANSFORM, PROJ4, W, H

SCR = HERE
D = os.path.join(HERE, "..", "..", "data", "asia1m")
NE = "C:/Users/anita/.local/share/cartopy/shapefiles/natural_earth/cultural/ne_10m_admin_1_states_provinces.shp"
KOREA_URL = "https://raw.githubusercontent.com/southkorea/southkorea-maps/master/kostat/2013/json/skorea_municipalities_geo_simple.json"

def s(v):
    if v is None: return None
    if isinstance(v, float) and np.isnan(v): return None
    v = str(v).replace('\xa0', ' ').strip()
    return v or None

def rec(name, local=None, prov=None, dist=None, dist_local=None):
    return dict(name=s(name), local=s(local), prov=s(prov), dist=s(dist), dist_local=s(dist_local))

# adapters: row -> rec(...)
def ocha(n):
    def f(r):
        return rec(r.get(f'adm{n}_name'), r.get(f'adm{n}_name1'),
                   r.get('adm1_name') if n > 1 else None,
                   r.get('adm2_name') if n > 2 else None,
                   r.get('adm2_name1') if n > 2 else None)
    return f
def hdx(n):
    def f(r):
        return rec(r.get(f'ADM{n}_EN'), r.get(f'ADM{n}_JA'), r.get('ADM1_EN'),
                   r.get('ADM2_EN') if n > 2 else None)
    return f
def gadm(n):
    def f(r):
        return rec(r.get(f'NAME_{n}'), r.get(f'NL_NAME_{n}') or r.get(f'VARNAME_{n}'), r.get('NAME_1'))
    return f
def china_adapter(r):
    return rec(r.get('NAME_3'), r.get('NL_NAME_3'), r.get('NAME_1'), r.get('PREF'), r.get('PREF_ZH'))
def wof(r):     return rec(r.get('name_eng') or r.get('name'), r.get('name_zho'))
def korea(r):   return rec(r.get('name_eng'), r.get('name'))
def ne_row(r):  return rec(r.get('name_en') or r.get('name'), r.get('name_local'), r.get('admin'))
def ne_dist(r): return rec(r.get('name_en') or r.get('name'), r.get('name_local'))

CONFIGS = [
    ('China',       f"{D}/china/CHN_adm3.shp", None, china_adapter),
    ('Mongolia',    f"{D}/mongolia/mng_admin1.shp", None, ocha(1)),
    ('North Korea', f"{D}/northkorea/prk_admin2.shp", None, ocha(2)),
    ('South Korea', KOREA_URL, None, korea),
    ('Japan',       f"{D}/japan/jpn_admbnda_adm2_2019.shp", None, hdx(2)),
    ('Taiwan',      f"{D}/taiwan/whosonfirst-data-admin-tw-region-polygon.shp", None, wof),
    ('Russia',      f"{D}/russia/rus_admbnda_adm2_gadm_2022_v02.shp", (100,38,135,66), hdx(2)),
    ('Vietnam',     f"{D}/vietnam/VNM_adm2.shp", None, gadm(2)),
    ('Laos',        f"{D}/laos/lao_admin2.shp", None, ocha(2)),
    ('Cambodia',    f"{D}/cambodia/khm_admin2.shp", None, ocha(2)),
    ('Thailand',    f"{D}/thailand/tha_admin2.shp", None, ocha(2)),
    ('Myanmar',     f"{D}/myanmar/mmr_admin3.shp", None, ocha(3)),
    ('Malaysia',    f"{D}/malaysia/mys_admin2.shp", None, ocha(2)),
    ('Singapore',   f"{D}/malaysia/whosonfirst-data-admin-sg-borough-polygon.shp", None, wof),
    ('Philippines', f"{D}/philippines/phl_admbnda_adm3_psa_namria_20231106.shp", None, hdx(3)),
    ('Indonesia',   f"{D}/indonesia/idn_admbnda_adm3_bps_20200401.shp", None, hdx(3)),
    ('East Timor',  f"{D}/indonesia/tls_admin1.shp", None, ocha(1)),
    ('Brunei',      NE, None, ne_row, "admin = 'Brunei'"),
    ('Hong Kong',   NE, None, ne_dist, "admin = 'Hong Kong S.A.R.'"),
    ('Macau',       NE, None, ne_dist, "admin = 'Macau S.A.R'"),
]

def attach_china_prefecture(g):
    O = gpd.read_file(f"{D}/china/chn_admbnda_adm2_ocha_2020.shp")[['ADM2_EN','ADM2_ZH','geometry']]
    O = O.to_crs(PROJ4)
    pts = g[['geometry']].copy()
    pts['geometry'] = g.geometry.representative_point()
    j = gpd.sjoin(pts, O, how='left', predicate='within')
    j = j[~j.index.duplicated(keep='first')]
    g = g.copy()
    g['PREF'] = j['ADM2_EN'].reindex(g.index).values
    g['PREF_ZH'] = j['ADM2_ZH'].reindex(g.index).values
    return g

def main():
    t0 = time.time()
    arr = np.zeros((H, W), np.int32)
    names = [None]; gid = 0
    for cfg in CONFIGS:
        country, path, bbox, adapter = cfg[0], cfg[1], cfg[2], cfg[3]
        where = cfg[4] if len(cfg) > 4 else None
        kw = {}
        if bbox: kw['bbox'] = bbox
        if where: kw['where'] = where
        g = gpd.read_file(path, **kw)
        if g.crs is None: g = g.set_crs("EPSG:4326")
        g = g.to_crs(PROJ4).reset_index(drop=True)
        if country == 'China':
            g = attach_china_prefecture(g)
        shapes = []
        for _, row in g.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty: continue
            gid += 1
            names.append({**adapter(row), 'country': country})
            shapes.append((geom, gid))
        tmp = rasterize(shapes, out_shape=(H, W), transform=TRANSFORM, fill=0, dtype=np.int32)
        m = tmp > 0
        arr[m] = tmp[m]
        print(f"[{time.time()-t0:.0f}s] {country}: {len(shapes)} units (gid={gid})")
    np.save(f"{SCR}/adm_all.npy", arr)
    json.dump(names, open(f"{SCR}/adm_all_names.json", "w", encoding="utf-8"), ensure_ascii=False)
    print(f"[{time.time()-t0:.0f}s] saved adm_all: {gid} units, painted {int((arr>0).sum())} px")

if __name__ == "__main__":
    main()
