"""
Read HydroRIVERS Asia once, keep the basins we care about, cache to pickle.

Slow step (reads a 207MB shapefile). Run once; build.py iterates on the cache.

    python prep.py [--basins 40] [--min-dis 5.0]
"""
import argparse, sys, time
import geopandas as gpd
import pandas as pd

SHP = 'C:/Users/anita/projects/maps/data/HydroRIVERS_v10_as_shp/HydroRIVERS_v10_as.shp'
CACHE = 'reaches.pkl'

ATTRS = ['HYRIV_ID', 'NEXT_DOWN', 'MAIN_RIV', 'LENGTH_KM', 'DIST_DN_KM',
         'DIST_UP_KM', 'CATCH_SKM', 'UPLAND_SKM', 'ENDORHEIC', 'DIS_AV_CMS',
         'ORD_STRA', 'ORD_CLAS', 'ORD_FLOW']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--basins', type=int, default=40,
                    help='keep the N largest basins by upstream area')
    ap.add_argument('--min-upland', type=float, default=200.0,
                    help='drop reaches with less upstream area than this (km2). '
                         'NOT discharge: discharge falls on exactly the drying '
                         'rivers we care about, so a discharge cut amputates '
                         'their tails and hides the effect. Upland area only '
                         'ever grows downstream, so it prunes headwaters safely.')
    a = ap.parse_args()

    t0 = time.time()
    print('pass 1: reading attributes to pick basins...', flush=True)
    at = gpd.read_file(SHP, columns=ATTRS, ignore_geometry=True)
    print(f'  {len(at):,} reaches, {time.time()-t0:.0f}s', flush=True)

    outlets = at[at.NEXT_DOWN == 0].sort_values('UPLAND_SKM', ascending=False)
    keep = outlets.head(a.basins)
    basins = set(keep.MAIN_RIV)
    print(f'  keeping {len(basins)} basins, '
          f'{(keep.ENDORHEIC > 0).sum()} of them endorheic, '
          f'smallest {keep.UPLAND_SKM.min():,.0f} km2', flush=True)

    # what survives the cut, so we know the render budget up front
    sel = at[at.MAIN_RIV.isin(basins) & (at.UPLAND_SKM >= a.min_upland)]
    print(f'  {len(sel):,} reaches survive upland >= {a.min_upland} km2', flush=True)

    print('pass 2: reading geometry (slow)...', flush=True)
    t1 = time.time()
    g = gpd.read_file(SHP, columns=ATTRS)
    print(f'  read {len(g):,} in {time.time()-t1:.0f}s', flush=True)

    g = g[g.MAIN_RIV.isin(basins) & (g.UPLAND_SKM >= a.min_upland)].copy()
    g = g.reset_index(drop=True)

    g.to_pickle(CACHE)
    print(f'wrote {CACHE}: {len(g):,} reaches, {g.LENGTH_KM.sum():,.0f} km, '
          f'{time.time()-t0:.0f}s total', flush=True)

    # per-basin summary so we can sanity-check the drying signal survives the cut
    rows = []
    for mr, sub in g.groupby('MAIN_RIV'):
        term = sub.loc[sub.DIST_DN_KM.idxmin()]
        rows.append(dict(MAIN_RIV=mr, endorheic=int(term.ENDORHEIC),
                         upland=term.UPLAND_SKM, length_km=sub.DIST_DN_KM.max(),
                         peak_cms=sub.DIS_AV_CMS.max(), term_cms=term.DIS_AV_CMS,
                         reaches=len(sub),
                         # must be 0.0 -- anything else means we cut the tail off
                         tail_km=sub.DIST_DN_KM.min()))
    s = pd.DataFrame(rows).sort_values('upland', ascending=False)
    s['pct_lost'] = (1 - s.term_cms / s.peak_cms) * 100
    s.to_csv('basins.csv', index=False)
    pd.set_option('display.width', 200)
    print('\n' + s.to_string(index=False, float_format=lambda v: f'{v:,.1f}'))


if __name__ == '__main__':
    sys.exit(main())
