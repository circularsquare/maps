"""
NYC subway eigenvector centrality map for the lecture.

Each station is a node; edges connect adjacent stations on the same route.
Adjacency is determined by projecting stations onto each route's line
geometry, splitting branches via linemerge, and connecting consecutive
stations within each component.
"""
import os
import time
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from shapely.ops import linemerge

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, '..'))
DATA_DIR = os.path.join(PROJ_DIR, 'data', 'nystreets')
OUT_DIR = os.path.join(PROJ_DIR, 'maidlecture')
OUT_PATH = os.path.join(OUT_DIR, 'subway_eigencentrality.png')

LINES_PATH = os.path.join(DATA_DIR, 'geo_export_6c9929cb-3ce9-4cb5-862d-dc06efa13f97.shp')
STATIONS_PATH = os.path.join(DATA_DIR, 'MTA_Subway_Stations.csv')

TARGET_CRS = "EPSG:32618"


def t(label, t0):
    print(f"  [{time.time() - t0:5.1f}s] {label}")


def main():
    t0 = time.time()

    print("Loading subway lines ...")
    lines = gpd.read_file(LINES_PATH).to_crs(TARGET_CRS)
    t(f"{len(lines)} line records", t0)

    print("Loading stations ...")
    sdf = pd.read_csv(STATIONS_PATH)
    stations = gpd.GeoDataFrame(
        sdf,
        geometry=gpd.points_from_xy(sdf['GTFS Longitude'], sdf['GTFS Latitude']),
        crs="EPSG:4326",
    ).to_crs(TARGET_CRS)
    # Collapse complexes: many physical platforms share a Complex ID
    # (e.g. Times Sq-42 St). Use Complex ID as node id so transfers within
    # one complex collapse into a single node.
    stations['routes_list'] = stations['Daytime Routes'].fillna('').str.split()
    # one row per Complex ID (pick first; aggregate routes)
    by_complex = (
        stations.groupby('Complex ID')
        .agg({
            'Stop Name': 'first',
            'geometry': 'first',
            'routes_list': lambda lst: sorted({r for sub in lst for r in sub}),
        })
        .reset_index()
    )
    by_complex = gpd.GeoDataFrame(by_complex, geometry='geometry', crs=stations.crs)
    by_complex = by_complex.set_index('Complex ID')
    t(f"{len(stations)} platforms -> {len(by_complex)} complexes", t0)

    # Map route -> geometry (skip '5 Peak' etc.)
    route_geoms = {}
    for _, row in lines.iterrows():
        svc = row['service']
        if pd.isna(svc) or ' ' in str(svc):
            continue
        route_geoms[svc] = row.geometry
    t(f"routes with geometry: {sorted(route_geoms.keys())}", t0)

    print("Building station adjacency graph ...")
    G = nx.Graph()
    for cid, st in by_complex.iterrows():
        G.add_node(cid,
                   x=st.geometry.x, y=st.geometry.y,
                   name=st['Stop Name'],
                   routes=st['routes_list'])

    edges_added = 0
    for route, geom in route_geoms.items():
        on_route = by_complex[by_complex['routes_list'].apply(lambda lst: route in lst)]
        if len(on_route) < 2:
            continue
        merged = linemerge(geom) if geom.geom_type == 'MultiLineString' else geom
        components = list(merged.geoms) if merged.geom_type == 'MultiLineString' else [merged]

        # assign each station to nearest component
        comp_of = {}
        for cid, st in on_route.iterrows():
            best_d, best_c = float('inf'), 0
            for ci, comp in enumerate(components):
                d = st.geometry.distance(comp)
                if d < best_d:
                    best_d, best_c = d, ci
            comp_of[cid] = best_c

        for ci, comp in enumerate(components):
            ranked = sorted(
                ((comp.project(by_complex.loc[cid].geometry), cid)
                 for cid in on_route.index if comp_of[cid] == ci),
                key=lambda x: x[0],
            )
            for i in range(len(ranked) - 1):
                a, b = ranked[i][1], ranked[i + 1][1]
                if a != b and not G.has_edge(a, b):
                    G.add_edge(a, b, route=route)
                    edges_added += 1
    t(f"added {edges_added} edges, graph has {G.number_of_edges()} unique edges", t0)

    # Largest CC (Staten Island Railway is its own component)
    cc = max(nx.connected_components(G), key=len)
    H = G.subgraph(cc).copy()
    t(f"largest CC: {H.number_of_nodes()} stations, {H.number_of_edges()} edges", t0)

    print("Computing eigenvector centrality ...")
    ec = nx.eigenvector_centrality_numpy(H, weight=None)
    vals = np.array(list(ec.values()))
    t(f"min={vals.min():.3e} max={vals.max():.3e}", t0)

    # Top-K printout for sanity
    top = sorted(ec.items(), key=lambda kv: -kv[1])[:12]
    print("  top-12 stations by eigenvector centrality:")
    for cid, c in top:
        print(f"    {c:.4f}  {H.nodes[cid]['name']:30s}  routes={H.nodes[cid]['routes']}")

    # ---- render ----
    print("Rendering ...")
    sids = list(H.nodes())
    cs = np.array([ec[s] for s in sids])
    xs = np.array([H.nodes[s]['x'] for s in sids])
    ys = np.array([H.nodes[s]['y'] for s in sids])

    cmin, cmax = cs.min(), cs.max()
    gamma = 0.55
    norm = PowerNorm(gamma=gamma, vmin=cmin, vmax=cmax)
    cmap = plt.get_cmap('inferno')

    width_m, height_m = xs.max() - xs.min(), ys.max() - ys.min()
    aspect = width_m / height_m
    fig_h = 14
    fig_w = max(5, fig_h * aspect)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200, facecolor='black')
    ax.set_facecolor('black')

    # graph edges as straight lines, faint gray
    for u, v in H.edges():
        ax.plot([H.nodes[u]['x'], H.nodes[v]['x']],
                [H.nodes[u]['y'], H.nodes[v]['y']],
                color='#555', linewidth=0.7, alpha=0.6, zorder=1)

    # stations colored by centrality
    nc = (cs - cmin) / (cmax - cmin + 1e-12)
    sizes = 6 + 110 * np.power(nc, 0.7)
    ax.scatter(xs, ys, c=cs, cmap=cmap, norm=norm, s=sizes,
               edgecolors='white', linewidths=0.25, zorder=3)

    ax.set_aspect('equal')
    ax.axis('off')
    pad = 800
    ax.set_xlim(xs.min() - pad, xs.max() + pad)
    ax.set_ylim(ys.min() - pad, ys.max() + pad)
    plt.tight_layout(pad=0)

    os.makedirs(OUT_DIR, exist_ok=True)
    plt.savefig(OUT_PATH, dpi=200, facecolor='black',
                bbox_inches='tight', pad_inches=0.05)
    t(f"saved: {OUT_PATH}", t0)


if __name__ == '__main__':
    main()
