"""
Manhattan eigenvector centrality map for the lecture.
Loads g_proj_d.pkl (NYC drive network in EPSG:32618), filters to the
Manhattan polygon, computes eigenvector centrality on the undirected
simple graph (largest CC), and renders edges colored by centrality.
"""
import os
import pickle
import time
import numpy as np
import networkx as nx
import osmnx as ox
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import PowerNorm
from shapely.geometry import Point

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GRAPH_PATH = os.path.join(SCRIPT_DIR, 'g_proj_d.pkl')
OUT_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'maidlecture'))
OUT_PATH = os.path.join(OUT_DIR, 'manhattan_eigencentrality.png')
TARGET_CRS = "EPSG:32618"


def t(label, t0):
    print(f"  [{time.time() - t0:6.1f}s] {label}")


def main():
    t0 = time.time()
    print(f"Loading {GRAPH_PATH} ...")
    with open(GRAPH_PATH, 'rb') as f:
        G = pickle.load(f)
    t(f"loaded ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)", t0)

    print("Fetching Manhattan polygon (cached by osmnx if available) ...")
    gdf = ox.geocode_to_gdf("Manhattan, New York, USA")
    gdf = gdf.to_crs(TARGET_CRS)
    poly = gdf.geometry.iloc[0]
    t("polygon ready", t0)

    print("Filtering nodes inside Manhattan ...")
    minx, miny, maxx, maxy = poly.bounds
    keep = []
    for n, d in G.nodes(data=True):
        x, y = d['x'], d['y']
        if minx <= x <= maxx and miny <= y <= maxy:
            if poly.contains(Point(x, y)):
                keep.append(n)
    H = G.subgraph(keep).copy()
    t(f"filtered to {H.number_of_nodes()} nodes / {H.number_of_edges()} edges", t0)

    print("Building undirected simple graph and taking largest CC ...")
    H_simple = nx.Graph()
    H_simple.add_nodes_from(H.nodes(data=True))
    for u, v in H.edges():
        if u != v:
            H_simple.add_edge(u, v)
    biggest = max(nx.connected_components(H_simple), key=len)
    H_cc = H_simple.subgraph(biggest).copy()
    t(f"largest CC: {H_cc.number_of_nodes()} nodes, {H_cc.number_of_edges()} edges", t0)

    print("Computing eigenvector centrality (numpy) ...")
    ec = nx.eigenvector_centrality_numpy(H_cc, weight=None)
    vals = np.array(list(ec.values()))
    t(f"centrality computed (min={vals.min():.2e}, max={vals.max():.2e})", t0)

    print("Building edge segments from MultiDiGraph geometry ...")
    biggest_set = set(biggest)
    segments = []
    edge_c = []
    for u, v, k, d in H.edges(keys=True, data=True):
        if u not in biggest_set or v not in biggest_set:
            continue
        c = 0.5 * (ec[u] + ec[v])
        if 'geometry' in d and d['geometry'] is not None:
            coords = list(d['geometry'].coords)
        else:
            coords = [(H.nodes[u]['x'], H.nodes[u]['y']),
                      (H.nodes[v]['x'], H.nodes[v]['y'])]
        segments.append(coords)
        edge_c.append(c)
    edge_c = np.array(edge_c)
    t(f"prepared {len(segments)} edge segments", t0)

    print("Rendering ...")
    cmin, cmax = edge_c.min(), edge_c.max()
    gamma = 0.45  # power transform: emphasize the high-centrality tail
    norm = PowerNorm(gamma=gamma, vmin=cmin, vmax=cmax)
    cmap = plt.get_cmap('inferno')
    colors = cmap(norm(edge_c))
    nc = (edge_c - cmin) / (cmax - cmin + 1e-12)
    linewidths = 0.25 + 1.6 * np.power(nc, gamma)

    xs = np.array([d['x'] for _, d in H_cc.nodes(data=True)])
    ys = np.array([d['y'] for _, d in H_cc.nodes(data=True)])
    pad = 200
    width_m = (xs.max() - xs.min())
    height_m = (ys.max() - ys.min())
    aspect = width_m / height_m
    fig_h = 16
    fig_w = max(3, fig_h * aspect)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=220, facecolor='black')
    ax.set_facecolor('black')
    lc = LineCollection(segments, colors=colors, linewidths=linewidths,
                        antialiased=True, capstyle='round')
    ax.add_collection(lc)
    ax.set_xlim(xs.min() - pad, xs.max() + pad)
    ax.set_ylim(ys.min() - pad, ys.max() + pad)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout(pad=0)

    os.makedirs(OUT_DIR, exist_ok=True)
    plt.savefig(OUT_PATH, dpi=220, facecolor='black',
                bbox_inches='tight', pad_inches=0)
    t(f"saved: {OUT_PATH}", t0)


if __name__ == '__main__':
    main()
