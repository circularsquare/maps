"""
Build subway graph from MTA shapefiles, find shortest paths for each
unique O-D station pair, and export everything as JSON for the web viz.

Graph model:
  - Each route gets its own layer: nodes are (x, y, route)
  - Edge weight = travel time in seconds (distance / train speed)
  - Station stop penalty: +60s for passing through a station
  - Transfer penalty: +180s for switching between routes at a station

Outputs:
  data.json  - { "stations": {...}, "paths": { "origId-destId": [[lon,lat],...] }, "od": [...] }
"""
import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import json
import csv
import time
import os
from multiprocessing import Pool, cpu_count
from scipy.spatial import KDTree
from pyproj import Transformer

# --- Config ---
LINES_SHP = "../data/nystreets/geo_export_6c9929cb-3ce9-4cb5-862d-dc06efa13f97.shp"
STATIONS_CSV = "../data/nystreets/MTA_Subway_Stations.csv"
OD_CSV = "data/od_wednesday_oct.csv"
OUT_JSON = "data.json"
TARGET_CRS = "EPSG:32618"

TRAIN_SPEED = 13.4       # m/s (~30 mph average including accel/decel)
STOP_PENALTY = 60         # seconds per station stop
TRANSFER_PENALTY = 180    # seconds to transfer between lines
SNAP_RADIUS = 850         # meters — generous to account for shapefile inaccuracy

transformer_to_ll = Transformer.from_crs(TARGET_CRS, "EPSG:4326", always_xy=True)
transformer_to_proj = Transformer.from_crs("EPSG:4326", TARGET_CRS, always_xy=True)


def build_subway_graph():
    """
    Build a layered subway graph where each route is its own set of nodes.
    After building the full-detail graph, contract degree-2 pass-through nodes
    so only stations and junctions remain (~1-2k nodes instead of ~227k).
    Detailed geometry is preserved on edges for visualization.
    """
    print("Loading subway lines...")
    lines_gdf = gpd.read_file(LINES_SHP).to_crs(TARGET_CRS).explode(index_parts=False)

    # --- 1. Build per-route track graph (detailed) ---
    G_detail = nx.Graph()
    route_nodes = {}  # route -> set of (x,y) nodes

    for _, row in lines_gdf.iterrows():
        line = row.geometry
        route = row['service']
        coords = list(line.coords)
        if route not in route_nodes:
            route_nodes[route] = set()
        for a, b in zip(coords, coords[1:]):
            a2, b2 = a[:2], b[:2]
            if a2 != b2:
                dist = np.linalg.norm(np.array(a2) - np.array(b2))
                na = (a2[0], a2[1], route)
                nb = (b2[0], b2[1], route)
                G_detail.add_edge(na, nb, length=dist)
                route_nodes[route].add(a2)
                route_nodes[route].add(b2)

    print(f"  Detail graph: {G_detail.number_of_nodes()} nodes, {G_detail.number_of_edges()} edges, {len(route_nodes)} routes")

    # --- 2. Load stations ---
    print("Loading stations...")
    stations_df = pd.read_csv(STATIONS_CSV)
    complexes = stations_df.groupby('Complex ID').agg({
        'Stop Name': 'first',
        'GTFS Latitude': 'mean',
        'GTFS Longitude': 'mean',
        'Daytime Routes': lambda x: ' '.join(set(' '.join(x).split()))
    }).reset_index()

    complex_coords = {}
    complex_info = {}
    for _, row in complexes.iterrows():
        cid = str(int(row['Complex ID']))
        lon, lat = row['GTFS Longitude'], row['GTFS Latitude']
        x, y = transformer_to_proj.transform(lon, lat)
        complex_coords[cid] = (x, y)
        complex_info[cid] = {
            'name': row['Stop Name'],
            'lat': round(lat, 6),
            'lon': round(lon, 6),
        }
    print(f"  {len(complex_coords)} station complexes")

    # --- 3. Snap stations to nearest track node per route ---
    print("Snapping stations to route layers...")
    route_trees = {}
    route_node_lists = {}
    for route, nodes in route_nodes.items():
        node_list = list(nodes)
        route_trees[route] = KDTree(node_list)
        route_node_lists[route] = node_list

    # Mark station nodes as "important" (don't contract them)
    important_nodes = set()
    station_route_nodes = {}  # cid -> list of (x, y, route) nodes
    for cid, (sx, sy) in complex_coords.items():
        station_route_nodes[cid] = []
        for route, tree in route_trees.items():
            dist, idx = tree.query([sx, sy])
            if dist < SNAP_RADIUS:
                track_xy = route_node_lists[route][idx]
                track_node = (track_xy[0], track_xy[1], route)
                if track_node in G_detail:
                    station_route_nodes[cid].append(track_node)
                    important_nodes.add(track_node)

    snapped_count = sum(1 for v in station_route_nodes.values() if v)
    print(f"  Snapped {snapped_count}/{len(complex_coords)} stations to routes")

    # --- 4. Contract degree-2 pass-through nodes ---
    # Keep: station nodes, junctions (degree != 2), endpoints (degree 1)
    # Contract: everything else (degree-2 pass-throughs on a single route)
    print("Contracting pass-through nodes...")

    # also mark junctions and endpoints as important
    for node in G_detail.nodes():
        deg = G_detail.degree(node)
        if deg != 2:
            important_nodes.add(node)

    G = nx.Graph()
    visited_edges = set()

    for start_node in important_nodes:
        if start_node not in G_detail:
            continue
        for neighbor in G_detail.neighbors(start_node):
            edge_key = (min(start_node, neighbor), max(start_node, neighbor))
            if edge_key in visited_edges:
                continue

            # walk along chain of degree-2 nodes until we hit another important node
            chain = [start_node, neighbor]
            total_dist = G_detail.edges[start_node, neighbor]['length']
            visited_edges.add(edge_key)

            current = neighbor
            prev = start_node
            while current not in important_nodes:
                neighbors = [n for n in G_detail.neighbors(current) if n != prev]
                if not neighbors:
                    break
                nxt = neighbors[0]
                edge_key2 = (min(current, nxt), max(current, nxt))
                visited_edges.add(edge_key2)
                total_dist += G_detail.edges[current, nxt]['length']
                chain.append(nxt)
                prev = current
                current = nxt

            end_node = chain[-1]
            travel_time = total_dist / TRAIN_SPEED

            # store the chain geometry (in projected coords) for later conversion
            geom = [(n[0], n[1]) for n in chain]  # (x, y) — drop route from tuple
            G.add_edge(start_node, end_node,
                       weight=travel_time, type='track', geom=geom)

    print(f"  Contracted graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # --- 5. Add stop penalties ---
    for cid, nodes in station_route_nodes.items():
        for track_node in nodes:
            if track_node not in G:
                continue
            for neighbor in list(G.neighbors(track_node)):
                if G.edges[track_node, neighbor].get('type') == 'track':
                    G.edges[track_node, neighbor]['weight'] += STOP_PENALTY / 2

    # --- 6. Add transfer edges between routes at same station ---
    print("Adding transfer edges...")
    transfer_count = 0
    for cid, nodes in station_route_nodes.items():
        if len(nodes) < 2:
            continue
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                G.add_edge(nodes[i], nodes[j], weight=TRANSFER_PENALTY, type='transfer')
                transfer_count += 1
    print(f"  Added {transfer_count} transfer edges")

    # --- 7. Add boarding nodes ---
    for cid, nodes in station_route_nodes.items():
        if not nodes:
            continue
        boarding_node = ('boarding', cid)
        for track_node in nodes:
            G.add_edge(boarding_node, track_node, weight=0, type='boarding')

    print(f"  Final graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G, complex_coords, complex_info, station_route_nodes


def load_od_pairs():
    """Load unique O-D pairs and hourly ridership from the CSV."""
    print("Loading O-D data...")
    od_by_hour = {}
    unique_pairs = set()

    with open(OD_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            oid = row['origin_station_complex_id']
            did = row['destination_station_complex_id']
            hour = int(row['hour_of_day'])
            riders = float(row['estimated_average_ridership'])
            if oid == did:
                continue
            pair = (oid, did)
            unique_pairs.add(pair)
            if pair not in od_by_hour:
                od_by_hour[pair] = {}
            od_by_hour[pair][hour] = riders

    print(f"  {len(unique_pairs)} unique O-D pairs")
    return od_by_hour, unique_pairs


_worker_G = None
_worker_station_route_nodes = None
_worker_transformer = None

def _init_worker(G, station_route_nodes):
    """Initialize each worker process with a copy of the graph."""
    global _worker_G, _worker_station_route_nodes, _worker_transformer
    _worker_G = G
    _worker_station_route_nodes = station_route_nodes
    _worker_transformer = Transformer.from_crs(TARGET_CRS, "EPSG:4326", always_xy=True)

def _solve_pair(pair):
    """Find shortest path for a single O-D pair. Runs in worker process."""
    oid, did = pair
    G = _worker_G
    srn = _worker_station_route_nodes
    t = _worker_transformer

    if oid not in srn or not srn[oid]:
        return None
    if did not in srn or not srn[did]:
        return None

    try:
        path_nodes = nx.shortest_path(G, ('boarding', oid), ('boarding', did), weight='weight')
        path_ll = []
        for k in range(len(path_nodes) - 1):
            a, b = path_nodes[k], path_nodes[k + 1]
            edge = G.edges[a, b]
            if edge.get('type') == 'track' and 'geom' in edge:
                geom = edge['geom']
                a_xy = (a[0], a[1])
                if len(geom) > 1 and np.linalg.norm(np.array(geom[0]) - np.array(a_xy)) > np.linalg.norm(np.array(geom[-1]) - np.array(a_xy)):
                    geom = geom[::-1]
                for x, y in geom:
                    lon, lat = t.transform(x, y)
                    path_ll.append([round(lon, 6), round(lat, 6)])
            else:
                if not isinstance(a[0], str):
                    lon, lat = t.transform(a[0], a[1])
                    path_ll.append([round(lon, 6), round(lat, 6)])
        last = path_nodes[-1]
        if not isinstance(last[0], str):
            lon, lat = t.transform(last[0], last[1])
            path_ll.append([round(lon, 6), round(lat, 6)])
        deduped = [path_ll[0]] if path_ll else []
        for p in path_ll[1:]:
            if p != deduped[-1]:
                deduped.append(p)
        if len(deduped) >= 2:
            return (f"{oid}-{did}", deduped)
    except nx.NetworkXNoPath:
        pass
    return None


def find_paths(G, station_route_nodes, unique_pairs):
    """Find shortest paths for all O-D pairs using multiprocessing."""
    ncpus = cpu_count()
    print(f"Finding paths using {ncpus} workers...")
    pairs_list = list(unique_pairs)
    total = len(pairs_list)
    paths = {}
    failed = 0
    t0 = time.time()

    with Pool(ncpus, initializer=_init_worker, initargs=(G, station_route_nodes)) as pool:
        for i, result in enumerate(pool.imap_unordered(_solve_pair, pairs_list, chunksize=200)):
            if result is not None:
                paths[result[0]] = result[1]
            else:
                failed += 1
            if (i + 1) % 2000 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (total - i - 1) / rate
                print(f"  [{(i+1)/total*100:5.1f}%] {i+1}/{total} pairs | {len(paths)} found, {failed} failed | {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining")

    elapsed = time.time() - t0
    print(f"  Found {len(paths)} paths, {failed} failed ({elapsed:.0f}s)")
    return paths


def build_od_array(od_by_hour):
    """Build compact O-D array for JSON: [origId, destId, hour, ridership]."""
    od_arr = []
    for (oid, did), hours in od_by_hour.items():
        for hour, riders in hours.items():
            od_arr.append([int(oid), int(did), hour, round(riders, 2)])
    od_arr.sort(key=lambda x: (x[2], -x[3]))
    return od_arr


def main():
    G, complex_coords, complex_info, station_route_nodes = build_subway_graph()
    od_by_hour, unique_pairs = load_od_pairs()
    paths = find_paths(G, station_route_nodes, unique_pairs)
    od_arr = build_od_array(od_by_hour)

    print(f"Writing {OUT_JSON}...")
    out = {
        'stations': complex_info,
        'paths': paths,
        'od': od_arr,
    }
    with open(OUT_JSON, 'w') as f:
        json.dump(out, f)

    import os
    size_mb = os.path.getsize(OUT_JSON) / 1024 / 1024
    print(f"Done! {OUT_JSON} is {size_mb:.1f} MB")


if __name__ == '__main__':
    main()
