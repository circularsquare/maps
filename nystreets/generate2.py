import geopandas as gpd
import networkx as nx
import osmnx as ox
import random
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, NoNorm, PowerNorm
import pickle
import numpy as np
import math
import pandas as pd
from scipy.spatial import KDTree
from itertools import combinations
from pyproj import Transformer
from shapely import LineString, MultiLineString
from shapely.geometry import Point, box
import multiprocessing
from functools import partial
import time
from collections import defaultdict


mta_colors = {
    'A': '#1373c2', 'C': '#0b4ab8', 'E': '#0b33b8', 'SIR': '#08179C', #A is 0027a6
    'B': '#ff8c19', 'D': '#ff7919', 'F': '#FF6319', 'M': '#e66325', 
    'G': '#6CBE45', 'J': '#996633', 'Z': '#825c35', 'L': '#A7A9AC',
    'N': '#f7d52a', 'Q': '#f2e422', 'R': '#fcBb0a', 'W': '#f0b618', 
    '1': '#d92121', '2': '#EE352E', '3': '#ed5247', 
    '4': '#21b559', '5': '#1dad46', '6': '#00933C', '7': '#B933AD', 
    'S': '#808183', 'SF': '#808183', 'ST': '#808183', 'SR': '#808183'}
subway_speeds = {
    'A': 1, 'C': 1, 'E': 1, 'SIR': 1,
    'B': 1, 'D': 1, 'F': 1, 'M': 1,
    'G': 1, 'J': 1, 'Z': 0.01, 'L': 1,
    'N': 1, 'Q': 1, 'R': 1, 'W': 1,
    '1': 1, '2': 1, '3': 1,
    '4': 1, '5': 1, '6': 1, '7': 1,
    'S': 1, 'SF': 1, 'ST': 1, 'SR': 1,
}
subway_lines = ['A', 'C', 'E', 'SIR', 'B', 'D', 'F', 'M', 'G', 'J', 'Z', 'L', 'N', 'Q', 'R', 'W', '1', '2', '3', '4', '5', '6', '7', 'S']
def route_offsets(route): # jitter by this amount by index
    if route in subway_lines:
        return subway_lines.index(route) * 0.2
    return -0.2
def jitter_line(line, route_id):
    offset = np.array(route_offsets(route_id))
    # Add the offset to every coordinate point in the line
    new_coords = [tuple(np.array(p) + offset) for p in line.coords]
    return LineString(new_coords)
def subdivide_line(line, max_length=30):
    if line.length <= max_length:
        return [line]
    segments = []
    num_segments = math.ceil(line.length / max_length)
    points = [line.interpolate(i / num_segments, normalized=True) for i in range(num_segments + 1)]
    for p1, p2 in zip(points[:-1], points[1:]):
        segments.append(LineString([p1, p2]))
    return segments


SUBWAY_SPEED = 1 # used in euclidean dist
TARGET_CRS = "EPSG:32618"

def jittered_weight(u, v, data):
    jitterSize = 1
    base_length = data['length']
    jitter = random.gauss(1, jitterSize)
    return base_length * jitter
def jitter_sqrt(u, v, data):
    strength = 1.0 # strength of 1 means 10 m path will be jittered by 1 * sqrt(10) = 3
                # or 100 m path will get jittered by 1 * sqrt (100) = 10
    sigma = strength * math.sqrt(data['length'])
    noise = np.abs(random.gauss(0, sigma)) # abs makes heuristic never overestimate
    return max(0.001, data['length'] + noise)
def euclidean_dist_heuristic(u, v):
    return np.linalg.norm(np.array(u) - np.array(v)) / SUBWAY_SPEED 
def euclidean_dist(u, v):
    return np.linalg.norm(np.array(u) - np.array(v))

def pathfind(source_target_pair, graph, weight_func, heuristic_func, exponent_val):
    source_node, target_node = source_target_pair
    source_node = tuple(source_node)
    target_node = tuple(target_node)
    try: 
        path = nx.astar_path(graph, source_node, target_node, 
                            weight=weight_func, heuristic=heuristic_func)
        # path = nx.shortest_path(graph, source_node, target_node, 
        #                 weight=weight_func)
        network_dist = sum(graph.edges[u, v]['length'] for u, v in zip(path[:-1], path[1:]))
        distance = np.linalg.norm(np.array(source_node) - np.array(target_node))
        directness = distance / (network_dist + 0.001)
        influence = directness ** exponent_val
        path_edges = list(zip(path[:-1], path[1:]))
        # Return all the edges from this path and the influence score for this path
        return (path_edges, influence)
    except nx.NetworkXNoPath:
        return None

def findUsage(num_iterations = 10000000):
    print("Downloading and building graph from OpenStreetMap...")
    place_name = "New York, New York"
    TARGET_CRS = "EPSG:32618"

    bbox = (-74.2768441,40.497615,-73.6334,40.981972)
    center_lon, center_lat = -73.920135,40.710313
    #center_lat, center_lon = 40.664279, -73.865331 # test near howard beach
    transformer = Transformer.from_crs("EPSG:4326", TARGET_CRS, always_xy = True)
    center_x, center_y = transformer.transform(center_lon, center_lat)
    center_point = np.array([center_x, center_y])

    # --- 1. Get data by uncommenting below 4 lines! ---
    # G_directed = ox.graph_from_bbox(bbox = bbox, network_type='drive')
    # G_proj = ox.project_graph(G_directed, to_crs=TARGET_CRS)
    # with open('g_proj_d.pkl', 'wb') as f:
    #     pickle.dump(G_proj, f)

    with open('g_proj_d.pkl', 'rb') as f:
        G_proj = pickle.load(f)

    true_lengths = {}
    # Iterate through the original, unsimplified G_proj to get the correct lengths
    for u_osmid, v_osmid, data in G_proj.edges(data=True):
        u_coords = (G_proj.nodes[u_osmid]['x'], G_proj.nodes[u_osmid]['y'])
        v_coords = (G_proj.nodes[v_osmid]['x'], G_proj.nodes[v_osmid]['y'])
        edge_key = tuple(sorted((u_coords, v_coords)))
        length = data['length']
        if edge_key not in true_lengths or length < true_lengths[edge_key]:
            true_lengths[edge_key] = length # if multiple edges with same endpoints, only keep shortest one

    G_initial = nx.Graph(G_proj)

    # --- 2. Convert Graph Nodes to Coordinate Tuples ---
    print("Converting graph nodes from OSM IDs to coordinates...")
    pos = {node: (data['x'], data['y']) for node, data in G_initial.nodes(data=True)}
    G = nx.Graph() # This will be our final graph with coordinate nodes
    for u, v, data in G_initial.edges(data=True):
        u_coords = pos[u]
        v_coords = pos[v]
        G.add_edge(u_coords, v_coords, **data)
    for u, v in G.edges():
        edge_key = tuple(sorted((u, v)))
        if edge_key in true_lengths:
            G.edges[u, v]['length'] = true_lengths[edge_key]
    street_nodes = list(G.nodes())
    print("Graph creation and length correction complete.")

    # # # --- 2.5. subway! ---
    # print('loading subway data...')

    # #lines_path = "data/nystreets/routes_nyc_subway_may2016.shp" # 'route_id'
    # lines_path = 'data/nystreets/geo_export_6c9929cb-3ce9-4cb5-862d-dc06efa13f97.shp' # 'service'
    # lines_gdf = gpd.read_file(lines_path)
    # lines_gdf = lines_gdf.to_crs(TARGET_CRS).explode()
    # #here
    # lines_gdf['route_id'] = lines_gdf['service'].map(lambda x: 'S' if len(x) == 2 else x) # call all shuttle  'S'
    # lines_gdf['route_id'] = lines_gdf['route_id'].map(lambda x: '5' if x=='5 Peak' else x) 

    # all_subdivided_lines_data = []

    # # Iterate through your original subway lines DataFrame
    # for index, row in lines_gdf.iterrows():
    #     original_line = row.geometry
    #     original_data = row.to_dict()
    #     new_segments = subdivide_line(original_line)
    #     # For each new, small segment, create a new row of data
    #     for seg in new_segments:
    #         new_row_data = original_data.copy()
    #         new_row_data['geometry'] = seg
    #         all_subdivided_lines_data.append(new_row_data)

    # lines_gdf_subdivided = gpd.GeoDataFrame(all_subdivided_lines_data, crs=TARGET_CRS)
    # print(f"Subdivision complete. Original {len(lines_gdf)} lines became {len(lines_gdf_subdivided)} segments.")
    # lines_gdf = lines_gdf_subdivided


    # stations_df = pd.read_csv("data/nystreets/MTA_Subway_Stations.csv")
    # stations_gdf = gpd.GeoDataFrame(
    #     stations_df,
    #     geometry=gpd.points_from_xy(stations_df['GTFS Longitude'], stations_df['GTFS Latitude']),
    #     crs="EPSG:4326").to_crs(TARGET_CRS)
    # stations_gdf['routes'] = stations_gdf['Daytime Routes'].str.split(' ')
    # stations_gdf.columns = stations_gdf.columns.str.lower().str.replace(' ', '_')

    # # prepare street graph G
    # for u, v, data in G.edges(data=True):
    #     G.edges[u, v]['type'] = 'street'

    # subG = nx.Graph()

    # # jitter lines
    # lines_gdf['geometry'] = lines_gdf.apply(
    #     lambda row: jitter_line(row.geometry, row['route_id']),
    #     axis=1)

    # all_subway_data = [] # We'll create a list of dictionaries
    # for index, row in lines_gdf.iterrows():
    #     line = row.geometry
    #     route_id = row['route_id'] 
    #     segments = [LineString(pair) for pair in zip(line.coords, line.coords[1:])]
    #     for seg in segments:
    #         all_subway_data.append({'geometry': seg, 'route_id': route_id})

    # # --- Part 2: Build the graph from our new, detailed list ---
    # for seg_data in all_subway_data:
    #     segment = seg_data['geometry']
    #     route_id = seg_data['route_id'] 
    #     start_node = (segment.coords[0])
    #     end_node = (segment.coords[-1])
    #     if start_node != end_node:
    #         travel_time = segment.length / (SUBWAY_SPEED * subway_speeds[route_id])
    #         subG.add_edge(start_node, end_node,
    #                     length=travel_time, 
    #                     type='subway',
    #                     route_id=route_id) 
    # endpoints_by_route = defaultdict(list)
    # for index, row in lines_gdf.iterrows():
    #     line = row.geometry
    #     route = row['route_id']
    #     if len(line.coords) > 1:
    #         endpoints_by_route[route].append(line.coords[0])
    #         endpoints_by_route[route].append(line.coords[-1])
    # gaps_bridged = 0
    # for route, endpoints in endpoints_by_route.items():
    #     if len(endpoints) < 2: continue
    #     unique_endpoints = list(set(endpoints))
    #     endpoint_tree = KDTree(unique_endpoints)
    #     gap_pairs = endpoint_tree.query_pairs(r=20) # radius for closing gaps
    #     for (i, j) in gap_pairs:
    #         node1 = unique_endpoints[i]
    #         node2 = unique_endpoints[j]
    #         dist = euclidean_dist(node1, node2)
    #         if dist > 20:
    #             print('oh no')
    #         subG.add_edge(node1, node2, length=dist/SUBWAY_SPEED, type='subway', route_id = route)
    #         gaps_bridged += 1
    # print(f"Bridged {gaps_bridged} gaps in the subway network.")

    # G = nx.compose(G, subG)

    
    # # connecting stations!
    # ENTER_STATION = 20 # meters from station to street (walking is ~ 1 m/s)
    # TRAIN_WAIT = 180 # meters from station to getting on train (applied both ways)

    # print("Connecting networks at station entrances...")
    # station_id_to_node = {}
    # for station in stations_gdf.itertuples():
    #     platform_coords = (station.geometry.x, station.geometry.y)
    #     G.add_node(platform_coords)
    #     station_id_to_node[station.station_id] = platform_coords

    # # subway_nodes = [n for n, d in G.nodes(data=True) if G.degree(n) > 0 and all(G.edges[n, neighbor].get('type') == 'subway' for neighbor in G.neighbors(n))]
    # # subway_tree = KDTree(subway_nodes)

    # street_tree = KDTree(street_nodes) # connect station to street
    # for station_id, platform_node in station_id_to_node.items():
    #     dist, idx = street_tree.query(platform_node)
    #     nearest_street_node = street_nodes[idx]
    #     entrance_time = dist + ENTER_STATION
    #     G.add_edge(platform_node, nearest_street_node, length = dist + ENTER_STATION, type='station_entrance')

    # g_nodes = list(G.nodes())
    # node_tree = KDTree(g_nodes)


    # nodes_by_route = defaultdict(list)
    # for u, v, data in G.edges(data=True):
    #     if data.get('type') == 'subway':
    #         route_id = data.get('route_id')
    #         if route_id:
    #             nodes_by_route[route_id].append(u)
    #             nodes_by_route[route_id].append(v)
    # for route, nodes in nodes_by_route.items():
    #     nodes_by_route[route] = list(set(nodes))

    # # 3. Loop through each station and find the best edge to split
    # for station in stations_gdf.itertuples():
    #     for route in station.routes:
    #         platform_coords = (station.geometry.x, station.geometry.y)
    #         platform_point = np.array(platform_coords)
    #         G.add_node(platform_coords)

    #         route_nodes = nodes_by_route.get(route)
    #         route_tree = KDTree(route_nodes)

    #         dist, idx = route_tree.query(platform_coords)
    #         closest_node = route_nodes[idx]
            
    #         best_edge = None
    #         min_dist_to_edge = float('inf')
    #         for neighbor in G.neighbors(closest_node):
    #             edge_geom = LineString([closest_node, neighbor])
    #             dist = edge_geom.distance(Point(platform_coords))
    #             if dist < min_dist_to_edge and G.edges[closest_node, neighbor]['type'] == 'subway':
    #                 min_dist_to_edge = dist
    #                 best_edge = (closest_node, neighbor)
                    
    #         if best_edge:
    #             # We found the correct edge in G to split
    #             u, v = best_edge
    #             edge_data = G.edges[u, v].copy() # Get its attributes
                
    #             # Create the new connection point
    #             new_node = tuple(LineString(best_edge).interpolate(LineString(best_edge).project(Point(platform_coords))).coords[0])
                
    #             # Perform the split-and-insert directly on G
    #             G.remove_edge(u, v)
    #             G.add_edge(u, new_node, **edge_data)
    #             G.add_edge(new_node, v, **edge_data)

    #             edist = euclidean_dist(new_node, platform_coords)
    #             if edist > 200:
    #                 print(f"platform track dist {edist:.0f}m for route {route} at {station.stop_name}")
    #                 print(euclidean_dist(u, platform_coords))
    #                 print(euclidean_dist(v, platform_coords))
    #                 print(u)
    #                 print(v)
    #                 print(new_node)
    #                 print(platform_coords)
                
    #             # Add the final connection from the platform to the new split point
    #             G.add_edge(platform_coords, new_node, 
    #                     length=TRAIN_WAIT, type='platform_access')

    # # adding stop time penalties
    # STOP_TIME = 30 # applied twice while passing a station
    # track_nodes_with_stops = set()
    # for station_id, platform_node in station_id_to_node.items():
    #     for neighbor in G.neighbors(platform_node):
    #         if G.edges[platform_node, neighbor].get('type') == 'platform_access':
    #             track_nodes_with_stops.add(neighbor)
    # for track_node in track_nodes_with_stops:
    #     for neighbor in G.neighbors(track_node):
    #         edge_data = G.edges[track_node, neighbor]
    #         if edge_data.get('type') == 'subway':
    #             #G.edges[u, v]['length'] = G.edges[u, v]['length']
    #             G.edges[track_node, neighbor]['length'] = G.edges[track_node,neighbor]['length'] + STOP_TIME

    
        
    # print("Multi-modal graph creation complete!")



    # --- 2.6 load population! --
    print('loading population...')
    kontur_filepath = "data/nystreets/kontur_population_US_20231101.gpkg"
    pop_gdf_info = gpd.read_file(kontur_filepath, rows=1)

    west, south, east, north = -74.253845, 40.497615, -73.656464, 40.981972
    bbox_gdf_latlon = gpd.GeoDataFrame(
        geometry=[box(west, south, east, north)],
        crs="EPSG:4326")
    bbox_gdf_projected = bbox_gdf_latlon.to_crs("EPSG:3857")
    projected_bbox_coords = tuple(bbox_gdf_projected.total_bounds)
    pop_gdf = gpd.read_file(
        kontur_filepath,
        bbox=projected_bbox_coords).to_crs(TARGET_CRS)
        
    pop_centers_gdf = pop_gdf.copy() # collapse to centers
    pop_centers_gdf['geometry'] = pop_centers_gdf.geometry.centroid
    pop_points = np.array([p.coords[0] for p in pop_centers_gdf.geometry])
    pop_values = pop_centers_gdf['population'].values
    pop_tree = KDTree(pop_points)
    print(f"Data loaded successfully. Found {len(pop_gdf)} population grid cells.")

    # --- 3. Run Pathfinding Simulation ---
    print("Selecting paths...")
    all_nodes = list(G.nodes)
    all_edges = list(G.edges)
    final_weights = []
    for u, v, data in G.edges(data=True):
        G.edges[u, v]['usage'] = 0

    shortDistance = 1000
    exponent = 2.3  
    proximity_sigma = 27000 # 1.6 km per mile

    # calculate probability of being picked of edges
    u_coords = np.array([u for u, v in all_edges])
    v_coords = np.array([v for u, v in all_edges])
    midpoints = (u_coords + v_coords)/2
    lengths = np.array(list(nx.get_edge_attributes(G, 'length').values()))
    # dist_center = np.linalg.norm(u_coords - center_point, axis=1)
    prox_weight = 1 #unused: np.exp(-(dist_center**2) / (2 * proximity_sigma**2))

    pop_distances, pop_indices = pop_tree.query(midpoints, k=1)
    edge_pops = pop_values[pop_indices]
    # normalize by total edge length in hexagon
    temp_df = pd.DataFrame({'edge_length': lengths, 'population': edge_pops, 'hexagon_id': pop_indices})
    hexagon_total_lengths = temp_df.groupby('hexagon_id')['edge_length'].sum()
    temp_df['hexagon_total_length'] = temp_df['hexagon_id'].map(hexagon_total_lengths)
    normalized_length_weight = temp_df['edge_length'] / temp_df['hexagon_total_length']
    # square root because weight will be applied once at source and once at destination
    final_weights = np.sqrt(temp_df['population']) * normalized_length_weight

    # do random selections
    probabilities = final_weights / np.sum(final_weights)
    edge_indices = np.arange(len(all_edges))
    all_edges_array = np.array(all_edges)
    chosen_indices = np.random.choice(edge_indices, size=(num_iterations, 2), p=probabilities)
    source_edges = all_edges_array[chosen_indices[:, 0]]
    target_edges = all_edges_array[chosen_indices[:, 1]]
    source_endpoint_choices = np.random.randint(2, size=num_iterations)
    target_endpoint_choices = np.random.randint(2, size=num_iterations)
    source_nodes = source_edges[np.arange(num_iterations), source_endpoint_choices]
    target_nodes = target_edges[np.arange(num_iterations), target_endpoint_choices]
    # calculate distances
    dists = np.linalg.norm(source_nodes - target_nodes, axis=1)
    # filter out bad paths
    acceptance_mask = dists <= shortDistance
    long_dists = dists[~acceptance_mask]
    probs_long = (shortDistance / long_dists) ** exponent
    rolls_long = np.random.random(size=len(long_dists))
    acceptance_mask[~acceptance_mask] = rolls_long < probs_long
    final_source_nodes = source_nodes[acceptance_mask]
    final_target_nodes = target_nodes[acceptance_mask]
    successful_selections = len(final_source_nodes)
    print(f"Sampling complete. {successful_selections} pairs were accepted for pathfinding.")

    print("Running pathfinding on accepted pairs...")
    start_time = time.time()
    successful_paths = 0
    failed_paths = 0

    parallel = successful_selections > 1200    
    if parallel:
        print('running parallel...')
        pathfinding_jobs = list(zip(final_source_nodes, final_target_nodes))
        worker_func = partial(pathfind, graph=G, weight_func = jitter_sqrt, 
            heuristic_func = euclidean_dist_heuristic, exponent_val=exponent)
        num_cores = multiprocessing.cpu_count()
        with multiprocessing.Pool(processes = num_cores - 2) as pool:
            results = pool.map(worker_func, pathfinding_jobs)
        successful_paths = 0
        for result in results:
            if result is not None:
                path_edges, influence = result
                successful_paths += 1
                for u, v in path_edges:
                    G.edges[u, v]['usage'] += influence
    else: 
        for i, (source_node, target_node) in enumerate(zip(final_source_nodes, final_target_nodes)):
            try:
                path = nx.astar_path(G, tuple(source_node), tuple(target_node), 
                                    weight=jitter_sqrt, heuristic=euclidean_dist_heuristic)
                network_dist = sum(G.edges[u, v]['length'] for u, v in zip(path[:-1], path[1:]))
                distance = euclidean_dist(source_node, target_node)
                directness = distance / (network_dist + 0.001)
                if directness >= SUBWAY_SPEED:
                    print ('directness ' + str(directness))
                for u, v in zip(path[:-1], path[1:]):
                    G.edges[u, v]['usage'] += directness**exponent
                successful_paths += 1
                if successful_paths > 0 and successful_paths % 1000 == 0:
                    print(f"  ...found {successful_paths}/{successful_selections} paths...")

            except nx.NetworkXNoPath:
                failed_paths += 1


    end_time = time.time()
    print(f"Pathfinding complete. {successful_paths}/{successful_selections} successful paths found. took {end_time-start_time:.2f} seconds, {(end_time-start_time)/successful_paths:5f} each.")
    
    # --- Step 4: Create Final GeoDataFrame ---
    # --- 4a: Start with the original street geometries from G_proj ---
    df_streets = ox.graph_to_gdfs(G_proj, nodes=False)
    print(df_streets.columns)
    df_streets = df_streets.reset_index()

    print('Filtering out detour paths...')
    pos = {node: (data['x'], data['y']) for node, data in G_proj.nodes(data=True)}
    def get_edge_key(row):
        return tuple(sorted((pos[row['u']], pos[row['v']])))
    df_streets['edge_key'] = df_streets.apply(get_edge_key, axis=1)

    def is_shortest_path(row):
        return np.isclose(row['length'], true_lengths.get(row['edge_key'], -1))
    df_streets = df_streets[df_streets.apply(is_shortest_path, axis=1)].copy()

    # Map the calculated 'usage' from your main graph G back to this GeoDataFrame
    print("Mapping usage counts to street geometries...")
    # First, create a simple lookup dictionary for usage from G.
    usage_lookup = {tuple(sorted((u, v))): data.get('usage', 0) for u, v, data in G.edges(data=True)}
    df_streets['usage'] = df_streets['edge_key'].map(usage_lookup).fillna(0)
    df_streets['type'] = 'street'
    df_streets['route_id'] = None
    df_streets = df_streets[['geometry', 'usage', 'type', 'route_id']] # Keep only needed columns

    # --- 4b: Create geometries for all NON-street edges (subway, transfers, etc.) ---
    # print("Creating geometries for subway and other network edges...")
    # other_edges_data = []
    # for u, v, data in G.edges(data=True):
    #     if data.get('type') != 'street':
    #         geom = LineString([u, v])
    #         other_edges_data.append({
    #             'geometry': geom,
    #             'usage': data.get('usage', 0),
    #             'type': data.get('type', 'unknown'),
    #             'route_id': data.get('route_id', None)
    #         })
    # df_other = gpd.GeoDataFrame(other_edges_data, crs=TARGET_CRS)
    # # --- 4c: Combine the two GeoDataFrames ---
    # print("Combining street and subway GeoDataFrames...")
    # df = pd.concat([df_streets, df_other], ignore_index=True)

    df = df_streets # comment this out if doing subway!!

    # Select only the columns you need for plotting to keep things clean
    df = df[['geometry', 'usage', 'type', 'route_id']]
    print("Final combined GeoDataFrame created successfully.")

    # save!
    df.to_file('usage_map2.gpkg', driver='GPKG')
    print('File saved.')

if __name__ == '__main__':
    
    findUsage(80000 * 100) 

    print('loading file...')
    df = gpd.read_file('usage_map2.gpkg')
    #df = gpd.read_file('usage_map.gpkg')

    print("Plotting final map...")
    imageSize = 100
    fig, ax = plt.subplots(figsize=(imageSize, imageSize))
    ax.set_facecolor('black')
    ax.set_axis_off()
    xmin, ymin, xmax, ymax = df.total_bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    cmap = 'magma'

    width_exp = 0.7
    maxWidth = 0.17
    vmin = 0.8
    maxUsage = df.usage.max()

    streets_df = df[(df['type'] != 'subway')].sort_values('usage', ascending=True)
    subway_df = df[df['type'] == 'subway'].copy()
    subway_df['routeColor'] = subway_df['route_id'].map(mta_colors).fillna('#FFFFFF')

    fig, ax = plt.subplots(figsize=(imageSize, imageSize))
    ax.set_facecolor('black')
    ax.set_axis_off()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    streets_df.plot(ax=ax, column='usage', cmap=cmap, 
        linewidth = maxWidth * imageSize * streets_df['usage']**width_exp / maxUsage**width_exp, 
        norm=LogNorm(vmin=vmin, vmax=df['usage'].max()*1.0), capstyle='round'    
    )
    plt.savefig('nystreets/osm' + '.png', pad_inches=0, facecolor='black')
    print('plot saved!')

    
    # streets_df.plot(
    #     ax=ax,
    #     column='usage',
    #     cmap=cmap, 
    #     linewidth = maxWidth * imageSize * streets_df['usage']**width_exp / maxUsage**width_exp, 
    #     norm=LogNorm(vmin=vmin, vmax=df['usage'].max()*1.0),
    #     capstyle='round'    
    # )
    
    # subway_df = subway_df.sort_values('usage', ascending=True)
    # subway_df.plot( ax=ax, color = subway_df['routeColor'],
    #     linewidth = maxWidth * imageSize * subway_df['usage']**width_exp / maxUsage**width_exp, 
    #     alpha=0.99, capstyle='round', 
    # )
    # plt.savefig('nystreets/osmsuball' + '.png', pad_inches=0, facecolor='black')
    # print('plot saved!')

    
    # for route in subway_lines:
    #     line_df = df[df.route_id == route]
    #     if not line_df.empty:
    #         fig_layer, ax_layer = plt.subplots(figsize=(imageSize, imageSize))
    #         ax_layer.set_facecolor('none')
    #         ax_layer.set_axis_off()
    #         ax_layer.set_xlim(xmin, xmax)
    #         ax_layer.set_ylim(ymin, ymax)

    #         line_df.plot(ax=ax_layer,
    #             color = mta_colors[route],
    #             linewidth = np.maximum(0.1, maxWidth * imageSize * line_df['usage']**width_exp / maxUsage**width_exp), # max is new
    #             capstyle='round')
    #         fig_layer.savefig('nystreets/lines/' + route + '.png', pad_inches=0, facecolor='none')
    #         plt.close(fig_layer)

    # print('line plots saved!')





# #PLOTTING ----------------------------------
#     #bbox_latlon = (-73.988, 40.745, -73.970, 40.758) 
#     bbox_latlon = (-73.961066,40.772774,-73.958652,40.774562) #77th st 
#     bbox_latlon = (-73.964950,40.767131,-73.963303,40.768549) #68th st
#     transformer = Transformer.from_crs("EPSG:4326", TARGET_CRS, always_xy=True)
#     min_lon, min_lat, max_lon, max_lat = bbox_latlon
#     bottom_left = transformer.transform(min_lon, min_lat)
#     top_right = transformer.transform(max_lon, max_lat)

#     # bottom_left = (586130, 4511430)
#     # top_right = (586165, 4511455)
#     area_of_interest = box(bottom_left[0], bottom_left[1], top_right[0], top_right[1])

#     print("Creating a subgraph for the target area...")
#     nodes_in_box = [n for n in G.nodes() if Point(n).within(area_of_interest)]
#     G_zoomed = G.subgraph(nodes_in_box)
#     print(f"Subgraph created with {G_zoomed.number_of_nodes()} nodes and {G_zoomed.number_of_edges()} edges.")

#     # Prepare Edges
#     edge_data = []
#     for u, v, data in G_zoomed.edges(data=True):
#         network_length = data.get('length', 0)
#         edist = np.linalg.norm(np.array(u) - np.array(v))
#         circuity = network_length / (edist + 1e-6)
#         edge_data.append({
#             'geometry': LineString([u,v]),
#             'type': data.get('type'),
#             'circuity': circuity
#         })
#     edges_gdf = gpd.GeoDataFrame(edge_data, crs=TARGET_CRS)
#     node_data = []
#     for node in G_zoomed.nodes():
#         node_data.append({
#             'geometry': Point(node),
#             'degree': G_zoomed.degree(node)
#         })
#     nodes_gdf = gpd.GeoDataFrame(node_data, crs=TARGET_CRS)

#     print("Plotting diagnostic map...")
#     fig, ax = plt.subplots(figsize=(50, 50))
#     ax.set_facecolor('white')
#     ax.set_axis_off()

#     edges_gdf.plot(ax=ax, column='circuity', cmap='coolwarm', linewidth=0.5, legend=True,
#                 legend_kwds={'label': "Edge Circuity (Network Length / Straight-Line Length)"}, alpha=0.5)
#     nodes_gdf.plot(ax=ax, column='degree', cmap='spring', markersize=8, legend=True,
#                 legend_kwds={'label': "Node Degree (Number of Connections)"}, alpha=0.4)

#     plt.savefig('debug_grand_central.png', bbox_inches='tight', pad_inches=0, facecolor='white', dpi=300)
#     print("Diagnostic plot saved to 'debug_grand_central.png'")


    # # PLOTTING
    # #     # START OTHER PLOTTING
    # # --- 2. Define a Wider Area of Interest for the 4 Train ---
    # bbox_latlon_4train = (-73.956356,40.630630,-73.886662,40.673673) 

    # # Project the bounding box to your script's CRS
    # transformer = Transformer.from_crs("EPSG:4326", TARGET_CRS, always_xy=True)
    # min_lon, min_lat, max_lon, max_lat = bbox_latlon_4train
    # bottom_left = transformer.transform(min_lon, min_lat)
    # top_right = transformer.transform(max_lon, max_lat)
    # area_of_interest_4train = box(bottom_left[0], bottom_left[1], top_right[0], top_right[1])

    # # --- 3. Filter for 4 Train Edges and Nodes within the Area ---
    # print("Filtering for 4 train network in the specified area...")
    # line_edges_in_area = []
    # line_nodes_in_area = set()

    # excluded = ['4', '5']
    # for u, v, data in G.edges(data=True):
    #     # Ensure it's a subway edge and is for the '4' train, and within the bbox
    #     if ((data.get('type') == 'subway' or data.get('type') == 'platform_access') and 
    #         not (data.get('route_id') in excluded)):
    #         line_geom = LineString([u, v])
    #         if line_geom.intersects(area_of_interest_4train):
    #             line_edges_in_area.append({'geometry': line_geom, 'usage': data.get('usage', 0), 'type': data.get('type')})
    #             line_nodes_in_area.add(u)
    #             line_nodes_in_area.add(v)

    # debug_lines_gdf = gpd.GeoDataFrame(line_edges_in_area, crs=TARGET_CRS)

    # # --- 4. Identify Non-Degree-2 Nodes on the 4 Train ---
    # special_nodes_data = []
    # for node_coords in line_nodes_in_area:
    #     degree = G.degree(node_coords)
    #     if degree != 0:
    #         special_nodes_data.append({
    #             'geometry': Point(node_coords),
    #             'degree': degree,
    #         })

    # special_nodes_gdf = gpd.GeoDataFrame(special_nodes_data, crs=TARGET_CRS)

    # # --- 5. Plot the Diagnostic Map ---
    # print("Plotting diagnostic map of 4 train special nodes...")
    # fig, ax = plt.subplots(figsize=(25, 25))
    # ax.set_facecolor('black')
    # ax.set_axis_off()

    # # Plot the 4 train line itself as a dim background
    # debug_lines_gdf.plot(ax=ax, color='grey', linewidth=0.8, alpha=0.6)
    # debug_lines_gdf[debug_lines_gdf['type'] == 'platform_access'].plot(ax=ax, color='red', linewidth=0.8, alpha=0.6)
    # print(len(debug_lines_gdf[debug_lines_gdf['type'] == 'platform_access']))

    # # Plot the special nodes, colored by their degree
    # if not special_nodes_gdf.empty:
    #     special_nodes_gdf.plot(ax=ax, markersize=10, alpha=0.4, 
    #         column = 'degree', cmap='viridis')

    # plt.savefig('debug_4train_special_nodes.png', bbox_inches='tight', pad_inches=0, facecolor='white', dpi=300)
    # print("Diagnostic plot saved to 'debug_4train_special_nodes.png'")
    # exit()
