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
from shapely import LineString
from shapely.geometry import Point, box


mta_colors = {
    'A': '#0039A6', 'C': '#0039A6', 'E': '#0039A6', 'SIR': '#0039A6',
    'B': '#FF6319', 'D': '#FF6319', 'F': '#FF6319', 'M': '#FF6319', 
    'G': '#6CBE45', 'J': '#996633', 'Z': '#996633', 'L': '#A7A9AC',
    'N': '#FCCC0A', 'Q': '#FCCC0A', 'R': '#FCCC0A', 'W': '#FCCC0A', 
    '1': '#EE352E', '2': '#EE352E', '3': '#EE352E', 
    '4': '#00933C', '5': '#00933C', '6': '#00933C', '7': '#B933AD', 
    'S': '#808183', 'SF': '#808183', 'ST': '#808183', 'SR': '#808183'}


SUBWAY_SPEED = 3 # used in euclidean dist

def jittered_weight(u, v, data):
    jitterSize = 0.03
    base_length = data['length']
    jitter = random.gauss(1, jitterSize)
    return base_length * jitter
def jitter_sqrt(u, v, data):
    strength = 0.5 # strength of 1 means 10 ft path will be jittered by 1 * sqrt(10) = 3
                # or 100 ft path will get jittered by 1 * sqrt (100) = 10
    base_length = data['length']
    if base_length <= 0:
        return 0.001
    sigma = strength * math.sqrt(base_length)
    noise = random.gauss(0, sigma)
    return max(0.001, base_length + noise)
def euclidean_dist(u, v):
    return np.linalg.norm(np.array(u) - np.array(v)) / SUBWAY_SPEED 


def findUsage(num_iterations = 10000000):
    print("Downloading and building graph from OpenStreetMap...")
    place_name = "New York, New York"
    TARGET_CRS = "EPSG:32618"

    bbox = (-74.253845,40.497615,-73.656464,40.981972)
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

    G_initial = nx.Graph(G_proj)
    print(f"Downloaded graph with {nx.number_connected_components(G_initial)} components.")

    # --- 2. Convert Graph Nodes to Coordinate Tuples ---
    print("Converting graph nodes from OSM IDs to coordinates...")
    pos = {node: (data['x'], data['y']) for node, data in G_initial.nodes(data=True)}
    G = nx.Graph() # This will be our final graph with coordinate nodes
    for u, v, data in G_initial.edges(data=True):
        u_coords = pos[u]
        v_coords = pos[v]
        G.add_edge(u_coords, v_coords, **data)
    street_nodes = list(G.nodes())

    # # --- 2.5. subway! ---
    print('loading subway data...')

    #lines_path = "data/nystreets/routes_nyc_subway_may2016.shp"
    lines_path = 'data/nystreets/geo_export_6c9929cb-3ce9-4cb5-862d-dc06efa13f97.shp'
    lines_gdf = gpd.read_file(lines_path)
    lines_gdf = lines_gdf.to_crs(TARGET_CRS).explode()
    lines_gdf['route_id'] = lines_gdf.service
    stations_df = pd.read_csv("data/nystreets/MTA_Subway_Stations.csv")
    stations_gdf = gpd.GeoDataFrame(
        stations_df,
        geometry=gpd.points_from_xy(stations_df['GTFS Longitude'], stations_df['GTFS Latitude']),
        crs="EPSG:4326").to_crs(TARGET_CRS)
    stations_gdf['routes'] = stations_gdf['Daytime Routes'].str.split(' ')
    stations_gdf.columns = stations_gdf.columns.str.lower().str.replace(' ', '_')

    # prepare street graph G
    for u, v, data in G.edges(data=True):
        G.edges[u, v]['type'] = 'street'

    subG = nx.Graph()

    all_subway_data = [] # We'll create a list of dictionaries
    for index, row in lines_gdf.iterrows():
        line = row.geometry
        route_symbol = row['route_id'] 
        segments = [LineString(pair) for pair in zip(line.coords, line.coords[1:])]
        for seg in segments:
            all_subway_data.append({'geometry': seg, 'route_id': route_symbol})

    # --- Part 2: Build the graph from our new, detailed list ---
    for seg_data in all_subway_data:
        segment = seg_data['geometry']
        route_symbol = seg_data['route_id'] 
        start_node = segment.coords[0][:2]
        end_node = segment.coords[-1][:2]
        if start_node != end_node:
            travel_time = segment.length / SUBWAY_SPEED
            subG.add_edge(start_node, end_node,
                        length=travel_time, 
                        type='subway',
                        route_id=route_symbol) 

    G = nx.compose(G, subG)

    # connecting stations!
    ENTER_STATION = 30 # meters from station to street (walking is ~ 1 m/s)
    TRAIN_WAIT = 300 # meters from station to getting on train (applied both ways)

    print("Connecting networks at station entrances...")
    station_id_to_node = {}
    for station in stations_gdf.itertuples():
        platform_coords = (station.geometry.x, station.geometry.y)
        G.add_node(platform_coords)
        station_id_to_node[station.station_id] = platform_coords

    subway_nodes = [n for n, d in G.nodes(data=True) if G.degree(n) > 0 and all(G.edges[n, neighbor].get('type') == 'subway' for neighbor in G.neighbors(n))]
    subway_tree = KDTree(subway_nodes)

    for station_id, platform_node in station_id_to_node.items():
        dist, idx = subway_tree.query(platform_node)
        nearest_track_node = subway_nodes[idx]
        G.add_edge(platform_node, nearest_track_node, length=TRAIN_WAIT, type='platform_access')

    street_tree = KDTree(street_nodes)
    for station_id, platform_node in station_id_to_node.items():
        dist, idx = street_tree.query(platform_node)
        nearest_street_node = street_nodes[idx]
        entrance_time = dist + ENTER_STATION
        G.add_edge(platform_node, nearest_street_node, length = dist + ENTER_STATION, type='station_entrance')
        
    print("\nMulti-modal graph creation complete!")

    # --- 2.6 load population! --
    print('loading population...')
    kontur_filepath = "data/nystreets/kontur_population_US_20231101.gpkg"
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
    print("Running pathfinding simulation...")
    all_nodes = list(G.nodes)
    all_edges = list(G.edges)
    final_weights = []
    for u, v, data in G.edges(data=True):
        G.edges[u, v]['usage'] = 0
        # G.edges[u, v]['length'] = G.edges[u, v]['length']


    shortDistance = 1000
    exponent = 2.3  
    proximity_sigma = 27000 # 1.6 km per mile

    # calculate probability of being picked of edges
    u_coords = np.array([u for u, v in all_edges])
    v_coords = np.array([v for u, v in all_edges])
    midpoints = (u_coords + v_coords)/2
    lengths = np.array(list(nx.get_edge_attributes(G, 'length').values()))
    # dist_center = np.linalg.norm(u_coords - center_point, axis=1)
    prox_weight = 1 #np.exp(-(dist_center**2) / (2 * proximity_sigma**2))

    pop_distances, pop_indices = pop_tree.query(midpoints, k=1)
    edge_pops = pop_values[pop_indices]
    # normalize by total edge length in hexagon
    temp_df = pd.DataFrame({'edge_length': lengths, 'population': edge_pops, 'hexagon_id': pop_indices})
    hexagon_total_lengths = temp_df.groupby('hexagon_id')['edge_length'].sum()
    temp_df['hexagon_total_length'] = temp_df['hexagon_id'].map(hexagon_total_lengths)
    normalized_length_weight = temp_df['edge_length'] / temp_df['hexagon_total_length']
    # square root because weight will be applied once at source and once at destination
    pop_weight = ((temp_df['population'] / (temp_df['population'].max() + 1)) + 0.01)**0.5


    final_weights = lengths * prox_weight * pop_weight

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
    print("Performing vectorized rejection sampling...")
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

    print("\nRunning pathfinding on accepted pairs...")
    successful_paths = 0
    failed_paths = 0
    # Now we just loop through the pairs we've already selected
    for i, (source_node, target_node) in enumerate(zip(final_source_nodes, final_target_nodes)):
        try:
            path = nx.astar_path(G, tuple(source_node), tuple(target_node), 
                                weight=jitter_sqrt, heuristic=euclidean_dist)

            network_dist = sum(G.edges[u, v]['length'] for u, v in zip(path[:-1], path[1:]))
            distance = np.linalg.norm(np.array(source_node) - np.array(target_node))
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
    print(f"\nPathfinding complete. {successful_paths} successful paths found, {failed_paths} failed.")

    # --- Step 4: Create Final GeoDataFrame ---
    print("\nCreating final GeoDataFrame with original street geometries...")

    # --- 4a: Start with the original street geometries from G_proj ---
    df_streets = ox.graph_to_gdfs(G_proj, nodes=False)

    # Map the calculated 'usage' from your main graph G back to this GeoDataFrame
    print("Mapping usage counts to street geometries...")
    usage_counts = []
    for u_osmid, v_osmid, data in G_proj.edges(data=True, keys=False):
        # Get the coordinate-based nodes used in your main graph G
        u_coords = (G_initial.nodes[u_osmid]['x'], G_initial.nodes[u_osmid]['y'])
        v_coords = (G_initial.nodes[v_osmid]['x'], G_initial.nodes[v_osmid]['y'])
        # Look up the edge in G to find its usage
        edge_data = G.get_edge_data(u_coords, v_coords)
        if edge_data:
            usage_counts.append(edge_data.get('usage', 0))
        else: # This might happen if the edge was removed from G, though unlikely here
            usage_counts.append(0)
    df_streets['usage'] = usage_counts
    df_streets['type'] = 'street'
    df_streets['route_id'] = None

    # --- 4b: Create geometries for all NON-street edges (subway, transfers, etc.) ---
    print("Creating geometries for subway and other network edges...")
    other_edges_data = []
    for u, v, data in G.edges(data=True):
        if data.get('type') != 'street':
            geom = LineString([u, v])
            other_edges_data.append({
                'geometry': geom,
                'usage': data.get('usage', 0),
                'type': data.get('type', 'unknown'),
                'route_id': data.get('route_id', None)
            })
    # Create a GeoDataFrame from the other edges, if any exist
    if other_edges_data:
        df_other = gpd.GeoDataFrame(other_edges_data, crs=TARGET_CRS)
        # --- 4c: Combine the two GeoDataFrames ---
        print("Combining street and subway GeoDataFrames...")
        df = pd.concat([df_streets, df_other], ignore_index=True)
    else:
        df = df_streets

    # Select only the columns you need for plotting to keep things clean
    df = df[['geometry', 'usage', 'type', 'route_id']]

    print("Final combined GeoDataFrame created successfully.")
    
    # save!
    df.to_file('usage_map_ds.gpkg', driver='GPKG')
    print('File saved.')


findUsage(2000000)

print('loading file...')
df = gpd.read_file('usage_map_ds.gpkg')
#df = gpd.read_file('usage_map_s_big.gpkg')

print("Plotting final map...")
imageSize = 80
fig, ax = plt.subplots(figsize=(imageSize, imageSize))
ax.set_facecolor('black')
ax.set_axis_off()
maxUsage = df.usage.max()

cmap = 'magma'

width_exp = 0.7
streets_df = df[(df['type'] != 'subway') & (df['usage'] > 0)].sort_values('usage', ascending=True)
streets_df.plot(
    ax=ax,
    column='usage',
    cmap=cmap, 
    linewidth = 0.3 * imageSize * streets_df['usage']**width_exp / maxUsage**width_exp, 
    norm=LogNorm(vmin=0.8, vmax=streets_df['usage'].max()*0.8),
    capstyle='round'    
)

subway_df = df[df['type'] == 'subway'].copy()
subway_df['routeColor'] = subway_df['route_id'].map(mta_colors).fillna('#FFFFFF')
subway_df = subway_df[subway_df['usage'] > 0].sort_values('usage', ascending=True)
# if not subway_df.empty:
#     subway_df.plot(
#         ax=ax,
#         color = subway_df['routeColor'],
#         linewidth = 0.3 * imageSize * subway_df['usage']**width_exp / maxUsage**width_exp, 
#         alpha=0.9,
#         #capstyle='round', 
#     )
subway_df.plot(
    ax=ax,
    column='usage',
    cmap=cmap, 
    linewidth = 0.3 * imageSize * subway_df['usage']**width_exp / maxUsage**width_exp, 
    norm=LogNorm(vmin=0.8, vmax=subway_df['usage'].max()*0.8),
    capstyle='round'    
)



plt.tight_layout()
plt.savefig('nystreets/osmsub' + '.png', bbox_inches='tight', pad_inches=0,
facecolor='black')
print('plot saved!')



# lines_path = "data/nystreets/routes_nyc_subway_may2016.shp"
# lines_gdf = gpd.read_file(lines_path)
# print(lines_gdf.route_id.unique())
