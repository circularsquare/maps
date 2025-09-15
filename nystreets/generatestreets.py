
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.collections import LineCollection
import geopandas
import geopandas as gpd
from cartopy import crs as ccrs 
import cartopy.feature as cfeature
import cartopy
import os
from geodatasets import get_path
import cartopy.io.shapereader as shpreader
import random
from shapely.geometry import LineString, MultiLineString
import pickle
import momepy
import networkx as nx
from shapely.geometry import MultiPoint
from shapely.ops import split
import math
import osmnx

def subdivide_line(line, max_length):
    if line.length <= max_length:
        return [line]
    segments = []
    num_segments = math.ceil(line.length / max_length)
    points = [line.interpolate(i / num_segments, normalized=True) for i in range(num_segments + 1)]
    for p1, p2 in zip(points[:-1], points[1:]):
        segments.append(LineString([p1, p2]))
    return segments

cache_dir = os.path.join(os.getcwd(), 'cartopycache')
cartopy.config['pre_existing_data_dir'] = cache_dir

#crs = ccrs.AlbersEqualArea(-59, 0, 0, 0, (0, -40))
crs = ccrs.Mercator()
fig, ax = plt.subplots(subplot_kw={"projection": crs}, figsize=(40,40))
MAX_SEGMENT_LENGTH = 100
TARGET_CRS = "EPSG:2263"
PRECISION = 1.0 # feet

path = 'data/nystreets/DCM_StreetCenterLine.shp'
df = geopandas.read_file(path)
if df.crs.is_geographic:
    df = df.to_crs("EPSG:2263")



place_name = "New York, New York"


# --- Step 1: Initial Data Preparation ---
print("\n--- Step 1: Loading and Cleaning Initial Data ---")
df = gpd.read_file(path)
if df.crs != TARGET_CRS:
    print(f"Projecting data to CRS {TARGET_CRS}...")
    df = df.to_crs(TARGET_CRS)
df = df.explode(index_parts=False)
df = df[df.geom_type == 'LineString'].copy()
print(f"Initial cleanup complete. Found {len(df)} LineString segments.")

# --- Phase 2: Topology Cleaning (THE NEW STEP) ---
# print("\n--- Phase 1.5: Closing Gaps in Network Topology ---")
# print(f"Searching for and closing gaps within a {5}-foot tolerance...")
# df = momepy.extend_lines(df, tolerance=5)

# --- Step 2: Geometric Approximation (Segmentation) ---
print(f"\n--- Step 2: Subdividing segments by vertices ---")
all_segments = []
# Iterate over each complex line in the original GeoDataFrame
for index, row in df.iterrows():
    line = row.geometry
    segments = [LineString(pair) for pair in zip(line.coords, line.coords[1:])]
    all_segments.extend(segments)
df = gpd.GeoDataFrame(geometry=all_segments, crs=df.crs)

print(f"Number of segments after splitting at vertices: {len(df)}")

# --- Step 3: Topological Cleaning (Finding Intersections) ---
print("\n--- Step 3: Finding Intersections with momepy ---")
print("Converting to graph to find and create nodes at all intersections...")
# This process is fast but simplifies geometry, which is why Step 2 was necessary.
preliminary_graph = momepy.gdf_to_nx(df, approach='primal')
nodes_gdf, df = momepy.nx_to_gdf(preliminary_graph)
print("Intersection analysis complete.")

# --- Step 4: (Optional but Recommended) Simplification ---
print("\n--- Step 4: Simplifying the Network ---")
print("Removing false nodes (degree-2 nodes) to simplify geometry...")
#df = momepy.remove_false_nodes(df)
print("Simplification complete.")

# --- Final Cleanup and Graph Creation ---
print("\n--- Step 5: Building Final Graph & Snapping Nodes ---")
df = df.explode(index_parts=False)
df = df[df.geom_type == 'LineString'].copy()

# Create the final graph object
G = nx.Graph()

print("Looping through final geometries to build graph...")
for index, row in df.iterrows():
    line = row.geometry
    start_coords = line.coords[0][:2]
    end_coords = line.coords[-1][:2]

    # Snap the coordinates to the grid defined by PRECISION
    start_node = (round(start_coords[0] / PRECISION) * PRECISION, 
                  round(start_coords[1] / PRECISION) * PRECISION)
    end_node = (round(end_coords[0] / PRECISION) * PRECISION, 
                round(end_coords[1] / PRECISION) * PRECISION)
    if start_node != end_node:
        G.add_edge(start_node, end_node, weight=line.length)


# graphpath = 'nystreets/graph3.gpickle'
# with open(graphpath, 'wb') as f:
#     pickle.dump(G, f)

# graphpath = 'nystreets/graph3.gpickle'
# with open(graphpath, 'rb') as f:
#     G = pickle.load(f)

print(f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
print("number of components " + str(nx.number_connected_components(G)))
avg_degree = (2 * G.number_of_edges()) / G.number_of_nodes()
print(f"Average node degree: {avg_degree:.4f}") 
sorted_components = sorted(nx.connected_components(G), key=len, reverse=True)
for i, component in enumerate(sorted_components):
    component_size = len(component)
    print(f"Component {i+1}: {component_size} nodes")
    if i==4:
        break


# Get the list of all nodes to pick from
all_nodes = list(G.nodes)

# Add a 'usage' attribute to each edge, initializing to 0
for u, v in G.edges():
    G.edges[u, v]['usage'] = 0

num_iterations = 50
failed_iterations = 0
failed_destinations = []
for i in range(num_iterations):
    if i % 100 == 0:
        print(f"Running iteration {i}/{num_iterations}...")
    # Randomly pick two distinct nodes
    source_node = random.choice(all_nodes)
    target_node = random.choice(all_nodes)
    if source_node == target_node:
        continue # Skip if the same node is picked
    try: # Find the shortest path using Dijkstra's algorithm, weighted by length
        path = nx.shortest_path(G, source=source_node, target=target_node, weight='weight')
        # Increment the usage count for each edge in the path
        for j in range(len(path) - 1):
            u = path[j]
            v = path[j+1]
            G.edges[u, v]['usage'] += 1
    except nx.NetworkXNoPath:
        failed_iterations += 1
        failed_destinations.append(source_node)
        failed_destinations.append(target_node)
        continue
print('paths found: ' + str(num_iterations-failed_iterations) + '/' + str(num_iterations))

usage_counts = []
unfoundCounter = 0
foundCounter = 0
# Map the usage counts from the graph back to the original GeoDataFrame
for index, row in df.iterrows():
    start_coords = row.geometry.coords[0][:2]
    end_coords = row.geometry.coords[-1][:2]
    start_node = (round(start_coords[0] / PRECISION) * PRECISION, 
                  round(start_coords[1] / PRECISION) * PRECISION)
    end_node = (round(end_coords[0] / PRECISION) * PRECISION, 
                round(end_coords[1] / PRECISION) * PRECISION)
    # Check if the edge exists in the graph (it should)
    edge_data = G.get_edge_data(start_node, end_node)
    if edge_data is not None:
        usage = edge_data.get('usage', 0)
        usage_counts.append(usage)
    else:
        unfoundCounter += 1
        usage_counts.append(0)
print(str(unfoundCounter) + ' not found and ' + str(foundCounter) + ' found') 

# Add the usage counts as a new column in the GeoDataFrame
print('plotting...')
df['usage'] = usage_counts
from matplotlib.colors import LogNorm

df.plot(ax=ax, 
    column='usage', cmap='viridis', 
    linewidth=2, norm=LogNorm(vmin=1, vmax=max(2, df['usage'].max())), )

#print(failed_destinations)    
fx, fy = zip(*failed_destinations)
ax.scatter(fx, fy, color='red', s=5, alpha=0.7)
for component in sorted_components:
    gx, gy = zip(*component)
    ax.scatter(gx, gy, s=1, alpha=0.3)

edge_lines_2d = []
for node1, node2 in G.edges():
    p1 = (node1[0], node1[1])  # (x1, y1)
    p2 = (node2[0], node2[1])  # (x2, y2)
    edge_lines_2d.append([p1, p2])

lc = LineCollection(edge_lines_2d, colors='gray', linewidths=0.5, alpha=1.0)
ax.add_collection(lc)

ax.set_facecolor('black')
ax.set_axis_off()
plt.tight_layout()
plt.savefig('nystreets/paths.png')
# plt.show()


def plotShapefile(path):
    df = geopandas.read_file(path)
    df = df.to_crs(crs)
    df.plot(ax=ax, edgecolor = '#63538d', facecolor='none', zorder = 10, 
        rasterized=False, antialiased = True, linewidth = 0.5)
# plotShapefile(path)
# plt.tight_layout()
# plt.savefig('nystreets/streets.png')






'''
streets_exploded_gdf = df.explode(index_parts=False) 
all_segments = []
for index, row in streets_exploded_gdf.iterrows():
    line = row.geometry
    segments = [LineString(pair) for pair in zip(line.coords, line.coords[1:])]
    all_segments.extend(segments)
segmented_gdf = geopandas.GeoDataFrame(geometry=all_segments, crs=crs)
df = segmented_gdf.explode(index_parts=False)


precision = 5 # snap nodes onto this many meter grid
G = nx.Graph()
for index, row in df.iterrows():
    line = row.geometry
    start_node = (round(line.coords[0][0]/precision) * precision,
                    round(line.coords[0][1]/precision) * precision)
    end_node = (round(line.coords[-1][0]/precision) * precision,
                round(line.coords[-1][1]/precision) * precision)
    if start_node==end_node: 
        continue
    G.add_edge(line.coords[0], line.coords[-1], weight=line.length, id=index)

print(f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
print("number of components " + str(nx.number_connected_components(G)))
avg_degree = (2 * G.number_of_edges()) / G.number_of_nodes()
print(f"Average node degree: {avg_degree:.4f}") 

#preliminary_graph = momepy.gdf_to_nx(segmented_gdf, approach='primal')
nodes_gdf, edges_gdf = momepy.nx_to_gdf(G)
df = momepy.remove_false_nodes(edges_gdf)
G = momepy.gdf_to_nx(df, approach='primal')

'''




'''
# --- Step 1: Initial Data Preparation ---
print("\n--- Step 1: Loading and Cleaning Initial Data ---")
df = gpd.read_file(path)
if df.crs != TARGET_CRS:
    print(f"Projecting data to CRS {TARGET_CRS}...")
    df = df.to_crs(TARGET_CRS)
df = df.explode(index_parts=False)
df = df[df.geom_type == 'LineString'].copy()
print(f"Initial cleanup complete. Found {len(df)} LineString segments.")


# --- Step 2: Build and Clean Network with spaghetti ---
print("\n--- Step 2: Building and Cleaning Network with spaghetti ---")

# 2a. Instantiate the Network object from your GeoDataFrame
print("  Instantiating spaghetti network...")
ntw = spaghetti.Network(in_data=df)
 
# 2b. Snap intersections within the given tolerance.
print(f"  Snapping intersections with a tolerance of {PRECISION} feet...")
ntw.snap_intersections(tolerance=PRECISION)

# 2c. **THE KEY STEP:** Split lines at all true intersections.
print("  Splitting edges at all true intersections...")
ntw.split_edges_at_intersections()

# 2d. Extract the cleaned network back into a GeoDataFrame
print("  Extracting cleaned network into a new GeoDataFrame...")
noded_gdf = ntw.extract_network()
print(f"Spaghetti processing complete. Final noded dataset has {len(noded_gdf)} segments.")

# --- Step 3: Final Graph Building ---
# The input 'noded_gdf' is now both geometrically and topologically correct.
print("\n--- Step 3: Building Final NetworkX Graph ---")
G = nx.Graph()

for index, row in noded_gdf.iterrows():
    line = row.geometry
    if not line.is_empty and line.geom_type == 'LineString':
        # Get the exact start and end coordinates from the cleaned geometry.
        # No rounding/snapping is needed here because spaghetti already handled it.
        start_node = line.coords[0][:2]
        end_node = line.coords[-1][:2]
        
        if start_node != end_node:
            G.add_edge(
                start_node, 
                end_node, 
                weight=line.length,
                # Optional: store the geometry itself for easy plotting of paths later
                geometry=line 
            )
df = noded_gdf
'''