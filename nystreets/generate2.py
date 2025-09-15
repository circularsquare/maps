import geopandas as gpd
import networkx as nx
import osmnx as ox
import random
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pickle
import numpy as np

# todo: weight on road length, not node coutn
# todo: add jersey?

def jittered_weight(u, v, data):
    #'data' is a dictionary of the edge's attributes.
    jitterSize = 0.03
    base_length = data['length']
    jitter = random.gauss(1, jitterSize)
    return base_length * jitter




# --- 1. Get Network Data from OpenStreetMap ---
print("Downloading and building graph from OpenStreetMap...")
place_name = "New York, New York"
TARGET_CRS = "EPSG:2263"

# Download the drivable street network as a directed graph
G_directed = ox.graph_from_place(place_name, network_type='drive')
# Project the graph to our target CRS (units in feet)
G_proj = ox.project_graph(G_directed, to_crs=TARGET_CRS)
# Convert to a simple undirected graph for our simulation
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
print("Graph conversion complete.")


# --- 3. Run Pathfinding Simulation ---
print("\nRunning pathfinding simulation...")
all_nodes = list(G.nodes)
for u, v in G.edges():
    G.edges[u, v]['usage'] = 0
shortDistance = 0.25 * 5280
exponent = 2

i = 0
num_iterations = 20000
failed_iterations = 0

while i < num_iterations:    
    source_node, target_node = random.sample(all_nodes, 2)
    dist = np.linalg.norm(np.array(source_node) - np.array(target_node))
    if dist > shortDistance:
        prob = (shortDistance / dist) ** exponent
        if random.random() > prob:
            continue
    
    try:
        path = nx.shortest_path(G, source=source_node, target=target_node, 
        weight=jittered_weight)
        for u, v in zip(path[:-1], path[1:]):
            G.edges[u, v]['usage'] += 1
        i += 1
        if i > 0 and i % 500 == 0:
            print(f"  ...ran {i}/{num_iterations} iterations.")
    except nx.NetworkXNoPath:
        failed_iterations += 1

print(f"Simulation complete. Paths found: {num_iterations - failed_iterations}/{num_iterations}")

# --- 4. Prepare the Final GeoDataFrame (This is your "df") ---
print("\nPreparing final GeoDataFrame for plotting...")
# We use an osmnx helper to get the geometries from the projected graph.
# This is where your new 'df' comes from.
df = ox.graph_to_gdfs(G_proj, nodes=False)


# --- 5. Map Usage Data Back to the GeoDataFrame ---
print("Mapping usage counts back to GeoDataFrame...")
usage_counts = []
for index, row in df.iterrows():
    start_node = row.geometry.coords[0][:2]
    end_node = row.geometry.coords[-1][:2]
    
    edge_data = G.get_edge_data(start_node, end_node)
    
    if edge_data:
        usage_counts.append(edge_data.get('usage', 0))
    else:
        usage_counts.append(0)

df['usage'] = usage_counts


# --- 6. Plot the Final Result ---
print("Plotting final map...")
fig, ax = plt.subplots(figsize=(40, 40))
ax.set_facecolor('black')
ax.set_axis_off()

# Plot only edges that were actually used
used_edges_df = df[df['usage'] > 0]

used_edges_df.plot(
    ax=ax,
    column='usage',
    cmap='inferno',
    linewidth = 5 * used_edges_df['usage']**0.6 / used_edges_df['usage'].max()**0.6, 
    norm=LogNorm(vmin=0.5, vmax=used_edges_df['usage'].max())
)

plt.tight_layout()
plt.savefig('nystreets/paths_osm.png', bbox_inches='tight', pad_inches=0,
facecolor='black')
print("\nPlot saved to nystreets/paths_osm.png")


df.to_file('usage_map.gpkg', driver='GPKG')