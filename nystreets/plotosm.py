import osmnx as ox
import matplotlib.pyplot as plt
import os

# --- Configuration ---
place_name = "New York, New York"
image_dpi = 1000 # Dots per inch for high-quality saved images


print(f"Downloading 'drive' network for {place_name}...")
G_drive = ox.graph_from_place(place_name, network_type='drive')

print("Plotting 'drive' network...")
fig, ax = ox.plot_graph(
    G_drive,
    show=False,
    close=False, # IMPORTANT: Set to False so we can save it ourselves
    bgcolor='white',
    edge_color='black',
    node_size=0,
    edge_linewidth=0.3
)
# Explicitly save the figure using matplotlib's savefig
path = 'nystreets/nyc_drive_network.png'
fig.savefig(
    path,
    dpi=image_dpi,
    bbox_inches='tight',
    pad_inches=0,
    facecolor='black'
)
plt.close(fig) # Close the figure to free up memory
print(f"Saved '{path}'")




# ----------------------------------------------------------------------
# --- 2. Plot the Walking Network ---
# ----------------------------------------------------------------------
print(f"\nDownloading 'walk' network for {place_name}...")
G_walk = ox.graph_from_place(place_name, network_type='walk')

print("Plotting 'walk' network...")
fig, ax = ox.plot_graph(
    G_walk,
    show=False,
    close=False, # IMPORTANT: Set to False
    bgcolor='white',
    edge_color='cyan',
    node_size=0,
    edge_linewidth=0.3
)

# Explicitly save the figure
path = 'nystreets/nyc_walk_network.png'
fig.savefig(
    path,
    dpi=image_dpi,
    bbox_inches='tight',
    pad_inches=0,
    facecolor='black'
)
plt.close(fig)
print(f"Saved '{path}'")
print("\nDone!")