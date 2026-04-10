import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import sys

crop = sys.argv[1].upper() if len(sys.argv) > 1 else "BARL"

df = pd.read_csv("data/Global_CSV/spam2020V2r0_global_production/spam2020V2r0_global_P_TA.csv")

col = f"{crop}_A"
if col not in df.columns:
    print(f"Column {col} not found. Available crops:")
    print([c.replace("_A", "") for c in df.columns if c.endswith("_A")])
    sys.exit(1)

# filter to pixels with nonzero production
mask = df[col] > 0
x = df.loc[mask, "x"].values
y = df.loc[mask, "y"].values
vals = df.loc[mask, col].values

print(f"{crop}: {mask.sum()} pixels with production, total {vals.sum():.0f} mt")

fig = plt.figure(figsize=(16, 8))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
ax.set_global()
ax.add_feature(cfeature.LAND, facecolor="#f0f0f0", edgecolor="none")
ax.add_feature(cfeature.OCEAN, facecolor="white")
ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor="#cccccc")
ax.add_feature(cfeature.COASTLINE, linewidth=0.3, edgecolor="#999999")

norm = mcolors.LogNorm(vmin=max(vals.min(), 1), vmax=vals.max())

sc = ax.scatter(
    x, y,
    c=vals,
    s=0.05,
    marker=",",
    norm=norm,
    cmap="YlGn",
    transform=ccrs.PlateCarree(),
    rasterized=True,
)

cbar = plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.02)
cbar.set_label("Production (metric tons)")

crop_names = {
    "BARL": "Barley", "WHEA": "Wheat", "RICE": "Rice", "MAIZ": "Maize",
    "SOYB": "Soybean", "COFF": "Coffee", "COCO": "Cocoa", "COTT": "Cotton",
    "SUGC": "Sugarcane", "POTA": "Potato", "TEAS": "Tea", "RUBB": "Rubber",
    "SORG": "Sorghum", "SUNF": "Sunflower", "RAPE": "Rapeseed",
}
name = crop_names.get(crop, crop)

ax.set_title(f"Global {name} Production (SPAM 2020)", fontsize=14, fontweight="bold")

plt.tight_layout()
out = f"plots/{crop.lower()}_production.png"
plt.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved to {out}")
