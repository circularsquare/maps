"""Convert MapSPAM production CSV to compact binary + GeoJSON for web use."""
import pandas as pd
import struct
import json
import sys
import os

crop = sys.argv[1].upper() if len(sys.argv) > 1 else "BARL"
col = f"{crop}_A"

df = pd.read_csv("data/Global_CSV/spam2020V2r0_global_production/spam2020V2r0_global_P_TA.csv",
                  usecols=["x", "y", col])

mask = df[col] > 0
df = df[mask].reset_index(drop=True)
print(f"{crop}: {len(df)} pixels with production")

# Write compact binary: [float32 lon, float32 lat, float32 value] per point
out_bin = f"data/{crop.lower()}_production.bin"
with open(out_bin, "wb") as f:
    for _, row in df.iterrows():
        f.write(struct.pack("<fff", row["x"], row["y"], row[col]))

size_mb = os.path.getsize(out_bin) / 1024 / 1024
print(f"Saved {out_bin} ({size_mb:.1f} MB)")

# Also write GeoJSON for tippecanoe later
out_json = f"data/{crop.lower()}_production.geojson"
features = []
for _, row in df.iterrows():
    features.append({
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [round(row["x"], 4), round(row["y"], 4)]},
        "properties": {"v": round(row[col], 1)}
    })
geojson = {"type": "FeatureCollection", "features": features}
with open(out_json, "w") as f:
    json.dump(geojson, f)
size_mb = os.path.getsize(out_json) / 1024 / 1024
print(f"Saved {out_json} ({size_mb:.1f} MB)")
