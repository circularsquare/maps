"""Combine all per-crop dot binaries into one file: (float32 lon, lat, crop_id) per dot."""
import struct
import os
import json

CROPS = [
    "bana","barl","bean","cass","chic","citr","cnut","coco","coff",
    "cott","cowp","grou","lent","maiz","mill","ocer","ofib","oilp",
    "onio","ooil","opul","orts","pige","plnt","pmil","pota","rape",
    "rcof","rest","rice","rubb","sesa","sorg","soyb","sugb","sugc",
    "sunf","swpo","teas","temf","toba","toma","trof","vege","whea","yams"
]

NAMES = {
    "bana":"Banana","barl":"Barley","bean":"Bean","cass":"Cassava","chic":"Chickpea",
    "citr":"Citrus","cnut":"Coconut","coco":"Cocoa","coff":"Arabica Coffee",
    "cott":"Cotton","cowp":"Cowpea","grou":"Groundnut","lent":"Lentil","maiz":"Maize",
    "mill":"Small Millet","ocer":"Other Cereals","ofib":"Other Fibre","oilp":"Oil Palm",
    "onio":"Onion","ooil":"Other Oil Crops","opul":"Other Pulses","orts":"Other Roots",
    "pige":"Pigeon Pea","plnt":"Plantain","pmil":"Pearl Millet","pota":"Potato",
    "rape":"Rapeseed","rcof":"Robusta Coffee","rest":"Rest of Crops","rice":"Rice",
    "rubb":"Rubber","sesa":"Sesame","sorg":"Sorghum","soyb":"Soybean","sugb":"Sugarbeet",
    "sugc":"Sugarcane","sunf":"Sunflower","swpo":"Sweet Potato","teas":"Tea",
    "temf":"Temperate Fruit","toba":"Tobacco","toma":"Tomato","trof":"Tropical Fruit",
    "vege":"Other Vegetables","whea":"Wheat","yams":"Yams"
}

# Assign colors: evenly spaced hues
import colorsys
legend = []
for i, code in enumerate(CROPS):
    hue = i / len(CROPS)
    r, g, b = colorsys.hls_to_rgb(hue, 0.45, 0.7)
    color = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

    bin_path = f"data/{code}_dots.bin"
    n_dots = os.path.getsize(bin_path) // 8
    legend.append({
        "code": code,
        "name": NAMES[code],
        "color": color,
        "dots": n_dots,
    })

# Write legend
with open("data/legend.json", "w") as f:
    json.dump(legend, f, indent=2)
print(f"Wrote data/legend.json ({len(legend)} crops)")

# Write combined binary
total = 0
with open("data/all_dots.bin", "wb") as out:
    for i, code in enumerate(CROPS):
        bin_path = f"data/{code}_dots.bin"
        with open(bin_path, "rb") as inp:
            data = inp.read()
        n = len(data) // 8  # 2 floats per dot
        for j in range(n):
            lon, lat = struct.unpack_from("<ff", data, j * 8)
            out.write(struct.pack("<ffB", lon, lat, i))
        total += n
        print(f"  {code}: {n} dots")

size_mb = os.path.getsize("data/all_dots.bin") / 1024 / 1024
print(f"Wrote data/all_dots.bin: {total} dots, {size_mb:.1f} MB")
