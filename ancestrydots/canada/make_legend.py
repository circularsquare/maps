"""
Build legend.json for the Canada map from ancestry_colors_ca.csv.

Shape matches what index.html's buildLegend() expects:
  [{label, group, subgroup, color, population}, ...]
Canada has no subgroups, so subgroup is "". Colors are the HLS values from the CSV
converted to hex. Populations are the national counts (from make_colors.py).

Usage:
    python make_legend.py
"""

from __future__ import annotations

import colorsys
import csv
import json
from pathlib import Path

HERE = Path(__file__).parent
COLORS = HERE / "ancestry_colors_ca.csv"
OUT = HERE / "data" / "processed" / "legend.json"

# group render order for the legend (mirrors the JS GROUP_ORDER we set for Canada)
GROUP_ORDER = ["western", "eastern", "american", "native", "african", "afro_carib",
               "latino", "mena", "s_c_asian", "se_asian", "east_asian", "pacific", "other"]


def hex_from_hls(hue, sat, lit):
    r, g, b = colorsys.hls_to_rgb(hue / 360, lit, sat)
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


def main():
    items = []
    with open(COLORS, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [n.strip() for n in reader.fieldnames]
        for row in reader:
            items.append({
                "label": row["label"].strip(),
                "group": row["group"].strip(),
                "subgroup": "",
                "color": hex_from_hls(float(row["hue"]), float(row["sat"]), float(row["lit"])),
                "population": int(row["population"]),
            })
    rank = {g: i for i, g in enumerate(GROUP_ORDER)}
    items.sort(key=lambda x: (rank.get(x["group"], 99), -x["population"]))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False)
    print(f"wrote {len(items)} legend entries -> {OUT}")


if __name__ == "__main__":
    main()
