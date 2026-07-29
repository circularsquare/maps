"""
Merge the US and Canada legends into one legend.json for the combined North America map.

- Colours + groups: the Canada (this-session) palette WINS for any label present in
  both, since we deliberately tuned it here; US-only labels keep their US colour.
- Population: summed across both countries for shared labels (the legend shows counts).

Inputs:
  ../data/processed/legend.json            (US)
  ../canada/data/processed/legend.json     (Canada, session colours)
Output:
  data/processed/legend.json               (combined)
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent
US = HERE.parent / "data" / "processed" / "legend.json"
CA = HERE.parent / "canada" / "data" / "processed" / "legend.json"
OUT = HERE / "data" / "processed" / "legend.json"

GROUP_ORDER = ["western", "eastern", "american", "native", "african", "afro_carib",
               "latino", "mena", "s_c_asian", "se_asian", "east_asian", "pacific",
               "other", "no_ancestry"]


def main():
    us = {e["label"]: e for e in json.load(open(US, encoding="utf-8"))}
    ca = {e["label"]: e for e in json.load(open(CA, encoding="utf-8"))}

    merged = {}
    for label in set(us) | set(ca):
        u, c = us.get(label), ca.get(label)
        base = c or u                      # Canada wins for colour/group
        pop = (u["population"] if u else 0) + (c["population"] if c else 0)
        merged[label] = {
            "label": label,
            "group": base["group"],
            "subgroup": base.get("subgroup", ""),
            "color": base["color"],
            "population": pop,
        }

    rank = {g: i for i, g in enumerate(GROUP_ORDER)}
    items = sorted(merged.values(), key=lambda x: (rank.get(x["group"], 99), -x["population"]))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(items, open(OUT, "w", encoding="utf-8"), ensure_ascii=False)

    shared = set(us) & set(ca)
    print(f"US={len(us)} CA={len(ca)} -> merged={len(items)} ({len(shared)} shared, "
          f"colour taken from Canada for those)")


if __name__ == "__main__":
    main()
