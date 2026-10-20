"""T-776: the world market's rows, kept beside the registry.

`app/src/game/registry.json` is the index the app bundles (what is needed before a zone opens);
each zone's market row lives in `app/src/game/market-rows.json`, keyed by zone id in the registry's
order, which the demand workers fetch once a market pack opens (docs/architecture.md). A row is
written by `market_zones.py` (from the sim's `market_pack`); its `country` is copied into the zone's
index entry, which the new game sheet and the joint solve read.
"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def path(root=ROOT):
    return Path(root) / "app/src/game/market-rows.json"


def load(root=ROOT):
    """{zone id: row}; empty if the file is not there yet."""
    p = path(root)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def save(reg, rows, root=ROOT):
    """Write the rows of the zones `reg` registers, in its order, one row a line (a rebuild's diff is
    per zone). Rows of zones it lacks are dropped. Returns the rows written."""
    keep = {c["id"]: rows[c["id"]] for c in reg["cities"] if c["id"] in rows}
    lines = [f"{json.dumps(k)}:{json.dumps(v, ensure_ascii=False, separators=(',', ':'))}" for k, v in keep.items()]
    text = "{\n" + ",\n".join(lines) + "\n}\n"
    p = path(root)
    part = p.with_suffix(".json.part")
    part.write_text(text, encoding="utf-8")
    os.replace(part, p)
    return keep


def with_countries(reg, rows):
    """Copy each row's country into its zone's index entry (`country`)."""
    for c in reg["cities"]:
        cc = (rows.get(c["id"]) or {}).get("country")
        if cc:
            c["country"] = cc
    return reg
