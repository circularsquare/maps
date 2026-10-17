"""T-537, pipeline stage 5: the build_city config of a planned city, generated from its world plan
row (pipeline/world_plan.json). Every field comes from the row or a rule (docs/adding-a-city.md 3
"A planned city"), and the row's country module (its `module`) puts its `plan_hooks(row)` keys over
them (T-555); build_city.CITIES takes every planned row whose id it does not already have.

    python pipeline/plan_config.py <id> [plan.json]     print the generated config
"""
import json
import os

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
PLAN = os.path.join(HERE, "world_plan.json")
WB_VULN = "https://api.worldbank.org/v2/country/{iso2}/indicator/SL.EMP.VULN.ZS?format=json&date=2010:2025&per_page=100"
# The jobs proxy's population share follows the share of work that is informal, between tier0's
# JOB_BLEND (pop 0.2) and the blend set by hand for Mumbai and Lagos (pop 0.5, T-292, T-459).
INFORMAL_BLEND = {"nres": 0.2, "osm": 0.3, "pop": 0.5}


def load(path=PLAN):
    """(rows, unit sources) of a plan file."""
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    return doc["metros"], doc["unit_sources"]


def vulnerable_share(iso2):
    """ILO's modelled vulnerable employment (own-account and family workers) as a share of all
    employment, World Bank SL.EMP.VULN.ZS, the latest year; cached per country."""
    import build_city as b
    import tier0
    path = tier0.raw(f"worldbank_{iso2}_SL.EMP.VULN.ZS.json")
    if not os.path.exists(path):
        for attempt in range(4):
            try:
                r = requests.get(WB_VULN.format(iso2=iso2), headers=b.UA, timeout=60)
                r.raise_for_status()
                break
            except requests.RequestException:
                if attempt == 3:
                    raise
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(r.json(), f)
    with open(path) as f:
        rows = [x for x in (json.load(f)[1] or []) if x["value"] is not None]
    return (max(rows, key=lambda x: x["date"])["value"] / 100.0) if rows else None


def job_blend(iso2):
    """The proxy's blend: population share = vulnerable employment share, held to 0.2-0.5, the
    other two moving with it between JOB_BLEND and INFORMAL_BLEND; JOB_BLEND without a figure."""
    import tier0
    v = vulnerable_share(iso2)
    lo, hi = tier0.JOB_BLEND, INFORMAL_BLEND
    t = 0.0 if v is None else (min(max(v, lo["pop"]), hi["pop"]) - lo["pop"]) / (hi["pop"] - lo["pop"])
    return {k: round(lo[k] + t * (hi[k] - lo[k]), 4) for k in lo}


def water_cell(bbox):
    """25 m, or 50 m where the extent's mask would pass 150M pixels at 25 m (japan_zones' rule)."""
    import math
    w, s, e, n = bbox
    k = math.cos(math.radians((s + n) / 2))
    px = ((e - w) * 111.32 * k + 4) * ((n - s) * 111.32 + 4) / 0.025 ** 2
    return 50.0 if px > 1.5e8 else 25.0


def country_hooks(row):
    """The keys the row's country module supplies (its `plan_hooks`, T-555); none without a module."""
    if not row.get("module"):
        return {}
    import importlib
    mod = importlib.import_module(row["module"])
    if not hasattr(mod, "plan_hooks"):
        raise SystemExit(f"{row['id']}: pipeline/{row['module']}.py has no plan_hooks(row)")
    return mod.plan_hooks(row)


def with_country_keys(cfg):
    """cfg with its row's country keys put over the defaults, once: a planned config carries its row
    ("plan_row") until then, since the country modules import build_city and so cannot be imported
    while build_city makes CITIES (build_city.Cities merges on lookup)."""
    row = cfg.pop("plan_row", None)
    if row is not None:
        hooks = country_hooks(row)
        if "boundary_file" in hooks:
            cfg.pop("latin_names")  # it reads osm_units' unit names, which a country's own outline does not write
        cfg.update(hooks)
    return cfg


def config(row, sources):
    """The CITIES entry for a planned row: tier 0's global defaults, then the country module's keys."""
    return with_country_keys(global_config(row, sources))


def global_config(row, sources):
    """The CITIES entry for a planned row from global sources only, carrying its row."""
    cid, ex = row["id"], list(row["extracts"])
    return {
        "plan_row": row,
        "tier0": True,
        "origin": tuple(row["origin"]),
        "water": {"cell_m": water_cell(row["bbox"]), "min_width_px": 2},
        "boundary_notes": "pipeline/world_plan.json",
        "boundary_file": lambda: __import__("osm_units").boundary_file(cid, row, sources),
        "job_blend": lambda: job_blend(row["iso2"]),
        "water_source": lambda bbox: __import__("osm_water").source(ex)(bbox),
        "extract": ex[0] if len(ex) == 1 else ex,
        "iso2": row["iso2"],
        "curve": "borrowed",
        "region_name": row["name"],
        "utc_offset": row["utc_offset"],
        "latin_names": lambda: __import__("osm_units").latin_names(cid),
    }


def planned(path=PLAN):
    """{id: config} for every planned row of the plan file, the country keys not yet merged."""
    if not os.path.exists(path):
        return {}
    rows, sources = load(path)
    return {r["id"]: global_config(r, sources) for r in rows if r.get("status") == "planned"}


if __name__ == "__main__":
    import sys
    rows, sources = load(sys.argv[2] if len(sys.argv) > 2 else PLAN)
    row = next(r for r in rows if r["id"] == sys.argv[1])
    cfg = config(row, sources)
    cfg["job_blend"] = cfg["job_blend"]()
    hooks, mark = country_hooks(row), f"  [{row['module']}]"
    for k, v in cfg.items():
        print(f"  {k}: {'(callable)' if callable(v) else v}{mark if k in hooks else ''}")
