"""religiondots' geography modules, loaded by path, and its data folders, read-only.

languagedots shares religiondots' placement layers (data/geo/<cc>/), its sea clip (water.py),
its Kontur density-cap registry (kontur_cap.py) and its geography checks
(sources/geo_checks.py). Those are facts about WHERE PEOPLE LIVE, not about religion, and
redoing them per map would mean two copies drifting apart.

LOADED BY PATH, NEVER BY sys.path. religiondots has its own countries.py and taxonomy modules
whose names collide with ours; putting its folder on sys.path would let them shadow these.

READ-ONLY. Nothing here writes into religiondots. The one module that can write is water.py,
which caches each clip beside its data; `water_clip` below reads religiondots' cache when it is
current and otherwise points the module's cache at languagedots/data/geo/_waterclip first.
"""
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RD = HERE.parent / "religiondots"
RD_GEO = RD / "data" / "geo"
OUR_CACHE = HERE / "data" / "geo" / "_waterclip"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(f"rd_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_mods = {}


OUR_CAP = HERE / "kontur_cap.csv"


def module(name):
    if name not in _mods:
        path = {"water": RD / "water.py", "kontur_cap": RD / "kontur_cap.py",
                "geo_checks": RD / "sources" / "geo_checks.py"}[name]
        mod = _load(name, path)
        if name == "kontur_cap":
            # Kontur cap blocks: religiondots' registry, plus languagedots' own rows for blocks
            # religiondots never met (same columns as religiondots/kontur_cap.csv). Written here,
            # never into religiondots' file.
            import csv
            orig = mod.read_registry

            def merged(cc=None):
                rows = orig(cc)
                if OUR_CAP.exists():
                    with open(OUR_CAP, encoding="utf-8", newline="") as fh:
                        rows += [r for r in csv.DictReader(fh) if cc is None or r["cc"] == cc]
                return rows
            mod.read_registry = merged
        _mods[name] = mod
    return _mods[name]


def water_clip(place, cc, src):
    w = module("water")
    rd_cache, rd_meta = w.CACHE / f"{cc}_{Path(src).stem}.gpkg", w.CACHE / f"{cc}_{Path(src).stem}.json"
    fresh = (rd_cache.exists() and rd_meta.exists()
             and json.loads(rd_meta.read_text()) == w._stamp(src))
    if not fresh:
        OUR_CACHE.mkdir(parents=True, exist_ok=True)
        w.CACHE = OUR_CACHE
    return w.clip(place, cc, src)
