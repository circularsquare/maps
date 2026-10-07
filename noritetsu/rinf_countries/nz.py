"""New Zealand: not RINF but KiwiRail's own line register (open data, CC BY 4.0: line names,
km posts and named locations), which nz_register.py --convert writes into rinf.py's input files
(data/raw/nz/sections.json, points.json) plus names.json, read here. nz_register.py's
docstring says how; nz_sources.md has the sources, the numbers and the decisions.

- A line is one KiwiRail line ("North Island Main Trunk", "Wairarapa Line"), its RINF id
  KiwiRail's abbreviation (NIMT, WRAPA), named by KiwiRail's name (`id_name`). Only the
  stretches with scheduled passenger trains are written (nz_register.SCOPE).
- No line takes a number or a name from OSM's route=railway relations (`osm_rel`).
- OSM ways carry their line's name ("North Island Main Trunk Up Main"): `way_line` makes a
  line's traces prefer its own ways in pass 2.
- Sections end where another line's end meets this one (`cut_at_junctions`, the shared points
  nz_register lists).
- OSM stations a train route stops at become stops (`osm_stops`): KiwiRail's
  Station/Passenger flag misses Te Huia's and the new Auckland stations.
- Stop names are read without their platform (`plain_name`, rules/nz.py's PLATFORM_SUFFIX).
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "nz"


def _load(fn, default):
    p = RAW / fn
    return json.loads(p.read_text("utf-8")) if p.exists() else default


META = _load("names.json", {"lines": {}, "cut": [], "way_names": {}})


def id_name(lid, _uop_name=None):
    e = META["lines"].get(lid.split("#")[0])
    if not e:
        return None
    return (e["name"], e.get("name_en") or "")


# An OSM way's name as its line: "North Island Main Trunk Up Main", "North Island Main Trunk –
# Down Main", "North Auckland Line/Br 57", "East Link Up Main" -> the KiwiRail line id.
WAY_SUFFIX = re.compile(r"\s*/.*$|\s*[–-]?\s*\b(?:Up|Down|Centre|Center|Third|West|East|Middle)\s+"
                        r"Main$|\s+(?:Up|Down)\s+Main$")


def way_line(tags):
    n = tags.get("name")
    if not n:
        return None
    return META["way_names"].get(WAY_SUFFIX.sub("", n).strip())


def plain_name(tags):
    import build_model
    return build_model.plain_name(tags, "nz")


# Stops OSM maps no station node for, placed at KiwiRail's point (rinf.py's `stop_name`): Levin,
# the Capital Connection's stop between Shannon and Ōtaki (KiwiRail: Station/Passenger), has
# none in the 2026-10-03 extract. Only names here; every other point is left to rinf's rule.
AT_KIWIRAIL = {"Levin"}


def stop_name(p):
    return p.get("name") if p.get("name") in AT_KIWIRAIL else None


IM = {"KiwiRail": "KiwiRail", "DCC": "Dunedin City Council"}

COUNTRY = {
    "iso3": "NZL", "langs": ["en"],
    "osm_rel": lambda _tags: None,
    "id_name": id_name,
    "way_line": way_line,
    "cut_at_junctions": set(META["cut"]),
    "osm_stops": True,
    "plain_name": plain_name,
    "stop_name": stop_name,
    "tol_abs": 0.5,
    "im": IM,
}
