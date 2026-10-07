"""India: not RINF but Wikidata's IR line chains with the IR timetable's km, which
in_register.py --convert writes into rinf.py's own input files (data/raw/in/sections.json,
points.json) plus names.json, read here. in_register.py's docstring says how; in_sources.md
has the sources, the numbers and what is off.

- A line is one Wikidata line item ("Mathura-Vadodara Section", its RINF id the item's QID),
  or a stretch of the timetable's network no item covers, junction to junction ("T-GD-BNY").
  No line takes a number or name from OSM: `osm_rel` drops every relation (OSM India's 600
  route=railway relations are a mix of sections and whole corridors, 14 with a ref).
- `id_name`: the line's name from names.json (English, as IR and Wikidata write it).
- The timetable's km are whole kilometres per pair of calls, so a section can be about a
  kilometre off either way: `tol_abs` 1.5, as for Russia's tariff km.
- `direct_near_m`: an end-to-end retrace must pass every placed point of the section within
  1.5 km (Russia's guard against finding another line).
- The operator is the IR zone most of the line's stations belong to (Wikidata: station ->
  division -> zone), the zone code in_register writes as each section's manager.
- `station_en`: Wikidata's English label where OSM's name is not in Latin script.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "in"


def _load(fn, default):
    p = RAW / fn
    return json.loads(p.read_text("utf-8")) if p.exists() else default


NAMES = _load("names.json", {})
_PTS = _load("points.json", {"rows": []})["rows"]
EN_BY_UOPID = {r["uopid"].upper(): r.get("name_wd") or r["name"] for r in _PTS}

IM = {
    "NR": "Northern Railway", "NER": "North Eastern Railway",
    "NFR": "Northeast Frontier Railway", "ER": "Eastern Railway",
    "SER": "South Eastern Railway", "SCR": "South Central Railway", "SR": "Southern Railway",
    "CR": "Central Railway", "WR": "Western Railway", "SWR": "South Western Railway",
    "NWR": "North Western Railway", "WCR": "West Central Railway",
    "NCR": "North Central Railway", "SECR": "South East Central Railway",
    "ECoR": "East Coast Railway", "ECR": "East Central Railway",
    "SCoR": "South Coast Railway", "KR": "Konkan Railway", "MR": "Metro Railway, Kolkata",
}

LATIN = re.compile(r"[A-Za-z]")


def id_name(lid, _uop_name=None):
    e = NAMES.get(lid.split("#")[0])
    if not e:
        return None
    return (e["name"], e.get("name_en") or "")


def station_en(point, name):
    if LATIN.search(name or ""):
        return ""
    return EN_BY_UOPID.get((point.get("uopid") or "").upper(), "")


COUNTRY = {
    "iso3": "IND", "langs": ["en", "hi"],
    "osm_rel": lambda _tags: None,
    "id_name": id_name,
    "tol_abs": 1.5,
    "direct_near_m": 1500,
    "im": IM,
    "station_en": station_en,
}
