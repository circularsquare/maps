"""Moldova: not RINF but the CIS tariff guide (Тарифное руководство № 4), whose CFM sheet ("Млд")
bymd_register.py --cc md --convert writes into rinf.py's own input files (data/raw/rinf/md/
sections.json, points.json) plus names.json, read here. The same recipe as Ukraine's
(rinf_countries/ua.py); md_sources.md has the rest.

- A line is one tariff section, its RINF id the section's number ("39-007").
- `id_name`: the section's name from names.json, its two ends as shown (Romanian, OSM's names),
  "Ungheni — Chișinău". No English names (the names are Latin already), except in
  Transnistria, whose names are Russian: there OSM's Romanian name is the English one.
- Tariff km are whole kilometres: `tol_abs` 1.5, and `direct_near_m` as Russia's.
- Transnistria (OSM relation 65335, with Bender) is drawn as Moldova's and greyed (Anita,
  2026-10-04): no train runs there, so the timetable marks its sections not running. Its
  section rows carry the manager code "39P": CFM, whose sheet lists the track, and the
  Pridnestrovian Railway, which runs it. Both are named, so gtfs_served counts the line as
  CFM's (whose trains the feed carries) and greys it, rather than leaving it as an operator
  missing from the feed.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NAMES_FILE = ROOT / "data" / "raw" / "rinf" / "md" / "names.json"
NAMES = json.loads(NAMES_FILE.read_text("utf-8")) if NAMES_FILE.exists() else {}

IM = {"39": "Calea Ferată din Moldova", "39P": "Calea Ferată din Moldova; Приднестровская железная дорога"}


def id_name(lid, _uop_name=None):
    e = NAMES.get(lid)
    if not e:
        return None
    return (e["name"], e.get("name_en") or "")


def stop_name(point):
    """bymd_register already decided which points are stops (Book 2's passenger operation and a
    timetable call, an OSM train route stop or a halt): each is a stop under its own name."""
    if point.get("type") in ("10", "70"):
        return point.get("name") or None
    return None


COUNTRY = {
    "stop_name": stop_name,
    "iso3": "MDA", "langs": ["ro", "en"],
    "osm_rel": lambda _tags: None,
    "id_name": id_name,
    "tol_abs": 1.5,
    "direct_near_m": 1500,
    "im": IM,
}
