"""Ukraine: not RINF but the CIS tariff guide (Тарифное руководство № 4), whose Ukrainian sheets
ua_register.py --convert writes into rinf.py's own input files (data/raw/rinf/ua/sections.json,
points.json) plus names.json, read here. ua_register.py's docstring says how; ua_sources.md has
the rest. The same recipe as Russia's (rinf_countries/ru.py), with Ukrainian names.

- A line is one tariff section, its RINF id the section's number ("32-001"). No section takes
  a line number from OSM: `osm_rel` drops every relation (OSM's Ukrainian route=railway
  relations are corridors and tariff sections both, named in either language).
- `id_name`: the section's name from names.json, its two ends as shown (Ukrainian, OSM's or
  Wikidata's names), "Чаплине — Покровськ"; the English one from the same ends' English names.
- `station_en`: the English name ua_register found for a point (OSM's name:en, else
  Wikidata's), checked there against the Ukrainian name.
- Tariff km are whole kilometres: `tol_abs` 1.5, and `direct_near_m` as Russia's.
- The operator is the regional railway (a branch of АТ «Укрзалізниця»), by the road code
  ua_register writes as each section's manager.
- Crimea and the 2022-annexed area are not in the extract or the register (ua_register
  --clip, --convert): Crimea is built with Russia, the annexed area with no country.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NAMES_FILE = ROOT / "data" / "raw" / "rinf" / "ua" / "names.json"
NAMES = json.loads(NAMES_FILE.read_text("utf-8")) if NAMES_FILE.exists() else {}

IM = {
    "32": "Південно-Західна залізниця", "35": "Львівська залізниця",
    "40": "Одеська залізниця", "43": "Південна залізниця",
    "45": "Придніпровська залізниця", "48": "Донецька залізниця",
}


def id_name(lid, _uop_name=None):
    e = NAMES.get(lid)
    if not e:
        return None
    return (e["name"], e.get("name_en") or "")


_EN = None


def station_en(point, _name):
    global _EN
    if _EN is None:
        pf = NAMES_FILE.parent / "points.json"
        _EN = {r["uopid"]: r["name_en"] for r in
               (json.loads(pf.read_text("utf-8"))["rows"] if pf.exists() else [])
               if r.get("name_en")}
    return _EN.get(point.get("uopid"), "")


def stop_name(point):
    """ua_register already decided which points are stops (Book 2's passenger operation and a
    timetable call, an OSM train route stop or a halt): each is a stop under its own name, at
    the OSM station of that name if one is near, else at its own place. Without this a stop
    OSM maps with no station node (Ірпінь: unnamed stop positions only) became a junction and
    merged away, and the timetable's calls there found no station."""
    if point.get("type") in ("10", "70"):
        return point.get("name") or None
    return None


COUNTRY = {
    "stop_name": stop_name,
    "iso3": "UKR", "langs": ["uk", "en"],
    "osm_rel": lambda _tags: None,
    "id_name": id_name,
    "tol_abs": 1.5,
    "direct_near_m": 1500,
    "im": IM,
    "station_en": station_en,
}
