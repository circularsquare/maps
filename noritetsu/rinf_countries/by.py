"""Belarus: not RINF but the CIS tariff guide (Тарифное руководство № 4), whose Belarusian
Railway sheet ("Бел") bymd_register.py --cc by --convert writes into rinf.py's own input files
(data/raw/rinf/by/sections.json, points.json) plus names.json, read here. The same recipe as
Ukraine's (rinf_countries/ua.py); by_sources.md has the rest.

- A line is one tariff section, its RINF id the section's number ("13-090").
- `id_name`: the section's name from names.json, its two ends as shown (Belarusian, OSM's
  names), "Мінск-Сартавальны — Орша-Цэнтральная"; the English one from the same ends' English
  names where both have one.
- `station_en`: the English name bymd_register found for a point (OSM's name:en, else
  Wikidata's), checked there against the Belarusian and Russian names.
- Tariff km are whole kilometres: `tol_abs` 1.5, and `direct_near_m` as Russia's.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NAMES_FILE = ROOT / "data" / "raw" / "rinf" / "by" / "names.json"
NAMES = json.loads(NAMES_FILE.read_text("utf-8")) if NAMES_FILE.exists() else {}

IM = {"13": "Беларуская чыгунка"}


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
    """bymd_register already decided which points are stops: each is a stop under its own
    name (ua.py's stop_name)."""
    if point.get("type") in ("10", "70"):
        return point.get("name") or None
    return None


COUNTRY = {
    "stop_name": stop_name,
    "iso3": "BLR", "langs": ["be", "en"],
    "osm_rel": lambda _tags: None,
    "id_name": id_name,
    "tol_abs": 1.5,
    "direct_near_m": 1500,
    "im": IM,
    "station_en": station_en,
}
