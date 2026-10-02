"""Russia: not RINF but the CIS tariff guide (Тарифное руководство № 4), which ru_register.py
--convert writes into rinf.py's own input files (data/raw/rinf/ru/sections.json, points.json)
plus names.json, read here. ru_register.py's docstring says how; ru_sources.md has the rest.

- A line is one tariff section, its RINF id the section's number ("01-011"). No section takes
  a line number from OSM: `osm_rel` drops every relation, since Russia's route=railway
  relations are whole corridors (the Trans-Siberian, the BAM) and their refs would merge
  tariff sections into them.
- `id_name`: the section's name and English name from names.json ("Обухово — Чудово-Московское";
  English only where a Wikidata line item named for the same two ends has an English label).
- Tariff km are whole kilometres, so a merged section can be about a kilometre off at either
  end: `tol_abs` 1.5.
- The operator is the regional railway of RZD (or Crimea's, Yakutia's, the annexed ones),
  by the road code ru_register writes as each section's manager.
- `skip_line`: the 2022-annexed railways (Donetsk, Luhansk, Melitopol-Kherson) are LEFT OUT
  while ANNEX_RUNNING is False: OSM has no passenger route relation there to say which
  sections trains run on, and the tariff guide lists only what is allowed. Anita (2026-10-01):
  railways we cannot show trains on are not assigned to any country; they come back, with
  Russia (de facto), when a source says which trains run. Crimea is not affected.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NAMES_FILE = ROOT / "data" / "raw" / "rinf" / "ru" / "names.json"
NAMES = json.loads(NAMES_FILE.read_text("utf-8")) if NAMES_FILE.exists() else {}

# Anita's call: False leaves the annexed railways out altogether (no country), until a source
# says which trains run; True draws them as running (Book 2's passenger points as stops, every
# section counting towards completion). Also switch tools/build_regions.py's EXTRA_AREAS.
ANNEX_RUNNING = False

IM = {
    "01": "Октябрьская железная дорога", "17": "Московская железная дорога",
    "24": "Горьковская железная дорога", "28": "Северная железная дорога",
    "51": "Северо-Кавказская железная дорога", "58": "Юго-Восточная железная дорога",
    "61": "Приволжская железная дорога", "63": "Куйбышевская железная дорога",
    "76": "Свердловская железная дорога", "80": "Южно-Уральская железная дорога",
    "83": "Западно-Сибирская железная дорога", "88": "Красноярская железная дорога",
    "92": "Восточно-Сибирская железная дорога", "94": "Забайкальская железная дорога",
    "96": "Дальневосточная железная дорога", "10": "Калининградская железная дорога",
    "91": "Железные дороги Якутии", "97": "ИФР-1", "85": "Крымская железная дорога",
    "89": "Донецкая железная дорога", "84": "Луганская железная дорога",
    "82": "Мелитопольская-Херсонская железная дорога",
}


def id_name(lid, _uop_name=None):
    e = NAMES.get(lid)
    if not e:
        return None
    return (e["name"], e.get("name_en") or "")


def skip_line(base):
    return not ANNEX_RUNNING and bool(NAMES.get(base.split("#")[0], {}).get("annexed"))


COUNTRY = {
    "iso3": "RUS", "langs": ["ru", "en"],
    "osm_rel": lambda _tags: None,
    "id_name": id_name,
    "tol_abs": 1.5,
    "direct_near_m": 1500,
    "im": IM,
    "skip_line": skip_line,
}
