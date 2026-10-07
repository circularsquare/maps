"""Russia: not RINF but the CIS tariff guide (Тарифное руководство № 4), which ru_register.py
--convert writes into rinf.py's own input files (data/raw/rinf/ru/sections.json, points.json)
plus names.json, read here. ru_register.py's docstring says how; ru_sources.md has the rest.

- A line is one tariff section, its RINF id the section's number ("01-011"). No section takes
  a line number from OSM: `osm_rel` drops every relation, since Russia's route=railway
  relations are whole corridors (the Trans-Siberian, the BAM) and their refs would merge
  tariff sections into them.
- `id_name`: the section's name and English name from names.json ("Обухово — Чудово-Московское",
  "Obukhovo — Chudovo-Moskovskoye": the English built from its end stations' English names,
  else a Wikidata line item's English label).
- `station_en`: stations' English names, Wikidata's by ESR code, where OSM gave none.
- Tariff km are whole kilometres, so a merged section can be about a kilometre off at either
  end: `tol_abs` 1.5.
- The operator is the regional railway of RZD (or Crimea's, Yakutia's, the annexed ones),
  by the road code ru_register writes as each section's manager; Kazakhstan's railway for
  its two sections around Iletsk on Russian soil.
- Border crossings (ru_register.BORDER) end at borders.EXTRA's points, type 90.
- The 2022-annexed railways (Donetsk, Luhansk, Melitopol-Kherson). Anita (2026-10-01): left
  to no country until a source says which trains run there; (2026-10-04) "donetsk luhansk to
  russia makes sense if theres trains run". The source is poizdato.net's pages for the
  Russian-run suburban trains there (ru_register.py --annex-trains, annex_evidence), so
  ANNEX_RUNNING is True: they are built with Russia, and `suspended` greys every line no
  train runs over. ru_register writes a tariff section trains run over in part as two lines,
  the part they run over under its id and the rest under "<id>~" (names.json `annex_running`).
  With ANNEX_RUNNING False, `skip_line` leaves them all out again. Crimea is not affected.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NAMES_FILE = ROOT / "data" / "raw" / "rinf" / "ru" / "names.json"
NAMES = json.loads(NAMES_FILE.read_text("utf-8")) if NAMES_FILE.exists() else {}

# Anita's call: False leaves the annexed railways out altogether (no country); True builds them
# with Russia, drawn as running where a train in data/raw/ru/annex_trains.json runs and greyed
# elsewhere (2026-10-04). tools/build_regions.py's EXTRA_AREAS must match (ru: annex.geojson).
ANNEX_RUNNING = True

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
    # Kazakhstan's railway's two sections on Russian soil (ru_register.FOREIGN)
    "68": "Қазақстан темір жолы",
}


def id_name(lid, _uop_name=None):
    e = NAMES.get(lid)
    if not e:
        return None
    return (e["name"], e.get("name_en") or "")


def skip_line(base):
    return not ANNEX_RUNNING and bool(NAMES.get(base.split("#")[0], {}).get("annexed"))


def suspended(_ref, lids):
    """A line of the annexed railways no train runs over: greyed as not running."""
    def closed(lid):
        e = NAMES.get(lid.split("#")[0], {})
        return bool(e.get("annexed")) and not e.get("annex_running")
    return any(closed(lid) for lid in lids)


_EN = None


def station_en(point, name):
    """A section end's English name: Wikidata's label by its ESR code, which ru_register
    --convert wrote into points.json (`name_en`) after checking it against the tariff guide's
    name, checked again here against the name the station is shown under (OSM's, for a stop
    matched to an OSM station of another spelling or by distance alone)."""
    global _EN
    if _EN is None:
        sys.path.insert(0, str(ROOT))
        pf = NAMES_FILE.parent / "points.json"
        _EN = {r["uopid"]: r["name_en"] for r in
               (json.loads(pf.read_text("utf-8"))["rows"] if pf.exists() else [])
               if r.get("name_en")}
    en = _EN.get(point.get("uopid"))
    if not en:
        return ""
    from ru_register import EN_AGREE, en_score
    return en if en_score(name, en) >= EN_AGREE else ""


COUNTRY = {
    "iso3": "RUS", "langs": ["ru", "en"],
    "osm_rel": lambda _tags: None,
    "id_name": id_name,
    "tol_abs": 1.5,
    "direct_near_m": 1500,
    "im": IM,
    "skip_line": skip_line,
    "suspended": suspended,
    "station_en": station_en,
}
