"""Indonesia: not RINF but id.wikipedia's line articles (station lists with KAI's km posts),
which id_register.py --convert writes into rinf.py's own input files (data/raw/id/sections.json,
points.json) plus names.json, read here. id_register.py's docstring says how; id_sources.md has
the sources, the numbers and the decisions.

- A line is one id.wikipedia line article ("Jalur kereta api Cikampek–Cirebon–Kroya"), its RINF
  id the article's Wikidata item (or its title), named as the article is less "Jalur kereta api"
  (`id_name`). Whoosh (Kereta Cepat Jakarta–Bandung) is written in by hand (id_register.HAND).
- No line takes a number or a name from OSM: OSM Indonesia's 74 route=railway relations are
  KAI's own segment names ("Jalur Kereta Api Solo Balapan–Kertosono"), cut elsewhere than the
  articles (`osm_rel` drops them).
- KAI's km posts are to the metre, but a station's point (OSM's node) is anywhere along its
  platforms and yard, and a few sections are crow-fly estimates: `tol_abs` 1.0 km.
- `direct_near_m`: an end-to-end retrace must pass every placed point within 1.5 km.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "id"


def _load(fn, default):
    p = RAW / fn
    return json.loads(p.read_text("utf-8")) if p.exists() else default


NAMES = _load("names.json", {})

IM = {"KAI": "Kereta Api Indonesia", "KCIC": "Kereta Cepat Indonesia China"}


def id_name(lid, _uop_name=None):
    e = NAMES.get(lid.split("#")[0])
    if not e:
        return None
    return (e["name"], e.get("name_en") or "")


COUNTRY = {
    "iso3": "IDN", "langs": ["id", "en"],
    "osm_rel": lambda _tags: None,
    "id_name": id_name,
    "tol_abs": 1.0,
    "direct_near_m": 1500,
    "im": IM,
}
