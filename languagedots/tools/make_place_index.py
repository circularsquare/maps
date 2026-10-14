"""Build place_index.json: the first-level divisions (provinces, states, regions) of the world as
name, country and bounding box, for the place search's instant local answers (index.html, both
languagedots and religiondots; the same file, copied by this script into both folders).

Natural Earth 10m admin_1_states_provinces (public domain), the zip tools/make_admin1.py already
downloads into data/raw/ne_admin1/. Kept per division: its English name (`name_en`, else `name`;
skipped when neither is in Latin letters), its country, and its bounding box to 2 decimals. A
division wider than 180 degrees (it crosses the date line: Chukotka, Fiji, the Aleutians) gets the
box of its biggest part instead, so a fit does not frame the whole world.

    python tools/make_place_index.py

Format: {"c": {"de": "Germany", ...}, "a": [[name, "de", west, south, east, north], ...]}.
Published beside index.html (tools/deploy.py GZ_FILES), fetched only when the search box is first
opened. Country keys are ISO alpha-2 in lower case, as both maps key their countries, falling back to
the ISO alpha-3 where Natural Earth has none.
"""
import json
import re
from pathlib import Path

import geopandas as gpd

MAPS = Path(r"C:\Users\anita\projects\maps")
LD = MAPS / "languagedots"
RD = MAPS / "religiondots"
SRC = LD / "data/raw/ne_admin1/ne_10m_admin_1_states_provinces.zip"
OUTS = [LD / "place_index.json", RD / "data/processed/place_index.json"]

g = gpd.read_file(f"zip://{SRC}")
latin = re.compile(r"^[\u0020-\u024F\u1E00-\u1EFF]+$")

countries, rows = {}, []
for r in g.itertuples(index=False):
    name = r.name_en if isinstance(r.name_en, str) and r.name_en.strip() else r.name
    if not isinstance(name, str) or not name.strip() or not latin.match(name.strip()):
        continue
    if r.geometry is None or r.geometry.is_empty:
        continue
    cc = (r.iso_a2 if isinstance(r.iso_a2, str) and r.iso_a2 not in ("-99", "") else r.adm0_a3).lower()
    countries.setdefault(cc, r.admin)
    w, s, e, n = r.geometry.bounds
    if e - w > 180:
        big = max(r.geometry.geoms, key=lambda q: q.area)
        w, s, e, n = big.bounds
    rows.append([name.strip(), cc, round(w, 2), round(s, 2), round(e, 2), round(n, 2)])

doc = {"c": countries, "a": rows}
for out in OUTS:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out} ({len(rows):,} divisions, {out.stat().st_size / 1e3:.0f} KB)")
