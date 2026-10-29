"""Real line colours for the real networks (T-1770): `realnet.py net` and `colours` call `line_colours`,
which gives each line of a city's network a "#rrggbb" or None (the sim's `realsave` then falls back to
its palette).

Where a colour comes from, by source:
  GTFS feeds: the route's `route_color` (New York, Toronto, Paris, London's National Rail).
  OSM route relations (Sendai): the relation's or its route master's `colour` tag, else noritetsu.
  Trip files (London TfL, Seoul, Tokyo): noritetsu's line model (`noritetsu/dist/data/<cc>/lines.json`,
    read only), whose colours are OSM tags, hand tables (official swatches) or Wikidata. London and
    Seoul name the noritetsu line per route below; Tokyo matches by line name and operator.
A colour too dark or too light to show on either map is moved to the nearest one that does (`legible`).
"""
import json
import os
import re
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NORI = os.path.join(os.path.dirname(os.path.dirname(HERE)), "noritetsu", "dist", "data")

# noritetsu line ids, per route code of the trip files
LON_TFL = {  # route code -> noritetsu gb line id (name)
    "BAK": "m7666698", "CEN": "m7254206", "CIR": "m7671007", "DIS": "m7677096", "DLR": "m7772562", "ELL": "m9683131",  # Windrush
    "EZL": "m9753125", "GOB": "m9683820",  # Suffragette
    "HAM": "m7666927", "JUB": "m7673326", "MET": "m7673626", "NLL": "m112649",  # Mildmay
    "NOR": "m7702944", "PIC": "m7702940", "URL": "g2aabd5a007",  # Liberty
    "VIC": "m6354922", "WAC": "m7672075", "WAG": "m3208125",  # Weaver
    "WEL": "m9752360",  # Lioness
}
SEO_LINES = {  # route code -> noritetsu kr line id
    "1": "m8691899", "2": "m7625892", "3": "m7879839", "4": "m7625893", "5": "m7879871", "6": "kbe4993c6fe",
    "7": "m7922930", "8": "m7919019", "9": "kf9c192d0dc",
    "AR": "m7919001", "GC": "m8656365", "GG": "m8735483", "GJ": "m8667957", "GP": "kec8639f119",
    "I1": "m7854149", "I2": "k1c136ed565", "SB": "m11619514", "SH": "m8725316", "SL": "k324520f9b8", "SN": "m7728256",
    "UI": "kb116d42600", "UJ": "m7530882",
}
# Tokyo (s12 operator keys) -> a fragment of noritetsu's operator name
JP_OPERATOR = {
    "jr_east": "東日本旅客鉄道", "jr_central": "東海旅客鉄道", "toei": "東京都", "tokyo_metro": "東京地下鉄|東京メトロ",
    "yokohama": "横浜市", "keikyu": "京浜急行", "odakyu": "小田急", "keio": "京王", "tokyu": "東急|東京急行", "seibu": "西武",
    "tobu": "東武", "keisei": "京成電鉄", "shin_keisei": "新京成", "sotetsu": "相模鉄道|相鉄", "hokuso": "北総",
    "chiba_monorail": "千葉都市モノレール", "tama_monorail": "多摩都市モノレール", "yurikamome": "ゆりかもめ",
    "rinkai": "東京臨海高速", "toyo": "東葉高速", "saitama_rr": "埼玉高速", "minatomirai": "横浜高速",
    "tx": "首都圏新都市鉄道", "tokyo_monorail": "東京モノレール",
}

_CSS = None


def css_names():
    global _CSS
    if _CSS is None:
        try:
            from matplotlib.colors import CSS4_COLORS
            _CSS = {k.lower(): v for k, v in CSS4_COLORS.items()}
        except Exception:
            _CSS = {}
    return _CSS


def hexcol(v):
    """A GTFS or OSM colour ("0062CF", "#abc", "SeaGreen") as "#rrggbb", else None."""
    v = (v or "").strip()
    if not v:
        return None
    if re.fullmatch(r"#?[0-9a-fA-F]{6}", v):
        return "#" + v.lstrip("#").lower()
    if re.fullmatch(r"#[0-9a-fA-F]{3}", v):
        return "#" + "".join(c * 2 for c in v[1:]).lower()
    return css_names().get(v.lower())


def _oklab_l(rgb):
    c = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    l = 0.4122214708 * c[0] + 0.5363325363 * c[1] + 0.0514459929 * c[2]
    m = 0.2119034982 * c[0] + 0.6806995451 * c[1] + 0.1073969566 * c[2]
    s = 0.0883024619 * c[0] + 0.2817188376 * c[1] + 0.6299787005 * c[2]
    return 0.2104542553 * np.cbrt(l) + 0.7936177850 * np.cbrt(m) - 0.0040720468 * np.cbrt(s)


L_MIN, L_MAX = 0.40, 0.90


def legible(hx):
    """Mix a colour towards white until its OKLab lightness is at least L_MIN (the Northern line's
    black), or towards black until it is at most L_MAX (white); the game's maps are light and dark."""
    rgb = np.array([int(hx[i:i + 2], 16) / 255 for i in (1, 3, 5)])
    for _ in range(100):
        L = _oklab_l(rgb)
        if L < L_MIN:
            rgb = rgb + (1 - rgb) * 0.04
        elif L > L_MAX:
            rgb = rgb * 0.96
        else:
            break
    return "#" + "".join(f"{int(round(c * 255)):02x}" for c in rgb)


def gtfs_colours(folder):
    import pandas as pd
    p = os.path.join(folder, "routes.txt")
    if not os.path.exists(p):
        return {}
    r = pd.read_csv(p, dtype=str, keep_default_na=False)
    if "route_color" not in r:
        return {}
    return {rid: hexcol(c) for rid, c in zip(r["route_id"], r["route_color"]) if hexcol(c)}


def osm_colours(path):
    """{route key as realnet.py writes it ("m<master id>" or "r<relation id>"): colour}."""
    d = json.load(open(path, encoding="utf-8"))
    out = {}
    master_of = {}
    for e in d["elements"]:
        if e["type"] == "relation" and e.get("tags", {}).get("type") == "route_master":
            for m in e["members"]:
                master_of[m["ref"]] = e
    for e in d["elements"]:
        t = e.get("tags", {})
        if e["type"] != "relation" or t.get("type") not in ("route", "route_master"):
            continue
        c = hexcol(t.get("colour"))
        if t.get("type") == "route_master":
            if c:
                out["m%d" % e["id"]] = c
        elif c or e["id"] in master_of:
            out["r%d" % e["id"]] = c or hexcol(master_of[e["id"]].get("tags", {}).get("colour"))
    return {k: v for k, v in out.items() if v}


_NORI = {}


def nori(cc):
    """noritetsu's lines of a region: [dict], read only."""
    if cc not in _NORI:
        p = os.path.join(NORI, cc, "lines.json")
        _NORI[cc] = json.load(open(p, encoding="utf-8"))["lines"] if os.path.exists(p) else []
    return _NORI[cc]


def nori_by_id(cc, lid):
    for l in nori(cc):
        if l["id"] == lid:
            return hexcol(l.get("colour"))
    return None


def jpnorm(s):
    s = re.sub(r"[（(][^）)]*[）)]", "", s)
    s = s.replace("JR", "").replace("本線", "線").replace("　", "").replace(" ", "")
    return s


def nori_by_name(cc, name, op_frag, norm=jpnorm):
    """The colour of the noritetsu lines named `name` (normalised) whose operator holds `op_frag`: a hand
    table's if the lines have one, else the commonest; None where nothing matches or it is a tie."""
    want = norm(name)
    cands = [l for l in nori(cc) if hexcol(l.get("colour")) and norm(l["name"]) == want
             and (not op_frag or re.search(op_frag, l.get("operator", "")))]
    if not cands:
        return None
    table = [l for l in cands if str(l.get("colour_src", "")).startswith("table")]
    pool = table or cands
    cnt = Counter(hexcol(l["colour"]) for l in pool).most_common()
    if len(cnt) > 1 and cnt[0][1] == cnt[1][1]:
        return None
    return cnt[0][0]


def line_colours(cfg, lines):
    """One "#rrggbb" or None per line of a city's network (`cfg` is its realnet.CITIES entry)."""
    srcs = {s["name"]: s for s in cfg["sources"]}
    cache = {}
    out = []
    for l in lines:
        s = srcs.get(l["feed"], {})
        ad, route = s.get("adapter"), l["route"]
        c = None
        if ad == "gtfs":
            c = cache.setdefault(l["feed"], gtfs_colours(s["folder"])).get(route)
        elif ad == "osm":
            c = cache.setdefault(l["feed"], osm_colours(s["file"])).get(route)
            c = c or nori_by_id("jp", route)
            c = c or nori_by_name("jp", l["name"].split(" (")[0], None)
        elif ad == "n02":
            c = nori_by_name("jp", l["name"], re.escape(l["operator"].replace("　", "")))
        elif ad == "trips":
            if cfg.get("pack") == "lon":
                c = nori_by_id("gb", LON_TFL.get(route, ""))
            elif cfg.get("pack") == "seo":
                c = nori_by_id("kr", SEO_LINES.get(route, ""))
            elif cfg.get("pack") == "tyo":
                c = nori_by_name("jp", l["name"], JP_OPERATOR.get(l["operator"]))
        out.append(legible(c) if c else None)
    return out
