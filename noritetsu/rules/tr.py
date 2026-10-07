"""Türkiye's rules for build_model.py (country_rules): which route=train relations are named
trains rather than lines. tr_sources.md, "Lines vs named trains", has the reasoning.

What OSM Türkiye has (data/proc/tr, 2026-10-03; 100 route=train relations, 73 of them TCDD
Taşımacılık's):
  - YHT (Yüksek Hızlı Tren) services, one relation per direction and pair of ends: "Ankara -
    İstanbul YHT Hattı", "Konya - İstanbul YHT Hattı", "Ankara - Sivas YHT Hattı", "Ankara -
    Karaman YHT Hattı", "Eskişehir - Ankara Yüksek Hızlı Tren hattı". The YHT is TCDD's
    interval product (Ankara - İstanbul about 20 a day each way, Ankara - Konya about 10,
    Ankara - Sivas 4 or more): a rider uses it as a line, as Germany's ICE lines and Spain's
    AVE, so every YHT relation is a LINE, the once- or twice-a-day İstanbul - Sivas and
    İstanbul - Karaman pairs included (they are the same product over the same track).
  - The suburban and regional trains: Marmaray, Başkentray, İZBAN, Gaziray, Adaray, AYBAN,
    KonyaRay, Halkalı - Bahçeşehir, and the "Bölgesel" (regional) trains, "B32 Basmane-Ödemiş
    Bölgesel", "Sivas - Samsun Bölgesel Tren Hattı", "Mersin-Adana Bölgesel Tren Hattı",
    "Karaman - Konya Bölgesel Tren Hattı", "Ankara - Polatlı Bölgesel Tren Hattı". LINES: they
    are the regional service of their corridor, as Germany's RE lines, even where one runs
    once or twice a day (Basmane - Uşak).
  - The anahat (main-line) trains, each a single named train once a day (or a few times a
    week): Doğu Ekspresi (Ankara - Kars), Turistik Doğu Ekspresi (a seasonal tour train over
    the same track), Ankara Ekspresi (the İstanbul - Ankara sleeper), Ege, Erciyes, Güney
    Kurtalan, Van Gölü, Pamukkale, Toros, Göller, Güller, 6 Eylül, 17 Eylül Ekspresi, İzmir
    and Konya Mavi Treni, and the international Istanbul - Bucharest/Sofia train (INT ...).
    NAMED TRAINS: no percentage of their own; their track counts through the register lines.
"""
import re

from rules.shared import EU_TRAIN

# "Ekspresi"/"Ekspres" (Doğu Ekspresi, 6 Eylül Ekspresi), "Mavi Treni" (İzmir/Konya Mavi
# Treni), and any relation the operator tags service=international or night.
NAMED = re.compile(r"\bEkspres\w*|\bMavi Tren(?:i)?\b|\bExpress\b|\bTuristik\b", re.I)
LINE_WORDS = re.compile(r"\bYHT\b|Yüksek Hızlı|Hızlı Tren|Bölgesel|Banliyö|ray\b|İZBAN|"
                        r"Marmaray|\bB\d+\b", re.I)
NAMED_SERVICE = {"international", "night", "long_distance"}


STOP_ON_TRACK_M = 300    # a station this close to a route's own track is one of its stops


def extra_route_stops(ways, rels, stops, coords, stations, resolved, log):
    """Stops for the TCDD route relations that list none. OSM Türkiye maps the YHT services and
    the named trains as ways only ("Ankara - İstanbul YHT Hattı", 464 ways, no stop members),
    so build_model dropped every one as having under two stops. Such a route stops at each
    station TCDD sells tickets to (tr_register.tcdd_names) that lies within STOP_ON_TRACK_M
    of its own track; a YHT route only at a YHT station or one of the shared stations the YHT
    calls at (tr_register.hs_ok). A named train calls at fewer stations than that, but named
    trains count towards nothing; what matters is that the YHT lines exist."""
    import math
    import numpy as np
    import build_model as bm
    import tr_register as trr
    tc = trr.tcdd_names(lambda m: None) or {}
    cand = [(sid, s) for sid, s in stations.items()
            if trr.station_key(s.get("name")) in tc]
    if not cand:
        return {}
    cx = np.array([s["lon"] for _sid, s in cand])
    cy = np.array([s["lat"] for _sid, s in cand])
    out = {}
    for rid, (tags, members) in rels.items():
        if tags.get("type") != "route" or tags.get("route") != "train":
            continue
        if len([m for m in bm.stop_members(members) if m in resolved]) >= 2:
            continue
        rw = [r for ty, r, _ in members if ty == "w" and r in ways]
        if not rw:
            continue
        nodes = np.unique(np.concatenate([np.asarray(ways[w][1], dtype=np.int64) for w in rw]))
        pos, ok = coords.many(nodes)
        nodes, pos = nodes[ok], pos[ok]
        x, y = coords.x[pos] / 1e7, coords.y[pos] / 1e7
        if not x.size:
            continue
        hs = bool(re.search(r"\bYHT\b|Yüksek Hızlı", tags.get("name") or "", re.I))
        box = (cx >= x.min() - 0.01) & (cx <= x.max() + 0.01) & \
              (cy >= y.min() - 0.01) & (cy <= y.max() + 0.01)
        got = {}
        for i in np.nonzero(box)[0].tolist():
            sid, s = cand[i]
            if hs and not trr.hs_ok(s["name"]):
                continue
            d = np.hypot((x - s["lon"]) * math.cos(math.radians(s["lat"])) * 111320,
                         (y - s["lat"]) * 110570)
            j = int(np.argmin(d))
            if d[j] <= STOP_ON_TRACK_M:
                got[sid] = {int(nodes[j])}
        if len(got) >= 2:
            out[rid] = got
    log(f"TR rules: {len(out)} TCDD routes with no stop members given their stations as stops "
        f"({sum(len(v) for v in out.values())} stops)")
    return out


def looks_like_service(tags, name, name_en):
    """Is this OSM route=train relation (or route_master) a named train rather than a line?"""
    name = name or ""
    text = " ".join(t for t in (name, name_en, tags.get("ref")) if t)
    if NAMED.search(text) or EU_TRAIN.search(name) or EU_TRAIN.search(tags.get("ref") or ""):
        return True
    if set((tags.get("service") or "").split(";")) & NAMED_SERVICE:
        return not LINE_WORDS.search(text)
    return False
