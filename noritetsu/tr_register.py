"""Türkiye: register lines are OpenStreetMap's named track, kr_register's recipe by way of
gb_register's repairs. Sources, numbers and what is off: tr_sources.md.

    python extract.py --region tr --pbf data/raw/turkey-latest.osm.pbf
    python tr_register.py --construction data/raw/turkey-latest.osm.pbf   # needs the .pbf
    python tr_register.py --clip          # after every extract (and --construction)
    python tr_register.py --names         # the folded line names with km
    python build_model.py --region tr --register tr_register:data/raw/tr

THE LINE UNIT. TCDD numbers its lines (101 İstanbul - Demirköprü, 104 Gebze - Ankara, 109
Irmak - Zonguldak ... 205 Kayaş YHT - Sivas YHT; TCDD Taşımacılık's yearly statistics list
them with each line's train-km), but publishes no length or geometry per line. What OSM
Türkiye puts on the track is the name of the railway as tr.wikipedia and Wikidata have it:
"Ankara-Kars demiryolu", "Irmak-Zonguldak demiryolu", "Fevzipaşa-Kurtalan demiryolu",
"İzmir-Afyonkarahisar demiryolu", "Ankara - İstanbul yüksek hızlı demiryolu", "Marmaray",
"Başkentray". 98.4% of main and branch rail km carries a name (`python probe_kr_ways.py
--region tr`), every way one name, so the lines never overlap. Those names are the line unit,
as Korea's 경부선 and the UK's Cotswold Line are; Wikidata has the same items with lengths
(P2043) and OSM relation ids (P402), which are the outside check. They are bigger than TCDD's
codes (the Ankara-Kars railway is TCDD 108, 110, 115 and 117) but are what a rider reads.

What gb_register does to the UK's names is done here too, through gb_register's own functions
with Turkish settings (`adopt`, as ca_register adopts us_register): spellings of one name
folded (`fold_key`, NAME_ALIAS), a structure's name on the track ("Batıbel Tüneli", "Sakarya
Viyadüğü") read as no name, unnamed track named from its neighbours, short stray pieces
folded into the line round them, junction ends, and sections running past a line's own
stations on a parallel track dropped. Track whose name is a metro's or a tram's (railway=rail
"M12 Göztepe-Ümraniye Metro Hattı", "Sirkeci - Kazlıçeşme Raylı Sistem Hattı") stays an OSM
line; so do freight-only branches (FREIGHT).

WHICH STATIONS ARE PASSENGER STOPS. OSM Türkiye has a station node at nearly every crossing
loop TCDD has, most with no train stopping. A station is a passenger stop when TCDD
Taşımacılık sells tickets to it (its station list, data/raw/tr/tcdd_station_pairs.json, 593
entries, found among OSM's stations by name, `tcdd_names`), when an OSM passenger route stops
at it (Marmaray, Başkentray, İZBAN, Gaziray: card-fare suburban lines with no ticket list) or
when a train route's track ends at it (`route_termini`). Such a station goes on each named
line whose track passes within LIST_M of it or ends within END_M of it, except that a
high-speed line takes only YHT stations (`hs_ok`): the YHT runs beside Başkentray's and
Marmaray's tracks without stopping. Lines end at junctions as in gb_register, plus where a
line's end reaches another line over a short unnamed curve, unless a passenger station is
near the end (`junction_ends`).

AFTER EVERY EXTRACT: `--construction <pbf>` then `--clip` (the commands above). The clip also
gives each two-direction TCDD service a route_master (`pair_directions`).

The `path` argument is data/raw/tr; the OSM half is read from data/proc/tr (extract.py).
"""
import hashlib
import json
import math
import os
import pickle
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw" / "tr"
PROC = ROOT / "data" / "proc" / "tr"
REGION = "tr"

# =========================================================================== the extract clip

# Route relations in the extract that are no passenger service: lines planned or being built,
# closed decades ago, or an infrastructure relation tagged route=train. Each kept a
# railway=construction way in the extract (extract.py keeps one a route runs over).
NOT_SERVICE = {
    17991053: "Antalya-Kayseri YHD, planned",
    17989829: "Eskişehir-Antalya YHD, planned",
    17990470: "Gaziantep-Mardin YHD, planned",
    17992755: "Erzincan-Erzurum YHD, planned",
    17995025: "Erzurum-Kars YHD, planned",
    19378231: "Erzincan - Zara - Sivas Hızlı Tren Hattı, being built",
    19201418: "Sivas - Zara - Erzincan Hızlı Tren Hattı, being built",
    15715486: "Mersin-Adana-Gaziantep Hızlı Tren Hattı, being built",
    16457633: "Gebze - Halkalı Yüksek Standartlı Demiryolu, the infrastructure (Marmaray's)",
    20068906: "Çukurova Havalimanı şube demiryolu, being built",
    20068907: "çukurova havalimanı şube demiryolu, being built",
    18931414: "Samsun-Çarşamba Demiryolu, no passenger trains (TCDD 112, freight only 2024)",
    18775212: "Siirt-Kurtalan Demiryolu, being built",
    1318224: "Sütlaç - Çivril hattı, closed 1988",
    1606729: "Pythio-Edirne-Svilengrad, the pre-1971 route through Karaağaç",
    19203227: "Konya - Karaman Hızlı Tren Hattı, the infrastructure (no operator)",
    19230796: "M İncirli - Söğütlüçeşme Metro Hattı, planned",
    19235274: "T Esenler-Davutpaşa Tramvay Hattı, planned",
    19697950: "T Otogar-Stadyum 2. etap projesi (Kocaeli), planned",
    19664874: "Kocaeli M2 Kuzey Metro Hattı (Körfezray), planned",
    20186679: "Kocaeli M2 Kuzey Metro Hattı (Körfezray), planned",
    19675941: "M3 Gebze - Sabiha Gökçen, planned",
    19676169: "M4 İzmit-Gölcük, planned",
    19925415: "Ankara M5 Kızılay - İlkbahar, planned",
    18642363: "KonyaRay, its track all railway=construction (being built)",
    19675921: "M1 Darıca Sahil - Gebze OSB, its track all railway=construction",
    1308730: "Mersin-Adana Bölgesel, suspended since 2024-04-22 for the rebuilding (TCDD; "
             "the 2026 network statement has Mersin, Tarsus and Taşkent closed to passengers)",
}
# A metro, tram or light-rail route with at least this share of its track railway=construction
# is a line being built (İstanbul's M12, Konya's Adliye - Şehir Hastanesi tram): dropped. Train
# routes are listed by hand above, since the Mersin - Adana line is being rebuilt under
# traffic and OSM maps its running track as construction.
CONSTRUCTION_SHARE = 0.5
# Construction ways kept whatever runs over them: (name, west lon, east lon). Yenice - Adana -
# Ceyhan - Toprakkale - Osmaniye is being doubled under traffic (the Mersin - Adana - Gaziantep
# high-standard railway) and OSM maps much of it as railway=construction, Adana - Ceyhan
# wholly; the Toros and Erciyes Ekspresi and the Adana - İskenderun and İslahiye trains run
# there (TCDD sells tickets to Yakapınar, İncirlik, Ceyhan, Toprakkale). West of Yenice the
# line to Mersin is shut for the rebuilding (since 2024-04-22), and east of Osmaniye the same
# name is the new alignment through the Nurdağı tunnel, not yet open.
KEEP_CONSTRUCTION = [("Mersin-Adana-Gaziantep yüksek standartlı demiryolu", 35.04, 36.32)]


def boundary():
    """Türkiye's outline: OSM relation 174737 from polygons.openstreetmap.fr."""
    import shapely
    from shapely.geometry import shape
    g = json.loads((RAW / "tr_boundary.geojson").read_text(encoding="utf-8"))
    geom = shape(g["geometries"][0]) if g.get("type") == "GeometryCollection" else shape(g)
    shapely.prepare(geom)
    return geom


def construction_pass(pbf, log=print):
    """data/proc/tr/construction.json: the ids of the railway=construction ways in the .pbf,
    which extract.py keeps (retagged as the track they will be) where a route runs over them,
    and which `clip` drops again where only a NOT_SERVICE route did."""
    os.environ.setdefault("OSMIUM_POOL_THREADS", "2")
    import osmium
    out = []
    for w in osmium.FileProcessor(str(pbf), osmium.osm.WAY):
        if w.tags.get("railway") == "construction":
            out.append(w.id)
    (PROC / "construction.json").write_text(json.dumps(sorted(out)), encoding="utf-8")
    log(f"construction: {len(out)} railway=construction ways")


MASTER_BASE = 9_000_000_000   # synthetic route_master ids: this + the lower route id


def pair_directions(rels, log):
    """One line for the two directions of a TCDD service. OSM Türkiye maps each YHT and
    regional service as two route relations, "Ankara - İstanbul YHT Hattı" and "İstanbul -
    Ankara YHT Hattı", with no route_master, and build_model groups route_master-less routes
    by name, so each direction became a line of its own. A train route with no master whose
    name, its two ends swapped, is another such route's gets a route_master over both
    (id MASTER_BASE + the lower route id, so the line id is stable)."""
    in_master = {r for t, ms in rels.values() if t.get("type") == "route_master"
                 for ty, r, _ in ms if ty == "r"}

    def key(name):
        m = re.match(r"^\s*(.+?)\s*[-–]\s*(.+?)\s+(YHT Hattı|Yüksek Hızlı Tren hattı|"
                     r"Bölgesel Tren Hattı|Bölgesel)\s*$", name or "", re.I)
        return (m.group(1), m.group(2), m.group(3).lower()) if m else None
    by = {}
    for k, (t, ms) in rels.items():
        if t.get("type") == "route" and t.get("route") == "train" and k not in in_master:
            kk = key(t.get("name"))
            if kk:
                by[kk] = k
    made = 0
    for (a, b, kind), k in sorted(by.items(), key=lambda kv: kv[1]):
        o = by.get((b, a, kind))
        if o is None or o < k:
            continue
        mid = MASTER_BASE + k
        t = rels[k][0]
        rels[mid] = ({"type": "route_master", "route_master": "train",
                      "name": t.get("name"), "operator": t.get("operator", ""),
                      "network": t.get("network", "")},
                     [("r", k, ""), ("r", o, "")])
        made += 1
    log(f"  {made} two-direction services given a route_master")


def clip(log=print):
    """Rewrite data/proc/tr: out what lies abroad (Geofabrik's cut takes in a little of
    Greece, Bulgaria, Georgia and Iran), the NOT_SERVICE route relations, and the construction
    ways no remaining route runs over. A way goes if at least half its nodes are outside
    Türkiye, a stop if it is, a relation if no member is left."""
    import shapely
    with open(PROC / "ways.pkl", "rb") as f:
        ways = pickle.load(f)
    with open(PROC / "rels.pkl", "rb") as f:
        rels = pickle.load(f)
    with open(PROC / "stops.pkl", "rb") as f:
        stops = pickle.load(f)
    with open(PROC / "infra.pkl", "rb") as f:
        infra = pickle.load(f)
    c = np.load(PROC / "coords.npz")
    cid, cx, cy = c["id"], c["x"], c["y"]
    tr = boundary()
    out = ~shapely.contains_xy(tr, cx / 1e7, cy / 1e7)
    outside = set(cid[out].tolist())
    known = set(cid.tolist())
    keep_w, cut = {}, Counter()
    for wid, (tags, nodes) in ways.items():
        ns = [int(n) for n in nodes if int(n) in known]
        if ns and 2 * sum(n in outside for n in ns) >= len(ns):
            cut[tags.get("name") or "(unnamed)"] += 1
        else:
            keep_w[wid] = (tags, nodes)
    keep_s = {k: v for k, v in stops.items() if k not in outside
              and shapely.contains_xy(tr, v[1], v[2])}
    dropped_routes = {k for k in rels if k in NOT_SERVICE}
    for k in sorted(dropped_routes):
        log(f"  not a service: {k} {NOT_SERVICE[k]}")
    cons_p = PROC / "construction.json"
    cons = set(json.loads(cons_p.read_text(encoding="utf-8"))) if cons_p.exists() else set()
    if cons:
        wkm = {}
        for k, (t, ms) in rels.items():
            if t.get("type") != "route" or t.get("route") not in ("subway", "tram",
                                                                  "light_rail", "monorail"):
                continue
            tot = con = 0.0
            for ty, r, _ in ms:
                if ty != "w" or r not in keep_w:
                    continue
                if r not in wkm:
                    nodes = np.asarray(keep_w[r][1], dtype=np.int64)
                    pos = np.searchsorted(cid, nodes)
                    np.clip(pos, 0, cid.size - 1, out=pos)
                    pos = pos[cid[pos] == nodes]
                    x, y = cx[pos] / 1e7, cy[pos] / 1e7
                    wkm[r] = float(np.hypot(np.diff(x) * 0.78, np.diff(y)).sum())
                tot += wkm[r]
                if r in cons:
                    con += wkm[r]
            if tot and con >= CONSTRUCTION_SHARE * tot:
                dropped_routes.add(k)
                log(f"  being built: {k} {t.get('name')} ({con / tot:.0%} of its track)")
    rels = {k: v for k, v in rels.items() if k not in dropped_routes}
    # construction ways only a dropped route ran over
    if cons:
        on_route = {r for k, (t, ms) in rels.items() if t.get("type") == "route"
                    for ty, r, _ in ms if ty == "w"}
        def whitelisted(w):
            t, nodes = keep_w[w]
            for nm, w0, e0 in KEEP_CONSTRUCTION:
                if t.get("name") == nm:
                    nodes = np.asarray(nodes, dtype=np.int64)
                    pos = np.searchsorted(cid, nodes)
                    np.clip(pos, 0, cid.size - 1, out=pos)
                    pos = pos[cid[pos] == nodes]
                    if pos.size and w0 <= float(cx[pos].mean()) / 1e7 <= e0:
                        return True
            return False
        gone_c = [w for w in keep_w if w in cons and w not in on_route and not whitelisted(w)]
        for w in gone_c:
            cut["(construction) " + (keep_w[w][0].get("name") or "")] += 1
            del keep_w[w]
        log(f"  {len(gone_c)} construction ways dropped, "
            f"{sum(1 for w in keep_w if w in cons)} kept under a passenger route")
    else:
        log("  no construction.json: run --construction on the .pbf first")
    pair_directions(rels, log)
    kept = {("w", k) for k in keep_w} | {("n", k) for k in keep_s}
    routes = {k for k, (tags, members) in rels.items()
              if tags.get("type") == "route" and any((t, r) in kept for t, r, _ in members)}
    keep_r = {k: v for k, v in rels.items()
              if k in routes or any(t == "r" and r in routes for t, r, _ in v[1])}
    gone = set(ways) - set(keep_w)
    keep_i = {k: v for k, v in infra.items()
              if not (any(t == "w" and r in gone for t, r, _ in v[1])
                      and not any(t == "w" and r in keep_w for t, r, _ in v[1]))}
    for name, v in sorted(cut.items(), key=lambda kv: -kv[1])[:40]:
        log(f"  cut {v:4d}  {name}")
    log(f"TR clip: kept {len(keep_w)}/{len(ways)} ways, {len(keep_s)}/{len(stops)} stops, "
        f"{len(keep_r)}/{len(rels)} relations, {len(keep_i)}/{len(infra)} infra relations")
    for fn, obj in (("ways.pkl", keep_w), ("rels.pkl", keep_r), ("stops.pkl", keep_s),
                    ("infra.pkl", keep_i)):
        tmp = PROC / (fn + ".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(obj, f, protocol=4)
        os.replace(tmp, PROC / fn)


# =========================================================================== the names

def tr_lower(s):
    """Turkish lower case: İ -> i, I -> ı, then the rest."""
    return (s or "").replace("İ", "i").replace("I", "ı").lower()


# Track names that are one line under another spelling (exact, before folding). The folded
# spellings (dashes, case, "demiryolu"/"hattı") need no entry.
NAME_ALIAS = {
    # Gebze - Köseköy (lon 29.41-30.01): the old İstanbul - Ankara line rebuilt for 160 km/h,
    # which every train shares (TCDD's network statement, Ek-3.3 row 7, "Gebze - Köseköy HT").
    # OSM names it with the İstanbul-first spellings of the high-speed line, which are used
    # nowhere else; the new-built high-speed line starts at Köseköy and Doğançay.
    "İstanbul - Ankara yüksek hızlı demiryolu": "İstanbul - Ankara demiryolu",
    "İstanbul-Ankara yüksek hızlı demiryolu": "İstanbul - Ankara demiryolu",
    "İstanbul - Ankara yüksek hızlıdemiryolu": "İstanbul - Ankara demiryolu",
    "İstanbul - Ankara yüksek hızlı  demiryolu": "İstanbul - Ankara demiryolu",
    "İstanbul - Ankara Hızlı Tren hatti": "İstanbul - Ankara demiryolu",
    "Ankara – İstanbul yüksek hızlı demiryolu": "Ankara - İstanbul yüksek hızlı demiryolu",
    # the shared approach into Sivas station from the west (lon 36.92-37.01), where the
    # high-speed line joins the Ankara - Kars railway
    "Sivas şehiriçi yaklaşım hattı": "Ankara-Kars demiryolu",
    # the high-speed line's own curve onto it, which every YHT to Sivas runs over
    "Yapı YHT Kavşağı": "Ankara - Sivas yüksek hızlı demiryolu",
    "İzmir – Eğirdir demiryolu": "İzmir-Alsancak-Eğirdir demiryolu",
    "İzmir-Eğirdir demiryolu": "İzmir-Alsancak-Eğirdir demiryolu",
    "Bağdat Demiryolu / Rêya hesinî li Bexda / سكة حديد بغداد": "Bağdat Demiryolu",
    "Mersin-Adana-Gaziantep Hızlı Tren Hattı": "Mersin-Adana-Gaziantep yüksek standartlı demiryolu",
    "ბაქო-თბილისი-ყარსის რკინიგზა": "Bakü-Tiflis-Kars Demiryolu Hattı",
    # stray spellings, a few hundred metres each
    "Ankara - Sivas Yüksek Hızlı Tren Hattı İnşaatı": "Ankara - Sivas yüksek hızlı demiryolu",
    "Çetinkaya-Malatya demiryolu": "Malatya-Çetinkaya demiryolu",
    "Konya – Ulukışla – Yenice yüksek standartlı demiryolu":
        "Konya – Ulukışla yüksek standartlı demiryolu",
    "Kütahya-Seyitören şube demiryolu": "Kütahya-Seyitömer şube demiryolu",
    "Afyon - Konya tren hattı": "Eskişehir-Konya demiryolu",
    "Bilecik - Bursa - Bandırma yüksek hızlı demiryolu":
        "Osmaneli – Bursa – Bandırma hızlı tren hattı",
    "Ödemiş Demiryolu": "Torbalı-Ödemiş demiryolu",
    "İzmir-Aydın Demiryolu": "İzmir-Alsancak-Eğirdir demiryolu",
    "Kasaba Demiryolu": "İzmir-Afyonkarahisar demiryolu",
    # a station's or a bridge's name on the track: no name
    "Ankara YHT": "",
    "*": "",
    "Karkamış Köprüsü جسر جرابلس": "",
    # Wikidata's labels, for the English names (wikidata_names)
    "İstanbul-Pythio demiryolu": "İstanbul – Pythion demiryolu",
    "Ulukışla-Boğazköprü demiryolu": "Boğazköprü-Ulukışla demiryolu",
}

# Words that say "railway" or "line" and nothing about which: folded away.
TRACK_WORDS = re.compile(r"\b(?:demiryolu|demiryolları|demir yolu|hattı|hatti|hat|tren|treni|"
                         r"şube|konvansiyonel)\b")

# A structure's or a yard's name on the track, read as no name (the way then takes its
# neighbours' line): "Batıbel Tüneli", "Sakarya Viyadüğü", "Çerikli viyadüğü", "Yapı YHT
# Kavşağı", "Sincan OSB Köprülü Kavşağı".
JUNK_NAME = re.compile(r"\b(?:Tüneli|Tünel|Viyadüğü|Viyadük|Köprüsü|Köprü|Kavşağı|Kavşak|"
                       r"İstasyonu|Garı|Gar|Makas|Makası|Sayding|Saydingi|Depo|Deposu|Yolu|"
                       r"Yolları|üçgen|Üçgen|Müselles|turning loop|Junction|Bağlantı Hattı|"
                       r"bağlantı hattı|bağlantısı|bağlantı kolu)\s*$", re.IGNORECASE)

# Rail-tagged track that is a metro's, a tram's or a light railway's: stays an OSM line.
# (gb_register calls .match, so the pattern carries its own leading ".*".)
METRO_ON_RAIL = re.compile(r"^M\d+\b|^T\d+\b|.*(?:\bMetro|\bTramvay|Raylı Sistem|\bHRS\b|"
                           r"Hafif Raylı)", re.IGNORECASE)

# Freight-only lines: no passenger train in TCDD Taşımacılık's 2024 train-km by line (its
# statistics, table 2.3.3; tcddt_2024_istatistik.pdf) and none in OSM. Left out of the
# register; their track stays drawn.
FREIGHT = {
    "Muratlı-Tekirdağ şube demiryolu",           # TCDD 150, freight only
    "Çobanisa-Kemalpaşa Şube Demiryolu",         # 151
    "Kayseri Kuzey Geçiş Varyantı",              # 154, freight bypass of Kayseri
    "Bakü-Tiflis-Kars Demiryolu Hattı",          # 158 Km 6+125 - Ahılkelek, freight only
    "Akçagöze-Başpınar varyantı",                # 161
    "Bozdemir-Mazıdağı Şube Demiryolu",          # 162
    "Kütahya-Seyitömer şube demiryolu",          # 135
    "Tavşanlı-Tunçbilek şube demiryolu",         # 136
    "Samsun-Azot şube demiryolu",
    "Hanlı-Bostankaya varyantı",                 # 114: 135 passenger train-km in 2024
    "Boğazköprü - Gömeç hattı (Kayseri bypass)",
}

# Lines TCDD's statistics show no passenger train on in 2024 although their stations are on
# its ticket list (international): built, greyed as not running (`suspended`).
SUSPENDED = {
    "Van-Sufiyan demiryolu",                     # TCDD 122 Van - Kapıköy: freight only 2024
}


def fold_key(name):
    """One key for the spellings of one line: "İstanbul - Ankara demiryolu", "İstanbul-Ankara
    demiryolu" and "İstanbul-Ankara Demiryolu"; "Alayunt - Balıkesir Demiryolu" and
    "Alayunt-Balıkesir demiryolu". "Muratlı-Tekirdağ şube demiryolu" is "muratlı tekirdağ"."""
    k = tr_lower(unicodedata.normalize("NFC", name))
    k = re.sub(r"[-‐-―−~/,.'’()]", " ", k)
    k = TRACK_WORDS.sub(" ", k)
    return " ".join(k.split())


def line_id(name):
    h = hashlib.blake2b(f"tr|{name}".encode("utf-8"), digest_size=5)
    return "t" + h.hexdigest()


# =========================================================================== stations

LIST_M = 400          # a passenger station goes on each named line whose track is this close
END_M = 2000          # ... and on a line whose track ends this close to it (kr's MATCH_M too)
CONNECT_KM = 5.0      # a line's dead end this far over unnamed track from another line is a junction
STATION_END_M = 1500  # a line end this close to a passenger station ends there, not at a junction
ANCHOR_END_M = 1000   # ... and is a stop node of it even on another line's track, this close
TCDD_NEAR_M = 6000    # a TCDD ticket station's own point may be this far from OSM's node
NAME_WORDS = re.compile(r"\b(?:gar|garı|istasyonu|istasyon|tren|durağı|durak|d|gari)\b")

# TCDD ticket names that are another OSM name (both by station_key).
TCDD_ALIAS = {
    "afyon a çetinkaya": "ali çetinkaya",
    # the YHT's stops on the Gebze - Köseköy line are the old stations, mapped without "YHT"
    "izmit yht": "izmit",
    "hereke yht": "hereke",
    "derince yht": "derince",
    "diliskelesi yht": "diliskelesi",
    "yarımca yht": "yarımca",
    "kadınhan": "kadınhanı",
    "kemaliye çaltı": "kemaliyeçaltı",
}
# Not passenger stations although TCDD lists them: junctions, sidings and points.
TCDD_NOT = re.compile(r"\bKM\b|KM\.|MAKAS|SAYDİNG|MÜSELLES|\bHT\b|\bMR\b|S HAT|VARYANT")


def station_key(name):
    """A station name for matching TCDD's list to OSM: Turkish lower case, no "Garı",
    "İstasyonu", "Durağı", dots and dashes as spaces."""
    k = tr_lower(unicodedata.normalize("NFC", name or ""))
    k = re.sub(r"[-‐-―−/,.'’]", " ", k)
    k = NAME_WORDS.sub(" ", k)
    return " ".join(k.split())


def tcdd_names(log):
    """{station_key: [(lon, lat) or None]} for TCDD Taşımacılık's ticket stations, each name
    also under the part in brackets ("İSTANBUL(HALKALI)" is halkalı, "İZMİR (BASMANE)"
    basmane) and the part before it ("SELÇUKLU YHT (KONYA)" is selçuklu yht)."""
    p = RAW / "tcdd_station_pairs.json"
    if not p.exists():
        log("TR: no tcdd_station_pairs.json; every OSM station counts")
        return None
    out = defaultdict(list)
    n = 0
    for s in json.loads(p.read_text(encoding="utf-8")):
        if not s.get("domestic") or TCDD_NOT.search(s.get("name") or ""):
            continue
        lon, lat = s.get("longitude") or 0, s.get("latitude") or 0
        pt = (lon, lat) if 25 < lon < 45 and 35 < lat < 43 else None
        nm = s["name"]
        keys = {station_key(nm)}
        m = re.match(r"^(.*?)\s*\((.*)\)\s*$", nm)
        if m:
            keys |= {station_key(m.group(1)), station_key(m.group(2))}
        for k in keys:
            k = TCDD_ALIAS.get(k, k)
            if k:
                out[k].append(pt)
        n += 1
    log(f"TR: TCDD Taşımacılık's ticket list, {n} stations ({len(out)} name keys)")
    return out


TERMINUS_M = 400      # a station this close to the end of a route's track is its terminus


def route_termini(st, rail, log):
    """Stations at the dead ends of each OSM train route's track."""
    import build_model as bm
    ways, coords = _S["ways"], _S["coords"]
    ids = [sid for sid in st if rail[sid]]
    if not ids:
        return set()
    sx = np.array([st[s]["lon"] for s in ids])
    sy = np.array([st[s]["lat"] for s in ids])
    out = set()
    for rid, (tags, members) in _S["rels"].items():
        if tags.get("type") != "route" or tags.get("route") != "train":
            continue
        deg = Counter()
        for ty, r, _ in members:
            if ty == "w" and r in ways:
                ns = ways[r][1]
                deg[int(ns[0])] += 1
                deg[int(ns[-1])] += 1
                for n in ns[1:-1]:
                    deg[int(n)] += 2
        for n, k in deg.items():
            if k != 1:
                continue
            p = coords.get(n)
            if p is None:
                continue
            d = np.hypot((sx - p[0]) * math.cos(math.radians(p[1])) * 111320,
                         (sy - p[1]) * 110570)
            j = int(np.argmin(d))
            if d[j] <= TERMINUS_M:
                out.add(ids[j])
    return out


def passenger_filter(st, node_st, by_key, by_base, stops, log):
    """Keep, of kr_register's stations, the rail ones that are passenger stops (the module
    docstring): on TCDD's ticket list, or a stop of an OSM passenger route. Metro, tram and
    light-rail stations are kept as they are (they never go on register track)."""
    import build_model as bm
    tc = tcdd_names(log)
    if tc is None:
        return st, node_st, by_key, by_base
    route_st = set()
    for rid, (tags, members) in _S["rels"].items():
        if tags.get("type") != "route" or tags.get("route") not in bm.ROUTE_KINDS:
            continue
        for n in bm.stop_members(members):
            if n in node_st:
                route_st.add(node_st[n])
    # TCDD's own points are often missing (0, 0) or wrong (Alp, in Erzincan, is put at
    # 30.2 E), so a name with one OSM rail station is that station; where OSM has several of
    # a name (Yeniköy), the one within TCDD_NEAR_M of TCDD's point, else the nearest to it,
    # else all of them.
    rail = {}
    for sid, s in st.items():
        tags = stops.get(sid, ({}, 0, 0))[0]
        mode_other = any(tags.get(m) == "yes" for m in ("subway", "tram", "light_rail",
                                                        "monorail", "funicular"))
        rail[sid] = not ((mode_other and tags.get("train") != "yes")
                         or tags.get("railway") == "tram_stop")
    cands = defaultdict(list)
    for sid, s in st.items():
        if rail[sid]:
            cands[station_key(s["name"])].append(sid)
    on_list = set()
    for k, pts in tc.items():
        cs = cands.get(k, [])
        if len(cs) <= 1:
            on_list.update(cs)
            continue
        good = [p for p in pts if p is not None]
        if not good:
            on_list.update(cs)
            continue
        dist = {c: min(kr.dist_m(st[c]["lon"], st[c]["lat"], *p) for p in good) for c in cs}
        near = [c for c in cs if dist[c] <= TCDD_NEAR_M]
        on_list.update(near or [min(cs, key=dist.get)])
    # (The 2026 decree's public-service lines are NOT taken as evidence: it names Adana -
    # Mersin and Gaziantep - Karkamış, which are shut, Mersin for rebuilding and Karkamış
    # closed to passengers in the 2026 network statement.)
    n_pso = 0
    # a station at either end of an OSM passenger route's track: its terminus (Mersin, for
    # the Mersin - Adana regional train, which lists no stops)
    term = route_termini(st, rail, log)
    n_term = len(term - on_list)
    on_list |= term
    keep, n_tc, n_rt, n_gone = {}, 0, 0, 0
    for sid, s in st.items():
        if not rail[sid]:
            keep[sid] = s
            continue
        if sid in on_list:
            keep[sid] = s
            n_tc += 1
        elif sid in route_st:
            keep[sid] = s
            n_rt += 1
        else:
            n_gone += 1
    node_st = {n: s for n, s in node_st.items() if s in keep}
    by_key = defaultdict(list, {k: [s for s in v if s in keep] for k, v in by_key.items()})
    by_base = defaultdict(list, {k: [s for s in v if s in keep] for k, v in by_base.items()})
    _S["tcdd"] = tc
    _S["on_list"] = on_list
    log(f"TR: passenger stations: {n_tc} on TCDD's ticket list or a route's terminus "
        f"({n_term} by a terminus alone), {n_rt} more as a route's stop; {n_gone} OSM rail "
        f"stations left out as no passenger stop")
    found = {station_key(st[s]["name"]) for s in on_list}
    missing = sorted(k for k in tc if k not in found and "(" not in k)
    log(f"TR: {len(missing)} TCDD name keys found no OSM station: {', '.join(missing[:120])}")
    return keep, node_st, by_key, by_base


def is_highspeed_line(name):
    n = tr_lower(name)
    return "yüksek hızlı" in n or "hızlı tren" in n


# Stations the YHT stops at that carry no "YHT" in their name (station_key).
YHT_SHARED = {"ankara", "eskişehir", "konya", "sivas", "karaman", "halkalı", "bakırköy",
              "söğütlüçeşme", "bostancı", "pendik", "gebze", "arifiye", "eryaman yht",
              "kayaş"}


def hs_ok(name):
    k = station_key(name)
    return "yht" in k.split() or k in YHT_SHARED or "hızlı" in k


def tr_lists(path, log):
    """gb_register's lists from OSM's routes, plus each passenger station on every named line
    whose track passes within LIST_M of it (a high-speed line only YHT stations)."""
    lists = gb.route_lists(log)
    ways, coords, st = _S["ways"], _S["coords"], _S["st"]
    by_line = defaultdict(list)
    for w, (t, _n) in ways.items():
        ln = gb.register_name(t)
        if ln:
            by_line[ln].append(w)
    segs = {}
    for ln, ws in by_line.items():
        xs, ys = [], []
        for w in ws:
            pos, ok = coords.many(np.asarray(ways[w][1], dtype=np.int64))
            pos = pos[ok]
            if pos.size < 2:
                continue
            x, y = coords.x[pos] / 1e7, coords.y[pos] / 1e7
            xs.append(np.column_stack([x[:-1], y[:-1], x[1:], y[1:]]))
        if xs:
            segs[ln] = np.vstack(xs)
    added = 0
    on_list = _S.get("on_list", set(st))
    for sid, s in st.items():
        # only the stations TCDD sells tickets to (or the decree names, or a route ends at):
        # a suburban route's own stops go on the lines its own track runs on (route_lists),
        # not on the main line beside it (Marmaray's, Başkentray's, Gaziray's stations), and
        # trams never
        if sid not in on_list:
            continue
        lon, lat = s["lon"], s["lat"]
        kx = math.cos(math.radians(lat)) * 111320
        for ln, a in segs.items():
            box = ((a[:, 0] - lon) * kx) ** 2 + ((a[:, 1] - lat) * 110570) ** 2
            if box.min() > (LIST_M + 3000) ** 2:
                # cheap reject: no vertex within a few km
                continue
            ax, ay = (a[:, 0] - lon) * kx, (a[:, 1] - lat) * 110570
            bx, by = (a[:, 2] - lon) * kx, (a[:, 3] - lat) * 110570
            dx, dy = bx - ax, by - ay
            L2 = np.maximum(dx * dx + dy * dy, 1e-9)
            t = np.clip(-(ax * dx + ay * dy) / L2, 0, 1)
            if np.min(np.hypot(ax + t * dx, ay + t * dy)) > LIST_M:
                continue
            if is_highspeed_line(ln) and not hs_ok(s["name"]):
                continue
            if s["name"] not in lists[ln]:
                lists[ln].add(s["name"])
                added += 1
    # A line's own terminus: OSM often ends a line's named track in the station throat, and a
    # big station's node can sit in its building well off the tracks (Karkamış, 600 m).
    ends = defaultdict(list)
    for ln, ws in by_line.items():
        inc = Counter()
        for w in ws:
            nl = np.asarray(ways[w][1]).tolist()
            inc[nl[0]] += 1
            inc[nl[-1]] += 1
            for x in nl[1:-1]:
                inc[x] += 2
        for n, k in inc.items():
            if k == 1:
                p = coords.get(n)
                if p is not None:
                    ends[ln].append(p)
    n_end = 0
    for sid in on_list:
        s = st.get(sid)
        if s is None:
            continue
        for ln, pts in ends.items():
            if s["name"] in lists[ln]:
                continue
            if is_highspeed_line(ln) and not hs_ok(s["name"]):
                continue
            if any(kr.dist_m(s["lon"], s["lat"], x, y) <= END_M for x, y in pts):
                lists[ln].add(s["name"])
                n_end += 1
    log(f"TR: {added} station-line pairs added by nearness ({LIST_M} m), {n_end} more as a "
        f"line's end ({END_M} m)")
    return ({k: [sorted(v)] for k, v in lists.items()}, defaultdict(list), {})


# =========================================================================== adoption

import gb_register as gb       # noqa: E402
import kr_register as kr       # noqa: E402

_S = gb._S


def base_name(tags):
    """gb_register.base_name with Türkiye's freight lines left out."""
    n = _gb_base_name(tags)
    return "" if n in FREIGHT else n


FILL_MAX_KM = 25.0    # an unnamed stretch at most this long takes the one name round it


def fill_runs(ways, coords, log):
    """Unnamed runs of track inside one named line. OSM Türkiye names the open line but often
    not the tracks through a station, which are split at every switch: a run of two or more
    unnamed ways, where gb_register.propagate fills a single way only (its neighbours at both
    ends must be named). The Alayunt-Balıkesir railway came out in 14 pieces. Each connected
    run of unnamed main track (no service tag, or crossover) under FILL_MAX_KM whose ends touch
    exactly one line's track takes that line's name; a run touching two lines (a junction
    station) is left alone."""
    cand = []
    for w, (t, _n) in ways.items():
        if t.get("railway") not in gb.TRACK_KIND or t.get("usage") in gb.NOT_PASSENGER:
            continue
        if t.get("service") not in (None, "crossover"):
            continue
        if gb.register_name(t) or gb.tidy(t.get("name")):
            continue
        cand.append(w)
    cset = set(cand)
    parent = {w: w for w in cand}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    at = defaultdict(list)
    for w, (t, nodes) in ways.items():
        for n in np.asarray(nodes).tolist():
            at[n].append(w)
    first = {}
    for w in cand:
        for n in np.asarray(ways[w][1]).tolist():
            if n in first:
                parent[find(w)] = find(first[n])
            else:
                first[n] = w
    runs = defaultdict(list)
    for w in cand:
        runs[find(w)].append(w)
    km = gb.way_km({w: ways[w] for w in cand}, coords)
    byobj = _S.setdefault("byobj", {})
    n_runs = n_ways = 0
    filled_km = 0.0
    for ws in runs.values():
        k = sum(km[w] for w in ws)
        if k > FILL_MAX_KM:
            continue
        names = set()
        for w in ws:
            for n in np.asarray(ways[w][1]).tolist():
                for o in at.get(n, ()):
                    if o not in cset:
                        nm = gb.register_name(ways[o][0])
                        if nm:
                            names.add(nm)
        if len(names) != 1:
            continue
        nm = next(iter(names))
        for w in ws:
            byobj[id(ways[w][0])] = nm
        n_runs += 1
        n_ways += len(ws)
        filled_km += k
    log(f"TR: {n_runs} unnamed runs of track named from the one line round them ({n_ways} "
        f"ways, {filled_km:,.0f} km of track)")


def junction_ends(log):
    """gb_register's junction ends, plus a line's dead end that reaches another line's track
    over at most CONNECT_KM of unnamed track: the high-speed line's ends at Doğançay and
    near Sivas meet the old line through connecting curves OSM leaves unnamed (or names
    "Konvansiyonel ile yüksek hızlı demiryolu bağlantı hattı", read as no name), and without
    a junction there the stretch from the line's last station to it belonged to no line."""
    import heapq
    out = _gb_junction_ends(log)
    ways, coords = _S["ways"], _S["coords"]
    inc = defaultdict(Counter)
    on = defaultdict(set)
    adj = defaultdict(list)
    for w, (t, nodes) in ways.items():
        nl = np.asarray(nodes).tolist()
        ln = gb.register_name(t)
        if ln:
            c = inc[ln]
            c[nl[0]] += 1
            c[nl[-1]] += 1
            for n in nl[1:-1]:
                c[n] += 2
            for n in nl:
                on[n].add(ln)
            continue
        if t.get("railway") not in gb.TRACK_KIND or t.get("usage") in ("industrial", "military"):
            continue
        pos, ok = coords.many(np.asarray(nodes, dtype=np.int64))
        prev = None
        for n, p, good in zip(nl, pos.tolist(), ok.tolist()):
            if not good:
                prev = None
                continue
            xy = (coords.x[p] / 1e7, coords.y[p] / 1e7)
            if prev is not None:
                d = kr.dist_m(*prev[1], *xy) / 1000
                adj[prev[0]].append((n, d))
                adj[n].append((prev[0], d))
            prev = (n, xy)
    added = 0
    # gb_register.build_stations stores st before it asks for the junction ends
    pst_ids = [s for s in _S.get("on_list", ()) if s in _S.get("st", {})]
    pst = [_S["st"][s] for s in pst_ids]
    px = np.array([s["lon"] for s in pst]) if pst else np.zeros(0)
    py = np.array([s["lat"] for s in pst]) if pst else np.zeros(0)
    for ln, c in inc.items():
        for n, k in c.items():
            if k != 1 or n in out:
                continue
            dist, heap, hit = {n: 0.0}, [(0.0, n)], False
            while heap:
                d, u = heapq.heappop(heap)
                if d > CONNECT_KM:
                    break
                if u != n and on.get(u, set()) - {ln}:
                    hit = True
                    break
                if d > dist.get(u, INF):
                    continue
                for v, w in adj.get(u, ()):
                    nd = d + w
                    if nd < dist.get(v, INF):
                        dist[v] = nd
                        heapq.heappush(heap, (nd, v))
            if hit:
                out[n].add(ln)
                added += 1
    # A line whose track ends near a passenger station starts there (tr_lists puts the
    # station on it, END_M): the Kars - Akyaka line leaves the Ankara - Kars line 0.8 km east
    # of Kars station, and a junction there made Kars - Duraklı a junction-ended section that
    # no OSM route runs over (the regional train has no relation), so it was dropped.
    # Where the end is not on another line's track (a gap in the named track, the line going
    # on beyond it: the İzmir - Eğirdir railway through Goncalı), the station is also given
    # that end node as a stop node, so both pieces of the line meet at it.
    removed = 0
    node_st = _S["node_st"]
    for n in list(out):
        p = coords.get(n)
        if p is None or not pst:
            continue
        d = np.hypot((px - p[0]) * math.cos(math.radians(p[1])) * 111320,
                     (py - p[1]) * 110570)
        near = sorted(np.nonzero(d <= STATION_END_M)[0].tolist(), key=lambda i: d[i])
        for ln in list(out[n]):
            ok = [i for i in near if not is_highspeed_line(ln) or hs_ok(pst[i]["name"])]
            if ok:
                out[n].discard(ln)
                removed += 1
                if n not in node_st and (not (on.get(n, set()) - {ln})
                                         or d[ok[0]] <= ANCHOR_END_M):
                    node_st[n] = pst_ids[ok[0]]
        if not out[n]:
            del out[n]
    log(f"TR: {added} more line ends that reach another line over unnamed track "
        f"(<= {CONNECT_KM} km) taken as junctions; {removed} line ends within "
        f"{STATION_END_M} m of a passenger station left to end there instead")
    return out


INF = float("inf")


def load_osm(log):
    out = _gb_load_osm(log)
    fill_runs(_S["ways"], _S["coords"], log)
    return out


def border_points(log):
    try:
        import borders
        return [p for p in borders.load(canonical_only=True) if REGION in p["countries"]]
    except Exception as e:                          # noqa: BLE001
        log(f"TR: no border points ({e})")
        return []


def adopt():
    """Point gb_register's (and through it kr_register's) settings at Türkiye's."""
    global _gb_base_name, _gb_load_osm, _gb_junction_ends
    if gb._orig:
        return
    _gb_load_osm = gb.load_osm
    gb.load_osm = load_osm
    _gb_junction_ends = gb.junction_ends
    gb.junction_ends = junction_ends
    gb.REGION = REGION
    gb.RAW = RAW
    gb.NAME_ALIAS = NAME_ALIAS
    gb.STATION_ALIAS = {}
    gb.METRO_ON_RAIL = METRO_ON_RAIL
    gb.JUNK_NAME = JUNK_NAME
    gb.UPDOWN = re.compile(r"\s*\((?:\d\.?\s*(?:hat|anahat)|gidiş|dönüş)[^)]*\)\s*$", re.I)
    gb.TRACK_WORDS = TRACK_WORDS
    gb.NOREF_SHARE = 0.0           # Türkiye's track has no ELR-like refs: nothing is "no ELR"
    gb.KEEP_NOREF = set()
    gb.fold_key = fold_key
    gb.line_id = line_id
    gb.border_points = border_points
    _gb_base_name = gb.base_name
    gb.base_name = base_name
    gb.load_lists = tr_lists
    gb.adopt()
    kr.MATCH_M = END_M
    orig = gb._orig["build_stations"]

    def build_stations(stops, log):
        st, node_st, by_key, by_base = orig(stops, log)
        return passenger_filter(st, node_st, by_key, by_base, stops, log)
    gb._orig["build_stations"] = build_stations
    kr.load_lists = tr_lists


_gb_base_name = None
_gb_load_osm = None
_gb_junction_ends = None


def wikidata_names():
    """{fold_key: (English label, length km)} from data/raw/tr/wd_lines.json."""
    p = RAW / "wd_lines.json"
    out = {}
    if not p.exists():
        return out
    for r in json.loads(p.read_text(encoding="utf-8")):
        lab = r.get("trlab") or ""
        if not lab:
            continue
        k = fold_key(NAME_ALIAS.get(lab, lab))
        en = r.get("enlab") or ""
        out.setdefault(k, en)
    return out


def build(path, log):
    adopt()
    lines, stations, geoms = gb.build(path, log)

    def r(sid):
        if sid.startswith("gj"):
            return "tj" + sid[2:]
        if sid.startswith("g") and sid[1:].lstrip("-").isdigit():
            return "t" + sid[1:]
        return sid
    en = wikidata_names()
    out_st = {}
    for sid, s in stations.items():
        nid = r(sid)
        s["id"] = nid
        out_st[nid] = s
    out_geoms = {}
    n_susp = 0
    for l in lines:
        l["src"] = "tr"
        l["sections"] = [[r(a), r(b), *rest] for a, b, *rest in l["sections"]]
        l["display"] = [r(x) for x in l["display"]]
        if isinstance(l.get("highspeed_sections"), dict):
            l["highspeed_sections"] = {"|".join(r(x) for x in k.split("|")): v
                                       for k, v in l["highspeed_sections"].items()}
        out_geoms[l["id"]] = {"|".join(r(x) for x in k.split("|")): v
                              for k, v in geoms[l["id"]].items()}
        l["name_en"] = l.get("name_en") or en.get(fold_key(l["name"]), "")
        if not l.get("operator"):
            l["operator"] = "TCDD"
        if l["name"] in SUSPENDED:
            l["suspended"] = True
            n_susp += 1
    log(f"TR: {len(lines)} register lines, {sum(l['km'] for l in lines):,.0f} km, "
        f"{len(out_st)} stations; {n_susp} lines suspended")
    # gb_register.build's section ends, under Türkiye's ids, for split_pieces
    _S["ends"] = [(r(sid), nm, x, y) for sid, nm, x, y in _S.get("ends", ())]
    return lines, out_st, out_geoms


# =========================================================================== lines in pieces

# Lines in pieces (tr_sources.md "Lines in pieces"), through pieces.py with gb_register's
# track graph under Türkiye's settings: a gap is bridged over the track between the pieces
# where trains run across, the rest is one line per piece. SLACK 2.0 rather than the UK's 1.5:
# the YHT from Pamukova runs north to Doğançay and onto the old line, then back west through
# Arifiye to its own track at Sapanca, 42.8 km for 22.8 km crow-fly. OWN_COST: the line's own
# named track costs half, so a gap that is only a short joint named for the other line (the
# old line's own track Köseköy - Sapanca stops 0.2 km short of the rest, where the YHT's
# junction is) is bridged over the line's own track, not the YHT's beside it. `dense`
# (pieces.Rules): a station on straight track with no OSM vertex near it (Pamukova YHT) still
# joins the track graph. KEEP_WHOLE: a gap that is track OSM does not have, on a line trains
# run through, stays one line in pieces until the track is mapped.
SLACK = 2.0
OWN_COST = 0.5
KEEP_WHOLE = {
    # Yenice - Şehitlik, Şakirpaşa - Adana, Ceyhan - Osmaniye: OSM maps the rebuilt line between
    # as railway=construction pieces that do not join, which the extract leaves out (Open,
    # "Adana - Ceyhan"); the Toros and Erciyes Ekspresi and Adana - İskenderun run through
    "Mersin-Adana-Gaziantep yüksek standartlı demiryolu",
}
# {line id: [the ids of the pieces split off it]}, filled by split_pieces; build_model writes it
# into aliases.json as `pieces`.
LINE_PIECES = {}


def rules():
    import pieces
    return pieces.Rules(tag="TR", id_prefix="t", lat=39.0, slack=SLACK, own_cost=OWN_COST,
                        keep_whole=KEEP_WHOLE, piece_name=pieces.english_piece_name,
                        dense=True, cut_ok=bridge_stop_ok)


def bridge_stop_ok(line, station):
    """A bridge over another line's track stops at that line's stations, but a high-speed line
    only at YHT stations (`hs_ok`, as tr_lists places them): the YHT runs through Karaköy and
    the old Sapanca station without stopping, and stops at Arifiye."""
    return not is_highspeed_line(line["name"]) or hs_ok(station["name"])


def split_pieces(lines, stations, geoms, reg_ways, state, log):
    """The build_model hook: pieces.split_pieces with Türkiye's rules over gb_register's
    track graph (gb_register's settings are Türkiye's here, `adopt`)."""
    import pieces
    r = rules()
    pieces.split_pieces(lines, stations, geoms, reg_ways, state, log, r, LINE_PIECES,
                        lambda: gb.track_graph(log, r), _S.get("ends", ()))


def names_report():
    import build_model as bm
    adopt()
    ways, _r, _s, cid, cx, cy = bm.load(REGION, print)
    coords = bm.Coords(cid, cx, cy)
    canon, fill, _h, _nr = gb.name_table(ways, coords, print)
    km = gb.way_km(ways, coords)
    by = Counter()
    for w, (t, _n) in ways.items():
        if t.get("railway") in gb.TRACK_KIND and not t.get("service") \
                and t.get("usage") not in gb.NOT_PASSENGER:
            n = gb.tidy(t.get("name"))
            by[canon.get(fold_key(n), n) if n else "(unnamed)"] += km[w]
    for n, v in by.most_common():
        print(f"  {v:8.1f}  {n}{'  [FREIGHT]' if n in FREIGHT else ''}"
              f"{'  [metro]' if n and METRO_ON_RAIL.search(n) else ''}")


if __name__ == "__main__":
    if "--clip" in sys.argv:
        clip()
    elif "--construction" in sys.argv:
        construction_pass(ROOT / sys.argv[sys.argv.index("--construction") + 1])
    elif "--names" in sys.argv:
        names_report()
    else:
        print(__doc__)
