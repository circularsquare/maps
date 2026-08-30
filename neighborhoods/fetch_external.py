"""
Stage 5b: per-city OFFICIAL polygon sources, for cities OSM has no shapes for.

Who's On First (fetch_wof.py) is the generic fallback and it is spent — measured, it gave
Cairo 0 shapes, Nairobi 0, Johannesburg 0, Lima 0, Hong Kong 0 and Copenhagen 7, because
its 1,682 useful shapes went to cities that already had coverage. The cities that actually
need help need naming, one at a time, from whoever publishes that city's own geography.

Four are configured. Each was found by research and each download was verified by hand
before being written in here; the per-city notes record what the source is and what it is
NOT, because the failure modes differ and the next person should not have to rediscover
them.

WHY PER-CITY AND NOT A GENERIC GAZETTEER. There is no world source at this granularity —
that is the whole finding. Overture's divisions theme is OpenStreetMap plus geoBoundaries,
so it cannot beat what we already hold; GADM forbids redistribution and renders Zamalek,
a Nile island, as a pentagon. What exists instead is national and municipal open data,
in four different formats with four different name conventions.

THE JOIN IS NAME + CONTAINMENT, never name alone. Denmark has seven Frederiksbergs and
none of them is Copenhagen's; `Amager` as a bydel exists only in Jutland. Requiring the
OSM point to fall inside the polygon turns a name coincidence into evidence, which is the
same rule §1.3a settled on for borrowing.

TWO OUTPUTS, AND THE SECOND IS THE POINT. `shapes` hands a polygon to a unit the corpus
already has. `units` is the inversion — the polygons no unit claimed, promoted to units of
their own, which is what reaches a city OSM has barely mapped: Cairo gains 26 kism the
survey does not name at all. §1.7 has the design; the short version is that the §3 level
rule is APPLIED to these layers rather than waived for them, and it rejects Kuala Lumpur's
at 5,504 people per unit, correctly, because that layer is a housing-estate register
carrying a wholesale market and an industrial estate.

Usage:
    python fetch_external.py --plan
    python fetch_external.py
    python fetch_external.py --only Q1748
"""

import argparse
import collections
import hashlib
import json
import io
import pathlib
import re
import sys
import unicodedata
import urllib.parse
import urllib.request
import zipfile

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import osmgeom
import pick_levels as pl
from fetch_osm import cities

HERE = pathlib.Path(__file__).parent
CACHE = HERE / "cache" / "external"
OUT = HERE / "data" / "external_shapes.json"
BASE = HERE / "data" / "base.json"

UA = "maps-neighborhoods/0.1 (https://anita.garden; anitaxinchen@gmail.com)"
TIMEOUT = 300

# A match covering this much of the city is the CITY, not a district — the failure mode
# is a unit named after its own city adopting the whole outline.
#
# NOT a ratio-to-median, which is what this was first written as, borrowed from the WOF
# stage. That guard is right for a world gazetteer where a neighbourhood can match a
# country, and wrong here: these are curated municipal layers in which every polygon is a
# legitimate unit, so the median is dragged down by the many small ones and normal
# districts trip it. At 12x-median it rejected Nørrebro (5.5 km²), Østerbro, Valby and
# Gentofte — all real — and cost Copenhagen 13 of its 54 matches.
MAX_CITY_FRAC = 0.5

# Containment is the gate, but an OSM point is sometimes placed at a station or a
# landmark just outside its own polygon. Where nothing contains the point, a single
# candidate within this distance is accepted; two or more is an ambiguity we refuse,
# which is what keeps the seven Danish Frederiksbergs out.
NEAR_TOLERANCE_DEG = 0.02  # ~2 km

# A source polygon overlapping an OSM unit's own outline by at least this much IS that
# unit arriving from a second direction, however differently the two spell its name — so
# it must not be promoted to a unit of its own. Same threshold and same reasoning as the
# Who's On First duplicate test (`WOF_DUP_IOU` in build.py); the name join is the primary
# defence and this is the backstop for the transliterations it misses.
UNIT_DUP_IOU = 0.9


# ---------------------------------------------------------------- name normalisation

_ARABIC_PREFIX = re.compile(r"^\s*(قسم|مركز|شياخة|حي)\s*")
# Malay local-authority data abbreviates relentlessly and inconsistently.
_MALAY = {"tmn": "taman", "kg": "kampung", "kpg": "kampung", "bdr": "bandar",
          "sg": "sungai", "kwsn": "kawasan", "jln": "jalan", "bkt": "bukit",
          "sek": "seksyen", "pkn": "pekan"}
# Suffixed compass variants are the single biggest source of false misses in Indian data:
# our "Borivali East" against the source's "BORIVALI".
_COMPASS = re.compile(r"\b(east|west|north|south|e|w|n|s)\b")


def _strip_accents(s):
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch))


def norm_latin(s):
    return re.sub(r"[^a-z0-9]", "", _strip_accents((s or "").casefold()))


def norm_compass(s):
    """Latin, with compass suffixes dropped. For Mumbai."""
    return re.sub(r"[^a-z0-9]", "", _COMPASS.sub("", _strip_accents((s or "").casefold())))


def norm_malay(s):
    toks = [_MALAY.get(t, t) for t in re.split(r"[^a-z0-9]+", _strip_accents((s or "").casefold())) if t]
    return "".join(toks)


def norm_arabic(s):
    """Strip the administrative prefix, unify the letters that vary, drop ALL whitespace.

    The whitespace rule is not cosmetic: the published Egyptian names have missing spaces
    (`قسم مصرالجديدة` for `مصر الجديدة`), so anything space-sensitive misses most of them.
    """
    s = _ARABIC_PREFIX.sub("", s or "")
    s = _strip_accents(s)
    for a, b in (("أ", "ا"), ("إ", "ا"), ("آ", "ا"), ("ى", "ي"), ("ة", "ه"),
                 ("ؤ", "و"), ("ئ", "ي")):
        s = s.replace(a, b)
    return re.sub(r"\s+", "", s)


# ---------------------------------------------------------------------- the sources

def _dawa_url():
    """DAWA rejects an unfiltered national query (180 MB+), so the bbox is mandatory."""
    poly = [[[12.2, 55.55], [12.78, 55.55], [12.78, 55.87], [12.2, 55.87], [12.2, 55.55]]]
    return "https://api.dataforsyningen.dk/steder?" + urllib.parse.urlencode(
        {"format": "geojson", "hovedtype": "Bebyggelse", "polygon": json.dumps(poly)}
    )


SOURCES = {
    # ---------------------------------------------------------------- Copenhagen
    "Q1748": {
        "city": "Copenhagen",
        "source": "Danske Stednavne (DAWA /steder), Klimadatastyrelsen",
        "licence": "CC BY 4.0",
        "attribution": "Klimadatastyrelsen (KDS)",
        "url": _dawa_url(),
        "nameField": "primærtnavn",
        "norm": norm_latin,
        "keep": lambda p: p.get("undertype") in ("bydel", "by"),
        "note": (
            "The best of the four: 54 of 56 units matched in testing, taking Copenhagen "
            "from 2% polygons to near-complete. Only `bydel` and `by` are kept — the same "
            "response carries 172 `kolonihave` (allotment gardens) and 48 `spredtBebyggelse`, "
            "which are not neighbourhoods. Reaches Hellerup, Bagsværd, Skovlunde and "
            "Frederiksberg, which no Copenhagen-municipality source can, because they are "
            "separate kommuner."
        ),
        # `id` is a DAWA UUID and is the stable key; the name is only the fallback.
        "units": {
            "key": "ext=bydel", "label": "bydel", "class": "informal", "idField": "id",
            "why": (
                "`informal` and not `official`: DAWA is the national place-NAME register "
                "(`primærnavnestatus: officielt` certifies the name, not a division), so "
                "these are the districts Copenhageners use rather than a kommune's own "
                "administrative split. Same class as London's `area of London`."
            ),
        },
    },
    # ---------------------------------------------------------------- Kuala Lumpur
    "Q1865": {
        "city": "Kuala Lumpur",
        "source": "DBKL City Planning System, Sempadan Taman layer",
        "licence": "not stated by DBKL; Malaysia's Terms of Use for Government Open Data "
                   "1.0 permits redistribution and commercial use with attribution",
        "attribution": "Dewan Bandaraya Kuala Lumpur, City Planning System (CPS)",
        "url": ("https://cps.dbkl.gov.my/server/rest/services/Hosted/SEMP_PENTADBIRAN/"
                "FeatureServer/4/query?where=1%3D1&outFields=*&returnGeometry=true"
                "&outSR=4326&f=geojson"),
        "nameField": "na_kawasan",
        "norm": norm_malay,
        "note": (
            "333 polygons tiling 237 km² of KL's 243 km². Only ~28% of our KL units match "
            "and that is expected, not a fault: DBKL's writ stops at the city boundary, "
            "while our 25 km radius pulls in Selangor — Bukit Jelutong, Subang, Kota "
            "Damansara, Bandar Utama and Batu Tiga are all outside it by design. The "
            "neighbouring councils publish an identical `Sempadan Taman` layer (MBSJ, "
            "MBSA, MPAJ) if the Selangor fringe is ever worth sweeping."
        ),
        # ArcGIS `objectid` renumbers when a layer is republished, so it is NOT usable as
        # a stable key and the normalised name is used instead — `na_kawasan` is unique
        # across all 333 features.
        "units": {
            "key": "ext=taman", "label": "taman", "class": "informal",
            "why": (
                "Configured but REJECTED by the level rule at 5,505 people per unit "
                "(§1.7), and the rejection is correct: the layer holds `Pasar Borong "
                "Kuala Lumpur` (a wholesale market) and `Kwsn Industri Batu Caves` (an "
                "industrial estate) beside its housing estates. Left declared so the "
                "verdict is visible rather than the source silently doing nothing."
            ),
        },
    },
    # ---------------------------------------------------------------- Mumbai
    "Q1156": {
        "city": "Mumbai",
        "source": "Greater Mumbai revenue village boundaries",
        "licence": "CC BY 4.0 as asserted by the publishing repository; upstream "
                   "provenance unstated, almost certainly the Maharashtra revenue layer",
        "attribution": "mumbai_spatial_data (sanjanakrishnan)",
        "url": ("https://raw.githubusercontent.com/sanjanakrishnan/mumbai_spatial_data/"
                "main/mumbai_village_boundaries.geojson"),
        "nameField": "village_na",
        "norm": norm_compass,
        "note": (
            "124 polygons forming a gap-free partition of the MCGM area. Matches ~36% of "
            "`place=suburb` and ~1% of `place=neighbourhood`, which is the granularity "
            "story: these are district-scale, and most of our Mumbai records are finer. "
            "Names are UPPERCASE and some are split (`BANDRA-A`..`BANDRA-I`), so the "
            "normaliser drops compass suffixes and the index carries hyphen-split parts. "
            "CAUTION: these are historical land-revenue extents, so the shape of Powai or "
            "Kandivali is tighter than colloquial usage. Do not treat as authoritative."
        ),
        # No id field at all — `village_na` is the only property and it is unique.
        "units": {
            "key": "ext=village", "label": "revenue village", "class": "official",
            # One feature carries a whitespace-only name. An unnamed unit cannot be
            # browsed or guessed, the same reason build() skips OSM elements without one.
            "exclude": [r"^\s*$"],
        },
    },
    # ---------------------------------------------------------------- Cairo
    "Q85": {
        "city": "Cairo",
        "source": "HDX / OCHA COD-AB Egypt, admin level 2 (kism), sourced from CAPMAS",
        "licence": "CC BY 3.0 IGO",
        "attribution": "OCHA ROMENA / CAPMAS, via the Humanitarian Data Exchange",
        "url": ("https://data.humdata.org/dataset/b90d81ba-7c7a-4283-9899-827480d80a79/"
                "resource/2080e95f-51c7-45b6-b5ab-613ff1cfc041/download/"
                "egy_admin_boundaries.geojson.zip"),
        "zipMember": "egy_admin2.geojson",
        "nameField": "adm2_name1",          # Arabic; adm2_name is the English transliteration
        "altNameFields": ["adm2_name", "adm2_ref_name"],
        "norm": norm_arabic,
        "keep": lambda p: p.get("adm1_name") in ("Cairo", "Giza", "Kalyoubia"),
        "note": (
            "79 kism polygons across the three governorates Greater Cairo spans — a "
            "Cairo-Governorate-only source would miss Dokki, Agouza and Imbaba. Matches "
            "~37% of our units because kism is a DISTRICT level and many of our records "
            "are sub-district (Kit Kat, Abbasiya, Mit Okba).\n"
            "Chosen over better-fitting alternatives on licence alone. A shiakha-level "
            "source exists with 823 bilingual full-resolution polygons, but it lives on "
            "an individual's ArcGIS account with no stated licence, republishing data "
            "CAPMAS asserts copyright over. GADM forbids redistribution outright and "
            "draws Zamalek as a pentagon. CAPMAS and the Egypt Survey Authority publish "
            "nothing downloadable at all — both were checked."
        ),
        # `adm2_pcode` (EG1309) is CAPMAS's own code and survives a reissue of the file.
        "units": {
            "key": "ext=kism", "label": "kism", "class": "official",
            "idField": "adm2_pcode",
            "enField": "adm2_name",
            # KISM ONLY. `adm2_name1` mixes two scales under one admin level, exactly the
            # §3.4 problem: قسم is an urban district and مركز a RURAL one, so Markaz
            # Ausim and Markaz El-Badrashein are farmland administered from Giza, not
            # Cairo districts. `norm_arabic` strips both prefixes for matching, so the
            # distinction has to be re-read from the raw name here.
            "exclude": [r"^\s*مركز\b"],
        },
    },
}


def fetch(qid, cfg, refresh=False):
    CACHE.mkdir(parents=True, exist_ok=True)
    dest = CACHE / f"{qid}.json"
    if dest.exists() and not refresh:
        return json.loads(dest.read_text("utf-8"))
    print(f"  fetching {cfg['city']} — {cfg['source']}", flush=True)
    req = urllib.request.Request(cfg["url"], headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        raw = r.read()
    if cfg.get("zipMember"):
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            raw = z.read(cfg["zipMember"])
    fc = json.loads(raw.decode("utf-8"))
    dest.write_text(json.dumps(fc, ensure_ascii=False), "utf-8")
    print(f"    {len(fc.get('features') or [])} features, {len(raw)/1e6:.1f} MB", flush=True)
    return fc


def index_source(cfg, fc):
    """-> (normalised name -> [feature record], [feature record]).

    ONE record per kept source polygon, and the index holds those same objects rather
    than copies, so a match can be attributed back to the feature it came from. That
    attribution is what §1.7's inversion needs: the polygons no unit claimed are the
    ones worth promoting to units of their own, and they cannot be identified from a
    name-keyed index alone — a feature appears in it once per name field and once per
    hyphen-split part.
    """
    keep = cfg.get("keep") or (lambda p: True)
    fields = [cfg["nameField"]] + list(cfg.get("altNameFields") or [])
    idx, feats = collections.defaultdict(list), []
    for f in fc.get("features") or []:
        props = f.get("properties") or {}
        if not keep(props):
            continue
        try:
            g = osmgeom.sg.shape(f["geometry"])
        except Exception:
            continue
        if not g.is_valid:
            g = g.buffer(0)
        if g.is_empty or g.area <= 0:
            continue
        rec = {"name": props.get(cfg["nameField"]), "g": g, "props": props}
        feats.append(rec)
        for field in fields:
            v = props.get(field)
            if not v:
                continue
            idx[cfg["norm"](v)].append(rec)
            # Hyphen-split parts: Mumbai's `DADAR-NAIGAON` must answer to both halves.
            for part in re.split(r"[-–]", str(v)):
                part = part.strip()
                if len(part) > 3:
                    idx[cfg["norm"](part)].append(rec)
    return idx, feats


# ------------------------------------------------- promoting source polygons to units

def synth_id(qid, key):
    """A stable `i` for a unit that has no OSM object behind it.

    §6 makes `i` a join key that must survive a refetch — the browser joins a dot to its
    outline on it, and the quiz will collapse on `dupOf` — so an id that churned between
    builds would reshuffle the deck for no reason. Hashing keeps it short and keeps the
    city in it, so two sources cannot collide.

    The key hashed is the source's own stable id where it HAS one and the raw name
    otherwise, which is the opposite of the obvious default and is measured: of the four
    sources only DAWA (`id`, a UUID) and HDX (`adm2_pcode`, CAPMAS's own code) publish
    one. DBKL's `objectid` is an ArcGIS row number that renumbers whenever the layer is
    republished, and Mumbai's file has no id property at all — for those two the name is
    strictly the more stable choice.

    THE RAW NAME, NEVER `cfg["norm"]`. The normalisers exist to make a LOOSE match — the
    Mumbai one deletes compass words so that our "Borivali East" reaches the source's
    "BORIVALI" — and looseness is the one property an identity key must not have. Keyed
    on the normalised name, `PAHADI GOREGAON-WEST` and `PAHADI GOREGAON-EAST` are one
    id: two polygons, one record, and a join key that silently resolves to whichever was
    written last.
    """
    return "x" + hashlib.sha1(f"{qid}|{key}".encode("utf-8")).hexdigest()[:10]


def _iou(a, b):
    """Intersection over union. Cheap bbox reject first — this runs over every source
    polygon against every OSM outline in the city."""
    if not a.intersects(b):
        return 0.0
    u = a.union(b).area
    return a.intersection(b).area / u if u > 0 else 0.0


def osm_outlines(qid, corpus):
    """This city's surveyed OSM polygons, for the duplicate backstop. `{}` if unfetched."""
    path = HERE / "cache" / "geom" / f"{qid}.json"
    if not path.exists():
        return {}
    out = {}
    for el in json.loads(path.read_text("utf-8")).get("elements", []):
        uid = f"{el['type'][0]}{el['id']}"
        if uid not in corpus:
            continue
        g = osmgeom.shape_of(el)
        if g is None or g.is_empty or g.geom_type not in ("Polygon", "MultiPolygon"):
            continue
        out[uid] = g
    return out


def create_units(qid, cfg, city, feats, claimed, units, shape):
    """The polygons no OSM unit claimed, promoted to units of their own.

    THIS IS THE INVERSION. The match loop above asks "does this city's own source have a
    shape for a unit OSM already knows?"; this asks the complementary question, "does it
    know a district OSM does not?", and for Cairo the answer is 26 kism — Abdeen,
    Ezbakeya, El-Darb El-Ahmar, El-Sahel — that no `place=*` node in the corpus names.

    Three gates, and each one is doing measurable work:

    - **The radius**, the same `city.radiusKm` every OSM unit is held to. DAWA's mandatory
      bbox is bigger than Copenhagen: 296 kept polygons, only 122 within 15 km. Without
      this, 174 Zealand villages become Copenhagen neighbourhoods.
    - **`exclude`**, per source, for a layer that mixes two scales the way §3.4's Delhi
      does. Cairo's file puts urban قسم and rural مركز on one admin level and
      `norm_arabic` strips both prefixes, so only the raw name can separate them.
    - **`UNIT_DUP_IOU`** against the city's own OSM outlines, because a name miss is not
      proof of distinctness. The name join is the primary defence; this catches the
      transliteration it fumbles.

    Returns (unit records, geometry by id, counts).
    """
    ucfg = cfg["units"]
    excl = [re.compile(p, re.I) for p in ucfg.get("exclude") or []]
    corpus = {u["i"] for u in units}
    outlines = osm_outlines(qid, corpus)

    def id_key(rec):
        if ucfg.get("idField"):
            return str(rec["props"].get(ucfg["idField"]))
        return str(rec["name"] or "").strip()

    # A source is entitled to give two polygons the same name — the identity key just has
    # to survive it. Colliding keys are extended with the rounded representative point,
    # which distinguishes them and stays put across a refetch as long as the boundary
    # does; the uncollided majority keep the bare key, so no id churns for this.
    seen = collections.Counter(id_key(r) for r in feats)

    out, geoms = [], {}
    n_far = n_excl = n_dup = 0
    n_layer = n_core = 0
    for rec in feats:
        name = str(rec["name"] or "").strip()
        g = rec["g"]
        # EXCLUDED BEFORE COUNTED, the order §3.4 settled on: filtering at display only
        # would leave the division estimate dividing by units nobody can see.
        if not name or any(r.search(name) for r in excl):
            n_excl += 1
            continue
        c = g.representative_point()       # guaranteed inside, unlike a centroid
        if pl.haversine_km((c.x, c.y), (city["lon"], city["lat"])) > city["radiusKm"]:
            n_far += 1
            continue
        in_core = None if shape is None else bool(shape.contains(c))
        # LAYER STATISTICS COUNT EVERY POLYGON IN RADIUS, claimed or not, because the
        # division estimate (§3.1) asks how many parts the layer cuts the city into and
        # that does not change because OSM happens to name some of them too. Counting
        # only the promoted ones would divide the city's population by the source's
        # failure to be joined — §3.4's error with the sign flipped.
        n_layer += 1
        if in_core:
            n_core += 1

        if id(rec) in claimed:
            continue
        if any(_iou(og, g) >= UNIT_DUP_IOU for og in outlines.values()):
            n_dup += 1
            continue
        key = id_key(rec)
        if seen[key] > 1:
            key = f"{key}@{c.x:.4f},{c.y:.4f}"
        uid = synth_id(qid, key)
        en = rec["props"].get(ucfg["enField"]) if ucfg.get("enField") else None
        out.append({
            "c": qid, "i": uid, "k": ucfg["key"], "n": name,
            "en": en if en and str(en) != name else None,
            "x": round(c.x, 5), "y": round(c.y, 5),
            "core": None if in_core is None else int(in_core),
            "src": qid,
        })
        geoms[uid] = osmgeom.sg.mapping(g)

    # §6 makes `i` the join key between a dot and its outline, so a collision is not a
    # cosmetic problem — it is two places sharing one record and one of the two outlines
    # being dropped. Cheap to check, and it caught a real one: Mumbai's PAHADI GOREGAON
    # East and West.
    dupe = [i for i, n in collections.Counter(u["i"] for u in out).items() if n > 1]
    if dupe:
        raise SystemExit(f"{qid}: synthetic id collision on {dupe} — fix the id key")

    # §3.1's division estimate, unchanged and deliberately so — see the docstring above
    # and §1.7. The denominator is core units where the boundary supports it, matching
    # the city population it is divided into.
    enough_core = (shape is not None and n_core >= pl.MIN_CORE_UNITS
                   and n_core >= pl.MIN_CORE_FRAC * n_layer)
    denom = n_core if enough_core else (n_layer or None)
    mean_pop = city["pop"] / denom if denom else None
    layer = {
        "key": ucfg["key"], "label": ucfg["label"], "class": ucfg["class"],
        "n": len(out), "nLayer": n_layer, "core": n_core,
        "meanPop": None if mean_pop is None else int(mean_pop),
        "verdict": pl.verdict(n_layer, None, mean_pop is not None and mean_pop >= pl.BAND_MIN),
        "source": cfg["source"], "licence": cfg["licence"],
        "attribution": cfg["attribution"],
    }
    return out, geoms, layer, {"outsideRadius": n_far, "excluded": n_excl, "iouDupe": n_dup}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="comma-separated city QIDs")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--plan", action="store_true")
    args = ap.parse_args()
    only = set(args.only.split(",")) if args.only else None

    if args.plan:
        print(f"{len(SOURCES)} configured sources:")
        for qid, c in SOURCES.items():
            print(f"  {c['city']:16s} {qid:8s} {c['licence'][:46]}")
            print(f"      {c['url'][:110]}")
        return

    if not BASE.exists():
        sys.exit("no data/base.json — run build.py first")
    base = json.loads(BASE.read_text("utf-8"))
    by_city = collections.defaultdict(list)
    for u in base["units"]:
        by_city[u["c"]].append(u)

    seeds = {c["qid"]: c for c in cities()}

    shapes, stats, layers, new_units = {}, {}, {}, []
    for qid, cfg in SOURCES.items():
        if only and qid not in only:
            continue
        # OSM UNITS ONLY. `base.json` is this stage's input and, since §1.7, also
        # downstream of its output: build.py writes the units created here back into it.
        # Left in, they match the very polygons they were made from, mark them claimed,
        # and the second run creates nothing — the stage would work exactly once and then
        # quietly stop. `src` is what makes the pass idempotent.
        units = [u for u in by_city.get(qid) or [] if not u.get("src")]
        if not units:
            print(f"  {cfg['city']}: no units in base.json, skipping")
            continue
        fc = fetch(qid, cfg, args.refresh)
        idx, feats = index_source(cfg, fc)
        n_src = len(feats)

        city_shape = pl.city_shape(qid)
        city_area = 0.0
        if city_shape is not None:
            try:
                city_area = city_shape.context.area
            except AttributeError:
                city_area = getattr(city_shape, "area", 0.0)

        # WHICH SOURCE POLYGONS A UNIT TOOK, so the complement can be promoted below.
        claimed = set()
        hit = near = miss = outside = oversize = 0
        for u in units:
            cands = idx.get(cfg["norm"](u["n"]))
            if not cands:
                miss += 1
                continue
            pt = osmgeom.sg.Point(u["x"], u["y"])
            # CONTAINMENT IS THE GATE. Denmark has seven Frederiksbergs; only one contains
            # Copenhagen's point. Smallest containing candidate wins, so a district does
            # not beat the neighbourhood inside it.
            inside = [r for r in cands if r["g"].contains(pt)]
            rule = "external-name-containment"
            if inside:
                rec = min(inside, key=lambda r: r["g"].area)
            else:
                nearby = [r for r in cands if r["g"].distance(pt) <= NEAR_TOLERANCE_DEG]
                # Exactly one, or we refuse: ambiguity here is precisely the wrong-city
                # match the containment gate exists to stop.
                if len({r["name"] for r in nearby}) != 1:
                    outside += 1
                    continue
                rec = min(nearby, key=lambda r: r["g"].distance(pt))
                rule = "external-name-near"
                near += 1
            if city_area > 0 and rec["g"].area / city_area > MAX_CITY_FRAC:
                oversize += 1
                if rule == "external-name-near":
                    near -= 1
                continue
            shapes[u["i"]] = {
                "c": qid,
                "g": osmgeom.sg.mapping(rec["g"]),
                "srcName": rec["name"],
                "source": cfg["source"],
                "rule": rule,
            }
            claimed.add(id(rec))
            hit += 1
        stats[qid] = {"city": cfg["city"], "units": len(units), "sourceFeatures": n_src,
                      "matched": hit, "viaNear": near, "nameMiss": miss,
                      "notContained": outside, "coversCity": oversize}

        print(f"  {cfg['city']:16s} {n_src:5d} source polys -> {hit:5d}/{len(units)} units "
              f"({hit/len(units)*100:4.0f}%)   name-miss {miss}, not-contained {outside}, "
              f"covers-city {oversize}, via-near {near}")

        # ---------------------------------------------- promote the unclaimed polygons
        if cfg.get("units") and (city := seeds.get(qid)):
            # The survey elements are what make `trustworthy` able to REJECT a boundary
            # (§4.4a). Passing an empty list silently accepts every one of them, so
            # Istanbul's one-neighbourhood boundary would be trusted here and nowhere
            # else in the pipeline.
            spath = pl.SURVEY / f"{qid}.json"
            els = json.loads(spath.read_text("utf-8")).get("elements", []) if spath.exists() else []
            shape, _ = pl.trustworthy(city, city_shape, els)
            made, geoms, layer, counts = create_units(
                qid, cfg, city, feats, claimed, units, shape)
            layers[qid] = layer
            stats[qid]["units_" + layer["key"]] = counts
            if layer["verdict"] in ("keep", "keep-est"):
                new_units.extend(made)
                shapes.update({i: {"c": qid, "g": g, "source": cfg["source"],
                                   "rule": "external-unit"} for i, g in geoms.items()})
            print(f"      {layer['key']}: {layer['nLayer']} in radius, {layer['core']} core, "
                  f"~{layer['meanPop'] or 0:,}/unit -> {layer['verdict']}"
                  f"{'' if layer['verdict'] in ('keep', 'keep-est') else ' (no units created)'}"
                  f"; {layer['n']} new units, {counts['iouDupe']} iou-dupes, "
                  f"{counts['excluded']} excluded, {counts['outsideRadius']} out of radius")

    OUT.write_text(json.dumps({
        "meta": {
            "note": "Per-city official polygon sources; see fetch_external.py SOURCES for "
                    "the provenance and caveats of each. `shapes` attaches geometry to an "
                    "OSM unit that already exists; `units` are source polygons no unit "
                    "claimed, promoted to units of their own (§1.7), and `layers` carries "
                    "the level-rule verdict that decides whether build.py takes them.",
            "sources": {q: {k: v for k, v in c.items()
                            if k in ("city", "source", "licence", "attribution", "url", "note")}
                        for q, c in SOURCES.items()},
        },
        "stats": stats,
        "layers": layers,
        "units": new_units,
        "shapes": shapes,
    }, ensure_ascii=False, separators=(",", ":")), "utf-8")
    print(f"\nwrote {OUT} — {len(shapes):,} shapes over {len(stats)} cities, "
          f"{len(new_units):,} units created over "
          f"{sum(1 for l in layers.values() if l['verdict'] in ('keep', 'keep-est'))} layers, "
          f"{OUT.stat().st_size/1e6:.1f} MB")


if __name__ == "__main__":
    main()
