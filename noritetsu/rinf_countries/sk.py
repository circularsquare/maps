"""Slovakia: ŽSR (Železnice Slovenskej republiky) is the one infrastructure manager in RINF
(0056_IM). Its line id is "SK_2601", ŽSR's own internal number for a stretch of line, which
riders never see. Riders know the timetable (KCP) number, "trať 120 Bratislava - Žilina": it
heads each table of ZSSK's timetable, is on departure boards, and is how sk.wikipedia and
Wikidata's P1671 number lines. So, as in Czechia, the ref is the timetable number.

There is no rule from ŽSR's id to the KCP number (SK_2601 is 180, SK_2701 is 120), so every
number comes from OSM's route=railway relations, which carry the KCP number in `ref`
("Táto relácia reprezentuje trať v cestovnom poriadku": this relation is the line in the
timetable). OSM ALSO HAS ŽSR's NETWORK-STATEMENT NUMBERS, on route=tracks relations
("105A" Košice - Kraľovany, "traťový úsek podľa zoznamu železničných tratí ŽSR"); `sk_rel`
ignores every route=tracks relation, and every route=railway relation outside the KCP range
100-199 or another network's (MÁV's, PKP's, ÖBB's, SŽ's lines reach over the border, and
PKP's Id-12 numbers 487-489 and 559 are mapped on ŽSR's border stubs). Rimavská Sobota -
Poltár (175) has no ref in OSM and is recognised by its Wikidata item.

The name is the timetable's form, "120 Bratislava – Žilina", the route read from the
relation's sk.wikipedia title (else its description). Wikidata is not read by the build
("wikidata": None): its labels are "železničná trať A – B" and rinf.py would prefer them; the
items' lengths (P2043, the sk.wikipedia infobox "dĺžka") are what check_model.REGISTER["sk"]
uses, pulled once into data/raw/sk_wikidata.json.

WHAT ŽSR's RINF GETS WRONG, corrected in `sk_fix` (rinf.py's `fix` hook) before anything
reads it:

- 89 points are typed 110 ("technical change"), among them real passenger stations:
  Piešťany, Pezinok, Bytča, Považská Bystrica, Senica, Levoča, Lipany, Sabinov... As 110 they
  were junctions, merged away, so line 120 had no stop between Leopoldov and Trenčín and
  Predmier - Horný Hričov ran past Bytča. Those named like a station are retyped 10 and then
  matched to OSM stations as any station is; yard groups ("Barca St. 1"), km posts, AH/EE
  points and Modrý most keep 110.
- A dozen passenger points lie 1-3.4 km from their station (RINF's Prakovce zastávka sits on
  Gelnica zastávka, Malá Maňa on Maňa zastávka, Turzovka zastávka 2.6 km off). Their RINF
  coordinate is dropped, and rinf.py places them at the OSM station of their exact name.
- Three section lengths are not lengths: Čremošné - Horná Štubňa obec on 170 is 5856.0 (metres),
  Horná Štubňa obec - Odb. Dolná Štubňa 238.421 and Varín - Žilina-Teplička 214.0/201.0 are
  kilometre positions. Metres are converted; positions are dropped (the section keeps its
  trace and has no RINF length).
- A junction standing at a stop, joined to it by a 0 km section (Lastovce odb. - Lastovce on
  191), becomes that stop; left apart, build_model dropped the 0 km section and 191 fell into
  two pieces.
- ŽSR's ids do not follow the timetable lines. SK_2902 runs Fiľakovo - Lučenec - Zvolen -
  Kremnica - Horná Štubňa - Martin - Vrútky, which is 160, 150's last stretch, 171 and 170;
  SK_3104 Zvolen - Banská Bystrica - Podbrezová is 170 then 172; SK_3012 carries 144
  (Prievidza - Nitrianske Pravno) on the end of 140. `SPLITS` cuts them at the stations where
  the timetable lines meet and gives each piece its number (`FIXED`). Where two timetable
  numbers share track, it goes to the line it completes: Zvolen - Hronská Dúbrava to 150
  (Nové Zámky - Zvolen, the through line) rather than 171, and Odb. Dolná Štubňa - Diviaky
  to 170 rather than 171, so 171 runs Hronská Dúbrava - Kremnica - Horná Štubňa.

ŽSR's RINF section lengths leave out the track inside stations (they run between station
limits), like MÁV's: they add up to 3,151 km against 3,255 km of crow-fly distance between
the same points, and line 120 is 157.9 km in RINF against 203 km published. Hence `tol_abs`,
as in Hungary; check_model's comparison against RINF's own lengths reads about 1.2 for that
reason, and the published lengths in REGISTER are the check that matters.

Eighteen timetable lines have no passenger trains (`SUSPENDED`, below); rinf.py's `suspended`
hook flags them and not_running.py greys every section, as Japan's closed lines are.
"""
import math
import re
from collections import defaultdict

KCP_MIN, KCP_MAX = 100, 199
ZSR = "Železnice Slovenskej republiky"
# Relations that carry their timetable number only in Wikidata.
REL_BY_WIKIDATA = {"Q803094": "175"}       # Rimavská Sobota - Poltár, no ref in OSM


def route_of(tags):
    """The route part of a relation's name: its sk.wikipedia title, else its description.
    Neither has the number; the KCP relations have no `name`."""
    wp = tags.get("wikipedia") or ""
    m = re.match(r"sk:Železničná trať\s+(.+)", wp)
    if m:
        return m.group(1).strip()
    d = (tags.get("description") or tags.get("name") or "").strip()
    d = re.sub(r"\s*\([^)]*\)", "", d)                  # "Bratislava - Žilina - (Košice)"
    d = re.sub(r"\s+-\s+", " – ", d).strip(" –-")
    return d


def sk_rel(tags):
    """(ref, name) of an OSM line relation, or None to ignore it (module docstring)."""
    if tags.get("route") != "railway":
        return None
    ref = (tags.get("ref") or "").strip()
    if not ref and tags.get("wikidata") in REL_BY_WIKIDATA:
        ref = REL_BY_WIKIDATA[tags["wikidata"]]
    if not re.fullmatch(r"\d{3}", ref) or not KCP_MIN <= int(ref) <= KCP_MAX:
        return None
    if tags.get("network") != ZSR and tags.get("wikidata") not in REL_BY_WIKIDATA:
        return None                       # "165 Gemerské spojky", a disused link, untagged
    route = route_of(tags)
    return ref, (f"{ref} {route}" if route else ref)


# ------------------------------------------------------------------ corrections (sk_fix)

# Type-110 points that are not stations: yard groups, km posts, an automatic block point (AH),
# a power-station siding (EE), and Modrý most, a junction in Bratislava.
NOT_A_STATION = re.compile(r"\bSt\.\s*\d|^km\s|\bkm\s[\d,]+$|^AH\s|^EE\s|^Modrý most$")

# Passenger points whose RINF coordinate is more than NAME_M (1 km) from the OSM station of
# their name, measured 2026-10-01; most are where a neighbouring stop is.
RELOCATE = {"Dulov", "Gelnica zastávka", "Kežmarok-Pradiareň", "Kraľovany zastávka", "Krivany",
            "Malá Maňa", "Maňa zastávka", "Medzilaborce", "Prakovce zastávka",
            "Strážky zastávka", "Turzovka zastávka", "Švošov"}

# One ŽSR id that is several timetable lines: cut at `cut` (point names), and each connected
# piece is named by one of its sections (two point names) and takes that piece's number.
SPLITS = {
    "SK_2902": {"cut": {"Zvolen osobná stanica", "Hronská Dúbrava", "Odb. Dolná Štubňa"},
                "pieces": {("Lučenec", "Tomášovce"): "160",
                           ("Zvolen osobná stanica", "Hronská Dúbrava"): "150",
                           ("Kremnica", "Kremnické Bane"): "171",
                           ("Martin", "Priekopa"): "170"}},
    "SK_3104": {"cut": {"Banská Bystrica"},
                "pieces": {("Zvolen mesto odb.", "Banská Bystrica"): "170",
                           ("Banská Bystrica", "Šalková"): "172"}},
    "SK_3012": {"cut": {"Prievidza"},
                "pieces": {("Prievidza", "Nedožery"): "144",
                           ("Oslany", "Bystričany"): "140"}},
}
FIXED = {f"{lid}-{ref}": ref for lid, sp in SPLITS.items() for ref in sp["pieces"].values()}
# Zvolen osobná stanica - Zvolen mesto odb., 0.3 km, where 170 leaves Zvolen; it lies beside
# 160's relation, so the relations alone leave it unnumbered.
FIXED["SK_3111"] = "170"


def _crow_km(a, b):
    if "lon" not in a or "lon" not in b:
        return None
    dx = (b["lon"] - a["lon"]) * math.cos(math.radians((a["lat"] + b["lat"]) / 2)) * 111320
    dy = (b["lat"] - a["lat"]) * 110570
    return math.hypot(dx, dy) / 1000


def sk_fix(secs, points):
    """rinf.py's `fix` hook: correct ŽSR's RINF in place (module docstring). Returns log lines."""
    out = []
    n110 = 0
    for p in points.values():
        if p.get("type") == "110" and p.get("name") and not NOT_A_STATION.search(p["name"]):
            p["type"] = "10"
            n110 += 1
    out.append(f"{n110} type-110 points named like stations retyped as stations")
    moved = []
    for p in points.values():
        if p.get("name") in RELOCATE and "lon" in p:
            del p["lon"], p["lat"]
            moved.append(p["name"])
    out.append(f"{len(moved)} points lose their RINF coordinate: {', '.join(sorted(moved))}")
    for s in secs:
        a, b = points.get(s["a"], {}), points.get(s["b"], {})
        crow, km = _crow_km(a, b), s["km"]
        if crow is None or km is None:
            continue
        if km > 50 and 0.5 * crow <= km / 1000 <= 3 * crow + 1:
            s["km"] = km / 1000
            out.append(f"{s['line']} {a.get('name')} - {b.get('name')}: {km} read as metres")
        elif km > 3 * crow + 3:
            s["km"] = None
            out.append(f"{s['line']} {a.get('name')} - {b.get('name')}: {km} km for "
                       f"{crow:.1f} km crow-fly is not a length; dropped")
    # A junction at the same place as a stop, joined to it by a 0 km section ("Lastovce odb." -
    # Lastovce, 0.019 km, 0 m apart): the junction becomes the stop. Left apart, build_model
    # dropped the 0 km junction-ended section as unridden and 191 fell into two pieces.
    stop_types = {"10", "20", "30", "70"}
    same = {}
    for s in secs:
        a, b = points.get(s["a"], {}), points.get(s["b"], {})
        crow = _crow_km(a, b)
        if s["km"] is None or s["km"] >= 0.05 or crow is None or crow >= 0.1:
            continue
        ta, tb = a.get("type") in stop_types, b.get("type") in stop_types
        if ta != tb:
            junction, stop = (s["b"], s["a"]) if ta else (s["a"], s["b"])
            same[junction] = stop
    if same:
        secs[:] = [s for s in secs
                   if same.get(s["a"]) != s["b"] and same.get(s["b"]) != s["a"]]
        for s in secs:
            s["a"], s["b"] = same.get(s["a"], s["a"]), same.get(s["b"], s["b"])
        out.append("junctions merged into the stop they stand at: " + ", ".join(
            f"{points[j].get('name')} -> {points[k].get('name')}" for j, k in same.items()))
    # split ids
    for lid, sp in SPLITS.items():
        mine = [s for s in secs if s["base"] == lid]
        name = {op: points.get(op, {}).get("name") for s in mine for op in (s["a"], s["b"])}
        parent = {}

        def find(x):
            while parent.setdefault(x, x) != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        by_node = defaultdict(list)
        for i, s in enumerate(mine):
            for op in (s["a"], s["b"]):
                if name[op] not in sp["cut"]:
                    by_node[op].append(i)
        for ids in by_node.values():
            for i in ids[1:]:
                parent[find(i)] = find(ids[0])
        piece_ref = {}
        for i, s in enumerate(mine):
            pair = {name[s["a"]], name[s["b"]]}
            for ends, ref in sp["pieces"].items():
                if set(ends) == pair:
                    piece_ref[find(i)] = ref
        n = defaultdict(int)
        for i, s in enumerate(mine):
            ref = piece_ref.get(find(i))
            if ref is None:
                out.append(f"{lid}: {name[s['a']]} - {name[s['b']]} is in no named piece; "
                           f"left on {lid}")
                continue
            s["line"] = s["base"] = f"{lid}-{ref}"
            n[ref] += 1
        out.append(f"{lid} split into " + ", ".join(f"{r} ({k} sections)" for r, k in n.items()))
    return out


# Ids with no passenger trains that would still be built, because they run between two points
# that are OSM stations (a section between two stops is never questioned):
# - Varín - Žilina-Teplička, two single sections whose "lengths" are km positions, and
#   Žilina-Teplička - Žilina: the freight bypass and the Teplička marshalling yard; and the
#   yard's other links (SK_2606, 2607, 2610, 2612), whose traces run along 180 between Varín
#   and Potok odb., where rinf.py took 180's own section for a second track pair, dropped it,
#   and left 180 in two pieces;
# - SK_3321 Haniska pri Košiciach - Maťovce and SK_3341 Čierna nad Tisou: the 1520 mm broad
#   gauge (ŠRT), freight only.
SKIP = {"SK_2601L7543", "SK_2601L7544", "SK_2609", "SK_2606", "SK_2607", "SK_2610", "SK_2612",
        "SK_3321", "SK_3341"}


# Timetable lines with no passenger trains: sk.wikipedia's "Zoznam železničných tratí na
# Slovensku" marks them "pravidelná osobná prevádzka prerušená" (retrieved 2026-10-01), and
# OSM has no passenger route on any of them (0.00-0.06 of each line, where every line with
# trains is 0.47-1.00). The page also lists 115, 122 and 136, which are not built (no track
# or not in RINF), and 135 Nové Zámky - Komárom, which OSM has an Os service on and stays.
# They stay on the map greyed as not running (not_running.py; Anita's rule for closures).
# SK_3141 Chvatimech - Hronec, the Hronec freight branch, has no number and no OSM route.
SUSPENDED = {"112", "113", "117", "124", "134", "142", "144", "161", "163", "164", "165",
             "166", "167", "168", "186", "187", "192", "195"}
SUSPENDED_IDS = {"SK_3141"}


def sk_suspended(ref, rinf_ids):
    return ref in SUSPENDED or any(i in SUSPENDED_IDS for i in rinf_ids)


COUNTRY = {
    "iso3": "SVK", "wikidata": None, "langs": ["sk"],
    "osm_rel": sk_rel, "fix": sk_fix, "fixed": FIXED,
    "skip_line": lambda lid: lid in SKIP, "suspended": sk_suspended,
    # Station track is left out of ŽSR's section lengths (docstring).
    "tol_abs": 1.0,
    "im": {"0056_IM": "Železnice Slovenskej republiky"},
}
