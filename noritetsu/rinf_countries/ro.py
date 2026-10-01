"""Romania: CFR SA (Compania Națională de Căi Ferate "CFR", IM code 0053) is the only manager
in RINF. Its line ids ("300", "100A", "316", "301Bb") are CFR Infrastructură's own line
numbers from its network statement, and they are NOT the numbers riders know.

WHICH NUMBER.  Riders, the CFR Călători timetable, ro.wikipedia and OpenStreetMap number lines
by the timetable's "secții": "Magistrala CFR 300" and "Calea ferată 202" on ro.wikipedia,
"Secția 202 Simeria - Filiași" in OSM (route=railway, mostly operator CFR Călători, with the
2011-2021 numbers in old_ref: 100 was 900, 101 was 901). The two schemes share digits but not
meanings: CFR SA's 203 is Copșa Mică - Sibiu - Podu Olt - Piatra Olt, the timetable's 203 is
Bartolomeu - Zărnești; CFR SA's 316 is Brașov - Deda - Războieni, which the timetable splits
into 400 (Brașov - Deda) and 405 (Deda - Târgu Mureș - Războieni). So the ref is the TIMETABLE
number, and CFR SA's id never becomes one by rule.

OSM ALSO CARRIES CFR SA's NUMBERS ("Linia 316 Brașov - Războieni", route=tracks, operator CFR
Infrastructură) on the same `ref` key, plus Hungarian, Serbian and Bulgarian relations at the
borders ("101: Püspökladány–Biharkeresztes"). `ro_rel` keeps only the Secția relations (and
the timetable's 300, which has no name), so the ref of a relation is always a timetable number.

HOW AN ID GETS ITS NUMBER
- Most ids lie along one Secția relation for most of their length, and rinf.py numbers them
  from it (60% of the traced line within 30 m of the relation).
- Ids the relations do not reach are in FIXED: where the Secția relation is mapped only in
  part (108 Roșiori - Turnu Măgurele has 11 ways), or not at all (Căciulați - Snagov Plajă),
  or where the id lies on two (CFR SA's 320 Blaj - Praid is 63% on 307 and runs on to Praid
  where the relation stops). Every number in FIXED is the Secția relation's own, by its end
  stations, or ro.wikipedia's list of magistrale and their branches.
- Ids that CFR SA files as one line but the timetable splits are cut by rinf.py's `fix` hook
  into one id per timetable number, "<id>:<number>", read back by `ro_ref` (SPLIT below).
- Yard links, port lines and Bucharest's freight belt (301x, 304x, 314x, 708x, 813x...) get
  no number and are left out (`skip_line`): they are not lines anyone rides, and joined to a
  numbered line they would turn its junctions into branch points.
- No CFR SA id may stay unnumbered under a name that is also a timetable number: rinf.py
  hashes an unnumbered id the same way as a number, so CFR SA's 307 (I. L. Caragiale - Gura
  Palangii) took the timetable 307's line id and Blaj - Praid vanished from the build. It is
  in FIXED as 303, which it is.

STATIONS.  RINF writes a halt's kind after its name ("Aghires PO" for punct de oprire, "Apa
HM" for haltă de mișcare, "Alius hcv."), where OSM writes "Aghireș hc", "Apa" or "Halta
Balda", so rinf.py matched neither by name. `ro_fix` moves the kind into brackets ("Aghires
(PO)"): rinf.name_variants then tries the name both with and without it. It also writes out
CFR's abbreviations (G-ral, Tr., I. L.). CFR's RINF coordinates are often 0.5-1.5 km from the
station, so `name_m` is 1500 m: measured, a cleaned name finds its station within 1000 m for
1,553 points, 1500 m for 1,632, 2000 m for 1,679, and the further ones are more often the next
halt of the same name. The big stations are further off still (Buzău 2.1 km, Satu Mare 2.4,
Drăgășani 4.1, Beclean pe Someș 4.2, Oradea 4.1, which by prefix took Oradea Est 0.5 km
away), and were section ends no OSM route confirms, so build_model dropped the sections
either side. `relocate` drops the coordinate of a point whose OSM station of exactly its name
is within RELOCATE_M though nothing matching is within name_m, and rinf.py places it there:
113 points, 1,738 of 2,019 passenger points now matched, from 1,450 with neither change.

WHERE TWO TIMETABLE LINES SHARE TRACK the register gives it to one: București - Ploiești is
CFR SA's 300 and so Magistrala 300, though the timetable's 500 and 1000 start at București
Nord too; Beclean pe Someș - Dej is 400 though 401 (Ilva Mică - Cluj) runs over it. Those
lines read short against the timetable's figure by that stretch (check_model notes them).

NAMES are "Magistrala 300 București – Oradea" for the main lines (100 to 800, and 1000) and
"Secția 202 Simeria – Filiași" for the rest, the route taken from the Secția relation's own
name; the English name is "Line 202 (Simeria – Filiași)". Wikidata is not used: its P1671
numbers mix the 2011-2021 and current timetables and CFR SA's ids ("100" is labelled
"Magistrala CFR 900", "210" is both Alba Iulia - Zlatna and Sibiu - Vințu de Jos).
"""
import pickle
import re
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent

# Timetable numbers run 100-128, 200-225, 300-318, 400-423, 500-518, 600-609, 700-706,
# 800-806 and 1000, with a letter for a few branches (116a, 122A, 200A, 221A).
SECTIA = re.compile(r"(\d{3,4})([A-Za-z]?)")


def ro_rel(tags):
    """(ref, name) of a timetable (Secția) relation, or None for every other relation."""
    ref = (tags.get("ref") or "").strip()
    name = (tags.get("name") or "").strip()
    m = SECTIA.fullmatch(ref)
    if not m:
        return None
    if name.startswith("Secția"):
        return ref.upper(), name
    if ref == "300" and not name and tags.get("route") == "railway":
        return "300", ""                      # the timetable's 300 carries no name
    return None


# The route part of a Secția relation's name, where the name cannot be read as it is: hyphens
# with no spaces ("Sântana-Brad") are separators here but part of place names elsewhere
# ("Bicaz-Chei"), and two names carry typos or notes.
ROUTE_FIX = {
    "116A": "Buziaș – Jamu Mare",
    "219": "Vâlcele – Râmnicu Vâlcea",
    "300": "București – Brașov – Cluj-Napoca – Oradea",
    "309": "Turda – Abrud",
    "314": "Oradea – Vașcău",
    "317": "Sântana – Brad",
    "402": "Oradea – Satu Mare – Halmeu",
    "407": "Târgu Mureș – Band – Lechința / Miheșu de Câmpie",
    "422": "Carei – Tiborszállás",
    "514": "Vama – Moldovița",
    "603": "Bârlad – Fălciu Nord",
    "806": "Dorobanțu / Palas – Sitorman",
}
_ROUTES = None


def ro_routes():
    """Timetable number key -> its route, from the extract's Secția relations (the one with
    the most ways where a number has two: 402's broad-gauge Porumbești - Halmeu)."""
    global _ROUTES
    if _ROUTES is not None:
        return _ROUTES
    ip = _ROOT / "data" / "proc" / "ro" / "infra.pkl"
    infra = {}
    if ip.exists():
        with open(ip, "rb") as f:
            infra = pickle.load(f)
    best = {}
    for tags, members in infra.values():
        got = ro_rel(tags)
        if not got or not got[1]:
            continue
        ref, name = got
        n = sum(1 for m in members if m[0] == "w")
        if n > best.get(ref, (0, ""))[0]:
            best[ref] = (n, name)
    _ROUTES = {}
    for ref, (_n, name) in best.items():
        r = re.sub(r"^Secția\s+\S+?:?\s+", "", name)
        r = re.sub(r"\s*\(.*?\)\s*", " ", r).strip(" .")
        r = re.sub(r"\s+[-–]\s+", " – ", r)
        _ROUTES[ref] = r
    _ROUTES.update(ROUTE_FIX)
    return _ROUTES


def is_magistrala(ref):
    return bool(re.fullmatch(r"[1-8]00|1000", ref or ""))


class RoName(str):
    """rinf.py writes a numbered line's name as COUNTRY["name"].format(ref=ref)."""
    def format(self, *args, **kw):
        ref = kw.get("ref", args[0] if args else "")
        route = ro_routes().get(ref.upper())
        word = "Magistrala" if is_magistrala(ref) else "Secția"
        return f"{word} {ref} {route}" if route else f"{word} {ref}"


class RoNameEn(str):
    def format(self, *args, **kw):
        ref = kw.get("ref", args[0] if args else "")
        route = ro_routes().get(ref.upper())
        return f"Line {ref} ({route})" if route else f"Line {ref}"


# CFR SA ids whose timetable number the Secția relations cannot give (module docstring).
# Each is the Secția relation's number by its end stations.
FIXED = {
    "110B": "108",    # Roșiori - Turnu Măgurele; relation 108 is 11 ways
    "111A": "109",    # Alexandria - Zimnicea (Roșiori - Zimnicea)
    "115": "113",     # Golenți - Poiana Mare
    "127": "120",     # Jebel - Liebling
    "128": "121",     # Jebel - Giera
    "134": "127",     # Cărpiniș - Ionel
    "135": "128",     # Jimbolia - Lovrin
    "212": "206",     # Șibot - Cugir
    "217": "213",     # Timișoara Est - Radna: relation 213 stops at Remetea Mică, 53%
    "221": "217",     # Nerău - Lovrin, the far end of Timișoara - Nerău
    "317": "403",     # Hărman - Întorsura Buzăului, 76% on 403
    "320": "307",     # Blaj - Praid, 63% on 307
    "324": "314",     # Holod - Oradea Est, the near end of Oradea - Vașcău
    "332": "313",     # Grăniceri - Nădab
    "334": "312",     # Cheresig - Oradea Vest
    "401A": "417",    # Bixad - Botiz (Satu Mare - Bixad)
    "423": "410",     # Vișeu de Jos - Borșa
    "424": "418",     # Ilva Mică - Rodna Veche
    "429": "411",     # Sighetu Marmației - Câmpulung la Tisa (to Teresva)
    "503": "507",     # Panciu - Mărășești
    "510": "512",     # Dorohoi - Leorda
    "512": "516",     # Dornișoara - Floreni
    "513": "514",     # Moldovița - Vama
    "520": "518",     # Siret - Dornești
    "608": "604",     # Crasna - Huși
    "703": "706",     # Căciulați - Snagov Plajă; no relation (ro.wikipedia, Wikidata 706)
    "810": "803",     # Medgidia - Negru Vodă
    "307": "303",     # I. L. Caragiale - Filipeștii de Pădure - Gura Palangii; unnumbered, its
                      # line id would be the same hash as the timetable's 307 (Blaj - Praid)
    "200B": "200",    # Bârzava Nouă - Văradia, 200's new alignment beside the old one
    "806": "701",     # Post Amara - Slobozia Veche: 701's trains run Post Amara - Slobozia -
                      # Țăndărei; filed under 802 by the relation, it cut 701 in two
    # Pieces the passenger trains run over that lie on two relations at once
    "301N": "101",    # Chitila - București Nord, where CFR SA's 101 (Craiova - Chitila) ends
    "143B": "221",    # Ram. Filiași - Filiași, where 143 (Târgu Jiu - Ram. Filiași) ends
    "200R": "200A",   # Coșlariu - Ram. Coșlariu, between 201 (Teiuș - Coșlariu) and 200A
    "201": "200A",    # Teiuș - Coșlariu
    # Ploiești: the trains run Brazi - Ploiești Triaj and on to Ploiești Vest (300) or
    # Ploiești Sud (500); both pieces also lie on the 1000 relation (București - Ploiești),
    # which shares 300's track all the way and so is no register line of its own
    "304A": "300",
    "304I": "500",
}

# CFR SA ids the timetable splits: the points to cut at, and the timetable number of each
# piece, named by a point inside it (RINF's spelling, no diacritics). Read off the Secția
# relations each piece lies on (ro_sources.md). A piece marked SKIP is left out.
SKIP = "skip"
SPLIT = {
    "100A": (["Timisoara Nord"], {"Craiova": "100", "Jimbolia": "119"}),
    "123": (["Buzias HM"], {"Lugoj": "116", "Jamu Mare hcv.": "116A"}),
    "130": (["Oravita"], {"Iam hcv.": "124", "Berzovia HM": "123"}),
    "200A": (["Vintu de Jos"], {"Ram. Coslariu": "200A", "Aurel Vlaicu HM": "200"}),
    "203": (["Sibiu", "Podu Olt"], {"Copsa Mica": "208", "Talmaciu HM": "200", "Piatra Olt": "201"}),
    "219": (["Periam HM"], {"Aradu Nou": "216", "Ram. Satu Nou": "217"}),
    "222": (["Periam HM"], {"Sanandrei HM": "217", "Valcani hcv.": "216"}),
    "316": (["Deda"], {"Brasov": "400", "Razboieni": "405"}),
    "412": (["Dej Calatori"], {"Baia Mare": "400", "Apahida HM": "401"}),
    "511": (["Gura Humorului HM"], {"Ram. Floreni": "502", "Darmanesti": "513"}),
    "804A": (["Tandarei"], {"Fetesti": "702", "Slobozia Noua HM": "701"}),
    # Ciulnița - Slobozia Veche is 802 (Slobozia - Călărași); Slobozia Veche - Slobozia Nouă
    # carries 701 on from Post Amara (806) to Țăndărei (804A)
    "807A": (["Slobozia Veche"], {"Ciulnita": "802", "Slobozia Noua HM": "701"}),
    # Târgu Jiu - Ram. Filiași is 221 (on to Filiași over 143B); the 1 km on to Gura Motrului
    # leads only to Turceni's freight curves, and as part of 221 it made Ram. Filiași a branch
    # point, which cost 221 its 14 km into Filiași (junction-ended, and no OSM route there)
    "143": (["Ram. Filiasi"], {"Targu Jiu": "221", "Gura Motrului HM": SKIP}),
}
def split_ids(secs, points):
    """Re-file the sections of every id in SPLIT under "<id>:<number>". Returns log lines."""
    out = []
    for lid, (cuts, nums) in SPLIT.items():
        mine = [s for s in secs if s["base"] == lid]
        name = {op: points.get(op, {}).get("name") for s in mine for op in (s["a"], s["b"])}
        cut_ops = {op for op, n in name.items() if n in cuts}
        missing = set(cuts) - {name[op] for op in cut_ops}
        if missing:
            out.append(f"SPLIT {lid}: no point named {sorted(missing)}; not split")
            continue
        parent = {}

        def find(x):
            while parent.setdefault(x, x) != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        # sections are joined through every point they share except the cut points
        for s in mine:
            for op in (s["a"], s["b"]):
                if op not in cut_ops:
                    parent[find(("s", s["sol"]))] = find(("p", op))
        here = defaultdict(set)
        for s in mine:
            here[find(("s", s["sol"]))] |= {name[s["a"]], name[s["b"]]}
        n = defaultdict(int)
        for s in mine:
            got = {num for p, num in nums.items() if p in here[find(("s", s["sol"]))]}
            if len(got) != 1:
                out.append(f"SPLIT {lid}: {name[s['a']]} - {name[s['b']]} is in a piece named "
                           f"by {sorted(got) or 'no point'}; left on {lid}")
                continue
            num = got.pop()
            s["line"] = s["base"] = f"{lid}:{num}"
            n[num] += 1
        out.append(f"{lid} split into " + ", ".join(f"{r} ({k} sections)" for r, k in n.items()))
    return out


# A halt's kind, written after its name in RINF ("Aghires PO") and sometimes in OSM.
KIND = re.compile(r"\s+(PO|P\.O\.|HM|PM|P\.M\.|hcv\.?|Hc\.?|P\.\s?Aj\.\s?M\.)\s*$")
# CFR's abbreviations, written out as OSM writes the names.
ABBREV = [(r"\bG-ral\.?\s*", "General "), (r"\bGral\.\s*", "General "),
          (r"\bGen\.\s*", "General "), (r"\bTr\.\s*", "Turnu "), (r"\bC-tin\b", "Constantin"),
          (r"^I\.\s*L\.\s*", "Ion Luca "), (r"\bMaresal\b", "Mareșal")]


# RINF names OSM spells otherwise, where neither prefix nor distance finds the station.
ALIAS = {"Podu Iloaiei": "Podu Iloaei"}


def clean_name(name):
    """"Aghires PO" -> "Aghires (PO)", "G-ral T.Mosoiu PO" -> "General T.Mosoiu (PO)"."""
    n = (name or "").strip()
    n = ALIAS.get(n, n)
    for a, b in ABBREV:
        n = re.sub(a, b, n)
    m = KIND.search(n)
    if m:
        n = f"{n[:m.start()].strip()} ({m.group(1)})"
    return n


RELOCATE_M = 6000


def relocate(secs, points):
    """Drop the RINF coordinate of a passenger point whose station is too far for rinf.py to
    find (module docstring, STATIONS): no OSM station of a matching name within `name_m`, but
    one of exactly its name within RELOCATE_M. rinf.py then places the point at the OSM station
    of its exact name (the one nearest a placed neighbour where the name repeats). An exact
    name only: by prefix "Caransebes Cazarma (PO)" would land on Caransebeș."""
    import build_model as bm
    import rinf
    _w, _r, stops, *_ = bm.load("ro", lambda m: None)
    ost = rinf.osm_stations(stops)
    idx = rinf.StationIndex(ost)
    used = {op for s in secs for op in (s["a"], s["b"])}
    # A station matched by prefix only counts if no other RINF point has its exact name: RINF's
    # Oradea lies 0.5 km from Oradea Est and 4.1 km from Oradea, and by prefix took Oradea Est.
    taken = defaultdict(set)
    for op, p in points.items():
        if op in used:
            for k in rinf.name_variants(p.get("name")):
                taken[k].add(op)
    moved = []
    for op, p in points.items():
        if p.get("type") not in rinf.PASSENGER_TYPES or "lon" not in p or op not in used:
            continue
        keys = rinf.name_variants(p.get("name"))
        near = sorted(idx.within(p["lon"], p["lat"], RELOCATE_M))

        def found(sid):
            ks = ost[sid]["keys"]
            if keys & ks:
                return True
            return (rinf.names_match(keys, ks)
                    and not any(taken.get(k, set()) - {op} for k in ks))
        if any(d <= COUNTRY["name_m"] and found(sid) for d, sid in near):
            continue
        hit = next(((d, sid) for d, sid in near if keys & ost[sid]["keys"]), None)
        if hit:
            del p["lon"], p["lat"]
            moved.append(f"{p['name']} {hit[0] / 1000:.1f}")
    return [f"{len(moved)} points lose their RINF coordinate, which lies this many km from the "
            f"OSM station of their name: {', '.join(sorted(moved))}"]


# Halts whose OSM station the trace cannot use (measured 2026-10-01): Lalașinț's is 1 km off,
# on 200's other alignment (Bata - Lalașinț traced 16.9 km for RINF's 3.9), and 200 lost
# Văradia - Lalașinț, 12.3 km. As a junction it is merged away and the line runs through.
NOT_STOPS = {"Lalasint (PO)"}


def ro_fix(secs, points):
    """rinf.py's `fix` hook (module docstring): split ids, clean point names, then relocate
    the points whose coordinate is too far off. Returns log lines. SPLIT is read on RINF's own
    names, so the split comes first."""
    out = split_ids(secs, points)
    n = 0
    for p in points.values():
        new = clean_name(p.get("name"))
        if new != (p.get("name") or "").strip():
            p["name"] = new
            n += 1
    out.append(f"{n} point names cleaned (halt kind in brackets, abbreviations written out)")
    for p in points.values():
        if p.get("name") in NOT_STOPS:
            p["type"] = "80"
    out += relocate(secs, points)
    return out


def ro_ref(lid):
    """The number after "<id>:", for the pieces `split_ids` made; None for CFR SA's own ids."""
    m = re.fullmatch(r"[^:]+:(\w+)", lid or "")
    return m.group(1) if m else None


# CFR SA ids that are never lines: Bucharest's freight belt and yard links (301x), Ploiești's
# (304x), Brașov's (314x), Galați's (706x, 708x), Constanța's port (813A-C, 814x, 817A, 818A),
# industrial branches and connecting curves no passenger train uses. Read off the RINF chains
# and the absence of any Secția relation on them (ro_sources.md lists each).
NOT_LINES = {
    "100B", "100Ba", "101A", "102", "102A", "102B", "105A", "106B", "113A", "114", "121",
    "133A", "133B", "133C", "133D", "140", "143A", "200/210", "200/300",
    "200D", "202A", "202B", "202C", "203A", "207", "209", "220", "227",
    "301A", "301B", "301Ba", "301Bb", "301Bc", "301D", "301F", "301FD2",
    "301G", "301I", "301J", "301K", "301L", "301M", "301M1", "301P1", "301Q", "301S", "301T",
    "301V", "301V1", "301W", "301X", "301Y", "301Z", "301Z2", "302/303", "304C", "304E",
    "304F", "304N", "304O", "314A", "314B", "314C", "314F", "316A", "326", "335", "400A",
    "404C", "413", "415", "417", "500R", "602A", "603", "605B", "605R", "701A",
    "702S", "706A", "706B", "706K", "706K+F", "706M+L", "708A", "708C", "708D", "708E",
    "708G", "708H", "801B", "810A", "813A", "813B", "813C", "814", "814A", "814B",
    "817A", "818A",
}


def ro_skip(lid):
    return lid in NOT_LINES or lid.endswith(f":{SKIP}")


COUNTRY = {
    "iso3": "ROU", "wikidata": None, "langs": ["ro"],
    "osm_rel": ro_rel, "fixed": FIXED, "ref": ro_ref, "rule_certain": True,
    "fix": ro_fix, "skip_line": ro_skip, "name_m": 1500,
    "name": RoName("Secția {ref} <route>"), "name_en": RoNameEn("Line {ref} (<route>)"),
    "im": {"0053_IM": "CFR"},
}
