"""Italy: RINF carries RFI (0083_IM, 3,203 of 3,650 sections) and nine regional infrastructure
managers: FERROVIENORD (the Milano - Saronno network and Brescia - Iseo - Edolo), FER (Emilia-
Romagna), Ferrovie del Sud Est, Ferrotramviaria (Bari - Barletta), EAV (only Cancello -
Benevento and the Alifana, not the Circumvesuviana, Cumana or Circumflegrea), La Ferroviaria
Italiana (Arezzo), Ferrovie del Gargano, GTT (the Canavesana, filed twice: RFI's C262 is the
same track) and Ferrovie Udine Cividale. Not in RINF, so OSM lines or nothing: Trentino
Trasporti (Trento - Malè), SAD/STA (Merano - Malles, Renon), FAL, Ferrovie della Calabria, ARST,
the Circumetnea, EAV's Vesuviana and Flegree lines, ASTRAL's Roma - Lido and Roma - Viterbo,
Genova - Casella, the Vigezzina, Sangritana (TUA). it_sources.md has the list.

NUMBERS. RFI's ids are its own internal line codes: F1-F2 ... F77-F78 for the fundamental
lines (a pair, one number per direction), C1 ... C262 for complementary lines, N1 ... N8 for the
eight city nodes, AV/AC1-AV/AC2 ... for high speed. Nobody rides "line C121"; Italian lines are
known by their ends ("Ferrovia Empoli-Siena"). So lines carry no ref (`ref_display` writes none)
and `id_name` gives RFI-style names from NAMES: a complementary line by its ends ("Empoli –
Siena"); a fundamental line by the cities whose nodes it runs between ("Milano – Bologna" is
Milano Rogoredo - PM Lavino); a node line "Nodo di Milano"; high speed "AV Roma – Napoli". OSM's
route=railway refs are ignored (`osm_ref`). The regional managers' ids are names already
("Lecce-Gallipoli", "SALV" for Saronno - Laveno) and are mapped the same way.

GROUPS (`fixed` on GROUP): RFI files several it.wikipedia lines in pieces split at junctions;
those that are one line to a rider are joined: Domodossola - Novara (C29-C34), Mantova -
Monselice, Paola - Cosenza, Battipaglia - Potenza - Metaponto, Castel Bolognese - Ravenna,
Ferrara - Ravenna - Rimini, Cremona - Mantova, Vicenza - Treviso, Lecco - Sondrio - Tirano,
Savigliano - Saluzzo - Cuneo, Decimomannu - Iglesias, and FER's Modena - Sassuolo.

FIXES TO RINF (`it_fix`):
- Point type 140 (where two infrastructure managers meet) is read as a station: in Italy it is
  Foggia, Taranto, Lecce, Modena, Bologna Centrale, Udine, Cancello and 20 more; typed 140 they
  were never stops, and build_model dropped the junction-ended sections into them that no OSM
  route covers (Apricena - Foggia, 39 km of the Adriatica).
- Id "0000" (46 sections, 209 km: border stubs, gaps, the new Napoli - Cancello line) is shared
  out to the lines its pieces belong to (ZERO). load_rinf's split into pieces is then redone.
- Altavilla Tavernelle - Vicenza (7.667 km, it.wikipedia chainage) is added: RINF lacks it and
  Verona - Padova came out in two pieces.
- EAV's two sections in metres (955.0, 683.0) are read as metres.
- FSE's Triggiano and Capurso lie 30-40 km off in RINF and are placed by name instead.
- F61-F62 files C141's Vairano - Sesto Campano - Venafro too; dropped there (DUP_DROP).
- Point names in capitals are written in the usual case (`title_it`), accents restored
  ("MONDOVI`" is Mondovì). Matching is case- and accent-blind, so no match changes.

LEFT OUT (`skip_line`, SKIP): GTT's copy of the Canavesana, the Messina strait ferry berths
(M), and the Padova Interporto freight pieces of F25-F26, F31-F32 and C82.
"""
import math
import re
from collections import Counter, defaultdict

# RINF's line id "0000" is a bag of 46 sections (209 km) that belong to other lines or stand
# alone: border stubs, gaps in a line, the new Napoli - Cancello line. Each piece is given the
# id of the line it belongs to, by the uopids of its points (`it_fix`). A uopid set maps to the
# id its piece joins; "NEW:<key>" makes it a line of its own.
ZERO = [
    ({"IT03466", "EU00150", "EU00151"}, "NEW:OPICINA"),     # Villa Opicina - Slovenian border
    ({"IT00096", "IT00119"}, "C260"),                       # Germagnano - Ceres
    ({"IT09213", "IT09042", "IT09041", "IT09305"}, "NEW:NABA"),   # Napoli - Cancello - Dugenta
    ({"IT08209", "IT08029"}, "F53-F54"),                    # Orte - LL/DD junction
    ({"IT04501", "EU00126"}, "F9-F10"),                     # Ventimiglia - French border
    ({"IT12338", "IT12244"}, "C178"),                       # Bicocca - Catenanuova
    ({"IT01113", "EU00152"}, "F15-F16"),                    # Luino - Swiss border (Pino)
    ({"IT01003", "EU00153"}, "F13-F14"),                    # Domodossola - Iselle
    ({"IT07305", "IT07304"}, "C126"),                       # Pollenza - Tolentino
    ({"IT03015", "EU00116"}, "F37-F38"),                    # Tarvisio - Austrian border
    ({"IT02028", "IT02094"}, "F23-F24"),                    # Sommacampagna - Verona
    ({"IT02114", "EU00114"}, "C67"),                        # San Candido - Austrian border
    ({"IT02001", "EU00115"}, "F27-F28"),                    # Brennero - Austrian border
    ({"IT00139", "IT00017"}, "C34"),                        # Cressa Fontaneto - Baragge
    ({"IT08119", "IT08230"}, "F55-F56"),                    # Orte - Capena (Direttissima)
    ({"IT05014", "IT01854"}, "FER206"),                     # Parma - Parma Est junction
]


# (id, other id): sections of the first that the second also has are the second's.
DUP_DROP = [("F61-F62", "C141")]


def _crow_km(points, a, b):
    pa, pb = points.get(a, {}), points.get(b, {})
    if "lon" not in pa or "lon" not in pb:
        return None
    dx = (pb["lon"] - pa["lon"]) * math.cos(math.radians(pa["lat"])) * 111.32
    dy = (pb["lat"] - pa["lat"]) * 110.57
    return math.hypot(dx, dy)


SMALL = {"di", "del", "della", "delle", "dei", "degli", "dell", "e", "al", "alla", "sul",
         "sulla", "in", "a", "lato", "per"}
KEEP_UPPER = re.compile(r"^(?:PM|PC|PP|AV|AC|LL|DD|FS|FM|FA|FT|UM\d|PB|IM|BA|BN|PZ|VR|NA|"
                        r"LT|TS|UD|MO\d|CP|RFI|FER|FN|SM|SC|II|III|IV|N|S|M|GR)$")


def title_it(name):
    """RFI writes its points in capitals ("BIVIO/PC S.LUCIA", "LENTINI DIRAMAZIONE"); the ones
    that show on the map (junctions, and stations with no OSM station) are written in the
    usual case. Matching is case-blind, so this changes no match."""
    # RFI writes a final accent as an apostrophe or backtick: MONDOVI`, CANICATTI'.
    name = re.sub(r"([AEIOU])[`'](?=$|[\s.,/)-])",
                  lambda m: "ÀÈÌÒÙ"["AEIOU".index(m.group(1))], name)
    parts = re.split(r"([^0-9A-Za-zÀ-ÿ]+)", name)
    out = []
    for i, p in enumerate(parts):
        if i % 2 or not p.isalpha():
            out.append(p)
            continue
        nxt = parts[i + 1] if i + 1 < len(parts) else ""
        first = not any(x.isalpha() for x in parts[:i])
        if p == "D" and nxt.startswith("'"):
            out.append("D" if first else "d")          # Rocca d'Evandro
        elif KEEP_UPPER.match(p):
            out.append(p)
        elif not first and p.lower() in SMALL:
            out.append(p.lower())
        else:
            out.append(p[0] + p[1:].lower())
    return "".join(out)


def it_fix(secs, points):
    out = []
    n_tc = 0
    for p in points.values():
        nm = p.get("name") or ""
        if nm and nm == nm.upper() and any(c.isalpha() for c in nm):
            p["name"] = title_it(nm)
            n_tc += 1
    out.append(f"{n_tc} point names in capitals written in the usual case")
    # a section length in metres (EAV: Benevento Rione Libertà - Benevento Appia 955.0)
    for s in secs:
        c = _crow_km(points, s["a"], s["b"])
        if s["km"] and s["km"] > 50 and c is not None and c < s["km"] / 20:
            out.append(f"{s['base']} {s['label']}: {s['km']} read as metres")
            s["km"] /= 1000
    for p in points.values():
        if p.get("type") == "70%2520":
            p["type"] = "70"
        # 140 is a point where two infrastructure managers meet, and in Italy that is often a
        # big station: Foggia, Taranto, Lecce, Modena, Bologna Centrale, Udine, Cancello.
        # Typed so, it was never a stop, every section into it ended at a "junction", and
        # build_model dropped the ones no OSM route covers (Apricena - Foggia, 39 km of the
        # Adriatica). Read as a station; one with no OSM station of its name stays a junction.
        if p.get("type") == "140":
            p["type"] = "10"
        # FSE's Triggiano and Capurso are 30-40 km from where they are (RINF puts them near
        # Bitonto); with no coordinate rinf.py places them at the OSM station of their name.
        if p.get("uopid") in ("IT13117", "IT13118"):
            p.pop("lon", None)
            p.pop("lat", None)
            out.append(f"{p['name']}: RINF's coordinate dropped, placed by name")
    # id 0000, piece by piece
    uop = lambda op: points.get(op, {}).get("uopid")
    n = 0
    nodes_of = {}
    for s in secs:
        if s["base"] == "0000":
            nodes_of.setdefault(s["line"], set()).update((uop(s["a"]), uop(s["b"])))
    for s in secs:
        if s["base"] != "0000":
            continue
        piece = s["line"]
        nodes = nodes_of[piece]
        for keys, target in ZERO:
            if keys <= nodes:
                s["line"] = s["base"] = target.replace("NEW:", "")
                n += 1
                break
        else:
            out.append(f"0000 piece {piece} matched nothing: {s['label']}")
    out.append(f"{n} sections of id 0000 given to the lines they belong to")
    # Milano - Venezia: RINF has no section Altavilla Tavernelle - Vicenza, so F25-F26 came
    # out in two pieces. it.wikipedia's chainage: Altavilla-Tavernelle 191+471, Vicenza 199+138.
    by_uop = {p.get("uopid"): op for op, p in points.items()}
    a, b = by_uop.get("IT02444"), by_uop.get("IT02446")
    if a and b:
        secs.append({"sol": "it:altavilla-vicenza", "line": "F25-F26", "base": "F25-F26",
                     "a": a, "b": b, "km": round(199.138 - 191.471, 3), "im": "0083_IM",
                     "label": "ALTAVILLA TAVERNELLE - VICENZA (added in it.py)"})
        out.append("added Altavilla Tavernelle - Vicenza to F25-F26, 7.667 km")
    # The same section filed under two ids. Logged; dropped from the first id where it is
    # another line's track: F61-F62 (Roma - Cassino - Napoli) files Vairano - Sesto Campano -
    # Venafro, which is C141 Vairano - Isernia, beside its own Rocca d'Evandro - Venafro link.
    pair_ids = defaultdict(set)
    for s in secs:
        pair_ids[frozenset((uop(s["a"]), uop(s["b"])))].add(s["base"].split("#")[0])
    dups = Counter(tuple(sorted(v)) for v in pair_ids.values() if len(v) > 1)
    if dups:
        out.append("sections filed under two ids: " + ", ".join(
            f"{'/'.join(k)} {n}" for k, n in dups.most_common()))
    n0 = len(secs)
    secs[:] = [s for s in secs if not any(
        s["base"].split("#")[0] == a and b in pair_ids[frozenset((uop(s["a"]), uop(s["b"])))]
        for a, b in DUP_DROP)]
    if n0 != len(secs):
        out.append(f"{n0 - len(secs)} sections dropped as another id's track (DUP_DROP)")
    # load_rinf split ids into connected pieces before the 0000 sections joined them up, so
    # pieces are worked out again here. A piece of an id that is still in several pieces
    # gets its own base ("C10#2"), so that it is named on its own (NAMES).
    by_id = defaultdict(list)
    for s in secs:
        s["line"] = s["base"] = s["base"].split("#")[0]
        by_id[s["base"]].append(s)
    for lid, ss in by_id.items():
        parent = {}

        def find(x):
            while parent.setdefault(x, x) != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for s in ss:
            parent[find(s["a"])] = find(s["b"])
        comps = defaultdict(list)
        for s in ss:
            comps[find(s["a"])].append(s)
        if len(comps) < 2:
            continue
        order = sorted(comps.values(), key=lambda c: min(
            points.get(op, {}).get("uopid") or op for s in c for op in (s["a"], s["b"])))
        for k, c in enumerate(order):
            for s in c:
                s["line"] = s["base"] = f"{lid}#{k + 1}"
        ends = [" / ".join(sorted({points.get(op, {}).get("name", "?") for s in c
                                   for op in (s["a"], s["b"])}))[:120] for c in order]
        out.append(f"{lid} is {len(comps)} pieces: " + " | ".join(
            f"#{k + 1} {e}" for k, e in enumerate(ends)))
    return out


# RINF ids that are one line to a rider -> the key the group is named under.
GROUP = {
    # Domodossola - Novara: RFI files it in six pieces, C29-C32 being its own single track
    # beside the Simplon line between Domodossola and Cuzzago.
    "C29": "C29", "C30": "C29", "C31": "C29", "C32": "C29", "C33": "C29", "C34": "C29",
    "FER202": "FER202", "FER202_MO": "FER202",            # Modena - Sassuolo
    # One it.wikipedia line that RFI files in pieces split at junctions.
    "C73": "C73", "C74": "C73", "C75": "C73", "C213": "C73",      # Mantova - Monselice
    "C164": "C164", "C167": "C164",                               # Paola - Cosenza
    "C136": "C136", "C155": "C136",                               # Battipaglia - Metaponto
    "C106": "C106", "C107": "C106", "C108": "C106",               # Castel Bolognese - Ravenna
    "C99": "C99", "C100": "C99", "C101": "C99", "C102": "C99",    # Ferrara - Rimini
    "C62": "C62", "C63": "C62",                                   # Cremona - Mantova
    "C86": "C86", "C87": "C86", "C88": "C86",                     # Vicenza - Treviso
    "C40": "C40", "C39": "C40", "C37": "C40",                     # Lecco - Tirano
    "C245": "C245", "C243": "C245",                               # Savigliano - Cuneo
    "C189": "C189", "C190": "C189",                               # Decimomannu - Iglesias
}

# Pieces of ids that are in several pieces once id 0000 is shared out, and freight pieces left
# out (`skip_line`): Padova Interporto, filed under three ids, and the Messina strait ferry
# berths (M).
SKIP = {"SETTIMO_TORINESE-RIVAROLO_CANAVESE", "M", "M#1", "M#2",
        "F25-F26#2", "F31-F32#1", "C82#2"}

NAMES = {
    # high speed (AV/AC)
    "AV/AC1-AV/AC2": "AV Roma – Napoli",
    "AV/AC3-AV/AC4": "AV Torino – Milano",
    "AV/AC5-AV/AC6": "AV Milano – Bologna",
    "AV/AC7-AV/AC8": "AV Bologna – Firenze",
    "AV/AC9-AV/AC10": "AV Treviglio – Brescia",
    # fundamental lines, named by the cities whose node (N1-N8) they start from
    "F1-F2": "Torino – Modane",
    "F3-F4": "Torino – Arquata Scrivia",
    "F5-F6": "Milano – Arquata Scrivia",
    "F7-F8": "Genova – Savona",
    "F9-F10": "Savona – Ventimiglia",
    "F11-F12": "Milano – Torino",
    "F13-F14": "Gallarate – Domodossola – Iselle",
    "F15-F16": "Gallarate – Luino – Pino",
    "F17-F18": "Seregno – Como – Chiasso",
    "F19-F20": "Arquata Scrivia – Genova",
    "F21-F22": "Alessandria – Piacenza",
    "F23-F24": "Milano – Verona",
    "F25-F26": "Verona – Padova",
    "F27-F28": "Brennero – Verona",
    "F29-F30": "Verona – Bologna",
    "F31-F32": "Bologna – Padova",
    "F33-F34": "Padova – Venezia",
    "F35-F36": "Venezia – Trieste",
    "F37-F38": "Venezia – Udine – Tarvisio",
    "F39-F40": "Udine – Gorizia – Monfalcone",
    "F41-F42": "Milano – Bologna",
    "F43-F44": "Bologna – Firenze (Direttissima)",
    "F45-F46": "Genova – Pisa",
    "F47-F48": "Parma – La Spezia",
    "F49-F50": "Firenze – Pisa",
    "F51-F52": "Pisa – Roma",
    "F53-F54": "Firenze – Roma (linea lenta)",
    "F55-F56": "Firenze – Roma (Direttissima)",
    "F57-F58": "Roma – Formia – Napoli",
    "F59-F60": "Villa Literno – Pozzuoli",
    "F61-F62": "Roma – Cassino – Napoli",
    "F63-F64": "Bologna – Ancona",
    "F65-F66": "Orte – Ancona",
    "F67-F68": "Ancona – Foggia",
    "F69-F70": "Foggia – Bari",
    "F71-F72": "Dugenta – Benevento – Foggia",
    "F73-F74": "Napoli – Salerno",
    "F75-F76": "Salerno – Paola",
    "F77-F78": "Paola – Reggio Calabria",
    # node lines: RFI's own unit, every line inside one city's node
    "N1": "Nodo di Torino", "N2": "Nodo di Milano", "N3": "Nodo di Venezia",
    "N4": "Nodo di Genova", "N5": "Nodo di Bologna", "N6": "Nodo di Firenze",
    "N7": "Nodo di Roma", "N8": "Nodo di Napoli",
    # complementary lines
    "C1": "Castagnole delle Lanze – Asti", "C2": "Chivasso – Ivrea",
    "C4": "Casale Popolo – Casale Monferrato", "C5": "Chivasso – Casale Monferrato",
    "C6": "Valenza – Casale Monferrato", "C7": "Valenza – Alessandria",
    "C8": "Torino – Pinerolo", "C9": "Fossano – Cuneo", "C10": "Cuneo – Limone – Ventimiglia",
    "C11": "Trofarello – Fossano", "C12": "Fossano – Mondovì", "C13": "Mondovì – Ceva",
    "C14": "Ceva – San Giuseppe di Cairo", "C15": "Carmagnola – Bra",
    "C16": "Alessandria – Cantalupo", "C17": "Ovada – Acqui Terme",
    "C18": "Alessandria – Acqui Terme", "C19": "Acqui Terme – San Giuseppe di Cairo",
    "C20": "Vercelli – Mortara", "C21": "Pavia – Mortara", "C22": "Arona – Oleggio",
    "C23": "Oleggio – Novara", "C24": "Novara – Vignale", "C25": "Novara – Mortara",
    "C26": "Mortara – Torreberetti", "C27": "Torreberetti – Valenza",
    "C28": "Pavia – Torreberetti", "C29": "Domodossola – Novara",
    "C35": "Novara – Romagnano Sesia", "C36": "Oleggio – Laveno Mombello",
    "C37": "Sondrio – Tirano", "C38": "Colico – Chiavenna", "C39": "Colico – Sondrio",
    "C40": "Lecco – Colico", "C41": "Como – Molteno", "C42": "Monza – Molteno",
    "C43": "Molteno – Lecco", "C44": "Lecco – Ponte San Pietro",
    "C45": "Carnate – Calolziocorte", "C46": "Monza – Carnate",
    "C47": "Seregno – Ponte San Pietro", "C48": "Ponte San Pietro – Bergamo – Rovato",
    "C49": "Gallarate – Varese – Porto Ceresio", "C50": "Bivio Adda – Treviglio Ovest",
    "C51": "Treviglio – Bergamo", "C52": "Olmeneta – Brescia", "C53": "Olmeneta – Cremona",
    "C54": "Cremona – Castelvetro", "C55": "Castelvetro – Fidenza",
    "C56": "Castelvetro – Piacenza", "C57": "Treviglio – Olmeneta", "C58": "Milano – Mortara",
    "C60": "Casalpusterlengo – Pavia", "C61": "Codogno – Cremona", "C62": "Cremona – Piadena",
    "C63": "Piadena – Mantova", "C64": "San Zeno – Brescia", "C65": "Brescia – Piadena",
    "C66": "Piadena – Parma", "C67": "Fortezza – San Candido", "C68": "Bolzano – Merano",
    "C69": "Vicenza – Schio", "C70": "Verona – Mantova", "C71": "Mantova – Suzzara",
    "C72": "Suzzara – Modena", "C73": "Mantova – Nogara", "C74": "Nogara – Cerea",
    "C75": "Cerea – Legnago", "C76": "Ponte nelle Alpi – Calalzo",
    "C77": "Ponte nelle Alpi – Belluno", "C78": "Belluno – Montebelluna",
    "C79": "Castelfranco Veneto – Montebelluna", "C80": "Castelfranco Veneto – Camposampiero",
    "C81": "Camposampiero – Vigodarzere", "C82": "Vigodarzere – Padova",
    "C83": "Montebelluna – Treviso", "C84": "Treviso – Portogruaro",
    "C85": "Conegliano – Ponte nelle Alpi", "C86": "Vicenza – Cittadella",
    "C87": "Cittadella – Castelfranco Veneto", "C88": "Castelfranco Veneto – Treviso",
    "C89": "Bassano del Grappa – Cittadella", "C90": "Cittadella – Camposampiero",
    "C91": "Castelfranco Veneto – Bassano del Grappa",
    "C92": "Castelfranco Veneto – Maerne", "C93": "Savona – San Giuseppe di Cairo",
    "C94": "Udine – Palmanova", "C95": "Palmanova – Cervignano",
    "C96": "San Giuseppe di Cairo – Ferrania", "C97": "Ferrania – Savona",
    "C98": "Genova – Ovada", "C99": "Ferrara – Portomaggiore",
    "C100": "Portomaggiore – Lavezzola", "C101": "Lavezzola – Ravenna",
    "C102": "Ravenna – Rimini", "C103": "Faenza – Granarolo Faentino",
    "C104": "Granarolo Faentino – Lugo", "C105": "Lugo – Lavezzola",
    "C106": "Castel Bolognese – Lugo", "C107": "Lugo – Russi", "C108": "Russi – Ravenna",
    "C109": "Granarolo Faentino – Russi", "C110": "Pontassieve – Borgo San Lorenzo",
    "C111": "Faenza – Borgo San Lorenzo", "C112": "Firenze – Borgo San Lorenzo",
    "C113": "Lucca – Viareggio", "C115": "Pistoia – Porretta Terme",
    "C116": "Pistoia – Lucca", "C117": "Pisa – Lucca", "C118": "Prato – Pistoia",
    "C119": "Lucca – Aulla", "C120": "Campiglia Marittima – Piombino",
    "C121": "Empoli – Siena", "C122": "Siena – Asciano", "C123": "Asciano – Chiusi",
    "C124": "Porto d'Ascoli – Ascoli Piceno", "C125": "Pescara – Sulmona",
    "C126": "Civitanova Marche – Albacina", "C127": "Terontola – Foligno",
    "C129": "Roma – Viterbo", "C130": "Campoleone – Nettuno",
    "C131": "Sulmona – Avezzano", "C132": "Guidonia – Avezzano",
    "C133": "Avezzano – Roccasecca", "C134": "Ciampino – Albano Laziale",
    "C135": "Ciampino – Velletri", "C136": "Battipaglia – Potenza",
    "C137": "Caserta – Aversa", "C138": "Villa Literno – Cancello",
    "C139": "Carpinone – Campobasso", "C140": "Carpinone – Isernia",
    "C141": "Vairano – Isernia", "C143": "Cancello – Nola – Sarno",
    "C144": "Sarno – Bivio Santa Lucia", "C145": "Nocera Inferiore – Codola",
    "C146": "Nocera Inferiore – Salerno", "C147": "Salerno – Arechi",
    "C148": "Salerno – Mercato San Severino", "C150": "Bari – Brindisi",
    "C151": "Brindisi – Lecce", "C152": "Bari – Bitritto",
    "C153": "Foggia – Rocchetta Sant'Antonio", "C154": "Rocchetta Sant'Antonio – Potenza",
    "C155": "Potenza – Metaponto", "C156": "Metaponto – Taranto",
    "C157": "Taranto – Brindisi", "C158": "Bari – Gioia del Colle",
    "C159": "Gioia del Colle – Taranto", "C160": "Sibari – Metaponto",
    "C161": "Sibari – Catanzaro Lido", "C162": "Catanzaro Lido – Reggio Calabria",
    "C163": "Lamezia Terme – Catanzaro Lido", "C164": "Paola – Castiglione Cosentino",
    "C165": "San Lucido – Bivio Pantani", "C166": "Bivio Settimo – Bivio Sant'Antonello",
    "C167": "Castiglione Cosentino – Cosenza", "C168": "Sibari – Castiglione Cosentino",
    "C169": "Messina – Catania", "C170": "Catania – Lentini", "C171": "Lentini – Siracusa",
    "C172": "Lentini – Caltagirone", "C173": "Gela – Ragusa – Modica",
    "C174": "Modica – Siracusa", "C175": "Fiumetorto – Messina",
    "C176": "Palermo – Fiumetorto", "C177": "Roccapalumba – Caltanissetta Xirbi",
    "C178": "Caltanissetta Xirbi – Catania", "C180": "Fiumetorto – Roccapalumba",
    "C181": "Palermo – Punta Raisi", "C182": "Roccapalumba – Aragona Caldare",
    "C183": "Aragona Caldare – Agrigento", "C184": "Palermo – Giachery",
    "C185": "Ozieri Chilivani – Olbia", "C186": "Ozieri Chilivani – Sassari",
    "C187": "Decimomannu – Ozieri Chilivani", "C188": "Cagliari – Decimomannu",
    "C189": "Decimomannu – Villamassargia", "C190": "Villamassargia – Iglesias",
    "C191": "Villamassargia – Carbonia", "C192": "Alcamo Diramazione – Trapani",
    "C193": "Alessandria – Ovada", "C194": "Asciano – Monte Antico",
    "C195": "Asti – Acqui Terme", "C196": "Viterbo – Attigliano",
    "C198": "Barletta – Spinazzola", "C199": "Novara – Biella",
    "C201": "Caltanissetta Xirbi – Aragona Caldare", "C204": "Carini – Alcamo Diramazione",
    "C205": "Casarsa – Portogruaro", "C206": "Cavallermaggiore – Alba – Castagnole delle Lanze",
    "C207": "Cecina – Volterra", "C210": "Canicattì – Gela", "C211": "Giulianova – Teramo",
    "C212": "Isola della Scala – Cerea", "C213": "Legnago – Monselice",
    "C214": "Legnago – Rovigo", "C215": "Mercato San Severino – Montoro",
    "C216": "Mercato San Severino – Codola", "C217": "Cuneo – Bivio Madonna dell'Olmo",
    "C218": "Casale Monferrato – Mortara", "C220": "Agrigento – Porto Empedocle",
    "C221": "Bassano del Grappa – Primolano", "C224": "Rocchetta Sant'Antonio – Gioia del Colle",
    "C225": "Rovigo – Chioggia", "C226": "Sacile – Maniago", "C228": "Siena – Grosseto",
    "C229": "Sulmona – Carpinone", "C230": "Termoli – Campobasso", "C231": "Terni – Sulmona",
    "C232": "Trento – Primolano", "C234": "Palazzolo sull'Oglio – Paratico",
    "C235": "Ciampino – Frascati", "C236": "Foggia – Manfredonia",
    "C237": "Olbia – Golfo Aranci", "C239": "Fabriano – Pergola",
    "C241": "Sassari – Porto Torres", "C242": "Fidenza – Salsomaggiore Terme",
    "C243": "Cuneo – Saluzzo", "C244": "Santhià – Biella", "C245": "Savigliano – Saluzzo",
    "C246": "Bussoleno – Susa", "C247": "Torre Annunziata – Castellammare di Stabia",
    "C248": "Trofarello – Chieri", "C251": "Perugia Ponte San Giovanni – Città di Castello",
    "C252": "Perugia Ponte San Giovanni – Perugia Sant'Anna", "C260": "Torino – Ceres",
    "C262": "Settimo Torinese – Rivarolo Canavese",
    "TR0965": "Bologna Centrale – Bologna San Vitale",
    "OPICINA": "Villa Opicina – Sežana", "NABA": "Napoli – Cancello – Dugenta",
    # other infrastructure managers
    "BAAS": "Milano – Asso", "KSA": "Milano – Saronno", "SACL": "Saronno – Como",
    "SALV": "Saronno – Laveno", "SANO": "Saronno – Novara", "SASR": "Saronno – Seregno",
    "SCMXP": "Busto Arsizio – Malpensa Aeroporto", "LINK1": "Castellanza – Busto Arsizio",
    "FNI1": "Brescia – Iseo – Edolo", "FNI2": "Bornato – Rovato",
    "FER202": "Modena – Sassuolo", "FER203": "Reggio Emilia – Sassuolo",
    "FER204": "Reggio Emilia – Guastalla", "FER205": "Reggio Emilia – Ciano d'Enza",
    "FER206": "Parma – Suzzara", "FER207": "Suzzara – Ferrara", "FER208": "Ferrara – Codigoro",
    "FER208_PD": "Portomaggiore – Dogato", "FER209": "Bologna – Portomaggiore",
    "FER210": "Casalecchio di Reno – Vignola",
    "Bari-Taranto": "Bari – Martina Franca – Taranto",
    "Mungivacca-Putignano": "Bari Mungivacca – Putignano",
    "MartinaFranca-Lecce": "Martina Franca – Lecce", "Lecce-Gallipoli": "Lecce – Gallipoli",
    "Novoli-Gagliano": "Novoli – Gagliano del Capo",
    "Zollino-Gagliano": "Zollino – Gagliano del Capo", "Maglie-Otranto": "Maglie – Otranto",
    "Casarano-Gallipoli": "Casarano – Gallipoli",
    "FTL1": "Bari – Barletta", "FTL2": "Bari – Aeroporto", "FTL3": "Fesca San Girolamo – Cecilia",
    "CANCELLO-BENEVENTO": "Cancello – Benevento",
    "S.MARIA_CAPUA_VETERE-PIEDIMONTE_MATESE": "Santa Maria Capua Vetere – Piedimonte Matese",
    "Arezzo-Sinalunga": "Arezzo – Sinalunga", "Arezzo-Stia": "Arezzo – Stia",
    "UDINE-CIVIDALE": "Udine – Cividale", "1A123": "Foggia – Lucera",
    "2A123": "San Severo – Peschici",
    # groups (GROUP)
    "C73": "Mantova – Monselice", "C164": "Paola – Cosenza",
    "C136": "Battipaglia – Potenza – Metaponto", "C106": "Castel Bolognese – Ravenna",
    "C99": "Ferrara – Ravenna – Rimini", "C62": "Cremona – Mantova",
    "C86": "Vicenza – Treviso", "C40": "Lecco – Sondrio – Tirano",
    "C245": "Savigliano – Saluzzo – Cuneo", "C189": "Decimomannu – Iglesias",
    # pieces (it_fix)
    "F61-F62#1": "Roma – Cassino – Caserta", "F61-F62#2": "Cancello – Maddaloni Marcianise",
    "N3#1": "Nodo di Venezia", "N3#2": "Venezia Mestre – Carpenedo",
    "C224#1": "Rocchetta Sant'Antonio – San Nicola di Melfi",
    "C224#2": "Gravina in Puglia – Gioia del Colle",
    "F25-F26#1": "Verona – Padova", "F31-F32#2": "Bologna – Padova",
    "C226#1": "Sacile – Maniago", "C226#2": "Gemona del Friuli – Osoppo",
    "C10#1": "Cuneo – Limone", "C10#2": "Breil-sur-Roya – Ventimiglia",
    "C82#1": "Vigodarzere – Padova",
}


def it_id_name(lid, _uop):
    return NAMES.get(GROUP.get(lid, lid)) or NAMES.get(lid.split("#")[0])


COUNTRY = {
    "iso3": "ITA", "wikidata": "Q38", "langs": ["it", "en"],
    "fix": it_fix,
    "osm_ref": lambda _r: None,
    "fixed": dict(GROUP),
    "ref_display": lambda _k: "",
    "id_name": it_id_name,
    "im": {"0083_IM": "RFI", "0064_IM": "FERROVIENORD", "3525_IM": "FER",
           "3572_IM": "Ferrovie del Sud Est", "3857_IM": "Ferrotramviaria",
           "3856_IM": "EAV", "3456_IM": "La Ferroviaria Italiana",
           "PY63_IM": "Ferrovie del Gargano", "3908_IM": "GTT",
           "3379_IM": "Ferrovie Udine Cividale"},
    # The same Settimo - Rivarolo track is filed twice, under RFI (C262) and GTT.
    "skip_line": lambda lid: lid in SKIP,
}
