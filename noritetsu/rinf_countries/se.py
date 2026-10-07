"""Sweden: RINF's line ids are Trafikverket's stråk numbers, "01" to "99", and the line names
are Trafikverket's names for them ("Stråk 1 Västra stambanan", "Stråk 2 Södra stambanan";
Järnvägsnätsbeskrivning 2013 bilaga 3.6, and sv.wikipedia's "Järnväg i Sverige", which lists
all 96). A stråk is a corridor, not quite a line a rider knows: stråk 2 (Södra stambanan) also
holds Järna - Nyköping - Åby (Nyköpingsbanan), Nässjö - Ekenässjön (the Vetlanda branch) and
Alvesta - Gemla (Kust till kust-banan); stråk 7 (Stambanan genom övre Norrland) holds Luleå -
Boden (Malmbanan) and Vännäs - Umeå; the city stråk 22, 23, 24 and 27 ("Stockholm",
"Göteborg", "Malmö") hold the ends of every main line that enters those cities, and Citybanan.
se_sources.md has the sources and the checks.

LINES.  So `fix` files every RINF section under a named line, given here as chains of RINF
points (opName) in LINES: each chain claims the sections on the shortest path (RINF km) between
its consecutive points, over the whole network, whatever stråk they are filed under. A section
on two lines' chains is the first line's (one owner: Stockholm C - Karlberg is Ostkustbanan's,
so Mälarbanan starts at Tomteboda). A line's name is Trafikverket's stråk name; where a stråk
is split, the pieces take the name sv.wikipedia's line article gives them (Nyköpingsbanan,
Citybanan, Citytunneln, Söderåsbanan, Lommabanan, Trelleborgsbanan...). Sections no chain
claims go to their stråk's own line (`DEFAULT`: parallel tracks and curves, which build_model
keeps only where trains run), or, for a stråk with no line here, nowhere (`skip_line`).
FORCE gives a section to a line where the shortest path would take another track, LENGTH
corrects a RINF length that is not the track's, and FOLD takes out a junction point a line's
trains never pass (Lockarp).

FREIGHT: freight-only stretches inside passenger stråk (Svappavaarabanan, the Kattarp bypass,
Ockelbo - Storvik, Mälarbanan's old route via Fellingsbro...), claimed first and left out, as
Finland's are. Freight-only stråk (42 Piteåbanan, 44 Hällnäs–Storuman, 55 Örbyhus–Hallstavik,
64 Finspång–Kimstad...) have no line here and are left out by `skip_line`.

`cut_at_junctions`: sections end wherever another line meets, as in Portugal. Without it a
branch ended partway along the section of the line it joins (Haparandabanan at Buddbyn, inside
Malmbanan's Boden - Holmfors; Tjustbanan at Bjärka-Säby), so lines did not connect: 27 such
places. With it, Samtrafiken's feed (gtfs_served.py) decides which junction-ended sections have
trains.

STATIONS.  RINF lists Trafikverket's driftplatser, not halts: Rönninge, Tullinge, Stuvsta are
not in it. `osm_stops` makes every OSM station a train route stops at a stop on the section it
lies on. RINF names big stations in the genitive ("Halmstads central", "Bodens central",
"Hallsbergs personbangård") where OSM writes "Halmstad C", "Boden Central", "Hallsberg"; `fix`
adds the bare town name as a second name, so they match.

OSM's route=railway relations are not read (`osm_rel` returns None): all the names come from
LINES.
"""
import heapq
import re
from collections import defaultdict

# Ordered: an earlier line keeps a section a later chain also runs over.
LINES = [
    ("Västra stambanan", [
        ["Stockholms central", "Stockholms södra", "Årstaberg", "Älvsjö", "Huddinge",
         "Flemingsberg", "Södertälje syd övre", "Järna", "Gnesta", "Flen",
         "Katrineholms central", "Hallsbergs personbangård", "Hallsbergs rangerbangård", "Laxå",
         "Gårdsjö", "Skövde central", "Falköpings central", "Herrljunga central", "Alingsås",
         "Partille", "Sävedalen", "Göteborg Sävenäs", "Olskroken", "Göteborgs Central"],
        # the old main line through Södertälje, which the pendeltåg use
        ["Flemingsberg", "Tumba", "Södertälje hamn", "Södertälje syd undre", "Järna"],
        ["Södertälje hamn", "Södertälje centrum"],
        ["Södertälje syd övre", "Södertälje syd undre"]]),
    ("Citybanan", [["Tomteboda övre", "Stockholm Odenplan", "Stockholm City",
                    "Stockholms södra"]]),
    ("Ostkustbanan", [
        ["Stockholms central", "Karlberg", "Tomteboda övre", "Solna", "Ulriksdal", "Helenelund",
         "Upplands Väsby", "Märsta", "Knivsta", "Uppsala central", "Storvreta", "Örbyhus",
         "Tierp", "Skutskär", "Gävle central", "Strömsbro", "Vallvik", "Söderhamns västra",
         "Hudiksvall", "Sundsvalls central"]]),
    # To Örebro C, as Mälartåg's trains run. Trafikverket and sv.wikipedia end Mälarbanan at
    # Hovsta and file Hovsta - Örebro under stråk 9, Godsstråket; but Hovsta is no stop, and
    # ending there left Arboga - Hovsta a section to a junction that only Arboga - Örebro's
    # 50 km non-stop run crosses, which the timetable check does not take as evidence.
    ("Mälarbanan", [
        ["Tomteboda övre", "Sundbyberg", "Duvbo", "Spånga", "Bålsta", "Enköping",
         "Västerås central", "Kolbäck", "Köping", "Arboga", "Jädersbruk", "Alväng", "Hovsta",
         "Örebro central"]]),
    ("Södra stambanan", [
        ["Katrineholms central", "Strångsjö", "Åby södra", "Norrköpings central",
         "Linköpings central", "Mjölby", "Tranås", "Nässjö central", "Alvesta", "Älmhult",
         "Hässleholm", "Lund c", "Åkarp", "Arlöv", "Malmö godsbangård", "Malmö central"]]),
    ("Godsstråket genom Bergslagen", [
        ["Storvik", "Torsåker", "Fors", "Avesta Krylbo", "Fagersta central", "Frövi",
         "Örebro central", "Kumla", "Hallsbergs personbangård", "Skymossen", "Motala central",
         "Skänninge", "Mjölby"]]),
    ("Nyköpingsbanan", [["Järna", "Hölö", "Nyköpings central", "Åby södra"]]),
    ("Västkustbanan", [
        ["Göteborgs Central", "Gubbero", "Liseberg", "Almedal", "Mölndals nedre", "Kungsbacka",
         "Varbergs godsbangård", "Varbergs central", "Torebo", "Falkenberg personstation",
         "Halmstads central", "Eldsberga", "Laholm västra", "Ängelholm", "Kattarp",
         "Helsingborgs central", "Helsingborgs godsbangård", "Landskrona Östra", "Kävlinge",
         "Lund c"]]),
    ("Kust till kust-banan", [
        ["Almedal", "Mölndals övre", "Borås central", "Värnamo central", "Alvesta", "Gemla",
         "Växjö", "Emmaboda", "Kalmar södra", "Kalmar central"],
        ["Emmaboda", "Gullberna", "Karlskrona central"]]),
    ("Dalabanan", [
        ["Uppsala central", "Uppsala norra", "Brunna", "Sala", "Rosshyttan", "Avesta Krylbo",
         "Snickarbo", "Hedemora", "Säter", "Borlänge central", "Leksand", "Rättvik",
         "Mora central", "Morastrand"]]),
    ("Bergslagsbanan", [
        ["Gävle central", "Hagaström", "Sandviken", "Storvik", "Falun central", "Domnarvet",
         "Borlänge central", "Sellnäs", "Ludvika", "Grängesberg", "Ställdalen", "Hällefors",
         "Nykroppa", "Sandmon", "Kil"],
        ["Ställdalen", "Kopparberg", "Lindesberg", "Vedevåg", "Frövi"]]),
    ("Norra stambanan", [
        ["Strömsbro", "Oslättfors", "Ockelbo", "Holmsveden", "Kilafors", "Bollnäs", "Ljusdal",
         "Ovansjö", "Ånge"]]),
    ("Mittbanan", [
        ["Sundsvalls central", "Nacksta", "Stöde", "Torpshammar", "Fränsta", "Erikslund", "Ånge",
         "Moradal", "Bräcke", "Stavre", "Gällö", "Brunflo", "Östersunds central", "Nälden",
         "Åre", "Storlien", "Storlien gränsen"]]),
    ("Stambanan genom övre Norrland", [
        ["Bräcke", "Långsele", "Forsmo", "Mellansel", "Trehörningsjö", "Oxmyran", "Öreälv",
         "Vännäs", "Hällnäs", "Bastuträsk", "Jörn", "Nyfors", "Älvsbyn", "Bodens central"],
        ["Öreälv", "Nyåker", "Oxmyran"]]),
    ("Malmbanan", [
        ["Luleå", "Bodens central", "Buddbyn", "Holmfors", "Koijuvaara", "Gällivare central",
         "Råtsi", "Peuravaara", "Riksgränsen Vj-Bjf"],
        ["Peuravaara", "Kiruna malmbangård"]]),
    ("Haparandabanan", [
        ["Buddbyn", "Hundsjön", "Morjärv", "Kalix östra", "Haparanda södra",
         "Riksgräns Haparanda - Tornio"]]),
    ("Botniabanan", [
        ["Västeraspby", "Solum", "Örnsköldsviks central", "Nordmaling", "Gimonäs", "Umeå östra",
         "Umeå central"]]),
    ("Ådalsbanan", [
        ["Nacksta", "Birsta", "Timrå", "Härnösands central", "Kramfors", "Västeraspby",
         "Sollefteå", "Långsele"]]),
    ("Vännäs–Umeå", [["Vännäs", "Umeå godsbangård", "Umeå central"]]),
    ("Svealandsbanan", [
        ["Södertälje syd övre", "Läggesta", "Strängnäs", "Eskilstuna central", "Folkesta",
         "Rekarne", "Kungsör", "Valskog"]]),
    ("Sala–Oxelösund", [
        ["Sala", "Ransta", "Tillberga", "Västerås norra"],
        ["Kolbäck", "Strömsholm", "Kvicksund", "Rekarne"],
        ["Eskilstuna central", "Hälleforsnäs", "Flens övre", "Nyköping södra",
         "Nyköpings central"],
        ["Flen", "Flens övre"],
        ["Nyköping södra", "Oxelösund"]]),
    ("Nynäsbanan", [["Älvsjö", "Farsta strand", "Handen", "Västerhaninge",
                     "Nynäshamns centrum"]]),
    ("Norge/Vänerbanan", [
        ["Olskroken", "Gamlestaden", "Göteborg Marieholm", "Agnesberg", "Bohus", "Älvängen",
         "Alvhem", "Trollhättan", "Öxnered", "Skälebol", "Mellerud", "Åmål", "Säffle", "Kil"],
        ["Skälebol", "Ed", "Kornsjö gränsen"]]),
    ("Värmlandsbanan", [
        ["Laxå", "Hasselfors", "Degerfors", "Kristinehamn", "Karlstads central", "Kil",
         "Arvika", "Charlottenberg", "Charlottenberg gränsen"]]),
    ("Skånebanan", [
        ["Helsingborgs godsbangård", "Ättekulla", "Påarp", "Bjuv", "Åstorp", "Klippan",
         "Perstorp", "Tyringe", "Finja", "Hässleholm", "Vinslöv", "Kristianstads central"]]),
    ("Jönköpingsbanan", [
        ["Falköpings central", "Vartofta", "Sandhem", "Jönköpings central", "Huskvarna",
         "Tenhult", "Nässjö central"]]),
    ("Älvsborgsbanan", [
        ["Uddevalla central", "Öxnered", "Vänersborg central", "Herrljunga central", "Ljung",
         "Borgstena", "Knalleland", "Borås central"]]),
    ("Bohusbanan", [
        ["Olskroken", "Göteborg Kville", "Säve", "Ytterby", "Stenungsund", "Ljungskile",
         "Grohed", "Uddevalla central", "Munkedal", "Dingle", "Strömstad"]]),
    ("Viskadalsbanan", [["Borås central", "Viskafors", "Skene", "Horred", "Veddige",
                         "Varbergs godsbangård"]]),
    ("Kinnekullebanan", [["Gårdsjö", "Hova", "Mariestad", "Lidköping", "Håkantorp"]]),
    ("Fryksdalsbanan", [["Kil", "Bäckebron", "Sunne", "Torsby"]]),
    ("Stångådalsbanan", [
        ["Linköpings central", "Bjärka-Säby", "Rimforsa", "Kisa", "Vimmerby", "Hultsfred",
         "Berga", "Blomstermåla", "Rockneby", "Kalmar södra"],
        ["Berga", "Oskarshamn"]]),
    ("Tjustbanan", [["Bjärka-Säby", "Viresjö", "Åtvidaberg", "Västervik"]]),
    ("Nässjö–Hultsfred", [["Nässjö central", "Eksjö", "Hultsfred"]]),
    ("Nässjö–Vetlanda", [["Nässjö central", "Ekenässjön", "Vetlanda"]]),
    ("Vaggerydsbanan", [["Jönköpings central", "Månsarp", "Vaggeryd"]]),
    ("Nässjö–Halmstad", [
        ["Nässjö central", "Stolpen", "Vaggeryd", "Värnamo central", "Forsheda", "Torup",
         "Furet"]]),
    ("Blekinge kustbana", [
        ["Kristianstads central", "Fjälkinge", "Bromölla", "Sölvesborg", "Karlshamn",
         "Ronneby", "Nättraby", "Gullberna"]]),
    ("Markarydsbanan", [["Eldsberga", "Genevad", "Markaryd", "Bjärnum", "Hässleholm"]]),
    ("Rååbanan", [["Helsingborgs godsbangård", "Gantofta", "Billeberga", "Teckomatorp",
                   "Eslöv"]]),
    ("Söderåsbanan", [["Kävlinge", "Teckomatorp", "Svalöv", "Kågeröd", "Billesholm",
                       "Åstorp"]]),
    ("Lommabanan", [["Kävlinge", "Stävie", "Lomma", "Arlöv"]]),
    ("Citytunneln", [["Malmö central", "Triangeln", "Hyllie"]]),
    # Runs over the bridge to the border. RINF's border point (EU00141) is on Peberholm, which is
    # Danish; _oresund moves it to where the track crosses the border on the bridge (as
    # borders.MOVE and rinf_countries/dk.py do) and folds Peberholm out, so stråk 98's Swedish
    # part ends there. Before Denmark was built (2026-10-03) this line ended at Lernacken.
    ("Öresundsbanan", [["Hyllie", "Lernacken", "Peberholm gränsen"]]),
    ("Ystadbanan", [["Hyllie", "Svågertorp", "Skabersjö", "Svedala", "Ystad", "Tomelilla",
                     "Simrishamn"]]),
    ("Trelleborgsbanan", [["Svågertorp", "Västra Ingelstad", "Östra Grevie", "Trelleborg"]]),
    ("Kontinentalbanan", [["Malmö central", "Östervärn", "Fosieby", "Svågertorp"]]),
    ("Arlandabanan", [
        ["Skavstaby", "TRV-Atrain border Skavstaby", "Arlanda nedre", "Arlanda södra",
         "Arlanda norra"],
        ["Myrbacken", "TRV-Atrain border Myrbacken", "Arlanda central", "Arlanda nedre"]]),
    ("Bergslagspendeln", [
        ["Kolbäck", "Hallstahammar", "Surahammar", "Ängelsberg", "Fagersta central",
         "Söderbärke", "Smedjebacken", "Hagge", "Ludvika"]]),
    ("Västerdalsbanan", [["Repbäcken", "Mockfjärd", "Vansbro", "Malung"]]),
    ("Dal Västra Värmlands Järnväg", [["Mellerud", "Åsensbruk", "Håverud", "Billingsfors"]]),
    ("Skelleftebanan", [["Bastuträsk", "Finnforsfallet", "Skellefteå",
                         "Skelleftehamns övre"]]),
    # Trafikverket's stråk 69 ("Kristinehamn–Nykroppa, Daglösen–Persberg"). Tågab's train
    # Falun - Ludvika - Nykroppa - Kristinehamn runs here (Sundays only in autumn 2026,
    # sv.wikipedia "Bergslagsbanan"). Nykroppa - Storfors is filed under stråk 10.
    ("Kristinehamn–Nykroppa", [["Kristinehamn", "Spjutbäcken", "Storfors", "Nykroppa"]]),
    ("Inlandsbanan", [
        ["Mora central", "TRV-IBAB border Mora", "Orsa", "Sveg", "Åsarna central", "Svenstavik",
         "TRV-IBAB border Brunflo", "Brunflo"],
        ["Östersunds central", "TRV-IBAB border Östersund", "Lit", "Ulriksfors", "Hoting",
         "Dorotea", "Vilhelmina", "Storuman", "Sorsele", "Arvidsjaur", "Jokkmokk", "Porjus",
         "TRV-IBAB border Gällivare", "Gällivare central"]]),
]

# A stråk's own line, for its sections no chain claims (parallel track pairs and curves,
# which build_model keeps only where OSM trains run over them).
DEFAULT = {
    "01": "Västra stambanan", "02": "Södra stambanan", "03": "Västkustbanan",
    "04": "Kust till kust-banan", "05": "Ostkustbanan", "06": "Dalabanan",
    "07": "Stambanan genom övre Norrland", "08": "Norra stambanan",
    "09": "Godsstråket genom Bergslagen", "10": "Bergslagsbanan", "11": "Norge/Vänerbanan",
    "12": "Värmlandsbanan", "13": "Skånebanan", "14": "Jönköpingsbanan",
    "15": "Älvsborgsbanan", "16": "Mälarbanan", "17": "Svealandsbanan",
    "18": "Sala–Oxelösund", "19": "Nynäsbanan", "20": "Mittbanan", "21": "Malmbanan",
    "28": "Botniabanan", "29": "Haparandabanan", "30": "Arlandabanan", "31": "Ådalsbanan",
    "32": "Rååbanan", "33": "Markarydsbanan", "45": "Skelleftebanan", "53": "Västerdalsbanan",
    "63": "Bergslagspendeln", "65": "Stångådalsbanan", "66": "Tjustbanan",
    "70": "Fryksdalsbanan", "71": "Dal Västra Värmlands Järnväg", "73": "Bohusbanan",
    "75": "Kinnekullebanan", "77": "Viskadalsbanan", "80": "Nässjö–Hultsfred",
    "81": "Nässjö–Vetlanda", "83": "Vaggerydsbanan", "84": "Nässjö–Halmstad",
    "88": "Blekinge kustbana", "90": "Ystadbanan", "99": "Inlandsbanan",
}

# Sections given to a line before the chains run, by their two end points: where the shortest
# path would take another track. Hallsberg - Motala trains leave Hallsbergs personbangård
# straight for Skymossen; by RINF km the way round through the rangerbangård is shorter.
FORCE = [("Godsstråket genom Bergslagen", "Hallsbergs personbangård", "Skymossen")]

# RINF lengths that are not the track's. Hallsbergs personbangård - Skymossen is 10.78 km in
# RINF for 7.7 km of track (traced both piecewise and end to end; 5.4 km as the crow flies).
LENGTH = {("Hallsbergs personbangård", "Skymossen"): 7.7}

# A junction RINF routes a line through that its trains never pass: (from, via, [onward]).
# The section from-via is folded into each via-onward section, which then starts at `from`.
# Lockarp's point is on the Trelleborg line north of the triangle there; the line from
# Svågertorp joins the Trelleborg line south of it and has its own curve onto the Ystad line.
# Traced through the point, Svågertorp - Lockarp ran 4.9 km for RINF's 2.6 (doubling back up
# the Trelleborg line), was rejected once sections were cut at junctions, and Ystadbanan lost
# its way into Malmö.
FOLD = [("Svågertorp", "Lockarp", ["Skabersjö", "Västra Ingelstad"])]

# Freight only, claimed before LINES and left out (`skip_line`), as Finland's FREIGHT is. Each
# lies under no OSM train route and has no train in Samtrafiken's feed; where it ends at a
# station, build_model would otherwise keep it as running. sv.wikipedia for each:
FREIGHT = [
    ["Råtsi", "Svappavaara"],                  # Svappavaarabanan, LKAB's ore
    ["Gällivare central", "Koskullskulle"],    # to Malmberget's ore
    ["Koijuvaara", "Aitik"],                   # Boliden's Aitik mine
    ["Birsta", "Fillan"],                      # Sundsvall's harbour
    # Skånebanan's bypass via Kattarp, "trafikeras i regel inte" (normally unused)
    ["Åstorp", "Hasslarp", "Kattarp"],
    # Norra stambanan's old start: "Samtliga persontåg på banan går numera via Gävle"
    ["Ockelbo", "Åshammar", "Storvik"],
    # Mälarbanan's old route via Frövi, "kvar parallellt" since the Jädersbruk - Hovsta cut-off
    # opened in 1997; no train in the feed calls at Fellingsbro
    ["Jädersbruk", "Fellingsbro", "Frövi"],
    # filed under Inlandsbanan's stråk 99: toward Lycksele (stråk 44, Hällnäs–Storuman) and
    # toward Bollnäs (the Orsa - Bollnäs line); Inlandståget runs neither
    ["Storuman", "Gunnarn"],
    ["Orsa", "Born", "Kallholsfors"],
]
FREIGHT_NAME = "(freight)"

NAMES = {n for n, _c in LINES}

# Point names: RINF's genitive "<town>s central" is OSM's "<town> C" / "<town> Central".
STATION_WORD = re.compile(r"^(.*?)\s+(central|c|centrum|personbangård|personstation)$", re.I)


def _point_ops(points):
    by = defaultdict(set)
    for op, p in points.items():
        by[p.get("name")].add(op)
    return by


def _path(adj, srcs, dsts):
    """Shortest path by RINF km from any op in srcs to any in dsts: the sections on it."""
    dist = {op: 0.0 for op in srcs}
    prev = {}
    heap = [(0.0, op) for op in srcs]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist.get(u, float("inf")):
            continue
        if u in dsts:
            out = []
            while u in prev:
                u, s = prev[u]
                out.append(s)
            return out
        for v, s in adj[u]:
            nd = d + (s["km"] or 0.0) + 1e-6
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                prev[v] = (u, s)
                heapq.heappush(heap, (nd, v))
    return None


def _fold(secs, points, out):
    nm = lambda op: points.get(op, {}).get("name")
    for a, via, onward in FOLD:
        first = [s for s in secs if {nm(s["a"]), nm(s["b"])} == {a, via}]
        if len(first) != 1:
            raise SystemExit(f"rinf_countries/se.py: FOLD {a} - {via}: {len(first)} sections")
        f = first[0]
        op_a = f["a"] if nm(f["a"]) == a else f["b"]
        for o in onward:
            nxt = [s for s in secs if {nm(s["a"]), nm(s["b"])} == {via, o}]
            if len(nxt) != 1:
                raise SystemExit(f"rinf_countries/se.py: FOLD {via} - {o}: {len(nxt)} sections")
            s = nxt[0]
            if nm(s["a"]) == via:
                s["a"] = op_a
            else:
                s["b"] = op_a
            s["km"] = (s["km"] or 0.0) + (f["km"] or 0.0)
            out.append(f"{via} folded out of {a} - {o} ({s['km']:.2f} km)")
        secs.remove(f)


ORESUND_BORDER = (12.808962, 55.579239)   # bridge rail ways 1185526677/8 x boundary way 71417261
ORESUND_DK_KM = 5.4                        # Peberholm's west end to the border, Danish track


def _oresund(secs, points, out):
    """EU00141 moved to the border on the bridge (as borders.MOVE and dk.py move it); Peberholm
    (SEPhm, Danish) folded out, so EU00141 - SE00100 is stråk 98's Swedish part."""
    uop = {p.get("uopid"): op for op, p in points.items()}
    bp, phm = uop.get("EU00141"), uop.get("SEPhm")
    if bp is None or phm is None:
        return
    points[bp]["lon"], points[bp]["lat"] = ORESUND_BORDER
    first = [s for s in secs if {s["a"], s["b"]} == {bp, phm}]
    nxt = [s for s in secs if phm in (s["a"], s["b"]) and bp not in (s["a"], s["b"])]
    if len(first) != 1 or len(nxt) != 1:
        raise SystemExit(f"rinf_countries/se.py: Öresund: {len(first)}, {len(nxt)} sections")
    s = nxt[0]
    if s["a"] == phm:
        s["a"] = bp
    else:
        s["b"] = bp
    s["km"] = (s["km"] or 0.0) + (first[0]["km"] or 0.0) - ORESUND_DK_KM
    secs.remove(first[0])
    out.append(f"EU00141 moved to the border on the bridge, Peberholm folded out: {s['km']:.3f} km")


def se_fix(secs, points):
    out = []
    _oresund(secs, points, out)
    _fold(secs, points, out)
    ops = _point_ops(points)
    adj = defaultdict(list)
    for s in secs:
        adj[s["a"]].append((s["b"], s))
        adj[s["b"]].append((s["a"], s))
    owner = {}
    nm = lambda op: points.get(op, {}).get("name")
    for s in secs:
        ends = {nm(s["a"]), nm(s["b"])}
        for name, a, b in FORCE:
            if ends == {a, b}:
                owner[s["sol"]] = name
        for (a, b), km in LENGTH.items():
            if ends == {a, b}:
                out.append(f"{a} - {b}: RINF's {s['km']} km read as {km}")
                s["km"] = km
    claims = defaultdict(list)
    for name, chains in [(FREIGHT_NAME, FREIGHT)] + LINES:
        for chain in chains:
            missing = [w for w in chain if not ops.get(w)]
            if missing:
                raise SystemExit(f"rinf_countries/se.py: {name}: no RINF point named {missing}")
            for a, b in zip(chain[:-1], chain[1:]):
                got = _path(adj, ops[a], ops[b])
                if got is None:
                    raise SystemExit(f"rinf_countries/se.py: {name}: no path {a} -> {b}")
                for s in got:
                    if name not in claims[s["sol"]]:
                        claims[s["sol"]].append(name)
    # A section on two lines' chains is the first line's in LINES (FREIGHT first of all).
    n_shared = 0
    for s in secs:
        cl = claims.get(s["sol"])
        if not cl or s["sol"] in owner:
            continue
        owner[s["sol"]] = cl[0]
        n_shared += len(cl) > 1
    out.append(f"{n_shared} sections on two lines' chains, given to the first")
    n_default = n_left = 0
    left = defaultdict(float)
    for s in secs:
        name = owner.get(s["sol"])
        if name is None:
            name = DEFAULT.get(s["base"].split("#")[0])
            if name:
                n_default += 1
            else:
                n_left += 1
                left[s["base"].split("#")[0]] += s["km"] or 0.0
                name = f"(stråk {s['base']})"
        s["line"] = s["base"] = name
    out.append(f"{len(owner)} sections on {len(NAMES)} named lines by LINES, {n_default} more "
               f"by their stråk (DEFAULT), {n_left} left out: "
               + ", ".join(f"{k} {v:.0f} km" for k, v in sorted(left.items())))
    # pieces of one name that do not touch, numbered as rinf.py numbers an id's pieces
    by = defaultdict(list)
    for s in secs:
        by[s["base"]].append(s)
    for name, ss in by.items():
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
        if len(comps) < 2 or name not in NAMES:
            continue
        order = sorted(comps.values(), key=lambda c: -sum(s["km"] or 0 for s in c))
        for k, c in enumerate(order):
            for s in c:
                s["line"] = name if k == 0 else f"{name}#{k + 1}"
        out.append(f"{name} is {len(comps)} pieces: " + " | ".join(
            f"{sum(s['km'] or 0 for s in c):.1f} km "
            + "/".join(sorted({points[op].get('name', '?') for s in c for op in (s['a'], s['b'])}))[:80]
            for c in order))
    # the town's bare name beside RINF's genitive form, for matching OSM stations
    n_alias = 0
    for p in points.values():
        m = STATION_WORD.match(p.get("name") or "")
        if m:
            base = m.group(1)
            alias = {base} | ({base[:-1]} if base.endswith("s") else set())
            p["name"] = " | ".join([p["name"], *sorted(alias)])
            n_alias += 1
    out.append(f"{n_alias} station names given the bare town name as a second name")
    return out


def se_id_name(lid, _uop=None):
    return lid.split("#")[0] if lid.split("#")[0] in NAMES else None


COUNTRY = {
    "iso3": "SWE", "wikidata": "Q34", "langs": ["sv", "en"],
    "fix": se_fix, "id_name": se_id_name,
    "skip_line": lambda lid: lid.split("#")[0] not in NAMES,
    "osm_rel": lambda _tags: None,
    "osm_stops": True,
    "cut_at_junctions": True,
    "im": {"0074_IM": "Trafikverket", "3779_IM": "Inlandsbanan AB", "LQB6_IM": "A-Train",
           "3872_IM": "Øresundsbro Konsortiet"},
}
