"""The line lists wafrica_register.py converts: one entry per passenger line, its stations in
order. Point syntax: wafrica_register.py's docstring; what runs and why: <cc>_sources.md."""


def L(lid, name, name_en, im, pts, more=(), suspended=False, listed_only=False, kind="rail",
      colour="", network="", note=""):
    """`listed_only`: the line's trains stop at its listed stations alone, so OSM stations it
    passes are not made stops (an express line beside an older one). `kind` other than "rail"
    is set on the built line (subway, light_rail)."""
    return {"id": lid, "name": name, "name_en": name_en, "im": im, "pts": list(pts),
            "more": [list(m) for m in more], "suspended": suspended,
            "listed_only": listed_only, "kind": kind, "colour": colour, "network": network,
            "note": note}


# Border points: id -> (lon, lat, [countries]). No passenger train crosses a border in the
# region (wafrica_survey.md), so none.
BORDERS = {}

# Route relations that are no scheduled passenger service; wafrica_register --clip drops them.
NOT_SERVICE = {
    "ng": {
        8527053: "Port Harcourt - Kano: the old Cape gauge express, not running (ng_sources.md)",
        18278186: "Kaduna - Zaria - Kano standard gauge: under construction",
        13035324: "Port Harcourt Monorail: never opened",
        # Abuja's metro: built as greyed register lines (ng-abj-yellow, ng-abj-blue); its OSM
        # routes would otherwise be drawn as running lines over them.
        8441841: "Abuja metro Yellow Line: no service reported in 2025-2026",
        8441842: "Abuja metro Yellow Line (other direction)",
        8442723: "Abuja metro Blue Line: no service reported in 2025-2026",
        8442724: "Abuja metro Blue Line (other direction)",
        8464133: "Abuja metro Yellow Line route_master",
        9542499: "Abuja metro Blue Line route_master",
        # Routes the register lines here are, under other names, so merge_sources cannot
        # twin them: each would be drawn a second time over its register line.
        13184642: "Express Train : Lagos <-> Ibadan: the register's Lagos – Ibadan",
        9285998: "Itakpe - Warri Rail Line: the register's Warri – Itakpe",
        6441947: "Abuja-Kaduna Railway: the register's Abuja – Kaduna",
    },
    "cg": {
        8360001: "La Gazelle Pointe Noire - Brazzaville: the register's Pointe-Noire – "
                 "Brazzaville",
        8360346: "CFCO Bilinga - Dolisie: a local train over the old Mayombe line; none "
                 "reported since the 2023 restart (cg_sources.md)",
        8359854: "COMILOG Mbinda - Mont Belo: no passenger train (cg_sources.md)",
    },
    "sn": {
        8530208: "Petit train de banlieue Dakar - Thiès: no service since the TER (sn_sources.md)",
        13645076: "TER AIBD - Dakar: the register's TER line",
        13645077: "TER Dakar - AIBD: the register's TER line",
        13645078: "TER route_master",
    },
    "gh": {
        14435104: "Nsawam - Accra: no service reported in 2025-2026 (gh_sources.md)",
        14435107: "Accra - Nsawam: as above",
        14435108: "Accra - Nsawam shuttle route_master",
    },
    "bf": {
        8530177: "Abidjan - Ouagadougou: the through train has not run since 2020; the "
                 "register has Ouagadougou – Bobo-Dioulasso (bf_sources.md)",
        10185190: "Ligne d'Abidjan à Ouagadougou route_master",
    },
    "cm": {
        8503524: "Douala-Yaoundé: the register's Douala – Yaoundé",
        8503527: "Yaoundé-Ngaoundéré: the register's Yaoundé – Ngaoundéré",
        11701793: "an unnamed one-way route=train, no stops",
    },
    "ao": {
        5414666: "Train: Luanda - Malanje: the register's Baía – Malanje",
        8477951: "Namibe - Lubango: the register's Namibe – Lubango",
        8477956: "Lobito - Luau: the register's Lobito – Luau",
        401053: "Matadi-Kinshasa: DR Congo's",
    },
    "cd": {
        401053: "Matadi-Kinshasa: the register's Kinshasa – Matadi, greyed (suspended since "
                "about April 2026; cd_sources.md)",
        1281986: "Ligne urbaine de l'Aéroport: the register's urban line",
        1281981: "Ligne urbaine de Kasangulu: not running",
        1789177: "Branche de Ango Ango (Matadi port): not running",
        4603619: "an unnamed route=train",
        # SNCC and the east: named trains once or twice a month, below the weekly bar
        # (cd_sources.md); Kisangani - Ubundu and Lubumbashi - Sakania likewise unreported.
        401389: "Lubumbashi-Tenke (SNCC)", 8480450: "Kamina - Kindu (SNCC)",
        8480451: "Lubumbashi - Ilebo (SNCC)", 14599641: "Lubumbashi - Kalemie (SNCC)",
        1124872: "Kisangani-Ubundu (SNCC)", 14361616: "Ubundu - Kisangani (SNCC)",
        3711965: "Lubumbashi - Sakania", 14599406: "Sakania - Lubumbashi",
    },
}


def _gauge(g):
    return lambda t: t.get("gauge") == g


def _named(n):
    return lambda t: t.get("name") == n


# Parallel railways: per country, (line id, rule on a way's tags) in order; a way the first
# matching rule names is that line's own track (wafrica_register.py, "PARALLEL RAILWAYS").
OWN = {
    # Lagos, Ebute Metta - Agbado: OSM maps the standard gauge double track ("Lagos–Ibadan
    # SGR", the only standard gauge pair it draws there) and NRC's Cape gauge track (gauge=1067,
    # unnamed) 90 m apart. The Cape gauge mass transit keeps to 1067; the SGR to its named
    # track; the Red Line to standard gauge (in rinf.py's pass 2, which takes one name per
    # way, that is the unnamed 1435 track: its Oyingbo end). Where two gauges share a node,
    # --clip splits it (split_gauges); a station still snaps to both tracks, so the
    # preference is needed as well. Built, the Red Line and the SGR each lie on one track of
    # OSM's pair: LAMATA's line has its own tracks in the corridor (two lines, counted
    # apart), which OSM has not drawn separately (ng_sources.md, "Build").
    "ng": [("ng-lag-iba", _named("Lagos–Ibadan SGR")),
           ("ng-red", _gauge("1435")),
           ("ng-iddo", _gauge("1067"))],
    # Dakar: the TER's standard gauge "TER Dakar-AIBD" beside the metre gauge Dakar - Niger.
    "sn": [("sn-ter", _named("TER Dakar-AIBD"))],
    # Tema: the standard gauge Tema - Mpakadan line and GRCL's Cape gauge both end at Tema
    # Harbour.
    "gh": [("gh-tema-adome", _gauge("1435")), ("gh-adome-mpakadan", _gauge("1435")),
           ("gh-accra-tema", _gauge("1067"))],
}

NRC = "Nigerian Railway Corporation"
LAMATA = "Lagos Metropolitan Area Transport Authority"

LINES = {}

# Nigeria (ng_sources.md). Running: the three standard gauge lines, NRC's two Cape gauge
# commuter services and LAMATA's Red Line; Abuja's metro greyed. LAMATA's Blue Line is
# mapped as railway=subway, which rinf.py's track graph does not take: it is built from its OSM
# route relation (15668733/4) as an OSM line.
LINES["ng"] = [
    # NRC's stations (LITS): every one listed, so OSM's Cape gauge and Red Line stations beside
    # the track are not made stops of it (listed_only).
    L("ng-lag-iba", "Lagos – Ibadan", "Lagos – Ibadan", NRC,
      ["Mobolaji Johnson Station", "Agege@3.3262,6.6195", "Agbado@3.3031,6.6871",
       "Kajola Station", "Papalanto Station", "Professor Wole Shoyinka Station, Laderin",
       "Olodo@3.59529,7.27938", "Omi-Adio Station", "Obafemi Awolowo Station"],
      listed_only=True, network="Lagos–Ibadan Train Service"),
    # From Idu to Kubwa the trains run over the metro Blue Line's track (OSM maps it as one
    # railway=rail), past metro halts they do not call at: listed_only. AKTS also calls at
    # Asham and Katari, which OSM does not have and no coordinate was found for.
    L("ng-abj-kad", "Abuja – Kaduna", "Abuja – Kaduna", NRC,
      ["Idu@7.3425,9.0470", "Kubwa@7.3253,9.1640", "Jere@7.4540,9.5707", "Rijana",
       "Kakau", "Kaduna - Rigasa"], listed_only=True, network="Abuja–Kaduna Train Service"),
    L("ng-war-ita", "Warri – Itakpe", "Warri – Itakpe", NRC,
      ["Ujevwu Station", "Agbarho Station", "Okpara Station", "Abraka Station",
       "Agbor Station", "Igbanke Station", "Igueben-Ekhen Station", "Uromi Station",
       "Agenebode Station", "Itogbo Station", "Ajaokuta Station", "Adogo Station",
       "Eganyi Station", "Itakpe Station"], network="Warri–Itakpe Train Service"),
    # The Cape gauge mass transit: Iddo's terminus has no OSM station (filled at its
    # buffer stops). OSM's Cape gauge track stops 2.6 km past Agbado and picks up again only
    # at Ijoko's sidings, 4 km on; a trace over the gap takes the SGR beside it. The line is
    # drawn to where OSM's track ends, so the last 4 km to Ijoko are not on the map.
    L("ng-iddo", "Iddo – Ijoko", "Iddo – Ijoko", NRC,
      # NRC's commuter stops, listed (the corridor's SGR and Red Line stations lie within
      # 120 m of the Cape gauge track); Ebute Metta Junction is OSM's unnamed station there.
      ["Iddo@3.3824,6.4702", "Ebute Metta Junction@3.3638,6.5059", "Yaba@3.3700,6.5115",
       "Mushin", "Oshodi", "Ikeja@3.3374,6.5919", "Agege@3.3262,6.6195", "Iju",
       "Agbado@3.3031,6.6871", "~Cape gauge track ends (OSM)@3.28957,6.70761"],
      listed_only=True, network="Lagos Mass Transit Train"),
    L("ng-ph-aba", "Port Harcourt – Aba", "Port Harcourt – Aba", NRC,
      ["Port Harcourt Station", "Aba Railway Station"]),
    # LAMATA's Red Line: its eight stations (lamata.lagosstate.gov.ng), on the SGR's track to
    # Yaba, then its own to Oyingbo.
    L("ng-red", "Red Line", "Red Line", LAMATA,
      ["Agbado@3.3031,6.6871", "Iju", "Agege LRMT Station", "Ikeja@3.3374,6.5919",
       "Oshodi", "Mushin", "Yaba@3.3700,6.5115", "Oyingbo"],
      listed_only=True, colour="#E30613", network="Lagos Rail Mass Transit"),
    # Abuja Rail Mass Transit: relaunched May 2024, free rides to the end of 2024, no report
    # of trains since (ng_sources.md): greyed.
    L("ng-abj-yellow", "Yellow Line", "Yellow Line", "Abuja Rail Mass Transit",
      ["Abuja Metro", "Stadium", "Kukwaba I", "Kukwaba II", "Wupa", "Idu@7.3428,9.0464",
       "Bassanjiwa", "Airport@7.2724,9.0067"], suspended=True, listed_only=True,
      kind="light_rail", colour="#F2C500", network="Abuja Rail Mass Transit"),
    L("ng-abj-blue", "Blue Line", "Blue Line", "Abuja Rail Mass Transit",
      ["Idu@7.3428,9.0464", "Gwagwa", "Deidei", "Kagini", "Gbazango"], suspended=True,
      listed_only=True, kind="light_rail", colour="#0072BC", network="Abuja Rail Mass Transit"),
]

# Gabon (ga_sources.md): the Transgabonais, SETRAG. All 23 of SETRAG's stations are OSM
# stations on the line ("Owendo Viré" is Owendo's passenger station); the express skips some,
# the omnibus calls at all, so every one is a stop.
LINES["ga"] = [
    L("ga-transgabonais", "Transgabonais : Owendo – Franceville", "Trans-Gabon Railway: "
      "Owendo – Franceville", "SETRAG",
      ["Owendo Viré", "Ntoum", "M'Bel", "Ndjolé", "Booué", "Lastourville", "Moanda",
       "Franceville"], network="SETRAG"),
]

# Republic of the Congo (cg_sources.md): the Congo-Océan main line, CFCO's weekly "La
# Gazelle". Stops: every OSM station on the traced line, which are the stops of OSM's Gazelle
# route (8360001, dropped by --clip as the register line's twin). The points pin the trace to
# the Mayombe realignment (Les Saras - Mvouti - Les Bandas), which the train takes, not the
# old line via Nkoungi and Nemba.
LINES["cg"] = [
    L("cg-cfco", "Pointe-Noire – Brazzaville", "Pointe-Noire – Brazzaville", "CFCO",
      ["Pointe-Noire", "Bilinga", "Les Saras", "Mvouti", "Gare Les Bandas", "Moukondo",
       "Dolisie", "Mont-Belo", "Loudima", "Nkayi@13.2830,-4.1893", "Madingou", "Mindouli",
       "Kibouende", "Gare de Mfilou", "Gare de Brazzaville"],
      network="Chemin de fer Congo-Océan"),
]

# Senegal (sn_sources.md): the TER, SETER for SENTER, Dakar - Diamniadio (2021) and on to the
# airport (28 Sept 2026). Its 14 stations as OSM's TER route lists them; listed, so the metre
# gauge's halts beside it (Baux Maraîchers is both) are not taken twice.
LINES["sn"] = [
    L("sn-ter", "TER Dakar – AIBD", "Dakar Regional Express Train: Dakar – AIBD", "SETER",
      ["Dakar@-17.4335,14.6760", "Colobane", "Hann", "Dalifort",
       "Baux Maraîchers@-17.4036,14.7397", "Pikine", "Thiaroye", "Yeumbeul@-17.3565,14.7649",
       "Mbao", "PNR", "Rufisque", "Bargny", "Diamniadio", "Aérogare Blaise Diagne"],
      listed_only=True, network="Train Express Régional"),
]

# Ghana (gh_sources.md). GRDA's Tema - Mpakadan standard gauge: trains Tema - Afienya three
# times a day since 1 Oct 2025, fares published on to Adome (written "Adorme" in the press);
# beyond Adome to Mpakadan no train is reported (greyed). OSM has none of the line's new
# stations but Tema Harbour and Kpong: Tema Industrial Area, Ashaiman, Afienya, Shai Hills and
# Doryumu have no coordinate, so they are not stops here. Adome is placed by hand (fill) on the
# track nearest Adomi (Atimpoku), 3 km east of it past the Volta bridge, where GRDA's list puts it
# between Kpong and Mpakadan: approximate.
# GRCL's Accra - Tema Cape gauge train, one return a day: stops are OSM's along it.
LINES["gh"] = [
    L("gh-tema-adome", "Tema – Adome", "Tema – Adome", "Ghana Railway Development Authority",
      ["Tema Harbour", "Kpong", "Adome@0.1222,6.2299"], listed_only=True,
      network="Tema–Mpakadan Railway"),
    L("gh-adome-mpakadan", "Adome – Mpakadan", "Adome – Mpakadan",
      "Ghana Railway Development Authority",
      ["Adome@0.1222,6.2299", "Mpakadan@0.0912,6.3292"], suspended=True, listed_only=True,
      network="Tema–Mpakadan Railway"),
    L("gh-accra-tema", "Accra – Tema", "Accra – Tema", "Ghana Railway Company",
      ["Accra@-0.21102,5.54892", "Baatsona", "Addogonno", "Tema Harbour"],
      network="Ghana Railway Company"),
]

# Burkina Faso (bf_sources.md): Sitarail's weekly train since 17 Nov 2023, calling at the four
# stations its communiqué names. Bobo-Dioulasso - Banfora - Niangoloko, where the press (not
# the communiqué) says it runs on to: greyed.
LINES["bf"] = [
    L("bf-ouaga-bobo", "Ouagadougou – Bobo-Dioulasso", "Ouagadougou – Bobo-Dioulasso",
      "Sitarail", ["Gare de Ouagadougou", "Gare de Koudougou", "Gare de Siby",
                   "Bobo-Dioulasso@-4.3068,11.1782"], listed_only=True,
      network="Sitarail"),
    L("bf-bobo-niangoloko", "Bobo-Dioulasso – Niangoloko", "Bobo-Dioulasso – Niangoloko",
      "Sitarail", ["Bobo-Dioulasso@-4.3068,11.1782", "Gare de Banfora",
                   "Gare de Niangoloko"], suspended=True, listed_only=True, network="Sitarail"),
]

# Cameroon (cm_sources.md): Camrail's three routes, every OSM station on them a stop (the
# omnibus trains call at all). Douala - Kumba starts at Douala Bessengué, Camrail's Douala
# station: OSM has no Bonabéri station across the Wouri, where seat61 says the trains leave.
LINES["cm"] = [
    L("cm-dla-yde", "Douala – Yaoundé", "Douala – Yaoundé", "Camrail",
      ["Douala Bessengué", "Edéa voyageurs", "Gare voyageur de Messondo", "Eséka", "Makak",
       "Otélé", "Ngoumou", "Yaoundé"], network="Camrail"),
    L("cm-yde-ngd", "Yaoundé – Ngaoundéré", "Yaoundé – Ngaoundéré", "Camrail",
      ["Yaoundé", "Gare d'Obala", "Nanga Eboko", "Belabo", "Ngaoundal", "Ngaoundéré"],
      network="Camrail"),
    L("cm-dla-kumba", "Douala – Mbanga – Kumba", "Douala – Mbanga – Kumba", "Camrail",
      ["Douala Bessengué", "Mbanga", "Gare de Kumba"], network="Camrail"),
]

# Angola (ao_sources.md): the three state railways, Cape gauge, every OSM station on a traced
# line a stop.
CFL, CFB, CFM = "Caminho de Ferro de Luanda", "Caminho de Ferro de Benguela", \
    "Caminho de Ferro de Moçâmedes"
LINES["ao"] = [
    L("ao-cfl-sub", "Luanda (Bungo) – Baía", "Luanda (Bungo) – Baía", CFL,
      ["Bungo", "Rotunda", "Filda", "Viana", "Entroncamento", "Baía"], network=CFL),
    L("ao-cfl-malanje", "Baía – Malanje", "Baía – Malanje", CFL,
      ["Baía", "Catete", "Zenza do Itombe", "N'dalantando", "Lucala", "Cacuso", "Malanje"],
      network=CFL),
    L("ao-cfl-dondo", "Zenza do Itombe – Dondo", "Zenza do Itombe – Dondo", CFL,
      ["Zenza do Itombe", "Cassoalala", "Dondo"], suspended=True, network=CFL),
    L("ao-cfb-benguela", "Lobito – Benguela", "Lobito – Benguela", CFB,
      ["Lobito", "Catumbela", "Luongo", "Aeroporto@13.4973,-12.4891", "Benguela"],
      network=CFB),
    L("ao-cfb-main", "Lobito – Huambo – Luau", "Lobito – Huambo – Luau", CFB,
      ["Lobito", "Catumbela", "Luongo", "Mina", "Cubal", "Ganda", "Huambo", "Cuíto",
       "Camacupa", "Luena", "Luau"], network=CFB),
    L("ao-cfm-namibe", "Namibe – Lubango", "Namibe – Lubango", CFM,
      ["Moçâmedes (Km 0)", "Caraculo (Km 76)", "Bibala (Km162)", "Lubango (Km 246)"],
      network=CFM),
    L("ao-cfm-menongue", "Lubango – Matala – Menongue", "Lubango – Matala – Menongue", CFM,
      ["Lubango (Km 246)", "Matala (Km 424)", "Menongue (Km 756)"], network=CFM),
]

# DR Congo (cd_sources.md): ONATRA/SCTP's Kinshasa urban train, relaunched 19 Aug 2026 between
# the Gare Centrale (OSM's "Kinshasa Est") and N'djili; Kinshasa - Matadi greyed (suspended
# about April 2026, no resumption found).
LINES["cd"] = [
    L("cd-kin-urbain", "Train urbain : Gare Centrale – Ndjili", "Kinshasa urban train: "
      "Central Station – N'djili", "SCTP", ["Kinshasa Est", "Limete Amicongo", "Masina Mapela",
                                             "Ndjili Aéroport"], network="SCTP"),
    L("cd-kin-matadi", "Kinshasa – Matadi", "Kinshasa – Matadi", "SCTP",
      ["Kinshasa Est", "Kimwenza", "Kasangulu", "Kisantu–Inkisi",
       "Gare centrale@14.855,-5.262", "Lukala", "Kenge", "Matadi"], suspended=True,
      network="SCTP"),
]
