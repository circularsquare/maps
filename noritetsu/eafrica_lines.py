"""The line lists eafrica_register.py converts (nafrica_lines.py's format): one entry per
passenger line, its stations in order. Point syntax: nafrica_register.py's docstring. Sources
and reasoning: <cc>_sources.md. Mauritius has no list (light rail only: OSM lines)."""
from nafrica_lines import L

# Border points: id -> (lon, lat, [countries]). A list point "#ID" ends a line there;
# borders.EXTRA carries the same point as "e" + ID (handoff_notes/eafrica_build.md).
BORDERS = {}
# OSM's own boundary at a border point where the clip's outline is too coarse:
# id -> {"line": ((lon, lat), (lon, lat)), a segment of OSM's boundary way; "ref": {cc: a point
# inside cc}; "km": how far from the point the clip is redone by this segment}
# (eafrica_register.keep_border_ways).
BORDER_LINES = {}

# Route relations that are no scheduled passenger service, or that OSM maps before they open;
# eafrica_register --clip <cc> drops them (nafrica_register.clip).
NOT_SERVICE = {cc: {} for cc in ("ke", "et", "dj", "mz", "zm", "zw", "tz", "mg", "mw", "ug",
                                 "mu")}

NOT_SERVICE["ke"] = {
    1789208: "Mombasa-Nairobi Railway (metre gauge): no passenger train since the SGR opened in "
             "2017 (ke_sources.md)",
    13180874: "Mombasa-Nairobi Railway, the other direction: as above",
    10167882: "Madaraka Express Nairobi - Suswa: not in any 2026 timetable; the register's "
              "Nairobi - Suswa SGR is greyed",
}

# OSM routes named only by their system: eafrica_register --clip gives each a route_master
# with this name (eafrica_register.name_routes).
ROUTE_NAMES = {
    "ke": {
        8489646: "NCR Nairobi – Syokimau",
        8489647: "NCR Nairobi – Kikuyu",
        8489648: "NCR Nairobi – Ruiru",
        13180873: "NCR Nairobi – Embakasi Village",
        13180875: "NCR Nairobi – Lukenya",
    },
    # Mauritius' Metro Express: the masters' names keep "Phoenix" from before the line reached
    # Curepipe, behind a "⟷" build_model does not read as a direction mark
    "mu": {
        10567273: "Metro Express: Port Louis – Curepipe",
        15351074: "Metro Express: Rose Hill – Réduit",
    },
}

# OSM stations renamed before anything reads them: {cc: {stop id: name}}
# (eafrica_register.tidy). For a station mapped twice under two names.
STATION_NAMES = {}

# Countries whose OSM track is in pieces a few metres apart: `eafrica_register --join <cc>`.
JOIN = set()

LINES = {cc: [] for cc in NOT_SERVICE}

# ------------------------------------------------------------------ Kenya (ke_sources.md)
# Station names as eafrica_register.tidy leaves them ("Kibera Train station" -> "Kibera").
KR = "Kenya Railways"
LINES["ke"] = [
    # The Madaraka Express, three trains a day each way; the stops seat61 gives (June 2026).
    L("sgr", "Mombasa – Nairobi SGR", "Mombasa – Nairobi SGR", KR,
      ["Mombasa SGR Terminus", "Mariakani@39.4606,-3.8602", "Miasenyi@38.9217,-3.6573",
       "Voi@38.5753,-3.4029", "Mtito Andei SGR@38.1707,-2.6903", "Kibwezi@37.9371,-2.4068",
       "Emali@37.4726,-2.0844", "Athi River@36.9905,-1.4680", "Nairobi Terminus"],
      listed_only=True),
    # SGR phase 2A: opened 2019, no train in any 2026 timetable.
    L("sgr2a", "Nairobi – Suswa SGR", "Nairobi – Suswa SGR", KR,
      ["Nairobi Terminus", "Ongata Rongai", "Ngong@36.6814,-1.3504", "Maai Mahiu",
       "Suswa@36.3204,-1.0445"], suspended=True, listed_only=True),
    # Nairobi Commuter Rail over the old main line, and its branches (OSM's NCR routes are
    # OSM lines over these). Every OSM station on them is a stop. The main line passes
    # Nairobi Terminus, where the link trains call at its metre-gauge platform.
    L("lukenya", "Nairobi – Athi River – Lukenya", "Nairobi – Athi River – Lukenya", KR,
      ["Nairobi Central", "Makadara", "Imara Daima", "Nairobi Terminus",
       "Athi River@36.9755,-1.4490", "Lukenya"]),
    # The spur on to Syokimau: the SGR shuttle and NCR trains Nairobi - Syokimau.
    L("syokimau", "Nairobi Terminus – Syokimau", "Nairobi Terminus – Syokimau", KR,
      ["Nairobi Terminus", "Syokimau"]),
    L("embakasi", "Makadara – Embakasi Village", "Makadara – Embakasi Village", KR,
      ["Makadara", "Donholm", "Pipeline", "Embakasi Village"]),
    # Commuter trains to Ruiru, the weekly "safari train" (Fri out, Sun back) to Nanyuki.
    L("nanyuki", "Nairobi – Thika – Nanyuki", "Nairobi – Thika – Nanyuki", KR,
      ["Makadara", "Dandora", "Mwiki", "Githurai", "Kahawa West", "Ruiru", "Thika", "Makuyu",
       "Sagana", "Karatina", "Nano Moru", "Nanyuki"]),
    L("limuru", "Nairobi – Kikuyu – Limuru", "Nairobi – Kikuyu – Limuru", KR,
      ["Nairobi Central", "Dagoretti", "Kikuyu", "Limuru"]),
    # The Kisumu train: suspended July 2025, festive specials only since (greyed).
    L("kisumu", "Limuru – Nakuru – Kisumu", "Limuru – Nakuru – Kisumu", KR,
      ["Limuru", "Uplands (Lari)", "Kijabe", "Naivasha", "Gilgil", "Nakuru", "Njoro", "Molo",
       "Londiani", "Kipkelion@35.4255,-0.2141", "Fort Ternan", "Muhoroni", "Kisumu"],
      suspended=True),
    # The shuttle from the old Mombasa station to Miritini, beside the SGR terminus. OSM has
    # no Mombasa station: --fill adds it where the main line ends, by Mwembe Tayari.
    L("mombasa", "Mombasa – Miritini", "Mombasa – Miritini", KR,
      ["Mombasa Central@39.6617,-4.0578", "Changamwe", "Miritini MGR"]),
]

# ------------------------------------------------------------------ Ethiopia (et_sources.md)
# The Addis Ababa - Djibouti Railway: one train every second day each way, Furi-Labu - Dire
# Dawa and Dire Dawa - Nagad with a night at Dire Dawa (seat61, Nov 2025; ethiopiarailway.com,
# July 2026). Sebeta - Furi-Labu (13.5 km) has no passenger train: left off. Stops: those of
# OSM's train route (6281977) and Metehara; not Addis Ababa-Kality (Indode, the freight
# terminal) or the passing loops. Dewele's stop is the customs and immigration station, mapped
# twice under two names (renamed here). The border point: where OSM's track (ways 967769108,
# 1197034121) crosses OSM's boundary (way 31304862, from the OSM API), 4.0 km past Dewele along the track;
# the app's outline crosses 0.8 km short of it, so near it the clip sides by OSM's boundary.
BORDERS["XDJET1"] = (42.642891, 11.090904, ["dj", "et"])
BORDER_LINES["XDJET1"] = {"line": ((42.63976, 11.09379), (42.64418, 11.08972)),
                          "ref": {"et": (42.6375, 11.0687), "dj": (42.7363, 11.1496)},
                          "km": 3.0}
STATION_NAMES["et"] = {6851442203: "Dewele", -809971740: "Dewele"}
EDR = "Ethio-Djibouti Railway"
LINES["et"] = [
    L("adr", "Addis Ababa – Djibouti Railway", "Addis Ababa – Djibouti Railway", EDR,
      ["Furi-Labu", "Bishoftu", "Mojo", "Adama", "Metehara", "Awash", "Asebot", "Mieso",
       "Mullu", "Bike", "Erer", "Dire Dawa", "Dewele@42.6376,11.0687", "#XDJET1"],
      listed_only=True),
]
# Addis Ababa's light rail: OSM's route masters, named "AA-LRT : Ayat <-> Tor Hailoch", which
# build_model reads as "AA-LRT" for both.
ROUTE_NAMES["et"] = {5697658: "Addis Ababa LRT East–West", 5697659: "Addis Ababa LRT North–South"}
# OSM's one train route is the register line again (rules/et.py SKIP_ROUTES).

# ------------------------------------------------------------------ Djibouti (dj_sources.md)
# The same railway's Djibouti end: the train calls at Ali Sabieh and ends at Nagad (Holhol is
# a passing loop; the port station is freight).
LINES["dj"] = [
    L("adr", "Chemin de fer Addis-Abeba – Djibouti", "Addis Ababa – Djibouti Railway", EDR,
      ["#XDJET1", "Ali Sabieh", "Gare de Nagad"], listed_only=True),
]

# ------------------------------------------------------------------ Mozambique (mz_sources.md)
# CFM's passenger lines (CFM "Transporte de passageiros", seat61 Jan 2026), and CDN's Nampula -
# Cuamba. CFM publishes no stop lists: every OSM station on a line is a stop. Each branch
# starts where OSM's track leaves the line it comes off (--fork).
CFM = "Caminhos de Ferro de Moçambique"
CDN = "Corredor de Desenvolvimento do Norte"
NOT_SERVICE["mz"] = {
    2117373: "Komatipoort - Maputo: no through train, passengers change at Ressano Garcia "
             "(za_sources.md)",
    14496057: "Linha Marromeu - Beira: no passenger train on the Marromeu branch (mz_sources.md)",
    14507272: "Linha Harare - Beira: no train over the border since 2000s; Beira - Machipanda is "
              "the register's",
}
LINES["mz"] = [
    # Maputo - Ressano Garcia daily (seat61), and Maputo's commuter trains to Matola and Machava
    L("ressano", "Linha de Ressano Garcia", "Ressano Garcia Line", CFM,
      ["Maputo Central", "Infulene", "Machava", "Moamba", "Ressano García"]),
    # daily on the Goba line (CFM); no train crosses into Eswatini
    L("goba", "Linha de Goba", "Goba Line", CFM,
      ["~Goba junction@32.46780,-25.90138", "Boane", "Goba"]),
    # Maputo - Chicualacuala twice a week; the commuter trains to Marracuene and Manhiça
    L("limpopo", "Linha do Limpopo", "Limpopo Line", CFM,
      ["Infulene", "Marracuene", "Manhiça", "Magude", "Chókwè", "Mabalane", "Chicualacuala"]),
    # Beira - Machipanda twice a week since Dec 2023 (AIM); Chimoio's station is "Comboios"
    L("machipanda", "Linha de Machipanda", "Machipanda Line", CFM,
      ["Beira", "Dondo", "Nhamatanda", "Gondola", "Comboios", "Manica", "Machipanda"]),
    # Beira - Moatize twice a week (seat61 Jan 2026), over the Sena line to Dona Ana
    # (Mutarara) and the Tete line on; it leaves the Machipanda line at Dondo
    L("sena", "Linha de Sena: Dondo – Moatize", "Sena Line: Dondo – Moatize", CFM,
      ["Dondo", "Inhamitanga", "Caia", "Sena", "Mutarara@35.0698,-17.4362", "Doa", "Moatize"]),
    # CDN's Nampula - Cuamba twice a week (seat61 Jan 2026)
    L("nacala", "Linha de Nacala: Nampula – Cuamba", "Nacala Line: Nampula – Cuamba", CDN,
      ["Nampula", "Iapala", "Malema", "Cuamba"]),
]

# ------------------------------------------------------------------ Zambia (zm_sources.md)
# The Tunduma - Nakonde crossing: where OSM's track (way 200454396) crosses OSM's boundary
# (way 363623831), 0.8 km east of Nakonde station. The app's outline runs 2.5 km further west
# and put Nakonde in Tanzania, so near the point the clip sides by OSM's boundary instead.
BORDERS["XTZZM1"] = (32.763516, -9.315615, ["tz", "zm"])
BORDER_LINES["XTZZM1"] = {"line": ((32.76545, -9.32314), (32.75568, -9.28513)),
                          "ref": {"zm": (32.75789, -9.30943), "tz": (32.76776, -9.31833)},
                          "km": 4.0}
ZRL = "Zambia Railways"
TAZARA = "TAZARA"
NOT_SERVICE["zm"] = {
    8473287: "Mulobezi Train: 'occasional mixed freight and passenger', no timetable; the "
             "register's Livingstone - Mulobezi line is greyed (zm_sources.md)",
}
LINES["zm"] = [
    # The weekly Livingstone - Kitwe train (seat61 Apr 2026), at the stops it calls at.
    L("main", "Livingstone – Lusaka – Kitwe", "Livingstone – Lusaka – Kitwe", ZRL,
      ["Livingstone", "Kalomo", "Choma", "Pemba", "Monze", "Mazabuka", "Kafue", "Lusaka",
       "Kabwe", "Kapiri Mposhi@28.676,-13.970", "Ndola", "Kitwe"], listed_only=True),
    # TAZARA's Zambian half, the weekly Mukuba Express since Feb 2026; every OSM station on
    # it is a stop (the ordinary train's pattern; the express calls at fewer).
    L("tazara", "TAZARA: New Kapiri Mposhi – Nakonde", "TAZARA: New Kapiri Mposhi – Nakonde",
      TAZARA, ["New Kapiri Mposhi", "Mkushi", "Serenje", "Mpika", "Kasama", "Nakonde",
               "#XTZZM1"]),
    # The Mulobezi branch: no timetable found (greyed).
    L("mulobezi", "Livingstone – Mulobezi", "Livingstone – Mulobezi", ZRL,
      ["Livingstone", "Makunka", "Saala", "Mulobezi"], suspended=True),
]

# ------------------------------------------------------------------ Zimbabwe (zw_sources.md)
# NRZ's two trains back since 17 Oct 2025, each weekly each way; the trunk line between them
# is greyed (Bulawayo - Harare suspended since 2020; managing session's call). NRZ publishes no
# stop lists: every OSM station on a line is a stop.
NRZ = "National Railways of Zimbabwe"
NOT_SERVICE["zw"] = {
    5419655: "Chicualacuala - Bulawayo: suspended since 2020 (zw_sources.md)",
    8468165: "Bulawayo - Francistown: suspended since 2020",
    8472183: "Gweru - Masvingo: no passenger train",
    8472184: "Harare - Chinhoyi: no passenger train",
    8472185: "Harare - Shamva: no passenger train",
    8472186: "Bulawayo - Beitbridge: suspended since 2014",
    8472187: "Bulawayo - Chiredzi: suspended since 2020",
    14507272: "Linha Harare - Beira: no train over the border",
}
LINES["zw"] = [
    L("vicfalls", "Bulawayo – Victoria Falls", "Bulawayo – Victoria Falls", NRZ,
      ["Bulawayo", "Nyamandolovu", "Sawmill", "Gwaai", "Dete", "Hwange",
       "Hwange - Thomson Junction", "Victoria Falls"]),
    L("mutare", "Harare – Mutare", "Harare – Mutare", NRZ,
      ["Harare", "Marondera", "Macheke", "Headlands", "Rusape", "Odzi", "Mutare"]),
    L("trunk", "Bulawayo – Gweru – Harare", "Bulawayo – Gweru – Harare", NRZ,
      ["Bulawayo", "Heany", "Shangani", "Gweru", "Kwekwe", "Kadoma", "Chegutu", "Norton",
       "Harare"], suspended=True),
]

# ------------------------------------------------------------------ Tanzania (tz_sources.md)
# TRC's lines as its route diagrams (trc_routes.pdf) cut them, its SGR, and TAZARA's
# Tanzanian half to the Tunduma - Nakonde border point (Zambia's block). Metre-gauge lines:
# every OSM station on them is a stop; the SGR calls at its own stations only.
TRC = "Tanzania Railways Corporation"
STATION_NAMES["tz"] = {-864642709: "Dar es Salaam (Magufuli)"}   # "New Central Railway Station"
MRUAZI = "~Mruazi@38.61076,-5.24470"   # where the Link line meets the Tanga line (--fork)
LINES["tz"] = [
    # SGR, several trains a day (TRC timetable from 3 Jan 2026)
    L("sgr", "SGR: Dar es Salaam – Dodoma", "SGR: Dar es Salaam – Dodoma", TRC,
      ["Dar es Salaam (Magufuli)", "Pugu@39.1252,-6.8845", "Soga@38.8592,-6.8352",
       "Ruvu@38.6734,-6.8101", "Ngerengere@38.1351,-6.7781", "Morogoro@37.6645,-6.7543",
       "Mkata@37.3482,-6.7581", "Kilosa@37.0308,-6.8167", "Kidete@36.6909,-6.6347",
       "Gulwe@36.3914,-6.4404", "Igandu@36.1434,-6.3629", "Dodoma@35.7343,-6.2116"],
      listed_only=True),
    # Dar - Kigoma 2-3 a week (TRC's 2023 timetable; The Citizen). OSM's metre-gauge main
    # line ends at Kamata, 1 km short of the old central station, whose throat is yard track
    # beside the SGR's terminus: the line starts at Kamata.
    L("central", "Central Line: Dar es Salaam – Kigoma", "Central Line: Dar es Salaam – Kigoma",
      TRC, ["Kamata", "Pugu@39.1200,-6.8802", "Ruvu@38.6615,-6.8094",
            "Ngerengere@38.1204,-6.7630", "Morogoro@37.6724,-6.8224", "Kilosa@36.9844,-6.8314",
            "Gulwe@36.4113,-6.4496", "Dodoma@35.7494,-6.1839", "Makutopora", "Manyoni",
            "Itigi", "Tabora", "Urambo", "Kaliua", "Uvinza", "Kigoma"]),
    # weekly Dar - Mwanza
    L("mwanza", "Mwanza Line: Tabora – Mwanza", "Mwanza Line: Tabora – Mwanza", TRC,
      ["Tabora", "Kakola", "Nzubaka", "Bukene", "Isaka", "Shinyanga", "Malampaka",
       "Mwanza South", "Mwanza"]),
    # weekly Dar - Mpanda
    L("mpanda", "Mpanda Line: Kaliua – Mpanda", "Mpanda Line: Kaliua – Mpanda", TRC,
      ["Kaliua", "Uyumbu", "Lumbe", "Ugalla", "Mpanda"]),
    # twice a week Dar - Moshi - Arusha, over the Link line and the Tanga line
    L("link", "Link Line: Ruvu – Mruazi", "Link Line: Ruvu – Mruazi", TRC,
      ["Ruvu@38.6615,-6.8094", "Kidomole", "Mvave", MRUAZI]),
    L("arusha", "Tanga Line: Mruazi – Moshi – Arusha", "Tanga Line: Mruazi – Moshi – Arusha",
      TRC, [MRUAZI, "Korogwe", "Mombo", "Same", "Moshi", "Kikuletwa", "Usa River, Arusha",
            "Arusha"]),
    # no passenger train in TRC's timetable (greyed)
    L("tanga", "Tanga Line: Tanga – Mruazi", "Tanga Line: Tanga – Mruazi", TRC,
      ["Tanga", "Muheza", MRUAZI], suspended=True),
    L("singida", "Singida Line: Manyoni – Singida", "Singida Line: Manyoni – Singida", TRC,
      ["Manyoni", "Issuna", "Ikungi", "Singida"], suspended=True),
    # TAZARA's Mukuba Express, weekly since 10 Feb 2026; Dar's TAZARA commuter trains to
    # Mwakanga run over its first 20 km
    L("tazara", "TAZARA: Dar es Salaam – Tunduma", "TAZARA: Dar es Salaam – Tunduma", TAZARA,
      ["Dar es Salaam", "Yombo", "Ifakara", "Mlimba", "Makambako", "Mbeya", "Mpemba",
       "#XTZZM1"]),
]

# ------------------------------------------------------------------ Madagascar (mg_sources.md)
# Madarail's northern network (TCE, MLA) and the FCE. Every OSM station on a line is a stop.
# OSM has no Toamasina station: --fill adds it on the line by the port, where the station is.
MADARAIL = "Madarail"
FCE = "Fianarantsoa-Côte Est"
NOT_SERVICE["mg"] = {
    8503490: "Moramanga - Ambatondrazaka: no timetable found; the register's MLA line is greyed "
             "(mg_sources.md)",
}
LINES["mg"] = [
    # Antananarivo's urban train since 16 Dec 2025, two pairs a day (Railway Gazette)
    L("urbain", "Train urbain : Soarano – Ambohimanambola",
      "Urban train: Soarano – Ambohimanambola", MADARAIL,
      ["Soarano", "Mandroseza", "Ambohimanambola"]),
    # the rest of the TCE to Moramanga: no passenger train (greyed)
    L("tce-w", "TCE : Ambohimanambola – Moramanga", "TCE: Ambohimanambola – Moramanga",
      MADARAIL, ["Ambohimanambola", "Manjakandriana", "Anjiro", "Moramanga"], suspended=True),
    # weekly Moramanga - Toamasina since June 2023 (newsmada)
    L("tce-e", "TCE : Moramanga – Toamasina", "TCE: Moramanga – Toamasina", MADARAIL,
      ["Moramanga", "Andasibe", "Fanovana", "Brickaville", "Ambila Lemaitso",
       "Andranokoditra", "Tampina", "Toamasina@49.4160,-18.1633"]),
    L("mla", "MLA : Moramanga – Ambatondrazaka", "MLA: Moramanga – Ambatondrazaka", MADARAIL,
      ["Moramanga", "Amboasary", "Andaingo", "Ambatondrazaka"], suspended=True),
    # 2-3 a week when the line is open (it shuts for months after cyclones)
    L("fce", "FCE : Fianarantsoa – Manakara", "FCE: Fianarantsoa – Manakara", FCE,
      ["Fianarantsoa", "Sahambavy", "Tolongoina", "Manakara"]),
]

# ------------------------------------------------------------------ Malawi (mw_sources.md)
# CEAR's weekly passenger train (resumed Aug 2022): Limbe - Balaka, on to Nayuchi at the
# Mozambique border and back (managing session: counted as running). CEAR's 2015 timetable km:
# Limbe 0, Blantyre 8, Nkaya 96, Balaka 112, Nkaya - Nayuchi 99. Every OSM station is a stop.
# No train crosses at Nayuchi. XMWMZ1 is there only for the clip: the app's outline puts
# Nayuchi station in Mozambique, so near it the clip sides by OSM's boundary (way 1413800808);
# no line ends at it and it needs no borders.EXTRA entry.
BORDERS["XMWMZ1"] = (35.877936, -14.981979, ["mw", "mz"])
BORDER_LINES["XMWMZ1"] = {"line": ((35.88745, -14.96187), (35.8761, -14.98586)),
                          "ref": {"mw": (35.87313, -14.97926), "mz": (35.8900, -14.9890)},
                          "km": 3.0}
CEAR = "Central East African Railways"
NOT_SERVICE["mw"] = {
    14482643: "Balaka - Bilila: freight only (mw_sources.md)",
    14482644: "Bilila - Balaka: as above",
}
LINES["mw"] = [
    L("limbe", "Limbe – Balaka", "Limbe – Balaka", CEAR,
      ["Limbe", "Blantyre", "Lirangwe", "Nkaya", "Balaka"]),
    L("nayuchi", "Nkaya – Nayuchi", "Nkaya – Nayuchi", CEAR,
      ["Nkaya", "Liwonde", "Nansama", "Namanja", "Nayuchi"]),
]

# ------------------------------------------------------------------ Uganda (ug_sources.md)
# URC's Kampala - Mukono commuter trains, weekdays (Matooke Republic, 31 Aug 2026). OSM maps
# Kampala, Namboole and Namanve; Mukono's halt is not mapped: --fill adds it where the main
# line passes closest to the town (2.8 km south of the centre; its exact place is unknown).
URC = "Uganda Railways Corporation"
LINES["ug"] = [
    L("mukono", "Kampala – Mukono", "Kampala – Mukono", URC,
      ["Kampala", "Namboole", "Namanve", "Mukono@32.7626,0.3293"]),
]
