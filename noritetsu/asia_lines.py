"""The line lists asia_register.py converts (lk_register.py's point syntax: "Name", "Name=km",
"Name@lon,lat", "~Junction@lon,lat", "#BORDER"). One entry per passenger line, its stations in
order. Sources and the reasoning for every call: kh_, la_, ph_, mm_, mn_, np_sources.md."""
from lk_register import L

# ============================================================== border points
# id -> (lon, lat, [countries]). Each is where OSM's track crosses OSM's national boundary
# (asia agent, 2026-10-08, from the extract's boundary relation). The Thai two are already in
# borders.EXTRA (th agent); the others are proposed for it (handoff_notes/asia_build.md).
BORDERS = {
    "xNongKhaiThanaleng": (102.715092, 17.880451, ["la", "th"]),     # Friendship Bridge
    "xAranyaprathetPoipet": (102.550138, 13.661698, ["kh", "th"]),   # Poipet - Ban Khlong Luk
    "xBotenMohan": (101.687244, 21.179426, ["cn", "la"]),            # in the Friendship Tunnel
    "xNaushkiSukhbaatar": (106.096798, 50.334187, ["mn", "ru"]),     # way 27366656
    "xZamynUudErenhot": (111.946562, 43.690326, ["cn", "mn"]),       # way 232643093, middle of 3
    "xEreentsavSolovyevsk": (115.742761, 49.885818, ["mn", "ru"]),   # ways 755589400/1435673744
    "xJaynagarInarwa": (86.137098, 26.606175, ["in", "np"]),         # way 1433667171
}

# ============================================================== Cambodia
# Royal Railway's two daily trains (seat61, read 2026-10-08): Phnom Penh - Battambang and
# Phnom Penh - Sihanoukville. Battambang - Poipet has had no train since the pandemic (a short
# 2024 revival ended; the border has been shut since the 2025 fighting): its own line, greyed.
RRC = "Royal Railway"
KH = [
    L("north", "ខ្សែផ្លូវដែកភាគខាងជើង", ["Phnom Penh", "Bat Doeng", "Kraing Skea", "Pursat", "Maung Russey",
                                    "Phnom Thipdey", "Battambang"],
      RRC, name_en="Northern Line", colour="#c0392b"),
    L("north-poipet", "ខ្សែផ្លូវដែកភាគខាងជើង (បាត់ដំបង - ប៉ោយប៉ែត)",
      ["Battambang", "Mongkol Borey", "Sisophon", "Poipet", "#xAranyaprathetPoipet"],
      RRC, name_en="Northern Line (Battambang - Poipet)", suspended=True, colour="#c0392b",
      note="no train since 2020 apart from a few months in 2024; border closed"),
    L("south", "ខ្សែផ្លូវដែកភាគខាងត្បូង",
      ["Phnom Penh", "Takeo", "Touk Meas", "Kep", "Kampot", "Prey Nob", "Sihanoukville"],
      RRC, name_en="Southern Line", colour="#1f4e9c"),
]

# ============================================================== Laos
# The Laos-China Railway at its passenger stations with en.wikipedia's km from Boten (the
# register's chainage), and SRT's metre-gauge line over the Friendship Bridge to Khamsavath,
# which asia_register gives Thailand's Nong Khai line id (`join`), so Nong Khai -> Khamsavath is
# one ride. Its trains: SRT 133/134 Bangkok - Khamsavath, 147/148 Udon Thani - Khamsavath.
LCR = "ບໍລິສັດ ທາງລົດໄຟ ລາວ-ຈີນ"   # Laos-China Railway Company
SRT = "การรถไฟแห่งประเทศไทย"          # State Railway of Thailand
LA = [
    L("lcr", "ທາງລົດໄຟ ລາວ-ຈີນ",
      ["#xBotenMohan", "Boten=0", "Namor=28", "Muang Xai=67", "Muang Nga=113",
       "Luang Prabang=168", "Kasi=239", "Vang Vieng=283", "Phonhong=342", "Vientiane=406"],
      LCR, name_en="Laos-China Railway", colour="#d62728", listed_only=True,
      note="passenger stations only; OSM's other stations on the line are passing loops"),
    L("thanaleng", "Thanaleng line",
      ["#xNongKhaiThanaleng", "Thanaleng", "Khamsavath"], SRT,
      name_en="Thanaleng line", listed_only=True),
]

# ============================================================== the Philippines
# PNR's metre-gauge South Main Line, cut where trains run and where they do not (ph_sources.md
# "Build"). LRT-1, LRT-2 and MRT-3 are OSM lines.
PNR = "Philippine National Railways"
PH = [
    L("ipc", "South Main Line (Calamba - Lucena)",
      ["Calamba", "Masili", "San Pablo", "Tiaong", "Candelaria",
       "Lucena@121.61345,13.92684"], PNR,
      note="the Inter-Provincial Commuter, Calamba - San Pablo - Lucena"),
    L("bicol", "South Main Line (Lupi - Naga)",
      ["Lupi@122.90751,13.78829", "Sipocot", "Libmanan@123.05792,13.69367", "Naga"], PNR,
      note="the Bicol Commuter, Lupi Viejo - Sipocot - Naga"),
    L("legazpi", "South Main Line (Naga - Legazpi)",
      ["Naga", "Pili", "Iriga", "Polangui", "Ligao", "Travesia Guinobatan", "Legazpi"], PNR,
      suspended=True, note="no train since Typhoon Uwan (10 Nov 2025) damaged the Guinobatan bridge"),
    # Tutuban - Calamba and Tutuban - Governor Pascual (closed 27-28 March 2024 for the North-South
    # Commuter Railway's works) are not built: OSM has no rail track left there to draw them on.
    L("bicol-mid", "South Main Line (Lucena - Lupi)",
      ["Lucena@121.61345,13.92684", "Agdangan", "Gumaca", "Hondagua", "Tagkawayan",
       "Del Gallego", "Ragay", "Lupi@122.90751,13.78829"], PNR, suspended=True,
      note="no regular train since the Bicol Express stopped (2014)"),
]

# ============================================================== Mongolia
# UBTZ's broad-gauge lines (mn_sources.md "Build"). Stations OSM names only in Cyrillic are
# written "Name@lon,lat" (the engine's key folds Cyrillic away, so the coordinate finds them).
UBTZ = "Улаанбаатар төмөр зам"
TMR = "Транс-Монголын төмөр зам"
UB = "Ulan Bator"
SALKHIT = "Salkhit@105.86748,49.19910"
DARKHAN = "Darkhan@105.93024,49.48461"
MN = [
    L("north", f"{TMR} (Сүхбаатар – Улаанбаатар)",
      ["#xNaushkiSukhbaatar", "Sükhbaatar", "Dulaan", DARKHAN, "Darkhan-2", SALKHIT,
       "Nomgon@105.92405,49.03666", "Züünkharaa", "Batsümber@106.74834,48.36329", UB],
      UBTZ, name_en="Trans-Mongolian Railway (Sükhbaatar - Ulaanbaatar)", colour="#1f4e9c"),
    L("south", f"{TMR} (Улаанбаатар – Замын-Үүд)",
      [UB, "Amgalan", "Bagakhangai", "Naran-elgen", "Choir", "Airag", "Sainshand", "Urgun",
       "Ulaanuul", "Zamyn-Üüd",
       "#xZamynUudErenhot"],
      UBTZ, name_en="Trans-Mongolian Railway (Ulaanbaatar - Zamyn-Üüd)", colour="#1f4e9c"),
    L("erdenet", "Салхит – Эрдэнэт", [SALKHIT, "Khutul", "Orkhontuul", "Erdenet"], UBTZ,
      name_en="Salkhit - Erdenet", colour="#2e9e4f"),
    L("sharyngol", "Дархан – Шарын гол", [DARKHAN, "Sharyngol"], UBTZ,
      name_en="Darkhan - Sharyn Gol", colour="#e07b00"),
    L("choibalsan", "Эрээнцав – Чойбалсан",
      ["#xEreentsavSolovyevsk", "Ereentsav@115.73532,49.87327", "Khukh nuur@115.55365,49.60448",
       "Khavirga@115.20324,48.81927", "Kherlen@114.89632,48.43610",
       "Bayantümen@114.56554,48.11365"],
      UBTZ, name_en="Ereentsav - Choibalsan", suspended=True, colour="#8a4baf",
      note="no scheduled passenger train found"),
]

# ============================================================== Nepal
# Nepal Railway Company's broad-gauge line from the Indian border (Jaynagar) to Bhangaha
# (OSM's "Bijalpura"), np_sources.md "Build". India's Jaynagar - border piece is proposed to
# in_register (handoff_notes/asia_build.md).
NRC = "नेपाल रेलवे कम्पनी"    # Nepal Railway Company
NP = [
    L("janakpur", "जयनगर–जनकपुर–भंगाहा रेलमार्ग",
      ["#xJaynagarInarwa", "Inarwa", "Khajuri", "Bideha", "Perbaha", "Janakpur Railway station",
       "Kurtha", "लोहारपट्टी रेलवे स्टेशन@85.84656,26.79665", "Bijalpura"], NRC,
      name_en="Jaynagar - Janakpur - Bhangaha railway", colour="#c0392b"),
]

# ============================================================== Myanmar
# Myanma Railways' metre-gauge lines as en.wikipedia's "List of railway stations in Myanmar"
# heads them, each point an OSM station at its own coordinate (OSM's names are Burmese with a
# name:en; spellings differ from the list's). What runs: mm_sources.md "Build". Unknown is
# greyed. Yangon's Circular and suburban routes are OSM lines.
MR = "မြန်မာ့မီးရထား"            # Myanma Railways
YGN = "Yangon Central Railway@96.16194,16.78107"
BAGO = "Bago@96.47490,17.33494"
THAZI = "Thazi Station@96.05796,20.85366"
MDY = "Mandalay Station@96.08623,21.97689"
PYAY = "Pyay@95.21667,18.82051"
PYIN = "Pyin Oo Lwin Station@96.46552,22.03670"
GOK = "Goke Hteik Station@96.85567,22.33790"
KALAW = "Kalaw Station@96.56863,20.62913"
SHWENYAUNG = "Shwenyaung Station@96.94253,20.76570"
MLM = "Mawlamyine@97.63604,16.47148"
YWATAUNG = "Ywar Htaung@95.97127,21.90490"
NABA = "Na Bar Station@96.18103,24.25247"
TDG = "Taungdwingyi Station@95.54041,20.00188"
PYINMANA = "Pyinmana Station@96.20669,19.73618"
MONYWA = "Monywa@95.13663,22.11567"
PAKOKKU = "Pakokku Station@95.06503,21.34945"
BAGAN = "Bagan Station@94.93522,21.15443"
KPD = "Kyaukpadaung Station@95.12578,20.83096"
MEIKTILA = "Meiktila Station@95.86074,20.87940"
MYINGYAN = "Myingyan Station@95.39602,21.45929"


def _mm(lid, name, pts, suspended=True, note=""):
    return L(lid, name, pts, MR, name_en=name, suspended=suspended, note=note)


MM = [
    # running (mm_sources.md "Build": what runs)
    _mm("ygn-mdy", "Yangon–Mandalay line",
        [YGN, BAGO, "Taungoo Station@96.43730,18.93535", PYINMANA,
         "Yamethin Station@96.13690,20.42877", THAZI, "Kyaukse Station@96.13518,21.60772", MDY],
        suspended=False),
    _mm("ygn-mlm", "Yangon–Mawlamyine line",
        [BAGO, "Thaton@97.35820,16.91587", "Moke Ta Ma@97.59457,16.53225", MLM],
        suspended=False, note="from Bago, where it leaves the Mandalay line"),
    _mm("ygn-pyay", "Yangon–Pyay line",
        [YGN, "Insein@96.10685,16.88497", "Hmawbi@96.04880,17.10082",
         "Gyobingauk@95.64826,18.23108", PYAY], suspended=False),
    _mm("pathein", "Kyangin–Hinthada–Pathein line",
        ["Pathein@94.74135,16.77599", "Hinthada@95.45176,17.64631", "Kyangin@95.24559,18.33287"],
        suspended=False),
    _mm("thazi-shwenyaung", "Thazi–Shwenyaung line",
        [THAZI, "Pyi Nyaung Station@96.39545,20.78308", KALAW, SHWENYAUNG], suspended=False),
    _mm("lashio-gokteik", "Mandalay–Lashio line (Pyin Oo Lwin – Gokteik)",
        [PYIN, "Nawnghkio Station@96.79414,22.33642", GOK], suspended=False),
    # greyed: no evidence of a passenger train since 2021
    _mm("lashio-mdy", "Mandalay–Lashio line (Mandalay – Pyin Oo Lwin)", [MDY, PYIN]),
    _mm("lashio", "Mandalay–Lashio line (Gokteik – Lashio)",
        [GOK, "Kyaukme Station@97.03121,22.54389", "Hsipaw Station@97.29433,22.61890",
         "Lashio Station@97.73112,22.97334"]),
    # from Sagaing: OSM has no track over the Irrawaddy (the Inwa bridge) to Mandalay
    _mm("myitkyina", "Mandalay–Myitkyina line",
        ["Sagaing@95.98558,21.88070", YWATAUNG, "Kanbalu@95.51482,23.20428", "Wuntho@95.69231,23.89994", NABA,
         "Mohnyin Station@96.36099,24.77861", "Myitkyina Station@97.39869,25.38033"]),
    _mm("katha", "Naba–Katha line", [NABA, "Katha Station@96.33732,24.17698"]),
    _mm("madaya", "Madaya line", [MDY, "Madaya Station@96.10945,22.20970"]),
    _mm("monywa", "Sagaing–Monywa line", [YWATAUNG, MONYWA]),
    _mm("ye-u", "Monywa–Ye-U line", [MONYWA, "Ye Htwet@95.18489,22.53587", "Ye-U@95.41544,22.75169"]),
    _mm("tanintharyi", "Tanintharyi line",
        [MLM, "Thanbyuzayat Station@97.72222,15.97419", "Ye@97.84364,15.24365",
         "Dawei Station@98.22579,14.09178"]),
    # to Taunggyi: OSM's track on to Nansang and Mong Nai is not joined to it
    _mm("taunggyi", "Shwenyaung–Taunggyi line", [SHWENYAUNG, "Taung Gyi Station@97.05166,20.74767"]),
    _mm("lawksawk", "Shwenyaung–Lawksawk line", [SHWENYAUNG, "Lawksawk Station@96.87718,21.24450"]),
    # from Aungban (Aungpan), where it leaves the Shwenyaung line
    _mm("loikaw", "Loikaw line", ["Aungpan Station@96.63220,20.65416", "Loikaw Station@97.21981,19.66603"]),
    _mm("pyinmana-tdg", "Pyinmana–Taungdwingyi line", [PYINMANA, TDG]),
    _mm("bagan", "Taungdwingyi–Bagan line", [TDG, KPD, BAGAN]),
    _mm("myingyan", "Thazi–Myingyan line", [THAZI, MEIKTILA, MYINGYAN]),
    _mm("pakokku", "Pakokku–Kalay line", [BAGAN, PAKOKKU, "Kalay Station@94.02933,23.18940"]),
    _mm("aunglan", "Pyay–Aunglan line", [PYAY, "Hlay Wun Station@95.33318,19.64332", TDG]),
]

LINES = {"kh": KH, "la": LA, "ph": PH, "mn": MN, "np": NP, "mm": MM}

# OSM stop nodes that are no stop of a line here, dropped after --clip
# (asia_register.drop_stops): {cc: {node id: why}}.
_RAILBUS = "the Ulaanbaatar railbus's halt (the railbus is not built; main-line trains pass)"
DROP_STOPS = {
    "mn": {
        10073823020: _RAILBUS, 10073815635: _RAILBUS, 10073799147: _RAILBUS,
        10073778041: _RAILBUS, 10073778042: _RAILBUS, 10073768618: _RAILBUS,
        10073768619: _RAILBUS, 10073799146: _RAILBUS, 10073823019: _RAILBUS,
        10074677642: _RAILBUS, 10074677643: _RAILBUS,
        8301672403: "a second node of Orkhon station, 64 m from the first",
        -229560493: "Tsomog's station building, beside its station node",
    },
}
NOT_SERVICE = {
    "kh": {}, "la": {},
    "ph": {
        8545505: "PNR Shuttle Service Governor Pascual - FTI: ended March 2024",
        10015475: "PNR Shuttle Service FTI - Governor Pascual: ended March 2024",
        9165727: "PNR Metro North Commuter: ended March 2024",
        9165728: "PNR Metro North Commuter: ended March 2024",
        9312509: "Metro Manila Subway: under construction",
        9352993: "Metro Manila Subway: under construction",
        9312512: "Metro Manila Subway's route_master",
    },
    "mn": {
        19785754: "Ulaanbaatar Railbus (route=tram): no evidence it runs in 2026 (mn_sources.md)",
    },
}

# Lines that continue a neighbour's register line over a border and take its id:
# {cc: {list id: (neighbour, border point)}}.
JOIN = {"la": {"thanaleng": ("th", "xNongKhaiThanaleng")}}

# Outside numbers for a path over the built register lines: (label, (lon, lat), (lon, lat),
# km, source).
PATH_CHECKS = {
    "kh": [("Phnom Penh - Battambang", (104.91604, 11.57246), (103.19440, 13.09793), 273.0,
            "seat61, Royal Railway's fare distance"),
           ("Phnom Penh - Sihanoukville", (104.91604, 11.57246), (103.51542, 10.64367), 263.0,
            "seat61")],
    "mn": [("Trans-Mongolian, border to border", (106.096798, 50.334187), (111.946562, 43.690326),
            1110.0, "en.WP Trans-Mongolian Railway, 1,110 km in Mongolia")],
}
