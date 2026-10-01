"""Lithuania: LTG Infra (IM 1224) alone. RINF's id is the line's own name with the places run
together in ASCII, "Kyviskes-Vilnius-Kaisiadorys-Kaunas-KazluRuda", "Mazeikiai-LV" (to the
Latvian border). LTG has no public line numbers: Wikidata's P1671 on Lithuanian lines are
pairs like "524-525" on items for single sections ("Kaunas - Palemonas railway line"), which
join nothing here, and OSM's few route=railway relations carry a name in `ref` ("Vilnius -
Kena"), so `osm_ref` returns None and nothing is numbered. Lines are named from the id
(`id_name`, needs rinf.py to read it for lines without a number), with the Lithuanian letters
put back from `NAMES`: "Kyviškės–Vilnius–Kaišiadorys–Kaunas–Kazlų Rūda".

Points' names are the same ASCII ("KazluRuda", "Valstybessiena" for valstybės siena, the state
border); rinf.norm folds "Kazlų Rūda" to the same key, so they match OSM's stations as they
are. Every point has a coordinate, on geo:hasGeometry as "POINT (+22.695635 56.196012)"; the
leading plus needs rinf.py's WKT pattern to allow it, or none is read.

Rail Baltica's standard-gauge Kaunas - Polish border line is in RINF as
"Palemonas-KazluRuda-Sestokai-Mockava-PL"; the broad-gauge line beside it from Kazlų Rūda is
"KazluRuda-Mockava-PL". lt_sources.md has which the trains use.
"""

NAMES = {
    "Kyviskes-Vilnius-Kaisiadorys-Kaunas-KazluRuda": "Kyviškės–Vilnius–Kaišiadorys–Kaunas–Kazlų Rūda",
    "Palemonas-Gaiziunai-Radviliskis-Siauliai-Klaipeda": "Palemonas–Gaižiūnai–Radviliškis–Šiauliai–Klaipėda",
    "Palemonas-KazluRuda-Sestokai-Mockava-PL": "Palemonas–Kazlų Rūda–Šeštokai–Mockava (Rail Baltica)",
    "KazluRuda-Mockava-PL": "Kazlų Rūda–Šeštokai–Mockava",
    "KazluRuda-Kybartai-RU": "Kazlų Rūda–Kybartai",
    "Kaisiadorys-Gaiziunai": "Kaišiadorys–Gaižiūnai",
    "Lentvaris-Marcinkonys": "Lentvaris–Varėna–Marcinkonys",
    "NaujojiVilnia-Turmantas-LV": "Naujoji Vilnia–Turmantas",
    # The three in KEEP_ONLY are named for the part that is built.
    "Vilnius-Stasylos-BY": "Vilnius–Oro uostas",
    "Kyviskes-Kena-BY": "Kyviškės–Kena",
    "Kyviskes-Vaidotai-Paneriai": "Kyviškės–Vaidotai–Paneriai",
    "Vaidotai-Valciunai": "Vaidotai–Valčiūnai",
    "SeniejiTrakai-Trakai": "Senieji Trakai–Trakai",
    "Sestokai-Alytus": "Šeštokai–Alytus",
    "Radviliskis-Rokiskis-LV": "Radviliškis–Panevėžys",
    "Radviliskis-Pagegiai-RU": "Radviliškis–Tauragė–Pagėgiai",
    "Radviliskis-Pakruojis-Petrasiunai": "Radviliškis–Pakruojis–Petrašiūnai",
    "Siauliai-Joniskis-LV": "Šiauliai–Joniškis",
    "Silenai-Jonaitiskiai": "Šilėnai–Jonaitiškiai",
    "Kuziai-Mazeikiai-Bugeniai": "Kužiai–Mažeikiai–Bugeniai",
    "Mazeikiai-LV": "Mažeikiai–Reņģe",
    "Klaipeda-Pagegiai": "Klaipėda–Šilutė",
    "Kretinga-Darbenai": "Kretinga–Darbėnai",
    "Akmene-Alkiskiai": "Akmenė–Alkiškiai",
    "Jonava-Rizgonys": "Jonava–Rizgonys",
    "Palemonas-Rokai-Jiesia": "Palemonas–Rokai–Jiesia",
    "Rokai-Jiesia": "Rokai–Jiesia",
    "Rimkai-Draugyste": "Rimkai–Draugystė",
    "Svencioneliai-Utena": "Švenčionėliai–Utena",
}


def lt_id_name(lid, _uop):
    return NAMES.get(lid) or (lid or "").replace("-", "–") or None


# WHICH LINES CARRY PASSENGERS. OSM has 14 Lithuanian train routes, so for most lines the
# check is LTG Link's own route list (ltglink.lt answers 403 to scripts; lt.wikibooks's
# "Traukinių tvarkaraščiai/2025" copies it, and a search of ltglink.lt confirms Kaunas -
# Kybartai and Vilnius - Kena): Vilnius - Kaunas, Šiauliai, Klaipėda, Trakai, Varėna
# (Marcinkonys), Ignalina/Visaginas, Kena, the airport; Kaunas - Šiauliai, Marijampolė,
# Kybartai; Klaipėda - Šiauliai, Radviliškis; Kretinga - Klaipėda - Šilutė; Šiauliai -
# Mažeikiai, Šiauliai - Panevėžys; Vilnius - Rīga, Vilnius - Mockava (- Kraków).
#
# Lines with no passenger train whose ends are stations, so build_model would never question
# them: Jonava - Rizgonys (the Achema works), the Vilnius freight bypass Kyviškės - Vaidotai -
# Paneriai, Radviliškis - Pakruojis - Petrašiūnai (its 2.4 km to Durpynas lies beside the
# main line), the Šiauliai bypass Šilėnai - Jonaitiškiai, and Radviliškis - Tauragė - Pagėgiai.
# And Rimkai - Draugystė, Klaipėda's port branch (Draugystė is a freight station), which OSM's
# Kretinga - Klaipėda - Šilutė route lies close enough beside to read as ridden.
FREIGHT = {"Jonava-Rizgonys", "Kyviskes-Vaidotai-Paneriai", "Radviliskis-Pakruojis-Petrasiunai",
           "Silenai-Jonaitiskiai", "Radviliskis-Pagegiai-RU", "Rimkai-Draugyste"}

# Lines ridden over part of their length, kept only between these points: the old Minsk line
# as far as Vilnius Airport (Oro uostas), Radviliškis - Rokiškis as far as Panevėžys
# (Šiauliai - Panevėžys trains; nothing runs on to Kupiškis and Rokiškis), and Klaipėda -
# Pagėgiai as far as Šilutė. Past those, the RINF stops (Kirtimai, Valčiūnai, Jašiūnai;
# Karsakiškis ... Tindžiuliai; Pagėgiai) are OSM stations, so the sections between them were
# kept unquestioned.
KEEP_ONLY = {
    "Vilnius-Stasylos-BY": {"Vilnius", "Orouostas"},
    # Kretinga - Klaipėda - Šilutė trains; Šilutė - Pagėgiai (36 km, Stoniškiai merged away
    # as a non-stop) has none.
    "Klaipeda-Pagegiai": {"Klaipeda", "Rimkai", "Gruzeikiai", "Dituva", "Priekule", "Vilkyciai",
                          "Kukorai", "Silute"},
    "Radviliskis-Rokiskis-LV": {"Radviliskis", "Seduva", "Laba", "Labuciai", "Gustonys",
                                "Panevezys"},
}


def lt_fix(secs, points):
    """rinf.py's `fix`: cut the lines in KEEP_ONLY back to their ridden part."""
    before = len(secs)
    keep = []
    for s in secs:
        names = KEEP_ONLY.get(s["base"])
        if names and not ({points.get(s["a"], {}).get("name"),
                           points.get(s["b"], {}).get("name")} <= names):
            continue
        keep.append(s)
    secs[:] = keep
    return [f"KEEP_ONLY left out {before - len(secs)} sections past the ridden part"]


COUNTRY = {
    # Wikidata's route numbers here join nothing (docstring), so none are fetched.
    "iso3": "LTU", "wikidata": None, "langs": ["lt"],
    "osm_ref": lambda _r: None,
    "id_name": lt_id_name,
    "skip_line": lambda lid: lid in FREIGHT,
    "fix": lt_fix,
    "im": {"1224_IM": "LTG Infra"},
}
