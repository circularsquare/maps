"""The Bahamas, 2010 census country of citizenship (island reports, Table 9.0) -> node
(sources/bs_census.py, sources/bs.md). No language question: each citizenship is drawn on its
country's language, the convention of sources/fr_build.COUNTRY_LANG, with these departures for
the Bahamas: Canada -> English (COUNTRY_LANG's French is France's own immigrant mix), Belgium and
Switzerland by their larger language.

  BAHAMAS (not white): Bahamian Creole (Glottolog baha1260), the vernacular of Afro-Bahamians.
  BAHAMAS (white): English; White Bahamian English is an English dialect, not a creole.
  OTHER COMMONWEALTH / NON-COMMONWEALTH COUNTRIES: pooled citizenships, unnamed -> `other`.
Spelling variants in the reports (JAMACIA, TRINADAD...) are merged in sources/bs_census.py.
"""
EN = "indoeuropean.germanic.english"
ES = "indoeuropean.romance.spanish"
NAMES = {
    "BAHAMAS (not white)": "creole.english_based.bahamian",
    "BAHAMAS (white)": EN,
    "HAITI": "creole.french_based.haitian",
    "JAMAICA": "creole.english_based.jamaican",
    "DOMINICA": "creole.french_based.antillean",
    "ST. LUCIA": "creole.french_based.antillean",
    # the English-speaking Caribbean on its own creole, as Bahamians are on theirs (Glottolog:
    # baja1265, creo1235, trin1276, turk1310, anti1245 which covers St Kitts, vinc1243, gren1247)
    "BARBADOS": "creole.english_based.bajan",
    "GUYANA": "creole.english_based.guyanese",
    "TRINIDAD AND TOBAGO": "creole.english_based.trinidadian",
    "TURKS AND CAICOS ISLANDS": "creole.english_based.turks_caicos",
    "ANTIGUA/BARBUDA": "creole.english_based.antiguan",
    "ST KITTS/NEVIS": "creole.english_based.antiguan",
    "ST. VINCENT AND THE GRENADINES": "creole.english_based.vincentian",
    "GRENADA": "creole.english_based.grenadian",
    # English-speaking countries and territories (COUNTRY_LANG: English); the Caymans and BVI
    # have a handful of nationals here and Belize's Kriol speakers are a minority of Belizeans
    **{k: EN for k in ("U.S.A", "CANADA", "BERMUDA", "UNITED KINGDOM", "IRELAND", "AUSTRALIA",
                       "NEW ZEALAND", "CAYMAN ISLANDS", "BRITISH VIRGIN ISLANDS", "BELIZE")},
    # Spanish-speaking
    **{k: ES for k in ("CUBA", "DOMINICAN REPUBLIC", "MEXICO", "COSTA RICA", "EL SALVADOR",
                       "GUATEMALA", "HONDURAS", "NICARAGUA", "PANAMA", "ARGENTINA", "CHILE",
                       "COLOMBIA", "ECUADOR", "PERU", "URUGUAY", "VENEZUELA", "SPAIN")},
    "BRAZIL": "indoeuropean.romance.portuguese",
    "PORTUGAL": "indoeuropean.romance.portuguese",
    "FRANCE": "indoeuropean.romance.french",
    "BELGIUM": "indoeuropean.germanic.continental.dutch",
    "ITALY": "indoeuropean.romance.italian",
    "ROMANIA": "indoeuropean.romance.romanian",
    "GERMANY": "indoeuropean.germanic.continental.german",
    "AUSTRIA": "indoeuropean.germanic.continental.german",
    "SWITZERLAND": "indoeuropean.germanic.continental.german",
    "NETHERLANDS AND HOLLAND": "indoeuropean.germanic.continental.dutch",
    "SWEDEN": "indoeuropean.germanic.north.swedish",
    "NORWAY": "indoeuropean.germanic.north.norwegian",
    "DENMARK": "indoeuropean.germanic.north.danish",
    "GREECE": "indoeuropean.hellenic.greek",
    "POLAND": "indoeuropean.slavic.west.polish",
    "RUSSIA": "indoeuropean.slavic.east.russian",
    "UKRAINE": "indoeuropean.slavic.east.ukrainian",
    "BULGARIA": "indoeuropean.slavic.south.bulgarian",
    "ESTONIA": "uralic.estonian",
    "PHILIPPINES": "austronesian.philippine.tagalog",
    "INDONESIA": "austronesian.malayic.indonesian",
    "CHINA": "sinotibetan.sinitic.mandarin",
    "HONG KONG": "sinotibetan.sinitic.cantonese",
    "JAPAN": "japonic.japanese",
    "INDIA": "indoeuropean.indoaryan.central.hindi",
    "SRI LANKA": "indoeuropean.indoaryan.sinhala",
    "EGYPT": "afroasiatic.egyptian_arabic",
    "NIGERIA": "afroasiatic.chadic.hausa",
    "GHANA": "nigercongo.kwa.twi",
    "KENYA": "nigercongo.bantu.swahili",
    "UGANDA": "nigercongo.bantu.ganda",
    "ZIMBABAWE": "nigercongo.bantu.shona",
    "SOUTH AFRICA": "nigercongo.bantu.nguni.zulu",
    "SWAZILAND": "nigercongo.bantu.nguni.siswati",
    "BOTSWANA": "nigercongo.bantu.sotho_tswana.setswana",
    "OTHER COMMONWEALTH": "other",
    "NON-COMMONWEALTH COUNTRIES": "other",
}


def resolve(label):
    if label in NAMES:
        return NAMES[label]
    raise KeyError(f"bs2010: unmapped label {label!r}")
