"""Austria, Volkszaehlung 2001, Umgangssprache (everyday language) -> node. sources/at_vz2001.py.

Keyed by Statistik Austria's labels as data/normalized/at.csv carries them: Tabelle 5's ten
columns and Tabelle 14's 45 leaves (the Oesterreich volume's spelling). The question asked for
the language(s) usually spoken in private life; a double answer is folded onto its non-German
half, so "Deutsch" is German alone and every other label is "this language, alone or with
German". The census imputed non-response: no "not stated" row.

CALLS (sources/at.md says more):
  "Burgenland-Kroatisch" (19,412): a new leaf `slavic.south.burgenland_croatian` beside Croatian,
    as Statistik Austria prints it apart from "Kroatisch" (the language of immigrants from
    Croatia and Bosnia). Glottolog: Burgenland Croatian, burg1244, a Chakavian dialect under
    Serbian-Croatian-Bosnian. A sibling, not a child, of Croatian (a child would turn Croatian
    into a group, drawn washed out).
  "Windisch" (568): a new leaf `slavic.south.windisch` beside Slovenian. The census's own box for
    Carinthian Slovene dialect speakers who did not want to be counted as Slovene; no Glottolog
    entry of its own (Slovenian, slov1268, lists Prekmurski but not this). Its own node because
    the census printed it as an answer.
  "Romanes" (6,273): Romani, variety not stated (`romani.romani`, as hu2022). Statistik Austria
    notes that Romanian speakers often ticked it by mistake (Hauptergebnisse I, Oesterreich, p. 17).
  "Kroatisch", "Serbisch", "Bosnisch": three leaves, as the census asked.
  "Russisch,Ukrainisch,Weißrussisch" (8,446): one category in the census; drawn on Russian, as
    ch2000 draws BFS's same merge. Most of it is Russian; the map cannot unpick it.
  "Indisch" (Indian, 3,582): a country, not a language (Punjabi, Malayalam, Hindi, Tamil...), so
    `other`, as cy2021 and zm2022.
  "Philippinisch": Filipino, as lu2021's "Philippin".
  "Chinesisch": Sinitic, as de2023.
  "Holländisch/Flämisch": Dutch (Flemish is Dutch).
  "sonstige europäische Sprachen", "andere asiatische Sprachen", "Andere Sprachen, unbekannt":
    cross-family remainders, `other`. "sonstige afrikanische Sprachen": `africa_other`.
  "Englisch" (58,582) is drawn as measured. Statistik Austria says some of the 33,000 Austrians
    naming it probably gave a foreign language they know, against the instructions; the census
    does not say how many, and 25,000 foreign citizens named it too.
"""
IE = "indoeuropean"
GE = f"{IE}.germanic"
RO = f"{IE}.romance"
SL = f"{IE}.slavic"

NAMES = {
    "Deutsch": f"{GE}.continental.german",
    # Tabelle 5's columns: the recognised minorities, Croatian and Windisch
    "Burgenland-Kroatisch": f"{SL}.south.burgenland_croatian",
    "Kroatisch": f"{SL}.south.croatian",
    "Romanes": f"{IE}.indoaryan.romani.romani",
    "Slowakisch": f"{SL}.west.slovak",
    "Slowenisch": f"{SL}.south.slovenian",
    "Tschechisch": f"{SL}.west.czech",
    "Ungarisch": "uralic.hungarian",
    "Windisch": f"{SL}.south.windisch",
    # Tabelle 14: languages of former Yugoslavia and Turkey
    "Bosnisch": f"{SL}.south.bosnian",
    "Mazedonisch": f"{SL}.south.macedonian",
    "Serbisch": f"{SL}.south.serbian",
    "Türkisch": "turkic.turkish",
    "Kurdisch": f"{IE}.iranian.kurdish",
    # English, French, Italian
    "Englisch": f"{GE}.english",
    "Französisch": f"{RO}.french",
    "Italienisch": f"{RO}.italian",
    # other European
    "Albanisch": f"{IE}.albanian.albanian",
    "Bulgarisch": f"{SL}.south.bulgarian",
    "Dänisch": f"{GE}.north.danish",
    "Finnisch": "uralic.finnish",
    "Griechisch": f"{IE}.hellenic.greek",
    "Holländisch/Flämisch": f"{GE}.continental.dutch",
    "Norwegisch": f"{GE}.north.norwegian",
    "Polnisch": f"{SL}.west.polish",
    "Portugiesisch": f"{RO}.portuguese",
    "Rumänisch": f"{RO}.romanian",
    "Russisch,Ukrainisch,Weißrussisch": f"{SL}.east.russian",
    "Schwedisch": f"{GE}.north.swedish",
    "Spanisch": f"{RO}.spanish",
    "sonstige europäische Sprachen": "other",
    # African
    "Arabisch": "afroasiatic.arabic",
    "sonstige afrikanische Sprachen": "africa_other",
    # Asian
    "Chinesisch": "sinotibetan.sinitic",
    "Hebräisch": "afroasiatic.hebrew",
    "Indisch": "other",
    "Indonesisch": "austronesian.malayic.indonesian",
    "Japanisch": "japonic.japanese",
    "Koreanisch": "koreanic.korean",
    "Persisch": f"{IE}.iranian.persian",
    "Philippinisch": "austronesian.philippine.filipino",
    "Thailändisch": "kradai.thai",
    "Vietnamesisch": "austroasiatic.vietnamese",
    "andere asiatische Sprachen": "other",
    "Andere Sprachen, unbekannt": "other",
}
NOT_STATED = set()


def resolve(label):
    if label in NOT_STATED:
        return None
    if label not in NAMES:
        raise KeyError(f"at2001: unmapped label {label!r}")
    return NAMES[label]
