"""Tajikistan 2010 census, native language by nationality (sources/tj_census.py) -> node.

Keyed by data/normalized/tj.csv's `source_category`: the table's three named columns ("Tajik",
"Russian", "other languages") and "own language: <nationality>" for the column "language of one's
own nationality", which names a different language on every row. Same shape as kg2022.

CALLS.
  own language, Lakai, Kongrat, Durmen, Katagan, Yuz, Ming, Kesamir, Semiz: a node each, beside
    Uzbek (tree.d/tj.txt). The census keeps these Uzbek tribal groups apart from Uzbeks as
    nationalities, and its language dictionary evidently holds a language for each: 60,392 Lakai
    named their own nationality's language and 5,127 "other languages", while the Barlos, a tribe
    with no language of its own in the dictionary, put all 5,267 under "other languages". So
    "own language: Lakai" is a separate answer from Uzbek, and a dialect gets a node (spec 3).
    Glottolog has no entry below Northern Uzbek for any of them; the labels say "Uzbek dialect".
    117,609 people, 1.6%.
  own language, Afghan: `other.afghan`, a named leaf under `other`. The census names no language.
    Tajikistan's "Afghans" are Afghan nationals (Dari or Pashto) and the Parya of the Hisor valley,
    an Indo-Aryan-speaking group who call themselves Afghan; Soviet usage made "Afghan language"
    Pashto. Iranian and Indo-Aryan share no node short of Indo-European, so, as Azerbaijan's
    "Jewish" and Kyrgyzstan's "India and Pakistan", it is a leaf. 2,320 people.
  own language, Arab: Arabic. Tajikistan's Arabs are the old Central Asian Arab community of
    Khatlon, whose own speech is Tajiki Arabic (Glottolog taji1248); the census says only "the
    language of the Arabs", and `afroasiatic.arabic` holds every Arabic. 4,089 people.
  own language, Lyuli (Roma): Romani, as Kyrgyzstan's Roma. Central Asia's Lyuli mostly speak
    Tajik or Uzbek; the 338 who named their own nationality's language named "Gypsy".
  own language, Turk: Turkish, as Kyrgyzstan's (Meskhetian Turks). 316 people.
  own language, Chinese: `sinitic`, the node every country uses for Chinese not split. 786 people.
  own language, Persian (Iranian): Persian. 395 people.
  own language, Jew: `other.jewish`, as Azerbaijan. 13 people. Central Asian (Bukharan) Jews: none
    named their own language.
  own language, American: English. own language, Mordvin: `uralic.mordvin` (not said which).
  own language, peoples of India and Pakistan: `other.india_pakistan`, as Kyrgyzstan. 217 people.
  "other languages": `other`. It holds whatever a nationality named that was neither its own,
    Tajik nor Russian: most likely Uzbek for the Barlos (5,267) and Lakai (5,127), and possibly a
    Pamiri language for some of the 13,581 Tajiks, but the table does not say. 29,904 people.
  "Tajik" and "Russian": another nationality's people naming Tajik or Russian.
"""
IR = "indoeuropean.iranian"
SL = "indoeuropean.slavic"
ND = "nakhdaghestanian"
UR = "uralic"
AD = "abkhazadyghe"

LANGS = {
    "Tajik": f"{IR}.tajik",
    "Russian": f"{SL}.east.russian",
    "other languages": "other",
}
# nationality (sources/tj_census.py NATS' English) -> its own language's node
OWN = {
    "Tajik": f"{IR}.tajik",
    "Uzbek": "turkic.uzbek",
    "Russian": f"{SL}.east.russian",
    "Tatar": "turkic.tatar",
    "Kyrgyz": "turkic.kyrgyz",
    "Ukrainian": f"{SL}.east.ukrainian",
    "German": "indoeuropean.germanic.continental.german",
    "Turkmen": "turkic.turkmen",
    "Korean": "koreanic.korean",
    "Kazakh": "turkic.kazakh",
    "Jew": "other.jewish",
    "Ossetian": f"{IR}.ossetian",
    "Belarusian": f"{SL}.east.belarusian",
    "Bashkir": "turkic.bashkir",
    "Armenian": "indoeuropean.armenian.armenian",
    "Mordvin": f"{UR}.mordvin",
    "Azerbaijani": "turkic.azerbaijani",
    "Chuvash": "turkic.chuvash",
    "Afghan": "other.afghan",
    "Lyuli (Roma)": "indoeuropean.indoaryan.romani.romani",
    "Bulgarian": f"{SL}.south.bulgarian",
    "Georgian": "kartvelian.georgian",
    "Moldovan": "indoeuropean.romance.moldovan",
    "Turk": "turkic.turkish",
    "Pole": f"{SL}.west.polish",
    "Udmurt": f"{UR}.udmurt",
    "Mari": f"{UR}.mari",
    "Greek": "indoeuropean.hellenic.greek",
    "Uyghur": "turkic.uyghur",
    "Lithuanian": "indoeuropean.baltic.lithuanian",
    "Persian (Iranian)": f"{IR}.persian",
    "Dargin": f"{ND}.dargwa",
    "Latvian": "indoeuropean.baltic.latvian",
    "Lezgin": f"{ND}.lezgic.lezgian",
    "Arab": "afroasiatic.arabic",
    "Kabardian": f"{AD}.kabardian",
    "Avar": f"{ND}.avarandic.avar",
    "Karakalpak": "turkic.karakalpak",
    "Buryat": "mongolic.buryat",
    "Estonian": f"{UR}.estonian",
    "Chechen": f"{ND}.nakh.chechen",
    "Kumyk": "turkic.kumyk",
    "Ingush": f"{ND}.nakh.ingush",
    "Circassian": f"{AD}.circassian",
    "Khakas": "turkic.khakas",
    "Finn": f"{UR}.finnish",
    "Komi-Permyak": f"{UR}.komi_permyak",
    "Tabasaran": f"{ND}.lezgic.tabasaran",
    "Chinese": "sinotibetan.sinitic",
    "Kurd": f"{IR}.kurdish",
    "Abaza": f"{AD}.abaza",
    "American": "indoeuropean.germanic.english",
    "Romanian": "indoeuropean.romance.romanian",
    "English": "indoeuropean.germanic.english",
    "Vietnamese": "austroasiatic.vietnamese",
    "Dutch": "indoeuropean.germanic.continental.dutch",
    "Spaniard": "indoeuropean.romance.spanish",
    "Karelian": f"{UR}.karelian",
    "Slovak": f"{SL}.west.slovak",
    "French": "indoeuropean.romance.french",
    "Italian": "indoeuropean.romance.italian",
    "Japanese": "japonic.japanese",
    "Dungan": "sinotibetan.sinitic.dungan",
    "Ming": "turkic.ming",
    "Durmen": "turkic.durmen",
    "Lakai": "turkic.lakai",
    "Kongrat": "turkic.kongrat",
    "Katagan": "turkic.katagan",
    "Yuz": "turkic.yuz",
    "Semiz": "turkic.semiz",
    "Kesamir": "turkic.kesamir",
    "peoples of India and Pakistan": "other.india_pakistan",
}
NAMES = {**LANGS, **{f"own language: {k}": v for k, v in OWN.items()}}


def resolve(label):
    if label not in NAMES:
        raise KeyError(f"tj2010: unmapped label {label!r}")
    return NAMES[label]
