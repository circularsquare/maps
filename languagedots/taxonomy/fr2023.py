"""France, 2023: language labels written by sources/fr_build.py -> node.

France's census asks no language, so every label here is the build's own, not a census
category (sources/fr.md says how each is derived):
  * "French": everyone the proxies do not place elsewhere.
  * regional languages from regional surveys (sources/fr_regional.py): Breton, Gallo, Basque,
    Alsatian, Lorraine Franconian (Platt), Occitan, Corsican, Catalan; overseas, the creoles of
    the Antilles, Guyane and Reunion, and Shimaore and Kibushi in Mayotte.
  * immigrant languages: INSEE's immigrants by country of birth, each country on its main
    language (sources/fr_build.py COUNTRY_LANG). Algeria and Morocco are split between Arabic and
    Berber. "Chinese" (born in China) is the Sinitic group: the source names a country, not one
    of its languages.
"""
IE = "indoeuropean"
GE = f"{IE}.germanic"
RO = f"{IE}.romance"
SL = f"{IE}.slavic"
IR = f"{IE}.iranian"
IA = f"{IE}.indoaryan"
NC = "nigercongo"

NAMES = {
    "French": f"{RO}.french",
    # regional languages, metropolitan
    "Breton": f"{IE}.celtic.breton",
    "Gallo": f"{RO}.gallo",
    "Basque": "isolate.basque",
    "Alsatian": f"{GE}.continental.alsatian",
    "Lorraine Franconian": f"{GE}.continental.lorraine_franconian",
    "Occitan": f"{RO}.occitan",
    "Corsican": f"{RO}.corsican",
    "Catalan": f"{RO}.catalan",
    # overseas
    "Antillean Creole": "creole.french_based.antillean",
    "Guianese Creole": "creole.french_based.guianese",
    "Reunion Creole": "creole.french_based.reunionese",
    "Shimaore": f"{NC}.bantu.shimaore",
    "Kibushi": "austronesian.kibushi",
    "Comorian": f"{NC}.bantu.comorian",
    # immigrant languages, Europe
    "Portuguese": f"{RO}.portuguese",
    "Italian": f"{RO}.italian",
    "Spanish": f"{RO}.spanish",
    "Romanian": f"{RO}.romanian",
    "German": f"{GE}.continental.german",
    "Dutch": f"{GE}.continental.dutch",
    "Luxembourgish": f"{GE}.continental.luxembourgish",
    "English": f"{GE}.english",
    "Danish": f"{GE}.north.danish",
    "Swedish": f"{GE}.north.swedish",
    "Norwegian": f"{GE}.north.norwegian",
    "Icelandic": f"{GE}.north.icelandic",
    "Irish": f"{IE}.celtic.irish",
    "Polish": f"{SL}.west.polish",
    "Czech": f"{SL}.west.czech",
    "Slovak": f"{SL}.west.slovak",
    "Russian": f"{SL}.east.russian",
    "Ukrainian": f"{SL}.east.ukrainian",
    "Bulgarian": f"{SL}.south.bulgarian",
    "Croatian": f"{SL}.south.croatian",
    "Bosnian": f"{SL}.south.bosnian",
    "Serbian": f"{SL}.south.serbian",
    "Slovenian": f"{SL}.south.slovenian",
    "Macedonian": f"{SL}.south.macedonian",
    "Greek": f"{IE}.hellenic.greek",
    "Albanian": f"{IE}.albanian.albanian",
    "Armenian": f"{IE}.armenian.armenian",
    "Latvian": f"{IE}.baltic.latvian",
    "Lithuanian": f"{IE}.baltic.lithuanian",
    "Hungarian": "uralic.hungarian",
    "Finnish": "uralic.finnish",
    "Estonian": "uralic.estonian",
    "Maltese": "afroasiatic.maltese",
    "Georgian": "kartvelian.georgian",
    "Turkish": "turkic.turkish",
    "Azerbaijani": "turkic.azerbaijani",
    "Kazakh": "turkic.kazakh",
    "Kyrgyz": "turkic.kyrgyz",
    "Turkmen": "turkic.turkmen",
    "Uzbek": "turkic.uzbek",
    # North Africa and the Middle East
    "Arabic": "afroasiatic.arabic",
    "Moroccan Arabic": "afroasiatic.darija",
    "Hassaniya": "afroasiatic.hassaniya",
    "Kabyle": "afroasiatic.berber.kabyle",
    "Tachelhit": "afroasiatic.berber.tachelhit",
    "Tamazight": "afroasiatic.berber.tamazight",
    "Tarifit": "afroasiatic.berber.tarifit",
    "Hebrew": "afroasiatic.hebrew",
    "Persian": f"{IR}.persian",
    "Dari": f"{IR}.dari",
    "Tajik": f"{IR}.tajik",
    # sub-Saharan Africa
    "Wolof": f"{NC}.atlantic.wolof",
    "Fula": f"{NC}.atlantic.fulah",
    "Bambara": f"{NC}.mande.bambara",
    "Mandinka": f"{NC}.mande.mandinka",
    "Lingala": f"{NC}.bantu.lingala",
    "Kirundi": f"{NC}.bantu.kirundi",
    "Kinyarwanda": f"{NC}.bantu.kinyarwanda",
    "Swahili": f"{NC}.bantu.swahili",
    "Luganda": f"{NC}.bantu.ganda",
    "Bemba": f"{NC}.bantu.bemba.bemba",
    "Shona": f"{NC}.bantu.shona",
    "Nyanja": f"{NC}.bantu.nyanja_sena.nyanja",
    "Emakhuwa": f"{NC}.bantu.makhuwa.emakhuwa",
    "Setswana": f"{NC}.bantu.sotho_tswana.setswana",
    "Sesotho": f"{NC}.bantu.sotho_tswana.sesotho",
    "Oshiwambo": f"{NC}.bantu.oshiwambo",
    "Zulu": f"{NC}.bantu.nguni.zulu",
    "Swati": f"{NC}.bantu.nguni.siswati",
    "Fang": f"{NC}.bantu.fang",
    "Sango": f"{NC}.sango",
    "Twi": f"{NC}.kwa.twi",
    "Ewe": f"{NC}.kwa.ewe",
    "Fon": f"{NC}.kwa.fon",
    "Moore": f"{NC}.gur.moore",
    "Malagasy": "austronesian.malagasy",
    "Hausa": "afroasiatic.chadic.hausa",
    "Somali": "afroasiatic.cushitic.lowland.somali",
    "Amharic": "afroasiatic.ethiosemitic.amharic",
    "Tigrinya": "afroasiatic.ethiosemitic.tigrinya",
    "Dinka": "nilosaharan.nilotic.dinka",
    "Morisyen": "creole.french_based.morisyen",
    "Seychellois Creole": "creole.french_based.seselwa",
    "Kabuverdianu": "creole.portuguese_based.kabuverdianu",
    "Guinea-Bissau Kriol": "creole.portuguese_based.guinea_bissau_kriol",
    "Krio": "creole.english_based.krio",
    "Liberian English": "creole.english_based.liberian",
    # the Americas
    "Haitian Creole": "creole.french_based.haitian",
    "Jamaican Creole": "creole.english_based.jamaican",
    "Papiamento": "creole.portuguese_based.papiamento",
    "Ndyuka": "creole.english_based.ndyuka",
    # Asia and the Pacific
    "Chinese": "sinotibetan.sinitic",
    "Mandarin": "sinotibetan.sinitic.mandarin",
    "Japanese": "japonic.japanese",
    "Korean": "koreanic.korean",
    "Mongolian": "mongolic.mongolian",
    "Vietnamese": "austroasiatic.vietnamese",
    "Khmer": "austroasiatic.khmer",
    "Lao": "kradai.lao",
    "Thai": "kradai.thai",
    "Burmese": "sinotibetan.burmish.burmese",
    "Dzongkha": "sinotibetan.tibetic.dzongkha",
    "Malay": "austronesian.malayic.malay",
    "Indonesian": "austronesian.malayic.indonesian",
    "Tagalog": "austronesian.philippine.tagalog",
    "Tetun": "austronesian.timoric.tetun_prasa",
    "Hindi": f"{IA}.central.hindi",
    "Urdu": f"{IA}.central.urdu",
    "Punjabi": f"{IA}.northwestern.punjabi",
    "Bengali": f"{IA}.eastern.bengali",
    "Nepali": f"{IA}.pahari.eastern.nepali",
    "Dhivehi": f"{IA}.dhivehi",
    "Tamil": "dravidian.southern.tamil",
    "Fijian": "austronesian.oceanic.fijian",
    "Samoan": "austronesian.oceanic.samoan",
    "Tongan": "austronesian.oceanic.tongan",
    "Gilbertese": "austronesian.oceanic.gilbertese",
    "Marshallese": "austronesian.oceanic.marshallese",
    "Nauruan": "austronesian.oceanic.nauruan",
    "Tuvaluan": "austronesian.oceanic.tuvaluan",
    "Palauan": "austronesian.palauan",
    "Tok Pisin": "creole.english_based.tok_pisin",
    "Solomon Islands Pijin": "creole.english_based.pijin",
    "Bislama": "creole.english_based.bislama",
}
EXTRA_NODES = []


def resolve(label):
    if label[:1].islower():   # a node id, from sources/origin_mix.py
        return label
    if label not in NAMES:
        raise KeyError(f"fr2023: unmapped label {label!r}")
    return NAMES[label]
