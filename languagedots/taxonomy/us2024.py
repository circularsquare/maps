"""United States, ACS 2020-2024 5-year: language spoken at home -> node.

Keyed by the label each row carries in data/normalized/us*.csv: "Speak only English" from C16001,
and every other label is a PUMS language code's own name (ACSPUMS2020_2024CodeLists.xlsx,
sheet Language), since countries/us.py splits each tract's C16001 group by its PUMA's mix of
those codes (sources/us_acs.py). Every one of the 125 codes is here.

Remainders (spec §3.2), each on the narrowest node containing what the code holds (the code
list names every detailed language inside a code):
  - "Chinese" (2.1M) holds unspecified Chinese plus Hakka, Wu, Gan, Xiang, Min Bei and Min Dong.
    It sits on the Chinese (Sinitic) group itself, so it is drawn as "Chinese, language not
    named", which is what it is: the variety was not named, or was one of six the code merges.
  - "Other Indo-Iranian languages" (Tajik, Balochi, Sindhi, Bhojpuri, Romani...) and "Other
    Indo-European languages" (Catalan, Welsh, Belarusian, Slovenian...): Indo-European.
  - "India N.E.C." (an answer such as "Indian"): Indo-European languages and Dravidian ones
    are both possible, so `other`.
  - "Other Languages of Asia" spans Turkic, Mongolic, Munda, Mon-Khmer, Tibeto-Burman, Tai and
    Ainu: `other`. "Other Philippine", "Other Eastern Malayo-Polynesian" (Palauan is not EMP,
    but it is in the code; the narrowest node holding both is Austronesian): their groups.
  - "Other Afro-Asiatic" (Hausa, Berber, Tigre...): Afroasiatic. "Other Bantu", "Other Mande",
    "Other Niger-Congo": their groups. "Other languages of Africa" (Khoisan, "Nigeria N.E.C."):
    `other`.
  - "Other Native North American languages" (Cree, Cheyenne, the Salish languages, Tlingit,
    Haida, the Alaskan Athabaskan languages, Cherokee...) and "Other Central and South American languages" (K'iche', Mam, Quechua,
    Zapotec...): `americas_other`, "Other indigenous languages of the Americas", a root
    of its own because no family node holds either (tree.d/us.txt).
  - "Other English-based Creole languages" (Gullah, Hawai'i Creole English, Krio, Nigerian
    Pidgin...): English-based creoles. "Other and unspecified languages": `other`.

Labels that name a cluster rather than one language get a leaf of their own, labelled with the
cluster (the census names nothing finer): Chin, Karen, Apache, Dakota, Manding, Gbe, Edoid.
Two codes are named for less than they hold:
  - "Aleut languages" holds Aleut, Inupiaq, the Yupik languages, Inuktitut and Greenlandic,
    i.e. the whole Eskimo-Aleut family; it sits on that family's node, not on an "Aleut" leaf.
  - "Uto-Aztecan languages" is the family's own name (Hopi, Tohono O'odham, Comanche, Shoshoni,
    two Nahuatls...): the family node.
"Filipino" stays apart from Tagalog: in the ACS it is often the answer of someone naming their
nationality's language, not a variety, so it may hide Cebuano or Ilocano speakers.
"""
IE = "indoeuropean"
IA = "indoeuropean.indoaryan"
ROM = "indoeuropean.romance"
GER = "indoeuropean.germanic.continental"
SL = "indoeuropean.slavic"
AN = "austronesian"
NC = "nigercongo"

NAMES = {
    "Speak only English": "indoeuropean.germanic.english",
    "Jamaican Creole English": "creole.english_based.jamaican",
    "Other English-based Creole languages": "creole.english_based",
    "Haitian Creole": "creole.french_based.haitian",
    "Kabuverdianu": "creole.portuguese_based.kabuverdianu",
    "German": f"{GER}.german",
    "Swiss German": f"{GER}.swiss_german",
    "Pennsylvania German": f"{GER}.pennsylvania_german",
    "Yiddish": f"{GER}.yiddish",
    "Dutch": f"{GER}.dutch",
    "Afrikaans": f"{GER}.afrikaans",
    "Swedish": "indoeuropean.germanic.north.swedish",
    "Danish": "indoeuropean.germanic.north.danish",
    "Norwegian": "indoeuropean.germanic.north.norwegian",
    "Italian": f"{ROM}.italian",
    "French": f"{ROM}.french",
    "Cajun French": f"{ROM}.cajun_french",
    "Spanish": f"{ROM}.spanish",
    "Portuguese": f"{ROM}.portuguese",
    "Romanian": f"{ROM}.romanian",
    "Irish": "indoeuropean.celtic.irish",
    "Greek": "indoeuropean.hellenic.greek",
    "Albanian": "indoeuropean.albanian.albanian",
    "Russian": f"{SL}.east.russian",
    "Ukrainian": f"{SL}.east.ukrainian",
    "Czech": f"{SL}.west.czech",
    "Slovak": f"{SL}.west.slovak",
    "Polish": f"{SL}.west.polish",
    "Bulgarian": f"{SL}.south.bulgarian",
    "Macedonian": f"{SL}.south.macedonian",
    "Serbocroatian": f"{SL}.south.serbocroatian",
    "Bosnian": f"{SL}.south.bosnian",
    "Croatian": f"{SL}.south.croatian",
    "Serbian": f"{SL}.south.serbian",
    "Lithuanian": "indoeuropean.baltic.lithuanian",
    "Latvian": "indoeuropean.baltic.latvian",
    "Armenian": "indoeuropean.armenian.armenian",
    "Farsi": "indoeuropean.iranian.persian",
    "Dari": "indoeuropean.iranian.dari",
    "Kurdish": "indoeuropean.iranian.kurdish",
    "Pashto": "indoeuropean.iranian.pashto",
    "India N.E.C.": "other",
    "Hindi": f"{IA}.central.hindi",
    "Urdu": f"{IA}.central.urdu",
    "Bengali": f"{IA}.eastern.bengali",
    "Punjabi": f"{IA}.northwestern.punjabi",
    "Konkani": f"{IA}.southern.konkani",
    "Marathi": f"{IA}.southern.marathi",
    "Gujarati": f"{IA}.gujarati.gujarati",
    "Nepali": f"{IA}.pahari.eastern.nepali",
    "Sinhala": f"{IA}.sinhala",
    "Other Indo-Iranian languages": IE,
    "Other Indo-European languages": IE,
    "Finnish": "uralic.finnish",
    "Hungarian": "uralic.hungarian",
    "Turkish": "turkic.turkish",
    "Mongolian": "mongolic.mongolian",
    "Telugu": "dravidian.southcentral.telugu",
    "Kannada": "dravidian.southern.kannada",
    "Malayalam": "dravidian.southern.malayalam",
    "Tamil": "dravidian.southern.tamil",
    "Khmer": "austroasiatic.khmer",
    "Vietnamese": "austroasiatic.vietnamese",
    "Chinese": "sinotibetan.sinitic",
    "Mandarin": "sinotibetan.sinitic.mandarin",
    "Min Nan Chinese": "sinotibetan.sinitic.min_nan",
    "Cantonese": "sinotibetan.sinitic.cantonese",
    "Tibetan": "sinotibetan.tibetic.tibetan",
    "Burmese": "sinotibetan.burmish.burmese",
    "Chin languages": "sinotibetan.kukichin.chin",
    "Karen languages": "sinotibetan.karen",
    "Thai": "kradai.thai",
    "Lao": "kradai.lao",
    "Iu Mien": "hmongmien.iu_mien",
    "Hmong": "hmongmien.hmong",
    "Japanese": "japonic.japanese",
    "Korean": "koreanic.korean",
    "Malay": f"{AN}.malayic.malay",
    "Indonesian": f"{AN}.malayic.indonesian",
    "Other Languages of Asia": "other",
    "Filipino": f"{AN}.philippine.filipino",
    "Tagalog": f"{AN}.philippine.tagalog",
    "Cebuano": f"{AN}.philippine.cebuano",
    "Ilocano": f"{AN}.philippine.ilocano",
    "Other Philippine languages": f"{AN}.philippine",
    "Chamorro": f"{AN}.chamorro",
    "Marshallese": f"{AN}.oceanic.marshallese",
    "Chuukese": f"{AN}.oceanic.chuukese",
    "Samoan": f"{AN}.oceanic.samoan",
    "Tongan": f"{AN}.oceanic.tongan",
    "Hawaiian": f"{AN}.oceanic.hawaiian",
    "Other Eastern Malayo-Polynesian languages": AN,
    "Arabic": "afroasiatic.arabic",
    "Hebrew": "afroasiatic.hebrew",
    "Assyrian Neo-Aramaic": "afroasiatic.assyrian",
    "Chaldean Neo-Aramaic": "afroasiatic.chaldean",
    "Amharic": "afroasiatic.ethiosemitic.amharic",
    "Tigrinya": "afroasiatic.ethiosemitic.tigrinya",
    "Oromo": "afroasiatic.cushitic.lowland.oromo",
    "Somali": "afroasiatic.cushitic.lowland.somali",
    "Other Afro-Asiatic languages": "afroasiatic",
    "Nilo-Saharan languages": "nilosaharan",
    "Swahili": f"{NC}.bantu.swahili",
    "Ganda": f"{NC}.bantu.ganda",
    "Shona": f"{NC}.bantu.shona",
    "Other Bantu languages": f"{NC}.bantu",
    "Manding languages": f"{NC}.mande.manding",
    "Other Mande languages": f"{NC}.mande",
    "Fulah": f"{NC}.atlantic.fulah",
    "Wolof": f"{NC}.atlantic.wolof",
    "Akan (incl. Twi)": f"{NC}.kwa.akan",
    "Ga": f"{NC}.kwa.ga",
    "Gbe languages": f"{NC}.kwa.gbe",
    "Yoruba": f"{NC}.voltaniger.yoruba",
    "Edoid languages": f"{NC}.voltaniger.edoid",
    "Igbo": f"{NC}.voltaniger.igbo",
    "Other Niger-Congo languages": NC,
    "Other languages of Africa": "other",
    "Aleut languages": "eskimoaleut",
    "Ojibwa": "algic.ojibwe",
    "Apache languages": "nadene.apache",
    "Navajo": "nadene.navajo",
    "Dakota languages": "siouan.dakota",
    "Uto-Aztecan languages": "utoaztecan",
    "Other Native North American languages": "americas_other",
    "Other Central and South American languages": "americas_other",
    "Other and unspecified languages": "other",
}


def resolve(label):
    return NAMES[label]
