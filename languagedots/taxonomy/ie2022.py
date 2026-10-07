"""Ireland, Census of Population 2022 (CSO) -> node. Keys are the labels sources/ie_census.py writes
into data/normalized/ie.csv: SAPS T2_5's Polish/French/Spanish, PxStat F5029's 63 further labels
(language other than English or Irish spoken at home), and the two built parts, Irish and English.

CALLS (sources/ie.md says more):
  * Irish is drawn only for those who speak it daily outside the education system (71,968); the
    census's 1.87M "can speak Irish" are mostly school learners and are drawn as English.
  * English is the remainder: the census never asks about English as a home language.
  * F5029 prints Filipino and Tagalog apart, and Bosnian, Croatian and Serbian apart; each keeps
    the node every other country uses.
  * "Chinese, nec" names no variety, so it sits on Chinese (sinitic), drawn as unnamed.
  * "Other Northern European" (1,311) goes on Indo-European: after Swedish, Danish, Finnish,
    Estonian, Latvian and Lithuanian are named, northern Europe's remaining languages are
    Norwegian, Icelandic, Faroese and Britain's Welsh, Scots and Gaelics; Sami would be a few
    people at most. "Other Southern European" (Maltese, Basque, Catalan, Slovene...), "Other
    Eastern European" (Belarusian, Romani, Tatar...) and "Other Asian" cross families and go on
    `other`; "Other African" on `africa_other`, as Germany's does.
  * "Other stated languages (incl. not stated)" (29,744) is a mixed remainder: `other`.
  * Irish cant is Shelta, beside Irish (Anita, 2026-10-05, on the UK's Irish Traveller Cant).
"""
IE = "indoeuropean"
GE = f"{IE}.germanic"
RO = f"{IE}.romance"
SL = f"{IE}.slavic"
BA = f"{IE}.baltic"
IA = f"{IE}.indoaryan"

NAMES = {
    "English (everyone not counted under another language)": f"{GE}.english",
    "Irish (speaks it daily outside the education system)": f"{IE}.celtic.irish",
    "Polish": f"{SL}.west.polish",
    "French": f"{RO}.french",
    "Lithuanian": f"{BA}.lithuanian",
    "German": f"{GE}.continental.german",
    "Russian": f"{SL}.east.russian",
    "Spanish": f"{RO}.spanish",
    "Romanian": f"{RO}.romanian",
    "Chinese, nec": "sinotibetan.sinitic",
    "Latvian": f"{BA}.latvian",
    "Portuguese": f"{RO}.portuguese",
    "Arabic": "afroasiatic.arabic",
    "Italian": f"{RO}.italian",
    "Yoruba": "nigercongo.voltaniger.yoruba",
    "Slovak": f"{SL}.west.slovak",
    "Malayalam": "dravidian.southern.malayalam",
    "Urdu": f"{IA}.central.urdu",
    "Hungarian": "uralic.hungarian",
    "Filipino": "austronesian.philippine.filipino",
    "Tagalog": "austronesian.philippine.tagalog",
    "Czech": f"{SL}.west.czech",
    "Dutch": f"{GE}.continental.dutch",
    "Hindi": f"{IA}.central.hindi",
    "Igbo": "nigercongo.voltaniger.igbo",
    "Bengali": f"{IA}.eastern.bengali",
    "Irish sign language": "signlanguage.isl",
    "Afrikaans": f"{GE}.continental.afrikaans",
    "Ukrainian": f"{SL}.east.ukrainian",
    "Swedish": f"{GE}.north.swedish",
    "Bulgarian": f"{SL}.south.bulgarian",
    "Albanian": f"{IE}.albanian.albanian",
    "Malay": "austronesian.malayic.malay",
    "Thai": "kradai.thai",
    "Turkish": "turkic.turkish",
    "Swahili": "nigercongo.bantu.swahili",
    "Punjabi": f"{IA}.northwestern.punjabi",
    "Tamil": "dravidian.southern.tamil",
    "Estonian": "uralic.estonian",
    "Japanese": "japonic.japanese",
    "Somali": "afroasiatic.cushitic.lowland.somali",
    "Bosnian": f"{SL}.south.bosnian",
    "Croatian": f"{SL}.south.croatian",
    "Shona": "nigercongo.bantu.shona",
    "Vietnamese": "austroasiatic.vietnamese",
    "Pashto": f"{IE}.iranian.pashto",
    "Telugu": "dravidian.southcentral.telugu",
    "Greek": f"{IE}.hellenic.greek",
    "Persian": f"{IE}.iranian.persian",
    "Lingala": "nigercongo.bantu.lingala",
    "Korean": "koreanic.korean",
    "Edo": "nigercongo.voltaniger.edoid.edo",          # new leaf, tree.d/ie.txt
    "Serbian": f"{SL}.south.serbian",
    "Finnish": "uralic.finnish",
    "Danish": f"{GE}.north.danish",
    "Kurdish": f"{IE}.iranian.kurdish",
    "Nepali": f"{IA}.pahari.eastern.nepali",
    "Sign Language (Not Specified)": "signlanguage",
    "Irish cant": f"{IE}.celtic.shelta",
    "Georgian": "kartvelian.georgian",
    "Hebrew": "afroasiatic.hebrew",
    "Nyanja (Chichewa)": "nigercongo.bantu.nyanja_sena.nyanja",
    "Other Northern European": IE,
    "Other Southern European": "other",
    "Other Eastern European": "other",
    "Other Asian": "other",
    "Other African": "africa_other",
    "Other stated languages (incl. not stated)": "other",
}


def resolve(label):
    if label not in NAMES:
        raise KeyError(f"ie2022: unmapped label {label!r}")
    return NAMES[label]
