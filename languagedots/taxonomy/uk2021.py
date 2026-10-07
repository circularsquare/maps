"""United Kingdom main language -> node. Keys are sources/uk_census.py's prefixed categories:
"ew:" a TS024 leaf (England and Wales 2021), "sc:" UV212 (Scotland 2022), "ni:" a Northern
Ireland 2021 label (MAIN_LANGUAGE_1000 at Data Zone, MS-B13 for the rest).

Calls worth knowing (sources/uk.md says more):
  * "English (English or Welsh in Wales)" depends on place (spec §3.3). In England it is English.
    In Wales the form's tick box was "English or Welsh", so the answer names one of two
    languages and the census never says which. Anita's ruling on ask 004 (2026-10-04): split
    it. sources/uk_census.py's split_wales() does, per Output Area, by the same census's "can
    you speak Welsh?" (Welsh = speakers, capped at the box; English = the rest), and writes
    the two parts as the "English or Welsh (Wales): split as ..." keys below, tier `derived`.
    No Welsh unit keeps the original label; resolve() refuses one rather than guess.
    Since 2026-10-06 sources/uk_home_use.py keeps only the daily speakers as Welsh, and raises
    Gaelic, Scots and Northern Ireland's Irish to home or daily use ("... beyond main
    language" keys), taking the people from English.
  * Scotland's "Gaelic" is Scottish Gaelic (NRS's own usage); NISRA's and ONS's "Gaelic (Not
    otherwise specified)" stays on Celtic (the tree has no Goidelic level; Irish sits directly
    under Celtic, as us.txt made it).
  * Regional remainders whose members cross families ("Any other South Asian language", "Any
    other African language", "Any other Nigerian language", "Any other West African language",
    "Any other West or Central Asian language", "Any other East Asian language", "Oceanic or
    Australian language", "North or South American language", "Any other Eastern European
    language (non EU)", "Any other European language (EU)", which can hold Basque) go on
    `other`: nothing narrower is sure to contain them. Scotland's "Other language" (272,820,
    everything but English, Scots, Gaelic and sign) has no finer table; since 2026-10-06 it is
    split by country of birth into the 2011 census's languages (SC_OTHER below).
  * "Northern European language (non EU)" (7,876) goes on North Germanic: the non-EU north is
    Norway, Iceland and the Faroes. ONS does not publish its members; any Sami speakers in it
    would be a few dozen.
  * Creoles follow br.txt/us.txt's `creole` root: "English-based Caribbean Creole" names a group,
    not a language, so it sits on English-based creoles; Krio is a leaf under it; "Any other
    Caribbean Creole" (French and other lexifiers too) and NISRA's "Creole (Not otherwise
    specified)" sit on the root.
  * "Tagalog or Filipino" goes on Tagalog: Filipino is standardised Tagalog, one answer.
  * "Bengali (with Sylheti and Chatgaya)" is ONS's own merge and goes on Bengali; the census
    cannot separate Sylheti, the majority of British Bangladeshis' speech, so the map cannot.
  * "Any sign communication system", "Any other sign language" and NISRA's "Sign Language (Not
    otherwise specified)" and Scotland's "Sign Language" sit on the sign-language root; BSL,
    Irish Sign Language and Makaton, which are named, have nodes.
"""
IE = "indoeuropean"
IA = "indoeuropean.indoaryan"
GE = "indoeuropean.germanic"
RO = "indoeuropean.romance"
SL = "indoeuropean.slavic"
BA = "indoeuropean.baltic"
CE = "indoeuropean.celtic"

EW = {
    "English (English or Welsh in Wales)": f"{GE}.english",   # England only; see resolve()
    "English or Welsh (Wales): split as Welsh": f"{CE}.welsh",       # split_wales(), derived
    "English or Welsh (Wales): split as English": f"{GE}.english",   # split_wales(), derived
    "Welsh or Cymraeg (in England only)": f"{CE}.welsh",
    "Other UK language: Gaelic (Irish)": f"{CE}.irish",
    "Other UK language: Gaelic (Scottish)": f"{CE}.scottishgaelic",
    "Other UK language: Manx Gaelic": f"{CE}.manx",
    "Other UK language: Gaelic (Not otherwise specified)": CE,
    "Other UK language: Cornish": f"{CE}.cornish",
    "Other UK language: Scots": f"{GE}.scots",
    "Other UK language: Ulster Scots": f"{GE}.ulsterscots",
    "Other UK language: Romany English": f"{IA}.romani.angloromani",
    "Other UK language: Irish Traveller Cant": f"{IE}.celtic.shelta",   # beside Irish (Anita, 2026-10-05)
    "French": f"{RO}.french",
    "Portuguese": f"{RO}.portuguese",
    "Spanish": f"{RO}.spanish",
    "Other European language (EU): Italian": f"{RO}.italian",
    "Other European language (EU): German": f"{GE}.continental.german",
    "Other European language (EU): Polish": f"{SL}.west.polish",
    "Other European language (EU): Slovak": f"{SL}.west.slovak",
    "Other European language (EU): Czech": f"{SL}.west.czech",
    "Other European language (EU): Romanian": f"{RO}.romanian",
    "Other European language (EU): Lithuanian": f"{BA}.lithuanian",
    "Other European language (EU): Latvian": f"{BA}.latvian",
    "Other European language (EU): Hungarian": "uralic.hungarian",
    "Other European language (EU): Bulgarian": f"{SL}.south.bulgarian",
    "Other European language (EU): Greek": f"{IE}.hellenic.greek",
    "Other European language (EU): Dutch": f"{GE}.continental.dutch",
    "Other European language (EU): Swedish": f"{GE}.north.swedish",
    "Other European language (EU): Danish": f"{GE}.north.danish",
    "Other European language (EU): Finnish": "uralic.finnish",
    "Other European language (EU): Estonian": "uralic.estonian",
    "Other European language (EU): Slovenian": f"{SL}.south.slovenian",
    "Other European language (EU): Maltese": "afroasiatic.maltese",
    "Other European language (EU): Any other European language (EU)": "other",
    "Other European language (non EU): Albanian": f"{IE}.albanian.albanian",
    "Other European language (non EU): Ukrainian": f"{SL}.east.ukrainian",
    "Other European language (non EU): Any other Eastern European language (non EU)": "other",
    "Other European language (non EU): Northern European language (non EU)": f"{GE}.north",
    "Other European language (EU and non-EU): Bosnian, Croatian, Serbian, and Montenegrin":
        f"{SL}.south.serbocroatian",
    "Other European language (non-national): Any Romani language": f"{IA}.romani",
    "Other European language (non-national): Yiddish": f"{GE}.continental.yiddish",
    "Russian": f"{SL}.east.russian",
    "Turkish": "turkic.turkish",
    "Arabic": "afroasiatic.arabic",
    "West or Central Asian language: Hebrew": "afroasiatic.hebrew",
    "West or Central Asian language: Kurdish": "indoeuropean.iranian.kurdish",
    "West or Central Asian language: Persian or Farsi": "indoeuropean.iranian.persian",
    "West or Central Asian language: Pashto": "indoeuropean.iranian.pashto",
    "West or Central Asian language: Any other West or Central Asian language": "other",
    "South Asian language: Urdu": f"{IA}.central.urdu",
    "South Asian language: Hindi": f"{IA}.central.hindi",
    "South Asian language: Panjabi": f"{IA}.northwestern.punjabi",
    "South Asian language: Pakistani Pahari (with Mirpuri and Potwari)":
        f"{IA}.northwestern.pahari_pothwari",
    "South Asian language: Bengali (with Sylheti and Chatgaya)": f"{IA}.eastern.bengali",
    "South Asian language: Gujarati": f"{IA}.gujarati.gujarati",
    "South Asian language: Marathi": f"{IA}.southern.marathi",
    "South Asian language: Telugu": "dravidian.southcentral.telugu",
    "South Asian language: Tamil": "dravidian.southern.tamil",
    "South Asian language: Malayalam": "dravidian.southern.malayalam",
    "South Asian language: Sinhala": f"{IA}.sinhala",
    "South Asian language: Nepalese": f"{IA}.pahari.eastern.nepali",
    "South Asian language: Any other South Asian language": "other",
    "East Asian language: Mandarin Chinese": "sinotibetan.sinitic.mandarin",
    "East Asian language: Cantonese Chinese": "sinotibetan.sinitic.cantonese",
    "East Asian language: All other Chinese": "sinotibetan.sinitic",
    "East Asian language: Japanese": "japonic.japanese",
    "East Asian language: Korean": "koreanic.korean",
    "East Asian language: Vietnamese": "austroasiatic.vietnamese",
    "East Asian language: Thai": "kradai.thai",
    "East Asian language: Malay": "austronesian.malayic.malay",
    "East Asian language: Tagalog or Filipino": "austronesian.philippine.tagalog",
    "East Asian language: Any other East Asian language": "other",
    "Oceanic or Australian language": "other",
    "North or South American language": "other",
    "Caribbean Creole: English-based Caribbean Creole": "creole.english_based",
    "Caribbean Creole: Any other Caribbean Creole": "creole",
    "African language: Amharic": "afroasiatic.ethiosemitic.amharic",
    "African language: Tigrinya": "afroasiatic.ethiosemitic.tigrinya",
    "African language: Somali": "afroasiatic.cushitic.lowland.somali",
    "African language: Krio": "creole.english_based.krio",
    "African language: Akan": "nigercongo.kwa.akan",
    "African language: Yoruba": "nigercongo.voltaniger.yoruba",
    "African language: Igbo": "nigercongo.voltaniger.igbo",
    "African language: Swahili or Kiswahili": "nigercongo.bantu.swahili",
    "African language: Luganda": "nigercongo.bantu.ganda",
    "African language: Lingala": "nigercongo.bantu.lingala",
    "African language: Shona": "nigercongo.bantu.shona",
    "African language: Afrikaans": f"{GE}.continental.afrikaans",
    "African language: Any other Nigerian language": "other",
    "African language: Any other West African language": "other",
    "African language: Any other African language": "other",
    "Sign language: British Sign Language": "signlanguage.bsl",
    "Sign language: Any other sign language": "signlanguage",
    "Sign language: Any sign communication system": "signlanguage",
    "Other language": "other",
}

SC = {
    "English": f"{GE}.english",
    "Scots": f"{GE}.scots",
    "Gaelic": f"{CE}.scottishgaelic",
    "Gaelic: used at home, beyond main language": f"{CE}.scottishgaelic",   # uk_home_use.py, derived
    "Scots: used at home, beyond main language": f"{GE}.scots",             # uk_home_use.py, derived
    "Sign Language": "signlanguage",
    "Other language": "other",
}

NI = {
    "English": f"{GE}.english",
    "Polish": f"{SL}.west.polish",
    "Lithuanian": f"{BA}.lithuanian",
    "Irish": f"{CE}.irish",
    "Irish: speaks daily, beyond main language": f"{CE}.irish",   # uk_home_use.py, derived
    "Romanian": f"{RO}.romanian",
    "Portuguese": f"{RO}.portuguese",
    "Arabic": "afroasiatic.arabic",
    "Bulgarian": f"{SL}.south.bulgarian",
    "Chinese (not otherwise specified)": "sinotibetan.sinitic",
    "Slovak": f"{SL}.west.slovak",
    "Hungarian": "uralic.hungarian",
    "Spanish": f"{RO}.spanish",
    "Latvian": f"{BA}.latvian",
    "Russian": f"{SL}.east.russian",
    "Tetun": "austronesian.tetun",
    "Malayalam": "dravidian.southern.malayalam",
    "Tagalog/Filipino": "austronesian.philippine.tagalog",
    "Cantonese": "sinotibetan.sinitic.cantonese",
    # MS-B13, Northern Ireland only: shared out over each Data Zone's "Other languages"
    "Italian": f"{RO}.italian",
    "French": f"{RO}.french",
    "German": f"{GE}.continental.german",
    "Mandarin Chinese": "sinotibetan.sinitic.mandarin",
    "Czech": f"{SL}.west.czech",
    "British Sign Language": "signlanguage.bsl",
    "Urdu": f"{IA}.central.urdu",
    "Hindi": f"{IA}.central.hindi",
    "Somali": "afroasiatic.cushitic.lowland.somali",
    "Bengali": f"{IA}.eastern.bengali",
    "Telugu": "dravidian.southcentral.telugu",
    "Turkish": "turkic.turkish",
    "Greek": f"{IE}.hellenic.greek",
    "Ulster-Scots": f"{GE}.ulsterscots",
    "Dutch": f"{GE}.continental.dutch",
    "Tamil": "dravidian.southern.tamil",
    "Persian/Farsi": "indoeuropean.iranian.persian",
    "Thai": "kradai.thai",
    "Panjabi (Not otherwise specified)": f"{IA}.northwestern.punjabi",
    "Kurdish (Not otherwise specified)": "indoeuropean.iranian.kurdish",
    "Nepalese": f"{IA}.pahari.eastern.nepali",
    "Indonesian Malay": "austronesian.malayic.indonesian",
    "Marathi": f"{IA}.southern.marathi",
    "Swedish": f"{GE}.north.swedish",
    "Makaton": "signlanguage.makaton",
    "Ukrainian": f"{SL}.east.ukrainian",
    "Shona": "nigercongo.bantu.shona",
    "Bosnian, Croatian, Serbian and Montenegrian": f"{SL}.south.serbocroatian",
    "Tigrinya": "afroasiatic.ethiosemitic.tigrinya",
    "Japanese": "japonic.japanese",
    "Pashto": "indoeuropean.iranian.pashto",
    "Kannada": "dravidian.southern.kannada",
    "Irish Sign Language": "signlanguage.isl",
    "Gaelic (Not otherwise specified)": CE,
    "Albanian (Not otherwise specified)": f"{IE}.albanian.albanian",
    "Afrikaans": f"{GE}.continental.afrikaans",
    "Vietnamese": "austroasiatic.vietnamese",
    "Welsh/Cymraeg": f"{CE}.welsh",
    "Korean": "koreanic.korean",
    "Slovenian": f"{SL}.south.slovenian",
    "Gujarati": f"{IA}.gujarati.gujarati",
    "Danish": f"{GE}.north.danish",
    "Igbo": "nigercongo.voltaniger.igbo",
    "Yoruba": "nigercongo.voltaniger.yoruba",
    "Catalan": f"{RO}.catalan",
    "Norwegian": f"{GE}.north.norwegian",
    "Finnish": "uralic.finnish",
    "Malay": "austronesian.malayic.malay",
    "Swahili/Kiswahili": "nigercongo.bantu.swahili",
    "Rhaeto-Romance": f"{RO}.rhaetoromance",
    "Amharic": "afroasiatic.ethiosemitic.amharic",
    "Estonian": "uralic.estonian",
    "Sinhala": f"{IA}.sinhala",
    "Fijian": "austronesian.oceanic.fijian",
    "Hakka": "sinotibetan.sinitic.hakka",
    "Ndebele (Not otherwise specified)": "nigercongo.bantu.nguni",
    "Hebrew": "afroasiatic.hebrew",
    "Akan": "nigercongo.kwa.akan",
    "Sign Language (Not otherwise specified)": "signlanguage",
    "Burmese": "sinotibetan.burmish.burmese",
    "Ndebele (Zimbabwe)": "nigercongo.bantu.nguni.ndebele_zw",
    "Bisayan (Not otherwise specified)": "austronesian.philippine.bisayan",
    "Konkani": f"{IA}.southern.konkani",
    "Setswana": "nigercongo.bantu.sotho_tswana.setswana",   # za.txt's Sotho-Tswana level
    "Icelandic": f"{GE}.north.icelandic",
    "Swiss German": f"{GE}.continental.swiss_german",
    "Creole (Not otherwise specified)": "creole",
    "Ndebele (South Africa)": "nigercongo.bantu.nguni.ndebele_za",
    "Hausa": "afroasiatic.chadic.hausa",
    "Sindhi": f"{IA}.northwestern.sindhi",
    "Philippine (Not otherwise specified)": "austronesian.philippine",
    "Xhosa": "nigercongo.bantu.nguni.xhosa",
    "Fula": "nigercongo.atlantic.fulah",
    "Zulu": "nigercongo.bantu.nguni.zulu",
    "Oriya": f"{IA}.eastern.odia",
    "Armenian": f"{IE}.armenian.armenian",
    "Scots": f"{GE}.scots",
    "Siswati": "nigercongo.bantu.nguni.siswati",
    "Georgian/Kartuli": "kartvelian.georgian",
    "Cambodian/Khmer": "austroasiatic.khmer",
    "Any Other Language": "other",
}

# Scotland's 2022 "Other language" is split (sources/uk_scot_other.py, 2026-10-06) into the
# languages of the 2011 census's AT_002_2011 "Language used at home other than English
# (detailed)", by country of birth, fitted to a national estimate built from that table. Its
# labels, stripped of trailing spaces; drawn as "sc:Other language: <label>", tier `derived`.
# Labels that 2022 counted outside "Other language" (its own English, Scots, Scottish Gaelic and
# Sign Language boxes) are in SC_OTHER_OUTSIDE and are not drawn from this table.
SC_OTHER_OUTSIDE = {"English only", "Scots", "Gaelic (Scottish)",
                    "Gaelic (Not otherwise specified)",   # 2011 had one "Gaelic" box; read as Scottish
                    "British Sign Language", "Makaton", "Deaf-Blind Manual Alphabet",
                    "Other Sign Language", "Sign Language (Not otherwise specified)"}
SC_OTHER = {
    **{k: v for k, v in NI.items() if k in {
        "Afrikaans", "Akan", "Amharic", "Armenian", "Bengali", "Bisayan (Not otherwise specified)",
        "Bulgarian", "Burmese", "Cantonese", "Catalan", "Creole (Not otherwise specified)", "Czech",
        "Danish", "Dutch", "Estonian", "Fijian", "Finnish", "French", "Georgian/Kartuli", "German", "Greek", "Gujarati", "Hakka", "Hausa", "Hebrew", "Hindi",
        "Hungarian", "Icelandic", "Igbo", "Indonesian Malay", "Italian", "Japanese", "Kannada",
        "Konkani", "Korean", "Kurdish (Not otherwise specified)", "Latvian", "Lingala",
        "Lithuanian", "Malay", "Malayalam", "Maltese", "Mandarin Chinese", "Marathi",
        "Ndebele (Not otherwise specified)", "Nepalese", "Norwegian", "Oriya", "Pashto",
        "Persian/Farsi", "Philippine (Not otherwise specified)", "Polish", "Portuguese",
        "Romanian", "Russian", "Setswana", "Shona", "Sindhi", "Sinhala", "Siswati", "Slovak",
        "Slovenian", "Somali", "Spanish", "Swahili/Kiswahili", "Swedish", "Swiss German",
        "Tagalog/Filipino", "Tamil", "Telugu", "Thai", "Tigrinya", "Turkish", "Ukrainian",
        "Vietnamese", "Welsh/Cymraeg", "Xhosa", "Yiddish", "Yoruba", "Zulu", "Arabic"}},
    "Gaelic (Irish)": f"{CE}.irish",
    "Lingala": "nigercongo.bantu.lingala",
    "Maltese": "afroasiatic.maltese",
    "Yiddish": f"{GE}.continental.yiddish",
    "Albanian (Gheg/Kosovan)": f"{IE}.albanian.albanian",   # the tree has no Gheg node
    "Albanian (Not otherwise specified)": f"{IE}.albanian.albanian",
    "Assamese": f"{IA}.eastern.assamese",
    "Azeri": "turkic.azerbaijani",
    "Basque/Euskara": "isolate.basque",
    "Bemba": "nigercongo.bantu.bemba.bemba",
    "Berber (Not otherwise specified)": "afroasiatic.berber",
    "Bini (Not otherwise specified)": "nigercongo.voltaniger.edoid.edo",   # Bini is Edo
    "Edo/Bini": "nigercongo.voltaniger.edoid.edo",
    "Bosnian": f"{SL}.south.bosnian",
    "Croatian": f"{SL}.south.croatian",
    "Serbian": f"{SL}.south.serbian",
    "Serbo-Croat (Not otherwise specified)": f"{SL}.south.serbocroatian",
    "Macedonian": f"{SL}.south.macedonian",
    "Cebuano": "austronesian.philippine.cebuano",
    "Hiligaynon": "austronesian.philippine.hiligaynon",
    "Ilocano": "austronesian.philippine.ilocano",
    "Chechen": "nakhdaghestanian.nakh.chechen",
    "Chichewa/Nyanja": "nigercongo.bantu.nyanja_sena.chewa",   # one language; Malawi's name
    "Chinese (Not otherwise specified)": "sinotibetan.sinitic",
    "Min Nan Chinese": "sinotibetan.sinitic.min_nan",
    "Ebira": "nigercongo.voltaniger.ebira",
    "Efik-Ibibio": "nigercongo.ibibio_efik",
    "Esan": "nigercongo.voltaniger.edoid.esan",
    "Urhobo": "nigercongo.voltaniger.edoid.urhobo",
    "Yekhee": "nigercongo.voltaniger.edoid.yekhee",
    "Ewe": "nigercongo.kwa.ewe",
    "Ga": "nigercongo.kwa.ga",
    "Faroese": f"{GE}.north.faroese",
    "French Creole (Not otherwise specified)": "creole.french_based",
    "Morisyen": "creole.french_based.morisyen",
    "Frisian": f"{GE}.frisian",
    "Fulfulde-Pulaar": "nigercongo.atlantic.fulah",
    "Wolof": "nigercongo.atlantic.wolof",
    "Gikuyu": "nigercongo.bantu.gikuyu",
    "Guere": "nigercongo.kru.guere",
    "Herero": "nigercongo.bantu.herero",
    "Hindko": f"{IA}.northwestern.hindko",
    "Iban": "austronesian.malayic.iban",
    "Idoma": "nigercongo.voltaniger.idoma",
    "Igala": "nigercongo.voltaniger.igala",
    "Ikwere": "nigercongo.voltaniger.ikwere",
    "Ijo (Not otherwise specified)": "nigercongo.ijoid",
    "Kalabari Ijo": "nigercongo.ijoid.kalabari",
    "Iranian": f"{IE}.iranian",   # as printed; Persian, Kurdish and Pashto are named beside it
    "Jamaican": "creole.english_based.jamaican",
    "Kachchi": f"{IA}.northwestern.kachchi",   # new node, tree.d/uk.txt
    "Kashmiri": f"{IA}.dardic.kashmiri",
    "Kazakh": "turkic.kazakh",
    "Uzbek": "turkic.uzbek",
    "Kinyarwanda": "nigercongo.bantu.kinyarwanda",
    "Kirundi": "nigercongo.bantu.kirundi",
    "Kodava": "dravidian.southern.kodava",
    "Tulu": "dravidian.southern.tulu",
    "Krio": "creole.english_based.krio",
    "Nigerian Pidgin": "creole.english_based.nigerian_pidgin",
    "Pidgin (Not otherwise specified)": "creole.english_based",
    "Kurdish (Sorani)": f"{IE}.iranian.kurdish",   # the tree does not split Kurdish
    "Luganda": "nigercongo.bantu.ganda",
    "Lusoga": "nigercongo.bantu.soga",
    "Runyakitara": "nigercongo.bantu.runyakitara",   # new node, tree.d/uk.txt
    "Luo/Dholuo": "nilosaharan.nilotic.luo",
    "Luxembourgish": f"{GE}.continental.luxembourgish",
    "Maghrebi Arabic": "afroasiatic.arabic",   # spans Darija, Algerian, Tunisian, Libyan
    "Sudanese Arabic": "afroasiatic.sudanese_arabic",
    "Maldivian": f"{IA}.dhivehi",
    "Manding/Mandenkan": "nigercongo.mande.manding",
    "Maori": "austronesian.oceanic.maori",
    "Mirpuri": f"{IA}.northwestern.pahari_pothwari",   # as ONS's "Pakistani Pahari (with Mirpuri and Potwari)"
    "Potwari": f"{IA}.northwestern.pahari_pothwari",
    "Mongolian": "mongolic.mongolian",
    "Newar": "sinotibetan.newaric.newar",
    "Panjabi (India)": f"{IA}.northwestern.punjabi",
    "Punjabi (Not otherwise specified)": f"{IA}.northwestern.punjabi",
    "Urdu": f"{IA}.central.urdu",
    "Romani": f"{IA}.romani.romani",
    "Romany (Not otherwise specified)": f"{IA}.romani",
    "Romany English": f"{IA}.romani.angloromani",
    "Traveller Irish": f"{IE}.celtic.shelta",
    "Saurashtra": f"{IA}.gujarati.saurashtra",
    "Sotho (Not otherwise specified)": "nigercongo.bantu.sotho_tswana",
    "Sotho/Sesotho": "nigercongo.bantu.sotho_tswana.sesotho",
    "Tibetan": "sinotibetan.tibetic.tibetan",
    "Tiv": "nigercongo.tiv",
    "Tonga (Not otherwise specified)": "nigercongo.bantu",   # Zambia's, Malawi's or Zimbabwe's; Tongan islanders would say Tongan
    "Tsonga": "nigercongo.bantu.tswa_ronga.tsonga",
    "Tumbuka": "nigercongo.bantu.tumbuka.tumbuka",
    "Other languages": "other",   # the table's "Other languages (1)": 10 or fewer people each, and unspecified
}

NAMES = {**{f"ew:{k}": v for k, v in EW.items()},
         **{f"sc:{k}": v for k, v in SC.items()},
         **{f"sc:Other language: {k}": v for k, v in SC_OTHER.items()},
         **{f"ni:{k}": v for k, v in NI.items()}}

def resolve(name, unit=""):
    if name == "ew:English (English or Welsh in Wales)" and str(unit).startswith("W"):
        raise ValueError(f"unsplit 'English or Welsh' box in Welsh unit {unit}: "
                         "re-run sources/uk_census.py")
    return NAMES[name]
