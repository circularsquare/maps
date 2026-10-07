"""Canada, Census of Population 2021, mother tongue -> node. Keys are the leaf labels of the
Census Profile's mother-tongue block (sources/ca_census.py), exactly as StatCan prints them.

Calls worth knowing (sources/ca.md says more):
  * "X, n.o.s." (not otherwise specified) is someone who named X without the variety the census
    splits X into. Where StatCan's members of X are separate languages or varieties with rows of
    their own, the answer sits on the group, as us2024 and uk2021 put unspecified Chinese
    (spec §3.2): Cree, Slavey, Tutchone, Low German, Chinese, Creole. Malagasy is the exception:
    Merina is a sibling leaf, `austronesian.merina`, as ru.txt does for Mari and Mordvin, so that
    `austronesian.malagasy` stays a leaf for Madagascar. Where the tree
    already has X as a leaf (Ojibwe, us.txt) or no row names the language X usually means
    (Dene: there is no Dene Suline/Chipewyan row, so "Dene" is that language's answer), it sits
    on the leaf. "Aramaic, n.o.s." gets a leaf of its own, Aramaic, because the tree files
    Assyrian and Chaldean Neo-Aramaic directly under Afroasiatic.
  * "Iranian Persian" and "Persian (Farsi), n.o.s." both go on Persian (Farsi), the node us2024
    and uk2021 use for "Farsi" and "Persian or Farsi". Dari has its own row and node.
  * "Oriya, n.o.s." merges into Odia: Oriya is the older spelling of the same name.
  * "Ndebele" does not say which of the two; it sits on Nguni, which holds both.
  * "X languages, n.i.e." (not included elsewhere: a named language the list has no row for)
    sits on X, the narrowest node the census put it under. "Indigenous languages, n.i.e./n.o.s."
    go on americas_other; "African, n.o.s." and "Other languages, n.i.e." on `other`.
  * "Mina" follows StatCan's own classification (Chadic); see tree.d/ca.txt.
  * The five multiple-response rows are not languages; countries/ca.py shares them out with
    MULTIPLE below.
"""
EN = "indoeuropean.germanic.english"
FR = "indoeuropean.romance.french"
IE = "indoeuropean"
IA = "indoeuropean.indoaryan"
IR = "indoeuropean.iranian"
GE = "indoeuropean.germanic"
CW = "indoeuropean.germanic.continental"
RO = "indoeuropean.romance"
SL = "indoeuropean.slavic"
AA = "afroasiatic"
AN = "austronesian"
PH = "austronesian.philippine"
NC = "nigercongo"
ST = "sinotibetan"
CR = "algic.creeinnu"
ND = "nadene.northern"

NAMES = {
    "English": EN,
    "French": FR,
    # ---- Indigenous: Algonquian ----
    "Blackfoot": "algic.blackfoot",
    "Atikamekw": f"{CR}.atikamekw",
    "Ililimowin (Moose Cree)": f"{CR}.cree.moose",
    "Inu Ayimun (Southern East Cree)": f"{CR}.cree.southeast",
    "Iyiyiw-Ayimiwin (Northern East Cree)": f"{CR}.cree.northeast",
    "Nehinawewin (Swampy Cree)": f"{CR}.cree.swampy",
    "Nehiyawewin (Plains Cree)": f"{CR}.cree.plains",
    "Nihithawiwin (Woods Cree)": f"{CR}.cree.woods",
    "Cree, n.o.s.": f"{CR}.cree",
    "Innu (Montagnais)": f"{CR}.innu",
    "Naskapi": f"{CR}.naskapi",
    "Mi'kmaq": "algic.eastern.mikmaq",
    "Wolastoqewi (Malecite)": "algic.eastern.wolastoqey",
    "Anicinabemowin (Algonquin)": "algic.algonquin",
    "Oji-Cree": "algic.ojicree",
    "Anishinaabemowin (Chippewa)": "algic.chippewa",
    "Daawaamwin (Odawa)": "algic.odawa",
    "Saulteau (Western Ojibway)": "algic.saulteaux",
    "Ojibway, n.o.s.": "algic.ojibwe",
    "Algonquian languages, n.i.e.": "algic",
    "Michif": "algic.michif",
    # ---- Athabaskan ----
    "Dakelh (Carrier)": f"{ND}.dakelh",
    "Dane-zaa (Beaver)": f"{ND}.danezaa",
    "Dene, n.o.s.": f"{ND}.dene",
    "Gwich'in": f"{ND}.gwichin",
    "Deh Gah Ghotie Zhatie (South Slavey)": f"{ND}.slavey.south",
    "Satuotine Yati (North Slavey)": f"{ND}.slavey.north",
    "Slavey, n.o.s.": f"{ND}.slavey",
    "Kaska (Nahani)": f"{ND}.tahltan.kaska",
    "Tahltan": f"{ND}.tahltan.tahltan",
    "Tlicho (Dogrib)": f"{ND}.tlicho",
    "Tse'khene (Sekani)": f"{ND}.sekani",
    "Tsilhqot'in (Chilcotin)": f"{ND}.tsilhqotin",
    "Tsuu T'ina (Sarsi)": f"{ND}.tsuutina",
    "Northern Tutchone": f"{ND}.tutchone.north",
    "Southern Tutchone": f"{ND}.tutchone.south",
    "Tutchone, n.o.s.": f"{ND}.tutchone",
    "Wetsuwet'en-Babine": f"{ND}.wetsuweten",
    "Tlingit": "nadene.tlingit",
    "Athabaskan languages, n.i.e.": "nadene",
    # ---- the rest of the Indigenous list ----
    "Haida": "isolate.haida",
    "Inuinnaqtun": "eskimoaleut.westcanadian.inuinnaqtun",
    "Inuvialuktun": "eskimoaleut.westcanadian.inuvialuktun",
    "Inuktitut": "eskimoaleut.inuktitut",
    "Inuktut (Inuit) languages, n.i.e.": "eskimoaleut",
    "Cayuga": "iroquoian.cayuga",
    "Mohawk": "iroquoian.mohawk",
    "Oneida": "iroquoian.oneida",
    "Iroquoian languages, n.i.e.": "iroquoian",
    "Ktunaxa (Kutenai)": "isolate.ktunaxa",
    "Halkomelem": "salishan.halkomelem",
    "Lillooet": "salishan.lillooet",
    "Ntlakapamux (Thompson)": "salishan.thompson",
    "Secwepemctsin (Shuswap)": "salishan.shuswap",
    "Squamish": "salishan.squamish",
    "Straits": "salishan.straits",
    "Syilx (Okanagan)": "salishan.okanagan",
    "Salish languages, n.i.e.": "salishan",
    "Assiniboine": "siouan.assiniboine",
    "Dakota": "siouan.dakota",
    "Stoney": "siouan.stoney",
    "Siouan languages, n.i.e.": "siouan",
    "Gitxsan (Gitksan)": "tsimshianic.gitxsan",
    "Nisga'a": "tsimshianic.nisgaa",
    "Tsimshian": "tsimshianic.tsimshian",
    "Haisla": "wakashan.haisla",
    "Heiltsuk": "wakashan.heiltsuk",
    "Kwak'wala (Kwakiutl)": "wakashan.kwakwala",
    "Nuu-chah-nulth (Nootka)": "wakashan.nuuchahnulth",
    "Wakashan languages, n.i.e.": "wakashan",
    "Indigenous languages, n.i.e.": "americas_other",
    "Indigenous languages, n.o.s.": "americas_other",
    # ---- Afroasiatic ----
    "Kabyle": f"{AA}.berber.kabyle",
    "Tamazight": f"{AA}.berber.tamazight",
    "Berber languages, n.i.e.": f"{AA}.berber",
    "Hausa": f"{AA}.chadic.hausa",
    "Mina": f"{AA}.chadic.mina",
    "Coptic": f"{AA}.coptic",
    "Bilen": f"{AA}.cushitic.agaw.bilen",
    "Oromo": f"{AA}.cushitic.lowland.oromo",
    "Somali": f"{AA}.cushitic.lowland.somali",
    "Cushitic languages, n.i.e.": f"{AA}.cushitic",
    "Amharic": f"{AA}.ethiosemitic.amharic",
    "Arabic": f"{AA}.arabic",
    "Assyrian Neo-Aramaic": f"{AA}.assyrian",
    "Chaldean Neo-Aramaic": f"{AA}.chaldean",
    "Aramaic, n.o.s.": f"{AA}.aramaic",
    "Harari": f"{AA}.ethiosemitic.harari",
    "Hebrew": f"{AA}.hebrew",
    "Maltese": f"{AA}.maltese",
    "Tigrigna": f"{AA}.ethiosemitic.tigrinya",
    "Semitic languages, n.i.e.": AA,   # the tree has no Semitic level
    # ---- Austroasiatic, Austronesian ----
    "Khmer (Cambodian)": "austroasiatic.khmer",
    "Vietnamese": "austroasiatic.vietnamese",
    "Austro-Asiatic languages, n.i.e": "austroasiatic",
    "Bikol": f"{PH}.bikol",
    "Bisaya, n.o.s.": f"{PH}.bisayan",
    "Cebuano": f"{PH}.cebuano",
    "Fijian": f"{AN}.oceanic.fijian",
    "Hiligaynon": f"{PH}.hiligaynon",
    "Ilocano": f"{PH}.ilocano",
    "Indonesian": f"{AN}.malayic.indonesian",
    "Kankanaey": f"{PH}.kankanaey",
    "Kinaray-a": f"{PH}.kinaraya",
    "Merina": f"{AN}.merina",
    "Malagasy, n.o.s.": f"{AN}.malagasy",
    "Malay": f"{AN}.malayic.malay",
    "Pampangan (Kapampangan, Pampango)": f"{PH}.kapampangan",
    "Pangasinan": f"{PH}.pangasinan",
    "Tagalog (Pilipino, Filipino)": f"{PH}.tagalog",
    "Waray-Waray": f"{PH}.waray",
    "Austronesian languages, n.i.e.": AN,
    # ---- creoles ----
    "Haitian Creole": "creole.french_based.haitian",
    "Jamaican English Creole": "creole.english_based.jamaican",
    "Krio": "creole.english_based.krio",
    "Morisyen": "creole.french_based.morisyen",
    "Sango": f"{NC}.sango",
    "Creole, n.o.s.": "creole",
    "Creole languages, n.i.e.": "creole",
    # ---- Dravidian, Kartvelian, Hmong-Mien ----
    "Kannada": "dravidian.southern.kannada",
    "Malayalam": "dravidian.southern.malayalam",
    "Tamil": "dravidian.southern.tamil",
    "Telugu": "dravidian.southcentral.telugu",
    "Tulu": "dravidian.southern.tulu",
    "Dravidian languages, n.i.e.": "dravidian",
    "Georgian": "kartvelian.georgian",
    "Hmong-Mien languages": "hmongmien",
    # ---- Indo-European ----
    "Albanian": f"{IE}.albanian.albanian",
    "Armenian": f"{IE}.armenian.armenian",
    "Latvian": f"{IE}.baltic.latvian",
    "Lithuanian": f"{IE}.baltic.lithuanian",
    "Belarusian": f"{SL}.east.belarusian",
    "Bulgarian": f"{SL}.south.bulgarian",
    "Czech": f"{SL}.west.czech",
    "Macedonian": f"{SL}.south.macedonian",
    "Polish": f"{SL}.west.polish",
    "Russian": f"{SL}.east.russian",
    "Rusyn": f"{SL}.east.rusyn",
    "Bosnian": f"{SL}.south.bosnian",
    "Croatian": f"{SL}.south.croatian",
    "Serbian": f"{SL}.south.serbian",
    "Serbo-Croatian, n.i.e.": f"{SL}.south.serbocroatian",
    "Slovak": f"{SL}.west.slovak",
    "Slovene (Slovenian)": f"{SL}.south.slovenian",
    "Ukrainian": f"{SL}.east.ukrainian",
    "Slavic languages, n.i.e.": SL,
    "Irish": f"{IE}.celtic.irish",
    "Scottish Gaelic": f"{IE}.celtic.scottishgaelic",
    "Welsh": f"{IE}.celtic.welsh",
    "Celtic languages, n.i.e.": f"{IE}.celtic",
    "Frisian": f"{GE}.frisian",
    "German": f"{CW}.german",
    "Pennsylvania German": f"{CW}.pennsylvania_german",
    "Swiss German": f"{CW}.swiss_german",
    "Yiddish": f"{CW}.yiddish",
    "Afrikaans": f"{CW}.afrikaans",
    "Dutch": f"{CW}.dutch",
    "Low German, n.o.s.": f"{CW}.lowgerman",
    "Low Saxon": f"{CW}.lowgerman.lowsaxon",
    "Plautdietsch": f"{CW}.lowgerman.plautdietsch",
    "Vlaams (Flemish)": f"{CW}.flemish",
    "Danish": f"{GE}.north.danish",
    "Icelandic": f"{GE}.north.icelandic",
    "Norwegian": f"{GE}.north.norwegian",
    "Swedish": f"{GE}.north.swedish",
    "Germanic languages, n.i.e.": GE,
    "Greek": f"{IE}.hellenic.greek",
    "Assamese": f"{IA}.eastern.assamese",
    "Bengali": f"{IA}.eastern.bengali",
    "Gujarati": f"{IA}.gujarati.gujarati",
    "Hindi": f"{IA}.central.hindi",
    "Kacchi": f"{IA}.northwestern.kachchhi",
    "Kashmiri": f"{IA}.dardic.kashmiri",
    "Konkani": f"{IA}.southern.konkani",
    "Marathi": f"{IA}.southern.marathi",
    "Nepali": f"{IA}.pahari.eastern.nepali",
    "Odia": f"{IA}.eastern.odia",
    "Oriya, n.o.s.": f"{IA}.eastern.odia",       # the older spelling of Odia
    "Punjabi (Panjabi)": f"{IA}.northwestern.punjabi",
    "Rohingya": f"{IA}.eastern.rohingya",
    "Sindhi": f"{IA}.northwestern.sindhi",
    "Sinhala (Sinhalese)": f"{IA}.sinhala",
    "Urdu": f"{IA}.central.urdu",
    "Indo-Aryan languages, n.i.e.": IA,
    "Baluchi": f"{IR}.balochi",
    "Kurdish": f"{IR}.kurdish",
    "Parsi": f"{IR}.parsi",
    "Pashto": f"{IR}.pashto",
    "Dari": f"{IR}.dari",
    "Iranian Persian": f"{IR}.persian",
    "Persian (Farsi), n.o.s.": f"{IR}.persian",
    "Iranian languages, n.i.e.": IR,
    "Indo-Iranian languages, n.i.e.": IE,  # no Indo-Iranian level in the tree
    "Catalan": f"{RO}.catalan",
    "Italian": f"{RO}.italian",
    "Portuguese": f"{RO}.portuguese",
    "Romanian": f"{RO}.romanian",
    "Spanish": f"{RO}.spanish",
    "Italic (Romance) languages, n.i.e.": RO,
    "Indo-European languages, n.i.e.": IE,
    # ---- East Asia ----
    "Japanese": "japonic.japanese",
    "Korean": "koreanic.korean",
    "Mongolian": "mongolic.mongolian",
    # ---- Niger-Congo, Nilo-Saharan ----
    "Akan (Twi)": f"{NC}.kwa.akan",
    "Bamanankan": f"{NC}.mande.bambara",
    "Edo": f"{NC}.voltaniger.edoid",
    "Éwé": f"{NC}.kwa.gbe",
    "Fulah (Pular, Pulaar, Fulfulde)": f"{NC}.atlantic.fulah",
    "Ga": f"{NC}.kwa.ga",
    "Ganda": f"{NC}.bantu.ganda",
    "Gikuyu": f"{NC}.bantu.gikuyu",
    "Igbo": f"{NC}.voltaniger.igbo",
    "Kinyarwanda (Rwanda)": f"{NC}.bantu.kinyarwanda",
    "Lingala": f"{NC}.bantu.lingala",
    "Luba-Kasai": f"{NC}.bantu.luba_kasai",
    "Mòoré": f"{NC}.gur.moore",
    "Mwani": f"{NC}.bantu.mwani",
    "Ndebele": f"{NC}.bantu.nguni",
    "Rundi (Kirundi)": f"{NC}.bantu.kirundi",
    "Shona": f"{NC}.bantu.shona",
    "Soninke": f"{NC}.mande.soninke",
    "Sotho-Tswana languages": f"{NC}.bantu.sotho_tswana",   # za.txt's node
    "Swahili": f"{NC}.bantu.swahili",
    "Wojenaka": f"{NC}.mande.wojenaka",
    "Wolof": f"{NC}.atlantic.wolof",
    "Yoruba": f"{NC}.voltaniger.yoruba",
    "Niger-Congo languages, n.i.e.": NC,
    "Dinka": "nilosaharan.nilotic.dinka",
    "Nuer": "nilosaharan.nilotic.nuer",
    "Nilo-Saharan languages, n.i.e.": "nilosaharan",
    "African, n.o.s.": "other",
    # ---- sign languages ----
    "American Sign Language": "signlanguage.asl",
    "Quebec Sign Language": "signlanguage.lsq",
    "Sign languages, n.i.e.": "signlanguage",
    # ---- Sino-Tibetan ----
    "Hakka": f"{ST}.sinitic.hakka",
    "Mandarin": f"{ST}.sinitic.mandarin",
    "Min Dong": f"{ST}.sinitic.min_dong",
    "Min Nan (Chaochow, Teochow, Fukien, Taiwanese)": f"{ST}.sinitic.min_nan",
    "Wu (Shanghainese)": f"{ST}.sinitic.wu",
    "Yue (Cantonese)": f"{ST}.sinitic.cantonese",
    "Chinese, n.o.s.": f"{ST}.sinitic",
    "Chinese languages, n.i.e.": f"{ST}.sinitic",
    "Burmese": f"{ST}.burmish.burmese",
    "Kuki-Chin languages": f"{ST}.kukichin",
    "S'gaw Karen": f"{ST}.karen.sgaw",
    "Karenic languages, n.i.e.": f"{ST}.karen",
    "Tibetan": f"{ST}.tibetic.tibetan",
    "Tibeto-Burman languages, n.i.e.": ST,
    "Sino-Tibetan languages, n.i.e.": ST,
    # ---- Kra-Dai, Turkic, Uralic ----
    "Lao": "kradai.lao",
    "Thai": "kradai.thai",
    "Tai-Kadai languages, n.i.e.": "kradai",
    "Azerbaijani": "turkic.azerbaijani",
    "Kazakh": "turkic.kazakh",
    "Turkish": "turkic.turkish",
    "Uyghur": "turkic.uyghur",
    "Uzbek": "turkic.uzbek",
    "Turkic languages, n.i.e.": "turkic",
    "Estonian": "uralic.estonian",
    "Finnish": "uralic.finnish",
    "Hungarian": "uralic.hungarian",
    "Other languages, n.i.e.": "other",
}

# The multiple-response rows: each person shared equally across the languages they named
# (spec §3.6, the exact 1/k split, since the row says which combination). NONOFFICIAL is a
# non-official language the census does not name; countries/ca.py shares it over the unit's
# own single-response non-official languages. "(s)": a person who named two non-official
# languages besides English counts here as one; the profile does not say how many.
NONOFFICIAL = "*nonofficial"
MULTIPLE = {
    "English and French": {EN: 1 / 2, FR: 1 / 2},
    "English and non-official language(s)": {EN: 1 / 2, NONOFFICIAL: 1 / 2},
    "French and non-official language(s)": {FR: 1 / 2, NONOFFICIAL: 1 / 2},
    "English, French and non-official language(s)": {EN: 1 / 3, FR: 1 / 3, NONOFFICIAL: 1 / 3},
    "Multiple non-official languages": {NONOFFICIAL: 1.0},
}


def resolve(label):
    return NAMES.get(label)
