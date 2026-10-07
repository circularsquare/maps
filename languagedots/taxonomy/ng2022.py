"""Nigeria, Afrobarometer R4-R9 (2008-2022), home language -> node.

Keyed by the answers sources/ng_afro.py writes to data/normalized/ng.csv: the survey's coded
labels (two spellings or two names of one language merged there, in CODED), and the languages
its "Other (specify)" free text names (VERBATIM there). The tree, and the Glottolog check of
every family and branch, is taxonomy/tree.d/ng.txt.

Remainders:
  * "Other Nigerian language": a free-text answer naming a language only one respondent gave,
    a place that holds several languages, or nothing identifiable. `africa_other`.
  * "Gwoza": coded by the survey in R5 (7 respondents, Borno); Gwoza is an LGA whose people
    speak several Chadic languages (Glavda, Guduf, Dghwede, Lamang). On `afroasiatic.chadic`,
    the narrowest node holding them all, so drawn as an unnamed Chadic language.
  * "English": the four English-at-home respondents who gave no ethnic group (the rest are
    drawn on their group's language; sources/ng_afro.py, english_by_ethnicity).
"""
VN = "nigercongo.voltaniger"
ED = "nigercongo.voltaniger.edoid"
IJ = "nigercongo.ijoid"
CR = "nigercongo.crossriver"
PL = "nigercongo.plateau"
KA = "nigercongo.kainji"
JU = "nigercongo.jukunoid"
BD = "nigercongo.bantoid"
AD = "nigercongo.adamawa"
CH = "afroasiatic.chadic"

NAMES = {
    "Hausa": f"{CH}.hausa",
    "Yoruba": f"{VN}.yoruba",
    "Igbo": f"{VN}.igbo",
    "Fula": "nigercongo.atlantic.fulah",
    "Kanuri": "nilosaharan.kanuri",
    "English": "indoeuropean.germanic.english",
    "Nigerian Pidgin": "creole.english_based.nigerian_pidgin",
    "Tiv": "nigercongo.tiv",
    "Ekajuk": "nigercongo.ekajuk",
    "Zarma": "nilosaharan.songhay.zarma",
    "Shuwa Arabic": "afroasiatic.arabic.shuwa",
    "Busa": "nigercongo.mande.busa",
    "Bariba": "nigercongo.gur.bariba",
    # Volta-Niger
    "Nupe": f"{VN}.nupe", "Igala": f"{VN}.igala", "Ibaji": f"{VN}.ibaji",
    "Idoma": f"{VN}.idoma", "Agatu": f"{VN}.agatu", "Alago": f"{VN}.alago", "Yala": f"{VN}.yala",
    "Etulo": f"{VN}.etulo", "Igede": f"{VN}.igede", "Yace": f"{VN}.yace",
    "Ebira": f"{VN}.ebira", "Gbagyi": f"{VN}.gbagyi", "Bassa-Nge": f"{VN}.bassa_nge",
    "Ganagana": f"{VN}.ganagana", "Gade": f"{VN}.gade",
    "Ika": f"{VN}.ika", "Ukwuani": f"{VN}.ukwuani", "Ikwere": f"{VN}.ikwere",
    "Ekpeye": f"{VN}.ekpeye", "Ogba": f"{VN}.ogba", "Ohafia": f"{VN}.ohafia",
    "Edo": f"{ED}.edo", "Urhobo": f"{ED}.urhobo", "Isoko": f"{ED}.isoko", "Esan": f"{ED}.esan",
    "Yekhee": f"{ED}.yekhee", "Okpela": f"{ED}.okpela", "Okpe": f"{ED}.okpe",
    "Epie": f"{ED}.epie", "Degema": f"{ED}.degema", "Emai-Iuleha-Ora (Owan)": f"{ED}.emai",
    # Ijoid
    "Ijaw": f"{IJ}.ijaw", "Kalabari": f"{IJ}.kalabari", "Okrika": f"{IJ}.okrika",
    "Nembe": f"{IJ}.nembe", "Ibani": f"{IJ}.ibani",
    # Cross River
    "Ibibio": f"{CR}.ibibio", "Efik": f"{CR}.efik", "Anaang": f"{CR}.anaang",
    "Oron": f"{CR}.oron", "Obolo": f"{CR}.obolo", "Ogoni": f"{CR}.ogoni",
    "Khana": f"{CR}.khana", "Eleme": f"{CR}.eleme", "Abua": f"{CR}.abua",
    "Ogbia": f"{CR}.ogbia", "Yakurr": f"{CR}.yakurr", "Mbembe": f"{CR}.mbembe",
    # Plateau
    "Tarok": f"{PL}.tarok", "Berom": f"{PL}.berom", "Tyap": f"{PL}.tyap", "Jju": f"{PL}.jju",
    "Gyong": f"{PL}.gyong", "Aten": f"{PL}.aten", "Hyam (Jaba)": f"{PL}.hyam",
    "Adara": f"{PL}.adara", "Eggon": f"{PL}.eggon", "Mada": f"{PL}.mada",
    "Ninzo": f"{PL}.ninzo", "Numana": f"{PL}.numana", "Irigwe": f"{PL}.irigwe",
    "Izere": f"{PL}.izere", "Migili": f"{PL}.migili", "Koro": f"{PL}.koro",
    # Kainji
    "Kambari": f"{KA}.kambari", "Bassa": f"{KA}.bassa", "Kurama": f"{KA}.kurama",
    "Laru": f"{KA}.laru", "Lela (Dakarkari)": f"{KA}.lela",
    # Jukunoid
    "Jukun": f"{JU}.jukun", "Kuteb": f"{JU}.kuteb", "Jibu": f"{JU}.jibu",
    "Tigon": f"{JU}.tigon", "Etkywan (Ichen)": f"{JU}.etkywan",
    # Bantoid
    "Ejagham": f"{BD}.ejagham", "Jarawa": f"{BD}.jarawa", "Mambila": f"{BD}.mambila",
    "Ndoola": f"{BD}.ndoola", "Bekwarra": f"{BD}.bekwarra", "Bette-Bendi": f"{BD}.bette_bendi",
    # Adamawa
    "Mumuye": f"{AD}.mumuye", "Chamba": f"{AD}.chamba", "Dza (Jenjo)": f"{AD}.dza",
    "Yungur": f"{AD}.yungur", "Waja": f"{AD}.waja", "Tula": f"{AD}.tula",
    # Chadic
    "Ngas": f"{CH}.ngas", "Tangale": f"{CH}.tangale", "Bura-Pabir": f"{CH}.bura_pabir",
    "Marghi": f"{CH}.marghi", "Karekare": f"{CH}.karekare", "Bade": f"{CH}.bade",
    "Kamwe": f"{CH}.kamwe", "Mwaghavul": f"{CH}.mwaghavul", "Gwandara": f"{CH}.gwandara",
    "Tera": f"{CH}.tera", "Bole": f"{CH}.bole", "Bachama": f"{CH}.bachama",
    "Zaar": f"{CH}.zaar", "Goemai": f"{CH}.goemai", "Kibaku (Chibok)": f"{CH}.kibaku",
    "Huba (Kilba)": f"{CH}.huba", "Glavda": f"{CH}.glavda", "Piapung": f"{CH}.piapung",
    # remainders
    "Gwoza": CH,
    "Other Nigerian language": "africa_other",
}

EXTRA_NODES = []


def resolve(name):
    return NAMES.get(name)
