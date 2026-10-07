"""Kenya, Afrobarometer R4 and R6-R9 (2008-2022), home language -> node.

Keyed by the answers sources/ke_afro.py writes to data/normalized/ke.csv: the survey's coded
labels (spellings merged there, in CODED), the languages its "Other (specify)" free text names
(VERBATIM there), and the four languages the two combined card labels are split into (Meru,
Embu from "Meru/Embu"; Maasai, Samburu from "Masai/Samburu"). The tree, and the Glottolog
check of every family and branch, is taxonomy/tree.d/ke.txt.

Clusters drawn as one language. The card's Luhya, Kalenjin and Mijikenda each name a cluster
that Glottolog files as a subgroup (Luyia luyi1234, Kalenjin kale1246, Mijikenda miji1238).
They are leaves here, because that is the answer 900-1,500 respondents gave each; a group node
would draw them washed out as "language not named". The few who wrote a variety (Maragoli,
Bukusu, Digo, Duruma, Nandi, Kipsigis) are counted in the cluster by sources/ke_afro.py.

Remainders:
  * "Other African language": a free-text answer naming a language only one respondent gave,
    or one not identifiable (Bokom, Shelshel, Munyaya, Watta, Malakote, Kisagalla, Nyarwanda,
    Nyasa, Nubi). `africa_other`.
  * "Other language": "Indian", "Hindu", "Asian South", "Punjabi" (R4 free text): South Asian
    answers no narrower node holds. `other`.
  * "Arabic": `afroasiatic.arabic`, as everywhere else on this map.
"""
BA = "nigercongo.bantu"
NI = "nilosaharan.nilotic"
LC = "afroasiatic.cushitic.lowland"

NAMES = {
    # Bantu
    "Kikuyu": f"{BA}.gikuyu",
    "Swahili": f"{BA}.swahili",
    "Kamba": f"{BA}.kamba",
    "Luhya": f"{BA}.luhya",
    "Kisii": f"{BA}.gusii",
    "Mijikenda": f"{BA}.mijikenda",
    "Meru": f"{BA}.meru",
    "Embu": f"{BA}.embu",
    "Taita": f"{BA}.taita",
    "Kuria": f"{BA}.kuria",
    "Suba": f"{BA}.suba",
    "Pokomo": f"{BA}.pokomo",
    "Bajuni": f"{BA}.bajuni",
    "Sheng": f"{BA}.sheng",
    # Nilotic
    "Luo": f"{NI}.luo",
    "Kalenjin": f"{NI}.kalenjin",
    "Pokot": f"{NI}.pokot",
    "Sabaot": f"{NI}.sabaot",
    "Okiek": f"{NI}.okiek",
    "Maasai": f"{NI}.maasai",
    "Samburu": f"{NI}.samburu",
    "Turkana": f"{NI}.turkana",
    "Teso": f"{NI}.teso",
    # Cushitic
    "Somali": f"{LC}.somali",
    "Borana": f"{LC}.oromo",                 # Borana Oromo (bora1271); Gabra is its dialect
    "Orma": f"{LC}.orma",
    "Garre": f"{LC}.garre",
    "Rendille": f"{LC}.rendille",
    # others
    "Arabic": "afroasiatic.arabic",
    "English": "indoeuropean.germanic.english",
    "Gujarati": "indoeuropean.indoaryan.gujarati.gujarati",
    "Other African language": "africa_other",
    "Other language": "other",
}

EXTRA_NODES = []


def resolve(name):
    return NAMES.get(name)
