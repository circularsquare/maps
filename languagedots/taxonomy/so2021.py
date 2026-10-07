"""Somalia: REACH JMCNA 2021 household main language (sources/so_jmcna.py, sources/so.md).

Glottolog: Maay maay1238 is a language of its own beside Somali soma1255 (both Lowland East
Cushitic); Benaadir bena1268 is a dialect of Somali, drawn as a sibling (not a child) because
the survey offers it as an answer beside Standard Somali, and a child would turn Somali into a
group node. Mushungulu mush1238 is a dialect of Zigula (Bantu), Bajuni baju1245 and Chimwiini
(the survey's "Bravanese") are Swahili varieties, drawn as siblings of Swahili as Kenya does
for Bajuni. English and Italian are drawn as measured: the question is home language, not
ability.
"""
NAMES = {
    "Standard / Northern Somali": "afroasiatic.cushitic.lowland.somali",
    "Benaadir Somali": "afroasiatic.cushitic.lowland.benaadir",
    "Maay Somali": "afroasiatic.cushitic.lowland.maay",
    "Oromo": "afroasiatic.cushitic.lowland.oromo",
    "Arabic": "afroasiatic.arabic",
    "English": "indoeuropean.germanic.english",
    "Italian": "indoeuropean.romance.italian",
    "Bravanese (Chimwiini / Chimbalazi)": "nigercongo.bantu.chimwiini",
    "Kibajuni": "nigercongo.bantu.bajuni",
    "Mushunguli": "nigercongo.bantu.mushungulu",
    "Other": "other",
}


def resolve(label):
    if label in NAMES:
        return NAMES[label]
    raise KeyError(f"so2021: unmapped label {label!r}")
