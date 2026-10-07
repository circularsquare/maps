"""Uganda: the answers of sources/ug_census.py -> node.

The census (NPHC 2024) asked tribe or nationality, not language. sources/ug_census.py turns each
subcounty's sampled tribes into home languages with Afrobarometer's shares (R4-R9), and its
answer names are the ones below. The levels and glottocodes are in tree.d/ug.txt.

Calls (the ethnic group -> language half lives in sources/ug_census.py's OWN):
  * Rufumbira is a sibling of Kinyarwanda, Labwor (Lebthur) of Acholi: dialects in Glottolog,
    named apart on the survey's card and by Uganda's census.
  * Luhya: Kenya's cluster leaf; Uganda's Babukusu and Maragoli and the survey's "Baluya".
  * "Other Ugandan language" (africa_other): the census's "Other Ugandan" and the named groups
    whose language could not be identified (Gimara, Reli, Shana, Vonoma, Bakingwe).
  * Non-Ugandans by nationality: Rwanda -> Kinyarwanda, Burundi -> Kirundi, Somalia -> Somali;
    every other African nationality on "Other African language" (africa_other), the rest on
    `other`. A nationality does not name a language (South Sudanese refugees alone speak
    Bari, Kakwa, Kuku, Dinka, Nuer, Madi, Acholi, Arabic...).
"""
B = "nigercongo.bantu"
NIL = "nilosaharan.nilotic"
CS = "nilosaharan.centralsudanic"

NAMES = {
    "Ganda": f"{B}.ganda", "Soga": f"{B}.soga", "Nyankore": f"{B}.nyankole",
    "Chiga": f"{B}.chiga", "Masaaba": f"{B}.masaaba", "Nyoro": f"{B}.nyoro",
    "Tooro": f"{B}.tooro", "Konzo": f"{B}.konzo", "Gwere": f"{B}.gwere",
    "Saamia": f"{B}.saamia", "Nyole": f"{B}.nyole", "Ruuli": f"{B}.ruuli",
    "Kenyi": f"{B}.kenyi", "Gungu": f"{B}.gungu", "Bwisi": f"{B}.bwisi", "Amba": f"{B}.amba",
    "Rufumbira": f"{B}.rufumbira", "Kinyarwanda": f"{B}.kinyarwanda",
    "Kirundi": f"{B}.kirundi", "Tagwenda": f"{B}.tagwenda", "Songora": f"{B}.songora",
    "Nyara": f"{B}.nyara", "Luhya": f"{B}.luhya", "Haya": f"{B}.haya", "Hehe": f"{B}.hehe",
    "Swahili": f"{B}.swahili",
    "Acholi": f"{NIL}.acholi", "Labwor": f"{NIL}.labwor", "Lango": f"{NIL}.lango",
    "Alur": f"{NIL}.alur", "Adhola": f"{NIL}.adhola", "Kumam": f"{NIL}.kumam",
    "Chope": f"{NIL}.chope", "Luo (Dholuo)": f"{NIL}.luo",
    "Teso": f"{NIL}.teso", "Karamojong": f"{NIL}.karamojong", "Kakwa": f"{NIL}.kakwa",
    "Kuku": f"{NIL}.kuku", "Kupsabiny": f"{NIL}.kupsabiny", "Sabaot": f"{NIL}.sabaot",
    "Pokot": f"{NIL}.pokot",
    "Lugbara": f"{CS}.lugbara", "Aringa": f"{CS}.aringa", "Ndo": f"{CS}.ndo",
    "Lendu": f"{CS}.lendu", "Mvuba": f"{CS}.mvuba", "Ma'di": "nilosaharan.madi",
    "Ik": "nilosaharan.kuliak.ik",
    "Nubi": "afroasiatic.arabic.nubi", "Somali": "afroasiatic.cushitic.lowland.somali",
    "English": "indoeuropean.germanic.english",
    "Other Ugandan language": "africa_other", "Other African language": "africa_other",
    "Other language": "other",
}


def resolve(name):
    return NAMES.get(name)
