"""Ghana: the answers of sources/gh_census.py -> node.

The census asked ethnicity, not language; sources/gh_census.py turns each district's count of
the nine major groups into home languages with Afrobarometer's shares (R4-R9, 13,030
respondents who named both), and splits Ga from Dangme and Nzema from Akan by the census's own
literacy table. The answers below are that script's names, which follow Glottolog
(data/raw/glottolog/languages.csv; tree.d/gh.txt has the levels). Glottocodes beside each
language are in sources/gh_census.py's HOME.

Calls:
  * Akan is one node, `akan`: the survey codes Akan once, with no Twi/Fante split, and the
    literacy table's Asante Twi, Akuapem Twi and Fante are dialects of it.
  * Dagaare goes on `dagara`, bf.txt's "Dagara (Dagaare)": the survey's Dagaare, Dagaari,
    Dagarti answers are Glottolog's Central and Southern Dagaare; Northern Dagara is the Burkina
    side. One node for the cluster, as Burkina's census has one answer.
  * Nanuni (Nanumba) answers are on Dagbani, of which it is a dialect. Nankani is on Farefare,
    Krobo on Dangme, Kanjaga on Buli, Wassa, Fante, Assin and Bono on Akan: dialects.
  * Remainders: "Guan (not named)" sits on the `guan` group, "Gurunsi (not named)" on bf.txt's
    `gurunsi` leaf, "Gur/Kwa/Mande (not named)" on those groups: a survey answer of "Other"
    whose verbatim names nothing identifiable, put on the narrowest node holding what the
    respondent's census group speaks. "Other African language" (the Others group's) on
    `africa_other`.
"""
GUR = "nigercongo.gur"
KWA = "nigercongo.kwa"
MANDE = "nigercongo.mande"

NAMES = {
    "Akan": f"{KWA}.akan", "Nzema": f"{KWA}.nzema", "Sehwi": f"{KWA}.sehwi",
    "Anufo": f"{KWA}.anufo", "Ga": f"{KWA}.ga", "Dangme": f"{KWA}.adangme",
    "Ewe": f"{KWA}.ewe",
    "Gonja": f"{KWA}.guan.gonja", "Krache": f"{KWA}.guan.krache",
    "Gikyode": f"{KWA}.guan.gikyode", "Chumburung": f"{KWA}.guan.chumburung",
    "Nawuri": f"{KWA}.guan.nawuri", "Efutu": f"{KWA}.guan.efutu",
    "Larteh": f"{KWA}.guan.larteh", "Guan (not named)": f"{KWA}.guan",
    "Lelemi": f"{KWA}.gtm.lelemi", "Siwu": f"{KWA}.gtm.siwu", "Sekpele": f"{KWA}.gtm.sekpele",
    "Tafi": f"{KWA}.gtm.tafi", "Bowiri": f"{KWA}.gtm.tuwuli", "Adele": f"{KWA}.gtm.adele",
    "Kwa (not named)": KWA,
    "Dagbani": f"{GUR}.dagbani", "Mampruli": f"{GUR}.mampruli", "Gurene": f"{GUR}.farefare",
    "Talni": f"{GUR}.talni", "Nabit": f"{GUR}.nabit", "Kusaal": f"{GUR}.kusaal",
    "Dagaare": f"{GUR}.dagara", "Waali": f"{GUR}.waali", "Birifor": f"{GUR}.birifor",
    "Buli": f"{GUR}.buli", "Hanga": f"{GUR}.hanga", "Safaliba": f"{GUR}.safaliba",
    "Mooré": f"{GUR}.moore",
    "Konkomba": f"{GUR}.konkomba", "Bimoba": f"{GUR}.bimoba", "Ntcham (Basari)": f"{GUR}.ntcham",
    "Gourmanchéma": f"{GUR}.gourmanchema", "Akaselem (Chamba)": f"{GUR}.akaselem",
    "Kabiyè": f"{GUR}.kabiye",
    "Kasem": f"{GUR}.kasem", "Sisaala": f"{GUR}.sisaala", "Tampulma": f"{GUR}.tampulma",
    "Deg": f"{GUR}.deg", "Tem": f"{GUR}.tem", "Chala": f"{GUR}.chala",
    "Gurunsi (not named)": f"{GUR}.gurunsi",
    "Nafaanra": f"{GUR}.nafana", "Lobi": f"{GUR}.lobi", "Gur (not named)": GUR,
    "Bissa": f"{MANDE}.bissa", "Dyula": f"{MANDE}.dyula", "Mande (not named)": MANDE,
    "Hausa": "afroasiatic.chadic.hausa", "Fula": "nigercongo.atlantic.fulah",
    "Zarma": "nilosaharan.songhay.zarma", "English": "indoeuropean.germanic.english",
    "Other African language": "africa_other",
}


def resolve(name):
    return NAMES.get(name)
