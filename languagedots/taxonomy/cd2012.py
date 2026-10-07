"""DR Congo, Enquete 1-2-3 2005 + 2012: ethnic group ("tribe") of the household head -> node.

Keyed by the survey's own field names, which USCB's Data Dictionary keeps; NOT by USCB's ISO
renamings, several of which are wrong ("Twa" became Plains Bira, "Bale (Londu)" Lendu is right
but "Mbunda" took Angola's code where Kwilu's Mbunda are the Mbuun). Each ethnic group is read
as its own language (AGENT_BRIEF §2, ethnicity only). Places quoted are where the survey's heads
of that group were sampled (sources/cd_e123.py prints them); families checked against Glottolog.

Merges, each the same people under two spellings or clan names of one language:
  * Lulua and the Luba-Kasai clans (Bakwa Kalonji, Bakwa Dishi, Bakwa Mulumba, Bakwanga) all
    speak Tshiluba (Luba-Kasai); the clan names are not speech varieties.
  * Kanioka and Bena-Kanioka ("people of Kanioka"), both 63-80% in Lomami: Kanyok.
  * Leele and Selele (Basilele), both 91-96% in Kasai: the Lele, "Bashilele" being their name
    with the people prefix.
  * Tshokwo and Tshoko: Chokwe; Tshoko is 61% in Kasai (Tshikapa, a Chokwe area) and 17% in
    Kwango, where Chokwe live along the Angolan border.
The Kongo groups are NOT merged: Ndibu, Manyanga, Ntandu, Mbata, Lemfu and the rest are named
speech varieties of Kikongo, so each gets a node under a Kongo group (taxonomy/tree.d/cd.txt).

"Luba" alone is Luba-Kasai (Tshiluba) everywhere, Katanga included. The survey prints "Luba
Shaba" apart, and the Kiluba heartland answered that: Manono, Kamina, Kabongo and Kaniama give
208, 151, 127 and 122 "Luba Shaba" heads against 0, 0, 2 and 18 "Luba". In Katanga "Luba" comes
from Lubumbashi (114), Likasi (76) and Mutshatsha (53), the mining and railway towns where Kasai
Luba settled. A first build that sent Katanga's "Luba" to Luba-Katanga drew it at 7.5 million.

"Hutu (Ruzizi)" is 94% in Nord-Kivu, not the Ruzizi plain: Kinyarwanda, as the Banyarwanda's.
"Twa" have no language of their own (they speak their neighbours'), and 43% are in Kasai, 30% in
Mai-Ndombe and 23% in Equateur, so no one language fits: `africa_other`, with "Other".
"Other" (10.2% of heads) sits on `africa_other`: it is groups the survey did not list, nearly
all Congolese, of every family.
"""
BT = "nigercongo.bantu"
KG = f"{BT}.kongo_dialects"
UB = "nigercongo.ubangian"
CS = "nilosaharan.centralsudanic"
LUBA_KASAI = f"{BT}.luba_kasai"
LUBA_KATANGA = f"{BT}.luba_katanga"

NAMES = {
    # ---- Kongo (H.10): Kikongo speech varieties, each its own node; Yombe a language of its own
    "Yombe": f"{KG}.yombe",
    "Ndibu": f"{KG}.ndibu",
    "Manyanga": f"{KG}.manyanga",
    "Ntandu": f"{KG}.ntandu",
    "Mbata": f"{KG}.mbata",
    "Lemfu": f"{KG}.lemfu",
    "Besi Ngombe": f"{KG}.besingombe",        # 68% Kongo-Central, 30% Kinshasa
    "Bakongo du Sud-Est du fleuve": f"{KG}.kongo_se",
    "Mboma": f"{KG}.mboma",                   # 46% Kongo-Central (Boma); 37% Sud-Ubangi, unexplained
    # ---- Kwilu, Kwango, Mai-Ndombe
    "Yaka": f"{BT}.yaka",
    "Suku": f"{BT}.suku",
    "Pelende": f"{BT}.pelende",               # Glottolog: a Yaka dialect; 95% Kwango
    "Mbala": f"{BT}.mbala",
    "Pende": f"{BT}.pende",
    "Yansi": f"{BT}.yansi",
    "Mbunda": f"{BT}.mbuun",                  # 87% Kwilu: the Mbuun, not Angola's Mbunda
    "Kwese": f"{BT}.kwese",
    "Ngongo": f"{BT}.ngongo",                 # 81% Kwilu: Glottolog's Ngongo language of Kwilu
    "Teke (Tic)": f"{BT}.teke",
    "Sakata": f"{BT}.sakata",
    "Sengele": f"{BT}.sengele",
    "Bomaa": f"{BT}.boma",
    "Tere": f"{BT}.tere",                     # 99% Mai-Ndombe; not in Glottolog under this name
    "Kundu": f"{BT}.kundu",                   # 90% Mai-Ndombe; not in Glottolog under this name
    # ---- Equateur, Tshuapa, Mongala, the Ubangi provinces (Bantu)
    "Mongo s.a.I": f"{BT}.mongo",
    "Konda (Ekonda)": f"{BT}.ekonda",         # Glottolog: a Mongo dialect; printed apart
    "Ntomba": f"{BT}.ntomba",
    "Nunu": f"{BT}.nunu",                     # Bobangi-Nunu of the Equateur river
    "Libinja": f"{BT}.libinza",
    "Mpama": f"{BT}.mpama",
    "Ngombe": f"{BT}.ngombe",
    "Mbuja (Mbudja)": f"{BT}.budza",
    "Bangando": f"{BT}.ngando",               # 92% Tshuapa: Glottolog's Ngando of DR Congo
    "Kutshu (Nkut.)": f"{BT}.nkutu",
    # ---- Kasai, Sankuru, Lomami
    "Lulua": LUBA_KASAI,
    "Bakwa Kalonji": LUBA_KASAI,
    "Bakwa Dishi": LUBA_KASAI,
    "Bakwa Mulumba": LUBA_KASAI,
    "Bakwanga": LUBA_KASAI,
    "Luba Shaba": LUBA_KATANGA,
    "Luba": LUBA_KASAI,                       # everywhere; see the docstring
    "Luntu": f"{BT}.luntu",
    "Kete": f"{BT}.kete",
    "Kanioka": f"{BT}.kanyok",
    "Bena-Kanioka": f"{BT}.kanyok",
    "Songye": f"{BT}.songe",
    "Tetela": f"{BT}.tetela",
    "Kuba (Bushoong)": f"{BT}.bushoong",
    "Leele": f"{BT}.lele",
    "Selele (Basilele)": f"{BT}.lele",
    "Ndengese": f"{BT}.dengese",
    "Tshokwo": f"{BT}.chokwe_lunda.chokwe",
    "Tshoko": f"{BT}.chokwe_lunda.chokwe",
    # ---- Katanga
    "Lunda": f"{BT}.chokwe_lunda.lunda",
    "Ndembe": f"{BT}.chokwe_lunda.ndembu",     # 76% Lualaba: Lunda-Ndembu
    "Bemba": f"{BT}.bemba.bemba",
    "Tabwe": f"{BT}.bemba.tabwa",
    "Zela": f"{BT}.zela",
    "Sanga": f"{BT}.sanga",
    "Kaonde": f"{BT}.luban.kaonde",
    "Hemba": f"{BT}.hemba",
    # ---- Maniema, the Kivus
    "Kusu (Mukusu, Bakusu)": f"{BT}.kusu",
    "Lega": f"{BT}.lega",                     # USCB: Lega-Shabunda
    "Rega": f"{BT}.lega_mwenga",              # USCB: Lega-Mwenga
    "Songola": f"{BT}.songola",
    "Bangu-Bangu": f"{BT}.bangubangu",
    "Benye Nonda": f"{BT}.benye_nonda",       # 95% Maniema; not in Glottolog under this name
    "Ngengele": f"{BT}.ngengele",
    "Mukumu": f"{BT}.komo",                   # Kumu (Komo of DR Congo), 77% Maniema
    "Nande (Mundande)": f"{BT}.nande",
    "Shi (Bashi)": f"{BT}.shi",
    "Havu": f"{BT}.havu",
    "Hunde": f"{BT}.hunde",
    "Fulero": f"{BT}.fuliiru",
    "Vira": f"{BT}.vira",
    "Bembe": f"{BT}.bembe",
    "Nyanga": f"{BT}.nyanga",
    "Rwanda (Kivu)": f"{BT}.kinyarwanda",
    "Hutu (Ruzizi)": f"{BT}.kinyarwanda",
    # ---- Tshopo, the Uele, Ituri (Bantu)
    "Lokele": f"{BT}.lokele",
    "Topoke": f"{BT}.poke",
    "Olombo (Turumbu)": f"{BT}.lombo",
    "Lengola": f"{BT}.lengola",
    "Mbosa": f"{BT}.mbesa",                   # 96% Tshopo; Glottolog's Mbesa
    "Bango": f"{BT}.babango",                 # 98% Tshopo; cf.txt's Babango (bbm)
    "Budu": f"{BT}.budu",
    "Boa (Benge, Bayew)": f"{BT}.bwa",
    "Benja": f"{BT}.benza",                   # 99% Bas-Uele; Glottolog's Benza
    "Nyari": f"{BT}.nyali",                   # 98% Ituri; Nyali is Bantu (D.33)
    # ---- Ubangian
    "Azande (Zande)": f"{UB}.zande.zande",
    "Ngbandi": f"{UB}.ngbandi.ngbandi_north",
    "Mbanja": f"{UB}.banda.mbandja",
    "Mono": f"{UB}.banda.mono",
    "Mayogo": f"{UB}.mundu_baka.mayogo",
    "Mba": f"{UB}.mundu_baka.mba",            # 79% Tshopo
    # 63% Sud-Ubangi, 33% Nord-Ubangi: the Ngbaka of Gemena, Ngbaka Minagende; USCB's code agrees
    "Ngbaka (Gwakamabo)": "nigercongo.gbaya.ngbaka_minagende",
    # ---- Central Sudanic, Nilotic
    "Lugbala": f"{CS}.lugbara",
    "Logo": f"{CS}.logo",
    "Bale (Londu)": f"{CS}.lendu",
    "Mamvu (Mamve)": f"{CS}.mamvu",
    "Makere": f"{CS}.makere",                 # Glottolog: a Mangbetu dialect; printed apart
    "Alur": "nilosaharan.nilotic.alur",
    # ---- no language of their own, and the unnamed
    "Twa": "africa_other",
    "Other": "africa_other",
}


def resolve(label, province=None):
    """`province` is accepted for countries/cd.py's call shape; no label depends on it."""
    return NAMES.get(label)
