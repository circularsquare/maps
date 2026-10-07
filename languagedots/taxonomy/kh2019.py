"""Cambodia, General Population Census 2019, mother tongue (sources/kh_gpcc.py).

Labels as NIS prints them: Table 2.7.1 of the final report for the five non-minority rows, and
Table 2.3 of "Ethnic Minorities in Cambodia" for the 23 minority languages and their Other. The
spellings are NIS's romanisations of the Khmer names; the usual English name is in the node.
Families and branches checked against Glottolog (data/raw/glottolog):

  * Chamic (Austronesian): Cham (Western Cham, west2650), Charai = Jarai (jara1266), Rodae =
    Rade (rade1240).
  * Bahnaric (Austroasiatic): Tumpuon = Tampuan (tamp1251); Prov = Brao (lave1249); Kroeng =
    Kreung (Glottolog's Krung, krun1240, a dialect of Brao), Kavet (kave1238) and Lorn = Lun
    (lunb1239), both also Brao dialects; Punorng = Bunong (Central Mnong, cent1992); Steang =
    Stieng (stie1250); Kroul = Kraol (krao1238); Mael = Mel and Khonh = Khaonh, which Glottolog
    joins as Mel-Khaonh (melk1242, with Khaonh khao1245 a dialect of it).
    Kreung, Kavet and Lun sit beside Brao, not under it: a child would make Brao a group node
    and wash out every Brao dot ("language not named"), and NIS prints them as five answers.
    Khloeng: not in Glottolog under that name. Cambodian sources list the Khleung (Kaleung)
    among the Brao groups of Ratanak Kiri; put beside them under Bahnaric on that account,
    with less certainty than the rest.
  * Katuic: Kuoy = Kuy (kuyy1240).
  * Pearic: Por = Pear (pear1247), Suoy (suoy1242), Sa-ouch = Sa'och (saoc1239).
  * Morn = Mon, on australia's existing `austroasiatic.mon`.
  * Directly under Austroasiatic, branch not verified: Thmoon (Thmon), Ro-ong, Ka-chrook,
    Kanh-Chok. None is in Glottolog under a matching name. All four are small highland answers
    the census files with the minority languages (1,164, 573, 266 and 16 people); Cambodian
    sources describe them as Mon-Khmer, which is all that is claimed here. Ka-chrook may be
    Kaco' (kaco1239, North Bahnaric) and Kanh-Chok another spelling of it; not assumed.
  * Other minority language (Table 2.3's Other, the questionnaire's code 29 "native language
    other", 7,413): the unnamed indigenous remainder, on `seasia_other` with Thailand's, never on
    `other` (AGENT_BRIEF §3).
  * Other foreign language (10,641, derived in the normaliser): Table 2.7.1's Other less the
    minority remainder; the form's foreign codes are French, English, Korean and Japanese. On
    `other`.
  * Vietnam: the Vietnamese language. Chinese: on `sinitic`, as every country files Chinese not
    split by variety (most of Cambodia's older Chinese community is Teochew-speaking, the newer
    one in Preah Sihanouk Mandarin-speaking; the census does not say).
"""

AA = "austroasiatic"
BAH = "austroasiatic.bahnaric"
PEA = "austroasiatic.pearic"
CHM = "austronesian.chamic"

NAMES = {
    # Table 2.7.1
    "Khmer": f"{AA}.khmer",
    "Vietnam": f"{AA}.vietnamese",
    "Chinese": "sinotibetan.sinitic",
    "Lao": "kradai.lao",
    "Thai": "kradai.thai",
    "Other foreign language": "other",
    # Table 2.3
    "Charai": f"{CHM}.jarai",
    "Cham": f"{CHM}.cham",
    "Rodae": f"{CHM}.rade",
    "Tumpuon": f"{BAH}.tampuan",
    "Prov": f"{BAH}.brao",
    "Kroeng": f"{BAH}.kreung",
    "Kavet": f"{BAH}.kavet",
    "Lorn": f"{BAH}.lun",
    "Khloeng": f"{BAH}.khleung",
    "Punorng": f"{BAH}.bunong",
    "Steang": f"{BAH}.stieng",
    "Kroul": f"{BAH}.kraol",
    "Mael": f"{BAH}.mel",
    "Khonh": f"{BAH}.khaonh",
    "Kuoy": f"{AA}.katuic.kuy",
    "Por": f"{PEA}.pear",
    "Suoy": f"{PEA}.suoy",
    "Sa-ouch": f"{PEA}.saoch",
    "Morn": f"{AA}.mon",
    "Thmoon": f"{AA}.thmon",
    "Ro-ong": f"{AA}.ro_ong",
    "Ka-chrook": f"{AA}.ka_chrook",
    "Kanh-Chok": f"{AA}.kanh_chok",
    "Other minority language": "seasia_other",
}

# Table 2.7.1's two rows that Table 2.3 and the derived foreign row replace; and the universe
NOT_DRAWN = {"Total", "Other", "Minority Languages"}


def resolve(label):
    return NAMES[label]
