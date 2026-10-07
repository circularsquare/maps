"""Central African Republic RGPH03 2003, language commonly spoken (USCB tabulation) -> node.

Keyed by the census's own French field names, which USCB's Data Dictionary preserves; NOT by
USCB's ISO renamings, several of which are wrong (sources/cf_uscb.py lists them). The census
prints its languages in ethnic blocks (Arabe-Peul, Sara, Mboum, Gbaya, Mandja, Banda,
Ngbaka-Bantou, Yakoma-Sango, Zande-Nzakara); where a name is in no reference, the block and the
communes its speakers live in decide, and the comment says so. Shares are of the label's
national total; places are communes and prefectures.

`Fulfuldé` also holds the children under three, who were not asked (sources/cf_uscb.py);
countries/cf.py draws only the estimate of real speakers left after removing them.

Remainders: "Autres langues locales" (56,755: other Central African languages) sits on
`africa_other`, kept apart from "Langues non centrafricaines" (301,955, 54% of it in Bangui,
where French is the obvious candidate but is not named), which sits on `other`.
"""
UB = "nigercongo.ubangian"
BA = f"{UB}.banda"
GB = "nigercongo.gbaya"
AD = "nigercongo.adamawa"
BT = "nigercongo.bantu"
AT = "nigercongo.atlantic"
CS = "nilosaharan.centralsudanic"

NAMES = {
    # ---- the Arabe-Peul block
    "Arabe": "afroasiatic.arabic",
    "Haoussa": "afroasiatic.chadic.hausa",
    "Fulfuldé": f"{AT}.fulah",
    "Fulata": f"{AT}.fulata",          # Fulani of Chad and Sudan origin ("Fellata"); its own answer
    "Peulh": f"{AT}.peulh",            # French for Fula, printed apart from Fulfuldé
    "Mbororo": f"{AT}.mbororo",        # the herders' Fulfulde, printed apart
    # ---- the Sara block (ethnic, not genetic: Runga is Maban)
    "Runga": "nilosaharan.maban.runga",          # 67% in Ouandja commune, Vakaga
    "Binga": f"{CS}.binga",
    "Yulu": f"{CS}.yulu",
    "Yamegi": f"{CS}.gula",            # USCB: Gula (CAR), kcm; Yamegi is a Gula name
    "Barar": f"{CS}.barar",            # not in Glottolog; in the census's Sara block
    "Irri": f"{CS}.birri",             # USCB: Birri; Glottolog's "Irri" is an Edoid dialect of Nigeria
    "Kresh": f"{CS}.kresh",            # 87% in Ouham (Bede, Ouaki), Kresh refugees' area
    "Sara": f"{CS}.sara",
    "Ngama": f"{CS}.ngam",             # 86% in Sido, on the Chad border
    # ---- the Mboum block
    "Talé": f"{AD}.tale",              # Glottolog: a dialect of Kare; printed apart, 91% Ouham-Pende
    "Karé": f"{AD}.kare",
    "Pana": f"{AD}.pana",
    "Mboum": f"{AD}.mbum",
    # ---- the Gbaya block
    "Gbaya": f"{GB}.gbaya",
    # 76% in Bocaranga commune, Ouham-Pende, inside the census's Gbaya block: Gbaya Kara, the
    # Northwest Gbaya of Bocaranga, not the Central Sudanic Kara of Vakaga that USCB took it for
    "Kara": f"{GB}.kara",
    "Bokoto": f"{GB}.bokoto",
    "Lay": f"{GB}.lai",                # Glottolog's Lai, a Northwest Gbaya dialect
    # Buli to Mboundjia: not in Glottolog; filed in the census's Gbaya block, 4,000 people together
    "Buli": f"{GB}.buli",
    "Bokaré": f"{GB}.bokare",
    "Suma": f"{GB}.suma",
    "Gbanou": f"{GB}.gbanu",
    "Budigiri": f"{GB}.budigiri",
    "Gbaguiri": f"{GB}.gbaguiri",
    "Gbadok": f"{GB}.gbadok",
    "Tongo": f"{GB}.tongo",
    "Bouar": f"{GB}.bouar",            # 14 people; a town's name used as a language's
    "Boda": f"{GB}.boda",
    "Mboundjia": f"{GB}.mboundjia",
    # "Kaka" closes the census's Gbaya block, but 97% of its 6,527 are in Mambere-Kadei (85% in
    # Basse-Boumbe, by Gamboula on the Cameroon border), which is where Kako, a Bantu A.90
    # language, is spoken; Gbaya has no "Kaka" variety in Glottolog. Kako.
    "Kaka": f"{BT}.kako",
    # ---- the Mandja block
    "Mandjia": f"{GB}.mandja",         # USCB: Mangbetu (mdj), wrong; Mandja is Glottolog's Manza
    "Ngbaka-Mandjia": f"{GB}.ngbaka_manza",
    "Ngbaka-Minaguendé": f"{GB}.ngbaka_minagende",
    "Ali": f"{GB}.ali",
    "Boffi": f"{GB}.bofi",
    "Séré": f"{UB}.sere",
    # ---- the Banda block
    # "Yaka": 63% in Basse-Kotto and 32% in Ouaka, in the census's Banda block. That is Yakpa, a
    # Banda variety (Glottolog yakp1238), not the Aka of the Lobaye forest that USCB's code axk
    # names (the census prints Aka separately, in the Lobaye).
    "Yaka": f"{BA}.yakpa",
    "Kpatili": f"{UB}.kpatili",
    "Banda": f"{BA}.banda",            # a named answer, so a leaf, not the group (spec §3)
    "Ka": f"{BA}.ka",
    "Ndi": f"{BA}.ndi",
    "Banda-Banda": f"{BA}.banda_banda",
    "Baba": f"{BA}.baba",              # 93 people in Kemo and Ouaka; not Grassfields Baba (USCB)
    "Dakpa": f"{BA}.dakpa",
    "Gbi": f"{BA}.gbi",
    "Yanguéré": f"{BA}.yangere",
    "Langbassi": f"{BA}.langbashe",
    "Langba": f"{BA}.langba",
    "Ngbougou": f"{BA}.ngbugu",
    "Bidjori": f"{BA}.bidjori",        # 40 people; not in Glottolog; in the Banda block
    # ---- the Ngbaka-Bantou block
    "Pomo": f"{BT}.pomo",
    "Bonzio": f"{BT}.bonzio",
    "Bamitaba": f"{BT}.bomitaba",
    "Bobongo": f"{BT}.babango",
    "Kpala": f"{UB}.mundu_baka.kpala",
    "Aka": f"{BT}.aka",                # 99% in Lobaye and Sangha-Mbaere: the BaAka's language
    "Bobangui": f"{BT}.bobangi",
    "Kari": f"{BT}.kari",
    "Bodo": f"{BT}.bodo",
    "Mondjombo": f"{UB}.mundu_baka.monzombo",   # USCB: Mbangala (mxg) of Angola, wrong
    "Ngbaka-Mabo": f"{UB}.mundu_baka.ngbaka_mabo",
    "Gbanziri": f"{UB}.mundu_baka.gbanziri",
    "Boraka": f"{UB}.mundu_baka.buraka",
    "Issongo": f"{BT}.mbati",          # Isongo is the Mbati's own name; USCB: Manza (mzv), wrong
    "Mbimou": f"{BT}.mpiemo",
    # ---- the Yakoma-Sango block
    "Sango Riverain": f"{UB}.ngbandi.sango_riverain",
    "Ngbandjiri": f"{UB}.ngbandi.ngbandjiri",
    "Yakoma": f"{UB}.ngbandi.yakoma",
    "Langue nationale (Sango parlé)": "nigercongo.sango",
    # ---- the Zande-Nzakara block
    "Zandé": f"{UB}.zande.zande",
    "Nzakara": f"{UB}.zande.nzakara",
    # ---- remainders
    "Autres langues locales": "africa_other",
    "Langues non centrafricaines": "other",
}


def resolve(name):
    return NAMES[name]
