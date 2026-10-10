"""Chad, MICS6 2019 (sources/td_mics.py): the labels written to data/normalized/td.csv.

Upper case: HC1B "Langue maternelle du chef de ménage" as the file prints it, each answer a node
of its own; Ngambaye and Sar are now nodes beside the census's Sara (siblings, not children, so
the Central African Republic's and the diaspora's Sara stay a leaf; regroup.txt then draws them all under a
Sara group, plain Sara as "Sara (language not given)"). Mixed case: where a head who
answered "other" is sent by their ethnic group (HC2): Tableau 5.10 rows as taxonomy/td2009.py
spells and maps them, and the remainders:
  * "Sara (autres langues sara)": the census's Sara row less Ngambay and Sar (Mbay, Gulay, Gor,
    Laka, Mango, Nangnda, Ngam, Kaba...), on the Sara leaf.
  * "Mouloui/Mousgoum": the Massa group's only member that is neither Massa nor Mousseye
    (Annexe 2; census code 28, no printed row), on cm.txt's Musgum.
  * "Assongori/Mararit": the Tama group's unprinted languages, both Tamaic (Glottolog tama1329),
    on the Nilo-Saharan root, the narrowest node holding both (ca2021 does the same).
  * "Autres langues nationales": the other groups' unprinted languages, on africa_other.
"""

C = "afroasiatic.chadic"
CS = "nilosaharan.centralsudanic"
AD = "nigercongo.adamawa"
MB = "nilosaharan.maban"

NAMES = {
    # HC1B
    "FRANCAIS": "indoeuropean.romance.french",
    "ARABE TCHADIEN": "afroasiatic.arabic.shuwa",
    "SAR": f"{CS}.sar",
    "NGAMBAYE": f"{CS}.ngambay",
    "GORANE": "nilosaharan.tubu",
    "KANEMBOU": "nilosaharan.kanembu",
    "MABA/OUADDAI": f"{MB}.maba",
    "MOUNDANG": f"{AD}.mundang",
    "MASSA": f"{C}.massa",
    "PEUL": "nigercongo.atlantic.fulah",
    "LELE": f"{C}.lele",
    "TOUPOURI": f"{AD}.tupuri",
    "ZAGHAWA": "nilosaharan.zaghawa",
    # "other" by the head's ethnic group: census rows (as td2009.py)
    "Arabe local": "afroasiatic.arabic.shuwa",
    "Gorane": "nilosaharan.tubu",
    "Zaghawa/Béri/Bideyat": "nilosaharan.zaghawa",
    "Peul/Foulfouldé/Bodoré": "nigercongo.atlantic.fulah",
    "Boulala": f"{CS}.bilala",
    "Kéra": f"{C}.kera",
    "Massalit": f"{MB}.masalit",
    "Mimi": f"{MB}.mimi",
    "Marba": f"{C}.marba",
    "Mesmé": f"{C}.mesme",
    "Karo/Kado": f"{C}.karo",
    "Lamé/Pévé": f"{C}.peve",
    "Sara Kaba": f"{CS}.sara_kaba",
    "Daye": f"{AD}.day",
    "Mboum": f"{AD}.mbum",
    "Mousseye": f"{C}.musey",
    "Tama": "nilosaharan.tama",
    "Barma/Baguirmi": f"{CS}.bagirmi",
    "Toumak/Ndom": f"{C}.tumak",
    "Dadjo": "nilosaharan.daju",
    "Moubi": f"{C}.mubi",
    "Gabri": f"{C}.gabri",
    "Kabalaye": f"{C}.kabalai",
    "Nangtchéré": f"{C}.nancere",
    "Rounga": f"{MB}.runga",
    "Kim": f"{AD}.kim",
    # remainders
    "Sara (autres langues sara)": f"{CS}.sara",
    "Mouloui/Mousgoum": f"{C}.musgum",
    "Assongori/Mararit": "nilosaharan",
    "Autres langues nationales": "africa_other",
}

NOT_DRAWN = {"NON REPONSE"}


def resolve(label):
    return NAMES[label]
