"""Chad, RGPH2 2009, first national language spoken (sources/td_rgph.py, Tableau 5.10).

Labels as INSEED prints them; nodes and Glottolog checks in taxonomy/tree.d/td.txt. Every
printed row is a node of its own. "Autres 1ere langues nationales parlées" is the unnamed
remainder (Annexe 3 lists some 70 languages in it, among them Toubou, Kanouri, Kotoko, the
Hadjaraï languages and Niellim), on africa_other.
"""

C = "afroasiatic.chadic"
CS = "nilosaharan.centralsudanic"
AD = "nigercongo.adamawa"
MB = "nilosaharan.maban"

NAMES = {
    "Arabe local": "afroasiatic.arabic.shuwa",
    "Sara": f"{CS}.sara",
    "Gorane": "nilosaharan.tubu",
    "Kanembou": "nilosaharan.kanembu",
    "Maba/Ouaddaï": f"{MB}.maba",
    "Moundang": f"{AD}.mundang",
    "Mousseye": f"{C}.musey",
    "Boulala": f"{CS}.bilala",
    "Zaghawa/Béri/Bideyat": "nilosaharan.zaghawa",
    "Marba": f"{C}.marba",
    "Massa": f"{C}.massa",
    "Peul/Foulfouldé/Bodoré": "nigercongo.atlantic.fulah",
    "Barma/Baguirmi": f"{CS}.bagirmi",
    "Massalit": f"{MB}.masalit",
    "Mimi": f"{MB}.mimi",
    "Rounga": f"{MB}.runga",
    "Tama": "nilosaharan.tama",
    "Dadjo": "nilosaharan.daju",
    "Moubi": f"{C}.mubi",
    "Mesmé": f"{C}.mesme",
    "Gabri": f"{C}.gabri",
    "Kabalaye": f"{C}.kabalai",
    "Kéra": f"{C}.kera",
    "Kim": f"{AD}.kim",
    "Lamé/Pévé": f"{C}.peve",
    "Lélé": f"{C}.lele",
    "Nangtchéré": f"{C}.nancere",
    "Toupouri": f"{AD}.tupuri",
    "Karo/Kado": f"{C}.karo",
    "Daye": f"{AD}.day",
    "Mboum": f"{AD}.mbum",
    "Sara Kaba": f"{CS}.sara_kaba",
    "Toumak/Ndom": f"{C}.tumak",
    "Autres 1ere langues nationales parlées": "africa_other",
}

NOT_DRAWN = {"Total"}


def resolve(label):
    return NAMES[label]
