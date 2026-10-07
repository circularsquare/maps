"""Côte d'Ivoire RGPH 2021, "langue la plus parlée" by région (Tableau 4.20 of the thematic
report, tome 1) -> node.

Keyed by Tableau 4.20's column heads (sources/ci_rgph.py). One answer per person: the Ivorian
national language they speak most; French is not an answer.

Remainders:
  * "Ensemble des autres langues nationales parlées" (3,850,778 nationally, 18.1%): the other
    82 labels the census coded, printed nationally only (annex 20). Every one is a language of
    Côte d'Ivoire and Niger-Congo (Kwa: Abron, Adjoukrou, Ebrié, Abidji, Abouré, Alladian,
    Appolo...; Kru: Dida, Wobé, Kroumen, Godié, Bakwé...; Mande:
    Mahou, Koyaka, Worodougouka, Toura, Gban...; Gur: Tagbana, Djimini, Niarafolo, Nafana...;
    Atlantic: Peul, Foula), plus "Naturalisé" (9,408, naturalised citizens, language not given)
    and "Autre langue nationale à préciser" (13). So it sits on `nigercongo`, the narrowest node
    holding all of them. Since ask 011 it is no longer drawn: sources/ci_model.py shares each
    région's remainder out into those 82 labels (SHARED below), and countries/ci.py draws that.
  * "Aucune langue nationale parlée" (722,167, 3.4%; 7.5% in Abidjan): Ivorians who speak no
    Ivorian language. French is the obvious candidate, since the question leaves it out, but the
    census does not name a language, so it sits on `other`.
"""
KWA = "nigercongo.kwa"
MANDE = "nigercongo.mande"
GUR = "nigercongo.gur"
KRU = "nigercongo.kru"

NAMES = {
    "Baoulé": f"{KWA}.baoule",
    "Agni": f"{KWA}.anyi",
    "Akyé ou Attié": f"{KWA}.attie",
    "Abbey": f"{KWA}.abbey",
    "Dioula": f"{MANDE}.dyula",
    "Malinké ou Malinka": f"{MANDE}.maninka",
    "Yacouba ou Dan": f"{MANDE}.dan",
    "Gouro": f"{MANDE}.guro",
    "Senoufo": f"{GUR}.senufo",
    "Lobi": f"{GUR}.lobi",
    "Koulango": f"{GUR}.kulango",
    "Bété": f"{KRU}.bete",
    "Guéré": f"{KRU}.guere",
    "Ensemble des autres langues nationales parlées": "nigercongo",
    "Aucune langue nationale parlée": "other",
}

# SHARE-OUT of "Ensemble des autres langues nationales parlées" (ask 011, Anita 2026-10-05; every
# row `modelled`, sources/ci_model.py). Keyed by annex 20's labels, as ci_model.csv prints them.
# Branch: the language's own (Glottolog, conventional levels as above); where Glottolog has no
# entry, the census's ethnic macro-group of most of its speakers (annex 27). Where the two differ
# the language wins: Yaouré and Ngain (Beng) are Mande languages of mostly Akan people; Ahizi
# (Aïzi) is Kru in Glottolog (aizi1248) though its people are counted Akan; Samogho is a Mande
# cluster though its people are counted Gur. The Senufo languages the census prints apart
# (Tagbana, Djimini, Niarafolo, Nafana, Palaka, Tchebara, Fodonon, Koufoulo) sit flat under Gur
# beside "Senoufo", which is a leaf. Conja (48% Gur) and Doma (51% Akan) have no branch any
# source gives, so they are leaves directly under Niger-Congo. Gagou and Gban are two names of
# one language (Glottolog gagu1242 "Gban", also called Gagu) and share a node.
ATL = "nigercongo.atlantic"
SHARED = {
    # Kwa
    "ABIDJI": f"{KWA}.abidji", "ABOURE": f"{KWA}.aboure", "ABRON": f"{KWA}.abron",
    "ADJOUKROU": f"{KWA}.adjoukrou", "ALLADIAN": f"{KWA}.alladian",
    "APPOLO ou N'ZIMA": f"{KWA}.nzema", "AVIKAM ou BRIGNAN": f"{KWA}.avikam",
    "EBRIE": f"{KWA}.ebrie", "EGA": f"{KWA}.ega", "EHOTILE": f"{KWA}.ehotile",
    "ESSOUMA": f"{KWA}.essouma", "KROBOU": f"{KWA}.krobou", "MBATTO ou GOUA": f"{KWA}.mbatto",
    "ANDOH": f"{KWA}.ando", "SOUAMINLIN": f"{KWA}.souaminlin",
    # Kru
    "AHIZI": f"{KRU}.aizi", "BAKWE": f"{KRU}.bakwe", "DIDA": f"{KRU}.dida",
    "GODIE": f"{KRU}.godie", "KODIA": f"{KRU}.kodia", "KOUYA": f"{KRU}.kouya",
    "KROUMEN": f"{KRU}.kroumen", "NEYO": f"{KRU}.neyo", "GNABOUA ou NIABOUA": f"{KRU}.nyabwa",
    "NIEDEBOUA": f"{KRU}.niedeboua", "OUBI": f"{KRU}.oubi", "WANE": f"{KRU}.wane",
    "WOBE": f"{KRU}.wobe", "GUEBIE": f"{KRU}.guebie", "KOTROHOU": f"{KRU}.kotrohou",
    "KOUZIE": f"{KRU}.kouzie", "SOKYIA": f"{KRU}.sokya", "WINNIN": f"{KRU}.winnin",
    # Mande
    "KOYAKA ou KOYARA": f"{MANDE}.koyaka", "MAHOUKA ou MAHOU": f"{MANDE}.mahou",
    "WORODOUGOUKA": f"{MANDE}.worodougou", "BARALAKA": f"{MANDE}.baralaka",
    "FINANGA": f"{MANDE}.finanga", "KARANDJAN": f"{MANDE}.karandjan",
    "NIGBI": f"{MANDE}.nigbi", "ODIENNEKA": f"{MANDE}.odienneka",
    "N'GARADOUGOUKA": f"{MANDE}.ngaradougou", "DJAMALA": f"{MANDE}.djamala",
    "GANDJE": f"{MANDE}.gandje", "KOMARA ou KAMARA": f"{MANDE}.komara",
    "OUADOUGOU": f"{MANDE}.ouadougou", "OUODOUGOU": f"{MANDE}.ouodougou",
    "KORO": f"{MANDE}.koro", "GBIN": f"{MANDE}.gbin", "BAMBARA": f"{MANDE}.bambara",
    "TOURA": f"{MANDE}.toura", "MONA ou MOUAN": f"{MANDE}.mwan", "OUAN": f"{MANDE}.wan",
    "GAGOU": f"{MANDE}.gban", "GBAN": f"{MANDE}.gban", "KLA": f"{MANDE}.kla",
    "SIA": f"{MANDE}.sia", "YOHOURE ou YAOURE": f"{MANDE}.yaoure", "NGAIN": f"{MANDE}.ngain",
    "SAMOGHO": f"{MANDE}.samogho",
    # Gur (Senufo languages flat, see above)
    "TAGBANA": f"{GUR}.tagbana", "DJIMINI": f"{GUR}.djimini", "NIARAFOLO": f"{GUR}.niarafolo",
    "NAFANA": f"{GUR}.nafana", "PALAKA": f"{GUR}.palaka", "TCHEBARA": f"{GUR}.tchebara",
    "FODONON": f"{GUR}.fodonon", "KOUFOULO": f"{GUR}.koufoulo", "GBONZRON": f"{GUR}.gbonzron",
    "MANGORO": f"{GUR}.mangoro", "LOHRON": f"{GUR}.lorhon", "BIRIFOR": f"{GUR}.birifor",
    "DEGHA": f"{GUR}.deg", "KOMONO": f"{GUR}.komono", "GOUIN ou KIRMA": f"{GUR}.cerma",
    "SITI": f"{GUR}.siti",
    # Atlantic: two answers, two nodes, as cf.txt has them
    "PEUL": f"{ATL}.peulh", "FOULA": f"{ATL}.fulah",
    # no branch known
    "CONJA": "nigercongo.conja", "DOMA": "nigercongo.doma",
    # not languages: naturalised citizens (no language given) and an unnamed national language
    "NATURALISE": "other",
    "AUTRE LANGUE NATIONALE A PRECISER": "nigercongo",
}
NAMES.update(SHARED)


def resolve(name):
    return NAMES[name]
