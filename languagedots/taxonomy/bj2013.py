"""Benin, RGPH-4 2013 primary household language (IPUMS 10% sample via CLEAR Global), read as
first language through Tableau 8's ethnic clusters (sources/bj_census.py) -> node.

Labels are CLEAR Global's language names (its glottocodes checked against
data/raw/glottolog/languages.csv), after sources/bj_census.py's relabelling of three wrong ones
(Lokpa, Agouna, Toli/Seto/Kogbe) and the northern "Defi" code. Classification:
  * Gbe (Kwa, Glottolog atla1278 > ... > Gbe): Fon, Gun, Aja, Gen, Ayizo, Maxi, Weme, Tofin,
    Saxwe, Xwela, Kotafon, Ci, Defi, Agouna, and Toli/Seto/Kogbe (one IPUMS-group label with no
    glottocode). Siblings of Fon and Ewe under Kwa, not children of us.txt's `kwa.gbe` leaf
    (a child would turn it into a group, drawn washed out). "Agu (Ewe)" is CLEAR's code for
    IPUMS's "Ewe" answer: Ewe's leaf.
  * Yoruboid (Volta-Niger as conventionally grouped): Yoruba, and Ede Nago, Ede Cabe, Ede
    Idaca, Ifè, Ede Ije, Manigri-Kambolé, which IPUMS and Glottolog give as languages of
    their own; siblings of Yoruba.
  * Gur: Baatonum (gur.bariba), Yom, Lukpa, Kabiyè, Ditammari, Waama, Biali, Nateni,
    Mbelime, Moba, Gourmanchéma, Mòoré, Senoufo; Miyobe under Gur as most readers know it
    (Glottolog leaves it unclassified within Atlantic-Congo).
  * Anii (anii1245): Ghana-Togo Mountain (Kwa), under gh.txt's `kwa.gtm`.
  * Boko (boko1266) and Boo (booo1257, a Busa dialect in Glottolog, a census answer of its
    own): Mande leaves beside ng.txt's Busa.
  * Dendi (dend1243): Songhay, a leaf beside Zarma.
  * "Defi (northern code, language not identified)": the Gur group node, drawn as "language
    not named" (sources/bj_census.py says why).
"""
G = "nigercongo.gur"
K = "nigercongo.kwa"
V = "nigercongo.voltaniger"

NAMES = {
    # Gbe
    "Fon": f"{K}.fon",
    "Gun": f"{K}.gun",
    "Aja (Benin)": f"{K}.aja",
    "Gen": f"{K}.gen",
    "Ayizo Gbe": f"{K}.ayizo",
    "Maxi Gbe": f"{K}.maxi",
    "Weme Gbe": f"{K}.weme",
    "Tofin Gbe": f"{K}.tofin",
    "Saxwe Gbe": f"{K}.saxwe",
    "Xwela Gbe": f"{K}.xwela",
    "Kotafon Gbe": f"{K}.kotafon",
    "Ci Gbe": f"{K}.ci",
    "Defi Gbe": f"{K}.defi",
    "Agouna (Gbe)": f"{K}.agouna",
    "Toli, Seto and Kogbe (Gbe)": f"{K}.toli",
    "Agu (Ewe)": f"{K}.ewe",
    "Anii": f"{K}.gtm.anii",
    "Akan": f"{K}.akan",
    # Yoruboid
    "Yoruba": f"{V}.yoruba",
    "Ede Nago": f"{V}.ede_nago",
    "Ede Cabe": f"{V}.ede_cabe",
    "Ede Idaca": f"{V}.ede_idaca",
    "Ifè": f"{V}.ife",
    "Ede Ije": f"{V}.ede_ije",
    "Manigri-Kambolé Ede Nago": f"{V}.manigri",
    "Igbo": f"{V}.igbo",
    # Gur
    "Baatonum": f"{G}.bariba",
    "Yom": f"{G}.yom",
    "Lokpa (Lukpa)": f"{G}.lukpa",
    "Kabiyé": f"{G}.kabiye",
    "Miyobe": f"{G}.miyobe",
    "Ditammari": f"{G}.ditammari",
    "Waama": f"{G}.waama",
    "Biali": f"{G}.biali",
    "Nateni": f"{G}.nateni",
    "Mbelime": f"{G}.mbelime",
    "Moba": f"{G}.moba",
    "Gourmanchéma": f"{G}.gourmanchema",
    "Mossi": f"{G}.moore",
    "Syenara Senoufo": f"{G}.senufo",
    "Defi (northern code, language not identified)": G,
    # Mande
    "Boko (Benin)": "nigercongo.mande.boko",
    "Boo": "nigercongo.mande.boo",
    "Bambara": "nigercongo.mande.bambara",
    "Bozo": "nigercongo.mande.bozo",
    "Soninke": "nigercongo.mande.soninke",
    "Dagaari Dioula": "nigercongo.mande.dyula",
    # others
    "Borgu Fulfulde": "nigercongo.atlantic.fulah",
    "Wolof": "nigercongo.atlantic.wolof",
    "Lingala-Bangala": "nigercongo.bantu.lingala",
    "Dendi (Benin)": "nilosaharan.songhay.dendi",
    "Zarma": "nilosaharan.songhay.zarma",
    "Songhay": "nilosaharan.songhay.songhay",
    "Hausa": "afroasiatic.chadic.hausa",
    "Standard Arabic": "afroasiatic.arabic",
    "French": "indoeuropean.romance.french",
    "English": "indoeuropean.germanic.english",
}


def resolve(name):
    return NAMES.get(name)
