"""Sierra Leone 2015 PHC, main language (P10, "main language NAME speaks") -> node.

Keyed by the rows of sources/sl_census.py's sl.csv: Table 3.22's 15 local languages and
"Other", with its "Foreign language" row split into English, French and Arabic by CLEAR
Global's sample shares (the census form coded the three apart; the report prints one row).

Labels -> languages (Glottolog codes from data/raw/glottolog/languages.csv, matching CLEAR's):
  * Mende (mend1266), Loko (loko1255), Kono (kono1268, Sierra Leone's Kono, a Vai-Kono
    language; Guinea's unrelated Kono is gn.txt's `kono_guinea`), Vai (vaii1241), Koranko =
    Kuranko (kura1250, gn's node), Susu (susu1250), Yalunka (yalu1240): Mande.
  * Madingo: the census's spelling of Mandingo. CLEAR codes it Mandinka (mand1436), and it
    goes on sn's `mandinka` node. Many of Sierra Leone's Mandingo trace to Guinea's Maninka
    rather than Gambia's Mandinka; the census does not say which, and the node is the Manding
    language the sample was coded to.
  * Temne (timn1235), Limba (limb1267), Sherbro (sher1258), Krim (krim1238, a dialect of
    Bom-Kim; a node because the census names it), Kissi (kiss1245), Fullah = Fula (fula1264):
    Atlantic.
  * Krio (krio1253): English-based creole, the existing node.

Remainders:
  * "Other" (5,499): a language neither among the fifteen local ones nor foreign; the census
    does not say which. A bare "other", so `other`.
  * No answer (121,417) is not drawn; it is in `gap`.
"""
AT = "nigercongo.atlantic"
MANDE = "nigercongo.mande"

NAMES = {
    "Mende": f"{MANDE}.mende",
    "Temne": f"{AT}.temne",
    "Krio": "creole.english_based.krio",
    "Limba": f"{AT}.limba",
    "Kono": f"{MANDE}.kono",
    "Koranko": f"{MANDE}.kuranko",
    "Fullah": f"{AT}.fulah",
    "Susu": f"{MANDE}.susu",
    "Kissi": f"{AT}.kissi",
    "Loko": f"{MANDE}.loko",
    "Madingo": f"{MANDE}.mandinka",
    "Sherbro": f"{AT}.sherbro",
    "Yalunka": f"{MANDE}.yalunka",
    "Krim": f"{AT}.krim",
    "Vai": f"{MANDE}.vai",
    "English": "indoeuropean.germanic.english",
    "French": "indoeuropean.romance.french",
    "Arabic": "afroasiatic.arabic",
    "Other": "other",
}


def resolve(name):
    return NAMES.get(name)
