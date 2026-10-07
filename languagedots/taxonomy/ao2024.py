"""Angola, RGPH 2024, mother tongue (língua materna) of the population aged 2 and over -> node.
Keyed by INE's printed label, as sources/ao_rgph.py writes it.

INE GROUPED SOME LANGUAGES BEFORE PRINTING (national volume p.54, footnote 5), and each printed
label is drawn as the language it names:
  * Kimbundu (Glottolog kimb1241) includes Mbongala, Songo and Ngoya.
  * Umbundu (umbu1257) includes Mukubale.
  * Cokue (Chokwe/Kioko) (chok1245) includes Lunda.
  * Olunyaneka (Nhaneka) (nyan1305) includes Humbi, Handa, Mucilengue, Sela and Kimbali.
    Glottolog files Humbe and Handa as Nkhumbi dialects, yet INE prints Muhumbi (Nkhumbi,
    nkhu1238) as a column of its own; both are drawn as printed.
  * Nganguela (Glottolog's Nyemba, nyem1238) includes Luchazi, Mbunda and Ukumbi. Drawn as
    Ngangela, under the Chokwe-Lunda group where zm.txt has Luchazi and Mbunda.
  * Oxikwanhama (Kwanhama) includes Herero and Mudimba. Drawn as Kwanyama, its own node beside
    na.txt's Oshiwambo (Kwanyama is Oshiwambo's largest variety; making Oshiwambo a group would
    wash Namibia's out).
  * Ifyoti (Fiote) is the Kongo of Cabinda (Woyo, Vili, Yombe); its own node beside Kongo.

GROUPS AND REMAINDERS, on the narrowest node holding everything filed there (spec §3.2):
  * Khoisan: the !Xun and Khwe of the south-east, two families; on the `khoisan` root, drawn
    washed out as "language not named".
  * Criolo: Portuguese-based creole not specified (Cabo Verde's, Guinea-Bissau's, São Tomé's
    are all Portuguese-based); on `creole.portuguese_based`.
  * Outras línguas: per the footnote, Angola's other national languages (Kisumbe, Kiswahili,
    Mwalabi, Sikabunda, Muko, Lucumai...) and Gestual, sign language. Mostly indigenous, so
    on `africa_other`; the sign-language users inside it cannot be told apart.
  * Malanje prints no foreign-language column, and its `Outras` equals the national volume's
    Malanje other + foreign, so sources/ao_rgph.py labels it apart; on `other`, the only node
    holding both.

THE FOREIGN LANGUAGES. The provincial volumes print one `Línguas estrangeiras` column (bar
Cuanza Sul, Huíla and Namibe, which print the nine); the nine are shared out within each
province by the national volume's provincial mix, `derived` (sources/ao_rgph.py).

NOT DRAWN: Não sabe (147,468 people).
"""
BANTU = "nigercongo.bantu"
NAMES = {
    "Português": "indoeuropean.romance.portuguese",
    "Kimbundu": f"{BANTU}.kimbundu",
    "Umbundu": f"{BANTU}.umbundu",
    "Cokue (Chokwe/Kioko)": f"{BANTU}.chokwe_lunda.chokwe",
    "Kikongo": f"{BANTU}.kongo",
    "Olunyaneka (Nhaneka)": f"{BANTU}.nyaneka_nkhumbi.nyaneka",
    "Nganguela": f"{BANTU}.chokwe_lunda.ngangela",
    "Oxikwanhama (Kwanhama)": f"{BANTU}.kwanyama",
    "Ifyoti (Fiote)": f"{BANTU}.fiote",
    "Muhumbi": f"{BANTU}.nyaneka_nkhumbi.nkhumbi",
    "Luvale": f"{BANTU}.chokwe_lunda.luvale",
    "Khoisan": "khoisan",
    "Mandarim": "sinotibetan.sinitic.mandarin",
    "Inglês": "indoeuropean.germanic.english",
    "Francês": "indoeuropean.romance.french",
    "Espanhol": "indoeuropean.romance.spanish",
    "Alemão": "indoeuropean.germanic.continental.german",
    "Russo": "indoeuropean.slavic.east.russian",
    "Árabe": "afroasiatic.arabic",
    "Lingala": f"{BANTU}.lingala",
    "Criolo": "creole.portuguese_based",
    "Outras línguas": "africa_other",
    "Outras línguas (Malanje: com as línguas estrangeiras)": "other",
    "Não sabe": None,
    "Número de pessoas com 2 ou mais anos": None,
}
EXCLUDED = ("Não sabe", "Número de pessoas com 2 ou mais anos")


def resolve(name):
    return NAMES[name]
