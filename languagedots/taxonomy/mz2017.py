"""Mozambique, IV RGPH 2017, mother tongue (língua materna) of the population aged 5 and over -> node.
Keyed by the canonical label sources/mz_rgph.py writes (its SPELLINGS merges INE's spelling
variants: CINHANJA/CINYANJA, ELOMWE/ELOMWUE, OUTRAS/OURAS/Outas).

EACH PROVINCE PRINTS ITS OWN SHORT LIST. Quadro 22 names five to seven languages per province
(Niassa: Emakhuwa, Ciyao, Cinhanja, Elomwe, Xichangana) and files every other Mozambican
language under `Outras línguas moçambicanas`; the national table names nine. Every printed label
is drawn as the language it names:
  * Xichangana is the Mozambican name of the language South Africa's census calls Xitsonga
    (Glottolog tson1249, ISO tso); drawn on za.txt's node, so the two censuses meet at the border
    in one colour.
  * Cinyanja on zm.txt's Nyanja. In Tete it is the speech Malawi and Zambia call Chewa; the
    census prints Nyanja, so Nyanja it is.
  * Cishona (Tete, 48,811) on the Shona leaf; Cindau, Cimanika and Chitewe (Manica, Sofala) on
    leaves of their own beside it, as INE prints them apart.
  * Chibalke is Barwe (Glottolog barw1243), Coti is Koti (koti1238), Lolo/Malolo is Lolo
    (lolo1261), Kimwani is Mwani (mwan1247), Bitonga is Gitonga (gito1238), Xirhonga is
    Ronga (rong1268), Xitshwa (the southern files' spelling) is Xitswa, CICOPI/CHICHOPI is
    Chopi (chop1243), and INE's KISWALHILI is Swahili (Cabo Delgado; the national table
    counts it as foreign, sources/mz_rgph.py NATIONAL_DIFF).

REMAINDERS, on the narrowest node holding everything filed there (spec §3.2):
  * `Outras línguas moçambicanas` (947,455 over the provincial tables, 4.3%): the province's
    other national languages. Every language spoken natively in Mozambique is Bantu (Glottolog: all of the
    country's indigenous languages sit under Narrow Bantu), so `nigercongo.bantu`, drawn washed
    out as "Bantu, language not named". Not `africa_other`, which is for a remainder whose family
    is unknown; this one's family is known. Angola's `Outras línguas` went on `africa_other`
    because sign language was filed inside it; here mute people have their own row.
  * `Outras línguas estrangeiras` (86,124 in the provincial tables): foreign languages, on `other`.

NOT DRAWN: `Desconhecida` (mother tongue not known: 672,976 in the provincial tables, 3.0%;
290,465 of them in Cabo Delgado, 15.6% of that province; the national table prints 407,927, see
sources/mz_rgph.py NATIONAL_DIFF) and `Mudo` (people unable to speak, 4,173), as Zambia's
"unable to speak" is not drawn.
"""
BANTU = "nigercongo.bantu"
NAMES = {
    "Português": "indoeuropean.romance.portuguese",
    "Emakhuwa": f"{BANTU}.makhuwa.emakhuwa",
    "Elomwe": f"{BANTU}.makhuwa.elomwe",
    "Echuwabo": f"{BANTU}.makhuwa.echuwabo",
    "Lolo/Malolo": f"{BANTU}.makhuwa.lolo",
    "Coti": f"{BANTU}.makhuwa.koti",
    "Xichangana": f"{BANTU}.tswa_ronga.tsonga",
    "Xitswa": f"{BANTU}.tswa_ronga.tswa",
    "Xironga": f"{BANTU}.tswa_ronga.ronga",
    "Cicopi": f"{BANTU}.chopi.cicopi",
    "Bitonga": f"{BANTU}.chopi.gitonga",
    "Cinyanja": f"{BANTU}.nyanja_sena.nyanja",
    "Cisena": f"{BANTU}.nyanja_sena.sena",
    "Cinyungwe": f"{BANTU}.nyanja_sena.nyungwe",
    "Chibalke": f"{BANTU}.nyanja_sena.barwe",
    "Cindau": f"{BANTU}.ndau",
    "Chitewe": f"{BANTU}.tewe",
    "Cimanika": f"{BANTU}.manyika",
    "Cishona": f"{BANTU}.shona",
    "Ciyao": f"{BANTU}.yao",
    "Shimakonde": f"{BANTU}.makonde",
    "Kimwani": f"{BANTU}.mwani",
    "Kiswahili": f"{BANTU}.swahili",
    "Outras línguas moçambicanas": BANTU,
    "Outras línguas estrangeiras": "other",
    "Mudo": None,
    "Desconhecida": None,
}
EXCLUDED = ("Mudo", "Desconhecida")


def resolve(name):
    return NAMES[name]
