"""Nigeria, MICS6 2021 (language of the household head) + Afrobarometer R4-R9 (2008-2022, home
language) -> node.

Keyed by the answers sources/ng_mics.py writes to data/normalized/ng.csv (and ng_afro.py to
ng_afro.csv, the comparison build). MICS's nine named answers are written under the
Afrobarometer's names for the same languages (FULANI as "Fula"; HAUSA, IGBO, YORUBA, KANURI,
IJAW, TIV, IBIBIO, EDO as themselves), so no new key: each was checked to mean the same narrow
language in both (MICS's Ijaw excludes Kalabari and Okrika, its Ibibio Efik and Anaang, its Edo
Esan and Urhobo; sources/ng_mics.py). The rest are the Afrobarometer's coded labels (two
spellings or two names of one language merged in ng_afro.py's CODED) and the languages its
"Other (specify)" free text names (VERBATIM there). The tree, and the Glottolog check of every
family and branch, is taxonomy/tree.d/ng.txt.

Remainders:
  * "Other Nigerian language": a free-text answer naming a language only one respondent gave,
    a place that holds several languages, or nothing identifiable; since 2026-10-09 also the
    part of MICS's "other language" the Afrobarometer has too few respondents to split
    (ng_mics.K_OTHER). `africa_other`. Since 2026-10-19 most of it is named by homeland or by
    migrants' origin (ng_mics.name_unnamed; HOMELAND below); what is left is mainly Oyo's and
    Osun's households from abroad or with no homeland language in the state.
  * "Gwoza": coded by the survey in R5 (7 respondents, Borno); Gwoza is an LGA whose people
    speak several Chadic languages (Glavda, Guduf, Dghwede, Lamang). On `afroasiatic.chadic`,
    the narrowest node holding them all, so drawn as an unnamed Chadic language.
  * "English": the four English-at-home respondents who gave no ethnic group (the rest are
    drawn on their group's language; sources/ng_afro.py, english_by_ethnicity).
"""
VN = "nigercongo.voltaniger"
ED = "nigercongo.voltaniger.edoid"
IJ = "nigercongo.ijoid"
CR = "nigercongo.crossriver"
PL = "nigercongo.plateau"
KA = "nigercongo.kainji"
JU = "nigercongo.jukunoid"
BD = "nigercongo.bantoid"
AD = "nigercongo.adamawa"
CH = "afroasiatic.chadic"

NAMES = {
    "Hausa": f"{CH}.hausa",
    "Yoruba": f"{VN}.yoruba",
    "Igbo": f"{VN}.igbo",
    "Fula": "nigercongo.atlantic.fulah",
    "Kanuri": "nilosaharan.kanuri",
    "English": "indoeuropean.germanic.english",
    "Nigerian Pidgin": "creole.english_based.nigerian_pidgin",
    "Tiv": "nigercongo.tiv",
    "Ekajuk": "nigercongo.ekajuk",
    "Zarma": "nilosaharan.songhay.zarma",
    "Shuwa Arabic": "afroasiatic.arabic.shuwa",
    "Busa": "nigercongo.mande.busa",
    "Bariba": "nigercongo.gur.bariba",
    # Volta-Niger
    "Nupe": f"{VN}.nupe", "Igala": f"{VN}.igala", "Ibaji": f"{VN}.ibaji",
    "Idoma": f"{VN}.idoma", "Agatu": f"{VN}.agatu", "Alago": f"{VN}.alago", "Yala": f"{VN}.yala",
    "Etulo": f"{VN}.etulo", "Igede": f"{VN}.igede", "Yace": f"{VN}.yace",
    "Ebira": f"{VN}.ebira", "Gbagyi": f"{VN}.gbagyi", "Bassa-Nge": f"{VN}.bassa_nge",
    "Ganagana": f"{VN}.ganagana", "Gade": f"{VN}.gade",
    "Ika": f"{VN}.ika", "Ukwuani": f"{VN}.ukwuani", "Ikwere": f"{VN}.ikwere",
    "Ekpeye": f"{VN}.ekpeye", "Ogba": f"{VN}.ogba", "Ohafia": f"{VN}.ohafia",
    "Edo": f"{ED}.edo", "Urhobo": f"{ED}.urhobo", "Isoko": f"{ED}.isoko", "Esan": f"{ED}.esan",
    "Yekhee": f"{ED}.yekhee", "Okpela": f"{ED}.okpela", "Okpe": f"{ED}.okpe",
    "Epie": f"{ED}.epie", "Degema": f"{ED}.degema", "Emai-Iuleha-Ora (Owan)": f"{ED}.emai",
    # Ijoid
    "Ijaw": f"{IJ}.ijaw", "Kalabari": f"{IJ}.kalabari", "Okrika": f"{IJ}.okrika",
    "Nembe": f"{IJ}.nembe", "Ibani": f"{IJ}.ibani",
    # Cross River
    "Ibibio": f"{CR}.ibibio", "Efik": f"{CR}.efik", "Anaang": f"{CR}.anaang",
    "Oron": f"{CR}.oron", "Obolo": f"{CR}.obolo", "Ogoni": f"{CR}.ogoni",
    "Khana": f"{CR}.khana", "Eleme": f"{CR}.eleme", "Abua": f"{CR}.abua",
    "Ogbia": f"{CR}.ogbia", "Yakurr": f"{CR}.yakurr", "Mbembe": f"{CR}.mbembe",
    # Plateau
    "Tarok": f"{PL}.tarok", "Berom": f"{PL}.berom", "Tyap": f"{PL}.tyap", "Jju": f"{PL}.jju",
    "Gyong": f"{PL}.gyong", "Aten": f"{PL}.aten", "Hyam (Jaba)": f"{PL}.hyam",
    "Adara": f"{PL}.adara", "Eggon": f"{PL}.eggon", "Mada": f"{PL}.mada",
    "Ninzo": f"{PL}.ninzo", "Numana": f"{PL}.numana", "Irigwe": f"{PL}.irigwe",
    "Izere": f"{PL}.izere", "Migili": f"{PL}.migili", "Koro": f"{PL}.koro",
    # Kainji
    "Kambari": f"{KA}.kambari", "Bassa": f"{KA}.bassa", "Kurama": f"{KA}.kurama",
    "Laru": f"{KA}.laru", "Lela (Dakarkari)": f"{KA}.lela",
    # Jukunoid
    "Jukun": f"{JU}.jukun", "Kuteb": f"{JU}.kuteb", "Jibu": f"{JU}.jibu",
    "Tigon": f"{JU}.tigon", "Etkywan (Ichen)": f"{JU}.etkywan",
    # Bantoid
    "Ejagham": f"{BD}.ejagham", "Jarawa": f"{BD}.jarawa", "Mambila": f"{BD}.mambila",
    "Ndoola": f"{BD}.ndoola", "Bekwarra": f"{BD}.bekwarra", "Bette-Bendi": f"{BD}.bette_bendi",
    # Adamawa
    "Mumuye": f"{AD}.mumuye", "Chamba": f"{AD}.chamba", "Dza (Jenjo)": f"{AD}.dza",
    "Yungur": f"{AD}.yungur", "Waja": f"{AD}.waja", "Tula": f"{AD}.tula",
    # Chadic
    "Ngas": f"{CH}.ngas", "Tangale": f"{CH}.tangale", "Bura-Pabir": f"{CH}.bura_pabir",
    "Marghi": f"{CH}.marghi", "Karekare": f"{CH}.karekare", "Bade": f"{CH}.bade",
    "Kamwe": f"{CH}.kamwe", "Mwaghavul": f"{CH}.mwaghavul", "Gwandara": f"{CH}.gwandara",
    "Tera": f"{CH}.tera", "Bole": f"{CH}.bole", "Bachama": f"{CH}.bachama",
    "Zaar": f"{CH}.zaar", "Goemai": f"{CH}.goemai", "Kibaku (Chibok)": f"{CH}.kibaku",
    "Huba (Kilba)": f"{CH}.huba", "Glavda": f"{CH}.glavda", "Piapung": f"{CH}.piapung",
    # Afrobarometer R3 (2005) and R10 (2024) answers, added 2026-10-19
    "Izi-Ezaa-Ikwo-Mgbo": f"{VN}.izi_ezaa", "Ukpet-Ehom": f"{CR}.ukpet_ehom",
    "Bokyi": f"{BD}.bokyi", "Itsekiri": f"{VN}.itsekiri", "Kupa": f"{VN}.kupa",
    "Nyankpa (Yeskwa)": f"{PL}.nyankpa", "Gun": "nigercongo.kwa.gun",
    # remainders
    "Gwoza": CH,
    "Other Nigerian language": "africa_other",
}

# Languages no survey names, drawn on the part of MICS's "other language" the Afrobarometer
# cannot split, by their homelands (sources/ng_homeland.py; sources/ng.md §8). Keyed by
# Glottolog's name, which is how sources/ng_mics.py writes them to ng.csv.
# >>> generated by sources/ng_homeland.py, do not edit by hand
HOMELAND = {
    'Abanyom': 'nigercongo.bantoid.abanyom',   # aban1242
    'Abon': 'nigercongo.bantoid.abon',   # abon1238
    'Abureni': 'nigercongo.crossriver.abureni',   # abur1244
    'Afade': 'afroasiatic.chadic.afade',   # afad1236
    'Agwagwune': 'nigercongo.crossriver.agwagwune',   # agwa1247
    'Ajiri': 'nigercongo.plateau.ajiri',   # eloy1241
    'Akpa': 'nigercongo.voltaniger.akpa',   # akpa1238
    'Akpes': 'nigercongo.voltaniger.akpes',   # akpe1248
    'Akuku': 'nigercongo.voltaniger.edoid.akuku',   # akuk1235
    'Akum': 'nigercongo.plateau.akum',   # akum1238
    'Akye': 'nigercongo.plateau.akye',   # akee1238
    'Alege': 'nigercongo.bantoid.alege',   # aleg1238
    'Ambo': 'nigercongo.bantoid.ambo',   # ambo1249
    'Amo': 'nigercongo.kainji.amo',   # amoo1242
    'Aninka': 'nigercongo.plateau.aninka',   # anin1242
    'Arum': 'nigercongo.plateau.arum',   # alum1246
    'Ashe': 'nigercongo.plateau.ashe',   # ashe1269
    'Asu (Nigeria)': 'nigercongo.voltaniger.asu_nigeria',   # asun1235
    'Atsam': 'nigercongo.kainji.atsam',   # atsa1241
    'Awak': 'nigercongo.adamawa.awak',   # awak1250
    'Ayere': 'nigercongo.voltaniger.ayere',   # ayer1245
    'Ayu': 'nigercongo.plateau.ayu',   # ayuu1242
    'Baa': 'nigercongo.adamawa.baa',   # kwaa1262
    'Baan': 'nigercongo.crossriver.baan',   # baan1241
    'Bada (Nigeria)': 'nigercongo.bantoid.bada_nigeria',   # bada1258
    'Bali (Nigeria)': 'nigercongo.adamawa.bali_nigeria',   # bali1245
    'Bangwinji': 'nigercongo.adamawa.bangwinji',   # bang1348
    'Basa-Gurmana': 'nigercongo.kainji.basa_gurmana',   # basa1281
    'Bata': 'afroasiatic.chadic.bata',   # bata1314
    'Batu': 'nigercongo.bantoid.batu',   # batu1255
    'Bauchi': 'nigercongo.kainji.bauchi',   # bauc1238
    'Begbere-Ejar': 'nigercongo.plateau.begbere_ejar',   # begb1241
    'Belneng': 'afroasiatic.chadic.belneng',   # beln1234
    'Bijim': 'nigercongo.plateau.bijim',   # biji1246
    'Bile': 'nigercongo.bantoid.bile',   # bile1244
    'Bina (Nigeria)': 'nigercongo.kainji.bina_nigeria',   # bina1270
    'Bitare': 'nigercongo.bantoid.bitare',   # bita1251
    'Bo-Rukul': 'nigercongo.plateau.bo_rukul',   # boru1244
    'Boga': 'afroasiatic.chadic.boga',   # boga1251
    'Boghom': 'afroasiatic.chadic.boghom',   # bogh1241
    'Boko (Benin)': 'nigercongo.mande.boko_benin',   # boko1266
    'Bu': 'nigercongo.plateau.bu',   # buuu1244
    'Bu (Zaranda)': 'afroasiatic.chadic.bu_zaranda',   # zara1252
    'Buduma': 'afroasiatic.chadic.buduma',   # budu1265
    'Bukwen': 'nigercongo.bantoid.bukwen',   # bukw1238
    'Buli (Bauchi State)': 'afroasiatic.chadic.buli_bauchi_state',   # buli1260
    'Bumaji': 'nigercongo.bantoid.bumaji',   # buma1244
    'Burak': 'nigercongo.adamawa.burak',   # bura1271
    'Buru-Angwe': 'nigercongo.bantoid.buru_angwe',   # buru1299
    'Cakfem-Mushere': 'afroasiatic.chadic.cakfem_mushere',   # cakf1236
    'Cara': 'nigercongo.plateau.cara',   # cara1270
    'Cen': 'nigercongo.plateau.cen',   # cenn1238
    'Chaari': 'afroasiatic.chadic.chaari',   # dans1239
    'Che': 'nigercongo.plateau.che',   # chee1238
    'Cicipu': 'nigercongo.kainji.cicipu',   # cici1237
    'Cineni': 'afroasiatic.chadic.cineni',   # cine1238
    'Ciwogai': 'afroasiatic.chadic.ciwogai',   # ciwo1236
    'Como Karim': 'nigercongo.jukunoid.como_karim',   # como1258
    'Daba': 'afroasiatic.chadic.daba',   # nucl1683
    'Dadiya': 'nigercongo.adamawa.dadiya',   # dadi1249
    'Dass': 'afroasiatic.chadic.dass',   # dass1243
    'Deko-Dusanga': 'nigercongo.crossriver.deko_dusanga',   # doko1243
    'Dendi (Benin)': 'nilosaharan.songhay.dendi_benin',   # dend1243
    'Deno': 'afroasiatic.chadic.deno',   # deno1239
    'Dera (Nigeria)': 'afroasiatic.chadic.dera_nigeria',   # dera1248
    'Dghwede': 'afroasiatic.chadic.dghwede',   # dghw1239
    'Dibo': 'nigercongo.voltaniger.dibo',   # dibo1247
    'Dijim-Bwilim': 'nigercongo.adamawa.dijim_bwilim',   # diji1241
    'Dir-Nyamzak-Mbarimi': 'afroasiatic.chadic.dir_nyamzak_mbarimi',   # lund1276
    'Diri': 'afroasiatic.chadic.diri',   # diri1259
    'Dong': 'nigercongo.bantoid.dong',   # dong1293
    'Duguri': 'nigercongo.bantoid.duguri',   # dugu1249
    'Duhwa': 'afroasiatic.chadic.duhwa',   # duhw1236
    'Duwai': 'afroasiatic.chadic.duwai',   # duwa1244
    'Dyarim': 'afroasiatic.chadic.dyarim',   # dyar1234
    'Dzodinka': 'nigercongo.bantoid.grassfields.dzodinka',   # dzod1238
    'Dũya': 'nigercongo.plateau.duya',   # idun1241
    'Eastern Acipa': 'nigercongo.kainji.eastern_acipa',   # east2403
    'Ebughu': 'nigercongo.crossriver.ebughu',   # ebug1241
    'Efai': 'nigercongo.crossriver.efai',   # efai1241
    'Efutop': 'nigercongo.bantoid.efutop',   # efut1242
    'Egene': 'nigercongo.voltaniger.edoid.egene',   # enge1239
    'Ehueun': 'nigercongo.voltaniger.edoid.ehueun',   # ehue1238
    'Ejuele': 'nigercongo.plateau.ejuele',   # doka1242
    'Eki': 'nigercongo.crossriver.eki',   # ekii1241
    'Ekit': 'nigercongo.crossriver.ekit',   # ekit1246
    'Enwan (Akwa Ibom State)': 'nigercongo.crossriver.enwan_akwa_ibom_state',   # enwa1245
    'Enwan (Edo State)': 'nigercongo.voltaniger.edoid.enwan_edo_state',   # enwa1246
    'Eruwa': 'nigercongo.voltaniger.edoid.eruwa',   # eruw1238
    'Esimbi': 'nigercongo.bantoid.esimbi',   # esim1238
    'Etebi': 'nigercongo.crossriver.etebi',   # eteb1238
    'Eten': 'nigercongo.plateau.eten',   # eten1239
    'Evant': 'nigercongo.bantoid.evant',   # evan1238
    'Fali': 'afroasiatic.chadic.fali',   # fali1285
    'Fam': 'nigercongo.bantoid.fam',   # famm1241
    'Firan': 'nigercongo.plateau.firan',   # fira1238
    'Fungwa': 'nigercongo.kainji.fungwa',   # fung1245
    'Fyam': 'nigercongo.plateau.fyam',   # fyam1238
    'Fyer': 'afroasiatic.chadic.fyer',   # fyer1241
    "Ga'anda": 'afroasiatic.chadic.ga_anda',   # gaan1243
    'Gaa': 'nigercongo.bantoid.gaa',   # gaaa1245
    'Galambu': 'afroasiatic.chadic.galambu',   # gala1264
    'Gbiri-Niragu': 'nigercongo.kainji.gbiri_niragu',   # gbir1241
    'Gengle-Kugama': 'nigercongo.adamawa.gengle_kugama',   # kuga1239
    'Gera': 'afroasiatic.chadic.gera',   # gera1246
    'Geruma': 'afroasiatic.chadic.geruma',   # geru1240
    'Ghotuo': 'nigercongo.voltaniger.edoid.ghotuo',   # ghot1243
    'Giiwo': 'afroasiatic.chadic.giiwo',   # giiw1236
    'Gude': 'afroasiatic.chadic.gude',   # gude1246
    'Gudu': 'afroasiatic.chadic.gudu',   # gudu1250
    'Guduf-Gava': 'afroasiatic.chadic.guduf_gava',   # gudu1252
    'Gupa': 'nigercongo.voltaniger.gupa',   # gupa1248
    'Gurmana': 'nigercongo.kainji.gurmana',   # gurm1246
    'Guruntum-Mbaaru': 'afroasiatic.chadic.guruntum_mbaaru',   # guru1271
    'Gvoko': 'afroasiatic.chadic.gvoko',   # gvok1239
    'Gyaazi': 'afroasiatic.chadic.gyaazi',   # nucl1692
    'Gyem': 'nigercongo.kainji.gyem',   # gyem1238
    'Hasha': 'nigercongo.plateau.hasha',   # hash1238
    'Hdi': 'afroasiatic.chadic.hdi',   # hdii1240
    'Horom': 'nigercongo.plateau.horom',   # horo1245
    'Hun-Saare': 'nigercongo.kainji.hun_saare',   # huns1239
    'Hungworo': 'nigercongo.kainji.hungworo',   # hung1276
    'Hwana': 'afroasiatic.chadic.hwana',   # hwan1240
    'Hya': 'afroasiatic.chadic.hya',   # hyaa1239
    'Hõne': 'nigercongo.jukunoid.hone',   # hone1235
    'Ibino': 'nigercongo.crossriver.ibino',   # ibin1241
    'Ibuoro': 'nigercongo.crossriver.ibuoro',   # ibuo1242
    'Iceve-Maci': 'nigercongo.bantoid.iceve_maci',   # icev1238
    'Idere': 'nigercongo.crossriver.idere',   # ider1238
    'Idon': 'nigercongo.plateau.idon',   # idon1238
    'Iguta': 'nigercongo.kainji.iguta',   # igut1238
    'Igwe': 'nigercongo.voltaniger.edoid.igwe',   # igwe1238
    'Ikhin-Arokho': 'nigercongo.voltaniger.edoid.ikhin_arokho',   # ikhi1234
    'Iko': 'nigercongo.crossriver.iko',   # ikoo1245
    'Ikpeshi': 'nigercongo.voltaniger.edoid.ikpeshi',   # ikpe1243
    'Iku-Gora-Ankwa': 'nigercongo.plateau.iku_gora_ankwa',   # ikug1238
    'Ikulu': 'nigercongo.plateau.ikulu',   # ikul1238
    'Ilue': 'nigercongo.crossriver.ilue',   # ilue1241
    'Ito': 'nigercongo.crossriver.ito',   # itoo1242
    'Itu Mbon Uzo': 'nigercongo.crossriver.itu_mbon_uzo',   # itum1245
    'Iyayu': 'nigercongo.voltaniger.edoid.iyayu',   # iyay1238
    'Iyive': 'nigercongo.bantoid.iyive',   # iyiv1238
    'Iyongiyong': 'nigercongo.crossriver.iyongiyong',   # bakp1238
    'Izora': 'nigercongo.kainji.izora',   # izor1238
    'Jakattoe': 'afroasiatic.chadic.jakattoe',   # jort1240
    'Janji': 'nigercongo.kainji.janji',   # janj1240
    'Jara': 'afroasiatic.chadic.jara',   # jara1274
    'Jarawa (Nigeria)': 'nigercongo.bantoid.jarawa_nigeria',   # jara1263
    'Jere': 'nigercongo.kainji.jere',   # jere1244
    'Jiba': 'nigercongo.jukunoid.jiba',   # jiba1237
    'Jilbe': 'afroasiatic.chadic.jilbe',   # jilb1238
    'Jimi (Nigeria)': 'afroasiatic.chadic.jimi_nigeria',   # jimi1255
    'Jirim': 'nigercongo.bantoid.jirim',   # diri1253
    'Ju': 'afroasiatic.chadic.ju',   # juuu1243
    'Kaan': 'nigercongo.adamawa.kaan',   # kaan1247
    'Kadung': 'nigercongo.plateau.kadung',   # vagh1248
    'Kakanda': 'nigercongo.voltaniger.kakanda',   # kaka1264
    'Kam': 'nigercongo.adamawa.kam',   # kamm1249
    'Kamantan': 'nigercongo.plateau.kamantan',   # kama1358
    'Kami (Nigeria)': 'nigercongo.voltaniger.kami_nigeria',   # kami1258
    'Kamo': 'nigercongo.adamawa.kamo',   # kamo1254
    'Kaningdon-Nindem': 'nigercongo.plateau.kaningdon_nindem',   # kani1277
    'Kanufi': 'nigercongo.plateau.kanufi',   # kanu1277
    'Kapya': 'nigercongo.plateau.kapya',   # kapy1238
    'Kariya': 'afroasiatic.chadic.kariya',   # kari1316
    'Kholok': 'afroasiatic.chadic.kholok',   # khol1240
    'Kinuku': 'nigercongo.kainji.kinuku',   # kinu1239
    'Kiong': 'nigercongo.crossriver.kiong',   # kion1242
    'Kir-Balar': 'afroasiatic.chadic.kir_balar',   # kirb1236
    'Kirya-Konzel': 'afroasiatic.chadic.kirya_konzel',   # kiry1234
    'Kizamani': 'nigercongo.kainji.kizamani',   # kiza1239
    'Koenoem': 'afroasiatic.chadic.koenoem',   # koen1239
    'Kohumono': 'nigercongo.crossriver.kohumono',   # kohu1244
    'Koma Ndera': 'nigercongo.adamawa.koma_ndera',   # koma1266
    'Kono (Nigeria)': 'nigercongo.kainji.kono_nigeria',   # kono1264
    'Korop': 'nigercongo.crossriver.korop',   # koro1304
    'Kpan': 'nigercongo.jukunoid.kpan',   # kpan1246
    'Kpasam': 'nigercongo.adamawa.kpasam',   # kpas1242
    'Kugbo': 'nigercongo.crossriver.kugbo',   # kugb1241
    'Kukele': 'nigercongo.crossriver.kukele',   # kuke1242
    'Kulere': 'afroasiatic.chadic.kulere',   # kule1247
    'Kulung (Nigeria)': 'nigercongo.bantoid.kulung_nigeria',   # kulu1255
    'Kumba': 'nigercongo.adamawa.kumba',   # kumb1238
    'Kushi': 'afroasiatic.chadic.kushi',   # kush1236
    'Kutto': 'afroasiatic.chadic.kutto',   # kutt1236
    'Kuturmi': 'nigercongo.plateau.kuturmi',   # kutu1262
    'Kwaami': 'afroasiatic.chadic.kwaami',   # kwaa1269
    'Kyak': 'nigercongo.adamawa.kyak',   # kyak1243
    'Kyenga': 'nigercongo.mande.kyenga',   # kyen1242
    'Kyoli': 'nigercongo.plateau.kyoli',   # cori1240
    'Labir': 'nigercongo.bantoid.labir',   # labi1245
    'Lala-Roba': 'nigercongo.adamawa.lala_roba',   # lala1261
    'Lamang': 'afroasiatic.chadic.lamang',   # lama1288
    'Lame': 'nigercongo.bantoid.lame',   # lame1257
    'Lamja-Dengsa-Tola': 'nigercongo.bantoid.lamja_dengsa_tola',   # lamj1245
    "Lamnso'": 'nigercongo.bantoid.grassfields.lamnso',   # lamn1239
    'Leelau': 'nigercongo.adamawa.leelau',   # leel1242
    'Lemoro': 'nigercongo.kainji.lemoro',   # lemo1242
    'Lenyima': 'nigercongo.crossriver.lenyima',   # leny1238
    'Lere': 'nigercongo.kainji.lere',   # lere1241
    'Leyigha': 'nigercongo.crossriver.leyigha',   # leyi1238
    'Limbum': 'nigercongo.bantoid.grassfields.limbum',   # limb1268
    'Longuda': 'nigercongo.adamawa.longuda',   # long1389
    'Loo': 'nigercongo.adamawa.loo',   # looo1238
    'Lopa': 'nigercongo.kainji.lopa',   # lopa1238
    'Lubila': 'nigercongo.crossriver.lubila',   # lubi1238
    'Lufu': 'nigercongo.plateau.lufu',   # lufu1237
    'Maaka': 'afroasiatic.chadic.maaka',   # maak1236
    'Mafa': 'afroasiatic.chadic.mafa',   # mafa1239
    'Mak (Nigeria)': 'nigercongo.adamawa.mak_nigeria',   # makn1235
    'Mama': 'nigercongo.bantoid.mama',   # mama1272
    'Mangas': 'afroasiatic.chadic.mangas',   # mang1416
    'Marghi South': 'afroasiatic.chadic.marghi_south',   # marg1266
    'Mashi (Nigeria)': 'nigercongo.bantoid.mashi_nigeria',   # mash1269
    'Mbay': 'nilosaharan.centralsudanic.mbay',   # mbay1241
    'Mbe': 'nigercongo.bantoid.mbe',   # mbee1249
    'Mboi': 'nigercongo.adamawa.mboi',   # mboi1246
    'Mbula-Bwazza': 'nigercongo.bantoid.mbula_bwazza',   # mbul1260
    'Mburku': 'afroasiatic.chadic.mburku',   # mbur1239
    'Mindat': 'afroasiatic.chadic.mindat',   # mund1334
    'Mingang Doso': 'nigercongo.adamawa.mingang_doso',   # ming1254
    'Miship': 'afroasiatic.chadic.miship',   # mish1244
    'Miya': 'afroasiatic.chadic.miya',   # miya1266
    'Mom Jango': 'nigercongo.adamawa.mom_jango',   # momj1237
    'Montol': 'afroasiatic.chadic.montol',   # mont1280
    'Moo': 'nigercongo.adamawa.moo',   # mooo1239
    'Mvano': 'nigercongo.bantoid.mvano',   # mvan1238
    'Mághdì': 'nigercongo.adamawa.maghdi',   # magh1238
    'Nde-Nsele-Nta': 'nigercongo.bantoid.nde_nsele_nta',   # nden1250
    'Ndoe': 'nigercongo.bantoid.ndoe',   # ndoe1238
    'Ndunda': 'nigercongo.bantoid.ndunda',   # ndun1251
    'Ndunic': 'nigercongo.plateau.ndunic',   # ahwa1235
    'Ngamo': 'afroasiatic.chadic.ngamo',   # ngam1282
    'Nggwahyi': 'afroasiatic.chadic.nggwahyi',   # nggw1242
    'Ngizim': 'afroasiatic.chadic.ngizim',   # ngiz1242
    'Ngwaba': 'afroasiatic.chadic.ngwaba',   # ngwa1251
    'Nkari': 'nigercongo.crossriver.nkari',   # nkar1238
    'Nkem-Nkum': 'nigercongo.bantoid.nkem_nkum',   # nkem1242
    'Nkoroo': 'nigercongo.ijoid.nkoroo',   # nkor1239
    'Nkukoli': 'nigercongo.crossriver.nkukoli',   # nkuk1238
    'Nnam': 'nigercongo.bantoid.nnam',   # nnam1238
    'Nteng': 'afroasiatic.chadic.nteng',   # nucl1698
    'Nungu': 'nigercongo.plateau.nungu',   # nung1292
    'Nyam': 'afroasiatic.chadic.nyam',   # nyam1285
    'Nyong': 'nigercongo.adamawa.nyong',   # nyon1241
    'Nzanyi': 'afroasiatic.chadic.nzanyi',   # nzan1240
    'Obanliku': 'nigercongo.bantoid.obanliku',   # oban1244
    'Obulom-Ochichi': 'nigercongo.crossriver.obulom_ochichi',   # obul1239
    'Odual': 'nigercongo.crossriver.odual',   # odua1238
    'Ogbogolo': 'nigercongo.crossriver.ogbogolo',   # ogbo1242
    'Ogbronuagum': 'nigercongo.crossriver.ogbronuagum',   # ogbr1243
    'Oko-Eni-Osayen': 'nigercongo.oko_eni_osayen',   # okoe1238
    'Okobo': 'nigercongo.crossriver.okobo',   # okob1241
    'Okodia': 'nigercongo.ijoid.okodia',   # okod1238
    'Okpamheri': 'nigercongo.voltaniger.edoid.okpamheri',   # okpa1238
    'Okpe (Northwestern Edo)': 'nigercongo.voltaniger.edoid.okpe_northwestern_edo',   # okpe1249
    'Oloma': 'nigercongo.voltaniger.edoid.oloma',   # olom1241
    'Olulumo-Ikom': 'nigercongo.crossriver.olulumo_ikom',   # olul1245
    'Oring': 'nigercongo.crossriver.oring',   # orin1239
    'Oruma': 'nigercongo.ijoid.oruma',   # orum1241
    'Ososo': 'nigercongo.voltaniger.edoid.ososo',   # osos1238
    'Otank': 'nigercongo.bantoid.otank',   # otan1238
    "Pa'a": 'afroasiatic.chadic.pa_a',   # paaa1242
    'Pan': 'afroasiatic.chadic.pan',   # kofy1242
    'Pana (Central African Republic)': 'nigercongo.adamawa.pana_central_african_republic',   # pana1293
    'Panawa': 'nigercongo.kainji.panawa',   # pana1298
    'Peere': 'nigercongo.adamawa.peere',   # peer1241
    'Pero': 'afroasiatic.chadic.pero',   # pero1241
    'Pesse': 'afroasiatic.chadic.pesse',   # polc1243
    'Piti': 'nigercongo.kainji.piti',   # piti1243
    'Piya-Kwonci': 'afroasiatic.chadic.piya_kwonci',   # piya1245
    'Pongu': 'nigercongo.kainji.pongu',   # pong1250
    'Psikye': 'afroasiatic.chadic.psikye',   # psik1239
    'Pye': 'nigercongo.plateau.pye',   # peee1238
    'Reshe': 'nigercongo.kainji.reshe',   # resh1242
    'Rishiwa': 'nigercongo.kainji.rishiwa',   # rish1234
    'Rjili': 'nigercongo.plateau.rjili',   # tanj1247
    'Robambami': 'nigercongo.crossriver.robambami',   # agoi1246
    'Ron': 'afroasiatic.chadic.ron',   # ronn1241
    'Sanga (Nigeria)': 'nigercongo.kainji.sanga_nigeria',   # sang1329
    'Sasaru': 'nigercongo.voltaniger.edoid.sasaru',   # sasa1247
    'Shall-Zwall': 'nigercongo.plateau.shall_zwall',   # shal1242
    'Shama-Sambuga': 'nigercongo.kainji.shama_sambuga',   # sham1278
    'Shanga': 'nigercongo.mande.shanga',   # shan1282
    'Shiki': 'nigercongo.bantoid.shiki',   # shik1242
    'Shoo-Minda-Nye': 'nigercongo.jukunoid.shoo_minda_nye',   # shoo1247
    'Siri': 'afroasiatic.chadic.siri',   # siri1278
    'Sukur': 'afroasiatic.chadic.sukur',   # suku1272
    'Sur': 'nigercongo.plateau.sur',   # surr1238
    'Sya': 'afroasiatic.chadic.sya',   # shaa1247
    'Tal': 'afroasiatic.chadic.tal',   # tall1250
    'Tala': 'afroasiatic.chadic.tala',   # tala1295
    'Tambas': 'afroasiatic.chadic.tambas',   # tamb1267
    'Tedaga': 'nilosaharan.tedaga',   # teda1241
    'Teme': 'nigercongo.adamawa.teme',   # teme1252
    'Tha': 'nigercongo.adamawa.tha',   # thaa1239
    'Tita': 'nigercongo.tita',   # tita1240
    'Tiyaa': 'nigercongo.plateau.tiyaa',   # yaaa1242
    'Toro': 'nigercongo.plateau.toro',   # toro1249
    'Tso': 'nigercongo.adamawa.tso',   # tsoo1241
    'Tsuvadi': 'nigercongo.kainji.tsuvadi',   # tsuv1239
    'Tunzu': 'nigercongo.kainji.tunzu',   # tunz1235
    'Ubaghara': 'nigercongo.crossriver.ubaghara',   # ubag1245
    'Ubang': 'nigercongo.bantoid.ubang',   # uban1243
    'Uda': 'nigercongo.crossriver.uda',   # udaa1238
    'Uhami': 'nigercongo.voltaniger.edoid.uhami',   # uham1238
    'Ukaan': 'nigercongo.ukaan',   # ukaa1243
    'Ukpe-Bayobiri': 'nigercongo.bantoid.ukpe_bayobiri',   # ukpe1247
    'Ukue': 'nigercongo.voltaniger.edoid.ukue',   # ukue1238
    'Ukwa': 'nigercongo.crossriver.ukwa',   # ukwa1238
    'Ulukwumi': 'nigercongo.voltaniger.ulukwumi',   # uluk1257
    'Umon': 'nigercongo.crossriver.umon',   # umon1238
    'Uneme': 'nigercongo.voltaniger.edoid.uneme',   # unem1238
    'Usaghade': 'nigercongo.crossriver.usaghade',   # usag1244
    'Ut-Main': 'nigercongo.kainji.ut_main',   # kagf1238
    'Utugwang-Irungene-Afrike': 'nigercongo.bantoid.utugwang_irungene_afrike',   # putu1241
    'Uvbie': 'nigercongo.voltaniger.edoid.uvbie',   # uvbi1238
    'Vemgo-Mabas': 'afroasiatic.chadic.vemgo_mabas',   # vemg1240
    'Vori': 'nigercongo.kainji.vori',   # suru1258
    'Vute': 'nigercongo.bantoid.vute',   # vute1244
    'Waka': 'nigercongo.adamawa.waka',   # waka1275
    'Wandala': 'afroasiatic.chadic.wandala',   # wand1278
    'Wannu': 'nigercongo.jukunoid.wannu',   # wann1241
    'Warji-Gala': 'afroasiatic.chadic.warji_gala',   # warj1253
    'Wom (Nigeria)': 'nigercongo.adamawa.wom_nigeria',   # womn1235
    'Yamba': 'nigercongo.bantoid.grassfields.yamba',   # yamb1251
    'Yara': 'nigercongo.kainji.yara',   # cind1241
    'Yendang': 'nigercongo.adamawa.yendang',   # yend1241
    'Yiwom': 'afroasiatic.chadic.yiwom',   # yiwo1237
    'Yotti': 'nigercongo.adamawa.yotti',   # yott1234
    'Yukuben': 'nigercongo.plateau.yukuben',   # yuku1243
    'Zhire': 'nigercongo.plateau.zhire',   # zhir1238
    'Zizilivakan': 'afroasiatic.chadic.zizilivakan',   # zizi1238
    'Zul': 'afroasiatic.chadic.zul',   # zull1239
    'Zumbun': 'afroasiatic.chadic.zumbun',   # zumb1240
    'Àhàn': 'nigercongo.voltaniger.ahan',   # ahan1244
}
# <<< end of generated block
NAMES.update(HOMELAND)

EXTRA_NODES = []


def resolve(name):
    return NAMES.get(name)
