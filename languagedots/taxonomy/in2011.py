"""Census of India 2011 C-16 mother-tongue code -> node.

Keyed by the six-digit mother-tongue code, not the name: the names repeat (Bagri appears
under both HINDI and PUNJABI) and the code is what the table is built on.

`resolve(code, state)` takes the state too, for the one label whose meaning depends on where
it was returned (Pahari, below). Everything else ignores it.

UNNAMED REMAINDERS. Every one of the census's 123 languages has an `xxx999` "Others" row: mother
tongues returned under that language and too small to list. They are drawn on the narrowest
node that contains everything the census files under that language, never guessed into one
member of it. For most languages that is the language's own node. For HINDI it is Indo-Aryan
as a whole, because the census's Hindi spans four of Masica's zones; that row is 16.7M people,
14.9M of them in Bihar, where they are very probably Bajjika and Angika speakers, but the
census does not say so and this map does not either (spec §3).
"""
IA = "indoeuropean.indoaryan"

CODES = {
    # 001 ASSAMESE
    "001002": f"{IA}.eastern.assamese",
    "001999": f"{IA}.eastern.assamese",
    # 002 BENGALI
    "002007": f"{IA}.eastern.bengali",
    "002011": f"{IA}.eastern.chakma",
    "002015": f"{IA}.eastern.hajong",
    "002044": f"{IA}.eastern.rajbanshi",
    "002999": f"{IA}.eastern",
    # 003 BODO
    "003001": "sinotibetan.boro_garo.bodo",
    "003003": "sinotibetan.boro_garo.kachari",
    "003004": "sinotibetan.boro_garo.mech",
    "003999": "sinotibetan.boro_garo",
    # 004 DOGRI
    "004001": f"{IA}.northwestern.dogri",
    "004999": f"{IA}.northwestern.dogri",
    # 005 GUJARATI. Gujrao, Pattani and Ponchi are Gujarati varieties of particular
    # communities; Saurashtra is the language of the Saurashtrians of Madurai, a Gujarati
    # offshoot kept apart because its geography is the point (Tamil Nadu, not Gujarat).
    "005018": f"{IA}.gujarati.gujarati",
    "005019": f"{IA}.gujarati.gujarati",
    "005052": f"{IA}.gujarati.gujarati",
    "005054": f"{IA}.gujarati.gujarati",
    "005057": f"{IA}.gujarati.saurashtra",
    "005999": f"{IA}.gujarati",
    # 006 HINDI. The census's Hindi is a political umbrella over four zones.
    "006030": f"{IA}.eastcentral.awadhi",
    "006040": f"{IA}.pahari.western.baghati",                    # Baghati Pahari
    "006042": f"{IA}.eastcentral.bagheli",
    "006045": f"{IA}.rajasthani.bagri",
    "006066": f"{IA}.rajasthani.lambadi",                # Banjari
    "006086": f"{IA}.pahari.western.bhadrawahi",
    "006089": f"{IA}.bhil.bhagoria",                              # Bhagoria, a Bhil variety of Malwa
    "006096": f"{IA}.pahari.western.gaddi",
    "006102": f"{IA}.bihari.bhojpuri",
    "006119": f"{IA}.rajasthani.bishnoi",                        # Bishnoi
    "006123": f"{IA}.central.braj",
    "006125": f"{IA}.central.bundeli",
    "006133": f"{IA}.pahari.western.chambeali",
    "006142": f"{IA}.eastcentral.chhattisgarhi",
    "006149": f"{IA}.pahari.western.churahi",                    # Churahi
    "006173": f"{IA}.rajasthani.dhundhari",
    "006195": f"{IA}.pahari.central.garhwali",
    "006198": f"{IA}.rajasthani.gawari",                        # Gawari
    "006207": f"{IA}.rajasthani.gujari",
    "006231": f"{IA}.pahari.western.handuri",                    # Handuri
    "006232": f"{IA}.rajasthani.harauti",
    "006235": f"{IA}.central.haryanvi",
    "006240": f"{IA}.central.hindi",
    "006265": f"{IA}.pahari.western.jaunsari",
    "006291": f"{IA}.pahari.western.kangri",
    "006311": f"{IA}.central.hindi",                     # Khari Boli, the dialect Hindi is built on
    "006320": f"{IA}.bihari.khortha",
    "006336": f"{IA}.pahari.western.kullui",
    "006340": f"{IA}.pahari.central.kumaoni",
    "006345": f"{IA}.bihari.kurmali",
    "006353": f"{IA}.rajasthani.lambadi",
    "006355": f"{IA}.eastcentral.chhattisgarhi",         # Laria, the Chhattisgarhi of western Odisha
    "006358": f"{IA}.central.lodhi",
    "006376": f"{IA}.bihari.magahi",
    "006391": f"{IA}.rajasthani.malvi",
    "006394": f"{IA}.pahari.western.mandeali",
    "006400": f"{IA}.rajasthani.marwari",
    "006408": f"{IA}.rajasthani.mewari",
    "006409": f"{IA}.rajasthani.mewati",
    "006420": f"{IA}.bihari.sadri",                      # Nagpuria
    "006432": f"{IA}.rajasthani.nimadi",
    "006438": f"{IA}.pahari.western.padari",                    # Padari
    "006439": None,                                      # Pahari: by state, see resolve()
    "006449": f"{IA}.bihari.palmuha",                            # Palmuha
    "006451": f"{IA}.bihari.panchpargania",
    "006452": f"{IA}.eastcentral.chhattisgarhi",         # Pando/Pandwani
    "006454": f"{IA}.pahari.western.pangwali",                    # Pangwali
    "006466": f"{IA}.central.pawari",
    "006476": f"{IA}.puran",                                   # Puran Bhasha
    "006489": f"{IA}.rajasthani.rajasthani",             # people who answered "Rajasthani" itself
    "006503": f"{IA}.bihari.sadri",
    "006530": f"{IA}.pahari.western.sirmauri",
    "006535": f"{IA}.rajasthani.sondwari",
    "006541": f"{IA}.rajasthani.lambadi",                # Sugali, the Lambadi of Andhra
    "006547": f"{IA}.eastcentral.surgujia",
    "006548": f"{IA}.bihari.surjapuri",
    "006999": f"{IA}",
    # 007 KANNADA
    "007002": "dravidian.southern.badaga",
    "007016": "dravidian.southern.kannada",
    "007027": "dravidian.southern.kurumba",
    "007041": "dravidian.southern.kannada",              # Prakritha Bhasha
    "007999": "dravidian.southern",
    # 008 KASHMIRI. Siraji and Kishtwari are Kashmiri-related varieties of Doda;
    # Dardi is the census's word for Shina-speaking Dards of Ladakh and Gurez.
    "008005": f"{IA}.dardic.kashmiri",
    "008010": f"{IA}.dardic.kashmiri",
    "008018": f"{IA}.dardic.kashmiri",
    "008019": f"{IA}.dardic.shina",
    "008999": f"{IA}.dardic",
    # 009 KONKANI
    "009011": f"{IA}.southern.konkani",
    "009013": f"{IA}.southern.konkani",
    "009016": f"{IA}.southern.konkani",
    "009020": f"{IA}.southern.konkani",
    "009028": f"{IA}.rajasthani.lambadi",                # Gorboli/Goru: the Banjara speech of Maharashtra
    "009999": f"{IA}.southern.konkani",
    # 010 MAITHILI
    "010008": f"{IA}.bihari.maithili",
    "010011": f"{IA}.bihari.maithili",
    "010013": f"{IA}.bihari.maithili",                   # Thati
    "010014": f"{IA}.tharu.tharu",
    "010999": f"{IA}.bihari.maithili",
    # 011 MALAYALAM
    "011016": "dravidian.southern.malayalam",
    "011023": "dravidian.southern.paniya",
    "011026": "dravidian.southern.ravula",
    "011999": "dravidian.southern",
    # 012 MANIPURI
    "012003": "sinotibetan.meitei",
    "012999": "sinotibetan.meitei",
    # 013 MARATHI
    "013004": f"{IA}.southern.marathi",                  # Are
    "013060": f"{IA}.southern.marathi",                  # Koli
    "013071": f"{IA}.southern.marathi",
    "013999": f"{IA}.southern",
    # 014 NEPALI
    "014011": f"{IA}.pahari.eastern.nepali",
    "014999": f"{IA}.pahari.eastern",
    # 015 ODIA
    "015006": f"{IA}.eastern.bhatri",
    "015007": f"{IA}.eastern.odia",                      # Bhuiya (Odia-speaking)
    "015010": f"{IA}.eastern.odia",                      # Bhumijali
    "015014": f"{IA}.eastern.desia",
    "015043": f"{IA}.eastern.odia",
    "015051": f"{IA}.eastern.desia",                     # Proja (Odia)
    "015055": f"{IA}.eastern.odia",                      # Relli
    "015058": f"{IA}.eastern.sambalpuri",
    "015999": f"{IA}.eastern",
    # 016 PUNJABI. Bagri is Rajasthani by every classification, wherever the census files it.
    "016002": f"{IA}.rajasthani.bagri",
    "016005": f"{IA}.pahari.western.bhateali",                    # Bhateali, of Chamba
    "016006": f"{IA}.pahari.western.bilaspuri",
    "016038": f"{IA}.northwestern.punjabi",
    "016999": f"{IA}.northwestern",
    # 017 SANSKRIT
    "017002": f"{IA}.sanskrit",
    "017999": f"{IA}.sanskrit",
    # 018 SANTALI. Karmali and Mahili are Santali-filed Munda varieties.
    "018010": "austroasiatic.munda.santali",
    "018011": "austroasiatic.munda.mahali",
    "018040": "austroasiatic.munda.santali",
    "018999": "austroasiatic.munda",
    # 019 SINDHI
    "019002": f"{IA}.northwestern.sindhi",               # Bhatia
    "019008": f"{IA}.northwestern.kachchhi",
    "019014": f"{IA}.northwestern.sindhi",
    "019999": f"{IA}.northwestern",
    # 020 TAMIL
    "020006": "dravidian.southern.irula",
    "020009": "dravidian.southern.kaikadi",
    "020015": "dravidian.southern.korava",
    "020027": "dravidian.southern.tamil",
    "020029": "dravidian.southern.yerukula",
    "020999": "dravidian.southern",
    # 021 TELUGU
    "021046": "dravidian.southcentral.telugu",
    "021048": "dravidian.southcentral.telugu",           # Vadari
    "021999": "dravidian.southcentral",
    # 022 URDU
    "022015": f"{IA}.central.urdu",
    "022016": f"{IA}.central.urdu",                      # Bhansari
    "022999": f"{IA}.central.urdu",
    # 023 ADI. Talgalo is a Galo variety.
    "023003": "sinotibetan.tani.adi",
    "023006": "sinotibetan.tani.galo",
    "023010": "sinotibetan.tani.adi",
    "023040": "sinotibetan.tani.galo",
    "023999": "sinotibetan.tani",
    # 024 AFGHANI/KABULI/PASHTO
    "024001": "indoeuropean.iranian.pashto",
    "024999": "indoeuropean.iranian",
    "025001": "sinotibetan.kukichin.anal",
    "025999": "sinotibetan.kukichin.anal",
    "026002": "sinotibetan.naga.angami",
    "026999": "sinotibetan.naga.angami",
    "027001": "sinotibetan.naga.ao",
    "027003": "sinotibetan.naga.ao",
    "027005": "sinotibetan.naga.ao",
    "027999": "sinotibetan.naga.ao",
    "028001": "afroasiatic.arabic",
    "028999": "afroasiatic.arabic",
    "029002": "sinotibetan.tibetic.balti",
    "029999": "sinotibetan.tibetic.balti",
    # 030 BHILI/BHILODI
    "030002": f"{IA}.bhil.baori",                              # Baori
    "030003": f"{IA}.bhil.barel",
    "030006": f"{IA}.bhil.bhilali",
    "030007": f"{IA}.bhil.bhili",
    "030011": f"{IA}.bhil.chodhari",
    "030015": f"{IA}.bhil.dhodia",                              # Dhodia
    "030020": f"{IA}.bhil.gamit",
    "030021": f"{IA}.bhil.garasia",
    "030035": f"{IA}.bhil.kokna",
    "030049": f"{IA}.bhil.mawchi",
    "030066": f"{IA}.bhil.paradhi",                              # Paradhi
    "030068": f"{IA}.bhil.pawri",
    "030070": f"{IA}.bhil.rathi",                              # Rathi
    "030073": f"{IA}.bhil.tadavi",                              # Tadavi
    "030075": f"{IA}.bhil.varli",
    "030076": f"{IA}.bhil.vasava",
    "030078": f"{IA}.bhil.wagdi",
    "030999": f"{IA}.bhil",
    # 031 BHOTIA. Bauti is the Bhoti of Ladakh and Kargil (99,974 of its 100,000 are in J&K).
    "031001": "sinotibetan.tibetic.bhotia",
    "031011": "sinotibetan.tibetic.bhotia",
    "031999": "sinotibetan.tibetic",
    "032002": "austroasiatic.munda.bhumij",
    "032999": "austroasiatic.munda.bhumij",
    "033003": f"{IA}.eastern.bishnupriya",
    "033999": f"{IA}.eastern.bishnupriya",
    "034001": "sinotibetan.naga.chakhesang",
    "035001": "sinotibetan.naga.chokri",
    "036001": "sinotibetan.naga.chang",
    "037001": "dravidian.southern.kodava",
    "037003": "dravidian.southern.kodava",
    "038001": "sinotibetan.boro_garo.deori",
    "039002": "sinotibetan.boro_garo.dimasa",
    "039999": "sinotibetan.boro_garo.dimasa",
    "040001": "indoeuropean.germanic.english",
    # 041 GADABA. Two unrelated languages share the name: Gutob (Munda) and Ollari/Konekor
    # (Dravidian). 33,342 of the 40,965 are in Odisha's Koraput, which is Gutob country.
    "041001": "austroasiatic.munda.gutob",
    "041999": "austroasiatic.munda.gutob",
    "042001": "sinotibetan.kukichin.gangte",
    "043005": "sinotibetan.boro_garo.garo",
    "043999": "sinotibetan.boro_garo.garo",
    # 044 GONDI. Dorli, Kalari and Maria/Muria are Gondi varieties.
    "044002": "dravidian.southcentral.gondi",
    "044007": "dravidian.southcentral.gondi",
    "044013": "dravidian.southcentral.gondi",
    "044017": "dravidian.southcentral.gondi",
    "044999": "dravidian.southcentral.gondi",
    "045001": f"{IA}.eastern.halbi",
    "045999": f"{IA}.eastern.halbi",
    "046003": "sinotibetan.kukichin.halam",
    "046999": "sinotibetan.kukichin.halam",
    "047001": "sinotibetan.kukichin.hmar",
    "048004": "austroasiatic.munda.ho",
    "048007": "austroasiatic.munda.ho",                  # Lohara
    "049002": "dravidian.southcentral.kui",              # Jatapu, a Kui-Kuvi variety
    "049999": "dravidian.southcentral.kui",
    "050001": "austroasiatic.munda.juang",
    "051003": "sinotibetan.naga.rongmei",                # Kabui
    "051005": "sinotibetan.naga.rongmei",
    "051999": "sinotibetan.naga.rongmei",
    "052002": "sinotibetan.karbi",
    # 053 KHANDESHI
    "053002": f"{IA}.bhil.khandeshi",                    # Ahirani
    "053004": f"{IA}.bhil.dangi",                              # Dangi
    "053005": f"{IA}.bhil.khandeshi",                    # Gujari (Khandesh)
    "053006": f"{IA}.bhil.khandeshi",
    "053999": f"{IA}.bhil.khandeshi",
    "054005": "austroasiatic.munda.kharia",
    "054999": "austroasiatic.munda.kharia",
    "055007": "austroasiatic.khasian.khasi",
    "055009": "austroasiatic.khasian.lyngngam",
    "055012": "austroasiatic.khasian.pnar",
    "055015": "austroasiatic.khasian.war",
    "055999": "austroasiatic.khasian",
    "056003": "sinotibetan.naga.khezha",
    "056999": "sinotibetan.naga.khezha",
    "057001": "sinotibetan.naga.khiamniungan",
    "057999": "sinotibetan.naga.khiamniungan",
    "058001": "dravidian.southcentral.kui",              # Khond/Kondh
    "058006": "dravidian.southcentral.kuvi",
    "059003": "sinotibetan.westhimalayish.kinnauri",
    "059999": "sinotibetan.westhimalayish.kinnauri",
    "060002": "dravidian.northern.kurukh",               # Kisan, a Kurukh variety
    "061002": "sinotibetan.boro_garo.koch",
    "061999": "sinotibetan.boro_garo.koch",
    "062003": "austroasiatic.munda.koda",
    "062999": "austroasiatic.munda.koda",
    "063001": "dravidian.central.kolami",
    "064001": "sinotibetan.kukichin.kom",
    "065002": "dravidian.southcentral.konda",            # Kodu
    "065003": "dravidian.southcentral.konda",
    "065999": "dravidian.southcentral.konda",
    "066001": "sinotibetan.naga.konyak",
    "067001": "austroasiatic.munda.korku",
    "067004": "austroasiatic.munda.korku",               # Muwasi
    "067999": "austroasiatic.munda.korku",
    "068003": "austroasiatic.munda.korwa",               # Koraku
    "068999": "austroasiatic.munda.korwa",
    "069002": "dravidian.southcentral.koya",
    "070001": "dravidian.southcentral.kui",
    "070999": "dravidian.southcentral.kui",
    "071008": "sinotibetan.kukichin.kuki",
    "071999": "sinotibetan.kukichin",
    "072024": "dravidian.northern.kurukh",
    "072999": "dravidian.northern.kurukh",
    "073003": "sinotibetan.tibetic.ladakhi",
    "074002": "sinotibetan.westhimalayish.lahuli",
    "074999": "sinotibetan.westhimalayish.lahuli",
    # 075 LAHNDA
    "075001": f"{IA}.northwestern.saraiki",              # Bahawalpuri
    "075006": f"{IA}.northwestern.saraiki",              # Multani
    "075999": f"{IA}.northwestern",
    "076002": "sinotibetan.kukichin.mara",
    "076999": "sinotibetan.kukichin.mara",
    "077001": "sinotibetan.boro_garo.tiwa",
    "078001": "sinotibetan.lepcha",
    "079002": "sinotibetan.naga.liangmai",
    "079999": "sinotibetan.naga.liangmai",
    "080002": "sinotibetan.kiranti.limbu",
    "080999": "sinotibetan.kiranti.limbu",
    "081001": "sinotibetan.naga.lotha",
    "082005": "sinotibetan.kukichin.mizo",
    "082999": "sinotibetan.kukichin.mizo",
    "083002": "dravidian.northern.malto",                # Pahariya (Sauria Paharia)
    "083004": "dravidian.northern.malto",                # Kulehiya
    "083999": "dravidian.northern.malto",
    "084001": "sinotibetan.naga.mao",
    "084003": "sinotibetan.naga.poumai",
    "084999": "sinotibetan.naga.mao",
    "085001": "sinotibetan.naga.maram",
    "086001": "sinotibetan.naga.maring",
    "087003": "sinotibetan.tani.mising",
    "088007": "sinotibetan.mishmi",
    "088999": "sinotibetan.mishmi",
    "089004": "sinotibetan.burmish.marma",
    "089999": "sinotibetan.burmish.marma",
    "090003": "sinotibetan.eastbodish.monpa",
    "091006": "austroasiatic.munda.mundari",             # Kol
    "091009": "austroasiatic.munda.mundari",
    "091999": "austroasiatic.munda",
    "092005": "austroasiatic.munda.mundari",
    "092999": "austroasiatic.munda.mundari",
    "093001": "austroasiatic.nicobarese",
    "094001": "sinotibetan.tani.apatani",
    "094007": "sinotibetan.tani.nyishi",
    "094008": "sinotibetan.tani.tagin",
    "094999": "sinotibetan.tani",
    "095003": "sinotibetan.naga.nocte",
    "095999": "sinotibetan.naga.nocte",
    "096002": "sinotibetan.kukichin.paite",
    "096999": "sinotibetan.kukichin.paite",
    "097001": "dravidian.central.parji",
    "097999": "dravidian.central.parji",
    "098001": "sinotibetan.kukichin.lai",
    "100001": "sinotibetan.naga.phom",
    "101006": "sinotibetan.naga.pochuri",
    "101999": "sinotibetan.naga.pochuri",
    "102002": "sinotibetan.boro_garo.rabha",
    "102999": "sinotibetan.boro_garo.rabha",
    "103003": "sinotibetan.kiranti.rai",                     # Rai: a community name over several Kiranti languages
    "103999": "sinotibetan.kiranti",
    "104001": "sinotibetan.naga.rengma",
    "105004": "sinotibetan.naga.sangtam",
    "105999": "sinotibetan.naga.sangtam",
    "106004": "austroasiatic.munda.sora",
    "106999": "austroasiatic.munda.sora",
    "107002": "sinotibetan.naga.sumi",
    "108001": "sinotibetan.tibetic.sherpa",
    "109005": f"{IA}.dardic.shina",
    "109999": f"{IA}.dardic.shina",
    "111001": "sinotibetan.tamangic.tamang",
    "112008": "sinotibetan.naga.tangkhul",
    "112999": "sinotibetan.naga.tangkhul",
    "113024": "sinotibetan.naga.tangsa",
    "113999": "sinotibetan.naga.tangsa",
    "114015": "sinotibetan.kukichin.thadou",
    "114999": "sinotibetan.kukichin.thadou",
    "115008": "sinotibetan.tibetic.tibetan",
    "115011": "sinotibetan.tibetic.purgi",
    "115999": "sinotibetan.tibetic",
    "116007": "sinotibetan.boro_garo.kokborok",
    "116011": "sinotibetan.boro_garo.reang",
    "116014": "sinotibetan.boro_garo.kokborok",
    "116999": "sinotibetan.boro_garo",
    "117009": "dravidian.southern.tulu",
    "117999": "dravidian.southern.tulu",
    "118002": "sinotibetan.kukichin.vaiphei",
    "119001": "sinotibetan.naga.wancho",
    "120001": "sinotibetan.naga.yimchungru",             # Chirr
    "120004": "sinotibetan.naga.yimchungru",             # Tikhir
    "120005": "sinotibetan.naga.yimchungru",
    "120999": "sinotibetan.naga.yimchungru",
    "121001": "sinotibetan.naga.zeliang",
    "122003": "sinotibetan.naga.zeme",
    "122999": "sinotibetan.naga.zeme",
    "123001": "sinotibetan.kukichin.zou",
    "124999": "other",
}


# nodes resolve() returns that are not CODES values, for taxonomy/build.py's check
EXTRA_NODES = [f"{IA}.northwestern.pahari_pothwari", f"{IA}.pahari.western.pahari",
               f"{IA}.eastern.sylheti"]

# BENGALI IN THE SYLHETI-SPEAKING DISTRICTS (Anita, 2026-10-07: "barak: yeah lets do it"). C-16
# files Sylheti under Bengali, as Bangladesh's census does; Bangladesh's Sylhet is drawn as Sylheti
# by place (countries/bd.py, sources/bd.md §4a), and the same people live across the border. The
# Barak Valley (Cachar 18316, Karimganj 18317, Hailakandi 18318; 2.93M Bengali) is Sylheti-
# speaking, as is north Tripura (North Tripura 16292, which in 2011 held today's Unakoti:
# Dharmanagar, Kailashahar, Kumarghat; 465k), part of greater Sylhet's speech area. The rest of
# Tripura (Comilla-Noakhali speech) and Assam's scattered Bengali stay Bengali. Every Bengali there
# goes over, as in Sylhet; Silchar town's Bengalis from elsewhere are not taken out (no figure).
SYLHETI_DISTRICTS = {"18316", "18317", "18318", "16292"}


def resolve(code, state=None, district=None):
    """Node for a mother-tongue code. `state` is the two-digit census state code, `district` the
    five-digit state + district code."""
    if code == "006439":
        # "Pahari" means two different things. In Jammu and Kashmir (977,860) it is
        # Pahari-Pothwari of Poonch and Rajouri, a Lahnda variety; in Himachal (2,190,065) and
        # elsewhere it is the Western Pahari of Mahasu and Shimla.
        return f"{IA}.northwestern.pahari_pothwari" if state == "01" else f"{IA}.pahari.western.pahari"
    if code == "002007" and district in SYLHETI_DISTRICTS:
        return f"{IA}.eastern.sylheti"
    return CODES[code]
