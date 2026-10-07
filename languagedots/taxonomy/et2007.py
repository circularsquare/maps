"""Ethiopia 2007 census, mother tongue (CSA Table 3.2, via USCB) -> node.

Keyed by CSA's own field names, which USCB's Data Dictionary preserves; NOT by USCB's ISO
renamings, several of which are wrong (sources/et_uscb.py). Where a CSA name is not a
language name in any reference, the call rests on where its speakers live (the zone they are
counted in), and the comment says so. Shares quoted are of the label's national total.

`Other Ethiopian Language` (119,659; 53% in Afder zone, Somali region) and `Other Foreign
Language` (20,102) are unnamed remainders spanning several families, so both sit on `other`.
There is no "not stated" cell: the 91 categories add up to the census population exactly.
"""
AS = "afroasiatic.ethiosemitic"
LE = "afroasiatic.cushitic.lowland"
HE = "afroasiatic.cushitic.highland"
AG = "afroasiatic.cushitic.agaw"
OM = "afroasiatic.omotic"
NS = "nilosaharan"

NAMES = {
    # ---- Ethiopian Semitic
    "Amarigna": f"{AS}.amharic",
    "Tigrigna": f"{AS}.tigrinya",
    "Guragiegna": f"{AS}.gurage",          # the census's one "Gurage", over Sebat Bet, Soddo and Mesqan alike
    # "Shitagna" is how USCB transcribed CSA's field; 83% of its 880,818 are in Silti zone and
    # 4% in neighbouring Gurage, and Silt'e has no other line in the table. It is Silt'e.
    "Shitagna": f"{AS}.silte",
    "Hareriegna": f"{AS}.harari",
    "Argobigna": f"{AS}.argobba",
    # ---- Lowland East Cushitic
    "Oromigna": f"{LE}.oromo",
    "Somaligna": f"{LE}.somali",
    "Affarigna": f"{LE}.afar",
    "Irobigna": f"{LE}.saho",               # Irob is the Saho of north-east Tigray
    "Konsogna": f"{LE}.konso",
    # Debosgna: 88% in Segen zone, 70,419 people. Segen's other languages all have lines of their
    # own (Konso, Koore, Burji, Gidole, Dirasha), and Debase is CSA's name for the Gawwada of Ale.
    "Debosgna": f"{LE}.gawwada",
    "Gedoligna": f"{LE}.gidole",           # Gidole and Dirasha are one language (Glottolog `gdl`),
    "Derashigna": f"{LE}.dirasha",         # but the census printed both, so both are drawn
    "Dasenechgna": f"{LE}.daasanach",
    "Tsemayigna": f"{LE}.tsamai",
    "Arborigna": f"{LE}.arbore",
    "Gedichogna": f"{LE}.baiso",           # Gidicho island, Lake Abaya; its people speak Baiso
    # Mosiye and Mashile are both names Ethnologue lists for Bussa (Konsoid), and both lines sit
    # in Segen zone (74% and 70%); two answers, two nodes.
    "Mossigna": f"{LE}.mosiye",
    "Mashiligna": f"{LE}.mashile",
    # Kusume: 96% in Segen zone, not identified with a language in Glottolog or Ethnologue.
    # Every other Segen language bar Koore is Lowland East Cushitic, so it is placed there.
    "Kusumegna": f"{LE}.kusume",
    # ---- Highland East Cushitic
    "Sidamigna": f"{HE}.sidama",
    "Hadiyigna": f"{HE}.hadiyya",
    "Gedeogna": f"{HE}.gedeo",
    "Kembatigna": f"{HE}.kambaata",
    "Alabigna": f"{HE}.alaba",
    "Tembarogna": f"{HE}.timbaro",          # Timbaro, a Kambaata variety; 96% in Kembata Tembaro
    "Marekogna": f"{HE}.mareko",
    "Qebenigna": f"{HE}.qebena",
    "Burjigna": f"{HE}.burji",
    "Dongigna": f"{HE}.donga",              # Donga, a Kambaata variety; 94% in Kembata Tembaro
    # ---- Agaw
    "Agew-Awinigigna": f"{AG}.awngi",
    "Agew-Kamyrnya": f"{AG}.xamtanga",       # Kamyr: Xamtanga of Wag Hemra
    # Felashigna, "the language of the Falasha" (Beta Israel), whose old language Kayla is
    # Agaw. Its 946 speakers are scattered across the south and east, nowhere near Gondar, so
    # the label may not mean what it says; drawn as named, on Agaw, which is what the name says.
    "Felashigna": f"{AG}.felashigna",
    # ---- Omotic
    "Welaitigna": f"{OM}.ometo.wolaytta",
    "Gamogna": f"{OM}.ometo.gamo",
    "Goffigna": f"{OM}.ometo.gofa",
    "Dawurogna": f"{OM}.ometo.dawro",
    "Kontigna": f"{OM}.ometo.konta",        # 93% in Konta special woreda
    "Maliegna": f"{OM}.ometo.male",
    "Oydigna": f"{OM}.ometo.oyda",
    "Basketigna": f"{OM}.ometo.basketo",
    "Charigna": f"{OM}.ometo.chara",
    "Zeysegna": f"{OM}.ometo.zayse",
    "Qechemigna": f"{OM}.ometo.kachama",
    # Koregna (156,749, 98% in Segen zone, where Amaro is) is Koore. USCB gave the ISO name
    # Koorete to the other line, Koyrigna (2,473, scattered over Borena, Gamo Gofa and Guji);
    # Koyra is another name for Koorete, but the census asked them apart, so it is its own node.
    "Koregna": f"{OM}.ometo.koore",
    "Koyrigna": f"{OM}.ometo.koyra",
    "Keffagna": f"{OM}.gonga.kafa",
    "Shekacho": f"{OM}.gonga.shekkacho",
    "Shinashigna": f"{OM}.gonga.shinasha",
    "Benchigna": f"{OM}.bench",
    "Yemsagna": f"{OM}.yem",
    "Maogna": f"{OM}.mao",                  # the census's one "Mao"; 55% in West Wellega
    "Shekogna": f"{OM}.dizoid.sheko",
    "Dizigna": f"{OM}.dizoid.dizi",
    "Naogna": f"{OM}.dizoid.nayi",
    "Arigna": f"{OM}.south.aari",
    "Hamerigna": f"{OM}.south.hamer",
    "Benagna": f"{OM}.south.banna",         # 96% in South Omo: Banna, Hamer's neighbour
    "Karogna": f"{OM}.south.karo",
    "Dimegna": f"{OM}.south.dime",
    # Demegna: 89% in South Omo (10,155), where Dime is spoken; the census also has a scattered
    # Dimegna (574). Probably the same people under two spellings, but asked apart; two nodes.
    "Demegna": f"{OM}.south.demegna",
    # ---- Nilo-Saharan
    "Nuwerigna": f"{NS}.nilotic.nuer",
    "Anyiwakgna": f"{NS}.nilotic.anuak",
    "Nyangatomigna": f"{NS}.nilotic.nyangatom",
    "Me'enigna": f"{NS}.surmic.meen",
    "Surmagna": f"{NS}.surmic.suri",
    "Mursygna": f"{NS}.surmic.mursi",
    "Bodigna": f"{NS}.surmic.bodi",
    "Murlegna": f"{NS}.surmic.murle",
    "Mejengerigna": f"{NS}.surmic.majang",  # zero everywhere in 2007
    "Messengogna": f"{NS}.surmic.mesengo",  # Mesengo is another name for Majang; asked apart
    "Bachagna": f"{NS}.surmic.bacha",       # Bacha is a name for Kwegu (Ethnologue); USCB agrees
    "Koygogna": f"{NS}.surmic.koygo",       # Koegu, also Kwegu by name; 37% in South Omo
    "Zlmamigna": f"{NS}.surmic.zilmamu",    # 68% in Bench Maji
    "Gumuzigna": f"{NS}.gumuz",
    "Bertagna": f"{NS}.berta.berta",
    "Fedashigna": f"{NS}.berta.fadashi",
    "Komigna": f"{NS}.koman.komo",
    "Qewamigna": f"{NS}.koman.gwama",
    # UPO (1,751): 54% in Etang special woreda, Gambela, which is where Opo (Opuo) is spoken.
    # USCB's ISO match, Ignaciano, is a language of Bolivia.
    "UPO": f"{NS}.koman.opo",
    "Kunamigna": f"{NS}.kunama",
    # ---- other languages
    "English": "indoeuropean.germanic.english",
    "Other Ethiopian Language": "other",
    "Other Foreign Language": "other",
    # Named, but not identified with a language; each gets a node of its own under `other`.
    # Brayligna is spread over Amhara's zones, where blindness from trachoma is commonest, and
    # reads as "Braille"; the rest are spread thinly over the south or Addis Ababa.
    "Merigna": "other.merigna",
    "Brayligna": "other.brayligna",
    "Guagugna": "other.guagugna",
    "Wergigna": "other.wergigna",
    "Gebatogna": "other.gebatogna",
    "Shegna": "other.shegna",               # all in Addis Ababa; not She (a Bench variety of Sheka)
}


def resolve(name):
    return NAMES[name]
