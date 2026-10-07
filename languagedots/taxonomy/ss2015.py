"""South Sudan, High Frequency South Sudan Survey waves 1 (2015) and 2 (2016), tribe of the household
head (sources/ss_hfs.py) -> node. Each group read as its language (AGENT_BRIEF section 2,
ethnicity only); sources/ss.md section 4 has the retention question.

Keys are the card's labels verbatim, and "Other: <verbatim>" for C.8.1's write-ins. Glottolog
codes from data/raw/glottolog/languages.csv; no code where Glottolog has no entry.

Calls worth knowing:
  * Dinka (dink1262, a family of five Dinka languages in Glottolog): one node, as Canada's census
    and every other source here have it. Wave 2's one "Rek" (Warrap), a Dinka section and dialect
    group named apart on the card: a sibling leaf, not a child (a child would turn Dinka into a
    group drawn as "language not named").
  * Kakwa and "Kakowa": both on the card, and Kakowa's 34 heads are Kakwa's people (30 in Central
    Equatoria, 18 born in Yei county, 2 in Morobo); a spelling variant, merged.
  * "Atuot" and "Atwot, Atuot": spelling variants, merged on Reel (reel1238, the Atuot's language).
  * "Jurchol, Jo-Luo, Jur Chol" and "Luo": the card's "Luo" was answered in Western Bahr el Ghazal
    by heads born in Jur River county (42 of 59), the Jur Chol's home; Dholuo (Kenya) is not
    meant. Both on Luwo (luwo1239), as two names for one people.
  * "Shatt", "Other: Jur Shat" and "Thuri": Jur Shatt is another name for the Thuri (thur1255),
    Western Nilotic, in Western Bahr el Ghazal (the Daju-speaking Shatt live in Sudan's Nuba
    Mountains, and all five "Shatt" heads were born in Raja or Wau counties). One node.
  * "Boya" and "Larim": two names for one people and language, Narim (nari1240, Surmic). One node.
  * "Balanda Viri" and "Bviri": the same name, Belanda Viri (bela1255). Merged.
  * "Lulubo" and "Olubo": both Olu'bo (olub1238). Merged.
  * "Kaliko": Keliko (keli1248).
  * "Moro Nuba": 3 heads, born in Mundri East, Lainya and Wau counties, none in Sudan's Nuba
    Mountains where the Moro (moro1285) live; read as the card's neighbouring "Moru, Moro"
    mis-tapped, and drawn on Moru (moru1253). The Moru write-ins (Miza, Kodo, Kederi, Agi, sections
    and dialects of Moru, and their spellings) are on Moru too.
  * Lotuko (Otuho, otuh1238) and the Lotuxo groups the card names apart: Lokoya (loko1254),
    Lango of South Sudan (lang1343, not Uganda's Lango), Lopit (lopi1242), Dongotono (dong1294),
    Logir (logi1239); Ifoto, Imatong and Ketebo have no Glottolog entry and are named by the
    respondents as their own: each its own Nilotic leaf.
  * Bari-speaking groups the card names apart: Bari (bari1284), Kuku (kuku1285), Pojulu,
    Mundari, Nyangwara (no entries of their own in Glottolog), Nyepu (nyep1239). Each a leaf.
  * Baka is South Sudan's Central Sudanic Baka (baka1274), not Cameroon's: a node of its own.
  * Write-ins mapped to the card's own groups: Zande (Azande), Bele (Jur Beli), Forege
    (Feroghe), Indiri and Indry (Indri), Pojulu, Kuku, Yulu, Dogo (Ndogo, Western Bahr el Ghazal).
  * Arabs: "Arab", "An Arab", "Arabic", Misiriya, Habaniya, Halba ("Halbaya", "Bini halba"),
    Baggara Arab peoples: Sudanese Arabic, sd.txt's node.
  * Darfur(fur), Four: Fur. Zagawa, Zakawa: Zaghawa. Maslaty: Masalit. Falata, Palata: Fula
    (the Fellata, as Sudan reads them); "Palata hahosa" (Fellata Hausa): Hausa. Baganda, Muganda,
    Mugandan, Bagada, Magadan: Ganda. Mukonjo: Konzo. Lubara, Lugera, Terego/lugara, Lugbwara:
    Lugbara. Kiswahili: Swahili. Banda: cf.txt's Banda.
  * `africa_other`: "Nuba", "Noba", "Noba mari" (Nuba names the peoples of a region whose
    languages are in four families); "Darfur" alone (a region); "Fertit" (a collective name for
    the small peoples of western Bahr el Ghazal); Tunyjur (sd.txt's Tunjur precedent); "Arua
    district" (a Ugandan district, several languages); and names not identified: Abania, Ajigo,
    Balanda Bagari, Beeri, Bukere, Dongulai, Ingisa, Ranga, Shita, Wadi, Wadii, Wira.
  * "Not stated": not drawn.
"""
NI = "nilosaharan.nilotic"
CS = "nilosaharan.centralsudanic"
SU = "nilosaharan.surmic"
UB = "nigercongo.ubangian"
AO = "africa_other"

NAMES = {
    # ---- Nilotic ----
    "Dinka, Jieeng, Muonyjang": f"{NI}.dinka",
    "Rek": f"{NI}.rek",
    "Nuer, Naath": f"{NI}.nuer",
    "Atuot": f"{NI}.reel",
    "Atwot, Atuot": f"{NI}.reel",
    "Shilluk, Chollo, Collo": f"{NI}.shilluk",
    "Pari, Paeri": f"{NI}.pari",
    "Acholi": f"{NI}.acholi",
    "Jurchol, Jo-Luo, Jur Chol": f"{NI}.luwo",
    "Luo": f"{NI}.luwo",
    "Balanda Bor": f"{NI}.belanda_bor",
    "Thuri": f"{NI}.thuri",
    "Shatt": f"{NI}.thuri",
    "Other: Jur Shat": f"{NI}.thuri",
    "Bari": f"{NI}.bari",
    "Kakwa": f"{NI}.kakwa",
    "Kakowa": f"{NI}.kakwa",
    "Kuku": f"{NI}.kuku",
    "Other: Kuku": f"{NI}.kuku",
    "Pojulu": f"{NI}.pojulu",
    "Other: Pojulu": f"{NI}.pojulu",
    "Mundari": f"{NI}.mundari",
    "Nyangwara": f"{NI}.nyangwara",
    "Nyepu": f"{NI}.nyepu",
    "Toposa": f"{NI}.toposa",
    "Lotuko, Lotuka": f"{NI}.otuho",
    "Lokoya": f"{NI}.lokoya",
    "Lango": f"{NI}.lango_ss",
    "Lopit": f"{NI}.lopit",
    "Dongotono, Dongotona": f"{NI}.dongotono",
    "Logir": f"{NI}.logir",
    "Ifoto": f"{NI}.ifoto",
    "Imatong": f"{NI}.imatong",
    "Ketebo": f"{NI}.ketebo",
    # ---- Surmic ----
    "Didinga": f"{SU}.didinga",
    "Tenet": f"{SU}.tennet",
    "Boya": f"{SU}.narim",
    "Larim": f"{SU}.narim",
    "Murle": f"{SU}.murle",
    # ---- Central Sudanic ----
    "Moru, Moro": f"{CS}.moru",
    "Moro Nuba": f"{CS}.moru",
    "Other: Moro Miza": f"{CS}.moru",
    "Other: MoroMiza": f"{CS}.moru",
    "Other: Moru Miza": f"{CS}.moru",
    "Other: Moru miza": f"{CS}.moru",
    "Other: Muru miza": f"{CS}.moru",
    "Other: Moro kodo": f"{CS}.moru",
    "Other: Moru Kodo": f"{CS}.moru",
    "Other: Moro kadere": f"{CS}.moru",
    "Other: Moru Kederi": f"{CS}.moru",
    "Other: Moru Agyi": f"{CS}.moru",
    "Madi": "nilosaharan.madi",
    "Avukaya": f"{CS}.avokaya",
    "Kaliko": f"{CS}.keliko",
    "Logo": f"{CS}.logo",
    "Lugbwara": f"{CS}.lugbara",
    "Other: Lubara": f"{CS}.lugbara",
    "Other: Lugera": f"{CS}.lugbara",
    "Other: Terego /lugara": f"{CS}.lugbara",
    "Baka": f"{CS}.baka_ss",
    "Jur Beli, Jurbiel, Bel": f"{CS}.beli",
    "Other: Bele": f"{CS}.beli",
    "Jur Modo": f"{CS}.jur_modo",
    "Bongo": f"{CS}.bongo",
    "Lulubo": f"{CS}.olubo",
    "Olubo": f"{CS}.olubo",
    "Nyamusa": f"{CS}.nyamusa",
    "Kresh": f"{CS}.kresh",
    "Binga": f"{CS}.binga",
    "Yulu": f"{CS}.yulu",
    "Other: Yulu": f"{CS}.yulu",
    # ---- other Nilo-Saharan ----
    "Ngulgule": "nilosaharan.njalgulgule",
    "Other: Jongurgule": "nilosaharan.njalgulgule",
    "Other: Darfur(fur)": "nilosaharan.fur",
    "Other: Four": "nilosaharan.fur",
    "Other: Zagawa": "nilosaharan.zaghawa",
    "Other: Zakawa /from North Sudan": "nilosaharan.zaghawa",
    "Other: Maslaty": "nilosaharan.maban.masalit",
    # ---- Ubangian ----
    "Azande": f"{UB}.zande.zande",
    "Other: Zande": f"{UB}.zande.zande",
    "Makaraka": f"{UB}.zande.makaraka",
    "Balanda Viri": f"{UB}.belanda_viri",
    "Bviri": f"{UB}.belanda_viri",
    "Ndogo": f"{UB}.ndogo",
    "Other: Dogo": f"{UB}.ndogo",
    "Bai": f"{UB}.bai",
    "Gollo": f"{UB}.golo",
    "Indri": f"{UB}.indri",
    "Other: Indiri": f"{UB}.indri",
    "Other: Indry": f"{UB}.indri",
    "Sere": f"{UB}.sere",
    "Feroghe": f"{UB}.feroge",
    "Other: Forege": f"{UB}.feroge",
    "Mangaya": f"{UB}.mangayat",
    "Mundu": f"{UB}.mundu_baka.mundu",
    "Other: Banda": f"{UB}.banda.banda",
    # ---- from beyond South Sudan ----
    "Other: Arab": "afroasiatic.sudanese_arabic",
    "Other: An Arab": "afroasiatic.sudanese_arabic",
    "Other: Arabic": "afroasiatic.sudanese_arabic",
    "Other: Misiriya": "afroasiatic.sudanese_arabic",
    "Other: Habaniya": "afroasiatic.sudanese_arabic",
    "Other: Halbaya/from Sudan": "afroasiatic.sudanese_arabic",
    "Other: Bini halba": "afroasiatic.sudanese_arabic",
    "Other: Falata": "nigercongo.atlantic.fulah",
    "Other: Palata": "nigercongo.atlantic.fulah",
    "Other: Palata hahosa": "afroasiatic.chadic.hausa",
    "Other: Baganda": "nigercongo.bantu.ganda",
    "Other: Muganda": "nigercongo.bantu.ganda",
    "Other: Mugandan": "nigercongo.bantu.ganda",
    "Other: Bagada": "nigercongo.bantu.ganda",
    "Other: Magadan": "nigercongo.bantu.ganda",
    "Other: Mukonjo": "nigercongo.bantu.konzo",
    "Other: Kiswahili": "nigercongo.bantu.swahili",
    # ---- not identified, or not a language ----
    "Fertit": AO,
    "Shita": AO,
    "Other: Nuba": AO,
    "Other: Noba": AO,
    "Other: Noba mari": AO,
    "Other: Darfur": AO,
    "Other: Tunyjur": AO,
    "Other: Arua district": AO,
    "Other: Abania": AO,
    "Other: Ajigo": AO,
    "Other: Balanda Bagari": AO,
    "Other: Beeri": AO,
    "Other: Bukere": AO,
    "Other: Dongulai": AO,
    "Other: Ingisa": AO,
    "Other: Ranga": AO,
    "Other: Wadi": AO,
    "Other: Wadii": AO,
    "Other: Wira": AO,
}

EXCLUDED = {"Not stated": "not drawn; in `gap`"}


def resolve(name):
    return NAMES.get(name)
