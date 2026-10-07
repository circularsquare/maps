"""Philippines 2020 CPH ETHNICITY -> the language node it is drawn as. A proxy: the census's
language question was published nationally only (sources/ph.md). Keyed by the labels
sources/ph_census.py writes: PSA's own column headings, plus five place splits it makes.

Every ethnic group is read as its language (AGENT_BRIEF §2, ethnicity), after the source script
has moved each group's non-speakers to the unit's lingua franca by the national household-
language table. Families and middle levels checked against Glottolog (data/raw/glottolog):

  * Every local group is Austronesian. The big regional languages sit flat under
    `austronesian.philippine`, as the existing nodes (tagalog, cebuano, ilocano, hiligaynon,
    bikol, waray, kapampangan, pangasinan, kinaraya, kankanaey) already do in other countries.
  * Clusters with many census subgroups get a group node: Kalinga (40 subgroups), Itneg, Ifugao,
    Baliwon (all inside `northern_luzon`, Glottolog's Northern Luzon minus Ilocano, Pangasinan
    and Kankanaey, which keep their flat ids), Manobo, Mansakan, Mangyan, Sambalic, Panay
    Bukidnon, and "Agta and Dumagat" (Luzon's Negrito peoples; NOT one Glottolog unit: Northern
    Luzon Agta, Bikol Agta and Manide are in different branches; grouped because the census
    files them as one family of names and a reader knows them so).
  * A cluster's generic label (Kalinga, Itneg, Ifugao, Manobo, Mangyan, Agta, Dumagat, Panay
    Bukidnon) sits on the group node, drawn as "language not named": the census offered the
    subgroups and these people did not pick one.
  * Sama-Bajaw (sama1302) is its own Austronesian branch outside Philippine: `austronesian.
    sama_bajaw`, with Yakan and Jama Mapun.
  * Chavacano is a Spanish-based creole (Glottolog chav1241 files it under Indo-European);
    `creole.spanish_based`, one leaf per census variety.
  * Bisaya/Binisaya -> Cebuano: the name Cebuano speakers in Mindanao, Leyte and Negros Oriental
    use for their language (Glottolog lists Binisaya under Cebuano). The national language table
    agrees: Bisaya + Cebuano + Boholano households are 0.97 of what the three ethnic groups imply.
  * Boholano has a leaf of its own beside Cebuano: a Cebuano dialect, but the language table
    prints it as an answer (309,977 households).
  * Ilonggo -> Hiligaynon; Karay-a -> Kinaray-a; Capizeño -> Capiznon (capi1239); Masbateño ->
    Masbatenyo; Cuyonen -> Cuyonon; Bantoanon -> Asi (Bantoanon); Teduray -> Tiruray (tiru1241).
  * No language of their own: Caviteño and Batangan (Tagalogs of Cavite and Batangas) -> Tagalog;
    Bago (La Union's mixed Ilocano-Igorot people; 631 households speak "Bago") -> Ilocano;
    Dumagat in Mindanao (the Lumad name for lowland Visayan settlers) -> Cebuano; Cotabateño
    and the generic Aeta, Ata/Negrito, Bagobo, Bukidnon of the Visayas and "Other Local
    Ethnicity" -> `austronesian.philippine`, not named.
  * Other Local Ethnicity in Aklan (88% of the province) -> Aklanon, and in Zambales (23%) ->
    Sambal: the ethnicity list has neither people, and the language table's 133,121 households of
    "Bukidnon/Binukid-Akeanon/Aklanon" are about Aklan's household count. Bukidnon-Akeanon also
    -> Aklanon.
  * Groups no source places in a language family (Imalawa, Ibukid, Kaunana, Magkunana,
    Kabayukan, Kailawan) get leaves under `austronesian.philippine`; Eskaya -> `other.eskayan`
    (Glottolog: artificial language).
  * Foreign ethnicities -> the nationality's language where it has one (American, British,
    Australian, Canadian, South African -> English; Chinese and Taiwanese -> Chinese, no variety
    named); Indian -> Indo-Aryan, not named; Swiss, Singaporean, Afghan and Other Foreign ->
    `other`.
  * Not Reported (18,590) is not drawn: countries/ph.py's gap.
"""
P = "austronesian.philippine"
NL = f"{P}.northern_luzon"
MAN = f"{P}.manobo"
MSK = f"{P}.mansakan"
MGY = f"{P}.mangyan"
SAM = f"{P}.sambalic"
AGT = f"{P}.agta"
PB = f"{P}.panay_bukidnon"
SB = "austronesian.sama_bajaw"
CH = "creole.spanish_based"

# (node id, legend label) for every node this file adds; tree.d/ph.txt lists the same
NODES = {}


def _n(nid, label):
    NODES[nid] = label
    return nid


NAMES = {
    # ---- regional languages, flat ----
    "Tagalog": f"{P}.tagalog",
    "Caviteño": f"{P}.tagalog",
    "Batangan": f"{P}.tagalog",
    "Bisaya/ Binisaya": f"{P}.cebuano",
    "Cebuano": f"{P}.cebuano",
    "Dumagat (Mindanao)": f"{P}.cebuano",
    "Boholano": _n(f"{P}.boholano", "Boholano"),
    "Ilonggo": f"{P}.hiligaynon",
    "Ilocano": f"{P}.ilocano",
    "Bago": f"{P}.ilocano",
    "Bikol/Bicol": f"{P}.bikol",
    "Waray": f"{P}.waray",
    "Kapampangan": f"{P}.kapampangan",
    "Pangasinan": f"{P}.pangasinan",
    "Karay-a": f"{P}.kinaraya",
    "Kankanaey": f"{P}.kankanaey",
    "Maguindanao": _n(f"{P}.maguindanao", "Maguindanao"),
    "Maranao": _n(f"{P}.maranao", "Maranao"),
    "Iranun/ Iraynun": _n(f"{P}.iranun", "Iranun"),
    "Tausog/ Tausug": _n(f"{P}.tausug", "Tausug"),
    "Capizeño": _n(f"{P}.capiznon", "Capiznon"),
    "Masbateño/ Masbatenon": _n(f"{P}.masbatenyo", "Masbatenyo"),
    "Surigaonon": _n(f"{P}.surigaonon", "Surigaonon"),
    "Romblomanon": _n(f"{P}.romblomanon", "Romblomanon"),
    "Cuyonen/ Cuyunon": _n(f"{P}.cuyonon", "Cuyonon"),
    "Bantoanon": _n(f"{P}.asi", "Asi (Bantoanon)"),
    "Agutaynen": _n(f"{P}.agutaynen", "Agutaynen"),
    "Other Local Ethnicity (Aklan)": _n(f"{P}.aklanon", "Aklanon"),
    "Bukidnon-Akeanon": f"{P}.aklanon",
    "Subanen/ Subanon": _n(f"{P}.subanen", "Subanen"),
    "Kolibugan": _n(f"{P}.kolibugan", "Kolibugan"),
    "Blaan": _n(f"{P}.blaan", "Blaan"),
    "T'boli/Tboli": _n(f"{P}.tboli", "Tboli"),
    "T'duray/ Teduray": _n(f"{P}.tiruray", "Teduray (Tiruray)"),
    "Lambanguian": _n(f"{P}.lambangian", "Lambangian"),
    "Sangir/Sangil": _n(f"{P}.sangil", "Sangil"),
    "Molbog": _n(f"{P}.molbog", "Molbog"),
    "Palawan-o": _n(f"{P}.palawano", "Palawano"),
    "Palawan-O- Ken-ey": _n(f"{P}.palawano_keney", "Palawano (Ken-ey)"),
    "Palawan-O- Tao't-Bato": _n(f"{P}.palawano_taotbato", "Palawano (Tao't Bato)"),
    "Palawani": _n(f"{P}.palawani", "Palawani"),
    "Tagbanua": _n(f"{P}.tagbanwa", "Tagbanwa"),
    "Tagbanua-Tandulanen": _n(f"{P}.tandulanen", "Tagbanwa (Tandulanen)"),
    # Calamian and Kalamianen: two spellings of the Calamian Tagbanwa
    "Tagbanua-Calamian": _n(f"{P}.calamian_tagbanwa", "Calamian Tagbanwa"),
    "Tagbanua-Kalamianen": f"{P}.calamian_tagbanwa",
    "Batak": _n(f"{P}.batak", "Batak (Palawan)"),
    "Ati": _n(f"{P}.inati", "Inati (Ati)"),
    "Ata": _n(f"{P}.ata", "Ata"),
    "Mamanwa": _n(f"{P}.mamanwa", "Mamanwa"),
    "Ivatan": _n(f"{P}.ivatan", "Ivatan"),
    "Ibatan": _n(f"{P}.ibatan", "Ibatan"),
    "Karulano": _n(f"{P}.karolano", "Karolano"),
    "Bagobo Klata": _n(f"{P}.giangan", "Giangan (Bagobo-Klata)"),
    # Guiangan and Diangan: spellings of Giangan
    "Guiangan": f"{P}.giangan",
    "Diangan": f"{P}.giangan",
    "Imalawa": _n(f"{P}.imalawa", "Imalawa"),
    "Ibukid": _n(f"{P}.ibukid", "Ibukid"),
    "Kaunana": _n(f"{P}.kaunana", "Kaunana"),
    "Magkunana": _n(f"{P}.magkunana", "Magkunana"),
    "Kabayukan": _n(f"{P}.kabayukan", "Kabayukan"),
    "Kailawan/ Kaylawan": _n(f"{P}.kailawan", "Kailawan"),
    # ---- not named ----
    "Other Local Ethnicity": P,
    "Aeta": P,
    "Aeta/Ayta": P,
    "Ayta": P,
    "Ata/Negrito": P,
    "Bagobo": P,
    "Cotabateño": P,
    "Bukidnon (Visayas)": P,
    # ---- Sambalic ----
    "Other Local Ethnicity (Zambales)": _n(f"{SAM}.sambal", "Sambal"),
    "Aeta/Ayta-Sambal": _n(f"{SAM}.ayta_sambal", "Ayta (Sambal)"),
    "Aeta/Ayta- Mag-Indi": _n(f"{SAM}.mag_indi", "Mag-Indi Ayta"),
    "Aeta/Ayta- Mang-Antsi": _n(f"{SAM}.mag_antsi", "Mag-antsi Ayta"),
    "Aeta/Ayta-Abelling/Abellen": _n(f"{SAM}.abellen", "Abellen Ayta"),
    "Abelling/Aberling": f"{SAM}.abellen",
    "Aeta/Ayta-Ambala": _n(f"{SAM}.ambala", "Ambala Ayta"),
    "Aeta/Ayta-Magbukun": _n(f"{SAM}.magbukun", "Magbukun Ayta"),
    # ---- Agta and Dumagat ----
    "Agta": AGT,
    "Dumagat": AGT,
    "Agta-Tabangnon/ Tabangnon": _n(f"{AGT}.tabangnon", "Agta (Tabangnon)"),
    "Agta-Cimaron": _n(f"{AGT}.cimaron", "Agta (Cimaron)"),
    "Agta-Taboy": _n(f"{AGT}.taboy", "Agta (Taboy)"),
    "Agta-Agay": _n(f"{AGT}.agay", "Agta (Agay)"),
    "Agta-Dupanigan": _n(f"{AGT}.dupaninan", "Agta (Dupaninan)"),
    "Agta-Labin": _n(f"{AGT}.labin", "Agta (Labin)"),
    "Agta-Isigiran": _n(f"{AGT}.isigiran", "Agta (Isigiran)"),
    "Agta-Dumagat": _n(f"{AGT}.agta_dumagat", "Agta (Dumagat)"),
    "Alta": _n(f"{AGT}.alta", "Alta"),
    "Dumagat/ Remontado": _n(f"{AGT}.remontado", "Dumagat (Remontado)"),
    "Dumagat-Kabolowen": _n(f"{AGT}.kabolowen", "Dumagat (Kabolowen)"),
    "Dumagat- Edimala": _n(f"{AGT}.edimala", "Dumagat (Edimala)"),
    "Dumagat-Tagebolus": _n(f"{AGT}.tagebolus", "Dumagat (Tagebolus)"),
    # Kabihug and Kabihug/Manide: one people, the Manide of Camarines Norte
    "Kabihug/ Manide": _n(f"{AGT}.manide", "Manide (Kabihug)"),
    "Kabihug": f"{AGT}.manide",
    "Parananum": _n(f"{AGT}.paranan", "Paranan"),
    # ---- Mangyan ----
    "Mangyan": MGY,
    "Iraya Mangyan": _n(f"{MGY}.iraya", "Iraya"),
    "Alangan Mangyan": _n(f"{MGY}.alangan", "Alangan"),
    "Tadyawan Mangyan": _n(f"{MGY}.tadyawan", "Tadyawan"),
    "Buhid Mangyan": _n(f"{MGY}.buhid", "Buhid"),
    # PSA prints "Buhid Mangyan" twice; the second column (169 people) is merged with the first
    "Buhid Mangyan [2]": f"{MGY}.buhid",
    "Bangon Mangyan": _n(f"{MGY}.bangon", "Bangon"),
    "Hanunuo Mangyan": _n(f"{MGY}.hanunoo", "Hanunoo"),
    "Tau-buid Mangyan": _n(f"{MGY}.tawbuid", "Tawbuid"),
    "Ratagnon Mangyan": _n(f"{MGY}.ratagnon", "Ratagnon"),
    "Gubatnon Mangyan": _n(f"{MGY}.gubatnon", "Gubatnon"),
    "Sibuyan Mangyan-Tagabukid": _n(f"{MGY}.sibuyan_tagabukid", "Sibuyan Mangyan (Tagabukid)"),
    # ---- Panay and Negros highlanders ----
    "Panay Bukidnon": PB,
    "Bukidnon- Iraynon": _n(f"{PB}.iraynon", "Iraynon"),
    "Bukidnon- Ituman": _n(f"{PB}.ituman", "Ituman"),
    # Bukidnon-Pan-Anayon and Pan-Ayanon: two spellings of the Pan-ayanon
    "Bukidnon- Pan-Anayon": _n(f"{PB}.panayanon", "Pan-ayanon"),
    "Pan-Ayanon": f"{PB}.panayanon",
    "Bukidnon-Halowodnon": _n(f"{PB}.halawodnon", "Halawodnon"),
    # Bukidnon-Magahat and Magahats: the Magahat of Negros
    "Bukidnon-Magahat": _n(f"{PB}.magahat", "Magahat"),
    "Magahats": f"{PB}.magahat",
    # ---- Mansakan ----
    "Mandaya": _n(f"{MSK}.mandaya", "Mandaya"),
    "Mansaka": _n(f"{MSK}.mansaka", "Mansaka"),
    "Kagan/ Kalagan": _n(f"{MSK}.kagan", "Kagan Kalagan"),
    "Tagakaulo": _n(f"{MSK}.tagakaulo", "Tagakaulo"),
    "Mangguangan": _n(f"{MSK}.mangguangan", "Mangguangan"),
    "Davaweño": _n(f"{MSK}.davawenyo", "Davawenyo"),
    # ---- Manobo ----
    "Manobo": MAN,
    "Bukidnon": _n(f"{MAN}.binukid", "Binukid (Bukidnon)"),
    "Talaandig": _n(f"{MAN}.talaandig", "Talaandig"),
    "Bukidnon-Tagoloanon": _n(f"{MAN}.bukidnon_tagoloanon", "Bukidnon (Tagoloanon)"),
    "Higaonon/ Higa-onon": _n(f"{MAN}.higaonon", "Higaonon"),
    "Higaonon-Tagoloanon": _n(f"{MAN}.higaonon_tagoloanon", "Higaonon (Tagoloanon)"),
    "Kamiguin": _n(f"{MAN}.kinamiguin", "Kinamiguin"),
    "Cagayanen": _n(f"{MAN}.kagayanen", "Kagayanen"),
    "Matigsalog": _n(f"{MAN}.matigsalug", "Matigsalug"),
    "Tigwahanon": _n(f"{MAN}.tigwahanon", "Tigwahanon"),
    "Umayamnon": _n(f"{MAN}.umayamnon", "Umayamnon"),
    "Talaingod": _n(f"{MAN}.talaingod", "Talaingod"),
    "Dibabawon": _n(f"{MAN}.dibabawon", "Dibabawon"),
    "Banwaon": _n(f"{MAN}.banwaon", "Banwaon"),
    "Langilan": _n(f"{MAN}.langilan", "Langilan"),
    "Tinananen": _n(f"{MAN}.tinananen", "Tinananen"),
    "Obu-Manuvu": _n(f"{MAN}.obo", "Obo Manobo"),
    "Ubo Monuvu/ Manobo-Ubo/ Ubo Manobo/ Ubo Manuvu/ Ubo/Menuvu": _n(f"{MAN}.ubo", "Ubo Manobo"),
    # Bagobo Tagabawa and Tagabawa: one people
    "Bagobo Tagabawa": _n(f"{MAN}.tagabawa", "Tagabawa"),
    "Tagabawa": f"{MAN}.tagabawa",
    "Manobo-Ata": _n(f"{MAN}.ata_manobo", "Ata Manobo"),
    "Manobo-Dulangan": _n(f"{MAN}.dulangan", "Dulangan Manobo"),
    "Manobo-Dulungan-Lambangian": _n(f"{MAN}.dulangan_lambangian", "Dulangan Manobo (Lambangian)"),
    "Manobo-Aromanon": _n(f"{MAN}.aromanon", "Aromanon Manobo"),
    "Manobo-Pulanguihon": _n(f"{MAN}.pulangiyen", "Pulangiyen Manobo"),
    "Manobo-Kirenteken": _n(f"{MAN}.kirenteken", "Kirenteken Manobo"),
    "Manobo-Dunggoanon": _n(f"{MAN}.dunggoanon", "Dunggoanon Manobo"),
    "Manobo-Blit": _n(f"{MAN}.blit", "Blit Manobo"),
    "Manobo- Blit-Tasaday": _n(f"{MAN}.tasaday", "Tasaday"),
}

# Arumanen (Aromanen/Eromanen) Manobo: the generic and ten subgroups, one leaf each
AROMANEN = "Aromanen-Manobo/ Eromanen-Manobo"
NAMES[AROMANEN] = _n(f"{MAN}.arumanen", "Arumanen Manobo")
for sub in ("Ilianen", "Kulmanen", "Livunganen", "Mulitaan", "Lahitanen", "Kirenteken",
            "Dibabeen", "Direrayaan", "Isoroken", "Pulengien"):
    NAMES[f"{AROMANEN} {sub}"] = _n(f"{MAN}.arumanen_{sub.lower()}", f"Arumanen Manobo ({sub})")

# ---- Northern Luzon ----
NAMES.update({
    "Ibaloy": _n(f"{NL}.ibaloi", "Ibaloi"),
    "Kalanguya": _n(f"{NL}.kalanguya", "Kalanguya"),
    "Kalanguya-Ikalahan": _n(f"{NL}.ikalahan", "Kalanguya (Ikalahan)"),
    "Kalanguya-Yattuka": _n(f"{NL}.yattuka", "Kalanguya (Yattuka)"),
    "Kankanaey- Hak'ki": _n(f"{NL}.hakki", "Kankanaey (Hak'ki)"),
    "Applai": _n(f"{NL}.applai", "Applai"),
    "Applai-Kachakran/ Kadaclan": _n(f"{NL}.kadaclan", "Applai (Kadaclan)"),
    "Karao": _n(f"{NL}.karao", "Karao"),
    "Iwak": _n(f"{NL}.iwak", "Iwak"),
    "Isinai": _n(f"{NL}.isinai", "Isinai"),
    "Bontok": _n(f"{NL}.bontok", "Bontok"),
    "Bontok-Majukayong": _n(f"{NL}.majukayong", "Bontok (Majukayong)"),
    "Balangao": _n(f"{NL}.balangao", "Balangao"),
    "Balangao-Lias": _n(f"{NL}.lias", "Balangao (Lias)"),
    # Isnag, Isneg and Isneg/Isnag: spellings of one answer
    "Isnag": _n(f"{NL}.isnag", "Isnag"),
    "Isneg": f"{NL}.isnag",
    "Isneg/ Isnag": f"{NL}.isnag",
    "Yapayao": _n(f"{NL}.yapayao", "Yapayao"),
    "Ibanag": _n(f"{NL}.ibanag", "Ibanag"),
    "Itawes": _n(f"{NL}.itawis", "Itawis"),
    "Yogad": _n(f"{NL}.yogad", "Yogad"),
    "Gaddang": _n(f"{NL}.gaddang", "Gaddang"),
    "Malaueg": _n(f"{NL}.malaueg", "Malaueg"),
    "Bugkalot/ Ilongot/ Egongot": _n(f"{NL}.bugkalot", "Bugkalot (Ilongot)"),
    # Ifugao: generic on the group, the four named varieties as leaves
    "Ifugao": f"{NL}.ifugao",
    "Tuwali": _n(f"{NL}.ifugao.tuwali", "Tuwali"),
    "Tuwali-Kele-i": _n(f"{NL}.ifugao.kelei", "Tuwali (Kele-i)"),
    "Ayangan": _n(f"{NL}.ifugao.ayangan", "Ayangan"),
    "Ayangan-Henanga": _n(f"{NL}.ifugao.henanga", "Ayangan (Henanga)"),
    # Baliwon: generic on the group
    "Baliwon": f"{NL}.baliwon",
    "Baliwon-I-Sadanga": _n(f"{NL}.baliwon.sadanga", "Baliwon (Sadanga)"),
    "Baliwon-Fiallig/Fialika": _n(f"{NL}.baliwon.fiallig", "Baliwon (Fiallig)"),
    "Baliwon-Gaddang": _n(f"{NL}.baliwon.gaddang", "Baliwon (Gaddang)"),
    "Baliwon-Miligan": _n(f"{NL}.baliwon.miligan", "Baliwon (Miligan)"),
    # Itneg / Tinggian: Itneg, Itneg/Tinguian and Tingguian are the generic, on the group
    "Itneg": f"{NL}.itneg",
    "Itneg/ Tinguian": f"{NL}.itneg",
    "Tingguian": f"{NL}.itneg",
    # Kalinga: Kalinga and Calinga are the generic, on the group
    "Kalinga": f"{NL}.kalinga",
    "Calinga": f"{NL}.kalinga",
})
_n(f"{NL}.ifugao", "Ifugao")
_n(f"{NL}.baliwon", "Baliwon")
_n(f"{NL}.itneg", "Itneg (Tinggian)")
_n(f"{NL}.kalinga", "Kalinga")
for sub in ("Adasen", "Balatok", "Banao", "Belwang", "Binongan", "Gubang", "Inlaud", "Illaud",
            "Mabaka", "Maeng", "Masadiit", "Muyadan"):
    lab = f"Itneg/ Tinguian- {sub}" if sub != "Illaud" else "Itneg/ Tinguian-Illaud"
    # Inlaud and Illaud: two spellings of the lowland Itneg
    nid = f"{NL}.itneg.{'inlaud' if sub == 'Illaud' else sub.lower()}"
    NAMES[lab] = nid if sub == "Illaud" else _n(nid, f"Itneg ({sub})")
KALINGA = ["Ab-abaan", "Aciga", "Ableg/Dalupa", "Ammacian", "Balatoc", "Balinciagao",
           "Ballayangan", "Banao", "Bangad", "Basao", "Biga", "Buaya", "Butbut", "Cagaluan",
           "Culminga", "Dacalan", "Dallac", "Dananao", "Dangtalan", "Dao-Angan", "Dugpa", "Gaang",
           "Gaddang", "Guilayon", "Guina-ang", "Gubang", "Limos", "Lubo", "Lubuagan", "Mabaca",
           "Mabongtot", "Magaogao", "Malbong", "Mangali", "Minanga", "Nanong", "Pangol",
           "Pinukpuk", "Poswoy", "Salegseg", "Sumadel", "Taloctoc", "Tanglag", "Tobog",
           "Tongrayan", "Tulgao", "Uma"]
KALINGA_NO_SPACE = {"Aciga", "Ableg/Dalupa", "Ammacian", "Balatoc", "Balinciagao", "Ballayangan",
                    "Dangtalan", "Lubuagan", "Mabongtot", "Magaogao", "Tongrayan"}
for sub in KALINGA:
    lab = f"Kalinga-{sub}" if sub in KALINGA_NO_SPACE else f"Kalinga- {sub}"
    slug = sub.lower().replace("/", "_").replace("-", "")
    NAMES[lab] = _n(f"{NL}.kalinga.{slug}", f"Kalinga ({sub})")

# ---- Sama-Bajaw ----
NAMES.update({
    "Sama/Samal": _n(f"{SB}.sama", "Sama"),
    "Sama Bangingi": _n(f"{SB}.bangingi", "Sama Bangingi"),
    "Sama Badjao": _n(f"{SB}.sama_badjao", "Sama Badjao"),
    # Badjao and Bajau: spellings of one answer
    "Badjao": _n(f"{SB}.badjao", "Badjao"),
    "Bajau": f"{SB}.badjao",
    "Sama Dilaut/ Sama Laut": _n(f"{SB}.sama_dilaut", "Sama Dilaut"),
    "Yakan": _n(f"{SB}.yakan", "Yakan"),
    "Jama Mapun": _n(f"{SB}.jama_mapun", "Jama Mapun"),
})

# ---- Chavacano ----
NAMES.update({
    "Zamboangeño": _n(f"{CH}.chavacano", "Chavacano (Zamboangueño)"),
    "Caviteño-Chavacano": _n(f"{CH}.caviteno", "Caviteño Chabacano"),
    "Cotabateño-Chavacano": _n(f"{CH}.cotabateno", "Cotabateño Chavacano"),
    "Davao-Chavacano": _n(f"{CH}.davaoeno", "Davao Chavacano"),
})

# ---- other ----
NAMES.update({
    "Eskaya": _n("other.eskayan", "Eskayan"),
    "American": "indoeuropean.germanic.english",
    "British": "indoeuropean.germanic.english",
    "Australian": "indoeuropean.germanic.english",
    "Canadian": "indoeuropean.germanic.english",
    "South African": "indoeuropean.germanic.english",
    "Spanish": "indoeuropean.romance.spanish",
    "French": "indoeuropean.romance.french",
    "Italian": "indoeuropean.romance.italian",
    "German": "indoeuropean.germanic.continental.german",
    "Japanese": "japonic.japanese",
    "South Korean": "koreanic.korean",
    "North Korean": "koreanic.korean",
    "Chinese": "sinotibetan.sinitic",
    "Taiwanese": "sinotibetan.sinitic",
    "Indonesian": "austronesian.malayic.indonesian",
    "Turkish": "turkic.turkish",
    "Indian": "indoeuropean.indoaryan",
    "Swiss": "other",
    "Singaporean": "other",
    "Afghan": "other",
    "Other Foreign Ethnicity": "other",
})

# group nodes that hold no census label of their own
_n(NL, "Northern Luzon")
_n(MAN, "Manobo")
_n(MSK, "Mansakan")
_n(MGY, "Mangyan")
_n(SAM, "Sambalic")
_n(AGT, "Agta and Dumagat")
_n(PB, "Panay Bukidnon (Sulod)")
_n(SB, "Sama-Bajaw")

EXCLUDED = {"Not Reported"}


def resolve(label):
    return NAMES.get(label)
