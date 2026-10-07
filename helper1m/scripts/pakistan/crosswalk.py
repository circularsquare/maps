"""Which polygons make up each census tehsil-tier unit: the hand-made part of the join.

The 2023 census tabulates 591 tehsil-tier units (tehsils, talukas, sub-divisions, sub-tehsils)
under 136 districts. No published boundary file has that set. The best fit is OpenStreetMap's
admin_level=7 relations, which were drawn in 2023 and carry most of the recent splits (Lahore's
five tehsils, Peshawar's seven, Mohmand's seven, Tando Allahyar's three...); OCHA's COD-AB v01
(2022) still has older lines in many places. So OSM is the base, COD fills OSM's holes, and
every place the two sets disagree is settled below.

Rules, in order:
  1. A district in WHOLE_DISTRICT is one unit at tehsil level: its census units cannot be placed
     on any polygon set with confidence.
  2. GROUPS: hand-made pairings, each a list of census units and the polygons that hold them.
     A group with several census units on one side is a merge to the common unit -- never a
     split invented for the census side.
  3. Everything else pairs one census unit with one OSM tehsil by name inside the district
     (spelling folded; difflib ratio >= 0.75; 1:1 both ways). Anything left over on either side
     is a hard error, so a new boundary or census release cannot slip through silently.

A polygon term is an OSM tehsil name (as OSM spells it, within the district), or a tuple:
  ("osm&cod", osm_name, cod_pcode)  OSM tehsil cut down to a COD tehsil  (COD has the finer line)
  ("osm-cod", osm_name, cod_pcode)  OSM tehsil minus a COD tehsil
  ("cod-osm", cod_pcode)            COD tehsil minus every OSM tehsil (fills a hole in OSM)
"""

# Table 9's district names (religiondots' district polygons) -> Table 1's
DIST_T9_TO_T1 = {
    "MALAKAND PROTECTED AREA": "MALAKAND DISTRICT",
    "TANDO ALLAHYAR DISTRICT": "TANDO AHYAR DISTRICT",       # Table 1's spelling
}

# OSM tehsils whose district is not the one they mostly overlap on COD's 2022 lines
OSM_DISTRICT_OVERRIDE = {
    "Tohmulk Tehsil": "KHARAN DISTRICT",   # the census prints TOHMULK SUB-TEHSIL under Kharan
}

KARACHI = ["KARACHI CENTRAL DISTRICT", "KARACHI EAST DISTRICT", "KARACHI SOUTH DISTRICT",
           "KARACHI WEST DISTRICT", "KORANGI DISTRICT", "MALIR DISTRICT", "KEAMARI DISTRICT"]

# district -> why it is one unit at tehsil level
WHOLE_DISTRICT = {
    **{k: "Karachi's 2023 sub-divisions share names with OSM's 2022 local-government towns "
          "but not their lines (name-matched pairs disagree with Kontur by up to 4x, e.g. "
          "Baldia 949k census vs 215k Kontur); COD has the 2001 towns" for k in KARACHI},
    "BARKHAN DISTRICT": "the census prints one tehsil for the whole district; OSM has two",
    "DERA BUGTI DISTRICT": "8 census units against 3 OSM tehsils; six sub-tehsils with no "
                           "polygon and no reliable location",
    "DUKI DISTRICT": "4 census units, OSM has one tehsil for the district",
    "GWADAR DISTRICT": "Suntser sub-tehsil has no polygon and no locatable headquarters",
    "NUSHKI DISTRICT": "Dak sub-tehsil has no polygon; OSM has one tehsil for the district",
    "SOHBATPUR DISTRICT": "6 census tehsils, OSM has one tehsil for the district",
    "SURAB DISTRICT": "4 census sub-divisions, OSM has one tehsil for the district",
}

# district -> [([census units], [polygon terms], reason)]
GROUPS = {
    # ---------------- Khyber Pakhtunkhwa
    "BANNU DISTRICT": [(["WAZIR SUB-DIVISION"], ["Gumatti Tehsil"],
                        "same unit, renamed (745 vs 821 km2, 37,262 census vs 37,840 Kontur)")],
    "BATAGRAM DISTRICT": [(["AI TEHSIL"], ["Allai Tehsil"], "same unit (Table 9 prints ALLAI)")],
    "BUNER DISTRICT": [(["MANDANR TEHSIL"], ["Chamla Tehsil"],
                        "same unit under its seat's name (325 vs 304 km2)")],
    "KHYBER DISTRICT": [(["BARA TEHSIL"], ["Bara Tehsil", "Tirah Tehsil"],
                         "Tirah is not a census tehsil; it was cut from Bara")],
    "KOLAI PALAS KOHISTAN DISTRICT": [(["BATTAIRA SUB-DIVISION"], ["Battera Kolai Tehsil"],
                                       "spelling")],
    "LAKKI MARWAT DISTRICT": [
        (["BETTANI SUB-DIVISION"], ["Bettani Tehsil", ("cod-osm", "PK51903")], "OSM + COD rest"),
        (["SARAI NAURANG TEHSIL"], [("cod-osm", "PK51902")], "OSM has no polygon; COD"),
        (["LAKKI MARWAT TEHSIL", "GHAZNI KHEL TEHSIL"], [("cod-osm", "PK51901")],
         "OSM has no polygon; COD predates Ghazni Khel (1,388 + 1,153 census km2 vs 2,318 COD)"),
    ],
    "LOWER DIR DISTRICT": [
        (["SAMARBAGH SUB-DIVISION"], ["Samarbagh Tehsil", "Munda Tehsil"],
         "census area 419 km2 = 277 + 150; Munda is not a census unit"),
        (["TEMERGARA SUB-DIVISION"], ["Timergara Tehsil", "Balambat Tehsil"],
         "census area 576 km2 = 306 + 241; Balambat is not a census unit"),
    ],
    "LOWER KOHISTAN DISTRICT": [(["BANKAND RANOLIA TEHSIL"], ["Ranolia Bankad Tehsil"], "word order")],
    "MALAKAND DISTRICT": [
        (["SWAT RANI ZAI SUB-DIVISION"], ["Swat Ranizai Tehsil", "Thana Baizai Tehsil"],
         "Thana Baizai is not a census unit; 672 census km2 vs 393 + 241, Kontur 493k vs 473k"),
    ],
    "NORTH WAZIRISTAN DISTRICT": [(["DATTA KHEL TEHSIL"], ["Datta Khel Tehsil", "Shawal Tehsil"],
                                   "Shawal is not a census tehsil; areas put it in Datta Khel "
                                   "(1,807 census km2 vs 1,725 + 187)")],
    "ORAKZAI DISTRICT": [
        (["CENTRAL TEHSIL"], ["Central Orakzai Tehsil"], "name"),
        (["LOWER TEHSIL"], ["Lower Orakzai Tehsil"], "name"),
        (["UPPER TEHSIL"], ["Upper Orakzai Tehsil"], "name"),
    ],
    "SOUTH WAZIRISTAN DISTRICT": [(["LADHA TEHSIL", "MAKIN TEHSIL", "SARAROGHA TEHSIL", "SHAKTOI TEHSIL"],
                                   ["Ladha Tehsil", "Makin Tehsil"],
                                   "OSM has no Sararogha or Shaktoi; Sararogha village lies in "
                                   "OSM's Makin, areas 1,683 census vs 1,545 OSM km2")],
    "SWAT DISTRICT": [
        (["BEHRAIN TEHSIL"], ["Behrain Tehsil", "Kalam Tehsil"],
         "Kalam is not a census tehsil; 2,899 census km2 vs 949 + 2,160"),
        (["MATTA TEHSIL"], ["Matta Shamizai Tehsil", "Matta Sebujani Tehsil"],
         "OSM splits Matta; 684 census km2 vs 295 + 385"),
    ],
    "TORGHAR DISTRICT": [(["JUDBA TEHSIL", "DAUR MERA TEHSIL"], ["Judba Tehsil"],
                          "OSM has no Daur Mera; 63 + 86 census km2 vs 206")],
    "UPPER CHITRAL DISTRICT": [(["MASTUJ SUB-DIVISION"],
                                ["Mastuj Tehsil", "Buni Tehsil", "Torkhow-Molkhow Tehsil"],
                                "the census prints one sub-division for the whole district")],
    "UPPER DIR DISTRICT": [(["DIR SUB-DIVISION", "LARJUM SUB-DIVISION", "SHARINGAL SUB-DIVISION"],
                            ["Dir Tehsil", "Larjam Tehsil", "Sheringal Tehsil", "Barawal Tehsil",
                             "Kalkot Tehsil", "Khall Tehsil"],
                            "OSM has 7 tehsils for the census's 4 sub-divisions and the areas do "
                            "not say which of Barawal, Kalkot, Khall belongs where; Wari matches")],
    "UPPER KOHISTAN DISTRICT": [(["HARBAN BHASHA TEHSIL"], ["Bhasah Harban Tehsil"], "word order")],
    # ---------------- Punjab
    "CHAKWAL DISTRICT": [(["TALA GANG TEHSIL"], ["Talagang Tehsil", "Multan Khurd Tehsil"],
                          "Multan Khurd tehsil was cut from Talagang after the census")],
    "DERA GHAZI KHAN DISTRICT": [
        (["KOH-E-SULEMAN TEHSIL"], ["Tribal Area"], "the de-excluded tribal area (5,339 vs 5,488 km2)"),
        (["KOT CHHUTTA TEHSIL"], [("osm&cod", "Dera Ghazi Khan Tehsil", "PK60703")],
         "OSM folds Kot Chhutta into Dera Ghazi Khan; COD has its line"),
        (["DERA GHAZI KHAN TEHSIL"], [("osm-cod", "Dera Ghazi Khan Tehsil", "PK60703")],
         "the rest of OSM's Dera Ghazi Khan"),
    ],
    "GUJRANWALA DISTRICT": [(["WAZIRABAD TEHSIL"], ["Wazirabad Tehsil", "Ali Pur Chatha Tehsil"],
                             "Ali Pur Chatha is not a census tehsil; 1,196 census km2 vs 530 + 649")],
    "GUJRAT DISTRICT": [(["GUJRAT TEHSIL"], ["Gujrat Tehsil", "Jalalpur Jattan Tehsil"],
                         "Jalalpur Jattan is not a census tehsil; 1,463 census km2 vs 730 + 721")],
    "KHUSHAB DISTRICT": [
        (["NOWSHERA TEHSIL"], [("osm&cod", "Khushab Tehsil", "PK61602")],
         "OSM folds Nowshera into Khushab; COD has its line"),
        (["KHUSHAB TEHSIL"], [("osm-cod", "Khushab Tehsil", "PK61602")], "the rest"),
    ],
    "RAJANPUR DISTRICT": [(["DE-EXCLUDED AREA RAJANPUR"], ["Tribal Area"], "same area")],
    "RAWALPINDI DISTRICT": [
        (["RAWALPINDI TEHSIL"], ["Rawalpindi City Tehsil", "Rawalpindi Saddar Tehsil",
                                 "Rawalpindi Cantonment Tehsil", "Chaklala Cantonment"],
         "the census's one Rawalpindi tehsil; 1,682 census km2 vs 1,594"),
        (["KAR SAYADDAN TEHSIL"], ["Kallar Syedan Tehsil"], "spelling"),
    ],
    "SARGODHA DISTRICT": [(["SHAHPUR TEHSIL"], ["Shahpur Saddar Tehsil"], "name")],
    # ---------------- Sindh
    "BADIN DISTRICT": [(["GOLARCHI (S.F.RAHU) TALUKA"], ["Shaheed Fazil Rahu Taluka"], "same taluka")],
    "HYDERABAD DISTRICT": [(["HYDERABAD TALUKA"], ["Hyderabad Rural Taluka"], "same taluka")],
    "KAMBAR SHAHDAD KOT DISTRICT": [(["KAMBAR ALI KHAN TALUKA"], ["Qambar Taluka"], "same taluka")],
    "KHAIRPUR DISTRICT": [(["MIRWAH TALUKA"], ["Thari Mirwah Taluka"], "same taluka")],
    "TANDO AHYAR DISTRICT": [(["TANDO AHYAR TALUKA"], ["Tando Allahyar Taluka"], "Table 1's spelling")],
    # ---------------- Islamabad
    "ISLAMABAD DISTRICT": [(["ISLAMABAD TEHSIL"], [("cod-osm", "PK40101")], "OSM has no tehsil")],
    # ---------------- Balochistan. Sub-tehsils OSM lacks are put inside the OSM tehsil that holds
    # their headquarters village (OSM place node), else where the census areas leave room.
    "AWARAN DISTRICT": [(["AWARAN SUB-DIVISION", "GISHKORE TEHSIL", "JHAL JAO TEHSIL", "KORAK JHAO TEHSIL"],
                         ["Awaran Tehsil", "Jhal Jhao Tehsil"],
                         "OSM has no Gishkore or Korak Jhao and neither can be placed")],
    "CHAGAI DISTRICT": [(["DALBANDIN SUB-DIVISION", "YAK MACHH SUB-TEHSIL"], ["Dalbandin Tehsil"],
                         "Yakmach village is in OSM's Dalbandin; 7,791 + 7,572 census km2 vs 15,018")],
    "HARNAI DISTRICT": [(["SHAHRIG TEHSIL", "KHOAST SUB-DIVISION"], ["Shahrag Tehsil"],
                         "Khost village is in OSM's Shahrag")],
    "KACHHI DISTRICT": [
        (["DHADAR SUB-DIVISION", "BALANARI SUB-TEHSIL"], ["Dhadar Tehsil"],
         "areas: 976 + 402 census km2 vs 1,830"),
        (["MACH SUB-DIVISION", "KHATTAN SUB-TEHSIL"], ["Machh Tehsil"],
         "areas: 708 + 277 census km2 vs 1,423"),
    ],
    "KALAT DISTRICT": [(["KALAT SUB-DIVISION", "GAZG SUB-TEHSIL", "JOHAN SUB-TEHSIL"], ["Qalat Tehsil"],
                        "areas: 3,788 + 1,390 + 1,328 census km2 vs 8,963; Mangochar matches alone")],
    "KECH DISTRICT": [
        (["TURBAT SUB-DIVISION", "HOSHAB SUB-TEHSIL"], ["Turbat Tehsil"], "Hoshab town is in OSM's Turbat"),
        (["BULAIDA SUB-DIVISION", "ZAMORAN SUB-TEHSIL"], ["Buleda Tehsil"],
         "areas: 1,997 + 1,462 census km2 vs 3,977; the only tehsil with room"),
    ],
    "KHARAN DISTRICT": [
        (["KHARAN SUB-DIVISION", "PATKAIN SUB-TEHSIL"], ["Kharan Tehsil"],
         "areas: 2,941 + 2,131 census km2 vs 4,279"),
        (["TOHMULK SUB-TEHSIL"], ["Tohmulk Tehsil"], "name; OSM's polygon straddles COD's Washuk line"),
    ],
    "KHUZDAR DISTRICT": [
        (["WADH SUB-DIVISION", "ARANJI SUB-TEHSIL", "ORNACH SUB-DIVISION"], ["Wadh Tehsil"],
         "areas: Wadh is 9,250 OSM km2 against 2,118 census; Aranji and Ornach lie south of Khuzdar"),
        (["MOOLA SUB-TEHSIL", "KARAKH SUB-DIVISION"], ["Mola Tehsil"],
         "Karkh village is on OSM's Mola/Khuzdar line; Mola has the room (4,186 vs 3,283 km2)"),
        (["NAL SUB-DIVISION", "GRESHA SUB-TEHSIL"], ["Naal Tehsil"], "Greshak village is in OSM's Naal"),
    ],
    "KILLA SAIFULLAH DISTRICT": [
        (["KILLA SAIFULLAH SUB-DIVISION", "BADINI SUB-TEHSIL", "SHINKI SUB-TEHSIL"], ["Qilla Saifullah Tehsil"],
         "areas: 1,103 census km2 against 8,537 OSM"),
        (["MUSLIM BAGH TEHSIL", "KAN MEHTARZAI SUB-TEHSIL"], ["Muslim Bagh Tehsil"],
         "Kan Mehtarzai is the pass above Muslim Bagh"),
    ],
    "KOHLU DISTRICT": [(["KOHLU SUB-DIVISION", "GRISANI SUB-TEHSIL", "TAMBOO SUB-TEHSIL"], ["Kohlu Tehsil"],
                        "areas: 231 + 174 + 536 census km2 vs 927")],
    "LASBELA DISTRICT": [(["SONMIANI/WINDER TEHSIL"], ["Sonmiani Tehsil"], "name")],
    "LORALAI DISTRICT": [(["BORI SUB-DIVISION"], ["Loralai Tehsil"], "Bori is Loralai's tehsil")],
    "MASTUNG DISTRICT": [(["MASTUNG SUB-DIVISION", "KHAD KOCHA SUB-TEHSIL"], ["Mastung Tehsil"],
                          "Khad Kocha lies between Mastung and Quetta; Kontur 154k vs 162k census "
                          "fits better with it than Dasht")],
    "MUSAKHEL DISTRICT": [(["DRUG TEHSIL", "MUSAKHEL TEHSIL", "TIYAR ESSOT SUB-TEHSIL", "TOISAR TEHSIL",
                            "ZIMRI PLASEEN SUB-TEHSIL"], ["Drug Tehsil", "Musakhel Tehsil"],
                           "three census units with no polygon, between Drug and Musakhel")],
    "NASIRABAD DISTRICT": [(["DERA MURAD JAMALI SUB-DIVISION", "LANDHI TEHSIL", "MIR HASSAN KHOSA TEHSIL"],
                            ["Dera Murad Jamali Tehsil"], "areas: 281 census km2 against 982 OSM")],
    "PANJGUR DISTRICT": [
        (["JAHEEN PAROME TEHSIL"], ["Paroom Tehsil"], "name"),
        (["PANJGUR SUB-DIVISION", "GOWARGO SUB-DIVISION", "KAG SUB-TEHSIL"], ["Panjgur Tehsil", "Gowargo Tehsil"],
         "Kag (Kallag in Table 9) cannot be placed between the two"),
    ],
    "PISHIN DISTRICT": [
        (["KAREZAT SUB-DIVISION", "BOSTAN TEHSIL"], ["Karezat Tehsil"], "Bostan town is in OSM's Karezat"),
        (["SARANAN TEHSIL", "NANA SAHIB TEHSIL"], ["Saranan Tehsil"],
         "areas: 83 + 804 census km2 vs 1,032; Kontur 101k vs 110k"),
    ],
    "QUETTA DISTRICT": [
        (["SUB-DIVISION CITY", "SUB-DIVISION SARIAB"], ["Quetta City Tehsil"], "Sariab is southern Quetta city"),
        (["SUB-DIVISION SADDAR TEHSIL", "SUB-DIVISION KUCHLAK"], ["Quetta Saddar Tehsil"],
         "Kuchlak town is in OSM's Quetta Saddar"),
    ],
    "ZHOB DISTRICT": [
        (["ZHOB SUB-DIVISION", "SAMBAZA SUB-TEHSIL"], ["Zhob Tehsil"], "Sambaza village is in OSM's Zhob"),
        (["KAKAR KHURASAN SUB-DIVISION", "ASHWAT SUB-TEHSIL", "KASHATOO SUB-TEHSIL"], ["Kakar Khurasan Tehsil"],
         "areas: 1,286 census km2 against 4,605 OSM, Zhob has none to spare"),
    ],
}
