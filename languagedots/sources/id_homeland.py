"""Indonesia: regency and kecamatan homelands for the big languages that had even seeds inside
their province (2026-10-19, session eca0185b-id; sources/id.md §12). Placement only.

    python sources/id_homeland.py        prints each prior's coverage check

Javanese, Sundanese, Madurese, Cirebonese and Betawi are counted by province (2010 census L4.5,
or the share-out for Cirebonese and Betawi). Inside a province they were spread by population
(Cirebonese and Betawi by one Glottolog point each), so Sundanese filled Cirebon, Indramayu and
Serang, and East Java's Madurese off Madura spread over the whole province. No regency table of
language or ethnicity for these provinces is open (sources/id.md §2, §12), so these priors are
homeland knowledge, recorded as such: for each language, a rough share of a place's
regional-language speakers who speak it. countries/id.py uses the prior as that language's seed
in the kecamatan rake, times (1 - the 2020 regency share using no regional language), as every
regional language's seed is; the rake then meets the 2010 province counts and Kontur's people,
so the priors only say where, never how many.

Format: PRIORS[province][node] = (default, {code: share}), where a code is a 2010 regency (4
digits) or kecamatan (7 digits); a kecamatan's own entry beats its regency's, which beats the
default. REST[province] = a node whose prior is 1 minus the others' in the same place (floor
REST_FLOOR): Javanese in Central and East Java.

Sources of the knowledge, per block below: the Sundanese/Javanese line in Brebes and Cilacap
(Suyanto, NUSA 13(4) 2018, which also gives Central Java's 578,164 Sundanese speakers); the
Madurese of the Tapal Kuda (Pasuruan to Banyuwangi: the "Pendalungan" culture area of the
literature on Jember and Probolinggo); Banten Javanese in northern Serang, Cilegon, Kota Serang
and north-west Tangerang; the Cirebon-Indramayu dialect area, with Sundanese in south Indramayu
(Haurgeulis, Gantar) and south-west Cirebon; Betawi in the northern and western Bekasi,
Depok and Tangerang fringe. The numbers are judgement at 5-10 point resolution.
"""

J = "austronesian.javanese"
S = "austronesian.sundanese"
M = "austronesian.madurese"
C = "austronesian.cirebonese"
B = "austronesian.malayic.betawi"
ML = "austronesian.malayic.malay"
BJ = "austronesian.malayic.banjar"
KR = "austronesian.malayic.kerinci"
REST_FLOOR = 0.02


def _k(codes, v):
    return {c: v for c in codes}


# --- West Java (32) ----------------------------------------------------------------------
IM_SOUTH_SUNDA = {"3212010": 0.7, "3212011": 0.8, "3212020": 0.5}    # Haurgeulis, Gantar, Kroya
IM_SOUTH_MIX = {"3212030": 0.2, "3212040": 0.2, "3212210": 0.15}     # Gabuswetan, Cikedung, Anjatan
CB_SW = {"3209011": 0.4, "3209070": 0.3, "3209100": 0.2, "3209101": 0.15, "3209121": 0.15}
#        Pasaleman, Sedong, Beber, Greged, Dukupuntang
SB_PANTURA = ["3213220", "3213210", "3213200", "3213201", "3213190", "3213191"]
#        Blanakan, Legonkulon, Pusakanagara, Pusakajaya, Pamanukan, Sukasari (Subang's coast)
BKS_BETAWI = ["3216090", "3216100", "3216150", "3216081", "3216082", "3216070", "3216010"]
#        Babelan, Tarumajaya, Muara Gembong, Tambun S/U, Cibitung, Setu
BKS_SUNDA = ["3216030", "3216031", "3216021", "3216050", "3216130", "3216121", "3216120",
             "3216110", "3216140"]
#        Cibarusah, Bojongmangu, Serang Baru, Kedungwaringin, Pebayuran, Sukakarya, Sukatani,
#        Tambelang, Cabangbungin (eastern Bekasi, Sundanese villages)

# --- Banten (36) -------------------------------------------------------------------------
SRG_NORTH = ["3604220", "3604240", "3604250", "3604260", "3604261", "3604270", "3604271",
             "3604130", "3604121", "3604120", "3604210", "3604211"]
#        Kramatwatu, Ciruas, Pontang, Carenang, Binuang, Tirtayasa, Tanara, Kragilan, Kibin,
#        Cikande, Bojonegara, Pulo Ampel: Banten Javanese
SRG_MIX = ["3604180", "3604200", "3604190"]                          # Waringinkurung, Anyar, Mancak
TNG_NW = ["3603150", "3603151", "3603140", "3603141", "3603160", "3603161", "3603162"]
#        Kronjo, Mekar Baru, Kresek, Gunung Kaler, Mauk, Kemiri, Sukadiri: Javanese
TNG_NE = ["3603170", "3603180", "3603181", "3603190", "3603200", "3603210"]
#        Rajeg, Sepatan, Sepatan Timur, Pakuhaji, Teluknaga, Kosambi: Betawi (Betawi Ora)
TNG_MID = ["3603120", "3603121", "3603130"]                           # Pasarkemis, Sindang Jaya, Balaraja
TNG_EAST = ["3603050", "3603051", "3603060", "3603070", "3603081"]
#        Curug, Kelapa Dua, Legok, Pagedangan, Cisauk (beside Tangerang Selatan)

# --- Central Java (33) -------------------------------------------------------------------
BREBES_SUNDA = {"3329010": 0.95, "3329020": 0.8, "3329090": 0.7, "3329080": 0.5,
                "3329070": 0.4, "3329120": 0.3, "3329100": 0.2, "3329050": 0.1, "3329060": 0.1}
#        Salem, Bantarkawung, Banjarharjo, Ketanggungan, Larangan, Kersana, Losari, Sirampog,
#        Tonjong
CILACAP_SUNDA = {"3301010": 0.95, "3301020": 0.7, "3301030": 0.6, "3301040": 0.5,
                 "3301060": 0.4, "3301050": 0.3, "3301070": 0.15}
#        Dayeuhluhur, Wanareja, Majenang, Cimanggu, Cipari, Karangpucung, Sidareja

# --- East Java (35): Madurese --------------------------------------------------------------
MADURA = {"3526": 0.97, "3527": 0.98, "3528": 0.98, "3529": 0.97}
TAPAL_KUDA = {"3512": 0.92, "3511": 0.92, "3513": 0.85, "3574": 0.5, "3509": 0.55, "3508": 0.3,
              "3514": 0.2, "3575": 0.35, "3510": 0.15, "3578": 0.2, "3525": 0.08, "3515": 0.08,
              "3507": 0.05, "3573": 0.08}
JEMBER = {**_k(["3509240", "3509250", "3509260", "3509270", "3509280", "3509220", "3509230",
                "3509080", "3509070", "3509090", "3509210", "3509200", "3509190", "3509180",
                "3509170"], 0.85),
          #  Kalisat, Ledokombo, Sumberjambe, Sukowono, Jelbuk, Arjasa, Pakusari, Mayang, Silo,
          #  Mumbulsari, Sukorambi, Panti, Bangsalsari, Tanggul, Sumber Baru (north and east)
          **_k(["3509710", "3509720", "3509730", "3509060", "3509100", "3509110", "3509120"], 0.5),
          #  the town (Kaliwates, Sumbersari, Patrang), Tempurejo, Jenggawah, Ajung, Rambipuji
          **_k(["3509010", "3509020", "3509030", "3509040", "3509050", "3509130", "3509140",
                "3509150", "3509160"], 0.12)}
          #  Kencong, Gumuk Mas, Puger, Wuluhan, Ambulu, Balung, Umbulsari, Semboro, Jombang
          #  (the Javanese south-west)
LUMAJANG = {**_k(["3508190", "3508200", "3508120", "3508180", "3508110", "3508130", "3508100",
                  "3508140", "3508170"], 0.6),
            #  Klakah, Ranuyoso, Randuagung, Kedungjajang, Jatiroto, Sukodono, Rowokangkung,
            #  Padang, Gucialit (north-east)
            "3508160": 0.02}                                          # Senduro (Tengger)
PASURUAN = {**_k(["3514230", "3514240", "3514220", "3514200", "3514210", "3514050",
                  "3514170", "3514160"], 0.65),
            #  Lekok, Nguling, Grati, Rejoso, Winongan, Lumbang, Kraton, Rembang (the east)
            "3514040": 0.02, "3514030": 0.05}                         # Tosari, Puspo (Tengger)
MALANG = _k(["3507110", "3507100", "3507240", "3507260", "3507090", "3507080", "3507070",
             "3507140", "3507130", "3507120", "3507060", "3507050", "3507150", "3507230"], 0.2)
#        Wajak, Poncokusumo, Tumpang, Jabung, Ampelgading, Tirto Yudo, Dampit, Gondanglegi,
#        Bululawang, Turen, Sumbermanjing, Gedangan, Pagelaran, Tajinan (east and south-east)
BANYUWANGI = {"3510210": 0.9, "3510200": 0.5, "3510180": 0.3, "3510050": 0.4, "3510080": 0.3,
              "3510090": 0.3, "3510040": 0.2,
              #  Wongsorejo, Kalipuro, Banyuwangi town, Muncar, Glenmore, Kalibaru, Tegaldlimo
              **_k(["3510170", "3510190", "3510130", "3510120", "3510160", "3510140",
                    "3510171"], 0.02)}
              #  Glagah, Giri, Kabat, Rogojampi, Songgon, Singojuruh, Licin (Osing, counted Jawa)
PROBOLINGGO = {"3513010": 0.02, "3513020": 0.1, "3513230": 0.3, "3513240": 0.4,
               "3513220": 0.3, "3513210": 0.6, "3513030": 0.5, "3513040": 0.7}
#        Sukapura (Tengger), Sumber, Tongas, Sumberasih, Lumbang, Wonomerto, Kuripan, Bantaran
#        (the west, towards Pasuruan); the rest of the regency at 0.85


PRIORS = {
    "32": {
        S: (0.95, {"3209": 0.06, "3274": 0.08, "3212": 0.03, "3210": 0.88, "3208": 0.92,
                   "3213": 0.85, "3215": 0.8, "3216": 0.45, "3275": 0.35, "3276": 0.4,
                   "3201": 0.85, "3271": 0.85, "3214": 0.93,
                   **IM_SOUTH_SUNDA, **IM_SOUTH_MIX, **CB_SW, **_k(SB_PANTURA, 0.4),
                   **_k(BKS_BETAWI, 0.2), **_k(BKS_SUNDA, 0.8)}),
        C: (0.002, {"3209": 0.88, "3274": 0.75, "3212": 0.92, "3210": 0.08, "3213": 0.06,
                    "3215": 0.05, "3208": 0.05,
                    "3212010": 0.25, "3212011": 0.15, "3212020": 0.45,
                    **{k: 0.75 for k in IM_SOUTH_MIX},
                    **{k: 0.95 - v for k, v in CB_SW.items()},
                    **_k(SB_PANTURA, 0.5)}),
        J: (0.01, {"3209": 0.04, "3274": 0.1, "3212": 0.04, "3213": 0.05, "3215": 0.08,
                   "3216": 0.15, "3275": 0.25, "3276": 0.2, "3201": 0.04, "3271": 0.05,
                   "3273": 0.04, "3277": 0.04}),
        B: (0.001, {"3216": 0.3, "3275": 0.35, "3276": 0.35, "3201": 0.07, "3271": 0.04,
                    "3215": 0.03, **_k(BKS_BETAWI, 0.55), **_k(BKS_SUNDA, 0.1)}),
    },
    "36": {
        S: (0.95, {"3603": 0.85, "3604": 0.95, "3671": 0.35, "3672": 0.15, "3673": 0.4,
                   "3674": 0.35, "3673060": 0.05,
                   **_k(SRG_NORTH, 0.1), **_k(SRG_MIX, 0.5), **_k(TNG_NW, 0.2),
                   **_k(TNG_NE, 0.15), **_k(TNG_MID, 0.5), **_k(TNG_EAST, 0.5)}),
        J: (0.01, {"3603": 0.03, "3604": 0.03, "3671": 0.2, "3672": 0.8, "3673": 0.55,
                   "3674": 0.15, "3673060": 0.9,
                   **_k(SRG_NORTH, 0.85), **_k(SRG_MIX, 0.45), **_k(TNG_NW, 0.6),
                   **_k(TNG_NE, 0.15), **_k(TNG_MID, 0.3)}),
        B: (0.001, {"3603": 0.05, "3671": 0.4, "3674": 0.5, "3672": 0.02,
                    **_k(TNG_NE, 0.65), **_k(TNG_NW, 0.15), **_k(TNG_MID, 0.2),
                    **_k(TNG_EAST, 0.3)}),
    },
    "33": {
        S: (0.002, {"3329": 0.01, "3301": 0.01, **BREBES_SUNDA, **CILACAP_SUNDA}),
    },
    "35": {
        M: (0.02, {**MADURA, **TAPAL_KUDA, **JEMBER, **LUMAJANG, **PASURUAN, **MALANG,
                   **BANYUWANGI, **PROBOLINGGO}),
    },
}

# --- Outside Java (2026-10-19, second stage; sources/id.md §13). Regency grain only. --------
# Malay here is BPS's "Melayu" (Riau, Jambi, coastal North Sumatra, Riau Islands, West
# Kalimantan's coast and rivers, Berau and Bulungan); Javanese the transmigration and plantation
# settlements; Banjar its homeland in South Kalimantan and its migrant coasts (Indragiri Hilir,
# Tanjung Jabung, Sampit and Pangkalan Bun, Samarinda and Balikpapan); Bugis the coasts it
# settled (Tanjung Jabung, Kotabaru and Tanah Bumbu, East Kalimantan and Nunukan-Tarakan), where
# its Glottolog point in Bone is too far to place it.
U = "austronesian.buginese"
PRIORS.update({
    "12": {    # North Sumatra: Javanese and Malay on the east coast plain
        J: (0.01, {"1212": 0.5, "1213": 0.5, "1218": 0.5, "1208": 0.45, "1207": 0.4,
                   "1222": 0.4, "1223": 0.4, "1219": 0.35, "1209": 0.35, "1274": 0.4,
                   "1276": 0.4, "1275": 0.25, "1273": 0.25, "1272": 0.15, "1220": 0.1,
                   "1221": 0.1}),
        ML: (0.005, {"1213": 0.3, "1212": 0.15, "1218": 0.25, "1219": 0.5, "1208": 0.25,
                     "1272": 0.6, "1207": 0.2, "1222": 0.15, "1223": 0.2, "1275": 0.15,
                     "1276": 0.15, "1274": 0.15, "1271": 0.2, "1204": 0.05}),
    },
    "14": {    # Riau: Malay everywhere; Javanese in the plantation regencies; Banjar in Inhil
        ML: (0.6, {"1403": 0.35, "1471": 0.3, "1473": 0.4, "1408": 0.7, "1410": 0.85,
                   "1409": 0.6, "1406": 0.7, "1401": 0.85, "1402": 0.55, "1404": 0.5,
                   "1405": 0.45, "1407": 0.55}),
        J: (0.2, {"1405": 0.4, "1404": 0.35, "1402": 0.35, "1407": 0.3, "1406": 0.2,
                  "1409": 0.25, "1403": 0.2, "1471": 0.2, "1473": 0.2, "1408": 0.15,
                  "1410": 0.1, "1401": 0.1}),
        BJ: (0.01, {"1403": 0.35, "1408": 0.03, "1471": 0.02}),
    },
    "15": {    # Jambi: Malay; Javanese, Banjar and Bugis on the Tanjung Jabung coast. Kerinci
               # and Sungai Penuh are Kerinci's: since 2026-10-21 the Kerinci speakers the
               # census coded Melayu are moved to Kerinci (countries/id.py KERINCI_FROM_MALAY,
               # sources/id.md §14), so Malay is low there and Kerinci high
        ML: (0.75, {"1506": 0.4, "1507": 0.4, "1501": 0.03, "1572": 0.08}),
        KR: (0.002, {"1501": 0.95, "1572": 0.9}),
        J: (0.2, {"1506": 0.3, "1507": 0.3, "1508": 0.3, "1502": 0.25, "1505": 0.25,
                  "1501": 0.05, "1572": 0.05}),
        BJ: (0.005, {"1506": 0.15, "1507": 0.15}),
        U: (0.005, {"1506": 0.15, "1507": 0.15}),
    },
    "16": {    # South Sumatra: Javanese transmigrants (OKU Timur, Musi Rawas, Banyuasin ...)
        J: (0.1, {"1609": 0.5, "1605": 0.35, "1607": 0.3, "1602": 0.25, "1606": 0.25,
                  "1603": 0.15, "1604": 0.08, "1601": 0.15, "1608": 0.05, "1610": 0.08,
                  "1611": 0.03, "1671": 0.08, "1672": 0.15, "1673": 0.03, "1674": 0.15}),
    },
    "18": {    # Lampung: Javanese most in the centre and east; Sundanese in the south
        J: (0.7, {"1801": 0.35, "1806": 0.5, "1807": 0.5, "1802": 0.6, "1803": 0.6,
                  "1808": 0.6, "1872": 0.85, "1804": 0.85, "1805": 0.85, "1810": 0.85,
                  "1809": 0.7, "1812": 0.75, "1811": 0.7, "1871": 0.55}),
        S: (0.07, {"1803": 0.15, "1802": 0.12, "1801": 0.08, "1809": 0.1}),
    },
    "21": {    # Riau Islands: Malay but in Batam; Javanese in Batam and Bintan
        ML: (0.85, {"2171": 0.35, "2172": 0.6}),
        J: (0.05, {"2171": 0.3, "2102": 0.15, "2101": 0.1, "2172": 0.1}),
    },
    "61": {    # West Kalimantan: Malay on the coast and rivers, Dayak inland (Dayak keeps points)
        ML: (0.3, {"6101": 0.85, "6111": 0.85, "6106": 0.6, "6112": 0.55, "6104": 0.55,
                   "6171": 0.5, "6172": 0.4, "6105": 0.3, "6107": 0.3, "6108": 0.35,
                   "6109": 0.25, "6110": 0.25, "6102": 0.15, "6103": 0.05}),
        M: (0.005, {"6112": 0.15, "6171": 0.12, "6104": 0.1, "6106": 0.05, "6101": 0.02,
                    "6172": 0.02}),
        J: (0.03, {"6112": 0.08, "6171": 0.08, "6106": 0.08, "6105": 0.06, "6107": 0.05}),
    },
    "62": {    # Central Kalimantan: Banjar on the southern coast and towns, Dayak upriver
        BJ: (0.2, {"6202": 0.45, "6201": 0.4, "6203": 0.35, "6210": 0.35, "6206": 0.5,
                   "6208": 0.35, "6271": 0.35, "6204": 0.3, "6205": 0.25, "6209": 0.2,
                   "6212": 0.15, "6207": 0.1, "6211": 0.05, "6213": 0.08}),
        J: (0.1, {"6201": 0.3, "6202": 0.2, "6208": 0.25, "6206": 0.2, "6207": 0.25,
                  "6203": 0.15, "6210": 0.15, "6271": 0.15, "6213": 0.05, "6211": 0.05}),
    },
    "63": {    # South Kalimantan: Banjar but in the south-east coast's migrant regencies
        BJ: (0.95, {"6301": 0.75, "6302": 0.5, "6310": 0.5, "6309": 0.85, "6372": 0.85}),
        J: (0.02, {"6301": 0.2, "6302": 0.15, "6310": 0.2, "6304": 0.06, "6372": 0.1,
                   "6303": 0.04, "6309": 0.08}),
        U: (0.005, {"6302": 0.2, "6310": 0.2}),
    },
    "64": {    # East Kalimantan, with Kalimantan Utara as the layer's residual unit "65"
        ML: (0.1, {"6405": 0.4, "6403": 0.25, "6404": 0.2, "65": 0.25}),
        BJ: (0.1, {"6472": 0.3, "6471": 0.2, "6403": 0.25, "6401": 0.25, "6409": 0.2,
                   "6405": 0.08, "65": 0.05}),
        J: (0.25, {"6471": 0.3, "6409": 0.35, "6401": 0.3, "6404": 0.3, "65": 0.2,
                   "6402": 0.15}),
        U: (0.15, {"6474": 0.3, "6471": 0.2, "6472": 0.2, "6404": 0.25, "6409": 0.25,
                   "6401": 0.3, "65": 0.35, "6402": 0.03}),
    },
})
REST = {"33": J, "35": J}


def prior(p, node, kec):
    """Prior shares of `node` for kecamatan codes `kec` (7-digit; regency and residual codes
    fall back to their regency or the default), or None when the node has no prior in p."""
    if p in REST and REST[p] == node:
        others = [prior(p, n, kec) for n in PRIORS.get(p, {})]
        return [max(REST_FLOOR, 1 - sum(o[i] for o in others)) for i in range(len(kec))]
    if p not in PRIORS or node not in PRIORS[p]:
        return None
    default, by = PRIORS[p][node]
    return [by.get(k, by.get(k[:4], default)) for k in kec]


def nodes(p):
    return set(PRIORS.get(p, {})) | ({REST[p]} if p in REST else set())


def _check():
    """Every code named is a unit or regency of the hex layer, in the right province, and every
    prior is a share."""
    import csv
    from pathlib import Path
    units = Path(__file__).resolve().parent.parent / "data" / "geo" / "id" / "id_units.csv"
    with open(units, encoding="utf-8") as f:
        us = {r["unit"] for r in csv.DictReader(f)}
    regs = {u[:4] for u in us if len(u) >= 4}
    bad = []
    for p, d in PRIORS.items():
        for n, (default, by) in d.items():
            for code, v in by.items():
                ok = (code in us) if len(code) in (2, 7) else (code in regs)
                # "65", Kalimantan Utara, is the layer's residual unit inside 2010's East
                # Kalimantan (64)
                if not ok or code[:2].replace("65", "64") != p or not 0 <= v <= 1:
                    bad.append((p, n, code, v))
    if bad:
        raise SystemExit(f"id_homeland: codes not in the layer or shares out of range: {bad}")
    for p, d in PRIORS.items():
        kec = sorted(u for u in us if u[:2] == p and len(u) == 7)
        tot = [sum(prior(p, n, [k])[0] for n in d) for k in kec]
        print(f"  {p}: {len(d)} priors, {len(kec)} kecamatan; sum of priors per kecamatan "
              f"{min(tot):.2f}-{max(tot):.2f}")


if __name__ == "__main__":
    _check()
