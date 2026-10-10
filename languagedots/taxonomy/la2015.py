"""Laos 2015 census ETHNIC GROUP -> the language node it is drawn as. A proxy: the census asked no
language question. Keyed by Table P2.7's English spelling, as sources/la_census.py writes it.

Anita allowed the proxy on 2026-10-05 (AGENT_BRIEF section 2). The village file gives ten
categories; sources/la_census.py splits them into these 49 groups on the 2011 agricultural
census's village groups and rakes to P2.7's national totals (sources/la.md). Since 2026-10-09
sources/la_mics.py moves part of each non-Tai group onto Lao (SHIFT below).

Glottolog checks (data/raw/glottolog, values.csv's Austroasiatic subclassification): Prai is
`phai1238` under Mal-Phai (Khmuic); Phong-Kniang `phon1246`, O'du and Then `thee1239` together
under Khmuic; Bit `bitt1240`, Lamet `lame1256` and Samtao `samt1238` in Palaungic; the Vietic
Phong dialect `phon1243` sits under Hung with Tum `toum1239` and Liha; Nguon `nguo1239` beside
Muong; May is a Chut dialect `mayy1239`; Kri is in Maleng `male1282`. Makong is Bru
(`mang1379`, Eastern Bru), Tri a Bru dialect `trii1240`, Kriang `ngeq1245`, Pacoh `paco1243`,
Katang `kata1264`, Kuy `kuyy1240`; Jeh `jehh1245`, Alak `alak1253`, Oy `oyyy1238` (Cheng
`jeng1241` is an Oy dialect), Nyaheun `nyah1249`, Lave a Brao variety.
"""
AA = "austroasiatic"
KD = "kradai"
HM = "hmongmien"
LL = "sinotibetan.loloish"

NAMES = {
    # Tai (Kra-Dai)
    "Lao": f"{KD}.lao",
    "Tai": f"{KD}.tai_vietnam",        # Tai Dam, Tai Daeng, Tai Khao: Vietnam's Thai node
    "Phouthay": f"{KD}.phu_thai",
    "Lue": f"{KD}.tai_lue",
    "Nhoaun": f"{KD}.northern_thai",   # Nyuan = Tai Yuan, Northern Thai (nod)
    "Yang": f"{KD}.giay",              # the Yang of Phongsaly speak Nhang (Giay)
    "Xaek": f"{KD}.saek",
    "Thaineau": f"{KD}.tai_nua",
    # Khmuic
    "Khmou": f"{AA}.khmuic.khmu",
    "Pray": f"{AA}.khmuic.prai",
    "Xingmoun": f"{AA}.khmuic.puoc",
    "Phong (Khmuic category)": f"{AA}.khmuic.phong_kniang",
    "Thaen": f"{AA}.khmuic.then",
    "Oedou": f"{AA}.khmuic.iduh",
    # Palaungic: leaves under Austroasiatic, as Vietnam's Khang
    "Bid": f"{AA}.bit",
    "Lamed": f"{AA}.lamet",
    "Samtao": f"{AA}.samtao",
    # Vietic: leaves under Austroasiatic, as Vietnam's Muong. Census "Phong" is split by the
    # census's own ethno-linguistic category (sources/la_census.py ALSO_IN)
    "Phong (Vietic category)": f"{AA}.phong_vietic",
    "Toum": f"{AA}.tum",
    "Ngouan": f"{AA}.nguon",
    "Moy": f"{AA}.may",
    "Kree": f"{AA}.kri",
    # Katuic
    "Katang": f"{AA}.katuic.katang",
    "Makong": f"{AA}.katuic.bru",
    "Tri": f"{AA}.katuic.tri",
    "Ta-oy": f"{AA}.katuic.taoih",
    "Katu": f"{AA}.katuic.katu",
    "Kriang": f"{AA}.katuic.kriang",
    "Xuay": f"{AA}.katuic.kuy",
    "Pacoh": f"{AA}.katuic.pacoh",
    # Bahnaric (and Khmer, which K4D files with them)
    "Yrou": f"{AA}.bahnaric.laven",
    "Trieng": f"{AA}.bahnaric.gie_trieng",
    "Yae": f"{AA}.bahnaric.jeh",
    "Brao": f"{AA}.bahnaric.brao",
    "Harak": f"{AA}.bahnaric.alak",
    "Oy": f"{AA}.bahnaric.oy",
    "Cheng": f"{AA}.bahnaric.cheng",
    "Sadang": f"{AA}.bahnaric.sedang",
    "Nhaheun": f"{AA}.bahnaric.nyaheun",
    "Lavy": f"{AA}.bahnaric.lavi",
    "Khmer": f"{AA}.khmer",
    # Hmong-Mien. "Ewmien" is the Iu Mien group, which in Laos takes in the Lanten (Kim Mun):
    # Vietnam's Dao node, which says both.
    "Hmong": f"{HM}.hmong",
    "Ewmien": f"{HM}.dao",
    # Loloish and Chinese
    "Akha": f"{LL}.akha",
    "Pounoy": f"{LL}.phunoi",
    "Lahou": f"{LL}.lahu",
    "Syla": f"{LL}.sila",
    "Hayi": f"{LL}.hani",              # Ha Nhi = Hani
    "Lolo": f"{LL}.lolo_vietnam",
    "Hor": "sinotibetan.sinitic.mandarin",   # Yunnanese Haw Chinese; Mandarin not split by subgroup (Anita, 2026-10-05)
    # Other ethnic groups, not stated, and the 45,538 foreign citizens: one residual per village
    "Other, not stated and foreigners": "other",
}

EXTRA_NODES = []

# sources/la_mics.py moves each non-Tai group's Lao-speaking share (LSIS III 2023, children's
# home language) onto rows labelled "Lao-speaking <group>"; they are drawn as Lao.
SHIFT = "Lao-speaking "


def resolve(label):
    if label.startswith(SHIFT) and label[len(SHIFT):] in NAMES:
        return NAMES["Lao"]
    return NAMES.get(label)
