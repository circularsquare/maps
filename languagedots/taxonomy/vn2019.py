"""Vietnam 2019 census ETHNIC GROUP (dan toc) -> the language node it is drawn as. A proxy: the
census asked no language question. Keyed by Table 2's Vietnamese spelling (NFC).

Anita allowed the proxy on 2026-10-05 (AGENT_BRIEF section 2); every row is `derived`.
sources/vn_census.py splits each minority by the 2024 survey of the 53 minorities (Bieu 3.9,
household language of family communication) into three labels, resolved here:
  "<group>"                                      -> the group's language (NAMES)
  "<group> / Vietnamese at home"                 -> Vietnamese
  "<group> / another minority language at home"  -> seasia_other (the survey does not say which)
sources/vn.md has the reasoning and the misfits. Glottolog checks (data/raw/glottolog, by ISO
code): every leaf below sits in the family and branch its node says, except as commented.
"""
AA = "austroasiatic"
KD = "kradai"
HM = "hmongmien"
CH = "austronesian.chamic"
LL = "sinotibetan.loloish"
SI = "sinotibetan.sinitic"
VI = f"{AA}.vietnamese"

HOME_VI = " / Vietnamese at home"
HOME_OTHER = " / another minority language at home"

NAMES = {
    "Kinh": VI,
    # Vietic (Glottolog Vietic: Muong mtq, Tho/Cuoi tou, Chut scb). Leaves directly under
    # Austroasiatic beside Vietnamese, which already sits there in every other fragment.
    "Mường": f"{AA}.muong",
    "Thổ": f"{AA}.tho",
    "Chứt": f"{AA}.chut",
    # Tai (Kra-Dai)
    "Tày": f"{KD}.tay",
    "Thái": f"{KD}.tai_vietnam",       # Tai Dam (blt) and Tai Don; not Thailand's Thai
    "Nùng": f"{KD}.nung",
    "Sán Chay": f"{KD}.cao_lan",       # the Cao Lan language (mlc); the San Chi half speak a
                                       # Chinese variety the survey does not separate
    "Giáy": f"{KD}.giay",              # a Northern Tai (Bouyei) dialect in Glottolog
    "Bố Y": f"{KD}.bouyei",
    "Lào": f"{KD}.lao",
    "Lự": f"{KD}.tai_lue",
    # Kra (Kra-Dai): La Chi (lbt), La Ha (lha), Gelao, Qabiao (laq)
    "La Chí": f"{KD}.lachi",
    "La Ha": f"{KD}.laha",
    "Cơ Lao": f"{KD}.gelao",
    "Pu Péo": f"{KD}.qabiao",
    # Hmong-Mien
    "Mông": f"{HM}.hmong",
    "Dao": f"{HM}.dao",                # Iu Mien (ium) and Kim Mun (mji); no source splits them
    "Pà Thẻn": f"{HM}.hmongic.pa_hng",  # Pa-Hng (pha), Hmongic
    # Chamic (Austronesian)
    "Gia Rai": f"{CH}.jarai",
    "Ê Đê": f"{CH}.rade",
    "Chăm": f"{CH}.cham",
    "Raglay": f"{CH}.roglai",
    "Chu Ru": f"{CH}.chru",
    # Austroasiatic
    "Khmer": f"{AA}.khmer",
    "Ba Na": f"{AA}.bahnaric.bahnar",
    "Xơ Đăng": f"{AA}.bahnaric.sedang",
    "Cơ Ho": f"{AA}.bahnaric.koho",
    "Hrê": f"{AA}.bahnaric.hre",
    "Mnông": f"{AA}.bahnaric.bunong",
    "Xtiêng": f"{AA}.bahnaric.stieng",
    "Gié Triêng": f"{AA}.bahnaric.gie_trieng",
    "Mạ": f"{AA}.bahnaric.maa",
    "Chơ Ro": f"{AA}.bahnaric.chrau",
    "Co": f"{AA}.bahnaric.cua",
    "Brâu": f"{AA}.bahnaric.brao",
    "Rơ Măm": f"{AA}.bahnaric.romam",
    "Bru Vân Kiều": f"{AA}.katuic.bru",
    "Cơ Tu": f"{AA}.katuic.katu",
    "Tà Ôi": f"{AA}.katuic.taoih",
    "Khơ Mú": f"{AA}.khmuic.khmu",
    "Xinh Mun": f"{AA}.khmuic.puoc",   # Ksingmul (puo)
    "Ơ Đu": f"{AA}.khmuic.iduh",       # O'du (tyh)
    "Kháng": f"{AA}.khang",            # Palaungic (kjm), a leaf like Wa and Blang
    "Mảng": f"{AA}.mang",              # Mang (zng), Mangic
    # Loloish. Lo Lo speak Mantsi (nty), which Glottolog files under Mondzish beside Ngwi; it is
    # put with the Yi languages here, as the Lo Lo are a Yi people and readers know it so.
    "Hà Nhì": f"{LL}.hani",
    "La Hủ": f"{LL}.lahu",
    "Lô Lô": f"{LL}.lolo_vietnam",
    "Phù Lá": f"{LL}.phula",
    "Si La": f"{LL}.sila",
    "Cống": f"{LL}.cong",
    # Chinese. Hoa on `sinitic` itself, as every country files Chinese not split by variety (the
    # Hoa speak Cantonese, Teochew, Hakka, Hokkien...). Ngai speak Hakka. San Diu speak a Yue
    # variety (often called Shan Yao); no Glottolog leaf, so its own leaf under Chinese.
    "Hoa": SI,
    "Ngái": f"{SI}.hakka",
    "Sán Dìu": f"{SI}.san_diu",
    "Người nước ngoài": "other",       # foreign citizens, 3,553: language unknown
}

EXCLUDED = {"Không xác định"}          # not stated, 349 people: in `gap`, not drawn
EXTRA_NODES = ["seasia_other"]


def resolve(label):
    if label.endswith(HOME_VI):
        return VI
    if label.endswith(HOME_OTHER):
        return "seasia_other"
    if label in EXCLUDED:
        return None
    return NAMES.get(label)
