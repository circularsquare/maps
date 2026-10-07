"""Cameroon: home language from five pooled Afrobarometer rounds (2013-2022), by unit.

    python sources/cm_afro.py --fetch   extract Cameroon's rows from religiondots' merged .sav
                                        files (read-only) -> data/raw/cm/ab_cm_language.csv
    python sources/cm_afro.py           -> data/normalized/cm.csv   (unit x answer, counts)
                                           data/normalized/cm_department.csv (department x
                                           answer shares, placement only)

Cameroon's 2005 census asked neither language nor ethnicity, so this is the survey route of
AGENT_BRIEF §2: shares from the survey times COD-PS 2025 unit totals (religiondots' cm_lookup.csv,
the base its religion map stands on), every row `modelled`. Units are religiondots' twelve: the
ten regions with Yaoundé (Mfoundi) and Douala (Wouri) cut out, because every round samples the
two cities apart.

LINGUA FRANCAS (ask 018). French, English, Cameroonian Pidgin and Fulfulde are taken from
LF_ROUNDS only; every other answer from all five rounds among the non-lingua-franca answers,
scaled to what the four leave (Tanzania's construction, sources/tz_afro.py). Rounds 5 and 6 ask
"Which language is your home language?"; rounds 7 to 9 "language spoken in home", and the answer
moves: French 16.6, 15.6 -> 30.4, 37.9, 49.2% of all respondents, English 0.9, 0.5 -> 7.2, 11.2,
16.4% (95% of Nord-Ouest in round 9). That is the language used, not the first language the rest
of the map draws, so LF_ROUNDS was [5, 6]. Since Anita's ruling (2026-10-05) it is "R7Q2A": the
four are set from R7's separate mother-tongue question (r7_targets). The record is sources/cm.md.
"""
import os
import re
import sys
import unicodedata
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
RD = HERE.parent / "religiondots"
AB_DIR = RD / "data" / "raw" / "afrobarometer"
RD_GEO = RD / "data" / "geo" / "cm"
LOOKUP = RD_GEO / "cm_lookup.csv"            # 12 units, COD-PS 2025
DEPTS = RD_GEO / "cm_departments.csv"        # COD-AB department -> unit
ADM2 = RD / "data" / "raw" / "cm" / "shp" / "cmr_admin2.shp"
RAW = HERE / "data" / "raw" / "cm"
EXTRACT = RAW / "ab_cm_language.csv"
OUT = HERE / "data" / "normalized" / "cm.csv"
OUT_DEPT = HERE / "data" / "normalized" / "cm_department.csv"

N_UNITS = 12
SOURCE_ID = "afrobarometer_r5_r9_cameroon"

# (round, file, language, verbatim, weight, ethnic group, its verbatim, interview language)
ROUNDS = [
    (5, "merged-round-5-data-34-countries-2011-2013-last-update-july-2015_0.sav",
     "Q2", "Q2OTHER", "withinwt", "Q84", "Q84OTHER", "Q103"),
    (6, "merged_r6_data_2016_36countries2.sav", "Q2", "Q2OTHER", "withinwt", "Q87", "Q87OTHER",
     "Q103"),
    (7, "r7_merged_data_34ctry.release.sav", "Q2B", "Q2BOTHER", "withinwt", "Q84", "Q84OTHER",
     "Q103"),
    (8, "afrobarometer_release-dataset_merge-34ctry_r8_en_2023-03-01.sav",
     "Q2", "Q2OTHER", "withinwt_hh", "Q81", "Q81OTHER", "Q103"),
    (9, "R9.Merge_39ctry.20Nov23.final_.release_Updated.4Jun25-3.sav",
     "Q2", "Q2OTHER", "withinwt_hh", "Q84A", "Q84AOTHER", "Q102"),
]
LANG_LABEL = ("language of respondent", "language spoken in home")
ETH_LABEL = ("tribe or ethnic group", "ethnic community")


def say(ok, msg):
    print(("  ok  " if ok else "  FAIL") + "  " + msg)
    if not ok:
        raise SystemExit("check failed: " + msg)


def gkey(s):
    """A label as bare letters. Round 6 is read as LATIN1, so its UTF-8 accents arrive as `Ã©`;
    undo that first (religiondots' sources/cm.py)."""
    s = str(s)
    try:
        s = s.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z]", "", s.casefold())


# ---------------------------------------------------------------- extract

def fetch():
    import pyreadstat
    out = []
    for rnd, name, q, qo, wt, eth, etho, il in ROUNDS:
        p = AB_DIR / name
        if not p.exists():
            raise SystemExit(f"{p} missing: religiondots keeps the merged rounds")
        try:
            _, meta = pyreadstat.read_sav(str(p), metadataonly=True)
            enc = {}
        except Exception:  # noqa: BLE001  R6 is not valid UTF-8
            _, meta = pyreadstat.read_sav(str(p), metadataonly=True, encoding="LATIN1")
            enc = {"encoding": "LATIN1"}
        up = {c.upper(): c for c in meta.column_names}
        locs = [c for c in ("DISTRICT", "LOCATION.LEVEL.1") if c in up]
        want = list(dict.fromkeys(["COUNTRY", "REGION", "RESPNO", "URBRUR", q, qo, wt, eth, etho,
                                   il] + locs))
        lab = str(meta.column_names_to_labels.get(up[q.upper()], "")).casefold()
        say(any(t in lab for t in LANG_LABEL), f"R{rnd} {q} is the home-language question "
            f"({lab!r})")
        elab = str(meta.column_names_to_labels.get(up[eth.upper()], "")).casefold()
        say(any(t in elab for t in ETH_LABEL), f"R{rnd} {eth} is the ethnic group ({elab!r})")
        df, meta = pyreadstat.read_sav(str(p), usecols=[up[c.upper()] for c in want], **enc)
        c = {k: up[k.upper()] for k in want}
        vl = meta.variable_value_labels
        cl = df[c["COUNTRY"]].map(vl.get(c["COUNTRY"], {})).astype(str).str.strip().str.casefold()
        sub = df[cl == "cameroon"]

        def lab_of(k):
            m = vl.get(c[k], {})
            return sub[c[k]].map(m) if m else sub[c[k]]
        w = pd.to_numeric(sub[c[wt]], errors="coerce")
        say(len(sub) >= 1150 and 0.98 <= w.sum() / len(sub) <= 1.02,
            f"R{rnd}: {len(sub):,} Cameroonian respondents; {wt} averages {w.sum() / len(sub):.3f}")
        o = pd.DataFrame({
            "round": rnd, "respno": sub[c["RESPNO"]].astype(str),
            "region_code": pd.to_numeric(sub[c["REGION"]], errors="coerce").astype("Int64"),
            "region": lab_of("REGION"),
            "district": lab_of(locs[0]) if locs else "",
            "urb": lab_of("URBRUR"),
            "lang": lab_of(q), "verbatim": sub[c[qo]].astype(str).str.strip(),
            "eth": lab_of(eth), "eth_verbatim": sub[c[etho]].astype(str).str.strip(),
            "intlang": lab_of(il), "w": w,
        })
        say(o["lang"].notna().all(), f"R{rnd}: every answer code has a label")
        out.append(o)
    a = pd.concat(out, ignore_index=True)
    RAW.mkdir(parents=True, exist_ok=True)
    a.to_csv(EXTRACT, index=False)
    print(f"wrote {EXTRACT} ({len(a):,} respondents)")


# ---------------------------------------------------------------- answers

FRENCH, ENGLISH, PIDGIN, FULFULDE = "French", "English", "Cameroonian Pidgin", "Fulfulde"
LINGUA_FRANCAS = [FRENCH, ENGLISH, PIDGIN, FULFULDE]
# Which rounds set each unit's lingua-franca shares (docstring). Ask 018's switch: "R7Q2A"
# (Anita's ruling, 2026-10-05) takes R7's mother-tongue question (r7_targets); [5, 6] was the
# reading before it; [5, 6, 7, 8, 9] draws every answer as given.
LF_ROUNDS = "R7Q2A"
NON_ANSWERS = {"dontknow", "refusedtoanswer", "refused", "missing"}
OTHER_CM = "Other Cameroonian language"
# remainders that sit on a group node: a place or people holding several languages of one group
BAMILEKE = "Bamileke (language not named)"
GRASSFIELDS = "Grassfields (language not named)"
BANTU = "Bantu (language not named)"
NOT_A_LANGUAGE = "several languages"      # "plusieurs langues": dropped like a non-answer

# The card's labels (keyed through gkey) -> the answer they are counted as. Where the card names
# a Bamileke language by its chief town, the town's language (taxonomy/tree.d/cm.txt):
# Bandjoun Ghomala', Bafang Fe'fe', Dschang Yemba, Bangangté Medumba.
CODED = {
    "french": FRENCH, "english": ENGLISH, "pidgin": PIDGIN,
    "foufoulde": FULFULDE, "fufulde": FULFULDE,
    # Bantu (zone A)
    "ewondo": "Ewondo", "eton": "Eton", "bene": "Bene",
    "bulu": "Bulu", "bula": "Bulu",          # R7-R9's spelling; the same Beti of Sud
    "fong": "Fang",                          # R5-R6's spelling: all 13 answers from Beti-Fang
    "bassa": "Basaa", "yabassi": "Basaa",    # Yabassi: Basaa of Nkam (3, ethnic group Bassa)
    "douala": "Duala", "batanga": "Batanga", "bakundu": "Bakundu", "bafia": "Bafia",
    "maka": "Makaa", "pol": "Pol",
    # Grassfields
    "bamoun": "Bamun", "bamileke": BAMILEKE,
    "bandjoun": "Ghomala'", "bafang": "Fe'fe'", "dschang": "Yemba",
    "bagangte": "Medumba", "bangangte": "Medumba",
    "mbouda": "Mbouda", "bangwa": "Ngwe",    # the Bangwa of Lebialem speak Ngwe (Nweh)
    "banso": "Lamnso'", "lamnso": "Lamnso'",
    "bafut": "Bafut", "mankon": "Mankon", "ngueba": "Ngemba",
    "njikwa": "Ngwo",                        # Njikwa, Momo: the Ngwo language
    "bayangi": "Kenyang",                    # Banyangi, the Kenyang speakers of Manyu
    "tikari": "Tikar",
    "mobakoh": "Mubako",                     # Bali Gashu and Balikumbat answers; see tree.d
    # Chadic, Adamawa, others of the north
    "mafa": "Mafa", "kapsiki": "Kapsiki", "massa": "Massa", "guiziga": "Guiziga",
    "haoussa": "Hausa", "kotoko": "Kotoko", "mousgoum": "Musgum", "djimi": "Jimi",
    "guidar": "Guidar", "guider": "Guidar",  # Guider town, Mayo-Louti: 9 of 14 ethnic Guider
    "moudan": "Mundang", "moudang": "Mundang", "toupouri": "Tupuri",
    "fali": "Fali", "gbaya": "Gbaya",
}
# Card labels whose language depends on place (AGENT_BRIEF §3, India's Pahari):
#   Yamba: in Nord-Ouest the Yamba of Donga-Mantung; elsewhere (Ouest, Menoua, Bamileke
#   respondents) Yemba, Dschang's language, which the card does not name in rounds 5-6.
#   Mboum: in Nord-Ouest (Donga-Mantung, ethnic group Wimbum) Limbum; elsewhere Mbum.
PLACE_DEPENDENT = {"yamba": ("Yamba", "Yemba"), "mboum": ("Limbum", "Mbum")}
NW = "CM007"
READ_VERBATIM = {"other", "others"}

# Every free-text answer (upper-cased, spaces collapsed) -> the answer it is counted as. A
# Bamileke or Grassfields chiefdom is read as its language. A language a single respondent names,
# and nothing else does, goes on OTHER_CM afterwards (single_verbatims).
VERBATIM = {
    # lingua francas
    "FOULBE": FULFULDE, "FULAMI": FULFULDE, "FULANE": FULFULDE, "PEULH": FULFULDE,
    "MBORORO": FULFULDE,               # the Mbororo speak Fulfulde
    "FRANCAIS ET EWONDO": "Ewondo", "FRANAIS ET BANSOA": BAMILEKE,  # the local language named
    "PLUSIEURS": NOT_A_LANGUAGE, "PLUSIEURS LANGUES": NOT_A_LANGUAGE,
    # Bantu
    "ETON": "Eton", "BENE": "Bene", "BULU": "Bulu", "FANG": "Fang", "BATANGA": "Batanga",
    "NTOUMOU": "Ntumu", "NTUMOU": "Ntumu",       # Ntumu, a Fang variety of the Vallée du Ntem
    "BASSA": "Basaa", "BASSO": "Basaa", "YABASSI": "Basaa", "DOUALA": "Duala",
    "BAKOKO": "Bakoko", "LE BAKOKO": "Bakoko", "ABO": "Abo",
    "MAKAA": "Makaa", "KAKO": "Kako", "KAKOO": "Kako",
    "NZIME": "Koonzime", "MEZIME": "Koonzime", "MEZIMA": "Koonzime",
    "BADJOUE": "Bajwe'e",
    "BANEN": "Tunen", "TUNEN": "Tunen", "TOUNENE": "Tunen",
    "YAMBASSA": "Yambassa", "LEMANDE": "Nomaande",
    "MBO": "Mbo", "MBO'O": "Mbo",
    "BAKOSSI": "Akoose", "BAKOSI": "Akoose", "AKOSSE": "Akoose",
    "BAKWERI": "Mokpwe", "BAKWERE": "Mokpwe", "BALCWERI": "Mokpwe", "BAKWARI": "Mokpwe",
    "OROKO": "Oroko", "BALOUNDO": "Oroko",       # Balondo, an Oroko people
    "MBA VELE": "Mvele", "NANGA": "Mvele", "NNAGA EBOKO": "Mvele",  # Nanga-Eboko, Mvele country
    "SAWA": BANTU, "MBAMOISE": BANTU, "GOUIFE": BANTU,  # coast; Mbam (several languages)
    # Grassfields: Bamileke
    "GHOMALA": "Ghomala'", "BAFOUSSAM": "Ghomala'", "BAHAM": "Ghomala'",
    "BAHAM/BAFOUSSAM": "Ghomala'", "BAMEDJOU": "Ghomala'", "BAMENDJOU": "Ghomala'",
    "BAMOUGOUM": "Ghomala'", "BAMEGOUM": "Ghomala'", "BAYANGAM": "Ghomala'", "BATIE": "Ghomala'",
    "BAPA": "Ghomala'", "BADENKOP": "Ghomala'", "BAMEKA": "Ghomala'", "BALENG": "Ghomala'",
    "FEFE": "Fe'fe'", "LE NUFI": "Fe'fe'", "FONDJOMEKWET": "Fe'fe'", "FONDJOMOUKOUE": "Fe'fe'",
    "FOTOUNI": "Fe'fe'", "FOUNDJOUK": "Fe'fe'", "BAWAN": "Fe'fe'",   # Haut-Nkam chiefdoms
    "YEMBA": "Yemba", "DSCHANG": "Yemba", "TCHANG": "Yemba",
    "MEDOUMBA": "Medumba", "BALENGOU": "Medumba",
    "BATCHAM": "Ngiemboon", "NGYEMBOON": "Ngiemboon",
    "BABADJOU": "Ngombale", "BABADJON": "Ngombale", "GOMBALE": "Ngombale",
    "MGOMBALE": "Ngombale", "NGOBALE": "Ngombale", "NGOMBALE": "Ngombale",
    "BOUDA": "Mbouda",
    "NWEH": "Ngwe",
    # chiefdoms not placed on one language: Bangou, Babanjou, Bafunda, Bafung, Baton, Balati,
    # Bambele, Bamendjing, Womela, Guemba, Djanch, Geba, Mefi, Banossi
    # Grassfields: the rest
    "KOM": "Kom", "BIKOM": "Kom", "OKU": "Oku",
    "ELAM EQUO": "Oku", "ELAM EQU0": "Oku", "ELAM EGUO": "Oku",   # all ethnic group Oku, Bui
    "META": "Meta'", "METSA": "Meta'", "METU": "Meta'",
    "MOGHAMO": "Moghamo", "BATIBO": "Moghamo", "AMBO-BATIBO": "Moghamo", "MOGHAGNO": "Moghamo",
    "NGI": "Ngie", "NGIE": "Ngie", "NGUO": "Ngwo", "NGWOH": "Ngwo",
    "OSHIE": "Oshie", "OSHE": "Oshie",
    "AGHEM": "Aghem", "WEH": "Weh",
    "ESIMBI": "Esimbi", "ESSIMBI": "Esimbi",
    "BEFANG": "Befang", "OBANG": "Befang",   # Obang: Glottolog's Obang (Befang)
    "BAFANG": "Befang",          # the one free-text Bafang is in Menchum, ethnic group Befang
    "AMBELE": "Ambele", "BAYA": "Gbaya",
    "NCHANI": "Ncane", "NCHANIH": "Ncane", "NONI": "Noni",
    "WIMBUM": "Limbum", "WIBUM": "Limbum", "LIMBUM": "Limbum", "LIMBUN": "Limbum",
    "NLIMBOM": "Limbum", "NKAMBE": "Limbum", "MBU": "Limbum",    # Nkambe and Ndu: Wimbum towns
    "MBEM": "Yamba",                          # Mbem, Donga-Mantung: Yamba
    "NSO": "Lamnso'",
    "NGEMBA": "Ngemba", "GHEMBA": "Ngemba", "MBATU": "Ngemba", "MENDAKWE": "Ngemba",
    "BAFANJI": "Bafanji", "BAFMEN": "Mmen", "BAMESSING": "Kenswei Nsei", "BABUNGO": "Vengo",
    "BUM": "Bum",
    "BALI NYONGA": "Mungaka", "BALINYONGYA": "Mungaka", "MONGHAKA": "Mungaka",
    "MUGAKA": "Mungaka",
    "MUBAKO": "Mubako", "BALIKUMBAT": "Mubako", "MANKONG": "Mubako",
    "MOUDANI": "Mundani", "MOUDINI": "Mundani", "MUNDANI": "Mundani",
    "BAMENDA": GRASSFIELDS, "LEBIALEM": GRASSFIELDS, "WIDIKUM": GRASSFIELDS,
    "MBESAH": GRASSFIELDS, "BESA": GRASSFIELDS,   # Donga-Mantung; not Bui's Mbessa
    # other Bantoid
    "TIKAR": "Tikar", "KANGYANG": "Kenyang",
    "EJAGHAM": "Ejagham", "EDJAGHAM": "Ejagham", "EJAGAM": "Ejagham", "EJAKA": "Ejagham",
    "EJAKAM": "Ejagham", "KEYKA": "Ejagham",
    "MAMBILA": "Mambila",
    # Chadic
    "MANDARA": "Wandala", "WANDALA": "Wandala",
    "MOUSGOUM": "Musgum", "MOUSGOUN": "Musgum", "MOUGOUM": "Musgum", "MUSGUM": "Musgum",
    "KOTOKO": "Kotoko", "MASSA": "Massa", "GUIZIGA": "Guiziga", "GIZZIGARE": "Guiziga",
    "GABSIKI": "Kapsiki", "MEFEWELE": "Kapsiki",   # ethnic group Kapsiki
    "DABA": "Daba", "DABALA": "Daba", "GOUDOUERI": "Daba",   # ethnic group Daba, Mayo-Louti
    "GAVAR": "Gavar", "HINA": "Hina", "MADA": "Mada", "PODOKO": "Podoko",
    "ZOULGO": "Zulgo", "ZOULOU": "Zulgo", "LELE": "Lele", "MATAL": "Matal", "BATA": "Bata",
    # Adamawa
    "MOUNDANG": "Mundang", "DII": "Dii", "DOUROU": "Dii", "DU": "Dii",   # Duru, the Dii
    "MBOUM": "Mbum", "MAMBAY": "Mambai", "MBOUTE": "Vute",
    # Saharan, Arabic, Central Sudanic
    "KANOURI": "Kanuri", "BORNOU": "Kanuri",
    "ARABE": "Shuwa Arabic", "ARABE CHOA": "Shuwa Arabic", "ARABE SHOUA": "Shuwa Arabic",
    "CHOUA": "Shuwa Arabic",
    "SARA": "Sara", "NGAMBAYE": "Ngambay",
    # outside Cameroon
    "IBO": "Igbo", "DBO": "Igbo",            # "DBO", Meme, ethnic group Ibo
    "JUKUM": "Jukun",
}
for _k in ("BANGOU", "BABANJOU", "BAFOUNDA", "BAFUNG", "BATON", "BALATI", "BAMBELE",
           "BAMENDJING", "WOMELA", "GUEMBA", "DJANCH", "GEBA", "MEFI", "BANOSSI"):
    VERBATIM[_k] = BAMILEKE
# not identifiable as a language of one group, or a place or people of several
for _k in ("ASHEM", "ASUMB0", "BABIS", "BOBIS", "BABOUTE", "BACHIOTIA", "BAFAW", "BAFO",
           "BAKASSI", "BALONG", "BAMBILI", "BAMUMBU", "BANVELE", "BAPOUKOU", "BATOMOW",
           "BAYAYUWI", "BAYENAWA", "BELIBONACHI", "BENEGUN", "BENGWI", "BO'O", "BOBLISE",
           "BONGKEN", "BU", "BUDA", "DJEM", "ESU", "ETENGA", "EWODI", "ISU", "KABALAKA", "KAMBU",
           "KLUM", "KOLE", "KOSHIN", "LAKA", "LAMBO", "LAME", "LUTU", "MABI", "MAKIA", "MALIMBA",
           "MANDOUA", "MANGUISSA", "MBANG", "MBE MBE", "MBESSA", "MEKAF", "MENKA", "MESAGIE",
           "MISANJE", "MOCKBIE", "MOZEUILLE", "MPONGO", "MUDELE", "NCHANG", "NDALE", "NDI",
           "NGAMBE", "NGEI", "NGOA", "NGOU", "NOUGOUNOU", "NSIE", "NTEM", "NTENAKO", "NYOKON",
           "PINYIN", "SANAGA", "SO'O", "SORO", "VONVON", "YAMBEN", "YANGUELE", "YEBEKOL",
           ):
    VERBATIM.setdefault(_k, OTHER_CM)
VERBATIM["BALI"] = None     # decided by place in answer(): Nord-Ouest Mubako, else Mungaka


# ---------------------------------------------------------------- units

# Every REGION label over the five rounds, keyed through gkey(), -> the unit's pcode
# (religiondots' sources/cm.py NORM).
NORM = {
    "adamaoua": "CM001", "adamawa": "CM001",
    "centre": "CM002", "centreyaounde": "CM002007", "yaounde": "CM002007", "mfoundi": "CM002007",
    "est": "CM003", "east": "CM003",
    "extremenord": "CM004", "extremenorth": "CM004",
    "littoral": "CM005", "littoraldouala": "CM005004", "douala": "CM005004", "wouri": "CM005004",
    "nord": "CM006", "north": "CM006",
    "nordouest": "CM007", "northwest": "CM007",
    "ouest": "CM008", "west": "CM008",
    "sud": "CM009",
    "sudouest": "CM010",
}
REGION_OF = {"CM002007": "CM002", "CM005004": "CM005"}
DEPT_ALIAS = {"kakey": "kadey", "mkam": "nkam", "koupeetmanengouba": "kupemanenguba",
              "ngoketundjia": "ngoketunjia"}
LOCATION_ROUNDS = [6, 7, 9]


def units(a):
    lut = pd.read_csv(LOOKUP, dtype={"geo_id": str})
    say(len(lut) == N_UNITS, f"religiondots' cm_lookup.csv: {len(lut)} units, "
        f"{int(lut['pop'].sum()):,} people (COD-PS 2025)")
    a["geo_id"] = a["region"].map(gkey).map(NORM)
    say(a["geo_id"].notna().all(), "every REGION label names a unit "
        f"({sorted(set(a.loc[a['geo_id'].isna(), 'region']))})")
    return a, lut


def departments(a):
    """Rounds 6, 7, 9 carry the department; it must agree with the unit (religiondots' check)."""
    d = pd.read_csv(DEPTS, dtype=str)
    say(len(d) == 58, f"{len(d)} COD-AB departments")
    dept = dict(zip(d["department"].map(gkey), d["adm2_pcode"]))
    unit = dict(zip(d["adm2_pcode"], d["unit"]))
    a["adm2"] = None
    for rnd in LOCATION_ROUNDS:
        m = a["round"] == rnd
        k = a.loc[m, "district"].map(gkey).map(lambda s: DEPT_ALIAS.get(s, s))
        p = k.map(dept)
        say(p.notna().all(), f"R{rnd}: every department label names a COD-AB department "
            f"({sorted(set(a.loc[m & p.isna().reindex(a.index, fill_value=False), 'district']))})")
        bad = p.map(unit) != a.loc[m, "geo_id"]
        say(not bad.any(), f"R{rnd}: {int(m.sum()):,} departments, all in the unit REGION names")
        a.loc[m, "adm2"] = p
    return a


# ---------------------------------------------------------------- shares

def answer(row):
    lab = gkey(row["lang"])
    if lab in READ_VERBATIM:
        v = " ".join(str(row["verbatim"]).split()).upper()
        if not v or v == "NAN":
            return OTHER_CM
        if v not in VERBATIM:
            raise SystemExit(f"verbatim {v!r} (R{row['round']}, {row['region']}) is not in "
                             "VERBATIM: decide it there")
        if v == "BALI":
            return "Mubako" if row["geo_id"] == NW else "Mungaka"
        return VERBATIM[v]
    if lab in NON_ANSWERS:
        return "non-answer"
    if lab in PLACE_DEPENDENT:
        nw, other = PLACE_DEPENDENT[lab]
        return nw if row["geo_id"] == NW else other
    if lab not in CODED:
        raise SystemExit(f"card label {row['lang']!r} (R{row['round']}) is not in CODED")
    return CODED[lab]


def single_verbatims(a):
    """A language named only in free text, by one respondent in the pool: on OTHER_CM."""
    keep = set(CODED.values()) | {v for v, _ in PLACE_DEPENDENT.values()} \
        | {o for _, o in PLACE_DEPENDENT.values()} | {OTHER_CM, BAMILEKE, GRASSFIELDS, BANTU}
    n = a.groupby("answer").size()
    one = sorted(k for k, v in n.items() if v == 1 and k not in keep)
    a.loc[a["answer"].isin(one), "answer"] = OTHER_CM
    return one


R7_LABELS = {FRENCH: ["French"], ENGLISH: ["English"], PIDGIN: ["Pidgin"],
             FULFULDE: ["Foufouldé"]}


def r7_targets(a):
    """Ask 018 (Anita, 2026-10-05): each unit's lingua-franca shares from R7's mother-tongue
    question (Q2A), shrunk (wafr_afro.shrink) towards the national Q2A share for French,
    English and Pidgin, and for Fulfulde, which has a homeland, towards the unit's R5-R6 share
    x the national Q2A / R5-R6 ratio. -> {geo_id: Series LF -> share}."""
    sys.path.insert(0, str(HERE / "sources"))
    from wafr_afro import r7_mother, shrink
    t = r7_mother("Cameroon", R7_LABELS)
    reg = a[a["round"] == 7].groupby("region")["geo_id"].agg(lambda x: set(x))
    say(all(len(v) == 1 for v in reg) and set(reg.index) == set(t.index) - {"_national"},
        f"R7's {len(reg)} REGION labels are one unit each")
    reg = {k: next(iter(v)) for k, v in reg.items()}
    old = {g: shares(b, [5, 6]) for g, b in a.groupby("geo_id")}
    s56 = a[a["round"].isin([5, 6])]
    out = {}
    for k in LINGUA_FRANCAS:
        if k == FULFULDE:
            ratio = (t.loc["_national", k + "_A"] / t.loc["_national", "n"]) / \
                    (s56.loc[s56["answer"] == k, "w"].sum() / s56["w"].sum())
            prior = pd.Series({r: old[g][k] * ratio for r, g in reg.items()})
            out[k] = shrink(t, k, prior=prior).rename(reg)
        else:
            out[k] = shrink(t, k).rename(reg)
        nat = t.loc["_national", k + "_A"] / t.loc["_national", "n"]
        print(f"  R7 Q2A {k}: {nat:.2%} nationally; drawn by unit " +
              ", ".join(f"{g} {v:.1%}" for g, v in out[k].items()))
    out[PIDGIN] = pidgin_l1(a)
    return {g: pd.Series({k: out[k][g] for k in LINGUA_FRANCAS}) for g in reg.values()}


# Cameroon Pidgin as a first language (Anita, 2026-10-06; the ask 019 route: a cited estimate,
# placed by a stated rule). Neba, Chibaka & Atindogbé (2006, African Study Monographs 27(2):
# 39-61): about 5% of Cameroonians speak it natively; Leclerc's CEFAN page gives the same 5%.
# 5% of the COD-PS 2025 total, spread over the units in proportion to each unit's pooled R5-R9
# Pidgin-at-home share x its population. R7's mother-tongue question (0.8%) is not used for it.
PIDGIN_L1_SHARE = 0.05


def pidgin_l1(a):
    """-> Series geo_id -> Pidgin L1 share. The level is PIDGIN_L1_SHARE of the country; the
    pattern is the pooled R5-R9 weighted Pidgin-at-home share of each unit."""
    lut = pd.read_csv(LOOKUP, dtype=str)
    pop = lut.set_index("geo_id")["pop"].astype(float)
    t = a.groupby("geo_id")["w"].sum()
    p = a[a["answer"] == PIDGIN].groupby("geo_id")["w"].sum().reindex(pop.index).fillna(0.0)
    home = p / t.reindex(pop.index)
    f = PIDGIN_L1_SHARE * pop.sum() / (home * pop).sum()
    s = home * f
    say(s.max() < 0.5, f"Pidgin L1: home-share pattern x {f:.2f} to reach "
        f"{PIDGIN_L1_SHARE:.0%} ({PIDGIN_L1_SHARE * pop.sum():,.0f})")
    nm = dict(zip(lut["geo_id"], lut["name"]))
    print("  Pidgin L1 by unit: " + ", ".join(f"{nm[g]} {v:.1%}" for g, v in
                                              s.sort_values(ascending=False).items() if v > 0))
    return s


def shares(b, lf_rounds, lf=None):
    """Lingua francas from lf_rounds (or `lf`, a Series of set shares); every other answer
    from all rounds among the rest, scaled to what the lingua francas leave. b: respondents of
    one place."""
    if lf is None:
        s = b[b["round"].isin(lf_rounds)]
        n = s["w"].sum()
        lf = pd.Series({k: (s.loc[s["answer"] == k, "w"].sum() / n if n else 0.0)
                        for k in LINGUA_FRANCAS})
    rest = b[~b["answer"].isin(LINGUA_FRANCAS)]
    r = rest.groupby("answer")["w"].sum()
    r = r / r.sum() if r.sum() else r
    return pd.concat([lf, r * (1 - lf.sum())])


def main():
    if "--fetch" in sys.argv:
        fetch()
    a = pd.read_csv(EXTRACT, dtype=str, keep_default_na=False)
    a["round"] = a["round"].astype(int)
    a["w"] = a["w"].astype(float)
    print(f"  {len(a):,} respondents in the extract, rounds {sorted(a['round'].unique())}")
    a, lut = units(a)
    a = departments(a)
    a["answer"] = a.apply(answer, axis=1)

    t = a.groupby(["answer", "round"])["w"].sum().unstack(fill_value=0) / a.groupby("round")["w"].sum()
    print("  lingua francas, weighted share of each round's answers (%):")
    for k in LINGUA_FRANCAS:
        print(f"    {k:20s}" + "".join(f"  R{r} {100 * t.loc[k, r]:5.1f}" for r in t.columns))

    n0 = len(a)
    a = a[~a["answer"].isin({"non-answer", NOT_A_LANGUAGE})].copy()
    print(f"  {n0 - len(a)} non-answers and 'several languages' dropped")
    one = single_verbatims(a)
    print(f"  {len(one)} languages named in free text by one respondent -> {OTHER_CM}: {one}")
    ct = pd.crosstab(a["answer"], a["round"])
    ct["all"] = ct.sum(axis=1)
    print(ct.sort_values("all", ascending=False).to_string())

    pop = lut.set_index("geo_id")["pop"].astype(float)
    total = int(pop.sum())
    nm = dict(zip(lut["geo_id"], lut["name"]))
    tg = r7_targets(a) if LF_ROUNDS == "R7Q2A" else {}
    sh = {g: shares(b, LF_ROUNDS, tg.get(g)) for g, b in a.groupby("geo_id")}
    say(set(sh) == set(pop.index), "every unit has respondents")
    say(all(abs(s.sum() - 1) < 1e-9 for s in sh.values()), "every unit's shares sum to 1")
    n_u = a.groupby("geo_id").size()
    n_lf = a[a["round"].isin([7] if LF_ROUNDS == "R7Q2A" else LF_ROUNDS)].groupby("geo_id").size()
    print(f"  respondents per unit: min {n_u.min()} ({nm[n_u.idxmin()]}), median "
          f"{int(n_u.median())}, max {n_u.max()} ({nm[n_u.idxmax()]}); in R{LF_ROUNDS}: min "
          f"{n_lf.min()} ({nm[n_lf.idxmin()]}), median {int(n_lf.median())}")

    rows = []
    for g, s in sh.items():
        s = s[s > 0]
        f = (s * pop[g]).to_numpy()
        base = np.floor(f)
        k = int(round(pop[g] - base.sum()))
        base[np.argsort(-(f - base))[:k]] += 1
        for (ans, x), c in zip(s.items(), base.astype(int)):
            rows.append((g, ans, x, c))
    df = pd.DataFrame(rows, columns=["geo_id", "answer", "share", "count"])
    say(int(df["count"].sum()) == total, f"drawn total {int(df['count'].sum()):,} = COD-PS 2025")
    nat = df.groupby("answer")["count"].sum().sort_values(ascending=False)
    print("\n  national, as drawn:")
    for k, v in nat.items():
        if v / total >= 0.002:
            print(f"    {k:32s} {v:>12,}  {v / total:6.2%}")
    print(f"    ... {int((nat / total < 0.002).sum())} answers under 0.2%")
    print("\n  lingua francas by unit (%): " )
    for k in LINGUA_FRANCAS:
        x = df[df["answer"] == k].set_index("geo_id")["share"]
        print(f"    {k:20s} " + ", ".join(f"{nm[g]} {100 * v:.0f}" for g, v in x.sort_values(
            ascending=False).items() if v >= 0.01))

    n_by = a.groupby(["geo_id", "answer"]).size()
    df["note"] = [f"share {r.share:.4f}; {n_by.get((r.geo_id, r.answer), 0)} respondents "
                  f"(of {n_u[r.geo_id]})" for r in df.itertuples()]
    df = df[df["count"] > 0]
    res = pd.DataFrame({
        "geo_id": df["geo_id"], "geo_level": "unit", "geo_name": df["geo_id"].map(nm),
        "source_category": df["answer"], "count": df["count"], "tier": "modelled",
        "source_id": SOURCE_ID, "year": "2013-2022", "note": df["note"],
    })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.sort_values(["geo_id", "count"], ascending=[True, False]).to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(res)} rows, {res['source_category'].nunique()} answers)")
    a.to_csv(RAW / "ab_cm_resolved.csv", index=False)
    split_half(a, nm)
    department_shares(a, sh, nm)


# ---------------------------------------------------------------- departments (placement only)

K = 8.0          # prior weight, in respondents: one enumeration area (Nigeria's choice)
NEAREST = 3      # sampled departments an unsampled department borrows from


def department_shares(a, sh, nm):
    """Each language's share in each COD-AB department, for placing dots inside a unit: the unit
    construction department by department (lingua francas from LF_ROUNDS respondents, the rest
    from all), each shrunk to the unit's share with K respondents' weight. A department nobody
    was placed in borrows the inverse-square-distance mean of the NEAREST sampled departments of
    its unit. Placement only: the counts are the unit's."""
    import geopandas as gpd
    g = gpd.read_file(ADM2, engine="pyogrio")
    d = pd.read_csv(DEPTS, dtype=str)
    g = g.drop(columns=["adm2_name"]).merge(
        d[["adm2_pcode", "unit", "department"]].rename(columns={"department": "adm2_name"}),
        on="adm2_pcode")
    say(len(g) == 58, f"{len(g)} COD-AB departments with a unit")
    c = g.to_crs(32633).representative_point()
    g["x"], g["y"] = c.x, c.y
    b = a[a["adm2"].notna()]
    print(f"\n  departments: {len(b):,} respondents (R{LOCATION_ROUNDS}) in "
          f"{b['adm2'].nunique()} of 58 departments")
    rows = []
    old_u = {u: shares(bu, [5, 6]) for u, bu in a.groupby("geo_id")}
    for u, gu in g.groupby("unit"):
        us = sh[u]
        own = {}
        for dp, bd in b[b["geo_id"] == u].groupby("adm2"):
            if LF_ROUNDS == "R7Q2A":
                # R7's mother tongue is read at unit level only: the department's R5-R6 pattern,
                # scaled so the unit's level is R7's
                ou = old_u[u]
                s_lf = bd[bd["round"].isin([5, 6])]
                n_lf = s_lf["w"].sum()
                lf = pd.Series({k: ((s_lf.loc[s_lf["answer"] == k, "w"].sum() + K * ou[k])
                                    / (n_lf + K)) * us.get(k, 0.0) / ou[k] if ou[k] > 0
                                else us.get(k, 0.0) for k in LINGUA_FRANCAS})
                lf = lf.clip(upper=0.95)
            else:
                s_lf = bd[bd["round"].isin(LF_ROUNDS)]
                n_lf = s_lf["w"].sum()
                lf = pd.Series({k: (s_lf.loc[s_lf["answer"] == k, "w"].sum()
                                    + K * us.get(k, 0.0)) / (n_lf + K) for k in LINGUA_FRANCAS})
            rest_u = us.drop(LINGUA_FRANCAS, errors="ignore")
            rest_u = rest_u / max(1e-12, rest_u.sum())
            r = bd[~bd["answer"].isin(LINGUA_FRANCAS)].groupby("answer")["w"].sum()
            r = r.reindex(rest_u.index.union(r.index)).fillna(0.0)
            rest = (r + K * rest_u.reindex(r.index).fillna(0.0)) / (r.sum() + K)
            own[dp] = pd.concat([lf, rest * (1 - lf.sum())])
        samp = list(own)
        for r in gu.itertuples():
            if r.adm2_pcode in own:
                v, how = own[r.adm2_pcode], "sampled"
            elif samp:
                sx = gu.set_index("adm2_pcode").loc[samp]
                d2 = ((sx["x"] - r.x) ** 2 + (sx["y"] - r.y) ** 2).to_numpy()
                near = np.argsort(d2)[:NEAREST]
                w = 1.0 / np.maximum(d2[near], 1e6)
                v = sum(own[samp[i]].mul(w[j]) for j, i in enumerate(near)) / w.sum()
                v = v.fillna(0.0)
                how = "borrowed"
            else:
                v, how = us, "unit"
            for k, x in v.items():
                if x > 0:
                    rows.append((u, r.adm2_pcode, r.adm2_name, k, float(x), how))
    t = pd.DataFrame(rows, columns=["geo_id", "adm2_pcode", "adm2_name", "source_category",
                                    "share", "basis"])
    tot = t.groupby("adm2_pcode")["share"].sum()
    say((tot - 1).abs().max() < 1e-6, "every department's shares sum to 1")
    print(f"  department basis: {t.drop_duplicates('adm2_pcode')['basis'].value_counts().to_dict()}")
    t.to_csv(OUT_DEPT, index=False)
    print(f"wrote {OUT_DEPT} ({len(t)} rows)")
    for u in ("CM007", "CM004"):
        tu = t[t["geo_id"] == u]
        print(f"  {nm[u]}:")
        for dn, x in tu.groupby("adm2_name"):
            top = x.sort_values("share", ascending=False).head(4)
            print(f"    {dn:18s} " + ", ".join(f"{r.source_category} {r.share:.0%}"
                                              for r in top.itertuples()) + f"  [{x['basis'].iloc[0]}]")


def split_half(a, nm):
    """R5-R6 against R7-R9, unit shares of every non-lingua-franca answer 1.5%+ of the pool."""
    x = a[~a["answer"].isin(LINGUA_FRANCAS)]

    def sh(b):
        t = b.groupby(["geo_id", "answer"])["w"].sum()
        return t.div(t.groupby(level="geo_id").sum(), level="geo_id").unstack(fill_value=0)
    h1, h2 = sh(x[x["round"] <= 6]), sh(x[x["round"] >= 7])
    idx = h1.index.intersection(h2.index)
    cols = h1.columns.union(h2.columns)
    h1 = h1.reindex(index=idx, columns=cols, fill_value=0)
    h2 = h2.reindex(index=idx, columns=cols, fill_value=0)
    natw = x.groupby("answer")["w"].sum() / x["w"].sum()
    print(f"\n  split-half, R5-R6 against R7-R9, non-lingua-franca answers, Pearson r across "
          f"{len(idx)} units:")
    for k in natw[natw >= 0.015].sort_values(ascending=False).index:
        r = np.corrcoef(h1[k], h2[k])[0, 1]
        print(f"    {k:28s} {natw[k]:6.1%}   r = {r:+.3f}")
        if k in (OTHER_CM, GRASSFIELDS, BANTU):
            print("          not asserted: a remainder, and rounds 5-6 left the free text blank")
        elif natw[k] >= 0.03 and k not in SPLIT_EXEMPT:
            say(r > 0.8, f"{k}: the two halves agree on where it is (r {r:+.3f})")
        elif k in SPLIT_EXEMPT:
            print(f"          not asserted: {SPLIT_EXEMPT[k]}")


# Measured 2026-10-05. Massa is in Extrême-Nord and Nord in both halves, but its Extrême-Nord share
# runs 1, 11, 3, 24, 1% by round (rural Mayo-Danay enumeration areas sampled or not) and R8 alone
# found 9% in Nord, so r is +0.675 over 12 units. Pooling five rounds is the remedy, not a reason
# to drop it.
SPLIT_EXEMPT = {"Massa": "its level swings with which Mayo-Danay areas a round sampled"}


if __name__ == "__main__":
    main()
