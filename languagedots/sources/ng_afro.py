"""Nigeria: home language from six pooled Afrobarometer rounds (2008-2022), by state.

    python sources/ng_afro.py --fetch   extract Nigeria's rows from religiondots' six merged
                                        .sav files (read-only) -> data/raw/ng/ab_ng_language.csv
    python sources/ng_afro.py           -> data/normalized/ng_afro.csv (state x answer, counts;
                                                                        the comparison build)
                                           data/normalized/ng_lga.csv  (LGA x answer, weighted
                                                                        respondents, placement)

SINCE 2026-10-09 the map is drawn by sources/ng_mics.py (MICS6 2021 for the ten languages MICS
names), which imports this file's harmonising steps for the split of MICS's "other language".
This script's own state build now writes ng_afro.csv, kept as the comparison (sources/ng.md §0).

Nigeria's censuses have asked no language (or ethnicity) question since 1963, so this is the
survey route of AGENT_BRIEF §2: shares from the survey with the most regions, times a population
base, every row `modelled`. The record is sources/ng.md.

THE QUESTION. R4-R6 "Which language is your home language?" (variable label "Language of
respondent"); R7-R9 "Language spoken in home". One answer each. The card lists Nigeria's main
languages and "Other (specify)"; the verbatim of "Other" is in the release, and is used.

LINGUA FRANCAS (ask 018, 2026-10-05). English, Pidgin and Hausa outside Hausaland are drawn at
R7's separate mother-tongue question (Q2A); see state_shares. English and Pidgin answers are
first moved to the respondent's ethnic group's language, so those languages carry the rest.

R4's COMBINED LABEL. R4 coded 172 respondents to "Ijaw/Kalabari/Okirika/Andoni/Ogoni/Nembe";
53 have a verbatim (Okirika, Andoni, Kalabari) and are read from it. The other 119 are shared,
within their state, across the six answers the label names, in proportion to those answers'
weighted counts in the same state in the other rows. All six are named answers in R5-R9.

THE VERBATIMS. A free-text answer is put on the language it names (Glottolog, language level):
a dialect or a town is put on its language ("Agbor" -> Ika, "Auchi" -> Yekhee, "Effon" ->
Yoruba). A language named by a single respondent, and anything not identifiable, goes on
"Other Nigerian language". Each mapping is in VERBATIM below.

POPULATION. COD-PS 2022 state populations (NPC/UNFPA projection off the 2006 census), from
religiondots' ng_lookup.csv, 216,798,930 people. The same base religiondots draws Nigeria on.
"""
import os
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
RAW = HERE / "data" / "raw" / "ng"
EXTRACT = RAW / "ab_ng_language.csv"
LOOKUP = RD / "data" / "geo" / "ng" / "ng_lookup.csv"
ADM2 = RD / "data" / "raw" / "ng" / "shp" / "nga_admin2.shp"
OUT = HERE / "data" / "normalized" / "ng_afro.csv"   # ng.csv is sources/ng_mics.py's since 2026-10-09
OUT_LGA = HERE / "data" / "normalized" / "ng_lga.csv"

CODPS_2022 = 216_798_930
N_STATES = 37
SOURCE_ID = "afrobarometer_r4_r9_nigeria"

# (round, file, language column, verbatim column, weight column, LGA column or None,
#  ethnic group column, its verbatim column)
ROUNDS = [
    (4, "merged_r4_data.sav", "Q3", "Q3OTHER", "Withinwt", "DISTRICT", "Q79", "Q79OTHER"),
    (5, "merged-round-5-data-34-countries-2011-2013-last-update-july-2015_0.sav",
     "Q2", "Q2OTHER", "withinwt", None, "Q84", "Q84OTHER"),
    (6, "merged_r6_data_2016_36countries2.sav", "Q2", "Q2OTHER", "withinwt", "LOCATION.LEVEL.1",
     "Q87", "Q87OTHER"),
    (7, "r7_merged_data_34ctry.release.sav", "Q2B", "Q2BOTHER", "withinwt", "LOCATION.LEVEL.1",
     "Q84", "Q84OTHER"),
    (8, "afrobarometer_release-dataset_merge-34ctry_r8_en_2023-03-01.sav",
     "Q2", "Q2OTHER", "withinwt_hh", None, "Q81", "Q81OTHER"),
    (9, "R9.Merge_39ctry.20Nov23.final_.release_Updated.4Jun25-3.sav",
     "Q2", "Q2OTHER", "withinwt_hh", "LOCATION.LEVEL.1", "Q84A", "Q84AOTHER"),
]
ETH_LABEL = ("tribe or ethnic group", "ethnic community")
N_RESP = {4: 2324, 5: 2400, 6: 2400, 7: 1600, 8: 1599, 9: 1600}
LANG_LABEL = ("language of respondent", "language spoken in home")

ENGLISH_PIDGIN = ["English", "Nigerian Pidgin"]
EP_ROUNDS = [7, 8, 9]
NON_ANSWERS = {"Don't know", "Refused", "Missing"}
OTHER_NG = "Other Nigerian language"
COMBINED = "Ijaw/Kalabari/Okirika/Andoni/Ogoni/Nembe"
COMBINED_PARTS = ["Ijaw", "Kalabari", "Okrika", "Obolo", "Ogoni", "Nembe"]
# coded labels whose verbatim is read instead of the code
READ_VERBATIM = {"Other", "Others", COMBINED, "Okpe"}

# Coded labels that are one answer under two spellings or two names.
CODED = {
    "Fulani": "Fula",
    "Pidgin English": "Nigerian Pidgin",
    "Babur": "Bura-Pabir", "Bura": "Bura-Pabir",          # one language, Bura-Pabir (bura1267)
    "Kataf (Atyap)": "Tyap", "Kataf": "Tyap",
    "Okrika": "Okrika",
    "Nwangavul": "Mwaghavul",
    "Anang": "Anaang",
    "Yakhor": "Yakurr",                                    # all 16 in Cross River (R5): Yakö
    "Ejagam": "Ejagham",
    "Ugep": "Yakurr",                                      # Ugep is the Yakurr town
    "Higgi": "Kamwe",
    "Sayawa": "Zaar",
    "Bajju": "Jju",
    "Kagoma": "Gyong",
}

# Every verbatim (title-cased) -> the answer it is counted as. OTHER_NG where it names a place
# that holds several languages, a word that is not a language, or nothing identifiable.
VERBATIM = {
    # Volta-Niger
    "Ibo": "Igbo", "Ehugbo": "Igbo", "Afikpo": "Igbo", "Aniocha": "Igbo", "Nsukwa": "Igbo",
    "Ezi Delta Igbo": "Igbo", "Delta-Igbo": "Igbo", "Ecthie": "Igbo", "Etche": "Igbo",
    "Ohafia": "Ohafia",
    "Agbor": "Ika", "Ika": "Ika", "Ikah": "Ika", "Akumazi": "Ika",
    "Kwale": "Ukwuani", "Ukwani": "Ukwuani", "Ukwuani": "Ukwuani", "Ukwuani Language": "Ukwuani",
    "Ukuani": "Ukwuani", "Ukwale": "Ukwuani", "Ndokwa": "Ukwuani",
    "Ekpeye": "Ekpeye", "Epeye": "Ekpeye", "Ahoada": "Ekpeye", "Ahuda": "Ekpeye",
    "Ogba": "Ogba", "Egi": "Ogba",
    "Awori": "Yoruba", "Effon": "Yoruba", "Ilawe": "Yoruba",
    "Igara": "Igala", "Ibaji": "Ibaji",
    "Ibira": "Ebira",
    "Gwari": "Gbagyi", "Gbagi": "Gbagyi", "Gbagri": "Gbagyi", "Gbagyi": "Gbagyi",
    "Ga'De": "Gade", "Gade": "Gade", "Gaide": "Gade",
    "Dibbo": "Ganagana", "Gana Gana": "Ganagana", "Ganagana": "Ganagana",
    "Kakanda": "Kakanda",
    "Bassa Nge": "Bassa-Nge",
    "Alago": "Alago", "Arago": "Alago",
    "Igede": "Igede", "Egede": "Igede",
    "Etulo": "Etulo", "Iteulo": "Etulo",
    "Afo": "Eloyi",
    "Yala": "Yala", "Iyala": "Yala", "Iyalla": "Yala", "Cross River Yala": "Yala",
    "Yache": "Yace",
    # Edoid
    "Bini": "Edo",
    "Auchi": "Yekhee", "Etsako": "Yekhee", "Etsako Language": "Yekhee", "Etsakor": "Yekhee",
    "Agbede": "Yekhee", "Ibie": "Yekhee", "Uzairue": "Yekhee",
    "Okpella": "Okpela", "Okpe": "Okpe",
    "Epie": "Epie", "Epia": "Epie", "Eapie": "Epie", "Ipia": "Epie",
    "Owan": "Emai-Iuleha-Ora (Owan)",
    "Isoko": "Isoko", "Isoko Language": "Isoko",
    "Afemai": OTHER_NG, "Sasaro": OTHER_NG, "Somorika": OTHER_NG, "Ososo": OTHER_NG,
    "Ikpeshi": OTHER_NG, "North Ivie": OTHER_NG, "Uzeba": OTHER_NG, "Okpamiri": OTHER_NG,
    # Ijoid
    "Izon": "Ijaw",
    "Kalabari": "Kalabari", "Kalagbari": "Kalabari", "Kakabari": "Kalabari",
    "Okirika": "Okrika", "Okrika": "Okrika",
    "Brass": "Nembe",
    "Ibani": "Ibani", "Igbani": "Ibani",
    # Cross River
    "Andoni": "Obolo", "Obolo": "Obolo",
    "Annang": "Anaang",
    "Khana": "Khana", "Eleme": "Eleme", "Oron": "Oron",
    "Calabar": "Efik",
    "Yakuri": "Yakurr", "Yakur": "Yakurr", "Ugep": "Yakurr",
    "Bette": "Bette-Bendi", "Bethe": "Bette-Bendi", "Betem": "Bette-Bendi", "Obudu": "Bette-Bendi",
    "Bekwarra": "Bekwarra", "Bekwaira": "Bekwarra", "Bikwara": "Bekwarra",
    "Mbembe": "Mbembe", "Adun": "Mbembe",
    "Abua": "Abua",
    # Bantoid, Ekoid, Jukunoid
    "Ejagam": "Ejagham", "Ejagham": "Ejagham",
    "Ekajuk": "Ekajuk", "Ekajiko": "Ekajuk",
    "Jarawa": "Jarawa",
    "Mambilla": "Mambila",
    "Ndola": "Ndoola", "Ndala": "Ndoola",
    "Jukun": "Jukun", "Jukum": "Jukun", "Jukun Wanu": "Jukun", "Jukun Donga": "Jukun",
    "Kona": "Jukun",
    "Kuteb": "Kuteb",
    "Jibu": "Jibu", "Jibawa": "Jibu",
    "Tigun": "Tigon", "Tigum": "Tigon", "Tikum": "Tigon",
    "Ichen": "Etkywan (Ichen)",
    "Jenjo": "Dza (Jenjo)",
    # Plateau, Kainji
    "Birom": "Berom", "Berom": "Berom", "Borom": "Berom", "Burum": "Berom",
    "Eggon": "Eggon",
    "Tarok": "Tarok",
    "Bajju": "Jju", "Baju": "Jju", "Bejju": "Jju", "Beju": "Jju", "Boju": "Jju",
    "Kataf": "Tyap", "Kataf(Atyap)": "Tyap", "Attakar": "Tyap", "Fantswam": "Tyap",
    "Kagoro": "Tyap", "Kagoru": "Tyap",
    "Kagoma": "Gyong", "Kakoma": "Gyong", "Gwong": "Gyong",
    "Adara": "Adara", "Kadara": "Adara",
    "Jaba": "Hyam (Jaba)", "Ham": "Hyam (Jaba)",
    "Ninzom": "Ninzo",
    "Numana": "Numana",
    "Mada": "Mada",
    "Irigwe": "Irigwe",
    "Afizere": "Izere", "Izere": "Izere",
    "Aten": "Aten",
    "Migili": "Migili", "Nigili": "Migili",
    "Koro": "Koro",
    "Chawai": OTHER_NG,
    "Kurama": "Kurama",
    "Kambari": "Kambari",
    "Zuru": "Lela (Dakarkari)", "Zuro": "Lela (Dakarkari)", "Dakarkari": "Lela (Dakarkari)",
    "Dakkarci": "Lela (Dakarkari)",
    "Amoh": OTHER_NG,
    "Basa": "Bassa", "Bassa": "Bassa",
    "Buji": OTHER_NG,
    # Adamawa
    "Mumuye": "Mumuye", "Mumuyhe": "Mumuye", "Muye": "Mumuye", "Mamunye": "Mumuye",
    "Mumugar": "Mumuye",
    "Waja": "Waja",
    "Chamba": "Chamba", "Chamber": "Chamba",
    "Yungur": "Yungur",
    "Lunguda": OTHER_NG,
    "Tula": "Tula",
    "Awak": OTHER_NG,
    # Chadic
    "Angas": "Ngas", "Ngas": "Ngas",
    "Tangale": "Tangale", "Tangali": "Tangale",
    "Bura": "Bura-Pabir", "Babur": "Bura-Pabir",
    "Margi": "Marghi", "Marghi": "Marghi",
    "Karekare": "Karekare", "Kari-Kari": "Karekare",
    "Bada": "Bade",
    "Higgi": "Kamwe", "Michika": "Kamwe", "Michica": "Kamwe", "Michinka": "Kamwe",
    "Minchika": "Kamwe",
    "Mwaghavul": "Mwaghavul", "Mwaghawul": "Mwaghavul", "Mwangavul": "Mwaghavul",
    "Magavoul": "Mwaghavul",
    "Mupun": "Mwaghavul",                                  # a Mwaghavul dialect (Glottolog)
    "Gwandara": "Gwandara",
    "Tera": "Tera",
    "Bole": "Bole", "Bolawa": "Bole", "Balawa": "Bole",
    "Bachama": "Bachama",
    "Sayawa": "Zaar", "Siyawa": "Zaar",
    "Gomai": "Goemai", "Ankwai": "Goemai", "Ankwe": "Goemai", "Aukwe": "Goemai",
    "Ankwoi": "Goemai", "Gimai Ankwen": "Goemai", "Diemak": "Goemai", "Doemark": "Goemai",
    "Chibok": "Kibaku (Chibok)",
    "Kilba": "Huba (Kilba)", "Kilba Babar": "Huba (Kilba)",
    "Glanda": "Glavda",
    "Piapun": "Piapung", "Pyapus": "Piapung",
    "Gwoza": "Gwoza",
    "Pero": OTHER_NG, "Ron": OTHER_NG, "Kulere": OTHER_NG, "Kwalla": OTHER_NG,
    "Fali": OTHER_NG,
    # Saharan, Songhay, Arabic, Mande, Gur, Gbe
    "Kanuri": "Kanuri", "Manga": "Kanuri",
    "Zabarmanci": "Zarma", "Zabarmanchi": "Zarma",
    "Shuwa": "Shuwa Arabic", "Shuwa Arab": "Shuwa Arabic",
    "Bokobaru": "Busa", "Bisa": "Busa", "Bussa": "Busa", "Bogobara": "Busa", "Boko": "Busa",
    "Baruba": "Bariba",
    "Egun": OTHER_NG,
    # Tiv
    "Tiv": "Tiv",
    # Places holding several languages, a former state, or nothing identifiable
    "Ogoja": OTHER_NG, "Obubra": OTHER_NG, "Ogae": OTHER_NG, "Ogaja": OTHER_NG,
    "Ikom": OTHER_NG, "Delta": OTHER_NG, "Bendel": OTHER_NG, "Opobo": OTHER_NG,
    "Egbema": OTHER_NG, "Egbma": OTHER_NG, "Omuma": OTHER_NG, "Bille": OTHER_NG,
    "Kagara": OTHER_NG, "Nzam": OTHER_NG, "Olukumi": OTHER_NG,
    "Abawa": OTHER_NG, "Abi": OTHER_NG, "Aduge": OTHER_NG, "Aka": OTHER_NG,
    "Akwandara": OTHER_NG, "Anian": OTHER_NG, "Awa": OTHER_NG, "Azoqo": OTHER_NG,
    "Baco": OTHER_NG, "Bako": OTHER_NG, "Badukke": OTHER_NG, "Bagwom": OTHER_NG,
    "Bakamuka": OTHER_NG, "Bakor": OTHER_NG, "Banbawa": OTHER_NG, "Bandawa": OTHER_NG,
    "Basange": OTHER_NG, "Basayi": OTHER_NG, "Bawaje": OTHER_NG, "Bawara": OTHER_NG,
    "Bazanfara": OTHER_NG, "Beltil": OTHER_NG, "Boiyana": OTHER_NG, "Buburmi": OTHER_NG,
    "Buh": OTHER_NG, "Cham": OTHER_NG, "Dador": OTHER_NG, "Dainawa": OTHER_NG,
    "Egbueuakaa": OTHER_NG, "Eika": OTHER_NG, "Ekeya": OTHER_NG, "Ekim": OTHER_NG,
    "Ekpari": OTHER_NG, "Etan": OTHER_NG, "Fakama": OTHER_NG, "Fakawa": OTHER_NG,
    "Fakun": OTHER_NG, "Faliya": OTHER_NG, "Femawa": OTHER_NG, "Gbile": OTHER_NG,
    "Geum": OTHER_NG, "Gomu": OTHER_NG, "Ikpena": OTHER_NG, "Jaja": OTHER_NG,
    "Kaba": OTHER_NG, "Kaika": OTHER_NG, "Kaka": OTHER_NG, "Kalakala": OTHER_NG,
    "Kambu": OTHER_NG, "Kami": OTHER_NG, "Kantana": OTHER_NG, "Karfawa": OTHER_NG,
    "Kirfawa": OTHER_NG, "Karmo": OTHER_NG, "Katongu": OTHER_NG, "Kuturmi": OTHER_NG,
    "Lanta": OTHER_NG, "Lawabe": OTHER_NG, "Lerenci": OTHER_NG, "Libbo": OTHER_NG,
    "Lumama": OTHER_NG, "Magu": OTHER_NG, "Mbube": OTHER_NG, "Minyam": OTHER_NG,
    "Miyawa": OTHER_NG, "Muryang": OTHER_NG, "Nbula": OTHER_NG, "Ndoro": OTHER_NG,
    "Nedere": OTHER_NG, "Ningawa": OTHER_NG, "Nkim": OTHER_NG, "Nkwane": OTHER_NG,
    "Nyandang": OTHER_NG, "Obodu": OTHER_NG, "Ogale": OTHER_NG, "Ogato": OTHER_NG,
    "Ogbia": "Ogbia", "Ogbolo": OTHER_NG, "Okon": OTHER_NG, "Once": OTHER_NG,
    "Ovia": OTHER_NG, "Panso": OTHER_NG, "Plumdi": OTHER_NG, "Pyakum": OTHER_NG,
    "Rendere": OTHER_NG, "Tanlyn": OTHER_NG, "Tarah": OTHER_NG, "Taro": OTHER_NG,
    "Tehl": OTHER_NG, "U Tonko": OTHER_NG, "Utenko": OTHER_NG, "Ufiyer": OTHER_NG,
    "Uwa": OTHER_NG, "Wurkum": OTHER_NG, "Wurkun": OTHER_NG, "Youm": OTHER_NG,
    "Zamji": OTHER_NG,
}

# Every Afrobarometer REGION label -> COD-AB state name (religiondots' sources/ng.py NORM).
NORM = {
    "abia": "Abia", "adamawa": "Adamawa", "akwa ibom": "Akwa Ibom", "akwa-ibom": "Akwa Ibom",
    "anambra": "Anambra", "bauchi": "Bauchi", "bayelsa": "Bayelsa", "benue": "Benue",
    "borno": "Borno", "cross river": "Cross River", "cross-river": "Cross River",
    "delta": "Delta", "ebonyi": "Ebonyi", "edo": "Edo", "ekiti": "Ekiti", "enugu": "Enugu",
    "fct": "Federal Capital Territory", "fct abuja": "Federal Capital Territory",
    "gombe": "Gombe", "imo": "Imo", "jigawa": "Jigawa", "kaduna": "Kaduna", "kano": "Kano",
    "katsina": "Katsina", "kebbi": "Kebbi", "kogi": "Kogi", "kwara": "Kwara", "lagos": "Lagos",
    "nasarawa": "Nasarawa", "nassarawa": "Nasarawa", "niger": "Niger", "ogun": "Ogun",
    "ondo": "Ondo", "osun": "Osun", "oyo": "Oyo", "plateau": "Plateau", "rivers": "Rivers",
    "sokoto": "Sokoto", "taraba": "Taraba", "yobe": "Yobe", "zamfara": "Zamfara",
}


def say(ok, msg):
    print(("  ok  " if ok else "  FAIL") + "  " + msg)
    if not ok:
        raise SystemExit("check failed: " + msg)


def ckey(s):
    s = unicodedata.normalize("NFKC", str(s)).replace("’", "'")
    return " ".join(s.split()).strip().casefold()


# ---------------------------------------------------------------- extract

def _read(path, cols=None):
    import pyreadstat
    try:
        return pyreadstat.read_sav(str(path), usecols=cols)
    except Exception:  # noqa: BLE001  R6 is not valid UTF-8
        return pyreadstat.read_sav(str(path), usecols=cols, encoding="LATIN1")


def fetch():
    """Nigeria's rows of the six merged rounds, read from religiondots' downloads."""
    import pyreadstat
    out = []
    for rnd, name, q, qo, wt, lga, eth, etho in ROUNDS:
        p = AB_DIR / name
        if not p.exists():
            raise SystemExit(f"{p} missing: run `python sources/ng.py --fetch` in religiondots "
                             "(the six merged rounds, ~280 MB, are kept there)")
        try:
            _, meta = pyreadstat.read_sav(str(p), metadataonly=True)
        except Exception:  # noqa: BLE001
            _, meta = pyreadstat.read_sav(str(p), metadataonly=True, encoding="LATIN1")
        up = {c.upper(): c for c in meta.column_names}
        want = ["COUNTRY", "REGION", "RESPNO", q, qo, wt, eth, etho] + ([lga] if lga else [])
        cols = [up[c.upper()] for c in want]
        lab = str(meta.column_names_to_labels.get(up[q.upper()], "")).casefold()
        say(any(t in lab for t in LANG_LABEL), f"R{rnd} {q} is the home-language question "
            f"({lab!r})")
        elab = str(meta.column_names_to_labels.get(up[eth.upper()], "")).casefold()
        say(any(t in elab for t in ETH_LABEL), f"R{rnd} {eth} is the ethnic group ({elab!r})")
        df, meta = _read(p, cols)
        c = {k: up[k.upper()] for k in want}
        clab = meta.variable_value_labels.get(c["COUNTRY"], {})
        sub = df[df[c["COUNTRY"]].map(clab).astype(str).str.strip().str.casefold() == "nigeria"]
        say(len(sub) == N_RESP[rnd], f"R{rnd}: {len(sub):,} Nigerian respondents")
        ql = meta.variable_value_labels.get(c[q], {})
        gl = meta.variable_value_labels.get(c["REGION"], {})
        ll = meta.variable_value_labels.get(c[lga], {}) if lga else {}
        w = pd.to_numeric(sub[c[wt]], errors="coerce")
        say(0.98 <= w.sum() / len(sub) <= 1.02, f"R{rnd} {wt} is a within-country weight "
            f"(mean {w.sum() / len(sub):.3f})")
        o = pd.DataFrame({
            "round": rnd,
            "respno": sub[c["RESPNO"]].astype(str),
            "region": sub[c["REGION"]].map(gl),
            "lga": (sub[c[lga]].map(ll) if ll else sub[c[lga]]) if lga else "",
            "lang": sub[c[q]].map(ql),
            "verbatim": sub[c[qo]].astype(str).str.strip(),
            "eth": sub[c[eth]].map(meta.variable_value_labels.get(c[eth], {})),
            "eth_verbatim": sub[c[etho]].astype(str).str.strip(),
            "w": w,
        })
        say(o["lang"].notna().all(), f"R{rnd}: every answer code has a label")
        out.append(o)
    a = pd.concat(out, ignore_index=True)
    RAW.mkdir(parents=True, exist_ok=True)
    a.to_csv(EXTRACT, index=False)
    print(f"wrote {EXTRACT} ({len(a):,} respondents)")


# ---------------------------------------------------------------- harmonise

def answer(row):
    lab = str(row["lang"]).strip()
    v = str(row["verbatim"]).strip()
    v = "" if v.lower() in ("", "nan", "none") else " ".join(v.split()).title()
    if lab in READ_VERBATIM and v:
        if v not in VERBATIM:
            raise SystemExit(f"verbatim {v!r} (R{row['round']}, {row['region']}) is not in "
                             "VERBATIM: decide it there")
        return VERBATIM[v]
    if lab in ("Other", "Others"):
        return OTHER_NG
    return CODED.get(lab, lab)


# The ethnic group card's labels where they are not already an answer's name.
ETHNIC = {
    "Fulani": "Fula", "Gwari": "Gbagyi", "Jaba": "Hyam (Jaba)", "Buju": "Jju", "Bajju": "Jju",
    "Birom": "Berom", "Kataf": "Tyap", "Bura": "Bura-Pabir", "Itsekiri": "Itsekiri",
}
# Ethnic-group verbatims of English-at-home respondents not already in VERBATIM.
ETH_VERBATIM = {
    "Obolo Andoni": "Obolo", "Adoni Obolo": "Obolo", "Ofia": "Ohafia",
    "Asari Toru": "Kalabari", "Abonema": "Kalabari", "Abam": "Igbo", "Bendel Igbo": "Igbo",
    "Ahudar": "Ekpeye", "Iyachie": "Yace", "Embembe": "Mbembe", "Dakarchi": "Lela (Dakarkari)",
    "Ikale": "Yoruba", "Ijebu Ode": "Yoruba", "Bazabarma": "Zarma", "Ejahum": "Ejagham",
    "Baburu": "Bura-Pabir", "Ewgon": "Eggon", "Efik": "Efik", "Esan": "Esan", "Yakurr": "Yakurr",
    "Ogoni": "Ogoni", "Okrika": "Okrika", "Mbula": OTHER_NG, "Boki": OTHER_NG,
    "Igarra": OTHER_NG, "Cross Riverian": OTHER_NG, "Abrakapo": OTHER_NG, "Abrimiba": OTHER_NG,
    "Kanukwu": OTHER_NG, "Alifokpa": OTHER_NG, "Ovenum": OTHER_NG, "Alin": OTHER_NG,
    "Bamono": OTHER_NG, "Agboja": OTHER_NG, "Nmuezugo": OTHER_NG, "Egah": OTHER_NG,
    "Efema": OTHER_NG, "Utugwu": OTHER_NG, "Piti": OTHER_NG, "Kaninkon": OTHER_NG,
    # Pidgin-at-home respondents' groups (ask 018, 2026-10-05)
    "Bonny": "Ibani",                                      # Bonny's language is Ibani
    "Heggi": "Kamwe", "Jukun-Kona": "Jukun",
    "Izzi": OTHER_NG,                                      # Izii, an Igboid language of its own
    "Gure": OTHER_NG, "Echoji": OTHER_NG, "Isen- Udm": OTHER_NG, "Jassa.": OTHER_NG,
}


def english_by_ethnicity(a, which="English"):
    """English (and Pidgin) answers are moved to the language of the respondent's own ethnic
    group; the lingua franca's own share is then set from R7's mother-tongue question in
    state_shares (ask 018).

    Nearly every English-at-home answer came in an English-language interview (153 of 166 in
    R7, 180 of 199 in R8, 116 of 120 in R9), 40% of them from rural respondents, and English
    went from 0 answers in R4 and R6 (not on the card) to 7-13% of R7-R9. Read as a FIRST
    language that would put 47% of Cross River and 34% of the FCT on English. So, as for
    English in El Salvador and Ecuador (AGENT_BRIEF §2, learned second languages), English is
    not drawn as a first language where the same respondent names a Nigerian ethnic group; it
    is drawn on that group's language. Those who gave no group (don't know, "Nigerian only")
    stay on English. Flagged to the supervisor: English is also a real first language for some
    urban Nigerians, and this call is the one most worth reversing if Anita disagrees.
    """
    m = a["answer"] == which
    lab = a["eth"].astype(str).str.strip()
    v = a["eth_verbatim"].astype(str).str.strip().str.split().str.join(" ").str.title()
    known = set(a["answer"]) | set(CODED.values())
    out = []
    missing = sorted({v[i] for i in a.index[m] if lab[i] in ("Other", "Others")
                      and v[i] not in ("", "Nan", "None")
                      and ETH_VERBATIM.get(v[i], VERBATIM.get(v[i])) is None})
    if missing:
        raise SystemExit(f"ethnic verbatims of {which} answers not in ETH_VERBATIM: {missing}")
    for i in a.index[m]:
        e = lab[i]
        if e in ("Other", "Others") and v[i] not in ("", "Nan", "None"):
            t = ETH_VERBATIM.get(v[i], VERBATIM.get(v[i]))
        elif e in ETHNIC:
            t = ETHNIC[e]
        elif CODED.get(e, e) in known and e not in ("Other", "Others"):
            t = CODED.get(e, e)
        else:
            t = which
        out.append(t)
    a.loc[m, "answer"] = out
    moved = pd.Series(out)
    print(f"  {which} answers: {int(m.sum())}; moved to their ethnic group's language: "
          f"{int((moved != which).sum())} (top: "
          f"{moved[moved != which].value_counts().head(6).to_dict()}); no group given (left out "
          f"of the shares): {int((moved == which).sum())}")
    return a


def load():
    a = pd.read_csv(EXTRACT, dtype=str, keep_default_na=False)
    a["round"] = a["round"].astype(int)
    a["w"] = a["w"].astype(float)
    say(len(a) == sum(N_RESP.values()), f"{len(a):,} respondents in the extract")
    a = a[~a["lang"].isin(NON_ANSWERS)].copy()
    print(f"  {sum(N_RESP.values()) - len(a)} non-answers dropped (don't know, refused)")
    a["answer"] = a.apply(answer, axis=1)
    a["state"] = a["region"].map(ckey).map(NORM)
    bad = sorted(a.loc[a["state"].isna(), "region"].unique())
    say(not bad, f"every REGION label is a state ({bad})")
    return a


def single_verbatims(a):
    """Languages named only in free text, by one respondent in the pool: on OTHER_NG."""
    coded = set(CODED.values()) | {CODED.get(x, x) for x in a["lang"].unique()}
    n = a.groupby("answer").size()
    one = [k for k, v in n.items() if v == 1 and k not in coded and k != OTHER_NG]
    a.loc[a["answer"].isin(one), "answer"] = OTHER_NG
    return one


def split_combined(a):
    """R4's combined Rivers/Bayelsa label, no verbatim: shared across its six answers."""
    m = a["answer"] == COMBINED
    rest = a[~m & a["answer"].isin(COMBINED_PARTS)]
    ref = rest.groupby(["state", "answer"])["w"].sum()
    nat = rest.groupby("answer")["w"].sum()
    rows = []
    fallback = []
    for _, r in a[m].iterrows():
        s = ref.get(r["state"]) if r["state"] in ref.index.get_level_values(0) else None
        if s is None or s.sum() == 0:
            # a migrant outside the delta: the six answers' national weighted counts
            s = nat
            fallback.append(r["state"])
        for ans, v in (s / s.sum()).items():
            rr = r.copy()
            rr["answer"], rr["w"] = ans, r["w"] * v
            rows.append(rr)
    print(f"  R4 combined label: {int(m.sum())} respondents without a verbatim shared across "
          f"{', '.join(COMBINED_PARTS)} by their weighted counts in the same state "
          f"({a.loc[m, 'state'].value_counts().to_dict()}); on the national split, as no "
          f"answer of the six was given there otherwise: {fallback}")
    return pd.concat([a[~m], pd.DataFrame(rows)], ignore_index=True)


HAUSALAND = 0.5   # a state is Hausaland where R7's mother-tongue Hausa share is at least this

# Nigerian Pidgin as a first language (Anita, 2026-10-06; the ask 019 route: a cited estimate,
# placed by a stated rule). Ethnologue 26th ed. (2023): 4.7 million L1 users (2020 figure).
# Spread over the states in proportion to each state's R7-R9 Pidgin-at-home share x its
# population (pidgin_pattern), so the national total is the estimate.
PIDGIN_L1 = 4_700_000


def pidgin_pattern(a):
    """Each state's weighted share of R7-R9 respondents answering Pidgin at home, read before
    those answers are moved to the respondent's ethnic group's language. Placement of the
    Ethnologue estimate only; the level is PIDGIN_L1."""
    b = a[a["round"].isin(EP_ROUNDS)]
    t = b.groupby("state")["w"].sum()
    p = b[b["answer"] == "Nigerian Pidgin"].groupby("state")["w"].sum()
    return (p.reindex(t.index).fillna(0.0) / t).rename("pidgin_home")


def state_shares(a, pid=None, pop=None):
    """Ask 018 (Anita, 2026-10-05): lingua francas at Afrobarometer R7's mother-tongue (Q2A).

    Every answer's share comes from R4-R9 among the respondents not left on English or Pidgin
    (whose English and Pidgin answers were first moved to their ethnic group's language).
    Then:
    - English = R7's Q2A English share in the state, shrunk to the national 1.9% by
      wafr_afro.K_SHRINK respondents (R7 has 16-128 a state).
    - Pidgin: R7's Q2A share is 0 (no R7 respondent named Pidgin as mother tongue; the 41 who
      speak it at home name Igbo, Edo, Ijaw, Esan... or "other"). Since 2026-10-06 (Anita) it
      is drawn from Ethnologue's 4.7 million L1 instead, spread by pidgin_pattern (PIDGIN_L1).
    - Hausa outside Hausaland (states under HAUSALAND by R7 Q2A) = R7's Q2A Hausa share,
      shrunk to the pooled share x the Q2A/Q2B ratio of those states' R7 respondents. In
      Hausaland the pooled share stands.
    The other answers are scaled to what is left.
    """
    sys.path.insert(0, str(HERE / "sources"))
    from wafr_afro import r7_mother, shrink
    t = r7_mother("Nigeria", {"English": ["English"], "Pidgin": ["Pidgin English"],
                              "Hausa": ["Hausa"]})
    say(t.loc["_national", "Pidgin_A"] == 0, "R7: no respondent names Pidgin as mother tongue")
    reg = {g: NORM[ckey(g)] for g in t.index if g != "_national"}
    say(len(set(reg.values())) == N_STATES, "R7's REGION labels are the 37 states")
    rest = a[~a["answer"].isin(ENGLISH_PIDGIN)]
    r = rest.groupby(["state", "answer"])["w"].sum()
    r = r.div(r.groupby(level="state").sum(), level="state").unstack(fill_value=0.0)

    eng = shrink(t, "English").rename(reg)
    raw_h = (t["Hausa_A"] / t["n"]).drop("_national").rename(reg)
    out_s = raw_h.index[raw_h < HAUSALAND]
    tt = t.drop(index="_national").rename(reg)
    ratio = tt.loc[out_s, "Hausa_A"].sum() / tt.loc[out_s, "Hausa_B"].sum()
    prior_r = pd.Series({g: r.loc[reg[g], "Hausa"] * ratio for g in reg})
    hau = shrink(t, "Hausa", prior=prior_r).rename(reg)
    print(f"  R7 Q2A English nationally {t.loc['_national', 'English_A'] / t.loc['_national', 'n']:.2%};"
          f" Hausa outside Hausaland ({len(out_s)} states): Q2A/Q2B ratio {ratio:.2f}")
    pl1 = None
    if pid is not None:
        home = (pid.reindex(pop.index).fillna(0.0) * pop)
        f = PIDGIN_L1 / home.sum()
        pl1 = pid.reindex(pop.index).fillna(0.0) * f
        say(pl1.max() < 0.5, f"Pidgin L1: home-share pattern x {f:.3f} to reach {PIDGIN_L1:,}")
        print("  Pidgin L1 by state (top): " + ", ".join(
            f"{k} {v:.1%}" for k, v in pl1.sort_values(ascending=False).head(12).items()))
    for st in r.index:
        set_ = {"English": eng[st]}
        if pl1 is not None and pl1[st] > 0:
            set_["Nigerian Pidgin"] = pl1[st]
        if st in out_s:
            set_["Hausa"] = hau[st]
        fixed = sum(set_.values())
        others = [c for c in r.columns if c not in set_]
        r.loc[st, others] = r.loc[st, others] / r.loc[st, others].sum() * (1 - fixed)
        for k, v in set_.items():
            r.loc[st, k] = v
    sh = r.stack()
    sh.index.names = ["state", "answer"]
    return sh[sh > 0].sort_index()


def main():
    if "--fetch" in sys.argv:
        fetch()
    a = load()
    ct = pd.crosstab(a["answer"], a["round"])
    ct["all"] = ct.sum(axis=1)
    print(ct.sort_values("all", ascending=False).head(40).to_string())
    pid = pidgin_pattern(a)
    a = english_by_ethnicity(a, "English")
    a = english_by_ethnicity(a, "Nigerian Pidgin")
    one = single_verbatims(a)
    print(f"  {len(one)} languages named in free text by one respondent -> {OTHER_NG}")
    a = split_combined(a)

    lut = pd.read_csv(LOOKUP, dtype={"geo_id": str})
    say(len(lut) == N_STATES and int(lut["pop"].sum()) == CODPS_2022,
        f"ng_lookup.csv: {len(lut)} states, {int(lut['pop'].sum()):,} people (COD-PS 2022)")
    say(sorted(a["state"].unique()) == sorted(lut["name"]), "the 37 states, both ways")
    pop = lut.set_index("name")["pop"].astype(float)
    gid = dict(zip(lut["name"], lut["geo_id"]))

    sh = state_shares(a, pid, pop)
    s = sh.groupby(level="state").sum()
    say((s - 1).abs().max() < 1e-9, "every state's shares sum to 1")
    n_state = a.groupby("state").size()
    print(f"  respondents per state: min {n_state.min()} ({n_state.idxmin()}), median "
          f"{int(n_state.median())}, max {n_state.max()} ({n_state.idxmax()})")
    ep_n = a[a["round"].isin(EP_ROUNDS)].groupby("state").size()
    print(f"  R7-R9 respondents per state (English, Pidgin): min {ep_n.min()} "
          f"({ep_n.idxmin()}), median {int(ep_n.median())}")

    df = sh.rename("share").reset_index()
    df["count"] = df["share"] * df["state"].map(pop)
    # integer counts that sum to each state's population (largest remainder)
    out = []
    for st, g in df.groupby("state"):
        f = g["count"].to_numpy()
        base = np.floor(f)
        k = int(round(pop[st] - base.sum()))
        base[np.argsort(-(f - base))[:k]] += 1
        g = g.assign(count=base.astype(int))
        out.append(g)
    df = pd.concat(out)
    say(int(df["count"].sum()) == CODPS_2022, f"drawn total {int(df['count'].sum()):,}")
    nat = df.groupby("answer")["count"].sum().sort_values(ascending=False)
    print("\n  national, as drawn:")
    for k, v in nat.head(30).items():
        print(f"    {k:28s} {v:>12,}  {v / CODPS_2022:6.2%}")

    n_by = a.groupby(["state", "answer"]).size()
    df["note"] = [f"share {r.share:.4f}; {n_by.get((r.state, r.answer), 0)} respondents "
                  f"(of {n_state[r.state]})" for r in df.itertuples()]
    df = df[df["count"] > 0]
    res = pd.DataFrame({
        "geo_id": df["state"].map(gid), "geo_level": "state", "geo_name": df["state"],
        "source_category": df["answer"], "count": df["count"], "tier": "modelled",
        "source_id": SOURCE_ID, "year": "2008-2022", "note": df["note"],
    })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.sort_values(["geo_id", "count"], ascending=[True, False]).to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(res)} rows, {res['source_category'].nunique()} answers)")

    split_half(a)
    clear_check(sh, gid)
    lga_table(a, gid)


def split_half(a):
    """R4-R6 against R7-R9, state shares of every answer that is 1%+ nationally.

    A test of the geography, not the level: does the survey put each language in the same
    states in two halves fourteen years apart? Pidgin is left out (on R7-R9 only).
    """
    def shares(b):
        t = b.groupby(["state", "answer"])["w"].sum()
        return t.div(t.groupby(level="state").sum(), level="state").unstack(fill_value=0)
    x = a[~a["answer"].isin(ENGLISH_PIDGIN)]
    h1, h2 = shares(x[x["round"] <= 6]), shares(x[x["round"] >= 7])
    nat = x.groupby("answer")["w"].sum() / x["w"].sum()
    print("\n  split-half, R4-R6 against R7-R9, Pearson r across the 37 states:")
    for k in nat[nat >= 0.01].sort_values(ascending=False).index:
        r = np.corrcoef(h1.get(k, 0).reindex(h1.index, fill_value=0),
                        h2.get(k, 0).reindex(h1.index, fill_value=0))[0, 1]
        print(f"    {k:26s} {nat[k]:6.1%}   r = {r:+.3f}")
        if nat[k] >= 0.03:
            say(r > 0.9, f"{k}: the two halves agree on where it is (r {r:+.3f})")


def clear_check(sh, gid):
    """CLEAR Global's admin1 shares (HDX nigeria-languages): R9 alone in 31 states, REACH's
    household surveys in Adamawa, Borno, Yobe (2021) and Katsina, Sokoto, Zamfara (2022)."""
    c = pd.read_csv(RAW / "clearglobal_language_use_NGA_admin1.csv")
    name = {"Hausa": "Hausa", "Yoruba": "Yoruba", "Igbo": "Igbo", "Central Kanuri": "Kanuri",
            "Kanuri": "Kanuri", "Fula": "Fula", "Eastern Fula": "Fula", "Nigerian Fulfulde": "Fula"}
    inv = {v: k for k, v in gid.items()}
    c["state"] = c["location_code"].map(inv)
    c["answer"] = c["language_name"].map(name)
    c = c[c["answer"].notna()].groupby(["state", "answer", "dataset_name"])[
        "proportion_value"].sum().reset_index()
    print("\n  against CLEAR Global's admin1 shares (big languages, points):")
    worst = []
    for r in c.itertuples():
        mine = sh.get((r.state, r.answer), 0.0)
        src = "REACH" if "REACH" in r.dataset_name else "AB R9"
        worst.append((abs(mine - r.proportion_value), r.state, r.answer, mine,
                      r.proportion_value, src))
    for d, st, ans, mine, theirs, src in sorted(worst, reverse=True)[:12]:
        print(f"    {st:12s} {ans:8s} drawn {mine:6.1%}  CLEAR {theirs:6.1%}  ({src})")
    reach = [w for w in worst if w[5] == "REACH"]
    print("    REACH states: " + "; ".join(f"{st} {ans} {mine:.0%}/{theirs:.0%}"
                                          for _, st, ans, mine, theirs, _ in sorted(reach)))


# ---------------------------------------------------------------- LGAs, for placement only

# Afrobarometer LGA label (folded) -> COD-AB adm2_name, where the plain fold does not match
# inside the state. Filled from the misses printed below; each one read, none fuzzy-matched.
LGA_ALIAS = {
    ("Akwa Ibom", "etinam"): "etinan",
    ("Bayelsa", "yenagoa"): "yenegoa",
    ("Benue", "katsinala"): "katsina ala",
    ("Cross River", "yakur"): "yakurr",
    ("Cross River", "ugep south"): "yakurr",            # Ugep is Yakurr's headquarters
    ("Delta", "ugheli north"): "ughelli north",
    ("Delta", "ugheli south"): "ughelli south",
    ("Edo", "orhionmwo"): "orhionmwon",
    ("Edo", "uhonmwode"): "uhunmwonde",
    ("Federal Capital Territory", "amac"): "abuja municipal",   # Abuja Municipal Area Council
    ("Gombe", "shogom"): "shomgom",
    ("Imo", "ezinitte mbaise"): "ezinihitte",
    ("Jigawa", "malam maduri"): "malam madori",
    ("Jigawa", "mallam madori"): "malam madori",
    ("Jigawa", "sule tankar kar"): "sule tankarkar",
    ("Kaduna", "jamaa"): "jemaa",
    ("Kaduna", "jema a"): "jemaa",
    ("Kaduna", "zangon kataf"): "zango kataf",
    ("Kano", "nassarawa"): "nasarawa",
    ("Kano", "tundun wada"): "tudun wada",
    ("Katsina", "maiduwa"): "maiadua",
    ("Kogi", "kabba bunu (oyi)"): "kabba bunu",
    ("Kwara", "patigi"): "pategi",
    ("Lagos", "oshodi isholo"): "oshodi isolo",
    ("Nasarawa", "nassarawa"): "nasarawa",
    ("Nasarawa", "nassarawa eggon"): "nasarawa eggon",
    ("Niger", "kotangora"): "kontagora",
    ("Ogun", "ado odo otta"): "ado odo ota",
    ("Ogun", "ikene"): "ikenne",
    ("Ogun", "sagamu"): "shagamu",
    ("Osun", "atakomosa west"): "atakumosa west",
    ("Oyo", "atisbo"): "atigbo",                       # COD-AB spells Atisbo "Atigbo"
    ("Plateau", "barkin ladi"): "barikin ladi",
    ("Plateau", "langtan south"): "langtang south",
    ("Plateau", "quan pan"): "quaan pan",
    ("Rivers", "abual odua"): "abua odual",
    ("Rivers", "obio akpor"): "obia akpor",
    ("Rivers", "ogba egbem"): "ogba egbema ndoni",
    ("Rivers", "okirika"): "okrika",
    ("Rivers", "port harcourt city"): "port harcourt",
    ("Rivers", "port harourt"): "port harcourt",
    ("Sokoto", "wamakko"): "wamako",
}
# Left unplaced on purpose (their respondents still count in their state; only the LGA is
# unknown): "Obioma Ngwa" (Abia; Obi Ngwa or Osisioma Ngwa), "Uquo Ibeno(Esit Eket" (two LGAs
# named), "Akoko" (Ondo has four), "Tai Eleme" (the LGA split into Tai and Eleme), and R6's
# "Gboko" under Niger (a Benue LGA; its 16 respondents answered Nupe, so the LGA is wrong).

# R7's LGA value labels are shifted by one state: 336 of its 1,600 respondents carry an LGA of
# the state before theirs in alphabetical order (Ebonyi respondents in Delta's Warri South,
# Kano's in Kaduna's Soba, Katsina's in Kano's Rano and Wudil), and their answers follow REGION,
# not the LGA (all eight "Warri South" answered Igbo). So R7's LGA column is not used at all:
# a label that happens to fall inside its own state is no more trustworthy than the rest.
LGA_ROUNDS = [4, 6, 9]


def lfold(s):
    s = ckey(s).replace("-", " ").replace("/", " ").replace("'", "").replace(".", " ")
    return " ".join(s.split())


def lga_table(a, gid):
    import geopandas as gpd
    adm2 = gpd.read_file(ADM2, engine="pyogrio", ignore_geometry=True)
    say(len(adm2) == 774, f"COD-AB: {len(adm2)} LGAs")
    st_pc = {v: k for k, v in gid.items()}
    by_state = {}
    for r in adm2.itertuples():
        by_state.setdefault(r.adm1_pcode, {})[lfold(r.adm2_name)] = r.adm2_pcode
    b = a[(a["lga"].astype(str).str.strip() != "") & a["round"].isin(LGA_ROUNDS)].copy()
    b["pc1"] = b["state"].map(gid)
    miss = {}
    pcs = []
    for r in b.itertuples():
        k = lfold(r.lga)
        k = LGA_ALIAS.get((r.state, k), k)
        pc = by_state[r.pc1].get(k)
        if pc is None:
            miss.setdefault((r.state, k), 0)
            miss[(r.state, k)] += 1
        pcs.append(pc)
    b["lga_pcode"] = pcs
    if miss:
        print(f"\n  {len(miss)} LGA labels with no COD-AB LGA in their state:")
        for (st, k), n in sorted(miss.items()):
            print(f"    ({st!r}, {k!r}): n={n}   [{', '.join(sorted(by_state[gid[st]])[:0])}]")
    b = b[b["lga_pcode"].notna()]
    print(f"  {len(b):,} respondents placed in {b['lga_pcode'].nunique()} LGAs "
          f"(rounds {sorted(b['round'].unique())})")
    t = b.groupby(["pc1", "lga_pcode", "answer"])["w"].sum().reset_index()
    t.columns = ["geo_id", "lga_pcode", "source_category", "w"]
    t.to_csv(OUT_LGA, index=False)
    print(f"wrote {OUT_LGA} ({len(t)} rows)")
    return miss


if __name__ == "__main__":
    main()
