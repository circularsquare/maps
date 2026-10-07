"""Tanzania: home language from five pooled Afrobarometer rounds (2008-2022), by region.

    python sources/tz_afro.py --fetch   extract Tanzania's rows from religiondots' merged .sav
                                        files (read-only) -> data/raw/tz/ab_tz_language.csv
    python sources/tz_afro.py           -> data/normalized/tz.csv   (unit x answer, counts)

                                        data/normalized/tz_district.csv (district x answer
                                        shares, placement only)

Tanzania's census asks neither language nor ethnicity, so this is the survey route of
AGENT_BRIEF §2: shares from the survey times the 2022 census region totals, every row
`modelled`. Rounds 4 and 6-9; round 5 left out (pre-2012 regions, no district). Round 8 is
decoded by REGION code, round 4 placed by district. Swahili's share per unit comes from
SW_ROUNDS (since 2026-10-05 R7 alone, whose "lang" is Q2A, the mother-tongue question, per
Anita's ruling on ask 018; it was R7-R9's "language spoken in home"); every other answer's
from all rounds among the non-Swahili answers. The record is sources/tz.md.
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
RD_GEO = RD / "data" / "geo" / "tz"
LOOKUP = RD_GEO / "tz_lookup.csv"          # 30 units, NBS 2022 census populations
DISTRICTS = RD_GEO / "tz_districts.csv"    # COD-AB 2018 district names -> unit
RAW = HERE / "data" / "raw" / "tz"
EXTRACT = RAW / "ab_tz_language.csv"
OUT = HERE / "data" / "normalized" / "tz.csv"

CENSUS_2022 = 61_741_120
N_UNITS = 30
SOURCE_ID = "afrobarometer_r4_r9_tanzania"

# (round, file, language, verbatim, weight, ethnic group, its verbatim, interview language)
ROUNDS = [
    (4, "merged_r4_data.sav", "Q3", "Q3OTHER", "Withinwt", "Q79", "Q79OTHER", "Q103"),
    (5, "merged-round-5-data-34-countries-2011-2013-last-update-july-2015_0.sav",
     "Q2", "Q2OTHER", "withinwt", "Q84", "Q84OTHER", "Q103"),
    (6, "merged_r6_data_2016_36countries2.sav", "Q2", "Q2OTHER", "withinwt", "Q87", "Q87OTHER",
     "Q103"),
    # R7: Q2A, "Respondent's mother tongue", not Q2B "Language spoken in home" (Anita's ruling on
    # ask 018, 2026-10-05: lingua francas at R7's mother-tongue question)
    (7, "r7_merged_data_34ctry.release.sav", "Q2A", "Q2AOTHER", "withinwt", "Q84", "Q84OTHER",
     "Q103"),
    (8, "afrobarometer_release-dataset_merge-34ctry_r8_en_2023-03-01.sav",
     "Q2", "Q2OTHER", "withinwt_hh", "Q81", "Q81OTHER", "Q103"),
    (9, "R9.Merge_39ctry.20Nov23.final_.release_Updated.4Jun25-3.sav",
     "Q2", "Q2OTHER", "withinwt_hh", "Q84A", "Q84AOTHER", "Q102"),
]
LANG_LABEL = ("language of respondent", "language spoken in home", "mother tongue")
INTERVIEWER = {7: "Q112"}   # interviewer number, kept where the round has one (R7)
ETH_LABEL = ("tribe or ethnic group", "ethnic community")


def say(ok, msg):
    print(("  ok  " if ok else "  FAIL") + "  " + msg)
    if not ok:
        raise SystemExit("check failed: " + msg)


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
        iv = [INTERVIEWER[rnd]] if rnd in INTERVIEWER else []
        want = list(dict.fromkeys(["COUNTRY", "REGION", "RESPNO", "URBRUR", q, qo, wt, eth, etho,
                                   il] + locs + iv))
        lab = str(meta.column_names_to_labels.get(up[q.upper()], "")).casefold()
        say(any(t in lab for t in LANG_LABEL), f"R{rnd} {q} is the home-language question "
            f"({lab!r})")
        elab = str(meta.column_names_to_labels.get(up[eth.upper()], "")).casefold()
        say(any(t in elab for t in ETH_LABEL), f"R{rnd} {eth} is the ethnic group ({elab!r})")
        df, meta = pyreadstat.read_sav(str(p), usecols=[up[c.upper()] for c in want], **enc)
        c = {k: up[k.upper()] for k in want}
        vl = meta.variable_value_labels
        cl = df[c["COUNTRY"]].map(vl.get(c["COUNTRY"], {})).astype(str).str.strip().str.casefold()
        sub = df[cl == "tanzania"]

        def lab_of(k):
            m = vl.get(c[k], {})
            return sub[c[k]].map(m) if m else sub[c[k]]
        w = pd.to_numeric(sub[c[wt]], errors="coerce")
        say(0.98 <= w.sum() / len(sub) <= 1.02, f"R{rnd}: {len(sub):,} Tanzanian respondents; "
            f"{wt} averages {w.sum() / len(sub):.3f}")
        o = pd.DataFrame({
            "round": rnd, "respno": sub[c["RESPNO"]].astype(str),
            "region_code": pd.to_numeric(sub[c["REGION"]], errors="coerce").astype("Int64"),
            "region": lab_of("REGION"),
            "district": lab_of(locs[0]) if locs else "",
            "urb": lab_of("URBRUR"),
            "lang": lab_of(q), "verbatim": sub[c[qo]].astype(str).str.strip(),
            "eth": lab_of(eth), "eth_verbatim": sub[c[etho]].astype(str).str.strip(),
            "intlang": lab_of(il), "w": w,
            "interviewer": lab_of(iv[0]).astype(str) if iv else "",
        })
        say(o["lang"].notna().all(), f"R{rnd}: every answer code has a label")
        out.append(o)
    a = pd.concat(out, ignore_index=True)
    RAW.mkdir(parents=True, exist_ok=True)
    a.to_csv(EXTRACT, index=False)
    print(f"wrote {EXTRACT} ({len(a):,} respondents)")


# ---------------------------------------------------------------- answers

SWAHILI = "Swahili"
# Which rounds set each unit's Swahili share (docstring, "SWAHILI"). [7, 8, 9] draws the answers
# as given under the current wording; [4, 6] is the older "home language" wording, which Kenya
# (sources/ke.md) uses. Ask 018 is open with Anita; this is the switch.
#
# Anita's ruling on ask 018 (2026-10-05): Swahili is drawn at R7's mother-tongue question (Q2A,
# now R7's "lang" in the extract), so SW_ROUNDS = [7]. Before that it was [7, 8, 9] under the
# home wording (Q2B in R7), 65.8% nationally.
SW_ROUNDS = [7]
DRAWN_ROUNDS = [4, 6, 7, 8, 9]
# Two R7 interviewers recorded Swahili as nearly every respondent's mother tongue (TAN15 98 of
# 100, TAN25 82 of 96) where the other interviewers on their team, in the same regions and
# often the same districts, recorded it for 0-20%: the mother tongues of Dodoma, Kigoma,
# Kilimanjaro, Manyara and Morogoro respondents (Gogo, Ha, Chaga...) cannot be one interviewer's
# 98% Swahili. Their R7 respondents are left out (checked in drop_interviewers).
DROP_INTERVIEWERS = {"TAN15", "TAN25"}
NON_ANSWERS = {"Don't know", "Refused", "Missing", "Refused To Answer"}
OTHER_TZ = "Other African language"

# The card's labels -> the answer they are counted as. Ki- is the Swahili prefix for a
# language; each is the language of that name (Glottolog in taxonomy/tree.d/tz.txt).
CODED = {
    "Swahili": SWAHILI, "Kiswahili": SWAHILI, "English": "English",
    "Kisukuma": "Sukuma", "Kiha": "Ha", "Kigogo": "Gogo", "Kihaya": "Haya",
    "Kichaga": "Chaga", "Kinyamwezi": "Nyamwezi", "Kimakonde": "Makonde",
    "Kinyakyusa": "Nyakyusa", "Kifipa": "Fipa", "Kinyaturu": "Nyaturu", "Kizigua": "Zigua",
    "Kiluguru": "Luguru", "Kihehe": "Hehe", "Kijita": "Jita", "Kisambaa": "Shambala",
    "Kipare": "Pare", "Kimasai": "Maasai", "Kiyao": "Yao", "Kingoni": "Ngoni",
    "Kizaramo": "Zaramo", "Kibena": "Bena", "Kikurya": "Kuria", "Kimwera": "Mwera",
    "Kindali": "Ndali", "Kinyiramba": "Nyiramba", "Kinyiha": "Nyiha", "Kinyambo": "Nyambo",
    "Kihangaza": "Hangaza", "Kirangi": "Rangi", "Kindendeule": "Ndendeule",
    "Kingindo": "Ngindo", "Kimeru": "Meru (Rwa)", "Kipogoro": "Pogoro", "Kisafwa": "Safwa",
    "Kikaguru": "Kaguru", "Kijaluo": "Luo", "Kijaruo": "Luo", "Kikwere": "Kwere",
    "Kindengereko": "Ndengereko", "Kimakuwa": "Makhuwa", "Kimakua": "Makhuwa",
    "Kinguu": "Nguu", "Kizinza": "Zinza", "Kisubi": "Shubi", "Kimatengo": "Matengo",
    "Kikinga": "Kinga", "Kindamba": "Ndamba", "Kimanyema": "Manyema",
    "Kizanaki": "Zanaki", "Kikerewe": "Kerewe", "Kiarusha": "Arusha", "Kikwaya": "Kwaya",
    "Kinyamwanga": "Nyamwanga", "Kishirazi": "Shirazi", "Kisumbwa": "Sumbwa",
    "Kisimbiti": "Suba-Simbiti", "Kisangu": "Sangu", "Kitongwe": "Tongwe",
    "Kipangwa": "Pangwa", "Kimambwe": "Mambwe",
    # Iraqw: three spellings, and Mbulu, the Swahili exonym (the Iraqw are "Wambulu", after
    # Mbulu district); one language, Glottolog iraq1241
    "Kiiraqi": "Iraqw", "Kiiraq": "Iraqw", "Kiiraqw": "Iraqw", "Kimbulu": "Iraqw",
    # on R7's mother-tongue card only
    "Kidigo": "Digo", "Mzigua": "Zigua", "Kitumbatu": SWAHILI,   # Tumbatu, as in VERBATIM
}
READ_VERBATIM = {"Other", "Others"}

# Every free-text answer (upper-cased) -> the answer it is counted as. A language a single
# respondent names, and nothing else does, goes on OTHER_TZ afterwards (single_verbatims).
VERBATIM = {
    # spellings of card languages
    "KINYAMWANGA": "Nyamwanga", "KINYIHA": "Nyiha", "KINYIA": "Nyiha", "KIMAMBWE": "Mambwe",
    "KIKEREWE": "Kerewe", "KIKINGA": "Kinga", "KINYAMBO": "Nyambo", "MNYAMBO": "Nyambo",
    "KISANGU": "Sangu", "KISAFWA": "Safwa", "KIPANGWA": "Pangwa", "KIRANGI": "Rangi",
    "KURANGI": "Rangi", "KIMBULU": "Iraqw", "KIIRAKI": "Iraqw", "KIIRAQ": "Iraqw",
    "KIMBULU/KIIRAKI": "Iraqw", "KIMAKUA": "Makhuwa", "KIMAKUWA": "Makhuwa",
    "MMAKUWA": "Makhuwa", "KIARUSHA": "Arusha", "KIBENA": "Bena", "KINDALI": "Ndali",
    "KIJALUO": "Luo", "KIJARUO": "Luo", "KILUO": "Luo", "KISUMBWA": "Sumbwa",
    "KINDAMBA": "Ndamba", "KIMWERA": "Mwera", "KIMWELA": "Mwera", "KIMATENGO": "Matengo",
    "KIHANGAZA": "Hangaza", "KIANGAZA": "Hangaza", "KINGINDO": "Ngindo", "KIGINDO": "Ngindo",
    "KIZINZA": "Zinza", "KIZINZAA": "Zinza", "KIZANAKI": "Zanaki",
    "KINDENGEREKO": "Ndengereko", "KIKAGURU": "Kaguru", "KIKAGULU": "Kaguru",
    "KAKAGURU": "Kaguru", "KINGUU": "Nguu", "KITONGWE": "Tongwe", "KINYIRAMBA": "Nyiramba",
    "KIZIGUA": "Zigua", "KIKWAYA": "Kwaya", "KIZARAMO": "Zaramo", "KINGONI": "Ngoni",
    "KINDENDEULE": "Ndendeule", "KINYATURU": "Nyaturu", "KIKWERE": "Kwere",
    "KISIMBITI": "Suba-Simbiti", "KIMERU": "Meru (Rwa)", "KIFIPA": "Fipa",
    "KINYAMWEZI": "Nyamwezi", "KIYAO": "Yao", "KPOGORO": "Pogoro", "KISHIRAZI": "Shirazi",
    "KISHUBI": "Shubi", "KIKAHE": "Chaga",            # Kahe, a Chaga variety (kahe1238)
    "KIMATAMBWE": "Makonde",                            # Matambwe, a Makonde group
    "KISUBA": "Suba-Simbiti",       # Tanzania's Suba (Mara) speak Suba-Simbiti (suba1252)
    # Swahili varieties named in free text: counted in Swahili, as Kenya counts a named
    # variety in its card cluster (Tumbatu, Makunduchi = Unguja's Kae, Pemba: Glottolog
    # dialects of swah1253)
    "KITUMBATU": SWAHILI, "KIMAKUNDUCHI": SWAHILI, "KIPEMBA": SWAHILI,
    "KIGUNYA": "Bajuni",            # Gunya is Bajuni's own name; respondents' group "Mgunya"
    # Fipa-Mambwe and Mbozi languages (Glottolog Mbozi)
    "KIPIMBWE": "Pimbwe", "KIPIMBWI": "Pimbwe", "KIMALILA": "Malila", "KIMALILI": "Malila",
    "KIBUNGU": "Bungu", "KILUNGWA": "Rungwa", "KILAMBYA": "Lambya",
    # languages of their own
    "KIBONDEI": "Bondei", "KIMATUMBI": "Matumbi", "KISANDAWE": "Sandawe", "KIDIGO": "Digo",
    "KISONJO": "Sonjo", "KIIKIZU": "Ikizu", "KIKIIZU": "Ikizu", "KIKIZU": "Ikizu",
    "KISIZAKI": "Ikizu",            # Sizaki, a dialect of Ikizu (siza1240)
    "KIBEMBE": "Bembe", "KIMBEMBE": "Bembe", "KIKONONGO": "Konongo", "KIKIMBU": "Kimbu",
    "KIIKOMA": "Ikoma", "KINYASA": "Nyasa", "KIMAGOMA": "Magoma", "KIKUTU": "Kutu",
    "KISAGALA": "Sagala", "KISAGARA": "Sagala", "KIKARA": "Kara", "KIKALA": "Kara",
    "KIWANJI": "Wanji", "KIUWANJI": "Wanji", "KIMUWANJI": "Wanji", "KIKABWA": "Kabwa",
    "KIMBUGWE": "Mbugwe", "KIMANDA": "Manda", "KIKISI": "Kisi", "KISEGEJU": "Segeju",
    "KIDOE": "Doe", "KIGWENO": "Gweno", "KIVIDUNDA": "Vidunda", "KIBENDE": "Bende",
    "KINYISANZU": "Isanzu", "KISANZU": "Isanzu",
    "KILURI": "Kwaya", "KIRURI": "Kwaya", "KIRYERI": "Kwaya",   # Ruri, a Kwaya dialect
    # Datooga: Barabaig and Taturu are Datooga sections; Mang'ati is the Maasai name for them
    "KIDATOOGA": "Datooga", "KITATURU": "Datooga", "KIBARBAIK": "Datooga",
    "KIMANG'ATI": "Datooga",
    "KIGOROO": "Gorowa",
    # Alagwa, whose speakers call themselves Wasi; the R8 answers came with ethnic group "Mwasi"
    "KICHASI": "Alagwa", "CHASI": "Alagwa", "KIASI": "Alagwa",
    "KITAITA": "Taita", "KIBORANA": "Borana", "KINYANKOLE": "Nyankole",
    "KIHINDI": "Other language",    # "Indian": no narrower node holds it
    # R7 mother-tongue free text (2026-10-05). Speaker names (Mzaramo, Muha) are read by
    # verbatim_key; these are spellings it cannot reach, checked against the respondent's
    # ethnic group where one was given.
    "KAYAO": "Yao", "KIHAYO": "Yao",                    # both ethnic group Myao
    "KAZARAMO": "Zaramo", "KIZARAMU": "Zaramo", "MZARAMU": "Zaramo", "KIRARAMO": "Zaramo",
    "MZALAMO": "Zaramo",
    "KIBENE": "Bena", "KIGIHA": "Ha", "KIHANJI": "Wanji", "KIKIZO": "Ikizu", "MWIKIZU": "Ikizu",
    "KIKUA": "Makhuwa", "KIMAKUHA": "Makhuwa", "KILAMBIA": "Lambya", "KIMATUMBWI": "Matumbi",
    "KINDENDEULI": "Ndendeule", "KINENGELEKO": "Ndengereko", "MDENGELEKO": "Ndengereko",
    "MNENGELEKO": "Ndengereko", "MUNDENGELEKO": "Ndengereko", "KINGINDU": "Ngindo",
    "KINYAHA": "Nyiha", "KINYISANZI": "Isanzu", "MUJISANZU": "Isanzu", "KIPOGOLO": "Pogoro",
    "MPOGOLO": "Pogoro", "KIRUGULU": "Luguru", "KISAFA": "Safwa", "KISANDAWI": "Sandawe",
    "KIZAANAKI": "Zanaki", "KIZUGUA": "Zigua", "MBARIBAIKI": "Datooga", "MLULI": "Kwaya",
    "MWASI": "Alagwa",
    "KIMPOTO": "Mpoto", "KIPOTO": "Mpoto",               # Mpoto (mpot1240), Lake Nyasa, Ruvuma
    "KIARABU": "Arabic", "WAARABU": "Arabic",
    # Unguja villages and "Zanzibari": the Swahili of Zanzibar, as Makunduchi above
    "KIBWEJUU": SWAHILI, "KIMATEMWI": SWAHILI, "MZANZIBARI": SWAHILI,
    "KINGAZIJA": OTHER_TZ,          # Comorian (Ngazidja), one respondent
    # not identifiable as a language, or a place or a clan: Kikine (Mara), Kinyimbo, Kimamba,
    # Kilongo, Kindyewe, Kilindi (a district), Kimalinyi (a town), Kituwan, Kiyusi, Kinata,
    # Kipagebi, Kimdale, Kisunva, Kitatuu, Kimazuruni, Kijahidiri, Kindonde, Kibunga,
    # Kinyagatwa, Kinyagazwa, Kikamba, Kimbungo, Kial-Yorobi, Kishihiri, Kirufiji, Kinyantuzu,
    # Mnyambu
}
for _k in ("KIKINE", "KINYIMBO", "KIMAMBA", "KILONGO", "KINDYEWE", "KILINDI", "KIMALINYI",
           "KITUWAN", "KIYUSI", "KINATA", "KIPAGEBI", "KIMDALE", "KISUNVA", "KITATUU",
           "KIMAZURUNI", "KIJAHIDIRI", "KINDONDE", "KIBUNGA", "KINYAGATWA", "KINYAGAZWA",
           "KIKAMBA", "KIMBUNGO", "KIAL-YOROBI", "KISHIHIRI", "KIRUFIJI", "KINYANTUZU",
           "MNYAMBU",
           # R7 mother tongue: not identifiable, or a clan or place
           "BAHASAN", "GERE", "KIBURUSHI", "KILIKHARUSI", "KIMAMBOYE", "KIMAZURUI", "MAZURUI",
           "KIMBUGU", "KIMBUNGA JAMII YA WANGONI", "KIMUNINDI", "KININDI", "KIMWIRA",
           "KINYAGATU", "KINYAGWATWA", "LAJMI", "MDUSHI", "MLWILWA", "MMBURU", "MSWETA"):
    VERBATIM[_k] = OTHER_TZ


# ---------------------------------------------------------------- units

# Every REGION label the rounds use -> the unit name in religiondots' tz_lookup.csv (its
# sources/tz.py NORM, which proves `Mrwara` and `Unfuja Kusini` by code).
NORM = {
    "arusha": "Arusha", "coast(pwani)": "Pwani", "pwani": "Pwani",
    "dar es salaam": "Dar es Salaam", "dar-es-salaam": "Dar es Salaam",
    "dares salaam": "Dar es Salaam",
    "dodoma": "Dodoma", "geita": "Geita", "iringa": "Iringa", "kagera": "Kagera",
    "kaskazini pemba": "Kaskazini Pemba", "north pemba": "Kaskazini Pemba",
    "pemba kaskazini": "Kaskazini Pemba",
    "kaskazini unguja": "Kaskazini Unguja", "north unguja": "Kaskazini Unguja",
    "unguja kaskazini": "Kaskazini Unguja",
    "katavi": "Katavi", "kigoma": "Kigoma", "kilimanjaro": "Kilimanjaro",
    "kusini pemba": "Kusini Pemba", "south pemba": "Kusini Pemba", "pemba kusini": "Kusini Pemba",
    "kusini unguja": "Kusini Unguja", "south unguja": "Kusini Unguja",
    "unfuja kusini": "Kusini Unguja",
    "lindi": "Lindi", "manyara": "Manyara", "mara": "Mara",
    "mbeya": "Mbeya and Songwe", "songwe": "Mbeya and Songwe",
    "mjini magharibi": "Mjini Magharibi", "urban west": "Mjini Magharibi",
    "morogoro": "Morogoro", "mrwara": "Mtwara", "mtwara": "Mtwara", "mwanza": "Mwanza",
    "njombe": "Njombe", "rukwa": "Rukwa", "ruvuma": "Ruvuma", "shinyanga": "Shinyanga",
    "simiyu": "Simiyu", "singida": "Singida", "tabora": "Tabora", "tanga": "Tanga",
}
# R4 (June 2008) predates the March 2012 regions; its districts place it. An old region's
# district may only land in that region or a region carved from it.
SPLIT_CHILDREN = {
    "Mwanza": {"Mwanza", "Geita", "Simiyu"}, "Shinyanga": {"Shinyanga", "Geita", "Simiyu"},
    "Kagera": {"Kagera", "Geita"}, "Iringa": {"Iringa", "Njombe"}, "Rukwa": {"Rukwa", "Katavi"},
}
DISTRICT_ALIAS = {"ARUMERU": "Meru"}
R4_AMBIGUOUS = {"MAGU"}     # Busega was split off Magu into Simiyu in 2012


def ckey(s):
    return " ".join(str(s).split()).strip().casefold()


def letters(s):
    return re.sub(r"[^a-z]", "", unicodedata.normalize("NFKD", str(s)).casefold())


def district_unit(label, cod):
    """religiondots' sources/tz.py rule: the unit whose COD-AB 2018 district names start with
    the label's first word (or, for names of 5+ letters, that the first word starts with);
    None unless exactly one unit matches."""
    lab = str(label).strip()
    if not lab or lab.lower() == "nan":
        return None
    word = letters(DISTRICT_ALIAS.get(lab.upper(), lab.replace("'", " ").split()[0]))
    if not word:
        return None
    hits = set()
    for d, u in cod:
        first = letters(str(d).split()[0])
        if letters(d).startswith(word) or (len(first) >= 5 and word.startswith(first)):
            hits.add(u)
    return next(iter(hits)) if len(hits) == 1 else None


def units(a):
    """Each respondent's unit (geo_id). Labelled rounds by REGION label; R8 (no labels in the
    merged file) by code, decoded from the other rounds; R4 by district."""
    lut = pd.read_csv(LOOKUP, dtype={"geo_id": str})
    say(len(lut) == N_UNITS and int(lut["pop"].sum()) == CENSUS_2022,
        f"religiondots' tz_lookup.csv: {len(lut)} units, {int(lut['pop'].sum()):,} people "
        "(NBS 2022 census)")
    name_id = dict(zip(lut["name"], lut["geo_id"]))
    a["unit_name"] = a["region"].map(ckey).map(NORM)
    lab = a[a["region"].str.strip() != ""]
    bad = sorted(set(lab.loc[lab["unit_name"].isna(), "region"]))
    say(not bad, f"every REGION label names a unit ({bad})")
    per = lab.groupby("region_code")["unit_name"].unique()
    clash = {c: list(v) for c, v in per.items() if len(v) != 1}
    say(not clash, f"every REGION code names one unit in every labelled round ({clash})")
    code_unit = {c: v[0] for c, v in per.items()}
    r8 = a["round"] == 8
    say(a.loc[r8, "region"].str.strip().eq("").all(), "R8 has no REGION labels (decoded by code)")
    a.loc[r8, "unit_name"] = a.loc[r8, "region_code"].map(code_unit)
    say(a.loc[r8, "unit_name"].notna().all(), "every R8 REGION code is decoded")
    # R8's sample design against R7 and R9's, per unit
    sh = pd.crosstab(a["unit_name"], a["round"], normalize="columns")
    r = np.corrcoef(sh[8], (sh[7] + sh[9]) / 2)[0, 1]
    say(r > 0.95, f"R8's decoded sample shares per unit against R7/R9's: r = {r:.3f}")

    dl = pd.read_csv(DISTRICTS, dtype=str)
    nm = dict(zip(lut["geo_id"], lut["name"]))
    cod = list(zip(dl["district"], dl["unit"]))
    r4 = a["round"] == 4
    amb = r4 & a["district"].str.upper().isin(R4_AMBIGUOUS)
    du = a.loc[r4 & ~amb, "district"].map(lambda s: district_unit(s, cod))
    say(du.notna().all(), f"every R4 district decodes to one unit "
        f"({sorted(set(a.loc[du.index[du.isna()], 'district']))})")
    new = du.map(nm)
    ok = [n in SPLIT_CHILDREN.get(o, {o}) for n, o in zip(new, a.loc[du.index, "unit_name"])]
    say(all(ok), "every R4 district lies in its old region or a region carved from it")
    print(f"  R4: {int((new != a.loc[du.index, 'unit_name']).sum())} respondents placed in a "
          f"region created after 2008; {int(amb.sum())} in Magu dropped (split across regions)")
    a.loc[du.index, "unit_name"] = new
    a = a[~amb].copy()
    a["geo_id"] = a["unit_name"].map(name_id)
    say(a["geo_id"].notna().all(), "every respondent has a unit")
    # R6, R7, R9 also carry a district: it must agree with the region label
    for rnd in (6, 7, 9):
        d = a[a["round"] == rnd]
        u = d["district"].map(lambda s: district_unit(s, cod))
        m = u.notna()
        dis = int((m & (u != d["geo_id"])).sum())
        say(dis <= 0.01 * m.sum(), f"R{rnd}: {int(m.sum()):,} of {len(d):,} districts match a "
            f"COD-AB name; {dis} disagree with the region label")
    return a, lut


# ---------------------------------------------------------------- shares

def verbatim_key(v):
    """R7's mother-tongue free text often names the speaker, not the language (Mzaramo, Muha,
    Mshirazi, Wasukuma) or drops the Ki- (Zaramo): try the Ki- form of the same stem."""
    if v in VERBATIM:
        return v
    coded_up = {k.upper(): k for k in CODED}
    stems = [v]
    for p in ("WA", "MU", "M"):
        if v.startswith(p) and len(v) >= len(p) + 2:
            stems.append(v[len(p):])
    for s in stems:
        for k in ("KI" + s, s):
            if k in VERBATIM:
                return k
            if k in coded_up:
                VERBATIM[k] = CODED[coded_up[k]]
                return k
    return v


def answer(row):
    lab = str(row["lang"]).strip()
    if lab in READ_VERBATIM:
        v = " ".join(str(row["verbatim"]).split()).upper()
        if not v or v == "NAN":
            return OTHER_TZ
        v = verbatim_key(v)
        if v not in VERBATIM:
            raise SystemExit(f"verbatim {v!r} (R{row['round']}, {row['region']}) is not in "
                             "VERBATIM: decide it there")
        return VERBATIM[v]
    if lab in NON_ANSWERS:
        return lab
    if lab not in CODED:
        raise SystemExit(f"card label {lab!r} (R{row['round']}) is not in CODED: decide it there")
    return CODED[lab]


def drop_interviewers(a):
    """R7: leave out DROP_INTERVIEWERS' respondents, after checking each recorded Swahili as
    mother tongue for 80%+ of respondents while the rest of R7's interviewers in the same
    regions recorded it for under a quarter. Prints every R7 interviewer's share."""
    r7 = a[a["round"] == 7]
    sw = r7.assign(s=(r7["answer"] == SWAHILI) * r7["w"]).groupby("interviewer")["s"].sum() \
        / r7.groupby("interviewer")["w"].sum()
    print("  R7 Q2A Swahili by interviewer: " + ", ".join(
        f"{k} {v:.0%}" for k, v in sw.sort_values(ascending=False).items() if v > 0))
    for iv in sorted(DROP_INTERVIEWERS):
        regs = set(r7.loc[r7["interviewer"] == iv, "geo_id"])
        mates = r7[r7["geo_id"].isin(regs) & ~r7["interviewer"].isin(DROP_INTERVIEWERS)]
        m = mates.loc[mates["answer"] == SWAHILI, "w"].sum() / mates["w"].sum()
        say(sw[iv] >= 0.8 and m < 0.25, f"R7 interviewer {iv}: Swahili mother tongue {sw[iv]:.0%}"
            f" of respondents; other interviewers in the same {len(regs)} units {m:.0%}")
    drop = (a["round"] == 7) & a["interviewer"].isin(DROP_INTERVIEWERS)
    print(f"  {int(drop.sum())} R7 respondents of {sorted(DROP_INTERVIEWERS)} left out")
    return a[~drop].copy()


def single_verbatims(a):
    """A language named only in free text, by one respondent in the pool: on OTHER_TZ."""
    coded = set(CODED.values())
    n = a.groupby("answer").size()
    one = sorted(k for k, v in n.items() if v == 1 and k not in coded
                 and k not in (OTHER_TZ, "Other language"))
    a.loc[a["answer"].isin(one), "answer"] = OTHER_TZ
    return one


def unit_shares(a):
    """Swahili from SW_ROUNDS; every other answer from all drawn rounds among the rest."""
    s = a[a["round"].isin(SW_ROUNDS)]
    sw = (s[s["answer"] == SWAHILI].groupby("geo_id")["w"].sum()
          / s.groupby("geo_id")["w"].sum()).fillna(0)
    rest = a[a["answer"] != SWAHILI]
    r = rest.groupby(["geo_id", "answer"])["w"].sum()
    r = r.div(r.groupby(level="geo_id").sum(), level="geo_id")
    gids = sorted(set(a["geo_id"]))
    sw = sw.reindex(gids).fillna(0)
    # a unit where every non-Swahili answer is missing would leave a hole: none is, asserted
    have = set(r.index.get_level_values(0))
    say(set(gids) <= have | set(sw[sw >= 1].index), "every unit has non-Swahili answers to "
        "fill what Swahili leaves")
    r = r.mul(1 - sw, level="geo_id")
    e = pd.Series(sw.values, index=pd.MultiIndex.from_arrays([sw.index, [SWAHILI] * len(sw)],
                                                             names=["geo_id", "answer"]))
    return pd.concat([e, r]).sort_index()


def main():
    if "--fetch" in sys.argv:
        fetch()
    a = pd.read_csv(EXTRACT, dtype=str, keep_default_na=False)
    a["round"] = a["round"].astype(int)
    a["w"] = a["w"].astype(float)
    a["region_code"] = a["region_code"].astype(int)
    print(f"  {len(a):,} respondents in the extract, rounds {sorted(a['round'].unique())}")
    a["answer"] = a.apply(answer, axis=1)

    sw = (a.assign(s=(a["answer"] == SWAHILI) * a["w"]).groupby("round")["s"].sum()
          / a.groupby("round")["w"].sum())
    print("  Swahili, weighted share of each round's answers: "
          + ", ".join(f"R{r} {v:.1%}" for r, v in sw.items()))

    n5 = int((a["round"] == 5).sum())
    a = a[a["round"].isin(DRAWN_ROUNDS)].copy()
    print(f"  R5 left out ({n5:,} respondents): its 26 old regions cannot place the five split "
          "in 2012, and it has no district column (religiondots' sources/tz.py)")
    a, lut = units(a)
    n0 = len(a)
    a = a[~a["answer"].isin(NON_ANSWERS)].copy()
    print(f"  {n0 - len(a)} non-answers dropped")
    a = drop_interviewers(a)
    one = single_verbatims(a)
    print(f"  {len(one)} languages named in free text by one respondent -> {OTHER_TZ}: {one}")
    ct = pd.crosstab(a["answer"], a["round"])
    ct["all"] = ct.sum(axis=1)
    print(ct.sort_values("all", ascending=False).to_string())

    pop = lut.set_index("geo_id")["pop"].astype(float)
    sh = unit_shares(a)
    s = sh.groupby(level="geo_id").sum()
    say((s - 1).abs().max() < 1e-9 and len(s) == N_UNITS, "every unit's shares sum to 1")
    n_u = a.groupby("geo_id").size()
    n_sw = a[a["round"].isin(SW_ROUNDS)].groupby("geo_id").size()
    nm = dict(zip(lut["geo_id"], lut["name"]))
    print(f"  respondents per unit: min {n_u.min()} ({nm[n_u.idxmin()]}), median "
          f"{int(n_u.median())}, max {n_u.max()} ({nm[n_u.idxmax()]}); in R{SW_ROUNDS} (Swahili): "
          f"min {n_sw.min()} ({nm[n_sw.idxmin()]}), median {int(n_sw.median())}")

    df = sh.rename("share").reset_index()
    df["count"] = df["share"] * df["geo_id"].map(pop)
    out = []
    for g_id, g in df.groupby("geo_id"):
        f = g["count"].to_numpy()
        base = np.floor(f)
        k = int(round(pop[g_id] - base.sum()))
        base[np.argsort(-(f - base))[:k]] += 1
        out.append(g.assign(count=base.astype(int)))
    df = pd.concat(out)
    say(int(df["count"].sum()) == CENSUS_2022, f"drawn total {int(df['count'].sum()):,}")
    nat = df.groupby("answer")["count"].sum().sort_values(ascending=False)
    print("\n  national, as drawn:")
    for k, v in nat.items():
        print(f"    {k:24s} {v:>12,}  {v / CENSUS_2022:6.2%}")
    print("\n  Swahili share by unit: " + ", ".join(
        f"{nm[g]} {v:.0%}" for g, v in df[df["answer"] == SWAHILI]
        .set_index("geo_id")["share"].sort_values().items()))

    n_by = a.groupby(["geo_id", "answer"]).size()
    df["note"] = [f"share {r.share:.4f}; {n_by.get((r.geo_id, r.answer), 0)} respondents "
                  f"(of {n_u[r.geo_id]})" for r in df.itertuples()]
    df = df[df["count"] > 0]
    res = pd.DataFrame({
        "geo_id": df["geo_id"], "geo_level": "region", "geo_name": df["geo_id"].map(nm),
        "source_category": df["answer"], "count": df["count"], "tier": "modelled",
        "source_id": SOURCE_ID, "year": "2008-2022", "note": df["note"],
    })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.sort_values(["geo_id", "count"], ascending=[True, False]).to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(res)} rows, {res['source_category'].nunique()} answers)")
    a.to_csv(RAW / "ab_tz_resolved.csv", index=False)
    split_half(a, nm)
    district_shares(a, df, nm)


# ---------------------------------------------------------------- districts (placement only)

ADM2 = RD.parent / "religiondots" / "data" / "raw" / "tz" / "adm2" / "tza_admbnda_adm2_20181019.shp"
OUT_DIST = HERE / "data" / "normalized" / "tz_district.csv"
K = 8.0          # prior weight, in respondents: one enumeration area (Nigeria's choice)
NEAREST = 3      # sampled districts an unsampled district borrows from
URBAN_WORDS = ("MANISPAA", "MUNICIPAL", "MJINI", "MJI", "JIJI", "CITY COUNCIL", "URBAN",
               "TOWNSHIP AUTHORITY")
RURAL_WORDS = ("VIJIJINI", "RURAL", "'R'")
# survey label (after the urban/rural words are cut) -> COD-AB 2018 base name
DIST_ALIAS = {
    "BUMBULI": "LUSHOTO",        # split from Lushoto in 2013, after COD-AB's boundaries
    "BUTIAMA": "BUTIAM", "SINGEA": "SONGEA", "SINYANGA": "SHINYANGA", "CHAKECHAKE": "CHAKE CHAKE",
    "KASKAZINI 'A'": "KASKAZINI A",
    "ARUMERU": "MERU",           # split into Arusha and Meru districts; religiondots' alias
}
# R4's "KASKAZINI" (North Unguja, A or B unsaid) is left without a district.


def dkey(s):
    """A district name -> (base, urban?)."""
    s = " ".join(str(s).upper().replace("’", "'").split())
    s = s.replace("'", "") if s != "KASKAZINI 'A'" and not s.endswith(" 'R'") else s
    urban = False
    for w in URBAN_WORDS:
        if s.endswith(" " + w):
            s, urban = s[: -len(w) - 1], True
    for w in RURAL_WORDS:
        if s.endswith(" " + w):
            s = s[: -len(w) - 1]
    s = DIST_ALIAS.get(s, s)
    return s.strip(), urban


def district_shares(a, df, nm):
    """Each language's share in each COD-AB district, for placing dots inside a unit.

    Respondents with a district label (R4 DISTRICT; R6, R7, R9 LOCATION.LEVEL.1; R8 has none)
    are joined to COD-AB 2018 districts by name within their unit. Swahili's district share
    comes from SW_ROUNDS respondents, every other language's from the non-Swahili respondents
    of all drawn rounds, scaled to what Swahili leaves: the unit-level construction, district by
    district. Each is shrunk to the unit's own share with K respondents' weight. A district no
    respondent was placed in borrows the inverse-square-distance mean of the NEAREST sampled
    districts of its unit. Placement only: the counts are the unit's."""
    import geopandas as gpd
    g = gpd.read_file(ADM2, engine="pyogrio")
    say(len(g) == 170, f"COD-AB 2018 admin2: {len(g)} districts")
    dl = pd.read_csv(DISTRICTS, dtype=str)
    unit_of = dict(zip(dl["district"], dl["unit"]))
    g["unit"] = g["ADM2_EN"].map(unit_of)
    say(g["unit"].notna().all(), "every COD-AB district has a unit (religiondots' tz_districts.csv)")
    c = g.to_crs(32737).representative_point()
    g["x"], g["y"] = c.x, c.y
    keys = {}
    for r in g.itertuples():
        keys.setdefault((r.unit, dkey(r.ADM2_EN)), []).append(r.ADM2_PCODE)
    say(all(len(v) == 1 for v in keys.values()), "COD-AB district keys are unique in their unit")

    def match(unit, label):
        if not str(label).strip() or str(label).upper().strip() == "KASKAZINI":
            return None
        b, u = dkey(label)
        for k in ((b, u), (b, not u)):
            if (unit, k) in keys:
                return keys[(unit, k)][0]
        return None
    a = a.copy()
    a["adm2"] = [match(u, l) for u, l in zip(a["geo_id"], a["district"])]
    lab = a[(a["round"] != 8) & (a["district"].str.strip() != "")]
    miss = lab[lab["adm2"].isna()]
    print(f"\n  districts: {int(lab['adm2'].notna().sum()):,} of {len(lab):,} labelled respondents "
          f"joined to a COD-AB district in their unit; {a['adm2'].nunique()} of 170 districts "
          f"sampled. Unjoined labels: {miss.groupby('district').size().to_dict()}")
    say(len(miss) <= 0.02 * len(lab), "98%+ of district labels join")

    unit_share = df.pivot_table(index="geo_id", columns="answer", values="share", fill_value=0.0)
    rows = []
    for u, gu in g.groupby("unit"):
        b = a[(a["geo_id"] == u) & a["adm2"].notna()]
        us = unit_share.loc[u]
        sw_u = float(us.get(SWAHILI, 0.0))
        langs = [k for k in us.index if us[k] > 0 and k != SWAHILI]
        rest_u = us[langs] / max(1e-12, us[langs].sum())
        own = {}
        for d, bd in b.groupby("adm2"):
            s = bd[bd["round"].isin(SW_ROUNDS)]
            n_s = s["w"].sum()
            sw = (s.loc[s["answer"] == SWAHILI, "w"].sum() + K * sw_u) / (n_s + K)
            r = bd[bd["answer"] != SWAHILI]
            t = r.groupby("answer")["w"].sum().reindex(langs).fillna(0.0)
            rest = (t + K * rest_u) / (t.sum() + K)
            own[d] = pd.concat([pd.Series({SWAHILI: sw}), rest * (1 - sw)])
        samp = list(own)
        for r in gu.itertuples():
            if r.ADM2_PCODE in own:
                v, how = own[r.ADM2_PCODE], "sampled"
            elif samp:
                sx = gu.set_index("ADM2_PCODE").loc[samp]
                d2 = ((sx["x"] - r.x) ** 2 + (sx["y"] - r.y) ** 2).to_numpy()
                near = np.argsort(d2)[:NEAREST]
                w = 1.0 / np.maximum(d2[near], 1e6)
                v = sum(own[samp[i]] * w[j] for j, i in enumerate(near)) / w.sum()
                how = "borrowed"
            else:
                v, how = us[us > 0], "unit"
            for k, x in v.items():
                if x > 0:
                    rows.append((u, r.ADM2_PCODE, r.ADM2_EN, k, float(x), how))
    t = pd.DataFrame(rows, columns=["geo_id", "adm2_pcode", "adm2_name", "source_category",
                                    "share", "basis"])
    tot = t.groupby("adm2_pcode")["share"].sum()
    say((tot - 1).abs().max() < 1e-6, "every district's shares sum to 1")
    b = t.drop_duplicates("adm2_pcode")["basis"].value_counts().to_dict()
    print(f"  district basis: {b}")
    t.to_csv(OUT_DIST, index=False)
    print(f"wrote {OUT_DIST} ({len(t)} rows)")
    # a look: Mara and Morogoro, the most mixed units, by district
    for u in ("TZ20", "TZ05"):
        tu = t[t["geo_id"] == u]
        print(f"  {nm[u]}:")
        for d, x in tu.groupby("adm2_name"):
            top = x.sort_values("share", ascending=False).head(4)
            print(f"    {d:16s} " + ", ".join(f"{r.source_category} {r.share:.0%}"
                                              for r in top.itertuples()) + f"  [{x['basis'].iloc[0]}]")


def split_half(a, nm):
    """R4+R6 against R7-R9, unit shares of every non-Swahili answer 2%+ of the pool: do the two
    halves put each language in the same units?"""
    x = a[a["answer"] != SWAHILI]

    def shares(b):
        t = b.groupby(["geo_id", "answer"])["w"].sum()
        return t.div(t.groupby(level="geo_id").sum(), level="geo_id").unstack(fill_value=0)
    h1, h2 = shares(x[x["round"] <= 6]), shares(x[x["round"] >= 7])
    idx = h1.index.intersection(h2.index)
    cols = h1.columns.union(h2.columns)
    h1 = h1.reindex(index=idx, columns=cols, fill_value=0)
    h2 = h2.reindex(index=idx, columns=cols, fill_value=0)
    natw = x.groupby("answer")["w"].sum() / x["w"].sum()
    print(f"\n  split-half, R4+R6 against R7-R9, non-Swahili answers, Pearson r across "
          f"{len(idx)} units:")
    for k in natw[natw >= 0.02].sort_values(ascending=False).index:
        r = np.corrcoef(h1[k], h2[k])[0, 1]
        print(f"    {k:16s} {natw[k]:6.1%}   r = {r:+.3f}")
        if natw[k] >= 0.03:
            say(r > 0.8, f"{k}: the two halves agree on where it is (r {r:+.3f})")


if __name__ == "__main__":
    main()
