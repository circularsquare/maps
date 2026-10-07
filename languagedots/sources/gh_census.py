"""Ghana: 2021 PHC ethnicity by district, read as language with Afrobarometer's retention shares.

    python sources/gh_census.py --fetch   two PxWeb cubes from GSS StatsBank -> data/raw/gh/
    python sources/gh_census.py           -> data/normalized/gh.csv

THE CENSUS. Ghana's 2021 PHC asked no language question (its language items are literacy).
It asked ethnicity, which GSS defines as "a grouping defined by common language, culture and
history with which a person identifies, or by mother tongue" (Vol 3C §3.2), and StatsBank
publishes it for Ghanaians only, as nine major groups plus Others, on 261 districts with the
six metros split into 17 sub-metros: `ethnic_table.px`. The same geography religiondots draws
Ghana on, so its boundary lookup is reused (read-only).

THE NINE GROUPS ARE CLUSTERS, NOT LANGUAGES. Mole-Dagbani is Dagbani, Dagaare, Gurene, Kusaal,
Mampruli, Waali and more; Guan is a dozen languages and many Guan communities now speak Akan;
Ga-Dangme is two languages. So each district's count of a group is shared across the languages
its members speak at home, in the shares Afrobarometer measures for that group (six pooled
rounds, 2008-2022, 13,169 Ghanaian respondents who each named an ethnic group and a home
language; sources/gh_afro.py). That one step is the retention check of AGENT_BRIEF §2 and the
split of each cluster into languages at once: a Guan respondent who speaks Akan at home moves
the Guan count towards Akan.

THE SHARES ARE LOCAL WHERE THE SURVEY CAN BE. P(language | group) is estimated at four nested
levels and each is shrunk towards the one above (a respondent-count prior of K): national ->
old region (Afrobarometer's 10 regions before 2019) -> 2021 region -> district. Respondents
are put in a 2021 region by their round's own region code (R8, R9) or by their district's
name (R4, R6, R7, R9); R5 has no district. A district with a few dozen respondents of a group
moves its shares a long way; one without any takes its region's.

GA AND DANGME. Afrobarometer codes the two as one answer, "Ga/Dangbe". The census's literacy
cube (`GHLang_table.px`, people literate in each Ghanaian language, by district) names them
apart, so each district's Ga/Dangme count is split by its own Ga : Dangme literacy ratio
(region's ratio where the district has fewer than 200 literate in the two).

Every row is `modelled`: the census counts the people and the group, the survey supplies the
language.
"""
import importlib.util
import json
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
RAW = HERE / "data" / "raw" / "gh"
OUT = HERE / "data" / "normalized" / "gh.csv"
AB = RAW / "ab_gh_language.csv"

API = "https://statsbank.statsghana.gov.gh/api/v1/en/PHC%202021%20StatsBank/"
CUBES = {
    "ethnic_table.json": ("Population/ethnic_table.px", "Ethnicity"),
    "ghlang_table.json": ("Education%20and%20Literacy/GHLang_table.px",
                          "Ghanaian_language_of_literacy"),
}
GHANAIANS = 30_484_536          # Vol 3C Table 5.5, all Ghanaians
SOURCE_ID = "gss_phc2021_ethnic_x_afrobarometer_r4_r9"
K = 30                          # shrinkage prior, in weighted respondents
GROUPS = ["Akan", "Ga-Dangme", "Ewe", "Guan", "Gurma", "Mole-Dagbani", "Grusi", "Mande",
          "Others"]


def say(ok, msg):
    print(("  ok   " if ok else "  FAIL ") + msg)
    if not ok:
        raise SystemExit(msg)


def _rd_gh():
    """religiondots' Ghana module, for its cube axes and geography nesting (read-only)."""
    spec = importlib.util.spec_from_file_location("rd_gh", RD / "sources" / "gh.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- fetch

def fetch():
    import requests
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    RAW.mkdir(parents=True, exist_ok=True)
    ua = {"User-Agent": "Mozilla/5.0"}
    for name, (path, dim) in CUBES.items():
        dest = RAW / name
        if dest.exists() and dest.stat().st_size > 10_000:
            print("already have", dest)
            continue
        # StatsBank omits its TLS intermediate certificate; religiondots' sources/gh.py says
        # why verification is off for this one host and the cube is checked structurally.
        meta = requests.get(API + path, timeout=120, verify=False, headers=ua).json()
        codes = [v["code"] for v in meta["variables"]]
        say(dim in codes and "Geographic_Area" in codes, f"{name}: axes {codes}")
        # an axis that eliminates gives its total when left out; one that does not (the
        # literacy cube's Locality) is asked for its first value, which is the total
        q = {"query": [{"code": c, "selection": {"filter": "all", "values": ["*"]}}
                       for c in (dim, "Geographic_Area")],
             "response": {"format": "json-stat2"}}
        for v in meta["variables"]:
            if v["code"] not in (dim, "Geographic_Area") and not v.get("elimination"):
                say(v["valueTexts"][0].lower().startswith("all"),
                    f"{name}: {v['code']} first value is a total ({v['valueTexts'][0]})")
                q["query"].append({"code": v["code"],
                                   "selection": {"filter": "item", "values": [v["values"][0]]}})
        r = requests.post(API + path, json=q, timeout=300, verify=False, headers=ua)
        r.raise_for_status()
        doc = r.json()
        say("value" in doc and {dim, "Geographic_Area"} <= set(doc["id"])
            and all(n == 1 for d, n in zip(doc["id"], doc["size"])
                    if d not in (dim, "Geographic_Area")),
            f"{name}: a json-stat2 cube {doc.get('size')}")
        dest.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
        print(f"  {dest} {dest.stat().st_size:,} bytes")


def read_cube(name, dim, total_cat):
    rd = _rd_gh()
    doc = json.loads((RAW / name).read_text(encoding="utf-8"))
    ids, sizes, codes, labels = rd._axes(doc)
    stride, acc = {}, 1
    for d, n in zip(reversed(ids), reversed(sizes)):
        stride[d] = acc
        acc *= n
    geo, cats = codes["Geographic_Area"], codes[dim]
    lab = labels[dim]
    rows = []
    for g in geo:
        for c in cats:
            v = doc["value"][stride[dim] * cats.index(c) + stride["Geographic_Area"] * geo.index(g)]
            rows.append((labels["Geographic_Area"][g], lab[c], 0 if v is None else int(v)))
    df = pd.DataFrame(rows, columns=["geo", "cat", "count"])
    total = df[df["cat"] == total_cat].set_index("geo")["count"].to_dict()
    level, region_of, parent_of = rd._nest(list(dict.fromkeys(df["geo"])), total)
    df["level"] = df["geo"].map(level)
    df["region"] = df["geo"].map(region_of).fillna("")
    return df, rd.REGIONS


# ---------------------------------------------------------------- names

def norm_name(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().upper()
    s = re.sub(r"\(.*?\)", " ", s)
    s = re.sub(r"\b(MUNICIPAL|METROPOLITAN|METRO|DISTRICT|ASSEMBLY|AREA|MA|DA|MMDA)\b", " ", s)
    s = re.sub(r"[^A-Z]+", " ", s)
    s = " ".join(s.split())
    s = re.sub(r"^(A M A|K M A|T M A|AMA|KMA|TMA|STMA|CCMA)\b ?", "", s)  # metro prefixes
    return s


OLD_REGION = {  # 2021 region -> the 2018 region it was carved from (Afrobarometer R4-R7)
    "Western": "Western", "Western North": "Western", "Central": "Central",
    "Greater Accra": "Greater Accra", "Volta": "Volta", "Oti": "Volta", "Eastern": "Eastern",
    "Ashanti": "Ashanti", "Ahafo": "Brong Ahafo", "Bono": "Brong Ahafo",
    "Bono East": "Brong Ahafo", "Northern": "Northern", "Savannah": "Northern",
    "North East": "Northern", "Upper East": "Upper East", "Upper West": "Upper West",
}
# a bare metro acronym in Afrobarometer's district column ("A M A") -> the metro's name
METRO_BARE = {("Greater Accra", "A"): "ACCRA", ("Ashanti", "K"): "KUMASI",
              ("Greater Accra", "T"): "TEMA", ("Northern", "T"): "TAMALE"}


# ---------------------------------------------------------------- the survey's answers

# Every home-language answer (coded label or "Other" verbatim, title-cased) -> the language
# drawn. A dialect or town goes on its language. "Ga/Dangme" and "Akan" are split later by
# the census's own literacy table (module docstring). Names follow Glottolog.
LANG = {
    "Akan": "Akan", "Fanti": "Akan", "Fante": "Akan", "Wassa": "Akan", "Assin": "Akan",
    "Brosa": "Akan", "Brossa": "Akan", "Blosa": "Akan",        # Abron (Bono) speech, Akan
    "Nzema": "Nzema", "Sefwi": "Sehwi",
    "Chokosi": "Anufo", "Chakosi": "Anufo", "Chukusi": "Anufo", "Tokosi": "Anufo",
    "Ga/Dangbe": "Ga/Dangme", "Krobo": "Dangme",               # Krobo is a Dangme dialect
    "Ewe": "Ewe", "Ewe/Anlo": "Ewe",
    "English": "English", "Hausa": "Hausa",
    "Fulani": "Fula", "Zamrama": "Zarma", "Zabrama": "Zarma",
    # Guan
    "Gonja": "Gonja", "Guan": "Guan (not named)",
    "Krakyi": "Krache", "Krachi": "Krache", "Krachie": "Krache", "Kratsi": "Krache",
    "Akyode": "Gikyode", "Achode": "Gikyode", "Gekyode": "Gikyode",
    "Nchumuru": "Chumburung", "Tsumuru": "Chumburung", "Chimullu": "Chumburung",
    "Nawuri": "Nawuri", "Nawurt": "Nawuri", "Efutu": "Efutu", "Larteh": "Larteh",
    "Atwede": "Guan (not named)", "Atweti": "Guan (not named)", "Okre": "Guan (not named)",
    "Kyakwesi": "Guan (not named)", "Nkura": "Guan (not named)", "Dwan": "Guan (not named)",
    # Ghana-Togo Mountain languages (the census files their speakers under Guan)
    "Buem": "Lelemi", "Lelemi": "Lelemi", "Siwu": "Siwu", "Akpafu Dialect": "Siwu",
    "Sekpele": "Sekpele", "Likpe": "Sekpele", "Tegbor": "Tafi", "Bowire": "Bowiri",
    "Bowiri": "Bowiri", "Adele": "Adele",
    # Mole-Dagbani (Oti-Volta)
    "Dagbani": "Dagbani", "Dagomba": "Dagbani", "Nanum": "Dagbani",   # Nanuni, a dialect
    "Mampruli": "Mampruli", "Mamprusi": "Mampruli", "Mamprugu": "Mampruli",
    "Mamprigu": "Mampruli",
    "Frafra": "Gurene", "Fafra": "Gurene", "Gruni": "Gurene", "Guruni": "Gurene",
    "Frafrah": "Gurene", "Nankani": "Gurene", "Nankana": "Gurene", "Nankane": "Gurene",
    "Gurune Or Frafra": "Gurene", "Gurune/Frafra": "Gurene", "Gurune /Frafra": "Gurene",
    "Gurune / Frafra": "Gurene", "Frafra Or Gurune": "Gurene",
    "Talensi": "Talni", "Taln": "Talni", "Talansi": "Talni",
    "Nabdam": "Nabit", "Nabt": "Nabit", "Nabd": "Nabit",
    "Kusaal": "Kusaal", "Kusasi": "Kusaal", "Kusal": "Kusaal", "Kusasi/Kusaase": "Kusaal",
    "Kusaasi": "Kusaal", "Kosaase": "Kusaal", "Kusasi And Twi": "Kusaal",
    "Dagaare": "Dagaare", "Dagaari": "Dagaare", "Dagaree": "Dagaare", "Dagate": "Dagaare",
    "Dagaate": "Dagaare", "Dagarti": "Dagaare", "Tei And Dagare": "Dagaare",
    "Waala": "Waali", "Waale": "Waali", "Wala": "Waali", "Waali": "Waali",
    "Wa Dormu": "Waali",
    "Brefo/Birfuo": "Birifor", "Brefo": "Birifor", "Birifori": "Birifor", "Brifo": "Birifor",
    "Brifor": "Birifor",
    "Buli": "Buli", "Bulsa": "Buli", "Builsa": "Buli", "Bule": "Buli", "Bulisa": "Buli",
    "Kanjaga": "Buli",
    "Moshie": "Mooré", "Mossi": "Mooré", "Moosi": "Mooré", "Moar": "Mooré",
    "Hanga": "Hanga", "Safalba": "Safaliba",
    # Gurma
    "Konkonba": "Konkomba", "Konkomba": "Konkomba", "Kokonba": "Konkomba",
    "Likpakpaln": "Konkomba", "Likpakpaa": "Konkomba", "Likpkpaa": "Konkomba",
    "Lakpakaw": "Konkomba", "Lekpekpe": "Konkomba",
    "Bimoba": "Bimoba", "Binmuba": "Bimoba", "Bemobe": "Bimoba", "Bemuba": "Bimoba",
    "Bomoba": "Bimoba", "Bemoba": "Bimoba",
    "Basare": "Ntcham (Basari)", "Basari": "Ntcham (Basari)", "Baasare": "Ntcham (Basari)",
    "Basseli": "Ntcham (Basari)", "Kasari": "Ntcham (Basari)",
    "Gruma": "Gourmanchéma", "Groma": "Gourmanchéma", "Gurma": "Gourmanchéma",
    "Chamba": "Akaselem (Chamba)", "Kabre": "Kabiyè",
    # Grusi
    "Kasem": "Kasem", "Kasena": "Kasem", "Kassena": "Kasem", "Kasina": "Kasem",
    "Kassem": "Kasem", "Kaseem": "Kasem", "Kassim": "Kasem", "Kassina Nankani": "Kasem",
    "Sisila": "Sisaala", "Sissali": "Sisaala", "Sissala": "Sisaala", "Sissale": "Sisaala",
    "Sisala": "Sisaala", "Sisaasla": "Sisaala",
    "Tampulima": "Tampulma", "Tampulma": "Tampulma", "Tampuma": "Tampulma",
    "Tapulsi": "Tampulma",
    "Mo": "Deg", "Moli": "Deg", "Mor": "Deg", "Deg-(Moo)": "Deg",
    "Kotokoli": "Tem", "Kotonkoli": "Tem",
    "Tsala": "Chala", "Challa": "Chala",
    "Grussi": "Gurunsi (not named)", "Grusi": "Gurunsi (not named)",
    "Gurusi": "Gurunsi (not named)", "Grushie": "Gurunsi (not named)",
    "Grushi": "Gurunsi (not named)", "Grusa": "Gurunsi (not named)",
    # other Gur
    "Banda": "Nafaanra", "Nafaana": "Nafaanra", "Nafana": "Nafaanra", "Fantara": "Nafaanra",
    "Lobi": "Lobi",
    # Mande
    "Bissa": "Bissa", "Busanga": "Bissa", "Bosanga": "Bissa", "Bisa": "Bissa",
    "Buzanga": "Bissa", "Bisab": "Bissa", "Busi": "Bissa",
    "Wangara": "Dyula", "Wangra": "Dyula",
    "Kabre ": "Kabiyè",
}
# coded labels that are not answers
NON_ANSWERS = {"Refused", "Don't know", "Refused To Answer"}
OTHER_LABELS = {"Other", "Others", "Other Northern Languages"}

# A combined label split by the respondent's own ethnic group
WAALE_ETH = {"Waali", "Wale", "Waala", "Waale", "Wala"}

# Remainder for an "Other" answer whose verbatim names nothing identifiable: the narrowest
# node holding what the respondent's census group speaks.
REMAINDER = {"Akan": "Kwa (not named)", "Ga-Dangme": "Kwa (not named)",
             "Ewe": "Kwa (not named)", "Guan": "Guan (not named)",
             "Gurma": "Gur (not named)", "Mole-Dagbani": "Gur (not named)",
             "Grusi": "Gurunsi (not named)", "Mande": "Mande (not named)",
             "Others": "Other African language"}

# Afrobarometer's ethnic labels and verbatims -> the census's nine groups. The census files
# its sub-groups as GSS does (Vol 3C names only the nine), and the district cube confirms the
# arguable ones: Chereponi is 66% Akan (the Chokosi, Anufo speakers), Bawku 17% Mande (the
# Bisa), Banda 58% Mande, Builsa North 96% Mole-Dagbani, Bunkpurugu 86% Gurma (the Bimoba),
# Kasena Nankana 90% Grusi.
GROUP_OF_LANG = {
    "Akan": "Akan", "Nzema": "Akan", "Sehwi": "Akan", "Anufo": "Akan",
    "Ga/Dangme": "Ga-Dangme", "Dangme": "Ga-Dangme", "Ewe": "Ewe",
    "Gonja": "Guan", "Guan (not named)": "Guan", "Krache": "Guan", "Gikyode": "Guan",
    "Chumburung": "Guan", "Nawuri": "Guan", "Efutu": "Guan", "Larteh": "Guan",
    "Lelemi": "Guan", "Siwu": "Guan", "Sekpele": "Guan", "Tafi": "Guan", "Bowiri": "Guan",
    "Adele": "Guan",
    "Dagbani": "Mole-Dagbani", "Mampruli": "Mole-Dagbani", "Gurene": "Mole-Dagbani",
    "Talni": "Mole-Dagbani", "Nabit": "Mole-Dagbani", "Kusaal": "Mole-Dagbani",
    "Dagaare": "Mole-Dagbani", "Waali": "Mole-Dagbani", "Birifor": "Mole-Dagbani",
    "Buli": "Mole-Dagbani", "Mooré": "Mole-Dagbani", "Hanga": "Mole-Dagbani",
    "Safaliba": "Mole-Dagbani",
    "Konkomba": "Gurma", "Bimoba": "Gurma", "Ntcham (Basari)": "Gurma",
    "Gourmanchéma": "Gurma", "Akaselem (Chamba)": "Gurma", "Kabiyè": "Gurma",
    "Kasem": "Grusi", "Sisaala": "Grusi", "Tampulma": "Grusi", "Deg": "Grusi", "Tem": "Grusi",
    "Chala": "Grusi", "Gurunsi (not named)": "Grusi",
    "Nafaanra": "Mande", "Bissa": "Mande", "Dyula": "Mande", "Lobi": "Others",
    "Hausa": "Others", "Fula": "Others", "Zarma": "Others", "English": "Others",
}
ETH = {  # coded ethnic labels (and verbatims) that are not a language name above
    "Ga/Adangbe": "Ga-Dangme", "Ewe/Anlo": "Ewe", "Ewe/Anglo": "Ewe", "Dagomba": "Mole-Dagbani",
    "Kusasi": "Mole-Dagbani", "Frafra": "Mole-Dagbani", "Frafri": "Mole-Dagbani",
    "Dagaaba": "Mole-Dagbani", "Dagaati": "Mole-Dagbani", "Dagarti": "Mole-Dagbani",
    "Dagao": "Mole-Dagbani", "Dagaate": "Mole-Dagbani", "Dagate": "Mole-Dagbani",
    "Waali": "Mole-Dagbani", "Wale": "Mole-Dagbani", "Waala": "Mole-Dagbani",
    "Waale": "Mole-Dagbani", "Waalu": "Mole-Dagbani", "Nandom": "Mole-Dagbani",
    "Mamprusi": "Mole-Dagbani", "Mole-Dagbani": "Mole-Dagbani", "Mosi": "Mole-Dagbani",
    "Moshie": "Mole-Dagbani", "Other Northern tribes": None,
    "Konkomba": "Gurma", "Konkonba": "Gurma", "Gurma": "Gurma", "Gruma": "Gurma",
    "Groma/Grumah": "Gurma", "Kokonba": "Gurma",
    "Grusi": "Grusi", "Kasina": "Grusi", "Sissila": "Grusi", "Sisila": "Grusi",
    "Kontonkoli": "Grusi", "Kotokoli": "Grusi",
    "Mande": "Mande", "Buzanga": "Mande", "Busanga": "Mande", "Wangara": "Mande",
    "Banda": "Mande",
    "Hausa": "Others", "Fulani": "Others", "Yoroba": "Others",
    "Akyem": "Akan", "Asante": "Akan", "Ashanti": "Akan", "Bono": "Akan", "Brong": "Akan",
    "Ahanta": "Akan", "Agona": "Akan", "Akuapim": "Akan", "Kwahu": "Akan", "Aduana": "Akan",
    "Asante Akyem": "Akan", "Ningo": "Ga-Dangme", "Tafi": "Guan", "Salaga": "Guan",
    "Agonja": "Guan", "Bowuri": "Guan", "Tsalla": "Grusi", "Nanung": "Mole-Dagbani",
    "Namba": "Mole-Dagbani", "Namuba": "Mole-Dagbani", "Taleng": "Mole-Dagbani",
    "Gurune": "Mole-Dagbani", "Dagaba": "Mole-Dagbani", "Daghaba": "Mole-Dagbani",
    "Dagaw": "Mole-Dagbani", "Dagare": "Mole-Dagbani", "Mampruga": "Mole-Dagbani",
    "Mampriga": "Mole-Dagbani", "Moose": "Mole-Dagbani", "Moshi": "Mole-Dagbani",
    "Mooshe": "Mole-Dagbani", "Moo": "Grusi", "Tamplima": "Grusi", "Tampluma": "Grusi",
    "Tampulinsi": "Grusi", "Tampumba": "Grusi", "Kesana(Grushie)": "Grusi", "Kasim": "Grusi",
    "Gurissi": "Grusi", "Gurucchi": "Grusi", "Likpapaln": "Gurma", "Lekpakpali": "Gurma",
    "Binmoba": "Gurma", "Baasara": "Gurma", "Baasere": "Gurma", "Bassel": "Gurma",
    "Busaga": "Mande", "Busagan": "Mande", "Kussanga": "Mande", "Wangala": "Mande",
    "Wangle": "Mande", "Wangal": "Mande", "Wangrani": "Mande", "Wagra": "Mande",
    "Zamarama": "Others", "Zabarama": "Others", "Brefuo": "Mole-Dagbani",
    "Brefo/Wala": "Mole-Dagbani", "Kotokolie": "Grusi", "Kotokoli Tem": "Grusi",
    "Kotokonli": "Grusi", "Anufo": "Akan", "Chakwasi": "Akan", "Pantra": "Mande",
    "Fantra": "Mande",
}


def say_lang(row):
    """The answer a respondent's home language is drawn as, or None (no answer)."""
    lab = str(row["lang"]).strip()
    if lab in NON_ANSWERS:
        return None
    if lab in OTHER_LABELS:
        v = " ".join(str(row["verbatim"]).split()).title()
        if v in ("", "Nan", "None"):
            return REMAINDER.get(row["group"], "Other African language")
        return LANG.get(v, REMAINDER.get(row["group"], "Other African language"))
    if lab == "Dagaare/Waale":
        return "Waali" if str(row["eth_key"]) in WAALE_ETH else "Dagaare"
    if lab not in LANG:
        raise SystemExit(f"coded language {lab!r} has no LANG entry")
    return LANG[lab]


def group_of(row):
    lab = str(row["eth"]).strip()
    if lab.startswith("Other"):
        v = " ".join(str(row["eth_verbatim"]).split()).title()
        if lab.startswith("Other Northern") and v in ("", "Nan", "None"):
            return None
        key = v
    else:
        key = lab
    if key in ETH:
        return ETH[key]
    if key in LANG and LANG[key] in GROUP_OF_LANG:
        return GROUP_OF_LANG[LANG[key]]
    if key == "Akan":
        return "Akan"
    return None          # national identity only, refused, or a name not placed


# ---------------------------------------------------------------- the model

def survey(units):
    """Afrobarometer respondents with a census group, an answer and their zones."""
    a = pd.read_csv(AB, keep_default_na=False)
    a["eth_key"] = [" ".join(str(v).split()).title() if str(e).startswith("Other") else e
                    for e, v in zip(a["eth"], a["eth_verbatim"])]
    a["group"] = a.apply(group_of, axis=1)
    a["answer"] = a.apply(say_lang, axis=1)
    n0 = a["w"].sum()
    a = a[a["group"].notna() & a["answer"].notna()].copy()
    print(f"  {len(a):,} respondents with a census group and an answer "
          f"({a['w'].sum() / n0:.1%} of the weight)")

    # zones
    a["region"] = a["region"].astype(str).str.strip().str.title()
    fix = {"Brong Ahafo": "Brong Ahafo", "North East": "North East",
           "Bono East": "Bono East", "Western North": "Western North"}
    a["region"] = a["region"].map(lambda r: fix.get(r, r))
    new = set(OLD_REGION)
    a["old"] = a["region"].map(lambda r: OLD_REGION.get(r, r))
    bad = set(a["old"]) - set(OLD_REGION.values())
    say(not bad, f"every respondent has a 2018 region (unplaced: {sorted(bad)})")
    a["new"] = a["region"].where(a["region"].isin(new) & (a["round"] >= 8), "")

    by_norm = {}
    for u in units.itertuples():
        for key in {norm_name(u.geo), norm_name(u.parent)}:
            by_norm.setdefault((u.old, key), set()).add(u.geo)
    import difflib
    cache = {}

    def match(old, name):
        n = norm_name(name)
        if (old, n) in cache:
            return cache[(old, n)]
        raw = " ".join(re.sub(r"[^A-Z]+", " ", str(name).upper()).split())
        if not n:
            m = re.match(r"^([AKT]) ?M ?A$", raw)
            n = METRO_BARE.get((old, m.group(1)), "") if m else ""
        keys = [k for (o, k) in by_norm if o == old]
        hit = set()
        if (old, n) in by_norm:
            hit = by_norm[(old, n)]
        else:
            pre = [k for k in keys if k.startswith(n + " ") or (n.startswith(k + " ") and
                                                                len(k) > 3)]
            if not pre:
                pre = difflib.get_close_matches(n, keys, n=1, cutoff=0.82)
            for k in pre:
                hit |= by_norm[(old, k)]
        cache[(old, n)] = sorted(hit)
        return cache[(old, n)]

    a["units"] = [match(o, d) if str(d).strip() not in ("", "nan") else []
                  for o, d in zip(a["old"], a["district"])]
    has = a["district"].astype(str).str.strip().ne("")
    placed = a["units"].map(len) > 0
    print(f"  district placed for {placed.sum():,} of {has.sum():,} respondents who name one")
    # a respondent placed in districts all of one 2021 region gets that region
    reg_of = dict(zip(units["geo"], units["region"]))
    for i in a.index[(a["new"] == "") & placed]:
        rs = {reg_of[u] for u in a.at[i, "units"]}
        if len(rs) == 1:
            a.at[i, "new"] = rs.pop()
    print(f"  2021 region known for {(a['new'] != '').mean():.1%} of respondents")
    return a


def shrink(counts, prior, k):
    """counts: Series answer -> weight; prior: Series answer -> share."""
    n = counts.sum()
    idx = prior.index.union(counts.index)
    c = counts.reindex(idx, fill_value=0.0)
    p = prior.reindex(idx, fill_value=0.0)
    return (c + k * p) / (n + k), n


# Where each language's speakers live, for the survey's shares to be placed within a region:
# Glottolog's point for the language (CC BY; the glottocode beside it) and a radius in km
# that is roughly its home area. A language not listed is spread evenly (Akan, English,
# Hausa, Ewe, the remainders, and languages whose Glottolog point is outside Ghana and whose
# Ghanaian speakers are scattered: Mooré, Tem, Kabiyè, Akaselem, Gourmanchéma, Fula, Zarma,
# Dyula). Ga/Dangme and Akan/Nzema are split afterwards by the literacy table instead.
HOME = {
    "Dagbani": (9.64745, -0.43227, 70),        # dagb1246
    "Mampruli": (10.3884, -0.74675, 45),       # mamp1244
    "Gurene": (11.0918, -0.81109, 30),         # fare1241 Farefare
    "Talni": (10.80, -0.80, 20),               # taln1239 shares Farefare's point; Tongo
    "Nabit": (10.857778, -0.671111, 20),       # nabi1240
    "Kusaal": (10.9703, -0.38756, 35),         # kusa1250
    "Dagaare": (10.4226, -2.52265, 55),        # sout2789 Central Dagaare
    "Waali": (10.0212, -2.31263, 30),          # wali1263 Wali (Ghana)
    "Birifor": (9.2873, -2.72672, 40),         # sout2790 Southern Birifor
    "Buli": (10.5763, -1.26748, 25),           # buli1254 Buli (Ghana)
    "Hanga": (9.33929, -1.57833, 30),          # hang1258
    "Safaliba": (8.93768, -2.58467, 25),       # safa1243
    "Konkomba": (9.82224, 0.2783, 60),         # konk1269
    "Bimoba": (10.4575, 0.06121, 30),          # bimo1239
    "Ntcham (Basari)": (9.23971, 0.599275, 45),  # ntch1242
    "Anufo": (10.2839, 0.56021, 35),           # anuf1239
    "Kasem": (11.0824, -1.39076, 30),          # kase1253
    "Sisaala": (10.63451, -1.79707, 45),       # tumu1242 Tumulung Sisaala
    "Tampulma": (9.75577, -1.35467, 35),       # tamp1252
    "Deg": (8.43846, -2.28724, 45),            # degg1238
    "Chala": (8.00089, 0.50896, 25),           # chal1269
    "Nafaanra": (8.00522, -2.51001, 35),       # nafa1258 Nafanan
    "Lobi": (9.96122, -3.336, 40),             # lobi1245
    "Bissa": (11.3434, -0.3744, 40),           # biss1248
    "Gonja": (8.48801, -0.72757, 100),         # gonj1241
    "Krache": (7.93475, 0.01925, 35),          # krac1238
    "Gikyode": (8.39478, 0.56975, 30),         # giky1238
    "Chumburung": (8.15211, -0.2755, 35),      # chum1261
    "Nawuri": (8.45116, 0.04947, 30),          # nawu1242
    "Efutu": (5.34699, -0.62723, 15),          # efut1241
    "Larteh": (5.9367, -0.07822, 15),          # lart1238
    "Lelemi": (7.34549, 0.50746, 20),          # lele1264
    "Siwu": (7.23811, 0.44362, 15),            # siwu1238
    "Sekpele": (7.16751, 0.5886, 15),          # sekp1241
    "Tafi": (6.79522, 0.39875, 12),            # tafi1243
    "Bowiri": (7.31095, 0.38008, 15),          # tuwu1238 Tuwuli
    "Adele": (8.16975, 0.619185, 25),          # adel1244
    "Sehwi": (6.3172, -2.73146, 40),           # sehw1238
}
FLOOR = 0.02        # a home kernel never falls below this: migrants live everywhere


def kernels(units, cent):
    """unit x language: 1 for an evenly spread language, else FLOOR + a Gaussian of distance."""
    lat = units["geo"].map(lambda u: cent[u][1]).to_numpy()
    lon = units["geo"].map(lambda u: cent[u][0]).to_numpy()
    out = {}
    for lang, (la, lo, r) in HOME.items():
        dy = (lat - la) * 111.0
        dx = (lon - lo) * 111.0 * np.cos(np.radians((lat + la) / 2))
        out[lang] = FLOOR + np.exp(-0.5 * (dx * dx + dy * dy) / (r * r))
    return pd.DataFrame(out, index=units["geo"])


def _norm(p):
    p = p[p > 0]
    return p / p.sum()


def shares(a, units, piv, cent):
    """P(answer | group) for every unit.

    The survey's shares are taken at four nested levels (national, 2018 region, 2021 region,
    district), each shrunk towards the one above with a prior worth K respondents. The
    prior is not flat inside a level: each language's share is modulated by HOME, so a
    region's prior already knows Efutu is spoken at Winneba and not at Bole, and a region's
    measured shares are passed down to its districts in proportion to where each language
    lives. Spread evenly, Kusaal would sit in Bolgatanga as thickly as in Bawku.
    """
    kern = kernels(units, cent)
    out, diag = {}, []
    rows = []
    for r in a[a["units"].map(len) > 0].itertuples():
        for u in r.units:
            rows.append((u, r.group, r.answer, r.w / len(r.units)))
    ad = pd.DataFrame(rows, columns=["unit", "group", "answer", "w"])
    for g in GROUPS:
        s = a[a["group"] == g]
        say(len(s) >= 50, f"{g}: {len(s):,} respondents nationally")
        nat = s.groupby("answer")["w"].sum() / s["w"].sum()
        c = piv[g].reindex(units["geo"]).astype(float)
        k = pd.DataFrame(1.0, index=units["geo"], columns=nat.index)
        for lang in nat.index:
            if lang in kern.columns:
                k[lang] = kern[lang]
        kbar = (k.mul(c, axis=0).sum() / c.sum())
        prior_d = k.mul(nat / kbar, axis=1)
        prior_d = prior_d.div(prior_d.sum(axis=1), axis=0)

        def prior_of(mask):
            cc = c[mask.to_numpy()]
            pd_ = prior_d[mask.to_numpy()]
            if cc.sum() > 0:
                return pd_.mul(cc, axis=0).sum() / cc.sum()
            return pd_.mean()

        pri_old = {o: prior_of(units["old"] == o) for o in units["old"].unique()}
        pri_new = {r: prior_of(units["region"] == r) for r in units["region"].unique()}
        post_old, post_new = {}, {}
        for o, pr in pri_old.items():
            d = s[s["old"] == o].groupby("answer")["w"].sum()
            post_old[o] = shrink(d, pr, K)
        for r, pr in pri_new.items():
            o = units.loc[units["region"] == r, "old"].iloc[0]
            p_old, _ = post_old[o]
            prior = _norm(p_old * (pr / pri_old[o]).reindex(p_old.index).fillna(1.0))
            d = s[s["new"] == r].groupby("answer")["w"].sum()
            post_new[r] = shrink(d, prior, K)
        for u in units.itertuples():
            p_new, n_new = post_new[u.region]
            ratio = (prior_d.loc[u.geo] / pri_new[u.region]).reindex(p_new.index).fillna(1.0)
            prior = _norm(p_new * ratio)
            d = ad[(ad["group"] == g) & (ad["unit"] == u.geo)].groupby("answer")["w"].sum()
            p_d, n_d = shrink(d, prior, K)
            out[(u.geo, g)] = _norm(p_d)
            diag.append((u.geo, g, post_old[u.old][1], n_new, n_d))
    return out, pd.DataFrame(diag, columns=["unit", "group", "n_old", "n_new", "n_dist"])


# Ask 018 (Anita, 2026-10-05): lingua francas at Afrobarometer R7's mother-tongue question.
LINGUA_FRANCAS = ["English", "Hausa", "Akan"]


def r7_mother(a):
    """R7 respondents of `a` with their Q2A (mother tongue) answer, read as say_lang reads."""
    sys.path.insert(0, str(HERE / "sources"))
    import wafr_afro
    df, meta = wafr_afro._read(wafr_afro.AFB / wafr_afro.R7_FILE,
                               usecols=["COUNTRY", "RESPNO", "Q2A", "Q2AOTHER"],
                               apply_value_formats=True)
    say("tongue" in str(meta.column_names_to_labels["Q2A"]).casefold(), "R7 Q2A is mother tongue")
    df = df[df["COUNTRY"].astype(str).str.strip() == "Ghana"]
    m = dict(zip(df["RESPNO"].astype(str), zip(df["Q2A"].astype(str), df["Q2AOTHER"].astype(str))))
    b = a[a["round"] == 7].copy()
    say(b["respno"].astype(str).isin(m).all(), f"every R7 respondent ({len(b)}) found in R7's Q2A")
    b["lang"] = [m[str(r)][0] for r in b["respno"]]
    b["verbatim"] = [m[str(r)][1] for r in b["respno"]]
    b["answer"] = b.apply(say_lang, axis=1)
    return b[b["answer"].notna()]


def calibrate(sh, a):
    """Each lingua franca's P(answer | group) set to R7's Q2A share among the group's R7
    respondents, shrunk (wafr_afro.K_SHRINK) towards the pooled share x the national Q2A/pooled
    ratio; applied as a factor to every unit's P, the group's other answers scaled to fill."""
    from wafr_afro import K_SHRINK
    b = r7_mother(a)
    print(f"\n  ask 018: {len(b):,} R7 respondents with a group and a mother tongue")
    pooled = a.groupby(["group", "answer"])["w"].sum()
    pooled = pooled.div(pooled.groupby(level="group").sum(), level="group")
    n7 = b.groupby("group")["w"].sum()
    x7 = b.groupby(["group", "answer"])["w"].sum()
    fac = {}
    for lf in LINGUA_FRANCAS:
        rho = (x7.xs(lf, level="answer").sum() / b["w"].sum()) / \
              (a.loc[a["answer"] == lf, "w"].sum() / a["w"].sum())
        for g in GROUPS:
            p0 = pooled.get((g, lf), 0.0)
            if p0 <= 0:
                continue
            q = (x7.get((g, lf), 0.0) + K_SHRINK * p0 * rho) / (n7.get(g, 0.0) + K_SHRINK)
            fac[(g, lf)] = q / p0
            print(f"    {lf:8s} {g:13s} pooled {p0:6.2%}  R7 Q2A {x7.get((g, lf), 0.0) / max(n7.get(g, 0), 1e-9):6.2%}"
                  f" (n {n7.get(g, 0):5.0f})  drawn {q:6.2%}")
    out = {}
    for (u, g), p in sh.items():
        p = p.copy()
        fixed = {lf: min(p[lf] * fac[(g, lf)], 1.0) for lf in LINGUA_FRANCAS
                 if lf in p.index and (g, lf) in fac}
        others = [x for x in p.index if x not in fixed]
        tot = sum(fixed.values())
        if others and p[others].sum() > 0:
            p[others] = p[others] / p[others].sum() * (1 - tot)
        for k, v in fixed.items():
            p[k] = v
        out[(u, g)] = _norm(p)
    return out


def literacy_split(df, lit, regions_of, pair, pooled, floor=200):
    """Split answer `pooled` in each unit by the unit's literacy counts in `pair`."""
    names, cats = pair
    out = []
    reg_lit = lit.groupby("region")[cats].sum()
    for r in df.itertuples():
        if r.answer != pooled:
            out.append((r.unit, r.answer, r.count))
            continue
        row = lit.loc[r.unit, cats] if r.unit in lit.index else None
        if row is None or row.sum() < floor:
            row = reg_lit.loc[regions_of[r.unit]]
        sh = row / row.sum()
        for nm, cs in zip(names, cats):
            out.append((r.unit, nm, r.count * float(sh[cs])))
    return pd.DataFrame(out, columns=["unit", "answer", "count"])


def main():
    eth, regions = read_cube("ethnic_table.json", "Ethnicity", "Total")
    lit, _ = read_cube("ghlang_table.json", "Ghanaian_language_of_literacy", "Literate")
    nat = eth[eth["level"] == "country"].set_index("cat")["count"]
    say(nat["Total"] == GHANAIANS, f"Ghanaians {nat['Total']:,} = Vol 3C Table 5.5")
    say(nat[GROUPS].sum() == nat["Total"], "the nine groups sum to the total")

    # the drawn units: districts, and sub-metros in place of their metro
    drawn = eth[eth["level"].isin(["district", "submetro"])]
    par = {}
    for g in drawn.loc[drawn["level"] == "submetro", "geo"].unique():
        m = re.match(r"^([A-Za-z]+)-", g)
        reg = drawn.loc[drawn["geo"] == g, "region"].iloc[0]
        parent = [p for p in eth.loc[(eth["level"] == "metro") & (eth["region"] == reg),
                                     "geo"].unique()
                  if p.rstrip().endswith(f"({m.group(1)})")]
        say(len(parent) == 1, f"{g}: one metro parent {parent}")
        par[g] = parent[0]
    units = drawn[["geo", "region"]].drop_duplicates().reset_index(drop=True)
    units["parent"] = units["geo"].map(lambda g: par.get(g, g))
    units["old"] = units["region"].map(OLD_REGION)
    say(len(units) == 272, f"{len(units)} drawn units (255 districts + 17 sub-metros)")
    piv = drawn.pivot_table(index="geo", columns="cat", values="count", aggfunc="sum")
    say(int(piv["Total"].sum()) == GHANAIANS, "the drawn units sum to all Ghanaians")
    say(bool((piv[GROUPS].sum(axis=1) == piv["Total"]).all()), "each unit's groups sum to it")

    a = survey(units)
    import geopandas as gpd
    geo = gpd.read_file(RD / "data" / "geo" / "gh" / "gh_districts.gpkg")
    pts = geo.to_crs(32630).geometry.representative_point().to_crs(4326)
    cent = {u: (p.x, p.y) for u, p in zip(geo["unit"], pts)}
    say(set(cent) == set(units["geo"]), "religiondots' 272 polygons are the 272 drawn units")
    sh, diag = shares(a, units, piv, cent)
    sh = calibrate(sh, a)

    rows = []
    for u in units["geo"]:
        for g in GROUPS:
            c = piv.at[u, g]
            if c <= 0:
                continue
            for ans, p in sh[(u, g)].items():
                rows.append((u, ans, c * p))
    df = pd.DataFrame(rows, columns=["unit", "answer", "count"])
    df = df.groupby(["unit", "answer"], as_index=False)["count"].sum()

    # Ga/Dangme, and Akan/Nzema, apart by the census's literacy table
    lt = lit[lit["level"].isin(["district", "submetro"])].pivot_table(
        index="geo", columns="cat", values="count", aggfunc="sum")
    lt["region"] = lt.index.map(dict(zip(units["geo"], units["region"])))
    say(lt["region"].notna().all() and len(lt) == 272, "literacy cube on the same 272 units")
    reg_of = dict(zip(units["geo"], units["region"]))
    df = literacy_split(df, lt, reg_of, (["Ga", "Dangme"], ["Ga", "Dangme"]), "Ga/Dangme")
    df.loc[df["answer"] == "Nzema", "answer"] = "Akan"
    df = df.groupby(["unit", "answer"], as_index=False)["count"].sum()
    df = literacy_split(df, lt, reg_of,
                        (["Akan", "Akan", "Akan", "Nzema"],
                         ["Asante_Twi", "Akwapim_Twi", "Fante", "Nzema"]), "Akan")
    df = df.groupby(["unit", "answer"], as_index=False)["count"].sum()
    say(abs(df["count"].sum() - GHANAIANS) < 1, f"drawn {df['count'].sum():,.0f} = Ghanaians")

    tot = df.groupby("answer")["count"].sum().sort_values(ascending=False)
    print((tot / 1e3).round(0).astype(int).to_string())
    diag.to_csv(RAW / "gh_model_diag.csv", index=False)

    df["count"] = df["count"].round(1)
    df = df[df["count"] > 0]
    out = pd.DataFrame({
        "geo_id": df["unit"], "geo_level": "district", "geo_name": df["unit"],
        "source_category": df["answer"], "count": df["count"], "tier": "modelled",
        "source_id": SOURCE_ID, "year": "2021 (census), 2008-2022 (survey)",
        "note": df["unit"].map(lambda u: "submetro of " + par[u] if u in par else ""),
    })
    out.loc[out["geo_id"].isin(par), "geo_level"] = "submetro"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(out):,} rows, {out['geo_id'].nunique()} units)")


if __name__ == "__main__":
    if "--fetch" in sys.argv:
        fetch()
    else:
        main()
