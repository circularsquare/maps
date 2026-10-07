"""Scotland: the 2022 census's "Other language" (272,820 people, every main language but English,
Scots, Scottish Gaelic and sign) split into languages by country of birth, fitted to a national
estimate from the 2011 census's detailed language table. Called by sources/uk_census.py.

    python sources/uk_fetch.py        downloads the inputs below into data/raw/uk/
    python sources/uk_scot_other.py   prints the checks and the national split (uk_census.py
                                      runs the same code and writes uk_units.csv)

Anita, 2026-10-06: split it by country of birth per area, "ideally with some sort of national
level estimate to fit". sources/uk.md §9 has the record.

INPUTS (NRS, Open Government Licence v3.0)
  * UV212 main language, 46,363 Output Areas (uk_census.scotland()): "Other language" per OA.
  * UV204 country of birth, 2022, 79 categories by 355 electoral wards (NRS SuperWEB2 extract
    on the UK Data Service CKAN), and nationally. 26 categories name one country; the rest are
    groups ("Other EU member countries", "Other Middle East", "South America").
  * UV204b country of birth, 14 regions, by Output Area (the OA zip).
  * AT_002_2011 "Language used at home other than English (detailed)", Scotland, 2011: 180
    languages. AT_003_2011 "Country of birth (detailed)", Scotland, 2011: about 230 countries.
    Both from the Wayback Machine (NRS's additional tables, the 2014 site); NRS's own site no
    longer serves them.
  * OA_TO_HIGHER_AREAS.csv in NRS's Census 2022 Index: OA -> ward and council area.

THE METHOD
  1. Every country of birth c -> its languages, origin_mix.mix(c, "uk"), each node put in a
     "bin": the AT_002 label whose node is the node or its nearest ancestor (Arabic varieties go
     to "Arabic"). English, Scots, Scottish Gaelic and sign are dropped (2022 counted them
     outside "Other language"), and so is any node with no AT_002 label (Saraiki, which
     Pakistanis in Scotland evidently reported as Punjabi or Urdu). UK-born and Channel
     Islands-born are left out.
  2. 2011 calibration: expected_2011(b) = sum over c of born_2011(c) x mix(c)(b);
     r(b) = AT_002(b) / expected_2011(b). r folds in everything the mix does not know: the
     children born in Scotland who speak the language (Urdu's r is far above 1), the immigrants
     who speak English at home (Hindi's is below), and how people named the language (most
     China-born wrote "Chinese", not "Mandarin").
     But 2011 asked which language other than English people USE at home, and 2022 asks their
     MAIN language: home use overstates French (r 1.81), German, Italian, Urdu. So:
  2b. The main-language ratio, from England and Wales's 2021 census, which asked Scotland's
     2022 question: r_EW(node) = TS024 main language / sum over 190 countries of birth
     (ONS country_of_birth_190a) of born x mix, for TS024's one-language leaves. A language
     takes min(r_EW, r_2011): a main language is a language used at home, so Scotland's own
     2011 ratio bounds it (this caps Gujarati, whose English ratio is East African Asians').
     A language TS024 does not name takes r_2011 x the median r_EW / r_2011 (0.568).
  3. 2022 national estimate: T(b) = r(b) x expected_2022(b), with 2022's grouped categories
     split into countries in 2011's proportions within the group. A language no country of
     birth explains (Welsh, Irish, Shelta, "Other languages") keeps its 2011 count grown at the
     explained languages' rate. The estimate before scaling is the check: 259,920 against the
     272,820 counted. All T scaled to 2022's "Other language" total.
  4. Ward step: seed(ward, b) = r(b) x expected_2022(ward, b); iterative proportional fitting
     to rows = each ward's "Other language" (summed from its OAs) and columns = T. This step
     sets the counts: it is the proxy that changes counts, and every row is `derived`.
  5. OA step, placement only: each ward's fitted count of b is spread over its OAs by where
     the people born in the regions that feed b live (UV204b: born in EU countries, Africa,
     Middle East and Asia...), fitted to each OA's own "Other language" count. Every OA keeps
     its UV212 total exactly.

WHAT IT ASSUMES: that how a country's emigrants and their children answered in 2011 still holds
in 2022, and that the 2011 home-language question ("Do you use a language other than English at
home?") and 2022's main-language question pick out the same people for immigrant languages. The
second is weakest for the languages of Western Europe, which people born in Scotland use at
home without it being their main language (French, German, Spanish); step 2b replaces it with
England and Wales's 2021 main-language ratios, which assume England's emigrant mix per country
behaves like Scotland's. sources/uk.md §9.
"""
import csv
import io
import os
import sys
import zipfile
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (str(HERE), str(ROOT / "taxonomy"), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)
RAW = ROOT / "data" / "raw" / "uk"
F_AT002 = RAW / "sc2011_AT_002_2011.xls"
F_AT003 = RAW / "sc2011_AT_003_2011.xls"
F_WARD = RAW / "sc2022_UV204_Electoral_Ward_2022.csv"
F_CTRY = RAW / "sc2022_UV204_ctry.csv"
F_INDEX = RAW / "sc_census_2022_index.zip"
F_OA = RAW / "sc_Census-2022-Output-Area-v1.zip"

DROP_NODES = ("indoeuropean.germanic.english", "indoeuropean.germanic.scots",
              "indoeuropean.celtic.scottishgaelic", "signlanguage")
EPS_WARD = 0.02     # a flat share added to every ward seed, so a ward with "Other language" but
EPS_OA = 0.01       # no matching birthplace still takes some; likewise within a ward's OAs
MIN_EXPECTED = 20   # a bin expected for fewer than 20 people in 2011 has no reliable r
MIN_CELL = 0.02     # OA x language cells under this many people are folded back into the OA

# ---------------------------------------------------------------------------------------------
# AT_003_2011's country names -> ISO 3166 (None: UK and Crown dependencies, left out)
# ---------------------------------------------------------------------------------------------
ISO = {
    "Afghanistan": "AF", "Africa (not otherwise specified)": None, "Albania": "AL",
    "Algeria": "DZ", "Angola": "AO", "Antigua and Barbuda": "AG", "Argentina": "AR",
    "Armenia": "AM", "Aruba": "AW", "Asia (Except Middle East)(not otherwise specified)": None,
    "Australia": "AU", "Austria": "AT", "Azerbaijan": "AZ", "Bahamas, The": "BS", "Bahrain": "BH",
    "Bangladesh": "BD", "Barbados": "BB", "Belarus": "BY", "Belgium": "BE", "Belize": "BZ",
    "Benin": "BJ", "Bermuda": "BM", "Bhutan": "BT", "Bolivia": "BO",
    "Bosnia and Herzegovina": "BA", "Botswana": "BW", "Brazil": "BR",
    "British Virgin Islands": "VG", "Brunei": "BN", "Bulgaria": "BG", "Burma": "MM",
    "Burundi": "BI", "Cambodia": "KH", "Cameroon": "CM", "Canada": "CA", "Cape Verde": "CV",
    "Caribbean (not otherwise specified)": None, "Cayman Islands": "KY", "Chad": "TD",
    "Chile": "CL", "China": "CN", "Colombia": "CO", "Congo": "CG",
    "Congo (Democratic Republic)": "CD", "Costa Rica": "CR", "Croatia": "HR", "Cuba": "CU",
    "Cyprus (European Union)": "CY", "Cyprus (Non-European Union)": "CY",
    "Cyprus (not otherwise specified)": "CY", "Czech Republic": "CZ", "Denmark": "DK",
    "Dominica": "DM", "Dominican Republic": "DO", "Ecuador": "EC", "Egypt": "EG",
    "El Salvador": "SV", "Equatorial Guinea": "GQ", "Eritrea": "ER", "Estonia": "EE",
    "Ethiopia": "ET", "Europe (not otherwise specified)": None, "Falkland Islands": "FK",
    "Faroe Islands": "FO", "Fiji": "FJ", "Finland": "FI", "France": "FR", "Gabon": "GA",
    "Gambia, The": "GM", "Georgia": "GE", "Germany": "DE", "Ghana": "GH", "Gibraltar": "GI",
    "Greece": "GR", "Grenada": "GD", "Guatemala": "GT", "Guinea": "GN", "Guinea-Bissau": "GW",
    "Guyana": "GY", "Honduras": "HN",
    "Hong Kong (Special Administrative Region of China)": "HK", "Hungary": "HU",
    "Iceland": "IS", "India": "IN", "Indonesia": "ID", "Iran": "IR", "Iraq": "IQ",
    "Israel": "IL", "Italy": "IT", "Ivory Coast": "CI", "Jamaica": "JM", "Japan": "JP",
    "Jordan": "JO", "Kazakhstan": "KZ", "Kenya": "KE", "Kiribati": "KI", "Korea (North)": "KP",
    "Korea (South)": "KR", "Kosovo": "XK", "Kuwait": "KW", "Kyrgyzstan": "KG", "Laos": "LA",
    "Latvia": "LV", "Lebanon": "LB", "Lesotho": "LS", "Liberia": "LR", "Libya": "LY",
    "Lithuania": "LT", "Luxembourg": "LU",
    "Macao (Special Administrative Region of China)": "MO", "Macedonia": "MK",
    "Madagascar": "MG", "Malawi": "MW", "Malaysia": "MY", "Maldives": "MV", "Malta": "MT",
    "Mauritius": "MU", "Mexico": "MX", "Middle East (not otherwise specified)": None,
    "Moldova": "MD", "Monaco": "MC", "Mongolia": "MN", "Montenegro": "ME", "Montserrat": "MS",
    "Morocco": "MA", "Mozambique": "MZ", "Namibia": "NA", "Nepal": "NP", "Netherlands": "NL",
    "Netherlands Antilles": "CW", "New Zealand": "NZ", "Nicaragua": "NI", "Nigeria": "NG",
    "North America (not otherwise specified)": None, "Norway": "NO",
    "Occupied Palestinian Territories": "PS", "Oman": "OM", "Pakistan": "PK", "Panama": "PA",
    "Papua New Guinea": "PG", "Paraguay": "PY", "Peru": "PE", "Philippines": "PH",
    "Poland": "PL", "Portugal": "PT", "Puerto Rico": "PR", "Qatar": "QA",
    "Republic of Ireland": "IE", "Romania": "RO", "Russia": "RU", "Rwanda": "RW",
    "Sao Tome and Principe": "ST", "Saudi Arabia": "SA", "Senegal": "SN", "Serbia": "RS",
    "Seychelles": "SC", "Sierra Leone": "SL", "Singapore": "SG", "Slovakia": "SK",
    "Slovenia": "SI", "Solomon Islands": "SB", "Somalia": "SO", "South Africa": "ZA",
    "South America (not otherwise specified)": None, "Spain": "ES", "Sri Lanka": "LK",
    "St Helena": "SH", "St Kitts and Nevis": "KN", "St Lucia": "LC",
    "St Vincent and the Grenadines": "VC", "Sudan": "SD", "Surinam": "SR", "Swaziland": "SZ",
    "Sweden": "SE", "Switzerland": "CH", "Syria": "SY", "Taiwan": "TW", "Tajikistan": "TJ",
    "Tanzania": "TZ", "Thailand": "TH", "Togo": "TG", "Tonga": "TO",
    "Trinidad and Tobago": "TT", "Tunisia": "TN", "Turkey": "TR", "Turkmenistan": "TM",
    "Uganda": "UG", "Ukraine": "UA", "Union of Soviet Socialist Republics (not otherwise specified)": "SU",
    "United Arab Emirates": "AE", "United States of America": "US", "Uruguay": "UY",
    "Uzbekistan": "UZ", "Vanuatu": "VU", "Venezuela": "VE", "Vietnam": "VN", "Yemen": "YE",
    "Yugoslavia (not otherwise specified)": "YU", "Zambia": "ZM", "Zimbabwe": "ZW",
    # UK and Crown dependencies: not immigrants' languages
    "England": None, "Scotland": None, "Wales": None, "Northern Ireland": None,
    "Great Britain (not otherwise specified)": None, "United Kingdom (not otherwise specified)": None,
    "UK part not specified": None, "Isle of Man": None, "Jersey": None, "Guernsey": None,
    "Channel Islands (not otherwise specified)": None, "Other countries": None,
}

# ---------------------------------------------------------------------------------------------
# 2022 UV204 categories -> (UV204b OA region, the 2011 countries they hold). A one-country
# category lists that country; a group lists its 2011 members, weighted by their 2011 counts.
# ---------------------------------------------------------------------------------------------
EU_OLD = "Europe: Other Europe: EU Countries: Other member countries in March 2022"
EU_ACC = "Europe: Other Europe: EU Countries: Other EU accession countries March 2022"
NONEU = "Europe: Other Europe: Non-EU countries"
AFR, ASIA, AMER, OCEA = ("Africa", "Middle East and Asia", "The Americas and the Caribbean",
                         "Antartica and Oceania and Other")
IRL = "Europe: Other Europe: EU Countries: Republic of Ireland"
P = "Europe: Other Europe: "
EUM = P + "EU Member countries in March 2022: "
COB22 = {
    EUM + "Republic of Ireland": (IRL, ["Republic of Ireland"]),
    **{EUM + k: (EU_OLD, [v]) for k, v in {
        "France": "France", "Germany": "Germany", "Greece": "Greece", "Italy": "Italy",
        "Netherlands": "Netherlands", "Spain": "Spain", "Czech Republic": "Czech Republic",
        "Hungary": "Hungary", "Latvia": "Latvia", "Lithuania": "Lithuania", "Poland": "Poland",
        "Romania": "Romania", "Slovakia": "Slovakia"}.items()},
    EUM + "Other EU member countries": (EU_OLD, [
        "Portugal", "Sweden", "Denmark", "Belgium", "Finland", "Austria", "Luxembourg",
        "Bulgaria", "Malta", "Estonia", "Slovenia", "Croatia", "Cyprus (European Union)",
        "Cyprus (not otherwise specified)"]),
    # the EU's candidate countries in March 2022 other than Turkey, which has its own row
    P + "Accession countries March 2022": (EU_ACC, ["Albania", "Montenegro", "Macedonia", "Serbia"]),
    P + "Non EU countries: Russia": (NONEU, ["Russia"]),
    P + "Non EU countries: Turkey": (NONEU, ["Turkey"]),
    P + "Non EU countries: Other European countries (Non EU)": (NONEU, [
        "Norway", "Switzerland", "Ukraine", "Bosnia and Herzegovina", "Iceland", "Azerbaijan",
        "Kosovo", "Belarus", "Georgia", "Faroe Islands", "Moldova", "Armenia", "Gibraltar",
        "Cyprus (Non-European Union)", "Union of Soviet Socialist Republics (not otherwise specified)",
        "Yugoslavia (not otherwise specified)"]),
    "Africa: North Africa": (AFR, ["Libya", "Egypt", "Algeria", "Sudan", "Morocco", "Tunisia"]),
    "Africa: Nigeria": (AFR, ["Nigeria"]),
    "Africa: Other Central and Western Africa": (AFR, [
        "Ghana", "Congo", "Cameroon", "Sierra Leone", "Gambia, The", "Angola",
        "Congo (Democratic Republic)", "Ivory Coast", "Guinea", "Senegal", "Liberia", "St Helena",
        "Togo", "Cape Verde", "Chad", "Gabon", "Benin", "Guinea-Bissau",
        "Sao Tome and Principe", "Equatorial Guinea"]),
    "Africa: Kenya": (AFR, ["Kenya"]),
    "Africa: South Africa": (AFR, ["South Africa"]),
    "Africa: Zimbabwe": (AFR, ["Zimbabwe"]),
    "Africa: Other South and Eastern Africa": (AFR, [
        "Zambia", "Somalia", "Uganda", "Malawi", "Tanzania", "Mauritius", "Eritrea", "Botswana",
        "Burundi", "Ethiopia", "Rwanda", "Namibia", "Swaziland", "Mozambique", "Seychelles",
        "Lesotho", "Madagascar"]),
    "Middle East and Asia: Middle East: Iran": (ASIA, ["Iran"]),
    "Middle East and Asia: Middle East: Iraq": (ASIA, ["Iraq"]),
    "Middle East and Asia: Middle East: Other Middle East": (ASIA, [
        "Saudi Arabia", "United Arab Emirates", "Kuwait", "Oman", "Bahrain", "Israel", "Syria",
        "Lebanon", "Yemen", "Jordan", "Occupied Palestinian Territories", "Qatar"]),
    "Middle East and Asia: Eastern Asia: Hong Kong (Special Administrative Region of China)":
        (ASIA, ["Hong Kong (Special Administrative Region of China)"]),
    "Middle East and Asia: Eastern Asia: China": (ASIA, ["China"]),
    "Middle East and Asia: Eastern Asia: Other Eastern Asia": (ASIA, [
        "Japan", "Taiwan", "Korea (South)", "Macao (Special Administrative Region of China)",
        "Mongolia", "Korea (North)"]),
    "Middle East and Asia: Southern Asia: Bangladesh": (ASIA, ["Bangladesh"]),
    "Middle East and Asia: Southern Asia: India": (ASIA, ["India"]),
    "Middle East and Asia: Southern Asia: Pakistan": (ASIA, ["Pakistan"]),
    "Middle East and Asia: Southern Asia: Other Southern Asia": (ASIA, [
        "Sri Lanka", "Nepal", "Afghanistan", "Maldives", "Bhutan"]),
    "Middle East and Asia: South-East Asia: Malaysia": (ASIA, ["Malaysia"]),
    "Middle East and Asia: South-East Asia: Philippines": (ASIA, ["Philippines"]),
    "Middle East and Asia: South-East Asia: Singapore": (ASIA, ["Singapore"]),
    "Middle East and Asia: East Asia: Other South-East Asia": (ASIA, [
        "Thailand", "Indonesia", "Vietnam", "Brunei", "Burma", "Cambodia", "Laos"]),
    "Middle East and Asia: Central Asia": (ASIA, [
        "Kazakhstan", "Uzbekistan", "Turkmenistan", "Kyrgyzstan", "Tajikistan"]),
    "The Americas and the Caribbean: North America: Canada": (AMER, ["Canada"]),
    "The Americas and the Caribbean: North America: United States of America":
        (AMER, ["United States of America"]),
    "The Americas and the Caribbean: North America: Other North America": (AMER, ["Bermuda"]),
    "The Americas and the Caribbean: Central America": (AMER, [
        "Mexico", "Belize", "Guatemala", "Costa Rica", "Honduras", "Panama", "El Salvador",
        "Nicaragua"]),
    "The Americas and the Caribbean: South America": (AMER, [
        "Brazil", "Venezuela", "Argentina", "Colombia", "Chile", "Peru", "Guyana", "Bolivia",
        "Falkland Islands", "Ecuador", "Uruguay", "Paraguay", "Surinam"]),
    "The Americas and the Caribbean: The Caribbean": (AMER, [
        "Trinidad and Tobago", "Jamaica", "Barbados", "Bahamas, The", "Cuba",
        "Dominican Republic", "Cayman Islands", "St Vincent and the Grenadines", "St Lucia",
        "Grenada", "Netherlands Antilles", "Puerto Rico", "Aruba", "Antigua and Barbuda",
        "St Kitts and Nevis", "Montserrat", "Dominica", "British Virgin Islands"]),
    "Antarctica and Oceania: Australia": (OCEA, ["Australia"]),
    "Antarctica and Oceania: New Zealand": (OCEA, ["New Zealand"]),
    "Antarctica and Oceania: Other Antarctia and Oceania": (OCEA, [
        "Fiji", "Papua New Guinea", "Solomon Islands", "Vanuatu", "Kiribati", "Tonga"]),
}
# 2022 categories that are UK-born, totals, or too vague to carry a language
COB22_SKIP = {
    "All people", "Europe: Total", "Europe: United Kingdom: Total",
    "Europe: United Kingdom: England", "Europe: United Kingdom: Northern Ireland",
    "Europe: United Kingdom: Scotland", "Europe: United Kingdom: Wales",
    "Europe: United Kingdom: UK part not specified", "Europe: Channel Islands and Isle of Man",
    EUM.rstrip(": ") + ": Total", P + "Non EU countries: Total", "Africa: Total",
    "Middle East and Asia: Total", "Middle East and Asia: Middle East: Total",
    "Middle East and Asia: Eastern Asia: Total", "Middle East and Asia: Southern Asia: Total",
    "Middle East and Asia: South-East Asia: Total", "The Americas and the Caribbean: Total",
    "The Americas and the Caribbean: North America: Total", "Antarctica and Oceania: Total",
    "Other",
}


# ---------------------------------------------------------------------------------------------
def read_at(path):
    """AT_00x_2011: the alphabetic listing (columns 3-4) -> {label: count}."""
    d = pd.read_excel(path, header=None)
    out = {}
    for a, b in zip(d[3], d[4]):
        if isinstance(b, (int, float)) and b == b and isinstance(a, str):
            lab = a.strip()
            lab = lab[:-2].strip() if lab[-2:] in (" 1", " 2", " 3") else lab   # footnote marks
            out[lab] = int(b)
    return out


def read_superweb(path, geo_col):
    """UKDS's SuperWEB2 csv -> long DataFrame (geo, label, count)."""
    lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith('"Counting"'))
    rows = []
    for rec in csv.reader(lines[start + 1:]):
        if len(rec) < 4 or rec[0] != "Individuals":
            continue
        rows.append((rec[1].strip(), rec[2].strip(), int(rec[3])))
    return pd.DataFrame(rows, columns=[geo_col, "label", "count"])


def ward_codes():
    """UV204's ward names -> 2022 ward codes. The extract prints non-ASCII letters as '?' and
    tells the two pairs of same-named wards apart by council in brackets; asserted 355 <-> 355."""
    import re
    with zipfile.ZipFile(F_INDEX) as z:
        lu = pd.read_csv(io.BytesIO(z.read(
            "Census_2022_Index/Higher_Geographies_LookUps/Electoral Ward 2022 Lookup.csv")),
            encoding="utf-8-sig")
        ca = pd.read_csv(io.BytesIO(z.read(
            "Census_2022_Index/Higher_Geographies_LookUps/Council Area 2019 Lookup.csv")),
            encoding="utf-8-sig")
        oa = pd.read_csv(io.BytesIO(z.read("Census_2022_Index/OA_TO_HIGHER_AREAS.csv")),
                         usecols=["OA2022", "CA2019", "EW2022"])
    lu.columns, ca.columns = ["code", "name"], ["code", "name"]
    lu["name"] = lu["name"].str.strip()
    w_ca = oa.drop_duplicates("EW2022").set_index("EW2022")["CA2019"].map(
        ca.set_index("code")["name"])
    dup = lu["name"].duplicated(keep=False)
    lu.loc[dup, "name"] = lu.loc[dup, "name"] + " (" + lu.loc[dup, "code"].map(w_ca) + ")"

    def key(s):
        return re.sub(r"[^A-Za-z0-9() ]", "?", s).rstrip("?").strip()
    k = {}
    for c, n in zip(lu["code"], lu["name"]):
        k.setdefault(key(n), []).append(c)
    assert all(len(v) == 1 for v in k.values()), "ward names collide after normalising"
    return {kk: v[0] for kk, v in k.items()}, key, oa


def bins():
    """AT_002 label -> node, for the labels 2022 would count as "Other language"."""
    import uk2021
    at2 = read_at(F_AT002)
    at2 = {("Other languages" if k.startswith("Other languages") else k): v for k, v in at2.items()}
    at2.pop("All people aged 3 and over")
    labels = set(at2) - uk2021.SC_OTHER_OUTSIDE
    missing = labels - set(uk2021.SC_OTHER)
    assert not missing, f"AT_002 labels with no node in uk2021.SC_OTHER: {sorted(missing)}"
    return {k: uk2021.SC_OTHER[k] for k in labels}, at2


def bin_of(node, node_bins):
    """The AT_002 label for a mix node: its own or its nearest ancestor's; Arabic varieties
    (siblings of afroasiatic.arabic in the tree) to Arabic. None: dropped."""
    if node.startswith(DROP_NODES):
        return None
    leaf = node.split(".")[-1]
    if node.startswith("afroasiatic.") and (leaf.endswith("_arabic") or leaf == "darija"):
        if leaf == "sudanese_arabic":
            return node_bins.get(node)
        return node_bins["afroasiatic.arabic"]
    parts = node.split(".")
    for k in range(len(parts), 0, -1):
        a = ".".join(parts[:k])
        if a in node_bins:
            return node_bins[a]
    return None


def country_mixes(node_bins, at3):
    """2011 country name -> {AT_002 label: share}, shares of everyone born there (may sum < 1:
    English and unbinned languages dropped)."""
    import origin_mix
    out, unknown = {}, []
    for name, n in at3.items():
        if name not in ISO:
            unknown.append(name)
            continue
        iso = ISO[name]
        if iso is None:
            continue
        try:
            m = origin_mix.mix(iso, "uk")
        except (KeyError, SystemExit) as e:
            print(f"  no language mix for {name} ({iso}): {e}; left out ({n} born there in 2011)")
            continue
        b = {}
        for node, s in m.items():
            lab = bin_of(node, node_bins)
            if lab is not None:
                b[lab] = b.get(lab, 0.0) + s
        out[name] = b
    assert not unknown, f"AT_003 countries with no ISO entry: {unknown}"
    return out


EW_NAME = {   # ONS country_of_birth_190a tails that differ from AT_003's names
    "Portugal (including Madeira and the Azores)": "Portugal",
    "Spain (including Canary Islands)": "Spain", "Czechia": "Czech Republic",
    "North Macedonia": "Macedonia", "The Gambia": "Gambia, The", "Eswatini": "Swaziland",
    "United States": "United States of America", "Myanmar (Burma)": "Burma",
    "The Bahamas": "Bahamas, The", "St Helena, Ascension and Tristan da Cunha": "St Helena",
    "Ireland": "Republic of Ireland", "Cyprus": "Cyprus (not otherwise specified)",
    "Union of Soviet Socialist Republics not otherwise specified":
        "Union of Soviet Socialist Republics (not otherwise specified)",
    "Yugoslavia not otherwise specified": "Yugoslavia (not otherwise specified)",
}
EW_ISO_EXTRA = {"South Sudan": "SS", "Czechoslovakia not otherwise specified": "QT"}


def ew_ratios(country_mix_iso):
    """England and Wales, Census 2021, the same main-language question as Scotland 2022:
    r_EW(node) = TS024 main language (node) / sum over 190 countries of birth of born x mix.
    -> {node: r}, for TS024's one-language leaves with 200+ expected people."""
    import json
    import uk2021
    import uk_census
    d = json.load(open(RAW / "ew_ctry_country_of_birth_190a.json", encoding="utf-8"))
    born = {}
    for o in d["observations"]:
        lab = o["dimensions"][1]["option"].split(": ")[-1]
        born[lab] = born.get(lab, 0) + o["observation"]
    ts, _ = uk_census.read_ts024("ctry")
    # the ctry file also carries England and Wales together (K04000001); England + Wales only
    ts = ts[ts["geo_id"].isin(["E92000001", "W92000004"])]
    obs = ts.groupby("category")["count"].sum()
    assert 55e6 < obs.sum() < 60e6, obs.sum()
    leaf_node = {}
    for leaf, n in obs.items():
        node = uk2021.EW.get(leaf) or uk2021.EW.get(leaf.split(": ")[-1])
        if leaf == "English (English or Welsh in Wales)" or node is None or node == "other" \
                or node.startswith(DROP_NODES):
            continue
        leaf_node[leaf] = node
    node_obs = {}
    for leaf, node in leaf_node.items():
        node_obs[node] = node_obs.get(node, 0) + obs[leaf]
    ew_bins = {n: n for n in node_obs}
    exp = {}
    skipped = 0
    for lab, n in born.items():
        name = EW_NAME.get(lab, lab)
        iso = ISO.get(name, EW_ISO_EXTRA.get(lab))
        if iso is None:
            skipped += n
            continue
        for node, s in country_mix_iso(iso).items():
            b = bin_of(node, ew_bins)
            if b is not None:
                exp[b] = exp.get(b, 0.0) + n * s
    r = {n: node_obs[n] / exp[n] for n in node_obs if exp.get(n, 0) >= 200}
    return r, node_obs, exp


def ipf(seed, rows, cols, iters=500, tol=1e-6):
    """Iterative proportional fitting of a 2-d array to row and column totals."""
    x = seed.copy()
    for _ in range(iters):
        rs = x.sum(axis=1)
        x *= np.divide(rows, rs, out=np.zeros_like(rows), where=rs > 0)[:, None]
        cs = x.sum(axis=0)
        x *= np.divide(cols, cs, out=np.zeros_like(cols), where=cs > 0)[None, :]
        if np.abs(x.sum(axis=1) - rows).max() < tol * max(rows.max(), 1):
            break
    return x


def split(other_oa, verbose=True):
    """other_oa: Series OA code -> UV212 "Other language" count.
    Returns DataFrame (unit, label, count) with every OA's total equal to its input."""
    lab_node, at2_raw = bins()                       # label -> node
    # labels on one node (Mirpuri and Potwari, Edo/Bini and Bini, the two Punjabis, Kurdish and
    # Sorani) are one bin, named by the largest of them
    by_node = {}
    for lab, node in lab_node.items():
        by_node.setdefault(node, []).append(lab)
    node_bins, at2 = {}, {}
    for node, labs in by_node.items():
        rep = max(labs, key=lambda k: (at2_raw[k], k))
        node_bins[node] = rep
        at2[rep] = sum(at2_raw[k] for k in labs)
    node_bins_rev = {v: k for k, v in node_bins.items()}
    at3 = read_at(F_AT003)
    at3 = {("Spain" if k.startswith("Spain") else "Czech Republic" if k.startswith("Czech Republic")
            else k): v for k, v in at3.items()}
    at3.pop("All people")
    cm = country_mixes(node_bins, at3)
    labels = sorted(node_bins_rev)
    L = {lab: i for i, lab in enumerate(labels)}

    def vec(d):
        v = np.zeros(len(labels))
        for k, s in d.items():
            v[L[k]] += s
        return v

    # ---- 2011 calibration ----
    exp11 = sum(at3[c] * vec(cm[c]) for c in cm)
    obs11 = np.array([at2[lab] for lab in labels], float)
    explained = exp11 >= MIN_EXPECTED
    r = np.where(explained, obs11 / np.where(explained, exp11, 1), 0.0)

    # ---- 2022 categories -> mixes ----
    mix22 = {}
    for cat, (_, members) in COB22.items():
        w = np.array([at3.get(m, 0) for m in members], float)
        w = w / w.sum() if w.sum() else np.full(len(members), 1 / len(members))
        mix22[cat] = sum(wi * vec(cm.get(m, {})) for wi, m in zip(w, members))

    nat = read_superweb(F_CTRY, "geo").set_index("label")["count"]
    unknown = set(nat.index) - set(COB22) - COB22_SKIP
    assert not unknown, f"2022 country-of-birth categories not handled: {unknown}"
    exp22 = sum(nat[c] * mix22[c] for c in COB22)

    # ---- the main-language ratio, from England and Wales 2021 ----
    import origin_mix
    cache = {}

    def mix_iso(iso):
        if iso not in cache:
            try:
                cache[iso] = origin_mix.mix(iso, "uk")
            except (KeyError, SystemExit):
                cache[iso] = {}
        return cache[iso]
    r_ew, ew_obs, ew_exp = ew_ratios(mix_iso)
    r11 = r.copy()
    has_ew = np.array([explained[i] and node_bins_rev[lab] in r_ew for i, lab in enumerate(labels)])
    ratio = np.array([r_ew[node_bins_rev[lab]] / r11[i] if has_ew[i] else np.nan
                      for i, lab in enumerate(labels)])
    k = float(np.nanmedian(ratio))
    # Scotland's own 2011 ratio caps it: a main language is a language used at home, so how many
    # of an origin's people and children used L at home in 2011 bounds how many have it as main
    # language (Gujarati: England's ratio is inflated by East African Asians, whom Scotland has
    # few of; 878 used it at home in 2011)
    r = np.where(has_ew, np.minimum([r_ew.get(node_bins_rev[lab], 0.0) for lab in labels], r11),
                 r11 * k)
    r = np.where(explained, r, 0.0)
    capped = has_ew & (r < np.array([r_ew.get(node_bins_rev[lab], 0.0) for lab in labels]) - 1e-12)
    if verbose:
        print(f"  England and Wales 2021 main language / country of birth ratio for "
              f"{has_ew.sum()} of {explained.sum()} explained languages "
              f"({obs11[has_ew].sum() / obs11[explained].sum():.1%} of their 2011 people); "
              f"the rest take Scotland 2011's ratio x {k:.3f} (median of r_EW / r_2011); "
              f"{capped.sum()} capped at Scotland's 2011 ratio: "
              + ", ".join(f"{labels[i]} {r_ew[node_bins_rev[labels[i]]]:.2f}->{r11[i]:.2f}"
                          for i in np.nonzero(capped)[0]))
        order = np.argsort(-obs11 * has_ew)
        print("    language                            r 2011 home   r used")
        for i in order[:25]:
            print(f"    {labels[i]:36} {r11[i]:10.2f} {r[i]:9.2f}")
    T = np.where(explained, r * exp22, 0.0)
    g = T[explained].sum() / obs11[explained].sum()
    T = np.where(explained, T, obs11 * g)
    total = float(other_oa.sum())
    if verbose:
        print(f"  2011: 'Other' languages {obs11.sum():,.0f} (AT_002, the labels 2022 counts as "
              f"Other language); {obs11[explained].sum():,.0f} in {explained.sum()} languages a "
              f"country of birth explains (expected {MIN_EXPECTED}+), the rest kept at 2011 x {g:.3f}")
        print(f"  2022 national estimate before scaling {T.sum():,.0f} against UV212's Other "
              f"language {total:,.0f} (ratio {total / T.sum():.3f})")
    T = T * total / T.sum()

    # ---- wards ----
    wcode, key, oa_lu = ward_codes()
    wd = read_superweb(F_WARD, "ward")
    wd["code"] = wd["ward"].map(lambda s: wcode.get(key(s)))
    assert wd["code"].notna().all(), wd.loc[wd["code"].isna(), "ward"].unique()[:5]
    assert wd["code"].nunique() == len(wcode) == 355
    unknown = set(wd["label"]) - set(COB22) - COB22_SKIP
    assert not unknown, unknown
    oa2w = oa_lu.set_index("OA2022")["EW2022"]
    assert set(other_oa.index) <= set(oa2w.index)
    other_w = other_oa.groupby(other_oa.index.map(oa2w)).sum()
    wards = sorted(wcode.values())
    W = {w: i for i, w in enumerate(wards)}
    cob_w = wd[wd["label"].isin(COB22)].pivot_table(index="code", columns="label",
                                                      values="count", aggfunc="sum").reindex(wards).fillna(0)
    regions = sorted({reg for reg, _ in COB22.values()})
    # per ward x region x label: the region's people's expected speakers (x r)
    E_reg = np.zeros((len(wards), len(regions), len(labels)))
    for cat, (reg, _) in COB22.items():
        E_reg[:, regions.index(reg), :] += np.outer(cob_w[cat].to_numpy(), mix22[cat] * r)
    E_w = E_reg.sum(axis=1)
    rows = other_w.reindex(wards).fillna(0).to_numpy(float)
    flat = np.outer(rows / rows.sum(), T)
    seed = E_w / max(E_w.sum(), 1e-9) * T.sum()
    seed[:, ~explained] = flat[:, ~explained]
    seed = seed + EPS_WARD * flat
    F = ipf(seed, rows, T.copy())
    if verbose:
        print(f"  wards: 355 joined; fitted, worst row gap {np.abs(F.sum(1) - rows).max():.4f}, "
              f"worst column gap {np.abs(F.sum(0) - T).max():.4f}")

    # ---- OAs within each ward ----
    with zipfile.ZipFile(F_OA) as z:
        t = z.read("UV204b - Country of birth (14) by sex by age (6).csv").decode("utf-8-sig").splitlines()
    rr = list(csv.reader(t[4:]))
    h0, h1, h2 = rr[0], rr[1], rr[2]
    cols = {h0[i]: i for i in range(1, len(h0)) if h1[i] == "All people" and h2[i] == "Total"}
    assert set(regions) <= set(cols), set(regions) - set(cols)
    born = {rec[0]: [0 if rec[cols[g]] in ("-", "") else int(rec[cols[g]]) for g in regions]
            for rec in rr[3:] if rec and rec[0].startswith("S00")}
    born = pd.DataFrame.from_dict(born, orient="index", columns=regions)
    assert len(born) == 46_363

    out = []
    before = np.zeros((len(wards), len(labels)))
    oa_w = pd.Series(other_oa.index.map(oa2w), index=other_oa.index)
    for w, oas in other_oa.groupby(oa_w).groups.items():
        oas = list(oas)
        o_tot = other_oa[oas].to_numpy(float)
        if o_tot.sum() == 0:
            continue
        i = W[w]
        b = born.reindex(oas).fillna(0).to_numpy(float)
        bw = b.sum(axis=0)
        share = np.divide(b, bw, out=np.zeros_like(b), where=bw > 0)     # OA's share of ward's born-in-region
        s = share @ E_reg[i]                                             # OA x label
        s = s / max(s.sum(), 1e-12) * F[i].sum()
        flat_o = np.outer(o_tot / o_tot.sum(), F[i])
        s[:, ~explained] = flat_o[:, ~explained]
        s = s + EPS_OA * flat_o
        x = ipf(s, o_tot, F[i].copy())
        before[i] = x.sum(axis=0)
        # tiny fractions: cells under MIN_CELL people are dropped and the rest refitted to the
        # same OA and ward totals, so the trim moves almost nobody between languages (the
        # scatter carries fractions along a Hilbert curve; this only keeps the file small)
        keep = x >= MIN_CELL
        if not keep.any(axis=1).all():                 # an OA left empty keeps its largest cell
            for o in np.nonzero(~keep.any(axis=1))[0]:
                keep[o, x[o].argmax()] = True
        x = ipf(np.where(keep, x, 0.0), o_tot, np.where(keep.any(axis=0), F[i], 0.0)
                * F[i].sum() / max(F[i][keep.any(axis=0)].sum(), 1e-12))
        x *= (o_tot / x.sum(axis=1))[:, None]          # OA totals exact
        oo, ll = np.nonzero(x > 0)
        out.append(pd.DataFrame({"unit": np.array(oas)[oo], "label": np.array(labels)[ll],
                                 "count": x[oo, ll]}))
    res = pd.concat(out, ignore_index=True)
    before = pd.Series(before.sum(axis=0), labels)
    chk = res.groupby("unit")["count"].sum() - other_oa[other_oa > 0]
    assert chk.abs().max() < 1e-6, chk.abs().max()
    assert set(res["unit"]) == set(other_oa[other_oa > 0].index)
    moved = (res.groupby("label")["count"].sum().reindex(before.index).fillna(0) - before).abs()
    if verbose:
        print(f"  trimming cells under {MIN_CELL} moved {moved.sum() / 2:,.0f} people between "
              f"languages within their OAs (largest change {moved.max():,.0f}, {moved.idxmax()})")

    if verbose:
        nat_res = res.groupby("label")["count"].sum().sort_values(ascending=False)
        print(f"  OAs: {res['unit'].nunique():,} with Other language, {len(res):,} rows; every OA "
              f"total kept; {res['label'].nunique()} languages")
        tab = pd.DataFrame({"2011": pd.Series(obs11, labels), "r": pd.Series(r, labels),
                            "2022 drawn": nat_res}).sort_values("2022 drawn", ascending=False)
        print("  language                              2011 at home     r    2022 drawn")
        for lab, row in tab.head(30).iterrows():
            print(f"  {lab:38} {row['2011']:>10,.0f} {row['r']:6.2f} {row['2022 drawn']:>11,.0f}")
    return res


def main():
    import uk_census
    units, _ = uk_census.scotland()
    split(units[units["category"] == "sc:Other language"].set_index("unit")["count"])


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
