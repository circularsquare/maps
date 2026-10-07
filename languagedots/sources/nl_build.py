"""Netherlands: build data/normalized/nl.csv and data/geo/nl/nl_weights.csv.

Run through `python sources/nl_cbs.py`. Every row is `derived`. sources/nl.md is the record.

Per gemeente (342, the 2026 classification), on CBS's 1 January 2026 population:
  1. Immigrant languages. People by country of origin (85458NED), split born abroad (first
     generation) and born in the Netherlands (second generation). Remainders ("Afrika
     (exclusief Marokko)" less its eight named countries, ...) are split into countries by the
     national mix of the unnamed countries (85384NED). Each country on its main language
     (fr_build.COUNTRY_LANG with NL_OVERRIDES), a few split by NIDI's figures (SPLITS).
     Share drawn on the language: first generation FIRST_GEN[group] (SCP surveys: 1 - share
     speaking Dutch often or always with their children), every origin without a figure at a
     common rate set so the first generation as a whole matches CBS's 44% "mostly another
     language at home" (SSW 2019); second generation k x the first-generation rate, k set so
     the second generation as a whole matches CBS's 16%.
  2. Regional languages. CBS SSW 2019 (Schmeets & Cornips, Statistische Trends 2021, table
     2.1), language most spoken at home, 15+, by province, applied to each gemeente's people
     15+; under-15s at Driessen's 2011 child-to-mother ratio. Labels by Glottolog: Frisian,
     Gronings, Westphalian (Glottolog's Westphalic holds Drents, Twents, Sallands, Achterhoeks,
     Stellingwerfs and Veluws), Limburgish, Zeeuws; Brabants, Hollandic, Town Frisian, Bildts
     and the other "dialect" answers outside Zeeland and Gelderland are Dutch in Glottolog.
  3. Dutch: everyone else.
"""
import os
import sys
import unicodedata

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "nl")
OUT = os.path.join(ROOT, "data", "normalized", "nl.csv")
WEIGHTS = os.path.join(ROOT, "data", "geo", "nl", "nl_weights.csv")
sys.path.insert(0, HERE)
from fr_build import COUNTRY_LANG  # noqa: E402
import nl_cbs  # noqa: E402

SOURCE_ID = "nl_cbs2026_x_surveys"
YEAR = 2026

# ---------------------------------------------------------------------------------------------
# country -> language, the Netherlands' own calls (sources/nl.md §2)
# ---------------------------------------------------------------------------------------------
NL_OVERRIDES = {
    "BE": "Dutch",            # Belgian-born in the Netherlands: overwhelmingly Flemish
    "ZA": "Afrikaans",        # emigration from South Africa to NL is largely Afrikaans-speaking
    "SX": "English", "BQ": "Papiamento", "AN": "Papiamento",
    "ID": "Dutch",            # Indonesia-born: mostly repatriated Indo-Dutch, see FIRST_GEN
    "YU": "Serbian", "SU": "Russian", "CS": "Czech",
    "HK": "Cantonese", "MO": "Cantonese",
}
# NIDI, Hogendoorn, "Taal en taligheid van mensen met een migratieachtergrond", Demos 39(4),
# 2023, p. 6 (SIM 2006/2011/2015, SING 2009 pooled): among people who understand an origin
# language other than Dutch, the one they understand best, first generation. Normalised over
# the languages named. Moroccan Berber is put wholly on Tarifit: the Moroccan Dutch came
# overwhelmingly from the Rif (sources/nl.md §2).
SPLITS = {
    "SR": {"Sranan Tongo": 64, "Sarnami": 29, "Javanese": 2},
    "MA": {"Moroccan Arabic": 51, "Tarifit": 43},
    "IQ": {"Arabic": 55, "Kurdish": 35},
    "AF": {"Dari": 79, "Pashto": 16},
    "TR": {"Turkish": 94, "Kurdish": 4},
}
SPLITS = {c: {k: v / sum(d.values()) for k, v in d.items()} for c, d in SPLITS.items()}

# First generation: share NOT speaking Dutch often or always with their children (SCP).
#   TR MA SR NC: SIM 2015, Bijlagen bij Integratie in zicht? (2016) table B3.2, 1st gen.
#   SO: SIM 2015, Gevlucht met weinig bagage (2017) fig. 3.9.
#   PL: SIM 2015, Bouwend aan een toekomst in Nederland (2018) fig. 4.3.
#   AF IQ IR: SING 2009, Vluchtelingengroepen in Nederland (2011) table 3.4.
#   SY: NSN 2019, Syrische statushouders op weg in Nederland (2020) table 1.
#   BG: Langer in Nederland (2015) table 3.2, with partner (no children column for Bulgarians).
FIRST_GEN = {"TR": 1 - .37, "MA": 1 - .54, "SR": 1 - .95, "NC": 1 - .80, "SO": 1 - .36,
             "PL": 1 - .14, "AF": 1 - .24, "IQ": 1 - .29, "IR": 1 - .35, "SY": 1 - .10,
             "BG": 1 - .10, "ID": 0.0}
# Borrowed: Eritrea and Ethiopia take Somalia's figure (refugees from the Horn, no SCP
# figure of their own); Ukraine, Russia, Romania, Hungary and the rest of central and eastern
# Europe take Poland's (recent labour and refugee migration). Everything else: the common rate.
BORROW = {"ER": "SO", "ET": "SO", "UA": "PL", "RU": "PL", "RO": "PL", "HU": "PL",
          "MOE_REST": "PL"}
CBS_FIRST, CBS_SECOND = 0.44, 0.16    # CBS SSW 2019, Statistische Trends 2021, table 2.2

# CBS 85458NED keys
NAMED = {"H008533": "AF", "H008544": "AW", "H008545": "AU", "H008552": "BE", "H008559": "BA",
         "H008562": "BR", "H008567": "BG", "H008571": "CA", "H008575": "CN", "H008579": "CO",
         "H008586": "CW", "H008592": "DE", "H008594": "EG", "H008597": "ER", "H008599": "ET",
         "H008603": "PH", "H008605": "FR", "H008612": "GH", "H008615": "EL", "H008627": "HU",
         "H008631": "IN", "H008632": "ID", "H008633": "IQ", "H008634": "IR", "H008636": "IT",
         "H008644": "CV", "H008673": "MA", "H008699": "NG", "H008706": "UA", "H008710": "PK",
         "H008718": "PL", "H008719": "PT", "H008723": "RO", "H008724": "RU", "H008736": "RS",
         "H008747": "SO", "H008749": "ES", "H008751": "SR", "H008753": "SY", "H008757": "TH",
         "H008766": "TR", "H008776": "UK", "H008778": "US", "H008780": "VN", "H008787": "ZA"}
EUROPE, EU, GIPS, MOE = "H007933", "H007935", "H007936", "H007937"
NC, AFR, AMO, ASI = "H007119", "H008860", "H008861", "H008862"
EU27 = {"BE", "BG", "CZ", "DK", "DE", "EE", "IE", "EL", "ES", "FR", "HR", "IT", "CY", "LV",
        "LT", "LU", "HU", "MT", "AT", "PL", "PT", "RO", "SI", "SK", "FI", "SE"}
MOE_SET = {"BG", "EE", "HU", "HR", "LV", "LT", "PL", "RO", "SI", "SK", "CZ"}
GIPS_SET = {"EL", "IT", "PT", "ES"}
# remainder -> (gemeente expression, the countries it holds nationally)
NC_SET = {"AW", "CW", "SX", "BQ", "AN"}

# CBS 85384NED Dutch country names -> ISO 3166 alpha-2 as fr_build writes it (UK, EL), with
# pseudo-codes for the dissolved states. Continent tags settle the remainders (checked against
# CBS's own continent totals in remainders()).
NAME_ISO = {
    "Afghanistan": "AF", "Albanië": "AL", "Algerije": "DZ", "Amerikaanse Maagdeneilanden": "VI",
    "Amerikaans-Samoa": "AS", "Andorra": "AD", "Angola": "AO", "Anguilla": "AI",
    "Antarctica": "AQ", "Antigua en Barbuda": "AG", "Argentinië": "AR", "Armenië": "AM",
    "Aruba": "AW", "Australië": "AU", "Azerbeidzjan": "AZ", "Bahama's": "BS", "Bahrein": "BH",
    "Bangladesh": "BD", "Barbados": "BB", "Belarus": "BY", "België": "BE", "Belize": "BZ",
    "Benin": "BJ", "Bermuda": "BM", "Bhutan": "BT", "Bolivia": "BO",
    "Bosnië-Herzegovina": "BA", "Botswana": "BW", "Brazilië": "BR",
    "Brits Territorium in de Indische Oceaan": "IO", "Britse Maagdeneilanden": "VG",
    "Brunei": "BN", "Bulgarije": "BG", "Burkina Faso": "BF", "Burundi": "BI",
    "Cambodja": "KH", "Canada": "CA", "Caribisch Nederland": "BQ", "Caymaneilanden": "KY",
    "Centraal-Afrikaanse Republiek": "CF", "Chili": "CL", "China": "CN", "Colombia": "CO",
    "Comoren": "KM", "Congo": "CG", "Congo (Democratische Republiek)": "CD",
    "Cookeilanden": "CK", "Costa Rica": "CR", "Cuba": "CU", "Curaçao": "CW", "Cyprus": "CY",
    "Denemarken": "DK", "Djibouti": "DJ", "Dominica": "DM", "Dominicaanse Republiek": "DO",
    "Duitsland": "DE", "Ecuador": "EC", "Egypte": "EG", "El Salvador": "SV",
    "Equatoriaal-Guinea": "GQ", "Eritrea": "ER", "Estland": "EE", "Eswatini": "SZ",
    "Ethiopië": "ET", "Faeröer-eilanden": "FO", "Falklandeilanden": "FK",
    "Federale Republiek Joegoslavië": "YU", "Fiji": "FJ", "Filippijnen": "PH",
    "Finland": "FI", "Frankrijk": "FR", "Frans-Guyana": "GF", "Frans-Polynesië": "PF",
    "Gabon": "GA", "Gambia": "GM", "Gazastrook en Westelijke Jordaanoever": "PS",
    "Georgië": "GE", "Ghana": "GH", "Gibraltar": "GI", "Grenada": "GD", "Griekenland": "EL",
    "Groenland": "GL", "Guadeloupe": "GP", "Guam": "GU", "Guatemala": "GT", "Guinee": "GN",
    "Guinee-Bissau": "GW", "Guyana": "GY", "Haïti": "HT", "Honduras": "HN",
    "Hongarije": "HU", "Hongkong": "HK", "Ierland": "IE", "IJsland": "IS", "India": "IN",
    "Indonesië": "ID", "Irak": "IQ", "Iran": "IR", "Israël": "IL", "Italië": "IT",
    "Ivoorkust": "CI", "Jamaica": "JM", "Japan": "JP", "Jemen": "YE",
    "Joegoslavië (oud)": "YU", "Jordanië": "JO", "Kaapverdië": "CV", "Kameroen": "CM",
    "Kanaaleilanden": "JE", "Katar": "QA", "Kazachstan": "KZ", "Kenia": "KE",
    "Kirgizië": "KG", "Kiribati": "KI", "Koeweit": "KW", "Kosovo": "XK", "Kroatië": "HR",
    "Laos": "LA", "Lesotho": "LS", "Letland": "LV", "Libanon": "LB", "Liberia": "LR",
    "Libië": "LY", "Liechtenstein": "LI", "Litouwen": "LT", "Luxemburg": "LU",
    "Macau": "MO", "Macedonië": "MK", "Madagaskar": "MG", "Malawi": "MW", "Maldiven": "MV",
    "Maleisië": "MY", "Mali": "ML", "Malta": "MT", "Man": "IM", "Marokko": "MA",
    "Marshall-eilanden": "MH", "Martinique": "MQ", "Mauritanië": "MR", "Mauritius": "MU",
    "Mayotte": "YT", "Mexico": "MX", "Micronesië": "FM", "Moldavië": "MD", "Monaco": "MC",
    "Mongolië": "MN", "Montenegro": "ME", "Montserrat": "MS", "Mozambique": "MZ",
    "Myanmar": "MM", "Namibië": "NA", "Nauru": "NR", "Nederlandse Antillen (oud)": "AN",
    "Nepal": "NP", "Nicaragua": "NI", "Nieuw-Caledonië": "NC_", "Nieuw-Zeeland": "NZ",
    "Niger": "NE", "Nigeria": "NG", "Niue": "NU", "Noordelijke Marianen": "MP",
    "Noord-Korea": "KP", "Noorwegen": "NO", "Norfolk": "NF", "Oekraïne": "UA",
    "Oezbekistan": "UZ", "Oman": "OM", "Oostenrijk": "AT", "Pakistan": "PK", "Palau": "PW",
    "Panama": "PA", "Papoea-Nieuw-Guinea": "PG", "Paraguay": "PY", "Peru": "PE",
    "Pitcairneilanden": "PN", "Polen": "PL", "Portugal": "PT", "Puerto Rico": "PR",
    "Republiek Noord-Macedonië": "MK", "Réunion": "RE", "Roemenië": "RO", "Rusland": "RU",
    "Rusland (oud)": "SU", "Rwanda": "RW", "Saint Kitts en Nevis": "KN",
    "Saint Pierre en Miquelon": "PM", "Salomonseilanden": "SB", "Samoa": "WS",
    "San Marino": "SM", "Sao Tomé en Principe": "ST", "Saoedi-Arabië": "SA",
    "Senegal": "SN", "Servië": "RS", "Servië en Montenegro": "YU", "Seychellen": "SC",
    "Sierra Leone": "SL", "Singapore": "SG", "Sint Lucia": "LC",
    "Sint Maarten (Nederlands deel)": "SX", "Sint Vincent en de Grenadines": "VC",
    "Sint-Helena": "SH", "Slovenië": "SI", "Slowakije": "SK", "Soedan": "SD",
    "Somalië": "SO", "Sovjet-Unie (oud)": "SU", "Spanje": "ES", "Sri Lanka": "LK",
    "Swaziland": "SZ", "Syrië": "SY", "Tadzjikistan": "TJ", "Taiwan": "TW",
    "Tanzania": "TZ", "Thailand": "TH", "Timor Leste": "TL", "Togo": "TG", "Tokelau": "TK",
    "Tonga": "TO", "Trinidad en Tobago": "TT", "Tsjaad": "TD", "Tsjechië": "CZ",
    "Tsjecho-Slowakije (oud)": "CS", "Tunesië": "TN", "Turkmenistan": "TM",
    "Turks- en Caicoseilanden": "TC", "Tuvalu": "TV", "Uganda": "UG", "Uruguay": "UY",
    "Vanuatu": "VU", "Vaticaanstad": "VA", "Venezuela": "VE", "Verenigd Koninkrijk": "UK",
    "Verenigde Arabische Emiraten": "AE", "Verenigde Staten van Amerika": "US",
    "Verre eilanden van de VS": "UM", "Vietnam": "VN", "Wallis en Futuna": "WF",
    "Zambia": "ZM", "Zimbabwe": "ZW", "Zuid-Afrika": "ZA", "Zuid-Korea": "KR",
    "Zuid-Soedan": "SS", "Zweden": "SE", "Zwitserland": "CH",
}
# small territories and pseudo-codes fr_build lacks
EXTRA_LANG = {
    "FR": "French", "VI": "English", "AS": "Samoan", "AI": "English", "AQ": "English", "BM": "English",
    "IO": "English", "VG": "English", "KY": "English", "CK": "English", "FO": "Danish",
    "FK": "English", "GF": "Guianese Creole", "PF": "French", "GI": "English",
    "GL": "Danish", "GP": "Antillean Creole", "GU": "English", "JE": "English",
    "IM": "English", "MQ": "Antillean Creole", "YT": "Shimaore", "NC_": "French",
    "NU": "English", "MP": "English", "NF": "English", "PN": "English", "PR": "Spanish",
    "RE": "Reunion Creole", "PM": "French", "SH": "English", "TK": "English",
    "TC": "English", "UM": "English", "WF": "French", "MS": "English", "SS": "Dinka",
}
# the continent each unnamed country sits in, for the remainders. Europe per CBS; Turkey,
# Cyprus? and the Caucasus are settled by the reconciliation below.
EUROPE_SET = EU27 | {"AL", "AD", "BA", "BY", "FO", "GI", "IS", "JE", "IM", "XK", "LI",
                     "MK", "MD", "MC", "ME", "NO", "UA", "AT", "RU", "SM", "RS", "UK",
                     "VA", "CH", "YU", "SU", "CS"}
AFRICA_SET = {"DZ", "AO", "BJ", "BW", "BF", "BI", "CF", "KM", "CG", "CD", "DJ", "EG", "GQ",
              "ER", "SZ", "ET", "GA", "GM", "GH", "GN", "GW", "CI", "CV", "CM", "KE", "LS",
              "LR", "LY", "MG", "MW", "ML", "MA", "MR", "MU", "YT", "MZ", "NA", "NE", "NG",
              "UG", "RE", "RW", "ST", "SN", "SC", "SL", "SH", "SD", "SO", "TZ", "TG", "TD",
              "TN", "ZM", "ZW", "ZA", "SS", "IO"}


DUTCH_NODE = "indoeuropean.germanic.continental.dutch"


def lang_of(iso):
    """The shared origin table (sources/origin_mix.py, 2026-10-05): {node: share}, Dutch as the
    label "Dutch". NIDI's SPLITS above are origin_mix overrides for the Netherlands now;
    NL_OVERRIDES and EXTRA_LANG are kept for the record (the uncited calls, Belgium -> Dutch and
    South Africa -> Afrikaans, went back to the home mix; Indonesia's FIRST_GEN of 0 still puts
    the Indonesia-born on Dutch)."""
    from origin_mix import mix
    return {("Dutch" if n == DUTCH_NODE else n): s for n, s in mix(iso, "nl").items()}


def non_dutch(iso):
    """Share of an origin's mix that is not Dutch."""
    return 1 - lang_of(iso).get("Dutch", 0.0)


def norm(s):
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()


# ---------------------------------------------------------------------------------------------
def load():
    g = pd.read_csv(nl_cbs.GEM_CSV).dropna(subset=["Bevolking_1"])
    p = g.pivot_table(index="RegioS", columns=["Geboorteland", "Herkomstland"],
                      values="Bevolking_1", aggfunc="sum").fillna(0)
    if len(p) != 342:
        sys.exit(f"!! {len(p)} gemeenten, expected 342")
    n = pd.read_csv(nl_cbs.NAT_CSV)
    names = pd.read_csv(nl_cbs.DIM_CSV.format(tid="85384NED", dim="Herkomstland"))
    names = names.set_index("Key")["Title"]
    n["name"] = n["Herkomstland"].map(names)
    return p, n


def national_countries(n):
    """born-abroad and born-in-NL people per ISO code, nationally (85384NED)."""
    n = n[n["name"].isin(NAME_ISO)].copy()
    n["iso"] = n["name"].map(NAME_ISO)
    out = n.pivot_table(index="iso", columns="Geboorteland", values="Bevolking_1",
                        aggfunc="sum").fillna(0)
    missing = [c for c in out.index if c not in EUROPE_SET | AFRICA_SET and False]
    for iso in out.index:
        lang_of(iso)    # raises KeyError on a country with no language
    return out


def remainders(p, nat):
    """Each remainder -> (series per gemeente per generation, national country shares)."""
    named = set(NAMED.values())
    gens = ("A051736", "A051735")
    sets = {
        "EU_REST": (EU27 - GIPS_SET - MOE_SET) - named,
        "GIPS_REST": GIPS_SET - named,
        "MOE_REST": MOE_SET - named,
        "NONEU_REST": (EUROPE_SET - EU27) - named,
        "NC_REST": NC_SET - named,
        "AFR_REST": (AFRICA_SET - {"MA"}) - named,
    }
    europe_or_africa = EUROPE_SET | AFRICA_SET | NC_SET | {"SR", "ID", "TR"}
    am_oc = {"AI", "AG", "AR", "BS", "BB", "BZ", "BM", "BO", "BR", "CA", "KY", "CL", "CO",
             "CR", "CU", "DM", "DO", "EC", "SV", "FK", "GF", "GL", "GD", "GP", "GT", "GY",
             "HT", "HN", "JM", "MQ", "MX", "MS", "NI", "PA", "PY", "PE", "PR", "KN", "PM",
             "LC", "VC", "TT", "TC", "UY", "VE", "VI", "VG", "US", "UM", "AS", "AU", "CK",
             "FJ", "PF", "GU", "KI", "MH", "FM", "NR", "NC_", "NZ", "NU", "MP", "NF", "PW",
             "PG", "PN", "WS", "SB", "TK", "TO", "TV", "VU", "WF", "AQ"}
    sets["AMO_REST"] = am_oc - named
    sets["ASI_REST"] = set(nat.index) - europe_or_africa - am_oc - named
    expr = {
        "EU_REST": lambda d: d[EU] - d[GIPS] - d[MOE] - d["H008552"] - d["H008592"] - d["H008605"],
        "GIPS_REST": lambda d: d[GIPS] - d["H008615"] - d["H008636"] - d["H008719"] - d["H008749"],
        "MOE_REST": lambda d: d[MOE] - d["H008567"] - d["H008627"] - d["H008718"] - d["H008723"],
        "NONEU_REST": lambda d: d[EUROPE] - d[EU] - d["H008559"] - d["H008706"] - d["H008724"]
        - d["H008736"] - d["H008776"],
        "NC_REST": lambda d: d[NC] - d["H008544"] - d["H008586"],
        "AFR_REST": lambda d: d[AFR] - sum(d[k] for k, v in NAMED.items() if v in AFRICA_SET
                                           and v != "MA"),
        "AMO_REST": lambda d: d[AMO] - sum(d[k] for k, v in NAMED.items() if v in am_oc),
        "ASI_REST": lambda d: d[ASI] - sum(d[k] for k, v in NAMED.items()
                                           if v in sets["ASI_REST"] | {"AF", "CN", "PH", "IN",
                                                                       "IQ", "IR", "PK", "SY",
                                                                       "TH", "VN"}),
    }
    out = {}
    print("remainders: gemeente table (summed) against the national countries assigned to it")
    for r, members in sets.items():
        shares, ser = {}, {}
        for gb in gens:
            d = p[gb]
            s = expr[r](d)
            if (s < -0.5).any():
                sys.exit(f"!! {r} {gb} negative in {list(s[s < -0.5].index[:5])}")
            ser[gb] = s.clip(lower=0)
            col = nat.reindex(sorted(members)).fillna(0)[gb]
            tot = col.sum()
            print(f"  {r:11s} {gb}: gemeenten {s.sum():>10,.0f}   national members "
                  f"{tot:>10,.0f}   diff {s.sum() - tot:>+8,.0f}")
            shares[gb] = (col / tot) if tot > 0 else col
        out[r] = (ser, shares)
    return out


# ---------------------------------------------------------------------------------------------
# regional languages: CBS SSW 2019 table 2.1, % of 15+ by province, language most spoken at home
# columns: dialect, Low Saxon, Frisian, Limburgish
SSW = {
    "Groningen": (0.0, 25.5, 1.9, 0.0), "Fryslan": (2.8, 2.8, 39.6, 0.0),
    "Drenthe": (0.0, 31.3, 0.9, 0.0), "Overijssel": (0.4, 23.6, 0.4, 0.0),
    "Flevoland": (1.7, 1.1, 0.0, 0.0), "Gelderland": (0.9, 10.2, 0.1, 0.7),
    "Utrecht": (2.3, 0.7, 0.2, 0.2), "Noord-Holland": (1.5, 0.2, 2.2, 0.1),
    "Zuid-Holland": (1.1, 0.1, 0.0, 0.1), "Zeeland": (29.6, 0.0, 0.0, 0.6),
    "Noord-Brabant": (25.0, 0.1, 0.1, 0.5), "Limburg": (0.6, 0.2, 0.2, 47.9),
}
SSW_OTHER = {"Groningen": 4.9, "Fryslan": 4.6, "Drenthe": 2.8, "Overijssel": 7.8,
             "Flevoland": 10.7, "Gelderland": 7.0, "Utrecht": 5.5, "Noord-Holland": 11.1,
             "Zuid-Holland": 12.3, "Zeeland": 8.9, "Noord-Brabant": 5.6, "Limburg": 5.1}
PROV_OF_COROP = {}
for _c, _p in [(range(1, 4), "Groningen"), (range(4, 7), "Fryslan"), (range(7, 10), "Drenthe"),
               (range(10, 13), "Overijssel"), (range(13, 17), "Gelderland"),
               (range(17, 18), "Utrecht"), (range(18, 25), "Noord-Holland"),
               (range(25, 31), "Zuid-Holland"), (range(31, 33), "Zeeland"),
               (range(33, 37), "Noord-Brabant"), (range(37, 40), "Limburg"),
               (range(40, 41), "Flevoland")]:
    for _i in _c:
        PROV_OF_COROP[f"CR{_i:02d}"] = _p
# Under-15s: Driessen 2011 (ITS, parents of pupils in groups 2 and 4, both parents born in NL),
# child speaks the regional language with the mother, against the adult share in the same
# provinces (CBS above): Frisian 37/39.6, Limburgish 39/47.9, Low Saxon 1/26.8 (Groningen,
# Drenthe, Overijssel mean), Zeeuws 10/29.6.
CHILD = {"Frisian": 37 / 39.6, "Limburgish": 39 / 47.9, "Gronings": 1 / 26.8,
         "Westphalian": 1 / 26.8, "Zeeuws": 10 / 29.6}
# Frisian inside Fryslân: De Fryske Taalatlas 2020 (Provinsje Fryslân, survey 2019, 18+), map
# 1.6, Frisian as mother tongue, band midpoints per gemeente. Relative weights only: the
# province total is CBS's. The Wadden islands were not surveyed and are drawn Dutch.
FRISIAN_BAND = {"dantumadiel": 85, "noardeast-fryslan": 75, "achtkarspelen": 75,
                "tytsjerksteradiel": 75, "opsterland": 75, "waadhoeke": 65,
                "smallingerland": 65, "de fryske marren": 65, "sudwest-fryslan": 55,
                "heerenveen": 55, "leeuwarden": 45, "ooststellingwerf": 45, "harlingen": 25,
                "weststellingwerf": 25, "ameland": 0, "schiermonnikoog": 0,
                "terschelling": 0, "vlieland": 0}
# Limburgish inside Limburg: Veldeke / R&M Matrix 2021, speaks Limburgish fluently, by region.
LIMB_EAST_SOUTH = {"heerlen", "kerkrade", "landgraaf", "brunssum", "simpelveld",
                   "voerendaal", "beekdaelen"}
LIMB_W = {"CR37": 60, "CR38": 74, "CR39": 76}
LIMB_E = 54
# Low Saxon inside Gelderland: the Veluwe and Achterhoek COROPs only (the river area and
# Arnhem-Nijmegen speak Low Franconian). Inside Fryslân: the Stellingwerven only.
GELD_LS = {"CR13", "CR14"}


def geography():
    import geopandas as gpd
    gem = gpd.read_file(os.path.join(RAW, "gemeente_2026.geojson"))
    corop = gpd.read_file(nl_cbs.COROP)
    gem = gem.set_crs(28992, allow_override=True)
    corop = corop.set_crs(28992, allow_override=True)
    pts = gpd.GeoDataFrame({"unit": gem["statcode"], "name": gem["statnaam"]},
                           geometry=gem.representative_point(), crs=28992)
    j = gpd.sjoin(pts, corop[["statcode", "geometry"]].rename(columns={"statcode": "corop"}),
                  how="left", predicate="within")
    if j["corop"].isna().any() or len(j) != 342:
        sys.exit("!! gemeente -> COROP join failed")
    j["prov"] = j["corop"].map(PROV_OF_COROP)
    j["key"] = j["name"].map(norm)
    return j.set_index("unit")[["name", "key", "corop", "prov"]]


def regional(geo, pop, u15):
    """{unit: {label: count}} for the regional languages."""
    a15 = pop - u15
    out = {u: {} for u in geo.index}

    def spread(units, total_adult, label, w=None):
        units = list(units)
        if not units or total_adult <= 0:
            return
        ww = pd.Series(1.0, index=units) if w is None else w.reindex(units).fillna(0)
        base = (ww * a15.reindex(units))
        if base.sum() <= 0:
            return
        rate = total_adult / base.sum()      # adult share where w = 1
        for u in units:
            r = rate * ww[u]
            n = r * a15[u] + r * CHILD.get(label, 1.0) * u15[u]
            out[u][label] = out[u].get(label, 0.0) + n

    for prov, (dia, ls, fy, li) in SSW.items():
        units = geo.index[geo["prov"] == prov]
        A = a15.reindex(units).sum()
        # Frisian
        if prov == "Fryslan":
            w = geo.loc[units, "key"].map(FRISIAN_BAND)
            if w.isna().any():
                sys.exit(f"!! Fryslan gemeente without a band: {list(w[w.isna()].index)}")
            spread(units, fy / 100 * A, "Frisian", w)
        elif prov == "Groningen":
            spread(geo.index[geo["key"] == "westerkwartier"], fy / 100 * A, "Frisian")
        elif prov != "Noord-Holland":
            # Noord-Holland's 2.2% "Fries" is read as West-Fries, a Hollandic (Dutch) dialect
            spread(units, fy / 100 * A, "Frisian")
        # Low Saxon
        if prov == "Groningen":
            spread(units, ls / 100 * A, "Gronings")
        elif prov == "Fryslan":
            spread(geo.index[geo["key"].isin({"ooststellingwerf", "weststellingwerf"})],
                   ls / 100 * A, "Westphalian")
        elif prov == "Gelderland":
            spread(units[geo.loc[units, "corop"].isin(GELD_LS)], ls / 100 * A, "Westphalian")
        else:
            spread(units, ls / 100 * A, "Westphalian")
        # Limburgish
        if prov == "Limburg":
            w = pd.Series({u: (LIMB_E if geo.loc[u, "key"] in LIMB_EAST_SOUTH
                               else LIMB_W[geo.loc[u, "corop"]]) for u in units})
            spread(units, li / 100 * A, "Limburgish", w)
        else:
            spread(units, li / 100 * A, "Limburgish")
        # "dialect": Zeeuws in Zeeland, Veluws (Westphalian) on the Veluwe; elsewhere Dutch
        if prov == "Zeeland":
            spread(units, dia / 100 * A, "Zeeuws")
        elif prov == "Gelderland":
            spread(units[geo.loc[units, "corop"] == "CR13"], dia / 100 * A, "Westphalian")
    return out


# ---------------------------------------------------------------------------------------------
def main():
    p, n = load()
    nat = national_countries(n)
    rem = remainders(p, nat)
    geo = geography()
    pop = p[("T001638", "T001040")]
    u15 = pd.read_csv(nl_cbs.U15_CSV).set_index("RegioS")["Bevolking_1"].reindex(pop.index)
    if u15.isna().any():
        sys.exit("!! under-15 table misses gemeenten")

    # people per (unit, origin code, generation): named countries + remainders' countries
    recs = []   # unit, iso, group(placement), gen, n
    group_of_key = {"H008552": "BEL", "H008592": "DEU", "H008718": "POL", "H008632": "IDN",
                    "H008673": "MAR", "H008751": "SUR", "H008766": "TUR"}

    def pgroup(iso):
        if iso == "BE":
            return "BEL"
        if iso == "DE":
            return "DEU"
        if iso == "PL":
            return "POL"
        if iso == "ID":
            return "IDN"
        if iso == "MA":
            return "MAR"
        if iso == "SR":
            return "SUR"
        if iso == "TR":
            return "TUR"
        if iso in NC_SET:
            return "NCAR"
        if iso in EUROPE_SET:
            return "EUO"
        if iso in AFRICA_SET:
            return "AFR"
        if iso in rem["AMO_REST"][1]["A051736"].index or iso in {"BR", "CA", "CO", "US", "AU"}:
            return "AMO"
        return "ASI"

    for gb in ("A051736", "A051735"):
        d = p[gb]
        for key, iso in NAMED.items():
            for u, v in d[key].items():
                if v > 0:
                    recs.append((u, iso, iso, gb, v))
        for r, (ser, shares) in rem.items():
            sh = shares[gb]
            for u, v in ser[gb].items():
                if v <= 0:
                    continue
                for iso, s in sh.items():
                    if s > 0:
                        recs.append((u, iso, r, gb, v * s))
    df = pd.DataFrame(recs, columns=["unit", "iso", "rate_key", "gen", "n"])
    # every foreign-origin person accounted for
    foreign = p["T001638"]["2012605"]
    got = df.groupby("unit")["n"].sum()
    bad = (got.reindex(foreign.index).fillna(0) - foreign).abs()
    print(f"foreign-origin people: table {foreign.sum():,.0f}, built {got.sum():,.0f}, "
          f"worst gemeente off by {bad.max():.2f}")
    if bad.max() > 1:
        sys.exit("!! foreign-origin people do not reconcile")
    df["group"] = df["iso"].map(pgroup)

    # first-generation rate per origin
    def first_rate_key(row_iso, rate_key):
        if row_iso in NC_SET:
            return "NC"
        if row_iso in FIRST_GEN:
            return row_iso
        if row_iso in BORROW:
            return BORROW[row_iso]
        if rate_key in BORROW:
            return BORROW[rate_key]
        return None

    df["fkey"] = [first_rate_key(i, r) for i, r in zip(df["iso"], df["rate_key"])]
    # The Dutch part of an origin's mix (all of Belgium's Flemish share, Suriname's Dutch) never
    # counts as foreign-language speakers: rates apply to the non-Dutch part, `nd`
    nd_of = {i: non_dutch(i) for i in df["iso"].unique()}
    df["nd"] = df["iso"].map(nd_of)
    first = df[(df.gen == "A051736")]
    target = CBS_FIRST * first["n"].sum()
    known = first[first.fkey.notna()]
    known_n = (known["n"] * known["nd"] * known["fkey"].map(lambda k: FIRST_GEN[k])).sum()
    rest = first[first.fkey.isna()]
    common = (target - known_n) / (rest["n"] * rest["nd"]).sum()
    print(f"first generation {first['n'].sum():,.0f}: target foreign {target:,.0f}; "
          f"SCP-rated origins {known['n'].sum():,.0f} give {known_n:,.0f}; "
          f"common rate for the other {(rest['n'] * rest['nd']).sum():,.0f} (non-Dutch part): "
          f"{common:.3f}")
    if not 0 < common < 1:
        sys.exit("!! common rate out of range")
    df["f1"] = df["fkey"].map(lambda k: FIRST_GEN.get(k, np.nan)).fillna(common)
    second = df[df.gen == "A051735"]
    k2 = CBS_SECOND * second["n"].sum() / (second["n"] * second["nd"] * second["f1"]).sum()
    print(f"second generation {second['n'].sum():,.0f}: k = {k2:.3f}")
    df["f"] = np.where(df.gen == "A051736", df["f1"], np.minimum(df["f1"] * k2, df["f1"]))
    df["speakers"] = df["n"] * df["f"] * df["nd"]

    # onto languages: the non-Dutch part of each origin's mix
    rows = []
    for (u, iso, grp), g in df.groupby(["unit", "iso", "group"]):
        s = g["speakers"].sum()
        if s <= 0:
            continue
        lang = {l: w for l, w in lang_of(iso).items() if l != "Dutch"}
        tot = sum(lang.values())
        for l, w in lang.items():
            rows.append((u, l, grp, s * w / tot))
    imm = pd.DataFrame(rows, columns=["unit", "label", "group", "count"])
    imm = imm[imm.label != "Dutch"]

    reg = regional(geo, pop, u15)
    out, wrows = [], []
    for u in pop.index:
        total = pop[u]
        iu = imm[imm.unit == u].groupby("label")["count"].sum()
        ru = pd.Series(reg[u], dtype=float)
        dutch = total - iu.sum() - ru.sum()
        if dutch < 0:
            sys.exit(f"!! {u}: Dutch negative")
        for lab, c in iu.items():
            out.append((u, lab, c, "immigrants by country of origin"))
        for lab, c in ru.items():
            out.append((u, lab, c, "regional language survey"))
        out.append((u, "Dutch", dutch, "remainder"))
    res = pd.DataFrame(out, columns=["geo_id", "source_category", "count", "part"])
    # rows under half a person (the origin mixes' long tails) go onto Dutch, so totals hold
    small = res["count"] < 0.5
    tiny = res[small].groupby("geo_id")["count"].sum()
    res = res[~small].copy()
    is_d = res["source_category"] == "Dutch"
    res.loc[is_d, "count"] += res.loc[is_d, "geo_id"].map(tiny).fillna(0.0)
    res.insert(1, "geo_level", "gemeente")
    res["tier"] = "derived"
    res["year"] = YEAR
    res["source_id"] = SOURCE_ID
    res = res[["geo_id", "geo_level", "source_category", "count", "tier", "part", "year",
               "source_id"]].sort_values(["geo_id", "source_category"])
    diff = (res.groupby("geo_id")["count"].sum() - pop).abs()
    print(f"gemeenten sum to CBS population within {diff.max():.1f} "
          f"({res['count'].sum():,.0f} of {pop.sum():,.0f})")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    res.to_csv(OUT, index=False, encoding="utf-8")
    imm.to_csv(WEIGHTS, index=False, encoding="utf-8")
    print("wrote", OUT, len(res), "and", WEIGHTS)

    # national and provincial summary; the SSW 'other language' check
    nat_lang = res.groupby("source_category")["count"].sum().sort_values(ascending=False)
    print("\nnational, top 30:")
    for lab, c in nat_lang.head(30).items():
        print(f"  {lab:22s} {c:>12,.0f}  {c / pop.sum():6.2%}")
    print("\nprovince: immigrant-language share (all ages) against SSW 2019 'other' (15+)")
    res["prov"] = res["geo_id"].map(geo["prov"])
    pp = pop.groupby(geo["prov"]).sum()
    im = res[res.part.str.startswith("immigrants")].groupby("prov")["count"].sum()
    for pr in SSW:
        print(f"  {pr:14s} built {im.get(pr, 0) / pp[pr]:6.1%}   SSW {SSW_OTHER[pr]:5.1f}%")
    reg_t = res[res.part.str.startswith("regional")].groupby(["prov", "source_category"])[
        "count"].sum()
    print("\nregional languages by province (share of all ages):")
    for (pr, lab), c in reg_t.items():
        print(f"  {pr:14s} {lab:12s} {c:>10,.0f}  {c / pp[pr]:6.1%}")


if __name__ == "__main__":
    main()
