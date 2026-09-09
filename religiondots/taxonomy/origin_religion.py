"""Country of origin -> religious composition. Shared by every country with a foreign half.

A country whose survey covers only citizens, or whose survey cannot reach its immigrants, has
to draw its foreign residents from somewhere else. The pattern, first built for Spain (§9y) and
now also Greece:

    magnitude    the host state's own count of foreign residents, by region x citizenship
    composition  Pew Research Center, "Religious Composition by Country, 2010-2020" (2025),
                 SEVEN families, per country of origin
    the split    inside Christianity and inside Islam, hand-authored here with a reason

spec §14.9 permits it and its preference clause is why it is used: for these people there is no
better source, and the alternative is not a worse map but one that omits them.

KEYED ON ISO 3166-1 ALPHA-2, and deliberately not on either source's country names. Eurostat
keys its census tables that way, INE writes Spanish names, Pew writes English ones, and every
pair of those disagrees about something ("Ivory Coast" / "Côte d'Ivoire", "Czechia" /
"Czech Republic", "Türkiye" / "Turkey"). One stable key with two translation tables beats three
sets of names matched pairwise. **Eurostat's two exceptions are worth knowing: it writes `UK`
for the United Kingdom and `EL` for Greece**, not GB and GR.

WHAT THIS CANNOT DO, and it belongs in every note_public that uses it:

  1. **It cannot see conversion, lapse or the second generation.** A Moroccan in Almería who no
     longer prays and a Colombian in Madrid who became evangelical are both counted as their
     country of origin says. Migrants secularise towards the host country, so these are upper
     bounds on adherence rather than estimates of it.
  2. **Pew is a NATIONAL composition and migrants are not a national sample.** Where the skew is
     documented and large it is corrected below, by name, with the reason. Where it is not,
     Pew's national figure stands — §14.9's standard is *documented rather than fitted*, and an
     undocumented adjustment to taste is exactly what that forbids.
  3. **It stops at the passport.** Anyone who naturalised is in the host survey's universe, not
     this one.

THE CASE THAT PROVES RULE 2 IS RIGHT, and it is Greece's. Albania is 374,917 of Greece's 758,472
foreign residents — 49% of them — so one coefficient decides the country's Muslim total. Pew puts
Albania at 59% Muslim; the literature on Albanian migrants in Greece documents Orthodox baptism
and name-changing as integration strategies and a disproportionately southern, Orthodox origin,
so the true share for that stream is certainly lower. It is left at Pew's figure anyway, because
the check says so: with it, Greece comes out **5.08% Muslim** against Pew's own independent
country estimate of **5.12%**; without Albanians it comes out 2.31%, less than half. The
undocumented adjustment would have been the error.
"""

# --------------------------------------------------------------------------------------
# ISO 3166-1 alpha-2 -> Pew's country name. Only countries that appear as a foreign
# residency in a drawn country need a row; `None` means Pew has no row and REGIONAL
# supplies a stand-in.
# --------------------------------------------------------------------------------------
PEW_BY_ISO = {
    # --- Europe
    "AD": None, "AL": "Albania", "AM": "Armenia", "AT": "Austria", "AZ": "Azerbaijan",
    "BA": "Bosnia-Herzegovina", "BE": "Belgium", "BG": "Bulgaria", "BY": "Belarus",
    "CH": "Switzerland", "CY": "Cyprus", "CZ": "Czech Republic", "DE": "Germany",
    "DK": "Denmark", "EE": "Estonia", "EL": "Greece", "ES": "Spain", "FI": "Finland",
    "FR": "France", "GB": "United Kingdom", "GE": "Georgia", "GR": "Greece",
    "HR": "Croatia", "HU": "Hungary", "IE": "Ireland", "IS": "Iceland", "IT": "Italy",
    "LI": None, "LT": "Lithuania", "LU": "Luxembourg", "LV": "Latvia", "MC": None,
    "MD": "Moldova", "ME": "Montenegro", "MK": "North Macedonia", "MT": "Malta",
    "NL": "Netherlands", "NO": "Norway", "PL": "Poland", "PT": "Portugal",
    "RO": "Romania", "RS": "Serbia", "RU": "Russia", "SE": "Sweden", "SI": "Slovenia",
    "SK": "Slovakia", "SM": None, "TR": "Turkey", "UA": "Ukraine",
    "UK": "United Kingdom", "VA": None, "XK": "Kosovo",
    # --- Africa
    "AO": "Angola", "BF": "Burkina Faso", "BI": "Burundi", "BJ": "Benin",
    "BW": "Botswana", "CD": "Democratic Republic of the Congo",
    "CF": "Central African Republic", "CG": "Republic of the Congo", "CI": "Ivory Coast",
    "CM": "Cameroon", "CV": "Cape Verde", "DJ": "Djibouti", "DZ": "Algeria",
    "EG": "Egypt", "EH": "Western Sahara", "ER": "Eritrea", "ET": "Ethiopia",
    "GA": "Gabon", "GH": "Ghana", "GM": "Gambia", "GN": "Guinea", "GQ": "Equatorial Guinea",
    "GW": "Guinea-Bissau", "KE": "Kenya", "KM": "Comoros", "LR": "Liberia", "LS": "Lesotho",
    "LY": "Libya", "MA": "Morocco", "MG": "Madagascar", "ML": "Mali", "MR": "Mauritania",
    "MU": "Mauritius", "MW": "Malawi", "MZ": "Mozambique", "NA": "Namibia", "NE": "Niger",
    "NG": "Nigeria", "RW": "Rwanda", "SC": "Seychelles", "SD": "Sudan", "SL": "Sierra Leone",
    "SN": "Senegal", "SO": "Somalia", "SS": "South Sudan", "ST": "Sao Tome and Principe",
    "SZ": "Eswatini", "TD": "Chad", "TG": "Togo", "TN": "Tunisia", "TZ": "Tanzania",
    "UG": "Uganda", "ZA": "South Africa", "ZM": "Zambia", "ZW": "Zimbabwe",
    # --- Americas
    "AG": None, "AR": "Argentina", "AW": "Aruba", "BB": "Barbados", "BO": "Bolivia",
    "BR": "Brazil", "BS": "Bahamas", "BZ": "Belize", "CA": "Canada", "CL": "Chile",
    "CO": "Colombia", "CR": "Costa Rica", "CU": "Cuba", "CW": "Curacao", "DM": None,
    "DO": "Dominican Republic", "EC": "Ecuador", "GD": "Grenada", "GT": "Guatemala",
    "GY": "Guyana", "HN": "Honduras", "HT": "Haiti", "JM": "Jamaica", "KN": None,
    "LC": "St. Lucia", "MX": "Mexico", "NI": "Nicaragua", "PA": "Panama", "PE": "Peru",
    "PY": "Paraguay", "SR": "Suriname", "SV": "El Salvador", "SX": None,
    "TT": "Trinidad and Tobago", "US": "United States", "UY": "Uruguay",
    "VC": "St. Vincent and the Grenadines", "VE": "Venezuela",
    # --- Asia
    "AE": "United Arab Emirates", "AF": "Afghanistan", "BD": "Bangladesh", "BH": "Bahrain",
    "BN": "Brunei", "BT": "Bhutan", "CN": "China", "ID": "Indonesia", "IL": "Israel",
    "IN": "India", "IQ": "Iraq", "IR": "Iran", "JO": "Jordan", "JP": "Japan",
    "KG": "Kyrgyzstan", "KH": "Cambodia", "KP": "North Korea", "KR": "South Korea",
    "KW": "Kuwait", "KZ": "Kazakhstan", "LA": "Laos", "LB": "Lebanon", "LK": "Sri Lanka",
    "MM": "Myanmar", "MN": "Mongolia", "MV": "Maldives", "MY": "Malaysia", "NP": "Nepal",
    "OM": "Oman", "PH": "Philippines", "PK": "Pakistan", "PS": "Palestinian territories",
    "QA": "Qatar", "SA": "Saudi Arabia", "SG": "Singapore", "SY": "Syria",
    "TH": "Thailand", "TJ": "Tajikistan", "TL": "Timor-Leste", "TM": "Turkmenistan",
    "TW": "Taiwan", "UZ": "Uzbekistan", "VN": "Vietnam", "YE": "Yemen",
    # --- Oceania
    "AU": "Australia", "FJ": "Fiji", "FM": "Federated States of Micronesia",
    "KI": "Kiribati", "MH": None, "NR": None, "NZ": "New Zealand",
    "PG": "Papua New Guinea", "PW": None, "SB": "Solomon Islands", "TO": "Tonga",
    "TV": None, "VU": "Vanuatu", "WS": "Samoa",
}

# --------------------------------------------------------------------------------------
# Stand-ins for the handful Pew has no row for: microstates below its 100,000 threshold and
# a couple of Caribbean territories. Percentages, same seven families as Pew's file.
# --------------------------------------------------------------------------------------
REGIONAL = {
    "AD": dict(Christians=89, Religiously_unaffiliated=9, Muslims=2),
    "LI": dict(Christians=85, Religiously_unaffiliated=11, Muslims=4),
    "MC": dict(Christians=86, Religiously_unaffiliated=12, Muslims=2),
    "SM": dict(Christians=92, Religiously_unaffiliated=7, Muslims=1),
    "VA": dict(Christians=100),
    "DM": dict(Christians=92, Religiously_unaffiliated=6, Other_religions=2),
    "AG": dict(Christians=93, Religiously_unaffiliated=5, Other_religions=2),
    "GD": dict(Christians=94, Religiously_unaffiliated=4, Other_religions=2),
    "KN": dict(Christians=93, Religiously_unaffiliated=5, Other_religions=2),
    "SX": dict(Christians=88, Religiously_unaffiliated=8, Other_religions=4),
    "NR": dict(Christians=95, Religiously_unaffiliated=3, Other_religions=2),
    # Pacific microstates below Pew's 100,000 threshold. All overwhelmingly Christian and
    # all with a handful of residents anywhere this map draws — Greece's census counts four
    # people from Palau and none at all from the other two — but a citizenship with no
    # composition stops the build, and stopping on four people would be silly.
    "MH": dict(Christians=97, Religiously_unaffiliated=2, Other_religions=1),
    "PW": dict(Christians=93, Religiously_unaffiliated=4, Other_religions=3),
    "TV": dict(Christians=97, Religiously_unaffiliated=2, Other_religions=1),
}

# --------------------------------------------------------------------------------------
# The Christian split, per origin country: shares OF THAT COUNTRY'S CHRISTIANS.
# --------------------------------------------------------------------------------------
CATH = "christianity.catholic.latin"
GRK = "christianity.catholic.eastern"          # Greek Catholic / Eastern rite
ORTH = "christianity.orthodox.canonical"
PROT = "christianity.protestant"
ORIENT = "christianity.oriental"

DEFAULT_CHRISTIAN = {CATH: 0.60, PROT: 0.35, ORTH: 0.05}

CHRISTIAN = {
    # ---- Orthodox Europe
    "RO": {"christianity.orthodox.canonical.romanian": 0.92, GRK: 0.04, CATH: 0.02,
           PROT: 0.02},
    "BG": {"christianity.orthodox.canonical.bulgarian": 0.94, PROT: 0.05, CATH: 0.01},
    # Ukraine's Greek Catholics are 8-10% nationally and heavily Galician; the migrant
    # streams to both Spain and Greece are western, so 0.12. A documented skew, per rule 2.
    "UA": {"christianity.orthodox.canonical.ukrainian": 0.85, GRK: 0.12, PROT: 0.03},
    "RU": {ORTH: 0.96, PROT: 0.03, CATH: 0.01},
    "MD": {"christianity.orthodox.canonical.romanian": 0.60, ORTH: 0.37, PROT: 0.03},
    "BY": {ORTH: 0.82, CATH: 0.16, PROT: 0.02},
    "GE": {"christianity.orthodox.canonical.georgian": 0.93, ORIENT: 0.05, CATH: 0.02},
    # THE GENERAL ARMENIAN NODE AND NOT A CATHOLICOSATE. Until 2026-09-08 this line read
    # `christianity.oriental.armenian-etchmiadzin`, which was ASARB's NORTH AMERICAN diocese
    # -- so an Armenian in Lyon or Thessaloniki was being filed on a US jurisdiction. Nothing
    # about a migrant's origin country says which of the two catholicosates they follow, and
    # the model must not invent one. ask/004-am.
    # The neighbouring ORIENT rows stay on the bare parent on purpose. GE, TR and AZ are
    # mostly Armenian and LY, SD, DJ mostly Coptic, but each is a MIXED residual in a model
    # rather than a category anybody published, and the ruling is about not asserting a
    # division the source makes no claim about.
    "AM": {"christianity.oriental.armenian": 0.94, CATH: 0.04, PROT: 0.02},
    "GR": {"christianity.orthodox.canonical.greek": 0.97, CATH: 0.02, PROT: 0.01},
    "EL": {"christianity.orthodox.canonical.greek": 0.97, CATH: 0.02, PROT: 0.01},
    "CY": {"christianity.orthodox.canonical.greek": 0.95, CATH: 0.04, PROT: 0.01},
    "RS": {"christianity.orthodox.canonical.serbian": 0.92, CATH: 0.06, PROT: 0.02},
    "ME": {"christianity.orthodox.canonical.serbian": 0.88, CATH: 0.10, PROT: 0.02},
    "MK": {"christianity.orthodox.canonical.macedonian": 0.95, CATH: 0.03, PROT: 0.02},
    "BA": {"christianity.orthodox.canonical.serbian": 0.54, CATH: 0.44, PROT: 0.02},
    "AL": {CATH: 0.52, "christianity.orthodox.canonical.albanian": 0.46, PROT: 0.02},
    "XK": {CATH: 0.62, "christianity.orthodox.canonical.serbian": 0.36, PROT: 0.02},
    "TR": {ORTH: 0.45, ORIENT: 0.40, CATH: 0.10, PROT: 0.05},
    "AZ": {ORTH: 0.80, ORIENT: 0.12, CATH: 0.05, PROT: 0.03},
    # ---- Catholic Europe
    "PL": {CATH: 0.97, ORTH: 0.02, PROT: 0.01},
    "LT": {CATH: 0.93, ORTH: 0.05, PROT: 0.02},
    "HR": {CATH: 0.95, ORTH: 0.04, PROT: 0.01},
    "SI": {CATH: 0.96, ORTH: 0.02, PROT: 0.02},
    "IT": {CATH: 0.94, ORTH: 0.04, PROT: 0.02},
    "PT": {CATH: 0.95, PROT: 0.04, ORTH: 0.01},
    "FR": {CATH: 0.86, PROT: 0.09, ORTH: 0.05},
    "IE": {CATH: 0.86, PROT: 0.12, ORTH: 0.02},
    "MT": {CATH: 0.98, PROT: 0.02},
    "LU": {CATH: 0.90, PROT: 0.07, ORTH: 0.03},
    "BE": {CATH: 0.86, PROT: 0.08, ORTH: 0.06},
    "AT": {CATH: 0.83, PROT: 0.10, ORTH: 0.07},
    "CZ": {CATH: 0.85, PROT: 0.13, ORTH: 0.02},
    "SK": {CATH: 0.79, GRK: 0.06, PROT: 0.14, ORTH: 0.01},
    "HU": {CATH: 0.69, GRK: 0.04, PROT: 0.26, ORTH: 0.01},
    "AD": {CATH: 0.97, PROT: 0.02, ORTH: 0.01},
    "MC": {CATH: 0.95, PROT: 0.03, ORTH: 0.02},
    "SM": {CATH: 0.97, PROT: 0.02, ORTH: 0.01},
    "VA": {CATH: 1.0},
    "LI": {CATH: 0.88, PROT: 0.10, ORTH: 0.02},
    # ---- Mixed and Protestant Europe
    "DE": {CATH: 0.47, PROT: 0.48, ORTH: 0.05},
    "NL": {CATH: 0.46, PROT: 0.51, ORTH: 0.03},
    "CH": {CATH: 0.54, PROT: 0.42, ORTH: 0.04},
    "LV": {PROT: 0.39, CATH: 0.31, ORTH: 0.30},
    "EE": {ORTH: 0.55, PROT: 0.41, CATH: 0.04},
    "GB": {PROT: 0.61, CATH: 0.32, ORTH: 0.07},
    "UK": {PROT: 0.61, CATH: 0.32, ORTH: 0.07},
    "SE": {PROT: 0.90, CATH: 0.05, ORTH: 0.05},
    "NO": {PROT: 0.91, CATH: 0.06, ORTH: 0.03},
    "DK": {PROT: 0.92, CATH: 0.04, ORTH: 0.04},
    "FI": {PROT: 0.94, ORTH: 0.05, CATH: 0.01},
    "IS": {PROT: 0.92, CATH: 0.07, ORTH: 0.01},
    # ---- Latin America and the Caribbean
    "CO": {CATH: 0.80, PROT: 0.20},
    "VE": {CATH: 0.85, PROT: 0.15},
    "HN": {CATH: 0.50, PROT: 0.50},
    "PE": {CATH: 0.82, PROT: 0.18},
    "EC": {CATH: 0.85, PROT: 0.15},
    "AR": {CATH: 0.85, PROT: 0.14, ORTH: 0.01},
    "BR": {CATH: 0.58, PROT: 0.42},
    "PY": {CATH: 0.90, PROT: 0.10},
    "BO": {CATH: 0.75, PROT: 0.25},
    "DO": {CATH: 0.70, PROT: 0.30},
    "CU": {CATH: 0.76, PROT: 0.23, ORTH: 0.01},
    "NI": {CATH: 0.62, PROT: 0.38},
    "MX": {CATH: 0.88, PROT: 0.12},
    "CL": {CATH: 0.70, PROT: 0.30},
    "UY": {CATH: 0.76, PROT: 0.23, ORTH: 0.01},
    "SV": {CATH: 0.55, PROT: 0.45},
    "GT": {CATH: 0.50, PROT: 0.50},
    "CR": {CATH: 0.70, PROT: 0.30},
    "PA": {CATH: 0.75, PROT: 0.25},
    "BZ": {CATH: 0.55, PROT: 0.45},
    "HT": {CATH: 0.60, PROT: 0.40},
    "JM": {PROT: 0.94, CATH: 0.06},
    "TT": {CATH: 0.45, PROT: 0.55},
    "GY": {PROT: 0.72, CATH: 0.28},
    "SR": {PROT: 0.55, CATH: 0.45},
    "BB": {PROT: 0.92, CATH: 0.08},
    "BS": {PROT: 0.92, CATH: 0.08},
    "DM": {CATH: 0.63, PROT: 0.37},
    "AG": {PROT: 0.90, CATH: 0.10},
    "GD": {CATH: 0.53, PROT: 0.47},
    "KN": {PROT: 0.90, CATH: 0.10},
    "LC": {CATH: 0.65, PROT: 0.35},
    "VC": {PROT: 0.88, CATH: 0.12},
    "AW": {CATH: 0.78, PROT: 0.22},
    "CW": {CATH: 0.78, PROT: 0.22},
    "SX": {CATH: 0.40, PROT: 0.60},
    # ---- North America
    "US": {PROT: 0.62, CATH: 0.35, ORTH: 0.03},
    "CA": {CATH: 0.55, PROT: 0.42, ORTH: 0.03},
    # ---- Africa
    "MA": {CATH: 0.55, PROT: 0.40, ORTH: 0.05},
    "DZ": {PROT: 0.70, CATH: 0.28, ORTH: 0.02},
    "TN": {CATH: 0.70, PROT: 0.25, ORTH: 0.05},
    "LY": {ORIENT: 0.55, CATH: 0.30, ORTH: 0.15},
    "EG": {"christianity.oriental.coptic": 0.90, ORTH: 0.04, CATH: 0.04, PROT: 0.02},
    "EH": {CATH: 0.60, PROT: 0.40},
    "SD": {CATH: 0.55, ORIENT: 0.25, PROT: 0.20},
    "SS": {CATH: 0.55, PROT: 0.45},
    "ER": {"christianity.oriental.eritrean": 0.85, CATH: 0.10, PROT: 0.05},
    "ET": {"christianity.oriental.ethiopian": 0.66, PROT: 0.32, CATH: 0.02},
    "SO": {CATH: 0.50, PROT: 0.40, ORIENT: 0.10},
    "DJ": {ORIENT: 0.50, CATH: 0.35, ORTH: 0.15},
    "SN": {CATH: 0.90, PROT: 0.10},
    "GM": {CATH: 0.45, PROT: 0.55},
    "ML": {CATH: 0.55, PROT: 0.45},
    "MR": {CATH: 0.60, PROT: 0.40},
    "GN": {CATH: 0.60, PROT: 0.40},
    "GW": {CATH: 0.65, PROT: 0.35},
    "SL": {PROT: 0.60, CATH: 0.40},
    "LR": {PROT: 0.85, CATH: 0.15},
    "CI": {CATH: 0.45, PROT: 0.55},
    "GH": {PROT: 0.76, CATH: 0.20, ORTH: 0.04},
    "NG": {PROT: 0.74, CATH: 0.26},
    "TG": {CATH: 0.58, PROT: 0.42},
    "BJ": {CATH: 0.62, PROT: 0.38},
    "BF": {CATH: 0.72, PROT: 0.28},
    "NE": {CATH: 0.45, PROT: 0.55},
    "TD": {CATH: 0.45, PROT: 0.55},
    "CF": {PROT: 0.55, CATH: 0.45},
    "CM": {CATH: 0.54, PROT: 0.46},
    "GQ": {CATH: 0.90, PROT: 0.10},
    "GA": {CATH: 0.72, PROT: 0.28},
    "CV": {CATH: 0.86, PROT: 0.14},
    "AO": {CATH: 0.56, PROT: 0.44},
    "CG": {CATH: 0.50, PROT: 0.50},
    "CD": {CATH: 0.50, PROT: 0.50},
    "KE": {PROT: 0.71, CATH: 0.28, ORTH: 0.01},
    "UG": {PROT: 0.60, CATH: 0.39, ORTH: 0.01},
    "TZ": {CATH: 0.50, PROT: 0.50},
    "RW": {CATH: 0.55, PROT: 0.45},
    "BI": {CATH: 0.72, PROT: 0.28},
    "ZA": {PROT: 0.87, CATH: 0.10, ORTH: 0.03},
    "ZM": {PROT: 0.80, CATH: 0.20},
    "ZW": {PROT: 0.82, CATH: 0.18},
    "MW": {PROT: 0.75, CATH: 0.25},
    "MZ": {PROT: 0.62, CATH: 0.38},
    "MG": {PROT: 0.55, CATH: 0.45},
    "NA": {PROT: 0.90, CATH: 0.10},
    "BW": {PROT: 0.90, CATH: 0.10},
    "LS": {CATH: 0.55, PROT: 0.45},
    "SZ": {PROT: 0.88, CATH: 0.12},
    "MU": {CATH: 0.85, PROT: 0.15},
    "SC": {CATH: 0.85, PROT: 0.15},
    "KM": {CATH: 0.70, PROT: 0.30},
    "ST": {CATH: 0.85, PROT: 0.15},
    # ---- Asia
    "PH": {CATH: 0.83, PROT: 0.13, "christianity.filipinoindependent": 0.04},
    "IN": {PROT: 0.58, CATH: 0.37, ORTH: 0.05},
    "PK": {PROT: 0.52, CATH: 0.48},
    "BD": {CATH: 0.55, PROT: 0.45},
    "LK": {CATH: 0.88, PROT: 0.12},
    "NP": {PROT: 0.90, CATH: 0.10},
    "CN": {PROT: 0.76, CATH: 0.24},
    "TW": {PROT: 0.65, CATH: 0.35},
    "KR": {PROT: 0.61, CATH: 0.39},
    "KP": {PROT: 0.70, CATH: 0.30},
    "JP": {PROT: 0.55, CATH: 0.40, ORTH: 0.05},
    "VN": {CATH: 0.80, PROT: 0.20},
    "ID": {PROT: 0.70, CATH: 0.30},
    "MY": {PROT: 0.55, CATH: 0.45},
    "SG": {PROT: 0.60, CATH: 0.40},
    "TH": {CATH: 0.45, PROT: 0.55},
    "MM": {PROT: 0.80, CATH: 0.20},
    "KH": {CATH: 0.45, PROT: 0.55},
    "LA": {PROT: 0.70, CATH: 0.30},
    "MN": {PROT: 0.80, CATH: 0.15, ORTH: 0.05},
    "KZ": {ORTH: 0.90, CATH: 0.07, PROT: 0.03},
    "KG": {ORTH: 0.85, PROT: 0.10, CATH: 0.05},
    "UZ": {ORTH: 0.85, PROT: 0.10, CATH: 0.05},
    "TM": {ORTH: 0.88, CATH: 0.07, PROT: 0.05},
    "TJ": {ORTH: 0.88, CATH: 0.07, PROT: 0.05},
    "IR": {ORIENT: 0.55, ORTH: 0.15, PROT: 0.20, CATH: 0.10},
    "IQ": {GRK: 0.65, ORIENT: 0.20, ORTH: 0.15},
    "SY": {"christianity.orthodox.canonical.antiochian": 0.55, ORIENT: 0.20, GRK: 0.20,
           PROT: 0.05},
    "LB": {GRK: 0.58, "christianity.orthodox.canonical.antiochian": 0.28, ORIENT: 0.09,
           PROT: 0.05},
    "JO": {"christianity.orthodox.canonical.greek": 0.60, GRK: 0.28, PROT: 0.12},
    "PS": {"christianity.orthodox.canonical.greek": 0.55, GRK: 0.30, PROT: 0.15},
    "IL": {GRK: 0.35, "christianity.orthodox.canonical.greek": 0.35, CATH: 0.20,
           PROT: 0.10},
    "SA": {CATH: 0.75, PROT: 0.20, ORIENT: 0.05},
    "AE": {CATH: 0.75, PROT: 0.20, ORIENT: 0.05},
    "KW": {CATH: 0.72, PROT: 0.20, ORIENT: 0.08},
    "QA": {CATH: 0.72, PROT: 0.20, ORIENT: 0.08},
    "BH": {CATH: 0.70, PROT: 0.22, ORIENT: 0.08},
    "OM": {CATH: 0.70, PROT: 0.22, ORIENT: 0.08},
    "YE": {CATH: 0.45, ORIENT: 0.35, PROT: 0.20},
    "AF": {PROT: 0.60, CATH: 0.40},
    "BT": {PROT: 0.85, CATH: 0.15},
    "MV": {CATH: 0.60, PROT: 0.40},
    "BN": {CATH: 0.55, PROT: 0.45},
    "TL": {CATH: 0.96, PROT: 0.04},
    # ---- Oceania
    "AU": {CATH: 0.44, PROT: 0.50, ORTH: 0.06},
    "NZ": {CATH: 0.35, PROT: 0.62, ORTH: 0.03},
    "FJ": {PROT: 0.70, CATH: 0.30},
    "PG": {PROT: 0.72, CATH: 0.28},
    "SB": {PROT: 0.82, CATH: 0.18},
    "VU": {PROT: 0.85, CATH: 0.15},
    "WS": {PROT: 0.80, CATH: 0.20},
    "TO": {PROT: 0.85, CATH: 0.15},
    "TV": {PROT: 0.95, CATH: 0.05},
    "KI": {CATH: 0.57, PROT: 0.43},
    "MH": {PROT: 0.92, CATH: 0.08},
    "FM": {CATH: 0.55, PROT: 0.45},
    "PW": {CATH: 0.60, PROT: 0.40},
    "NR": {PROT: 0.75, CATH: 0.25},
}

# --------------------------------------------------------------------------------------
# The Islamic split. Almost everything is Sunni, so only the exceptions are listed.
# --------------------------------------------------------------------------------------
DEFAULT_MUSLIM = {"islam.sunni": 1.0}

MUSLIM = {
    "IR": {"islam.shia": 0.90, "islam.sunni": 0.10},
    "IQ": {"islam.shia": 0.62, "islam.sunni": 0.38},
    "LB": {"islam.shia": 0.50, "islam.sunni": 0.50},
    # Syria's Alawites are not Twelver Shia and the tree has no node for them; `islam` the
    # parent is the honest place, per §6.6 and ru2012.py's "neither Sunni nor Shia".
    "SY": {"islam.sunni": 0.85, "islam": 0.15},
    "AZ": {"islam.shia": 0.75, "islam.sunni": 0.25},
    "BH": {"islam.shia": 0.60, "islam.sunni": 0.40},
    "YE": {"islam.sunni": 0.65, "islam.shia": 0.35},
    # Pakistan's Ahmadis are ~0.2% of the country and legally not Muslim there; §9t put the
    # node under Islam anyway, and this keeps that decision.
    "PK": {"islam.sunni": 0.85, "islam.shia": 0.145, "islam.ahmadiyya": 0.005},
    "IN": {"islam.sunni": 0.87, "islam.shia": 0.13},
    "AF": {"islam.sunni": 0.85, "islam.shia": 0.15},
    "TR": {"islam.sunni": 0.80, "alevism": 0.20},
    # Albania's Bektashi are a Sufi order with their own world headquarters in Tirana and
    # about a fifth of the country's Muslims; the tree has no node, so they sit on `islam`.
    "AL": {"islam.sunni": 0.80, "islam": 0.20},
}

# --------------------------------------------------------------------------------------
# Pew's `Other_religions` column, a different thing in every country. The default is the
# host country's own residual node, supplied by the caller.
# --------------------------------------------------------------------------------------
OTHER = {
    "CN": {"chinesefolk": 0.92, "daoism": 0.05, None: 0.03},
    "TW": {"chinesefolk": 0.80, "daoism": 0.15, None: 0.05},
    # Pew files Sikhs and Jains under Other for India. Both Spain's and Greece's Indians are
    # heavily Punjabi, so the Sikh share of this cell is above India's own — rule 2's
    # documented skew.
    "IN": {"sikhism": 0.85, "jainism": 0.07, None: 0.08},
    "PK": {"sikhism": 0.30, None: 0.70},
    "JP": {"shinto": 0.80, None: 0.20},
    "KR": {"eastasiannew": 0.60, "confucianism": 0.20, None: 0.20},
    "VN": {"caodaism": 0.55, None: 0.45},
    "NG": {"indigenous.african": 0.85, None: 0.15},
    "GH": {"indigenous.african": 0.85, None: 0.15},
    "SN": {"indigenous.african": 0.80, None: 0.20},
    "CI": {"indigenous.african": 0.80, None: 0.20},
    "CM": {"indigenous.african": 0.80, None: 0.20},
    "BJ": {"afrodiasporic": 0.70, "indigenous.african": 0.25, None: 0.05},
    "TG": {"afrodiasporic": 0.60, "indigenous.african": 0.35, None: 0.05},
    "CU": {"afrodiasporic": 0.85, None: 0.15},
    "HT": {"afrodiasporic": 0.90, None: 0.10},
    "BR": {"afrodiasporic": 0.55, "spiritualism": 0.35, None: 0.10},
    "JM": {"rastafari": 0.70, None: 0.30},
    "NP": {"indigenous": 0.60, None: 0.40},
    "ZA": {"indigenous.african": 0.80, None: 0.20},
    "ET": {"indigenous.african": 0.80, None: 0.20},
    "ER": {"indigenous.african": 0.80, None: 0.20},
    "IQ": {"yazidism": 0.70, "mandaeism": 0.20, None: 0.10},
    "SY": {"yazidism": 0.40, "druze": 0.50, None: 0.10},
    "LB": {"druze": 0.85, None: 0.15},
    "IL": {"druze": 0.70, None: 0.30},
    "IR": {"zoroastrianism": 0.40, "bahai": 0.45, None: 0.15},
    "LK": {"indigenous": 0.40, None: 0.60},
}

FAMILIES = ["Christians", "Muslims", "Religiously_unaffiliated", "Buddhists", "Hindus",
            "Jews", "Other_religions"]

SIMPLE = {
    "Religiously_unaffiliated": "unaffiliated",
    "Buddhists": "buddhism",
    "Hindus": "hinduism",
    "Jews": "judaism",
}


def composition(iso, pew_row, other_node):
    """{node: share of this citizenship}, summing to 1.

    `pew_row` is a dict of the seven Pew percentage columns (or a REGIONAL stand-in), and
    `other_node` is the host country's own residual node — `other.es`, `other.gr` — which is
    where an unresolvable slice of Pew's `Other religions` cell goes, per spec §3.11.
    """
    out = {}
    total = sum(v for v in pew_row.values() if v)
    if total <= 0:
        raise ValueError(f"empty Pew row for {iso}")
    for fam, pct in pew_row.items():
        if not pct:
            continue
        frac = pct / total
        if fam == "Christians":
            split = CHRISTIAN.get(iso, DEFAULT_CHRISTIAN)
        elif fam == "Muslims":
            split = MUSLIM.get(iso, DEFAULT_MUSLIM)
        elif fam == "Other_religions":
            split = OTHER.get(iso, {None: 1.0})
        else:
            split = {SIMPLE[fam]: 1.0}
        s = sum(split.values())
        for node, share in split.items():
            node = other_node if node is None else node
            out[node] = out.get(node, 0.0) + frac * share / s
    return out


def nodes(other_node):
    """Every node this model can emit — coverage.py's question, not the data's answer.

    A country with a foreign half is asked TWO questions: its own survey's religion item, and
    the census's citizenship item crossed with this file. So its coverage (spec §6.12) is the
    union of both, and it has to be computed rather than read off the counts: Greece has no
    Sikh residents from a country this table calls Sikh in any measurable number, and that is
    "asked, and essentially nobody" rather than "never asked".
    """
    out = set()
    for table, default in ((CHRISTIAN, DEFAULT_CHRISTIAN), (MUSLIM, DEFAULT_MUSLIM),
                           (OTHER, {None: 1.0})):
        out |= set(default)
        for split in table.values():
            out |= set(split)
    out |= set(SIMPLE.values())
    return {other_node if n is None else n for n in out}
