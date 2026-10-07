"""Italy: build data/normalized/it.csv and data/geo/it/it_weights.csv.

    python sources/it_istat.py --fetch     # ISTAT 2024 language tables, ASTAT 2024, ISPAT 2021
    python sources/it_istat.py             # the build

Italy's census asks no language. Anita's 2026-10-05 ruling for rich countries with no language
question (AGENT_BRIEF §2): the national language, plus regional languages from surveys, plus
immigrant languages proxied by citizenship. Per NUTS 3 province (107):

  1. population and citizenship: ISTAT resident population by citizenship and comune,
     1 Jan 2025 (religiondots' raw copy, read-only).
  2. measured minorities: South Tyrol's 2024 language-group declarations by comune (ASTAT,
     German / Italian / Ladin, as shares) on each comune's Italian citizens; Trentino's 2021
     Ladin, Mocheno and Cimbrian declarations by comune (ISPAT), as counts.
  3. immigrant languages: foreign citizens by citizenship, each on its country's main language,
     scaled so the region's total equals ISTAT 2024's "another language in the family" share
     (tav. 3); in Sardinia, Friuli, Aosta, South Tyrol and Trentino, foreign citizens x 61.5%
     (tav. 11: the share of non-Italian mother tongues who speak another language at home).
  4. local languages: tav. 3's "dialect" (only or mainly, plus half of "both Italian and
     dialect") as a share of those not speaking another language, on everyone left; each
     comune's dialect is drawn as the language it is (sources/it_regional.py).
  5. Slovene, Griko and Calabrian Greek: tav. 14 knowledge x 49.1% family use.
  6. Italian: everyone else.
Every row is `derived` except steps 2 (`measured`). The record is sources/it.md.
"""
import io
import json
import os
import re
import sys
import urllib.request
import zipfile

os.environ.setdefault("OMP_NUM_THREADS", "2")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "taxonomy"))
import it_regional as R  # noqa: E402

RAW = os.path.join(ROOT, "data", "raw", "it")
OUT = os.path.join(ROOT, "data", "normalized", "it.csv")
WEIGHTS = os.path.join(ROOT, "data", "geo", "it", "it_weights.csv")
RD = os.path.join(os.path.dirname(ROOT), "religiondots")
RD_LAU = os.path.join(RD, "data", "geo", "it", "it_lau.gpkg")
RCS = os.path.join(RD, "data", "raw", "it", "Dati_RCS_cittadinanza_2025.zip")
YEAR = 2025
SOURCE_ID = "it_rcs2025_x_istat2024"

FILES = {
    "istat_lingue_2024_tavole.xlsx":
        "https://www.istat.it/wp-content/uploads/2026/01/Tavole_Report_lingue-e-dialetti.xlsx",
    "istat_lingue_2024_report.pdf":
        "https://www.istat.it/wp-content/uploads/2026/01/REPORT_lingue-e-dialetti-def.pdf",
    # ASTAT, Sprachgruppenzugehörigkeit 2024, CC0, the province's WFS
    "astat_language_belonging_2024.csv":
        "https://geoservices1.civis.bz.it/geoserver/p_bz-Astat/ows?service=WFS&version=2.0.0"
        "&request=GetFeature&typeNames=p_bz-Astat:LanguageBelonging&outputFormat=csv",
    "ispat_minoranze_2021.pdf":
        "http://www.statistica.provincia.tn.it/binary/pat_statistica_new/popolazione/"
        "RilevazioneMinoranze_2021.1651135867.pdf",
}

# ---------------------------------------------------------------------------------------------
# citizenship (ISTAT's Italian names) -> ISO 3166 alpha-2 as Eurostat writes it (EL, UK), then
# fr_build.COUNTRY_LANG gives the language; ITALY_OVERRIDES differ from France on purpose.
# ---------------------------------------------------------------------------------------------
CIT_ISO = {
    "Romania": "RO", "Albania": "AL", "Marocco": "MA", "Cina": "CN", "Ucraina": "UA",
    "Bangladesh": "BD", "Egitto": "EG", "India": "IN", "Pakistan": "PK", "Filippine": "PH",
    "Nigeria": "NG", "Tunisia": "TN", "Senegal": "SN", "Perù": "PE", "Sri Lanka": "LK",
    "Moldova": "MD", "Polonia": "PL", "Ecuador": "EC", "Brasile": "BR", "Bulgaria": "BG",
    "Macedonia del Nord": "MK", "Ghana": "GH", "Federazione russa": "RU", "Georgia": "GE",
    "Germania": "DE", "Kosovo": "XK", "Costa d'Avorio": "CI", "Francia": "FR",
    "Repubblica Dominicana": "DO", "Spagna": "ES", "Gambia": "GM", "Serbia": "RS", "Mali": "ML",
    "Cuba": "CU", "Colombia": "CO", "El Salvador": "SV", "Regno Unito": "UK", "Turchia": "TR",
    "Iran": "IR", "Burkina Faso": "BF", "Algeria": "DZ", "Bosnia-Erzegovina": "BA",
    "Camerun": "CM", "Argentina": "AR", "Afghanistan": "AF", "Stati Uniti d'America": "US",
    "Venezuela": "VE", "Guinea": "GN", "Croazia": "HR", "Bolivia": "BO", "Paesi Bassi": "NL",
    "Bielorussia": "BY", "Somalia": "SO", "Slovacchia": "SK", "Ungheria": "HU",
    "Svizzera": "CH", "Grecia": "EL", "Iraq": "IQ", "Portogallo": "PT", "Giappone": "JP",
    "Siria": "SY", "Austria": "AT", "Belgio": "BE", "Etiopia": "ET", "Thailandia": "TH",
    "Eritrea": "ER", "Messico": "MX", "Repubblica ceca": "CZ", "Lituania": "LT", "Libano": "LB",
    "Togo": "TG", "Maurizio": "MU", "Irlanda": "IE", "Honduras": "HN",
    "Repubblica Democratica del Congo": "CD", "Cile": "CL", "Corea del Sud": "KR",
    "Indonesia": "ID", "Slovenia": "SI", "Svezia": "SE", "Capo Verde": "CV", "Congo": "CG",
    "Benin": "BJ", "Libia": "LY", "Lettonia": "LV", "Sudan": "SD", "Kirghizistan": "KG",
    "Kenya": "KE", "Canada": "CA", "Guinea-Bissau": "GW", "Sierra Leone": "SL",
    "Kazakhstan": "KZ", "Israele": "IL", "Paraguay": "PY", "Nepal": "NP", "Danimarca": "DK",
    "Montenegro": "ME", "Madagascar": "MG", "Vietnam": "VN", "Niger": "NE", "Australia": "AU",
    "Palestina": "PS", "Finlandia": "FI", "Armenia": "AM", "Giordania": "JO", "Tanzania": "TZ",
    "Uruguay": "UY", "Estonia": "EE", "Liberia": "LR", "Uzbekistan": "UZ", "Angola": "AO",
    "San Marino": "SM", "Norvegia": "NO", "Azerbaigian": "AZ", "Guatemala": "GT",
    "Dominica": "DM", "Malta": "MT", "Nicaragua": "NI", "Sudafrica": "ZA", "Burundi": "BI",
    "Taiwan": "TW", "Mauritania": "MR", "Ruanda": "RW", "Uganda": "UG", "Costa Rica": "CR",
    "Malaysia": "MY", "Gabon": "GA", "Myanmar/Birmania": "MM", "Ciad": "TD", "Haiti": "HT",
    "Cipro": "CY", "Panama": "PA", "Mozambico": "MZ", "Nuova Zelanda": "NZ", "Yemen": "YE",
    "Seychelles": "SC", "Lussemburgo": "LU", "Mongolia": "MN", "Zimbabwe": "ZW",
    "Zambia": "ZM", "Singapore": "SG", "Cambogia": "KH", "Sud Sudan": "SS",
    "Repubblica Centrafricana": "CF", "Guinea Equatoriale": "GQ", "Islanda": "IS",
    "Arabia Saudita": "SA", "Tagikistan": "TJ", "Giamaica": "JM", "Timor Leste": "TL",
    "Malawi": "MW", "Turkmenistan": "TM", "Laos": "LA", "Corea del Nord": "KP", "Kuwait": "KW",
    "Trinidad e Tobago": "TT", "Namibia": "NA", "Samoa": "WS", "Monaco": "MC",
    "Papua Nuova Guinea": "PG", "Liechtenstein": "LI", "Eswatini": "SZ",
    "Antigua e Barbuda": "AG", "Figi": "FJ", "Gibuti": "DJ", "Qatar": "QA", "Bhutan": "BT",
    "Botswana": "BW", "Emirati Arabi Uniti": "AE", "Guyana": "GY", "Oman": "OM",
    "Bahamas": "BS", "Belize": "BZ", "Bahrein": "BH", "Sao Tomé e Principe": "ST",
    "Santa Lucia": "LC", "Maldive": "MV", "Barbados": "BB", "Saint Kitts e Nevis": "KN",
    "Tonga": "TO", "Lesotho": "LS", "Suriname": "SR", "Grenada": "GD", "Andorra": "AD",
    "Stato della Città del Vaticano": "VA", "Vanuatu": "VU", "Isole Salomone": "SB",
    "Saint Vincent e Grenadine": "VC", "Brunei Darussalam": "BN", "Comore": "KM",
    "Stati Federati di Micronesia": "FM", "Isole Marshall": "MH", "Kiribati": "KI",
    "Palau": "PW",
    "Apolide": None,          # stateless: drawn as Italian (525 people)
}
ITALY_OVERRIDES = {
    "FR": "French",
    "IN": "Punjabi",   # Italy's Indians are overwhelmingly Punjabi (Sikh farm labour, Po valley)
    "LK": "Sinhala",   # Italy's Sri Lankans are mostly Sinhalese from the Negombo coast
    "CH": "German",    # France drew Swiss as French; Swiss citizens here are mostly German-Swiss
    "BE": "Dutch",     # likewise Belgium: France had French for its francophone emigration
    "CA": "English",
    "SM": "Italian", "VA": "Italian",
}


def lang_of(iso):
    """The shared origin table (sources/origin_mix.py, 2026-10-05): {label or node: share}.
    ITALY_OVERRIDES above is kept for the record only: the uncited ones (Sri Lanka, Switzerland,
    Belgium, Canada) went back to the home mix, India's Punjab share is an origin_mix override
    for Italy (Bertolani: 70% from Punjab), San Marino and the Vatican are Italian."""
    if iso is None or iso in ("SM", "VA"):
        return "Italian"
    from origin_mix import mix
    return {("Italian" if n == "indoeuropean.romance.italian" else n): s
            for n, s in mix(iso, "it").items()}


# ---------------------------------------------------------------------------------------------
def fetch():
    os.makedirs(RAW, exist_ok=True)
    for name, url in FILES.items():
        path = os.path.join(RAW, name)
        if os.path.exists(path):
            print(f"  have {name}")
            continue
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; "
                                                   "Win64; x64)"})
        data = urllib.request.urlopen(req, timeout=120).read()
        with open(path + ".tmp", "wb") as f:
            f.write(data)
        os.replace(path + ".tmp", path)
        print(f"  fetched {name}: {len(data):,} bytes")


def read_tav(sheet, first_col_rows=None):
    """An ISTAT tavola as {row label: [numbers]} (the percentage sheets)."""
    import openpyxl
    wb = openpyxl.load_workbook(os.path.join(RAW, "istat_lingue_2024_tavole.xlsx"),
                                read_only=True, data_only=True)
    out = {}
    for row in wb[sheet].iter_rows(values_only=True):
        if row and isinstance(row[0], str) and any(isinstance(v, (int, float)) for v in row[1:]):
            out.setdefault(row[0].strip(), [v if isinstance(v, (int, float)) else None
                                            for v in row[1:]])
    return out


def survey():
    """Tav. 3 home language by region; tav. 14 knowledge of protected languages; 6+ base."""
    t3 = read_tav("tav3")
    home = {}
    for reg, v in t3.items():
        i, d, m, a = v[0:4]
        if None in (i, d, m, a):
            continue
        s = i + d + m + a
        if not 96.5 <= s <= 100.5:
            sys.exit(f"!! tav3 {reg}: shares sum to {s}")
        # the rest to 100 is "not stated"; shares are taken among those who answered
        home[reg] = dict(I=i / s, D=d / s, M=m / s, A=a / s)
    t14 = read_tav("tav14")
    import openpyxl
    wb = openpyxl.load_workbook(os.path.join(RAW, "istat_lingue_2024_tavole.xlsx"),
                                read_only=True, data_only=True)
    hdr = None
    for row in wb["tav14"].iter_rows(values_only=True):
        cells = [" ".join(str(x).split()) if x else "" for x in (row or [])]
        if "Almeno una" in cells:
            hdr = cells[1:]
            break
    know = {reg: {h: (v / 100 if isinstance(v, (int, float)) else 0.0)
                  for h, v in zip(hdr, vals) if h} for reg, vals in t14.items()}
    # 6+ population, Italy (thousands): tav. 1 (segue), both sexes, total, in-family columns
    seg = None
    for row in wb["tav1 (segue)"].iter_rows(values_only=True):
        if row and isinstance(row[0], str) and row[0].strip() == "MASCHI E FEMMINE":
            seg = "mf"
        if seg == "mf" and row and isinstance(row[0], str) and row[0].strip() == "Totale":
            pop6 = sum(row[1:5]) * 1000
            break
    return home, know, pop6


# ---------------------------------------------------------------------------------------------
def population():
    """Comune x citizenship, 1 Jan 2025, joined to religiondots' comuni (LAU 2021)."""
    import geopandas as gpd
    lay = gpd.read_file(RD_LAU, ignore_geometry=True)
    lay["lau6"] = lay["lau"].astype(str).str.zfill(6)
    z = zipfile.ZipFile(RCS)
    d = pd.read_csv(z.open("Dati_cittadinanza_2025.csv"), sep=";", dtype=str, encoding="utf-8")
    d["n"] = d["Totale"].astype(int)
    prov = d[d["Codice Istat"].str.len() == 3]
    d = d[d["Codice Istat"].str.len() == 6].copy()
    unit = dict(zip(lay["lau6"], lay["unit"]))
    # comuni merged or moved since LAU 2021: the new comune -> one predecessor in the layer
    # (all in the same NUTS 3); the merged population is placed on the predecessors' comuni
    missing = sorted(set(d["Codice Istat"]) - set(unit))
    names = d.drop_duplicates("Codice Istat").set_index("Codice Istat")["Denominazione"]
    lay_by_name = lay.assign(nm=lay["name"].astype(str).str.lower())
    merged = {"025075": "025002",   # Setteville = Alano di Piave + Quero Vas (Belluno), 2025
              "028108": "028022"}   # Santa Caterina d'Este = Carceri + Vighizzolo d'Este, 2025
    for code in missing:
        nm = names[code]
        if code in merged:
            unit[code] = unit[merged[code]]
            print(f"  RCS comune {code} {nm} -> {unit[code]} (merger of LAU {merged[code]} and "
                  f"a neighbour)")
            continue
        parts = [p.strip().lower() for p in re.split(r" con | e |-", nm) if p.strip()]
        hit = lay_by_name[lay_by_name["nm"].isin(parts + [nm.lower()]) &
                          (lay_by_name["lau6"].str[:3] == code[:3])]
        if hit.empty:   # moved province (Montecopiolo, Sassofeltrio: Pesaro -> Rimini, 2021)
            hit = lay_by_name[lay_by_name["nm"] == nm.lower()]
        if hit.empty:   # a new comune split from another (Misiliscemi from Trapani, 2021):
            hit = lay_by_name[lay_by_name["lau6"].str[:3] == code[:3]]   # its province
        if hit["unit"].nunique() != 1:
            sys.exit(f"!! cannot place RCS comune {code} {nm}")
        unit[code] = hit["unit"].iloc[0]
        print(f"  RCS comune {code} {nm} -> {hit['unit'].iloc[0]} via {hit['name'].tolist()}")
    lay["rcs_name"] = lay["lau6"].map(names)
    d["unit"] = d["Codice Istat"].map(unit)
    d["nuts2"] = d["unit"].str[:4]
    gone = sorted(set(lay["lau6"]) - set(d["Codice Istat"]))
    print(f"  {len(gone)} LAU 2021 comuni absent from RCS 2025 (merged; placement only)")
    tot = d["n"].sum()
    if abs(tot - prov["n"].sum()) > 0:
        sys.exit(f"!! comuni {tot:,} against provinces {prov['n'].sum():,}")
    print(f"  RCS 2025: {tot:,} residents in {d['Codice Istat'].nunique():,} comuni, "
          f"{d['unit'].nunique()} NUTS 3; Italian citizens "
          f"{d.loc[d['Stato di cittadinanza'] == 'Italia', 'n'].sum():,}")
    unknown = sorted(set(d["Stato di cittadinanza"]) - set(CIT_ISO) - {"Italia"})
    if unknown:
        sys.exit(f"!! citizenships with no ISO code: {unknown}")
    return d, lay


REGION_OF_NUTS2 = {
    "ITC1": "Piemonte", "ITC2": "Valle d'Aosta/Vallée d'Aoste", "ITC3": "Liguria",
    "ITC4": "Lombardia", "ITH1": "Bolzano/Bozen", "ITH2": "Trento", "ITH3": "Veneto",
    "ITH4": "Friuli-Venezia Giulia", "ITH5": "Emilia-Romagna", "ITI1": "Toscana",
    "ITI2": "Umbria", "ITI3": "Marche", "ITI4": "Lazio", "ITF1": "Abruzzo", "ITF2": "Molise",
    "ITF3": "Campania", "ITF4": "Puglia", "ITF5": "Basilicata", "ITF6": "Calabria",
    "ITG1": "Sicilia", "ITG2": "Sardegna",
}


def comune_lookup(lay, region, name):
    # the layer has GISCO's names (some only Slovene: "Zgonik"); ISTAT's are bilingual
    # ("Sgonico-Zgonik"): match the layer name, ISTAT's, or ISTAT's Italian half
    key = name.lower()
    nm = lay["name"].astype(str).str.lower()
    rn = lay["rcs_name"].fillna("").astype(str).str.lower()
    m = lay[(lay["region"] == region) &
            ((nm == key) | (rn == key) | (rn.str.split("-").str[0].str.strip() == key) |
             (nm.str.split("/").str[0].str.strip() == key))]
    if len(m) != 1:
        sys.exit(f"!! comune {name!r} in {region}: {len(m)} matches")
    return m.iloc[0]["lau6"]


def zones(lay):
    """Each LAU comune's local-language label (its 'dialetto'), from it_regional.py."""
    lay["region"] = lay["nuts2"].map(REGION_OF_NUTS2)
    lay["zone"] = lay["region"].map(R.REGION_DIALECT).fillna("")
    for cap, lab in R.PROVINCE_DIALECT.items():
        units = lay.loc[lay["name"].astype(str).str.lower() == cap.lower(), "unit"].unique()
        if len(units) != 1:
            sys.exit(f"!! capoluogo {cap}: {len(units)} units")
        lay.loc[lay["unit"] == units[0], "zone"] = lab
    n = 0
    for (region, lab), names in R.COMUNE_LANG.items():
        for nm in names:
            lay.loc[lay["lau6"] == comune_lookup(lay, region, nm), "zone"] = lab
            n += 1
    print(f"  {n} comuni given a language of their own (it_regional.COMUNE_LANG)")
    return lay


# ---------------------------------------------------------------------------------------------
def astat(lay):
    """South Tyrol 2024: per comune, German / Italian / Ladin shares of declarations."""
    a = pd.read_csv(os.path.join(RAW, "astat_language_belonging_2024.csv"), dtype=str)
    a["lau6"] = a["ISTAT_CODE"].str.zfill(6)
    for c in ("DEU", "ITA", "LLD"):
        a[c] = a[c].astype(float) / 100
    s = a[["DEU", "ITA", "LLD"]].sum(axis=1)
    if (abs(s - 1) > 0.002).any():
        sys.exit("!! ASTAT shares do not sum to 100 in some comune")
    bz = set(lay.loc[lay["region"] == "Bolzano/Bozen", "lau6"])
    if set(a["lau6"]) != bz:
        sys.exit(f"!! ASTAT comuni {len(a)} against the layer's {len(bz)}")
    print(f"  ASTAT 2024: {len(a)} comuni, all South Tyrol's")
    return a.set_index("lau6")[["DEU", "ITA", "LLD"]]


def ispat(lay):
    """Trentino 2021: Ladin, Mocheno and Cimbrian declarations by comune (tables 2, 7, 12)."""
    import fitz
    doc = fitz.open(os.path.join(RAW, "ispat_minoranze_2021.pdf"))
    tables = {"Ladin": (9, 12, 15775), "Mocheno": (17, 18, None), "Cimbrian": (23, 24, 1111)}
    out = []
    for lab, (p0, p1, want) in tables.items():
        lines = []
        for p in range(p0, p1):
            lines += [x.strip() for x in doc[p].get_text().splitlines()]
        rows, i = [], 0
        while i + 3 < len(lines):
            nm, a, b, c = lines[i:i + 4]
            if (re.fullmatch(r"[\d.]+", a) and re.fullmatch(r"[\d.]+", b)
                    and re.fullmatch(r"[\d]+,\d", c) and not re.fullmatch(r"[\d.,]+", nm)):
                rows.append((nm, int(a.replace(".", "")), int(b.replace(".", ""))))
                i += 4
            else:
                i += 1
        total = [r for r in rows if r[0] == "Totale provincia"]
        rows = [r for r in rows if r[0] not in ("Totale provincia",)]
        got = sum(r[1] for r in rows)
        if total and got != total[0][1]:
            sys.exit(f"!! ISPAT {lab}: comuni sum to {got} against {total[0][1]}")
        if want and total and total[0][1] != want:
            sys.exit(f"!! ISPAT {lab}: total {total[0][1]}, expected {want}")
        tn = lay[lay["region"] == "Trento"]
        for nm, n, popc in rows:
            if nm == "Altri comuni":
                out.append((None, lab, n))
                continue
            cand = [nm, nm.split("-")[0]]
            m = tn[tn["name"].astype(str).str.lower().isin([c.lower() for c in cand])]
            if len(m) != 1:
                sys.exit(f"!! ISPAT {lab}: comune {nm!r} matched {len(m)}")
            out.append((m.iloc[0]["lau6"], lab, n))
        print(f"  ISPAT 2021 {lab}: {got:,} in {len(rows)} rows "
              f"(table total {total[0][1] if total else 'n/a'})")
    return pd.DataFrame(out, columns=["lau6", "label", "count"])


# ---------------------------------------------------------------------------------------------
def build():
    import fr_build  # noqa: F401  (COUNTRY_LANG)
    import it2025
    home, know, pop6 = survey()
    d, lay = population()
    lay = zones(lay)
    lay["region"] = lay["nuts2"].map(REGION_OF_NUTS2)
    d["region"] = d["nuts2"].map(REGION_OF_NUTS2)
    pop_tot = d["n"].sum()
    u6 = 1 - pop6 / pop_tot
    k_young = 1 - u6 * (1 - R.YOUNG_RATIO)
    print(f"  survey base 6+ {pop6:,.0f} of {pop_tot:,}: under-6 {u6:.2%}; dialect shares x "
          f"{k_young:.4f} for the under-6s (6-14 ratio {R.YOUNG_RATIO:.3f})")

    d["iso"] = d["Stato di cittadinanza"].map(lambda s: "IT" if s == "Italia" else CIT_ISO[s])
    # a country may split across languages (Algeria, Morocco: fr_build's shares)
    split = {}
    for i in d["iso"].unique():
        lg = "Italian" if i == "IT" else lang_of(i)
        split[i] = list(lg.items()) if isinstance(lg, dict) else [(lg, 1.0)]
    d["lang"] = d["iso"].map(lambda i: split[i])
    d = d.explode("lang")
    d["n"] = d["n"] * d["lang"].map(lambda t: t[1])
    d["lang"] = d["lang"].map(lambda t: t[0])
    for lab in d["lang"].unique():
        it2025.resolve(lab)
    foreign = d[d["iso"] != "IT"]
    ital_c = d[d["iso"] == "IT"].groupby("Codice Istat")["n"].sum()

    rows = []        # (unit, label, count, tier, part)
    wrows = []       # (lau6, label, weight) placement weights for local / measured languages
    unit_pop = d.groupby("unit")["n"].sum()
    reg_pop = d.groupby("region")["n"].sum()
    reg_for = foreign.groupby("region")["n"].sum()
    used = {u: 0.0 for u in unit_pop.index}

    # ---- measured: South Tyrol and Trentino -------------------------------------------------
    a = astat(lay)
    bz_ital = ital_c.reindex(d.loc[d["region"] == "Bolzano/Bozen", "Codice Istat"].unique())
    bz_ital.index = bz_ital.index.astype(str)
    for col, lab in (("DEU", "German"), ("ITA", "Italian"), ("LLD", "Ladin")):
        w = a[col].reindex(bz_ital.index).fillna(0) * bz_ital
        u = d.loc[d["region"] == "Bolzano/Bozen", "unit"].iloc[0]
        rows.append((u, lab, w.sum(), "measured", "South Tyrol language groups 2024"))
        used[u] += w.sum()
        wrows += [(c, f"bz:{lab}", v) for c, v in w.items() if v > 0]
    print(f"  South Tyrol citizens {bz_ital.sum():,}: " + ", ".join(
        f"{r[1]} {r[2]:,.0f}" for r in rows))
    tn = ispat(lay)
    tn_unit = d.loc[d["region"] == "Trento", "unit"].iloc[0]
    tn_layer = lay[lay["region"] == "Trento"]
    for lab, g in tn.groupby("label"):
        n = g["count"].sum()
        rows.append((tn_unit, lab, n, "measured", "Trentino minority declarations 2021"))
        used[tn_unit] += n
        named = g[g["lau6"].notna()]
        wrows += [(c, lab, v) for c, v in zip(named["lau6"], named["count"])]
        rest = g.loc[g["lau6"].isna(), "count"].sum()
        if rest:   # "altri comuni": spread on the province's other comuni by population
            others = tn_layer[~tn_layer["lau6"].isin(named["lau6"])]
            wrows += [(c, lab, rest * p / others["pop"].sum())
                      for c, p in zip(others["lau6"], others["pop"])]

    # ---- immigrant languages ----------------------------------------------------------------
    special = R.REGIONAL_IN_OTHER | {"Bolzano/Bozen", "Trento"}
    print("  immigrant languages by region: 'another language' x population against foreign "
          "citizens")
    scale = {}
    for reg in sorted(reg_pop.index):
        h = home[reg]
        if reg in special:
            scale[reg] = R.RETENTION
            how = "retention"
        else:
            scale[reg] = h["A"] * reg_pop[reg] / reg_for[reg]
            how = "survey"
        print(f"    {reg:30} other-language {h['A']:5.1%}  foreign {reg_for[reg] / reg_pop[reg]:5.1%}"
              f"  -> x{scale[reg]:.3f} of foreign citizens ({how})")
        if scale[reg] > 1.6:
            sys.exit(f"!! {reg}: scale {scale[reg]:.2f}")
    imm = foreign.assign(c=foreign["n"] * foreign["region"].map(scale))
    gi = imm.groupby(["unit", "lang"])["c"].sum()
    for (u, lab), n in gi.items():
        rows.append((u, lab, n, "derived", "immigrants by citizenship"))
        used[u] += n
    print(f"  immigrant languages: {gi.sum():,.0f} of {foreign['n'].sum():,} foreign citizens "
          f"({gi.sum() / foreign['n'].sum():.1%})")

    # ---- fixed-count minorities (tav. 14 x family use) ---------------------------------------
    fixed = {}
    for (reg, lab, col), names in R.KNOW_FIXED.items():
        share = know[reg][col] * R.FAMILY_USE
        n = share * reg_pop[reg]
        codes = [comune_lookup(lay, reg, nm) for nm in names]
        z = lay[lay["lau6"].isin(codes)]
        w = z.set_index("lau6")["ital"]
        by_unit = z.groupby("unit")["ital"].sum()
        for u, wu in by_unit.items():
            fixed[(u, lab)] = n * wu / by_unit.sum()
        wrows += [(c, lab, v) for c, v in w.items()]
        print(f"  {lab}: {know[reg][col]:.1%} of {reg} know it x {R.FAMILY_USE:.1%} = {n:,.0f}, "
              f"placed on {len(codes)} comuni in {by_unit.size} provinces")

    # ---- local languages ("dialetto") ---------------------------------------------------------
    lay_w = lay[lay["region"] != "Bolzano/Bozen"].copy()
    region_of_unit = d.drop_duplicates("unit").set_index("unit")["region"]
    imm_reg = imm.groupby("region")["c"].sum()
    for u in sorted(unit_pop.index):
        reg = region_of_unit[u]
        rest = unit_pop[u] - used[u]
        if reg == "Bolzano/Bozen":
            rows.append((u, "Italian", rest, "derived", "foreign citizens not drawn on their "
                                                        "language"))
            continue
        h = home[reg]
        if reg in R.REGIONAL_IN_OTHER:
            a_imm = imm_reg[reg] / reg_pop[reg]
            lam = (h["D"] + h["M"] / 2 + max(0.0, h["A"] - a_imm)) / (1 - a_imm)
        else:
            lam = (h["D"] + h["M"] / 2) / (h["I"] + h["D"] + h["M"])
        lam *= k_young
        local = lam * rest
        fx = {lab: n for (uu, lab), n in fixed.items() if uu == u}
        local_left = local - sum(fx.values())
        if local_left < 0:
            sys.exit(f"!! {u}: fixed minorities exceed the local-language share")
        z = lay_w[lay_w["unit"] == u]
        zw = z.groupby("zone")["ital"].sum()
        for lab, wz in zw.items():
            n = local_left * wz / zw.sum()
            if lab == "Italian":   # central dialects: Italian, placed on all Italian citizens
                continue
            rows.append((u, lab, n, "derived", "regional survey, dialect at home"))
            used[u] += n
            wrows += [(c, f"zone:{lab}", v) for c, v in
                      zip(z.loc[z["zone"] == lab, "lau6"], z.loc[z["zone"] == lab, "ital"])]
        for lab, n in fx.items():
            rows.append((u, lab, n, "derived", "regional survey, knowledge x family use"))
            used[u] += n
        rows.append((u, "Italian", unit_pop[u] - used[u], "derived", "everyone else"))

    df = pd.DataFrame(rows, columns=["geo_id", "source_category", "count", "tier", "part"])
    df = df.groupby(["geo_id", "source_category", "tier", "part"], as_index=False)["count"].sum()
    if (df["count"] < -0.5).any():
        sys.exit(f"!! negative rows:\n{df[df['count'] < 0]}")
    chk = df.groupby("geo_id")["count"].sum() - unit_pop
    # rows under half a person (the origin mixes' long tails) go onto Italian, so totals hold
    small = df["count"] <= 0.5
    tiny = df[small].groupby("geo_id")["count"].sum()
    df = df[~small].copy()
    rest = (df["source_category"] == "Italian") & (df["part"] == "everyone else")
    df.loc[rest, "count"] += df.loc[rest, "geo_id"].map(tiny).fillna(0.0)
    if chk.abs().max() > 1:
        sys.exit(f"!! units do not sum to RCS 2025 population: {chk.abs().max():,.0f}")
    df["geo_level"] = "nuts3"
    df["year"] = YEAR
    df["source_id"] = SOURCE_ID
    for lab in df["source_category"].unique():
        it2025.resolve(lab)
    df = df[["geo_id", "geo_level", "source_category", "count", "tier", "part", "year",
             "source_id"]]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT + ".tmp", index=False)
    os.replace(OUT + ".tmp", OUT)

    w = pd.DataFrame(wrows, columns=["lau6", "key", "weight"])
    w = w.groupby(["lau6", "key"], as_index=False)["weight"].sum()
    os.makedirs(os.path.dirname(WEIGHTS), exist_ok=True)
    w.to_csv(WEIGHTS + ".tmp", index=False)
    os.replace(WEIGHTS + ".tmp", WEIGHTS)

    nat = df.groupby("source_category")["count"].sum().sort_values(ascending=False)
    print(f"\nwrote {OUT}: {len(df):,} rows, {df['geo_id'].nunique()} provinces, "
          f"{df['count'].sum():,.0f} people, {len(nat)} languages; {WEIGHTS}: {len(w):,} rows")
    for lab, n in nat.head(40).items():
        print(f"  {lab:22} {n:12,.0f}  {n / nat.sum():6.2%}")
    print("by region, Italian / local / immigrant:")
    df["region"] = df["geo_id"].str[:4].map(REGION_OF_NUTS2)
    for reg, g in df.groupby("region"):
        t = g["count"].sum()
        itn = g.loc[g["source_category"] == "Italian", "count"].sum()
        loc = g.loc[g["part"].str.startswith(("regional", "South", "Trentino")) &
                    (g["source_category"] != "Italian"), "count"].sum()
        im = g.loc[g["part"] == "immigrants by citizenship", "count"].sum()
        print(f"  {reg:30} {t:11,.0f}  Italian {itn / t:6.1%}  local {loc / t:6.1%}  "
              f"immigrant {im / t:5.1%}")


if __name__ == "__main__":
    if "--fetch" in sys.argv:
        fetch()
    build()
