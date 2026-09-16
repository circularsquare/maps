"""Mozambique — INE, III RGPH 2007, religion by district, carried onto the 2017 census's totals.

Reads (or fetches) data/raw/mz/isd2007/ and data/raw/mz/q3_2017/, and writes
    data/normalized/mz_2007.csv       the 2007 table as printed: share x N, per 2007 district
    data/normalized/mz_districts.csv  what is drawn: 2007 district shares on 2017 totals

sources/mz.md §7 is the record. The short form:

## THE SOURCE

INE, *III Recenseamento Geral da População e Habitação 2007: Indicadores Sócio-Demográficos
Distritais*, one volume per province (2010-2012). Quadro 8.1 (printed as `Quadro 8.` in four
volumes) is *Distribuição percentual da população por religião segundo distritos*: the eight
answers of 2017's Quadro 11, Católica to Desconhecida, as one-decimal shares, with each
district's population `N`. Nine of the eleven volumes are in the Wayback Machine, on INE's 2015
Plone tree or its 2019 one; **Cabo Delgado and Manica are not captured at any timestamp**
(sources.md §scout-2026-09-15-africa-upgrades), so those two stay at province, drawn from 2017's
Quadro 11 by countries/mz.py.

## THE CONSTRUCTION: 2007 SHAPE, 2017 SIZE (Switzerland's, countries/ch.py)

A 2007 district's religion is known only as it was in 2007. So each province is fitted by
iterative proportional fitting to two sets of 2017 margins, with the 2007 table supplying only
the association between district and answer:

  * columns: the province's eight answers in 2017's Quadro 11 (data/normalized/mz.csv);
  * rows: 2017 population per 2007 district, from each province's 2017 Quadro 3
    (*População por idade, segundo área de residência, distrito e sexo*).

Both margins are exact in the output. Every row drawn from this file is `derived`.

**THE ROW MARGIN IS PER CLUSTER, NOT PER DISTRICT.** Mozambique made 23 districts in 2013 and
two more in 2016, mostly by raising one administrative post to a district, but some moved
posts between districts, so a 2017 district total is not always a sum of whole 2007 districts.
`clusters()` links 2007 districts and 2017 districts through the posts they share (COD-AB's
adm3 layer, with POST_MOVES for posts that changed district) and takes each connected group as
one row target. Inside a group, the 2007 districts keep their 2007 population proportions.
**Zambézia has no 2017 Quadro 3** (its captured set lacks every district table, Quadros 3, 6,
7 and 8), so the whole province is one group there and its districts keep 2007's proportions.

`growth_witness()` is the check that found the moved posts: 2017 over 2007 population per
group, against the province's own growth. Nampula-Rapale alone reads 0.82 against the
province's 1.38, and with Cidade de Nampula 1.37, because Anchilo post is in the city now.

## TRAPS

  * **N printed off its row.** In Tete, Sofala, Inhambane, Gaza and Maputo Cidade the text
    layer puts some districts' `N` on the line above the shares, or between the name and the
    shares (Tete's Moatize). The reader takes the table as one token stream and pairs the Nth
    count with the Nth row; the N column's sum to the Total row, and the N-weighted shares
    against the Total row, catch a wrong pairing.
  * **Gaza's printed Total row is not its districts.** See GAZA_TOTAL_ROW below.
  * **The volumes were made from one template.** Nampula's credits page says Niassa and
    Maputo Província; Zambézia's commentary talks about Tete. Only the table caption is
    checked for the province.

Usage:
    python sources/mz_2007.py --fetch     nine PDFs and eight xlsx from the Wayback Machine, ~7 MB
    python sources/mz_2007.py             read, check, fit, write
"""

import csv
import hashlib
import os
import re
import sys
import time
import unicodedata
from collections import defaultdict

from fetch_checks import check_body, wayback_raw   # shared, sources/fetch_checks.py
from mz import CATEGORIES, fold                    # the 2017 reader's eight answers, same spelling

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW07 = os.path.join(ROOT, "data", "raw", "mz", "isd2007")
RAW17 = os.path.join(ROOT, "data", "raw", "mz", "q3_2017")
ADM3 = os.path.join(ROOT, "data", "raw", "mz", "shp", "moz_admin3.shp")
NORM17 = os.path.join(ROOT, "data", "normalized", "mz.csv")
OUT07 = os.path.join(ROOT, "data", "normalized", "mz_2007.csv")
OUT = os.path.join(ROOT, "data", "normalized", "mz_districts.csv")

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]
LEVEL = "district2007"

T15 = ("http://www.ine.gov.mz/estatisticas/estatisticas-demograficas-e-indicadores-sociais/"
       "populacao/relatorio-de-indicadores-distritais-2007/")
T19 = "http://www.ine.gov.mz/operacoes-estatisticas/censos/censo-2007/rgph-2007/indicadores-distritais/"

# (province code, name, Wayback timestamp, URL, table page(s) 1-based, folded caption key, size)
VOLUMES = [
    ("01", "Niassa", "20150410195551",
     T15 + "indicadores-socio-demograficos-distritais-2007-niassa/at_download/file",
     (20,), "provinciadeniassa", 770_728),
    ("03", "Nampula", "20150410194648",
     T15 + "indicadores-socio-demograficos-distritais-2007-nampula/at_download/file",
     (21,), "provinciadenampula", 763_235),
    ("04", "Zambézia", "20150410195646",
     T15 + "indicadores-socio-demograficos-distritais-2007-provincia-de-zambezia/at_download/file",
     (22,), "provinciadezambezia", 812_230),
    ("05", "Tete", "20150410194849",
     T15 + "indicadores-socio-demograficos-distritais-2007-provincia-de-tete/at_download/file",
     (20,), "provinciadetete", 711_909),
    ("07", "Sofala", "20150410194919",
     T15 + "indicadores-socio-demograficos-distritais-2007-provincia-de-sofala/at_download/file",
     (18,), "provinciadesofala", 811_715),
    ("08", "Inhambane", "20150410195422",
     T15 + "indicadores-socio-demograficos-distritais-2007-provincia-de-inhambane/at_download/file",
     (19,), "provinciadeinhambane", 910_990),
    # Gaza's path repeats its own slug on INE's server; the capture is of that path.
    ("09", "Gaza", "20150410195119",
     T15 + "indicadores-socio-demograficos-distritais-2007-provincia-de-gazaindicadores-socio-"
           "demograficos-distritais-2007-provincia-de-gaza/at_download/file",
     (20,), "provinciadegaza", 881_946),
    ("10", "Maputo Província", "20150410194746",
     T15 + "indicadores-socio-demograficos-distritais-2007-maputo-provincia/at_download/file",
     (21,), "maputoprovincia", 838_613),
    # Cidade de Maputo is captured only on the 2019 tree, and its table runs over two pages.
    ("11", "Maputo Cidade", "20190201065532",
     T19 + "regiao-sol/indicadores-socio-demograficos-distritais-cidade-de-maputo.pdf/at_download/file",
     (15, 16), "maputocidade", 622_073),
]

R17 = "http://www.ine.gov.mz/iv-rgph-2017/"
# 2017 Quadro 3 per province: (code, Wayback timestamp, URL, size). No Zambézia: see the docstring.
Q3 = [
    ("01", "20191114002621", R17 + "niassa/quadro-3-populacao-por-idade-segundo-area-de-residencia-"
                                   "distrito-e-sexo-provincia-de-niassa-2017.xlsx", 20_289),
    ("03", "20191114010934", R17 + "nampula/quadro-3-populacao-por-idade-segundo-area-de-residencia-"
                                   "distrito-e-sexo-provincia-de-nampula-2017.xlsx", 24_176),
    ("05", "20191114015524", R17 + "tete/quadro-3-populacao-por-idade-segundo-area-de-residencia-"
                                   "distrito-e-sexo-provincia-de-tete-2017.xlsx", 20_530),
    ("07", "20191114025715", R17 + "sofala/quadro-3-populacao-por-idade-segundo-distrito-area-de-"
                                   "residencia-e-sexo-provincia-de-sofala-2017.xlsx", 19_321),
    ("08", "20191114031532", R17 + "inhambane/quadro-3-populacao-por-idade-segundo-area-de-"
                                   "residencia-distrito-e-sexo-provincia-de-inhambane-2017.xlsx", 19_299),
    ("09", "20191114035221", R17 + "gaza/quadro-3-populacao-por-idade-segundo-area-de-residencia-"
                                   "distrito-e-se.xlsx", 19_351),
    ("10", "20191015090808", R17 + "maputo-provincia/quadro-3-populacao-por-idade-segundo-distrito-"
                                   "area-de-residencia-e-sexo-maputo-provincia-2017.xlsx", 16_888),
    ("11", "20191114050419", R17 + "maputo-cidade/quadro-3-populacao-por-idade-segundo-distrito-e-"
                                   "sexo-maputo-cidade-2017.xlsx", 14_934),
]

# Each 2007 district as Quadro 8.1 prints it, in its printed order, with the COD-AB (2025)
# districts whose posts it held by default. The first pcode is the district's `geo_id`.
# Posts that changed district are in POST_MOVES, which overrides this.
DISTRICTS = {
    "01": [("Cidade de Lichinga", ["MZ0101"]), ("Cuamba", ["MZ0102"]), ("Lago", ["MZ0103"]),
           ("Distrito de Lichinga", ["MZ0104"]), ("Majune", ["MZ0105"]),
           ("Mandimba", ["MZ0106"]), ("Marrupa", ["MZ0107"]), ("Maua", ["MZ0108"]),
           ("Mavago", ["MZ0109"]), ("Mecanhelas", ["MZ0110"]), ("Mecula", ["MZ0111"]),
           ("Metarica", ["MZ0112"]), ("Meumbe", ["MZ0113"]), ("N´gauma", ["MZ0114"]),
           ("Nipepe", ["MZ0115"]), ("Sanga", ["MZ0116"])],
    "03": [("Cidade de Nampula", ["MZ0301"]), ("Angoche", ["MZ0302"]), ("Erati", ["MZ0303"]),
           ("Ilha de Moçambique", ["MZ0304"]), ("Lalaua", ["MZ0305"]), ("Malema", ["MZ0306"]),
           ("Meconta", ["MZ0307"]), ("Mecuburi", ["MZ0308"]), ("Memba", ["MZ0309"]),
           ("Mogincual", ["MZ0310", "MZ0323"]), ("Mogovolas", ["MZ0311"]),
           ("Moma", ["MZ0312", "MZ0322"]), ("Monapo", ["MZ0313"]), ("Mossuril", ["MZ0314"]),
           ("Muecate", ["MZ0315"]), ("Murrupula", ["MZ0316"]), ("Nacala Porto", ["MZ0317"]),
           ("Nacala-Velha", ["MZ0318"]), ("Nacaroa", ["MZ0319"]),
           ("Nampula-Rapale", ["MZ0320"]), ("Ribaue", ["MZ0321"])],
    "04": [("Cidade de Quelimane", ["MZ0401"]), ("Alto Molocue", ["MZ0402"]),
           ("Chinde", ["MZ0403", "MZ0419"]), ("Gilé", ["MZ0404"]), ("Gurue", ["MZ0405"]),
           ("Ile", ["MZ0406", "MZ0422"]), ("Inhassunge", ["MZ0407"]), ("Lugela", ["MZ0408"]),
           ("Maganja da Costa", ["MZ0409", "MZ0420"]), ("Milange", ["MZ0410", "MZ0421"]),
           ("Mocuba", ["MZ0411"]), ("Mopeia", ["MZ0412"]),
           ("Morrumbala", ["MZ0413", "MZ0418"]), ("Namacurra", ["MZ0414"]),
           ("Namarroi", ["MZ0415"]), ("Nicoadala", ["MZ0416"]), ("Pebane", ["MZ0417"])],
    "05": [("Cidade de Tete", ["MZ0501"]), ("Angónia", ["MZ0502"]), ("Cahora Bassa", ["MZ0503"]),
           ("Changara", ["MZ0504", "MZ0515"]), ("Chifunde", ["MZ0505"]), ("Chiuta", ["MZ0506"]),
           ("Macanga", ["MZ0507"]), ("Magoe", ["MZ0508"]), ("Marávia", ["MZ0509"]),
           ("Moatize", ["MZ0510"]), ("Mutarara", ["MZ0511", "MZ0514"]),
           ("Tsangano", ["MZ0512"]), ("Zumbo", ["MZ0513"])],
    "07": [("Cidade da beira", ["MZ0701"]), ("Buzi", ["MZ0702"]), ("Caia", ["MZ0703"]),
           ("Chemba", ["MZ0704"]), ("Cheringoma", ["MZ0705"]), ("Chibabava", ["MZ0706"]),
           ("Dondo", ["MZ0707"]), ("Gorongoza", ["MZ0708"]), ("Machanga", ["MZ0709"]),
           ("Maringue", ["MZ0710"]), ("Marromeu", ["MZ0711"]), ("Muanza", ["MZ0712"]),
           ("Nhamatanda", ["MZ0713"])],
    "08": [("Cidade de Inhambane", ["MZ0801"]), ("Funhalouro", ["MZ0802"]),
           ("Govuro", ["MZ0803"]), ("Homoine", ["MZ0804"]), ("Inharrime", ["MZ0805"]),
           ("Inhassoro", ["MZ0806"]), ("Jangamo", ["MZ0807"]), ("Mabote", ["MZ0808"]),
           ("Massinga", ["MZ0809"]), ("Cidade da Maxixe", ["MZ0810"]),
           ("Morrumbene", ["MZ0811"]), ("Panda", ["MZ0812"]), ("Vilanculos", ["MZ0813"]),
           ("Zavala", ["MZ0814"])],
    "09": [("Cidade de Xai-Xai", ["MZ0901"]), ("Bilene Macia", ["MZ0902"]),
           ("Chibuto", ["MZ0903"]), ("Chicualacuala", ["MZ0904", "MZ0914"]),
           ("Chigubo", ["MZ0905"]), ("Chokwe", ["MZ0906"]), ("Guijá", ["MZ0907"]),
           ("Mabalane", ["MZ0908"]), ("Mandlacaze", ["MZ0909"]), ("Massangena", ["MZ0910"]),
           ("Massingir", ["MZ0911"]), ("Distrito de Xai-Xai", ["MZ0912", "MZ0913"])],
    "10": [("Cidade da Matola", ["MZ1001"]), ("Boane", ["MZ1002"]), ("Magude", ["MZ1003"]),
           ("Manhiça", ["MZ1004"]), ("Marracuene", ["MZ1005"]), ("Matutuine", ["MZ1006"]),
           ("Moamba", ["MZ1007"]), ("Namaacha", ["MZ1008"])],
    "11": [("KaMpfumu (DM 1)", ["MZ1101"]), ("Nlhamankulu (DM 2)", ["MZ1102"]),
           ("KaMaxakeni (DM 3)", ["MZ1103"]), ("KaMavota (DM 4)", ["MZ1104"]),
           ("KaMubukwana (DM 5)", ["MZ1105"]), ("KaTembe (DM 6)", ["MZ1106"]),
           ("KaNyaka (DM 7)", ["MZ1107"])],
}

# COD-AB post (adm3 pcode) -> the 2007 district (its geo_id) it belonged to, where that is not
# the default above. Each is witnessed by growth_witness() here and the Kontur band in
# sources/mz_geo.py; sources/mz.md §7 has the evidence.
POST_MOVES = {
    # Anchilo is in Cidade de Nampula now. Rapale alone reads 2017/2007 = 0.82 against the
    # province's 1.38; the city and Rapale together read 1.37.
    "MZ030107": "MZ0320",
    # Meponda and Lussanhando are in Cidade de Lichinga now; Chimbonila (2007's Distrito de
    # Lichinga) keeps only Chimbunila and Lione. Alone: city 1.70, Chimbonila 0.76.
    "MZ010105": "MZ0104",
    "MZ010106": "MZ0104",
    # Chongoene district (2016) took Mazucane and Nguzene from Mandlakazi as well as Chongoene
    # post from the old Xai-Xai district. Alone: Mandlakazi 0.83, old Xai-Xai 1.29.
    "MZ091302": "MZ0909",
    "MZ091303": "MZ0909",
    # Maquival is in Quelimane now. Portuguese Wikipedia's Nicoadala article: "O posto
    # administrativo de Maquival, até então parte deste distrito, foi transferido para o
    # distrito e município de Quelimane em 2013" (read 2026-09-15; its 2007 figure, 231,850, is
    # the volume's N to the person). Zambézia has no 2017 district table, so the witness is
    # Kontur (sources/mz_geo.py): people per 2007 N over the province's reads Nicoadala 0.55
    # and Quelimane 1.26 with Maquival in the city, 0.90 and 0.84 with it in Nicoadala.
    "MZ040106": "MZ0416",
}

# The printed 2007 names whose folded form differs from COD-AB's name for their first pcode.
# Asserted both ways, so a mis-typed pcode above fails instead of pairing two wrong districts.
NAME_DIFFERS = {
    "Distrito de Lichinga": "Chimbonila", "Meumbe": "Muembe", "Nacala-Velha": "Nacala-a-Velha",
    "Nampula-Rapale": "Rapale", "Gorongoza": "Gorongosa", "Vilanculos": "Vilankulo",
    "Bilene Macia": "Bilene", "Mandlacaze": "Mandlakazi", "Distrito de Xai-Xai": "Limpopo",
    "KaMpfumu (DM 1)": "KaMpfumo", "Nlhamankulu (DM 2)": "Nhlamankulo",
    "KaMaxakeni (DM 3)": "KaMaxaqueni",
}

# 2017 Quadro 3 spellings (folded, prefix stripped) that are not COD-AB's.
Q3_ALIAS = {"ilhademogincual": "mogincual", "ilhademossuril": "mossuril",
            "maoatize": "moatize", "gorongoza": "gorongosa", "mandlakaze": "mandlakazi",
            "nlhamankulu": "nhlamankulo", "kamaxaquene": "kamaxaqueni"}

# GAZA'S PRINTED TOTAL ROW IS NOT ITS DISTRICTS. It reads Católica 37.5, Anglicana 15.4,
# Islâmica 15.8, Zione/Sião 19.8, Evangélica/Pentecostal 7.0, Sem religião 0.9, Outra 3.0,
# Desconhecida 0.6, while no district is over 22% Católica or 10% Anglicana. The twelve
# districts weighted by their N give Católica 15.38, Anglicana 3.01, Islâmica 0.89, Zione/Sião
# 37.55, Evangélica/Pentecostal 15.77, Sem religião 19.82, Outra 6.96, Desconhecida 0.63
# (measured 2026-09-15): the same eight numbers, each within 0.05, set in the wrong cells. So
# the districts are right and the Total row is mis-set; it is not used for anything but this
# check. check_volume asserts the printed row is GAZA_PRINTED and the districts match GAZA_MEANT.
GAZA_PRINTED = [37.5, 15.4, 15.8, 19.8, 7.0, 0.9, 3.0, 0.6]
GAZA_MEANT = [15.4, 3.0, 0.9, 37.5, 15.8, 19.8, 7.0, 0.6]
assert sorted(GAZA_PRINTED) == sorted(GAZA_MEANT)

# The parsed tables, pinned: SHA-1 of every (district, N, eight shares) in printed order,
# recorded 2026-09-15 after every other check in check_volume passed.
TABLE_DIGESTS = {
    "01": "75718c50cd89a2f1816f6e26c17d4a9f16b196a2",
    "03": "e3357ecc474119a929461a31388f6b22b8ef735d",
    "04": "7ee7fa9b033736d8e7dcecc57723cffe78f74f7e",
    "05": "e18c1d1171b4604820b694d76c583ab4cf343662",
    "07": "34547fdcbc1da7286f0e24f48d3f08e1259c5c38",
    "08": "1b1d01286cdd5d569fb146b5d8b61c569be40f35",
    "09": "494d61800fc9388fa6c30c284f829ccfed2fc287",
    "10": "6e843a594ac62154a1501fc2dc2b90d77f90d991",
    "11": "003f21e3011a22a13a5019c27e30dd167812bfb6",
}

# 2017 QUADRO 3'S DISTRICTS DO NOT ALWAYS SUM TO ITS OWN T O T A L. Tete's fifteen district rows
# sum to 2,551,824 against a printed 2,551,826, each row's Homens + Mulheres closing. Two people
# is INE's arithmetic, not a misread row (a misread district is off by tens of thousands), so
# the difference is pinned per province and main() spreads it over the groups by largest
# remainder, which puts both on the largest.
Q3_SLACK = {"05": 2}

TOL_ROW = 0.45      # eight one-decimal cells can sum to 100.0 +/- 0.4
TOL_WEIGHTED = 0.1  # N-weighted district shares against the Total row: 0.05 + 0.05 rounding
# A group's 2017/2007 growth over its province's. The low side is what finds a moved post
# (Rapale alone 0.59, 2007's Distrito de Lichinga alone 0.52). The high side is wide because
# suburbs really did grow that fast: Marracuene 2.57 against Maputo Província's 1.58 (1.63),
# Cidade de Tete 1.97 against 1.43 (1.38). Mandlakazi alone would read 0.73 and pass, so the
# Gaza moves rest on COD-AB's post list and the pair's reading, not on this band.
GROWTH_BAND = (0.65, 2.0)

SHARE = re.compile(r"^\d{1,3}\.\d$")
COUNT = re.compile(r"^\d{1,3}(,\d{3})+$")
PAGENO = re.compile(r"^\d{1,3}$")


def nfc(s):
    return unicodedata.normalize("NFC", " ".join(str(s).split()))


def strip_prefix(f):
    """A folded unit name without `cidadede`, `distritode`, `distritomunicipal` and the like."""
    return re.sub(r"^(cidadede|cidadeda|cidadedo|distritomunicipal|distritode|distritoda)", "", f)


def key07(name):
    return strip_prefix(fold(re.sub(r"\(DM \d\)", "", name)))


# ---------------------------------------------------------------- fetch


def _get(url, where):
    import requests
    for attempt in range(8):
        try:
            r = requests.get(url, timeout=180,
                             headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        except requests.RequestException as e:
            print(f"  {where}: {type(e).__name__}, retrying")
            time.sleep(5 * (attempt + 1))
            continue
        if r.status_code == 200:
            return r.content
        print(f"  {where}: HTTP {r.status_code}, retrying")
        time.sleep(5 * (attempt + 1))
    raise SystemExit(f"{where}: the Wayback Machine did not return {url}")


def _save(dest, body):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest + ".part", "wb") as fh:
        fh.write(body)
    os.replace(dest + ".part", dest)


def fetch():
    for code, name, ts, url, _, _, size in VOLUMES:
        dest = vol_path(code)
        if os.path.exists(dest) and os.path.getsize(dest) == size:
            print(f"have {name} 2007 volume")
            continue
        body = _get(wayback_raw(ts, url), name)
        print(f"got  {name} 2007 volume: {check_body(body, 'pdf', where=name, pin_size=size)}")
        _save(dest, body)
    for code, ts, url, size in Q3:
        dest = q3_path(code)
        if os.path.exists(dest) and os.path.getsize(dest) == size:
            print(f"have Quadro 3 {code}")
            continue
        body = _get(wayback_raw(ts, url), f"Quadro 3 {code}")
        print(f"got  Quadro 3 {code}: {check_body(body, 'xlsx', where=code, pin_size=size)}")
        _save(dest, body)


def vol_path(code):
    return os.path.join(RAW07, f"isd2007_{code}.pdf")


def q3_path(code):
    return os.path.join(RAW17, f"q3_2017_{code}.xlsx")


# ---------------------------------------------------------------- read 2007


def _lines(page):
    """The page's words grouped into lines by vertical centre: [(ykey, [(x0, text), ...])]."""
    rows = defaultdict(list)
    for x0, y0, x1, y1, w, *_ in page.get_text("words"):
        rows[round((y0 + y1) / 2 / 3)].append((x0, w))
    return [(k, sorted(rows[k])) for k in sorted(rows)]


def read_volume(code):
    """One volume's Quadro 8.1 -> (caption, [(name, N, [8 shares])]), the Total row first."""
    import fitz

    _, name, _, _, pages, _, _ = next(v for v in VOLUMES if v[0] == code)
    doc = fitz.open(vol_path(code))
    tokens, caption, header_seen = [], "", False
    for pno in pages:
        lines = _lines(doc[pno - 1])
        start = 0
        if not header_seen:
            # The line that STARTS with `Distritos`: Sofala's commentary above the table has
            # "católica ... distritos" in one sentence.
            hdr = next((i for i, (_, ws) in enumerate(lines)
                        if fold(ws[0][1]) == "distritos"
                        and {"catolica", "outra"} <= {fold(w) for _, w in ws}), None)
            if hdr is None:
                raise SystemExit(f"{name}: no `Distritos ... Católica` header on page {pno}")
            cap = next((i for i, (_, ws) in enumerate(lines)
                        if fold(" ".join(w for _, w in ws)).startswith("quadro8")), None)
            if cap is None or cap > hdr:
                raise SystemExit(f"{name}: no Quadro 8 caption above the header on page {pno}")
            caption = " ".join(w for _, ws in lines[cap:cap + 2] for _, w in ws)
            check_header_order(name, lines, hdr)
            header_seen, start = True, hdr + 1
        for _, ws in lines[start:]:
            words = [w for _, w in ws]
            if "ORFANDADE" in words:
                break
            tokens.extend(words)
        else:
            continue
        break

    rows, counts, label, cur = [], [], [], None
    for t in tokens:
        if COUNT.match(t):
            counts.append(int(t.replace(",", "")))
        elif cur is not None:
            if not SHARE.match(t):
                raise SystemExit(f"{name}: {t!r} inside the shares of {' '.join(label)!r}")
            cur.append(float(t))
            if len(cur) == 8:
                rows.append((nfc(" ".join(label)), cur))
                label, cur = [], None
        elif t == "100.0" and label:
            cur = []
        elif PAGENO.match(t) and not label:
            continue                         # a page number between the two pages of a table
        elif SHARE.match(t):
            raise SystemExit(f"{name}: share {t!r} with no row open (last label {label!r})")
        else:
            label.append(t)
    if cur is not None or label:
        raise SystemExit(f"{name}: table ends inside a row: {label!r} {cur!r}")
    if len(counts) != len(rows):
        raise SystemExit(f"{name}: {len(rows)} rows and {len(counts)} N values")
    return caption, [(r[0], n, r[1]) for r, n in zip(rows, counts)]


HEADER_STEMS = [("Católica", ("catolica",)), ("Anglicana", ("anglican",)),
                ("Islâmica", ("islamica",)), ("Zione/Sião", ("zione", "siao")),
                ("Evangélica/Pentecostal", ("evangel",)), ("Sem religião", ("sem",)),
                ("Outra", ("outra",)), ("Desconhecida", ("descon", "desco", "desc")),
                ("N", ("n",))]


def check_header_order(name, lines, hdr):
    """The eight answers and N stand left to right in 2017's order. The header wraps over up to
    three lines (`Evangélica/P` over `entecostal`), so each stem's leftmost x is compared."""
    ykey = lines[hdr][0]
    words = [(x, fold(w)) for k, ws in lines if ykey - 15 <= k <= ykey for x, w in ws]
    xs = []
    for label, stems in HEADER_STEMS:
        hits = [x for x, f in words if any(f == s if s == "n" else f.startswith(s) for s in stems)]
        if not hits:
            raise SystemExit(f"{name}: no header word for {label} in {[f for _, f in words]}")
        xs.append(min(hits))
    if xs != sorted(xs):
        raise SystemExit(f"{name}: header columns out of 2017's order: "
                         f"{list(zip([l for l, _ in HEADER_STEMS], xs))}")


def table_digest(rows):
    h = hashlib.sha1()
    for nm, n, sh in rows:
        h.update(f"{nm}|{n}|{'|'.join(f'{s:.1f}' for s in sh)}\n".encode("utf-8"))
    return h.hexdigest()


def read_2007():
    return {code: read_volume(code) for code, *_ in VOLUMES}


# ---------------------------------------------------------------- read 2017


def read_q3(code):
    """2017 Quadro 3 -> (province total, {folded district key: total})."""
    import openpyxl

    wb = openpyxl.load_workbook(q3_path(code), data_only=True, read_only=True)
    rows = [tuple(r) for r in wb.worksheets[0].iter_rows(values_only=True)]
    wb.close()
    title = fold(" ".join(str(c) for c in rows[0] if c is not None))
    if not title.startswith("quadro3populacaoporidade"):
        raise SystemExit(f"Quadro 3 {code}: title is {title[:60]!r}")
    if not any(fold(r[1]) == "total" for r in rows[:6] if len(r) > 1):
        raise SystemExit(f"Quadro 3 {code}: no TOTAL in column 2 of the header")
    prov, out = None, {}
    for i, r in enumerate(rows):
        f = fold(r[0])
        if f == "total" and prov is None:
            prov = int(r[1])
        elif f.startswith(("cidade", "distrito")) and isinstance(r[1], (int, float)):
            # (the header, `DISTRITO, ÁREA DE RESIDÊNCIA E SEXO`, has no number beside it)
            h, m = rows[i + 1], rows[i + 2]
            if fold(h[0]) != "homens" or fold(m[0]) != "mulheres" or int(h[1]) + int(m[1]) != int(r[1]):
                raise SystemExit(f"Quadro 3 {code} {r[0]}: Homens + Mulheres is not the total")
            k = strip_prefix(f)
            k = Q3_ALIAS.get(k, k)
            if k in out:
                raise SystemExit(f"Quadro 3 {code}: {k} twice")
            out[k] = int(r[1])
    if prov is None:
        raise SystemExit(f"Quadro 3 {code}: no T O T A L row")
    gap = prov - sum(out.values())
    if gap != Q3_SLACK.get(code, 0):
        raise SystemExit(f"Quadro 3 {code}: districts sum to {sum(out.values()):,}, T O T A L "
                         f"{prov:,}, a gap of {gap:+,} where Q3_SLACK pins {Q3_SLACK.get(code, 0):+,}")
    if gap:
        print(f"  Quadro 3 {code}: districts sum {gap:+,} short of T O T A L, as pinned")
    return prov, out


def read_q11_provinces():
    """data/normalized/mz.csv (sources/mz.py) -> {province code: {category or Total: count}}."""
    out = defaultdict(dict)
    with open(NORM17, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out[r["geo_id"][2:]][r["source_category"]] = int(r["count"])
    return out


def read_posts():
    """COD-AB adm3 attributes: [(post pcode, post name, adm2 pcode, adm2 name, adm1 pcode)]."""
    import geopandas as gpd
    g = gpd.read_file(ADM3, engine="fiona", ignore_geometry=True)
    if len(g) == 0:
        raise SystemExit(f"{ADM3}: zero features")
    return list(zip(g["adm3_pcode"], g["adm3_name"], g["adm2_pcode"], g["adm2_name"],
                    g["adm1_pcode"]))


def read_adm2_names():
    import geopandas as gpd
    g = gpd.read_file(ADM3.replace("admin3", "admin2"), engine="fiona", ignore_geometry=True)
    return dict(zip(g["adm2_pcode"], g["adm2_name"]))


def post_units(posts):
    """{post pcode: 2007 geo_id} for every post in the nine provinces."""
    default = {}
    for code, dl in DISTRICTS.items():
        for _, pcodes in dl:
            for p in pcodes:
                if p in default:
                    raise SystemExit(f"{p} is listed under two 2007 districts")
                default[p] = pcodes[0]
    out = {}
    for pp, _, a2, _, a1 in posts:
        if a1[2:] not in DISTRICTS:
            continue
        if a2 not in default:
            raise SystemExit(f"COD district {a2} is in no 2007 district")
        out[pp] = POST_MOVES.get(pp, default[a2])
    unused = set(default) - {a2 for _, _, a2, _, _ in posts}
    if unused:
        raise SystemExit(f"DISTRICTS names pcodes COD-AB does not have: {sorted(unused)}")
    bad = set(POST_MOVES) - set(out)
    if bad:
        raise SystemExit(f"POST_MOVES names posts COD-AB does not have: {sorted(bad)}")
    return out


# ---------------------------------------------------------------- check


def say(ok, msg, fails):
    print(f"  {'OK ' if ok else 'BAD'} {msg}")
    if not ok:
        fails.append(msg)


def check_volume(code, caption, rows, fails):
    _, name, _, _, _, capkey, _ = next(v for v in VOLUMES if v[0] == code)
    print(f"\n{name}: Quadro 8.1, {len(rows) - 1} districts")
    say(capkey in fold(caption), f"caption names the province: {caption!r}", fails)
    say(rows[0][0] == "Total", f"first row is Total ({rows[0][0]!r})", fails)
    want = [n for n, _ in DISTRICTS[code]]
    got = [r[0] for r in rows[1:]]
    say(got == want, "district names and order are DISTRICTS'"
        + ("" if got == want else f": got {got}"), fails)
    worst = max(abs(sum(sh) - 100.0) for _, _, sh in rows)
    say(worst <= TOL_ROW, f"every row's eight shares sum to 100.0 within {TOL_ROW} (worst {worst:.1f})",
        fails)
    total_n, dist = rows[0][1], rows[1:]
    say(sum(n for _, n, _ in dist) == total_n,
        f"district N sum to the Total row's N ({sum(n for _, n, _ in dist):,} against {total_n:,})",
        fails)
    weighted = [sum(n * sh[j] for _, n, sh in dist) / sum(n for _, n, _ in dist) for j in range(8)]
    printed = rows[0][2]
    if code == "09":
        say(printed == GAZA_PRINTED, f"Gaza's Total row is the pinned misprint {GAZA_PRINTED}", fails)
        printed = GAZA_MEANT
    gap = max(abs(w - p) for w, p in zip(weighted, printed))
    say(gap <= TOL_WEIGHTED, f"N-weighted district shares match the Total row within {TOL_WEIGHTED} "
        f"(worst {gap:.3f}; weighted {[round(w, 2) for w in weighted]})", fails)
    d = table_digest(rows)
    pin = TABLE_DIGESTS.get(code)
    say(pin == d, f"table digest {d} is the pinned {pin}", fails)


def check_names(adm2_names, fails):
    print("\n2007 names against COD-AB's name for their first pcode")
    differ = {}
    for code, dl in DISTRICTS.items():
        for nm, pcodes in dl:
            cod = adm2_names[pcodes[0]]
            if key07(nm) != strip_prefix(fold(cod)):
                differ[nm] = cod
    say(differ == NAME_DIFFERS, "the names that differ are exactly NAME_DIFFERS"
        + ("" if differ == NAME_DIFFERS else f": {differ}"), fails)


# ---------------------------------------------------------------- fit


def clusters(code, posts, units, q3):
    """Groups of 2007 districts that cover the same ground as a group of 2017 districts.

    -> [(sorted 2007 geo_ids, 2017 total)]. Nodes are 2007 districts and COD-AB districts,
    joined by every post they share; each connected component is one group."""
    parent = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    adm2_key = {}
    for pp, _, a2, a2name, a1 in posts:
        if a1[2:] != code:
            continue
        parent[find("u" + units[pp])] = find("d" + a2)
        adm2_key[a2] = strip_prefix(fold(a2name))
    groups = defaultdict(lambda: ([], []))
    for node in list(parent):
        g = groups[find(node)]
        (g[0] if node[0] == "u" else g[1]).append(node[1:])
    keys = set(adm2_key.values())
    if set(q3) != keys:
        raise SystemExit(f"Quadro 3 {code} districts {sorted(set(q3) - keys)} are not COD-AB's; "
                         f"COD-AB has {sorted(keys - set(q3))} Quadro 3 does not")
    return sorted((sorted(u), sum(q3[adm2_key[a]] for a in d)) for u, d in groups.values())


def rake_groups(seed, cols, groups, tol=1e-11, iters=20_000):
    """IPF of a district x answer seed to answer totals and to GROUP totals of districts.

    `seed` {district: [8 floats]}, `cols` [8 targets], `groups` [(districts, target)]. The
    column step is sources/td.py's; the row step scales every district in a group by one
    factor, so a group's districts keep their seed proportions to each other."""
    m = {d: list(v) for d, v in seed.items()}
    for _ in range(iters):
        for j, t in enumerate(cols):
            s = sum(v[j] for v in m.values())
            for v in m.values():
                v[j] *= t / s
        worst = 0.0
        for ds, t in groups:
            s = sum(sum(m[d]) for d in ds)
            worst = max(worst, abs(s - t) / t)
            for d in ds:
                m[d] = [x * t / s for x in m[d]]
        cworst = max(abs(sum(v[j] for v in m.values()) - t) / t for j, t in enumerate(cols) if t)
        if worst < tol and cworst < tol:
            return m
    raise SystemExit("raking did not converge")


def integerise(m, cols, groups):
    """Integers with every group total and every answer total exact.

    District totals first, by largest remainder inside each group; then each district's cells by
    largest remainder; then single people moved between two answers inside one district, from
    the cell rounded furthest up to the one rounded furthest down, until the answer totals hold.
    A move never changes a district's total."""
    out = {}
    for ds, t in groups:
        real = {d: sum(m[d]) for d in ds}
        base = {d: int(real[d]) for d in ds}
        for d in sorted(ds, key=lambda d: -(real[d] - base[d]))[:t - sum(base.values())]:
            base[d] += 1
        for d in ds:
            cells = [int(x) for x in m[d]]
            for j in sorted(range(8), key=lambda j: -(m[d][j] - cells[j]))[:base[d] - sum(cells)]:
                cells[j] += 1
            out[d] = cells
    for _ in range(100_000):
        diff = [sum(v[j] for v in out.values()) - t for j, t in enumerate(cols)]
        if not any(diff):
            return out
        hi = next(j for j, x in enumerate(diff) if x > 0)
        lo = next(j for j, x in enumerate(diff) if x < 0)
        d = max((d for d in out if out[d][hi] > 0),
                key=lambda d: (out[d][hi] - m[d][hi]) + (m[d][lo] - out[d][lo]))
        out[d][hi] -= 1
        out[d][lo] += 1
    raise SystemExit("integerise did not close the answer totals")


def growth_witness(code, groups, n07, q11_total, fails):
    """2017 over 2007 population per group, over the province's own. A group outside
    GROWTH_BAND means a post moved between districts and POST_MOVES does not know it."""
    prov = q11_total / sum(n07.values())
    lo, hi = GROWTH_BAND
    print(f"  2017/2007 growth: province {prov:.2f}; per group, over the province's:")
    worst = []
    for ds, t in groups:
        g = t / sum(n07[d] for d in ds) / prov
        worst.append((g, ds))
        print(f"    {g:5.2f}  {'+'.join(ds)}  {sum(n07[d] for d in ds):>9,} -> {t:>9,}")
    bad = [(round(g, 2), ds) for g, ds in worst if not lo <= g <= hi]
    say(not bad, f"every group grew within {GROWTH_BAND} of the province" + (f": {bad}" if bad else ""),
        fails)


# ---------------------------------------------------------------- write


def write07(tables):
    rows = []
    for code, (_, table) in tables.items():
        ids = [p[0] for _, p in DISTRICTS[code]]
        for gid, (nm, n, sh) in zip(ids, table[1:]):
            rows.append([gid, LEVEL, nm, "Total", n, "self_id", 2007, "mz_rgph_2007",
                         "the printed N, not a religion category"])
            for c, s in zip(CATEGORIES, sh):
                rows.append([gid, LEVEL, nm, c, round(n * s / 100, 2), "self_id", 2007,
                             "mz_rgph_2007", "one-decimal share times the printed N"])
    _write(OUT07, rows)


def write_fitted(fitted, names):
    rows = []
    for gid in sorted(fitted):
        cells = fitted[gid]
        rows.append([gid, LEVEL, names[gid], "Total", sum(cells), "self_id", 2017,
                     "mz_rgph_2017_on_2007", "2017 population of this 2007 district, fitted"])
        for c, v in zip(CATEGORIES, cells):
            rows.append([gid, LEVEL, names[gid], c, v, "self_id", 2017, "mz_rgph_2017_on_2007",
                         "2007 Quadro 8.1 shape on 2017 Quadro 11 and Quadro 3 totals"])
    _write(OUT, rows)


def _write(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".part", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(COLUMNS)
        w.writerows(rows)
    os.replace(path + ".part", path)
    print(f"wrote {path}  {len(rows)} rows")


# ---------------------------------------------------------------- main


def main():
    if "--fetch" in sys.argv:
        fetch()
    fails = []
    tables = read_2007()
    for code, (caption, rows) in tables.items():
        check_volume(code, caption, rows, fails)
    posts = read_posts()
    check_names(read_adm2_names(), fails)
    units = post_units(posts)
    q11 = read_q11_provinces()
    q3 = {code: read_q3(code) for code, *_ in Q3}

    fitted, names = {}, {}
    for code, (_, table) in tables.items():
        name = next(v[1] for v in VOLUMES if v[0] == code)
        ids = [p[0] for _, p in DISTRICTS[code]]
        n07 = {gid: n for gid, (_, n, _) in zip(ids, table[1:])}
        names.update({gid: nm for gid, (nm, _, _) in zip(ids, table[1:])})
        seed = {gid: [n * s / 100 for s in sh] for gid, (_, n, sh) in zip(ids, table[1:])}
        cols = [q11[code][c] for c in CATEGORIES]
        print(f"\n{name}: fit")
        if code in q3:
            prov17, dist17 = q3[code]
            say(prov17 == q11[code]["Total"],
                f"Quadro 3 total {prov17:,} is Quadro 11's {q11[code]['Total']:,}", fails)
            groups = clusters(code, posts, units, dist17)
            # Q3_SLACK: where Quadro 3's districts fall short of its own total, scale the group
            # targets onto the province total and round by largest remainder (Tete: +2).
            short = q11[code]["Total"] - sum(t for _, t in groups)
            if short:
                real = [t * q11[code]["Total"] / (q11[code]["Total"] - short) for _, t in groups]
                ints = [int(x) for x in real]
                for i in sorted(range(len(real)), key=lambda i: -(real[i] - ints[i]))[
                        :q11[code]["Total"] - sum(ints)]:
                    ints[i] += 1
                print(f"  group targets moved by {[b - t for (_, t), b in zip(groups, ints) if b != t]} "
                      f"to reach Quadro 11's {q11[code]['Total']:,}")
                groups = [(ds, b) for (ds, _), b in zip(groups, ints)]
            merged = [g for g in groups if len(g[0]) > 1]
            print(f"  {len(groups)} groups for {len(ids)} 2007 districts; joined: "
                  + ("; ".join("+".join(ds) for ds, _ in merged) or "none"))
            growth_witness(code, groups, n07, q11[code]["Total"], fails)
        else:
            groups = [(ids, q11[code]["Total"])]
            print("  no 2017 district table: one group, the province")
        m = rake_groups(seed, cols, groups)
        ints = integerise(m, cols, groups)
        say(all(sum(v[j] for v in ints.values()) == t for j, t in enumerate(cols)),
            "every answer's fitted districts sum to its 2017 Quadro 11 count, to the person", fails)
        say(all(sum(sum(ints[d]) for d in ds) == t for ds, t in groups),
            "every group's fitted districts sum to its 2017 total, to the person", fails)
        moved = max(abs(ints[d][j] / sum(ints[d]) - seed[d][j] / sum(seed[d])) * 100
                    for d in ids for j in range(8))
        print(f"  largest change in one district's share of one answer, 2007 to fitted: "
              f"{moved:.1f} points")
        fitted.update(ints)

    if fails:
        raise SystemExit("\n".join(["\nFAILED:"] + fails))
    write07(tables)
    write_fitted(fitted, names)


if __name__ == "__main__":
    main()
