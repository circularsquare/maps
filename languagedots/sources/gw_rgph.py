"""Guinea-Bissau RGPH 2009, principal ethnic language ("principal dialecto falado") by etnia,
national -> data/normalized/gw.csv, and each language's split across the nine regiões through
the etnias that speak it -> data/normalized/gw_place.csv (a placement weight only).

    python sources/gw_rgph.py [--fetch]

SOURCE. INE Guiné-Bissau, RGPH 2009, *Características socioculturais* (92 pp), the volume
religiondots draws Guinea-Bissau's religion from (religiondots/sources/gw.md). Live at
https://www.stat-guinebissau.com/Menu_principal/IV_RGPH/rgph1/caracteristicas_socio_cultural.pdf
(and in the Wayback Machine; some captures there are truncated at 1 MiB, hence the digest).
--fetch copies religiondots' verified download (read-only) when it is there, else downloads it.

QUESTIONS (Anexo 2, the individual form; definitions on PDF p18):
  P.15 "Qual é o principal Dialecto falado?" One write-in answer, coded. A "dialecto" is, in
       the volume's own definition, "the means of communication of an etnia"; Crioulo,
       Portuguese and foreign languages are "línguas" and are asked in P.16, so P.15 can only
       name an ethnic language. "Sem dialecto" is a person who named none as principal (69% of
       people of no etnia, 6% overall): in practice mostly people whose main language is Kriol.
  P.16 "Fala Crioulo? Português? Francês? Inglês? Espanhol? Russo? outra?" yes/no each, so
       languages known, several allowed (Crioulo 90.4%, Portuguese 27.1%). NOT DRAWN: it never
       asks which is first, and P.15 is a single-answer table from the same census, which the
       brief prefers (AGENT_BRIEF §2). It names no ethnic language at all.
Universe: Guinean nationals in ordinary households, 1,442,227, all ages.

TABLES READ (1-based PDF pages):
  Anexo Quadro 4, pp73-74   etnia x principal dialecto, counts, 17 columns. DRAWN (its Total
                            row is the national count per language).
  Anexo Quadro 5, pp75-76   the same, men; Quadro 5A p77, women (its first page is missing
                            from the volume; the continuation is there). Check 5.
  Anexo Quadro 6, pp78-79   the same, urban; 6A pp80-81, rural. Check 4.
  Anexo Quadro 2, p71       região x etnia, counts. The placement weight (below).
  Quadro 2, p26             etnia x age group. Check 7 (what NA is).
  Gráfico 5, p33            % of each etnia whose principal dialect is its own. Check 6.

NA. Quadro 4's last column "NA", 131,640 (9.1%), is no answer recorded. It is 6.6-9.9% of
every etnia and 100% of the 1,274 with no etnia recorded, and its share by etnia follows the
share of children (check 7), so it is mostly infants who speak no language yet. Not drawn: it
goes in `gap`.

PLACEMENT. Language is published only nationally. Inside the country a language's speakers are
spread over the regiões through the etnias that name it: speakers of language d in região r =
sum over etnias e of [Quadro 4: e's people naming d] x [Anexo Quadro 2: e's share living in r].
That moves people only inside the unit the census counted them in (AGENT_BRIEF §4.4), and
the national count per language stays INE's. It assumes an etnia's language choice is the
same in every região (Quadros 6/6A show it is not quite the same in town and country, check 4's
note), so a language's regional split is an estimate. Anexo Quadro 2 has two misprints, each a
dropped digit, found by religiondots (religiondots/sources/gw.md §5): Fula in Oio 2,980 for
23,980 and Mandinga in Cacheu 1,460 for 11,460. Corrected here, and check 3 shows each row and
column then closes.

CHECKS (all must pass):
  1. the PDF is the pinned 92-page volume (SHA-1, size, %%EOF)
  2. Anexo Quadro 4 parsed off pp73-74 = the transcription, 17 rows x 17 columns
  3. Quadro 4: every etnia row and every column closes on its printed total; its etnia totals =
     Anexo Quadro 2's; Anexo Quadro 2 (corrected) closes on rows and columns
  4. urban (Quadro 6) + rural (6A) = Quadro 4, cell by cell on the first page's 9 columns for
     every etnia and on the Total row of the continuation
  5. men (Quadro 5) + women (5A) = Quadro 4 on the continuation's Total row; men alone on both
     pages close
  6. each etnia's own-language share in Quadro 4 = Gráfico 5's bars (one decimal)
  7. NA share by etnia against the share aged 0-14 (Quadro 2): positive correlation
  8. the 9 regiões = religiondots' gw_hexes units; the placement sums back to the national table
"""
import argparse
import base64
import hashlib
import re
import shutil
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from rdlink import RD_GEO  # noqa: E402

RAW = HERE / "data" / "raw" / "gw"
OUT = HERE / "data" / "normalized" / "gw.csv"
OUT_PLACE = HERE / "data" / "normalized" / "gw_place.csv"
NAME = "caracteristicas_socio_cultural.pdf"
PDF = RAW / NAME
RD_PDF = HERE.parent / "religiondots" / "data" / "raw" / "gw" / NAME
URL = "https://www.stat-guinebissau.com/Menu_principal/IV_RGPH/rgph1/" + NAME
WAYBACK = ("http://web.archive.org/web/20230314200710id_/https://www.stat-guinebissau.com/"
           "Menu_principal/IV_RGPH/rgph1/" + NAME)
SIZE = 2_725_536
SHA1_B32 = "3SHQDVYZG7WACAPWVML6UUN4ZZZSEKXH"   # as religiondots pins it
PAGES = 92
# 0-based pages
P_Q2_AGE, P_A2 = 25, 70
P_Q4 = (72, 73)
P_Q5M = (74, 75)
P_Q5F_CONT = 76
P_Q6U = (77, 78)
P_Q6R = (79, 80)

NATIONAL = 1_442_227
REGIONS = ["Tombali", "Quinara", "Oio", "Biombo", "Bolama/Bijagós", "Bafatá", "Gabú",
           "Cacheu", "SAB"]
ETNIAS = ["Sem Etnia", "Balanta", "Fula", "Mandinga", "Manjaco", "Mancanha", "Papel",
          "Bijagos", "Beafada", "Felupe", "Mansoanca", "Balanta Mane", "Nalu", "Sussu",
          "Saracule", "ND"]
# Quadro 4's columns after Total, in page order (p73 then p74)
COLS = ["Sem dialecto", "Balanta", "Fula", "Mancanha", "Mandinga", "Manjaco", "Bijagos",
        "Papel", "Beafada", "Felupe", "Mansoanca", "Balanta Mane", "Nalu", "Sosso",
        "Saracule", "NA"]
NOT_DRAWN = {"NA"}

# Anexo Quadro 4 as printed: etnia -> (Total, then COLS). Check 2 asserts the page says this.
Q4 = {
    "Total":        (1442227, 85356, 283835, 371077, 34991, 184220, 105526, 24090, 106969,
                     41040, 22432, 15186, 11887, 8150, 10278, 5550, 131640),
    "Sem Etnia":    (32098, 22207, 1179, 2857, 286, 1215, 473, 140, 417,
                     177, 101, 68, 17, 37, 452, 27, 2445),
    "Balanta":      (323948, 14482, 274348, 1462, 557, 733, 743, 159, 762,
                     167, 127, 497, 560, 179, 80, 11, 29081),
    "Fula":         (410560, 6760, 1427, 356584, 396, 2673, 391, 149, 308,
                     316, 124, 126, 43, 136, 703, 78, 40346),
    "Mandinga":     (212269, 6516, 915, 5956, 1425, 172619, 866, 146, 372,
                     735, 167, 113, 48, 100, 1272, 58, 20961),
    "Manjaco":      (119808, 5803, 742, 508, 626, 1902, 99515, 159, 573,
                     65, 271, 73, 26, 35, 41, 23, 9446),
    "Mancanha":     (44829, 6515, 748, 363, 30915, 770, 2000, 83, 258,
                     73, 42, 41, 17, 12, 14, 3, 2975),
    "Papel":        (130651, 11331, 1754, 808, 376, 524, 665, 797, 103117,
                     120, 118, 54, 17, 36, 39, 4, 10891),
    "Bijagos":      (30294, 3580, 201, 238, 63, 168, 157, 22287, 824,
                     60, 49, 13, 5, 32, 35, 4, 2578),
    "Beafada":      (50543, 2961, 274, 746, 101, 1740, 318, 99, 91,
                     38980, 58, 24, 11, 56, 506, 2, 4576),
    "Felupe":       (24892, 809, 91, 142, 94, 145, 235, 21, 111,
                     26, 21346, 8, 7, 3, 41, 1, 1812),
    "Mansoanca":    (20456, 2358, 1187, 250, 54, 462, 81, 10, 75,
                     18, 10, 14115, 39, 14, 20, 7, 1756),
    "Balanta Mane": (14460, 428, 814, 87, 20, 642, 28, 1, 9,
                     4, 10, 20, 11059, 7, 1, 3, 1327),
    "Nalu":         (13420, 880, 69, 251, 24, 92, 11, 17, 20,
                     263, 1, 9, 19, 7409, 3251, 8, 1096),
    "Sussu":        (5318, 259, 57, 390, 45, 161, 17, 10, 23,
                     25, 2, 5, 3, 86, 3788, 7, 440),
    "Saracule":     (7407, 467, 29, 435, 9, 374, 26, 12, 9,
                     11, 6, 20, 16, 8, 35, 5314, 636),
    "ND":           (1274, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 1274),
}
# each etnia's own column in COLS (Gráfico 5's bars, check 6)
OWN = {"Balanta": "Balanta", "Fula": "Fula", "Mandinga": "Mandinga", "Manjaco": "Manjaco",
       "Mancanha": "Mancanha", "Papel": "Papel", "Bijagos": "Bijagos", "Beafada": "Beafada",
       "Felupe": "Felupe", "Mansoanca": "Mansoanca", "Balanta Mane": "Balanta Mane",
       "Nalu": "Nalu", "Sussu": "Sosso", "Saracule": "Saracule"}
GRAFICO5 = [86.9, 85.8, 84.7, 83.1, 81.3, 78.9, 77.1, 76.5, 73.6, 71.7, 71.2, 69.2, 69.0,
            69.0, 55.2]   # as printed, highest first (bar labels; one of them is Sem Etnia's
#                           69.2% "sem dialecto", per the p34 prose)

# Quadros 6 / 6A first page (Total + the first 8 COLS), urban and rural; continuation Totals.
URBAN_P1 = {
    "Total": (570048, 67417, 88290, 129848, 28011, 67155, 40030, 7418, 53623),
    "Sem Etnia": (23231, 16569, 748, 1899, 253, 874, 354, 100, 362),
    "Balanta": (106046, 11266, 83010, 814, 415, 385, 461, 106, 485),
    "Fula": (142638, 5539, 763, 121930, 274, 1399, 243, 72, 209),
    "Mandinga": (77809, 5870, 427, 2531, 751, 60438, 561, 81, 202),
    "Manjaco": (47254, 5029, 475, 294, 464, 1043, 36096, 118, 392),
    "Mancanha": (36674, 6037, 532, 271, 25271, 585, 1258, 55, 202),
    "Papel": (70078, 10395, 1189, 536, 310, 319, 547, 369, 51167),
    "Bijagos": (9138, 1317, 113, 93, 39, 71, 99, 6423, 370),
    "Beafada": (21627, 2191, 172, 490, 76, 1057, 150, 50, 63),
    "Felupe": (10874, 610, 65, 94, 34, 110, 136, 13, 84),
    "Mansoanca": (9413, 1297, 459, 195, 43, 206, 58, 8, 55),
    "Balanta Mane": (3199, 354, 248, 59, 13, 163, 18, 1, 7),
    "Nalu": (3199, 345, 36, 142, 19, 72, 10, 7, 6),
    "Sussu": (2343, 198, 29, 233, 40, 121, 16, 6, 11),
    "Saracule": (5568, 400, 24, 267, 9, 312, 23, 9, 8),
    "ND": (957, 0, 0, 0, 0, 0, 0, 0, 0),
}
RURAL_P1 = {
    "Total": (872179, 17939, 195545, 241229, 6980, 117065, 65496, 16672, 53346),
    "Sem Etnia": (8867, 5638, 431, 958, 33, 341, 119, 40, 55),
    "Balanta": (217902, 3216, 191338, 648, 142, 348, 282, 53, 277),
    "Fula": (267922, 1221, 664, 234654, 122, 1274, 148, 77, 99),
    "Mandinga": (134460, 646, 488, 3425, 674, 112181, 305, 65, 170),
    "Manjaco": (72554, 774, 267, 214, 162, 859, 63419, 41, 181),
    "Mancanha": (8155, 478, 216, 92, 5644, 185, 742, 28, 56),
    "Papel": (60573, 936, 565, 272, 66, 205, 118, 428, 51950),
    "Bijagos": (21156, 2263, 88, 145, 24, 97, 58, 15864, 454),
    "Beafada": (28916, 770, 102, 256, 25, 683, 168, 49, 28),
    "Felupe": (14018, 199, 26, 48, 60, 35, 99, 8, 27),
    "Mansoanca": (11043, 1061, 728, 55, 11, 256, 23, 2, 20),
    "Balanta Mane": (11261, 74, 566, 28, 7, 479, 10, 0, 2),
    "Nalu": (10221, 535, 33, 109, 5, 20, 1, 10, 14),
    "Sussu": (2975, 61, 28, 157, 5, 40, 1, 4, 12),
    "Saracule": (1839, 67, 5, 168, 0, 62, 3, 3, 1),
    "ND": (317, 0, 0, 0, 0, 0, 0, 0, 0),
}
URBAN_P2_TOTAL = (16851, 9520, 6912, 2352, 2199, 2582, 4184, 43656)
RURAL_P2_TOTAL = (24189, 12912, 8274, 9535, 5951, 7696, 1366, 87984)
MEN_P1_TOTAL = (698119, 43840, 133754, 182767, 16348, 89085, 48481, 11246, 50089)
MEN_P2_TOTAL = (19975, 11300, 7344, 5691, 4042, 5239, 2674, 66244)
WOMEN_P2_TOTAL = (21065, 11132, 7842, 6196, 4108, 5039, 2876, 65396)

# Anexo Quadro 2, região x etnia (REGIONS order), with religiondots' two corrections.
A2 = {
    "Sem Etnia":    (1215, 505, 1097, 612, 602, 2909, 3237, 3783, 18138),
    "Balanta":      (42276, 21329, 93737, 17983, 1403, 16094, 3734, 53072, 74320),
    "Fula":         (18861, 4788, 23980, 4065, 1154, 120183, 162970, 9142, 65417),
    "Mandinga":     (4386, 3041, 70739, 1503, 1462, 45818, 29017, 11460, 44843),
    "Manjaco":      (1042, 1650, 6561, 2487, 893, 4498, 916, 67726, 34035),
    "Mancanha":     (280, 749, 1391, 2539, 1676, 788, 452, 8304, 28650),
    "Papel":        (1437, 2088, 1587, 59995, 1805, 2191, 526, 4206, 56816),
    "Bijagos":      (1165, 1491, 147, 849, 20670, 202, 395, 260, 5115),
    "Beafada":      (5156, 22231, 2186, 451, 2004, 2567, 613, 744, 14591),
    "Felupe":       (58, 63, 158, 930, 110, 145, 55, 16713, 6660),
    "Mansoanca":    (711, 2215, 8029, 708, 60, 1878, 904, 1049, 4902),
    "Balanta Mane": (148, 17, 4530, 151, 5, 555, 153, 7064, 1837),
    "Nalu":         (10498, 218, 55, 149, 69, 159, 253, 109, 1910),
    "Sussu":        (2700, 175, 112, 60, 122, 257, 206, 106, 1580),
    "Saracule":     (128, 47, 437, 75, 55, 1920, 1331, 286, 3128),
    "ND":           (67, 17, 45, 108, 50, 78, 52, 100, 757),
}
A2_PRINTED = {("Fula", 2): 2980, ("Mandinga", 7): 1460}
A2_REGION_TOTAL = (90128, 60624, 214791, 92665, 32140, 200242, 204814, 184124, 362699)

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"}
NUM = re.compile(r"^\d+(,\d+)?$")


def say(ok, msg):
    print(("  ok  " if ok else "  FAIL") + "  " + msg)
    if not ok:
        raise SystemExit("check failed: " + msg)


def sha1_b32(body):
    return base64.b32encode(hashlib.sha1(body).digest()).decode()


def fetch():
    RAW.mkdir(parents=True, exist_ok=True)
    if PDF.exists() and PDF.stat().st_size == SIZE:
        print("already have", PDF)
        return
    if RD_PDF.exists() and RD_PDF.stat().st_size == SIZE:
        shutil.copyfile(RD_PDF, PDF)
        print(f"copied religiondots' verified download -> {PDF}")
        return
    for url in (URL, WAYBACK):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA),
                                        timeout=600) as r:
                body = r.read()
        except Exception as e:  # noqa: BLE001
            print(f"  {url}: {e}")
            continue
        if sha1_b32(body) == SHA1_B32:
            PDF.write_bytes(body)
            print(f"wrote {PDF} ({len(body):,} bytes)")
            return
        print(f"  {url}: not the pinned file ({len(body):,} bytes)")
    raise SystemExit("could not fetch the RGPH 2009 socio-cultural volume")


def lines_of(doc, pno):
    return [ln.strip() for ln in doc.load_page(pno).get_text().splitlines() if ln.strip()]


def table_rows(doc, pno):
    """label -> counts (every other number after the label), from the last header '%' on."""
    lines = lines_of(doc, pno)
    last = max(i for i, ln in enumerate(lines) if ln == "%")
    out, label, nums = {}, None, []
    for ln in lines[last + 1:]:
        if NUM.match(ln):
            nums.append(ln)
        else:
            if label is not None:
                out[label] = nums
            label, nums = ln, []
    if label is not None:
        out[label] = nums
    return {k: tuple(int(x) for x in v[0::2]) for k, v in out.items()
            if v and all("," not in x for x in v[0::2])}


def ints_on(doc, pno):
    return [int(ln) for ln in lines_of(doc, pno) if re.fullmatch(r"\d+", ln)]


def is_subsequence(seq, of):
    it = iter(of)
    return all(any(x == y for y in it) for x in seq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    a = ap.parse_args()
    if a.fetch:
        fetch()

    import fitz
    body = PDF.read_bytes()
    doc = fitz.open(PDF)
    say(len(body) == SIZE and sha1_b32(body) == SHA1_B32 and body.rstrip().endswith(b"%%EOF")
        and doc.page_count == PAGES,
        f"1. {NAME}: {len(body):,} bytes, SHA-1 pinned, %%EOF, {doc.page_count} pages")

    # 2. Quadro 4 = the transcription
    p1, p2 = table_rows(doc, P_Q4[0]), table_rows(doc, P_Q4[1])
    parsed = {e: p1[e] + p2[e] for e in Q4}
    say(parsed == Q4 and all(len(v) == 17 for v in Q4.values()),
        f"2. Anexo Quadro 4 parsed off pp73-74 = the transcription ({len(Q4)} rows x 17)")

    # 3. closure
    rows_close = all(v[0] == sum(v[1:]) for v in Q4.values())
    cols_close = all(Q4["Total"][j] == sum(Q4[e][j] for e in ETNIAS) for j in range(17))
    say(rows_close and cols_close and Q4["Total"][0] == NATIONAL,
        f"3a. Quadro 4: every etnia row and every column closes; total {NATIONAL:,}")
    say(all(sum(A2[e]) == Q4[e][0] for e in ETNIAS),
        "3b. Anexo Quadro 2 (corrected) etnia totals = Quadro 4's")
    say(all(sum(A2[e][r] for e in ETNIAS) == A2_REGION_TOTAL[r] for r in range(9))
        and sum(A2_REGION_TOTAL) == NATIONAL,
        "3c. Anexo Quadro 2 (corrected) closes on every região total")
    a2_ints = ints_on(doc, P_A2)
    printed = {k: v for k, v in A2.items()}
    for (e, r), v in A2_PRINTED.items():
        printed[e] = printed[e][:r] + (v,) + printed[e][r + 1:]
    seq = [x for e in ETNIAS for x in (Q4[e][0],) + printed[e]]
    say(is_subsequence(seq, a2_ints),
        "3d. Anexo Quadro 2 as printed (Fula/Oio 2,980, Mandinga/Cacheu 1,460) is on p71")

    # 4. urban + rural
    u1, r1 = table_rows(doc, P_Q6U[0]), table_rows(doc, P_Q6R[0])
    say(all(u1.get(e) == URBAN_P1[e] for e in URBAN_P1)
        and all(r1.get(e if e != "Sussu" else "Sosso") == RURAL_P1[e] for e in RURAL_P1),
        "4a. Quadros 6 and 6A first pages parsed = the transcription")
    say(all(URBAN_P1[e][j] + RURAL_P1[e][j] == Q4[e][j] for e in Q4 for j in range(9)),
        "4b. urban + rural = Quadro 4, every cell of the first page (17 rows x 9)")
    say(is_subsequence(URBAN_P2_TOTAL, ints_on(doc, P_Q6U[1]))
        and is_subsequence(RURAL_P2_TOTAL, ints_on(doc, P_Q6R[1]))
        and all(URBAN_P2_TOTAL[j] + RURAL_P2_TOTAL[j] == Q4["Total"][9 + j] for j in range(8)),
        "4c. urban + rural = Quadro 4 on the continuation's Total row")

    # 5. men + women
    m1 = table_rows(doc, P_Q5M[0])
    say(m1["Total"] == MEN_P1_TOTAL
        and is_subsequence(MEN_P2_TOTAL, ints_on(doc, P_Q5M[1]))
        and is_subsequence(WOMEN_P2_TOTAL, ints_on(doc, P_Q5F_CONT))
        and all(MEN_P2_TOTAL[j] + WOMEN_P2_TOTAL[j] == Q4["Total"][9 + j] for j in range(8))
        and MEN_P1_TOTAL[0] == sum(MEN_P1_TOTAL[1:]) + sum(MEN_P2_TOTAL),
        "5. men + women = Quadro 4 on the continuation's Total row; men close on both pages "
        f"({MEN_P1_TOTAL[0]:,})")

    # 6. Gráfico 5
    own = sorted((round(100 * Q4[e][1 + COLS.index(c)] / Q4[e][0], 1) for e, c in OWN.items()),
                 reverse=True)
    sem = round(100 * Q4["Sem Etnia"][1] / Q4["Sem Etnia"][0], 1)
    say(sorted(own + [sem], reverse=True) == GRAFICO5,
        "6. own-language share per etnia (and Sem Etnia's 'sem dialecto') = Gráfico 5's 15 bars")

    # 7. NA against children
    age = table_rows(doc, P_Q2_AGE)
    kids, na = [], []
    for e in ETNIAS[:-1]:
        key = e if e in age else {"Bijagos": "Bijagós"}.get(e, e)
        if key not in age:
            continue
        kids.append(age[key][1] / age[key][0])
        na.append(Q4[e][16] / Q4[e][0])
    r = float(np.corrcoef(kids, na)[0, 1])
    say(len(kids) >= 12 and r > 0.5,
        f"7. NA share vs share aged 0-14 over {len(kids)} etnias: r = {r:+.2f} "
        f"(NA {min(na):.1%}-{max(na):.1%}; 0-14 {min(kids):.1%}-{max(kids):.1%})")

    # 8. placement and join
    import geopandas as gpd
    units = set(gpd.read_file(RD_GEO / "gw" / "gw_hexes.gpkg", ignore_geometry=True)["unit"])
    say(units == set(REGIONS), f"8a. the 9 regiões = religiondots' gw_hexes units")
    # SAB (Bissau) is all urban: its people of etnia e take e's URBAN language rates (Quadro 6,
    # first page's 8 columns; the rest of e's urban people split as e's continuation columns
    # split nationally). Everything else of e goes to the other 8 regiões by Anexo Quadro 2.
    sab = REGIONS.index("SAB")
    say(all(A2[e][sab] <= URBAN_P1[e][0] for e in ETNIAS),
        "8c. every etnia's SAB population fits inside its urban population")
    alloc = {}   # (etnia, col index) -> SAB people
    for e in ETNIAS:
        u = URBAN_P1[e]
        if not u[0]:
            continue
        rest_u = u[0] - sum(u[1:])
        rest_n = sum(Q4[e][9:])
        for j in range(16):
            rate = (u[1 + j] / u[0] if j < 8 else
                    (rest_u / u[0]) * (Q4[e][1 + j] / rest_n if rest_n else 0.0))
            alloc[e, j] = A2[e][sab] * rate
    say(all(alloc[k] <= Q4[k[0]][1 + k[1]] + 1e-6 for k in alloc),
        "8d. no language's SAB share exceeds its national count in any etnia")
    rows = []
    for j, lang in enumerate(COLS):
        if lang in NOT_DRAWN:
            continue
        for ri, reg in enumerate(REGIONS):
            v = 0.0
            for e in ETNIAS:
                if not Q4[e][0]:
                    continue
                s = alloc.get((e, j), 0.0)
                if ri == sab:
                    v += s
                else:
                    outside = Q4[e][0] - A2[e][sab]
                    if outside:
                        v += (Q4[e][1 + j] - s) * A2[e][ri] / outside
            rows.append(dict(region=reg, source_category=lang, people=v))
    place = pd.DataFrame(rows)
    back = place.groupby("source_category")["people"].sum()
    say(all(abs(back[c] - Q4["Total"][1 + COLS.index(c)]) < 0.5 for c in back.index),
        "8b. each language's regional split sums back to its national count")

    out = [dict(geo_id="Guinea-Bissau", geo_level="country", geo_name="Guinea-Bissau",
                source_category=c, count=Q4["Total"][1 + j], tier="measured")
           for j, c in enumerate(COLS) if c not in NOT_DRAWN]
    df = pd.DataFrame(out)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, encoding="utf-8")
    print(f"wrote {OUT}: {len(df)} rows, {df['count'].sum():,} people "
          f"(NA {Q4['Total'][16]:,} not drawn)")
    place.to_csv(OUT_PLACE, index=False, encoding="utf-8")
    print(f"wrote {OUT_PLACE}: {len(place)} rows")
    # what the placement gives: top languages per região
    piv = place.pivot(index="region", columns="source_category", values="people")
    for reg in REGIONS:
        s = piv.loc[reg].sort_values(ascending=False)
        tot = s.sum()
        print(f"  {reg:15s} {tot:9,.0f}  " + ", ".join(
            f"{k} {100 * v / tot:.0f}" for k, v in s.items() if v / tot >= 0.03))


if __name__ == "__main__":
    main()
