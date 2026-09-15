"""Guinea-Bissau — RGPH 2009 (Terceiro RGPH), religion by região, Guinean nationals.

Reads (or fetches) data/raw/gw/caracteristicas_socio_cultural.pdf and writes
data/normalized/gw.csv. `sources/gw.md` is the write-up; `sources/gw_geo.py` builds the
polygons and the grid.

## §11w READ THE REGIONAL BOOKLETS; THE TABLE IS IN THE SOCIO-CULTURAL VOLUME'S ANNEX

§11w downloaded the nine per-region RGPH 2009 booklets, found no religion table in any of
them, and recorded that "the one analytical volume that mentions religion does so in
passing". That volume is INE's *Terceiro RGPH 2009: Características socioculturais* (92 pp,
Word 2007 export dated 2013-01-03), and its religion section is not a passing mention: it
has four religion tables, and the annex prints the région table in counts.

    Quadro 3 (body)   religion x sex, counts                     PDF p29
    Quadro 4 (body)   região x religion, shares, one decimal     PDF p30
    Quadro 5 (body)   etnia x religion, counts                   PDF p32
    Anexo Quadro 3    região x religion, counts and shares       PDF p72  <- drawn
    Anexo Quadro 7    the same table again, SAB column in full   PDF p82

Anexo Quadro 3 is drawn, and Quadro 7 is its duplicate: the page that carries Quadro 3 cuts
off the SAB percentage column at the right margin ("7,", "34", "40"), Quadro 7 prints it
whole, and the counts on the two pages are identical. **The drawn numbers are counts, exact:
no share is multiplied by anything.** All six columns equal UNSD table 28's Guinea-Bissau
2009 row to the person, ND being UNSD's `Unknown`.

## THE PROSE DISAGREES WITH THE TABLES, AND THE ANNEX WINS

The body text (PDF p30) says Gabú and Bafatá are *"77,1% e 86,5% respectivamente"* Muslim and
Oio *"47,1%"*. Every table, body Quadro 4 included, has Bafatá 77.1, Gabú 86.5 and Oio 42.1,
and 90,341 / 214,791 is 42.06%. The sources.md §11aq scout attributed the swap to Quadro 4;
it is in the sentences above Quadro 4, which itself agrees with the annex. `check()`
asserts Quadro 4 equal to Quadro 7 cell for cell.

A witness that does not come from the same tabulation settles which way round Bafatá and
Gabú are: Anexo Quadro 2 (região x etnia, counts) crossed with Quadro 5's national religion
rates per etnia predicts each região's religion from its ethnic mix alone. Gabú is 79.6% Fula,
Bafatá 60.0% Fula and 22.9% Mandinga, so the prediction puts Gabú above Bafatá on Muslim share,
as the annex does. `ethnic_witness()` asserts that and prints the whole comparison.

## THE QUESTIONNAIRE IS A WRITE-IN (ANEXO 2)

P.14, asked of *todos os recenseados*, is *"Qual é a sua Religião?"* with a blank line and a
two-digit code box: no printed answer list, so no no-religion box that names animism (the
Mozambique ruling, spec §6.3a) and no card whose column order could be swapped (Zambia,
§9db). `Sem religião` is people who said they had none; `Animista` is people who named
animism. The report adds (PDF p28) that *"existe uma percentagem significativa da população
que pratica duas religiões, o que não foi contemplado no estudo"*: one answer per person.

## THE UNIVERSE, AND WHO IS OUTSIDE IT

The table is **Guinean nationals living in ordinary households, as enumerated**, 1,442,227.

    enumerated in ordinary households     1,449,230   Quadro 1 (PDF p20), and PDF p3
      Guinean nationals                   1,442,227   <- the religion table
      foreign nationals                       1,933   Quadro 1; Anexo Quadro 1 by country
      nationality not recorded                5,070   the remainder, not printed as a row
    enumerated in collective households       3,696   PDF p3 ("orfanatos e casas religiosas")
    enumerated, all                       1,452,926
    post-enumeration survey omission 4.6%    +68,  ->  1,520,830, the headline figure

The counts are NOT corrected for the 4.6% omission. The note on PDF p3 recommends applying
its per-région weights for population totals, and says this report itself used the
uncorrected population; the correction is a per-région scalar, so it would change dot counts
by 3.8% (Cacheu) to 6.1% (SAB) and no share. Drawn as printed, so that every count still
equals the office's table and UNSD's.

ND, 228,718 or 15.9% of nationals, is described on PDF p28 as people who did not answer the
question (*"não responderam à esta pergunta"*). It is not drawn (spec §3.5) and is `gap`.

Usage:
    python sources/gw.py --fetch    one 2.7 MB PDF from stat-guinebissau.com (Wayback fallback)
    python sources/gw.py            normalise from data/raw/gw/
"""

import csv
import os
import re
import sys
import unicodedata

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "gw")
OUT = os.path.join(ROOT, "data", "normalized", "gw.csv")
sys.path.insert(0, os.path.join(ROOT, "tools"))

from fetch_checks import FetchCheckError, check_body, digest   # noqa: E402  shared, not copied

SOURCE_ID = "gw_rgph2009_socioculturais"
YEAR = 2009
BASIS = "self_id"
COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

URL = ("https://www.stat-guinebissau.com/Menu_principal/IV_RGPH/rgph1/"
       "caracteristicas_socio_cultural.pdf")
# The 2023-03-14 capture. Wayback also holds captures of this URL with digest UXLWBN4E...:
# those are the first 1,048,576 bytes of the same file, a truncated capture that still opens
# as 92 pages in MuPDF. Hence the trailer check and the digest check, not just `%PDF-`.
WAYBACK = ("http://web.archive.org/web/20230314200710id_/https://www.stat-guinebissau.com/"
           "Menu_principal/IV_RGPH/rgph1/caracteristicas_socio_cultural.pdf")
PDF = os.path.join(RAW, "caracteristicas_socio_cultural.pdf")
SIZE = 2_725_536
DIGEST = "3SHQDVYZG7WACAPWVML6UUN4ZZZSEKXH"   # SHA-1, base32, as the Wayback CDX writes it
PAGES = 92

# 0-based page indices.
PAGE_PES = 2         # PDF p3, the omission note: household and collective residents by região
PAGE_Q1 = 19         # PDF p20, Quadro 1, nationality
PAGE_Q1A = 21        # PDF p22, Quadro 1A, nationals by região and sex
PAGE_Q4 = 29         # PDF p30, Quadro 4, região x religion, shares
PAGE_A2 = 70         # PDF p71, Anexo Quadro 2, região x etnia, counts
PAGE_A3 = 71         # PDF p72, Anexo Quadro 3, região x religion, counts (drawn)
PAGE_A7 = 81         # PDF p82, Anexo Quadro 7, the same, SAB shares in full

REGIONS = ["Tombali", "Quinara", "Oio", "Biombo", "Bolama/Bijagós", "Bafatá", "Gabú",
           "Cacheu", "SAB"]
NAMES = {"SAB": "Sector Autónomo de Bissau"}
CATS = ["Animista", "Muçulmana", "Cristão", "Outra religião", "Sem religião", "ND"]

# Anexo Quadro 3, transcribed, and asserted equal to what is parsed off pages 72 and 82.
NATIONAL = {"Total": 1_442_227, "Animista": 215_130, "Muçulmana": 650_402,
            "Cristão": 318_021, "Outra religião": 414, "Sem religião": 29_542,
            "ND": 228_718}
A3 = {   # REGIONS order
    "Total":          (90128, 60624, 214791, 92665, 32140, 200242, 204814, 184124, 362699),
    "Animista":       (21748, 3767, 44683, 37193, 7897, 7814, 654, 62670, 28704),
    "Muçulmana":      (38722, 27753, 90341, 5828, 4773, 154402, 177265, 27218, 124100),
    "Cristão":        (13225, 11752, 33904, 27997, 9864, 13670, 5361, 56597, 145651),
    "Outra religião": (44, 2, 26, 6, 1, 6, 5, 279, 45),
    "Sem religião":   (369, 4289, 1829, 2346, 1340, 1739, 120, 5507, 12003),
    "ND":             (16020, 13061, 44008, 19295, 8265, 22611, 21409, 31853, 52196),
}

# PDF p3: residents of ordinary and collective households by região, as enumerated.
HOUSEHOLD = (91089, 60777, 215259, 93039, 32424, 200884, 205608, 185053, 365097)
COLLECTIVE = (222, 205, 838, 245, 746, 195, 316, 378, 551)
HOUSEHOLD_TOTAL, COLLECTIVE_TOTAL = 1_449_230, 3_696
FOREIGN = 1_933                                   # Quadro 1

# UNSD's labels -> the annex's rows.
UNSD = {"Animist": "Animista", "Muslim": "Muçulmana", "Christian": "Cristão",
        "Other": "Outra religião", "No Religion": "Sem religião", "Unknown": "ND"}

# The ethnic witness. Anexo Quadro 2 (região x etnia, counts, REGIONS order) and Quadro 5
# (etnia x religion, counts, CATS order). Transcribed; the checks are that every row and
# column closes on the printed totals, which a slipped digit cannot survive.
ETNIAS = ["Sem etnia", "Balanta", "Fula", "Mandinga", "Manjaco", "Mancanha", "Papel",
          "Bijagó", "Beafada", "Felupe", "Mansoanca", "Balanta Mane", "Nalu", "Sosso",
          "Saracule", "ND"]
A2 = {
    "Sem etnia":    (1215, 505, 1097, 612, 602, 2909, 3237, 3783, 18138),
    "Balanta":      (42276, 21329, 93737, 17983, 1403, 16094, 3734, 53072, 74320),
    "Fula":         (18861, 4788, 2980, 4065, 1154, 120183, 162970, 9142, 65417),
    "Mandinga":     (4386, 3041, 70739, 1503, 1462, 45818, 29017, 1460, 44843),
    "Manjaco":      (1042, 1650, 6561, 2487, 893, 4498, 916, 67726, 34035),
    "Mancanha":     (280, 749, 1391, 2539, 1676, 788, 452, 8304, 28650),
    "Papel":        (1437, 2088, 1587, 59995, 1805, 2191, 526, 4206, 56816),
    "Bijagó":       (1165, 1491, 147, 849, 20670, 202, 395, 260, 5115),
    "Beafada":      (5156, 22231, 2186, 451, 2004, 2567, 613, 744, 14591),
    "Felupe":       (58, 63, 158, 930, 110, 145, 55, 16713, 6660),
    "Mansoanca":    (711, 2215, 8029, 708, 60, 1878, 904, 1049, 4902),
    "Balanta Mane": (148, 17, 4530, 151, 5, 555, 153, 7064, 1837),
    "Nalu":         (10498, 218, 55, 149, 69, 159, 253, 109, 1910),
    "Sosso":        (2700, 175, 112, 60, 122, 257, 206, 106, 1580),
    "Saracule":     (128, 47, 437, 75, 55, 1920, 1331, 286, 3128),
    "ND":           (67, 17, 45, 108, 50, 78, 52, 100, 757),
}
A2_TOTALS = {"Sem etnia": 32098, "Balanta": 323948, "Fula": 410560, "Mandinga": 212269,
             "Manjaco": 119808, "Mancanha": 44829, "Papel": 130651, "Bijagó": 30294,
             "Beafada": 50543, "Felupe": 24892, "Mansoanca": 20456, "Balanta Mane": 14460,
             "Nalu": 13420, "Sosso": 5318, "Saracule": 7407, "ND": 1274}
Q5 = {
    "Sem etnia":    (1164, 13058, 13019, 6, 494, 4357),
    "Balanta":      (104022, 5800, 116870, 74, 14686, 82497),
    "Fula":         (1276, 361389, 3046, 9, 758, 44081),
    "Mandinga":     (1187, 184547, 3339, 20, 349, 22827),
    "Manjaco":      (39874, 6617, 53239, 266, 2680, 17132),
    "Mancanha":     (6058, 1096, 29468, 2, 1716, 6489),
    "Papel":        (40452, 2148, 57519, 10, 4714, 25808),
    "Bijagó":       (7722, 1098, 12964, 2, 1234, 7274),
    "Beafada":      (350, 43055, 1779, 12, 131, 5216),
    "Felupe":       (6265, 1083, 11189, 3, 2132, 4220),
    "Mansoanca":    (4842, 3194, 8108, 7, 411, 3894),
    "Balanta Mane": (1411, 5272, 5884, 0, 116, 1777),
    "Nalu":         (381, 11042, 553, 1, 58, 1385),
    "Sosso":        (67, 4442, 203, 0, 25, 581),
    "Saracule":     (33, 6374, 248, 2, 17, 733),
    "ND":           (26, 187, 593, 0, 21, 447),
}
Q5_TOTALS = dict(A2_TOTALS, Balanta=323949, Fula=410559)   # Quadro 5 prints these one apart

LABELS = [("Total", r"TOTAL"), ("Animista", r"Animista"), ("Muçulmana", r"Muçulmana"),
          ("Cristão", r"Crist[ãa]o"), ("Outra religião", r"Outra\s+[Rr]eligi[ãa]o"),
          ("Sem religião", r"Sem\s+[Rr]eligi[ãa]o"), ("ND", r"\bND\b")]
SPACES = dict.fromkeys([0x00A0, 0x2007, 0x2008, 0x2009, 0x202F, 0x205F], " ")


def despace(s):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(s)).translate(SPACES)).strip()


def fetch():
    import urllib.request

    os.makedirs(RAW, exist_ok=True)
    if os.path.exists(PDF) and os.path.getsize(PDF) == SIZE:
        print("already have", PDF)
        return
    ua = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/128.0 Safari/537.36"}
    for url in (URL, WAYBACK):
        try:
            req = urllib.request.Request(url, headers=ua)
            with urllib.request.urlopen(req, timeout=600) as r:
                body = r.read()
        except Exception as e:                       # noqa: BLE001 - try the archive next
            print(f"  {url[:70]}...: {e}")
            continue
        # §5a and [[reference_pdf_truncated_at_source]]: the truncated Wayback capture starts
        # `%PDF-1.5` and opens, so the digest is pinned (sources/fetch_checks.py::check_body).
        try:
            check_body(body, "pdf", where=f"{url[:70]}...", pin_digest=DIGEST)
        except FetchCheckError as e:
            print(f"  {e}")
            continue
        with open(PDF + ".part", "wb") as fh:
            fh.write(body)
        os.replace(PDF + ".part", PDF)
        print(f"wrote {PDF} ({len(body):,} bytes) from {url[:45]}")
        return
    raise SystemExit("neither stat-guinebissau.com nor the Wayback copy returned the PDF")


def _text(doc, pno):
    return despace(" ".join(doc.load_page(pno).get_text().splitlines()))


def read_table(doc, pno, marker):
    """A región x religion table off one page: {row label: [tokens]}, in LABELS order.

    Rows are found by label in order after `marker`; a token `100,` followed by `0` (a cell
    wrapped at the column edge) is joined back into `100,0`.
    """
    text = _text(doc, pno)
    at = text.find(marker)
    if at < 0:
        raise SystemExit(f"no {marker!r} on page index {pno}")
    spans, pos = [], at
    for key, rx in LABELS:
        m = re.compile(rx).search(text, pos)
        if not m:
            raise SystemExit(f"no {key!r} row after {marker!r} on page index {pno}")
        spans.append((key, m.start(), m.end()))
        pos = m.end()
    out = {}
    for k, (key, _s, e) in enumerate(spans):
        end = spans[k + 1][1] if k + 1 < len(spans) else len(text)
        merged = []
        for t in text[e:end].split():
            if merged and re.fullmatch(r"\d+,", merged[-1]) and re.fullmatch(r"\d+", t):
                merged[-1] += t
            else:
                merged.append(t)
        out[key] = merged
    return out


def counts_of(tokens):
    vals = tokens[0::2]
    if not all(re.fullmatch(r"\d+", v) for v in vals):
        return None
    return tuple(int(v) for v in vals)


def pct(t):
    return float(t.replace(",", "."))


def unsd_counts():
    try:
        import oracle
        rows = oracle.oracle("Guinea-Bissau", 2009)
    except SystemExit:
        rows = None
    if not rows:
        return None
    tot = next((v for k, v in rows.items() if "total" in k.lower()), None)
    return {c: n for c, n in tot.items() if c != oracle.TOTAL} if tot else None


def ethnic_witness(say):
    """Predict each região's religion from its ethnic mix and Quadro 5's national rates."""
    import numpy as np

    # ANEXO QUADRO 2 HAS TWO MISPRINTED CELLS, each a dropped digit. As printed, the Fula row
    # sums 21,000 short of its total and so does the Oio column; the Mandinga row and the
    # Cacheu column are both 10,000 short; every other row and column closes. Fula in Oio
    # printed 2,980 is 23,980, and Mandinga in Cacheu printed 1,460 is 11,460 (their printed
    # shares, 0.7% each, were computed from the misprints). Other splits of the same totals
    # across these four cells would also close, but only this one is a single slipped digit.
    # The drawn table is not affected: this is the witness.
    a2 = {e: list(v) for e, v in A2.items()}
    misprints = {("Fula", "Oio"): 23_980, ("Mandinga", "Cacheu"): 11_460}
    short = {("Fula", "Oio"): 21_000, ("Mandinga", "Cacheu"): 10_000}
    for (e, r), fixed in misprints.items():
        i = REGIONS.index(r)
        row_gap = A2_TOTALS[e] - sum(A2[e])
        col_gap = A3["Total"][i] - sum(A2[x][i] for x in ETNIAS)
        say(row_gap == col_gap == short[(e, r)] == fixed - A2[e][i],
            f"Anexo Quadro 2 misprint: {e} in {r} printed {A2[e][i]:,}; its row and its "
            f"column are both {row_gap:,} short, so it is {fixed:,}")
        a2[e][i] = fixed
    say(all(sum(a2[e]) == A2_TOTALS[e] and sum(Q5[e]) == Q5_TOTALS[e] for e in ETNIAS),
        "with those two cells, all 16 etnias close on their printed totals in both tables "
        "(Balanta and Fula are printed one person apart between the two)")
    say(all(sum(a2[e][r] for e in ETNIAS) == A3["Total"][r] for r in range(9)),
        "and Anexo Quadro 2's etnia columns sum to every região's total")
    say(all(sum(Q5[e][c] for e in ETNIAS) == NATIONAL[CATS[c]] for c in range(6)),
        "Quadro 5's etnia rows sum to every national religion count")

    reg = np.array([a2[e] for e in ETNIAS], dtype=float)            # etnia x região
    rate = np.array([[q / sum(Q5[e]) for q in Q5[e]] for e in ETNIAS])  # etnia x religion
    pred = (reg.T @ rate) / np.array(A3["Total"], dtype=float)[:, None]  # região x religion
    act = np.array([[A3[c][r] / A3["Total"][r] for c in CATS] for r in range(9)])

    print(f"\n  ethnic witness, predicted / actual share (%) by região:")
    print("    " + " " * 15 + "".join(f"{c[:9]:>16}" for c in CATS))
    for r, name in enumerate(REGIONS):
        print(f"    {name:<15}" + "".join(f"{100 * pred[r, c]:>8.1f}{100 * act[r, c]:>8.1f}"
                                         for c in range(6)))
    # The prediction applies NATIONAL rates per etnia, so it cannot see that a Papel or
    # Balanta living in Bissau is more often Christian and less often animist than one in
    # the villages: SAB is predicted 17.2% animist and 28.5% Christian against 7.9% and 40.2%
    # printed. That caps the correlations below 1 for a real reason, so the bar is 0.8 and
    # the label test is the swap test after it, not the correlation.
    for c in (0, 1, 2, 4, 5):
        rr = float(np.corrcoef(pred[:, c], act[:, c])[0, 1])
        if c in (0, 1, 2):
            say(rr >= 0.8, f"{CATS[c]:<13} across the 9 regiões, r(predicted, printed) = "
                f"{rr:+.3f}")
        else:
            print(f"      {CATS[c]:<13} r = {rr:+.3f} (printed, not asserted)")

    # A swap test only means something for two regiões that look different. Tombali and Oio
    # are printed within 3.3 pp of each other on every answer, so swapping them improves the
    # fit by noise and would change nothing on the map. A swap FAILS only if the two regiões
    # differ by 5 pp or more somewhere, which is the kind of swap a reader could see.
    sse = lambda m: float(((m - pred) ** 2).sum())
    base = sse(act)
    beaten, visible = [], []
    for i in range(9):
        for j in range(i + 1, 9):
            sw = act.copy()
            sw[[i, j]] = sw[[j, i]]
            if sse(sw) < base:
                gap = float(100 * np.abs(act[i] - act[j]).max())
                tag = (f"{REGIONS[i]}/{REGIONS[j]} (improves SSE {base - sse(sw):.4f} of "
                       f"{base:.4f}; the two differ by at most {gap:.1f} pp)")
                (visible if gap >= 5.0 else beaten).append(tag)
    if beaten:
        print("      swaps that fit better between regiões printed alike, not a finding: "
              + "; ".join(beaten))
    say(not visible, "no swap of two visibly different regiões fits the ethnic prediction "
        "better than the table as printed (36 pairs tried)"
        + (": " + "; ".join(visible) if visible else ""))
    ba, ga = REGIONS.index("Bafatá"), REGIONS.index("Gabú")
    say(pred[ga, 1] > pred[ba, 1] and act[ga, 1] > act[ba, 1],
        f"Gabú is more Muslim than Bafatá in the prediction ({100 * pred[ga, 1]:.1f} against "
        f"{100 * pred[ba, 1]:.1f}) and in the annex ({100 * act[ga, 1]:.1f} against "
        f"{100 * act[ba, 1]:.1f}); the prose on PDF p30 has them the other way round")


def check(doc):
    ok = True

    def say(good, msg):
        nonlocal ok
        ok &= bool(good)
        print(f"  {'OK ' if good else 'BAD'} {msg}")

    print("Guinea-Bissau — RGPH 2009, Características socioculturais, Anexo Quadro 3\n")
    say(doc.page_count == PAGES, f"the volume is {doc.page_count} pages (expected {PAGES})")
    with open(PDF, "rb") as fh:
        body = fh.read()
    say(digest(body) == DIGEST, f"file digest {digest(body)} (expected {DIGEST})")

    # 1. the two annex copies, parsed, equal the transcription
    a3 = read_table(doc, PAGE_A3, "Quadro nº 3")
    a7 = read_table(doc, PAGE_A7, "Quadro nº 7")
    for label, tab in (("Anexo Quadro 3", a3), ("Anexo Quadro 7", a7)):
        bad = [k for k in A3 if counts_of(tab[k]) != (NATIONAL[k],) + A3[k]]
        say(not bad, f"{label}: all 7 rows x 10 columns of counts equal the transcription"
            + (f"; differ: {bad}" if bad else ""))
        for k in bad:
            print(f"        {k}: page {counts_of(tab[k])}")

    # 2. every row and column closes
    say(all(sum(A3[k]) == NATIONAL[k] for k in A3),
        "every religion's 9 regiões sum to its national count")
    say(all(sum(A3[c][r] for c in CATS) == A3["Total"][r] for r in range(9))
        and sum(NATIONAL[c] for c in CATS) == NATIONAL["Total"],
        "every região's 6 answers sum to its total, and the nation's to 1,442,227")

    # 3. Quadro 7's printed shares are the counts' shares, and body Quadro 4 is the same table
    shares = {k: [pct(t) for t in a7[k][1::2]] for k in A3}
    worst = max(abs(shares[k][i] - 100.0 * n / t)
                for k in A3 for i, (n, t) in enumerate(zip((NATIONAL[k],) + A3[k],
                                                           (NATIONAL["Total"],) + A3["Total"])))
    say(all(len(v) == 10 for v in shares.values()) and worst <= 0.05 + 1e-9,
        f"Anexo Quadro 7's 70 printed shares are the counts' shares within {worst:.3f} pp")
    q4 = read_table(doc, PAGE_Q4, "Quadro 4")
    q4v = {k: [pct(t) for t in q4[k]] for k in A3}
    say(all(q4v[k] == shares[k] for k in A3),
        "body Quadro 4 (PDF p30) equals Anexo Quadro 7 cell for cell, Bafatá and Gabú included")

    # 4. the universe
    t1 = _text(doc, PAGE_Q1)
    say(all(f"{n}" in t1 for n in (HOUSEHOLD_TOTAL, NATIONAL["Total"], FOREIGN)),
        f"Quadro 1: {HOUSEHOLD_TOTAL:,} in ordinary households = {NATIONAL['Total']:,} "
        f"Guinean + {FOREIGN:,} foreign + {HOUSEHOLD_TOTAL - NATIONAL['Total'] - FOREIGN:,} "
        "with no nationality recorded")
    t1a = _text(doc, PAGE_Q1A).split()
    say(all(str(n) in t1a for n in A3["Total"]),
        "Quadro 1A (PDF p22) prints the same 9 região totals")
    t2 = _text(doc, PAGE_A2).split()
    say(all(str(n) in t2 for n in A3["Total"]),
        "Anexo Quadro 2 (PDF p71) prints the same 9 região totals")
    tp = _text(doc, PAGE_PES)
    spaced = lambda n: f"{n:,}".replace(",", ".")
    say(all(spaced(n) in tp for n in HOUSEHOLD) and sum(HOUSEHOLD) == HOUSEHOLD_TOTAL
        and all(str(n) in tp.split() for n in COLLECTIVE)
        and sum(COLLECTIVE) == COLLECTIVE_TOTAL,
        f"PDF p3: household residents by região sum to {HOUSEHOLD_TOTAL:,}, collective to "
        f"{COLLECTIVE_TOTAL:,}")
    say(all(h >= n for h, n in zip(HOUSEHOLD, A3["Total"])),
        "no região has more Guinean nationals than household residents ("
        + ", ".join(f"{REGIONS[i][:6]} {h - n:,}" for i, (h, n)
                    in enumerate(zip(HOUSEHOLD, A3["Total"]))) + " not in the table)")

    # 5. UNSD
    u = unsd_counts()
    if u is None:
        print("  -- oracle cache not present; UNSD check skipped")
    else:
        say({UNSD[k]: v for k, v in u.items()} == {c: NATIONAL[c] for c in CATS},
            f"UNSD table 28 Guinea-Bissau 2009 equals the annex to the person: {u}")

    # 6. the ethnic witness
    ethnic_witness(say)

    if not ok:
        raise SystemExit("reconciliation FAILED")


def emit():
    rows = []
    for i, r in enumerate(REGIONS):
        for c in ["Total"] + CATS:
            n = A3[c][i]
            if n <= 0:
                continue
            rows.append({
                "geo_id": r, "geo_level": "region", "geo_name": NAMES.get(r, r),
                "source_category": c, "count": n, "basis": BASIS, "year": YEAR,
                "source_id": SOURCE_ID,
                "note": (f"Anexo Quadro 3 count; {100.0 * n / A3['Total'][i]:.2f}% of the "
                         f"região's {A3['Total'][i]:,} Guinean nationals"),
            })
    return rows


def main():
    import fitz

    if "--fetch" in sys.argv:
        fetch()
    if not os.path.exists(PDF):
        raise SystemExit(f"{PDF} missing — run: python sources/gw.py --fetch")
    doc = fitz.open(PDF)
    check(doc)
    rows = emit()

    drawn = sum(NATIONAL[c] for c in CATS if c != "ND")
    enumerated = HOUSEHOLD_TOTAL + COLLECTIVE_TOTAL
    outside = enumerated - NATIONAL["Total"]
    print(f"\n  drawn {drawn:,} of {NATIONAL['Total']:,} nationals "
          f"({100.0 * drawn / NATIONAL['Total']:.2f}%); ND {NATIONAL['ND']:,} "
          f"({100.0 * NATIONAL['ND'] / NATIONAL['Total']:.2f}% of nationals)")
    print(f"  of the {enumerated:,} enumerated: ND {100.0 * NATIONAL['ND'] / enumerated:.2f}%, "
          f"outside the table {outside:,} ({100.0 * outside / enumerated:.2f}%), "
          f"undrawn together {100.0 * (NATIONAL['ND'] + outside) / enumerated:.2f}%")
    print(f"\n  {'região':<16}{'nationals':>10}   shares of answers given (ND excluded)")
    for i, r in enumerate(REGIONS):
        ans = A3["Total"][i] - A3["ND"][i]
        print(f"  {r:<16}{A3['Total'][i]:>10,}   "
              + "  ".join(f"{c[:4]} {100.0 * A3[c][i] / ans:5.1f}" for c in CATS[:5])
              + f"   ND {100.0 * A3['ND'][i] / A3['Total'][i]:.1f}% of all")
    for c in ("Animista", "Cristão", "Outra religião"):
        top = max(range(9), key=lambda i: A3[c][i])
        print(f"  {c}: largest in {REGIONS[top]}, {100.0 * A3[c][top] / NATIONAL[c]:.1f}% "
              "of the national count")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print("\nwrote", OUT, f"({len(rows)} rows)")


if __name__ == "__main__":
    main()
